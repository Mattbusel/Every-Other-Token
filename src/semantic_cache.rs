//! Semantic cache for prompt-response deduplication using cosine similarity.
//!
//! Prompts are embedded as bag-of-words TF vectors in a 256-dimensional space
//! (word → hash % 256). Cache hits are returned when cosine similarity is at or
//! above the configured threshold.

use std::time::Instant;

// ── Embedding ─────────────────────────────────────────────────────────────────

/// A dense 256-dimensional embedding vector.
#[derive(Debug, Clone)]
pub struct Embedding {
    /// Fixed-length 256-element vector.
    pub dims: Vec<f64>,
}

impl Embedding {
    /// Cosine similarity between two embeddings.
    /// Returns `0.0` if either vector is zero-length.
    pub fn cosine_similarity(a: &Embedding, b: &Embedding) -> f64 {
        let dot: f64 = a.dims.iter().zip(b.dims.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.dims.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.dims.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a < 1e-12 || norm_b < 1e-12 {
            return 0.0;
        }
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }

    /// Build a bag-of-words TF embedding: count word frequencies, normalize by L2 norm.
    /// Dimension = hash(word) % 256.
    pub fn from_text(text: &str) -> Embedding {
        let mut dims = vec![0.0f64; 256];
        for word in text.split_whitespace() {
            let w = word.to_lowercase();
            let w = w.trim_matches(|c: char| !c.is_alphabetic());
            if w.is_empty() {
                continue;
            }
            let idx = simple_hash(w) % 256;
            dims[idx] += 1.0;
        }
        // L2 normalize.
        let norm: f64 = dims.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for d in &mut dims {
                *d /= norm;
            }
        }
        Embedding { dims }
    }
}

/// FNV-1a-inspired hash for a string slice, returning a usize.
fn simple_hash(s: &str) -> usize {
    let mut h: usize = 0xcbf29ce484222325usize;
    for b in s.bytes() {
        h ^= b as usize;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ── CacheEntry ────────────────────────────────────────────────────────────────

/// A single entry in the semantic cache.
pub struct CacheEntry {
    /// Original prompt text.
    pub prompt: String,
    /// Cached response.
    pub response: String,
    /// Embedding of the prompt.
    pub embedding: Embedding,
    /// Wall-clock time when this entry was inserted.
    pub created_at: Instant,
    /// Number of times this entry has been returned as a hit.
    pub hits: u32,
}

// ── CacheStats ────────────────────────────────────────────────────────────────

/// Aggregate statistics for a [`SemanticCache`].
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Current number of entries.
    pub size: usize,
    /// Total number of `lookup` calls.
    pub total_lookups: u64,
    /// Number of cache hits.
    pub hits: u64,
    /// `hits / total_lookups`, or `0.0` if no lookups yet.
    pub hit_rate: f64,
    /// Mean cosine similarity on hit calls (0 if no hits).
    pub avg_similarity_on_hit: f64,
}

// ── SemanticCache ─────────────────────────────────────────────────────────────

/// Semantic prompt-response cache backed by cosine-similarity search.
pub struct SemanticCache {
    /// Stored entries, newest last.
    pub entries: Vec<CacheEntry>,
    /// Minimum cosine similarity required for a cache hit.
    pub threshold: f64,
    /// Maximum number of entries before LRU eviction.
    pub max_size: usize,
    // Internal stats.
    total_lookups: u64,
    total_hits: u64,
    similarity_sum_on_hit: f64,
}

impl SemanticCache {
    /// Create a new cache with the given threshold and capacity.
    pub fn new(threshold: f64, max_size: usize) -> Self {
        SemanticCache {
            entries: Vec::new(),
            threshold,
            max_size,
            total_lookups: 0,
            total_hits: 0,
            similarity_sum_on_hit: 0.0,
        }
    }

    /// Look up a prompt. Returns a reference to the cached response if a match
    /// with cosine similarity ≥ threshold exists; otherwise `None`.
    pub fn lookup(&mut self, prompt: &str) -> Option<&str> {
        self.total_lookups += 1;
        let query = Embedding::from_text(prompt);

        // Find best match.
        let mut best_idx: Option<usize> = None;
        let mut best_sim = -1.0f64;
        for (i, entry) in self.entries.iter().enumerate() {
            let sim = Embedding::cosine_similarity(&query, &entry.embedding);
            if sim >= self.threshold && sim > best_sim {
                best_sim = sim;
                best_idx = Some(i);
            }
        }

        if let Some(idx) = best_idx {
            self.entries[idx].hits += 1;
            self.total_hits += 1;
            self.similarity_sum_on_hit += best_sim;
            // Return reference — Rust requires we re-borrow after the mutable ops.
            return Some(self.entries[idx].response.as_str());
        }
        None
    }

    /// Insert a new prompt-response pair. Evicts if at capacity.
    pub fn insert(&mut self, prompt: &str, response: &str) {
        if self.entries.len() >= self.max_size {
            self.evict_lru();
        }
        let embedding = Embedding::from_text(prompt);
        self.entries.push(CacheEntry {
            prompt: prompt.to_string(),
            response: response.to_string(),
            embedding,
            created_at: Instant::now(),
            hits: 0,
        });
    }

    /// Remove the entry with the lowest composite score: hits + age_weight.
    /// Age weight = max_age_secs / age_secs so newer entries are preferred.
    pub fn evict_lru(&mut self) {
        if self.entries.is_empty() {
            return;
        }
        let now = Instant::now();
        // Score = hits. Tie-break: older = lower score.
        // We want to evict lowest hits, and among equal-hits the oldest.
        let evict_idx = self
            .entries
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let sa = a.hits as f64;
                let sb = b.hits as f64;
                // Lower hits first; if equal, older (larger elapsed) first.
                sa.partial_cmp(&sb)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        let age_a = now.duration_since(a.created_at).as_secs_f64();
                        let age_b = now.duration_since(b.created_at).as_secs_f64();
                        // larger age = evict first (reverse order)
                        age_b.partial_cmp(&age_a).unwrap_or(std::cmp::Ordering::Equal)
                    })
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        self.entries.remove(evict_idx);
    }

    /// Return aggregate cache statistics.
    pub fn stats(&self) -> CacheStats {
        let hit_rate = if self.total_lookups > 0 {
            self.total_hits as f64 / self.total_lookups as f64
        } else {
            0.0
        };
        let avg_similarity_on_hit = if self.total_hits > 0 {
            self.similarity_sum_on_hit / self.total_hits as f64
        } else {
            0.0
        };
        CacheStats {
            size: self.entries.len(),
            total_lookups: self.total_lookups,
            hits: self.total_hits,
            hit_rate,
            avg_similarity_on_hit,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_hits_with_cosine_one() {
        let mut cache = SemanticCache::new(0.95, 10);
        cache.insert("What is Rust?", "A systems language.");
        let result = cache.lookup("What is Rust?");
        assert_eq!(result, Some("A systems language."));
    }

    #[test]
    fn below_threshold_misses() {
        let mut cache = SemanticCache::new(0.99, 10);
        cache.insert("What is Rust?", "A systems language.");
        // Completely unrelated prompt.
        let result = cache.lookup("Tell me about quantum physics.");
        assert_eq!(result, None);
    }

    #[test]
    fn hit_counter_increments() {
        let mut cache = SemanticCache::new(0.95, 10);
        cache.insert("Hello world", "Hi there.");
        cache.lookup("Hello world");
        cache.lookup("Hello world");
        assert_eq!(cache.entries[0].hits, 2);
    }

    #[test]
    fn lru_eviction_removes_lowest_hits() {
        let mut cache = SemanticCache::new(0.95, 2);
        cache.insert("prompt one", "response one");
        cache.insert("prompt two", "response two");
        // Hit "prompt one" twice so it survives eviction.
        cache.lookup("prompt one");
        cache.lookup("prompt one");
        // Inserting a third entry should evict "prompt two" (0 hits).
        cache.insert("prompt three", "response three");
        assert_eq!(cache.entries.len(), 2);
        let prompts: Vec<&str> = cache.entries.iter().map(|e| e.prompt.as_str()).collect();
        assert!(!prompts.contains(&"prompt two"), "prompt two should have been evicted");
    }

    #[test]
    fn different_text_misses() {
        let mut cache = SemanticCache::new(0.99, 10);
        cache.insert("apple banana cherry", "fruit response");
        let result = cache.lookup("dog cat fish");
        assert_eq!(result, None);
    }

    #[test]
    fn stats_track_correctly() {
        let mut cache = SemanticCache::new(0.95, 10);
        cache.insert("hello world", "hi");
        cache.lookup("hello world"); // hit
        cache.lookup("something unrelated and completely different xyz"); // miss
        let stats = cache.stats();
        assert_eq!(stats.total_lookups, 2);
        assert_eq!(stats.hits, 1);
        assert!((stats.hit_rate - 0.5).abs() < 1e-9);
    }
}
