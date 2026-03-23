//! # Memory Retrieval
//!
//! Episodic memory store with recency + relevance + importance scoring.
//!
//! Provides a simple but effective vector of [`MemoryEntry`] items that can be
//! queried using a combined TF-IDF / recency / importance score.

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

// ---------------------------------------------------------------------------
// MemoryEntry
// ---------------------------------------------------------------------------

/// A single piece of episodic memory.
#[derive(Clone, Debug)]
pub struct MemoryEntry {
    /// Unique identifier assigned at store time.
    pub id: u64,
    /// The stored text content.
    pub content: String,
    /// Free-form tags for categorisation and retrieval.
    pub tags: Vec<String>,
    /// User-assigned importance score (0.0–1.0).
    pub importance: f64,
    /// Number of times this entry has been retrieved.
    pub access_count: u64,
    /// When this entry was first created.
    pub created_at: std::time::Instant,
    /// When this entry was last accessed.
    pub last_accessed: std::time::Instant,
}

// ---------------------------------------------------------------------------
// tfidf_score
// ---------------------------------------------------------------------------

/// Simple TF-IDF approximation.
///
/// For each word in `query`, counts its occurrences in `document` (TF),
/// then normalises by document length.  IDF is approximated as 1.0 for all
/// terms (we have only one document per call).
pub fn tfidf_score(query: &str, document: &str) -> f64 {
    if query.is_empty() || document.is_empty() {
        return 0.0;
    }
    let query_words: Vec<&str> = query.split_whitespace().collect();
    if query_words.is_empty() {
        return 0.0;
    }

    let doc_words: Vec<String> = document
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .collect();
    let doc_len = doc_words.len().max(1) as f64;

    let mut word_freq: HashMap<&str, u64> = HashMap::new();
    for w in &doc_words {
        // We need to match query words against lowercased doc words.
        for &qw in &query_words {
            if w == &qw.to_lowercase() {
                *word_freq.entry(qw).or_insert(0) += 1;
            }
        }
    }

    let score: f64 = word_freq.values().map(|&c| c as f64 / doc_len).sum();
    // Normalise by query length so longer queries don't get unfair advantage.
    score / query_words.len() as f64
}

// ---------------------------------------------------------------------------
// recency_score
// ---------------------------------------------------------------------------

/// Exponential decay recency score.
///
/// Returns `exp(-elapsed_secs / decay_secs)`, which is 1.0 when the entry was
/// just accessed and approaches 0.0 as time passes.
pub fn recency_score(last_accessed: &std::time::Instant, decay_secs: f64) -> f64 {
    let elapsed = last_accessed.elapsed().as_secs_f64();
    (-elapsed / decay_secs.max(f64::EPSILON)).exp()
}

// ---------------------------------------------------------------------------
// RetrievalScore
// ---------------------------------------------------------------------------

/// Breakdown of why an entry received a particular combined score.
#[derive(Debug, Clone)]
pub struct RetrievalScore {
    /// The entry this score was computed for.
    pub entry_id: u64,
    /// TF-IDF relevance component.
    pub relevance: f64,
    /// Recency decay component.
    pub recency: f64,
    /// User-specified importance component.
    pub importance: f64,
    /// Weighted combination of the three components.
    pub combined: f64,
}

// ---------------------------------------------------------------------------
// EpisodicMemory
// ---------------------------------------------------------------------------

/// In-memory episodic store with LRU-like eviction.
pub struct EpisodicMemory {
    entries: RwLock<Vec<MemoryEntry>>,
    next_id: AtomicU64,
    /// Maximum number of entries to retain before eviction.
    pub max_size: usize,
    /// Weight of the recency component (0.0–1.0).
    pub recency_weight: f64,
    /// Weight of the TF-IDF relevance component (0.0–1.0).
    pub relevance_weight: f64,
    /// Weight of the importance component (0.0–1.0).
    pub importance_weight: f64,
}

impl EpisodicMemory {
    /// Create a new memory store.
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: RwLock::new(Vec::new()),
            next_id: AtomicU64::new(1),
            max_size,
            recency_weight: 0.3,
            relevance_weight: 0.5,
            importance_weight: 0.2,
        }
    }

    /// Create a store with explicit weights (must sum to ≈ 1.0 for sensible results).
    pub fn with_weights(
        max_size: usize,
        recency_weight: f64,
        relevance_weight: f64,
        importance_weight: f64,
    ) -> Self {
        Self {
            entries: RwLock::new(Vec::new()),
            next_id: AtomicU64::new(1),
            max_size,
            recency_weight,
            relevance_weight,
            importance_weight,
        }
    }

    /// Store a new memory entry and return its assigned id.
    ///
    /// If the store is at `max_size` the least-relevant entries are evicted
    /// first.
    pub fn store(&self, content: &str, tags: Vec<String>, importance: f64) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let now = std::time::Instant::now();
        let entry = MemoryEntry {
            id,
            content: content.to_string(),
            tags,
            importance: importance.clamp(0.0, 1.0),
            access_count: 0,
            created_at: now,
            last_accessed: now,
        };

        if let Ok(mut entries) = self.entries.write() {
            entries.push(entry);
        }
        // Evict if over capacity.
        self.forget_least_relevant(self.max_size);
        id
    }

    /// Retrieve the `top_k` most relevant entries for `query`.
    pub fn retrieve(&self, query: &str, top_k: usize) -> Vec<MemoryEntry> {
        let entries = match self.entries.read() {
            Ok(e) => e,
            Err(_) => return Vec::new(),
        };

        let mut scored: Vec<(f64, usize)> = entries
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                let score = self.score_entry(entry, query);
                (score.combined, i)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        let result: Vec<MemoryEntry> = scored
            .iter()
            .map(|&(_, i)| entries[i].clone())
            .collect();

        // Update access metadata (best effort, separate lock acquisition).
        drop(entries);
        for entry in &result {
            self.update_access(entry.id);
        }

        result
    }

    /// Compute the retrieval score for `entry` against `query`.
    pub fn score_entry(&self, entry: &MemoryEntry, query: &str) -> RetrievalScore {
        const DECAY_SECS: f64 = 3600.0; // 1-hour half-life default
        let relevance = tfidf_score(query, &entry.content);
        let recency = recency_score(&entry.last_accessed, DECAY_SECS);
        let importance = entry.importance;
        let combined = self.relevance_weight * relevance
            + self.recency_weight * recency
            + self.importance_weight * importance;
        RetrievalScore {
            entry_id: entry.id,
            relevance,
            recency,
            importance,
            combined,
        }
    }

    /// Increment access count and refresh `last_accessed` for `entry_id`.
    pub fn update_access(&self, entry_id: u64) {
        if let Ok(mut entries) = self.entries.write() {
            for entry in entries.iter_mut() {
                if entry.id == entry_id {
                    entry.access_count += 1;
                    entry.last_accessed = std::time::Instant::now();
                    break;
                }
            }
        }
    }

    /// If the store holds more than `keep_n` entries, drop those with the
    /// lowest combined score (using an empty query for scoring).
    pub fn forget_least_relevant(&self, keep_n: usize) {
        let len = match self.entries.read() {
            Ok(e) => e.len(),
            Err(_) => return,
        };
        if len <= keep_n {
            return;
        }

        let entries_snap: Vec<MemoryEntry> = match self.entries.read() {
            Ok(e) => e.clone(),
            Err(_) => return,
        };

        let mut scored: Vec<(f64, u64)> = entries_snap
            .iter()
            .map(|e| (self.score_entry(e, "").combined, e.id))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Keep the top `keep_n` by score.
        let keep_ids: std::collections::HashSet<u64> =
            scored.iter().take(keep_n).map(|&(_, id)| id).collect();

        if let Ok(mut entries) = self.entries.write() {
            entries.retain(|e| keep_ids.contains(&e.id));
        }
    }

    /// Return all entries that have `tag` in their tag list.
    pub fn entries_by_tag(&self, tag: &str) -> Vec<MemoryEntry> {
        match self.entries.read() {
            Ok(entries) => entries
                .iter()
                .filter(|e| e.tags.iter().any(|t| t == tag))
                .cloned()
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Returns `(count, avg_importance, avg_access_count)`.
    pub fn summary_stats(&self) -> (usize, f64, f64) {
        match self.entries.read() {
            Ok(entries) => {
                let count = entries.len();
                if count == 0 {
                    return (0, 0.0, 0.0);
                }
                let avg_importance = entries.iter().map(|e| e.importance).sum::<f64>() / count as f64;
                let avg_access = entries.iter().map(|e| e.access_count as f64).sum::<f64>() / count as f64;
                (count, avg_importance, avg_access)
            }
            Err(_) => (0, 0.0, 0.0),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_and_retrieve_basic() {
        let mem = EpisodicMemory::new(100);
        let id = mem.store("The capital of France is Paris", vec!["geography".to_string()], 0.9);
        mem.store("Rust is a systems programming language", vec!["tech".to_string()], 0.7);
        let results = mem.retrieve("France capital", 1);
        assert!(!results.is_empty());
        assert_eq!(results[0].id, id);
    }

    #[test]
    fn eviction_respects_max_size() {
        let mem = EpisodicMemory::new(3);
        for i in 0..5 {
            mem.store(&format!("entry {i}"), vec![], 0.5);
        }
        let (count, _, _) = mem.summary_stats();
        assert!(count <= 3);
    }

    #[test]
    fn entries_by_tag() {
        let mem = EpisodicMemory::new(100);
        mem.store("tagged content", vec!["important".to_string()], 0.8);
        mem.store("other content", vec!["misc".to_string()], 0.5);
        let tagged = mem.entries_by_tag("important");
        assert_eq!(tagged.len(), 1);
    }

    #[test]
    fn tfidf_score_non_zero_for_matching_query() {
        let score = tfidf_score("Rust programming", "Rust is a great programming language");
        assert!(score > 0.0);
    }

    #[test]
    fn tfidf_score_zero_for_no_match() {
        let score = tfidf_score("Python", "Rust is fast");
        assert_eq!(score, 0.0);
    }

    #[test]
    fn recency_score_near_one_when_fresh() {
        let now = std::time::Instant::now();
        let score = recency_score(&now, 3600.0);
        assert!(score > 0.99);
    }

    #[test]
    fn summary_stats_empty() {
        let mem = EpisodicMemory::new(10);
        let (count, avg_imp, avg_acc) = mem.summary_stats();
        assert_eq!(count, 0);
        assert_eq!(avg_imp, 0.0);
        assert_eq!(avg_acc, 0.0);
    }

    #[test]
    fn update_access_increments_count() {
        let mem = EpisodicMemory::new(10);
        let id = mem.store("test", vec![], 0.5);
        mem.update_access(id);
        let entries = mem.entries.read().unwrap();
        let entry = entries.iter().find(|e| e.id == id).unwrap();
        assert!(entry.access_count >= 1);
    }
}
