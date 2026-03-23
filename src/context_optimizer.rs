//! Context window optimizer: deduplication, compression, and priority-based budgeting.

use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Priority constants
// ---------------------------------------------------------------------------

pub struct Priority;
impl Priority {
    pub const SYSTEM: u8 = 0;
    pub const CRITICAL: u8 = 1;
    pub const HIGH: u8 = 2;
    pub const NORMAL: u8 = 3;
    pub const LOW: u8 = 4;
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct ContextEntry {
    pub id: String,
    pub content: String,
    pub tokens: usize,
    pub priority: u8,
    pub created_at: u64,
}

impl ContextEntry {
    pub fn new(
        id: impl Into<String>,
        content: impl Into<String>,
        tokens: usize,
        priority: u8,
        created_at: u64,
    ) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            tokens,
            priority,
            created_at,
        }
    }
}

#[derive(Debug, Clone)]
pub enum DedupStrategy {
    /// Remove entries with byte-identical content.
    Exact,
    /// Remove entries whose Jaccard similarity on word 3-shingles exceeds the threshold.
    FuzzyHash { threshold: f64 },
    /// Remove entries that share the same SHA-256-like content hash (simple polynomial hash).
    ContentHash,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute a simple 64-bit polynomial hash of a string (FNV-1a).
fn content_hash(s: &str) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

/// Build the set of word 3-shingles from text.
fn word_shingles(text: &str, k: usize) -> HashSet<Vec<String>> {
    let words: Vec<String> = text
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .collect();
    if words.len() < k {
        let mut s = HashSet::new();
        s.insert(words);
        return s;
    }
    words.windows(k).map(|w| w.to_vec()).collect()
}

/// Jaccard similarity between two shingle sets.
fn jaccard(a: &HashSet<Vec<String>>, b: &HashSet<Vec<String>>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

// ---------------------------------------------------------------------------
// ContextOptimizer
// ---------------------------------------------------------------------------

pub struct ContextOptimizer {
    /// Token threshold above which compress_entry will truncate content.
    pub compress_threshold: usize,
}

impl Default for ContextOptimizer {
    fn default() -> Self {
        Self {
            compress_threshold: 200,
        }
    }
}

impl ContextOptimizer {
    pub fn new(compress_threshold: usize) -> Self {
        Self { compress_threshold }
    }

    /// Remove near-duplicate entries according to strategy.
    /// The first occurrence (by order) is kept.
    pub fn deduplicate(
        &self,
        entries: &[ContextEntry],
        strategy: &DedupStrategy,
    ) -> Vec<ContextEntry> {
        let mut kept: Vec<ContextEntry> = Vec::new();

        match strategy {
            DedupStrategy::Exact => {
                let mut seen: HashSet<String> = HashSet::new();
                for e in entries {
                    if seen.insert(e.content.clone()) {
                        kept.push(e.clone());
                    }
                }
            }
            DedupStrategy::ContentHash => {
                let mut seen: HashSet<u64> = HashSet::new();
                for e in entries {
                    let h = content_hash(&e.content);
                    if seen.insert(h) {
                        kept.push(e.clone());
                    }
                }
            }
            DedupStrategy::FuzzyHash { threshold } => {
                let shingles: Vec<HashSet<Vec<String>>> = entries
                    .iter()
                    .map(|e| word_shingles(&e.content, 3))
                    .collect();

                let mut removed: Vec<bool> = vec![false; entries.len()];
                for i in 0..entries.len() {
                    if removed[i] {
                        continue;
                    }
                    for j in (i + 1)..entries.len() {
                        if removed[j] {
                            continue;
                        }
                        if jaccard(&shingles[i], &shingles[j]) >= *threshold {
                            removed[j] = true;
                        }
                    }
                }
                for (i, e) in entries.iter().enumerate() {
                    if !removed[i] {
                        kept.push(e.clone());
                    }
                }
            }
        }

        kept
    }

    /// Greedy 0-1 knapsack: keep highest-priority entries within token budget.
    /// Lower priority number = higher importance (SYSTEM=0 is most important).
    pub fn prioritize(
        &self,
        entries: &mut Vec<ContextEntry>,
        budget_tokens: usize,
    ) -> Vec<ContextEntry> {
        // Sort ascending by priority value (0 = SYSTEM = most important first),
        // tie-break by creation time (oldest first keeps context coherence).
        entries.sort_by(|a, b| a.priority.cmp(&b.priority).then(a.created_at.cmp(&b.created_at)));

        let mut result = Vec::new();
        let mut used = 0usize;
        for e in entries.iter() {
            if used + e.tokens <= budget_tokens {
                used += e.tokens;
                result.push(e.clone());
            }
        }
        result
    }

    /// Truncate content to 60% of its length if tokens exceed compress_threshold.
    pub fn compress_entry(&self, entry: &ContextEntry) -> ContextEntry {
        if entry.tokens <= self.compress_threshold {
            return entry.clone();
        }
        let chars: Vec<char> = entry.content.chars().collect();
        let keep = (chars.len() as f64 * 0.6) as usize;
        let new_content: String = chars[..keep].iter().collect::<String>() + "...";
        // Estimate new token count proportionally (60% + a bit for ellipsis).
        let new_tokens = (entry.tokens as f64 * 0.6) as usize + 1;
        ContextEntry {
            id: entry.id.clone(),
            content: new_content,
            tokens: new_tokens,
            priority: entry.priority,
            created_at: entry.created_at,
        }
    }

    /// Full pipeline: dedup → compress → prioritize.
    pub fn optimize(
        &self,
        entries: Vec<ContextEntry>,
        budget_tokens: usize,
        dedup: &DedupStrategy,
    ) -> Vec<ContextEntry> {
        let deduped = self.deduplicate(&entries, dedup);
        let mut compressed: Vec<ContextEntry> =
            deduped.iter().map(|e| self.compress_entry(e)).collect();
        self.prioritize(&mut compressed, budget_tokens)
    }

    /// Number of tokens saved by optimization.
    pub fn token_savings(original: &[ContextEntry], optimized: &[ContextEntry]) -> usize {
        let orig_total: usize = original.iter().map(|e| e.tokens).sum();
        let opt_total: usize = optimized.iter().map(|e| e.tokens).sum();
        orig_total.saturating_sub(opt_total)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str, content: &str, tokens: usize, priority: u8) -> ContextEntry {
        ContextEntry::new(id, content, tokens, priority, 0)
    }

    #[test]
    fn test_exact_dedup() {
        let opt = ContextOptimizer::default();
        let entries = vec![
            entry("a", "hello world", 2, Priority::NORMAL),
            entry("b", "hello world", 2, Priority::HIGH),
            entry("c", "different text", 2, Priority::LOW),
        ];
        let result = opt.deduplicate(&entries, &DedupStrategy::Exact);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].id, "a");
        assert_eq!(result[1].id, "c");
    }

    #[test]
    fn test_content_hash_dedup() {
        let opt = ContextOptimizer::default();
        let entries = vec![
            entry("a", "hello world", 2, Priority::NORMAL),
            entry("b", "hello world", 2, Priority::NORMAL),
        ];
        let result = opt.deduplicate(&entries, &DedupStrategy::ContentHash);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_fuzzy_dedup() {
        let opt = ContextOptimizer::default();
        let entries = vec![
            entry("a", "the quick brown fox jumps over the lazy dog", 9, Priority::NORMAL),
            entry("b", "the quick brown fox jumps over the lazy cat", 9, Priority::NORMAL),
            entry("c", "completely unrelated text about mathematics and science", 8, Priority::NORMAL),
        ];
        // High threshold should keep all
        let result_strict = opt.deduplicate(&entries, &DedupStrategy::FuzzyHash { threshold: 0.99 });
        assert_eq!(result_strict.len(), 3);
        // Lower threshold merges first two (they share most shingles)
        let result_loose = opt.deduplicate(&entries, &DedupStrategy::FuzzyHash { threshold: 0.5 });
        assert!(result_loose.len() < 3);
    }

    #[test]
    fn test_prioritize_budget() {
        let opt = ContextOptimizer::default();
        let mut entries = vec![
            entry("a", "system prompt", 50, Priority::SYSTEM),
            entry("b", "critical info", 50, Priority::CRITICAL),
            entry("c", "normal info", 50, Priority::NORMAL),
            entry("d", "low priority", 50, Priority::LOW),
        ];
        let result = opt.prioritize(&mut entries, 120);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].id, "a");
        assert_eq!(result[1].id, "b");
    }

    #[test]
    fn test_compress_entry() {
        let opt = ContextOptimizer::new(10);
        let e = entry("a", "abcdefghijklmnopqrstuvwxyz", 20, Priority::NORMAL);
        let compressed = opt.compress_entry(&e);
        assert!(compressed.content.ends_with("..."));
        assert!(compressed.tokens < e.tokens);
        // Content should be ~60% of original chars
        let chars_kept = compressed.content.len() - 3; // subtract "..."
        assert!(chars_kept <= (26.0 * 0.6) as usize + 1);
    }

    #[test]
    fn test_compress_below_threshold_unchanged() {
        let opt = ContextOptimizer::new(200);
        let e = entry("a", "short text", 5, Priority::NORMAL);
        let compressed = opt.compress_entry(&e);
        assert_eq!(compressed.content, "short text");
    }

    #[test]
    fn test_optimize_pipeline() {
        let opt = ContextOptimizer::new(5);
        let entries = vec![
            entry("a", "system prompt here", 10, Priority::SYSTEM),
            entry("b", "system prompt here", 10, Priority::NORMAL), // duplicate
            entry("c", "low priority info", 10, Priority::LOW),
        ];
        let result = opt.optimize(entries.clone(), 15, &DedupStrategy::Exact);
        // "b" removed as duplicate, "a" compressed (tokens=10>5 threshold), budget 15
        // "a" compressed to ~7 tokens, "c" compressed to ~7 tokens; SYSTEM first, fits ~7
        assert!(!result.is_empty());
        assert_eq!(result[0].id, "a");
    }

    #[test]
    fn test_token_savings() {
        let original = vec![
            entry("a", "text", 100, Priority::NORMAL),
            entry("b", "text2", 200, Priority::NORMAL),
        ];
        let optimized = vec![entry("a", "text", 100, Priority::NORMAL)];
        assert_eq!(ContextOptimizer::token_savings(&original, &optimized), 200);
    }

    #[test]
    fn test_priority_constants() {
        assert!(Priority::SYSTEM < Priority::CRITICAL);
        assert!(Priority::CRITICAL < Priority::HIGH);
        assert!(Priority::HIGH < Priority::NORMAL);
        assert!(Priority::NORMAL < Priority::LOW);
    }

    #[test]
    fn test_empty_input() {
        let opt = ContextOptimizer::default();
        let result = opt.optimize(vec![], 1000, &DedupStrategy::Exact);
        assert!(result.is_empty());
        assert_eq!(ContextOptimizer::token_savings(&[], &[]), 0);
    }

    #[test]
    fn test_jaccard_identical() {
        let a = word_shingles("the quick brown fox", 3);
        let b = word_shingles("the quick brown fox", 3);
        assert!((jaccard(&a, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let a = word_shingles("apple banana cherry", 3);
        let b = word_shingles("dog elephant frog", 3);
        assert_eq!(jaccard(&a, &b), 0.0);
    }
}
