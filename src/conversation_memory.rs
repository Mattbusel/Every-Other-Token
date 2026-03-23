//! Persistent conversation memory with relevance decay over time.
//!
//! Records are scored by combining base importance, a configurable decay
//! function, and a recency boost that rewards recently-accessed entries.

use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// A single stored memory record.
#[derive(Debug, Clone)]
pub struct MemoryRecord {
    pub id: String,
    pub content: String,
    pub summary: String,
    pub tokens: usize,
    pub created_at: u64,
    pub last_accessed: u64,
    pub access_count: u32,
    pub importance: f64,
}

/// Controls how importance decays as time passes.
#[derive(Debug, Clone)]
pub enum DecayFunction {
    /// Importance halves every `half_life_ms` milliseconds.
    Exponential { half_life_ms: u64 },
    /// Importance decreases linearly at `rate_per_ms` per millisecond.
    Linear { rate_per_ms: f64 },
    /// Importance drops to zero after `threshold_ms` milliseconds.
    Step { threshold_ms: u64 },
}

// ---------------------------------------------------------------------------
// Decay / boost helpers
// ---------------------------------------------------------------------------

/// Compute the decay multiplier (0.0 – 1.0) based on age and decay function.
pub fn decay_factor(created_at: u64, now: u64, decay: &DecayFunction) -> f64 {
    let age_ms = now.saturating_sub(created_at) as f64;
    match decay {
        DecayFunction::Exponential { half_life_ms } => {
            let hl = (*half_life_ms).max(1) as f64;
            (-(age_ms / hl) * std::f64::consts::LN_2).exp()
        }
        DecayFunction::Linear { rate_per_ms } => {
            (1.0 - rate_per_ms * age_ms).max(0.0)
        }
        DecayFunction::Step { threshold_ms } => {
            if age_ms <= *threshold_ms as f64 { 1.0 } else { 0.0 }
        }
    }
}

/// Recency boost: 1.0 + 0.5 * exp(-hours_since_last_access).
pub fn recency_boost(last_accessed: u64, now: u64) -> f64 {
    let elapsed_ms = now.saturating_sub(last_accessed) as f64;
    let hours = elapsed_ms / 3_600_000.0;
    1.0 + 0.5 * (-hours).exp()
}

/// Compute the overall relevance score for a record.
pub fn relevance_score(record: &MemoryRecord, now: u64, decay: &DecayFunction) -> f64 {
    let df = decay_factor(record.created_at, now, decay);
    let rb = recency_boost(record.last_accessed, now);
    record.importance * df * rb
}

// ---------------------------------------------------------------------------
// ConversationMemory
// ---------------------------------------------------------------------------

/// In-memory store for conversation records with decay-based retrieval.
pub struct ConversationMemory {
    records: HashMap<String, MemoryRecord>,
    decay: DecayFunction,
}

impl ConversationMemory {
    /// Create a new store with the given decay function.
    pub fn new(decay: DecayFunction) -> Self {
        Self {
            records: HashMap::new(),
            decay,
        }
    }

    /// Store a new memory record, returning its generated ID.
    pub fn store(
        &mut self,
        content: &str,
        summary: &str,
        tokens: usize,
        importance: f64,
        now: u64,
    ) -> String {
        let id = Uuid::new_v4().to_string();
        let record = MemoryRecord {
            id: id.clone(),
            content: content.to_string(),
            summary: summary.to_string(),
            tokens,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            importance,
        };
        self.records.insert(id.clone(), record);
        id
    }

    /// Retrieve top-k records ranked by keyword overlap × relevance score.
    pub fn retrieve<'a>(&'a self, query: &str, top_k: usize, now: u64) -> Vec<(f64, &'a MemoryRecord)> {
        let query_words: Vec<&str> = query.split_whitespace().collect();

        let mut scored: Vec<(f64, &MemoryRecord)> = self
            .records
            .values()
            .map(|r| {
                let relevance = relevance_score(r, now, &self.decay);
                let overlap = keyword_overlap(&r.content, &query_words)
                    + keyword_overlap(&r.summary, &query_words);
                let score = overlap * relevance;
                (score, r)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    /// Update last_accessed timestamp and increment access_count.
    pub fn touch(&mut self, id: &str, now: u64) {
        if let Some(record) = self.records.get_mut(id) {
            record.last_accessed = now;
            record.access_count += 1;
        }
    }

    /// Remove a record by ID.  Returns `true` if it existed.
    pub fn forget(&mut self, id: &str) -> bool {
        self.records.remove(id).is_some()
    }

    /// Remove all records whose current relevance score is below `min_score`.
    /// Returns the number of records removed.
    pub fn prune_stale(&mut self, min_score: f64, now: u64) -> usize {
        let stale: Vec<String> = self
            .records
            .values()
            .filter(|r| relevance_score(r, now, &self.decay) < min_score)
            .map(|r| r.id.clone())
            .collect();
        let count = stale.len();
        for id in stale {
            self.records.remove(&id);
        }
        count
    }

    /// Human-readable summary of the store state.
    pub fn memory_summary(&self, now: u64) -> String {
        let n = self.records.len();
        if n == 0 {
            return "0 records".to_string();
        }

        let avg_importance = self.records.values().map(|r| r.importance).sum::<f64>() / n as f64;

        let oldest_created = self
            .records
            .values()
            .map(|r| r.created_at)
            .min()
            .unwrap_or(now);
        let age_days = now.saturating_sub(oldest_created) as f64 / 86_400_000.0;

        format!(
            "{} records, avg importance: {:.3}, oldest: {:.2} days",
            n, avg_importance, age_days
        )
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn keyword_overlap(text: &str, query_words: &[&str]) -> f64 {
    if query_words.is_empty() {
        return 0.0;
    }
    let lower = text.to_lowercase();
    let matches = query_words
        .iter()
        .filter(|w| lower.contains(&w.to_lowercase()))
        .count();
    matches as f64 / query_words.len() as f64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const T0: u64 = 1_000_000;

    fn make_memory() -> ConversationMemory {
        ConversationMemory::new(DecayFunction::Exponential { half_life_ms: 3_600_000 })
    }

    #[test]
    fn store_and_retrieve_basic() {
        let mut mem = make_memory();
        let id = mem.store("The sky is blue", "sky color", 5, 1.0, T0);
        let results = mem.retrieve("sky", 5, T0);
        assert!(!results.is_empty());
        assert_eq!(results[0].1.id, id);
    }

    #[test]
    fn forget_returns_true_then_false() {
        let mut mem = make_memory();
        let id = mem.store("hello world", "greeting", 2, 0.5, T0);
        assert!(mem.forget(&id));
        assert!(!mem.forget(&id));
    }

    #[test]
    fn touch_increments_access_count() {
        let mut mem = make_memory();
        let id = mem.store("test content", "test", 3, 0.8, T0);
        mem.touch(&id, T0 + 1000);
        let record = mem.records.get(&id).unwrap();
        assert_eq!(record.access_count, 1);
        assert_eq!(record.last_accessed, T0 + 1000);
    }

    #[test]
    fn prune_stale_removes_low_score_records() {
        let mut mem = ConversationMemory::new(DecayFunction::Step { threshold_ms: 1000 });
        mem.store("old content", "old", 3, 0.5, T0);
        // prune at T0 + 2000 — step threshold exceeded, score = 0.0
        let removed = mem.prune_stale(0.01, T0 + 2000);
        assert_eq!(removed, 1);
        assert_eq!(mem.records.len(), 0);
    }

    #[test]
    fn prune_stale_keeps_fresh_records() {
        let mut mem = ConversationMemory::new(DecayFunction::Step { threshold_ms: 10_000 });
        mem.store("fresh content", "fresh", 3, 1.0, T0);
        let removed = mem.prune_stale(0.01, T0 + 5_000);
        assert_eq!(removed, 0);
    }

    #[test]
    fn decay_exponential_halves() {
        let created = 0u64;
        let half_life = 1000u64;
        let decay = DecayFunction::Exponential { half_life_ms: half_life };
        let f_at_hl = decay_factor(created, half_life, &decay);
        assert!((f_at_hl - 0.5).abs() < 1e-9);
    }

    #[test]
    fn decay_linear_clamps_to_zero() {
        let decay = DecayFunction::Linear { rate_per_ms: 0.001 };
        assert_eq!(decay_factor(0, 2000, &decay), 0.0);
    }

    #[test]
    fn decay_step_before_after() {
        let decay = DecayFunction::Step { threshold_ms: 500 };
        assert_eq!(decay_factor(0, 400, &decay), 1.0);
        assert_eq!(decay_factor(0, 600, &decay), 0.0);
    }

    #[test]
    fn recency_boost_decreases_over_time() {
        let b0 = recency_boost(T0, T0);
        let b1 = recency_boost(T0, T0 + 3_600_000);
        assert!(b0 > b1);
        assert!(b0 >= 1.0);
    }

    #[test]
    fn memory_summary_format() {
        let mut mem = make_memory();
        mem.store("content a", "a", 4, 0.6, T0);
        mem.store("content b", "b", 4, 0.8, T0);
        let s = mem.memory_summary(T0);
        assert!(s.starts_with("2 records"));
    }

    #[test]
    fn retrieve_ranks_by_relevance() {
        let mut mem = make_memory();
        mem.store("rust programming language", "rust", 4, 1.0, T0);
        mem.store("baking bread with yeast", "bread", 4, 1.0, T0);
        let results = mem.retrieve("rust language", 5, T0);
        assert_eq!(results[0].1.summary, "rust");
    }

    #[test]
    fn empty_store_summary() {
        let mem = make_memory();
        assert_eq!(mem.memory_summary(T0), "0 records");
    }
}
