//! # Stage: Agent Memory System (Task 2.3)
//!
//! ## Responsibility
//! Persistent knowledge base for the self-modifying agent loop.  Stores and
//! retrieves four categories of knowledge:
//!
//! 1. **Modification records** — past changes with outcomes (success/failure,
//!    metric delta, rollback flag).  Agents query before starting to avoid
//!    repeating failed approaches.
//!
//! 2. **Code patterns** — reusable patterns (snippets, architectural decisions)
//!    keyed by tag set.  Agents consult before writing new code.
//!
//! 3. **Performance baselines** — named metric baselines (p50/p95/p99 latency,
//!    throughput, etc.) updated after each successful deployment.
//!
//! 4. **Dead ends** — approaches that have been tried and failed with a reason.
//!    The meta-task generator consults this to avoid generating tasks that would
//!    just reproduce known-bad approaches.
//!
//! In production this would be Redis-backed; this implementation provides a
//! fully in-memory store with the same interface so the rest of the system
//! compiles and tests without a Redis dependency.  A `RedisBackend` can be
//! swapped in by implementing the `MemoryBackend` trait.
//!
//! ## Guarantees
//! - Bounded: all collections have configurable capacity limits
//! - Non-panicking: all lookups return Option / Result
//! - Thread-safe: designed to be wrapped in `Arc<Mutex<AgentMemory>>`

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Outcome
// ---------------------------------------------------------------------------

/// Whether a modification attempt succeeded or failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Success,
    Failure,
    RolledBack,
    Pending,
}

impl std::fmt::Display for Outcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Outcome::Success => write!(f, "success"),
            Outcome::Failure => write!(f, "failure"),
            Outcome::RolledBack => write!(f, "rolled_back"),
            Outcome::Pending => write!(f, "pending"),
        }
    }
}

// ---------------------------------------------------------------------------
// ModificationRecord
// ---------------------------------------------------------------------------

/// A record of one past self-modification attempt.
#[derive(Debug, Clone)]
pub struct ModificationRecord {
    /// Unique identifier (task ID or agent-assigned UUID).
    pub id: String,
    /// Short description of what was attempted.
    pub description: String,
    /// Files that were changed.
    pub affected_files: Vec<String>,
    /// Final outcome.
    pub outcome: Outcome,
    /// Change in key metrics after the modification (positive = improvement).
    /// Key: metric name, Value: fractional delta (0.1 = 10% better).
    pub metric_deltas: HashMap<String, f64>,
    /// Lessons learned / notes for future agents.
    pub notes: String,
    /// Timestamp in ms.
    pub timestamp_ms: u64,
}

impl ModificationRecord {
    pub fn was_successful(&self) -> bool {
        self.outcome == Outcome::Success
    }
}

// ---------------------------------------------------------------------------
// CodePattern
// ---------------------------------------------------------------------------

/// A reusable code or architectural pattern.
#[derive(Debug, Clone)]
pub struct CodePattern {
    pub id: String,
    pub title: String,
    /// Full text of the pattern (code snippet, pseudo-code, or description).
    pub content: String,
    /// Searchable tags (e.g. "error-handling", "async", "backpressure").
    pub tags: HashSet<String>,
    /// How many times this pattern has been used (agents increment on use).
    pub use_count: u32,
    pub created_at_ms: u64,
}

impl CodePattern {
    /// Returns `true` if this pattern matches all provided tags.
    pub fn matches_all_tags(&self, tags: &[&str]) -> bool {
        tags.iter().all(|t| self.tags.contains(*t))
    }

    /// Returns `true` if this pattern matches any of the provided tags.
    pub fn matches_any_tag(&self, tags: &[&str]) -> bool {
        tags.iter().any(|t| self.tags.contains(*t))
    }
}

// ---------------------------------------------------------------------------
// PerformanceBaseline
// ---------------------------------------------------------------------------

/// Baseline performance values for a named metric.
#[derive(Debug, Clone)]
pub struct PerformanceBaseline {
    pub metric: String,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    /// Simple moving average of recent values.
    pub mean: f64,
    /// Standard deviation of recent values.
    pub std_dev: f64,
    /// Number of samples used to compute this baseline.
    pub sample_count: u64,
    pub updated_at_ms: u64,
}

impl PerformanceBaseline {
    /// Returns `true` if `value` is within `sigma` standard deviations of mean.
    pub fn is_normal(&self, value: f64, sigma: f64) -> bool {
        if self.std_dev == 0.0 {
            return (value - self.mean).abs() < f64::EPSILON;
        }
        let z = (value - self.mean).abs() / self.std_dev;
        z <= sigma
    }

    /// Fractional deviation from mean (positive = above, negative = below).
    pub fn deviation_fraction(&self, value: f64) -> f64 {
        if self.mean == 0.0 { return 0.0; }
        (value - self.mean) / self.mean
    }
}

// ---------------------------------------------------------------------------
// DeadEnd
// ---------------------------------------------------------------------------

/// An approach that has been tried and should not be retried.
#[derive(Debug, Clone)]
pub struct DeadEnd {
    /// Stable key identifying the approach (e.g. "increase-dedup-ttl-above-600s").
    pub key: String,
    /// Human-readable description of the failed approach.
    pub description: String,
    /// Why it failed.
    pub reason: String,
    /// Signals / metrics that triggered this approach (for context matching).
    pub related_signals: Vec<String>,
    pub recorded_at_ms: u64,
}

// ---------------------------------------------------------------------------
// MemoryConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Maximum modification records to retain (oldest evicted).
    pub max_modifications: usize,
    /// Maximum code patterns.
    pub max_patterns: usize,
    /// Maximum dead ends.
    pub max_dead_ends: usize,
    /// Maximum performance baseline entries.
    pub max_baselines: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_modifications: 1000,
            max_patterns: 500,
            max_dead_ends: 200,
            max_baselines: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// MemoryStats
// ---------------------------------------------------------------------------

/// Summary statistics for the current state of the memory store.
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub modification_count: usize,
    pub pattern_count: usize,
    pub baseline_count: usize,
    pub dead_end_count: usize,
    pub success_rate: f64,
}

// ---------------------------------------------------------------------------
// AgentMemory
// ---------------------------------------------------------------------------

/// In-memory knowledge base for the self-modifying agent loop.
pub struct AgentMemory {
    config: MemoryConfig,
    /// Modification history (ring buffer, front = oldest).
    modifications: VecDeque<ModificationRecord>,
    /// Code patterns keyed by ID.
    patterns: HashMap<String, CodePattern>,
    /// Performance baselines keyed by metric name.
    baselines: HashMap<String, PerformanceBaseline>,
    /// Dead ends keyed by their stable key.
    dead_ends: HashMap<String, DeadEnd>,
}

impl AgentMemory {
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            config,
            modifications: VecDeque::new(),
            patterns: HashMap::new(),
            baselines: HashMap::new(),
            dead_ends: HashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Modification records
    // -----------------------------------------------------------------------

    /// Record a modification attempt.  Returns the record ID.
    pub fn record_modification(&mut self, record: ModificationRecord) -> String {
        if self.modifications.len() >= self.config.max_modifications {
            self.modifications.pop_front();
        }
        let id = record.id.clone();
        self.modifications.push_back(record);
        id
    }

    /// Look up a modification record by ID.
    pub fn get_modification(&self, id: &str) -> Option<&ModificationRecord> {
        self.modifications.iter().find(|r| r.id == id)
    }

    /// Return all modification records, oldest first.
    pub fn modifications(&self) -> impl Iterator<Item = &ModificationRecord> {
        self.modifications.iter()
    }

    /// Return the N most recent modification records.
    pub fn recent_modifications(&self, n: usize) -> Vec<&ModificationRecord> {
        self.modifications.iter().rev().take(n).collect()
    }

    /// Return all modifications affecting any of the given files.
    pub fn modifications_for_files(&self, files: &[&str]) -> Vec<&ModificationRecord> {
        let set: HashSet<&&str> = files.iter().collect();
        self.modifications
            .iter()
            .filter(|r| r.affected_files.iter().any(|f| set.contains(&f.as_str())))
            .collect()
    }

    /// Return all failed modifications.
    pub fn failed_modifications(&self) -> Vec<&ModificationRecord> {
        self.modifications
            .iter()
            .filter(|r| !r.was_successful())
            .collect()
    }

    /// Success rate over all recorded modifications (0.0–1.0).
    /// Returns `None` if no modifications have been recorded.
    pub fn success_rate(&self) -> Option<f64> {
        if self.modifications.is_empty() {
            return None;
        }
        let successes = self
            .modifications
            .iter()
            .filter(|r| r.was_successful())
            .count();
        Some(successes as f64 / self.modifications.len() as f64)
    }

    // -----------------------------------------------------------------------
    // Code patterns
    // -----------------------------------------------------------------------

    /// Store a code pattern.  Overwrites any existing pattern with the same ID.
    pub fn store_pattern(&mut self, pattern: CodePattern) {
        if self.patterns.len() >= self.config.max_patterns
            && !self.patterns.contains_key(&pattern.id)
        {
            // Evict least-used pattern
            if let Some(lru_id) = self
                .patterns
                .iter()
                .min_by_key(|(_, p)| p.use_count)
                .map(|(id, _)| id.clone())
            {
                self.patterns.remove(&lru_id);
            }
        }
        self.patterns.insert(pattern.id.clone(), pattern);
    }

    /// Look up a pattern by ID.
    pub fn get_pattern(&self, id: &str) -> Option<&CodePattern> {
        self.patterns.get(id)
    }

    /// Return all patterns matching all provided tags.
    pub fn patterns_by_tags(&self, tags: &[&str]) -> Vec<&CodePattern> {
        self.patterns
            .values()
            .filter(|p| p.matches_all_tags(tags))
            .collect()
    }

    /// Increment the use count for a pattern.  Returns `true` if found.
    pub fn record_pattern_use(&mut self, id: &str) -> bool {
        if let Some(p) = self.patterns.get_mut(id) {
            p.use_count = p.use_count.saturating_add(1);
            true
        } else {
            false
        }
    }

    // -----------------------------------------------------------------------
    // Performance baselines
    // -----------------------------------------------------------------------

    /// Update (or insert) a performance baseline.
    pub fn update_baseline(&mut self, baseline: PerformanceBaseline) {
        if self.baselines.len() >= self.config.max_baselines
            && !self.baselines.contains_key(&baseline.metric)
        {
            // Evict oldest (by updated_at_ms)
            if let Some(oldest) = self
                .baselines
                .iter()
                .min_by_key(|(_, b)| b.updated_at_ms)
                .map(|(k, _)| k.clone())
            {
                self.baselines.remove(&oldest);
            }
        }
        self.baselines.insert(baseline.metric.clone(), baseline);
    }

    /// Get the baseline for a named metric.
    pub fn get_baseline(&self, metric: &str) -> Option<&PerformanceBaseline> {
        self.baselines.get(metric)
    }

    /// Return all baselines.
    pub fn baselines(&self) -> impl Iterator<Item = &PerformanceBaseline> {
        self.baselines.values()
    }

    /// Update a baseline using a new sample via Welford's online algorithm.
    /// If no baseline exists for the metric, a new one is created.
    pub fn incorporate_sample(
        &mut self,
        metric: &str,
        value: f64,
        p50: f64,
        p95: f64,
        p99: f64,
        now_ms: u64,
    ) {
        let entry = self.baselines.entry(metric.to_string()).or_insert_with(|| {
            PerformanceBaseline {
                metric: metric.to_string(),
                p50,
                p95,
                p99,
                mean: value,
                std_dev: 0.0,
                sample_count: 0,
                updated_at_ms: now_ms,
            }
        });

        // Welford online update
        entry.sample_count += 1;
        let n = entry.sample_count as f64;
        let delta = value - entry.mean;
        entry.mean += delta / n;
        let delta2 = value - entry.mean;
        // Accumulate M2 in std_dev field temporarily (recomputed below)
        let m2_prev = if entry.sample_count > 1 {
            entry.std_dev * entry.std_dev * (n - 2.0) / (n - 1.0) * (n - 1.0)
        } else {
            0.0
        };
        let m2 = m2_prev + delta * delta2;
        entry.std_dev = if entry.sample_count > 1 {
            (m2 / (n - 1.0)).sqrt()
        } else {
            0.0
        };
        entry.p50 = p50;
        entry.p95 = p95;
        entry.p99 = p99;
        entry.updated_at_ms = now_ms;
    }

    // -----------------------------------------------------------------------
    // Dead ends
    // -----------------------------------------------------------------------

    /// Record a dead-end approach.
    pub fn record_dead_end(&mut self, dead_end: DeadEnd) {
        if self.dead_ends.len() >= self.config.max_dead_ends
            && !self.dead_ends.contains_key(&dead_end.key)
        {
            // Evict oldest
            if let Some(oldest) = self
                .dead_ends
                .iter()
                .min_by_key(|(_, d)| d.recorded_at_ms)
                .map(|(k, _)| k.clone())
            {
                self.dead_ends.remove(&oldest);
            }
        }
        self.dead_ends.insert(dead_end.key.clone(), dead_end);
    }

    /// Returns `true` if the given approach key is a known dead end.
    pub fn is_dead_end(&self, key: &str) -> bool {
        self.dead_ends.contains_key(key)
    }

    /// Return dead ends related to any of the given signals.
    pub fn dead_ends_for_signals(&self, signals: &[&str]) -> Vec<&DeadEnd> {
        let signal_set: HashSet<&&str> = signals.iter().collect();
        self.dead_ends
            .values()
            .filter(|d| {
                d.related_signals
                    .iter()
                    .any(|s| signal_set.contains(&s.as_str()))
            })
            .collect()
    }

    /// Remove a dead end (e.g. after the underlying issue is resolved).
    pub fn remove_dead_end(&mut self, key: &str) -> bool {
        self.dead_ends.remove(key).is_some()
    }

    // -----------------------------------------------------------------------
    // Stats
    // -----------------------------------------------------------------------

    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            modification_count: self.modifications.len(),
            pattern_count: self.patterns.len(),
            baseline_count: self.baselines.len(),
            dead_end_count: self.dead_ends.len(),
            success_rate: self.success_rate().unwrap_or(0.0),
        }
    }

    /// Prune all modification records older than `max_age` (by timestamp_ms).
    pub fn prune_old_modifications(&mut self, cutoff_ms: u64) {
        self.modifications.retain(|r| r.timestamp_ms >= cutoff_ms);
    }

    /// Export all data as a JSON value (for MCP tool / dashboard).
    pub fn export_json(&self) -> serde_json::Value {
        let stats = self.stats();
        serde_json::json!({
            "stats": {
                "modifications": stats.modification_count,
                "patterns": stats.pattern_count,
                "baselines": stats.baseline_count,
                "dead_ends": stats.dead_end_count,
                "success_rate": stats.success_rate,
            },
            "recent_modifications": self.recent_modifications(10).iter().map(|r| serde_json::json!({
                "id": r.id,
                "description": r.description,
                "outcome": r.outcome.to_string(),
                "notes": r.notes,
                "timestamp_ms": r.timestamp_ms,
            })).collect::<Vec<_>>(),
            "dead_ends": self.dead_ends.values().map(|d| serde_json::json!({
                "key": d.key,
                "reason": d.reason,
            })).collect::<Vec<_>>(),
        })
    }
}

// ---------------------------------------------------------------------------
// MemoryBackend trait (for future Redis swap-in)
// ---------------------------------------------------------------------------

/// Abstraction over the storage backend.  The in-memory `AgentMemory` struct
/// satisfies this interface; a `RedisMemory` could be substituted at runtime.
pub trait MemoryBackend: Send + Sync {
    fn record_modification(&mut self, record: ModificationRecord) -> String;
    fn get_modification(&self, id: &str) -> Option<ModificationRecord>;
    fn recent_modifications(&self, n: usize) -> Vec<ModificationRecord>;
    fn success_rate(&self) -> Option<f64>;

    fn store_pattern(&mut self, pattern: CodePattern);
    fn get_pattern(&self, id: &str) -> Option<CodePattern>;

    fn update_baseline(&mut self, baseline: PerformanceBaseline);
    fn get_baseline(&self, metric: &str) -> Option<PerformanceBaseline>;

    fn record_dead_end(&mut self, dead_end: DeadEnd);
    fn is_dead_end(&self, key: &str) -> bool;

    /// How long a round-trip to the backend typically takes.
    fn estimated_latency(&self) -> Duration;
}

/// In-memory backend (wraps `AgentMemory`).
pub struct InMemoryBackend {
    inner: AgentMemory,
}

impl InMemoryBackend {
    pub fn new(config: MemoryConfig) -> Self {
        Self { inner: AgentMemory::new(config) }
    }

    pub fn inner(&self) -> &AgentMemory {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut AgentMemory {
        &mut self.inner
    }
}

impl MemoryBackend for InMemoryBackend {
    fn record_modification(&mut self, record: ModificationRecord) -> String {
        self.inner.record_modification(record)
    }

    fn get_modification(&self, id: &str) -> Option<ModificationRecord> {
        self.inner.get_modification(id).cloned()
    }

    fn recent_modifications(&self, n: usize) -> Vec<ModificationRecord> {
        self.inner.recent_modifications(n).into_iter().cloned().collect()
    }

    fn success_rate(&self) -> Option<f64> {
        self.inner.success_rate()
    }

    fn store_pattern(&mut self, pattern: CodePattern) {
        self.inner.store_pattern(pattern)
    }

    fn get_pattern(&self, id: &str) -> Option<CodePattern> {
        self.inner.get_pattern(id).cloned()
    }

    fn update_baseline(&mut self, baseline: PerformanceBaseline) {
        self.inner.update_baseline(baseline)
    }

    fn get_baseline(&self, metric: &str) -> Option<PerformanceBaseline> {
        self.inner.get_baseline(metric).cloned()
    }

    fn record_dead_end(&mut self, dead_end: DeadEnd) {
        self.inner.record_dead_end(dead_end)
    }

    fn is_dead_end(&self, key: &str) -> bool {
        self.inner.is_dead_end(key)
    }

    fn estimated_latency(&self) -> Duration {
        Duration::from_micros(1)
    }
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

#[cfg(test)]
pub fn make_record(id: &str, outcome: Outcome) -> ModificationRecord {
    ModificationRecord {
        id: id.to_string(),
        description: format!("modification {}", id),
        affected_files: vec!["src/foo.rs".to_string()],
        outcome,
        metric_deltas: HashMap::new(),
        notes: String::new(),
        timestamp_ms: 1000,
    }
}

#[cfg(test)]
pub fn make_pattern(id: &str, tags: &[&str]) -> CodePattern {
    CodePattern {
        id: id.to_string(),
        title: format!("Pattern {}", id),
        content: "// example pattern".to_string(),
        tags: tags.iter().map(|s| s.to_string()).collect(),
        use_count: 0,
        created_at_ms: 0,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn mem() -> AgentMemory {
        AgentMemory::new(MemoryConfig::default())
    }

    fn small_mem() -> AgentMemory {
        AgentMemory::new(MemoryConfig {
            max_modifications: 3,
            max_patterns: 3,
            max_dead_ends: 3,
            max_baselines: 3,
        })
    }

    // -------------------------------------------------------------------
    // Outcome display
    // -------------------------------------------------------------------

    #[test]
    fn test_outcome_display_success() { assert_eq!(Outcome::Success.to_string(), "success"); }
    #[test]
    fn test_outcome_display_failure() { assert_eq!(Outcome::Failure.to_string(), "failure"); }
    #[test]
    fn test_outcome_display_rolled_back() { assert_eq!(Outcome::RolledBack.to_string(), "rolled_back"); }
    #[test]
    fn test_outcome_display_pending() { assert_eq!(Outcome::Pending.to_string(), "pending"); }

    // -------------------------------------------------------------------
    // Modification records
    // -------------------------------------------------------------------

    #[test]
    fn test_record_modification_stores_record() {
        let mut m = mem();
        m.record_modification(make_record("r1", Outcome::Success));
        assert!(m.get_modification("r1").is_some());
    }

    #[test]
    fn test_get_modification_unknown_returns_none() {
        let m = mem();
        assert!(m.get_modification("nope").is_none());
    }

    #[test]
    fn test_success_rate_all_success() {
        let mut m = mem();
        m.record_modification(make_record("a", Outcome::Success));
        m.record_modification(make_record("b", Outcome::Success));
        assert!((m.success_rate().unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_success_rate_mixed() {
        let mut m = mem();
        m.record_modification(make_record("a", Outcome::Success));
        m.record_modification(make_record("b", Outcome::Failure));
        assert!((m.success_rate().unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_success_rate_empty_is_none() {
        let m = mem();
        assert!(m.success_rate().is_none());
    }

    #[test]
    fn test_modifications_evict_oldest_when_full() {
        let mut m = small_mem();
        m.record_modification(make_record("r1", Outcome::Success));
        m.record_modification(make_record("r2", Outcome::Success));
        m.record_modification(make_record("r3", Outcome::Success));
        m.record_modification(make_record("r4", Outcome::Success)); // evicts r1
        assert!(m.get_modification("r1").is_none());
        assert!(m.get_modification("r4").is_some());
    }

    #[test]
    fn test_recent_modifications_returns_n_newest() {
        let mut m = mem();
        for i in 0..5 {
            m.record_modification(make_record(&i.to_string(), Outcome::Success));
        }
        let recent = m.recent_modifications(3);
        assert_eq!(recent.len(), 3);
        // Should be 4, 3, 2 (newest first)
        assert_eq!(recent[0].id, "4");
    }

    #[test]
    fn test_modifications_for_files_returns_matching() {
        let mut m = mem();
        let mut r = make_record("r1", Outcome::Success);
        r.affected_files = vec!["src/router.rs".to_string()];
        m.record_modification(r);
        let results = m.modifications_for_files(&["src/router.rs"]);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_modifications_for_files_no_match() {
        let mut m = mem();
        m.record_modification(make_record("r1", Outcome::Success));
        let results = m.modifications_for_files(&["src/other.rs"]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_failed_modifications_filters_correctly() {
        let mut m = mem();
        m.record_modification(make_record("ok", Outcome::Success));
        m.record_modification(make_record("fail", Outcome::Failure));
        m.record_modification(make_record("rb", Outcome::RolledBack));
        let failed = m.failed_modifications();
        assert_eq!(failed.len(), 2);
        assert!(failed.iter().all(|r| !r.was_successful()));
    }

    #[test]
    fn test_prune_removes_old_records() {
        let mut m = mem();
        let mut r1 = make_record("old", Outcome::Success);
        r1.timestamp_ms = 100;
        let mut r2 = make_record("new", Outcome::Success);
        r2.timestamp_ms = 1000;
        m.record_modification(r1);
        m.record_modification(r2);
        m.prune_old_modifications(500);
        assert!(m.get_modification("old").is_none());
        assert!(m.get_modification("new").is_some());
    }

    // -------------------------------------------------------------------
    // Code patterns
    // -------------------------------------------------------------------

    #[test]
    fn test_store_and_retrieve_pattern() {
        let mut m = mem();
        m.store_pattern(make_pattern("p1", &["async", "backpressure"]));
        let p = m.get_pattern("p1").unwrap();
        assert_eq!(p.id, "p1");
    }

    #[test]
    fn test_get_unknown_pattern_returns_none() {
        let m = mem();
        assert!(m.get_pattern("missing").is_none());
    }

    #[test]
    fn test_patterns_by_tags_all_match() {
        let mut m = mem();
        m.store_pattern(make_pattern("p1", &["async", "channel"]));
        m.store_pattern(make_pattern("p2", &["async", "retry"]));
        let results = m.patterns_by_tags(&["async"]);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_patterns_by_tags_partial_match_excluded() {
        let mut m = mem();
        m.store_pattern(make_pattern("p1", &["async"]));
        m.store_pattern(make_pattern("p2", &["sync"]));
        let results = m.patterns_by_tags(&["async", "sync"]);
        assert!(results.is_empty(), "must match ALL tags");
    }

    #[test]
    fn test_record_pattern_use_increments_count() {
        let mut m = mem();
        m.store_pattern(make_pattern("p1", &[]));
        m.record_pattern_use("p1");
        m.record_pattern_use("p1");
        assert_eq!(m.get_pattern("p1").unwrap().use_count, 2);
    }

    #[test]
    fn test_record_pattern_use_unknown_returns_false() {
        let mut m = mem();
        assert!(!m.record_pattern_use("ghost"));
    }

    #[test]
    fn test_pattern_evicts_least_used_when_full() {
        let mut m = small_mem(); // max_patterns = 3
        m.store_pattern(make_pattern("p1", &[]));
        m.store_pattern(make_pattern("p2", &[]));
        m.store_pattern(make_pattern("p3", &[]));
        // p1 has use_count=0 — it's the LRU
        m.record_pattern_use("p2");
        m.record_pattern_use("p3");
        m.store_pattern(make_pattern("p4", &[])); // should evict p1
        assert!(m.get_pattern("p1").is_none());
        assert!(m.get_pattern("p4").is_some());
    }

    // -------------------------------------------------------------------
    // Performance baselines
    // -------------------------------------------------------------------

    #[test]
    fn test_update_and_get_baseline() {
        let mut m = mem();
        m.update_baseline(PerformanceBaseline {
            metric: "p95_latency_ms".to_string(),
            p50: 20.0, p95: 50.0, p99: 80.0,
            mean: 30.0, std_dev: 5.0,
            sample_count: 100, updated_at_ms: 0,
        });
        let b = m.get_baseline("p95_latency_ms").unwrap();
        assert!((b.p95 - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_get_unknown_baseline_returns_none() {
        let m = mem();
        assert!(m.get_baseline("nonexistent").is_none());
    }

    #[test]
    fn test_baseline_is_normal_within_sigma() {
        let b = PerformanceBaseline {
            metric: "x".into(), p50: 0.0, p95: 0.0, p99: 0.0,
            mean: 100.0, std_dev: 10.0, sample_count: 50, updated_at_ms: 0,
        };
        assert!(b.is_normal(105.0, 2.0)); // 0.5σ
        assert!(!b.is_normal(130.0, 2.0)); // 3σ
    }

    #[test]
    fn test_baseline_deviation_fraction() {
        let b = PerformanceBaseline {
            metric: "x".into(), p50: 0.0, p95: 0.0, p99: 0.0,
            mean: 100.0, std_dev: 10.0, sample_count: 50, updated_at_ms: 0,
        };
        assert!((b.deviation_fraction(110.0) - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_incorporate_sample_updates_mean() {
        let mut m = mem();
        m.incorporate_sample("lat", 100.0, 80.0, 100.0, 120.0, 0);
        m.incorporate_sample("lat", 200.0, 80.0, 100.0, 120.0, 1);
        let b = m.get_baseline("lat").unwrap();
        assert!((b.mean - 150.0).abs() < 1e-6);
    }

    #[test]
    fn test_incorporate_sample_count_increments() {
        let mut m = mem();
        m.incorporate_sample("lat", 100.0, 0.0, 0.0, 0.0, 0);
        m.incorporate_sample("lat", 100.0, 0.0, 0.0, 0.0, 1);
        m.incorporate_sample("lat", 100.0, 0.0, 0.0, 0.0, 2);
        assert_eq!(m.get_baseline("lat").unwrap().sample_count, 3);
    }

    // -------------------------------------------------------------------
    // Dead ends
    // -------------------------------------------------------------------

    #[test]
    fn test_record_and_check_dead_end() {
        let mut m = mem();
        m.record_dead_end(DeadEnd {
            key: "increase-dedup-ttl".into(),
            description: "tried longer TTL".into(),
            reason: "caused memory explosion".into(),
            related_signals: vec!["anomaly:dedup_hit_rate:Warn".into()],
            recorded_at_ms: 0,
        });
        assert!(m.is_dead_end("increase-dedup-ttl"));
        assert!(!m.is_dead_end("something-else"));
    }

    #[test]
    fn test_remove_dead_end() {
        let mut m = mem();
        m.record_dead_end(DeadEnd {
            key: "bad-approach".into(),
            description: "".into(),
            reason: "".into(),
            related_signals: vec![],
            recorded_at_ms: 0,
        });
        assert!(m.remove_dead_end("bad-approach"));
        assert!(!m.is_dead_end("bad-approach"));
    }

    #[test]
    fn test_remove_unknown_dead_end_returns_false() {
        let mut m = mem();
        assert!(!m.remove_dead_end("ghost"));
    }

    #[test]
    fn test_dead_ends_for_signals_returns_matching() {
        let mut m = mem();
        m.record_dead_end(DeadEnd {
            key: "approach-a".into(),
            description: "".into(),
            reason: "failed".into(),
            related_signals: vec!["anomaly:lat:Critical".into()],
            recorded_at_ms: 0,
        });
        let results = m.dead_ends_for_signals(&["anomaly:lat:Critical"]);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_dead_ends_for_signals_no_match() {
        let mut m = mem();
        m.record_dead_end(DeadEnd {
            key: "approach-a".into(),
            description: "".into(),
            reason: "".into(),
            related_signals: vec!["other:signal".into()],
            recorded_at_ms: 0,
        });
        let results = m.dead_ends_for_signals(&["anomaly:lat:Critical"]);
        assert!(results.is_empty());
    }

    // -------------------------------------------------------------------
    // Stats
    // -------------------------------------------------------------------

    #[test]
    fn test_stats_reflects_counts() {
        let mut m = mem();
        m.record_modification(make_record("r1", Outcome::Success));
        m.store_pattern(make_pattern("p1", &[]));
        m.update_baseline(PerformanceBaseline {
            metric: "x".into(), p50: 0.0, p95: 0.0, p99: 0.0,
            mean: 1.0, std_dev: 0.0, sample_count: 1, updated_at_ms: 0,
        });
        m.record_dead_end(DeadEnd {
            key: "d".into(), description: "".into(), reason: "".into(),
            related_signals: vec![], recorded_at_ms: 0,
        });
        let s = m.stats();
        assert_eq!(s.modification_count, 1);
        assert_eq!(s.pattern_count, 1);
        assert_eq!(s.baseline_count, 1);
        assert_eq!(s.dead_end_count, 1);
    }

    // -------------------------------------------------------------------
    // InMemoryBackend (trait impl)
    // -------------------------------------------------------------------

    #[test]
    fn test_in_memory_backend_round_trips_modification() {
        let mut b = InMemoryBackend::new(MemoryConfig::default());
        b.record_modification(make_record("x", Outcome::Success));
        let r = b.get_modification("x").unwrap();
        assert_eq!(r.id, "x");
    }

    #[test]
    fn test_in_memory_backend_latency_is_sub_ms() {
        let b = InMemoryBackend::new(MemoryConfig::default());
        assert!(b.estimated_latency() < Duration::from_millis(1));
    }

    #[test]
    fn test_in_memory_backend_is_dead_end() {
        let mut b = InMemoryBackend::new(MemoryConfig::default());
        b.record_dead_end(DeadEnd {
            key: "k".into(), description: "".into(), reason: "".into(),
            related_signals: vec![], recorded_at_ms: 0,
        });
        assert!(b.is_dead_end("k"));
    }

    // -------------------------------------------------------------------
    // Export JSON
    // -------------------------------------------------------------------

    #[test]
    fn test_export_json_contains_stats() {
        let mut m = mem();
        m.record_modification(make_record("r1", Outcome::Success));
        let json = m.export_json();
        assert!(json["stats"]["modifications"].as_u64().unwrap() > 0);
    }

    #[test]
    fn test_export_json_recent_modifications_capped_at_10() {
        let mut m = mem();
        for i in 0..15 {
            m.record_modification(make_record(&i.to_string(), Outcome::Success));
        }
        let json = m.export_json();
        assert!(json["recent_modifications"].as_array().unwrap().len() <= 10);
    }

    #[test]
    fn test_modification_record_was_successful() {
        let r = make_record("x", Outcome::Success);
        assert!(r.was_successful());
        let r2 = make_record("y", Outcome::Failure);
        assert!(!r2.was_successful());
    }
}
