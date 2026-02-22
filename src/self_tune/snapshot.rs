//! # Stage: Configuration Snapshotter (Task 1.6)
//!
//! ## Responsibility
//! Git-like versioning for the pipeline configuration.  Every time any tunable
//! parameter changes — whether via PID controller, manual override, or rollback —
//! a snapshot is created capturing the full parameter map, the change delta, the
//! source of the change, and the telemetry metrics active at that moment.
//!
//! The registry supports `rollback_to(id)`, `diff(a, b)`, and
//! `best_in_window(metric, since)` queries.  It also exposes the two MCP tools
//! `config_history` and `config_rollback` referenced in the task description.
//!
//! ## Guarantees
//! - Bounded: history is capped at `capacity` entries (oldest evicted first)
//! - Non-panicking: all indexing is bounds-checked; all division is guarded
//! - Thread-safe: designed to be wrapped in `Arc<Mutex<SnapshotRegistry>>`
//!
//! ## NOT Responsible For
//! - Persisting snapshots to disk or Redis (that is the agent memory system, 2.3)
//! - Applying parameter changes to the live pipeline (that is the controller, 1.2)

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Parameter map — mirrors Param enum from controller but as string keys so
// the snapshotter does not need to import the controller module.
// ---------------------------------------------------------------------------

/// A single parameter name / value pair.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamEntry {
    pub name: String,
    pub value: f64,
}

/// The complete set of pipeline parameters captured at one instant.
pub type ParamMap = HashMap<String, f64>;

// ---------------------------------------------------------------------------
// ChangeSource — who or what triggered this snapshot
// ---------------------------------------------------------------------------

/// Identifies what triggered this configuration change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeSource {
    /// PID controller automatic adjustment.
    Controller,
    /// Experiment engine promoting a winner.
    Experiment { experiment_name: String },
    /// Anomaly-triggered rollback.
    AnomalyRollback,
    /// Manual override from a human operator or MCP tool.
    Manual { operator: String },
    /// Auto-rollback because a metric degraded past threshold.
    AutoRollback { degraded_metric: String },
    /// System startup / initial state snapshot.
    Initial,
}

impl std::fmt::Display for ChangeSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChangeSource::Controller => write!(f, "controller"),
            ChangeSource::Experiment { experiment_name } => {
                write!(f, "experiment:{}", experiment_name)
            }
            ChangeSource::AnomalyRollback => write!(f, "anomaly-rollback"),
            ChangeSource::Manual { operator } => write!(f, "manual:{}", operator),
            ChangeSource::AutoRollback { degraded_metric } => {
                write!(f, "auto-rollback:{}", degraded_metric)
            }
            ChangeSource::Initial => write!(f, "initial"),
        }
    }
}

// ---------------------------------------------------------------------------
// SnapshotMetrics — lightweight copy of the most important telemetry values
// ---------------------------------------------------------------------------

/// Key metrics recorded alongside a snapshot.
#[derive(Debug, Clone, Default)]
pub struct SnapshotMetrics {
    pub p95_latency_ms: f64,
    pub drop_rate_pct: f64,
    pub cache_hit_rate: f64,
    pub error_rate: f64,
    pub throughput_rps: f64,
    /// Arbitrary extra metrics — e.g. cost, quality score.
    pub extras: HashMap<String, f64>,
}

impl SnapshotMetrics {
    /// Return the value of a named metric.  Returns `None` if unknown.
    pub fn get(&self, name: &str) -> Option<f64> {
        match name {
            "p95_latency_ms" => Some(self.p95_latency_ms),
            "drop_rate_pct" => Some(self.drop_rate_pct),
            "cache_hit_rate" => Some(self.cache_hit_rate),
            "error_rate" => Some(self.error_rate),
            "throughput_rps" => Some(self.throughput_rps),
            other => self.extras.get(other).copied(),
        }
    }
}

// ---------------------------------------------------------------------------
// ConfigDiff — difference between two snapshots
// ---------------------------------------------------------------------------

/// Describes what changed between two consecutive snapshots.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamDiff {
    pub name: String,
    pub before: f64,
    pub after: f64,
}

impl ParamDiff {
    pub fn delta(&self) -> f64 {
        self.after - self.before
    }

    pub fn pct_change(&self) -> f64 {
        if self.before == 0.0 {
            0.0
        } else {
            (self.after - self.before) / self.before.abs() * 100.0
        }
    }
}

/// Full diff between two snapshots.
#[derive(Debug, Clone)]
pub struct ConfigDiff {
    pub from_id: u64,
    pub to_id: u64,
    pub changes: Vec<ParamDiff>,
}

impl ConfigDiff {
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }
}

// ---------------------------------------------------------------------------
// ConfigSnapshot — one immutable point-in-time record
// ---------------------------------------------------------------------------

/// An immutable snapshot of the full pipeline configuration at one instant.
#[derive(Debug, Clone)]
pub struct ConfigSnapshot {
    /// Monotonically increasing identifier (1-based).
    pub id: u64,
    /// Unix timestamp (milliseconds) when the snapshot was taken.
    pub timestamp_ms: u64,
    /// Source of the change that produced this snapshot.
    pub source: ChangeSource,
    /// Full parameter map at this instant.
    pub params: ParamMap,
    /// Telemetry metrics active at this instant.
    pub metrics: SnapshotMetrics,
    /// Human-readable note (optional).
    pub note: Option<String>,
}

impl ConfigSnapshot {
    /// Compute the diff between `self` (the "before") and `other` (the "after").
    pub fn diff_to(&self, other: &ConfigSnapshot) -> ConfigDiff {
        let mut changes = Vec::new();

        // Parameters that exist in both
        for (name, &after) in &other.params {
            let before = self.params.get(name).copied().unwrap_or(0.0);
            if (after - before).abs() > f64::EPSILON {
                changes.push(ParamDiff {
                    name: name.clone(),
                    before,
                    after,
                });
            }
        }

        // Parameters removed in other (present in self but not in other)
        for (name, &before) in &self.params {
            if !other.params.contains_key(name) {
                changes.push(ParamDiff {
                    name: name.clone(),
                    before,
                    after: 0.0,
                });
            }
        }

        changes.sort_by(|a, b| a.name.cmp(&b.name));

        ConfigDiff {
            from_id: self.id,
            to_id: other.id,
            changes,
        }
    }
}

// ---------------------------------------------------------------------------
// SnapshotRegistry
// ---------------------------------------------------------------------------

/// Git-like history of configuration snapshots.
///
/// # Example
/// ```rust
/// use every_other_token::self_tune::snapshot::{
///     SnapshotRegistry, ChangeSource, SnapshotMetrics,
/// };
/// use std::collections::HashMap;
///
/// let mut reg = SnapshotRegistry::new(100);
/// let params = HashMap::from([("buf_dedup".to_string(), 128.0)]);
/// let id = reg.commit(params, ChangeSource::Initial, SnapshotMetrics::default(), None);
/// assert_eq!(id, 1);
/// ```
pub struct SnapshotRegistry {
    /// Ring buffer of snapshots (oldest at front).
    history: std::collections::VecDeque<ConfigSnapshot>,
    capacity: usize,
    next_id: u64,
    /// Simulated monotonic clock in ms — incremented by `tick_ms` on each commit.
    /// In production callers supply real timestamps; this default is for tests.
    clock_ms: u64,
}

impl SnapshotRegistry {
    /// Create a new registry with the given maximum history length.
    pub fn new(capacity: usize) -> Self {
        Self {
            history: std::collections::VecDeque::with_capacity(capacity.min(256)),
            capacity: capacity.max(1),
            next_id: 1,
            clock_ms: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn now_ms(&mut self) -> u64 {
        self.clock_ms += 1;
        self.clock_ms
    }

    fn evict_if_full(&mut self) {
        while self.history.len() >= self.capacity {
            self.history.pop_front();
        }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Record a new configuration snapshot.  Returns the snapshot ID.
    pub fn commit(
        &mut self,
        params: ParamMap,
        source: ChangeSource,
        metrics: SnapshotMetrics,
        note: Option<String>,
    ) -> u64 {
        self.evict_if_full();
        let id = self.next_id;
        self.next_id += 1;
        let timestamp_ms = self.now_ms();
        self.history.push_back(ConfigSnapshot {
            id,
            timestamp_ms,
            source,
            params,
            metrics,
            note,
        });
        id
    }

    /// Commit with an explicit timestamp (production use — callers supply real wall-clock ms).
    pub fn commit_at(
        &mut self,
        params: ParamMap,
        source: ChangeSource,
        metrics: SnapshotMetrics,
        note: Option<String>,
        timestamp_ms: u64,
    ) -> u64 {
        self.evict_if_full();
        let id = self.next_id;
        self.next_id += 1;
        self.history.push_back(ConfigSnapshot {
            id,
            timestamp_ms,
            source,
            params,
            metrics,
            note,
        });
        id
    }

    /// Return the total number of snapshots currently in history.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Return `true` if the history is empty.
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Look up a snapshot by ID.
    pub fn get(&self, id: u64) -> Option<&ConfigSnapshot> {
        // Linear scan — history is small (≤capacity).
        self.history.iter().find(|s| s.id == id)
    }

    /// Return the most recent snapshot.
    pub fn latest(&self) -> Option<&ConfigSnapshot> {
        self.history.back()
    }

    /// Return all snapshots in chronological order (oldest first).
    pub fn all(&self) -> impl Iterator<Item = &ConfigSnapshot> {
        self.history.iter()
    }

    /// Return snapshots since `timestamp_ms` (inclusive), chronological.
    pub fn since(&self, timestamp_ms: u64) -> impl Iterator<Item = &ConfigSnapshot> {
        self.history.iter().filter(move |s| s.timestamp_ms >= timestamp_ms)
    }

    /// Return snapshots from the last `window_ms` milliseconds.
    pub fn last_window(&self, window_ms: u64) -> impl Iterator<Item = &ConfigSnapshot> {
        let cutoff = self.clock_ms.saturating_sub(window_ms);
        self.history.iter().filter(move |s| s.timestamp_ms > cutoff)
    }

    // -----------------------------------------------------------------------
    // Diff
    // -----------------------------------------------------------------------

    /// Compute the parameter diff between two snapshots identified by ID.
    /// Returns `None` if either ID is not found.
    pub fn diff(&self, from_id: u64, to_id: u64) -> Option<ConfigDiff> {
        let from = self.get(from_id)?;
        let to = self.get(to_id)?;
        Some(from.diff_to(to))
    }

    // -----------------------------------------------------------------------
    // Rollback
    // -----------------------------------------------------------------------

    /// Create a new snapshot whose parameter map is copied from the snapshot
    /// identified by `target_id`.  This does NOT directly apply changes to
    /// the live pipeline; callers must read the returned snapshot and forward
    /// the parameters to the controller.
    ///
    /// Returns the new snapshot's ID, or an error string if `target_id` is
    /// not found.
    pub fn rollback_to(
        &mut self,
        target_id: u64,
        triggered_by: ChangeSource,
        current_metrics: SnapshotMetrics,
    ) -> Result<u64, String> {
        // Clone params out before the mutable borrow for commit.
        let params = self
            .get(target_id)
            .map(|s| s.params.clone())
            .ok_or_else(|| format!("snapshot {} not found", target_id))?;

        let note = Some(format!("rollback to snapshot {}", target_id));
        Ok(self.commit(params, triggered_by, current_metrics, note))
    }

    // -----------------------------------------------------------------------
    // Best-in-window
    // -----------------------------------------------------------------------

    /// Return the snapshot within the last `window_ms` milliseconds that
    /// minimises the given metric (lower-is-better: latency, error_rate,
    /// drop_rate_pct) or maximises it (higher-is-better: cache_hit_rate,
    /// throughput_rps).
    ///
    /// `higher_is_better` controls the direction.
    ///
    /// Returns `None` if the window is empty or the metric is unknown for all
    /// snapshots in the window.
    pub fn best_in_window(
        &self,
        metric: &str,
        window_ms: u64,
        higher_is_better: bool,
    ) -> Option<&ConfigSnapshot> {
        let cutoff = self.clock_ms.saturating_sub(window_ms);
        let candidates: Vec<&ConfigSnapshot> = self
            .history
            .iter()
            .filter(|s| s.timestamp_ms > cutoff)
            .filter(|s| s.metrics.get(metric).is_some())
            .collect();

        candidates.into_iter().max_by(|a, b| {
            let va = a.metrics.get(metric).unwrap_or(0.0);
            let vb = b.metrics.get(metric).unwrap_or(0.0);
            if higher_is_better {
                va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                // Lower is better → invert comparison
                vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
            }
        })
    }

    // -----------------------------------------------------------------------
    // MCP tool helpers
    // -----------------------------------------------------------------------

    /// Serialise the history as a JSON-compatible `Vec` of summary records.
    /// Each entry contains: id, timestamp_ms, source (string), note, and a
    /// flat map of changed parameters relative to the previous snapshot.
    pub fn config_history_json(&self) -> Vec<serde_json::Value> {
        let snapshots: Vec<&ConfigSnapshot> = self.history.iter().collect();
        snapshots
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let diff_changes: Vec<serde_json::Value> = if i == 0 {
                    vec![]
                } else {
                    let prev = snapshots[i - 1];
                    prev.diff_to(s)
                        .changes
                        .into_iter()
                        .map(|d| {
                            serde_json::json!({
                                "param": d.name,
                                "before": d.before,
                                "after": d.after,
                                "delta": d.delta(),
                            })
                        })
                        .collect()
                };
                serde_json::json!({
                    "id": s.id,
                    "timestamp_ms": s.timestamp_ms,
                    "source": s.source.to_string(),
                    "note": s.note,
                    "changes": diff_changes,
                    "metrics": {
                        "p95_latency_ms": s.metrics.p95_latency_ms,
                        "drop_rate_pct": s.metrics.drop_rate_pct,
                        "cache_hit_rate": s.metrics.cache_hit_rate,
                        "error_rate": s.metrics.error_rate,
                        "throughput_rps": s.metrics.throughput_rps,
                    }
                })
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn base_params() -> ParamMap {
        let mut m = HashMap::new();
        m.insert("buf_dedup".to_string(), 128.0);
        m.insert("buf_cache".to_string(), 256.0);
        m.insert("rate_limit".to_string(), 100.0);
        m
    }

    fn changed_params() -> ParamMap {
        let mut m = base_params();
        m.insert("buf_dedup".to_string(), 192.0); // changed
        m.insert("priority_interval_ms".to_string(), 50.0); // new
        m
    }

    fn metrics(p95: f64, err: f64) -> SnapshotMetrics {
        SnapshotMetrics {
            p95_latency_ms: p95,
            error_rate: err,
            ..Default::default()
        }
    }

    // -----------------------------------------------------------------------
    // Backend identity
    // -----------------------------------------------------------------------

    #[test]
    fn test_change_source_display_controller() {
        assert_eq!(ChangeSource::Controller.to_string(), "controller");
    }

    #[test]
    fn test_change_source_display_experiment() {
        let src = ChangeSource::Experiment {
            experiment_name: "exp_a".to_string(),
        };
        assert_eq!(src.to_string(), "experiment:exp_a");
    }

    #[test]
    fn test_change_source_display_manual() {
        let src = ChangeSource::Manual {
            operator: "alice".to_string(),
        };
        assert_eq!(src.to_string(), "manual:alice");
    }

    #[test]
    fn test_change_source_display_auto_rollback() {
        let src = ChangeSource::AutoRollback {
            degraded_metric: "p95_latency_ms".to_string(),
        };
        assert_eq!(src.to_string(), "auto-rollback:p95_latency_ms");
    }

    #[test]
    fn test_change_source_display_initial() {
        assert_eq!(ChangeSource::Initial.to_string(), "initial");
    }

    #[test]
    fn test_change_source_display_anomaly_rollback() {
        assert_eq!(ChangeSource::AnomalyRollback.to_string(), "anomaly-rollback");
    }

    // -----------------------------------------------------------------------
    // ParamDiff
    // -----------------------------------------------------------------------

    #[test]
    fn test_param_diff_delta_positive() {
        let d = ParamDiff { name: "x".into(), before: 100.0, after: 150.0 };
        assert!((d.delta() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_param_diff_delta_negative() {
        let d = ParamDiff { name: "x".into(), before: 200.0, after: 100.0 };
        assert!((d.delta() - (-100.0)).abs() < 1e-9);
    }

    #[test]
    fn test_param_diff_pct_change_normal() {
        let d = ParamDiff { name: "x".into(), before: 100.0, after: 125.0 };
        assert!((d.pct_change() - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_param_diff_pct_change_zero_before_returns_zero() {
        let d = ParamDiff { name: "x".into(), before: 0.0, after: 50.0 };
        assert_eq!(d.pct_change(), 0.0);
    }

    // -----------------------------------------------------------------------
    // ConfigSnapshot::diff_to
    // -----------------------------------------------------------------------

    #[test]
    fn test_diff_detects_changed_parameter() {
        let s1 = ConfigSnapshot {
            id: 1, timestamp_ms: 1, source: ChangeSource::Initial,
            params: base_params(), metrics: SnapshotMetrics::default(), note: None,
        };
        let s2 = ConfigSnapshot {
            id: 2, timestamp_ms: 2, source: ChangeSource::Controller,
            params: changed_params(), metrics: SnapshotMetrics::default(), note: None,
        };
        let diff = s1.diff_to(&s2);
        assert!(!diff.is_empty());
        let buf_dedup = diff.changes.iter().find(|d| d.name == "buf_dedup");
        assert!(buf_dedup.is_some());
        assert!((buf_dedup.unwrap().before - 128.0).abs() < 1e-9);
        assert!((buf_dedup.unwrap().after - 192.0).abs() < 1e-9);
    }

    #[test]
    fn test_diff_detects_new_parameter() {
        let s1 = ConfigSnapshot {
            id: 1, timestamp_ms: 1, source: ChangeSource::Initial,
            params: base_params(), metrics: SnapshotMetrics::default(), note: None,
        };
        let s2 = ConfigSnapshot {
            id: 2, timestamp_ms: 2, source: ChangeSource::Controller,
            params: changed_params(), metrics: SnapshotMetrics::default(), note: None,
        };
        let diff = s1.diff_to(&s2);
        let new_param = diff.changes.iter().find(|d| d.name == "priority_interval_ms");
        assert!(new_param.is_some());
        assert!((new_param.unwrap().before - 0.0).abs() < 1e-9);
        assert!((new_param.unwrap().after - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_diff_detects_removed_parameter() {
        // s2 does not have "rate_limit"
        let mut p2 = changed_params();
        p2.remove("rate_limit");
        let s1 = ConfigSnapshot {
            id: 1, timestamp_ms: 1, source: ChangeSource::Initial,
            params: base_params(), metrics: SnapshotMetrics::default(), note: None,
        };
        let s2 = ConfigSnapshot {
            id: 2, timestamp_ms: 2, source: ChangeSource::Controller,
            params: p2, metrics: SnapshotMetrics::default(), note: None,
        };
        let diff = s1.diff_to(&s2);
        let removed = diff.changes.iter().find(|d| d.name == "rate_limit");
        assert!(removed.is_some());
        assert!((removed.unwrap().before - 100.0).abs() < 1e-9);
        assert_eq!(removed.unwrap().after, 0.0);
    }

    #[test]
    fn test_diff_empty_when_identical() {
        let s1 = ConfigSnapshot {
            id: 1, timestamp_ms: 1, source: ChangeSource::Initial,
            params: base_params(), metrics: SnapshotMetrics::default(), note: None,
        };
        let s2 = ConfigSnapshot {
            id: 2, timestamp_ms: 2, source: ChangeSource::Controller,
            params: base_params(), metrics: SnapshotMetrics::default(), note: None,
        };
        let diff = s1.diff_to(&s2);
        assert!(diff.is_empty());
    }

    #[test]
    fn test_diff_changes_sorted_by_name() {
        let s1 = ConfigSnapshot {
            id: 1, timestamp_ms: 1, source: ChangeSource::Initial,
            params: base_params(), metrics: SnapshotMetrics::default(), note: None,
        };
        let s2 = ConfigSnapshot {
            id: 2, timestamp_ms: 2, source: ChangeSource::Controller,
            params: changed_params(), metrics: SnapshotMetrics::default(), note: None,
        };
        let diff = s1.diff_to(&s2);
        let names: Vec<&str> = diff.changes.iter().map(|d| d.name.as_str()).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted);
    }

    // -----------------------------------------------------------------------
    // SnapshotRegistry — basic operations
    // -----------------------------------------------------------------------

    #[test]
    fn test_registry_starts_empty() {
        let reg = SnapshotRegistry::new(10);
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn test_commit_returns_incrementing_ids() {
        let mut reg = SnapshotRegistry::new(10);
        let id1 = reg.commit(base_params(), ChangeSource::Initial, metrics(10.0, 0.0), None);
        let id2 = reg.commit(changed_params(), ChangeSource::Controller, metrics(12.0, 0.0), None);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn test_get_returns_correct_snapshot() {
        let mut reg = SnapshotRegistry::new(10);
        reg.commit(base_params(), ChangeSource::Initial, metrics(10.0, 0.0), None);
        let id = reg.commit(changed_params(), ChangeSource::Controller, metrics(12.0, 0.0), None);
        let snap = reg.get(id).unwrap();
        assert_eq!(snap.id, id);
        assert!(matches!(snap.source, ChangeSource::Controller));
    }

    #[test]
    fn test_get_unknown_id_returns_none() {
        let reg = SnapshotRegistry::new(10);
        assert!(reg.get(999).is_none());
    }

    #[test]
    fn test_latest_returns_most_recent() {
        let mut reg = SnapshotRegistry::new(10);
        reg.commit(base_params(), ChangeSource::Initial, metrics(10.0, 0.0), None);
        reg.commit(changed_params(), ChangeSource::Controller, metrics(12.0, 0.0), None);
        let latest = reg.latest().unwrap();
        assert_eq!(latest.id, 2);
    }

    #[test]
    fn test_latest_empty_registry_returns_none() {
        let reg = SnapshotRegistry::new(10);
        assert!(reg.latest().is_none());
    }

    // -----------------------------------------------------------------------
    // Capacity / eviction
    // -----------------------------------------------------------------------

    #[test]
    fn test_capacity_evicts_oldest() {
        let mut reg = SnapshotRegistry::new(3);
        let id1 = reg.commit(base_params(), ChangeSource::Initial, metrics(10.0, 0.0), None);
        reg.commit(base_params(), ChangeSource::Controller, metrics(11.0, 0.0), None);
        reg.commit(base_params(), ChangeSource::Controller, metrics(12.0, 0.0), None);
        // Adding a 4th should evict id1
        reg.commit(base_params(), ChangeSource::Controller, metrics(13.0, 0.0), None);
        assert_eq!(reg.len(), 3);
        assert!(reg.get(id1).is_none(), "oldest snapshot should be evicted");
    }

    #[test]
    fn test_capacity_one_always_contains_latest() {
        let mut reg = SnapshotRegistry::new(1);
        reg.commit(base_params(), ChangeSource::Initial, metrics(10.0, 0.0), None);
        let id = reg.commit(changed_params(), ChangeSource::Controller, metrics(11.0, 0.0), None);
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.latest().unwrap().id, id);
    }

    // -----------------------------------------------------------------------
    // Rollback
    // -----------------------------------------------------------------------

    #[test]
    fn test_rollback_to_creates_new_snapshot_with_old_params() {
        let mut reg = SnapshotRegistry::new(10);
        let id1 = reg.commit(base_params(), ChangeSource::Initial, metrics(10.0, 0.0), None);
        reg.commit(changed_params(), ChangeSource::Controller, metrics(20.0, 0.1), None);

        let rb_id = reg
            .rollback_to(id1, ChangeSource::AutoRollback { degraded_metric: "p95_latency_ms".into() }, metrics(15.0, 0.05))
            .unwrap();

        let rb_snap = reg.get(rb_id).unwrap();
        // Params should match snapshot 1
        assert_eq!(rb_snap.params.get("buf_dedup"), Some(&128.0));
        // New snapshot should NOT have priority_interval_ms (was added in changed_params)
        assert!(!rb_snap.params.contains_key("priority_interval_ms"));
        assert!(rb_snap.note.as_deref().unwrap_or("").contains("rollback"));
    }

    #[test]
    fn test_rollback_to_unknown_id_returns_err() {
        let mut reg = SnapshotRegistry::new(10);
        reg.commit(base_params(), ChangeSource::Initial, metrics(10.0, 0.0), None);
        let result = reg.rollback_to(999, ChangeSource::AnomalyRollback, metrics(10.0, 0.0));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("999"));
    }

    #[test]
    fn test_rollback_increments_id() {
        let mut reg = SnapshotRegistry::new(10);
        let id1 = reg.commit(base_params(), ChangeSource::Initial, metrics(10.0, 0.0), None);
        let id2 = reg.commit(changed_params(), ChangeSource::Controller, metrics(12.0, 0.0), None);
        let rb_id = reg.rollback_to(id1, ChangeSource::AnomalyRollback, metrics(10.0, 0.0)).unwrap();
        assert_eq!(rb_id, id2 + 1);
    }

    // -----------------------------------------------------------------------
    // Diff via registry
    // -----------------------------------------------------------------------

    #[test]
    fn test_registry_diff_between_two_snapshots() {
        let mut reg = SnapshotRegistry::new(10);
        let id1 = reg.commit(base_params(), ChangeSource::Initial, metrics(10.0, 0.0), None);
        let id2 = reg.commit(changed_params(), ChangeSource::Controller, metrics(12.0, 0.0), None);
        let diff = reg.diff(id1, id2).unwrap();
        assert!(!diff.is_empty());
    }

    #[test]
    fn test_registry_diff_unknown_from_returns_none() {
        let mut reg = SnapshotRegistry::new(10);
        let id = reg.commit(base_params(), ChangeSource::Initial, metrics(10.0, 0.0), None);
        assert!(reg.diff(999, id).is_none());
    }

    #[test]
    fn test_registry_diff_unknown_to_returns_none() {
        let mut reg = SnapshotRegistry::new(10);
        let id = reg.commit(base_params(), ChangeSource::Initial, metrics(10.0, 0.0), None);
        assert!(reg.diff(id, 999).is_none());
    }

    // -----------------------------------------------------------------------
    // since / last_window
    // -----------------------------------------------------------------------

    #[test]
    fn test_since_filters_by_timestamp() {
        let mut reg = SnapshotRegistry::new(10);
        // Internal clock increments on each commit: 1, 2, 3, 4
        reg.commit(base_params(), ChangeSource::Initial, metrics(10.0, 0.0), None); // ts=1
        reg.commit(base_params(), ChangeSource::Controller, metrics(11.0, 0.0), None); // ts=2
        reg.commit(base_params(), ChangeSource::Controller, metrics(12.0, 0.0), None); // ts=3
        reg.commit(base_params(), ChangeSource::Controller, metrics(13.0, 0.0), None); // ts=4
        let recent: Vec<_> = reg.since(3).collect();
        assert_eq!(recent.len(), 2, "should include ts=3 and ts=4");
    }

    #[test]
    fn test_last_window_returns_recent_entries() {
        let mut reg = SnapshotRegistry::new(10);
        reg.commit(base_params(), ChangeSource::Initial, metrics(10.0, 0.0), None); // ts=1
        reg.commit(base_params(), ChangeSource::Controller, metrics(11.0, 0.0), None); // ts=2
        reg.commit(base_params(), ChangeSource::Controller, metrics(12.0, 0.0), None); // ts=3
        // clock_ms is now 3; last_window(2) → cutoff = 3-2=1 → includes ts=2 and ts=3
        let window: Vec<_> = reg.last_window(2).collect();
        assert_eq!(window.len(), 2);
    }

    // -----------------------------------------------------------------------
    // best_in_window
    // -----------------------------------------------------------------------

    #[test]
    fn test_best_in_window_lower_is_better_latency() {
        let mut reg = SnapshotRegistry::new(10);
        reg.commit(base_params(), ChangeSource::Initial, metrics(50.0, 0.0), None); // p95=50
        reg.commit(base_params(), ChangeSource::Controller, metrics(30.0, 0.0), None); // p95=30 ← best
        reg.commit(base_params(), ChangeSource::Controller, metrics(40.0, 0.0), None); // p95=40
        let best = reg.best_in_window("p95_latency_ms", 100, false).unwrap();
        assert!((best.metrics.p95_latency_ms - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_best_in_window_higher_is_better_throughput() {
        let mut reg = SnapshotRegistry::new(10);
        let mut m1 = SnapshotMetrics::default(); m1.throughput_rps = 100.0;
        let mut m2 = SnapshotMetrics::default(); m2.throughput_rps = 500.0; // best
        let mut m3 = SnapshotMetrics::default(); m3.throughput_rps = 300.0;
        reg.commit(base_params(), ChangeSource::Initial, m1, None);
        reg.commit(base_params(), ChangeSource::Controller, m2, None);
        reg.commit(base_params(), ChangeSource::Controller, m3, None);
        let best = reg.best_in_window("throughput_rps", 100, true).unwrap();
        assert!((best.metrics.throughput_rps - 500.0).abs() < 1e-9);
    }

    #[test]
    fn test_best_in_window_empty_window_returns_none() {
        let mut reg = SnapshotRegistry::new(10);
        reg.commit(base_params(), ChangeSource::Initial, metrics(10.0, 0.0), None);
        // Window of 0ms → nothing qualifies (cutoff = clock - 0 = clock, filter is > cutoff)
        // Actually clock is 1 after one commit; window=0 means cutoff=1; filter is ts > 1 → nothing
        let best = reg.best_in_window("p95_latency_ms", 0, false);
        assert!(best.is_none());
    }

    #[test]
    fn test_best_in_window_unknown_metric_returns_none() {
        let mut reg = SnapshotRegistry::new(10);
        reg.commit(base_params(), ChangeSource::Initial, metrics(10.0, 0.0), None);
        let best = reg.best_in_window("nonexistent_metric", 100, true);
        assert!(best.is_none());
    }

    #[test]
    fn test_best_in_window_extras_metric() {
        let mut reg = SnapshotRegistry::new(10);
        let mut m1 = SnapshotMetrics::default();
        m1.extras.insert("cost_usd".to_string(), 0.05);
        let mut m2 = SnapshotMetrics::default();
        m2.extras.insert("cost_usd".to_string(), 0.02); // cheaper = better (lower)
        reg.commit(base_params(), ChangeSource::Initial, m1, None);
        reg.commit(base_params(), ChangeSource::Controller, m2, None);
        let best = reg.best_in_window("cost_usd", 100, false).unwrap();
        assert!((best.metrics.get("cost_usd").unwrap() - 0.02).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // commit_at
    // -----------------------------------------------------------------------

    #[test]
    fn test_commit_at_stores_explicit_timestamp() {
        let mut reg = SnapshotRegistry::new(10);
        let id = reg.commit_at(
            base_params(),
            ChangeSource::Initial,
            metrics(10.0, 0.0),
            None,
            999_999,
        );
        let snap = reg.get(id).unwrap();
        assert_eq!(snap.timestamp_ms, 999_999);
    }

    // -----------------------------------------------------------------------
    // config_history_json
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_history_json_returns_correct_count() {
        let mut reg = SnapshotRegistry::new(10);
        reg.commit(base_params(), ChangeSource::Initial, metrics(10.0, 0.0), None);
        reg.commit(changed_params(), ChangeSource::Controller, metrics(12.0, 0.0), None);
        let json = reg.config_history_json();
        assert_eq!(json.len(), 2);
    }

    #[test]
    fn test_config_history_json_first_entry_has_no_changes() {
        let mut reg = SnapshotRegistry::new(10);
        reg.commit(base_params(), ChangeSource::Initial, metrics(10.0, 0.0), None);
        let json = reg.config_history_json();
        let changes = json[0]["changes"].as_array().unwrap();
        assert!(changes.is_empty());
    }

    #[test]
    fn test_config_history_json_second_entry_has_changes() {
        let mut reg = SnapshotRegistry::new(10);
        reg.commit(base_params(), ChangeSource::Initial, metrics(10.0, 0.0), None);
        reg.commit(changed_params(), ChangeSource::Controller, metrics(12.0, 0.0), None);
        let json = reg.config_history_json();
        let changes = json[1]["changes"].as_array().unwrap();
        assert!(!changes.is_empty());
    }

    #[test]
    fn test_config_history_json_contains_source_string() {
        let mut reg = SnapshotRegistry::new(10);
        reg.commit(base_params(), ChangeSource::Initial, metrics(10.0, 0.0), None);
        let json = reg.config_history_json();
        assert_eq!(json[0]["source"].as_str().unwrap(), "initial");
    }

    #[test]
    fn test_config_history_json_empty_registry() {
        let reg = SnapshotRegistry::new(10);
        let json = reg.config_history_json();
        assert!(json.is_empty());
    }

    // -----------------------------------------------------------------------
    // all() iterator
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_iterates_in_chronological_order() {
        let mut reg = SnapshotRegistry::new(10);
        for i in 0..5u64 {
            let mut p = base_params();
            p.insert("counter".to_string(), i as f64);
            reg.commit(p, ChangeSource::Controller, metrics(10.0 + i as f64, 0.0), None);
        }
        let ids: Vec<u64> = reg.all().map(|s| s.id).collect();
        assert_eq!(ids, vec![1, 2, 3, 4, 5]);
    }

    // -----------------------------------------------------------------------
    // SnapshotMetrics::get
    // -----------------------------------------------------------------------

    #[test]
    fn test_snapshot_metrics_get_known_fields() {
        let m = SnapshotMetrics {
            p95_latency_ms: 15.0,
            drop_rate_pct: 1.5,
            cache_hit_rate: 0.7,
            error_rate: 0.02,
            throughput_rps: 250.0,
            extras: HashMap::new(),
        };
        assert_eq!(m.get("p95_latency_ms"), Some(15.0));
        assert_eq!(m.get("drop_rate_pct"), Some(1.5));
        assert_eq!(m.get("cache_hit_rate"), Some(0.7));
        assert_eq!(m.get("error_rate"), Some(0.02));
        assert_eq!(m.get("throughput_rps"), Some(250.0));
    }

    #[test]
    fn test_snapshot_metrics_get_unknown_returns_none() {
        let m = SnapshotMetrics::default();
        assert!(m.get("not_a_metric").is_none());
    }

    #[test]
    fn test_snapshot_metrics_get_extra() {
        let mut m = SnapshotMetrics::default();
        m.extras.insert("my_metric".to_string(), 42.0);
        assert_eq!(m.get("my_metric"), Some(42.0));
    }
}
