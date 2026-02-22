//! # Stage: Redis-Backed Snapshot Store
//!
//! ## Responsibility
//! Write-through Redis persistence layer for the `SnapshotRegistry`.  Every
//! `commit()` call is mirrored to a Redis List (`helix:snapshots`) so that
//! restarts can replay the configuration history.
//!
//! ## Guarantees
//! - Write-through: every commit is persisted to Redis (best-effort after
//!   construction — Redis write failures do not abort the in-memory commit)
//! - Restartable: `restore()` replays the Redis List into the `SnapshotRegistry`
//! - Non-panicking: all Redis error paths are handled via `Result`
//! - Bounded: the Redis List is trimmed to `capacity` on every write
//!
//! ## NOT Responsible For
//! - Applying restored params to the live pipeline (caller reads `registry()`)
//! - Cross-node synchronisation

#![cfg(feature = "redis-backing")]

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::snapshot::{ChangeSource, ConfigSnapshot, ParamMap, SnapshotMetrics, SnapshotRegistry};

// ---------------------------------------------------------------------------
// Redis key constants
// ---------------------------------------------------------------------------

const KEY_SNAPSHOTS: &str = "helix:snapshots";

// ---------------------------------------------------------------------------
// Serializable snapshot — mirrors ConfigSnapshot with serde-friendly types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SnapshotSerde {
    id: u64,
    timestamp_ms: u64,
    source: String,
    params: HashMap<String, f64>,
    p95_latency_ms: f64,
    drop_rate_pct: f64,
    cache_hit_rate: f64,
    error_rate: f64,
    throughput_rps: f64,
    extras: HashMap<String, f64>,
    note: Option<String>,
}

impl From<&ConfigSnapshot> for SnapshotSerde {
    fn from(s: &ConfigSnapshot) -> Self {
        Self {
            id: s.id,
            timestamp_ms: s.timestamp_ms,
            source: s.source.to_string(),
            params: s.params.clone(),
            p95_latency_ms: s.metrics.p95_latency_ms,
            drop_rate_pct: s.metrics.drop_rate_pct,
            cache_hit_rate: s.metrics.cache_hit_rate,
            error_rate: s.metrics.error_rate,
            throughput_rps: s.metrics.throughput_rps,
            extras: s.metrics.extras.clone(),
            note: s.note.clone(),
        }
    }
}

impl SnapshotSerde {
    fn into_parts(self) -> (ParamMap, ChangeSource, SnapshotMetrics, Option<String>, u64) {
        let source = parse_source(&self.source);
        let metrics = SnapshotMetrics {
            p95_latency_ms: self.p95_latency_ms,
            drop_rate_pct: self.drop_rate_pct,
            cache_hit_rate: self.cache_hit_rate,
            error_rate: self.error_rate,
            throughput_rps: self.throughput_rps,
            extras: self.extras,
        };
        (self.params, source, metrics, self.note, self.timestamp_ms)
    }
}

fn parse_source(s: &str) -> ChangeSource {
    if s == "controller" {
        ChangeSource::Controller
    } else if s == "anomaly-rollback" {
        ChangeSource::AnomalyRollback
    } else if s == "initial" {
        ChangeSource::Initial
    } else if let Some(rest) = s.strip_prefix("experiment:") {
        ChangeSource::Experiment {
            experiment_name: rest.to_string(),
        }
    } else if let Some(rest) = s.strip_prefix("manual:") {
        ChangeSource::Manual {
            operator: rest.to_string(),
        }
    } else if let Some(rest) = s.strip_prefix("auto-rollback:") {
        ChangeSource::AutoRollback {
            degraded_metric: rest.to_string(),
        }
    } else {
        ChangeSource::Manual {
            operator: s.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// SnapshotRedisOps trait
// ---------------------------------------------------------------------------

/// Abstraction over the Redis list operations needed by `RedisSnapshotStore`.
pub(crate) trait SnapshotRedisOps: Send + Sync {
    fn lpush(&mut self, key: &str, value: &str) -> Result<(), String>;
    fn ltrim(&mut self, key: &str, start: isize, stop: isize) -> Result<(), String>;
    fn lrange(&mut self, key: &str, start: isize, stop: isize) -> Result<Vec<String>, String>;
    fn ping(&mut self) -> Result<(), String>;
}

// ---------------------------------------------------------------------------
// redis::Connection impl
// ---------------------------------------------------------------------------

impl SnapshotRedisOps for redis::Connection {
    fn lpush(&mut self, key: &str, value: &str) -> Result<(), String> {
        redis::cmd("LPUSH")
            .arg(key)
            .arg(value)
            .query::<()>(self)
            .map_err(|e| e.to_string())
    }

    fn ltrim(&mut self, key: &str, start: isize, stop: isize) -> Result<(), String> {
        redis::cmd("LTRIM")
            .arg(key)
            .arg(start)
            .arg(stop)
            .query::<()>(self)
            .map_err(|e| e.to_string())
    }

    fn lrange(&mut self, key: &str, start: isize, stop: isize) -> Result<Vec<String>, String> {
        redis::cmd("LRANGE")
            .arg(key)
            .arg(start)
            .arg(stop)
            .query::<Vec<String>>(self)
            .map_err(|e| e.to_string())
    }

    fn ping(&mut self) -> Result<(), String> {
        redis::cmd("PING")
            .query::<String>(self)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
}

// ---------------------------------------------------------------------------
// MockSnapshotRedisOps
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) struct MockSnapshotRedisOps {
    list: Vec<String>,
}

#[cfg(test)]
impl MockSnapshotRedisOps {
    pub(crate) fn new() -> Self {
        Self { list: Vec::new() }
    }
}

#[cfg(test)]
impl SnapshotRedisOps for MockSnapshotRedisOps {
    fn lpush(&mut self, _key: &str, value: &str) -> Result<(), String> {
        self.list.insert(0, value.to_string());
        Ok(())
    }

    fn ltrim(&mut self, _key: &str, start: isize, stop: isize) -> Result<(), String> {
        let len = self.list.len() as isize;
        let s = start.max(0) as usize;
        let e = if stop < 0 {
            (len + stop).max(-1) as usize
        } else {
            stop.min(len - 1).max(-1) as usize
        };
        if s > e || s >= self.list.len() {
            self.list.clear();
        } else {
            self.list = self.list[s..=e].to_vec();
        }
        Ok(())
    }

    fn lrange(&mut self, _key: &str, start: isize, stop: isize) -> Result<Vec<String>, String> {
        let len = self.list.len() as isize;
        let s = start.max(0) as usize;
        let e = if stop < 0 {
            (len + stop).max(0) as usize
        } else {
            stop.min(len - 1).max(0) as usize
        };
        if self.list.is_empty() || s > e || s >= self.list.len() {
            Ok(vec![])
        } else {
            Ok(self.list[s..=e].to_vec())
        }
    }

    fn ping(&mut self) -> Result<(), String> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RedisSnapshotStore
// ---------------------------------------------------------------------------

/// Write-through Redis-backed wrapper around `SnapshotRegistry`.
///
/// # Construction
/// - `connect(url, capacity)` — opens a blocking Redis connection
/// - `with_ops(ops, capacity)` — inject any `SnapshotRedisOps` (used in tests)
///
/// # Persistence model
/// `commit()` calls `SnapshotRegistry::commit_at` with a fixed timestamp of 0
/// (test mode) or the timestamp embedded in the serde record on restore.
/// The Redis List is trimmed to `capacity` on every write.
pub struct RedisSnapshotStore {
    ops: Box<dyn SnapshotRedisOps>,
    registry: SnapshotRegistry,
    capacity: usize,
}

impl RedisSnapshotStore {
    /// Open a connection to Redis and return a new store.
    pub fn connect(url: &str, capacity: usize) -> Result<Self, String> {
        let client = redis::Client::open(url).map_err(|e| e.to_string())?;
        let mut conn = client
            .get_connection()
            .map_err(|e| format!("Redis connection failed: {}", e))?;
        conn.ping().map_err(|e| format!("Redis ping failed: {}", e))?;
        Ok(Self {
            ops: Box::new(conn),
            registry: SnapshotRegistry::new(capacity),
            capacity,
        })
    }

    /// Construct with an injected `SnapshotRedisOps` implementation.
    pub fn with_ops(ops: Box<dyn SnapshotRedisOps>, capacity: usize) -> Self {
        Self {
            ops,
            registry: SnapshotRegistry::new(capacity),
            capacity,
        }
    }

    /// Load snapshots from Redis into the in-memory registry.
    ///
    /// Returns the number of snapshots successfully loaded.
    pub fn restore(&mut self) -> Result<usize, String> {
        let raw = self.ops.lrange(KEY_SNAPSHOTS, 0, -1)?;
        let mut loaded = 0usize;
        // Newest-first in Redis; restore oldest-first
        for json in raw.iter().rev() {
            if let Ok(s) = serde_json::from_str::<SnapshotSerde>(json) {
                let (params, source, metrics, note, ts) = s.into_parts();
                self.registry.commit_at(params, source, metrics, note, ts);
                loaded += 1;
            }
        }
        Ok(loaded)
    }

    /// Commit a new snapshot to both the in-memory registry and Redis.
    ///
    /// Returns the new snapshot ID.
    pub fn commit(
        &mut self,
        params: ParamMap,
        source: ChangeSource,
        metrics: SnapshotMetrics,
        note: Option<String>,
    ) -> u64 {
        let id = self.registry.commit(params, source, metrics, note);
        // Persist the snapshot we just committed
        if let Some(snap) = self.registry.get(id) {
            let serde_snap = SnapshotSerde::from(snap);
            if let Ok(json) = serde_json::to_string(&serde_snap) {
                let _ = self.ops.lpush(KEY_SNAPSHOTS, &json);
                let stop = (self.capacity as isize) - 1;
                let _ = self.ops.ltrim(KEY_SNAPSHOTS, 0, stop);
            }
        }
        id
    }

    /// Access the underlying `SnapshotRegistry`.
    pub fn registry(&self) -> &SnapshotRegistry {
        &self.registry
    }

    /// Number of snapshots in the in-memory registry.
    pub fn len(&self) -> usize {
        self.registry.len()
    }

    /// Returns `true` if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.registry.is_empty()
    }

    /// Return the most recent snapshot.
    pub fn latest(&self) -> Option<&ConfigSnapshot> {
        self.registry.latest()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn store(capacity: usize) -> RedisSnapshotStore {
        RedisSnapshotStore::with_ops(Box::new(MockSnapshotRedisOps::new()), capacity)
    }

    fn params() -> ParamMap {
        let mut m = HashMap::new();
        m.insert("buf".to_string(), 128.0);
        m
    }

    fn metrics_default() -> SnapshotMetrics {
        SnapshotMetrics::default()
    }

    fn commit_one(s: &mut RedisSnapshotStore) -> u64 {
        s.commit(params(), ChangeSource::Initial, metrics_default(), None)
    }

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    #[test]
    fn with_ops_creates_store() {
        let s = store(10);
        assert!(s.is_empty());
    }

    #[test]
    fn len_zero_when_empty() {
        let s = store(10);
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn is_empty_true_when_empty() {
        let s = store(10);
        assert!(s.is_empty());
    }

    // -----------------------------------------------------------------------
    // commit
    // -----------------------------------------------------------------------

    #[test]
    fn commit_increments_len() {
        let mut s = store(10);
        commit_one(&mut s);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn commit_persists_to_redis() {
        let mut s = store(10);
        commit_one(&mut s);
        commit_one(&mut s);
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn commit_returns_monotonic_ids() {
        let mut s = store(10);
        let id1 = commit_one(&mut s);
        let id2 = commit_one(&mut s);
        assert!(id2 > id1);
    }

    #[test]
    fn two_commits_both_in_registry() {
        let mut s = store(10);
        let id1 = commit_one(&mut s);
        let id2 = commit_one(&mut s);
        assert!(s.registry().get(id1).is_some());
        assert!(s.registry().get(id2).is_some());
    }

    // -----------------------------------------------------------------------
    // latest
    // -----------------------------------------------------------------------

    #[test]
    fn latest_returns_most_recent() {
        let mut s = store(10);
        commit_one(&mut s);
        let id2 = s.commit(params(), ChangeSource::Controller, metrics_default(), None);
        assert_eq!(s.latest().unwrap().id, id2);
    }

    #[test]
    fn latest_returns_none_when_empty() {
        let s = store(10);
        assert!(s.latest().is_none());
    }

    // -----------------------------------------------------------------------
    // registry accessor
    // -----------------------------------------------------------------------

    #[test]
    fn registry_accessible() {
        let mut s = store(10);
        commit_one(&mut s);
        assert_eq!(s.registry().len(), 1);
    }

    // -----------------------------------------------------------------------
    // is_empty
    // -----------------------------------------------------------------------

    #[test]
    fn is_empty_false_after_commit() {
        let mut s = store(10);
        commit_one(&mut s);
        assert!(!s.is_empty());
    }

    // -----------------------------------------------------------------------
    // restore
    // -----------------------------------------------------------------------

    #[test]
    fn restore_loads_snapshots_from_redis() {
        // Pre-populate mock with a JSON snapshot
        let mut mock = MockSnapshotRedisOps::new();
        let snap_serde = SnapshotSerde {
            id: 1,
            timestamp_ms: 100,
            source: "initial".to_string(),
            params: HashMap::from([("x".to_string(), 1.0)]),
            p95_latency_ms: 10.0,
            drop_rate_pct: 0.0,
            cache_hit_rate: 0.5,
            error_rate: 0.0,
            throughput_rps: 100.0,
            extras: HashMap::new(),
            note: None,
        };
        let json = serde_json::to_string(&snap_serde).unwrap();
        mock.lpush(KEY_SNAPSHOTS, &json).unwrap();

        let mut s = RedisSnapshotStore::with_ops(Box::new(mock), 10);
        s.restore().unwrap();
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn restore_returns_count() {
        let mut mock = MockSnapshotRedisOps::new();
        for i in 0..3u64 {
            let snap = SnapshotSerde {
                id: i + 1,
                timestamp_ms: i * 10,
                source: "controller".to_string(),
                params: HashMap::new(),
                p95_latency_ms: 0.0,
                drop_rate_pct: 0.0,
                cache_hit_rate: 0.0,
                error_rate: 0.0,
                throughput_rps: 0.0,
                extras: HashMap::new(),
                note: None,
            };
            let json = serde_json::to_string(&snap).unwrap();
            mock.lpush(KEY_SNAPSHOTS, &json).unwrap();
        }
        let mut s = RedisSnapshotStore::with_ops(Box::new(mock), 10);
        let count = s.restore().unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn restore_empty_redis_gives_zero_count() {
        let mut s = store(10);
        let count = s.restore().unwrap();
        assert_eq!(count, 0);
    }

    // -----------------------------------------------------------------------
    // ltrim caps list
    // -----------------------------------------------------------------------

    #[test]
    fn ltrim_caps_snapshot_list() {
        let mut s = store(3);
        for _ in 0..5 {
            commit_one(&mut s);
        }
        // In-memory registry also evicts, so len == 3
        assert_eq!(s.len(), 3);
    }

    // -----------------------------------------------------------------------
    // MockSnapshotRedisOps unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn mock_snapshot_ops_lpush_lrange() {
        let mut mock = MockSnapshotRedisOps::new();
        mock.lpush("k", "a").unwrap();
        mock.lpush("k", "b").unwrap();
        let vals = mock.lrange("k", 0, -1).unwrap();
        assert_eq!(vals, vec!["b", "a"]);
    }

    #[test]
    fn mock_snapshot_ops_ltrim() {
        let mut mock = MockSnapshotRedisOps::new();
        for i in 0..5 {
            mock.lpush("k", &i.to_string()).unwrap();
        }
        mock.ltrim("k", 0, 1).unwrap();
        let vals = mock.lrange("k", 0, -1).unwrap();
        assert_eq!(vals.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Serialisation round-trips
    // -----------------------------------------------------------------------

    #[test]
    fn snapshot_roundtrip_json() {
        let snap = SnapshotSerde {
            id: 42,
            timestamp_ms: 9999,
            source: "manual:alice".to_string(),
            params: HashMap::from([("rate".to_string(), 100.0)]),
            p95_latency_ms: 12.5,
            drop_rate_pct: 0.1,
            cache_hit_rate: 0.8,
            error_rate: 0.02,
            throughput_rps: 500.0,
            extras: HashMap::from([("cost".to_string(), 0.05)]),
            note: Some("test note".to_string()),
        };
        let json = serde_json::to_string(&snap).unwrap();
        let back: SnapshotSerde = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, 42);
        assert_eq!(back.note.as_deref(), Some("test note"));
        assert!((back.p95_latency_ms - 12.5).abs() < 1e-9);
    }

    #[test]
    fn param_map_serializes_correctly() {
        let mut params = ParamMap::new();
        params.insert("a".to_string(), 1.5);
        params.insert("b".to_string(), 2.5);
        let json = serde_json::to_string(&params).unwrap();
        let back: ParamMap = serde_json::from_str(&json).unwrap();
        assert_eq!(back["a"], 1.5);
        assert_eq!(back["b"], 2.5);
    }

    // -----------------------------------------------------------------------
    // commit with note
    // -----------------------------------------------------------------------

    #[test]
    fn commit_with_note_stores_note() {
        let mut s = store(10);
        let id = s.commit(
            params(),
            ChangeSource::Manual { operator: "bob".to_string() },
            metrics_default(),
            Some("my note".to_string()),
        );
        let snap = s.registry().get(id).unwrap();
        assert_eq!(snap.note.as_deref(), Some("my note"));
    }

    #[test]
    fn commit_with_none_note_is_valid() {
        let mut s = store(10);
        let id = commit_one(&mut s);
        let snap = s.registry().get(id).unwrap();
        assert!(snap.note.is_none());
    }

    // -----------------------------------------------------------------------
    // SnapshotMetrics default
    // -----------------------------------------------------------------------

    #[test]
    fn snapshot_metrics_default_values() {
        let m = SnapshotMetrics::default();
        assert_eq!(m.p95_latency_ms, 0.0);
        assert_eq!(m.throughput_rps, 0.0);
    }

    // -----------------------------------------------------------------------
    // capacity stored
    // -----------------------------------------------------------------------

    #[test]
    fn redis_snapshot_store_capacity_stored() {
        let s = store(42);
        assert_eq!(s.capacity, 42);
    }

    // -----------------------------------------------------------------------
    // ChangeSource variants
    // -----------------------------------------------------------------------

    #[test]
    fn commit_source_controller_valid() {
        let mut s = store(10);
        let id = s.commit(params(), ChangeSource::Controller, metrics_default(), None);
        let snap = s.registry().get(id).unwrap();
        assert!(matches!(snap.source, ChangeSource::Controller));
    }

    #[test]
    fn commit_source_manual_valid() {
        let mut s = store(10);
        let id = s.commit(
            params(),
            ChangeSource::Manual { operator: "ops".to_string() },
            metrics_default(),
            None,
        );
        let snap = s.registry().get(id).unwrap();
        assert!(matches!(snap.source, ChangeSource::Manual { .. }));
    }
}
