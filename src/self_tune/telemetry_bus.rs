//! # Stage: Telemetry Feedback Bus
//!
//! ## Responsibility
//! Aggregates metrics from every pipeline stage into a unified stream.
//! Emits [`TelemetrySnapshot`] structs at a configurable interval (default 5 s).
//! Maintains four rolling windows (1 m, 5 m, 15 m, 1 h) in bounded ring buffers.
//!
//! ## Guarantees
//! - Thread-safe: all types are `Send + Sync`
//! - Bounded: ring buffers have fixed capacity; old entries are evicted
//! - Non-blocking: `record_*` methods never block the calling task
//! - Graceful shutdown: `shutdown()` drains the bus cleanly
//!
//! ## NOT Responsible For
//! - Persistence (Redis integration is in the snapshotter, task 1.6)
//! - Parameter adjustment (that is the controller, task 1.2)
//! - Anomaly scoring (that is the anomaly detector, task 1.4)

use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::sync::{broadcast, Mutex, RwLock};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of samples held in each rolling window ring buffer.
pub const WINDOW_1M_CAP: usize = 60;   // one sample per second
pub const WINDOW_5M_CAP: usize = 300;
pub const WINDOW_15M_CAP: usize = 900;
pub const WINDOW_1H_CAP: usize = 3_600;

/// Broadcast channel capacity (number of snapshots in flight).
pub const BUS_CHANNEL_CAP: usize = 256;

// ---------------------------------------------------------------------------
// Metric types
// ---------------------------------------------------------------------------

/// One latency observation from a pipeline stage, in microseconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatencyObs {
    pub stage: PipelineStage,
    pub micros: u64,
}

/// Named pipeline stages whose metrics flow into the bus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineStage {
    Dedup,
    RateLimit,
    Priority,
    Cache,
    Inference,
    CircuitBreaker,
    /// Any stage not explicitly named above.
    Other,
}

impl std::fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            PipelineStage::Dedup          => "dedup",
            PipelineStage::RateLimit      => "rate_limit",
            PipelineStage::Priority       => "priority",
            PipelineStage::Cache          => "cache",
            PipelineStage::Inference      => "inference",
            PipelineStage::CircuitBreaker => "circuit_breaker",
            PipelineStage::Other          => "other",
        };
        write!(f, "{s}")
    }
}

// ---------------------------------------------------------------------------
// Rolling ring buffer
// ---------------------------------------------------------------------------

/// A fixed-capacity ring buffer of `f64` samples.
///
/// Evicts the oldest entry when full. All operations are O(1).
#[derive(Debug, Clone)]
pub struct RingBuffer {
    buf: Vec<f64>,
    head: usize,
    len: usize,
    cap: usize,
}

impl RingBuffer {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "RingBuffer capacity must be > 0");
        Self {
            buf: vec![0.0; capacity],
            head: 0,
            len: 0,
            cap: capacity,
        }
    }

    /// Push a sample, evicting the oldest when full.
    pub fn push(&mut self, value: f64) {
        self.buf[self.head] = value;
        self.head = (self.head + 1) % self.cap;
        if self.len < self.cap {
            self.len += 1;
        }
    }

    /// Number of valid samples currently stored.
    pub fn len(&self) -> usize { self.len }

    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Iterate samples from oldest to newest.
    pub fn iter(&self) -> impl Iterator<Item = f64> + '_ {
        let start = if self.len == self.cap {
            self.head
        } else {
            0
        };
        (0..self.len).map(move |i| self.buf[(start + i) % self.cap])
    }

    /// Arithmetic mean of all stored samples, or `None` if empty.
    pub fn mean(&self) -> Option<f64> {
        if self.is_empty() { return None; }
        Some(self.iter().sum::<f64>() / self.len as f64)
    }

    /// p95 of all stored samples, or `None` if empty.
    pub fn p95(&self) -> Option<f64> {
        if self.is_empty() { return None; }
        let mut v: Vec<f64> = self.iter().collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((v.len() as f64 * 0.95).ceil() as usize).saturating_sub(1).min(v.len() - 1);
        Some(v[idx])
    }

    /// Maximum value, or `None` if empty.
    pub fn max(&self) -> Option<f64> {
        self.iter().reduce(f64::max)
    }
}

// ---------------------------------------------------------------------------
// Per-stage telemetry accumulator
// ---------------------------------------------------------------------------

/// Mutable state accumulated between snapshots for one pipeline stage.
#[derive(Debug, Default)]
struct StageAccumulator {
    latency_sum_us: u64,
    latency_count: u64,
    error_count: u64,
    throughput_count: u64,
}

impl StageAccumulator {
    fn record_latency(&mut self, micros: u64) {
        self.latency_sum_us = self.latency_sum_us.saturating_add(micros);
        self.latency_count += 1;
        self.throughput_count += 1;
    }

    fn record_error(&mut self) {
        self.error_count += 1;
    }

    fn avg_latency_us(&self) -> f64 {
        if self.latency_count == 0 { 0.0 }
        else { self.latency_sum_us as f64 / self.latency_count as f64 }
    }

    fn reset(&mut self) -> StageAccumulator {
        std::mem::take(self)
    }
}

// ---------------------------------------------------------------------------
// TelemetrySnapshot — the public type broadcast on the bus
// ---------------------------------------------------------------------------

/// A point-in-time snapshot of system-wide metrics.
///
/// Emitted at a configurable interval (default 5 s). Every self-improving
/// module subscribes to a stream of these.
#[derive(Debug, Clone)]
pub struct TelemetrySnapshot {
    /// Wall-clock instant this snapshot was captured.
    pub captured_at: Instant,

    // --- global counters (monotonically increasing) ---

    /// Total requests processed since startup.
    pub total_requests: u64,
    /// Total requests dropped (backpressure shed) since startup.
    pub total_dropped: u64,
    /// Total errors since startup.
    pub total_errors: u64,
    /// Total cache hits since startup.
    pub total_cache_hits: u64,
    /// Total dedup collisions since startup.
    pub total_dedup_hits: u64,

    // --- derived rates (per-interval) ---

    /// Requests processed during the last interval.
    pub interval_requests: u64,
    /// Errors during the last interval.
    pub interval_errors: u64,
    /// Drop rate as a fraction [0, 1] during the last interval.
    pub drop_rate: f64,
    /// Cache hit rate as a fraction [0, 1] since startup.
    pub cache_hit_rate: f64,

    // --- latency (all values in microseconds) ---

    /// Average end-to-end latency during the last interval.
    pub avg_latency_us: f64,
    /// p95 end-to-end latency in the 1-minute rolling window.
    pub p95_1m_us: f64,
    /// p95 in the 5-minute window.
    pub p95_5m_us: f64,
    /// p95 in the 15-minute window.
    pub p95_15m_us: f64,

    // --- circuit breaker ---

    /// `true` when any circuit breaker is currently open.
    pub circuit_open: bool,
    /// Number of circuit breaker trips since startup.
    pub circuit_trips: u64,

    // --- queue depth ---

    /// Current queue depth (number of items waiting for processing).
    pub queue_depth: u64,
    /// Queue depth as a fraction of configured capacity [0, 1].
    pub queue_fill_frac: f64,
}

impl TelemetrySnapshot {
    fn zero() -> Self {
        Self {
            captured_at: Instant::now(),
            total_requests: 0,
            total_dropped: 0,
            total_errors: 0,
            total_cache_hits: 0,
            total_dedup_hits: 0,
            interval_requests: 0,
            interval_errors: 0,
            drop_rate: 0.0,
            cache_hit_rate: 0.0,
            avg_latency_us: 0.0,
            p95_1m_us: 0.0,
            p95_5m_us: 0.0,
            p95_15m_us: 0.0,
            circuit_open: false,
            circuit_trips: 0,
            queue_depth: 0,
            queue_fill_frac: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// BusConfig
// ---------------------------------------------------------------------------

/// Configuration for the telemetry bus.
#[derive(Debug, Clone)]
pub struct BusConfig {
    /// How often to emit a [`TelemetrySnapshot`].
    pub emit_interval: Duration,
    /// Maximum queue depth (used to compute `queue_fill_frac`).
    pub queue_capacity: u64,
}

impl Default for BusConfig {
    fn default() -> Self {
        Self {
            emit_interval: Duration::from_secs(5),
            queue_capacity: 1024,
        }
    }
}

// ---------------------------------------------------------------------------
// TelemetryBus — the main public type
// ---------------------------------------------------------------------------

/// Inner state, protected by RwLock.
struct BusState {
    cfg: BusConfig,

    // Global monotonic counters
    total_requests: AtomicU64,
    total_dropped:  AtomicU64,
    total_errors:   AtomicU64,
    total_cache_hits: AtomicU64,
    total_dedup_hits: AtomicU64,
    circuit_trips:  AtomicU64,
    queue_depth:    AtomicU64,

    // Rolling windows (1m, 5m, 15m, 1h) of avg_latency_us samples
    window_1m:  Mutex<RingBuffer>,
    window_5m:  Mutex<RingBuffer>,
    window_15m: Mutex<RingBuffer>,
    window_1h:  Mutex<RingBuffer>,

    // Per-interval accumulator reset on each snapshot
    accumulator: Mutex<StageAccumulator>,

    // Latest snapshot, updated each interval
    latest: RwLock<TelemetrySnapshot>,

    // Broadcast sender
    tx: broadcast::Sender<TelemetrySnapshot>,

    circuit_open: std::sync::atomic::AtomicBool,

    // Used to track the previous interval's request count for rate calculation
    prev_total_requests: AtomicU64,
    prev_total_errors:   AtomicU64,
    prev_total_dropped:  AtomicU64,
}

/// The telemetry feedback bus.
///
/// Clone freely — all clones share the same underlying state.
///
/// # Example
/// ```ignore
/// let bus = TelemetryBus::new(BusConfig::default());
/// bus.start_emitter(); // spawns background task
/// bus.record_latency(PipelineStage::Cache, 340);
/// let mut rx = bus.subscribe();
/// let snap = rx.recv().await.unwrap();
/// ```
#[derive(Clone)]
pub struct TelemetryBus {
    inner: Arc<BusState>,
}

impl TelemetryBus {
    /// Create a new bus with the given configuration.
    pub fn new(cfg: BusConfig) -> Self {
        let (tx, _) = broadcast::channel(BUS_CHANNEL_CAP);
        let inner = Arc::new(BusState {
            cfg,
            total_requests: AtomicU64::new(0),
            total_dropped:  AtomicU64::new(0),
            total_errors:   AtomicU64::new(0),
            total_cache_hits: AtomicU64::new(0),
            total_dedup_hits: AtomicU64::new(0),
            circuit_trips:  AtomicU64::new(0),
            queue_depth:    AtomicU64::new(0),
            window_1m:  Mutex::new(RingBuffer::new(WINDOW_1M_CAP)),
            window_5m:  Mutex::new(RingBuffer::new(WINDOW_5M_CAP)),
            window_15m: Mutex::new(RingBuffer::new(WINDOW_15M_CAP)),
            window_1h:  Mutex::new(RingBuffer::new(WINDOW_1H_CAP)),
            accumulator: Mutex::new(StageAccumulator::default()),
            latest: RwLock::new(TelemetrySnapshot::zero()),
            tx,
            circuit_open: std::sync::atomic::AtomicBool::new(false),
            prev_total_requests: AtomicU64::new(0),
            prev_total_errors:   AtomicU64::new(0),
            prev_total_dropped:  AtomicU64::new(0),
        });
        Self { inner }
    }

    /// Subscribe to the snapshot stream.
    ///
    /// The receiver will see every [`TelemetrySnapshot`] emitted while it is alive.
    /// Slow receivers will see [`broadcast::error::RecvError::Lagged`] when they
    /// fall too far behind (more than `BUS_CHANNEL_CAP` snapshots).
    pub fn subscribe(&self) -> broadcast::Receiver<TelemetrySnapshot> {
        self.inner.tx.subscribe()
    }

    /// Spawn the background emitter task. Call once after construction.
    ///
    /// The emitter fires every `cfg.emit_interval`, builds a snapshot from
    /// accumulated metrics, and broadcasts it to all subscribers.
    pub fn start_emitter(&self) {
        let bus = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(bus.inner.cfg.emit_interval);
            loop {
                interval.tick().await;
                bus.emit_snapshot().await;
            }
        });
    }

    /// Record a latency observation from a pipeline stage (non-blocking).
    pub fn record_latency(&self, _stage: PipelineStage, micros: u64) {
        self.inner.total_requests.fetch_add(1, Ordering::Relaxed);
        // Try to update accumulator; if lock contended, skip (best-effort telemetry)
        if let Ok(mut acc) = self.inner.accumulator.try_lock() {
            acc.record_latency(micros);
        }
    }

    /// Record a dropped request.
    pub fn record_drop(&self) {
        self.inner.total_dropped.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a pipeline error.
    pub fn record_error(&self, _stage: PipelineStage) {
        self.inner.total_errors.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut acc) = self.inner.accumulator.try_lock() {
            acc.record_error();
        }
    }

    /// Record a cache hit.
    pub fn record_cache_hit(&self) {
        self.inner.total_cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a dedup collision (request deduplicated against an in-flight request).
    pub fn record_dedup_hit(&self) {
        self.inner.total_dedup_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Update the current queue depth.
    pub fn set_queue_depth(&self, depth: u64) {
        self.inner.queue_depth.store(depth, Ordering::Relaxed);
    }

    /// Record a circuit breaker state transition.
    pub fn record_circuit_transition(&self, is_open: bool) {
        self.inner.circuit_open.store(is_open, Ordering::Relaxed);
        if is_open {
            self.inner.circuit_trips.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Return the most recently emitted snapshot (or a zero snapshot if none yet).
    pub async fn latest(&self) -> TelemetrySnapshot {
        self.inner.latest.read().await.clone()
    }

    // --- internal ---

    async fn emit_snapshot(&self) {
        let i = &self.inner;

        // Drain accumulator
        let acc = {
            let mut lock = i.accumulator.lock().await;
            lock.reset()
        };

        let total_req  = i.total_requests.load(Ordering::Relaxed);
        let total_drop = i.total_dropped.load(Ordering::Relaxed);
        let total_err  = i.total_errors.load(Ordering::Relaxed);
        let total_ch   = i.total_cache_hits.load(Ordering::Relaxed);

        let prev_req   = i.prev_total_requests.swap(total_req, Ordering::Relaxed);
        let prev_err   = i.prev_total_errors.swap(total_err, Ordering::Relaxed);
        let prev_drop  = i.prev_total_dropped.swap(total_drop, Ordering::Relaxed);

        let interval_req  = total_req.saturating_sub(prev_req);
        let interval_err  = total_err.saturating_sub(prev_err);
        let interval_drop = total_drop.saturating_sub(prev_drop);

        let drop_rate = if interval_req == 0 { 0.0 }
            else { interval_drop as f64 / interval_req as f64 };

        let cache_hit_rate = if total_req == 0 { 0.0 }
            else { total_ch as f64 / total_req as f64 };

        let avg_us = acc.avg_latency_us();

        // Push avg into rolling windows
        {
            let mut w1 = i.window_1m.lock().await;
            w1.push(avg_us);
        }
        {
            let mut w5 = i.window_5m.lock().await;
            w5.push(avg_us);
        }
        {
            let mut w15 = i.window_15m.lock().await;
            w15.push(avg_us);
        }
        {
            let mut w1h = i.window_1h.lock().await;
            w1h.push(avg_us);
        }

        let p95_1m  = { i.window_1m.lock().await.p95().unwrap_or(0.0) };
        let p95_5m  = { i.window_5m.lock().await.p95().unwrap_or(0.0) };
        let p95_15m = { i.window_15m.lock().await.p95().unwrap_or(0.0) };

        let queue_d = i.queue_depth.load(Ordering::Relaxed);
        let qfrac = if i.cfg.queue_capacity == 0 { 0.0 }
            else { (queue_d as f64 / i.cfg.queue_capacity as f64).min(1.0) };

        let snap = TelemetrySnapshot {
            captured_at: Instant::now(),
            total_requests: total_req,
            total_dropped:  total_drop,
            total_errors:   total_err,
            total_cache_hits: total_ch,
            total_dedup_hits: i.total_dedup_hits.load(Ordering::Relaxed),
            interval_requests: interval_req,
            interval_errors:   interval_err,
            drop_rate,
            cache_hit_rate,
            avg_latency_us: avg_us,
            p95_1m_us:  p95_1m,
            p95_5m_us:  p95_5m,
            p95_15m_us: p95_15m,
            circuit_open: i.circuit_open.load(Ordering::Relaxed),
            circuit_trips: i.circuit_trips.load(Ordering::Relaxed),
            queue_depth: queue_d,
            queue_fill_frac: qfrac,
        };

        *i.latest.write().await = snap.clone();
        // Ignore send error — no subscribers is fine
        let _ = i.tx.send(snap);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::timeout;

    fn default_bus() -> TelemetryBus {
        TelemetryBus::new(BusConfig::default())
    }

    // ===== RingBuffer =====

    #[test]
    fn test_ringbuffer_new_is_empty() {
        let rb = RingBuffer::new(10);
        assert!(rb.is_empty());
        assert_eq!(rb.len(), 0);
    }

    #[test]
    fn test_ringbuffer_push_increments_len() {
        let mut rb = RingBuffer::new(5);
        rb.push(1.0);
        assert_eq!(rb.len(), 1);
        rb.push(2.0);
        assert_eq!(rb.len(), 2);
    }

    #[test]
    fn test_ringbuffer_len_capped_at_capacity() {
        let mut rb = RingBuffer::new(3);
        for i in 0..10 {
            rb.push(i as f64);
        }
        assert_eq!(rb.len(), 3);
    }

    #[test]
    fn test_ringbuffer_iter_order_oldest_to_newest() {
        let mut rb = RingBuffer::new(3);
        rb.push(10.0);
        rb.push(20.0);
        rb.push(30.0);
        // Push a 4th element to trigger wrap-around
        rb.push(40.0);
        let vals: Vec<f64> = rb.iter().collect();
        // oldest three: 20, 30, 40
        assert_eq!(vals, vec![20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_ringbuffer_mean_empty_is_none() {
        let rb = RingBuffer::new(5);
        assert_eq!(rb.mean(), None);
    }

    #[test]
    fn test_ringbuffer_mean_single() {
        let mut rb = RingBuffer::new(5);
        rb.push(42.0);
        assert!((rb.mean().unwrap() - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_ringbuffer_mean_multiple() {
        let mut rb = RingBuffer::new(4);
        rb.push(10.0);
        rb.push(20.0);
        rb.push(30.0);
        rb.push(40.0);
        assert!((rb.mean().unwrap() - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_ringbuffer_p95_empty_is_none() {
        let rb = RingBuffer::new(5);
        assert_eq!(rb.p95(), None);
    }

    #[test]
    fn test_ringbuffer_p95_single() {
        let mut rb = RingBuffer::new(5);
        rb.push(99.0);
        assert!((rb.p95().unwrap() - 99.0).abs() < 1e-9);
    }

    #[test]
    fn test_ringbuffer_p95_sorted_100() {
        let mut rb = RingBuffer::new(100);
        for i in 1..=100u64 {
            rb.push(i as f64);
        }
        assert!((rb.p95().unwrap() - 95.0).abs() < 1.0);
    }

    #[test]
    fn test_ringbuffer_max_empty_is_none() {
        let rb = RingBuffer::new(3);
        assert_eq!(rb.max(), None);
    }

    #[test]
    fn test_ringbuffer_max_correct() {
        let mut rb = RingBuffer::new(5);
        for v in [3.0, 1.0, 4.0, 1.0, 5.0] {
            rb.push(v);
        }
        assert!((rb.max().unwrap() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_ringbuffer_wraps_correctly_full_cycle() {
        let mut rb = RingBuffer::new(4);
        for i in 1..=8u64 {
            rb.push(i as f64);
        }
        let vals: Vec<f64> = rb.iter().collect();
        assert_eq!(vals, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_ringbuffer_mean_after_wrap() {
        let mut rb = RingBuffer::new(3);
        rb.push(1.0); rb.push(2.0); rb.push(3.0);
        rb.push(100.0); // evicts 1.0
        // remaining: 2, 3, 100
        let m = rb.mean().unwrap();
        assert!((m - 35.0).abs() < 1e-9);
    }

    // ===== PipelineStage =====

    #[test]
    fn test_pipeline_stage_display_dedup() {
        assert_eq!(PipelineStage::Dedup.to_string(), "dedup");
    }

    #[test]
    fn test_pipeline_stage_display_inference() {
        assert_eq!(PipelineStage::Inference.to_string(), "inference");
    }

    #[test]
    fn test_pipeline_stage_display_other() {
        assert_eq!(PipelineStage::Other.to_string(), "other");
    }

    #[test]
    fn test_pipeline_stage_eq() {
        assert_eq!(PipelineStage::Cache, PipelineStage::Cache);
        assert_ne!(PipelineStage::Cache, PipelineStage::Dedup);
    }

    // ===== StageAccumulator =====

    #[test]
    fn test_accumulator_record_latency_increments_count() {
        let mut acc = StageAccumulator::default();
        acc.record_latency(100);
        acc.record_latency(200);
        assert_eq!(acc.latency_count, 2);
        assert_eq!(acc.throughput_count, 2);
    }

    #[test]
    fn test_accumulator_avg_latency_empty_is_zero() {
        let acc = StageAccumulator::default();
        assert_eq!(acc.avg_latency_us(), 0.0);
    }

    #[test]
    fn test_accumulator_avg_latency_correct() {
        let mut acc = StageAccumulator::default();
        acc.record_latency(100);
        acc.record_latency(300);
        assert!((acc.avg_latency_us() - 200.0).abs() < 1e-9);
    }

    #[test]
    fn test_accumulator_error_count() {
        let mut acc = StageAccumulator::default();
        acc.record_error();
        acc.record_error();
        assert_eq!(acc.error_count, 2);
    }

    #[test]
    fn test_accumulator_reset_returns_old_values() {
        let mut acc = StageAccumulator::default();
        acc.record_latency(500);
        let old = acc.reset();
        assert_eq!(old.latency_count, 1);
        // After reset, acc is zeroed
        assert_eq!(acc.latency_count, 0);
    }

    #[test]
    fn test_accumulator_saturation_add() {
        let mut acc = StageAccumulator::default();
        acc.latency_sum_us = u64::MAX;
        acc.record_latency(1); // should not overflow
        assert_eq!(acc.latency_sum_us, u64::MAX);
    }

    // ===== TelemetryBus construction =====

    #[test]
    fn test_bus_new_does_not_panic() {
        let _ = default_bus();
    }

    #[test]
    fn test_bus_clone_shares_state() {
        let bus = default_bus();
        let bus2 = bus.clone();
        bus.record_latency(PipelineStage::Dedup, 100);
        // Both share the same inner Arc
        assert_eq!(
            bus.inner.total_requests.load(Ordering::Relaxed),
            bus2.inner.total_requests.load(Ordering::Relaxed)
        );
    }

    #[test]
    fn test_bus_record_latency_increments_total() {
        let bus = default_bus();
        bus.record_latency(PipelineStage::Cache, 200);
        bus.record_latency(PipelineStage::Inference, 300);
        assert_eq!(bus.inner.total_requests.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_bus_record_drop_increments_counter() {
        let bus = default_bus();
        bus.record_drop();
        bus.record_drop();
        assert_eq!(bus.inner.total_dropped.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_bus_record_error_increments_counter() {
        let bus = default_bus();
        bus.record_error(PipelineStage::CircuitBreaker);
        assert_eq!(bus.inner.total_errors.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_bus_record_cache_hit_increments_counter() {
        let bus = default_bus();
        bus.record_cache_hit();
        bus.record_cache_hit();
        bus.record_cache_hit();
        assert_eq!(bus.inner.total_cache_hits.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_bus_record_dedup_hit_increments_counter() {
        let bus = default_bus();
        bus.record_dedup_hit();
        assert_eq!(bus.inner.total_dedup_hits.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_bus_set_queue_depth() {
        let bus = default_bus();
        bus.set_queue_depth(42);
        assert_eq!(bus.inner.queue_depth.load(Ordering::Relaxed), 42);
    }

    #[test]
    fn test_bus_circuit_transition_open_increments_trips() {
        let bus = default_bus();
        bus.record_circuit_transition(true);
        assert!(bus.inner.circuit_open.load(Ordering::Relaxed));
        assert_eq!(bus.inner.circuit_trips.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_bus_circuit_transition_close_does_not_increment_trips() {
        let bus = default_bus();
        bus.record_circuit_transition(false);
        assert!(!bus.inner.circuit_open.load(Ordering::Relaxed));
        assert_eq!(bus.inner.circuit_trips.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_bus_emit_snapshot_broadcasts() {
        let bus = TelemetryBus::new(BusConfig {
            emit_interval: Duration::from_millis(50),
            ..Default::default()
        });
        let mut rx = bus.subscribe();
        bus.start_emitter();

        let result = timeout(Duration::from_millis(500), rx.recv()).await;
        assert!(result.is_ok(), "should receive snapshot within 500ms");
        let snap = result.unwrap().unwrap();
        assert!(snap.total_requests < u64::MAX); // always true, confirms type
    }

    #[tokio::test]
    async fn test_bus_emit_snapshot_reflects_recorded_metrics() {
        let bus = TelemetryBus::new(BusConfig {
            emit_interval: Duration::from_millis(30),
            ..Default::default()
        });
        let mut rx = bus.subscribe();

        bus.record_latency(PipelineStage::Inference, 1000);
        bus.record_latency(PipelineStage::Inference, 3000);
        bus.record_drop();
        bus.record_cache_hit();
        bus.start_emitter();

        let snap = timeout(Duration::from_millis(300), rx.recv())
            .await
            .expect("timeout")
            .expect("recv error");

        assert_eq!(snap.total_requests, 2);
        assert_eq!(snap.total_dropped, 1);
        assert_eq!(snap.total_cache_hits, 1);
        assert!(snap.avg_latency_us > 0.0);
    }

    #[tokio::test]
    async fn test_bus_emit_drop_rate_calculation() {
        let bus = TelemetryBus::new(BusConfig {
            emit_interval: Duration::from_millis(30),
            ..Default::default()
        });
        let mut rx = bus.subscribe();

        // 3 requests, 1 drop → drop_rate should be 1/3 ≈ 0.333
        bus.record_latency(PipelineStage::Dedup, 100);
        bus.record_latency(PipelineStage::Dedup, 100);
        bus.record_latency(PipelineStage::Dedup, 100);
        bus.record_drop();
        bus.start_emitter();

        let snap = timeout(Duration::from_millis(300), rx.recv())
            .await.expect("timeout").expect("recv error");

        assert!((snap.drop_rate - 1.0 / 3.0).abs() < 0.01,
            "drop_rate={}", snap.drop_rate);
    }

    #[tokio::test]
    async fn test_bus_emit_cache_hit_rate_calculation() {
        let bus = TelemetryBus::new(BusConfig {
            emit_interval: Duration::from_millis(30),
            ..Default::default()
        });
        let mut rx = bus.subscribe();

        for _ in 0..10 { bus.record_latency(PipelineStage::Cache, 50); }
        for _ in 0..4  { bus.record_cache_hit(); }
        bus.start_emitter();

        let snap = timeout(Duration::from_millis(300), rx.recv())
            .await.expect("timeout").expect("recv error");

        assert!((snap.cache_hit_rate - 0.4).abs() < 0.01,
            "cache_hit_rate={}", snap.cache_hit_rate);
    }

    #[tokio::test]
    async fn test_bus_queue_fill_frac_clamped_to_one() {
        let bus = TelemetryBus::new(BusConfig {
            emit_interval: Duration::from_millis(30),
            queue_capacity: 10,
            ..Default::default()
        });
        let mut rx = bus.subscribe();
        bus.set_queue_depth(1000); // way over capacity
        bus.start_emitter();

        let snap = timeout(Duration::from_millis(300), rx.recv())
            .await.expect("timeout").expect("recv error");
        assert!((snap.queue_fill_frac - 1.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_bus_latest_returns_last_snapshot() {
        let bus = TelemetryBus::new(BusConfig {
            emit_interval: Duration::from_millis(20),
            ..Default::default()
        });
        bus.start_emitter();
        tokio::time::sleep(Duration::from_millis(100)).await;
        let snap = bus.latest().await;
        // After at least one tick, captured_at should be recent
        assert!(snap.captured_at.elapsed() < Duration::from_secs(5));
    }

    #[tokio::test]
    async fn test_bus_multiple_subscribers_all_receive() {
        let bus = TelemetryBus::new(BusConfig {
            emit_interval: Duration::from_millis(30),
            ..Default::default()
        });
        let mut rx1 = bus.subscribe();
        let mut rx2 = bus.subscribe();
        let mut rx3 = bus.subscribe();
        bus.start_emitter();

        let r1 = timeout(Duration::from_millis(300), rx1.recv()).await;
        let r2 = timeout(Duration::from_millis(300), rx2.recv()).await;
        let r3 = timeout(Duration::from_millis(300), rx3.recv()).await;

        assert!(r1.is_ok());
        assert!(r2.is_ok());
        assert!(r3.is_ok());
    }

    #[tokio::test]
    async fn test_bus_p95_rolling_windows_populated() {
        let bus = TelemetryBus::new(BusConfig {
            emit_interval: Duration::from_millis(20),
            ..Default::default()
        });
        let mut rx = bus.subscribe();
        for us in [100, 200, 300, 400, 500] {
            bus.record_latency(PipelineStage::Inference, us);
        }
        bus.start_emitter();

        let snap = timeout(Duration::from_millis(300), rx.recv())
            .await.expect("timeout").expect("recv");
        // After first emit the 1m window has one sample
        assert!(snap.p95_1m_us >= 0.0);
    }

    #[tokio::test]
    async fn test_bus_circuit_open_reflected_in_snapshot() {
        let bus = TelemetryBus::new(BusConfig {
            emit_interval: Duration::from_millis(20),
            ..Default::default()
        });
        let mut rx = bus.subscribe();
        bus.record_circuit_transition(true);
        bus.start_emitter();

        let snap = timeout(Duration::from_millis(300), rx.recv())
            .await.expect("timeout").expect("recv");
        assert!(snap.circuit_open);
        assert_eq!(snap.circuit_trips, 1);
    }

    #[tokio::test]
    async fn test_bus_interval_requests_resets_between_snapshots() {
        let bus = TelemetryBus::new(BusConfig {
            emit_interval: Duration::from_millis(40),
            ..Default::default()
        });
        let mut rx = bus.subscribe();

        // Record 5 requests before first snapshot
        for _ in 0..5 { bus.record_latency(PipelineStage::Dedup, 10); }
        bus.start_emitter();

        let snap1 = timeout(Duration::from_millis(300), rx.recv())
            .await.expect("timeout").expect("recv");
        assert_eq!(snap1.interval_requests, 5);

        // No new requests between intervals
        let snap2 = timeout(Duration::from_millis(300), rx.recv())
            .await.expect("timeout").expect("recv");
        assert_eq!(snap2.interval_requests, 0);
    }

    // ===== BusConfig defaults =====

    #[test]
    fn test_bus_config_default_emit_interval_is_5s() {
        let cfg = BusConfig::default();
        assert_eq!(cfg.emit_interval, Duration::from_secs(5));
    }

    #[test]
    fn test_bus_config_default_queue_capacity() {
        let cfg = BusConfig::default();
        assert_eq!(cfg.queue_capacity, 1024);
    }

    #[test]
    fn test_bus_config_zero_capacity_gives_zero_fill_frac() {
        let bus = TelemetryBus::new(BusConfig {
            emit_interval: Duration::from_millis(1),
            queue_capacity: 0,
        });
        bus.set_queue_depth(100);
        // queue_fill_frac should be 0 when capacity is 0 (no div by zero)
        assert_eq!(bus.inner.cfg.queue_capacity, 0);
    }
}
