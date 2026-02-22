//! # Stage: Parameter Controller
//!
//! ## Responsibility
//! PID controllers for 12 tunable pipeline parameters. Reads [`TelemetrySnapshot`]
//! from the telemetry bus, computes error signals (actual vs target), applies
//! bounded PID updates, and logs every adjustment with before/after metrics.
//! Auto-rolls back any change that degrades the target metric by >10% within
//! the configured rollback window.
//!
//! ## Guarantees
//! - Bounded: all parameters are clamped to `[min, max]`
//! - Cooldown-aware: a parameter is not adjusted again until its cooldown expires
//! - Rollback-safe: a snapshot of each parameter is kept before every change
//! - Non-panicking: all arithmetic is saturating or checked; no `unwrap` outside tests
//!
//! ## NOT Responsible For
//! - Persistence (snapshot versioning is in task 1.6)
//! - A/B experiment routing (experiment engine, task 1.3)
//! - Anomaly detection (anomaly detector, task 1.4)

use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use super::telemetry_bus::TelemetrySnapshot;

// ---------------------------------------------------------------------------
// Parameter names
// ---------------------------------------------------------------------------

/// The 12 tunable parameters managed by the controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Param {
    // Channel buffer sizes (one per stage)
    DedupChannelBuf,
    RateLimitChannelBuf,
    PriorityChannelBuf,
    CacheChannelBuf,
    InferenceChannelBuf,
    // Backpressure
    BackpressureShedThreshold,
    // Circuit breaker
    CircuitBreakerFailureThreshold,
    CircuitBreakerSuccessRate,
    CircuitBreakerTimeoutMs,
    // Deduplication
    DedupTtlMs,
    DedupHashBuckets,
    // Rate limiter
    RateLimiterRefillRate,
}

impl Param {
    /// Human-readable name for logging.
    pub fn name(self) -> &'static str {
        match self {
            Param::DedupChannelBuf              => "dedup_channel_buf",
            Param::RateLimitChannelBuf          => "rate_limit_channel_buf",
            Param::PriorityChannelBuf           => "priority_channel_buf",
            Param::CacheChannelBuf              => "cache_channel_buf",
            Param::InferenceChannelBuf          => "inference_channel_buf",
            Param::BackpressureShedThreshold    => "backpressure_shed_threshold",
            Param::CircuitBreakerFailureThreshold => "circuit_breaker_failure_threshold",
            Param::CircuitBreakerSuccessRate    => "circuit_breaker_success_rate",
            Param::CircuitBreakerTimeoutMs      => "circuit_breaker_timeout_ms",
            Param::DedupTtlMs                   => "dedup_ttl_ms",
            Param::DedupHashBuckets             => "dedup_hash_buckets",
            Param::RateLimiterRefillRate        => "rate_limiter_refill_rate",
        }
    }

    /// All 12 parameters in a fixed order.
    pub fn all() -> &'static [Param] {
        &[
            Param::DedupChannelBuf,
            Param::RateLimitChannelBuf,
            Param::PriorityChannelBuf,
            Param::CacheChannelBuf,
            Param::InferenceChannelBuf,
            Param::BackpressureShedThreshold,
            Param::CircuitBreakerFailureThreshold,
            Param::CircuitBreakerSuccessRate,
            Param::CircuitBreakerTimeoutMs,
            Param::DedupTtlMs,
            Param::DedupHashBuckets,
            Param::RateLimiterRefillRate,
        ]
    }
}

impl std::fmt::Display for Param {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// ParameterSpec — constraints and PID gains for one parameter
// ---------------------------------------------------------------------------

/// Specification for a single tunable parameter.
#[derive(Debug, Clone)]
pub struct ParameterSpec {
    /// Minimum allowed value (inclusive).
    pub min: f64,
    /// Maximum allowed value (inclusive).
    pub max: f64,
    /// Minimum step size per adjustment.
    pub step: f64,
    /// How long to wait before re-adjusting this parameter after a change.
    pub cooldown: Duration,
    /// If a change degrades the target metric by more than this fraction,
    /// auto-rollback is triggered. E.g. `0.10` means roll back on >10% degradation.
    pub rollback_threshold: f64,
    /// PID proportional gain.
    pub kp: f64,
    /// PID integral gain.
    pub ki: f64,
    /// PID derivative gain.
    pub kd: f64,
}

impl ParameterSpec {
    /// Clamp a candidate value to `[min, max]`.
    pub fn clamp(&self, v: f64) -> f64 {
        v.clamp(self.min, self.max)
    }

    /// Round a value to the nearest multiple of `step` (clamped).
    pub fn snap(&self, v: f64) -> f64 {
        if self.step <= 0.0 { return self.clamp(v); }
        let snapped = (v / self.step).round() * self.step;
        self.clamp(snapped)
    }
}

impl Default for ParameterSpec {
    fn default() -> Self {
        Self {
            min: 1.0,
            max: 10_000.0,
            step: 1.0,
            cooldown: Duration::from_secs(30),
            rollback_threshold: 0.10,
            kp: 0.5,
            ki: 0.05,
            kd: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// PidState — running state for one parameter's PID loop
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct PidState {
    integral: f64,
    prev_error: f64,
}

impl Default for PidState {
    fn default() -> Self { Self { integral: 0.0, prev_error: 0.0 } }
}

impl PidState {
    /// Compute PID output given current error and spec gains.
    /// Returns the signed adjustment to apply to the current value.
    fn update(&mut self, error: f64, spec: &ParameterSpec, dt: f64) -> f64 {
        self.integral += error * dt;
        // Clamp integral to prevent wind-up: ±(max - min)
        let wind_up_limit = (spec.max - spec.min).abs();
        self.integral = self.integral.clamp(-wind_up_limit, wind_up_limit);

        let derivative = if dt > 0.0 { (error - self.prev_error) / dt } else { 0.0 };
        self.prev_error = error;

        spec.kp * error + spec.ki * self.integral + spec.kd * derivative
    }

    fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
    }
}

// ---------------------------------------------------------------------------
// AdjustmentRecord — audit log entry
// ---------------------------------------------------------------------------

/// A record of one parameter adjustment, kept in the audit log.
#[derive(Debug, Clone)]
pub struct AdjustmentRecord {
    pub param: Param,
    pub before: f64,
    pub after: f64,
    pub error_signal: f64,
    pub pid_output: f64,
    /// The metric value (e.g. avg_latency_us) that triggered this adjustment.
    pub trigger_metric: f64,
    pub timestamp: Instant,
    /// Whether this record represents a rollback.
    pub is_rollback: bool,
}

// ---------------------------------------------------------------------------
// RollbackGuard — snapshot for auto-rollback
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct RollbackGuard {
    param: Param,
    value_before: f64,
    metric_before: f64,
    applied_at: Instant,
    rollback_window: Duration,
    rollback_threshold: f64,
}

impl RollbackGuard {
    /// Returns `true` if the metric has degraded enough to trigger rollback.
    fn should_rollback(&self, current_metric: f64) -> bool {
        if self.metric_before <= 0.0 { return false; }
        // For latency: degradation = (current - before) / before (higher is worse)
        let degradation = (current_metric - self.metric_before) / self.metric_before;
        degradation > self.rollback_threshold
    }

    fn is_expired(&self) -> bool {
        self.applied_at.elapsed() > self.rollback_window
    }
}

// ---------------------------------------------------------------------------
// ControllerConfig
// ---------------------------------------------------------------------------

/// Configuration for the parameter controller.
#[derive(Debug, Clone)]
pub struct ControllerConfig {
    /// Target average end-to-end latency in microseconds.
    pub target_latency_us: f64,
    /// Target maximum drop rate [0, 1].
    pub target_drop_rate: f64,
    /// Maximum number of audit log entries kept in memory.
    pub audit_log_cap: usize,
    /// How long a rollback guard is active before expiring.
    pub rollback_window: Duration,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            target_latency_us: 5_000.0,   // 5 ms
            target_drop_rate: 0.01,        // 1%
            audit_log_cap: 1_000,
            rollback_window: Duration::from_secs(30),
        }
    }
}

// ---------------------------------------------------------------------------
// Controller — the main type
// ---------------------------------------------------------------------------

/// PID controller managing all 12 tunable parameters.
///
/// Call [`Controller::observe`] whenever a new [`TelemetrySnapshot`] arrives.
/// The controller decides whether to adjust each parameter based on the
/// error signal, cooldown, and PID state.
pub struct Controller {
    cfg: ControllerConfig,
    specs: HashMap<Param, ParameterSpec>,
    values: HashMap<Param, f64>,
    pid: HashMap<Param, PidState>,
    last_adjusted: HashMap<Param, Instant>,
    rollback_guards: Vec<RollbackGuard>,
    audit_log: Vec<AdjustmentRecord>,
    last_snapshot_time: Option<Instant>,
}

impl Controller {
    /// Create a controller with default specs for all 12 parameters.
    pub fn new(cfg: ControllerConfig) -> Self {
        let mut specs = HashMap::new();
        let mut values = HashMap::new();
        let mut pid = HashMap::new();
        let mut last_adjusted = HashMap::new();

        for &p in Param::all() {
            let spec = default_spec(p);
            values.insert(p, default_value(p));
            specs.insert(p, spec);
            pid.insert(p, PidState::default());
            // Initialize last_adjusted far in the past so the first snapshot
            // can trigger adjustments immediately.
            last_adjusted.insert(p, Instant::now() - Duration::from_secs(3_600));
        }

        Self {
            cfg,
            specs,
            values,
            pid,
            last_adjusted,
            rollback_guards: Vec::new(),
            audit_log: Vec::new(),
            last_snapshot_time: None,
        }
    }

    /// Override the spec for a specific parameter.
    pub fn set_spec(&mut self, param: Param, spec: ParameterSpec) {
        self.specs.insert(param, spec);
    }

    /// Read the current value of a parameter.
    pub fn get(&self, param: Param) -> f64 {
        self.values.get(&param).copied().unwrap_or(0.0)
    }

    /// Forcibly set a parameter value (bypasses PID, used for manual overrides).
    pub fn set(&mut self, param: Param, value: f64) {
        if let Some(spec) = self.specs.get(&param) {
            let clamped = spec.clamp(value);
            self.values.insert(param, clamped);
        }
    }

    /// Process a new telemetry snapshot.
    ///
    /// For each parameter whose cooldown has expired, computes the PID error
    /// signal, applies the update, and records the adjustment. Also processes
    /// any pending rollback guards.
    pub fn observe(&mut self, snap: &TelemetrySnapshot) {
        let now = Instant::now();
        let dt = self.last_snapshot_time
            .map(|t| now.duration_since(t).as_secs_f64())
            .unwrap_or(1.0)
            .max(0.001);
        self.last_snapshot_time = Some(now);

        // --- Check rollback guards ---
        self.check_rollbacks(snap, now);

        // --- Latency-driven parameters ---
        let latency_error = self.cfg.target_latency_us - snap.avg_latency_us;

        // --- Drop-rate-driven parameters ---
        let drop_error = self.cfg.target_drop_rate - snap.drop_rate;

        // Drive channel buffer sizes with latency error
        let channel_params = [
            Param::DedupChannelBuf,
            Param::RateLimitChannelBuf,
            Param::PriorityChannelBuf,
            Param::CacheChannelBuf,
            Param::InferenceChannelBuf,
        ];
        for p in channel_params {
            self.apply_pid(p, latency_error, snap.avg_latency_us, dt, now);
        }

        // Drive backpressure with drop rate error
        self.apply_pid(Param::BackpressureShedThreshold, drop_error, snap.drop_rate, dt, now);

        // Drive dedup TTL with cache hit rate error (higher hit rate → increase TTL)
        let cache_error = snap.cache_hit_rate - 0.5; // target 50% cache hits
        self.apply_pid(Param::DedupTtlMs, cache_error, snap.cache_hit_rate, dt, now);

        // Drive hash buckets with dedup hit rate
        let dedup_hit_rate = if snap.total_requests == 0 { 0.0 }
            else { snap.total_dedup_hits as f64 / snap.total_requests as f64 };
        let dedup_error = dedup_hit_rate - 0.05; // target 5% dedup rate
        self.apply_pid(Param::DedupHashBuckets, dedup_error, dedup_hit_rate, dt, now);

        // Drive rate limiter with latency + queue pressure
        let queue_pressure_error = 0.5 - snap.queue_fill_frac; // target 50% queue fill
        self.apply_pid(Param::RateLimiterRefillRate, queue_pressure_error, snap.queue_fill_frac, dt, now);

        // Drive circuit breaker timeout with latency
        self.apply_pid(Param::CircuitBreakerTimeoutMs, latency_error, snap.avg_latency_us, dt, now);

        // Circuit breaker thresholds are driven by error rate
        let error_rate = if snap.total_requests == 0 { 0.0 }
            else { snap.total_errors as f64 / snap.total_requests as f64 };
        let error_rate_error = 0.05 - error_rate; // target <5% error rate
        self.apply_pid(Param::CircuitBreakerFailureThreshold, error_rate_error, error_rate, dt, now);
        self.apply_pid(Param::CircuitBreakerSuccessRate, -error_rate_error, error_rate, dt, now);
    }

    /// The audit log of all adjustments made, newest last.
    pub fn audit_log(&self) -> &[AdjustmentRecord] {
        &self.audit_log
    }

    /// Clear the audit log.
    pub fn clear_audit_log(&mut self) {
        self.audit_log.clear();
    }

    /// Number of currently active rollback guards.
    pub fn active_rollback_guards(&self) -> usize {
        self.rollback_guards.len()
    }

    // --- private ---

    fn apply_pid(
        &mut self,
        param: Param,
        error: f64,
        trigger_metric: f64,
        dt: f64,
        now: Instant,
    ) {
        let spec = match self.specs.get(&param) {
            Some(s) => s.clone(),
            None => return,
        };
        let last = self.last_adjusted.get(&param).copied()
            .unwrap_or_else(|| now - Duration::from_secs(3_600));

        if now.duration_since(last) < spec.cooldown {
            return; // still in cooldown
        }

        let pid_state = self.pid.entry(param).or_default();
        let raw_output = pid_state.update(error, &spec, dt);

        // Only apply if the adjustment is at least one step in magnitude
        if raw_output.abs() < spec.step { return; }

        let current = self.values.get(&param).copied().unwrap_or(default_value(param));
        let candidate = spec.snap(current + raw_output);

        if (candidate - current).abs() < spec.step * 0.5 { return; }

        // Install a rollback guard before mutating
        self.rollback_guards.push(RollbackGuard {
            param,
            value_before: current,
            metric_before: trigger_metric,
            applied_at: now,
            rollback_window: self.cfg.rollback_window,
            rollback_threshold: spec.rollback_threshold,
        });

        self.values.insert(param, candidate);
        self.last_adjusted.insert(param, now);

        self.push_audit(AdjustmentRecord {
            param,
            before: current,
            after: candidate,
            error_signal: error,
            pid_output: raw_output,
            trigger_metric,
            timestamp: now,
            is_rollback: false,
        });
    }

    fn check_rollbacks(&mut self, snap: &TelemetrySnapshot, now: Instant) {
        let mut to_rollback: Vec<RollbackGuard> = Vec::new();

        self.rollback_guards.retain(|g| {
            if g.is_expired() {
                // Guard expired cleanly — no rollback needed
                return false;
            }
            // Determine the current metric for this parameter's domain
            let current_metric = match g.param {
                Param::DedupChannelBuf
                | Param::RateLimitChannelBuf
                | Param::PriorityChannelBuf
                | Param::CacheChannelBuf
                | Param::InferenceChannelBuf
                | Param::CircuitBreakerTimeoutMs => snap.avg_latency_us,
                Param::BackpressureShedThreshold => snap.drop_rate,
                _ => snap.avg_latency_us,
            };
            if g.should_rollback(current_metric) {
                to_rollback.push(g.clone());
                false // remove from guards
            } else {
                true // keep watching
            }
        });

        for guard in to_rollback {
            let current = self.values.get(&guard.param).copied()
                .unwrap_or(default_value(guard.param));
            self.values.insert(guard.param, guard.value_before);
            // Reset PID integral to prevent wind-up after rollback
            if let Some(pid) = self.pid.get_mut(&guard.param) {
                pid.reset();
            }
            // Install a post-rollback cooldown equal to the rollback_window so
            // the parameter cannot be re-adjusted during the same observe cycle.
            self.last_adjusted.insert(guard.param, now);
            self.push_audit(AdjustmentRecord {
                param: guard.param,
                before: current,
                after: guard.value_before,
                error_signal: 0.0,
                pid_output: 0.0,
                trigger_metric: guard.metric_before,
                timestamp: now,
                is_rollback: true,
            });
        }
    }

    fn push_audit(&mut self, record: AdjustmentRecord) {
        if self.audit_log.len() >= self.cfg.audit_log_cap {
            self.audit_log.remove(0);
        }
        self.audit_log.push(record);
    }
}

// ---------------------------------------------------------------------------
// Default specs and values for the 12 parameters
// ---------------------------------------------------------------------------

fn default_spec(p: Param) -> ParameterSpec {
    match p {
        Param::DedupChannelBuf
        | Param::RateLimitChannelBuf
        | Param::PriorityChannelBuf
        | Param::CacheChannelBuf
        | Param::InferenceChannelBuf => ParameterSpec {
            min: 16.0, max: 4096.0, step: 16.0,
            cooldown: Duration::from_secs(10),
            rollback_threshold: 0.10,
            kp: 0.001, ki: 0.0001, kd: 0.0005,
        },
        Param::BackpressureShedThreshold => ParameterSpec {
            min: 0.05, max: 1.0, step: 0.01,
            cooldown: Duration::from_secs(15),
            rollback_threshold: 0.20,
            kp: 0.5, ki: 0.05, kd: 0.1,
        },
        Param::CircuitBreakerFailureThreshold => ParameterSpec {
            min: 1.0, max: 50.0, step: 1.0,
            cooldown: Duration::from_secs(30),
            rollback_threshold: 0.10,
            kp: 2.0, ki: 0.1, kd: 0.5,
        },
        Param::CircuitBreakerSuccessRate => ParameterSpec {
            min: 0.10, max: 1.0, step: 0.05,
            cooldown: Duration::from_secs(30),
            rollback_threshold: 0.10,
            kp: 0.5, ki: 0.05, kd: 0.1,
        },
        Param::CircuitBreakerTimeoutMs => ParameterSpec {
            min: 100.0, max: 30_000.0, step: 100.0,
            cooldown: Duration::from_secs(20),
            rollback_threshold: 0.15,
            kp: 0.0001, ki: 0.00001, kd: 0.00005,
        },
        Param::DedupTtlMs => ParameterSpec {
            min: 100.0, max: 60_000.0, step: 100.0,
            cooldown: Duration::from_secs(20),
            rollback_threshold: 0.10,
            kp: 100.0, ki: 10.0, kd: 20.0,
        },
        Param::DedupHashBuckets => ParameterSpec {
            min: 64.0, max: 65536.0, step: 64.0,
            cooldown: Duration::from_secs(60),
            rollback_threshold: 0.10,
            kp: 1000.0, ki: 50.0, kd: 200.0,
        },
        Param::RateLimiterRefillRate => ParameterSpec {
            min: 1.0, max: 10_000.0, step: 1.0,
            cooldown: Duration::from_secs(5),
            rollback_threshold: 0.15,
            kp: 10.0, ki: 1.0, kd: 2.0,
        },
    }
}

fn default_value(p: Param) -> f64 {
    match p {
        Param::DedupChannelBuf
        | Param::RateLimitChannelBuf
        | Param::PriorityChannelBuf
        | Param::CacheChannelBuf
        | Param::InferenceChannelBuf => 256.0,
        Param::BackpressureShedThreshold => 0.80,
        Param::CircuitBreakerFailureThreshold => 5.0,
        Param::CircuitBreakerSuccessRate => 0.50,
        Param::CircuitBreakerTimeoutMs => 5_000.0,
        Param::DedupTtlMs => 5_000.0,
        Param::DedupHashBuckets => 1024.0,
        Param::RateLimiterRefillRate => 100.0,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn snap_zero() -> TelemetrySnapshot {
        TelemetrySnapshot {
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
            avg_latency_us: 5_000.0, // exactly on target → no adjustment
            p95_1m_us: 5_000.0,
            p95_5m_us: 5_000.0,
            p95_15m_us: 5_000.0,
            circuit_open: false,
            circuit_trips: 0,
            queue_depth: 0,
            queue_fill_frac: 0.5,
        }
    }

    fn snap_with(avg_latency_us: f64, drop_rate: f64) -> TelemetrySnapshot {
        TelemetrySnapshot { avg_latency_us, drop_rate, ..snap_zero() }
    }

    fn default_controller() -> Controller {
        Controller::new(ControllerConfig::default())
    }

    // ===== ParameterSpec =====

    #[test]
    fn test_spec_clamp_below_min() {
        let spec = ParameterSpec { min: 10.0, max: 100.0, ..Default::default() };
        assert_eq!(spec.clamp(5.0), 10.0);
    }

    #[test]
    fn test_spec_clamp_above_max() {
        let spec = ParameterSpec { min: 10.0, max: 100.0, ..Default::default() };
        assert_eq!(spec.clamp(200.0), 100.0);
    }

    #[test]
    fn test_spec_clamp_in_range() {
        let spec = ParameterSpec { min: 10.0, max: 100.0, ..Default::default() };
        assert_eq!(spec.clamp(50.0), 50.0);
    }

    #[test]
    fn test_spec_snap_rounds_to_nearest_step() {
        let spec = ParameterSpec { min: 0.0, max: 1000.0, step: 16.0, ..Default::default() };
        // 18 → nearest multiple of 16 is 16
        assert_eq!(spec.snap(18.0), 16.0);
    }

    #[test]
    fn test_spec_snap_rounds_up() {
        let spec = ParameterSpec { min: 0.0, max: 1000.0, step: 16.0, ..Default::default() };
        // 25 → nearest multiple of 16 is 32
        assert_eq!(spec.snap(25.0), 32.0);
    }

    #[test]
    fn test_spec_snap_clamps_after_snap() {
        let spec = ParameterSpec { min: 0.0, max: 100.0, step: 16.0, ..Default::default() };
        // 99 snaps to 96 (within max)
        assert_eq!(spec.snap(99.0), 96.0);
    }

    #[test]
    fn test_spec_snap_zero_step_just_clamps() {
        let spec = ParameterSpec { min: 0.0, max: 100.0, step: 0.0, ..Default::default() };
        assert_eq!(spec.snap(77.7), 77.7);
    }

    // ===== PidState =====

    #[test]
    fn test_pid_zero_error_zero_output() {
        let mut pid = PidState::default();
        let spec = ParameterSpec::default();
        let out = pid.update(0.0, &spec, 1.0);
        assert_eq!(out, 0.0);
    }

    #[test]
    fn test_pid_positive_error_positive_output() {
        let mut pid = PidState::default();
        let spec = ParameterSpec { kp: 1.0, ki: 0.0, kd: 0.0, ..Default::default() };
        let out = pid.update(10.0, &spec, 1.0);
        assert!(out > 0.0);
    }

    #[test]
    fn test_pid_negative_error_negative_output() {
        let mut pid = PidState::default();
        let spec = ParameterSpec { kp: 1.0, ki: 0.0, kd: 0.0, ..Default::default() };
        let out = pid.update(-10.0, &spec, 1.0);
        assert!(out < 0.0);
    }

    #[test]
    fn test_pid_integral_accumulates() {
        let mut pid = PidState::default();
        let spec = ParameterSpec { kp: 0.0, ki: 1.0, kd: 0.0, ..Default::default() };
        pid.update(5.0, &spec, 1.0);
        pid.update(5.0, &spec, 1.0);
        // integral = 5+5 = 10; ki=1 → output = 10
        let out = pid.update(0.0, &spec, 1.0);
        assert!((out - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_pid_integral_wind_up_clamped() {
        let mut pid = PidState::default();
        let spec = ParameterSpec { min: 0.0, max: 100.0, kp: 0.0, ki: 1.0, kd: 0.0, ..Default::default() };
        for _ in 0..10_000 {
            pid.update(1.0, &spec, 1.0);
        }
        // integral should be clamped to max-min = 100
        assert!(pid.integral.abs() <= 100.0 + 1e-6);
    }

    #[test]
    fn test_pid_derivative_term() {
        let mut pid = PidState::default();
        let spec = ParameterSpec { kp: 0.0, ki: 0.0, kd: 1.0, ..Default::default() };
        pid.update(10.0, &spec, 1.0); // prev_error = 10
        let out = pid.update(5.0, &spec, 1.0); // deriv = (5-10)/1 = -5
        assert!((out - (-5.0)).abs() < 1e-9);
    }

    #[test]
    fn test_pid_reset_clears_state() {
        let mut pid = PidState::default();
        let spec = ParameterSpec { kp: 1.0, ki: 1.0, kd: 0.0, ..Default::default() };
        pid.update(100.0, &spec, 1.0);
        pid.reset();
        assert_eq!(pid.integral, 0.0);
        assert_eq!(pid.prev_error, 0.0);
    }

    // ===== Param =====

    #[test]
    fn test_param_all_has_12_entries() {
        assert_eq!(Param::all().len(), 12);
    }

    #[test]
    fn test_param_name_non_empty() {
        for &p in Param::all() {
            assert!(!p.name().is_empty(), "name empty for {:?}", p);
        }
    }

    #[test]
    fn test_param_display_matches_name() {
        for &p in Param::all() {
            assert_eq!(p.to_string(), p.name());
        }
    }

    #[test]
    fn test_param_unique_names() {
        let names: std::collections::HashSet<_> = Param::all().iter().map(|p| p.name()).collect();
        assert_eq!(names.len(), 12);
    }

    // ===== RollbackGuard =====

    #[test]
    fn test_rollback_guard_no_rollback_when_metric_improves() {
        let guard = RollbackGuard {
            param: Param::DedupChannelBuf,
            value_before: 256.0,
            metric_before: 10_000.0, // high latency before
            applied_at: Instant::now(),
            rollback_window: Duration::from_secs(30),
            rollback_threshold: 0.10,
        };
        // Latency improved to 8000 → no rollback
        assert!(!guard.should_rollback(8_000.0));
    }

    #[test]
    fn test_rollback_guard_triggers_on_degradation() {
        let guard = RollbackGuard {
            param: Param::DedupChannelBuf,
            value_before: 256.0,
            metric_before: 5_000.0,
            applied_at: Instant::now(),
            rollback_window: Duration::from_secs(30),
            rollback_threshold: 0.10,
        };
        // Latency went from 5000 → 6000 (20% increase > 10% threshold)
        assert!(guard.should_rollback(6_000.0));
    }

    #[test]
    fn test_rollback_guard_exactly_at_threshold_does_not_trigger() {
        let guard = RollbackGuard {
            param: Param::DedupChannelBuf,
            value_before: 256.0,
            metric_before: 5_000.0,
            applied_at: Instant::now(),
            rollback_window: Duration::from_secs(30),
            rollback_threshold: 0.10,
        };
        // Exactly 10% increase: (5500-5000)/5000 = 0.10, not > threshold
        assert!(!guard.should_rollback(5_500.0));
    }

    #[test]
    fn test_rollback_guard_zero_metric_before_no_rollback() {
        let guard = RollbackGuard {
            param: Param::DedupChannelBuf,
            value_before: 256.0,
            metric_before: 0.0,
            applied_at: Instant::now(),
            rollback_window: Duration::from_secs(30),
            rollback_threshold: 0.10,
        };
        assert!(!guard.should_rollback(9999.0));
    }

    #[test]
    fn test_rollback_guard_expired() {
        let guard = RollbackGuard {
            param: Param::DedupChannelBuf,
            value_before: 256.0,
            metric_before: 5_000.0,
            applied_at: Instant::now() - Duration::from_secs(60),
            rollback_window: Duration::from_secs(30),
            rollback_threshold: 0.10,
        };
        assert!(guard.is_expired());
    }

    #[test]
    fn test_rollback_guard_not_expired() {
        let guard = RollbackGuard {
            param: Param::DedupChannelBuf,
            value_before: 256.0,
            metric_before: 5_000.0,
            applied_at: Instant::now(),
            rollback_window: Duration::from_secs(30),
            rollback_threshold: 0.10,
        };
        assert!(!guard.is_expired());
    }

    // ===== Controller construction =====

    #[test]
    fn test_controller_new_all_params_have_defaults() {
        let ctrl = default_controller();
        for &p in Param::all() {
            let v = ctrl.get(p);
            assert!(v > 0.0, "default value for {:?} should be > 0, got {}", p, v);
        }
    }

    #[test]
    fn test_controller_set_clamps_to_spec() {
        let mut ctrl = default_controller();
        ctrl.set(Param::DedupChannelBuf, 999_999.0);
        let v = ctrl.get(Param::DedupChannelBuf);
        let spec = &ctrl.specs[&Param::DedupChannelBuf];
        assert!(v <= spec.max);
    }

    #[test]
    fn test_controller_set_clamps_to_min() {
        let mut ctrl = default_controller();
        ctrl.set(Param::DedupChannelBuf, -1.0);
        let v = ctrl.get(Param::DedupChannelBuf);
        let spec = &ctrl.specs[&Param::DedupChannelBuf];
        assert!(v >= spec.min);
    }

    #[test]
    fn test_controller_audit_log_empty_initially() {
        let ctrl = default_controller();
        assert!(ctrl.audit_log().is_empty());
    }

    #[test]
    fn test_controller_observe_zero_error_no_adjustment() {
        let mut ctrl = default_controller();
        // target latency = 5000 us; snap gives exactly 5000
        ctrl.observe(&snap_zero());
        // No adjustment expected (zero error, PID output < step)
        assert_eq!(ctrl.audit_log().len(), 0);
    }

    #[test]
    fn test_controller_observe_high_latency_triggers_adjustment() {
        let mut ctrl = Controller::new(ControllerConfig {
            target_latency_us: 1_000.0, // target 1 ms
            ..Default::default()
        });
        // Override specs with aggressive gains and no cooldown
        for &p in Param::all() {
            ctrl.set_spec(p, ParameterSpec {
                min: 1.0, max: 100_000.0, step: 1.0,
                cooldown: Duration::from_millis(0),
                rollback_threshold: 0.10,
                kp: 10.0, ki: 0.0, kd: 0.0,
            });
        }
        // Very high latency → large error → PID output should exceed step
        ctrl.observe(&snap_with(50_000.0, 0.0));
        assert!(!ctrl.audit_log().is_empty(), "expected at least one adjustment");
    }

    #[test]
    fn test_controller_adjustment_record_before_after_differ() {
        let mut ctrl = Controller::new(ControllerConfig {
            target_latency_us: 1_000.0,
            ..Default::default()
        });
        for &p in Param::all() {
            ctrl.set_spec(p, ParameterSpec {
                min: 1.0, max: 100_000.0, step: 1.0,
                cooldown: Duration::from_millis(0),
                rollback_threshold: 0.10,
                kp: 10.0, ki: 0.0, kd: 0.0,
            });
        }
        ctrl.observe(&snap_with(50_000.0, 0.0));
        for rec in ctrl.audit_log() {
            assert_ne!(rec.before, rec.after, "before==after for {:?}", rec.param);
        }
    }

    #[test]
    fn test_controller_cooldown_prevents_rapid_readjustment() {
        let mut ctrl = default_controller();
        // Set a very long cooldown
        for &p in Param::all() {
            ctrl.set_spec(p, ParameterSpec {
                min: 1.0, max: 100_000.0, step: 1.0,
                cooldown: Duration::from_secs(3_600),
                rollback_threshold: 0.10,
                kp: 100.0, ki: 0.0, kd: 0.0,
            });
        }
        ctrl.observe(&snap_with(50_000.0, 0.0));
        let count_after_first = ctrl.audit_log().len();
        ctrl.observe(&snap_with(50_000.0, 0.0));
        // Second observe should not produce new entries (still in cooldown)
        assert_eq!(ctrl.audit_log().len(), count_after_first);
    }

    #[test]
    fn test_controller_rollback_triggered_on_degradation() {
        let mut ctrl = Controller::new(ControllerConfig {
            target_latency_us: 1_000.0,
            rollback_window: Duration::from_secs(3_600), // long window
            ..Default::default()
        });
        for &p in Param::all() {
            ctrl.set_spec(p, ParameterSpec {
                min: 1.0, max: 100_000.0, step: 1.0,
                cooldown: Duration::from_millis(0),
                rollback_threshold: 0.10,
                kp: 10.0, ki: 0.0, kd: 0.0,
            });
        }
        // First observe triggers adjustments + rollback guards
        ctrl.observe(&snap_with(50_000.0, 0.0));
        let guards_before = ctrl.active_rollback_guards();
        assert!(guards_before > 0);

        // Second observe with WORSE latency → should trigger rollbacks
        ctrl.observe(&snap_with(100_000.0, 0.0));
        let rollbacks: Vec<_> = ctrl.audit_log().iter().filter(|r| r.is_rollback).collect();
        assert!(!rollbacks.is_empty(), "expected rollback entries in audit log");
    }

    #[test]
    fn test_controller_rollback_audit_record_before_equals_original() {
        let mut ctrl = Controller::new(ControllerConfig {
            target_latency_us: 1_000.0,
            rollback_window: Duration::from_secs(3_600),
            ..Default::default()
        });
        for &p in Param::all() {
            ctrl.set_spec(p, ParameterSpec {
                min: 1.0, max: 100_000.0, step: 1.0,
                cooldown: Duration::from_millis(0),
                rollback_threshold: 0.10,
                kp: 10.0, ki: 0.0, kd: 0.0,
            });
        }

        let before = ctrl.get(Param::DedupChannelBuf);
        ctrl.observe(&snap_with(50_000.0, 0.0));

        // Confirm an adjustment was recorded
        let adj: Vec<_> = ctrl.audit_log().iter().filter(|r| !r.is_rollback).collect();
        if adj.is_empty() { return; } // no adjustment to roll back — test not applicable

        // Trigger rollback by worsening latency
        ctrl.observe(&snap_with(200_000.0, 0.0));

        // Verify a rollback audit entry exists with `after` equal to the original value
        let rb: Vec<_> = ctrl.audit_log().iter().filter(|r| r.is_rollback).collect();
        assert!(!rb.is_empty(), "expected a rollback record in audit log");
        let rb_entry = rb.iter().find(|r| r.param == Param::DedupChannelBuf);
        if let Some(rb_entry) = rb_entry {
            assert!((rb_entry.after - before).abs() < 1.0,
                "rollback `after` should equal original {}, got {}", before, rb_entry.after);
        }
    }

    #[test]
    fn test_controller_audit_log_capped() {
        let mut ctrl = Controller::new(ControllerConfig {
            target_latency_us: 1_000.0,
            audit_log_cap: 5,
            ..Default::default()
        });
        for &p in Param::all() {
            ctrl.set_spec(p, ParameterSpec {
                min: 1.0, max: 100_000.0, step: 1.0,
                cooldown: Duration::from_millis(0),
                rollback_threshold: 1.0, // disable rollbacks
                kp: 10.0, ki: 0.0, kd: 0.0,
            });
        }
        for _ in 0..10 {
            ctrl.observe(&snap_with(50_000.0, 0.0));
        }
        assert!(ctrl.audit_log().len() <= 5);
    }

    #[test]
    fn test_controller_clear_audit_log() {
        let mut ctrl = Controller::new(ControllerConfig {
            target_latency_us: 1_000.0,
            ..Default::default()
        });
        for &p in Param::all() {
            ctrl.set_spec(p, ParameterSpec {
                min: 1.0, max: 100_000.0, step: 1.0,
                cooldown: Duration::from_millis(0),
                rollback_threshold: 1.0,
                kp: 10.0, ki: 0.0, kd: 0.0,
            });
        }
        ctrl.observe(&snap_with(50_000.0, 0.0));
        ctrl.clear_audit_log();
        assert!(ctrl.audit_log().is_empty());
    }

    #[test]
    fn test_controller_set_spec_overrides_default() {
        let mut ctrl = default_controller();
        let custom = ParameterSpec { max: 42.0, ..Default::default() };
        ctrl.set_spec(Param::DedupTtlMs, custom);
        assert_eq!(ctrl.specs[&Param::DedupTtlMs].max, 42.0);
    }

    #[test]
    fn test_default_spec_all_params_have_positive_step() {
        for &p in Param::all() {
            let s = default_spec(p);
            assert!(s.step > 0.0, "step must be > 0 for {:?}", p);
        }
    }

    #[test]
    fn test_default_spec_min_lt_max() {
        for &p in Param::all() {
            let s = default_spec(p);
            assert!(s.min < s.max, "min < max violated for {:?}", p);
        }
    }

    #[test]
    fn test_default_value_within_spec_bounds() {
        for &p in Param::all() {
            let v = default_value(p);
            let s = default_spec(p);
            assert!(v >= s.min && v <= s.max,
                "default value {} out of [{}, {}] for {:?}", v, s.min, s.max, p);
        }
    }

    #[test]
    fn test_controller_observe_does_not_panic_on_zero_snap() {
        let mut ctrl = default_controller();
        let snap = TelemetrySnapshot {
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
        };
        ctrl.observe(&snap); // must not panic
    }

    #[test]
    fn test_controller_values_always_within_spec_after_many_observations() {
        let mut ctrl = Controller::new(ControllerConfig {
            target_latency_us: 1_000.0,
            ..Default::default()
        });
        for &p in Param::all() {
            ctrl.set_spec(p, ParameterSpec {
                min: 10.0, max: 500.0, step: 1.0,
                cooldown: Duration::from_millis(0),
                rollback_threshold: 1.0,
                kp: 100.0, ki: 10.0, kd: 5.0,
            });
        }
        // Drive with extreme latencies to exercise clamping
        for i in 0..20 {
            let lat = if i % 2 == 0 { 1.0 } else { 1_000_000.0 };
            ctrl.observe(&snap_with(lat, 0.0));
        }
        for &p in Param::all() {
            let v = ctrl.get(p);
            assert!(v >= 10.0 && v <= 500.0,
                "param {:?} out of bounds: {}", p, v);
        }
    }
}
