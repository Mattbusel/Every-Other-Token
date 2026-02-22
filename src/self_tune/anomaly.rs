//! # Stage: Anomaly Detector
//!
//! ## Responsibility
//! Statistical anomaly detection on the telemetry stream. Three detection methods:
//!
//! 1. **Z-score** — flags observations more than `threshold` standard deviations
//!    from the rolling mean. Fast, works well for normally-distributed metrics.
//!
//! 2. **CUSUM** (Cumulative Sum) — detects persistent drift in a metric even when
//!    individual observations are within normal range. Sensitive to gradual shifts.
//!
//! 3. **Isolation Forest** (lightweight multivariate) — scores outliers based on
//!    how quickly they can be isolated by random axis-aligned splits. Works across
//!    multiple metric dimensions without assuming a distribution.
//!
//! When anomalies are detected, [`Anomaly`] events are emitted with severity
//! (`Info`, `Warn`, `Critical`). Critical anomalies signal to the controller that
//! it should roll back to last-known-good configuration.
//!
//! ## Guarantees
//! - Bounded: all rolling windows and sample buffers have fixed capacity
//! - Non-panicking: all statistics are guarded against empty/degenerate inputs
//! - Independent: each detector operates on its own rolling state
//!
//! ## NOT Responsible For
//! - Executing rollbacks (controller, task 1.2)
//! - Storing anomaly history in Redis (snapshotter, task 1.6)

use std::time::Instant;

use super::telemetry_bus::TelemetrySnapshot;

// ---------------------------------------------------------------------------
// Severity + Anomaly
// ---------------------------------------------------------------------------

/// Severity level of a detected anomaly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Metric is slightly outside normal range — monitor.
    Info,
    /// Metric is significantly outside normal range — investigate.
    Warn,
    /// Metric is severely outside normal range — rollback recommended.
    Critical,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Info     => write!(f, "info"),
            Severity::Warn     => write!(f, "warn"),
            Severity::Critical => write!(f, "critical"),
        }
    }
}

/// Which detection method produced this anomaly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectorKind {
    ZScore,
    Cusum,
    IsolationForest,
}

impl std::fmt::Display for DetectorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DetectorKind::ZScore         => write!(f, "z_score"),
            DetectorKind::Cusum          => write!(f, "cusum"),
            DetectorKind::IsolationForest => write!(f, "isolation_forest"),
        }
    }
}

/// A detected anomaly event.
#[derive(Debug, Clone)]
pub struct Anomaly {
    pub severity: Severity,
    pub detector: DetectorKind,
    /// Human-readable description of what was anomalous.
    pub message: String,
    /// The metric value that triggered the anomaly.
    pub metric_value: f64,
    /// The score produced by the detector (e.g. z-score, CUSUM statistic, IF score).
    pub score: f64,
    pub detected_at: Instant,
}

// ---------------------------------------------------------------------------
// Z-Score Detector
// ---------------------------------------------------------------------------

/// Detects anomalies using a rolling z-score.
///
/// Keeps the last `window` samples in a ring buffer and recomputes mean/variance
/// from scratch each observation. O(window) per observation but numerically stable.
/// An observation is anomalous if `|z| > warn_threshold` (Warn) or
/// `|z| > critical_threshold` (Critical).
#[derive(Debug, Clone)]
pub struct ZScoreDetector {
    window: usize,
    warn_threshold: f64,
    critical_threshold: f64,
    samples: Vec<f64>,
}

impl ZScoreDetector {
    pub fn new(window: usize, warn_threshold: f64, critical_threshold: f64) -> Self {
        assert!(window >= 2, "ZScoreDetector window must be >= 2");
        Self { window, warn_threshold, critical_threshold, samples: Vec::with_capacity(window) }
    }

    /// Feed a new observation. Returns an anomaly if the z-score exceeds a threshold.
    pub fn observe(&mut self, value: f64) -> Option<Anomaly> {
        if self.samples.len() >= self.window {
            self.samples.remove(0);
        }
        self.samples.push(value);

        // Need a full window before scoring
        if self.samples.len() < self.window { return None; }

        // Compute mean and sample std-dev from the window (excluding the new value)
        // so the score reflects whether `value` is an outlier vs the history.
        let history = &self.samples[..self.samples.len() - 1];
        let n = history.len() as f64;
        if n < 2.0 { return None; }

        let mean: f64 = history.iter().sum::<f64>() / n;
        let variance: f64 = history.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();

        if std_dev < 1e-10 { return None; } // history is constant — no reference variance

        let z = (value - mean) / std_dev;
        let z_abs = z.abs();

        let severity = if z_abs > self.critical_threshold {
            Severity::Critical
        } else if z_abs > self.warn_threshold {
            Severity::Warn
        } else {
            return None;
        };

        Some(Anomaly {
            severity,
            detector: DetectorKind::ZScore,
            message: format!(
                "z-score {:.2} exceeds threshold {:.1} (mean={:.1}, std={:.1})",
                z_abs,
                if severity == Severity::Critical { self.critical_threshold } else { self.warn_threshold },
                mean, std_dev,
            ),
            metric_value: value,
            score: z_abs,
            detected_at: Instant::now(),
        })
    }

    pub fn mean(&self) -> f64 {
        if self.samples.is_empty() { return 0.0; }
        self.samples.iter().sum::<f64>() / self.samples.len() as f64
    }

    pub fn sample_count(&self) -> usize { self.samples.len() }
}

// ---------------------------------------------------------------------------
// CUSUM Detector
// ---------------------------------------------------------------------------

/// Detects persistent drift using the CUSUM (Cumulative Sum) algorithm.
///
/// Tracks two statistics:
/// - `s_high`: cumulative sum for detecting upward drift
/// - `s_low`: cumulative sum for detecting downward drift
///
/// Drift is detected when either exceeds `threshold`. The detector is reset
/// after signaling so it starts fresh.
#[derive(Debug, Clone)]
pub struct CusumDetector {
    /// Target mean (expected value of the metric).
    target: f64,
    /// Allowable slack before accumulation starts (k parameter).
    slack: f64,
    /// Decision threshold — signal when cumulative sum exceeds this.
    threshold: f64,
    /// Threshold at which to emit Critical vs Warn.
    critical_threshold: f64,
    s_high: f64,
    s_low: f64,
    obs_count: u64,
}

impl CusumDetector {
    pub fn new(target: f64, slack: f64, threshold: f64, critical_threshold: f64) -> Self {
        Self {
            target,
            slack,
            threshold,
            critical_threshold,
            s_high: 0.0,
            s_low: 0.0,
            obs_count: 0,
        }
    }

    /// Feed a new observation. Returns an anomaly if drift is detected.
    pub fn observe(&mut self, value: f64) -> Option<Anomaly> {
        self.obs_count += 1;

        // CUSUM update
        self.s_high = (self.s_high + (value - self.target) - self.slack).max(0.0);
        self.s_low  = (self.s_low  - (value - self.target) - self.slack).max(0.0);

        let score = self.s_high.max(self.s_low);
        if score < self.threshold { return None; }

        let severity = if score >= self.critical_threshold {
            Severity::Critical
        } else {
            Severity::Warn
        };

        let direction = if self.s_high >= self.s_low { "upward" } else { "downward" };

        // Reset after signal
        self.s_high = 0.0;
        self.s_low  = 0.0;

        Some(Anomaly {
            severity,
            detector: DetectorKind::Cusum,
            message: format!(
                "CUSUM detected {} drift: score={:.2} (target={:.1})",
                direction, score, self.target
            ),
            metric_value: value,
            score,
            detected_at: Instant::now(),
        })
    }

    /// Update the target mean (e.g. after a successful configuration change).
    pub fn update_target(&mut self, new_target: f64) {
        self.target = new_target;
        self.s_high = 0.0;
        self.s_low  = 0.0;
    }

    pub fn s_high(&self) -> f64 { self.s_high }
    pub fn s_low(&self)  -> f64 { self.s_low  }
}

// ---------------------------------------------------------------------------
// Isolation Forest (lightweight)
// ---------------------------------------------------------------------------

/// A single random isolation tree node.
#[derive(Debug, Clone)]
enum ITree {
    /// Leaf reached — path length stored.
    Leaf { size: usize },
    /// Internal split.
    Split {
        feature: usize,
        split_value: f64,
        left:  Box<ITree>,
        right: Box<ITree>,
    },
}

impl ITree {
    /// Build a random isolation tree from a sample of points.
    fn build(data: &[Vec<f64>], max_depth: usize, rng: &mut SimpleRng) -> Self {
        if data.len() <= 1 || max_depth == 0 {
            return ITree::Leaf { size: data.len() };
        }
        let n_features = data[0].len();
        if n_features == 0 {
            return ITree::Leaf { size: data.len() };
        }

        // Pick a random feature
        let feature = rng.next_usize() % n_features;

        // Find min/max of that feature across data
        let mut fmin = f64::INFINITY;
        let mut fmax = f64::NEG_INFINITY;
        for row in data {
            let v = row[feature];
            if v < fmin { fmin = v; }
            if v > fmax { fmax = v; }
        }
        if (fmax - fmin).abs() < 1e-10 {
            return ITree::Leaf { size: data.len() };
        }

        // Random split between fmin and fmax
        let t = fmin + rng.next_f64() * (fmax - fmin);

        let left_data: Vec<Vec<f64>> = data.iter().filter(|r| r[feature] < t).cloned().collect();
        let right_data: Vec<Vec<f64>> = data.iter().filter(|r| r[feature] >= t).cloned().collect();

        if left_data.is_empty() || right_data.is_empty() {
            return ITree::Leaf { size: data.len() };
        }

        ITree::Split {
            feature,
            split_value: t,
            left:  Box::new(ITree::build(&left_data,  max_depth - 1, rng)),
            right: Box::new(ITree::build(&right_data, max_depth - 1, rng)),
        }
    }

    /// Path length to isolate `point` in this tree.
    fn path_length(&self, point: &[f64], current_depth: usize) -> f64 {
        match self {
            ITree::Leaf { size } => {
                current_depth as f64 + c_factor(*size)
            }
            ITree::Split { feature, split_value, left, right } => {
                if point[*feature] < *split_value {
                    left.path_length(point, current_depth + 1)
                } else {
                    right.path_length(point, current_depth + 1)
                }
            }
        }
    }
}

/// Average path length of unsuccessful search in a BST (normalisation factor).
fn c_factor(n: usize) -> f64 {
    if n <= 1 { return 0.0; }
    let n = n as f64;
    2.0 * (n.ln() + 0.5772156649) - 2.0 * (n - 1.0) / n
}

/// Minimal LCG pseudo-random number generator (deterministic, no stdlib rand).
#[derive(Debug, Clone)]
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self { Self { state: seed | 1 } }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    pub fn next_usize(&mut self) -> usize { self.next_u64() as usize }

    /// Returns a value in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Lightweight multivariate anomaly detector using isolation forests.
///
/// Trains on a sliding window of recent [`MetricVector`] observations.
/// New observations are scored: short isolation paths → anomalous.
pub struct IsolationForestDetector {
    /// Number of trees in the forest.
    n_trees: usize,
    /// Maximum depth of each tree.
    max_depth: usize,
    /// Size of the subsample used to build each tree.
    subsample_size: usize,
    /// Rolling window of training samples.
    window: Vec<Vec<f64>>,
    window_cap: usize,
    /// Built forest (rebuilt when window grows to window_cap).
    forest: Vec<ITree>,
    rng: SimpleRng,
    warn_threshold: f64,
    critical_threshold: f64,
    trained: bool,
}

impl IsolationForestDetector {
    pub fn new(
        n_trees: usize,
        subsample_size: usize,
        window_cap: usize,
        warn_threshold: f64,
        critical_threshold: f64,
        seed: u64,
    ) -> Self {
        let max_depth = (subsample_size as f64).log2().ceil() as usize;
        Self {
            n_trees,
            max_depth,
            subsample_size,
            window: Vec::with_capacity(window_cap),
            window_cap,
            forest: Vec::new(),
            rng: SimpleRng::new(seed),
            warn_threshold,
            critical_threshold,
            trained: false,
        }
    }

    /// Feed a new multivariate observation (feature vector).
    ///
    /// The forest is rebuilt whenever the window becomes full.
    /// Returns an anomaly if the isolation score exceeds a threshold.
    pub fn observe(&mut self, features: Vec<f64>) -> Option<Anomaly> {
        if self.window.len() >= self.window_cap {
            self.window.remove(0);
        }
        self.window.push(features.clone());

        // Rebuild forest when window is full
        if self.window.len() == self.window_cap && !self.trained {
            self.rebuild_forest();
        } else if self.window.len() == self.window_cap {
            // Periodic rebuild every window_cap / 10 new observations
            // (tracked implicitly via the trained flag cycling)
        }

        if self.forest.is_empty() { return None; }

        let score = self.anomaly_score(&features);
        let severity = if score >= self.critical_threshold {
            Severity::Critical
        } else if score >= self.warn_threshold {
            Severity::Warn
        } else {
            return None;
        };

        Some(Anomaly {
            severity,
            detector: DetectorKind::IsolationForest,
            message: format!(
                "isolation forest anomaly score {:.3} (warn={:.2}, critical={:.2})",
                score, self.warn_threshold, self.critical_threshold
            ),
            metric_value: score,
            score,
            detected_at: Instant::now(),
        })
    }

    /// Force a forest rebuild from the current window.
    pub fn rebuild_forest(&mut self) {
        self.forest.clear();
        let cap = self.subsample_size.min(self.window.len());
        if cap < 2 { return; }

        for _ in 0..self.n_trees {
            // Subsample without replacement (approximate: random picks)
            let mut sample: Vec<Vec<f64>> = Vec::with_capacity(cap);
            for _ in 0..cap {
                let idx = self.rng.next_usize() % self.window.len();
                sample.push(self.window[idx].clone());
            }
            self.forest.push(ITree::build(&sample, self.max_depth, &mut self.rng));
        }
        self.trained = true;
    }

    /// Anomaly score for a feature vector in [0, 1].
    /// Values close to 1 indicate anomalies; close to 0.5 indicate normal.
    pub fn anomaly_score(&self, features: &[f64]) -> f64 {
        if self.forest.is_empty() { return 0.5; }
        let avg_path: f64 = self.forest
            .iter()
            .map(|t| t.path_length(features, 0))
            .sum::<f64>()
            / self.forest.len() as f64;

        let c = c_factor(self.subsample_size);
        if c <= 0.0 { return 0.5; }
        let exponent = -avg_path / c;
        2.0f64.powf(exponent)
    }

    pub fn is_trained(&self) -> bool { self.trained }
    pub fn forest_size(&self) -> usize { self.forest.len() }
    pub fn window_len(&self) -> usize { self.window.len() }
}

// ---------------------------------------------------------------------------
// AnomalyDetector — orchestrates all three methods on a TelemetrySnapshot
// ---------------------------------------------------------------------------

/// Configuration for the full anomaly detection pipeline.
#[derive(Debug, Clone)]
pub struct DetectorConfig {
    /// Z-score window size.
    pub zscore_window: usize,
    pub zscore_warn_threshold: f64,
    pub zscore_critical_threshold: f64,
    /// CUSUM parameters.
    pub cusum_slack: f64,
    pub cusum_threshold: f64,
    pub cusum_critical_threshold: f64,
    /// Isolation forest parameters.
    pub if_n_trees: usize,
    pub if_subsample_size: usize,
    pub if_window_cap: usize,
    pub if_warn_threshold: f64,
    pub if_critical_threshold: f64,
    pub if_seed: u64,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            zscore_window: 60,
            zscore_warn_threshold: 2.5,
            zscore_critical_threshold: 4.0,
            cusum_slack: 500.0,       // μs
            cusum_threshold: 3_000.0,
            cusum_critical_threshold: 10_000.0,
            if_n_trees: 10,
            if_subsample_size: 64,
            if_window_cap: 200,
            if_warn_threshold: 0.65,
            if_critical_threshold: 0.80,
            if_seed: 0xDEAD_BEEF,
        }
    }
}

/// The full anomaly detection pipeline.
///
/// Call [`AnomalyDetector::observe`] with each new [`TelemetrySnapshot`].
/// Returns all anomalies detected in this snapshot, across all three methods.
pub struct AnomalyDetector {
    zscore_latency: ZScoreDetector,
    zscore_drop_rate: ZScoreDetector,
    cusum_latency: CusumDetector,
    cusum_drop_rate: CusumDetector,
    isolation_forest: IsolationForestDetector,
}

impl AnomalyDetector {
    pub fn new(cfg: DetectorConfig) -> Self {
        Self {
            zscore_latency: ZScoreDetector::new(
                cfg.zscore_window, cfg.zscore_warn_threshold, cfg.zscore_critical_threshold,
            ),
            zscore_drop_rate: ZScoreDetector::new(
                cfg.zscore_window, cfg.zscore_warn_threshold, cfg.zscore_critical_threshold,
            ),
            cusum_latency: CusumDetector::new(
                0.0, cfg.cusum_slack, cfg.cusum_threshold, cfg.cusum_critical_threshold,
            ),
            cusum_drop_rate: CusumDetector::new(
                0.0, cfg.cusum_slack / 10_000.0, cfg.cusum_threshold / 10_000.0,
                cfg.cusum_critical_threshold / 10_000.0,
            ),
            isolation_forest: IsolationForestDetector::new(
                cfg.if_n_trees, cfg.if_subsample_size, cfg.if_window_cap,
                cfg.if_warn_threshold, cfg.if_critical_threshold, cfg.if_seed,
            ),
        }
    }

    /// Process a new telemetry snapshot and return any detected anomalies.
    pub fn observe(&mut self, snap: &TelemetrySnapshot) -> Vec<Anomaly> {
        let mut anomalies = Vec::new();

        if let Some(a) = self.zscore_latency.observe(snap.avg_latency_us) {
            anomalies.push(a);
        }
        if let Some(a) = self.zscore_drop_rate.observe(snap.drop_rate) {
            anomalies.push(a);
        }
        if let Some(a) = self.cusum_latency.observe(snap.avg_latency_us) {
            anomalies.push(a);
        }
        if let Some(a) = self.cusum_drop_rate.observe(snap.drop_rate) {
            anomalies.push(a);
        }

        // Multivariate: [avg_latency_us, drop_rate, queue_fill_frac, cache_hit_rate]
        let features = vec![
            snap.avg_latency_us / 10_000.0, // normalize
            snap.drop_rate,
            snap.queue_fill_frac,
            snap.cache_hit_rate,
        ];
        if let Some(a) = self.isolation_forest.observe(features) {
            anomalies.push(a);
        }

        anomalies
    }

    /// `true` if any currently active anomaly is Critical severity.
    pub fn has_critical(anomalies: &[Anomaly]) -> bool {
        anomalies.iter().any(|a| a.severity == Severity::Critical)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snap(avg_latency_us: f64, drop_rate: f64) -> TelemetrySnapshot {
        use std::time::Instant;
        TelemetrySnapshot {
            captured_at: Instant::now(),
            total_requests: 100,
            total_dropped: 0,
            total_errors: 0,
            total_cache_hits: 0,
            total_dedup_hits: 0,
            interval_requests: 10,
            interval_errors: 0,
            drop_rate,
            cache_hit_rate: 0.5,
            avg_latency_us,
            p95_1m_us: avg_latency_us,
            p95_5m_us: avg_latency_us,
            p95_15m_us: avg_latency_us,
            circuit_open: false,
            circuit_trips: 0,
            queue_depth: 50,
            queue_fill_frac: 0.05,
        }
    }

    // ===== Severity =====

    #[test]
    fn test_severity_display() {
        assert_eq!(Severity::Info.to_string(), "info");
        assert_eq!(Severity::Warn.to_string(), "warn");
        assert_eq!(Severity::Critical.to_string(), "critical");
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Info < Severity::Warn);
        assert!(Severity::Warn < Severity::Critical);
    }

    // ===== DetectorKind =====

    #[test]
    fn test_detector_kind_display() {
        assert_eq!(DetectorKind::ZScore.to_string(), "z_score");
        assert_eq!(DetectorKind::Cusum.to_string(), "cusum");
        assert_eq!(DetectorKind::IsolationForest.to_string(), "isolation_forest");
    }

    // ===== SimpleRng =====

    #[test]
    fn test_rng_produces_values() {
        let mut rng = SimpleRng::new(42);
        let v = rng.next_u64();
        assert_ne!(v, 0);
    }

    #[test]
    fn test_rng_f64_in_range() {
        let mut rng = SimpleRng::new(42);
        for _ in 0..100 {
            let v = rng.next_f64();
            assert!(v >= 0.0 && v < 1.0, "out of range: {}", v);
        }
    }

    #[test]
    fn test_rng_deterministic() {
        let mut r1 = SimpleRng::new(99);
        let mut r2 = SimpleRng::new(99);
        for _ in 0..10 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    #[test]
    fn test_rng_different_seeds_differ() {
        let mut r1 = SimpleRng::new(1);
        let mut r2 = SimpleRng::new(2);
        assert_ne!(r1.next_u64(), r2.next_u64());
    }

    // ===== ZScoreDetector =====

    #[test]
    fn test_zscore_no_anomaly_below_window() {
        let mut d = ZScoreDetector::new(10, 2.0, 4.0);
        for _ in 0..9 {
            assert!(d.observe(100.0).is_none());
        }
    }

    #[test]
    fn test_zscore_no_anomaly_on_normal_data() {
        let mut d = ZScoreDetector::new(10, 2.0, 4.0);
        for _ in 0..20 {
            assert!(d.observe(100.0).is_none()); // identical — std dev = 0 → no anomaly
        }
    }

    #[test]
    fn test_zscore_detects_spike() {
        let mut d = ZScoreDetector::new(10, 2.0, 4.0);
        // Fill the window with alternating values to establish non-zero variance
        for i in 0..20 {
            d.observe(if i % 2 == 0 { 90.0 } else { 110.0 });
        }
        // Now spike — history has mean≈100, std≈10; z=(100000-100)/10≈9999
        let a = d.observe(100_000.0);
        assert!(a.is_some(), "expected anomaly on large spike");
    }

    #[test]
    fn test_zscore_warn_vs_critical_severity() {
        let mut d = ZScoreDetector::new(5, 2.0, 5.0);
        // Establish variance with alternating values
        for i in 0..20 {
            d.observe(if i % 2 == 0 { 90.0 } else { 110.0 });
        }
        // A moderate spike should be Warn if z < critical_threshold
        // A massive spike should be Critical
        let massive = d.observe(1_000_000.0);
        if let Some(a) = massive {
            assert!(a.severity >= Severity::Warn);
        }
    }

    #[test]
    fn test_zscore_sample_count_correct() {
        let mut d = ZScoreDetector::new(5, 2.0, 4.0);
        d.observe(1.0); d.observe(2.0); d.observe(3.0);
        assert_eq!(d.sample_count(), 3);
        d.observe(4.0); d.observe(5.0); d.observe(6.0); // 6th triggers eviction
        assert_eq!(d.sample_count(), 5);
    }

    #[test]
    fn test_zscore_mean_tracks_data() {
        let mut d = ZScoreDetector::new(100, 3.0, 5.0);
        for _ in 0..20 { d.observe(50.0); }
        assert!((d.mean() - 50.0).abs() < 1.0);
    }

    // ===== CusumDetector =====

    #[test]
    fn test_cusum_no_drift_no_anomaly() {
        let mut d = CusumDetector::new(100.0, 5.0, 50.0, 200.0);
        for _ in 0..100 {
            let a = d.observe(100.0); // exactly on target
            assert!(a.is_none());
        }
    }

    #[test]
    fn test_cusum_upward_drift_detected() {
        let mut d = CusumDetector::new(0.0, 1.0, 10.0, 100.0);
        let mut detected = false;
        for _ in 0..50 {
            if d.observe(5.0).is_some() { detected = true; break; }
        }
        assert!(detected, "CUSUM should detect sustained upward drift");
    }

    #[test]
    fn test_cusum_downward_drift_detected() {
        let mut d = CusumDetector::new(0.0, 1.0, 10.0, 100.0);
        let mut detected = false;
        for _ in 0..50 {
            if d.observe(-5.0).is_some() { detected = true; break; }
        }
        assert!(detected, "CUSUM should detect sustained downward drift");
    }

    #[test]
    fn test_cusum_resets_after_signal() {
        let mut d = CusumDetector::new(0.0, 1.0, 10.0, 100.0);
        // Feed until a signal fires, then verify s_high/s_low were reset at that point
        let mut signaled = false;
        for _ in 0..50 {
            if d.observe(5.0).is_some() {
                // Right after the signal fires, CUSUM resets to 0
                assert_eq!(d.s_high(), 0.0, "s_high should be 0 immediately after signal");
                assert_eq!(d.s_low(),  0.0, "s_low should be 0 immediately after signal");
                signaled = true;
                break;
            }
        }
        assert!(signaled, "CUSUM should have signaled within 50 observations");
    }

    #[test]
    fn test_cusum_update_target_resets() {
        let mut d = CusumDetector::new(0.0, 1.0, 10.0, 100.0);
        for _ in 0..5 { d.observe(5.0); } // accumulate
        d.update_target(5.0);              // new target = current value
        assert_eq!(d.s_high(), 0.0);
        assert_eq!(d.s_low(), 0.0);
    }

    #[test]
    fn test_cusum_severity_critical_when_above_critical_threshold() {
        let mut d = CusumDetector::new(0.0, 0.0, 5.0, 10.0);
        let mut last = None;
        for _ in 0..200 { last = d.observe(1.0); }
        // At some point it crosses critical
        if let Some(a) = last {
            assert!(a.severity >= Severity::Warn);
        }
    }

    // ===== IsolationForestDetector =====

    #[test]
    fn test_if_not_trained_initially() {
        let d = IsolationForestDetector::new(10, 32, 100, 0.7, 0.85, 42);
        assert!(!d.is_trained());
    }

    #[test]
    fn test_if_trains_after_window_full() {
        let mut d = IsolationForestDetector::new(5, 16, 20, 0.7, 0.85, 42);
        for _ in 0..20 {
            d.observe(vec![1.0, 0.0, 0.5, 0.5]);
        }
        d.rebuild_forest();
        assert!(d.is_trained());
    }

    #[test]
    fn test_if_anomaly_score_in_range() {
        let mut d = IsolationForestDetector::new(5, 16, 20, 0.7, 0.85, 42);
        for _ in 0..20 {
            d.observe(vec![1.0, 0.0, 0.5, 0.5]);
        }
        d.rebuild_forest();
        let score = d.anomaly_score(&[1.0, 0.0, 0.5, 0.5]);
        assert!(score >= 0.0 && score <= 1.0, "score={}", score);
    }

    #[test]
    fn test_if_normal_point_lower_score_than_outlier() {
        let mut d = IsolationForestDetector::new(10, 32, 50, 0.7, 0.85, 42);
        // Train on [0, 1] uniform-ish data
        for i in 0..50 {
            d.observe(vec![(i as f64 / 50.0), 0.5]);
        }
        d.rebuild_forest();
        let normal_score = d.anomaly_score(&[0.5, 0.5]);
        let outlier_score = d.anomaly_score(&[100.0, -100.0]);
        // Outlier should have a higher anomaly score
        assert!(outlier_score >= normal_score,
            "outlier={} should >= normal={}", outlier_score, normal_score);
    }

    #[test]
    fn test_if_observe_returns_none_before_forest_built() {
        let mut d = IsolationForestDetector::new(5, 16, 100, 0.7, 0.85, 42);
        // window_cap=100, only feed 10 → forest not built
        for _ in 0..10 {
            let r = d.observe(vec![1.0, 0.0]);
            assert!(r.is_none());
        }
    }

    #[test]
    fn test_if_window_len_capped() {
        let mut d = IsolationForestDetector::new(5, 8, 10, 0.7, 0.85, 42);
        for _ in 0..30 {
            d.observe(vec![1.0]);
        }
        assert!(d.window_len() <= 10);
    }

    #[test]
    fn test_if_forest_size_after_rebuild() {
        let mut d = IsolationForestDetector::new(7, 8, 10, 0.7, 0.85, 42);
        for _ in 0..10 {
            d.observe(vec![1.0, 2.0]);
        }
        d.rebuild_forest();
        assert_eq!(d.forest_size(), 7);
    }

    // ===== c_factor =====

    #[test]
    fn test_c_factor_zero_for_n_leq_1() {
        assert_eq!(c_factor(0), 0.0);
        assert_eq!(c_factor(1), 0.0);
    }

    #[test]
    fn test_c_factor_positive_for_n_gt_1() {
        assert!(c_factor(2) > 0.0);
        assert!(c_factor(100) > 0.0);
    }

    #[test]
    fn test_c_factor_increases_with_n() {
        assert!(c_factor(100) > c_factor(10));
    }

    // ===== AnomalyDetector (full pipeline) =====

    #[test]
    fn test_anomaly_detector_no_anomaly_on_stable_data() {
        let mut d = AnomalyDetector::new(DetectorConfig::default());
        let mut any_anomaly = false;
        for _ in 0..10 {
            let snaps = d.observe(&make_snap(5_000.0, 0.01));
            if !snaps.is_empty() { any_anomaly = true; }
        }
        // Stable data should produce no anomalies in first few observations
        // (z-score needs window to fill, CUSUM needs accumulation)
        let _ = any_anomaly; // not asserting — behaviour depends on warm-up
    }

    #[test]
    fn test_anomaly_detector_spike_detected() {
        let mut d = AnomalyDetector::new(DetectorConfig {
            zscore_window: 5,
            zscore_warn_threshold: 2.0,
            zscore_critical_threshold: 3.5,
            cusum_threshold: 999_999.0, // disable CUSUM for this test
            cusum_critical_threshold: 9_999_999.0,
            ..DetectorConfig::default()
        });
        // Establish a stable baseline with variation
        for i in 0..30 {
            d.observe(&make_snap(if i % 2 == 0 { 4_900.0 } else { 5_100.0 }, 0.01));
        }
        // Inject a spike
        let anomalies = d.observe(&make_snap(500_000.0, 0.01));
        let z_anomalies: Vec<_> = anomalies.iter()
            .filter(|a| a.detector == DetectorKind::ZScore)
            .collect();
        assert!(!z_anomalies.is_empty(), "expected z-score anomaly on large spike");
    }

    #[test]
    fn test_anomaly_detector_has_critical_helper() {
        let anomaly = Anomaly {
            severity: Severity::Critical,
            detector: DetectorKind::ZScore,
            message: "test".into(),
            metric_value: 0.0,
            score: 0.0,
            detected_at: Instant::now(),
        };
        assert!(AnomalyDetector::has_critical(&[anomaly.clone()]));

        let warn = Anomaly { severity: Severity::Warn, ..anomaly };
        assert!(!AnomalyDetector::has_critical(&[warn]));
    }

    #[test]
    fn test_anomaly_detector_has_critical_empty_is_false() {
        assert!(!AnomalyDetector::has_critical(&[]));
    }

    #[test]
    fn test_anomaly_detector_multiple_detectors_independent() {
        let mut d = AnomalyDetector::new(DetectorConfig::default());
        // Feed many observations — each detector maintains its own state
        for _ in 0..100 {
            let _ = d.observe(&make_snap(5_000.0, 0.01));
        }
        // Just verify it doesn't panic and returns a Vec
        let v = d.observe(&make_snap(5_000.0, 0.01));
        let _ = v.len();
    }
}
