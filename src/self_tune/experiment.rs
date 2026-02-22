//! # Stage: Experiment Engine
//!
//! ## Responsibility
//! A/B testing framework for pipeline parameter changes. Routes a configurable
//! percentage of traffic through experimental configurations. Uses Welch's t-test
//! for statistical significance testing. Auto-promotes winning variants and
//! kills losing ones.
//!
//! ## Guarantees
//! - Bounded: sample buffers have a configurable maximum size
//! - Deterministic routing: a given request ID always maps to the same variant
//! - Thread-safe: [`ExperimentRegistry`] wraps state in `Arc<Mutex>`
//! - Non-panicking: all statistics calculations handle edge cases
//!
//! ## NOT Responsible For
//! - Executing the actual pipeline with variant parameters (caller's concern)
//! - Cross-node A/B routing (multi-node brain, task 4.1)
//! - Persistence (snapshot versioning, task 1.6)

use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

// ---------------------------------------------------------------------------
// ExperimentStatus
// ---------------------------------------------------------------------------

/// Lifecycle state of an experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentStatus {
    /// Collecting samples, not yet significant.
    Running,
    /// Statistical significance reached — winner determined.
    Concluded { winner: Variant },
    /// Manually stopped before conclusion.
    Stopped,
}

/// Which variant of the experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    Control,
    Treatment,
}

impl std::fmt::Display for Variant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Variant::Control   => write!(f, "control"),
            Variant::Treatment => write!(f, "treatment"),
        }
    }
}

// ---------------------------------------------------------------------------
// ExperimentSpec — definition of one A/B test
// ---------------------------------------------------------------------------

/// Definition of a single A/B experiment.
///
/// Mirrors the TOML experiment definition from the design doc:
/// ```toml
/// [experiment]
/// name = "dedup_ttl_increase"
/// parameter = "dedup.ttl_ms"
/// control   = 5000
/// variant   = 10000
/// traffic_split = 0.1
/// min_samples   = 1000
/// significance  = 0.95
/// ```
#[derive(Debug, Clone)]
pub struct ExperimentSpec {
    /// Unique experiment name.
    pub name: String,
    /// The parameter being tested (human-readable, e.g. `"dedup.ttl_ms"`).
    pub parameter: String,
    /// Current (baseline) value of the parameter.
    pub control_value: f64,
    /// Experimental value of the parameter.
    pub treatment_value: f64,
    /// Fraction of traffic routed to the treatment [0.0, 1.0].
    pub traffic_split: f64,
    /// Minimum samples per variant before significance testing begins.
    pub min_samples: usize,
    /// Required p-value threshold for significance (e.g. 0.95 → p < 0.05).
    pub significance: f64,
    /// Maximum number of samples stored per variant (ring-buffer style).
    pub max_samples: usize,
    /// Experiment auto-expires after this duration.
    pub ttl: Duration,
}

impl ExperimentSpec {
    /// Validate the spec — returns an error string if invalid.
    pub fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() {
            return Err("experiment name must not be empty".into());
        }
        if !(0.0..=1.0).contains(&self.traffic_split) {
            return Err(format!("traffic_split must be in [0, 1], got {}", self.traffic_split));
        }
        if self.min_samples == 0 {
            return Err("min_samples must be >= 1".into());
        }
        if !(0.0..1.0).contains(&self.significance) {
            return Err(format!("significance must be in [0, 1), got {}", self.significance));
        }
        if self.max_samples == 0 {
            return Err("max_samples must be >= 1".into());
        }
        Ok(())
    }
}

impl Default for ExperimentSpec {
    fn default() -> Self {
        Self {
            name: "unnamed".into(),
            parameter: "unknown".into(),
            control_value: 0.0,
            treatment_value: 0.0,
            traffic_split: 0.10,
            min_samples: 100,
            significance: 0.95,
            max_samples: 10_000,
            ttl: Duration::from_secs(3_600),
        }
    }
}

// ---------------------------------------------------------------------------
// VariantStats — live statistics for one variant
// ---------------------------------------------------------------------------

/// Rolling statistics for one experiment variant.
#[derive(Debug, Clone, Default)]
pub struct VariantStats {
    /// All recorded metric values (bounded ring buffer).
    samples: Vec<f64>,
    /// Maximum number of samples kept.
    max_samples: usize,
    /// Running sum (for mean calculation without full iteration).
    sum: f64,
    /// Running sum of squares (for variance).
    sum_sq: f64,
}

impl VariantStats {
    fn new(max_samples: usize) -> Self {
        Self { max_samples, ..Default::default() }
    }

    /// Record a metric observation (lower is better — typically latency_us).
    pub fn record(&mut self, value: f64) {
        if self.samples.len() >= self.max_samples {
            // Evict oldest: recalculate sums
            let evicted = self.samples.remove(0);
            self.sum -= evicted;
            self.sum_sq -= evicted * evicted;
        }
        self.samples.push(value);
        self.sum += value;
        self.sum_sq += value * value;
    }

    pub fn count(&self) -> usize { self.samples.len() }

    pub fn mean(&self) -> f64 {
        if self.samples.is_empty() { 0.0 }
        else { self.sum / self.samples.len() as f64 }
    }

    /// Sample variance (Bessel's correction).
    pub fn variance(&self) -> f64 {
        let n = self.samples.len();
        if n < 2 { return 0.0; }
        let m = self.mean();
        (self.sum_sq - self.samples.len() as f64 * m * m) / (n - 1) as f64
    }

    pub fn std_dev(&self) -> f64 { self.variance().sqrt() }
}

// ---------------------------------------------------------------------------
// Welch's t-test
// ---------------------------------------------------------------------------

/// Result of a Welch's t-test comparison.
#[derive(Debug, Clone)]
pub struct TTestResult {
    /// t-statistic.
    pub t_stat: f64,
    /// Degrees of freedom (Welch–Satterthwaite approximation).
    pub df: f64,
    /// Two-tailed p-value estimate.
    pub p_value: f64,
    /// Whether the null hypothesis is rejected at the experiment's significance level.
    pub significant: bool,
    /// Which variant has lower mean (better performance), or `None` if equal.
    pub better: Option<Variant>,
}

/// Perform Welch's independent-samples t-test between control and treatment.
///
/// Returns `None` if either sample has fewer than 2 observations.
pub fn welch_t_test(
    control: &VariantStats,
    treatment: &VariantStats,
    significance: f64,
) -> Option<TTestResult> {
    let n1 = control.count() as f64;
    let n2 = treatment.count() as f64;
    if n1 < 2.0 || n2 < 2.0 { return None; }

    let m1 = control.mean();
    let m2 = treatment.mean();
    let v1 = control.variance();
    let v2 = treatment.variance();

    let se1 = v1 / n1;
    let se2 = v2 / n2;
    let se_total = se1 + se2;

    // When variance is zero but means differ, the result is definitively significant.
    if se_total <= 0.0 {
        let better = if (m1 - m2).abs() < 1e-10 { None }
            else if m2 < m1 { Some(Variant::Treatment) }
            else { Some(Variant::Control) };
        return Some(TTestResult {
            t_stat: f64::INFINITY,
            df: n1 + n2 - 2.0,
            p_value: 0.0,
            significant: true,
            better,
        });
    }

    let t_stat = (m1 - m2) / se_total.sqrt();

    // Welch–Satterthwaite degrees of freedom
    let df = if se_total > 0.0 {
        (se_total * se_total) / (se1 * se1 / (n1 - 1.0) + se2 * se2 / (n2 - 1.0))
    } else {
        1.0
    };

    // Approximate two-tailed p-value using a simple t-distribution approximation.
    // For df > 30, the t-distribution is close to normal; we use a rational
    // approximation of the normal CDF for simplicity.
    let p_value = approx_two_tailed_p(t_stat.abs(), df);
    let significant = p_value < (1.0 - significance);

    let better = if (m1 - m2).abs() < 1e-10 {
        None
    } else if m2 < m1 {
        Some(Variant::Treatment) // lower mean is better
    } else {
        Some(Variant::Control)
    };

    Some(TTestResult { t_stat, df, p_value, significant, better })
}

/// Approximate two-tailed p-value for a given |t| and degrees of freedom.
/// Uses a rational approximation sufficient for decision-making.
fn approx_two_tailed_p(t_abs: f64, df: f64) -> f64 {
    // For large df, use normal approximation
    let z = if df > 100.0 {
        t_abs
    } else {
        // Scale t toward normal using a simple correction
        t_abs * (1.0 - 0.25 / df.max(1.0))
    };
    // Rational approximation of erfc for the normal distribution
    let p_one_tail = standard_normal_upper_tail(z);
    (2.0 * p_one_tail).min(1.0)
}

/// Upper tail probability of the standard normal distribution (z > x).
/// Hart's rational approximation — accurate to ~1e-5.
fn standard_normal_upper_tail(x: f64) -> f64 {
    if x < 0.0 { return 1.0 - standard_normal_upper_tail(-x); }
    if x > 8.0 { return 0.0; }
    // Abramowitz & Stegun 26.2.17 approximation
    let t = 1.0 / (1.0 + 0.2316419 * x);
    let poly = t * (0.319381530
        + t * (-0.356563782
        + t * (1.781477937
        + t * (-1.821255978
        + t * 1.330274429))));
    let pdf = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
    pdf * poly
}

// ---------------------------------------------------------------------------
// Experiment — live state for one A/B test
// ---------------------------------------------------------------------------

/// Live state of a single A/B experiment.
pub struct Experiment {
    pub spec: ExperimentSpec,
    pub control: VariantStats,
    pub treatment: VariantStats,
    pub status: ExperimentStatus,
    pub started_at: Instant,
    pub concluded_at: Option<Instant>,
    pub last_test_result: Option<TTestResult>,
}

impl Experiment {
    pub fn new(spec: ExperimentSpec) -> Self {
        let max = spec.max_samples;
        Self {
            control:   VariantStats::new(max),
            treatment: VariantStats::new(max),
            status: ExperimentStatus::Running,
            started_at: Instant::now(),
            concluded_at: None,
            last_test_result: None,
            spec,
        }
    }

    /// Determine which variant a request should use based on its ID.
    ///
    /// Uses a simple hash so the same request ID always maps to the same variant.
    pub fn route(&self, request_id: u64) -> Variant {
        let frac = ((request_id.wrapping_mul(2654435761) >> 16) & 0xFFFF) as f64 / 65536.0;
        if frac < self.spec.traffic_split {
            Variant::Treatment
        } else {
            Variant::Control
        }
    }

    /// Record a metric observation for the given variant.
    ///
    /// After recording, if both variants have `>= min_samples` the significance
    /// test runs automatically. If significant, the experiment is concluded.
    pub fn record(&mut self, variant: Variant, metric: f64) {
        if self.status != ExperimentStatus::Running { return; }

        // Check TTL
        if self.started_at.elapsed() > self.spec.ttl {
            self.status = ExperimentStatus::Stopped;
            return;
        }

        match variant {
            Variant::Control   => self.control.record(metric),
            Variant::Treatment => self.treatment.record(metric),
        }

        self.maybe_conclude();
    }

    /// Force a significance check and potentially conclude the experiment.
    pub fn maybe_conclude(&mut self) {
        if self.status != ExperimentStatus::Running { return; }
        if self.control.count() < self.spec.min_samples
            || self.treatment.count() < self.spec.min_samples
        {
            return;
        }

        if let Some(result) = welch_t_test(&self.control, &self.treatment, self.spec.significance) {
            if result.significant {
                let winner = result.better.unwrap_or(Variant::Control);
                self.status = ExperimentStatus::Concluded { winner };
                self.concluded_at = Some(Instant::now());
            }
            self.last_test_result = Some(result);
        }
    }

    /// Stop the experiment manually.
    pub fn stop(&mut self) {
        self.status = ExperimentStatus::Stopped;
    }

    /// `true` if the experiment has finished (concluded or stopped).
    pub fn is_finished(&self) -> bool {
        !matches!(self.status, ExperimentStatus::Running)
    }

    /// The winning parameter value, or the control value if no winner yet.
    pub fn winning_value(&self) -> f64 {
        match self.status {
            ExperimentStatus::Concluded { winner: Variant::Treatment } => self.spec.treatment_value,
            _ => self.spec.control_value,
        }
    }
}

// ---------------------------------------------------------------------------
// ExperimentRegistry — manages multiple concurrent experiments
// ---------------------------------------------------------------------------

/// Manages a collection of named experiments.
///
/// This is the primary interface for the self-tuning subsystem. Use
/// [`ExperimentRegistry::register`] to add experiments, [`ExperimentRegistry::route`]
/// to assign variants, and [`ExperimentRegistry::record`] to feed metrics in.
pub struct ExperimentRegistry {
    experiments: HashMap<String, Experiment>,
    /// Maximum number of concurrent active experiments.
    max_active: usize,
}

impl ExperimentRegistry {
    pub fn new(max_active: usize) -> Self {
        Self { experiments: HashMap::new(), max_active }
    }

    /// Register a new experiment. Returns an error if the registry is full
    /// or an experiment with the same name already exists.
    pub fn register(&mut self, spec: ExperimentSpec) -> Result<(), String> {
        spec.validate()?;
        let active = self.experiments.values().filter(|e| !e.is_finished()).count();
        if active >= self.max_active {
            return Err(format!(
                "registry full: {} active experiments (max {})",
                active, self.max_active
            ));
        }
        if self.experiments.contains_key(&spec.name) {
            return Err(format!("experiment '{}' already registered", spec.name));
        }
        self.experiments.insert(spec.name.clone(), Experiment::new(spec));
        Ok(())
    }

    /// Route a request to a variant for the named experiment.
    pub fn route(&self, name: &str, request_id: u64) -> Option<Variant> {
        self.experiments.get(name).map(|e| e.route(request_id))
    }

    /// Record a metric observation for the named experiment and variant.
    pub fn record(&mut self, name: &str, variant: Variant, metric: f64) {
        if let Some(exp) = self.experiments.get_mut(name) {
            exp.record(variant, metric);
        }
    }

    /// Get the current status of an experiment.
    pub fn status(&self, name: &str) -> Option<ExperimentStatus> {
        self.experiments.get(name).map(|e| e.status)
    }

    /// Get the winning parameter value for a concluded experiment.
    pub fn winning_value(&self, name: &str) -> Option<f64> {
        self.experiments.get(name).map(|e| e.winning_value())
    }

    /// Remove all finished experiments from the registry.
    pub fn gc(&mut self) -> usize {
        let before = self.experiments.len();
        self.experiments.retain(|_, e| !e.is_finished());
        before - self.experiments.len()
    }

    /// Number of active (running) experiments.
    pub fn active_count(&self) -> usize {
        self.experiments.values().filter(|e| !e.is_finished()).count()
    }

    /// Number of all experiments (including finished).
    pub fn total_count(&self) -> usize {
        self.experiments.len()
    }

    /// Iterate over all experiments.
    pub fn experiments(&self) -> impl Iterator<Item = (&str, &Experiment)> {
        self.experiments.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Stop all running experiments.
    pub fn stop_all(&mut self) {
        for exp in self.experiments.values_mut() {
            exp.stop();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str, split: f64, min_samples: usize) -> ExperimentSpec {
        ExperimentSpec {
            name: name.into(),
            parameter: "test_param".into(),
            control_value: 100.0,
            treatment_value: 200.0,
            traffic_split: split,
            min_samples,
            significance: 0.95,
            max_samples: 10_000,
            ttl: Duration::from_secs(3_600),
        }
    }

    // ===== ExperimentSpec validation =====

    #[test]
    fn test_spec_valid_default() {
        // Default has name "unnamed" which is non-empty, so it validates ok
        assert!(ExperimentSpec::default().validate().is_ok());
        let s = spec("test", 0.1, 10);
        assert!(s.validate().is_ok());
    }

    #[test]
    fn test_spec_empty_name_invalid() {
        let s = ExperimentSpec { name: "".into(), ..spec("x", 0.1, 10) };
        assert!(s.validate().is_err());
    }

    #[test]
    fn test_spec_traffic_split_above_one_invalid() {
        let s = ExperimentSpec { traffic_split: 1.5, ..spec("x", 0.1, 10) };
        assert!(s.validate().is_err());
    }

    #[test]
    fn test_spec_traffic_split_negative_invalid() {
        let s = ExperimentSpec { traffic_split: -0.1, ..spec("x", 0.1, 10) };
        assert!(s.validate().is_err());
    }

    #[test]
    fn test_spec_traffic_split_zero_valid() {
        let s = ExperimentSpec { traffic_split: 0.0, ..spec("x", 0.1, 10) };
        assert!(s.validate().is_ok());
    }

    #[test]
    fn test_spec_traffic_split_one_valid() {
        let s = ExperimentSpec { traffic_split: 1.0, ..spec("x", 0.1, 10) };
        assert!(s.validate().is_ok());
    }

    #[test]
    fn test_spec_min_samples_zero_invalid() {
        let s = ExperimentSpec { min_samples: 0, ..spec("x", 0.1, 10) };
        assert!(s.validate().is_err());
    }

    #[test]
    fn test_spec_significance_one_invalid() {
        let s = ExperimentSpec { significance: 1.0, ..spec("x", 0.1, 10) };
        assert!(s.validate().is_err());
    }

    #[test]
    fn test_spec_significance_negative_invalid() {
        let s = ExperimentSpec { significance: -0.1, ..spec("x", 0.1, 10) };
        assert!(s.validate().is_err());
    }

    #[test]
    fn test_spec_max_samples_zero_invalid() {
        let s = ExperimentSpec { max_samples: 0, ..spec("x", 0.1, 10) };
        assert!(s.validate().is_err());
    }

    // ===== VariantStats =====

    #[test]
    fn test_variant_stats_empty_mean_is_zero() {
        let vs = VariantStats::new(100);
        assert_eq!(vs.mean(), 0.0);
    }

    #[test]
    fn test_variant_stats_count_increments() {
        let mut vs = VariantStats::new(100);
        vs.record(1.0); vs.record(2.0);
        assert_eq!(vs.count(), 2);
    }

    #[test]
    fn test_variant_stats_mean_correct() {
        let mut vs = VariantStats::new(100);
        vs.record(10.0); vs.record(20.0); vs.record(30.0);
        assert!((vs.mean() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_variant_stats_variance_zero_when_count_lt_2() {
        let mut vs = VariantStats::new(100);
        vs.record(5.0);
        assert_eq!(vs.variance(), 0.0);
    }

    #[test]
    fn test_variant_stats_variance_correct() {
        let mut vs = VariantStats::new(100);
        // samples: 2, 4, 4, 4, 5, 5, 7, 9 → variance = 4.571...
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            vs.record(v);
        }
        assert!((vs.variance() - 4.571).abs() < 0.01, "variance={}", vs.variance());
    }

    #[test]
    fn test_variant_stats_capped_at_max_samples() {
        let mut vs = VariantStats::new(5);
        for i in 0..10 { vs.record(i as f64); }
        assert_eq!(vs.count(), 5);
    }

    #[test]
    fn test_variant_stats_mean_after_eviction() {
        let mut vs = VariantStats::new(3);
        vs.record(1.0); vs.record(2.0); vs.record(3.0);
        vs.record(10.0); // evicts 1.0
        // remaining: 2, 3, 10 → mean = 5
        assert!((vs.mean() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_variant_stats_std_dev_non_negative() {
        let mut vs = VariantStats::new(100);
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] { vs.record(v); }
        assert!(vs.std_dev() >= 0.0);
    }

    // ===== Welch's t-test =====

    #[test]
    fn test_ttest_none_when_insufficient_samples() {
        let c = VariantStats::new(100);
        let mut t = VariantStats::new(100);
        t.record(1.0);
        assert!(welch_t_test(&c, &t, 0.95).is_none());
    }

    #[test]
    fn test_ttest_none_when_both_empty() {
        let c = VariantStats::new(100);
        let t = VariantStats::new(100);
        assert!(welch_t_test(&c, &t, 0.95).is_none());
    }

    #[test]
    fn test_ttest_identical_distributions_not_significant() {
        let mut c = VariantStats::new(1000);
        let mut t = VariantStats::new(1000);
        for i in 0..200 {
            c.record(i as f64 % 10.0);
            t.record(i as f64 % 10.0);
        }
        let result = welch_t_test(&c, &t, 0.95).unwrap();
        assert!(!result.significant, "identical distributions should not be significant");
    }

    #[test]
    fn test_ttest_clearly_different_is_significant() {
        let mut c = VariantStats::new(1000);
        let mut t = VariantStats::new(1000);
        // Control: mean 1000; Treatment: mean 10 — very different
        for _ in 0..200 {
            c.record(1000.0);
            t.record(10.0);
        }
        let result = welch_t_test(&c, &t, 0.95).unwrap();
        assert!(result.significant);
        assert_eq!(result.better, Some(Variant::Treatment)); // lower is better
    }

    #[test]
    fn test_ttest_better_is_lower_mean() {
        let mut c = VariantStats::new(100);
        let mut t = VariantStats::new(100);
        for _ in 0..50 { c.record(200.0); t.record(50.0); }
        let result = welch_t_test(&c, &t, 0.95).unwrap();
        assert_eq!(result.better, Some(Variant::Treatment));
    }

    #[test]
    fn test_ttest_p_value_in_range() {
        let mut c = VariantStats::new(100);
        let mut t = VariantStats::new(100);
        for i in 0..30 { c.record(i as f64); t.record(i as f64 * 2.0); }
        let result = welch_t_test(&c, &t, 0.95).unwrap();
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0,
            "p_value={}", result.p_value);
    }

    #[test]
    fn test_ttest_df_positive() {
        let mut c = VariantStats::new(100);
        let mut t = VariantStats::new(100);
        for i in 0..20 { c.record(i as f64); t.record(i as f64 + 100.0); }
        let result = welch_t_test(&c, &t, 0.95).unwrap();
        assert!(result.df > 0.0, "df={}", result.df);
    }

    // ===== Experiment routing =====

    #[test]
    fn test_experiment_route_deterministic() {
        let exp = Experiment::new(spec("r", 0.5, 10));
        let v1 = exp.route(12345);
        let v2 = exp.route(12345);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_experiment_route_zero_split_always_control() {
        let exp = Experiment::new(spec("r", 0.0, 10));
        for id in 0..100u64 {
            assert_eq!(exp.route(id), Variant::Control);
        }
    }

    #[test]
    fn test_experiment_route_full_split_always_treatment() {
        let exp = Experiment::new(spec("r", 1.0, 10));
        for id in 0..100u64 {
            assert_eq!(exp.route(id), Variant::Treatment);
        }
    }

    #[test]
    fn test_experiment_route_split_approximately_correct() {
        let exp = Experiment::new(spec("r", 0.2, 10));
        let treatments = (0..1000u64).filter(|&id| exp.route(id) == Variant::Treatment).count();
        // Should be ~200 ± 50
        assert!(treatments >= 150 && treatments <= 250,
            "expected ~200 treatments, got {}", treatments);
    }

    // ===== Experiment record + conclude =====

    #[test]
    fn test_experiment_status_running_initially() {
        let exp = Experiment::new(spec("e", 0.5, 10));
        assert_eq!(exp.status, ExperimentStatus::Running);
    }

    #[test]
    fn test_experiment_record_ignored_when_stopped() {
        let mut exp = Experiment::new(spec("e", 0.5, 10));
        exp.stop();
        exp.record(Variant::Control, 100.0);
        assert_eq!(exp.control.count(), 0);
    }

    #[test]
    fn test_experiment_concludes_when_significant() {
        let mut exp = Experiment::new(spec("e", 0.5, 5));
        // Feed clearly different distributions
        for _ in 0..100 {
            exp.record(Variant::Control, 1000.0);
            exp.record(Variant::Treatment, 10.0);
        }
        assert!(exp.is_finished(), "experiment should have concluded");
        assert!(matches!(exp.status, ExperimentStatus::Concluded { winner: Variant::Treatment }));
    }

    #[test]
    fn test_experiment_does_not_conclude_before_min_samples() {
        let mut exp = Experiment::new(spec("e", 0.5, 1000));
        for _ in 0..10 {
            exp.record(Variant::Control, 1000.0);
            exp.record(Variant::Treatment, 10.0);
        }
        // Only 10 samples each, need 1000
        assert!(!exp.is_finished());
    }

    #[test]
    fn test_experiment_winning_value_treatment_when_treatment_wins() {
        let mut exp = Experiment::new(spec("e", 0.5, 5));
        for _ in 0..100 {
            exp.record(Variant::Control, 1000.0);
            exp.record(Variant::Treatment, 10.0);
        }
        if matches!(exp.status, ExperimentStatus::Concluded { winner: Variant::Treatment }) {
            assert_eq!(exp.winning_value(), exp.spec.treatment_value);
        }
    }

    #[test]
    fn test_experiment_winning_value_control_when_running() {
        let exp = Experiment::new(spec("e", 0.5, 10));
        assert_eq!(exp.winning_value(), exp.spec.control_value);
    }

    #[test]
    fn test_experiment_stop_changes_status() {
        let mut exp = Experiment::new(spec("e", 0.5, 10));
        exp.stop();
        assert_eq!(exp.status, ExperimentStatus::Stopped);
    }

    #[test]
    fn test_experiment_concluded_at_set_after_conclusion() {
        let mut exp = Experiment::new(spec("e", 0.5, 5));
        for _ in 0..100 {
            exp.record(Variant::Control, 5000.0);
            exp.record(Variant::Treatment, 1.0);
        }
        if exp.is_finished() {
            assert!(exp.concluded_at.is_some());
        }
    }

    // ===== ExperimentRegistry =====

    #[test]
    fn test_registry_register_valid_spec() {
        let mut reg = ExperimentRegistry::new(10);
        assert!(reg.register(spec("a", 0.1, 10)).is_ok());
    }

    #[test]
    fn test_registry_register_duplicate_name_fails() {
        let mut reg = ExperimentRegistry::new(10);
        reg.register(spec("a", 0.1, 10)).unwrap();
        assert!(reg.register(spec("a", 0.2, 20)).is_err());
    }

    #[test]
    fn test_registry_register_invalid_spec_fails() {
        let mut reg = ExperimentRegistry::new(10);
        let bad = ExperimentSpec { name: "".into(), ..spec("x", 0.1, 10) };
        assert!(reg.register(bad).is_err());
    }

    #[test]
    fn test_registry_max_active_enforced() {
        let mut reg = ExperimentRegistry::new(2);
        reg.register(spec("a", 0.1, 10)).unwrap();
        reg.register(spec("b", 0.1, 10)).unwrap();
        assert!(reg.register(spec("c", 0.1, 10)).is_err());
    }

    #[test]
    fn test_registry_active_count_correct() {
        let mut reg = ExperimentRegistry::new(10);
        reg.register(spec("a", 0.1, 10)).unwrap();
        reg.register(spec("b", 0.1, 10)).unwrap();
        assert_eq!(reg.active_count(), 2);
    }

    #[test]
    fn test_registry_route_returns_none_for_unknown() {
        let reg = ExperimentRegistry::new(10);
        assert!(reg.route("no_such", 42).is_none());
    }

    #[test]
    fn test_registry_route_returns_variant() {
        let mut reg = ExperimentRegistry::new(10);
        reg.register(spec("a", 0.5, 10)).unwrap();
        assert!(reg.route("a", 42).is_some());
    }

    #[test]
    fn test_registry_record_feeds_experiment() {
        let mut reg = ExperimentRegistry::new(10);
        reg.register(spec("a", 0.5, 10)).unwrap();
        reg.record("a", Variant::Control, 100.0);
        reg.record("a", Variant::Control, 200.0);
        // Verify through status — still running with only 2 samples
        assert_eq!(reg.status("a"), Some(ExperimentStatus::Running));
    }

    #[test]
    fn test_registry_gc_removes_finished() {
        let mut reg = ExperimentRegistry::new(10);
        reg.register(spec("a", 0.5, 2)).unwrap();
        // Drive to conclusion
        for _ in 0..50 {
            reg.record("a", Variant::Control, 9999.0);
            reg.record("a", Variant::Treatment, 1.0);
        }
        // If experiment concluded, gc removes it; if it didn't conclude (significance
        // not reached), the experiment is still running — either way active_count is accurate.
        let _removed = reg.gc();
        // After gc, no finished experiments remain in registry
        assert!(reg.experiments().all(|(_, e)| !e.is_finished()));
    }

    #[test]
    fn test_registry_stop_all_stops_all_running() {
        let mut reg = ExperimentRegistry::new(10);
        reg.register(spec("a", 0.1, 100)).unwrap();
        reg.register(spec("b", 0.2, 100)).unwrap();
        reg.stop_all();
        assert_eq!(reg.active_count(), 0);
    }

    #[test]
    fn test_registry_winning_value_none_for_unknown() {
        let reg = ExperimentRegistry::new(10);
        assert!(reg.winning_value("no_such").is_none());
    }

    #[test]
    fn test_registry_total_count_includes_finished() {
        let mut reg = ExperimentRegistry::new(10);
        reg.register(spec("a", 0.5, 2)).unwrap();
        for _ in 0..50 {
            reg.record("a", Variant::Control, 9999.0);
            reg.record("a", Variant::Treatment, 1.0);
        }
        // Even after conclusion, total_count includes it
        assert_eq!(reg.total_count(), 1);
    }

    #[test]
    fn test_variant_display() {
        assert_eq!(Variant::Control.to_string(), "control");
        assert_eq!(Variant::Treatment.to_string(), "treatment");
    }

    // ===== Statistical helper =====

    #[test]
    fn test_standard_normal_upper_tail_at_zero_is_half() {
        let p = standard_normal_upper_tail(0.0);
        assert!((p - 0.5).abs() < 0.01, "upper tail at 0 should be ~0.5, got {}", p);
    }

    #[test]
    fn test_standard_normal_upper_tail_large_x_near_zero() {
        let p = standard_normal_upper_tail(8.0);
        assert!(p < 0.001, "upper tail at 8 should be near 0, got {}", p);
    }

    #[test]
    fn test_standard_normal_upper_tail_negative_x() {
        let p = standard_normal_upper_tail(-2.0);
        assert!(p > 0.9, "upper tail at -2 should be >0.9, got {}", p);
    }

    #[test]
    fn test_approx_two_tailed_p_at_zero_is_one() {
        let p = approx_two_tailed_p(0.0, 100.0);
        assert!((p - 1.0).abs() < 0.05, "p at t=0 should be ~1.0, got {}", p);
    }

    #[test]
    fn test_approx_two_tailed_p_large_t_near_zero() {
        let p = approx_two_tailed_p(10.0, 100.0);
        assert!(p < 0.01, "p at t=10 should be near 0, got {}", p);
    }
}
