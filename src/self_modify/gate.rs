//! # Stage: Validation Gate (Task 2.2)
//!
//! ## Responsibility
//! Immune system for self-modifications.  Every proposed change runs through a
//! configurable sequence of checks before it can be accepted.
//!
//! Check pipeline (in order):
//! 1. `cargo test` — all unit and integration tests must pass
//! 2. `cargo clippy` — zero warnings / errors
//! 3. Benchmark regression — criterion run, <5% regression on named benchmarks
//! 4. Integration smoke test — configurable smoke-test command passes
//! 5. Staging metric validation — post-deploy metrics within acceptable bounds
//!
//! Trust levels:
//! - `0` — Always require human review; auto-gate only, never auto-merge
//! - `1` — Auto-merge when all checks pass
//! - `2` — Auto-deploy (merge + deploy) when all checks pass
//!
//! ## Guarantees
//! - Non-panicking: all check results are `Result`-typed
//! - Auditable: every run produces a `ValidationReport` with per-check results
//! - Configurable: each check can be individually enabled / disabled

use std::time::Duration;

// ---------------------------------------------------------------------------
// Check result types
// ---------------------------------------------------------------------------

/// The outcome of a single validation check.
#[derive(Debug, Clone, PartialEq)]
pub enum CheckStatus {
    Passed,
    Failed { reason: String },
    Skipped { reason: String },
}

impl CheckStatus {
    pub fn is_passed(&self) -> bool {
        matches!(self, CheckStatus::Passed)
    }

    pub fn is_failed(&self) -> bool {
        matches!(self, CheckStatus::Failed { .. })
    }

    pub fn is_skipped(&self) -> bool {
        matches!(self, CheckStatus::Skipped { .. })
    }
}

impl std::fmt::Display for CheckStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckStatus::Passed => write!(f, "PASS"),
            CheckStatus::Failed { reason } => write!(f, "FAIL: {}", reason),
            CheckStatus::Skipped { reason } => write!(f, "SKIP: {}", reason),
        }
    }
}

/// Result of one named check.
#[derive(Debug, Clone)]
pub struct CheckResult {
    pub name: String,
    pub status: CheckStatus,
    /// Wall-clock time the check took.
    pub duration: Duration,
    /// Any extra diagnostic output (stdout/stderr snippets, metric values).
    pub details: Vec<String>,
}

impl CheckResult {
    pub fn passed(name: impl Into<String>, duration: Duration) -> Self {
        Self { name: name.into(), status: CheckStatus::Passed, duration, details: vec![] }
    }

    pub fn failed(name: impl Into<String>, reason: impl Into<String>, duration: Duration) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Failed { reason: reason.into() },
            duration,
            details: vec![],
        }
    }

    pub fn skipped(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Skipped { reason: reason.into() },
            duration: Duration::ZERO,
            details: vec![],
        }
    }

    pub fn with_details(mut self, details: Vec<String>) -> Self {
        self.details = details;
        self
    }
}

// ---------------------------------------------------------------------------
// BenchmarkRegression
// ---------------------------------------------------------------------------

/// A single benchmark measurement for regression tracking.
#[derive(Debug, Clone)]
pub struct BenchmarkSample {
    pub name: String,
    /// Baseline time in nanoseconds.
    pub baseline_ns: f64,
    /// Current time in nanoseconds.
    pub current_ns: f64,
}

impl BenchmarkSample {
    /// Fractional regression (positive = slower, negative = improvement).
    pub fn regression_fraction(&self) -> f64 {
        if self.baseline_ns == 0.0 {
            0.0
        } else {
            (self.current_ns - self.baseline_ns) / self.baseline_ns
        }
    }

    pub fn pct_change(&self) -> f64 {
        self.regression_fraction() * 100.0
    }

    pub fn is_regression(&self, threshold: f64) -> bool {
        self.regression_fraction() > threshold
    }
}

// ---------------------------------------------------------------------------
// StagingMetricCheck
// ---------------------------------------------------------------------------

/// A named metric observed after staging deployment, with expected bounds.
#[derive(Debug, Clone)]
pub struct StagingMetric {
    pub name: String,
    pub observed: f64,
    /// Acceptable range (inclusive).
    pub min: f64,
    pub max: f64,
}

impl StagingMetric {
    pub fn passes(&self) -> bool {
        self.observed >= self.min && self.observed <= self.max
    }
}

// ---------------------------------------------------------------------------
// TrustLevel
// ---------------------------------------------------------------------------

/// Controls how much autonomy the gate grants on a passing run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrustLevel {
    /// All checks must pass but a human must approve before merging.
    ReviewRequired = 0,
    /// Auto-merge when all enabled checks pass.
    AutoMerge = 1,
    /// Auto-merge and auto-deploy when all enabled checks pass.
    AutoDeploy = 2,
}

impl std::fmt::Display for TrustLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrustLevel::ReviewRequired => write!(f, "review_required"),
            TrustLevel::AutoMerge => write!(f, "auto_merge"),
            TrustLevel::AutoDeploy => write!(f, "auto_deploy"),
        }
    }
}

// ---------------------------------------------------------------------------
// GateConfig
// ---------------------------------------------------------------------------

/// Configuration for one run of the validation gate.
#[derive(Debug, Clone)]
pub struct GateConfig {
    pub trust_level: TrustLevel,
    /// Enable the `cargo test` check.
    pub run_tests: bool,
    /// Enable the `cargo clippy` check.
    pub run_clippy: bool,
    /// Enable criterion benchmark regression check.
    pub run_benchmarks: bool,
    /// Enable integration smoke test.
    pub run_smoke: bool,
    /// Enable staging metric validation.
    pub run_staging_metrics: bool,
    /// Maximum allowed benchmark regression fraction (default 0.05 = 5%).
    pub bench_regression_threshold: f64,
    /// Optional override for test command (default: `cargo test --all-features`).
    pub test_command: Option<String>,
    /// Optional override for clippy command.
    pub clippy_command: Option<String>,
    /// Optional smoke test command.
    pub smoke_command: Option<String>,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            trust_level: TrustLevel::ReviewRequired,
            run_tests: true,
            run_clippy: true,
            run_benchmarks: true,
            run_smoke: false,
            run_staging_metrics: false,
            bench_regression_threshold: 0.05,
            test_command: None,
            clippy_command: None,
            smoke_command: None,
        }
    }
}

// ---------------------------------------------------------------------------
// ValidationReport
// ---------------------------------------------------------------------------

/// The complete output of one gate run.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Identifier for the change being validated (e.g. git commit SHA or task ID).
    pub change_id: String,
    pub config: GateConfig,
    pub checks: Vec<CheckResult>,
    /// Overall outcome: true iff all non-skipped checks passed.
    pub overall_passed: bool,
    pub total_duration: Duration,
    /// Action the gate recommends based on outcome + trust level.
    pub recommended_action: RecommendedAction,
}

/// What the gate recommends doing with this change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecommendedAction {
    /// All checks passed; trust level allows autonomous merge.
    AutoMerge,
    /// All checks passed; trust level allows autonomous merge + deploy.
    AutoDeploy,
    /// All checks passed but trust level requires human sign-off.
    AwaitReview,
    /// One or more checks failed; change must be rejected or fixed.
    Reject { failed_checks: Vec<String> },
}

impl std::fmt::Display for RecommendedAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecommendedAction::AutoMerge => write!(f, "auto_merge"),
            RecommendedAction::AutoDeploy => write!(f, "auto_deploy"),
            RecommendedAction::AwaitReview => write!(f, "await_review"),
            RecommendedAction::Reject { failed_checks } => {
                write!(f, "reject({})", failed_checks.join(","))
            }
        }
    }
}

impl ValidationReport {
    pub fn failed_checks(&self) -> Vec<&CheckResult> {
        self.checks.iter().filter(|c| c.status.is_failed()).collect()
    }

    pub fn passed_checks(&self) -> Vec<&CheckResult> {
        self.checks.iter().filter(|c| c.status.is_passed()).collect()
    }

    pub fn skipped_checks(&self) -> Vec<&CheckResult> {
        self.checks.iter().filter(|c| c.status.is_skipped()).collect()
    }

    /// Return a compact text summary for logs.
    pub fn summary(&self) -> String {
        format!(
            "Gate[{}]: {} — {} passed, {} failed, {} skipped — action={}",
            self.change_id,
            if self.overall_passed { "PASS" } else { "FAIL" },
            self.passed_checks().len(),
            self.failed_checks().len(),
            self.skipped_checks().len(),
            self.recommended_action,
        )
    }
}

// ---------------------------------------------------------------------------
// ValidationGate
// ---------------------------------------------------------------------------

/// The validation gate.  In production, `run()` would shell out to cargo etc.
/// In this implementation the check execution is injectable via `CheckRunner`
/// so that tests can fully exercise the gate logic without touching the filesystem.
pub struct ValidationGate {
    config: GateConfig,
}

impl ValidationGate {
    pub fn new(config: GateConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &GateConfig {
        &self.config
    }

    pub fn set_trust_level(&mut self, level: TrustLevel) {
        self.config.trust_level = level;
    }

    /// Run the gate using the provided `runner` for actual check execution.
    /// This is the testable entry-point.
    pub fn run(&self, change_id: impl Into<String>, runner: &dyn CheckRunner) -> ValidationReport {
        let change_id = change_id.into();
        let mut checks = Vec::new();
        let start = std::time::Instant::now();

        // 1. Tests
        if self.config.run_tests {
            let cmd = self
                .config
                .test_command
                .as_deref()
                .unwrap_or("cargo test --all-features");
            checks.push(runner.run_tests(cmd));
        } else {
            checks.push(CheckResult::skipped("tests", "disabled in config"));
        }

        // 2. Clippy
        if self.config.run_clippy {
            let cmd = self
                .config
                .clippy_command
                .as_deref()
                .unwrap_or("cargo clippy --all-features -- -D warnings");
            checks.push(runner.run_clippy(cmd));
        } else {
            checks.push(CheckResult::skipped("clippy", "disabled in config"));
        }

        // 3. Benchmarks — only run if tests passed (saves time on obvious failure)
        let tests_ok = checks.iter().all(|c| !c.status.is_failed());
        if self.config.run_benchmarks && tests_ok {
            let samples = runner.run_benchmarks();
            checks.push(self.evaluate_benchmarks(samples));
        } else if self.config.run_benchmarks {
            checks.push(CheckResult::skipped("benchmarks", "tests failed; skipping"));
        } else {
            checks.push(CheckResult::skipped("benchmarks", "disabled in config"));
        }

        // 4. Smoke test
        if self.config.run_smoke {
            let cmd = self.config.smoke_command.as_deref().unwrap_or("./scripts/smoke.sh");
            checks.push(runner.run_smoke(cmd));
        } else {
            checks.push(CheckResult::skipped("smoke", "disabled in config"));
        }

        // 5. Staging metrics
        if self.config.run_staging_metrics {
            let metrics = runner.staging_metrics();
            checks.push(self.evaluate_staging_metrics(metrics));
        } else {
            checks.push(CheckResult::skipped("staging_metrics", "disabled in config"));
        }

        let total_duration = start.elapsed();
        let overall_passed = checks.iter().all(|c| !c.status.is_failed());
        let recommended_action = self.recommend(&checks, overall_passed);

        ValidationReport {
            change_id,
            config: self.config.clone(),
            checks,
            overall_passed,
            total_duration,
            recommended_action,
        }
    }

    fn evaluate_benchmarks(&self, samples: Vec<BenchmarkSample>) -> CheckResult {
        let threshold = self.config.bench_regression_threshold;
        let regressions: Vec<&BenchmarkSample> =
            samples.iter().filter(|s| s.is_regression(threshold)).collect();

        if regressions.is_empty() {
            CheckResult::passed("benchmarks", Duration::ZERO)
        } else {
            let details: Vec<String> = regressions
                .iter()
                .map(|s| {
                    format!(
                        "{}: +{:.1}% ({:.0}ns → {:.0}ns)",
                        s.name,
                        s.pct_change(),
                        s.baseline_ns,
                        s.current_ns,
                    )
                })
                .collect();
            CheckResult::failed(
                "benchmarks",
                format!(
                    "{} benchmark(s) regressed >{:.0}%",
                    regressions.len(),
                    threshold * 100.0
                ),
                Duration::ZERO,
            )
            .with_details(details)
        }
    }

    fn evaluate_staging_metrics(&self, metrics: Vec<StagingMetric>) -> CheckResult {
        let failures: Vec<&StagingMetric> = metrics.iter().filter(|m| !m.passes()).collect();
        if failures.is_empty() {
            CheckResult::passed("staging_metrics", Duration::ZERO)
        } else {
            let details: Vec<String> = failures
                .iter()
                .map(|m| {
                    format!(
                        "{}: {:.3} not in [{:.3}, {:.3}]",
                        m.name, m.observed, m.min, m.max
                    )
                })
                .collect();
            CheckResult::failed(
                "staging_metrics",
                format!("{} staging metric(s) out of bounds", failures.len()),
                Duration::ZERO,
            )
            .with_details(details)
        }
    }

    fn recommend(&self, checks: &[CheckResult], overall_passed: bool) -> RecommendedAction {
        if !overall_passed {
            let failed: Vec<String> = checks
                .iter()
                .filter(|c| c.status.is_failed())
                .map(|c| c.name.clone())
                .collect();
            return RecommendedAction::Reject { failed_checks: failed };
        }
        match self.config.trust_level {
            TrustLevel::ReviewRequired => RecommendedAction::AwaitReview,
            TrustLevel::AutoMerge => RecommendedAction::AutoMerge,
            TrustLevel::AutoDeploy => RecommendedAction::AutoDeploy,
        }
    }
}

// ---------------------------------------------------------------------------
// CheckRunner trait — injectable for testing
// ---------------------------------------------------------------------------

/// Abstracts the execution of each check so tests can inject deterministic results.
pub trait CheckRunner {
    fn run_tests(&self, cmd: &str) -> CheckResult;
    fn run_clippy(&self, cmd: &str) -> CheckResult;
    fn run_benchmarks(&self) -> Vec<BenchmarkSample>;
    fn run_smoke(&self, cmd: &str) -> CheckResult;
    fn staging_metrics(&self) -> Vec<StagingMetric>;
}

/// A `CheckRunner` that always reports all checks as passing with zero regressions.
pub struct AlwaysPassRunner;

impl CheckRunner for AlwaysPassRunner {
    fn run_tests(&self, _cmd: &str) -> CheckResult {
        CheckResult::passed("tests", Duration::from_millis(500))
    }
    fn run_clippy(&self, _cmd: &str) -> CheckResult {
        CheckResult::passed("clippy", Duration::from_millis(200))
    }
    fn run_benchmarks(&self) -> Vec<BenchmarkSample> { vec![] }
    fn run_smoke(&self, _cmd: &str) -> CheckResult {
        CheckResult::passed("smoke", Duration::from_millis(100))
    }
    fn staging_metrics(&self) -> Vec<StagingMetric> { vec![] }
}

/// A `CheckRunner` with configurable failures.
pub struct ConfigurableRunner {
    pub test_result: Option<String>, // None = pass, Some(reason) = fail
    pub clippy_result: Option<String>,
    pub bench_samples: Vec<BenchmarkSample>,
    pub smoke_result: Option<String>,
    pub staging: Vec<StagingMetric>,
}

impl ConfigurableRunner {
    pub fn all_pass() -> Self {
        Self {
            test_result: None,
            clippy_result: None,
            bench_samples: vec![],
            smoke_result: None,
            staging: vec![],
        }
    }

    pub fn with_test_failure(mut self, reason: impl Into<String>) -> Self {
        self.test_result = Some(reason.into());
        self
    }

    pub fn with_clippy_failure(mut self, reason: impl Into<String>) -> Self {
        self.clippy_result = Some(reason.into());
        self
    }

    pub fn with_bench_regression(mut self, name: &str, baseline: f64, current: f64) -> Self {
        self.bench_samples.push(BenchmarkSample {
            name: name.to_string(),
            baseline_ns: baseline,
            current_ns: current,
        });
        self
    }

    pub fn with_staging_metric(mut self, name: &str, observed: f64, min: f64, max: f64) -> Self {
        self.staging.push(StagingMetric {
            name: name.to_string(),
            observed,
            min,
            max,
        });
        self
    }
}

impl CheckRunner for ConfigurableRunner {
    fn run_tests(&self, _cmd: &str) -> CheckResult {
        match &self.test_result {
            None => CheckResult::passed("tests", Duration::from_millis(400)),
            Some(r) => CheckResult::failed("tests", r.clone(), Duration::from_millis(400)),
        }
    }

    fn run_clippy(&self, _cmd: &str) -> CheckResult {
        match &self.clippy_result {
            None => CheckResult::passed("clippy", Duration::from_millis(100)),
            Some(r) => CheckResult::failed("clippy", r.clone(), Duration::from_millis(100)),
        }
    }

    fn run_benchmarks(&self) -> Vec<BenchmarkSample> {
        self.bench_samples.clone()
    }

    fn run_smoke(&self, _cmd: &str) -> CheckResult {
        match &self.smoke_result {
            None => CheckResult::passed("smoke", Duration::from_millis(50)),
            Some(r) => CheckResult::failed("smoke", r.clone(), Duration::from_millis(50)),
        }
    }

    fn staging_metrics(&self) -> Vec<StagingMetric> {
        self.staging.clone()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_gate() -> ValidationGate {
        ValidationGate::new(GateConfig::default())
    }

    fn gate_with_smoke() -> ValidationGate {
        ValidationGate::new(GateConfig { run_smoke: true, ..GateConfig::default() })
    }

    fn gate_with_staging() -> ValidationGate {
        ValidationGate::new(GateConfig {
            run_staging_metrics: true,
            ..GateConfig::default()
        })
    }

    // -------------------------------------------------------------------
    // CheckStatus
    // -------------------------------------------------------------------

    #[test]
    fn test_check_status_passed_is_passed() {
        assert!(CheckStatus::Passed.is_passed());
        assert!(!CheckStatus::Passed.is_failed());
        assert!(!CheckStatus::Passed.is_skipped());
    }

    #[test]
    fn test_check_status_failed_is_failed() {
        let s = CheckStatus::Failed { reason: "oops".into() };
        assert!(s.is_failed());
        assert!(!s.is_passed());
    }

    #[test]
    fn test_check_status_skipped_is_skipped() {
        let s = CheckStatus::Skipped { reason: "off".into() };
        assert!(s.is_skipped());
        assert!(!s.is_failed());
    }

    #[test]
    fn test_check_status_display_passed() {
        assert_eq!(CheckStatus::Passed.to_string(), "PASS");
    }

    #[test]
    fn test_check_status_display_failed() {
        let s = CheckStatus::Failed { reason: "x".into() };
        assert!(s.to_string().starts_with("FAIL:"));
    }

    #[test]
    fn test_check_status_display_skipped() {
        let s = CheckStatus::Skipped { reason: "y".into() };
        assert!(s.to_string().starts_with("SKIP:"));
    }

    // -------------------------------------------------------------------
    // BenchmarkSample
    // -------------------------------------------------------------------

    #[test]
    fn test_bench_regression_fraction_zero_baseline() {
        let s = BenchmarkSample { name: "x".into(), baseline_ns: 0.0, current_ns: 100.0 };
        assert_eq!(s.regression_fraction(), 0.0);
    }

    #[test]
    fn test_bench_regression_fraction_10pct() {
        let s = BenchmarkSample { name: "x".into(), baseline_ns: 100.0, current_ns: 110.0 };
        assert!((s.regression_fraction() - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_bench_regression_fraction_improvement_is_negative() {
        let s = BenchmarkSample { name: "x".into(), baseline_ns: 100.0, current_ns: 90.0 };
        assert!(s.regression_fraction() < 0.0);
    }

    #[test]
    fn test_bench_is_regression_above_threshold() {
        let s = BenchmarkSample { name: "x".into(), baseline_ns: 100.0, current_ns: 110.0 };
        assert!(s.is_regression(0.05)); // 10% > 5%
    }

    #[test]
    fn test_bench_is_not_regression_below_threshold() {
        let s = BenchmarkSample { name: "x".into(), baseline_ns: 100.0, current_ns: 103.0 };
        assert!(!s.is_regression(0.05)); // 3% < 5%
    }

    // -------------------------------------------------------------------
    // StagingMetric
    // -------------------------------------------------------------------

    #[test]
    fn test_staging_metric_passes_in_range() {
        let m = StagingMetric { name: "lat".into(), observed: 50.0, min: 0.0, max: 100.0 };
        assert!(m.passes());
    }

    #[test]
    fn test_staging_metric_fails_above_max() {
        let m = StagingMetric { name: "lat".into(), observed: 150.0, min: 0.0, max: 100.0 };
        assert!(!m.passes());
    }

    #[test]
    fn test_staging_metric_fails_below_min() {
        let m = StagingMetric { name: "thr".into(), observed: 10.0, min: 50.0, max: 500.0 };
        assert!(!m.passes());
    }

    #[test]
    fn test_staging_metric_passes_at_boundary() {
        let m = StagingMetric { name: "x".into(), observed: 100.0, min: 0.0, max: 100.0 };
        assert!(m.passes());
    }

    // -------------------------------------------------------------------
    // TrustLevel
    // -------------------------------------------------------------------

    #[test]
    fn test_trust_level_ord_auto_deploy_highest() {
        assert!(TrustLevel::AutoDeploy > TrustLevel::AutoMerge);
        assert!(TrustLevel::AutoMerge > TrustLevel::ReviewRequired);
    }

    #[test]
    fn test_trust_level_display() {
        assert_eq!(TrustLevel::ReviewRequired.to_string(), "review_required");
        assert_eq!(TrustLevel::AutoMerge.to_string(), "auto_merge");
        assert_eq!(TrustLevel::AutoDeploy.to_string(), "auto_deploy");
    }

    // -------------------------------------------------------------------
    // Gate — all pass
    // -------------------------------------------------------------------

    #[test]
    fn test_gate_all_pass_overall_passed() {
        let gate = default_gate();
        let report = gate.run("sha-abc", &AlwaysPassRunner);
        assert!(report.overall_passed);
    }

    #[test]
    fn test_gate_all_pass_review_required_recommends_await() {
        let gate = default_gate(); // trust = ReviewRequired
        let report = gate.run("sha-abc", &AlwaysPassRunner);
        assert_eq!(report.recommended_action, RecommendedAction::AwaitReview);
    }

    #[test]
    fn test_gate_all_pass_auto_merge_recommends_merge() {
        let mut gate = default_gate();
        gate.set_trust_level(TrustLevel::AutoMerge);
        let report = gate.run("sha-abc", &AlwaysPassRunner);
        assert_eq!(report.recommended_action, RecommendedAction::AutoMerge);
    }

    #[test]
    fn test_gate_all_pass_auto_deploy_recommends_deploy() {
        let mut gate = default_gate();
        gate.set_trust_level(TrustLevel::AutoDeploy);
        let report = gate.run("sha-abc", &AlwaysPassRunner);
        assert_eq!(report.recommended_action, RecommendedAction::AutoDeploy);
    }

    // -------------------------------------------------------------------
    // Gate — test failure
    // -------------------------------------------------------------------

    #[test]
    fn test_gate_test_failure_overall_fails() {
        let gate = default_gate();
        let runner = ConfigurableRunner::all_pass().with_test_failure("3 tests failed");
        let report = gate.run("sha-xyz", &runner);
        assert!(!report.overall_passed);
    }

    #[test]
    fn test_gate_test_failure_recommends_reject() {
        let gate = default_gate();
        let runner = ConfigurableRunner::all_pass().with_test_failure("test failed");
        let report = gate.run("sha-xyz", &runner);
        assert!(matches!(report.recommended_action, RecommendedAction::Reject { .. }));
    }

    #[test]
    fn test_gate_test_failure_reject_names_tests() {
        let gate = default_gate();
        let runner = ConfigurableRunner::all_pass().with_test_failure("fail");
        let report = gate.run("sha", &runner);
        if let RecommendedAction::Reject { failed_checks } = &report.recommended_action {
            assert!(failed_checks.contains(&"tests".to_string()));
        } else {
            panic!("expected Reject");
        }
    }

    #[test]
    fn test_gate_clippy_failure_overall_fails() {
        let gate = default_gate();
        let runner = ConfigurableRunner::all_pass().with_clippy_failure("warning: unused var");
        let report = gate.run("sha", &runner);
        assert!(!report.overall_passed);
    }

    // -------------------------------------------------------------------
    // Gate — benchmark regression
    // -------------------------------------------------------------------

    #[test]
    fn test_gate_bench_regression_fails() {
        let gate = default_gate();
        let runner = ConfigurableRunner::all_pass()
            .with_bench_regression("dedup_hot_path", 100.0, 120.0); // 20% > 5%
        let report = gate.run("sha", &runner);
        assert!(!report.overall_passed);
        let bench = report.checks.iter().find(|c| c.name == "benchmarks").unwrap();
        assert!(bench.status.is_failed());
    }

    #[test]
    fn test_gate_bench_within_threshold_passes() {
        let gate = default_gate();
        let runner = ConfigurableRunner::all_pass()
            .with_bench_regression("dedup_hot_path", 100.0, 103.0); // 3% < 5%
        let report = gate.run("sha", &runner);
        let bench = report.checks.iter().find(|c| c.name == "benchmarks").unwrap();
        assert!(bench.status.is_passed());
    }

    #[test]
    fn test_gate_bench_skipped_when_tests_fail() {
        let gate = default_gate();
        let runner = ConfigurableRunner::all_pass()
            .with_test_failure("fail")
            .with_bench_regression("x", 100.0, 200.0); // would fail but should be skipped
        let report = gate.run("sha", &runner);
        let bench = report.checks.iter().find(|c| c.name == "benchmarks").unwrap();
        assert!(bench.status.is_skipped());
    }

    #[test]
    fn test_gate_bench_regression_details_contain_name() {
        let gate = default_gate();
        let runner = ConfigurableRunner::all_pass()
            .with_bench_regression("my_bench", 100.0, 200.0);
        let report = gate.run("sha", &runner);
        let bench = report.checks.iter().find(|c| c.name == "benchmarks").unwrap();
        assert!(bench.details.iter().any(|d| d.contains("my_bench")));
    }

    // -------------------------------------------------------------------
    // Gate — staging metrics
    // -------------------------------------------------------------------

    #[test]
    fn test_gate_staging_metric_out_of_bounds_fails() {
        let gate = gate_with_staging();
        let runner = ConfigurableRunner::all_pass()
            .with_staging_metric("p95_latency_ms", 200.0, 0.0, 100.0);
        let report = gate.run("sha", &runner);
        let sm = report.checks.iter().find(|c| c.name == "staging_metrics").unwrap();
        assert!(sm.status.is_failed());
    }

    #[test]
    fn test_gate_staging_metric_in_bounds_passes() {
        let gate = gate_with_staging();
        let runner = ConfigurableRunner::all_pass()
            .with_staging_metric("p95_latency_ms", 50.0, 0.0, 100.0);
        let report = gate.run("sha", &runner);
        let sm = report.checks.iter().find(|c| c.name == "staging_metrics").unwrap();
        assert!(sm.status.is_passed());
    }

    // -------------------------------------------------------------------
    // Gate — disabled checks
    // -------------------------------------------------------------------

    #[test]
    fn test_gate_disabled_tests_skips_tests() {
        let gate = ValidationGate::new(GateConfig { run_tests: false, ..GateConfig::default() });
        let report = gate.run("sha", &AlwaysPassRunner);
        let tests = report.checks.iter().find(|c| c.name == "tests").unwrap();
        assert!(tests.status.is_skipped());
    }

    #[test]
    fn test_gate_disabled_benchmarks_skips_benchmarks() {
        let gate = ValidationGate::new(GateConfig { run_benchmarks: false, ..GateConfig::default() });
        let report = gate.run("sha", &AlwaysPassRunner);
        let bench = report.checks.iter().find(|c| c.name == "benchmarks").unwrap();
        assert!(bench.status.is_skipped());
    }

    #[test]
    fn test_gate_smoke_disabled_by_default() {
        let report = default_gate().run("sha", &AlwaysPassRunner);
        let smoke = report.checks.iter().find(|c| c.name == "smoke").unwrap();
        assert!(smoke.status.is_skipped());
    }

    #[test]
    fn test_gate_smoke_enabled_runs() {
        let gate = gate_with_smoke();
        let report = gate.run("sha", &AlwaysPassRunner);
        let smoke = report.checks.iter().find(|c| c.name == "smoke").unwrap();
        assert!(smoke.status.is_passed());
    }

    // -------------------------------------------------------------------
    // ValidationReport helpers
    // -------------------------------------------------------------------

    #[test]
    fn test_report_failed_checks_returns_only_failures() {
        let gate = default_gate();
        let runner = ConfigurableRunner::all_pass().with_test_failure("fail");
        let report = gate.run("sha", &runner);
        let failed = report.failed_checks();
        assert!(!failed.is_empty());
        assert!(failed.iter().all(|c| c.status.is_failed()));
    }

    #[test]
    fn test_report_passed_checks_returns_only_passed() {
        let report = default_gate().run("sha", &AlwaysPassRunner);
        let passed = report.passed_checks();
        assert!(passed.iter().all(|c| c.status.is_passed()));
    }

    #[test]
    fn test_report_summary_contains_change_id() {
        let report = default_gate().run("my-sha-123", &AlwaysPassRunner);
        assert!(report.summary().contains("my-sha-123"));
    }

    #[test]
    fn test_report_summary_contains_pass() {
        let report = default_gate().run("sha", &AlwaysPassRunner);
        assert!(report.summary().contains("PASS"));
    }

    #[test]
    fn test_report_summary_contains_fail_on_failure() {
        let runner = ConfigurableRunner::all_pass().with_test_failure("fail");
        let report = default_gate().run("sha", &runner);
        assert!(report.summary().contains("FAIL"));
    }

    #[test]
    fn test_report_has_five_check_entries() {
        // 5 checks: tests, clippy, benchmarks, smoke, staging_metrics
        let report = default_gate().run("sha", &AlwaysPassRunner);
        assert_eq!(report.checks.len(), 5);
    }

    // -------------------------------------------------------------------
    // RecommendedAction display
    // -------------------------------------------------------------------

    #[test]
    fn test_recommended_action_display_reject_includes_names() {
        let a = RecommendedAction::Reject { failed_checks: vec!["tests".into(), "clippy".into()] };
        let s = a.to_string();
        assert!(s.contains("tests"));
        assert!(s.contains("clippy"));
    }
}
