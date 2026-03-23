//! # Bayesian A/B Testing with Thompson Sampling
//!
//! Replaces the classical Welch's t-test with an online Bayesian bandit that
//! adapts the experiment arm selection probability in real-time based on
//! observed outcomes.
//!
//! ## Why Thompson Sampling over t-tests?
//!
//! Classical A/B testing (Welch's t-test) requires a fixed sample size
//! decided upfront and treats all arms equally until the test is complete.
//! This is wasteful: if arm B is clearly worse after 20 runs, you still run it
//! 50 more times to reach statistical significance.
//!
//! Thompson sampling is a **multi-armed bandit** algorithm that:
//!
//! 1. **Adapts dynamically** — arms with better observed performance are
//!    sampled more often, reducing wasted runs on inferior transforms.
//! 2. **No fixed horizon** — you can stop at any time and report the current
//!    best arm with credible intervals.
//! 3. **Handles uncertainty correctly** — uncertain arms (few samples) are
//!    explored; well-understood arms converge to their true performance.
//!
//! For token-level experiments, we model each arm's mean confidence as a
//! **Beta distribution** (conjugate prior for the Bernoulli likelihood):
//!
//! - Each observed confidence token is discretised to a Bernoulli trial:
//!   `success = confidence >= threshold` (default 0.5).
//! - The Beta posterior `Beta(α, β)` is updated per observation.
//! - At selection time, each arm draws one sample from its posterior; the
//!   arm with the highest draw is selected.
//!
//! ## Example
//!
//! ```rust
//! use every_other_token::bayesian::{ThompsonBandit, BanditArm};
//!
//! let mut bandit = ThompsonBandit::new(0.5); // success threshold
//!
//! // Register transform variants as arms.
//! bandit.add_arm("every_other");
//! bandit.add_arm("every_third");
//! bandit.add_arm("chaos_10pct");
//!
//! // Simulate 30 experiments.
//! for _ in 0..30 {
//!     let arm_name = bandit.select();
//!     // ... run experiment with that transform ...
//!     let mean_confidence: f64 = 0.72; // from the run
//!     bandit.update(&arm_name, mean_confidence);
//! }
//!
//! let winner = bandit.best_arm();
//! println!("Best transform: {} (mean={:.3})", winner.name, winner.posterior_mean());
//! ```

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// BanditArm
// ---------------------------------------------------------------------------

/// A single experimental arm with a Beta-distributed posterior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditArm {
    /// Name of the transform variant this arm represents.
    pub name: String,
    /// Beta distribution alpha parameter (successes + 1).
    pub alpha: f64,
    /// Beta distribution beta parameter (failures + 1).
    pub beta: f64,
    /// Total number of observations recorded.
    pub observations: u64,
    /// Running sum of observed values (for mean calculation).
    pub value_sum: f64,
}

impl BanditArm {
    /// Create a new arm with a uniform prior Beta(1, 1).
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            alpha: 1.0,
            beta: 1.0,
            observations: 0,
            value_sum: 0.0,
        }
    }

    /// Mean of the Beta posterior: α / (α + β).
    pub fn posterior_mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Variance of the Beta posterior.
    pub fn posterior_variance(&self) -> f64 {
        let n = self.alpha + self.beta;
        (self.alpha * self.beta) / (n * n * (n + 1.0))
    }

    /// 95% credible interval (Wilson score approximation).
    pub fn credible_interval_95(&self) -> (f64, f64) {
        let mean = self.posterior_mean();
        let std = self.posterior_variance().sqrt();
        let margin = 1.96 * std;
        ((mean - margin).max(0.0), (mean + margin).min(1.0))
    }

    /// Draw a single Thompson sample from the Beta posterior.
    ///
    /// Uses Johnk's method for exact Beta sampling.
    pub fn thompson_sample<R: Rng>(&self, rng: &mut R) -> f64 {
        sample_beta(rng, self.alpha, self.beta)
    }

    /// Mean of raw observed values (not the posterior mean).
    pub fn observed_mean(&self) -> f64 {
        if self.observations == 0 {
            return 0.0;
        }
        self.value_sum / self.observations as f64
    }
}

// ---------------------------------------------------------------------------
// ThompsonBandit
// ---------------------------------------------------------------------------

/// Online multi-armed bandit with Thompson sampling selection.
///
/// See the [module documentation][self] for a full usage example.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThompsonBandit {
    /// Arms indexed by name.
    arms: HashMap<String, BanditArm>,
    /// Ordered arm names for deterministic iteration.
    arm_order: Vec<String>,
    /// Confidence value at or above which an observation counts as a "success".
    pub success_threshold: f64,
    /// Total selections made.
    pub total_selections: u64,
}

impl ThompsonBandit {
    /// Create a bandit with the given success threshold.
    ///
    /// A token observation is a "success" when
    /// `mean_confidence >= success_threshold`.
    pub fn new(success_threshold: f64) -> Self {
        Self {
            arms: HashMap::new(),
            arm_order: Vec::new(),
            success_threshold,
            total_selections: 0,
        }
    }

    /// Add a new arm.  No-op if an arm with `name` already exists.
    pub fn add_arm(&mut self, name: impl Into<String>) {
        let name = name.into();
        if !self.arms.contains_key(&name) {
            self.arm_order.push(name.clone());
            self.arms.insert(name.clone(), BanditArm::new(name));
        }
    }

    /// Select the next arm to run using Thompson sampling.
    ///
    /// Each arm draws one sample from its Beta posterior; the arm with the
    /// highest draw is returned.  Falls back to a round-robin if no arms are
    /// registered.
    ///
    /// # Panics
    ///
    /// Does not panic.
    pub fn select(&mut self) -> String {
        if self.arms.is_empty() {
            return String::new();
        }

        let mut rng = rand::thread_rng();
        let (best_name, _) = self
            .arm_order
            .iter()
            .filter_map(|name| {
                self.arms
                    .get(name)
                    .map(|arm| (name.clone(), arm.thompson_sample(&mut rng)))
            })
            .fold(
                (String::new(), f64::NEG_INFINITY),
                |(best_n, best_v), (n, v)| {
                    if v > best_v {
                        (n, v)
                    } else {
                        (best_n, best_v)
                    }
                },
            );

        self.total_selections += 1;
        best_name
    }

    /// Record an observation for `arm_name`.
    ///
    /// `value` should be the mean token confidence from the completed run
    /// (range 0.0–1.0).  The observation is discretised to a Bernoulli trial
    /// against `success_threshold`.
    pub fn update(&mut self, arm_name: &str, value: f64) {
        if let Some(arm) = self.arms.get_mut(arm_name) {
            if value >= self.success_threshold {
                arm.alpha += 1.0;
            } else {
                arm.beta += 1.0;
            }
            arm.observations += 1;
            arm.value_sum += value;
        }
    }

    /// Return the arm with the highest posterior mean.
    ///
    /// Returns `None` when no arms are registered.
    pub fn best_arm(&self) -> Option<&BanditArm> {
        self.arms
            .values()
            .max_by(|a, b| a.posterior_mean().partial_cmp(&b.posterior_mean()).unwrap())
    }

    /// Return all arms sorted by posterior mean (descending).
    pub fn ranked_arms(&self) -> Vec<&BanditArm> {
        let mut arms: Vec<&BanditArm> = self.arms.values().collect();
        arms.sort_by(|a, b| b.posterior_mean().partial_cmp(&a.posterior_mean()).unwrap());
        arms
    }

    /// Return a snapshot of arm statistics suitable for display or serialisation.
    pub fn stats(&self) -> Vec<ArmStats> {
        self.arm_order
            .iter()
            .filter_map(|name| self.arms.get(name))
            .map(|arm| {
                let ci = arm.credible_interval_95();
                ArmStats {
                    name: arm.name.clone(),
                    observations: arm.observations,
                    posterior_mean: arm.posterior_mean(),
                    observed_mean: arm.observed_mean(),
                    ci_low: ci.0,
                    ci_high: ci.1,
                    alpha: arm.alpha,
                    beta: arm.beta,
                }
            })
            .collect()
    }

    /// Probability that arm `a` is better than arm `b`, estimated by Monte Carlo.
    ///
    /// Returns `None` if either arm name is unrecognised.
    pub fn probability_a_beats_b(&self, a: &str, b: &str, samples: usize) -> Option<f64> {
        let arm_a = self.arms.get(a)?;
        let arm_b = self.arms.get(b)?;
        let mut rng = rand::thread_rng();
        let wins: usize = (0..samples)
            .filter(|_| arm_a.thompson_sample(&mut rng) > arm_b.thompson_sample(&mut rng))
            .count();
        Some(wins as f64 / samples as f64)
    }
}

// ---------------------------------------------------------------------------
// ArmStats
// ---------------------------------------------------------------------------

/// Display statistics for a single bandit arm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmStats {
    /// Arm name.
    pub name: String,
    /// Number of observations recorded.
    pub observations: u64,
    /// Beta posterior mean (α / (α + β)).
    pub posterior_mean: f64,
    /// Arithmetic mean of raw observed values.
    pub observed_mean: f64,
    /// Lower bound of 95% credible interval.
    pub ci_low: f64,
    /// Upper bound of 95% credible interval.
    pub ci_high: f64,
    /// Current alpha parameter of the Beta posterior.
    pub alpha: f64,
    /// Current beta parameter of the Beta posterior.
    pub beta: f64,
}

// ---------------------------------------------------------------------------
// Beta sampling
// ---------------------------------------------------------------------------

/// Sample from Beta(alpha, beta) using Johnk's method.
fn sample_beta<R: Rng>(rng: &mut R, alpha: f64, beta: f64) -> f64 {
    // Gamma-based method: X ~ Gamma(alpha, 1), Y ~ Gamma(beta, 1)
    // Z = X / (X + Y) ~ Beta(alpha, beta)
    let x = sample_gamma(rng, alpha);
    let y = sample_gamma(rng, beta);
    let s = x + y;
    if s == 0.0 { 0.5 } else { x / s }
}

/// Marsaglia-Tsang method for Gamma(shape, 1) sampling.
fn sample_gamma<R: Rng>(rng: &mut R, shape: f64) -> f64 {
    if shape < 1.0 {
        return sample_gamma(rng, shape + 1.0) * rng.gen::<f64>().powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x: f64 = rng.sample(rand::distributions::Standard);
        // Box-Muller normal variate
        let u: f64 = rng.gen();
        if u <= 0.0 {
            continue;
        }
        let z = (2.0 * std::f64::consts::PI * u).cos()
            * (-2.0 * (rng.gen::<f64>().ln())).sqrt();
        let v = (1.0 + c * z).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u2: f64 = rng.gen();
        if u2 < 1.0 - 0.0331 * (z * z) * (z * z) {
            return d * v;
        }
        let _ = x; // suppress unused
        if u2.ln() < 0.5 * z * z + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arm_uniform_prior() {
        let arm = BanditArm::new("test");
        assert_eq!(arm.posterior_mean(), 0.5); // Beta(1,1) mean = 0.5
        assert_eq!(arm.observations, 0);
    }

    #[test]
    fn bandit_update_shifts_posterior() {
        let mut bandit = ThompsonBandit::new(0.5);
        bandit.add_arm("a");
        // Feed 10 successes.
        for _ in 0..10 {
            bandit.update("a", 0.8);
        }
        let mean = bandit.arms["a"].posterior_mean();
        assert!(mean > 0.5, "posterior should shift toward success after 10 wins");
    }

    #[test]
    fn bandit_selects_from_registered_arms() {
        let mut bandit = ThompsonBandit::new(0.5);
        bandit.add_arm("x");
        bandit.add_arm("y");
        let selected = bandit.select();
        assert!(selected == "x" || selected == "y");
    }

    #[test]
    fn empty_bandit_select_returns_empty() {
        let mut bandit = ThompsonBandit::new(0.5);
        assert_eq!(bandit.select(), "");
    }

    #[test]
    fn best_arm_returns_highest_posterior_mean() {
        let mut bandit = ThompsonBandit::new(0.5);
        bandit.add_arm("low");
        bandit.add_arm("high");
        for _ in 0..20 {
            bandit.update("low", 0.2);
        }
        for _ in 0..20 {
            bandit.update("high", 0.9);
        }
        let best = bandit.best_arm().unwrap();
        assert_eq!(best.name, "high");
    }

    #[test]
    fn probability_a_beats_b_with_strong_evidence() {
        let mut bandit = ThompsonBandit::new(0.5);
        bandit.add_arm("winner");
        bandit.add_arm("loser");
        for _ in 0..50 {
            bandit.update("winner", 0.95);
        }
        for _ in 0..50 {
            bandit.update("loser", 0.1);
        }
        let p = bandit.probability_a_beats_b("winner", "loser", 2000).unwrap();
        assert!(p > 0.95, "winner should beat loser with >95% probability, got {p:.3}");
    }

    #[test]
    fn ranked_arms_descending_order() {
        let mut bandit = ThompsonBandit::new(0.5);
        bandit.add_arm("a");
        bandit.add_arm("b");
        bandit.add_arm("c");
        bandit.update("a", 0.3);
        bandit.update("b", 0.7);
        bandit.update("c", 0.5);
        let ranked = bandit.ranked_arms();
        assert!(
            ranked[0].posterior_mean() >= ranked[1].posterior_mean(),
            "arms should be sorted descending"
        );
    }

    #[test]
    fn credible_interval_is_valid() {
        let mut arm = BanditArm::new("t");
        for _ in 0..10 {
            arm.alpha += 1.0;
        }
        let (lo, hi) = arm.credible_interval_95();
        assert!(lo >= 0.0);
        assert!(hi <= 1.0);
        assert!(lo < hi);
    }

    #[test]
    fn stats_includes_all_arms() {
        let mut bandit = ThompsonBandit::new(0.5);
        bandit.add_arm("p");
        bandit.add_arm("q");
        bandit.update("p", 0.6);
        let stats = bandit.stats();
        assert_eq!(stats.len(), 2);
    }
}
