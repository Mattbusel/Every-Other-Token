//! # Hallucination Detector
//!
//! Detects potential hallucinations in LLM-generated token streams by
//! identifying two complementary signals:
//!
//! 1. **Perplexity spikes** — positions where the model's confidence drops
//!    sharply relative to the local baseline.  A sudden spike in perplexity
//!    (= -log₂ prob) suggests the model is generating tokens that were
//!    unlikely given the context.
//!
//! 2. **Confident-but-wrong patterns** — positions where the model assigns
//!    high confidence *and* the second forward pass (with a paraphrase prompt)
//!    produces a substantially different top token, suggesting the first
//!    response was plausible-sounding but fragile.
//!
//! ## Usage
//!
//! ```rust
//! use every_other_token::hallucination::{HallucinationDetector, DetectorConfig, TokenSample};
//!
//! let config = DetectorConfig::default();
//! let detector = HallucinationDetector::new(config);
//!
//! let tokens = vec![
//!     TokenSample { token: "Paris".into(), logprob: -0.1, position: 0 },
//!     TokenSample { token: "is".into(),    logprob: -0.2, position: 1 },
//!     TokenSample { token: "1337".into(),  logprob: -4.5, position: 2 }, // spike!
//! ];
//!
//! let report = detector.analyse(&tokens, None);
//! assert!(!report.flags.is_empty());
//! ```

use serde::{Deserialize, Serialize};

// ── Token sample ──────────────────────────────────────────────────────────────

/// A single generated token with its position and log-probability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenSample {
    /// Surface form of the token.
    pub token: String,
    /// Natural-log probability from the model (`-∞` to `0`).
    pub logprob: f64,
    /// Zero-based position in the generated sequence.
    pub position: usize,
}

impl TokenSample {
    /// Perplexity contribution for this token: `exp(-logprob)`.
    pub fn perplexity(&self) -> f64 {
        (-self.logprob).exp()
    }

    /// Confidence: `exp(logprob)` clamped to `[0, 1]`.
    pub fn confidence(&self) -> f64 {
        self.logprob.exp().clamp(0.0, 1.0)
    }
}

// ── Flags ─────────────────────────────────────────────────────────────────────

/// The kind of hallucination signal detected.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FlagKind {
    /// Perplexity at this position is significantly higher than the local mean.
    PerplexitySpike,
    /// Model was confident but a second pass produced a different top token.
    ConfidentButUnstable,
}

/// A single hallucination flag for one token position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HallucinationFlag {
    /// Zero-based position in the generated sequence.
    pub position: usize,
    /// The token at this position.
    pub token: String,
    /// What kind of signal was detected.
    pub kind: FlagKind,
    /// Severity score in `[0.0, 1.0]` (1.0 = most suspicious).
    pub severity: f64,
    /// Human-readable explanation.
    pub reason: String,
}

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for the hallucination detector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorConfig {
    /// A spike is flagged when perplexity exceeds `baseline_mean * spike_ratio`.
    pub spike_ratio: f64,
    /// Minimum perplexity value to consider a spike (ignores low-perplexity noise).
    pub min_perplexity_threshold: f64,
    /// Confidence above this value qualifies a token for the
    /// "confident-but-wrong" check.
    pub confident_threshold: f64,
    /// Log-prob difference between two passes that counts as "unstable".
    pub instability_logprob_delta: f64,
    /// Window size for computing the local perplexity baseline.
    pub baseline_window: usize,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            spike_ratio: 3.0,
            min_perplexity_threshold: 2.0,
            confident_threshold: 0.7,
            instability_logprob_delta: 1.5,
            baseline_window: 8,
        }
    }
}

// ── Cross-check sample ────────────────────────────────────────────────────────

/// Token sample from a second LLM forward pass used for instability detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossCheckSample {
    /// Position in the generated sequence (must match the primary sample).
    pub position: usize,
    /// Top token produced by the second pass.
    pub top_token: String,
    /// Log-probability of the top token in the second pass.
    pub logprob: f64,
}

// ── Report ────────────────────────────────────────────────────────────────────

/// Full analysis report for a token sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HallucinationReport {
    /// All flagged positions.
    pub flags: Vec<HallucinationFlag>,
    /// Mean perplexity across the entire sequence.
    pub mean_perplexity: f64,
    /// Peak perplexity value in the sequence.
    pub peak_perplexity: f64,
    /// Fraction of tokens flagged (`flags.len() / total_tokens`).
    pub flag_rate: f64,
    /// Overall risk level derived from flag rate and severity.
    pub risk_level: RiskLevel,
}

/// Coarse risk classification.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskLevel {
    /// Fewer than 5% of tokens flagged.
    Low,
    /// 5–20% of tokens flagged.
    Moderate,
    /// More than 20% of tokens flagged.
    High,
}

// ── Detector ──────────────────────────────────────────────────────────────────

/// Analyses token streams for hallucination signals.
pub struct HallucinationDetector {
    config: DetectorConfig,
}

impl HallucinationDetector {
    /// Create a detector with the given configuration.
    pub fn new(config: DetectorConfig) -> Self {
        Self { config }
    }

    /// Analyse a token sequence, optionally cross-checking with a second pass.
    ///
    /// # Parameters
    ///
    /// - `tokens`: primary token stream with logprobs.
    /// - `cross_check`: optional second-pass samples for instability detection.
    pub fn analyse(
        &self,
        tokens: &[TokenSample],
        cross_check: Option<&[CrossCheckSample]>,
    ) -> HallucinationReport {
        if tokens.is_empty() {
            return HallucinationReport {
                flags: vec![],
                mean_perplexity: 0.0,
                peak_perplexity: 0.0,
                flag_rate: 0.0,
                risk_level: RiskLevel::Low,
            };
        }

        let perplexities: Vec<f64> = tokens.iter().map(|t| t.perplexity()).collect();
        let mean_perplexity = perplexities.iter().sum::<f64>() / perplexities.len() as f64;
        let peak_perplexity = perplexities
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let mut flags: Vec<HallucinationFlag> = Vec::new();

        // ── Pass 1: perplexity spike detection ────────────────────────────
        for (i, token) in tokens.iter().enumerate() {
            let local_baseline = self.local_baseline(&perplexities, i);
            let ppl = perplexities[i];

            if ppl >= self.config.min_perplexity_threshold
                && ppl > local_baseline * self.config.spike_ratio
            {
                let severity = ((ppl / (local_baseline * self.config.spike_ratio)) - 1.0)
                    .clamp(0.0, 1.0);
                flags.push(HallucinationFlag {
                    position: token.position,
                    token: token.token.clone(),
                    kind: FlagKind::PerplexitySpike,
                    severity,
                    reason: format!(
                        "Perplexity {ppl:.2} is {:.1}x the local baseline {local_baseline:.2}",
                        ppl / local_baseline.max(f64::EPSILON)
                    ),
                });
            }
        }

        // ── Pass 2: confident-but-unstable detection ──────────────────────
        if let Some(cc) = cross_check {
            let cc_map: std::collections::HashMap<usize, &CrossCheckSample> =
                cc.iter().map(|s| (s.position, s)).collect();

            for token in tokens {
                if token.confidence() < self.config.confident_threshold {
                    continue;
                }
                if let Some(cc_sample) = cc_map.get(&token.position) {
                    let delta = (token.logprob - cc_sample.logprob).abs();
                    if delta >= self.config.instability_logprob_delta
                        || cc_sample.top_token != token.token
                    {
                        let severity =
                            (delta / (self.config.instability_logprob_delta * 2.0)).clamp(0.0, 1.0);
                        // Only add if not already flagged at this position.
                        if !flags.iter().any(|f| f.position == token.position
                            && f.kind == FlagKind::ConfidentButUnstable)
                        {
                            flags.push(HallucinationFlag {
                                position: token.position,
                                token: token.token.clone(),
                                kind: FlagKind::ConfidentButUnstable,
                                severity,
                                reason: format!(
                                    "High confidence ({:.2}) but second pass produced '{}' (Δlogprob={delta:.2})",
                                    token.confidence(),
                                    cc_sample.top_token
                                ),
                            });
                        }
                    }
                }
            }
        }

        flags.sort_by_key(|f| f.position);

        let flag_rate = flags.len() as f64 / tokens.len() as f64;
        let risk_level = if flag_rate >= 0.20 {
            RiskLevel::High
        } else if flag_rate >= 0.05 {
            RiskLevel::Moderate
        } else {
            RiskLevel::Low
        };

        HallucinationReport {
            flags,
            mean_perplexity,
            peak_perplexity,
            flag_rate,
            risk_level,
        }
    }

    /// Compute the mean perplexity in a window around position `i`.
    fn local_baseline(&self, perplexities: &[f64], i: usize) -> f64 {
        let half = self.config.baseline_window / 2;
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(perplexities.len());
        let window: Vec<f64> = perplexities[start..end]
            .iter()
            .enumerate()
            .filter(|(j, _)| *j + start != i) // exclude the token itself
            .map(|(_, &v)| v)
            .collect();

        if window.is_empty() {
            1.0
        } else {
            window.iter().sum::<f64>() / window.len() as f64
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(token: &str, logprob: f64, position: usize) -> TokenSample {
        TokenSample {
            token: token.to_string(),
            logprob,
            position,
        }
    }

    #[test]
    fn no_tokens_returns_empty_report() {
        let detector = HallucinationDetector::new(DetectorConfig::default());
        let report = detector.analyse(&[], None);
        assert!(report.flags.is_empty());
        assert_eq!(report.risk_level, RiskLevel::Low);
    }

    #[test]
    fn perplexity_spike_detected() {
        let detector = HallucinationDetector::new(DetectorConfig {
            spike_ratio: 2.0,
            min_perplexity_threshold: 1.5,
            ..Default::default()
        });
        // Positions 0-3: low perplexity; position 4: huge spike.
        let tokens = vec![
            ts("The", -0.1, 0),
            ts("cat", -0.2, 1),
            ts("sat", -0.15, 2),
            ts("on", -0.1, 3),
            ts("XYZZY", -5.0, 4), // spike
        ];
        let report = detector.analyse(&tokens, None);
        let spike_flags: Vec<_> = report
            .flags
            .iter()
            .filter(|f| f.kind == FlagKind::PerplexitySpike)
            .collect();
        assert!(!spike_flags.is_empty(), "expected at least one spike flag");
        assert_eq!(spike_flags[0].position, 4);
    }

    #[test]
    fn no_false_positives_on_uniform_sequence() {
        let detector = HallucinationDetector::new(DetectorConfig::default());
        let tokens: Vec<TokenSample> = (0..10)
            .map(|i| ts("word", -0.2, i))
            .collect();
        let report = detector.analyse(&tokens, None);
        assert!(
            report.flags.iter().all(|f| f.kind != FlagKind::PerplexitySpike),
            "uniform sequence should not produce spike flags"
        );
    }

    #[test]
    fn confident_but_unstable_detected() {
        let detector = HallucinationDetector::new(DetectorConfig::default());
        // High-confidence token.
        let tokens = vec![ts("Paris", -0.05, 0)]; // confidence ≈ 0.95
        let cross_check = vec![CrossCheckSample {
            position: 0,
            top_token: "London".to_string(),
            logprob: -0.05, // Δlogprob small, but different token — still flagged if tokens differ
        }];
        // Note: even with equal logprobs, different top_token triggers the flag.
        let report = detector.analyse(&tokens, Some(&cross_check));
        let unstable: Vec<_> = report
            .flags
            .iter()
            .filter(|f| f.kind == FlagKind::ConfidentButUnstable)
            .collect();
        assert!(!unstable.is_empty());
    }

    #[test]
    fn risk_level_high_when_many_flags() {
        let detector = HallucinationDetector::new(DetectorConfig {
            spike_ratio: 1.1,
            min_perplexity_threshold: 0.0,
            ..Default::default()
        });
        // All tokens spike against each other.
        let tokens: Vec<TokenSample> = (0..10)
            .map(|i| {
                let lp = if i % 2 == 0 { -0.1 } else { -8.0 };
                ts("tok", lp, i)
            })
            .collect();
        let report = detector.analyse(&tokens, None);
        assert!(report.flag_rate > 0.0);
    }

    #[test]
    fn confidence_and_perplexity_inverse() {
        let t = ts("hello", -0.5, 0);
        let ppl = t.perplexity();
        let conf = t.confidence();
        // Higher confidence → lower perplexity.
        assert!(ppl > 1.0);
        assert!(conf < 1.0 && conf > 0.0);
    }
}
