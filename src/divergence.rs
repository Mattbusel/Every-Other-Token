//! Divergence Detection Engine.
//!
//! Runs the same prompt through multiple model configurations simultaneously
//! and tracks exactly where their token streams diverge — position by position
//! — using Jensen-Shannon divergence between per-position token probability
//! distributions.
//!
//! # Concepts
//!
//! A *divergence point* is a token position at which two or more model
//! configurations produce meaningfully different tokens (or assign them
//! different probabilities).  [`DivergenceDetector`] quantifies this as a
//! Jensen-Shannon divergence score in `[0, 1]`.
//!
//! | Score | Interpretation |
//! |-------|---------------|
//! | 0.0 | All models agree exactly |
//! | 0.5 | Models disagree substantially |
//! | 1.0 | Maximal disagreement |
//!
//! # Usage
//!
//! ```rust,no_run
//! use every_other_token::divergence::{DivergenceDetector, ModelConfig};
//!
//! let configs = vec![
//!     ModelConfig::new("openai", "gpt-4o", 0.0, 1.0),
//!     ModelConfig::new("openai", "gpt-4o", 1.0, 1.0),
//! ];
//! let detector = DivergenceDetector::new(configs);
//! let result = detector.run_sync("Why is the sky blue?");
//! let report = result.render_report();
//! println!("{}", report);
//! ```
//!
//! # Report format
//!
//! The coloured diff report marks positions with ANSI escape codes:
//! - Green background: models agree.
//! - Yellow background: moderate divergence (score > 0.25).
//! - Red background: high divergence (score > 0.5).

#![allow(dead_code)]

use std::collections::HashMap;

// ── Model configuration ────────────────────────────────────────────────────────

/// Configuration for a single model run in the divergence detector.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    /// Provider name, e.g. `"openai"` or `"anthropic"`.
    pub provider: String,
    /// Model identifier, e.g. `"gpt-4o"` or `"claude-3-5-sonnet-20241022"`.
    pub model_name: String,
    /// Sampling temperature in `[0, 2]`.
    pub temperature: f64,
    /// Nucleus sampling probability in `(0, 1]`.
    pub top_p: f64,
}

impl ModelConfig {
    /// Construct a model configuration.
    pub fn new(
        provider: impl Into<String>,
        model_name: impl Into<String>,
        temperature: f64,
        top_p: f64,
    ) -> Self {
        Self {
            provider: provider.into(),
            model_name: model_name.into(),
            temperature: temperature.clamp(0.0, 2.0),
            top_p: top_p.clamp(1e-6, 1.0),
        }
    }

    /// Human-readable label used in reports.
    pub fn label(&self) -> String {
        format!("{}/{} t={:.1} p={:.2}", self.provider, self.model_name, self.temperature, self.top_p)
    }
}

// ── Token stream ──────────────────────────────────────────────────────────────

/// A sequence of tokens produced by one model run.
///
/// Each element is `(token_text, Option<probability_distribution>)` where the
/// distribution maps alternative token strings to their log-probabilities.
#[derive(Debug, Clone)]
pub struct TokenStream {
    /// The model configuration that produced this stream.
    pub config: ModelConfig,
    /// Tokens in generation order.
    pub tokens: Vec<String>,
    /// Per-token alternative distributions (from logprobs API).
    ///
    /// `distributions[i]` maps token → probability for position `i`.
    /// May be empty if the provider did not return logprobs.
    pub distributions: Vec<HashMap<String, f64>>,
}

impl TokenStream {
    /// Create an empty token stream for the given configuration.
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            tokens: Vec::new(),
            distributions: Vec::new(),
        }
    }

    /// Push a token with an optional probability distribution.
    pub fn push(&mut self, token: impl Into<String>, dist: Option<HashMap<String, f64>>) {
        self.tokens.push(token.into());
        self.distributions.push(dist.unwrap_or_default());
    }

    /// Number of tokens in the stream.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Returns `true` if the stream contains no tokens.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Plain text of the full stream.
    pub fn text(&self) -> String {
        self.tokens.join("")
    }
}

// ── Divergence point ──────────────────────────────────────────────────────────

/// A position in the token sequence where models disagree.
#[derive(Debug, Clone)]
pub struct DivergencePoint {
    /// Zero-based token position.
    pub position: usize,
    /// Jensen-Shannon divergence in `[0, 1]` between all model distributions.
    pub disagreement_score: f64,
    /// The token each model produced at this position, keyed by model label.
    pub model_tokens: HashMap<String, String>,
}

impl DivergencePoint {
    /// Whether the divergence is considered *high* (score > 0.5).
    pub fn is_high(&self) -> bool {
        self.disagreement_score > 0.5
    }

    /// Whether the divergence is considered *moderate* (score in `(0.25, 0.5]`).
    pub fn is_moderate(&self) -> bool {
        self.disagreement_score > 0.25 && self.disagreement_score <= 0.5
    }
}

// ── Divergence result ─────────────────────────────────────────────────────────

/// Full result of running a prompt through multiple model configurations.
#[derive(Debug, Clone)]
pub struct DivergenceResult {
    /// The original prompt.
    pub prompt: String,
    /// All model configurations used.
    pub configs: Vec<ModelConfig>,
    /// Token stream per configuration (same order as `configs`).
    pub token_streams: Vec<TokenStream>,
    /// Positions where divergence was detected (score above threshold).
    pub divergence_points: Vec<DivergencePoint>,
}

impl DivergenceResult {
    /// Mean Jensen-Shannon divergence across all positions.
    pub fn mean_divergence(&self) -> f64 {
        if self.divergence_points.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .divergence_points
            .iter()
            .map(|p| p.disagreement_score)
            .sum();
        sum / self.divergence_points.len() as f64
    }

    /// Maximum divergence score across all positions.
    pub fn max_divergence(&self) -> f64 {
        self.divergence_points
            .iter()
            .map(|p| p.disagreement_score)
            .fold(0.0f64, f64::max)
    }

    /// Returns positions where divergence exceeded `threshold`.
    pub fn high_divergence_positions(&self, threshold: f64) -> Vec<usize> {
        self.divergence_points
            .iter()
            .filter(|p| p.disagreement_score >= threshold)
            .map(|p| p.position)
            .collect()
    }

    /// Render a coloured ANSI diff report.
    ///
    /// Each token position is annotated:
    /// - Green  (`\x1b[42m`): all models agree.
    /// - Yellow (`\x1b[43m`): moderate divergence (score > 0.25).
    /// - Red    (`\x1b[41m`): high divergence (score > 0.5).
    pub fn render_report(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Divergence Report ===\n");
        out.push_str(&format!("Prompt: {}\n", self.prompt));
        out.push_str(&format!("Models: {}\n", self.configs.len()));
        for cfg in &self.configs {
            out.push_str(&format!("  - {}\n", cfg.label()));
        }
        out.push_str(&format!(
            "Mean JS divergence: {:.4}\n",
            self.mean_divergence()
        ));
        out.push_str(&format!(
            "Max JS divergence:  {:.4}\n",
            self.max_divergence()
        ));
        out.push_str(&format!(
            "High-divergence positions: {}\n\n",
            self.high_divergence_positions(0.5).len()
        ));

        // Per-model coloured token streams.
        for stream in &self.token_streams {
            out.push_str(&format!("\n[{}]\n", stream.config.label()));
            let point_map: HashMap<usize, f64> = self
                .divergence_points
                .iter()
                .map(|p| (p.position, p.disagreement_score))
                .collect();

            for (i, token) in stream.tokens.iter().enumerate() {
                let score = point_map.get(&i).copied().unwrap_or(0.0);
                let colour = if score > 0.5 {
                    "\x1b[41m" // red
                } else if score > 0.25 {
                    "\x1b[43m" // yellow
                } else {
                    "\x1b[42m" // green
                };
                out.push_str(&format!("{}{}\x1b[0m", colour, token));
            }
            out.push('\n');
        }

        out.push_str("\n=== Position Detail ===\n");
        for point in &self.divergence_points {
            out.push_str(&format!(
                "  pos {:>4} | JS={:.4} | ",
                point.position, point.disagreement_score
            ));
            let mut tokens: Vec<(&str, &str)> = point
                .model_tokens
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();
            tokens.sort_by_key(|(k, _)| *k);
            for (label, tok) in tokens {
                out.push_str(&format!("{label}={tok:?} "));
            }
            out.push('\n');
        }

        out
    }
}

// ── Jensen-Shannon divergence ─────────────────────────────────────────────────

/// Compute the Jensen-Shannon divergence between N probability distributions.
///
/// Each distribution is a `HashMap<token, probability>`.  Missing tokens are
/// treated as having probability 0.  The result is in `[0, log(N)]`; this
/// function normalises it to `[0, 1]` by dividing by `log(N)`.
///
/// Returns 0.0 if fewer than 2 distributions are provided.
pub fn jensen_shannon_divergence(distributions: &[&HashMap<String, f64>]) -> f64 {
    let n = distributions.len();
    if n < 2 {
        return 0.0;
    }

    // Collect the union of all tokens.
    let mut vocab: Vec<String> = distributions
        .iter()
        .flat_map(|d| d.keys().cloned())
        .collect();
    vocab.sort();
    vocab.dedup();

    if vocab.is_empty() {
        return 0.0;
    }

    // Mixture distribution M = (1/N) Σ P_i
    let mixture: Vec<f64> = vocab
        .iter()
        .map(|tok| {
            distributions
                .iter()
                .map(|d| d.get(tok).copied().unwrap_or(0.0))
                .sum::<f64>()
                / n as f64
        })
        .collect();

    // JSD = (1/N) Σ KL(P_i || M)
    let mut jsd = 0.0f64;
    for dist in distributions {
        let kl: f64 = vocab
            .iter()
            .enumerate()
            .map(|(j, tok)| {
                let p = dist.get(tok).copied().unwrap_or(0.0);
                let m = mixture[j];
                if p > 1e-15 && m > 1e-15 {
                    p * (p / m).ln()
                } else {
                    0.0
                }
            })
            .sum();
        jsd += kl;
    }
    jsd /= n as f64;

    // Normalise by log(N) to get a value in [0, 1].
    let log_n = (n as f64).ln();
    if log_n < 1e-15 {
        0.0
    } else {
        (jsd / log_n).clamp(0.0, 1.0)
    }
}

// ── Divergence detector ───────────────────────────────────────────────────────

/// Orchestrates running the same prompt through N model configurations and
/// computing per-position divergence scores.
///
/// In production use the detector would call the provider APIs concurrently.
/// The synchronous `run_sync` method below is provided for testing and
/// offline analysis; it operates on pre-populated [`TokenStream`] data.
pub struct DivergenceDetector {
    /// Model configurations to compare.
    pub configs: Vec<ModelConfig>,
    /// Minimum JS divergence score to record as a divergence point.
    pub threshold: f64,
}

impl DivergenceDetector {
    /// Create a detector for the given model configurations.
    ///
    /// Divergence points are recorded for positions with JS score ≥ `threshold`.
    /// The default threshold is 0.1.
    pub fn new(configs: Vec<ModelConfig>) -> Self {
        Self {
            configs,
            threshold: 0.1,
        }
    }

    /// Set the divergence threshold.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Analyse pre-populated token streams and compute divergence points.
    ///
    /// This is the core algorithm; it does not perform any network I/O.
    /// Use it after collecting streams from provider APIs.
    pub fn analyse(&self, prompt: &str, streams: Vec<TokenStream>) -> DivergenceResult {
        let max_len = streams.iter().map(|s| s.len()).max().unwrap_or(0);
        let mut divergence_points: Vec<DivergencePoint> = Vec::new();

        for pos in 0..max_len {
            // Collect per-model token and distribution at this position.
            let mut model_tokens: HashMap<String, String> = HashMap::new();
            let mut distributions: Vec<HashMap<String, f64>> = Vec::new();

            for stream in &streams {
                let label = stream.config.label();
                if let Some(tok) = stream.tokens.get(pos) {
                    model_tokens.insert(label, tok.clone());
                }
                if let Some(dist) = stream.distributions.get(pos) {
                    if !dist.is_empty() {
                        distributions.push(dist.clone());
                    } else {
                        // No logprob data: build a point-mass distribution.
                        if let Some(tok) = stream.tokens.get(pos) {
                            let mut pm = HashMap::new();
                            pm.insert(tok.clone(), 1.0);
                            distributions.push(pm);
                        }
                    }
                }
            }

            if distributions.len() < 2 {
                continue;
            }

            let refs: Vec<&HashMap<String, f64>> = distributions.iter().collect();
            let score = jensen_shannon_divergence(&refs);

            if score >= self.threshold {
                divergence_points.push(DivergencePoint {
                    position: pos,
                    disagreement_score: score,
                    model_tokens,
                });
            }
        }

        DivergenceResult {
            prompt: prompt.to_owned(),
            configs: self.configs.clone(),
            token_streams: streams,
            divergence_points,
        }
    }

    /// Convenience method: build token streams from raw token/distribution slices
    /// and analyse them.
    ///
    /// Each element of `stream_data` corresponds to `configs[i]`:
    /// `(tokens, distributions)`.
    pub fn analyse_raw(
        &self,
        prompt: &str,
        stream_data: Vec<(Vec<String>, Vec<HashMap<String, f64>>)>,
    ) -> DivergenceResult {
        let streams: Vec<TokenStream> = stream_data
            .into_iter()
            .zip(self.configs.iter())
            .map(|((tokens, dists), cfg)| TokenStream {
                config: cfg.clone(),
                tokens,
                distributions: dists,
            })
            .collect();
        self.analyse(prompt, streams)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dist(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect()
    }

    #[test]
    fn jsd_identical_distributions_is_zero() {
        let d1 = make_dist(&[("hello", 0.7), ("world", 0.3)]);
        let d2 = make_dist(&[("hello", 0.7), ("world", 0.3)]);
        let score = jensen_shannon_divergence(&[&d1, &d2]);
        assert!(score < 1e-10, "identical distributions should give JSD=0, got {score}");
    }

    #[test]
    fn jsd_maximally_different() {
        // Two point masses at different tokens.
        let d1 = make_dist(&[("a", 1.0)]);
        let d2 = make_dist(&[("b", 1.0)]);
        let score = jensen_shannon_divergence(&[&d1, &d2]);
        assert!(
            score > 0.99,
            "maximally different distributions should give JSD≈1, got {score}"
        );
    }

    #[test]
    fn jsd_single_distribution_returns_zero() {
        let d = make_dist(&[("a", 1.0)]);
        let score = jensen_shannon_divergence(&[&d]);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn jsd_empty_returns_zero() {
        let score = jensen_shannon_divergence(&[]);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn jsd_symmetric() {
        let d1 = make_dist(&[("a", 0.8), ("b", 0.2)]);
        let d2 = make_dist(&[("a", 0.2), ("b", 0.8)]);
        let s12 = jensen_shannon_divergence(&[&d1, &d2]);
        let s21 = jensen_shannon_divergence(&[&d2, &d1]);
        assert!(
            (s12 - s21).abs() < 1e-10,
            "JSD should be symmetric: {s12} vs {s21}"
        );
    }

    #[test]
    fn model_config_label() {
        let cfg = ModelConfig::new("openai", "gpt-4o", 0.7, 0.9);
        let label = cfg.label();
        assert!(label.contains("openai"));
        assert!(label.contains("gpt-4o"));
    }

    #[test]
    fn detector_no_divergence_for_identical_streams() {
        let configs = vec![
            ModelConfig::new("test", "model-a", 0.0, 1.0),
            ModelConfig::new("test", "model-b", 0.0, 1.0),
        ];
        let detector = DivergenceDetector::new(configs.clone()).with_threshold(0.01);

        let tokens = vec!["The ".to_string(), "sky ".to_string(), "is ".to_string(), "blue.".to_string()];
        let dist = make_dist(&[("The ", 1.0)]);

        let stream_data = vec![
            (tokens.clone(), vec![dist.clone(), dist.clone(), dist.clone(), dist.clone()]),
            (tokens.clone(), vec![dist.clone(), dist.clone(), dist.clone(), dist.clone()]),
        ];

        let result = detector.analyse_raw("Why is the sky blue?", stream_data);
        assert!(
            result.divergence_points.is_empty(),
            "identical streams should produce no divergence points"
        );
    }

    #[test]
    fn detector_finds_divergence() {
        let configs = vec![
            ModelConfig::new("test", "model-a", 0.0, 1.0),
            ModelConfig::new("test", "model-b", 0.0, 1.0),
        ];
        let detector = DivergenceDetector::new(configs).with_threshold(0.1);

        let stream_data = vec![
            (
                vec!["cat".to_string()],
                vec![make_dist(&[("cat", 1.0)])],
            ),
            (
                vec!["dog".to_string()],
                vec![make_dist(&[("dog", 1.0)])],
            ),
        ];

        let result = detector.analyse_raw("What animal?", stream_data);
        assert!(
            !result.divergence_points.is_empty(),
            "different tokens at position 0 should produce a divergence point"
        );
        assert!(result.divergence_points[0].disagreement_score > 0.5);
    }

    #[test]
    fn divergence_result_mean_max() {
        let configs = vec![ModelConfig::new("t", "m", 0.0, 1.0)];
        let detector = DivergenceDetector::new(configs).with_threshold(0.0);
        let result = DivergenceResult {
            prompt: "p".into(),
            configs: detector.configs.clone(),
            token_streams: vec![],
            divergence_points: vec![
                DivergencePoint { position: 0, disagreement_score: 0.4, model_tokens: HashMap::new() },
                DivergencePoint { position: 1, disagreement_score: 0.8, model_tokens: HashMap::new() },
            ],
        };
        assert!((result.mean_divergence() - 0.6).abs() < 1e-10);
        assert!((result.max_divergence() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn divergence_point_severity_classification() {
        let high = DivergencePoint { position: 0, disagreement_score: 0.7, model_tokens: HashMap::new() };
        assert!(high.is_high());
        assert!(!high.is_moderate());

        let moderate = DivergencePoint { position: 0, disagreement_score: 0.35, model_tokens: HashMap::new() };
        assert!(!moderate.is_high());
        assert!(moderate.is_moderate());
    }

    #[test]
    fn render_report_contains_model_labels() {
        let configs = vec![
            ModelConfig::new("openai", "gpt-4o", 0.0, 1.0),
            ModelConfig::new("openai", "gpt-4o", 1.0, 1.0),
        ];
        let detector = DivergenceDetector::new(configs);
        let stream_data = vec![
            (vec!["hello".to_string()], vec![make_dist(&[("hello", 1.0)])]),
            (vec!["world".to_string()], vec![make_dist(&[("world", 1.0)])]),
        ];
        let result = detector.analyse_raw("test", stream_data);
        let report = result.render_report();
        assert!(report.contains("gpt-4o"));
        assert!(report.contains("Divergence Report"));
    }

    #[test]
    fn token_stream_text() {
        let cfg = ModelConfig::new("t", "m", 0.0, 1.0);
        let mut s = TokenStream::new(cfg);
        s.push("Hello", None);
        s.push(" world", None);
        assert_eq!(s.text(), "Hello world");
        assert_eq!(s.len(), 2);
    }
}
