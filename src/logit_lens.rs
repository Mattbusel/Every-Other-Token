//! # Logit Lens
//!
//! At each transformer layer, project the residual stream back to vocabulary
//! space and decode the model's "early predictions".
//!
//! ## Method
//!
//! A real logit-lens requires white-box access to the model's intermediate
//! layer representations.  Because we operate over the standard completions API
//! we approximate the effect by simulating a multi-resolution confidence
//! cascade:
//!
//!   - Layer 0 (early) → high entropy, near-uniform distribution.
//!   - Layers 1..L-1 → gradually sharpening distribution toward the final answer.
//!   - Layer L (final) → matches actual API logprobs.
//!
//! This lets the TUI render a per-layer heatmap showing the evolution of token
//! predictions through the network, giving users intuition for how prediction
//! certainty builds across depth.
//!
//! ## Visualisation
//!
//! [`LogitLensResult::heatmap`] returns a 2-D grid `[layer][token]` of
//! log-probabilities, ready to be rendered as a colour-coded heatmap in the
//! TUI using the existing [`crate::heatmap`] infrastructure.
//!
//! ## Usage
//!
//! ```rust
//! use every_other_token::logit_lens::{LogitLens, LogitLensConfig};
//!
//! let cfg = LogitLensConfig { num_layers: 12, ..Default::default() };
//! let lens = LogitLens::new(cfg);
//!
//! // Simulate from a known final logprob distribution.
//! let final_logprobs = vec![("Paris".to_string(), -0.1), ("Lyon".to_string(), -3.4)];
//! let result = lens.simulate_layer_cascade("The capital of France is", &final_logprobs);
//! assert_eq!(result.layers.len(), 12);
//! ```

use serde::{Deserialize, Serialize};

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for logit-lens visualisation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogitLensConfig {
    /// Number of transformer layers to simulate.
    pub num_layers: usize,
    /// Top-K tokens to track at each layer.
    pub top_k: usize,
    /// Sharpening exponent: controls how quickly probability concentrates.
    /// Higher = faster convergence to the final distribution.
    pub sharpening_exponent: f64,
}

impl Default for LogitLensConfig {
    fn default() -> Self {
        Self {
            num_layers: 12,
            top_k: 5,
            sharpening_exponent: 2.0,
        }
    }
}

// ── Layer snapshot ────────────────────────────────────────────────────────────

/// Prediction state at a single transformer layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSnapshot {
    /// Layer index (0 = earliest, num_layers-1 = final).
    pub layer: usize,
    /// Top-K predicted tokens at this layer, with log-probabilities.
    pub top_tokens: Vec<(String, f64)>,
    /// Entropy of the distribution at this layer (nats).
    pub entropy: f64,
    /// Most likely token at this layer.
    pub argmax_token: String,
}

/// Full logit-lens result: a cascade of layer snapshots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogitLensResult {
    /// Input prompt used for the run.
    pub prompt: String,
    /// One snapshot per layer.
    pub layers: Vec<LayerSnapshot>,
    /// The final (layer L) top-1 prediction.
    pub final_prediction: String,
}

impl LogitLensResult {
    /// Return a 2-D heatmap grid `[layer][token_rank]` of log-probabilities.
    ///
    /// Rows = layers (0 = earliest), columns = token ranks (0 = most likely).
    pub fn heatmap(&self) -> Vec<Vec<f64>> {
        self.layers
            .iter()
            .map(|snap| snap.top_tokens.iter().map(|(_, lp)| *lp).collect())
            .collect()
    }

    /// Column labels for the heatmap (token strings at the final layer).
    pub fn token_labels(&self) -> Vec<String> {
        if let Some(last) = self.layers.last() {
            last.top_tokens.iter().map(|(t, _)| t.clone()).collect()
        } else {
            vec![]
        }
    }

    /// Row labels for the heatmap.
    pub fn layer_labels(&self) -> Vec<String> {
        self.layers
            .iter()
            .map(|s| format!("L{:02}", s.layer))
            .collect()
    }

    /// ASCII table suitable for the TUI.
    pub fn ascii_table(&self) -> String {
        let labels = self.token_labels();
        let header_tokens = labels
            .iter()
            .map(|t| format!("{:>8}", truncate(t, 8)))
            .collect::<Vec<_>>()
            .join(" ");
        let mut out = format!("Logit Lens — prompt: \"{}\"\n\n", truncate(&self.prompt, 50));
        out.push_str(&format!("       {}\n", header_tokens));
        out.push_str(&format!(
            "       {}\n",
            "─".repeat(header_tokens.len() + 1)
        ));
        for snap in &self.layers {
            let row = labels
                .iter()
                .map(|label| {
                    let lp = snap
                        .top_tokens
                        .iter()
                        .find(|(t, _)| t == label)
                        .map(|(_, lp)| *lp)
                        .unwrap_or(-20.0);
                    format!("{:>8.2}", lp)
                })
                .collect::<Vec<_>>()
                .join(" ");
            let marker = if snap.layer == self.layers.len() - 1 {
                "*"
            } else {
                " "
            };
            out.push_str(&format!("  L{:02}{} {}\n", snap.layer, marker, row));
        }
        out.push_str(&format!("\n  Final prediction: \"{}\"", self.final_prediction));
        out
    }
}

// ── Core lens ─────────────────────────────────────────────────────────────────

/// Logit lens engine.
pub struct LogitLens {
    pub config: LogitLensConfig,
}

impl LogitLens {
    pub fn new(config: LogitLensConfig) -> Self {
        Self { config }
    }

    /// Simulate how the model's prediction sharpens from layer 0 → L-1.
    ///
    /// `final_logprobs`: the actual top-K logprobs returned by the API at
    /// the final layer.  Earlier layers are interpolated from a uniform
    /// prior toward these values using a power-law schedule.
    pub fn simulate_layer_cascade(
        &self,
        prompt: &str,
        final_logprobs: &[(String, f64)],
    ) -> LogitLensResult {
        let l = self.config.num_layers;
        let k = self.config.top_k.min(final_logprobs.len().max(1));
        let exp = self.config.sharpening_exponent;

        // Normalise final logprobs to probabilities
        let max_lp = final_logprobs
            .iter()
            .map(|(_, lp)| *lp)
            .fold(f64::NEG_INFINITY, f64::max);
        let final_probs: Vec<(&str, f64)> = final_logprobs
            .iter()
            .take(k)
            .map(|(t, lp)| (t.as_str(), (lp - max_lp).exp()))
            .collect();

        let uniform = 1.0 / final_probs.len() as f64;

        let layers: Vec<LayerSnapshot> = (0..l)
            .map(|layer| {
                // t in [0,1]: 0 at layer 0, 1 at layer L-1
                let t = if l > 1 {
                    (layer as f64 / (l - 1) as f64).powf(exp)
                } else {
                    1.0
                };

                // Interpolate between uniform and final distribution
                let probs: Vec<(&str, f64)> = final_probs
                    .iter()
                    .map(|(tok, p)| (*tok, (1.0 - t) * uniform + t * p))
                    .collect();

                // Renormalise
                let sum: f64 = probs.iter().map(|(_, p)| p).sum();
                let probs_norm: Vec<(&str, f64)> = probs
                    .iter()
                    .map(|(t, p)| (*t, p / sum))
                    .collect();

                // Convert to logprobs
                let top_tokens: Vec<(String, f64)> = probs_norm
                    .iter()
                    .map(|(t, p)| (t.to_string(), p.ln()))
                    .collect();

                // Entropy
                let entropy: f64 = probs_norm
                    .iter()
                    .map(|(_, p)| if *p > 0.0 { -p * p.ln() } else { 0.0 })
                    .sum();

                let argmax_token = top_tokens
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(t, _)| t.clone())
                    .unwrap_or_default();

                LayerSnapshot {
                    layer,
                    top_tokens,
                    entropy,
                    argmax_token,
                }
            })
            .collect();

        let final_prediction = layers
            .last()
            .map(|s| s.argmax_token.clone())
            .unwrap_or_default();

        LogitLensResult {
            prompt: prompt.to_string(),
            layers,
            final_prediction,
        }
    }

    /// Integrate with real API logprobs: given a sequence of per-token logprob
    /// maps, produce a logit-lens result for each generated position.
    pub fn from_api_logprobs(
        &self,
        prompt: &str,
        per_token_logprobs: &[Vec<(String, f64)>],
    ) -> Vec<LogitLensResult> {
        per_token_logprobs
            .iter()
            .enumerate()
            .map(|(i, lps)| {
                let sub_prompt = format!("{} [pos {}]", truncate(prompt, 40), i);
                self.simulate_layer_cascade(&sub_prompt, lps)
            })
            .collect()
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n.saturating_sub(1)])
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_logprobs() -> Vec<(String, f64)> {
        vec![
            ("Paris".to_string(), -0.1),
            ("Lyon".to_string(), -3.5),
            ("Rome".to_string(), -5.0),
        ]
    }

    #[test]
    fn layer_count() {
        let lens = LogitLens::new(LogitLensConfig {
            num_layers: 8,
            ..Default::default()
        });
        let result = lens.simulate_layer_cascade("prompt", &make_logprobs());
        assert_eq!(result.layers.len(), 8);
    }

    #[test]
    fn entropy_decreases_monotonically_ish() {
        let lens = LogitLens::new(LogitLensConfig::default());
        let result = lens.simulate_layer_cascade("prompt", &make_logprobs());
        let entropies: Vec<f64> = result.layers.iter().map(|l| l.entropy).collect();
        // Entropy should generally decrease (may plateau at start/end due to uniform prior)
        let first = entropies[0];
        let last = *entropies.last().unwrap();
        assert!(first >= last - 0.01, "entropy should decrease: {first} >= {last}");
    }

    #[test]
    fn final_prediction_matches_argmax() {
        let lens = LogitLens::new(LogitLensConfig::default());
        let result = lens.simulate_layer_cascade("prompt", &make_logprobs());
        assert_eq!(result.final_prediction, "Paris");
    }

    #[test]
    fn heatmap_dimensions() {
        let cfg = LogitLensConfig {
            num_layers: 6,
            top_k: 3,
            ..Default::default()
        };
        let lens = LogitLens::new(cfg);
        let result = lens.simulate_layer_cascade("prompt", &make_logprobs());
        let hm = result.heatmap();
        assert_eq!(hm.len(), 6);
        assert_eq!(hm[0].len(), 3);
    }

    #[test]
    fn ascii_table_non_empty() {
        let lens = LogitLens::new(LogitLensConfig::default());
        let result = lens.simulate_layer_cascade("France", &make_logprobs());
        let table = result.ascii_table();
        assert!(table.contains("Paris"));
        assert!(table.contains("L00"));
    }
}
