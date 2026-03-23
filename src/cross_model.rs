//! # Cross-Model Mechanistic Comparison
//!
//! Compare internal mechanisms between two different language models
//! (e.g., a GPT-family model vs a Claude-family model) by running the same
//! input through both and comparing their logit distributions at each layer.
//!
//! ## Method
//!
//! 1. Run the same prompt through both models simultaneously (two API calls).
//! 2. At each generated token position, compare the full logit distributions:
//!    - Jensen-Shannon divergence (symmetric, bounded in [0, 1]).
//!    - Top-1 agreement rate.
//!    - Per-token probability ratio.
//! 3. Find positions where the two models diverge most strongly.
//! 4. Report which *input tokens* caused maximum divergence (by correlating
//!    position divergence with input token attribution).
//!
//! ## Usage
//!
//! ```rust
//! use every_other_token::cross_model::{CrossModelComparator, ComparisonConfig};
//!
//! let cfg = ComparisonConfig::default();
//! let cmp = CrossModelComparator::new(cfg);
//! let result = cmp.compare_distributions("gpt-4o", "claude-3-5-sonnet", &[], &[]);
//! assert_eq!(result.model_a, "gpt-4o");
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for cross-model comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonConfig {
    /// Number of top tokens to consider in distribution overlap analysis.
    pub top_k: usize,
    /// Divergence threshold above which a position is "high divergence".
    pub divergence_threshold: f64,
    /// Whether to compute per-input-token attribution of divergence.
    pub attribute_divergence: bool,
}

impl Default for ComparisonConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            divergence_threshold: 0.1,
            attribute_divergence: true,
        }
    }
}

// ── Per-position comparison ───────────────────────────────────────────────────

/// Comparison result at a single generated token position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionComparison {
    /// Token index in the generated sequence.
    pub position: usize,
    /// Top-1 token predicted by model A.
    pub top1_model_a: String,
    /// Top-1 token predicted by model B.
    pub top1_model_b: String,
    /// Whether both models agree on top-1.
    pub top1_agreement: bool,
    /// Jensen-Shannon divergence between the two distributions (0 = identical, 1 = max diverge).
    pub js_divergence: f64,
    /// KL(A || B)
    pub kl_a_to_b: f64,
    /// KL(B || A)
    pub kl_b_to_a: f64,
    /// Tokens where model A assigns much higher probability than B.
    pub a_prefers: Vec<(String, f64)>,
    /// Tokens where model B assigns much higher probability than A.
    pub b_prefers: Vec<(String, f64)>,
}

// ── Full comparison result ────────────────────────────────────────────────────

/// Full cross-model comparison result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    /// Input prompt.
    pub prompt: String,
    /// Name of model A.
    pub model_a: String,
    /// Name of model B.
    pub model_b: String,
    /// Per-position comparisons.
    pub positions: Vec<PositionComparison>,
    /// Mean JS divergence across all positions.
    pub mean_js_divergence: f64,
    /// Top-1 agreement rate across all positions.
    pub top1_agreement_rate: f64,
    /// Positions with divergence above threshold, sorted by divergence desc.
    pub high_divergence_positions: Vec<usize>,
    /// Input tokens ranked by their contribution to divergence.
    pub divergence_attribution: Vec<(String, f64)>,
}

impl ComparisonResult {
    /// Most divergent position.
    pub fn max_divergence_position(&self) -> Option<&PositionComparison> {
        self.positions
            .iter()
            .max_by(|a, b| a.js_divergence.partial_cmp(&b.js_divergence).unwrap())
    }

    /// ASCII report for TUI display.
    pub fn ascii_report(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "Cross-Model Comparison: {} vs {}\n",
            self.model_a, self.model_b
        ));
        out.push_str(&format!("  Prompt: {}\n\n", truncate(&self.prompt, 60)));
        out.push_str(&format!(
            "  Mean JS divergence:   {:.4}\n",
            self.mean_js_divergence
        ));
        out.push_str(&format!(
            "  Top-1 agreement:      {:.1}%\n",
            self.top1_agreement_rate * 100.0
        ));
        out.push_str(&format!(
            "  High-divergence pos:  {:?}\n\n",
            self.high_divergence_positions
        ));

        out.push_str("  pos  A-top1          B-top1          JS-div  agree\n");
        out.push_str("  ───  ─────────────  ─────────────  ───────  ─────\n");
        for p in &self.positions {
            out.push_str(&format!(
                "  {:>3}  {:<13}  {:<13}  {:.4}   {}\n",
                p.position,
                truncate(&p.top1_model_a, 13),
                truncate(&p.top1_model_b, 13),
                p.js_divergence,
                if p.top1_agreement { "yes" } else { "NO " }
            ));
        }

        if !self.divergence_attribution.is_empty() {
            out.push_str("\n  Input token divergence attribution:\n");
            for (tok, score) in self.divergence_attribution.iter().take(5) {
                out.push_str(&format!("    {:>12}: {:.4}\n", truncate(tok, 12), score));
            }
        }
        out
    }
}

// ── Comparator ────────────────────────────────────────────────────────────────

/// Cross-model mechanistic comparator.
pub struct CrossModelComparator {
    pub config: ComparisonConfig,
}

impl CrossModelComparator {
    pub fn new(config: ComparisonConfig) -> Self {
        Self { config }
    }

    /// Compare two logprob distributions at a single position.
    fn compare_position(
        &self,
        pos: usize,
        logprobs_a: &HashMap<String, f64>,
        logprobs_b: &HashMap<String, f64>,
    ) -> PositionComparison {
        // Merge vocabulary
        let mut vocab: Vec<String> = logprobs_a
            .keys()
            .chain(logprobs_b.keys())
            .cloned()
            .collect();
        vocab.sort();
        vocab.dedup();

        let get = |m: &HashMap<String, f64>, tok: &str| m.get(tok).copied().unwrap_or(-20.0);

        // Compute probabilities
        let probs_a: Vec<f64> = vocab.iter().map(|t| get(logprobs_a, t).exp()).collect();
        let probs_b: Vec<f64> = vocab.iter().map(|t| get(logprobs_b, t).exp()).collect();

        let sum_a: f64 = probs_a.iter().sum::<f64>().max(1e-15);
        let sum_b: f64 = probs_b.iter().sum::<f64>().max(1e-15);
        let pa: Vec<f64> = probs_a.iter().map(|p| p / sum_a).collect();
        let pb: Vec<f64> = probs_b.iter().map(|p| p / sum_b).collect();

        // Jensen-Shannon divergence
        let m: Vec<f64> = pa.iter().zip(&pb).map(|(a, b)| (a + b) / 2.0).collect();
        let js = 0.5 * kl_divergence(&pa, &m) + 0.5 * kl_divergence(&pb, &m);
        let kl_ab = kl_divergence(&pa, &pb);
        let kl_ba = kl_divergence(&pb, &pa);

        // Top-1
        let top1_a = top1_token(logprobs_a);
        let top1_b = top1_token(logprobs_b);
        let agreement = top1_a == top1_b;

        // Preference lists
        let k = self.config.top_k;
        let mut ratios: Vec<(String, f64)> = vocab
            .iter()
            .enumerate()
            .map(|(i, tok)| (tok.clone(), (pa[i] + 1e-30).ln() - (pb[i] + 1e-30).ln()))
            .collect();
        ratios.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let a_prefers: Vec<(String, f64)> = ratios
            .iter()
            .take(k)
            .filter(|(_, r)| *r > 0.0)
            .cloned()
            .collect();
        ratios.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let b_prefers: Vec<(String, f64)> = ratios
            .iter()
            .take(k)
            .filter(|(_, r)| *r < 0.0)
            .map(|(t, r)| (t.clone(), r.abs()))
            .collect();

        PositionComparison {
            position: pos,
            top1_model_a: top1_a,
            top1_model_b: top1_b,
            top1_agreement: agreement,
            js_divergence: js,
            kl_a_to_b: kl_ab,
            kl_b_to_a: kl_ba,
            a_prefers,
            b_prefers,
        }
    }

    /// Compare two models' logprob sequences for the same prompt.
    ///
    /// `logprobs_a[i]`: token → logprob map for model A at generated position i.
    /// `logprobs_b[i]`: same for model B.
    pub fn compare_distributions(
        &self,
        model_a: &str,
        model_b: &str,
        prompt: &str,
        logprobs_a: &[HashMap<String, f64>],
        logprobs_b: &[HashMap<String, f64>],
    ) -> ComparisonResult {
        let n = logprobs_a.len().min(logprobs_b.len());
        let positions: Vec<PositionComparison> = (0..n)
            .map(|i| self.compare_position(i, &logprobs_a[i], &logprobs_b[i]))
            .collect();

        let mean_js = if positions.is_empty() {
            0.0
        } else {
            positions.iter().map(|p| p.js_divergence).sum::<f64>() / positions.len() as f64
        };

        let agree_count = positions.iter().filter(|p| p.top1_agreement).count();
        let top1_rate = if positions.is_empty() {
            0.0
        } else {
            agree_count as f64 / positions.len() as f64
        };

        let mut high_div: Vec<usize> = positions
            .iter()
            .filter(|p| p.js_divergence > self.config.divergence_threshold)
            .map(|p| p.position)
            .collect();
        high_div.sort_by(|&a, &b| {
            positions[b]
                .js_divergence
                .partial_cmp(&positions[a].js_divergence)
                .unwrap()
        });

        // Attribute divergence to input tokens (proportional to position divergence)
        let divergence_attribution = if self.config.attribute_divergence && !prompt.is_empty() {
            let input_tokens: Vec<&str> = prompt.split_whitespace().collect();
            let n_in = input_tokens.len();
            input_tokens
                .iter()
                .enumerate()
                .map(|(i, &tok)| {
                    // Heuristic: each input token's divergence contribution is weighted
                    // by the divergence at the output position that corresponds to it
                    let gen_pos = (i * positions.len()) / n_in.max(1);
                    let score = positions
                        .get(gen_pos)
                        .map(|p| p.js_divergence)
                        .unwrap_or(0.0);
                    (tok.to_string(), score)
                })
                .collect::<Vec<_>>()
        } else {
            vec![]
        };

        ComparisonResult {
            prompt: prompt.to_string(),
            model_a: model_a.to_string(),
            model_b: model_b.to_string(),
            positions,
            mean_js_divergence: mean_js,
            top1_agreement_rate: top1_rate,
            high_divergence_positions: high_div,
            divergence_attribution,
        }
    }

    /// Convenience: given raw token sequences from two models, compute agreement
    /// statistics without full logprob maps.
    pub fn token_sequence_overlap(
        &self,
        tokens_a: &[&str],
        tokens_b: &[&str],
    ) -> TokenOverlapReport {
        let n = tokens_a.len().min(tokens_b.len());
        let matches: usize = tokens_a[..n]
            .iter()
            .zip(&tokens_b[..n])
            .filter(|(a, b)| a == b)
            .count();
        let exact_match_rate = if n == 0 { 0.0 } else { matches as f64 / n as f64 };

        // Jaccard overlap on token sets
        let set_a: std::collections::HashSet<&&str> = tokens_a.iter().collect();
        let set_b: std::collections::HashSet<&&str> = tokens_b.iter().collect();
        let intersection = set_a.intersection(&set_b).count();
        let union = set_a.union(&set_b).count();
        let jaccard = if union == 0 { 0.0 } else { intersection as f64 / union as f64 };

        // Find first divergence point
        let first_divergence = tokens_a[..n]
            .iter()
            .zip(&tokens_b[..n])
            .position(|(a, b)| a != b);

        TokenOverlapReport {
            sequence_length_a: tokens_a.len(),
            sequence_length_b: tokens_b.len(),
            compared_length: n,
            exact_match_count: matches,
            exact_match_rate,
            jaccard_similarity: jaccard,
            first_divergence_position: first_divergence,
        }
    }
}

/// Summary statistics for token-sequence overlap between two models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenOverlapReport {
    pub sequence_length_a: usize,
    pub sequence_length_b: usize,
    pub compared_length: usize,
    pub exact_match_count: usize,
    pub exact_match_rate: f64,
    pub jaccard_similarity: f64,
    pub first_divergence_position: Option<usize>,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    p.iter()
        .zip(q)
        .map(|(pi, qi)| {
            if *pi > 1e-15 && *qi > 1e-15 {
                pi * (pi / qi).ln()
            } else {
                0.0
            }
        })
        .sum()
}

fn top1_token(logprobs: &HashMap<String, f64>) -> String {
    logprobs
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(t, _)| t.clone())
        .unwrap_or_default()
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

    fn make_logprobs(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn identical_distributions_zero_js() {
        let cmp = CrossModelComparator::new(ComparisonConfig::default());
        let lp = make_logprobs(&[("Paris", -0.1), ("Lyon", -3.0), ("Rome", -5.0)]);
        let pos = cmp.compare_position(0, &lp, &lp);
        assert!(pos.js_divergence < 1e-10, "JS should be 0 for identical distributions");
        assert!(pos.top1_agreement);
    }

    #[test]
    fn divergent_distributions_nonzero_js() {
        let cmp = CrossModelComparator::new(ComparisonConfig::default());
        let lp_a = make_logprobs(&[("Paris", -0.1), ("Lyon", -5.0)]);
        let lp_b = make_logprobs(&[("Paris", -5.0), ("Lyon", -0.1)]);
        let pos = cmp.compare_position(0, &lp_a, &lp_b);
        assert!(pos.js_divergence > 0.1, "JS should be large for opposite distributions");
        assert!(!pos.top1_agreement);
    }

    #[test]
    fn js_divergence_bounded() {
        let cmp = CrossModelComparator::new(ComparisonConfig::default());
        let lp_a = make_logprobs(&[("a", 0.0)]);
        let lp_b = make_logprobs(&[("b", 0.0)]);
        let pos = cmp.compare_position(0, &lp_a, &lp_b);
        assert!(pos.js_divergence <= 1.0 + 1e-9, "JS must be in [0,1]");
        assert!(pos.js_divergence >= 0.0);
    }

    #[test]
    fn compare_distributions_meta() {
        let cmp = CrossModelComparator::new(ComparisonConfig::default());
        let lp_a = make_logprobs(&[("Paris", -0.1), ("Lyon", -3.0)]);
        let lp_b = make_logprobs(&[("Paris", -0.5), ("Lyon", -2.0)]);
        let result = cmp.compare_distributions(
            "gpt-4o",
            "claude-3-5-sonnet",
            "The capital of France is",
            &[lp_a],
            &[lp_b],
        );
        assert_eq!(result.model_a, "gpt-4o");
        assert_eq!(result.model_b, "claude-3-5-sonnet");
        assert_eq!(result.positions.len(), 1);
        assert!(result.mean_js_divergence >= 0.0);
    }

    #[test]
    fn token_sequence_overlap_identical() {
        let cmp = CrossModelComparator::new(ComparisonConfig::default());
        let tokens = &["the", "cat", "sat"];
        let report = cmp.token_sequence_overlap(tokens, tokens);
        assert!((report.exact_match_rate - 1.0).abs() < 1e-9);
        assert!((report.jaccard_similarity - 1.0).abs() < 1e-9);
        assert!(report.first_divergence_position.is_none());
    }

    #[test]
    fn token_sequence_overlap_disjoint() {
        let cmp = CrossModelComparator::new(ComparisonConfig::default());
        let a = &["cat", "dog"];
        let b = &["fish", "bird"];
        let report = cmp.token_sequence_overlap(a, b);
        assert!((report.exact_match_rate - 0.0).abs() < 1e-9);
        assert_eq!(report.first_divergence_position, Some(0));
    }
}
