//! Cross-model structural comparison.
//!
//! [`CrossModelAnalyzer`] accepts token streams from multiple models and computes
//! statistical divergence measures between them:
//!
//! - Jensen-Shannon divergence between token-count distributions
//! - NxN Pearson correlation matrix across token-confidence sequences
//! - Agreement score (fraction of positions where all models agree on the token)
//! - Structural diff showing regions of high divergence
//!
//! ## Example
//!
//! ```rust
//! use every_other_token::comparison::CrossModelAnalyzer;
//! use every_other_token::TokenEvent;
//!
//! let mut analyzer = CrossModelAnalyzer::new();
//! // add streams, then …
//! let score = analyzer.agreement_score();
//! assert!((0.0..=1.0).contains(&score));
//! ```

use crate::TokenEvent;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// TokenDistribution
// ---------------------------------------------------------------------------

/// Descriptive statistics and a histogram over a sequence of token-count values.
#[derive(Debug, Clone)]
pub struct TokenDistribution {
    /// Arithmetic mean of the values.
    pub mean: f64,
    /// Population standard deviation.
    pub stddev: f64,
    /// Minimum value observed.
    pub min: f64,
    /// Maximum value observed.
    pub max: f64,
    /// Histogram buckets: each entry is `(bucket_start, count)`.
    /// Buckets are evenly spaced across `[min, max]`.
    pub histogram: Vec<(f64, usize)>,
}

impl TokenDistribution {
    /// Build a distribution from a slice of values.
    ///
    /// Returns `None` when `values` is empty.
    pub fn from_values(values: &[f64], num_buckets: usize) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        let num_buckets = num_buckets.max(1);

        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let stddev = variance.sqrt();

        let range = if (max - min).abs() < f64::EPSILON { 1.0 } else { max - min };
        let bucket_width = range / num_buckets as f64;

        let mut counts = vec![0usize; num_buckets];
        for &v in values {
            let idx = ((v - min) / bucket_width).floor() as usize;
            let idx = idx.min(num_buckets - 1);
            counts[idx] += 1;
        }

        let histogram = counts
            .into_iter()
            .enumerate()
            .map(|(i, c)| (min + i as f64 * bucket_width, c))
            .collect();

        Some(Self { mean, stddev, min, max, histogram })
    }
}

// ---------------------------------------------------------------------------
// ModelComparison
// ---------------------------------------------------------------------------

/// Captures token streams from multiple models for comparison.
#[derive(Debug, Clone, Default)]
pub struct ModelComparison {
    /// Model identifier to token stream mapping.
    pub streams: HashMap<String, Vec<TokenEvent>>,
}

impl ModelComparison {
    /// Create a new, empty `ModelComparison`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a model's token stream.
    pub fn add_stream(&mut self, model_id: String, tokens: Vec<TokenEvent>) {
        self.streams.insert(model_id, tokens);
    }
}

// ---------------------------------------------------------------------------
// DiffRegion
// ---------------------------------------------------------------------------

/// A contiguous region within the token sequence where models structurally diverge.
#[derive(Debug, Clone)]
pub struct DiffRegion {
    /// First token index of this region (inclusive).
    pub start_position: usize,
    /// Last token index of this region (inclusive).
    pub end_position: usize,
    /// Mean divergence score for this region (0.0 = identical, 1.0 = maximally different).
    pub divergence_score: f64,
    /// Model IDs that agree with each other in this region.
    pub models_agreeing: Vec<String>,
    /// Model IDs that disagree with the majority in this region.
    pub models_disagreeing: Vec<String>,
}

// ---------------------------------------------------------------------------
// ComparisonReport
// ---------------------------------------------------------------------------

/// Formatted report comparing model outputs.
#[derive(Debug, Clone)]
pub struct ComparisonReport {
    /// Human-readable summary lines.
    pub lines: Vec<String>,
}

impl ComparisonReport {
    /// Render the report as a single multi-line string.
    pub fn render(&self) -> String {
        self.lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// CrossModelAnalyzer
// ---------------------------------------------------------------------------

/// Registers token streams from multiple models and computes comparison metrics.
#[derive(Debug, Default)]
pub struct CrossModelAnalyzer {
    /// Ordered list of model IDs.
    model_ids: Vec<String>,
    /// Token event sequences in the same order as `model_ids`.
    streams: Vec<Vec<TokenEvent>>,
}

impl CrossModelAnalyzer {
    /// Create a new, empty analyzer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a model's token output stream.
    ///
    /// If the same `model_id` is added twice the second call overwrites the first.
    pub fn add_model_stream(&mut self, model_id: String, tokens: Vec<TokenEvent>) {
        if let Some(idx) = self.model_ids.iter().position(|id| id == &model_id) {
            self.streams[idx] = tokens;
        } else {
            self.model_ids.push(model_id);
            self.streams.push(tokens);
        }
    }

    /// Number of models currently registered.
    pub fn model_count(&self) -> usize {
        self.model_ids.len()
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Extract per-position confidence values for a stream (falling back to
    /// `importance` when confidence is absent).
    fn confidence_sequence(stream: &[TokenEvent]) -> Vec<f64> {
        stream
            .iter()
            .map(|e| e.confidence.map(|c| c as f64).unwrap_or(e.importance))
            .collect()
    }

    /// Build a normalised probability distribution over `values` using the
    /// supplied `num_buckets` for binning.  Returns an empty vec when
    /// `values` is empty.
    fn normalised_histogram(values: &[f64], num_buckets: usize) -> Vec<f64> {
        if values.is_empty() {
            return vec![];
        }
        let num_buckets = num_buckets.max(1);
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = if (max - min).abs() < f64::EPSILON { 1.0 } else { max - min };
        let bw = range / num_buckets as f64;

        let mut counts = vec![0usize; num_buckets];
        for &v in values {
            let idx = ((v - min) / bw).floor() as usize;
            counts[idx.min(num_buckets - 1)] += 1;
        }

        let total = values.len() as f64;
        counts.iter().map(|&c| c as f64 / total).collect()
    }

    // -----------------------------------------------------------------------
    // Public metrics
    // -----------------------------------------------------------------------

    /// Jensen-Shannon divergence between the token-confidence distributions
    /// of all registered models.
    ///
    /// Returns `0.0` when fewer than two models are registered.
    /// The value is in `[0.0, 1.0]` (log base-2 variant).
    pub fn compute_divergence(&self) -> f64 {
        if self.streams.len() < 2 {
            return 0.0;
        }
        const BUCKETS: usize = 20;

        let hists: Vec<Vec<f64>> = self
            .streams
            .iter()
            .map(|s| Self::normalised_histogram(&Self::confidence_sequence(s), BUCKETS))
            .collect();

        let n = hists.len() as f64;
        // Mixture distribution M = (1/n) * sum(P_i)
        let m: Vec<f64> = (0..BUCKETS)
            .map(|b| hists.iter().map(|h| h.get(b).copied().unwrap_or(0.0)).sum::<f64>() / n)
            .collect();

        let kl = |p: &Vec<f64>, q: &[f64]| -> f64 {
            p.iter()
                .zip(q.iter())
                .map(|(&pi, &qi)| {
                    if pi > 0.0 && qi > 0.0 {
                        pi * (pi / qi).log2()
                    } else {
                        0.0
                    }
                })
                .sum::<f64>()
        };

        let js = hists.iter().map(|p| kl(p, &m)).sum::<f64>() / n;
        // Clamp to [0.0, 1.0] to handle floating-point noise.
        js.max(0.0).min(1.0)
    }

    /// Compute the NxN Pearson correlation matrix between model confidence sequences.
    ///
    /// Entry `[i][j]` is the Pearson correlation between model `i` and model `j`.
    /// Returns an empty matrix when fewer than two models are registered.
    pub fn correlation_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.streams.len();
        if n < 2 {
            return vec![];
        }

        let seqs: Vec<Vec<f64>> = self.streams.iter().map(|s| Self::confidence_sequence(s)).collect();

        // Helper: Pearson r between two sequences (uses min-length prefix).
        let pearson = |a: &[f64], b: &[f64]| -> f64 {
            let len = a.len().min(b.len());
            if len == 0 {
                return 0.0;
            }
            let a = &a[..len];
            let b = &b[..len];
            let mean_a = a.iter().sum::<f64>() / len as f64;
            let mean_b = b.iter().sum::<f64>() / len as f64;
            let num: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - mean_a) * (y - mean_b)).sum();
            let den_a: f64 = a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>().sqrt();
            let den_b: f64 = b.iter().map(|y| (y - mean_b).powi(2)).sum::<f64>().sqrt();
            let den = den_a * den_b;
            if den.abs() < f64::EPSILON { 0.0 } else { (num / den).max(-1.0).min(1.0) }
        };

        (0..n)
            .map(|i| (0..n).map(|j| pearson(&seqs[i], &seqs[j])).collect())
            .collect()
    }

    /// Agreement score: fraction of token positions where every registered model
    /// produced the same token text.
    ///
    /// Returns `1.0` when fewer than two models are registered (trivially agree).
    /// Returns `0.0` when no tokens have been streamed.
    pub fn agreement_score(&self) -> f64 {
        if self.streams.len() < 2 {
            return 1.0;
        }
        let min_len = self.streams.iter().map(|s| s.len()).min().unwrap_or(0);
        if min_len == 0 {
            return 0.0;
        }
        let agreed = (0..min_len)
            .filter(|&pos| {
                let first = &self.streams[0][pos].original;
                self.streams[1..].iter().all(|s| &s[pos].original == first)
            })
            .count();
        agreed as f64 / min_len as f64
    }

    /// Returns regions where models structurally diverge.
    ///
    /// A window of `window_size` consecutive positions is scored; any window
    /// whose agreement fraction falls below `threshold` becomes a `DiffRegion`.
    /// Adjacent windows are merged when they overlap.
    pub fn structural_diff(&self) -> Vec<DiffRegion> {
        self.structural_diff_with_params(10, 0.8)
    }

    /// Parameterised variant of [`structural_diff`].
    pub fn structural_diff_with_params(&self, window_size: usize, threshold: f64) -> Vec<DiffRegion> {
        if self.streams.len() < 2 {
            return vec![];
        }
        let window_size = window_size.max(1);
        let min_len = self.streams.iter().map(|s| s.len()).min().unwrap_or(0);
        if min_len == 0 {
            return vec![];
        }

        let mut regions: Vec<DiffRegion> = vec![];

        let mut start = 0;
        while start + window_size <= min_len {
            let end = (start + window_size - 1).min(min_len - 1);
            let window_agreement = self.window_agreement(start, end);

            if window_agreement < threshold {
                let divergence = 1.0 - window_agreement;
                // Determine which models agree / disagree with the plurality.
                let (agreeing, disagreeing) = self.majority_split(start, end);

                // Merge with previous region if it overlaps or is adjacent.
                if let Some(last) = regions.last_mut() {
                    if last.end_position + 1 >= start {
                        last.end_position = end;
                        last.divergence_score = (last.divergence_score + divergence) / 2.0;
                        last.models_agreeing = agreeing;
                        last.models_disagreeing = disagreeing;
                        start += 1;
                        continue;
                    }
                }

                regions.push(DiffRegion {
                    start_position: start,
                    end_position: end,
                    divergence_score: divergence,
                    models_agreeing: agreeing,
                    models_disagreeing: disagreeing,
                });
            }
            start += 1;
        }

        regions
    }

    /// Mean agreement fraction over positions `[start, end]`.
    fn window_agreement(&self, start: usize, end: usize) -> f64 {
        if start > end {
            return 1.0;
        }
        let len = end - start + 1;
        let agreed = (start..=end)
            .filter(|&pos| {
                let first = &self.streams[0][pos].original;
                self.streams[1..].iter().all(|s| &s[pos].original == first)
            })
            .count();
        agreed as f64 / len as f64
    }

    /// Split model IDs into a majority-agreeing group and a disagreeing group
    /// based on the most-common token at each position in `[start, end]`.
    fn majority_split(&self, start: usize, end: usize) -> (Vec<String>, Vec<String>) {
        let n = self.streams.len();
        let mut agree_votes = vec![0usize; n];

        for pos in start..=end {
            // Build frequency map of original tokens at this position.
            let mut freq: HashMap<&str, usize> = HashMap::new();
            for s in &self.streams {
                *freq.entry(s[pos].original.as_str()).or_insert(0) += 1;
            }
            // Find plurality token.
            let plurality = freq
                .iter()
                .max_by_key(|(_, &c)| c)
                .map(|(&tok, _)| tok)
                .unwrap_or("");
            for (i, s) in self.streams.iter().enumerate() {
                if s[pos].original.as_str() == plurality {
                    agree_votes[i] += 1;
                }
            }
        }

        let total = end - start + 1;
        let half = total / 2;
        let mut agreeing = vec![];
        let mut disagreeing = vec![];
        for (i, &votes) in agree_votes.iter().enumerate() {
            if votes > half {
                agreeing.push(self.model_ids[i].clone());
            } else {
                disagreeing.push(self.model_ids[i].clone());
            }
        }
        (agreeing, disagreeing)
    }

    /// Build a `ComparisonReport` summarising all metrics.
    pub fn build_report(&self) -> ComparisonReport {
        let mut lines = vec!["=== Cross-Model Comparison Report ===".to_string()];

        lines.push(format!("Models registered: {}", self.model_ids.len()));
        for id in &self.model_ids {
            lines.push(format!("  - {}", id));
        }

        let agreement = self.agreement_score();
        lines.push(format!("Agreement score:    {:.4} ({:.1}%)", agreement, agreement * 100.0));

        let divergence = self.compute_divergence();
        lines.push(format!("JS divergence:      {:.4}", divergence));

        let corr = self.correlation_matrix();
        if !corr.is_empty() {
            lines.push("Correlation matrix:".to_string());
            for (i, row) in corr.iter().enumerate() {
                let cells: Vec<String> = row.iter().map(|v| format!("{:6.3}", v)).collect();
                lines.push(format!("  {:>20}  [{}]", self.model_ids[i], cells.join(", ")));
            }
        }

        let diffs = self.structural_diff();
        lines.push(format!("Divergent regions:  {}", diffs.len()));
        for region in &diffs {
            lines.push(format!(
                "  pos [{:4}..{:4}]  divergence={:.3}  agree={:?}  disagree={:?}",
                region.start_position,
                region.end_position,
                region.divergence_score,
                region.models_agreeing,
                region.models_disagreeing,
            ));
        }

        ComparisonReport { lines }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TokenEvent;

    fn make_event(original: &str, confidence: Option<f32>) -> TokenEvent {
        TokenEvent {
            text: original.to_string(),
            original: original.to_string(),
            index: 0,
            transformed: false,
            importance: 0.5,
            chaos_label: None,
            provider: None,
            confidence,
            perplexity: None,
            alternatives: vec![],
            is_error: false,
            arrival_ms: None,
        }
    }

    fn make_stream(tokens: &[(&str, f32)]) -> Vec<TokenEvent> {
        tokens.iter().map(|(t, c)| make_event(t, Some(*c))).collect()
    }

    #[test]
    fn test_add_model_stream_and_count() {
        let mut analyzer = CrossModelAnalyzer::new();
        analyzer.add_model_stream("gpt-4o".to_string(), make_stream(&[("hello", 0.9)]));
        analyzer.add_model_stream("claude".to_string(), make_stream(&[("hello", 0.8)]));
        assert_eq!(analyzer.model_count(), 2);
    }

    #[test]
    fn test_add_model_stream_overwrites_duplicate() {
        let mut analyzer = CrossModelAnalyzer::new();
        analyzer.add_model_stream("gpt".to_string(), make_stream(&[("a", 0.9)]));
        analyzer.add_model_stream("gpt".to_string(), make_stream(&[("b", 0.5)]));
        assert_eq!(analyzer.model_count(), 1);
        assert_eq!(analyzer.streams[0][0].original, "b");
    }

    #[test]
    fn test_agreement_score_identical() {
        let mut analyzer = CrossModelAnalyzer::new();
        let tokens: Vec<(&str, f32)> = vec![("the", 0.9), ("cat", 0.8), ("sat", 0.7)];
        analyzer.add_model_stream("m1".to_string(), make_stream(&tokens));
        analyzer.add_model_stream("m2".to_string(), make_stream(&tokens));
        assert!((analyzer.agreement_score() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_agreement_score_no_overlap() {
        let mut analyzer = CrossModelAnalyzer::new();
        analyzer.add_model_stream("m1".to_string(), make_stream(&[("a", 0.9), ("b", 0.8)]));
        analyzer.add_model_stream("m2".to_string(), make_stream(&[("x", 0.9), ("y", 0.8)]));
        assert!((analyzer.agreement_score() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_agreement_score_single_model() {
        let mut analyzer = CrossModelAnalyzer::new();
        analyzer.add_model_stream("m1".to_string(), make_stream(&[("a", 0.9)]));
        assert!((analyzer.agreement_score() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_divergence_identical_streams() {
        let mut analyzer = CrossModelAnalyzer::new();
        let tokens = make_stream(&[("a", 0.9), ("b", 0.8), ("c", 0.7)]);
        analyzer.add_model_stream("m1".to_string(), tokens.clone());
        analyzer.add_model_stream("m2".to_string(), tokens);
        let js = analyzer.compute_divergence();
        assert!((0.0..=1.0).contains(&js));
        // Identical distributions should yield near-zero divergence.
        assert!(js < 0.01, "Expected low divergence for identical streams, got {}", js);
    }

    #[test]
    fn test_compute_divergence_zero_for_single_model() {
        let mut analyzer = CrossModelAnalyzer::new();
        analyzer.add_model_stream("m1".to_string(), make_stream(&[("a", 0.9)]));
        assert_eq!(analyzer.compute_divergence(), 0.0);
    }

    #[test]
    fn test_correlation_matrix_shape() {
        let mut analyzer = CrossModelAnalyzer::new();
        analyzer.add_model_stream("m1".to_string(), make_stream(&[("a", 0.9), ("b", 0.5)]));
        analyzer.add_model_stream("m2".to_string(), make_stream(&[("a", 0.8), ("b", 0.6)]));
        analyzer.add_model_stream("m3".to_string(), make_stream(&[("a", 0.7), ("b", 0.4)]));
        let mat = analyzer.correlation_matrix();
        assert_eq!(mat.len(), 3);
        for row in &mat {
            assert_eq!(row.len(), 3);
        }
        // Diagonal should be 1.0.
        for i in 0..3 {
            assert!((mat[i][i] - 1.0).abs() < 1e-6, "Diagonal element {} is not 1.0", i);
        }
    }

    #[test]
    fn test_correlation_matrix_empty_for_single_model() {
        let mut analyzer = CrossModelAnalyzer::new();
        analyzer.add_model_stream("m1".to_string(), make_stream(&[("a", 0.9)]));
        assert!(analyzer.correlation_matrix().is_empty());
    }

    #[test]
    fn test_structural_diff_identical_streams() {
        let mut analyzer = CrossModelAnalyzer::new();
        let tokens: Vec<(&str, f32)> = (0..50).map(|i| ("tok", 0.5 + i as f32 * 0.001)).collect();
        analyzer.add_model_stream("m1".to_string(), make_stream(&tokens));
        analyzer.add_model_stream("m2".to_string(), make_stream(&tokens));
        // Identical streams should have no diff regions.
        let diffs = analyzer.structural_diff();
        assert!(diffs.is_empty(), "Expected no diff regions for identical streams");
    }

    #[test]
    fn test_structural_diff_fully_divergent() {
        let mut analyzer = CrossModelAnalyzer::new();
        let s1: Vec<(&str, f32)> = (0..30).map(|_| ("aaa", 0.9)).collect();
        let s2: Vec<(&str, f32)> = (0..30).map(|_| ("zzz", 0.1)).collect();
        analyzer.add_model_stream("m1".to_string(), make_stream(&s1));
        analyzer.add_model_stream("m2".to_string(), make_stream(&s2));
        let diffs = analyzer.structural_diff();
        assert!(!diffs.is_empty(), "Expected divergent regions for completely different streams");
    }

    #[test]
    fn test_build_report_contains_model_names() {
        let mut analyzer = CrossModelAnalyzer::new();
        analyzer.add_model_stream("gpt-4o".to_string(), make_stream(&[("hello", 0.9)]));
        analyzer.add_model_stream("claude-3".to_string(), make_stream(&[("hi", 0.8)]));
        let report = analyzer.build_report();
        let text = report.render();
        assert!(text.contains("gpt-4o"), "Report should mention gpt-4o");
        assert!(text.contains("claude-3"), "Report should mention claude-3");
        assert!(text.contains("Agreement score"), "Report should include agreement score");
        assert!(text.contains("JS divergence"), "Report should include JS divergence");
    }

    #[test]
    fn test_token_distribution_from_values() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let dist = TokenDistribution::from_values(&values, 5).expect("distribution");
        assert!((dist.mean - 3.0).abs() < 1e-9);
        assert!((dist.min - 1.0).abs() < 1e-9);
        assert!((dist.max - 5.0).abs() < 1e-9);
        assert_eq!(dist.histogram.len(), 5);
    }

    #[test]
    fn test_token_distribution_empty() {
        assert!(TokenDistribution::from_values(&[], 5).is_none());
    }
}
