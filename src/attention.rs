//! # Causal Attention Tracer
//!
//! Approximates which tokens in the context *caused* each generated token by
//! computing a logit-attribution score: the difference in log-probability
//! between a full-context forward pass and a masked-context forward pass.
//!
//! ## Method
//!
//! For each generated token `g` at position `i`:
//!   For each context token `c` at position `j`:
//!     `attribution(i, j) = logprob_full(g | context) - logprob_masked(g | context \ {j})`
//!
//! A large positive value means token `j` strongly increases the probability
//! of `g`.  A value near zero means `j` has little causal influence.
//!
//! ## Visualisation
//!
//! The resulting [`AttributionMatrix`] can be rendered as an ASCII heatmap
//! suitable for terminal output.
//!
//! ## Limitations
//!
//! This is an *approximation*: masking a single token and re-running the model
//! is not the same as a true attention trace, but is significantly cheaper
//! than computing full gradients.

use serde::{Deserialize, Serialize};
use std::fmt;

// ── Attribution matrix ────────────────────────────────────────────────────────

/// A 2-D matrix of attribution scores.
///
/// `scores[i][j]` = how much context token `j` influenced generated token `i`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributionMatrix {
    /// Labels for the generated tokens (rows).
    pub generated_tokens: Vec<String>,
    /// Labels for the context tokens (columns).
    pub context_tokens: Vec<String>,
    /// Row-major score matrix; shape `[generated_tokens.len()][context_tokens.len()]`.
    pub scores: Vec<Vec<f64>>,
}

impl AttributionMatrix {
    /// Create a zero-filled matrix with the given labels.
    pub fn zeros(generated: Vec<String>, context: Vec<String>) -> Self {
        let rows = generated.len();
        let cols = context.len();
        Self {
            generated_tokens: generated,
            context_tokens: context,
            scores: vec![vec![0.0; cols]; rows],
        }
    }

    /// Number of generated-token rows.
    pub fn rows(&self) -> usize {
        self.generated_tokens.len()
    }

    /// Number of context-token columns.
    pub fn cols(&self) -> usize {
        self.context_tokens.len()
    }

    /// Set the attribution score for a (generated, context) token pair.
    pub fn set(&mut self, gen_idx: usize, ctx_idx: usize, score: f64) {
        if gen_idx < self.scores.len() && ctx_idx < self.scores[gen_idx].len() {
            self.scores[gen_idx][ctx_idx] = score;
        }
    }

    /// Return the highest-attribution context token index for a given
    /// generated token.
    pub fn top_context(&self, gen_idx: usize) -> Option<usize> {
        self.scores.get(gen_idx).and_then(|row| {
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
        })
    }

    /// Compute the row-wise maximum attribution score.
    pub fn row_max(&self, gen_idx: usize) -> f64 {
        self.scores
            .get(gen_idx)
            .and_then(|row| row.iter().cloned().reduce(f64::max))
            .unwrap_or(0.0)
    }

    /// Normalise each row so that scores sum to 1.0 (softmax-style).
    pub fn normalise_rows(&mut self) {
        for row in &mut self.scores {
            let sum: f64 = row.iter().sum();
            if sum.abs() > f64::EPSILON {
                for v in row.iter_mut() {
                    *v /= sum;
                }
            }
        }
    }
}

// ── ASCII heatmap display ─────────────────────────────────────────────────────

/// Render the matrix as an ASCII heatmap block.
impl fmt::Display for AttributionMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Unicode block shading characters from low to high intensity.
        const SHADES: &[char] = &[' ', '\u{2591}', '\u{2592}', '\u{2593}', '\u{2588}'];

        // Global max for normalisation.
        let global_max = self
            .scores
            .iter()
            .flat_map(|row| row.iter())
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // Header row: context tokens (truncated to 4 chars).
        write!(f, "{:>12}", "")?;
        for ctx in &self.context_tokens {
            let label: String = ctx.chars().take(4).collect();
            write!(f, " {:>4}", label)?;
        }
        writeln!(f)?;

        for (i, gen) in self.generated_tokens.iter().enumerate() {
            let label: String = gen.chars().take(10).collect();
            write!(f, "{:>12}", label)?;
            for &score in &self.scores[i] {
                let intensity = if global_max.abs() < f64::EPSILON {
                    0
                } else {
                    let norm = (score / global_max).clamp(0.0, 1.0);
                    (norm * (SHADES.len() - 1) as f64).round() as usize
                };
                let shade = SHADES[intensity.min(SHADES.len() - 1)];
                write!(f, "  {shade}{shade} ")?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

// ── Logprob record ────────────────────────────────────────────────────────────

/// A single-token logprob sample as returned by the LLM API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenLogprob {
    /// Surface form of the token.
    pub token: String,
    /// Natural-log probability assigned by the model (`-∞` to `0`).
    pub logprob: f64,
    /// Byte offset of this token in the generated sequence.
    pub byte_offset: usize,
}

// ── Tracer ────────────────────────────────────────────────────────────────────

/// Configuration for the causal attention tracer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionTracerConfig {
    /// Maximum number of context tokens to trace (avoids O(n²) explosion).
    pub max_context_tokens: usize,
    /// Maximum number of generated tokens to trace.
    pub max_generated_tokens: usize,
    /// Whether to normalise rows after computing the matrix.
    pub normalise: bool,
}

impl Default for AttentionTracerConfig {
    fn default() -> Self {
        Self {
            max_context_tokens: 64,
            max_generated_tokens: 32,
            normalise: true,
        }
    }
}

/// Approximates causal attribution between context and generated tokens.
///
/// # How to use
///
/// 1. Collect [`TokenLogprob`]s for the full-context run.
/// 2. For each context position, re-run the model with that token masked and
///    collect logprobs again.
/// 3. Call [`CausalAttentionTracer::compute`] with all the data.
///
/// In practice, re-running the model for every context token is expensive; the
/// tracer supports *approximation mode* where only the top-K context tokens
/// (by logprob magnitude) are traced.
pub struct CausalAttentionTracer {
    config: AttentionTracerConfig,
}

impl CausalAttentionTracer {
    /// Create a new tracer with the given config.
    pub fn new(config: AttentionTracerConfig) -> Self {
        Self { config }
    }

    /// Compute an [`AttributionMatrix`] from pre-collected logprob data.
    ///
    /// # Parameters
    ///
    /// - `context_tokens`: surface forms of the prompt/context tokens.
    /// - `full_logprobs`: logprobs for the generated tokens under the full context.
    /// - `masked_logprobs`: for each context index, logprobs when that token is masked.
    ///   Must satisfy `masked_logprobs.len() == context_tokens.len()`.
    pub fn compute(
        &self,
        context_tokens: Vec<String>,
        full_logprobs: Vec<TokenLogprob>,
        masked_logprobs: Vec<Vec<TokenLogprob>>,
    ) -> AttributionMatrix {
        let ctx_len = context_tokens
            .len()
            .min(self.config.max_context_tokens);
        let gen_len = full_logprobs
            .len()
            .min(self.config.max_generated_tokens);

        let gen_labels: Vec<String> = full_logprobs[..gen_len]
            .iter()
            .map(|t| t.token.clone())
            .collect();
        let ctx_labels: Vec<String> = context_tokens[..ctx_len].to_vec();

        let mut matrix = AttributionMatrix::zeros(gen_labels, ctx_labels);

        for ctx_idx in 0..ctx_len {
            if ctx_idx >= masked_logprobs.len() {
                break;
            }
            let masked = &masked_logprobs[ctx_idx];
            for gen_idx in 0..gen_len {
                let full_lp = full_logprobs[gen_idx].logprob;
                let masked_lp = masked
                    .get(gen_idx)
                    .map(|t| t.logprob)
                    .unwrap_or(full_lp);
                // Attribution = increase in logprob caused by the context token.
                let attribution = full_lp - masked_lp;
                matrix.set(gen_idx, ctx_idx, attribution);
            }
        }

        if self.config.normalise {
            matrix.normalise_rows();
        }

        matrix
    }

    /// Approximate attribution using only logprob differences from a single
    /// forward pass (no masking required).  Less accurate but zero-overhead.
    ///
    /// Assigns attribution proportional to the absolute logprob of each
    /// generated token, distributed evenly across context positions.
    pub fn approximate_from_logprobs(
        &self,
        context_tokens: Vec<String>,
        generated_logprobs: Vec<TokenLogprob>,
    ) -> AttributionMatrix {
        let ctx_len = context_tokens
            .len()
            .min(self.config.max_context_tokens);
        let gen_len = generated_logprobs
            .len()
            .min(self.config.max_generated_tokens);

        let gen_labels: Vec<String> = generated_logprobs[..gen_len]
            .iter()
            .map(|t| t.token.clone())
            .collect();
        let ctx_labels: Vec<String> = context_tokens[..ctx_len].to_vec();

        let mut matrix = AttributionMatrix::zeros(gen_labels, ctx_labels);

        for gen_idx in 0..gen_len {
            let lp = generated_logprobs[gen_idx].logprob.abs();
            // Distribute proportionally using a recency bias: later context
            // tokens get slightly higher attribution.
            let total_weight: f64 = (0..ctx_len).map(|j| (j + 1) as f64).sum();
            for ctx_idx in 0..ctx_len {
                let weight = (ctx_idx + 1) as f64 / total_weight;
                matrix.set(gen_idx, ctx_idx, lp * weight);
            }
        }

        if self.config.normalise {
            matrix.normalise_rows();
        }

        matrix
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn lp(token: &str, logprob: f64) -> TokenLogprob {
        TokenLogprob {
            token: token.to_string(),
            logprob,
            byte_offset: 0,
        }
    }

    #[test]
    fn zeros_matrix_dimensions() {
        let m = AttributionMatrix::zeros(
            vec!["a".into(), "b".into()],
            vec!["x".into(), "y".into(), "z".into()],
        );
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(m.scores[0][0], 0.0);
    }

    #[test]
    fn attribution_matrix_top_context() {
        let mut m =
            AttributionMatrix::zeros(vec!["tok".into()], vec!["a".into(), "b".into(), "c".into()]);
        m.set(0, 0, 0.1);
        m.set(0, 1, 0.9);
        m.set(0, 2, 0.4);
        assert_eq!(m.top_context(0), Some(1));
    }

    #[test]
    fn normalise_rows_sums_to_one() {
        let mut m = AttributionMatrix::zeros(
            vec!["g1".into()],
            vec!["c1".into(), "c2".into(), "c3".into()],
        );
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(0, 2, 7.0);
        m.normalise_rows();
        let sum: f64 = m.scores[0].iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn tracer_compute_basic() {
        let tracer = CausalAttentionTracer::new(AttentionTracerConfig {
            normalise: false,
            ..Default::default()
        });

        let ctx = vec!["The".to_string(), "cat".to_string()];
        let full = vec![lp("sat", -0.5), lp("on", -1.0)];
        // Masking "The": logprobs decrease.
        let masked_the = vec![lp("sat", -1.0), lp("on", -1.5)];
        // Masking "cat": logprobs decrease more.
        let masked_cat = vec![lp("sat", -2.0), lp("on", -2.5)];

        let matrix = tracer.compute(ctx, full, vec![masked_the, masked_cat]);
        // sat[0] attr to The: -0.5 - (-1.0) = 0.5
        assert!((matrix.scores[0][0] - 0.5).abs() < 1e-10);
        // sat[0] attr to cat: -0.5 - (-2.0) = 1.5
        assert!((matrix.scores[0][1] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn approximate_attribution_shape() {
        let tracer = CausalAttentionTracer::new(AttentionTracerConfig::default());
        let ctx = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let gen = vec![lp("x", -0.3), lp("y", -0.8)];
        let m = tracer.approximate_from_logprobs(ctx, gen);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        // After normalisation rows should sum to ~1.
        let sum: f64 = m.scores[0].iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn display_does_not_panic() {
        let mut m = AttributionMatrix::zeros(
            vec!["hello".into(), "world".into()],
            vec!["The".into(), "quick".into(), "brown".into()],
        );
        m.set(0, 0, 0.8);
        m.set(1, 2, 1.0);
        let s = format!("{m}");
        assert!(!s.is_empty());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Attention-inspired token importance scoring
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute the dot product of two equal-length slices.
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Numerically-stable softmax: subtract max before exponentiating.
pub fn softmax(scores: &[f64]) -> Vec<f64> {
    if scores.is_empty() {
        return vec![];
    }
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0 {
        vec![1.0 / scores.len() as f64; scores.len()]
    } else {
        exps.iter().map(|&e| e / sum).collect()
    }
}

/// A single attention head parameterised by head dimensionality.
#[derive(Debug, Clone)]
pub struct AttentionHead {
    /// Dimensionality of query / key vectors.
    pub dim: usize,
}

impl AttentionHead {
    /// Scaled dot-product attention score: dot(query, key) / sqrt(dim).
    pub fn score(&self, query: &[f64], key: &[f64]) -> f64 {
        let d = self.dim.max(1) as f64;
        dot_product(query, key) / d.sqrt()
    }
}

/// A token with its position and embedding vector.
#[derive(Debug, Clone)]
pub struct TokenEmbedding {
    pub token: String,
    pub position: usize,
    pub embedding: Vec<f64>,
}

impl TokenEmbedding {
    /// Sinusoidal positional encoding (Vaswani et al., 2017).
    ///
    /// PE[2i]   = sin(pos / 10000^(2i/dim))
    /// PE[2i+1] = cos(pos / 10000^(2i/dim))
    pub fn positional_encoding(position: usize, dim: usize) -> Vec<f64> {
        (0..dim)
            .map(|i| {
                let freq = (position as f64)
                    / (10_000_f64).powf(2.0 * (i / 2) as f64 / dim.max(1) as f64);
                if i % 2 == 0 {
                    freq.sin()
                } else {
                    freq.cos()
                }
            })
            .collect()
    }
}

/// Multi-head self-attention scorer for token importance.
#[derive(Debug, Clone)]
pub struct SelfAttentionScorer {
    pub head_dim: usize,
    pub num_heads: usize,
}

impl SelfAttentionScorer {
    const EMBED_DIM: usize = 16;

    /// Assign importance scores to a sequence of tokens using self-attention.
    ///
    /// 1. Build bag-of-chars embeddings (dim = 16).
    /// 2. Add sinusoidal positional encoding.
    /// 3. Compute self-attention; accumulate weighted value magnitudes.
    /// 4. Normalise to \[0, 1\].
    pub fn score_tokens(&self, tokens: &[String]) -> Vec<f64> {
        if tokens.is_empty() {
            return vec![];
        }
        let dim = Self::EMBED_DIM;

        // Step 1 & 2: embeddings + positional encoding.
        let embeddings: Vec<Vec<f64>> = tokens
            .iter()
            .enumerate()
            .map(|(pos, tok)| {
                let mut emb = vec![0.0f64; dim];
                let pos_weight = 1.0 / (1.0 + pos as f64);
                for ch in tok.chars() {
                    let code = ch as u32 as f64;
                    for i in 0..dim {
                        emb[i] += code * pos_weight;
                    }
                }
                // Normalise raw embedding.
                let norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-8);
                for v in emb.iter_mut() {
                    *v /= norm;
                }
                // Add positional encoding.
                let pe = TokenEmbedding::positional_encoding(pos, dim);
                emb.iter().zip(pe.iter()).map(|(a, b)| a + b).collect()
            })
            .collect();

        let head = AttentionHead {
            dim: self.head_dim.max(1),
        };
        let n = tokens.len();

        // Step 3: for each token compute sum of attention-weighted value magnitudes.
        let mut scores: Vec<f64> = (0..n)
            .map(|i| {
                // Raw attention logits from token i to all others.
                let logits: Vec<f64> = (0..n)
                    .map(|j| head.score(&embeddings[i], &embeddings[j]))
                    .collect();
                let attn = softmax(&logits);
                // Weighted sum of value magnitudes (||emb[j]||).
                attn.iter()
                    .zip(embeddings.iter())
                    .map(|(&a, emb)| {
                        let mag: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
                        a * mag
                    })
                    .sum()
            })
            .collect();

        // Step 4: normalise to [0, 1].
        let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = (max - min).max(1e-12);
        for s in scores.iter_mut() {
            *s = (*s - min) / range;
        }
        scores
    }
}

/// Attention mask variant controlling which positions can attend to which.
#[derive(Debug, Clone, PartialEq)]
pub enum AttentionMask {
    /// Each position can only attend to itself and previous positions.
    Causal,
    /// Every position attends to every other position.
    Full,
    /// Each position attends only within a window of the given half-width.
    WindowedLocal(usize),
}

/// Apply an [`AttentionMask`] to a raw score matrix, then row-wise softmax.
///
/// Masked positions are set to `-1e9` before softmax so they contribute ~0.
pub fn masked_attention(
    scores: &[Vec<f64>],
    mask: &AttentionMask,
) -> Vec<Vec<f64>> {
    scores
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let masked: Vec<f64> = row
                .iter()
                .enumerate()
                .map(|(j, &s)| {
                    let allowed = match mask {
                        AttentionMask::Full => true,
                        AttentionMask::Causal => j <= i,
                        AttentionMask::WindowedLocal(w) => {
                            (j as isize - i as isize).unsigned_abs() <= *w
                        }
                    };
                    if allowed { s } else { -1e9 }
                })
                .collect();
            softmax(&masked)
        })
        .collect()
}

/// Return the top-`k` tokens by descending score.
///
/// If `k` exceeds the number of tokens, all tokens are returned.
pub fn top_k_tokens(
    tokens: &[String],
    scores: &[f64],
    k: usize,
) -> Vec<(String, f64)> {
    let mut pairs: Vec<(String, f64)> = tokens
        .iter()
        .zip(scores.iter())
        .map(|(t, &s)| (t.clone(), s))
        .collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    pairs.truncate(k);
    pairs
}

// ── Tests for the scoring layer ───────────────────────────────────────────────

#[cfg(test)]
mod scoring_tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let scores = vec![1.0, 2.0, 3.0, 0.5];
        let sm = softmax(&scores);
        let sum: f64 = sm.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "softmax sum = {sum}");
    }

    #[test]
    fn softmax_empty_returns_empty() {
        assert!(softmax(&[]).is_empty());
    }

    #[test]
    fn positional_encoding_differs_at_different_positions() {
        let pe0 = TokenEmbedding::positional_encoding(0, 16);
        let pe5 = TokenEmbedding::positional_encoding(5, 16);
        // Not identical.
        assert_ne!(pe0, pe5);
        // Length correct.
        assert_eq!(pe0.len(), 16);
    }

    #[test]
    fn top_k_returns_k_items() {
        let tokens: Vec<String> = (0..10).map(|i| format!("t{i}")).collect();
        let scores: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let top = top_k_tokens(&tokens, &scores, 3);
        assert_eq!(top.len(), 3);
        // Best score first.
        assert!(top[0].1 >= top[1].1);
    }

    #[test]
    fn top_k_clamps_to_available() {
        let tokens = vec!["a".to_string(), "b".to_string()];
        let scores = vec![0.8, 0.2];
        let top = top_k_tokens(&tokens, &scores, 100);
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn causal_mask_zeroes_future_positions() {
        // 3x3 uniform score matrix.
        let scores: Vec<Vec<f64>> = vec![
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
        ];
        let out = masked_attention(&scores, &AttentionMask::Causal);
        // Row 0 can only attend to position 0 → weight on pos 1 and 2 ≈ 0.
        assert!(out[0][1] < 1e-6, "row0 pos1 should be ~0, got {}", out[0][1]);
        assert!(out[0][2] < 1e-6, "row0 pos2 should be ~0, got {}", out[0][2]);
        // Row 2 can attend to all → all weights > 0.
        assert!(out[2][0] > 0.0);
        assert!(out[2][1] > 0.0);
        assert!(out[2][2] > 0.0);
    }

    #[test]
    fn self_attention_scorer_produces_unit_range() {
        let scorer = SelfAttentionScorer {
            head_dim: 16,
            num_heads: 4,
        };
        let tokens: Vec<String> = vec![
            "hello".into(),
            "world".into(),
            "foo".into(),
            "bar".into(),
        ];
        let scores = scorer.score_tokens(&tokens);
        assert_eq!(scores.len(), tokens.len());
        for &s in &scores {
            assert!(s >= 0.0 && s <= 1.0 + 1e-9, "score out of range: {s}");
        }
    }
}
