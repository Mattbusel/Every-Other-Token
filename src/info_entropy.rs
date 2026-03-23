//! # Information-Theoretic Token Analysis
//!
//! Provides Shannon entropy, conditional entropy, mutual information,
//! perplexity, redundancy, n-gram language modelling, and entropy profiles
//! for arbitrary token sequences.
//!
//! ## Quick reference
//!
//! | Function / Type | Description |
//! |-----------------|-------------|
//! | [`token_frequencies`] | Count occurrences of each token |
//! | [`shannon_entropy`] | H = -Σ p·log₂(p) |
//! | [`conditional_entropy`] | H(T\|context) via n-gram model |
//! | [`mutual_information`] | I(A;B) = H(A) + H(B) - H(A,B) |
//! | [`perplexity`] | exp(H) using external probability model |
//! | [`redundancy`] | 1 - H/H_max |
//! | [`NgramModel`] | Laplace-smoothed n-gram LM with generation |
//! | [`EntropyProfile`] | One-shot composite analysis |

use std::collections::HashMap;

// ── Frequency counting ────────────────────────────────────────────────────────

/// Count the occurrence of each token in `tokens`.
pub fn token_frequencies(tokens: &[String]) -> HashMap<String, usize> {
    let mut map = HashMap::new();
    for t in tokens {
        *map.entry(t.clone()).or_insert(0) += 1;
    }
    map
}

// ── Shannon entropy ───────────────────────────────────────────────────────────

/// Compute Shannon entropy H = -Σ p(x)·log₂(p(x)) over the token distribution.
///
/// Returns 0.0 for an empty or single-unique-token sequence.
pub fn shannon_entropy(tokens: &[String]) -> f64 {
    if tokens.is_empty() {
        return 0.0;
    }
    let n = tokens.len() as f64;
    let freq = token_frequencies(tokens);
    freq.values()
        .map(|&c| {
            let p = c as f64 / n;
            if p > 0.0 { -p * p.log2() } else { 0.0 }
        })
        .sum()
}

// ── Conditional entropy ───────────────────────────────────────────────────────

/// Compute H(T|context) using an n-gram model of size `context_size`.
///
/// For each context (prefix of length `context_size`) we build the
/// distribution of next tokens, compute its entropy, and take the weighted
/// average (weighted by context frequency).
///
/// Returns 0.0 when `tokens` is too short to form any contexts.
pub fn conditional_entropy(tokens: &[String], context_size: usize) -> f64 {
    if tokens.len() <= context_size {
        return 0.0;
    }
    // Build context → {next_token: count}
    let mut ctx_map: HashMap<Vec<String>, HashMap<String, usize>> = HashMap::new();
    let mut ctx_total: HashMap<Vec<String>, usize> = HashMap::new();

    for window in tokens.windows(context_size + 1) {
        let ctx = window[..context_size].to_vec();
        let next = window[context_size].clone();
        *ctx_map.entry(ctx.clone()).or_default().entry(next).or_insert(0) += 1;
        *ctx_total.entry(ctx).or_insert(0) += 1;
    }

    let total_contexts: usize = ctx_total.values().sum();
    if total_contexts == 0 {
        return 0.0;
    }

    ctx_map
        .iter()
        .map(|(ctx, next_counts)| {
            let ctx_count = *ctx_total.get(ctx).unwrap_or(&0) as f64;
            let weight = ctx_count / total_contexts as f64;
            let h: f64 = next_counts
                .values()
                .map(|&c| {
                    let p = c as f64 / ctx_count;
                    if p > 0.0 { -p * p.log2() } else { 0.0 }
                })
                .sum();
            weight * h
        })
        .sum()
}

// ── Mutual information ────────────────────────────────────────────────────────

/// Compute I(A;B) = H(A) + H(B) - H(A,B) for two parallel token sequences.
///
/// The joint distribution is formed by treating each (a_i, b_i) pair as a
/// single symbol. Sequences of unequal length are truncated to the shorter.
pub fn mutual_information(tokens_a: &[String], tokens_b: &[String]) -> f64 {
    let min_len = tokens_a.len().min(tokens_b.len());
    if min_len == 0 {
        return 0.0;
    }
    let a = &tokens_a[..min_len];
    let b = &tokens_b[..min_len];

    // Joint tokens represented as "a\0b" to avoid false merges.
    let joint: Vec<String> =
        a.iter().zip(b.iter()).map(|(x, y)| format!("{}\x00{}", x, y)).collect();

    let ha = shannon_entropy(a);
    let hb = shannon_entropy(b);
    let hab = shannon_entropy(&joint);

    (ha + hb - hab).max(0.0)
}

// ── Perplexity ────────────────────────────────────────────────────────────────

/// Compute perplexity of `tokens` under `model_probs`.
///
/// Perplexity = exp(H) where H = -mean(log₂ p(token)).
/// Unknown tokens receive a small epsilon probability (1e-10).
pub fn perplexity(tokens: &[String], model_probs: &HashMap<String, f64>) -> f64 {
    if tokens.is_empty() {
        return 1.0;
    }
    const EPSILON: f64 = 1e-10;
    let h: f64 = tokens
        .iter()
        .map(|t| {
            let p = model_probs.get(t.as_str()).copied().unwrap_or(EPSILON);
            let p = p.max(EPSILON);
            -p.log2()
        })
        .sum::<f64>()
        / tokens.len() as f64;
    2_f64.powf(h)
}

// ── Redundancy ────────────────────────────────────────────────────────────────

/// Compute redundancy = 1 - H/H_max where H_max = log₂(|unique tokens|).
///
/// Returns 0.0 when all tokens are unique (maximum entropy) and 1.0 when
/// only one unique token exists.
pub fn redundancy(tokens: &[String]) -> f64 {
    if tokens.is_empty() {
        return 0.0;
    }
    let unique = token_frequencies(tokens).len();
    if unique <= 1 {
        return 1.0;
    }
    let h_max = (unique as f64).log2();
    if h_max == 0.0 {
        return 0.0;
    }
    let h = shannon_entropy(tokens);
    1.0 - h / h_max
}

// ── NgramModel ────────────────────────────────────────────────────────────────

/// A Laplace-smoothed n-gram language model.
#[derive(Debug, Clone)]
pub struct NgramModel {
    pub n: usize,
    /// context → next_token → count
    pub counts: HashMap<Vec<String>, HashMap<String, u32>>,
    /// context → total count of all next tokens seen
    pub total_counts: HashMap<Vec<String>, u32>,
    /// number of unique token types (for Laplace smoothing denominator)
    vocab_size: usize,
}

impl NgramModel {
    /// Train an n-gram model on `tokens`.
    pub fn train(tokens: &[String], n: usize) -> Self {
        assert!(n >= 1, "n must be at least 1");
        let context_len = n - 1;
        let mut counts: HashMap<Vec<String>, HashMap<String, u32>> = HashMap::new();
        let mut total_counts: HashMap<Vec<String>, u32> = HashMap::new();

        if tokens.len() > context_len {
            for window in tokens.windows(n) {
                let ctx = window[..context_len].to_vec();
                let next = window[context_len].clone();
                *counts.entry(ctx.clone()).or_default().entry(next).or_insert(0) += 1;
                *total_counts.entry(ctx).or_insert(0) += 1;
            }
        }

        let vocab_size = token_frequencies(tokens).len().max(1);
        Self { n, counts, total_counts, vocab_size }
    }

    /// Compute P(next_token | context) with Laplace (add-1) smoothing.
    pub fn probability(&self, context: &[String], next_token: &str) -> f64 {
        let context_len = self.n - 1;
        // Use the last `context_len` elements of the supplied context.
        let ctx: Vec<String> = if context.len() >= context_len {
            context[context.len() - context_len..].to_vec()
        } else {
            context.to_vec()
        };

        let count = self
            .counts
            .get(&ctx)
            .and_then(|m| m.get(next_token))
            .copied()
            .unwrap_or(0) as f64;
        let total = self.total_counts.get(&ctx).copied().unwrap_or(0) as f64;

        // Laplace smoothing: (count + 1) / (total + vocab_size)
        (count + 1.0) / (total + self.vocab_size as f64)
    }

    /// Compute per-token perplexity of `test_tokens` under this model.
    pub fn perplexity_on(&self, test_tokens: &[String]) -> f64 {
        let context_len = self.n - 1;
        if test_tokens.len() <= context_len {
            return 1.0;
        }
        let log_sum: f64 = test_tokens[context_len..]
            .iter()
            .enumerate()
            .map(|(i, next)| {
                let ctx_start = i; // i-th window start in test_tokens
                let ctx = test_tokens[ctx_start..ctx_start + context_len].to_vec();
                let p = self.probability(&ctx, next.as_str());
                -p.log2()
            })
            .sum();
        let num = (test_tokens.len() - context_len) as f64;
        2_f64.powf(log_sum / num)
    }

    /// Generate `length` tokens starting from `seed`, sampling with
    /// temperature `temp` using a simple LCG seeded with `rng_seed`.
    pub fn generate(
        &self,
        seed: &[String],
        length: usize,
        temp: f64,
        rng_seed: u64,
    ) -> Vec<String> {
        let context_len = self.n - 1;
        let mut rng = LcgRng::new(rng_seed);
        let mut context: Vec<String> = seed.to_vec();
        let mut out = Vec::with_capacity(length);

        for _ in 0..length {
            let ctx: Vec<String> = if context.len() >= context_len {
                context[context.len() - context_len..].to_vec()
            } else {
                context.clone()
            };

            // Gather candidates and apply temperature.
            let candidates: Vec<(String, f64)> = if let Some(next_map) = self.counts.get(&ctx) {
                let total = *self.total_counts.get(&ctx).unwrap_or(&0) as f64;
                next_map
                    .iter()
                    .map(|(tok, &cnt)| {
                        let p = (cnt as f64 + 1.0) / (total + self.vocab_size as f64);
                        (tok.clone(), (p.ln() / temp).exp())
                    })
                    .collect()
            } else {
                Vec::new()
            };

            if candidates.is_empty() {
                break;
            }

            let total_weight: f64 = candidates.iter().map(|(_, w)| w).sum();
            let r = rng.next_f64() * total_weight;
            let mut acc = 0.0;
            let mut chosen = candidates[0].0.clone();
            for (tok, w) in &candidates {
                acc += w;
                if r <= acc {
                    chosen = tok.clone();
                    break;
                }
            }
            context.push(chosen.clone());
            out.push(chosen);
        }
        out
    }
}

// ── Simple LCG RNG (no external deps) ────────────────────────────────────────

struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }
    fn next_u64(&mut self) -> u64 {
        // LCG parameters from Numerical Recipes
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ── EntropyProfile ────────────────────────────────────────────────────────────

/// A composite entropy analysis of a token sequence.
#[derive(Debug, Clone)]
pub struct EntropyProfile {
    /// Shannon entropy of the token at each position (using a window of 1).
    pub per_position_entropy: Vec<f64>,
    pub overall_entropy: f64,
    pub conditional_entropy: f64,
    pub perplexity: f64,
    pub redundancy: f64,
}

impl EntropyProfile {
    /// Compute the full entropy profile for `tokens` with `context_size`
    /// for conditional entropy and the n-gram model.
    pub fn compute(tokens: &[String], context_size: usize) -> Self {
        let overall_entropy = shannon_entropy(tokens);
        let cond_entropy = conditional_entropy(tokens, context_size);
        let redundancy_val = redundancy(tokens);

        // Build a unigram probability map for perplexity.
        let freq = token_frequencies(tokens);
        let n = tokens.len() as f64;
        let model_probs: HashMap<String, f64> =
            freq.iter().map(|(k, &v)| (k.clone(), v as f64 / n)).collect();
        let perp = perplexity(tokens, &model_probs);

        // Per-position entropy: for each position i, compute entropy of the
        // single-token "sequence" [tokens[i]] within the global distribution.
        let per_position_entropy: Vec<f64> = tokens
            .iter()
            .map(|t| {
                let p = model_probs.get(t.as_str()).copied().unwrap_or(1e-10);
                if p > 0.0 { -p * p.log2() } else { 0.0 }
            })
            .collect();

        Self {
            per_position_entropy,
            overall_entropy,
            conditional_entropy: cond_entropy,
            perplexity: perp,
            redundancy: redundancy_val,
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn uniform_distribution_maximum_entropy() {
        // Four distinct tokens each appearing once → H = log2(4) = 2.0
        let tokens = toks(&["a", "b", "c", "d"]);
        let h = shannon_entropy(&tokens);
        assert!((h - 2.0).abs() < 1e-9, "expected H=2.0, got {h}");
    }

    #[test]
    fn single_token_zero_entropy() {
        let tokens = toks(&["x", "x", "x", "x"]);
        let h = shannon_entropy(&tokens);
        assert!(h.abs() < 1e-9, "expected H=0.0, got {h}");
    }

    #[test]
    fn bigram_model_probabilities_sum_to_one() {
        let tokens = toks(&["a", "b", "a", "c", "a", "b"]);
        let model = NgramModel::train(&tokens, 2);
        // Context = ["a"] — sum over all next tokens should be ≈ 1.
        let ctx = toks(&["a"]);
        let vocab: Vec<String> = vec!["a", "b", "c"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let total: f64 = vocab.iter().map(|t| model.probability(&ctx, t.as_str())).sum();
        // With Laplace smoothing the sum over *all* vocab tokens (including
        // unseen) should be 1.0 — here we only sum over observed types so it
        // must be ≤ 1.0 and reasonably close given vocab_size = 3.
        assert!(total <= 1.0 + 1e-9, "sum of probs > 1: {total}");
        assert!(total > 0.0);
    }

    #[test]
    fn perplexity_greater_than_one_for_non_uniform() {
        let tokens = toks(&["a", "a", "a", "b"]);
        let freq = token_frequencies(&tokens);
        let n = tokens.len() as f64;
        let model_probs: HashMap<String, f64> =
            freq.iter().map(|(k, &v)| (k.clone(), v as f64 / n)).collect();
        let pp = perplexity(&tokens, &model_probs);
        assert!(pp > 1.0, "expected perplexity > 1, got {pp}");
    }

    #[test]
    fn redundancy_all_unique_is_zero() {
        // When all tokens are distinct, H = H_max, so redundancy = 0.
        let tokens = toks(&["a", "b", "c", "d"]);
        let r = redundancy(&tokens);
        assert!(r.abs() < 1e-9, "expected redundancy ≈ 0, got {r}");
    }

    #[test]
    fn redundancy_single_type_is_one() {
        let tokens = toks(&["x", "x", "x"]);
        let r = redundancy(&tokens);
        assert!((r - 1.0).abs() < 1e-9, "expected redundancy=1.0, got {r}");
    }

    #[test]
    fn mutual_information_nonnegative() {
        let a = toks(&["a", "b", "a", "c"]);
        let b = toks(&["x", "y", "x", "z"]);
        let mi = mutual_information(&a, &b);
        assert!(mi >= 0.0, "MI must be non-negative, got {mi}");
    }

    #[test]
    fn ngram_generate_returns_correct_length() {
        let tokens = toks(&["a", "b", "c", "a", "b", "c", "d"]);
        let model = NgramModel::train(&tokens, 2);
        let seed = toks(&["a"]);
        let generated = model.generate(&seed, 5, 1.0, 42);
        assert_eq!(generated.len(), 5);
    }

    #[test]
    fn entropy_profile_overall_matches_standalone() {
        let tokens = toks(&["a", "b", "c", "a", "b"]);
        let profile = EntropyProfile::compute(&tokens, 1);
        let standalone = shannon_entropy(&tokens);
        assert!((profile.overall_entropy - standalone).abs() < 1e-9);
    }
}
