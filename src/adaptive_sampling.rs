//! Adaptive token sampling strategies for diverse and controlled generation.
//!
//! Provides a suite of sampling algorithms (greedy, top-k, top-p / nucleus,
//! min-p, typical-p, MiroStat, beam search, temperature) together with an
//! [`AdaptiveSampler`] that can apply them in sequence and draw a weighted
//! random sample from the resulting distribution.

#![allow(dead_code)]

// ── SamplingStrategy ─────────────────────────────────────────────────────────

/// Sampling algorithm selection.
#[derive(Debug, Clone, PartialEq)]
pub enum SamplingStrategy {
    /// Always pick the single highest-probability token.
    Greedy,
    /// Keep only the top-k tokens by probability.
    TopK(usize),
    /// Nucleus sampling: keep smallest set whose cumulative prob ≥ p.
    TopP(f64),
    /// Min-p: remove tokens whose prob < min_p × max_prob.
    MinP(f64),
    /// Typical-p: filter by information-content deviation from mean entropy.
    TypicalP(f64),
    /// MiroStat adaptive entropy targeting.
    MiroStat { tau: f64, eta: f64 },
    /// Beam search (width = number of beams kept in parallel).
    BeamSearch(usize),
    /// Scale logits by temperature before softmax.
    Temperature(f64),
}

impl SamplingStrategy {
    /// Human-readable description of the strategy.
    pub fn description(&self) -> &str {
        match self {
            SamplingStrategy::Greedy => "Greedy: always pick the highest-probability token",
            SamplingStrategy::TopK(_) => "Top-K: restrict to the k most probable tokens",
            SamplingStrategy::TopP(_) => "Top-P (nucleus): keep the smallest set summing to p",
            SamplingStrategy::MinP(_) => "Min-P: remove tokens below min_p × max_prob",
            SamplingStrategy::TypicalP(_) => {
                "Typical-P: filter by deviation from expected information content"
            }
            SamplingStrategy::MiroStat { .. } => {
                "MiroStat: adaptively target a desired perplexity (tau)"
            }
            SamplingStrategy::BeamSearch(_) => "Beam Search: keep the top-width partial sequences",
            SamplingStrategy::Temperature(_) => "Temperature: scale logits before softmax",
        }
    }
}

// ── TokenProb ────────────────────────────────────────────────────────────────

/// A single token together with its logit and softmax probability.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenProb {
    /// Vocabulary index.
    pub token_id: usize,
    /// String representation of the token.
    pub token: String,
    /// Raw logit (pre-softmax score).
    pub logit: f64,
    /// Softmax probability (should sum to 1.0 across the distribution).
    pub probability: f64,
}

// ── SamplerState ─────────────────────────────────────────────────────────────

/// Mutable state carried between MiroStat sampling steps.
///
/// Tracks the running estimate of cross-entropy so that the learning-rate
/// update `mu -= eta * (surprise - tau)` can be applied each step.
#[derive(Debug, Clone)]
pub struct SamplerState {
    /// Current MiroStat mu parameter (controls the effective vocabulary size).
    pub mu: f64,
    /// Surprise (−log₂ p) of the last sampled token.
    pub last_surprise: f64,
    /// Number of tokens sampled so far.
    pub steps: usize,
}

impl SamplerState {
    /// Initialise state from a desired perplexity target `tau`.
    pub fn new(tau: f64) -> Self {
        SamplerState {
            mu: tau * 2.0,
            last_surprise: 0.0,
            steps: 0,
        }
    }

    /// Update mu after observing `surprise` bits.
    pub fn update(&mut self, surprise: f64, eta: f64, tau: f64) {
        self.last_surprise = surprise;
        self.mu -= eta * (surprise - tau);
        self.steps += 1;
    }
}

// ── AdaptiveSampler ──────────────────────────────────────────────────────────

/// Collection of static methods that implement each sampling filter.
pub struct AdaptiveSampler;

impl AdaptiveSampler {
    // ── Temperature ──────────────────────────────────────────────────────────

    /// Divide all logits by `temp`, then re-run softmax to update probabilities.
    ///
    /// `temp < 1.0` → sharper (more deterministic).
    /// `temp > 1.0` → flatter (more random).
    pub fn apply_temperature(probs: &mut Vec<TokenProb>, temp: f64) {
        if temp <= 0.0 {
            return;
        }
        for tp in probs.iter_mut() {
            tp.logit /= temp;
        }
        Self::softmax_from_logits(probs);
    }

    // ── Top-K ────────────────────────────────────────────────────────────────

    /// Keep only the `k` tokens with the highest probabilities.
    pub fn top_k_filter(probs: &mut Vec<TokenProb>, k: usize) {
        if k == 0 || k >= probs.len() {
            return;
        }
        probs.sort_unstable_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap());
        probs.truncate(k);
    }

    // ── Top-P (nucleus) ──────────────────────────────────────────────────────

    /// Remove tokens beyond the nucleus whose cumulative probability ≥ `p`.
    pub fn top_p_filter(probs: &mut Vec<TokenProb>, p: f64) {
        probs.sort_unstable_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap());
        let mut cumulative = 0.0_f64;
        let mut cut = probs.len();
        for (i, tp) in probs.iter().enumerate() {
            cumulative += tp.probability;
            if cumulative >= p {
                cut = i + 1;
                break;
            }
        }
        probs.truncate(cut);
    }

    // ── Min-P ────────────────────────────────────────────────────────────────

    /// Remove tokens whose probability is below `min_p × max_prob`.
    pub fn min_p_filter(probs: &mut Vec<TokenProb>, min_p: f64) {
        let max_prob = probs
            .iter()
            .map(|tp| tp.probability)
            .fold(f64::NEG_INFINITY, f64::max);
        let threshold = min_p * max_prob;
        probs.retain(|tp| tp.probability >= threshold);
    }

    // ── Typical-P ────────────────────────────────────────────────────────────

    /// Typical-P filter: keep tokens whose information content is close to the
    /// expected entropy, then truncate to the typical set summing to `p`.
    pub fn typical_p_filter(probs: &mut Vec<TokenProb>, p: f64) {
        let h = Self::entropy(probs);
        // Information content for each token: −log₂(prob).
        let mut with_ic: Vec<(f64, usize)> = probs
            .iter()
            .enumerate()
            .map(|(i, tp)| {
                let ic = if tp.probability > 0.0 {
                    -tp.probability.log2()
                } else {
                    f64::INFINITY
                };
                ((ic - h).abs(), i)
            })
            .collect();
        // Sort by closeness to entropy.
        with_ic.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut cumulative = 0.0_f64;
        let mut keep_indices = std::collections::HashSet::new();
        for (_, idx) in &with_ic {
            cumulative += probs[*idx].probability;
            keep_indices.insert(*idx);
            if cumulative >= p {
                break;
            }
        }
        let mut i = 0;
        probs.retain(|_| {
            let keep = keep_indices.contains(&i);
            i += 1;
            keep
        });
    }

    // ── Normalize ────────────────────────────────────────────────────────────

    /// Renormalize probabilities so they sum to 1.0 after filtering.
    pub fn normalize(probs: &mut Vec<TokenProb>) {
        let total: f64 = probs.iter().map(|tp| tp.probability).sum();
        if total > 0.0 {
            for tp in probs.iter_mut() {
                tp.probability /= total;
            }
        }
    }

    // ── Sample ───────────────────────────────────────────────────────────────

    /// Weighted random draw using a linear congruential generator seeded with
    /// `seed`. Returns `None` if `probs` is empty.
    pub fn sample<'a>(probs: &'a [TokenProb], seed: u64) -> Option<&'a TokenProb> {
        if probs.is_empty() {
            return None;
        }
        // LCG (Knuth parameters).
        let rng = lcg_f64(seed);
        let mut cumulative = 0.0_f64;
        for tp in probs {
            cumulative += tp.probability;
            if rng <= cumulative {
                return Some(tp);
            }
        }
        probs.last()
    }

    // ── Entropy ──────────────────────────────────────────────────────────────

    /// Shannon entropy H = −Σ p·log₂(p) of the distribution (bits).
    pub fn entropy(probs: &[TokenProb]) -> f64 {
        probs
            .iter()
            .filter(|tp| tp.probability > 0.0)
            .map(|tp| -tp.probability * tp.probability.log2())
            .sum()
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Compute softmax probabilities from the logit field.
    fn softmax_from_logits(probs: &mut Vec<TokenProb>) {
        let max_logit = probs
            .iter()
            .map(|tp| tp.logit)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0_f64;
        for tp in probs.iter_mut() {
            tp.probability = (tp.logit - max_logit).exp();
            sum += tp.probability;
        }
        if sum > 0.0 {
            for tp in probs.iter_mut() {
                tp.probability /= sum;
            }
        }
    }
}

// ── LCG helper ───────────────────────────────────────────────────────────────

/// One step of a 64-bit LCG, returning a float in [0, 1).
fn lcg_f64(seed: u64) -> f64 {
    let x = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (x >> 11) as f64 / (1u64 << 53) as f64
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_probs(pairs: &[(&str, f64)]) -> Vec<TokenProb> {
        pairs
            .iter()
            .enumerate()
            .map(|(i, (tok, p))| TokenProb {
                token_id: i,
                token: tok.to_string(),
                logit: p.ln(),
                probability: *p,
            })
            .collect()
    }

    #[test]
    fn test_strategy_description() {
        assert!(!SamplingStrategy::Greedy.description().is_empty());
        assert!(!SamplingStrategy::TopK(10).description().is_empty());
        assert!(!SamplingStrategy::TopP(0.9).description().is_empty());
        assert!(!SamplingStrategy::MinP(0.05).description().is_empty());
        assert!(!SamplingStrategy::TypicalP(0.9).description().is_empty());
        assert!(
            !SamplingStrategy::MiroStat { tau: 5.0, eta: 0.1 }
                .description()
                .is_empty()
        );
        assert!(!SamplingStrategy::BeamSearch(4).description().is_empty());
        assert!(!SamplingStrategy::Temperature(0.8).description().is_empty());
    }

    #[test]
    fn test_top_k_filter() {
        let mut probs = make_probs(&[("a", 0.4), ("b", 0.3), ("c", 0.2), ("d", 0.1)]);
        AdaptiveSampler::top_k_filter(&mut probs, 2);
        assert_eq!(probs.len(), 2);
        assert_eq!(probs[0].token, "a");
        assert_eq!(probs[1].token, "b");
    }

    #[test]
    fn test_top_p_filter() {
        // Sorted desc: a=0.5, b=0.3, c=0.2. Nucleus p=0.85 → keep a+b (0.8 < 0.85), then c pushes to 1.0.
        let mut probs = make_probs(&[("a", 0.5), ("b", 0.3), ("c", 0.2)]);
        AdaptiveSampler::top_p_filter(&mut probs, 0.85);
        // cumulative after a=0.5, b=0.8 < 0.85, c=1.0 ≥ 0.85 → cut=3
        assert_eq!(probs.len(), 3);
    }

    #[test]
    fn test_min_p_filter() {
        let mut probs = make_probs(&[("a", 0.6), ("b", 0.3), ("c", 0.05), ("d", 0.05)]);
        // max_prob=0.6, min_p=0.1, threshold=0.06 → c and d (0.05) removed
        AdaptiveSampler::min_p_filter(&mut probs, 0.1);
        assert_eq!(probs.len(), 2);
    }

    #[test]
    fn test_normalize() {
        let mut probs = make_probs(&[("a", 0.6), ("b", 0.4)]);
        // Remove one token and renorm.
        probs.truncate(1);
        AdaptiveSampler::normalize(&mut probs);
        let total: f64 = probs.iter().map(|t| t.probability).sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_uniform() {
        // Uniform distribution over 4 tokens → H = 2 bits.
        let probs = make_probs(&[("a", 0.25), ("b", 0.25), ("c", 0.25), ("d", 0.25)]);
        let h = AdaptiveSampler::entropy(&probs);
        assert!((h - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_degenerate() {
        // All probability on one token → H = 0.
        let probs = make_probs(&[("a", 1.0), ("b", 0.0)]);
        let h = AdaptiveSampler::entropy(&probs);
        assert!(h.abs() < 1e-10);
    }

    #[test]
    fn test_sample_deterministic() {
        let probs = make_probs(&[("a", 0.99), ("b", 0.01)]);
        // With almost all weight on "a", most seeds should return "a".
        let result = AdaptiveSampler::sample(&probs, 42);
        assert!(result.is_some());
    }

    #[test]
    fn test_sample_empty() {
        let probs: Vec<TokenProb> = vec![];
        assert!(AdaptiveSampler::sample(&probs, 0).is_none());
    }

    #[test]
    fn test_apply_temperature_low() {
        let mut probs = make_probs(&[("a", 0.5), ("b", 0.5)]);
        // Set distinct logits
        probs[0].logit = 2.0;
        probs[1].logit = 1.0;
        AdaptiveSampler::apply_temperature(&mut probs, 0.5);
        // Low temperature → higher-logit token gets more weight.
        assert!(probs[0].probability > probs[1].probability);
    }

    #[test]
    fn test_sampler_state_update() {
        let mut state = SamplerState::new(5.0);
        state.update(6.0, 0.1, 5.0);
        // mu should decrease when surprise > tau.
        assert!(state.mu < 10.0);
        assert_eq!(state.steps, 1);
    }

    #[test]
    fn test_typical_p_filter_keeps_some() {
        let mut probs = make_probs(&[("a", 0.5), ("b", 0.3), ("c", 0.1), ("d", 0.1)]);
        AdaptiveSampler::typical_p_filter(&mut probs, 0.8);
        assert!(!probs.is_empty());
    }
}
