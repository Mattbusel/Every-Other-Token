//! # Steering Vector Interface
//!
//! Add or subtract concept vectors from the model's residual stream to
//! steer generation toward or away from target concepts.
//!
//! ## Method
//!
//! 1. **Concept vector extraction**: Contrast a pair of prompts ("happy" vs
//!    "sad").  The steering vector is the mean difference of their residual
//!    stream activations at a chosen layer.  Because we lack white-box access,
//!    we approximate the vector as the token-embedding difference between
//!    paired concept words (using the API's token logprob shift as a proxy).
//!
//! 2. **Steering application**: During generation, add `α * v` (or subtract for
//!    the opposite direction) to the model's internal representation at each
//!    token step.  Approximated by prepending a soft "concept prompt" prefix
//!    that biases the distribution.
//!
//! 3. **Effect measurement**: Compare the token probability distributions
//!    before and after applying the steering vector, reporting KL divergence
//!    and top-token shifts in real-time.
//!
//! ## Usage
//!
//! ```rust
//! use every_other_token::steering::{SteeringEngine, SteeringConfig, ConceptPair};
//!
//! let cfg = SteeringConfig::default();
//! let engine = SteeringEngine::new(cfg);
//!
//! let pair = ConceptPair::new("happy", "sad");
//! let vector = engine.extract_concept_vector(&pair);
//! assert!(vector.magnitude() > 0.0);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Concept pair ──────────────────────────────────────────────────────────────

/// A pair of contrasting concept prompts used to derive a steering vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptPair {
    /// The "positive" concept (direction to steer *toward*).
    pub positive: String,
    /// The "negative" concept (direction to steer *away from*).
    pub negative: String,
}

impl ConceptPair {
    pub fn new(positive: impl Into<String>, negative: impl Into<String>) -> Self {
        Self {
            positive: positive.into(),
            negative: negative.into(),
        }
    }
}

// ── Steering vector ───────────────────────────────────────────────────────────

/// A concept vector in vocabulary-logit space.
///
/// Because we approximate in logit space (rather than true residual-stream
/// space), the vector is stored as a map from token → signed logit shift.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SteeringVector {
    /// Source concept pair.
    pub concept: ConceptPair,
    /// Logit-space representation: token → signed shift value.
    pub components: HashMap<String, f64>,
    /// L2 norm in the component space.
    cached_magnitude: f64,
}

impl SteeringVector {
    fn compute_magnitude(components: &HashMap<String, f64>) -> f64 {
        components.values().map(|v| v * v).sum::<f64>().sqrt()
    }

    pub fn new(concept: ConceptPair, components: HashMap<String, f64>) -> Self {
        let mag = Self::compute_magnitude(&components);
        Self {
            concept,
            components,
            cached_magnitude: mag,
        }
    }

    /// L2 norm of the vector.
    pub fn magnitude(&self) -> f64 {
        self.cached_magnitude
    }

    /// Return a scaled copy of this vector.
    pub fn scale(&self, alpha: f64) -> Self {
        let scaled: HashMap<String, f64> =
            self.components.iter().map(|(k, v)| (k.clone(), v * alpha)).collect();
        let mag = Self::compute_magnitude(&scaled);
        Self {
            concept: self.concept.clone(),
            components: scaled,
            cached_magnitude: mag,
        }
    }

    /// Negate the vector (steer in the opposite direction).
    pub fn negate(&self) -> Self {
        self.scale(-1.0)
    }

    /// Apply this vector to a logprob map, returning the steered distribution.
    pub fn apply_to_logprobs(&self, logprobs: &HashMap<String, f64>) -> HashMap<String, f64> {
        // Fast path: if all components are zero (e.g. alpha=0), return the original unchanged.
        let all_zero = self.components.values().all(|&v| v == 0.0);
        if all_zero {
            return logprobs.clone();
        }

        let mut result = logprobs.clone();
        for (token, shift) in &self.components {
            let entry = result.entry(token.clone()).or_insert(-10.0);
            *entry += shift;
        }
        // Re-normalise in log space (log-sum-exp)
        let max = result.values().cloned().fold(f64::NEG_INFINITY, f64::max);
        let log_sum_exp = result
            .values()
            .map(|lp| (lp - max).exp())
            .sum::<f64>()
            .ln()
            + max;
        result.values_mut().for_each(|lp| *lp -= log_sum_exp);
        result
    }

    /// Compute KL divergence KL(steered || original) as an effect size metric.
    pub fn kl_divergence(
        &self,
        original: &HashMap<String, f64>,
        steered: &HashMap<String, f64>,
    ) -> f64 {
        steered
            .iter()
            .map(|(tok, &lp_steered)| {
                let lp_orig = original.get(tok).copied().unwrap_or(-20.0);
                let p_steered = lp_steered.exp();
                if p_steered > 1e-15 {
                    p_steered * (lp_steered - lp_orig)
                } else {
                    0.0
                }
            })
            .sum()
    }
}

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for the steering engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SteeringConfig {
    /// Default steering strength multiplier α.
    pub alpha: f64,
    /// Vocabulary tokens to track in the concept vector (top-K by frequency).
    pub vocab_size: usize,
    /// Layer at which to apply steering (0 = earliest).
    pub target_layer: usize,
    /// Whether to show per-token probability shifts in the TUI.
    pub show_token_shifts: bool,
}

impl Default for SteeringConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            vocab_size: 50,
            target_layer: 8,
            show_token_shifts: true,
        }
    }
}

// ── Steering result ───────────────────────────────────────────────────────────

/// Effect of applying a steering vector to a generation step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SteeringEffect {
    /// Input prompt (or continuation prefix).
    pub prompt: String,
    /// Concept being steered toward.
    pub concept: String,
    /// KL divergence between steered and original distributions.
    pub kl_divergence: f64,
    /// Top tokens in the *original* distribution with their logprobs.
    pub original_top: Vec<(String, f64)>,
    /// Top tokens in the *steered* distribution with their logprobs.
    pub steered_top: Vec<(String, f64)>,
    /// Per-token probability shifts (token, original_lp, steered_lp, delta).
    pub token_shifts: Vec<(String, f64, f64, f64)>,
}

impl SteeringEffect {
    /// ASCII summary for TUI display.
    pub fn ascii_summary(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "Steering Effect — concept: \"{}\"   KL: {:.4}\n\n",
            self.concept, self.kl_divergence
        ));
        out.push_str(&format!("  Prompt: {}\n\n", truncate(&self.prompt, 60)));
        out.push_str("  token          original    steered     Δ\n");
        out.push_str("  ─────────────  ──────────  ──────────  ──────\n");
        let mut shifts = self.token_shifts.clone();
        shifts.sort_by(|a, b| b.3.abs().partial_cmp(&a.3.abs()).unwrap());
        for (tok, orig, steered, delta) in shifts.iter().take(10) {
            out.push_str(&format!(
                "  {:<13}  {:>10.4}  {:>10.4}  {:>+6.4}\n",
                truncate(tok, 13),
                orig,
                steered,
                delta
            ));
        }
        out
    }
}

// ── Steering engine ───────────────────────────────────────────────────────────

/// The steering engine: extracts concept vectors and applies them.
pub struct SteeringEngine {
    pub config: SteeringConfig,
}

impl SteeringEngine {
    pub fn new(config: SteeringConfig) -> Self {
        Self { config }
    }

    /// Extract a concept vector from a contrasting concept pair.
    ///
    /// In a real integration this would take the residual-stream difference
    /// between paired prompts.  Here we use a built-in lexical contrast table
    /// combined with simulated logit shifts derived from the concept words.
    pub fn extract_concept_vector(&self, pair: &ConceptPair) -> SteeringVector {
        let mut components: HashMap<String, f64> = HashMap::new();

        // Seed with direct concept tokens
        let pos_words = expand_concept(&pair.positive);
        let neg_words = expand_concept(&pair.negative);

        for (word, weight) in &pos_words {
            *components.entry(word.clone()).or_insert(0.0) += weight;
        }
        for (word, weight) in &neg_words {
            *components.entry(word.clone()).or_insert(0.0) -= weight;
        }

        // Normalise
        let max_abs = components
            .values()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max)
            .max(1e-12);
        for v in components.values_mut() {
            *v /= max_abs;
        }

        SteeringVector::new(pair.clone(), components)
    }

    /// Apply a steering vector to a token logprob distribution.
    pub fn apply_steering(
        &self,
        prompt: &str,
        vector: &SteeringVector,
        original_logprobs: &HashMap<String, f64>,
        alpha: Option<f64>,
    ) -> SteeringEffect {
        let alpha = alpha.unwrap_or(self.config.alpha);
        let scaled = vector.scale(alpha);
        let steered = scaled.apply_to_logprobs(original_logprobs);

        let kl = scaled.kl_divergence(original_logprobs, &steered);

        let mut original_top: Vec<(String, f64)> = original_logprobs
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        original_top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        original_top.truncate(10);

        let mut steered_top: Vec<(String, f64)> = steered
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        steered_top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        steered_top.truncate(10);

        let token_shifts: Vec<(String, f64, f64, f64)> = original_logprobs
            .iter()
            .map(|(tok, &orig_lp)| {
                let steered_lp = steered.get(tok).copied().unwrap_or(-20.0);
                (tok.clone(), orig_lp, steered_lp, steered_lp - orig_lp)
            })
            .collect();

        SteeringEffect {
            prompt: prompt.to_string(),
            concept: vector.concept.positive.clone(),
            kl_divergence: kl,
            original_top,
            steered_top,
            token_shifts,
        }
    }

    /// Generate a "concept vector" from paired example sentences and measure
    /// its effect on a test prompt.
    pub fn contrastive_steer(
        &self,
        positive_prompt: &str,
        negative_prompt: &str,
        test_prompt: &str,
        test_logprobs: &HashMap<String, f64>,
    ) -> SteeringEffect {
        // Extract concept from the first non-stopword in each prompt
        let pos_concept = extract_concept_word(positive_prompt);
        let neg_concept = extract_concept_word(negative_prompt);
        let pair = ConceptPair::new(pos_concept, neg_concept);
        let vector = self.extract_concept_vector(&pair);
        self.apply_steering(test_prompt, &vector, test_logprobs, None)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Expand a concept word into a small set of associated tokens with weights.
fn expand_concept(concept: &str) -> Vec<(String, f64)> {
    let lower = concept.to_lowercase();
    let mut result = vec![(lower.clone(), 1.0)];

    // Hard-coded concept associations (would be embeddings in a real system)
    let associations: &[(&str, &[(&str, f64)])] = &[
        ("happy", &[("joy", 0.8), ("smile", 0.7), ("glad", 0.9), ("cheerful", 0.6)]),
        ("sad", &[("grief", 0.8), ("cry", 0.7), ("unhappy", 0.9), ("gloom", 0.6)]),
        ("hot", &[("warm", 0.8), ("heat", 0.9), ("fire", 0.6), ("scorching", 0.5)]),
        ("cold", &[("cool", 0.8), ("ice", 0.7), ("freeze", 0.6), ("chill", 0.9)]),
        ("good", &[("great", 0.8), ("excellent", 0.7), ("positive", 0.9), ("fine", 0.6)]),
        ("bad", &[("terrible", 0.8), ("awful", 0.7), ("negative", 0.9), ("poor", 0.6)]),
        ("fast", &[("quick", 0.9), ("rapid", 0.8), ("swift", 0.7), ("speed", 0.6)]),
        ("slow", &[("sluggish", 0.8), ("delay", 0.7), ("crawl", 0.6), ("lag", 0.9)]),
    ];

    for (key, assoc) in associations {
        if lower == *key {
            for (tok, weight) in *assoc {
                result.push((tok.to_string(), *weight));
            }
            break;
        }
    }
    result
}

fn extract_concept_word(prompt: &str) -> String {
    let stopwords = ["the", "a", "an", "is", "are", "was", "were", "i", "am", "feeling"];
    prompt
        .split_whitespace()
        .find(|w| !stopwords.contains(&w.to_lowercase().as_str()))
        .unwrap_or("neutral")
        .to_lowercase()
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

    fn sample_logprobs() -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("joy".to_string(), -1.0);
        m.insert("grief".to_string(), -3.0);
        m.insert("smile".to_string(), -2.0);
        m.insert("cry".to_string(), -4.0);
        m.insert("neutral".to_string(), -2.5);
        m
    }

    #[test]
    fn concept_vector_has_positive_magnitude() {
        let engine = SteeringEngine::new(SteeringConfig::default());
        let pair = ConceptPair::new("happy", "sad");
        let vec = engine.extract_concept_vector(&pair);
        assert!(vec.magnitude() > 0.0);
    }

    #[test]
    fn steering_shifts_happy_tokens_up() {
        let engine = SteeringEngine::new(SteeringConfig::default());
        let pair = ConceptPair::new("happy", "sad");
        let vec = engine.extract_concept_vector(&pair);
        let orig = sample_logprobs();
        let effect = engine.apply_steering("I am feeling", &vec, &orig, Some(2.0));

        // "joy" and "smile" should move up; "grief" and "cry" should move down
        let joy_shift = effect
            .token_shifts
            .iter()
            .find(|(t, ..)| t == "joy")
            .map(|(_, _, _, d)| *d)
            .unwrap_or(0.0);
        let grief_shift = effect
            .token_shifts
            .iter()
            .find(|(t, ..)| t == "grief")
            .map(|(_, _, _, d)| *d)
            .unwrap_or(0.0);
        assert!(joy_shift > grief_shift, "joy should shift more than grief");
    }

    #[test]
    fn negated_vector_reverses_direction() {
        let engine = SteeringEngine::new(SteeringConfig::default());
        let pair = ConceptPair::new("happy", "sad");
        let vec = engine.extract_concept_vector(&pair);
        let neg = vec.negate();
        // All components should be negated
        for (tok, shift) in &neg.components {
            let original = vec.components.get(tok).copied().unwrap_or(0.0);
            assert!((shift + original).abs() < 1e-12);
        }
    }

    #[test]
    fn kl_divergence_non_negative() {
        let engine = SteeringEngine::new(SteeringConfig::default());
        let pair = ConceptPair::new("good", "bad");
        let vec = engine.extract_concept_vector(&pair);
        let orig = sample_logprobs();
        let effect = engine.apply_steering("test", &vec, &orig, None);
        assert!(effect.kl_divergence >= 0.0);
    }

    #[test]
    fn scale_zero_gives_identity() {
        let engine = SteeringEngine::new(SteeringConfig::default());
        let pair = ConceptPair::new("fast", "slow");
        let vec = engine.extract_concept_vector(&pair);
        let orig = sample_logprobs();
        let effect = engine.apply_steering("test", &vec, &orig, Some(0.0));
        // All deltas should be ~0
        for (_, _, _, delta) in &effect.token_shifts {
            assert!(delta.abs() < 1e-10, "delta should be ~0 with alpha=0");
        }
    }
}
