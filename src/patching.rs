//! # Activation Patching
//!
//! Token-level activation patching for mechanistic interpretability.
//!
//! ## Method
//!
//! Run the model twice:
//!   1. **Clean run** – original prompt, record logprobs at each token position.
//!   2. **Corrupted run** – corrupted prompt (token substitutions), record logprobs.
//!
//! "Patching" means restoring the clean activation at a specific position into the
//! corrupted run and measuring how much the final output recovers.  Because we
//! operate over the API (no direct weight access), we approximate the patch effect
//! via a *logit-difference* metric:
//!
//!   `patch_effect(pos) = logprob_clean(target) - logprob_corrupted(target)`
//!
//! A large positive value at position `pos` means that token's activation carries
//! information critical to producing the target output.
//!
//! ## Usage
//!
//! ```rust
//! use every_other_token::patching::{ActivationPatcher, PatchConfig, CorruptionStrategy};
//!
//! let cfg = PatchConfig::default();
//! let patcher = ActivationPatcher::new(cfg);
//!
//! let prompt = "The capital of France is";
//! let corrupted = patcher.corrupt(prompt, CorruptionStrategy::RandomTokenSwap);
//! assert_ne!(prompt, corrupted.as_str());
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Corruption strategies ─────────────────────────────────────────────────────

/// How to corrupt the original prompt for the noisy run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CorruptionStrategy {
    /// Replace every token with a random token from a fixed vocabulary.
    RandomTokenSwap,
    /// Replace tokens with their antonyms / semantically distant words.
    SemanticNegation,
    /// Shuffle the token order uniformly at random.
    Shuffle,
    /// Zero-ablate: replace all tokens with a single neutral filler token ("the").
    ZeroAblate,
    /// Corrupt only a specific subset of positions (given as indices).
    PositionSubset(Vec<usize>),
}

impl Default for CorruptionStrategy {
    fn default() -> Self {
        Self::ZeroAblate
    }
}

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for activation patching experiments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchConfig {
    /// Which corruption strategy to apply to produce the corrupted run.
    pub corruption: CorruptionStrategy,
    /// Token positions to patch (empty = all positions).
    pub patch_positions: Vec<usize>,
    /// Target token whose log-probability is tracked as the metric.
    pub target_token: Option<String>,
    /// Number of independent patch trials to average over.
    pub num_trials: usize,
}

impl Default for PatchConfig {
    fn default() -> Self {
        Self {
            corruption: CorruptionStrategy::default(),
            patch_positions: vec![],
            target_token: None,
            num_trials: 1,
        }
    }
}

// ── Results ───────────────────────────────────────────────────────────────────

/// Per-position result of an activation patch experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionPatchResult {
    /// Token position that was patched.
    pub position: usize,
    /// The token string at this position in the clean prompt.
    pub clean_token: String,
    /// The token string at this position in the corrupted prompt.
    pub corrupted_token: String,
    /// Log-probability of the target token in the **clean** run.
    pub logprob_clean: f64,
    /// Log-probability of the target token in the **corrupted** run.
    pub logprob_corrupted: f64,
    /// Estimated patch effect: `logprob_clean - logprob_corrupted`.
    /// Positive → this position's clean activation helps produce the target.
    pub patch_effect: f64,
    /// Normalised patch effect (fraction of total recovery budget, 0–1).
    pub normalised_effect: f64,
}

/// Full result set for one patching experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchingResult {
    /// Original (clean) prompt.
    pub clean_prompt: String,
    /// Corrupted prompt used for the noisy baseline.
    pub corrupted_prompt: String,
    /// Target token being tracked.
    pub target_token: String,
    /// Per-position results, sorted by |patch_effect| descending.
    pub positions: Vec<PositionPatchResult>,
    /// Summary: which positions contributed most to recovery.
    pub top_positions: Vec<usize>,
}

impl PatchingResult {
    /// Return the positions sorted by patch effect (most positive first).
    pub fn ranked_positions(&self) -> Vec<&PositionPatchResult> {
        let mut refs: Vec<&PositionPatchResult> = self.positions.iter().collect();
        refs.sort_by(|a, b| b.patch_effect.partial_cmp(&a.patch_effect).unwrap());
        refs
    }

    /// Render a compact ASCII summary suitable for the TUI.
    pub fn ascii_summary(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "Activation Patching: target=\"{}\"\n",
            self.target_token
        ));
        out.push_str(&format!(
            "  clean:     {}\n",
            truncate(&self.clean_prompt, 60)
        ));
        out.push_str(&format!(
            "  corrupted: {}\n\n",
            truncate(&self.corrupted_prompt, 60)
        ));
        out.push_str("  pos  token           effect   norm\n");
        out.push_str("  ───  ─────────────  ───────  ────\n");
        for p in self.ranked_positions().iter().take(10) {
            out.push_str(&format!(
                "  {:>3}  {:<13}  {:>+7.3}  {:.2}\n",
                p.position,
                truncate(&p.clean_token, 13),
                p.patch_effect,
                p.normalised_effect,
            ));
        }
        out
    }

    /// Render a bar-chart heatmap of patch effects for each position.
    pub fn heatmap_bars(&self) -> Vec<(String, f64)> {
        self.positions
            .iter()
            .map(|p| (p.clean_token.clone(), p.patch_effect))
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

// ── Core patcher ──────────────────────────────────────────────────────────────

/// Activation patcher – simulates the clean/corrupted dual-run experiment.
pub struct ActivationPatcher {
    pub config: PatchConfig,
}

impl ActivationPatcher {
    pub fn new(config: PatchConfig) -> Self {
        Self { config }
    }

    /// Corrupt a prompt according to the configured strategy.
    ///
    /// In a real deployment this would call the tokeniser; here we operate on
    /// whitespace-split tokens as a transport-layer approximation.
    pub fn corrupt(&self, prompt: &str, strategy: CorruptionStrategy) -> String {
        let tokens: Vec<&str> = prompt.split_whitespace().collect();
        match strategy {
            CorruptionStrategy::ZeroAblate => {
                vec!["the"; tokens.len()].join(" ")
            }
            CorruptionStrategy::RandomTokenSwap => {
                let fillers = ["cat", "blue", "runs", "seven", "above", "cold"];
                tokens
                    .iter()
                    .enumerate()
                    .map(|(i, t)| {
                        // Deterministic pseudo-random based on position + length
                        if i % 3 == 0 {
                            fillers[i % fillers.len()]
                        } else {
                            t
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            }
            CorruptionStrategy::Shuffle => {
                let mut shuffled = tokens.clone();
                // Deterministic Fisher-Yates using position-based seed
                let n = shuffled.len();
                for i in (1..n).rev() {
                    let j = (i * 6364136223846793005 + 1442695040888963407) % (i + 1);
                    shuffled.swap(i, j);
                }
                shuffled.join(" ")
            }
            CorruptionStrategy::SemanticNegation => tokens
                .iter()
                .map(|t| negate_word(t))
                .collect::<Vec<_>>()
                .join(" "),
            CorruptionStrategy::PositionSubset(ref positions) => {
                let filler = "the";
                tokens
                    .iter()
                    .enumerate()
                    .map(|(i, t)| if positions.contains(&i) { filler } else { t })
                    .collect::<Vec<_>>()
                    .join(" ")
            }
        }
    }

    /// Simulate a patching experiment using pre-computed logprob tables.
    ///
    /// `clean_logprobs[pos]` and `corrupted_logprobs[pos]` should map target
    /// tokens to their log-probability at each position.  In a live integration
    /// these are obtained from two separate API calls with `logprobs=true`.
    pub fn run_patch_experiment(
        &self,
        clean_prompt: &str,
        corrupted_prompt: &str,
        target_token: &str,
        clean_logprobs: &[HashMap<String, f64>],
        corrupted_logprobs: &[HashMap<String, f64>],
    ) -> PatchingResult {
        let clean_tokens: Vec<&str> = clean_prompt.split_whitespace().collect();
        let corrupted_tokens: Vec<&str> = corrupted_prompt.split_whitespace().collect();

        let n = clean_tokens.len().min(corrupted_tokens.len());
        let patch_positions: Vec<usize> = if self.config.patch_positions.is_empty() {
            (0..n).collect()
        } else {
            self.config
                .patch_positions
                .iter()
                .cloned()
                .filter(|&p| p < n)
                .collect()
        };

        let mut positions: Vec<PositionPatchResult> = patch_positions
            .iter()
            .map(|&pos| {
                let lp_clean = clean_logprobs
                    .get(pos)
                    .and_then(|m| m.get(target_token))
                    .copied()
                    .unwrap_or(-10.0);
                let lp_corrupted = corrupted_logprobs
                    .get(pos)
                    .and_then(|m| m.get(target_token))
                    .copied()
                    .unwrap_or(-10.0);
                PositionPatchResult {
                    position: pos,
                    clean_token: clean_tokens.get(pos).unwrap_or(&"").to_string(),
                    corrupted_token: corrupted_tokens.get(pos).unwrap_or(&"").to_string(),
                    logprob_clean: lp_clean,
                    logprob_corrupted: lp_corrupted,
                    patch_effect: lp_clean - lp_corrupted,
                    normalised_effect: 0.0, // filled below
                }
            })
            .collect();

        // Normalise
        let total_effect: f64 = positions.iter().map(|p| p.patch_effect.max(0.0)).sum();
        if total_effect > 0.0 {
            for p in &mut positions {
                p.normalised_effect = p.patch_effect.max(0.0) / total_effect;
            }
        }

        let mut top: Vec<usize> = positions
            .iter()
            .enumerate()
            .filter(|(_, p)| p.patch_effect > 0.0)
            .map(|(i, _)| i)
            .collect();
        top.sort_by(|&a, &b| {
            positions[b]
                .patch_effect
                .partial_cmp(&positions[a].patch_effect)
                .unwrap()
        });
        top.truncate(5);

        PatchingResult {
            clean_prompt: clean_prompt.to_string(),
            corrupted_prompt: corrupted_prompt.to_string(),
            target_token: target_token.to_string(),
            positions,
            top_positions: top,
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn negate_word(word: &str) -> &str {
    let antonyms: &[(&str, &str)] = &[
        ("good", "bad"),
        ("bad", "good"),
        ("hot", "cold"),
        ("cold", "hot"),
        ("fast", "slow"),
        ("slow", "fast"),
        ("happy", "sad"),
        ("sad", "happy"),
        ("big", "small"),
        ("small", "big"),
        ("yes", "no"),
        ("no", "yes"),
        ("up", "down"),
        ("down", "up"),
        ("left", "right"),
        ("right", "left"),
    ];
    for (a, b) in antonyms {
        if word.eq_ignore_ascii_case(a) {
            return b;
        }
    }
    word
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corrupt_zero_ablate() {
        let patcher = ActivationPatcher::new(PatchConfig::default());
        let result = patcher.corrupt("hello world foo", CorruptionStrategy::ZeroAblate);
        assert_eq!(result, "the the the");
    }

    #[test]
    fn corrupt_semantic_negation() {
        let patcher = ActivationPatcher::new(PatchConfig::default());
        let result = patcher.corrupt("good hot fast", CorruptionStrategy::SemanticNegation);
        assert_eq!(result, "bad cold slow");
    }

    #[test]
    fn patch_experiment_basic() {
        let cfg = PatchConfig {
            patch_positions: vec![0, 1],
            target_token: Some("Paris".to_string()),
            ..Default::default()
        };
        let patcher = ActivationPatcher::new(cfg);

        let mut clean_lp = HashMap::new();
        clean_lp.insert("Paris".to_string(), -0.5);
        let mut corrupt_lp = HashMap::new();
        corrupt_lp.insert("Paris".to_string(), -3.0);

        let result = patcher.run_patch_experiment(
            "The capital of France is",
            "The the the the the",
            "Paris",
            &[clean_lp.clone(), clean_lp.clone()],
            &[corrupt_lp.clone(), corrupt_lp.clone()],
        );

        assert_eq!(result.positions.len(), 2);
        assert!((result.positions[0].patch_effect - 2.5).abs() < 1e-9);
    }

    #[test]
    fn ascii_summary_non_empty() {
        let result = PatchingResult {
            clean_prompt: "hello world".to_string(),
            corrupted_prompt: "the the".to_string(),
            target_token: "world".to_string(),
            positions: vec![PositionPatchResult {
                position: 0,
                clean_token: "hello".to_string(),
                corrupted_token: "the".to_string(),
                logprob_clean: -1.0,
                logprob_corrupted: -4.0,
                patch_effect: 3.0,
                normalised_effect: 1.0,
            }],
            top_positions: vec![0],
        };
        let summary = result.ascii_summary();
        assert!(summary.contains("world"));
        assert!(summary.contains("hello"));
    }
}
