//! # Prompt Sensitivity Analyzer
//!
//! For a given prompt, generates N variations (word substitutions, reorderings)
//! and measures how sensitive the model output is to each variation.
//! Reports a sensitivity score per prompt element.
//!
//! ## Method
//!
//! 1. Tokenise the prompt into words/elements.
//! 2. For each element, apply one or more mutations (swap, delete, substitute).
//! 3. Run the mutated prompt through the model and compare the output to the
//!    baseline (original prompt output) using a configurable distance metric.
//! 4. Aggregate per-element scores across all mutations.
//!
//! ## Usage
//!
//! ```rust
//! use every_other_token::sensitivity::{
//!     SensitivityAnalyzer, SensitivityConfig, MutationStrategy,
//! };
//!
//! let config = SensitivityConfig::default();
//! let analyzer = SensitivityAnalyzer::new(config);
//!
//! let prompt = "The quick brown fox";
//! let variations = analyzer.generate_variations(prompt);
//! assert!(!variations.is_empty());
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Mutation strategy ─────────────────────────────────────────────────────────

/// How to mutate individual prompt elements.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MutationStrategy {
    /// Replace a word with a synonym or random word from a small built-in list.
    WordSubstitution,
    /// Remove a word from the prompt.
    Deletion,
    /// Swap two adjacent words.
    AdjacentSwap,
    /// Reverse the order of all words in the sentence.
    Reversal,
    /// Apply all of the above.
    All,
}

impl Default for MutationStrategy {
    fn default() -> Self {
        Self::All
    }
}

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for the sensitivity analyzer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityConfig {
    /// Maximum number of variations to generate in total.
    pub max_variations: usize,
    /// Which mutation strategy to use.
    pub strategy: MutationStrategy,
    /// Distance metric for comparing outputs.
    pub distance_metric: DistanceMetric,
}

impl Default for SensitivityConfig {
    fn default() -> Self {
        Self {
            max_variations: 20,
            strategy: MutationStrategy::All,
            distance_metric: DistanceMetric::NormalisedEditDistance,
        }
    }
}

/// Metric used to compare two text outputs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Normalised Levenshtein edit distance in `[0, 1]`.
    NormalisedEditDistance,
    /// Jaccard distance on word sets: `1 - |A ∩ B| / |A ∪ B|`.
    JaccardWordDistance,
    /// Character-level overlap: `1 - (2 * |common_chars| / (|a| + |b|))`.
    CharacterOverlap,
}

// ── Variation ─────────────────────────────────────────────────────────────────

/// A single mutated version of the original prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptVariation {
    /// Zero-based index of the element that was mutated.
    pub element_index: usize,
    /// Surface form of the element that was mutated.
    pub element: String,
    /// The mutation applied.
    pub mutation: String,
    /// The resulting mutated prompt text.
    pub mutated_prompt: String,
}

// ── Element score ──────────────────────────────────────────────────────────────

/// Sensitivity score for a single prompt element.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElementScore {
    /// Zero-based index of the element in the original prompt.
    pub index: usize,
    /// Surface form of the element.
    pub element: String,
    /// Mean sensitivity score across all mutations applied to this element
    /// (higher = more sensitive = output changes more when this element changes).
    pub sensitivity: f64,
    /// Number of variations that targeted this element.
    pub variation_count: usize,
}

// ── Report ────────────────────────────────────────────────────────────────────

/// Full sensitivity analysis report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityReport {
    /// Per-element sensitivity scores, sorted by sensitivity descending.
    pub element_scores: Vec<ElementScore>,
    /// Overall mean sensitivity across all elements.
    pub mean_sensitivity: f64,
    /// Element with the highest sensitivity score.
    pub most_sensitive_element: Option<String>,
    /// Element with the lowest sensitivity score.
    pub least_sensitive_element: Option<String>,
    /// Total number of variations evaluated.
    pub variations_evaluated: usize,
}

// ── Synonym table (built-in, minimal) ─────────────────────────────────────────

fn built_in_substitution(word: &str) -> Option<&'static str> {
    match word.to_lowercase().as_str() {
        "quick" => Some("fast"),
        "fast" => Some("quick"),
        "good" => Some("excellent"),
        "bad" => Some("poor"),
        "big" => Some("large"),
        "small" => Some("tiny"),
        "smart" => Some("intelligent"),
        "the" => Some("a"),
        "a" => Some("the"),
        "run" => Some("execute"),
        "execute" => Some("run"),
        "important" => Some("critical"),
        "critical" => Some("essential"),
        _ => None,
    }
}

// ── Distance metrics ──────────────────────────────────────────────────────────

/// Compute normalised edit distance between two strings.
fn normalised_edit_distance(a: &str, b: &str) -> f64 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 && n == 0 {
        return 0.0;
    }
    if m == 0 {
        return 1.0;
    }
    if n == 0 {
        return 1.0;
    }

    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }
    for i in 1..=m {
        for j in 1..=n {
            dp[i][j] = if a_chars[i - 1] == b_chars[j - 1] {
                dp[i - 1][j - 1]
            } else {
                1 + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1])
            };
        }
    }

    dp[m][n] as f64 / m.max(n) as f64
}

/// Compute Jaccard distance on word sets.
fn jaccard_word_distance(a: &str, b: &str) -> f64 {
    let set_a: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let set_b: std::collections::HashSet<&str> = b.split_whitespace().collect();
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 {
        0.0
    } else {
        1.0 - intersection as f64 / union as f64
    }
}

/// Compute character-level overlap distance.
fn character_overlap_distance(a: &str, b: &str) -> f64 {
    let set_a: std::collections::HashSet<char> = a.chars().collect();
    let set_b: std::collections::HashSet<char> = b.chars().collect();
    let common = set_a.intersection(&set_b).count();
    let total = a.len() + b.len();
    if total == 0 {
        0.0
    } else {
        1.0 - (2.0 * common as f64) / total as f64
    }
}

// ── Analyzer ──────────────────────────────────────────────────────────────────

/// Generates prompt variations and measures output sensitivity.
pub struct SensitivityAnalyzer {
    config: SensitivityConfig,
}

impl SensitivityAnalyzer {
    /// Create a new analyzer with the given config.
    pub fn new(config: SensitivityConfig) -> Self {
        Self { config }
    }

    /// Tokenise a prompt into words (whitespace-split, preserving punctuation).
    pub fn tokenise(prompt: &str) -> Vec<String> {
        prompt
            .split_whitespace()
            .map(|w| w.to_string())
            .collect()
    }

    /// Generate variations for the prompt using the configured strategy.
    pub fn generate_variations(&self, prompt: &str) -> Vec<PromptVariation> {
        let words = Self::tokenise(prompt);
        if words.is_empty() {
            return vec![];
        }

        let mut variations: Vec<PromptVariation> = Vec::new();
        let strategy = &self.config.strategy;

        for (i, word) in words.iter().enumerate() {
            // Word substitution.
            if matches!(strategy, MutationStrategy::WordSubstitution | MutationStrategy::All) {
                if let Some(sub) = built_in_substitution(word) {
                    let mut mutated = words.clone();
                    mutated[i] = sub.to_string();
                    variations.push(PromptVariation {
                        element_index: i,
                        element: word.clone(),
                        mutation: format!("substitute '{word}' → '{sub}'"),
                        mutated_prompt: mutated.join(" "),
                    });
                }
            }

            // Deletion.
            if matches!(strategy, MutationStrategy::Deletion | MutationStrategy::All) {
                let mut mutated = words.clone();
                mutated.remove(i);
                variations.push(PromptVariation {
                    element_index: i,
                    element: word.clone(),
                    mutation: format!("delete '{word}'"),
                    mutated_prompt: mutated.join(" "),
                });
            }

            // Adjacent swap.
            if matches!(strategy, MutationStrategy::AdjacentSwap | MutationStrategy::All) {
                if i + 1 < words.len() {
                    let mut mutated = words.clone();
                    mutated.swap(i, i + 1);
                    variations.push(PromptVariation {
                        element_index: i,
                        element: word.clone(),
                        mutation: format!("swap '{word}' ↔ '{}'", words[i + 1]),
                        mutated_prompt: mutated.join(" "),
                    });
                }
            }

            if variations.len() >= self.config.max_variations {
                break;
            }
        }

        // Reversal (whole-prompt mutation, attributed to element 0).
        if matches!(strategy, MutationStrategy::Reversal | MutationStrategy::All)
            && variations.len() < self.config.max_variations
        {
            let reversed: Vec<String> = words.iter().cloned().rev().collect();
            variations.push(PromptVariation {
                element_index: 0,
                element: words[0].clone(),
                mutation: "reverse word order".to_string(),
                mutated_prompt: reversed.join(" "),
            });
        }

        variations.truncate(self.config.max_variations);
        variations
    }

    /// Compute the distance between two outputs using the configured metric.
    pub fn distance(&self, a: &str, b: &str) -> f64 {
        match self.config.distance_metric {
            DistanceMetric::NormalisedEditDistance => normalised_edit_distance(a, b),
            DistanceMetric::JaccardWordDistance => jaccard_word_distance(a, b),
            DistanceMetric::CharacterOverlap => character_overlap_distance(a, b),
        }
    }

    /// Build a [`SensitivityReport`] from a baseline output and a list of
    /// `(variation, model_output)` pairs.
    ///
    /// Pass this the results of calling the model on each variation produced by
    /// [`generate_variations`].
    pub fn build_report(
        &self,
        prompt: &str,
        baseline_output: &str,
        variation_outputs: &[(PromptVariation, String)],
    ) -> SensitivityReport {
        let words = Self::tokenise(prompt);
        let variations_evaluated = variation_outputs.len();

        // Accumulate per-element scores.
        let mut scores_by_element: HashMap<usize, Vec<f64>> = HashMap::new();

        for (variation, output) in variation_outputs {
            let dist = self.distance(baseline_output, output);
            scores_by_element
                .entry(variation.element_index)
                .or_default()
                .push(dist);
        }

        let mut element_scores: Vec<ElementScore> = scores_by_element
            .into_iter()
            .map(|(idx, scores)| {
                let mean = scores.iter().sum::<f64>() / scores.len() as f64;
                ElementScore {
                    index: idx,
                    element: words
                        .get(idx)
                        .cloned()
                        .unwrap_or_else(|| format!("<{idx}>")),
                    sensitivity: mean,
                    variation_count: scores.len(),
                }
            })
            .collect();

        element_scores.sort_by(|a, b| {
            b.sensitivity
                .partial_cmp(&a.sensitivity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mean_sensitivity = if element_scores.is_empty() {
            0.0
        } else {
            element_scores.iter().map(|e| e.sensitivity).sum::<f64>()
                / element_scores.len() as f64
        };

        let most_sensitive_element = element_scores.first().map(|e| e.element.clone());
        let least_sensitive_element = element_scores.last().map(|e| e.element.clone());

        SensitivityReport {
            element_scores,
            mean_sensitivity,
            most_sensitive_element,
            least_sensitive_element,
            variations_evaluated,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenise_basic() {
        let tokens = SensitivityAnalyzer::tokenise("The quick brown fox");
        assert_eq!(tokens, ["The", "quick", "brown", "fox"]);
    }

    #[test]
    fn generate_variations_non_empty() {
        let analyzer = SensitivityAnalyzer::new(SensitivityConfig::default());
        let vars = analyzer.generate_variations("The quick brown fox");
        assert!(!vars.is_empty());
    }

    #[test]
    fn deletion_variation_is_shorter() {
        let analyzer = SensitivityAnalyzer::new(SensitivityConfig {
            strategy: MutationStrategy::Deletion,
            max_variations: 100,
            ..Default::default()
        });
        let vars = analyzer.generate_variations("The quick brown");
        for v in &vars {
            let orig_words = 3usize;
            let mutated_words = v.mutated_prompt.split_whitespace().count();
            assert_eq!(mutated_words, orig_words - 1);
        }
    }

    #[test]
    fn swap_variation_same_length() {
        let analyzer = SensitivityAnalyzer::new(SensitivityConfig {
            strategy: MutationStrategy::AdjacentSwap,
            max_variations: 100,
            ..Default::default()
        });
        let vars = analyzer.generate_variations("a b c d");
        for v in &vars {
            assert_eq!(
                v.mutated_prompt.split_whitespace().count(),
                4,
                "swap should not change word count"
            );
        }
    }

    #[test]
    fn reversal_variation_reverses_words() {
        let analyzer = SensitivityAnalyzer::new(SensitivityConfig {
            strategy: MutationStrategy::Reversal,
            max_variations: 100,
            ..Default::default()
        });
        let vars = analyzer.generate_variations("alpha beta gamma");
        let rev = vars
            .iter()
            .find(|v| v.mutation.contains("reverse"))
            .expect("reversal variation should exist");
        assert_eq!(rev.mutated_prompt, "gamma beta alpha");
    }

    #[test]
    fn max_variations_respected() {
        let analyzer = SensitivityAnalyzer::new(SensitivityConfig {
            max_variations: 3,
            strategy: MutationStrategy::All,
            ..Default::default()
        });
        let vars = analyzer.generate_variations("The quick brown fox jumps");
        assert!(vars.len() <= 3);
    }

    #[test]
    fn normalised_edit_distance_identical() {
        assert_eq!(normalised_edit_distance("abc", "abc"), 0.0);
    }

    #[test]
    fn normalised_edit_distance_completely_different() {
        let d = normalised_edit_distance("abc", "xyz");
        assert!(d > 0.5);
    }

    #[test]
    fn jaccard_identical_sets() {
        assert_eq!(jaccard_word_distance("the cat sat", "the cat sat"), 0.0);
    }

    #[test]
    fn jaccard_disjoint_sets() {
        assert_eq!(jaccard_word_distance("abc def", "ghi jkl"), 1.0);
    }

    #[test]
    fn build_report_scores_elements() {
        let analyzer = SensitivityAnalyzer::new(SensitivityConfig::default());
        let prompt = "The quick fox";
        let baseline = "The fox is quick";

        let variations = vec![
            (
                PromptVariation {
                    element_index: 1,
                    element: "quick".into(),
                    mutation: "delete 'quick'".into(),
                    mutated_prompt: "The fox".into(),
                },
                "The fox".to_string(),
            ),
            (
                PromptVariation {
                    element_index: 0,
                    element: "The".into(),
                    mutation: "substitute 'The' → 'a'".into(),
                    mutated_prompt: "a quick fox".into(),
                },
                "a fox is quick".to_string(),
            ),
        ];

        let report = analyzer.build_report(prompt, baseline, &variations);
        assert_eq!(report.variations_evaluated, 2);
        assert!(!report.element_scores.is_empty());
        assert!(report.mean_sensitivity >= 0.0);
        assert!(report.most_sensitive_element.is_some());
    }
}
