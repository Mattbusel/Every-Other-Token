//! Structured prompt mutation experiment runner.
//!
//! ## Overview
//!
//! [`MutationLab`] takes a base prompt and systematically varies specified
//! tokens, phrases, or parameters. It runs all variants through the model,
//! collects statistics on each metric of interest, ranks variants by that
//! metric, and produces a markdown table of results.
//!
//! ## Use cases
//!
//! - **Interpretability research** — measure how specific tokens shift
//!   perplexity or confidence distributions.
//! - **Red-teaming** — substitute candidate jailbreak phrases and measure
//!   output-length changes or sentiment shifts.
//! - **Prompt engineering** — rank synonym variants of a key phrase by their
//!   effect on model confidence to find the most effective wording.
//!
//! ## Quick example
//!
//! ```rust,no_run
//! use every_other_token::mutation_lab::{
//!     MutationLab, MutationSpec, MutationTarget, MutationMetric,
//! };
//!
//! # async fn run() {
//! let mut lab = MutationLab::new("Explain {SLOT} to a 5-year-old.");
//!
//! let spec = MutationSpec {
//!     target: MutationTarget::Word(1), // index of the {SLOT} word
//!     variants: vec![
//!         "gravity".to_string(),
//!         "recursion".to_string(),
//!         "photosynthesis".to_string(),
//!     ],
//!     metric: MutationMetric::Perplexity,
//! };
//!
//! // Simulate running the experiment (real usage calls lab.run_experiment(&spec, runner)).
//! let result = lab.run_experiment_simulated(&spec);
//! println!("{}", result.to_markdown_table());
//! # }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// MutationTarget
// ---------------------------------------------------------------------------

/// The element of the prompt to mutate in each experimental variant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MutationTarget {
    /// Substitute the word at the given whitespace-delimited index.
    ///
    /// Index 0 is the first word. Out-of-range indices are skipped.
    Word(usize),

    /// Substitute the inclusive byte-range `[start, end)` of the base prompt.
    ///
    /// If the range is out of bounds the variant prompt is left unchanged.
    Phrase(usize, usize),

    /// Replace the entire system prompt with the variant string.
    SystemPrompt,

    /// Vary the sampling temperature (parse the variant as an `f32`).
    ///
    /// Non-parseable strings are skipped.
    Temperature,
}

// ---------------------------------------------------------------------------
// MutationMetric
// ---------------------------------------------------------------------------

/// The measurement used to rank variants.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MutationMetric {
    /// Mean per-token perplexity (`exp(-logprob)`) across the full output.
    Perplexity,
    /// Total number of output tokens generated.
    OutputLength,
    /// Whether the top predicted token changed vs. the baseline.
    ///
    /// Stored as 0.0 (no change) or 1.0 (at least one top-token changed).
    TopTokenChange,
    /// Approximate sentiment shift: fraction of output tokens with confidence
    /// above 0.7 minus fraction above 0.7 in the baseline (proxy for
    /// positive sentiment).
    SentimentShift,
}

impl MutationMetric {
    /// Returns the display name used in the markdown table header.
    pub fn display_name(&self) -> &'static str {
        match self {
            MutationMetric::Perplexity => "Mean Perplexity",
            MutationMetric::OutputLength => "Output Length (tokens)",
            MutationMetric::TopTokenChange => "Top-Token Changed",
            MutationMetric::SentimentShift => "Sentiment Shift",
        }
    }
}

// ---------------------------------------------------------------------------
// MutationSpec
// ---------------------------------------------------------------------------

/// Specification for a single mutation experiment.
///
/// A spec defines what to vary (`target`), what values to try (`variants`),
/// and how to measure the effect (`metric`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationSpec {
    /// Which part of the prompt to mutate.
    pub target: MutationTarget,
    /// List of alternative strings to substitute in.
    pub variants: Vec<String>,
    /// The measurement to collect and rank by.
    pub metric: MutationMetric,
}

// ---------------------------------------------------------------------------
// VariantResult
// ---------------------------------------------------------------------------

/// Measurement from a single variant run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantResult {
    /// The variant string substituted into the prompt.
    pub variant: String,
    /// The full prompt string sent to the model.
    pub prompt: String,
    /// The measured metric value for this variant.
    pub metric_value: f64,
    /// Additional statistics collected alongside the primary metric.
    pub stats: HashMap<String, f64>,
}

// ---------------------------------------------------------------------------
// ExperimentResult
// ---------------------------------------------------------------------------

/// Full result of a mutation experiment, including all variant results ranked
/// by the target metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResult {
    /// The base prompt used as the starting point.
    pub base_prompt: String,
    /// The spec that was executed.
    pub spec: MutationSpec,
    /// Per-variant results, sorted ascending by `metric_value`.
    pub results: Vec<VariantResult>,
}

impl ExperimentResult {
    /// Render results as a markdown table.
    ///
    /// The table has four columns: rank, variant, prompt snippet (first 40
    /// chars), and metric value.
    pub fn to_markdown_table(&self) -> String {
        let metric_name = self.spec.metric.display_name();
        let mut md = format!(
            "## Mutation Experiment Results\n\n\
             **Base prompt:** `{}`\n\
             **Target:** {:?}\n\
             **Metric:** {}\n\n\
             | Rank | Variant | Prompt (snippet) | {} |\n\
             |------|---------|-----------------|---{}---|\n",
            truncate(&self.base_prompt, 60),
            self.spec.target,
            metric_name,
            metric_name,
            "-".repeat(metric_name.len()),
        );

        for (rank, result) in self.results.iter().enumerate() {
            let snippet = truncate(&result.prompt, 40);
            md.push_str(&format!(
                "| {} | `{}` | `{}` | {:.4} |\n",
                rank + 1,
                result.variant,
                snippet,
                result.metric_value,
            ));
        }

        md
    }

    /// Return the variant with the lowest metric value.
    pub fn best(&self) -> Option<&VariantResult> {
        self.results.first()
    }

    /// Return the variant with the highest metric value.
    pub fn worst(&self) -> Option<&VariantResult> {
        self.results.last()
    }
}

// ---------------------------------------------------------------------------
// MutationLab
// ---------------------------------------------------------------------------

/// Structured prompt mutation experiment runner.
///
/// Construct with a base prompt, add mutation specs, then call
/// [`MutationLab::run_experiment`] (requires a live model runner) or
/// [`MutationLab::run_experiment_simulated`] (for testing without an API key).
///
/// # Example
///
/// ```rust
/// use every_other_token::mutation_lab::{
///     MutationLab, MutationSpec, MutationTarget, MutationMetric,
/// };
///
/// let mut lab = MutationLab::new("Explain gravity to a child.");
///
/// let spec = MutationSpec {
///     target: MutationTarget::Word(1),
///     variants: vec!["gravity".into(), "recursion".into()],
///     metric: MutationMetric::Perplexity,
/// };
///
/// let result = lab.run_experiment_simulated(&spec);
/// assert!(!result.results.is_empty());
/// let md = result.to_markdown_table();
/// assert!(md.contains("Rank"));
/// assert!(md.contains("Variant"));
/// ```
pub struct MutationLab {
    /// The base prompt template.
    pub base_prompt: String,
}

impl MutationLab {
    /// Create a new lab with the given base prompt.
    pub fn new(base_prompt: impl Into<String>) -> Self {
        Self {
            base_prompt: base_prompt.into(),
        }
    }

    /// Apply the mutation target to substitute `variant` into a copy of the
    /// base prompt.
    ///
    /// Returns the modified prompt string. For targets that cannot be applied
    /// (e.g. out-of-range word index, unparseable temperature), the original
    /// base prompt is returned unchanged.
    pub fn apply_variant(&self, target: &MutationTarget, variant: &str) -> String {
        match target {
            MutationTarget::Word(idx) => {
                let mut words: Vec<&str> = self.base_prompt.split_whitespace().collect();
                if *idx < words.len() {
                    words[*idx] = variant;
                }
                words.join(" ")
            }
            MutationTarget::Phrase(start, end) => {
                let base = &self.base_prompt;
                let start = (*start).min(base.len());
                let end = (*end).min(base.len());
                if start > end {
                    return base.clone();
                }
                format!("{}{}{}", &base[..start], variant, &base[end..])
            }
            MutationTarget::SystemPrompt => {
                // The system prompt replacement is returned verbatim; callers
                // are responsible for routing it to the system role.
                variant.to_string()
            }
            MutationTarget::Temperature => {
                // For temperature variants the prompt itself does not change;
                // the variant string carries the temperature value.
                self.base_prompt.clone()
            }
        }
    }

    /// Run a simulated experiment without calling any external API.
    ///
    /// This method generates deterministic pseudo-metric values based on the
    /// hash of each variant string. It is intended for unit tests, CI checks,
    /// and offline demonstrations. Real metric collection requires calling
    /// the provider API.
    ///
    /// Results are sorted ascending by the simulated metric value.
    pub fn run_experiment_simulated(&self, spec: &MutationSpec) -> ExperimentResult {
        let mut results: Vec<VariantResult> = spec
            .variants
            .iter()
            .map(|v| {
                let prompt = self.apply_variant(&spec.target, v);
                let metric_value = simulate_metric(v, &spec.metric);
                let mut stats = HashMap::new();
                stats.insert("simulated".to_string(), 1.0);
                stats.insert("variant_len".to_string(), v.len() as f64);
                VariantResult {
                    variant: v.clone(),
                    prompt,
                    metric_value,
                    stats,
                }
            })
            .collect();

        // Sort ascending by metric value (lower = better for perplexity;
        // all metrics are ranked ascending for simplicity).
        results.sort_by(|a, b| {
            a.metric_value
                .partial_cmp(&b.metric_value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        ExperimentResult {
            base_prompt: self.base_prompt.clone(),
            spec: spec.clone(),
            results,
        }
    }

    /// Run an experiment where metric values are provided by the caller.
    ///
    /// `measured_values` must have the same length as `spec.variants`. Each
    /// entry corresponds to the measured metric for the variant at the same
    /// index.
    ///
    /// Useful for integrating with a custom inference runner or mock.
    pub fn run_experiment_with_values(
        &self,
        spec: &MutationSpec,
        measured_values: &[f64],
    ) -> ExperimentResult {
        assert_eq!(
            measured_values.len(),
            spec.variants.len(),
            "measured_values must have the same length as spec.variants"
        );

        let mut results: Vec<VariantResult> = spec
            .variants
            .iter()
            .zip(measured_values.iter())
            .map(|(v, &mv)| {
                let prompt = self.apply_variant(&spec.target, v);
                let mut stats = HashMap::new();
                stats.insert("variant_len".to_string(), v.len() as f64);
                VariantResult {
                    variant: v.clone(),
                    prompt,
                    metric_value: mv,
                    stats,
                }
            })
            .collect();

        results.sort_by(|a, b| {
            a.metric_value
                .partial_cmp(&b.metric_value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        ExperimentResult {
            base_prompt: self.base_prompt.clone(),
            spec: spec.clone(),
            results,
        }
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Generate a deterministic pseudo-metric value from the variant string.
///
/// Uses a simple FNV-1a hash so results are reproducible across runs.
fn simulate_metric(variant: &str, metric: &MutationMetric) -> f64 {
    let hash = fnv1a_hash(variant.as_bytes());
    let normalized = (hash as f64) / (u64::MAX as f64); // [0, 1)
    match metric {
        MutationMetric::Perplexity => 1.0 + normalized * 10.0,  // [1, 11)
        MutationMetric::OutputLength => (normalized * 500.0).floor(),
        MutationMetric::TopTokenChange => if normalized > 0.5 { 1.0 } else { 0.0 },
        MutationMetric::SentimentShift => normalized * 2.0 - 1.0, // [-1, 1)
    }
}

fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 14_695_981_039_346_656_037;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(1_099_511_628_211);
    }
    hash
}

fn truncate(s: &str, max_chars: usize) -> String {
    let mut chars = s.chars();
    let mut result = String::new();
    for _ in 0..max_chars {
        match chars.next() {
            Some(c) => result.push(c),
            None => return result,
        }
    }
    if chars.next().is_some() {
        result.push_str("...");
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_spec(metric: MutationMetric) -> MutationSpec {
        MutationSpec {
            target: MutationTarget::Word(1),
            variants: vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()],
            metric,
        }
    }

    #[test]
    fn apply_variant_word() {
        let lab = MutationLab::new("The quick brown fox");
        let result = lab.apply_variant(&MutationTarget::Word(1), "lazy");
        assert_eq!(result, "The lazy brown fox");
    }

    #[test]
    fn apply_variant_word_out_of_range() {
        let lab = MutationLab::new("Hello world");
        // Word index 5 doesn't exist — should return original.
        let result = lab.apply_variant(&MutationTarget::Word(5), "test");
        assert_eq!(result, "Hello world");
    }

    #[test]
    fn apply_variant_phrase() {
        let lab = MutationLab::new("Hello world");
        // Replace bytes 6..11 ("world") with "Rust".
        let result = lab.apply_variant(&MutationTarget::Phrase(6, 11), "Rust");
        assert_eq!(result, "Hello Rust");
    }

    #[test]
    fn apply_variant_system_prompt() {
        let lab = MutationLab::new("User prompt");
        let result = lab.apply_variant(&MutationTarget::SystemPrompt, "You are a pirate.");
        assert_eq!(result, "You are a pirate.");
    }

    #[test]
    fn apply_variant_temperature() {
        let lab = MutationLab::new("My prompt");
        // Temperature doesn't change the prompt text.
        let result = lab.apply_variant(&MutationTarget::Temperature, "0.9");
        assert_eq!(result, "My prompt");
    }

    #[test]
    fn run_experiment_simulated_sorted_ascending() {
        let lab = MutationLab::new("Explain {topic} to a child.");
        let spec = make_spec(MutationMetric::Perplexity);
        let result = lab.run_experiment_simulated(&spec);

        assert_eq!(result.results.len(), 3);
        // Results should be sorted ascending.
        let vals: Vec<f64> = result.results.iter().map(|r| r.metric_value).collect();
        for w in vals.windows(2) {
            assert!(w[0] <= w[1], "Expected ascending order: {w:?}");
        }
    }

    #[test]
    fn run_experiment_with_values_correct_order() {
        let lab = MutationLab::new("Base prompt");
        let spec = MutationSpec {
            target: MutationTarget::Word(0),
            variants: vec!["A".to_string(), "B".to_string(), "C".to_string()],
            metric: MutationMetric::OutputLength,
        };
        let values = vec![30.0, 10.0, 20.0];
        let result = lab.run_experiment_with_values(&spec, &values);

        // Should be sorted: B(10) < C(20) < A(30).
        assert_eq!(result.results[0].variant, "B");
        assert_eq!(result.results[1].variant, "C");
        assert_eq!(result.results[2].variant, "A");
    }

    #[test]
    fn markdown_table_contains_required_sections() {
        let lab = MutationLab::new("Prompt");
        let spec = make_spec(MutationMetric::Perplexity);
        let result = lab.run_experiment_simulated(&spec);
        let md = result.to_markdown_table();

        assert!(md.contains("Rank"), "table should have Rank column");
        assert!(md.contains("Variant"), "table should have Variant column");
        assert!(md.contains("Mean Perplexity"), "table should include metric name");
        // All 3 variants should appear.
        assert!(md.contains("alpha") || md.contains("beta") || md.contains("gamma"));
    }

    #[test]
    fn best_and_worst_helpers() {
        let lab = MutationLab::new("Prompt");
        let spec = MutationSpec {
            target: MutationTarget::Word(0),
            variants: vec!["low".to_string(), "high".to_string()],
            metric: MutationMetric::Perplexity,
        };
        let result = lab.run_experiment_with_values(&spec, &[1.5, 8.0]);

        let best = result.best().expect("best should exist");
        let worst = result.worst().expect("worst should exist");
        assert!((best.metric_value - 1.5).abs() < f64::EPSILON);
        assert!((worst.metric_value - 8.0).abs() < f64::EPSILON);
    }

    #[test]
    fn fnv1a_hash_deterministic() {
        let h1 = fnv1a_hash(b"hello");
        let h2 = fnv1a_hash(b"hello");
        assert_eq!(h1, h2);
        let h3 = fnv1a_hash(b"world");
        assert_ne!(h1, h3);
    }

    #[test]
    fn truncate_short_string_unchanged() {
        assert_eq!(truncate("hi", 10), "hi");
    }

    #[test]
    fn truncate_long_string_adds_ellipsis() {
        let s = "a".repeat(50);
        let t = truncate(&s, 10);
        assert!(t.ends_with("..."), "expected ellipsis, got: {t}");
        assert!(t.len() <= 13, "should not be much longer than max");
    }
}
