//! Chain-of-thought step validation.
//!
//! Provides structures and logic for validating multi-step reasoning chains,
//! including cycle detection, reference validity, and confidence scoring.

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A single step in a chain-of-thought reasoning sequence.
#[derive(Debug, Clone)]
pub struct ReasoningStep {
    /// 1-based step number.
    pub step_number: u32,
    /// Text content of the step.
    pub content: String,
    /// Model confidence for this step in [0.0, 1.0].
    pub confidence: f64,
    /// Step numbers that this step explicitly references.
    pub references_prior: Vec<u32>,
}

/// A rule that can be applied to a reasoning chain.
#[derive(Debug, Clone)]
pub enum ChainValidationRule {
    /// The chain must have at least this many steps.
    MinSteps(usize),
    /// The chain must have at most this many steps.
    MaxSteps(usize),
    /// Every step must have confidence >= this value.
    ConfidenceFloor(f64),
    /// References must not form a cycle.
    NoCycles,
    /// All referenced step numbers must exist in the chain.
    ReferencesValid,
    /// No step may have empty content.
    StepsNonEmpty,
}

/// Result of validating a chain against a set of rules.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether all rules passed.
    pub passed: bool,
    /// Human-readable failure messages.
    pub failures: Vec<String>,
    /// Non-fatal warning messages.
    pub warnings: Vec<String>,
    /// Per-step scores (same order as input slice).
    pub step_scores: Vec<f64>,
}

// ---------------------------------------------------------------------------
// ChainValidator
// ---------------------------------------------------------------------------

/// Validates reasoning chains against a configurable set of rules.
pub struct ChainValidator {
    pub rules: Vec<ChainValidationRule>,
}

impl ChainValidator {
    pub fn new(rules: Vec<ChainValidationRule>) -> Self {
        Self { rules }
    }

    /// Validate `steps` against all configured rules and return a [`ValidationResult`].
    pub fn validate(&self, steps: &[ReasoningStep]) -> ValidationResult {
        let mut failures = Vec::new();
        let mut warnings = Vec::new();

        // Build a set of valid step numbers for O(1) lookup.
        let step_nums: HashSet<u32> = steps.iter().map(|s| s.step_number).collect();

        for rule in &self.rules {
            match rule {
                ChainValidationRule::MinSteps(min) => {
                    if steps.len() < *min {
                        failures.push(format!(
                            "MinSteps: expected >= {}, got {}",
                            min,
                            steps.len()
                        ));
                    }
                }
                ChainValidationRule::MaxSteps(max) => {
                    if steps.len() > *max {
                        failures.push(format!(
                            "MaxSteps: expected <= {}, got {}",
                            max,
                            steps.len()
                        ));
                    }
                }
                ChainValidationRule::ConfidenceFloor(floor) => {
                    for s in steps {
                        if s.confidence < *floor {
                            failures.push(format!(
                                "ConfidenceFloor: step {} has confidence {:.3} < {:.3}",
                                s.step_number, s.confidence, floor
                            ));
                        }
                    }
                }
                ChainValidationRule::NoCycles => {
                    if let Some(cycle_desc) = detect_cycle(steps) {
                        failures.push(format!("NoCycles: {}", cycle_desc));
                    }
                }
                ChainValidationRule::ReferencesValid => {
                    for s in steps {
                        for &r in &s.references_prior {
                            if !step_nums.contains(&r) {
                                failures.push(format!(
                                    "ReferencesValid: step {} references non-existent step {}",
                                    s.step_number, r
                                ));
                            }
                        }
                    }
                }
                ChainValidationRule::StepsNonEmpty => {
                    for s in steps {
                        if s.content.trim().is_empty() {
                            failures.push(format!(
                                "StepsNonEmpty: step {} has empty content",
                                s.step_number
                            ));
                        }
                    }
                }
            }
        }

        // Produce a warning if confidence is highly variable.
        if steps.len() > 1 {
            let avg = steps.iter().map(|s| s.confidence).sum::<f64>() / steps.len() as f64;
            let variance = steps
                .iter()
                .map(|s| (s.confidence - avg).powi(2))
                .sum::<f64>()
                / steps.len() as f64;
            if variance > 0.05 {
                warnings.push(format!(
                    "High confidence variance ({:.3}); chain may be inconsistent",
                    variance
                ));
            }
        }

        let step_scores: Vec<f64> = steps.iter().map(|s| s.confidence).collect();

        ValidationResult {
            passed: failures.is_empty(),
            failures,
            warnings,
            step_scores,
        }
    }
}

// ---------------------------------------------------------------------------
// Cycle detection (DFS)
// ---------------------------------------------------------------------------

/// Returns `Some(description)` if the reference graph contains a cycle, else `None`.
fn detect_cycle(steps: &[ReasoningStep]) -> Option<String> {
    // Build adjacency map: step_number -> referenced step numbers
    let adj: HashMap<u32, Vec<u32>> = steps
        .iter()
        .map(|s| (s.step_number, s.references_prior.clone()))
        .collect();

    let mut visited: HashSet<u32> = HashSet::new();
    let mut stack: HashSet<u32> = HashSet::new();

    for start in adj.keys().copied() {
        if !visited.contains(&start) {
            if dfs_has_cycle(start, &adj, &mut visited, &mut stack) {
                return Some(format!("cycle detected involving step {}", start));
            }
        }
    }
    None
}

fn dfs_has_cycle(
    node: u32,
    adj: &HashMap<u32, Vec<u32>>,
    visited: &mut HashSet<u32>,
    stack: &mut HashSet<u32>,
) -> bool {
    visited.insert(node);
    stack.insert(node);

    if let Some(neighbors) = adj.get(&node) {
        for &nb in neighbors {
            if !visited.contains(&nb) {
                if dfs_has_cycle(nb, adj, visited, stack) {
                    return true;
                }
            } else if stack.contains(&nb) {
                return true;
            }
        }
    }

    stack.remove(&node);
    false
}

// ---------------------------------------------------------------------------
// Score computation
// ---------------------------------------------------------------------------

/// Compute a holistic quality score for a reasoning chain.
///
/// Formula:
///   score = avg_confidence * completeness_penalty * length_bonus
///
/// * `completeness_penalty`: fraction of steps that have ≥ 1 reference (or are
///   the first step, which needs no reference). Steps with no back-references
///   beyond the first are penalised.
/// * `length_bonus`: log-scaled bonus for longer chains, capped at 1.0.
pub fn score_chain(steps: &[ReasoningStep]) -> f64 {
    if steps.is_empty() {
        return 0.0;
    }

    let avg_confidence = steps.iter().map(|s| s.confidence).sum::<f64>() / steps.len() as f64;

    // Completeness: every step except the first should reference something.
    let referenced_count = steps
        .iter()
        .enumerate()
        .filter(|(i, s)| *i == 0 || !s.references_prior.is_empty())
        .count();
    let completeness_penalty = referenced_count as f64 / steps.len() as f64;

    // Length bonus: ln(n+1) / ln(11) → 1.0 at n=10 steps.
    let length_bonus = ((steps.len() + 1) as f64).ln() / (11.0_f64).ln();
    let length_bonus = length_bonus.min(1.0);

    avg_confidence * completeness_penalty * length_bonus
}

// ---------------------------------------------------------------------------
// ChainBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing `Vec<ReasoningStep>`.
#[derive(Default)]
pub struct ChainBuilder {
    steps: Vec<ReasoningStep>,
}

impl ChainBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a step. Step numbers are assigned automatically (1-based).
    pub fn add_step(&mut self, content: &str, confidence: f64, refs: Vec<u32>) -> &mut Self {
        let step_number = (self.steps.len() + 1) as u32;
        self.steps.push(ReasoningStep {
            step_number,
            content: content.to_string(),
            confidence,
            references_prior: refs,
        });
        self
    }

    /// Consume the builder and return the completed chain.
    pub fn build(self) -> Vec<ReasoningStep> {
        self.steps
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_chain() -> Vec<ReasoningStep> {
        let mut b = ChainBuilder::new();
        b.add_step("Define the problem", 0.95, vec![]);
        b.add_step("Gather evidence", 0.90, vec![1]);
        b.add_step("Draw conclusion", 0.88, vec![1, 2]);
        b.build()
    }

    #[test]
    fn valid_chain_passes_all_rules() {
        let rules = vec![
            ChainValidationRule::MinSteps(2),
            ChainValidationRule::MaxSteps(10),
            ChainValidationRule::ConfidenceFloor(0.5),
            ChainValidationRule::NoCycles,
            ChainValidationRule::ReferencesValid,
            ChainValidationRule::StepsNonEmpty,
        ];
        let v = ChainValidator::new(rules);
        let result = v.validate(&valid_chain());
        assert!(result.passed, "failures: {:?}", result.failures);
    }

    #[test]
    fn cycle_detection_catches_cycle() {
        // Step 1 references step 3 → 1→3→2→1 cycle
        let steps = vec![
            ReasoningStep {
                step_number: 1,
                content: "A".into(),
                confidence: 0.9,
                references_prior: vec![3],
            },
            ReasoningStep {
                step_number: 2,
                content: "B".into(),
                confidence: 0.8,
                references_prior: vec![1],
            },
            ReasoningStep {
                step_number: 3,
                content: "C".into(),
                confidence: 0.7,
                references_prior: vec![2],
            },
        ];
        let v = ChainValidator::new(vec![ChainValidationRule::NoCycles]);
        let result = v.validate(&steps);
        assert!(!result.passed);
        assert!(result.failures.iter().any(|f| f.contains("cycle")));
    }

    #[test]
    fn invalid_references_caught() {
        let steps = vec![ReasoningStep {
            step_number: 1,
            content: "Intro".into(),
            confidence: 0.85,
            references_prior: vec![99], // step 99 does not exist
        }];
        let v = ChainValidator::new(vec![ChainValidationRule::ReferencesValid]);
        let result = v.validate(&steps);
        assert!(!result.passed);
        assert!(result.failures.iter().any(|f| f.contains("99")));
    }

    #[test]
    fn score_computation_is_reasonable() {
        let chain = valid_chain();
        let score = score_chain(&chain);
        // With high confidence and all steps referencing priors, score should be > 0.5
        assert!(score > 0.5, "score was {}", score);
        // Score should not exceed 1.0
        assert!(score <= 1.0, "score was {}", score);
    }

    #[test]
    fn empty_chain_scores_zero() {
        assert_eq!(score_chain(&[]), 0.0);
    }

    #[test]
    fn min_steps_rule_fails_correctly() {
        let chain = valid_chain(); // 3 steps
        let v = ChainValidator::new(vec![ChainValidationRule::MinSteps(5)]);
        let result = v.validate(&chain);
        assert!(!result.passed);
    }

    #[test]
    fn empty_content_fails_steps_non_empty() {
        let steps = vec![ReasoningStep {
            step_number: 1,
            content: "   ".into(),
            confidence: 0.9,
            references_prior: vec![],
        }];
        let v = ChainValidator::new(vec![ChainValidationRule::StepsNonEmpty]);
        let result = v.validate(&steps);
        assert!(!result.passed);
    }
}
