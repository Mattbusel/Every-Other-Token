//! Chain-of-thought extraction and validation.
//!
//! Parses raw model output into a sequence of [`ReasoningStep`]s, classifies
//! each step by type, checks logical flow, detects unsupported jumps, and
//! validates any arithmetic present in the text.

use std::fmt;

// ---------------------------------------------------------------------------
// Step type
// ---------------------------------------------------------------------------

/// Semantic category of a single reasoning step.
#[derive(Debug, Clone, PartialEq)]
pub enum StepType {
    Observation,
    Hypothesis,
    Deduction,
    Induction,
    Calculation,
    Assumption,
    Conclusion,
    Verification,
}

impl fmt::Display for StepType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StepType::Observation  => write!(f, "Observation"),
            StepType::Hypothesis   => write!(f, "Hypothesis"),
            StepType::Deduction    => write!(f, "Deduction"),
            StepType::Induction    => write!(f, "Induction"),
            StepType::Calculation  => write!(f, "Calculation"),
            StepType::Assumption   => write!(f, "Assumption"),
            StepType::Conclusion   => write!(f, "Conclusion"),
            StepType::Verification => write!(f, "Verification"),
        }
    }
}

// ---------------------------------------------------------------------------
// Core data structures
// ---------------------------------------------------------------------------

/// A single labelled step in a chain of thought.
#[derive(Debug, Clone)]
pub struct ReasoningStep {
    pub step_number: usize,
    pub content: String,
    pub step_type: StepType,
    pub confidence: f64,
    pub references: Vec<String>,
}

/// A full chain of reasoning steps with an optional final answer and quality.
#[derive(Debug, Clone)]
pub struct ReasoningChain {
    pub steps: Vec<ReasoningStep>,
    pub final_answer: Option<String>,
    pub overall_quality: f64,
}

/// Validated analysis of a [`ReasoningChain`].
#[derive(Debug, Clone)]
pub struct TracerResult {
    pub chain: ReasoningChain,
    pub has_logical_flow: bool,
    pub has_unsupported_jumps: bool,
    pub calculation_errors: usize,
    pub quality_score: f64,
    pub suggestions: Vec<String>,
}

// ---------------------------------------------------------------------------
// Tracer implementation
// ---------------------------------------------------------------------------

/// Analyses free-form reasoning text.
#[derive(Debug, Default)]
pub struct ReasoningTracer;

impl ReasoningTracer {
    pub fn new() -> Self {
        ReasoningTracer
    }

    // -----------------------------------------------------------------------
    // Extraction
    // -----------------------------------------------------------------------

    /// Split `text` into raw step strings using common markers.
    pub fn extract_steps(text: &str) -> Vec<ReasoningStep> {
        let mut raw: Vec<String> = Vec::new();

        // Try numbered list: "1.", "1)", "Step 1:"
        let numbered_re = Self::split_numbered(text);
        if numbered_re.len() > 1 {
            raw = numbered_re;
        } else {
            // Try transition words
            raw = Self::split_transition_words(text);
        }

        if raw.is_empty() {
            raw = vec![text.trim().to_string()];
        }

        raw.into_iter()
            .filter(|s| !s.trim().is_empty())
            .enumerate()
            .map(|(i, content)| {
                let (step_type, confidence) = Self::classify_step(&content);
                let references = Self::extract_references(&content);
                ReasoningStep {
                    step_number: i + 1,
                    content,
                    step_type,
                    confidence,
                    references,
                }
            })
            .collect()
    }

    fn split_numbered(text: &str) -> Vec<String> {
        let mut parts: Vec<String> = Vec::new();
        let mut current = String::new();
        let mut step_count = 0usize;

        for line in text.lines() {
            let trimmed = line.trim();
            // Matches "1.", "1)", "Step 1:", "Step 1 -"
            let is_marker = {
                let digits_dot = trimmed.len() > 1
                    && trimmed.chars().next().map_or(false, |c| c.is_ascii_digit())
                    && (trimmed.chars().nth(1) == Some('.')
                        || trimmed.chars().nth(1) == Some(')'));
                let step_kw = trimmed.to_lowercase().starts_with("step ");
                digits_dot || step_kw
            };

            if is_marker {
                if !current.trim().is_empty() {
                    parts.push(current.trim().to_string());
                }
                current = trimmed.to_string();
                step_count += 1;
            } else {
                if !current.is_empty() {
                    current.push(' ');
                }
                current.push_str(trimmed);
            }
        }
        if !current.trim().is_empty() {
            parts.push(current.trim().to_string());
        }
        if step_count < 2 {
            return Vec::new();
        }
        parts
    }

    fn split_transition_words(text: &str) -> Vec<String> {
        // Split on "First,", "Then,", "Finally,", "Therefore,", "Thus,", "Hence,"
        let markers = [
            "First,", "First ", "Then,", "Then ", "Next,", "Next ",
            "Finally,", "Finally ", "Therefore,", "Therefore ", "Thus,",
            "Thus ", "Hence,", "Hence ", "Additionally,", "Moreover,",
            "Consequently,",
        ];

        let mut split_points: Vec<usize> = Vec::new();
        for marker in &markers {
            let lower_text = text.to_lowercase();
            let lower_marker = marker.to_lowercase();
            let mut start = 0;
            while let Some(pos) = lower_text[start..].find(&lower_marker) {
                let abs = start + pos;
                // Only split at start of sentence (beginning or after ". ")
                if abs == 0
                    || text[..abs].ends_with(". ")
                    || text[..abs].ends_with(".\n")
                    || text[..abs].ends_with('\n')
                {
                    split_points.push(abs);
                }
                start = abs + 1;
            }
        }

        split_points.sort();
        split_points.dedup();

        if split_points.len() < 2 {
            return Vec::new();
        }

        let mut parts = Vec::new();
        let mut prev = 0usize;
        for &sp in &split_points[1..] {
            parts.push(text[prev..sp].trim().to_string());
            prev = sp;
        }
        parts.push(text[prev..].trim().to_string());
        parts
    }

    fn extract_references(text: &str) -> Vec<String> {
        // Simple: look for "step N" / "point N" references
        let mut refs = Vec::new();
        let lower = text.to_lowercase();
        let markers = ["step ", "point ", "from above", "as stated", "as noted"];
        for m in &markers {
            if lower.contains(m) {
                refs.push(m.trim().to_string());
            }
        }
        refs
    }

    // -----------------------------------------------------------------------
    // Classification
    // -----------------------------------------------------------------------

    /// Keyword-based classification with a confidence estimate.
    pub fn classify_step(text: &str) -> (StepType, f64) {
        let lower = text.to_lowercase();

        let matches = |kws: &[&str]| -> usize {
            kws.iter().filter(|&&kw| lower.contains(kw)).count()
        };

        let conclusion_kws   = ["therefore", "thus", "hence", "in conclusion", "so we", "the answer", "finally,", "conclude"];
        let deduction_kws    = ["because", "since", "given that", "it follows", "must be", "implies", "deduces"];
        let hypothesis_kws   = ["might", "could be", "perhaps", "possibly", "assume", "suppose", "hypothesis", "conjecture"];
        let observation_kws  = ["observe", "notice", "see that", "note that", "we can see", "looking at", "inspection"];
        let calculation_kws  = ["calculate", "compute", "equals", "=", "+", "-", "×", "÷", "result is", "sum", "product"];
        let induction_kws    = ["pattern", "always", "every", "in general", "inductively", "by induction", "for all"];
        let assumption_kws   = ["assume", "given", "let us say", "suppose", "wlog", "without loss"];
        let verification_kws = ["verify", "check", "confirm", "proof", "validate", "indeed", "correct"];

        let scores: [(StepType, usize); 8] = [
            (StepType::Conclusion,   matches(&conclusion_kws)),
            (StepType::Deduction,    matches(&deduction_kws)),
            (StepType::Hypothesis,   matches(&hypothesis_kws)),
            (StepType::Observation,  matches(&observation_kws)),
            (StepType::Calculation,  matches(&calculation_kws)),
            (StepType::Induction,    matches(&induction_kws)),
            (StepType::Assumption,   matches(&assumption_kws)),
            (StepType::Verification, matches(&verification_kws)),
        ];

        let best = scores.iter().max_by_key(|(_, s)| s).unwrap();
        if best.1 == 0 {
            return (StepType::Observation, 0.4);
        }

        let total: usize = scores.iter().map(|(_, s)| s).sum();
        let confidence = (best.1 as f64 / total as f64).min(1.0).max(0.4);
        (best.0.clone(), confidence)
    }

    // -----------------------------------------------------------------------
    // Calculation detection & validation
    // -----------------------------------------------------------------------

    /// Returns true if the step text contains arithmetic operators or equations.
    pub fn detect_calculations(step: &str) -> bool {
        let ops = ['+', '-', '*', '/', '=', '×', '÷'];
        let has_op = step.chars().any(|c| ops.contains(&c));
        let has_num = step.chars().any(|c| c.is_ascii_digit());
        has_op && has_num
    }

    /// Attempts to validate simple "X op Y = Z" arithmetic. Returns the error
    /// magnitude if validation fails, `None` if no equation is parseable.
    pub fn validate_calculation(step: &str) -> Option<f64> {
        // Look for pattern: <number> <op> <number> = <number>
        // Very simple tokenizer
        let tokens: Vec<&str> = step.split_whitespace().collect();
        for window in tokens.windows(5) {
            if window.len() == 5 && window[3] == "=" {
                let a: f64 = window[0].trim_matches(|c: char| !c.is_ascii_digit() && c != '-' && c != '.').parse().ok()?;
                let op = window[1];
                let b: f64 = window[2].trim_matches(|c: char| !c.is_ascii_digit() && c != '-' && c != '.').parse().ok()?;
                let expected: f64 = window[4].trim_matches(|c: char| !c.is_ascii_digit() && c != '-' && c != '.').parse().ok()?;
                let computed = match op {
                    "+" => a + b,
                    "-" => a - b,
                    "*" | "×" => a * b,
                    "/" | "÷" => {
                        if b == 0.0 { return None; }
                        a / b
                    }
                    _ => return None,
                };
                return Some((computed - expected).abs());
            }
        }
        None
    }

    // -----------------------------------------------------------------------
    // Flow checking
    // -----------------------------------------------------------------------

    /// Returns true if the chain follows an Observation → … → Conclusion order.
    pub fn check_logical_flow(steps: &[ReasoningStep]) -> bool {
        if steps.is_empty() {
            return false;
        }
        // Conclusions should not appear before any hypothesis or deduction
        let mut seen_premise = false;
        let mut conclusion_before_premise = false;
        for step in steps {
            match step.step_type {
                StepType::Observation
                | StepType::Hypothesis
                | StepType::Assumption
                | StepType::Induction
                | StepType::Deduction => {
                    seen_premise = true;
                }
                StepType::Conclusion if !seen_premise => {
                    conclusion_before_premise = true;
                }
                _ => {}
            }
        }
        !conclusion_before_premise
    }

    /// Returns indices of conclusion steps that appear without sufficient prior premises.
    pub fn find_unsupported_jumps(steps: &[ReasoningStep]) -> Vec<usize> {
        let mut unsupported = Vec::new();
        let mut premise_count = 0usize;

        for (i, step) in steps.iter().enumerate() {
            match step.step_type {
                StepType::Observation
                | StepType::Hypothesis
                | StepType::Assumption
                | StepType::Deduction
                | StepType::Induction
                | StepType::Calculation
                | StepType::Verification => {
                    premise_count += 1;
                }
                StepType::Conclusion => {
                    if premise_count < 2 {
                        unsupported.push(i);
                    }
                    // A conclusion can itself serve as a partial premise for later
                    premise_count = premise_count.saturating_add(1);
                }
            }
        }
        unsupported
    }

    // -----------------------------------------------------------------------
    // Answer extraction
    // -----------------------------------------------------------------------

    /// Extracts the final answer sentence from common delimiters.
    pub fn extract_answer(text: &str) -> Option<String> {
        let lower = text.to_lowercase();
        let markers = [
            "therefore,", "therefore ", "the answer is", "answer:", "conclusion:",
            "in conclusion,", "thus,", "hence,",
        ];
        for marker in &markers {
            if let Some(pos) = lower.find(marker) {
                let tail = text[pos + marker.len()..].trim();
                if !tail.is_empty() {
                    // Take up to first period or 200 chars
                    let end = tail.find('.').map_or(tail.len().min(200), |p| p + 1);
                    return Some(tail[..end].trim().to_string());
                }
            }
        }
        None
    }

    // -----------------------------------------------------------------------
    // Main entry point
    // -----------------------------------------------------------------------

    /// Full analysis of a reasoning text.
    pub fn trace(&self, text: &str) -> TracerResult {
        let steps = Self::extract_steps(text);
        let final_answer = Self::extract_answer(text);

        let has_logical_flow = Self::check_logical_flow(&steps);
        let unsupported = Self::find_unsupported_jumps(&steps);
        let has_unsupported_jumps = !unsupported.is_empty();

        // Count calculation errors
        let mut calculation_errors = 0usize;
        for step in &steps {
            if Self::detect_calculations(&step.content) {
                if let Some(err) = Self::validate_calculation(&step.content) {
                    if err > 1e-6 {
                        calculation_errors += 1;
                    }
                }
            }
        }

        // Quality heuristics
        let mut quality_score = 1.0_f64;
        let mut suggestions: Vec<String> = Vec::new();

        if !has_logical_flow {
            quality_score -= 0.2;
            suggestions.push("Reorder steps so observations come before conclusions.".to_string());
        }
        if has_unsupported_jumps {
            quality_score -= 0.15 * unsupported.len() as f64;
            suggestions.push(format!(
                "Step(s) {:?} jump to a conclusion without sufficient premises.",
                unsupported
            ));
        }
        if calculation_errors > 0 {
            quality_score -= 0.1 * calculation_errors as f64;
            suggestions.push(format!("{} arithmetic error(s) detected.", calculation_errors));
        }
        if steps.len() < 2 {
            quality_score -= 0.1;
            suggestions.push("Add more intermediate steps for clarity.".to_string());
        }
        if final_answer.is_none() {
            quality_score -= 0.1;
            suggestions.push("Add an explicit conclusion or answer statement.".to_string());
        }

        let avg_confidence = if steps.is_empty() {
            0.5
        } else {
            steps.iter().map(|s| s.confidence).sum::<f64>() / steps.len() as f64
        };
        let overall_quality = (quality_score * avg_confidence).clamp(0.0, 1.0);

        let chain = ReasoningChain {
            steps,
            final_answer,
            overall_quality,
        };

        TracerResult {
            chain,
            has_logical_flow,
            has_unsupported_jumps,
            calculation_errors,
            quality_score: quality_score.clamp(0.0, 1.0),
            suggestions,
        }
    }

    // -----------------------------------------------------------------------
    // Chain comparison
    // -----------------------------------------------------------------------

    /// Returns a [0, 1] similarity score between two chains.
    pub fn compare_chains(a: &ReasoningChain, b: &ReasoningChain) -> f64 {
        // Jaccard on step types
        let types_a: std::collections::HashSet<String> =
            a.steps.iter().map(|s| s.step_type.to_string()).collect();
        let types_b: std::collections::HashSet<String> =
            b.steps.iter().map(|s| s.step_type.to_string()).collect();

        let intersection = types_a.intersection(&types_b).count();
        let union = types_a.union(&types_b).count();
        let type_sim = if union == 0 {
            1.0
        } else {
            intersection as f64 / union as f64
        };

        // Length similarity
        let len_a = a.steps.len() as f64;
        let len_b = b.steps.len() as f64;
        let len_sim = if len_a == 0.0 && len_b == 0.0 {
            1.0
        } else {
            1.0 - (len_a - len_b).abs() / (len_a + len_b).max(1.0)
        };

        // Quality similarity
        let qual_sim = 1.0 - (a.overall_quality - b.overall_quality).abs();

        (type_sim + len_sim + qual_sim) / 3.0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_conclusion() {
        let (st, _) = ReasoningTracer::classify_step("Therefore the answer is 42.");
        assert_eq!(st, StepType::Conclusion);
    }

    #[test]
    fn test_detect_calculation() {
        assert!(ReasoningTracer::detect_calculations("3 + 4 = 7"));
        assert!(!ReasoningTracer::detect_calculations("no math here"));
    }

    #[test]
    fn test_validate_calculation_correct() {
        let err = ReasoningTracer::validate_calculation("3 + 4 = 7");
        assert!(err.is_some());
        assert!(err.unwrap() < 1e-6);
    }

    #[test]
    fn test_extract_answer() {
        let text = "We consider the problem. Therefore, the answer is 42.";
        assert!(ReasoningTracer::extract_answer(text).is_some());
    }

    #[test]
    fn test_trace_basic() {
        let tracer = ReasoningTracer::new();
        let text = "Step 1: We observe that x = 2. Step 2: Since x = 2, x + 3 = 5. Step 3: Therefore the answer is 5.";
        let result = tracer.trace(text);
        assert!(!result.chain.steps.is_empty());
    }

    #[test]
    fn test_compare_chains() {
        let a = ReasoningChain { steps: vec![], final_answer: None, overall_quality: 0.8 };
        let b = ReasoningChain { steps: vec![], final_answer: None, overall_quality: 0.8 };
        let sim = ReasoningTracer::compare_chains(&a, &b);
        assert!(sim >= 0.0 && sim <= 1.0);
    }
}
