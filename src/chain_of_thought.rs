//! # Chain-of-Thought
//!
//! CoT prompt construction, self-consistency aggregation, and reasoning
//! quality analysis.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// CoTStrategy
// ---------------------------------------------------------------------------

/// Strategy used to frame a chain-of-thought prompt.
#[derive(Debug, Clone)]
pub enum CoTStrategy {
    /// Ask the model to reason step-by-step without giving examples.
    ZeroShot,
    /// Provide worked examples before the task.
    FewShot {
        /// (input, reasoning_output) example pairs.
        examples: Vec<(String, String)>,
    },
    /// Explore multiple reasoning branches in parallel.
    TreeOfThought {
        /// Number of parallel branches to explore.
        branches: u32,
    },
    /// Sample multiple completions and pick the majority answer.
    SelfConsistency {
        /// Number of samples to take.
        samples: u32,
    },
}

// ---------------------------------------------------------------------------
// CoTStep
// ---------------------------------------------------------------------------

/// A single parsed reasoning step extracted from a model completion.
#[derive(Debug, Clone)]
pub struct CoTStep {
    /// Ordinal position of this step (1-based).
    pub index: u32,
    /// Short description / heading for this step.
    pub description: String,
    /// Full reasoning text for this step.
    pub reasoning: String,
    /// Estimated confidence (0.0–1.0) inferred from language cues.
    pub confidence: f64,
}

// ---------------------------------------------------------------------------
// parse_cot_steps
// ---------------------------------------------------------------------------

/// Parse reasoning steps from a model completion.
///
/// Recognises:
/// * `Step N:` / `Step N.` patterns
/// * Numbered lists `1.` / `1)`
/// * Discourse markers: "First", "Second", "Third", "Finally"
pub fn parse_cot_steps(text: &str) -> Vec<CoTStep> {
    let mut steps: Vec<CoTStep> = Vec::new();
    let mut index = 1u32;

    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();

        // Match "Step N:" or "Step N."
        let step_match = line
            .strip_prefix("Step ")
            .and_then(|rest| {
                let end = rest.find(|c: char| !c.is_ascii_digit())?;
                let _n: u32 = rest[..end].parse().ok()?;
                let after = rest[end..].trim_start_matches(':').trim_start_matches('.').trim();
                Some(after.to_string())
            });

        // Match numbered list "1." or "1)"
        let numbered_match = if step_match.is_none() {
            let dot_pos = line.find(|c: char| c == '.' || c == ')');
            dot_pos.and_then(|pos| {
                let num_part = &line[..pos];
                if num_part.chars().all(|c| c.is_ascii_digit()) && !num_part.is_empty() {
                    Some(line[pos + 1..].trim().to_string())
                } else {
                    None
                }
            })
        } else {
            None
        };

        // Match discourse markers
        const MARKERS: &[&str] = &["First,", "First ", "Second,", "Second ", "Third,", "Third ",
            "Fourth,", "Finally,", "Finally ", "Lastly,", "Lastly "];
        let marker_match = if step_match.is_none() && numbered_match.is_none() {
            MARKERS.iter().find_map(|&m| {
                if line.starts_with(m) {
                    Some(line[m.len()..].trim().to_string())
                } else {
                    None
                }
            })
        } else {
            None
        };

        if let Some(desc) = step_match.or(numbered_match).or(marker_match) {
            // Gather continuation lines.
            let mut reasoning = desc.clone();
            let mut j = i + 1;
            while j < lines.len() {
                let next = lines[j].trim();
                if next.is_empty() { break; }
                // Stop if the next line looks like another step heading.
                if next.starts_with("Step ") || next.starts_with("First") || next.starts_with("Second")
                    || next.starts_with("Finally") || is_numbered_list_item(next)
                {
                    break;
                }
                reasoning.push(' ');
                reasoning.push_str(next);
                j += 1;
            }
            // Estimate confidence from hedging language.
            let confidence = estimate_confidence(&reasoning);
            steps.push(CoTStep {
                index,
                description: desc,
                reasoning,
                confidence,
            });
            index += 1;
            i = j;
            continue;
        }

        i += 1;
    }

    steps
}

fn is_numbered_list_item(s: &str) -> bool {
    let end = s.find(|c: char| c == '.' || c == ')').unwrap_or(0);
    if end == 0 { return false; }
    s[..end].chars().all(|c| c.is_ascii_digit())
}

fn estimate_confidence(text: &str) -> f64 {
    let lower = text.to_lowercase();
    let hedges = ["might", "could", "perhaps", "maybe", "possibly", "uncertain", "not sure", "i think"];
    let strong = ["therefore", "clearly", "obviously", "definitely", "certainly", "must", "always"];
    let hedge_count = hedges.iter().filter(|&&h| lower.contains(h)).count();
    let strong_count = strong.iter().filter(|&&s| lower.contains(s)).count();
    let base = 0.7_f64;
    let adjusted = base - hedge_count as f64 * 0.1 + strong_count as f64 * 0.05;
    adjusted.clamp(0.1, 1.0)
}

// ---------------------------------------------------------------------------
// extract_final_answer
// ---------------------------------------------------------------------------

/// Extract the final answer from a completion that uses CoT framing.
///
/// Looks for (in order):
/// 1. "Therefore:" / "Therefore,"
/// 2. "Answer:"
/// 3. "= " followed by a value
/// 4. The last non-empty sentence as a fallback
pub fn extract_final_answer(text: &str) -> Option<String> {
    const MARKERS: &[&str] = &["Therefore:", "Therefore,", "Answer:", "The answer is:", "Thus,", "So,"];
    for marker in MARKERS {
        if let Some(pos) = text.find(marker) {
            let after = text[pos + marker.len()..].trim();
            if !after.is_empty() {
                return Some(after.lines().next().unwrap_or(after).trim().to_string());
            }
        }
    }

    // Look for "= <value>" pattern (arithmetic).
    for line in text.lines().rev() {
        if let Some(eq_pos) = line.rfind('=') {
            let value = line[eq_pos + 1..].trim();
            if !value.is_empty() && value.chars().any(|c| c.is_alphanumeric()) {
                return Some(value.to_string());
            }
        }
    }

    // Fallback: last non-empty sentence.
    text.split(['.', '!', '?'])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .last()
        .map(|s| s.to_string())
}

// ---------------------------------------------------------------------------
// CoTPromptBuilder
// ---------------------------------------------------------------------------

/// Constructs a full chain-of-thought prompt from task description and strategy.
#[derive(Debug, Clone)]
pub struct CoTPromptBuilder {
    /// The strategy to use.
    pub strategy: CoTStrategy,
    /// The task or question to answer.
    pub task_description: String,
    /// Optional hint for the number of reasoning steps to use.
    pub steps_hint: Option<u32>,
}

impl CoTPromptBuilder {
    /// Create a builder.
    pub fn new(strategy: CoTStrategy, task_description: impl Into<String>) -> Self {
        Self {
            strategy,
            task_description: task_description.into(),
            steps_hint: None,
        }
    }

    /// Set the steps hint.
    pub fn with_steps_hint(mut self, n: u32) -> Self {
        self.steps_hint = Some(n);
        self
    }

    /// Build the full prompt string.
    pub fn build_prompt(&self) -> String {
        self.build_with_context("")
    }

    /// Build the prompt, optionally prepending extra `context`.
    pub fn build_with_context(&self, context: &str) -> String {
        let mut parts: Vec<String> = Vec::new();

        if !context.is_empty() {
            parts.push(format!("Context:\n{context}\n"));
        }

        match &self.strategy {
            CoTStrategy::ZeroShot => {
                parts.push("Let's think step by step.\n".to_string());
                if let Some(n) = self.steps_hint {
                    parts.push(format!("Use exactly {n} steps.\n"));
                }
                parts.push(self.task_description.clone());
            }

            CoTStrategy::FewShot { examples } => {
                for (i, (inp, out)) in examples.iter().enumerate() {
                    parts.push(format!("Example {}:\nInput: {}\nReasoning: {}\n", i + 1, inp, out));
                }
                parts.push("Now solve the following step by step:".to_string());
                parts.push(self.task_description.clone());
            }

            CoTStrategy::TreeOfThought { branches } => {
                parts.push(format!(
                    "Explore {branches} different reasoning branches for the following task. \
                     After exploring each branch, select the best answer.\n"
                ));
                for b in 1..=*branches {
                    parts.push(format!("Branch {b}: [explore here]"));
                }
                parts.push(self.task_description.clone());
            }

            CoTStrategy::SelfConsistency { samples } => {
                parts.push(format!(
                    "You will answer the following question {samples} times independently, \
                     then report the most consistent answer.\n"
                ));
                parts.push(self.task_description.clone());
            }
        }

        parts.push("Therefore, the answer is:".to_string());
        parts.join("\n")
    }
}

// ---------------------------------------------------------------------------
// SelfConsistencyAggregator
// ---------------------------------------------------------------------------

/// Collects multiple sampled answers and determines the majority.
#[derive(Debug, Default)]
pub struct SelfConsistencyAggregator {
    answers: Vec<String>,
}

impl SelfConsistencyAggregator {
    /// Create an empty aggregator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one sampled answer.
    pub fn add_answer(&mut self, answer: String) {
        self.answers.push(answer);
    }

    /// Normalise an answer string for comparison.
    fn normalise(s: &str) -> String {
        s.trim().to_lowercase()
    }

    /// The most frequently occurring normalised answer, or `None` if empty.
    pub fn majority_answer(&self) -> Option<String> {
        if self.answers.is_empty() {
            return None;
        }
        let mut counts: HashMap<String, usize> = HashMap::new();
        for a in &self.answers {
            *counts.entry(Self::normalise(a)).or_insert(0) += 1;
        }
        counts
            .into_iter()
            .max_by_key(|&(_, c)| c)
            .map(|(k, _)| k)
    }

    /// Fraction of answers that match the majority (0.0–1.0).
    pub fn confidence(&self) -> f64 {
        if self.answers.is_empty() {
            return 0.0;
        }
        let majority = match self.majority_answer() {
            Some(m) => m,
            None => return 0.0,
        };
        let count = self
            .answers
            .iter()
            .filter(|a| Self::normalise(a) == majority)
            .count();
        count as f64 / self.answers.len() as f64
    }

    /// Returns `true` when every answer (after normalisation) is identical.
    pub fn all_agree(&self) -> bool {
        if self.answers.is_empty() {
            return true;
        }
        let first = Self::normalise(&self.answers[0]);
        self.answers.iter().all(|a| Self::normalise(a) == first)
    }
}

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// CoTAnalyzer
// ---------------------------------------------------------------------------

/// Analyses a model completion for reasoning quality signals.
pub struct CoTAnalyzer;

impl CoTAnalyzer {
    /// Count the number of recognisable reasoning steps.
    pub fn count_reasoning_steps(text: &str) -> u32 {
        parse_cot_steps(text).len() as u32
    }

    /// Returns `true` if the text contains arithmetic operators or numerals
    /// alongside operators.
    pub fn has_arithmetic(text: &str) -> bool {
        const OPS: &[char] = &['+', '-', '×', '÷', '=', '*', '/'];
        let has_op = text.chars().any(|c| OPS.contains(&c));
        let has_num = text.chars().any(|c| c.is_ascii_digit());
        has_op && has_num
    }

    /// Score reasoning depth on a 0.0–1.0 scale based on:
    /// * Number of steps (up to 10)
    /// * Presence of transition words
    /// * Presence of arithmetic
    pub fn reasoning_depth_score(text: &str) -> f64 {
        let steps = parse_cot_steps(text).len();
        let step_score = (steps as f64 / 10.0).min(1.0);

        const TRANSITIONS: &[&str] = &[
            "therefore", "thus", "because", "since", "consequently",
            "however", "furthermore", "additionally", "first", "finally",
            "as a result", "in conclusion",
        ];
        let lower = text.to_lowercase();
        let transition_count = TRANSITIONS.iter().filter(|&&t| lower.contains(t)).count();
        let transition_score = (transition_count as f64 / 5.0).min(1.0);

        let arithmetic_score = if Self::has_arithmetic(text) { 0.2 } else { 0.0 };

        ((step_score * 0.5 + transition_score * 0.3 + arithmetic_score) * (1.0 / 0.8)).min(1.0)
    }

    /// Identify potential logical gaps in a sequence of CoT steps.
    ///
    /// Gaps are flagged when:
    /// * A step has confidence below 0.4.
    /// * There is a confidence drop > 0.3 between consecutive steps.
    pub fn identify_logical_gaps(steps: &[CoTStep]) -> Vec<String> {
        let mut gaps = Vec::new();
        for step in steps {
            if step.confidence < 0.4 {
                gaps.push(format!(
                    "Step {} has low confidence ({:.2}): \"{}\"",
                    step.index, step.confidence, step.description
                ));
            }
        }
        for window in steps.windows(2) {
            let prev = &window[0];
            let next = &window[1];
            if prev.confidence - next.confidence > 0.3 {
                gaps.push(format!(
                    "Confidence drops sharply between step {} ({:.2}) and step {} ({:.2})",
                    prev.index, prev.confidence, next.index, next.confidence
                ));
            }
        }
        gaps
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cot_steps_step_prefix() {
        let text = "Step 1: Identify the variables.\nStep 2: Set up the equation.\nStep 3: Solve.";
        let steps = parse_cot_steps(text);
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0].index, 1);
    }

    #[test]
    fn parse_cot_steps_numbered_list() {
        let text = "1. Understand the problem.\n2. Break it down.\n3. Verify.";
        let steps = parse_cot_steps(text);
        assert_eq!(steps.len(), 3);
    }

    #[test]
    fn extract_final_answer_therefore() {
        let text = "We need to add. Therefore: 42";
        let ans = extract_final_answer(text);
        assert_eq!(ans.as_deref(), Some("42"));
    }

    #[test]
    fn extract_final_answer_equation() {
        let text = "x + 1 = 5\nSo x = 4";
        let ans = extract_final_answer(text);
        assert!(ans.is_some());
    }

    #[test]
    fn prompt_builder_zero_shot() {
        let b = CoTPromptBuilder::new(CoTStrategy::ZeroShot, "What is 2+2?");
        let p = b.build_prompt();
        assert!(p.contains("step by step"));
        assert!(p.contains("What is 2+2?"));
    }

    #[test]
    fn self_consistency_majority() {
        let mut agg = SelfConsistencyAggregator::new();
        agg.add_answer("42".to_string());
        agg.add_answer("42".to_string());
        agg.add_answer("41".to_string());
        assert_eq!(agg.majority_answer().as_deref(), Some("42"));
        assert!(agg.confidence() > 0.6);
    }

    #[test]
    fn self_consistency_all_agree() {
        let mut agg = SelfConsistencyAggregator::new();
        agg.add_answer("yes".to_string());
        agg.add_answer("YES".to_string());
        assert!(agg.all_agree());
    }

    #[test]
    fn analyzer_has_arithmetic() {
        assert!(CoTAnalyzer::has_arithmetic("2 + 2 = 4"));
        assert!(!CoTAnalyzer::has_arithmetic("no numbers here"));
    }

    #[test]
    fn analyzer_depth_score_zero_for_empty() {
        let score = CoTAnalyzer::reasoning_depth_score("");
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn gap_detection_low_confidence() {
        let steps = vec![
            CoTStep { index: 1, description: "maybe".to_string(), reasoning: "maybe it is".to_string(), confidence: 0.3 },
        ];
        let gaps = CoTAnalyzer::identify_logical_gaps(&steps);
        assert!(!gaps.is_empty());
    }
}
