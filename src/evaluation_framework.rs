//! LLM output evaluation metrics.
//!
//! Provides a suite of automatic evaluation metrics for scoring LLM responses
//! across dimensions such as groundedness, relevance, faithfulness, coherence,
//! toxicity, and bias.

use std::fmt;

// ---------------------------------------------------------------------------
// EvaluationMetric
// ---------------------------------------------------------------------------

/// The dimension being scored.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EvaluationMetric {
    Groundedness,
    Relevance,
    Faithfulness,
    Completeness,
    Coherence,
    Conciseness,
    Toxicity,
    Bias,
}

impl fmt::Display for EvaluationMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvaluationMetric::Groundedness => write!(f, "Groundedness"),
            EvaluationMetric::Relevance => write!(f, "Relevance"),
            EvaluationMetric::Faithfulness => write!(f, "Faithfulness"),
            EvaluationMetric::Completeness => write!(f, "Completeness"),
            EvaluationMetric::Coherence => write!(f, "Coherence"),
            EvaluationMetric::Conciseness => write!(f, "Conciseness"),
            EvaluationMetric::Toxicity => write!(f, "Toxicity"),
            EvaluationMetric::Bias => write!(f, "Bias"),
        }
    }
}

// ---------------------------------------------------------------------------
// MetricScore
// ---------------------------------------------------------------------------

/// Score for a single evaluation dimension.
#[derive(Debug, Clone)]
pub struct MetricScore {
    pub metric: EvaluationMetric,
    /// 0.0 (worst) – 1.0 (best).
    pub score: f64,
    pub explanation: String,
    /// Evaluator confidence in this score (0.0 – 1.0).
    pub confidence: f64,
}

impl MetricScore {
    pub fn new(metric: EvaluationMetric, score: f64, explanation: impl Into<String>, confidence: f64) -> Self {
        Self {
            metric,
            score: score.clamp(0.0, 1.0),
            explanation: explanation.into(),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

// ---------------------------------------------------------------------------
// EvaluationResult
// ---------------------------------------------------------------------------

/// Aggregated result of evaluating a single response.
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    pub overall_score: f64,
    pub metrics: Vec<MetricScore>,
    pub passed: bool,
    pub threshold: f64,
    pub feedback: Vec<String>,
}

impl EvaluationResult {
    /// Format a human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "Overall: {:.2} | Passed: {} | Threshold: {:.2}",
            self.overall_score, self.passed, self.threshold
        )
    }
}

// ---------------------------------------------------------------------------
// EvaluationCriteria
// ---------------------------------------------------------------------------

/// Configuration that drives an evaluation run.
#[derive(Debug, Clone)]
pub struct EvaluationCriteria {
    /// Which metrics to compute.
    pub metrics: Vec<EvaluationMetric>,
    /// Minimum average score to pass.
    pub min_threshold: f64,
    /// Any individual metric listed here will fail the response if its score
    /// falls below 0.5.
    pub fail_on: Vec<EvaluationMetric>,
}

impl Default for EvaluationCriteria {
    fn default() -> Self {
        Self {
            metrics: vec![
                EvaluationMetric::Groundedness,
                EvaluationMetric::Relevance,
                EvaluationMetric::Faithfulness,
                EvaluationMetric::Completeness,
                EvaluationMetric::Coherence,
                EvaluationMetric::Conciseness,
                EvaluationMetric::Toxicity,
                EvaluationMetric::Bias,
            ],
            min_threshold: 0.6,
            fail_on: vec![EvaluationMetric::Toxicity],
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| w.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| !w.is_empty())
        .collect()
}

fn word_set(text: &str) -> std::collections::HashSet<String> {
    tokenize(text).into_iter().collect()
}

fn jaccard(a: &std::collections::HashSet<String>, b: &std::collections::HashSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let intersection = a.intersection(b).count() as f64;
    let union = a.union(b).count() as f64;
    if union == 0.0 { 0.0 } else { intersection / union }
}

fn sentences(text: &str) -> Vec<&str> {
    text.split(|c| c == '.' || c == '!' || c == '?')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect()
}

// ---------------------------------------------------------------------------
// LlmEvaluator
// ---------------------------------------------------------------------------

/// Heuristic evaluator for LLM responses.
///
/// All methods are pure functions operating on strings — no external API calls
/// are required.
#[derive(Debug, Default)]
pub struct LlmEvaluator;

impl LlmEvaluator {
    pub fn new() -> Self {
        Self
    }

    // -----------------------------------------------------------------------
    // Individual metric methods
    // -----------------------------------------------------------------------

    /// Score how well the *response* is supported by *context* (keyword overlap
    /// plus a basic contradiction check).
    pub fn evaluate_groundedness(&self, response: &str, context: &str) -> MetricScore {
        let resp_set = word_set(response);
        let ctx_set = word_set(context);

        let overlap = jaccard(&resp_set, &ctx_set);

        // Simple contradiction heuristic: "not", "never", "no" near a context keyword
        let contradiction_words = ["not", "never", "no", "false", "incorrect", "wrong"];
        let contradiction_count = tokenize(response)
            .iter()
            .filter(|w| contradiction_words.contains(&w.as_str()))
            .count();
        let contradiction_penalty = (contradiction_count as f64 * 0.05).min(0.3);

        let score = (overlap - contradiction_penalty).clamp(0.0, 1.0);
        let explanation = format!(
            "Keyword overlap: {:.2}; contradiction penalty: {:.2}",
            overlap, contradiction_penalty
        );
        MetricScore::new(EvaluationMetric::Groundedness, score, explanation, 0.7)
    }

    /// Score query-response topical similarity.
    pub fn evaluate_relevance(&self, response: &str, query: &str) -> MetricScore {
        let resp_set = word_set(response);
        let query_set = word_set(query);
        let score = jaccard(&resp_set, &query_set);
        let explanation = format!("Query-response Jaccard similarity: {:.2}", score);
        MetricScore::new(EvaluationMetric::Relevance, score, explanation, 0.75)
    }

    /// Score how many claims in *response* can be verified against *source_docs*.
    pub fn evaluate_faithfulness(&self, response: &str, source_docs: &[&str]) -> MetricScore {
        if source_docs.is_empty() {
            return MetricScore::new(
                EvaluationMetric::Faithfulness,
                0.5,
                "No source documents provided; unable to verify claims.".to_string(),
                0.3,
            );
        }

        let resp_sentences = sentences(response);
        if resp_sentences.is_empty() {
            return MetricScore::new(EvaluationMetric::Faithfulness, 0.5, "Empty response.".to_string(), 0.5);
        }

        let all_source_tokens: std::collections::HashSet<String> = source_docs
            .iter()
            .flat_map(|d| word_set(d))
            .collect();

        let verified = resp_sentences.iter().filter(|s| {
            let s_set = word_set(s);
            let overlap = s_set.intersection(&all_source_tokens).count() as f64;
            overlap / (s_set.len().max(1) as f64) > 0.3
        }).count();

        let score = verified as f64 / resp_sentences.len() as f64;
        let explanation = format!(
            "{}/{} sentences have sufficient overlap with sources.",
            verified,
            resp_sentences.len()
        );
        MetricScore::new(EvaluationMetric::Faithfulness, score, explanation, 0.65)
    }

    /// Score how completely the response addresses the query's questions.
    pub fn evaluate_completeness(&self, response: &str, query: &str) -> MetricScore {
        // Count question words in the query and check coverage
        let question_words = ["what", "how", "why", "when", "where", "who", "which", "does", "is", "are", "can"];
        let query_tokens = tokenize(query);
        let resp_tokens = tokenize(response);

        let query_q_count = query_tokens.iter().filter(|t| question_words.contains(&t.as_str())).count();
        let query_keywords: std::collections::HashSet<_> = query_tokens.iter()
            .filter(|t| t.len() > 3 && !question_words.contains(&t.as_str()))
            .cloned()
            .collect();
        let resp_set: std::collections::HashSet<_> = resp_tokens.iter().cloned().collect();

        let keyword_coverage = if query_keywords.is_empty() {
            0.7
        } else {
            query_keywords.intersection(&resp_set).count() as f64 / query_keywords.len() as f64
        };

        // Bonus for length relative to query complexity
        let length_bonus = (resp_tokens.len() as f64 / (query_q_count as f64 * 30.0 + 1.0)).min(0.3);
        let score = (keyword_coverage * 0.7 + length_bonus).clamp(0.0, 1.0);

        let explanation = format!(
            "Keyword coverage: {:.2}; length bonus: {:.2}",
            keyword_coverage, length_bonus
        );
        MetricScore::new(EvaluationMetric::Completeness, score, explanation, 0.65)
    }

    /// Score logical flow and transition quality across sentences.
    pub fn evaluate_coherence(&self, response: &str) -> MetricScore {
        let sents = sentences(response);
        if sents.len() < 2 {
            return MetricScore::new(
                EvaluationMetric::Coherence,
                if sents.is_empty() { 0.0 } else { 0.7 },
                "Too few sentences to assess coherence.".to_string(),
                0.5,
            );
        }

        // Transition words that indicate logical flow
        let transition_words = [
            "however", "therefore", "thus", "furthermore", "additionally",
            "moreover", "consequently", "nevertheless", "alternatively", "finally",
            "first", "second", "third", "next", "then", "also", "because", "since",
        ];

        let transition_count = sents.iter().skip(1).filter(|s| {
            let first_tokens: Vec<_> = tokenize(s).into_iter().take(5).collect();
            first_tokens.iter().any(|t| transition_words.contains(&t.as_str()))
        }).count();

        let transition_ratio = transition_count as f64 / (sents.len() - 1) as f64;

        // Consecutive sentence overlap (shared vocabulary indicates topical continuity)
        let overlap_scores: Vec<f64> = sents.windows(2).map(|pair| {
            let a = word_set(pair[0]);
            let b = word_set(pair[1]);
            jaccard(&a, &b)
        }).collect();

        let avg_overlap = if overlap_scores.is_empty() {
            0.0
        } else {
            overlap_scores.iter().sum::<f64>() / overlap_scores.len() as f64
        };

        let score = (transition_ratio * 0.4 + avg_overlap * 0.6).clamp(0.0, 1.0);
        let explanation = format!(
            "Transition ratio: {:.2}; avg consecutive overlap: {:.2}",
            transition_ratio, avg_overlap
        );
        MetricScore::new(EvaluationMetric::Coherence, score, explanation, 0.6)
    }

    /// Score conciseness — penalises repetition and filler phrases.
    pub fn evaluate_conciseness(&self, response: &str) -> MetricScore {
        let tokens = tokenize(response);
        if tokens.is_empty() {
            return MetricScore::new(EvaluationMetric::Conciseness, 0.0, "Empty response.".to_string(), 1.0);
        }

        let filler_phrases = [
            "basically", "essentially", "in other words", "as i mentioned",
            "it is important to note that", "it should be noted", "needless to say",
            "at the end of the day", "going forward", "in terms of", "very", "really",
            "just", "literally", "absolutely", "definitely", "certainly", "obviously",
        ];

        let response_lower = response.to_lowercase();
        let filler_count = filler_phrases.iter().filter(|p| response_lower.contains(*p)).count();

        // Type-token ratio (TTR) — higher = less repetitive
        let unique: std::collections::HashSet<_> = tokens.iter().collect();
        let ttr = unique.len() as f64 / tokens.len() as f64;

        let filler_penalty = (filler_count as f64 * 0.05).min(0.4);
        let score = (ttr - filler_penalty).clamp(0.0, 1.0);

        let explanation = format!(
            "Type-token ratio: {:.2}; filler penalty: {:.2} ({} fillers)",
            ttr, filler_penalty, filler_count
        );
        MetricScore::new(EvaluationMetric::Conciseness, score, explanation, 0.7)
    }

    /// Detect harmful or toxic content.
    pub fn evaluate_toxicity(&self, response: &str) -> MetricScore {
        // A simple keyword-based toxicity signal.  In production this would
        // call a dedicated classifier.
        let toxic_patterns = [
            "hate", "kill", "murder", "violence", "threat", "abuse", "harass",
            "racist", "sexist", "offensive", "slur", "derogatory", "harmful",
            "dangerous", "illegal", "exploit",
        ];

        let response_lower = response.to_lowercase();
        let hit_count = toxic_patterns.iter().filter(|p| response_lower.contains(*p)).count();

        // Score is inverted: 1.0 = no toxicity detected
        let raw_penalty = (hit_count as f64 * 0.15).min(1.0);
        let score = 1.0 - raw_penalty;

        let explanation = if hit_count == 0 {
            "No toxic patterns detected.".to_string()
        } else {
            format!("{} potentially toxic term(s) detected.", hit_count)
        };
        MetricScore::new(EvaluationMetric::Toxicity, score, explanation, 0.6)
    }

    /// Detect one-sided or biased reasoning.
    pub fn evaluate_bias(&self, response: &str) -> MetricScore {
        // Heuristic: responses that present multiple perspectives score higher.
        let balance_indicators = [
            "on the other hand", "however", "alternatively", "some argue",
            "others believe", "critics", "proponents", "both", "while",
            "although", "despite", "conversely", "in contrast", "nevertheless",
        ];

        let response_lower = response.to_lowercase();
        let balance_count = balance_indicators.iter().filter(|p| response_lower.contains(*p)).count();

        // Loaded language check
        let loaded_words = [
            "always", "never", "everyone", "nobody", "obviously", "clearly",
            "undeniably", "irrefutably", "must", "only",
        ];
        let loaded_count = tokenize(response)
            .iter()
            .filter(|t| loaded_words.contains(&t.as_str()))
            .count();

        let balance_score = (balance_count as f64 * 0.2).min(0.6);
        let loaded_penalty = (loaded_count as f64 * 0.05).min(0.4);
        let score = (0.5 + balance_score - loaded_penalty).clamp(0.0, 1.0);

        let explanation = format!(
            "Balance indicators: {}; loaded language count: {}",
            balance_count, loaded_count
        );
        MetricScore::new(EvaluationMetric::Bias, score, explanation, 0.55)
    }

    // -----------------------------------------------------------------------
    // Aggregate evaluation
    // -----------------------------------------------------------------------

    /// Run all metrics specified in *criteria* and return an [`EvaluationResult`].
    pub fn evaluate(
        &self,
        response: &str,
        query: &str,
        context: &str,
        criteria: &EvaluationCriteria,
    ) -> EvaluationResult {
        let source_docs = if context.is_empty() { vec![] } else { vec![context] };

        let metric_scores: Vec<MetricScore> = criteria
            .metrics
            .iter()
            .map(|m| match m {
                EvaluationMetric::Groundedness => self.evaluate_groundedness(response, context),
                EvaluationMetric::Relevance => self.evaluate_relevance(response, query),
                EvaluationMetric::Faithfulness => {
                    self.evaluate_faithfulness(response, &source_docs.iter().map(|s: &&str| *s).collect::<Vec<_>>())
                }
                EvaluationMetric::Completeness => self.evaluate_completeness(response, query),
                EvaluationMetric::Coherence => self.evaluate_coherence(response),
                EvaluationMetric::Conciseness => self.evaluate_conciseness(response),
                EvaluationMetric::Toxicity => self.evaluate_toxicity(response),
                EvaluationMetric::Bias => self.evaluate_bias(response),
            })
            .collect();

        let overall_score = if metric_scores.is_empty() {
            0.0
        } else {
            metric_scores.iter().map(|s| s.score).sum::<f64>() / metric_scores.len() as f64
        };

        let mut feedback = Vec::new();
        let mut failed_on_criterion = false;

        for ms in &metric_scores {
            if criteria.fail_on.contains(&ms.metric) && ms.score < 0.5 {
                failed_on_criterion = true;
                feedback.push(format!(
                    "FAIL: {} scored {:.2} (below 0.5 threshold)",
                    ms.metric, ms.score
                ));
            } else if ms.score < criteria.min_threshold {
                feedback.push(format!(
                    "WARN: {} scored {:.2} (below min threshold {:.2})",
                    ms.metric, ms.score, criteria.min_threshold
                ));
            }
        }

        let passed = !failed_on_criterion && overall_score >= criteria.min_threshold;
        if passed && feedback.is_empty() {
            feedback.push("All metrics within acceptable ranges.".to_string());
        }

        EvaluationResult {
            overall_score,
            metrics: metric_scores,
            passed,
            threshold: criteria.min_threshold,
            feedback,
        }
    }

    /// Evaluate multiple (response, query, context) triples.
    pub fn batch_evaluate(
        &self,
        pairs: &[(&str, &str, &str)],
        criteria: &EvaluationCriteria,
    ) -> Vec<EvaluationResult> {
        pairs
            .iter()
            .map(|(response, query, context)| self.evaluate(response, query, context, criteria))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_groundedness_high_overlap() {
        let evaluator = LlmEvaluator::new();
        let context = "The Eiffel Tower is located in Paris, France.";
        let response = "The Eiffel Tower stands in Paris, France.";
        let score = evaluator.evaluate_groundedness(response, context);
        assert!(score.score > 0.3, "Expected reasonable groundedness score");
    }

    #[test]
    fn test_toxicity_clean() {
        let evaluator = LlmEvaluator::new();
        let score = evaluator.evaluate_toxicity("The weather is nice today.");
        assert!(score.score > 0.9);
    }

    #[test]
    fn test_full_evaluate_passes() {
        let evaluator = LlmEvaluator::new();
        let criteria = EvaluationCriteria {
            metrics: vec![EvaluationMetric::Relevance, EvaluationMetric::Toxicity],
            min_threshold: 0.3,
            fail_on: vec![EvaluationMetric::Toxicity],
        };
        let result = evaluator.evaluate(
            "Paris is the capital of France.",
            "What is the capital of France?",
            "France is a country in Europe. Its capital is Paris.",
            &criteria,
        );
        assert!(result.passed);
    }

    #[test]
    fn test_batch_evaluate() {
        let evaluator = LlmEvaluator::new();
        let criteria = EvaluationCriteria::default();
        let pairs = vec![
            ("Paris is the capital of France.", "What is the capital of France?", "France capital is Paris."),
            ("The sky is blue.", "Why is the sky blue?", "Light scattering makes the sky appear blue."),
        ];
        let results = evaluator.batch_evaluate(&pairs, &criteria);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_metric_display() {
        assert_eq!(EvaluationMetric::Groundedness.to_string(), "Groundedness");
        assert_eq!(EvaluationMetric::Toxicity.to_string(), "Toxicity");
    }
}
