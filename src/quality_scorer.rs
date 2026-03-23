//! Multi-dimensional response quality scorer.
//!
//! Scores LLM responses along five dimensions — completeness, coherence,
//! conciseness, accuracy (stub), and relevance — and aggregates them into a
//! single weighted overall score plus a human-readable report.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// QualityDimension
// ---------------------------------------------------------------------------

/// The five quality dimensions evaluated by `QualityScorer`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum QualityDimension {
    Completeness,
    Coherence,
    Conciseness,
    Accuracy,
    Relevance,
}

impl QualityDimension {
    pub fn label(&self) -> &'static str {
        match self {
            QualityDimension::Completeness => "Completeness",
            QualityDimension::Coherence => "Coherence",
            QualityDimension::Conciseness => "Conciseness",
            QualityDimension::Accuracy => "Accuracy",
            QualityDimension::Relevance => "Relevance",
        }
    }

    /// Weight used when computing the overall score.
    fn weight(&self) -> f64 {
        match self {
            QualityDimension::Completeness => 0.25,
            QualityDimension::Coherence => 0.20,
            QualityDimension::Conciseness => 0.15,
            QualityDimension::Accuracy => 0.15,
            QualityDimension::Relevance => 0.25,
        }
    }
}

// ---------------------------------------------------------------------------
// DimensionScore
// ---------------------------------------------------------------------------

/// A scored result for a single quality dimension.
#[derive(Debug, Clone)]
pub struct DimensionScore {
    pub dimension: QualityDimension,
    /// Score in [0.0, 1.0].
    pub score: f64,
    pub explanation: String,
}

// ---------------------------------------------------------------------------
// QualityReport
// ---------------------------------------------------------------------------

/// Aggregated quality report for a single prompt/response pair.
#[derive(Debug, Clone)]
pub struct QualityReport {
    pub prompt_id: String,
    pub scores: Vec<DimensionScore>,
    /// Weighted overall score in [0.0, 1.0].
    pub overall: f64,
    pub recommendation: String,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Tokenise text into lowercase alphabetic words.
fn tokenise(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphabetic())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

/// Split text into non-empty sentences (split on `.`, `!`, `?`).
fn sentences(text: &str) -> Vec<String> {
    text.split(|c| c == '.' || c == '!' || c == '?')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Build a term-frequency map from a token list.
fn tf_map(tokens: &[String]) -> HashMap<String, f64> {
    let mut map: HashMap<String, f64> = HashMap::new();
    for t in tokens {
        *map.entry(t.clone()).or_insert(0.0) += 1.0;
    }
    let total = tokens.len().max(1) as f64;
    map.values_mut().for_each(|v| *v /= total);
    map
}

/// Cosine similarity between two TF maps.
fn cosine_similarity(a: &HashMap<String, f64>, b: &HashMap<String, f64>) -> f64 {
    let dot: f64 = a
        .iter()
        .filter_map(|(k, v)| b.get(k).map(|bv| v * bv))
        .sum();
    let mag_a: f64 = a.values().map(|v| v * v).sum::<f64>().sqrt();
    let mag_b: f64 = b.values().map(|v| v * v).sum::<f64>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        0.0
    } else {
        (dot / (mag_a * mag_b)).min(1.0)
    }
}

/// Shared-word overlap between two token slices (Jaccard-like).
fn word_overlap(a: &[String], b: &[String]) -> f64 {
    let set_a: std::collections::HashSet<&String> = a.iter().collect();
    let set_b: std::collections::HashSet<&String> = b.iter().collect();
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

// ---------------------------------------------------------------------------
// QualityScorer
// ---------------------------------------------------------------------------

pub struct QualityScorer;

impl QualityScorer {
    pub fn new() -> Self {
        QualityScorer
    }

    // -----------------------------------------------------------------------
    // Individual dimension scorers
    // -----------------------------------------------------------------------

    /// Score completeness: how many prompt keywords (length ≥ 4) appear in the
    /// response?  Returns a ratio in [0.0, 1.0].
    pub fn score_completeness(prompt: &str, response: &str) -> f64 {
        let prompt_tokens = tokenise(prompt);
        let response_tokens: std::collections::HashSet<String> =
            tokenise(response).into_iter().collect();

        let keywords: Vec<&String> = prompt_tokens
            .iter()
            .filter(|w| w.len() >= 4)
            .collect();

        if keywords.is_empty() {
            return 1.0; // nothing to cover
        }

        let covered = keywords
            .iter()
            .filter(|k| response_tokens.contains(**k))
            .count();

        covered as f64 / keywords.len() as f64
    }

    /// Score coherence: average shared-word overlap between adjacent sentences.
    /// A proxy for smooth topic transitions.
    pub fn score_coherence(response: &str) -> f64 {
        let sents = sentences(response);
        if sents.len() < 2 {
            return 1.0; // single sentence is trivially coherent
        }

        let tokenised: Vec<Vec<String>> = sents.iter().map(|s| tokenise(s)).collect();
        let mut total = 0.0;
        let pairs = tokenised.len() - 1;
        for i in 0..pairs {
            total += word_overlap(&tokenised[i], &tokenised[i + 1]);
        }
        (total / pairs as f64).min(1.0)
    }

    /// Score conciseness: ratio of unique sentences to total sentences.
    /// Penalises repetition.
    pub fn score_conciseness(response: &str) -> f64 {
        let sents = sentences(response);
        if sents.is_empty() {
            return 1.0;
        }
        let unique: std::collections::HashSet<String> =
            sents.iter().map(|s| s.to_lowercase()).collect();
        unique.len() as f64 / sents.len() as f64
    }

    /// Accuracy is difficult to measure automatically without ground truth.
    /// Returns 0.75 as a conservative neutral estimate.
    pub fn score_accuracy(_prompt: &str, _response: &str) -> f64 {
        0.75
    }

    /// Score relevance: cosine similarity between TF vectors of prompt and response.
    pub fn score_relevance(prompt: &str, response: &str) -> f64 {
        let pt = tokenise(prompt);
        let rt = tokenise(response);
        let tf_p = tf_map(&pt);
        let tf_r = tf_map(&rt);
        cosine_similarity(&tf_p, &tf_r)
    }

    // -----------------------------------------------------------------------
    // Aggregate
    // -----------------------------------------------------------------------

    /// Score all five dimensions and return the individual results.
    pub fn score_all(prompt: &str, response: &str) -> Vec<DimensionScore> {
        let completeness = Self::score_completeness(prompt, response);
        let coherence = Self::score_coherence(response);
        let conciseness = Self::score_conciseness(response);
        let accuracy = Self::score_accuracy(prompt, response);
        let relevance = Self::score_relevance(prompt, response);

        vec![
            DimensionScore {
                dimension: QualityDimension::Completeness,
                score: completeness,
                explanation: format!(
                    "{}% of prompt keywords found in response",
                    (completeness * 100.0).round()
                ),
            },
            DimensionScore {
                dimension: QualityDimension::Coherence,
                score: coherence,
                explanation: format!(
                    "Average adjacent-sentence word overlap: {:.2}",
                    coherence
                ),
            },
            DimensionScore {
                dimension: QualityDimension::Conciseness,
                score: conciseness,
                explanation: format!(
                    "Unique sentence ratio: {:.2} (higher = less repetition)",
                    conciseness
                ),
            },
            DimensionScore {
                dimension: QualityDimension::Accuracy,
                score: accuracy,
                explanation: "Accuracy cannot be measured automatically without ground truth; \
                              conservative neutral estimate applied."
                    .to_string(),
            },
            DimensionScore {
                dimension: QualityDimension::Relevance,
                score: relevance,
                explanation: format!(
                    "TF-IDF cosine similarity between prompt and response: {:.3}",
                    relevance
                ),
            },
        ]
    }

    /// Compute a weighted average overall score from dimension scores.
    pub fn overall_score(scores: &[DimensionScore]) -> f64 {
        let mut total_weight = 0.0_f64;
        let mut weighted_sum = 0.0_f64;
        for ds in scores {
            let w = ds.dimension.weight();
            weighted_sum += ds.score * w;
            total_weight += w;
        }
        if total_weight == 0.0 {
            0.0
        } else {
            (weighted_sum / total_weight).min(1.0)
        }
    }

    /// Build a full `QualityReport` for a prompt/response pair.
    pub fn report(prompt_id: &str, prompt: &str, response: &str) -> QualityReport {
        let scores = Self::score_all(prompt, response);
        let overall = Self::overall_score(&scores);

        let recommendation = if overall >= 0.85 {
            "Excellent response. No changes recommended.".to_string()
        } else if overall >= 0.70 {
            "Good response. Minor improvements possible in lower-scoring dimensions.".to_string()
        } else if overall >= 0.50 {
            "Adequate response. Consider improving completeness and relevance.".to_string()
        } else {
            "Poor response. Significant revision recommended across multiple dimensions.".to_string()
        };

        QualityReport {
            prompt_id: prompt_id.to_string(),
            scores,
            overall,
            recommendation,
        }
    }
}

impl Default for QualityScorer {
    fn default() -> Self {
        QualityScorer::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const PROMPT: &str = "Explain the Rust ownership model and borrowing rules.";
    const GOOD_RESPONSE: &str = "Rust ownership ensures memory safety without a garbage \
        collector. Every value has an owner. When the owner goes out of scope the value \
        is dropped. Borrowing allows references to a value without taking ownership. \
        Mutable borrowing is exclusive; immutable borrowing can be shared.";
    const BAD_RESPONSE: &str = "Python is a great language. I like cats. Cats are nice.";

    #[test]
    fn completeness_good_response() {
        let score = QualityScorer::score_completeness(PROMPT, GOOD_RESPONSE);
        assert!(score > 0.5, "score={}", score);
    }

    #[test]
    fn completeness_bad_response() {
        let score = QualityScorer::score_completeness(PROMPT, BAD_RESPONSE);
        assert!(score < 0.5, "score={}", score);
    }

    #[test]
    fn coherence_single_sentence() {
        assert_eq!(QualityScorer::score_coherence("One sentence only."), 1.0);
    }

    #[test]
    fn coherence_related_sentences() {
        let text = "Rust uses ownership for memory safety. \
                    Memory safety avoids dangling pointers. \
                    Dangling pointers can cause undefined behaviour.";
        let score = QualityScorer::score_coherence(text);
        assert!(score > 0.0);
    }

    #[test]
    fn conciseness_no_repetition() {
        let score = QualityScorer::score_conciseness(GOOD_RESPONSE);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn conciseness_with_repetition() {
        let text = "Hello world. Hello world. Hello world.";
        let score = QualityScorer::score_conciseness(text);
        // 1 unique / 3 total
        assert!((score - 1.0 / 3.0).abs() < 0.01, "score={}", score);
    }

    #[test]
    fn relevance_on_topic_response() {
        let score = QualityScorer::score_relevance(PROMPT, GOOD_RESPONSE);
        assert!(score > 0.0);
    }

    #[test]
    fn relevance_off_topic_response() {
        let on = QualityScorer::score_relevance(PROMPT, GOOD_RESPONSE);
        let off = QualityScorer::score_relevance(PROMPT, BAD_RESPONSE);
        assert!(on > off, "on={}, off={}", on, off);
    }

    #[test]
    fn score_all_returns_five_dimensions() {
        let scores = QualityScorer::score_all(PROMPT, GOOD_RESPONSE);
        assert_eq!(scores.len(), 5);
    }

    #[test]
    fn overall_score_in_range() {
        let scores = QualityScorer::score_all(PROMPT, GOOD_RESPONSE);
        let overall = QualityScorer::overall_score(&scores);
        assert!(overall >= 0.0 && overall <= 1.0, "overall={}", overall);
    }

    #[test]
    fn overall_good_gt_bad() {
        let good_scores = QualityScorer::score_all(PROMPT, GOOD_RESPONSE);
        let bad_scores = QualityScorer::score_all(PROMPT, BAD_RESPONSE);
        let good = QualityScorer::overall_score(&good_scores);
        let bad = QualityScorer::overall_score(&bad_scores);
        assert!(good > bad, "good={}, bad={}", good, bad);
    }

    #[test]
    fn report_builds_correctly() {
        let report = QualityScorer::report("test-001", PROMPT, GOOD_RESPONSE);
        assert_eq!(report.prompt_id, "test-001");
        assert_eq!(report.scores.len(), 5);
        assert!(report.overall >= 0.0 && report.overall <= 1.0);
        assert!(!report.recommendation.is_empty());
    }

    #[test]
    fn weights_sum_to_one() {
        let dims = [
            QualityDimension::Completeness,
            QualityDimension::Coherence,
            QualityDimension::Conciseness,
            QualityDimension::Accuracy,
            QualityDimension::Relevance,
        ];
        let total: f64 = dims.iter().map(|d| d.weight()).sum();
        assert!((total - 1.0).abs() < 1e-10, "total weight={}", total);
    }
}
