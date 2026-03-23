//! Aggregate and merge partial results from multiple model calls.
//!
//! Supports several aggregation strategies: first-complete, best-confidence,
//! consensus by token overlap, weighted merge with sentence-level deduplication,
//! ensemble voting, and longest-output selection.

#![allow(dead_code)]

use std::collections::HashMap;

// ── PartialResult ─────────────────────────────────────────────────────────────

/// A single result produced by one model call or one parallel shard.
#[derive(Debug, Clone)]
pub struct PartialResult {
    /// Unique identifier for the source model / shard.
    pub source_id: String,
    /// The generated text content.
    pub content: String,
    /// Confidence score in [0, 1].
    pub confidence: f64,
    /// Number of tokens consumed (input + output).
    pub tokens: usize,
    /// Wall-clock latency for this call in milliseconds.
    pub latency_ms: u64,
}

// ── AggregationMethod ─────────────────────────────────────────────────────────

/// Strategy used to combine [`PartialResult`]s.
#[derive(Debug, Clone, PartialEq)]
pub enum AggregationMethod {
    /// Return the first result that arrives (lowest latency).
    FirstComplete,
    /// Return the result with the highest confidence score.
    BestConfidence,
    /// Return a result only when the majority (> threshold fraction) of sources agree.
    Consensus(f64),
    /// Interleave content from all results, weighted by confidence.
    WeightedMerge,
    /// Return the longest content by character count.
    Longest,
    /// For each sentence position, vote for the most common sentence across results.
    Ensemble,
}

// ── AggregationResult ─────────────────────────────────────────────────────────

/// The output of an aggregation operation.
#[derive(Debug, Clone)]
pub struct AggregationResult {
    /// The method that produced this result.
    pub method: AggregationMethod,
    /// The merged / selected output text.
    pub final_content: String,
    /// Aggregate confidence (method-dependent).
    pub confidence: f64,
    /// IDs of the sources that contributed.
    pub sources_used: Vec<String>,
    /// Total token count across all contributing sources.
    pub total_tokens: usize,
}

// ── ResultAggregator ──────────────────────────────────────────────────────────

/// Collects [`PartialResult`]s and aggregates them on demand.
#[derive(Debug, Default)]
pub struct ResultAggregator {
    results: Vec<PartialResult>,
}

impl ResultAggregator {
    /// Create a new, empty aggregator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a partial result to the collection.
    pub fn add_result(&mut self, result: PartialResult) {
        self.results.push(result);
    }

    /// Aggregate all collected results using the given method.
    ///
    /// Returns `None` if no results have been added.
    pub fn aggregate(&self, method: &AggregationMethod) -> Option<AggregationResult> {
        if self.results.is_empty() {
            return None;
        }
        match method {
            AggregationMethod::FirstComplete => {
                // Assume insertion order reflects arrival order.
                let r = &self.results[0];
                Some(AggregationResult {
                    method: method.clone(),
                    final_content: r.content.clone(),
                    confidence: r.confidence,
                    sources_used: vec![r.source_id.clone()],
                    total_tokens: r.tokens,
                })
            }
            AggregationMethod::BestConfidence => {
                let r = self
                    .results
                    .iter()
                    .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())?;
                Some(AggregationResult {
                    method: method.clone(),
                    final_content: r.content.clone(),
                    confidence: r.confidence,
                    sources_used: vec![r.source_id.clone()],
                    total_tokens: r.tokens,
                })
            }
            AggregationMethod::Consensus(threshold) => {
                let content = Self::consensus_check(&self.results, *threshold)?;
                let mean_conf = self.results.iter().map(|r| r.confidence).sum::<f64>()
                    / self.results.len() as f64;
                let total_tokens = self.results.iter().map(|r| r.tokens).sum();
                let sources: Vec<String> =
                    self.results.iter().map(|r| r.source_id.clone()).collect();
                Some(AggregationResult {
                    method: method.clone(),
                    final_content: content,
                    confidence: mean_conf,
                    sources_used: sources,
                    total_tokens,
                })
            }
            AggregationMethod::WeightedMerge => {
                let content = Self::weighted_merge(&self.results);
                let total_conf: f64 = self.results.iter().map(|r| r.confidence).sum();
                let weight_sum: f64 = self
                    .results
                    .iter()
                    .map(|r| r.confidence)
                    .sum::<f64>()
                    .max(1e-10);
                let confidence = total_conf / weight_sum;
                let total_tokens = self.results.iter().map(|r| r.tokens).sum();
                let sources: Vec<String> =
                    self.results.iter().map(|r| r.source_id.clone()).collect();
                Some(AggregationResult {
                    method: method.clone(),
                    final_content: content,
                    confidence,
                    sources_used: sources,
                    total_tokens,
                })
            }
            AggregationMethod::Longest => {
                let r = self
                    .results
                    .iter()
                    .max_by_key(|r| r.content.len())?;
                Some(AggregationResult {
                    method: method.clone(),
                    final_content: r.content.clone(),
                    confidence: r.confidence,
                    sources_used: vec![r.source_id.clone()],
                    total_tokens: r.tokens,
                })
            }
            AggregationMethod::Ensemble => {
                let content = Self::ensemble_vote(&self.results);
                let mean_conf = self.results.iter().map(|r| r.confidence).sum::<f64>()
                    / self.results.len() as f64;
                let total_tokens = self.results.iter().map(|r| r.tokens).sum();
                let sources: Vec<String> =
                    self.results.iter().map(|r| r.source_id.clone()).collect();
                Some(AggregationResult {
                    method: method.clone(),
                    final_content: content,
                    confidence: mean_conf,
                    sources_used: sources,
                    total_tokens,
                })
            }
        }
    }

    // ── Consensus ────────────────────────────────────────────────────────────

    /// Return the content that appears in more than `threshold` fraction of
    /// results (by token-level Jaccard overlap ≥ 0.5), or `None` if no such
    /// consensus exists.
    pub fn consensus_check(results: &[PartialResult], threshold: f64) -> Option<String> {
        if results.is_empty() {
            return None;
        }
        let n = results.len() as f64;
        for candidate in results {
            let candidate_words = word_set(&candidate.content);
            let agree = results
                .iter()
                .filter(|r| {
                    let other_words = word_set(&r.content);
                    jaccard(&candidate_words, &other_words) >= 0.5
                })
                .count() as f64;
            if agree / n > threshold {
                return Some(candidate.content.clone());
            }
        }
        None
    }

    // ── Weighted merge ───────────────────────────────────────────────────────

    /// Interleave sentences from all results, ordered by descending source
    /// confidence. Duplicate sentences (exact match) are dropped.
    pub fn weighted_merge(results: &[PartialResult]) -> String {
        let mut sorted = results.to_vec();
        sorted.sort_unstable_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut output: Vec<String> = Vec::new();

        // Collect all sentences in round-robin order across sorted sources.
        let sentences_per_source: Vec<Vec<String>> = sorted
            .iter()
            .map(|r| split_sentences(&r.content))
            .collect();

        let max_len = sentences_per_source.iter().map(|v| v.len()).max().unwrap_or(0);
        for i in 0..max_len {
            for sentences in &sentences_per_source {
                if let Some(s) = sentences.get(i) {
                    let trimmed = s.trim().to_string();
                    if !trimmed.is_empty() && !seen.contains(&trimmed) {
                        seen.insert(trimmed.clone());
                        output.push(trimmed);
                    }
                }
            }
        }
        output.join(" ")
    }

    // ── Ensemble vote ────────────────────────────────────────────────────────

    /// For each sentence position, pick the most common sentence across all
    /// results. Ties are broken by the first occurrence.
    pub fn ensemble_vote(results: &[PartialResult]) -> String {
        if results.is_empty() {
            return String::new();
        }
        let all_sentences: Vec<Vec<String>> =
            results.iter().map(|r| split_sentences(&r.content)).collect();
        let max_len = all_sentences.iter().map(|v| v.len()).max().unwrap_or(0);

        let mut output: Vec<String> = Vec::new();
        for pos in 0..max_len {
            let mut counts: HashMap<String, usize> = HashMap::new();
            for sentences in &all_sentences {
                if let Some(s) = sentences.get(pos) {
                    let trimmed = s.trim().to_string();
                    if !trimmed.is_empty() {
                        *counts.entry(trimmed).or_insert(0) += 1;
                    }
                }
            }
            if let Some((sentence, _)) = counts.iter().max_by_key(|(_, v)| *v) {
                output.push(sentence.clone());
            }
        }
        output.join(" ")
    }

    // ── Diversity ────────────────────────────────────────────────────────────

    /// Mean pairwise Jaccard distance of word sets across all results.
    ///
    /// Returns 0.0 for 0 or 1 result, 1.0 for completely disjoint results.
    pub fn result_diversity(results: &[PartialResult]) -> f64 {
        let n = results.len();
        if n < 2 {
            return 0.0;
        }
        let word_sets: Vec<std::collections::HashSet<String>> =
            results.iter().map(|r| word_set(&r.content)).collect();

        let mut total_dist = 0.0_f64;
        let mut pairs = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                total_dist += 1.0 - jaccard(&word_sets[i], &word_sets[j]);
                pairs += 1;
            }
        }
        if pairs == 0 {
            0.0
        } else {
            total_dist / pairs as f64
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Split text into sentences on `.`, `!`, or `?`.
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }
    let remainder = current.trim().to_string();
    if !remainder.is_empty() {
        sentences.push(remainder);
    }
    sentences
}

/// Return the set of lowercase words in `text`.
fn word_set(text: &str) -> std::collections::HashSet<String> {
    text.split_whitespace()
        .map(|w| w.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| !w.is_empty())
        .collect()
}

/// Jaccard similarity of two word sets: |A ∩ B| / |A ∪ B|.
fn jaccard(
    a: &std::collections::HashSet<String>,
    b: &std::collections::HashSet<String>,
) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let intersection = a.intersection(b).count() as f64;
    let union = a.union(b).count() as f64;
    if union == 0.0 {
        1.0
    } else {
        intersection / union
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(id: &str, content: &str, confidence: f64) -> PartialResult {
        PartialResult {
            source_id: id.to_string(),
            content: content.to_string(),
            confidence,
            tokens: content.split_whitespace().count(),
            latency_ms: 100,
        }
    }

    #[test]
    fn test_first_complete() {
        let mut agg = ResultAggregator::new();
        agg.add_result(make_result("m1", "Hello world.", 0.9));
        agg.add_result(make_result("m2", "Goodbye world.", 0.8));
        let result = agg.aggregate(&AggregationMethod::FirstComplete).unwrap();
        assert_eq!(result.final_content, "Hello world.");
        assert_eq!(result.sources_used, vec!["m1"]);
    }

    #[test]
    fn test_best_confidence() {
        let mut agg = ResultAggregator::new();
        agg.add_result(make_result("m1", "Low.", 0.3));
        agg.add_result(make_result("m2", "High.", 0.95));
        let result = agg.aggregate(&AggregationMethod::BestConfidence).unwrap();
        assert_eq!(result.final_content, "High.");
    }

    #[test]
    fn test_longest() {
        let mut agg = ResultAggregator::new();
        agg.add_result(make_result("m1", "Short.", 0.5));
        agg.add_result(make_result("m2", "This is a much longer sentence.", 0.4));
        let result = agg.aggregate(&AggregationMethod::Longest).unwrap();
        assert_eq!(result.final_content, "This is a much longer sentence.");
    }

    #[test]
    fn test_empty_aggregator() {
        let agg = ResultAggregator::new();
        assert!(agg.aggregate(&AggregationMethod::FirstComplete).is_none());
    }

    #[test]
    fn test_consensus_check_majority() {
        let results = vec![
            make_result("m1", "The cat sat on the mat.", 0.9),
            make_result("m2", "The cat sat on the mat.", 0.8),
            make_result("m3", "Something completely different.", 0.7),
        ];
        // 2/3 agree → threshold 0.5 should find consensus.
        let c = ResultAggregator::consensus_check(&results, 0.5);
        assert!(c.is_some());
    }

    #[test]
    fn test_consensus_check_no_majority() {
        let results = vec![
            make_result("m1", "apples oranges bananas", 0.9),
            make_result("m2", "elephants giraffes lions", 0.8),
            make_result("m3", "spaghetti carbonara pasta", 0.7),
        ];
        let c = ResultAggregator::consensus_check(&results, 0.6);
        // No two results share enough tokens.
        assert!(c.is_none());
    }

    #[test]
    fn test_weighted_merge_deduplication() {
        let results = vec![
            make_result("m1", "Sentence one. Sentence two.", 0.9),
            make_result("m2", "Sentence one. Sentence three.", 0.8),
        ];
        let merged = ResultAggregator::weighted_merge(&results);
        // "Sentence one." should appear only once.
        let count = merged.matches("Sentence one").count();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_ensemble_vote() {
        let results = vec![
            make_result("m1", "Alpha. Beta.", 0.9),
            make_result("m2", "Alpha. Gamma.", 0.8),
            make_result("m3", "Alpha. Beta.", 0.7),
        ];
        let voted = ResultAggregator::ensemble_vote(&results);
        // "Alpha" should appear (majority pos 0), "Beta" majority at pos 1.
        assert!(voted.contains("Alpha"));
        assert!(voted.contains("Beta"));
    }

    #[test]
    fn test_result_diversity_identical() {
        let results = vec![
            make_result("m1", "the quick brown fox", 0.9),
            make_result("m2", "the quick brown fox", 0.8),
        ];
        let div = ResultAggregator::result_diversity(&results);
        // Identical → Jaccard=1 → distance=0.
        assert!(div < 1e-10);
    }

    #[test]
    fn test_result_diversity_disjoint() {
        let results = vec![
            make_result("m1", "aaa bbb ccc", 0.9),
            make_result("m2", "xxx yyy zzz", 0.8),
        ];
        let div = ResultAggregator::result_diversity(&results);
        // Completely disjoint → distance=1.
        assert!((div - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_result_diversity_single() {
        let results = vec![make_result("m1", "only one", 0.9)];
        assert_eq!(ResultAggregator::result_diversity(&results), 0.0);
    }

    #[test]
    fn test_weighted_merge_agg_method() {
        let mut agg = ResultAggregator::new();
        agg.add_result(make_result("m1", "Hello there.", 0.9));
        agg.add_result(make_result("m2", "Goodbye there.", 0.5));
        let result = agg.aggregate(&AggregationMethod::WeightedMerge).unwrap();
        assert!(!result.final_content.is_empty());
        assert_eq!(result.sources_used.len(), 2);
    }
}
