//! Token sequence reranker using multiple scoring criteria.
//!
//! Supports TF-IDF relevance, diversity (cosine dissimilarity), novelty
//! (unseen-token fraction), quality heuristics, and Maximal Marginal Relevance
//! (MMR) for diverse subset selection.

use std::collections::{HashMap, HashSet};

// ── RankingCriterion ──────────────────────────────────────────────────────────

/// A criterion used when ranking token sequences.
#[derive(Debug, Clone)]
pub enum RankingCriterion {
    /// Semantic relevance to a query string.
    Relevance { query: String },
    /// Encourage diversity across the ranked set.
    Diversity,
    /// Penalise tokens that have already been seen.
    Novelty { seen: HashSet<String> },
    /// Intrinsic quality heuristics (length, content-word ratio, dedup).
    Quality,
    /// Recency boost — earlier positions in `timestamps` are treated as newer.
    Recency { timestamps: Vec<u64> },
}

// ── RankingScore ──────────────────────────────────────────────────────────────

/// Per-criterion scores for a single token sequence.
#[derive(Debug, Clone)]
pub struct RankingScore {
    pub relevance: f64,
    pub diversity: f64,
    pub novelty: f64,
    pub quality: f64,
    /// Weighted composite of all individual scores.
    pub composite: f64,
}

// ── ScoredSequence ────────────────────────────────────────────────────────────

/// A token sequence with its associated ranking score.
#[derive(Debug, Clone)]
pub struct ScoredSequence {
    pub tokens: Vec<String>,
    pub score: RankingScore,
    /// 1-based rank (1 = best).
    pub rank: usize,
}

// ── Reranker ──────────────────────────────────────────────────────────────────

/// Multi-criterion token-sequence reranker.
///
/// # Weights
/// The `weights` map controls how much each criterion contributes to the
/// composite score.  Keys match criterion names: `"relevance"`, `"diversity"`,
/// `"novelty"`, `"quality"`, `"recency"`.  Missing keys default to `1.0`.
pub struct Reranker {
    pub weights: HashMap<String, f64>,
}

impl Default for Reranker {
    fn default() -> Self {
        Self::new()
    }
}

impl Reranker {
    /// Create a reranker with equal weights for all criteria.
    pub fn new() -> Self {
        let mut weights = HashMap::new();
        weights.insert("relevance".to_string(), 1.0);
        weights.insert("diversity".to_string(), 1.0);
        weights.insert("novelty".to_string(), 1.0);
        weights.insert("quality".to_string(), 1.0);
        weights.insert("recency".to_string(), 1.0);
        Reranker { weights }
    }

    // ── individual scorers ────────────────────────────────────────────────────

    /// TF-IDF cosine similarity between `tokens` and the query words.
    pub fn score_relevance(tokens: &[String], query: &str) -> f64 {
        if tokens.is_empty() || query.is_empty() {
            return 0.0;
        }
        let query_terms: Vec<String> = query
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();

        let token_tf = Self::term_frequency(tokens);
        let query_tf = Self::term_frequency_str(&query_terms);

        // Simplified TF cosine (no IDF corpus needed for single-document comparison)
        Self::cosine_similarity_maps(&token_tf, &query_tf)
    }

    /// Diversity: 1 − max cosine similarity to any other sequence.
    pub fn score_diversity(tokens: &[String], other_sequences: &[Vec<String>]) -> f64 {
        if other_sequences.is_empty() {
            return 1.0;
        }
        let tv = Self::term_frequency(tokens);
        let max_sim = other_sequences
            .iter()
            .map(|other| {
                let ov = Self::term_frequency(other);
                Self::cosine_similarity_maps(&tv, &ov)
            })
            .fold(0.0_f64, f64::max);
        (1.0 - max_sim).clamp(0.0, 1.0)
    }

    /// Novelty: fraction of tokens NOT present in `seen_tokens`.
    pub fn score_novelty(tokens: &[String], seen_tokens: &HashSet<String>) -> f64 {
        if tokens.is_empty() {
            return 0.0;
        }
        let novel = tokens
            .iter()
            .filter(|t| !seen_tokens.contains(t.as_str()))
            .count();
        novel as f64 / tokens.len() as f64
    }

    /// Quality heuristic: blend of avg token length, content-word fraction, and
    /// unique-token fraction.
    pub fn score_quality(tokens: &[String]) -> f64 {
        if tokens.is_empty() {
            return 0.0;
        }

        // Average token length — longer tokens tend to be more informative
        let avg_len = tokens.iter().map(|t| t.len()).sum::<usize>() as f64
            / tokens.len() as f64;
        // Normalise: assume 8 chars = 1.0 (typical content word length)
        let len_score = (avg_len / 8.0).min(1.0);

        // Content-word fraction: words that are not common stopwords
        let content_fraction = {
            let content = tokens.iter().filter(|t| !Self::is_stopword(t)).count();
            content as f64 / tokens.len() as f64
        };

        // Unique-token fraction
        let unique: HashSet<&str> = tokens.iter().map(|t| t.as_str()).collect();
        let unique_fraction = unique.len() as f64 / tokens.len() as f64;

        // Weighted blend
        let score = 0.3 * len_score + 0.4 * content_fraction + 0.3 * unique_fraction;
        score.clamp(0.0, 1.0)
    }

    // ── rank ──────────────────────────────────────────────────────────────────

    /// Score all sequences by the given criteria, compute weighted composites,
    /// and return them sorted in descending order of composite score.
    pub fn rank(
        &self,
        sequences: Vec<Vec<String>>,
        criteria: &[RankingCriterion],
    ) -> Vec<ScoredSequence> {
        if sequences.is_empty() {
            return vec![];
        }

        let mut scored: Vec<ScoredSequence> = sequences
            .into_iter()
            .map(|tokens| {
                let mut rel = 0.0_f64;
                let mut div = 1.0_f64;
                let mut nov = 1.0_f64;
                let mut qual = Self::score_quality(&tokens);
                let mut rec = 0.0_f64;

                for criterion in criteria {
                    match criterion {
                        RankingCriterion::Relevance { query } => {
                            rel = Self::score_relevance(&tokens, query);
                        }
                        RankingCriterion::Diversity => {
                            // diversity is computed post-hoc below
                        }
                        RankingCriterion::Novelty { seen } => {
                            nov = Self::score_novelty(&tokens, seen);
                        }
                        RankingCriterion::Quality => {
                            qual = Self::score_quality(&tokens);
                        }
                        RankingCriterion::Recency { timestamps } => {
                            rec = Self::score_recency(&tokens, timestamps);
                        }
                    }
                }

                let score = RankingScore {
                    relevance: rel,
                    diversity: div,
                    novelty: nov,
                    quality: qual,
                    composite: 0.0,
                };
                ScoredSequence { tokens, score, rank: 0 }
            })
            .collect();

        // Compute pairwise diversity now that we have all sequences
        let all_seqs: Vec<Vec<String>> = scored.iter().map(|s| s.tokens.clone()).collect();
        for i in 0..scored.len() {
            let others: Vec<Vec<String>> = all_seqs
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, v)| v.clone())
                .collect();
            scored[i].score.diversity = Self::score_diversity(&scored[i].tokens, &others);
        }

        // Composite
        let w_rel = self.weights.get("relevance").copied().unwrap_or(1.0);
        let w_div = self.weights.get("diversity").copied().unwrap_or(1.0);
        let w_nov = self.weights.get("novelty").copied().unwrap_or(1.0);
        let w_qual = self.weights.get("quality").copied().unwrap_or(1.0);
        let w_rec = self.weights.get("recency").copied().unwrap_or(0.0);
        let total_w = w_rel + w_div + w_nov + w_qual + w_rec;
        let denom = if total_w == 0.0 { 1.0 } else { total_w };

        for s in &mut scored {
            s.score.composite = (w_rel * s.score.relevance
                + w_div * s.score.diversity
                + w_nov * s.score.novelty
                + w_qual * s.score.quality)
                / denom;
            s.score.composite = s.score.composite.clamp(0.0, 1.0);
        }

        // Sort descending
        scored.sort_by(|a, b| {
            b.score
                .composite
                .partial_cmp(&a.score.composite)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Assign ranks
        for (i, s) in scored.iter_mut().enumerate() {
            s.rank = i + 1;
        }

        scored
    }

    // ── MMR ───────────────────────────────────────────────────────────────────

    /// Maximal Marginal Relevance reranking.
    ///
    /// Greedily selects up to `k` sequences that maximise
    /// `λ * relevance − (1−λ) * max_sim_to_selected`.
    pub fn mmr_rerank(
        &self,
        sequences: Vec<Vec<String>>,
        query: &str,
        lambda: f64,
        k: usize,
    ) -> Vec<Vec<String>> {
        if sequences.is_empty() || k == 0 {
            return vec![];
        }
        let lambda = lambda.clamp(0.0, 1.0);
        let k = k.min(sequences.len());

        // Pre-compute relevance and term-frequency vectors
        let rel_scores: Vec<f64> = sequences
            .iter()
            .map(|seq| Self::score_relevance(seq, query))
            .collect();
        let tf_vecs: Vec<HashMap<String, f64>> = sequences
            .iter()
            .map(|seq| Self::term_frequency(seq))
            .collect();

        let mut selected_indices: Vec<usize> = Vec::with_capacity(k);
        let mut remaining: Vec<usize> = (0..sequences.len()).collect();

        while selected_indices.len() < k && !remaining.is_empty() {
            let best = remaining
                .iter()
                .map(|&i| {
                    let rel = rel_scores[i];
                    let max_sim = if selected_indices.is_empty() {
                        0.0
                    } else {
                        selected_indices
                            .iter()
                            .map(|&j| {
                                Self::cosine_similarity_maps(&tf_vecs[i], &tf_vecs[j])
                            })
                            .fold(0.0_f64, f64::max)
                    };
                    let mmr_score = lambda * rel - (1.0 - lambda) * max_sim;
                    (i, mmr_score)
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            if let Some((best_idx, _)) = best {
                selected_indices.push(best_idx);
                remaining.retain(|&i| i != best_idx);
            } else {
                break;
            }
        }

        selected_indices
            .into_iter()
            .map(|i| sequences[i].clone())
            .collect()
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    fn term_frequency(tokens: &[String]) -> HashMap<String, f64> {
        let mut tf: HashMap<String, f64> = HashMap::new();
        for token in tokens {
            *tf.entry(token.to_lowercase()).or_insert(0.0) += 1.0;
        }
        tf
    }

    fn term_frequency_str(words: &[String]) -> HashMap<String, f64> {
        let mut tf: HashMap<String, f64> = HashMap::new();
        for word in words {
            *tf.entry(word.clone()).or_insert(0.0) += 1.0;
        }
        tf
    }

    fn cosine_similarity_maps(a: &HashMap<String, f64>, b: &HashMap<String, f64>) -> f64 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }
        let dot: f64 = a.iter().map(|(k, v)| v * b.get(k).copied().unwrap_or(0.0)).sum();
        let mag_a: f64 = a.values().map(|v| v * v).sum::<f64>().sqrt();
        let mag_b: f64 = b.values().map(|v| v * v).sum::<f64>().sqrt();
        if mag_a == 0.0 || mag_b == 0.0 {
            0.0
        } else {
            (dot / (mag_a * mag_b)).clamp(0.0, 1.0)
        }
    }

    fn score_recency(_tokens: &[String], timestamps: &[u64]) -> f64 {
        if timestamps.is_empty() {
            return 0.0;
        }
        // Recency: use average timestamp; normalise by max
        let max_ts = *timestamps.iter().max().unwrap_or(&1);
        if max_ts == 0 {
            return 0.0;
        }
        let avg = timestamps.iter().sum::<u64>() as f64 / timestamps.len() as f64;
        (avg / max_ts as f64).clamp(0.0, 1.0)
    }

    fn is_stopword(token: &str) -> bool {
        const STOPS: &[&str] = &[
            "the", "a", "an", "is", "it", "in", "on", "at", "of", "to", "and",
            "or", "but", "for", "with", "that", "this", "are", "was", "be",
            "as", "by", "from", "not", "no", "so", "if", "he", "she", "we",
        ];
        let lower = token.to_lowercase();
        STOPS.contains(&lower.as_str())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(words: &[&str]) -> Vec<String> {
        words.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn relevance_ranking_puts_query_matching_first() {
        let reranker = Reranker::new();
        let sequences = vec![
            seq(&["unrelated", "content", "here"]),
            seq(&["machine", "learning", "model", "training"]),
            seq(&["machine", "learning", "deep", "neural"]),
        ];
        let criteria = [RankingCriterion::Relevance {
            query: "machine learning".to_string(),
        }];
        let ranked = reranker.rank(sequences, &criteria);
        // Top-ranked sequence should contain "machine" or "learning"
        let top = &ranked[0];
        assert!(
            top.tokens.contains(&"machine".to_string()) || top.tokens.contains(&"learning".to_string()),
            "Top-ranked sequence should be query-relevant: {:?}",
            top.tokens
        );
    }

    #[test]
    fn novelty_penalises_repeated_content() {
        let mut seen: HashSet<String> = HashSet::new();
        seen.insert("hello".to_string());
        seen.insert("world".to_string());

        let all_seen = seq(&["hello", "world"]);
        let all_novel = seq(&["brand", "new", "content"]);

        let nov_seen = Reranker::score_novelty(&all_seen, &seen);
        let nov_novel = Reranker::score_novelty(&all_novel, &seen);

        assert_eq!(nov_seen, 0.0);
        assert_eq!(nov_novel, 1.0);
    }

    #[test]
    fn mmr_selects_diverse_set() {
        let reranker = Reranker::new();
        let sequences = vec![
            seq(&["cat", "dog", "pet"]),
            seq(&["cat", "dog", "pet"]),   // nearly identical
            seq(&["space", "rocket", "orbit"]), // diverse
        ];
        let result = reranker.mmr_rerank(sequences, "cat", 2, 2);
        assert_eq!(result.len(), 2);
        // The diverse sequence should be selected
        let has_diverse = result
            .iter()
            .any(|s| s.contains(&"space".to_string()) || s.contains(&"rocket".to_string()));
        assert!(has_diverse, "MMR should select the diverse sequence");
    }

    #[test]
    fn composite_score_in_range() {
        let reranker = Reranker::new();
        let sequences = vec![
            seq(&["hello", "world"]),
            seq(&["foo", "bar", "baz"]),
        ];
        let criteria = [
            RankingCriterion::Relevance { query: "hello".to_string() },
            RankingCriterion::Quality,
        ];
        let ranked = reranker.rank(sequences, &criteria);
        for s in &ranked {
            assert!(
                s.score.composite >= 0.0 && s.score.composite <= 1.0,
                "composite out of range: {}",
                s.score.composite
            );
        }
    }

    #[test]
    fn score_relevance_identical_query() {
        let tokens = seq(&["machine", "learning"]);
        let score = Reranker::score_relevance(&tokens, "machine learning");
        assert!(score > 0.5, "Identical query should yield high relevance: {}", score);
    }

    #[test]
    fn score_diversity_lone_sequence() {
        let tokens = seq(&["hello", "world"]);
        let score = Reranker::score_diversity(&tokens, &[]);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn score_quality_longer_tokens_score_higher() {
        let short_tokens = seq(&["a", "b", "c"]);
        let long_tokens = seq(&["analysis", "calculation", "differentiation"]);
        let q_short = Reranker::score_quality(&short_tokens);
        let q_long = Reranker::score_quality(&long_tokens);
        assert!(q_long > q_short, "longer tokens should score higher: {} vs {}", q_long, q_short);
    }

    #[test]
    fn ranks_assigned_correctly() {
        let reranker = Reranker::new();
        let sequences = vec![
            seq(&["relevant", "machine", "learning"]),
            seq(&["unrelated"]),
        ];
        let criteria = [RankingCriterion::Relevance { query: "machine learning".to_string() }];
        let ranked = reranker.rank(sequences, &criteria);
        assert_eq!(ranked[0].rank, 1);
        assert_eq!(ranked[1].rank, 2);
    }
}
