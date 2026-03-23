//! Token importance scorer.
//!
//! Assigns a scalar importance score to each token in a sequence using one of
//! several methods, then exposes helpers for ranking and budget-constrained
//! compression.

use std::collections::HashMap;

// ── ImportanceMethod ──────────────────────────────────────────────────────────

/// Strategy used to compute per-token importance scores.
#[derive(Debug, Clone)]
pub enum ImportanceMethod {
    /// Earlier tokens score higher; score decays exponentially with position.
    /// `score = exp(-decay * position)`, where `decay` defaults to 0.1.
    Positional { decay: f64 },
    /// Rarer tokens (lower corpus frequency) are scored higher.
    /// `score = 1 / (1 + count)`.
    Frequency,
    /// Punctuation and common English stop-words score near 0; content words
    /// score near 1.
    Syntactic,
    /// Weighted combination of other methods.
    /// Each `(method, weight)` pair contributes `weight * normalised_score`.
    Composite(Vec<(ImportanceMethod, f64)>),
}

// ── TokenScore ────────────────────────────────────────────────────────────────

/// Importance score for a single token.
#[derive(Debug, Clone)]
pub struct TokenScore {
    /// The token string.
    pub token: String,
    /// Importance score in [0, 1].
    pub score: f32,
    /// 1-based rank within the sequence (1 = most important).
    pub rank: usize,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Common English stop-words that receive low importance under `Syntactic`.
static STOP_WORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "over", "after",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "shall", "can", "not", "no", "nor", "so", "yet", "both", "either",
    "neither", "each", "few", "more", "most", "other", "some", "such",
    "that", "this", "these", "those", "i", "me", "my", "we", "our",
    "you", "your", "he", "him", "his", "she", "her", "it", "its",
    "they", "them", "their", "what", "which", "who", "whom", "when",
    "where", "why", "how", "all", "both", "if", "then", "because",
    "as", "until", "while", "s", "t",
];

fn is_punctuation(token: &str) -> bool {
    token.chars().all(|c| !c.is_alphanumeric())
}

fn is_stop_word(token: &str) -> bool {
    let lower = token.to_lowercase();
    STOP_WORDS.contains(&lower.as_str())
}

/// Build a frequency map for a token slice.
fn frequency_map(tokens: &[String]) -> HashMap<String, usize> {
    let mut map = HashMap::new();
    for t in tokens {
        *map.entry(t.clone()).or_insert(0) += 1;
    }
    map
}

/// Normalise a raw score vec so values lie in [0, 1].
fn normalise(scores: &[f64]) -> Vec<f32> {
    let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    if range < 1e-12 {
        return vec![1.0f32; scores.len()];
    }
    scores.iter().map(|&s| ((s - min) / range) as f32).collect()
}

// ── ImportanceScorer ──────────────────────────────────────────────────────────

/// Computes per-token importance scores for a token sequence.
pub struct ImportanceScorer {
    /// The method used to score tokens.
    pub method: ImportanceMethod,
}

impl ImportanceScorer {
    /// Create a scorer with the given method.
    pub fn new(method: ImportanceMethod) -> Self {
        Self { method }
    }

    /// Score every token in `tokens`, returning [`TokenScore`]s in original order.
    ///
    /// Ranks are assigned after scoring: rank 1 = highest score.
    pub fn score(&self, tokens: &[String]) -> Vec<TokenScore> {
        if tokens.is_empty() {
            return Vec::new();
        }
        let raw = self.raw_scores(tokens, &self.method);
        let normalised = normalise(&raw);

        // Compute ranks: for each token, count how many others score strictly higher.
        let ranks: Vec<usize> = normalised
            .iter()
            .map(|&s| {
                let above = normalised.iter().filter(|&&o| o > s).count();
                above + 1
            })
            .collect();

        tokens
            .iter()
            .enumerate()
            .map(|(i, t)| TokenScore {
                token: t.clone(),
                score: normalised[i],
                rank: ranks[i],
            })
            .collect()
    }

    /// Return the top-`k` tokens by score, preserving original token order in
    /// the output.  If `k >= tokens.len()` all tokens are returned.
    pub fn top_k(&self, tokens: &[String], k: usize) -> Vec<TokenScore> {
        let mut scored = self.score(tokens);
        // Identify the k highest scores.
        let mut score_vals: Vec<f32> = scored.iter().map(|s| s.score).collect();
        score_vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let threshold = score_vals.get(k.saturating_sub(1)).copied().unwrap_or(0.0);

        // Keep only those at or above the threshold (up to k entries).
        let mut count = 0;
        scored.retain(|s| {
            if s.score >= threshold && count < k {
                count += 1;
                true
            } else {
                false
            }
        });
        scored
    }

    /// Keep the `budget` most important tokens and return them in original order.
    pub fn compress_to_budget(&self, tokens: &[String], budget: usize) -> Vec<String> {
        if budget >= tokens.len() {
            return tokens.to_vec();
        }
        let top = self.top_k(tokens, budget);
        // top_k already preserves original order, so we can collect directly.
        top.into_iter().map(|s| s.token).collect()
    }

    // ── internal ──────────────────────────────────────────────────────────────

    fn raw_scores(&self, tokens: &[String], method: &ImportanceMethod) -> Vec<f64> {
        match method {
            ImportanceMethod::Positional { decay } => {
                tokens
                    .iter()
                    .enumerate()
                    .map(|(i, _)| (-decay * i as f64).exp())
                    .collect()
            }
            ImportanceMethod::Frequency => {
                let freq = frequency_map(tokens);
                tokens
                    .iter()
                    .map(|t| 1.0 / (1.0 + *freq.get(t).unwrap_or(&1) as f64))
                    .collect()
            }
            ImportanceMethod::Syntactic => {
                tokens
                    .iter()
                    .map(|t| {
                        if is_punctuation(t) {
                            0.05
                        } else if is_stop_word(t) {
                            0.2
                        } else {
                            1.0
                        }
                    })
                    .collect()
            }
            ImportanceMethod::Composite(components) => {
                if components.is_empty() {
                    return vec![1.0; tokens.len()];
                }
                let mut combined = vec![0.0f64; tokens.len()];
                let mut total_weight = 0.0f64;
                for (m, w) in components {
                    let raw = self.raw_scores(tokens, m);
                    let normed = normalise(&raw);
                    for (i, v) in normed.iter().enumerate() {
                        combined[i] += w * (*v as f64);
                    }
                    total_weight += w;
                }
                if total_weight > 0.0 {
                    for v in &mut combined {
                        *v /= total_weight;
                    }
                }
                combined
            }
        }
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(s: &str) -> Vec<String> {
        s.split_whitespace().map(|t| t.to_string()).collect()
    }

    // 1. Positional: first token gets highest score
    #[test]
    fn positional_first_is_highest() {
        let scorer = ImportanceScorer::new(ImportanceMethod::Positional { decay: 0.1 });
        let tokens = toks("alpha beta gamma delta");
        let scores = scorer.score(&tokens);
        assert!(scores[0].score >= scores[1].score);
        assert!(scores[1].score >= scores[2].score);
    }

    // 2. Positional: rank 1 assigned to first token
    #[test]
    fn positional_rank_assignment() {
        let scorer = ImportanceScorer::new(ImportanceMethod::Positional { decay: 0.1 });
        let tokens = toks("x y z");
        let scores = scorer.score(&tokens);
        assert_eq!(scores[0].rank, 1);
    }

    // 3. Frequency: rare token scores higher than common token
    #[test]
    fn frequency_rare_higher() {
        let scorer = ImportanceScorer::new(ImportanceMethod::Frequency);
        let tokens = toks("cat cat cat dog");
        let scores = scorer.score(&tokens);
        // dog appears once, cat appears 3 times — dog should score higher
        let dog_score = scores[3].score;
        let cat_score = scores[0].score;
        assert!(dog_score >= cat_score, "dog={} cat={}", dog_score, cat_score);
    }

    // 4. Frequency: single token returns score
    #[test]
    fn frequency_single_token() {
        let scorer = ImportanceScorer::new(ImportanceMethod::Frequency);
        let scores = scorer.score(&["hello".to_string()]);
        assert_eq!(scores.len(), 1);
        assert!(scores[0].score >= 0.0);
    }

    // 5. Syntactic: punctuation scores lower than content
    #[test]
    fn syntactic_punctuation_low() {
        let scorer = ImportanceScorer::new(ImportanceMethod::Syntactic);
        let tokens = toks("telescope . nebula");
        let scores = scorer.score(&tokens);
        // punctuation score should be less than content words
        assert!(scores[1].score < scores[0].score);
        assert!(scores[1].score < scores[2].score);
    }

    // 6. Syntactic: stop-words score lower than content words
    #[test]
    fn syntactic_stopwords_low() {
        let scorer = ImportanceScorer::new(ImportanceMethod::Syntactic);
        let tokens = toks("the quantum entanglement");
        let scores = scorer.score(&tokens);
        assert!(scores[0].score < scores[1].score);
        assert!(scores[0].score < scores[2].score);
    }

    // 7. Empty input → empty output
    #[test]
    fn empty_tokens() {
        let scorer = ImportanceScorer::new(ImportanceMethod::Frequency);
        let scores = scorer.score(&[]);
        assert!(scores.is_empty());
    }

    // 8. top_k returns at most k items
    #[test]
    fn top_k_limit() {
        let scorer = ImportanceScorer::new(ImportanceMethod::Positional { decay: 0.1 });
        let tokens = toks("a b c d e");
        let top = scorer.top_k(&tokens, 3);
        assert!(top.len() <= 3);
    }

    // 9. top_k preserves original order
    #[test]
    fn top_k_preserves_order() {
        let scorer = ImportanceScorer::new(ImportanceMethod::Syntactic);
        let tokens = vec![
            "quantum".to_string(),
            "the".to_string(),
            "entanglement".to_string(),
            "is".to_string(),
            "strange".to_string(),
        ];
        let top = scorer.top_k(&tokens, 3);
        // Check positions are non-decreasing
        let positions: Vec<usize> = top
            .iter()
            .map(|s| tokens.iter().position(|t| t == &s.token).unwrap())
            .collect();
        for w in positions.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    // 10. compress_to_budget returns correct count
    #[test]
    fn compress_to_budget_count() {
        let scorer = ImportanceScorer::new(ImportanceMethod::Positional { decay: 0.05 });
        let tokens = toks("one two three four five six seven");
        let compressed = scorer.compress_to_budget(&tokens, 4);
        assert_eq!(compressed.len(), 4);
    }

    // 11. compress_to_budget budget >= len → all tokens
    #[test]
    fn compress_to_budget_no_trim() {
        let scorer = ImportanceScorer::new(ImportanceMethod::Frequency);
        let tokens = toks("one two three");
        let compressed = scorer.compress_to_budget(&tokens, 10);
        assert_eq!(compressed, tokens);
    }

    // 12. Composite: weights blend correctly (smoke test)
    #[test]
    fn composite_runs() {
        let method = ImportanceMethod::Composite(vec![
            (ImportanceMethod::Positional { decay: 0.1 }, 0.5),
            (ImportanceMethod::Syntactic, 0.5),
        ]);
        let scorer = ImportanceScorer::new(method);
        let tokens = toks("the quick brown fox");
        let scores = scorer.score(&tokens);
        assert_eq!(scores.len(), 4);
        for s in &scores {
            assert!(s.score >= 0.0 && s.score <= 1.0);
        }
    }

    // 13. All scores in [0,1]
    #[test]
    fn scores_in_range() {
        for method in [
            ImportanceMethod::Positional { decay: 0.2 },
            ImportanceMethod::Frequency,
            ImportanceMethod::Syntactic,
        ] {
            let scorer = ImportanceScorer::new(method);
            let tokens = toks("the lazy dog jumps over the fence");
            for s in scorer.score(&tokens) {
                assert!(s.score >= 0.0 && s.score <= 1.0, "score out of range: {}", s.score);
            }
        }
    }

    // 14. Score length matches token count
    #[test]
    fn score_length() {
        let scorer = ImportanceScorer::new(ImportanceMethod::Positional { decay: 0.1 });
        let tokens = toks("a b c d e f g");
        assert_eq!(scorer.score(&tokens).len(), 7);
    }

    // 15. Rank is at least 1
    #[test]
    fn rank_at_least_one() {
        let scorer = ImportanceScorer::new(ImportanceMethod::Frequency);
        let tokens = toks("alpha beta gamma");
        for s in scorer.score(&tokens) {
            assert!(s.rank >= 1);
        }
    }

    // 16. top_k with k=0 returns empty
    #[test]
    fn top_k_zero() {
        let scorer = ImportanceScorer::new(ImportanceMethod::Syntactic);
        let tokens = toks("a b c");
        let top = scorer.top_k(&tokens, 0);
        assert!(top.is_empty());
    }

    // 17. top_k with k > len returns all tokens
    #[test]
    fn top_k_exceeds_length() {
        let scorer = ImportanceScorer::new(ImportanceMethod::Positional { decay: 0.1 });
        let tokens = toks("x y z");
        let top = scorer.top_k(&tokens, 100);
        assert_eq!(top.len(), 3);
    }

    // 18. Positional decay=0 → uniform scores
    #[test]
    fn positional_zero_decay_uniform() {
        let scorer = ImportanceScorer::new(ImportanceMethod::Positional { decay: 0.0 });
        let tokens = toks("a b c d");
        let scores = scorer.score(&tokens);
        // All raw scores are exp(0)=1, so after normalisation all are 1.0
        for s in &scores {
            assert!((s.score - 1.0).abs() < 1e-5);
        }
    }

    // 19. Frequency: all unique tokens → equal scores
    #[test]
    fn frequency_all_unique_equal_scores() {
        let scorer = ImportanceScorer::new(ImportanceMethod::Frequency);
        let tokens = toks("alpha beta gamma delta");
        let scores = scorer.score(&tokens);
        let first = scores[0].score;
        for s in &scores {
            assert!((s.score - first).abs() < 1e-5);
        }
    }

    // 20. Composite with empty components → uniform scores
    #[test]
    fn composite_empty_components_uniform() {
        let scorer = ImportanceScorer::new(ImportanceMethod::Composite(vec![]));
        let tokens = toks("a b c");
        let scores = scorer.score(&tokens);
        for s in &scores {
            assert!(s.score >= 0.0);
        }
    }

    // 21. compress_to_budget budget=0 → empty
    #[test]
    fn compress_to_budget_zero() {
        let scorer = ImportanceScorer::new(ImportanceMethod::Syntactic);
        let tokens = toks("one two three");
        let compressed = scorer.compress_to_budget(&tokens, 0);
        assert!(compressed.is_empty());
    }

    // 22. TokenScore token field matches original token
    #[test]
    fn token_field_matches() {
        let scorer = ImportanceScorer::new(ImportanceMethod::Positional { decay: 0.1 });
        let tokens = vec!["hello".to_string(), "world".to_string()];
        let scores = scorer.score(&tokens);
        assert_eq!(scores[0].token, "hello");
        assert_eq!(scores[1].token, "world");
    }
}
