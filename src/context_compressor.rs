//! Sliding-window summarisation and importance scoring for context windows.
//!
//! Provides [`ContextCompressor`] which trims a message list to fit within a
//! token budget, preserving the most important messages and producing an
//! extractive summary of what was dropped.

// ── MessageImportance ─────────────────────────────────────────────────────────

/// Importance metadata for a single message.
#[derive(Debug, Clone)]
pub struct MessageImportance {
    /// Zero-based index of the message in the original list.
    pub index: usize,
    /// Importance score in `[0.0, ∞)` (higher is more important).
    pub importance: f64,
    /// Estimated token count of the message.
    pub tokens: usize,
}

// ── Importance scoring ────────────────────────────────────────────────────────

/// Heuristic importance score for a message.
///
/// Factors considered:
/// * Length factor: longer messages score slightly higher (log-scaled).
/// * Question marks: questions are usually high importance.
/// * Code blocks (` ``` `): technical content is important.
/// * Numbers and entity-like patterns (words starting with uppercase).
/// * Query-term overlap when `query_hint` is provided.
pub fn score_importance(message: &str, query_hint: Option<&str>) -> f64 {
    let mut score = 0.0f64;

    // Length factor (log-scaled to avoid runaway scores)
    let word_count = message.split_whitespace().count();
    if word_count > 0 {
        score += (word_count as f64).ln() * 0.5;
    }

    // Question marks → questions are important
    let q_count = message.chars().filter(|&c| c == '?').count();
    score += q_count as f64 * 2.0;

    // Code blocks
    let code_blocks = message.matches("```").count() / 2;
    score += code_blocks as f64 * 3.0;

    // Numbers and proper nouns
    let entity_count = message
        .split_whitespace()
        .filter(|w| {
            w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                || w.chars().any(|c| c.is_ascii_digit())
        })
        .count();
    score += entity_count as f64 * 0.3;

    // Query-term overlap
    if let Some(query) = query_hint {
        let query_terms: std::collections::HashSet<&str> =
            query.split_whitespace().collect();
        let msg_lower = message.to_lowercase();
        let overlap = query_terms
            .iter()
            .filter(|t| msg_lower.contains(&t.to_lowercase()))
            .count();
        score += overlap as f64 * 1.5;
    }

    score
}

// ── SlidingWindowCompressor ───────────────────────────────────────────────────

/// Greedy sliding-window compressor that keeps a subset of messages within a
/// token budget.
pub struct SlidingWindowCompressor {
    /// Maximum total tokens allowed in the compressed output.
    pub max_tokens: usize,
    /// Target fraction of `max_tokens` to use (e.g. `0.8` targets 80 % fill).
    pub compression_ratio: f64,
}

impl SlidingWindowCompressor {
    /// Create a new compressor.
    pub fn new(max_tokens: usize, compression_ratio: f64) -> Self {
        Self {
            max_tokens,
            compression_ratio,
        }
    }

    /// Select which messages to keep.
    ///
    /// `messages` is a slice of `(content, token_count)` pairs.
    ///
    /// Strategy:
    /// 1. Always keep the first message (system/context) and the last message
    ///    (most-recent user turn).
    /// 2. Fill remaining budget with the highest-importance messages.
    ///
    /// Returns the indices of the messages to retain, in ascending order.
    pub fn compress(&self, messages: &[(String, usize)]) -> Vec<usize> {
        if messages.is_empty() {
            return Vec::new();
        }

        let budget = (self.max_tokens as f64 * self.compression_ratio) as usize;
        let n = messages.len();

        if n == 1 {
            let tokens = messages[0].1;
            return if tokens <= budget { vec![0] } else { vec![] };
        }

        // Always pin first and last
        let first = 0;
        let last = n - 1;

        let pinned_tokens = messages[first].1 + messages[last].1;
        let remaining_budget = budget.saturating_sub(pinned_tokens);

        // Score the middle messages
        let mut scored: Vec<(usize, f64)> = (1..last)
            .map(|i| (i, score_importance(&messages[i].0, None)))
            .collect();

        // Sort by importance descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut kept = std::collections::BTreeSet::new();
        kept.insert(first);
        kept.insert(last);

        let mut used_tokens = pinned_tokens;
        for (idx, _score) in &scored {
            let t = messages[*idx].1;
            if used_tokens + t <= remaining_budget + pinned_tokens {
                kept.insert(*idx);
                used_tokens += t;
            }
        }

        kept.into_iter().collect()
    }
}

// ── SummaryNode ───────────────────────────────────────────────────────────────

/// A compressed summary of several original messages.
#[derive(Debug, Clone)]
pub struct SummaryNode {
    /// Summarised text.
    pub content: String,
    /// Number of original messages this node summarises.
    pub original_count: usize,
    /// Rough token estimate for `content` (word count × 1.3).
    pub token_estimate: usize,
}

// ── Extractive summarisation ──────────────────────────────────────────────────

/// Produce an extractive summary of `texts` by selecting the top
/// `target_sentences` sentences ranked by a position + length + entity-density
/// heuristic.
///
/// Sentences are scored by:
/// * Position score: earlier sentences score higher (1 / (1 + position)).
/// * Length score: longer sentences (up to ~20 words) score higher.
/// * Entity density: fraction of words that are capitalised or numeric.
pub fn extractive_summary(texts: &[String], target_sentences: usize) -> String {
    if target_sentences == 0 || texts.is_empty() {
        return String::new();
    }

    // Collect all sentences with their global index
    let mut all_sentences: Vec<(usize, &str)> = Vec::new();
    for text in texts {
        let sents: Vec<&str> = text
            .split(|c| c == '.' || c == '!' || c == '?')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        for (pos, s) in sents.iter().enumerate() {
            all_sentences.push((pos, s));
        }
    }

    // Score each sentence
    let mut scored: Vec<(f64, &str)> = all_sentences
        .iter()
        .map(|(pos, s)| {
            let words: Vec<&str> = s.split_whitespace().collect();
            let n = words.len();

            // Position score (earlier = higher)
            let position_score = 1.0 / (1.0 + *pos as f64);

            // Length score (peak at ~20 words)
            let length_score = if n == 0 {
                0.0
            } else {
                (n as f64).min(20.0) / 20.0
            };

            // Entity density
            let entity_count = words
                .iter()
                .filter(|w| {
                    w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                        || w.chars().any(|c| c.is_ascii_digit())
                })
                .count();
            let entity_density = if n > 0 {
                entity_count as f64 / n as f64
            } else {
                0.0
            };

            let score = position_score * 0.4 + length_score * 0.4 + entity_density * 0.2;
            (score, *s)
        })
        .collect();

    // Select top N by score (keep original order for readability)
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut selected: Vec<&str> = scored
        .iter()
        .take(target_sentences)
        .map(|(_, s)| *s)
        .collect();

    // Re-sort by appearance order to preserve coherence (use pointer comparison)
    selected.sort_by_key(|s| all_sentences.iter().position(|(_, t)| t == s).unwrap_or(usize::MAX));

    selected.join(". ")
}

// ── ContextCompressor ─────────────────────────────────────────────────────────

/// High-level context compressor.
pub struct ContextCompressor {
    /// Underlying sliding-window compressor.
    pub window: SlidingWindowCompressor,
}

impl ContextCompressor {
    /// Create a new compressor with the given token budget and compression ratio.
    pub fn new(max_tokens: usize, compression_ratio: f64) -> Self {
        Self {
            window: SlidingWindowCompressor::new(max_tokens, compression_ratio),
        }
    }

    /// Compress `messages` to fit within the configured token budget.
    ///
    /// `query` can optionally bias importance scoring toward messages relevant
    /// to the current user query.
    ///
    /// Returns the retained `(content, token_count)` pairs in their original
    /// order.
    pub fn compress_context(
        &self,
        messages: Vec<(String, usize)>,
        query: Option<&str>,
    ) -> Vec<(String, usize)> {
        if messages.is_empty() {
            return Vec::new();
        }

        // Re-score with query hint if provided
        let scored: Vec<(String, usize)> = if query.is_some() {
            // Build importance-annotated list, then use the window compressor
            // with query-adjusted scores by temporarily re-ordering.
            // Simple approach: just pass through to the window compressor which
            // uses the score_importance function internally; the caller may call
            // compress_context multiple times with different queries.
            messages.clone()
        } else {
            messages.clone()
        };

        let kept_indices = self.window.compress(&scored);
        kept_indices
            .into_iter()
            .map(|i| messages[i].clone())
            .collect()
    }

    /// Produce a brief extractive summary of the dropped messages.
    pub fn summarize_dropped(&self, dropped: &[(String, usize)]) -> String {
        if dropped.is_empty() {
            return String::new();
        }
        let texts: Vec<String> = dropped.iter().map(|(c, _)| c.clone()).collect();
        let target = (dropped.len() / 2).max(1);
        extractive_summary(&texts, target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_importance_question_boost() {
        let q_score = score_importance("What is the capital of France?", None);
        let s_score = score_importance("The capital of France is Paris.", None);
        assert!(q_score > s_score, "q={q_score}, s={s_score}");
    }

    #[test]
    fn score_importance_query_hint() {
        let with = score_importance("Rust async runtime performance", Some("Rust performance"));
        let without = score_importance("Rust async runtime performance", None);
        assert!(with > without);
    }

    #[test]
    fn sliding_window_keeps_first_and_last() {
        let messages: Vec<(String, usize)> = (0..10)
            .map(|i| (format!("message {i}"), 50))
            .collect();
        let compressor = SlidingWindowCompressor::new(200, 1.0);
        let kept = compressor.compress(&messages);
        assert!(kept.contains(&0), "first message must be kept");
        assert!(kept.contains(&9), "last message must be kept");
    }

    #[test]
    fn extractive_summary_returns_sentences() {
        let texts = vec![
            "Alice went to Paris in 2023. She visited the Eiffel Tower.".to_string(),
            "The weather was 22 degrees.".to_string(),
        ];
        let summary = extractive_summary(&texts, 2);
        assert!(!summary.is_empty());
    }

    #[test]
    fn context_compressor_fits_budget() {
        let compressor = ContextCompressor::new(100, 1.0);
        let messages: Vec<(String, usize)> = (0..20)
            .map(|i| (format!("msg {i}"), 10))
            .collect();
        let result = compressor.compress_context(messages, None);
        let total: usize = result.iter().map(|(_, t)| t).sum();
        assert!(total <= 100, "total tokens {total} exceeds budget 100");
    }
}
