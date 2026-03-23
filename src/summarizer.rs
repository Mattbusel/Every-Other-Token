//! Extractive summarization by scoring and selecting top sentences.
//!
//! Sentences are scored using a composite of position, length, keyword presence,
//! and TF-IDF signals, then the top-scoring sentences are returned in original order.

use std::collections::HashMap;

// ── Stopwords ─────────────────────────────────────────────────────────────────

const STOPWORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "and", "or", "but", "in",
    "on", "at", "to", "for", "of", "with", "by", "from", "as", "that",
    "this", "these", "those", "it", "its",
];

fn is_stopword(word: &str) -> bool {
    STOPWORDS.contains(&word)
}

// ── Sentence ──────────────────────────────────────────────────────────────────

/// A single sentence with metadata used during scoring.
#[derive(Debug, Clone)]
pub struct Sentence {
    /// Original sentence text (trimmed).
    pub text: String,
    /// Zero-based position in the source document.
    pub position: usize,
    /// Number of whitespace-delimited words.
    pub word_count: usize,
    /// Composite score in `[0, 1]`.
    pub score: f64,
}

// ── SentenceScorer ────────────────────────────────────────────────────────────

/// Scores individual sentences using multiple independent signals.
pub struct SentenceScorer;

impl SentenceScorer {
    /// U-shaped position score — first and last sentences score highest.
    ///
    /// Formula: `1.0 - 2*|position/total - 0.5| + 0.2`, clamped to `[0, 1]`.
    pub fn position_score(position: usize, total: usize) -> f64 {
        if total == 0 {
            return 1.0;
        }
        let t = position as f64 / total as f64;
        let raw = 1.0 - 2.0 * (t - 0.5).abs() + 0.2;
        raw.clamp(0.0, 1.0)
    }

    /// Gaussian length score centred at `ideal` words with σ = 10.
    pub fn length_score(word_count: usize, ideal: usize) -> f64 {
        let diff = word_count as f64 - ideal as f64;
        (-0.5 * (diff / 10.0).powi(2)).exp()
    }

    /// Fraction of `keywords` that appear (case-insensitive) in `sentence`.
    pub fn keyword_score(sentence: &str, keywords: &[&str]) -> f64 {
        if keywords.is_empty() {
            return 0.0;
        }
        let lower = sentence.to_lowercase();
        let hits = keywords
            .iter()
            .filter(|&&kw| lower.contains(&kw.to_lowercase()))
            .count();
        hits as f64 / keywords.len() as f64
    }

    /// Average TF-IDF score of content words in `sentence` across `all_sentences`.
    ///
    /// TF is computed within the sentence; IDF = log((N+1)/(df+1)) where N is
    /// the number of sentences and df is the document frequency of the word.
    pub fn tfidf_score(sentence: &str, all_sentences: &[&str]) -> f64 {
        let n = all_sentences.len() as f64;
        if n == 0.0 {
            return 0.0;
        }

        // Word frequency within the target sentence.
        let words: Vec<String> = sentence
            .split_whitespace()
            .map(|w| w.to_lowercase().trim_matches(|c: char| !c.is_alphabetic()).to_string())
            .filter(|w| !w.is_empty() && !is_stopword(w))
            .collect();

        if words.is_empty() {
            return 0.0;
        }

        let mut tf_map: HashMap<&str, f64> = HashMap::new();
        for w in &words {
            *tf_map.entry(w.as_str()).or_insert(0.0) += 1.0;
        }
        let total_words = words.len() as f64;
        for v in tf_map.values_mut() {
            *v /= total_words;
        }

        // Document frequency across all sentences.
        let mut df_map: HashMap<String, usize> = HashMap::new();
        for sent in all_sentences {
            let sent_words: std::collections::HashSet<String> = sent
                .split_whitespace()
                .map(|w| w.to_lowercase().trim_matches(|c: char| !c.is_alphabetic()).to_string())
                .filter(|w| !w.is_empty())
                .collect();
            for w in sent_words {
                *df_map.entry(w).or_insert(0) += 1;
            }
        }

        let score_sum: f64 = tf_map
            .iter()
            .map(|(w, tf)| {
                let df = *df_map.get(*w).unwrap_or(&0) as f64;
                let idf = ((n + 1.0) / (df + 1.0)).ln();
                tf * idf
            })
            .sum();

        (score_sum / tf_map.len() as f64).max(0.0)
    }

    /// Weighted composite score:
    /// `position*0.15 + length*0.10 + keyword*0.35 + tfidf*0.40`
    pub fn composite_score(
        sentence: &str,
        position: usize,
        total: usize,
        all_sentences: &[&str],
        keywords: &[&str],
    ) -> f64 {
        let word_count = sentence.split_whitespace().count();
        let p = Self::position_score(position, total);
        let l = Self::length_score(word_count, 20);
        let k = Self::keyword_score(sentence, keywords);
        let t = Self::tfidf_score(sentence, all_sentences);
        p * 0.15 + l * 0.10 + k * 0.35 + t * 0.40
    }
}

// ── ExtractiveSummarizer ──────────────────────────────────────────────────────

/// Selects the most informative sentences from a text.
pub struct ExtractiveSummarizer;

impl ExtractiveSummarizer {
    /// Split `text` into sentences on `.`, `!`, `?`.
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
        let trimmed = current.trim().to_string();
        if !trimmed.is_empty() {
            sentences.push(trimmed);
        }
        sentences
    }

    /// Score and return the top `target_sentences` sentences in original order.
    pub fn summarize(text: &str, target_sentences: usize, keywords: &[&str]) -> Vec<Sentence> {
        let raw = Self::split_sentences(text);
        let total = raw.len();
        let refs: Vec<&str> = raw.iter().map(String::as_str).collect();

        let mut scored: Vec<(usize, Sentence)> = raw
            .iter()
            .enumerate()
            .map(|(pos, s)| {
                let word_count = s.split_whitespace().count();
                let score = SentenceScorer::composite_score(s, pos, total, &refs, keywords);
                (
                    pos,
                    Sentence { text: s.clone(), position: pos, word_count, score },
                )
            })
            .collect();

        // Sort descending by score, pick top N.
        scored.sort_by(|a, b| b.1.score.partial_cmp(&a.1.score).unwrap_or(std::cmp::Ordering::Equal));
        let take = target_sentences.min(scored.len());
        let mut selected: Vec<(usize, Sentence)> = scored.into_iter().take(take).collect();

        // Restore original order.
        selected.sort_by_key(|(pos, _)| *pos);
        selected.into_iter().map(|(_, s)| s).collect()
    }

    /// Select `ratio * total` sentences (rounded up, minimum 1).
    pub fn summarize_ratio(text: &str, ratio: f64, keywords: &[&str]) -> Vec<Sentence> {
        let raw = Self::split_sentences(text);
        let total = raw.len();
        let target = ((total as f64 * ratio).ceil() as usize).max(1);
        Self::summarize(text, target, keywords)
    }

    /// Join selected sentences with a single space, preserving order.
    pub fn to_text(sentences: &[Sentence]) -> String {
        sentences.iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join(" ")
    }
}

// ── KeywordExtractor ──────────────────────────────────────────────────────────

/// Extracts the most important content words from a text using TF-IDF-style scoring.
pub struct KeywordExtractor;

impl KeywordExtractor {
    /// Return the top `top_n` content words by frequency, penalizing stopwords.
    pub fn extract(text: &str, top_n: usize) -> Vec<String> {
        let mut freq: HashMap<String, usize> = HashMap::new();
        for word in text.split_whitespace() {
            let w = word
                .to_lowercase()
                .trim_matches(|c: char| !c.is_alphabetic())
                .to_string();
            if !w.is_empty() {
                *freq.entry(w).or_insert(0) += 1;
            }
        }

        let mut scored: Vec<(String, f64)> = freq
            .into_iter()
            .map(|(w, count)| {
                let penalty = if is_stopword(&w) { 0.1 } else { 1.0 };
                (w, count as f64 * penalty)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(top_n).map(|(w, _)| w).collect()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SHORT_TEXT: &str =
        "The quick brown fox jumps over the lazy dog. \
         Rust is a systems programming language. \
         It is fast and memory safe.";

    #[test]
    fn short_text_extracts_one_sentence() {
        let result = ExtractiveSummarizer::summarize(SHORT_TEXT, 1, &[]);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn keyword_score_counts_occurrences() {
        let sentence = "Rust is fast and safe and great";
        let keywords = &["rust", "safe", "great", "missing"];
        let score = SentenceScorer::keyword_score(sentence, keywords);
        // 3 of 4 keywords present => 0.75
        assert!((score - 0.75).abs() < 1e-9, "score={score}");
    }

    #[test]
    fn ratio_based_selection() {
        // 3 sentences, ratio 0.67 → ceil(2.01) = 3, but let's use 0.34 → ceil(1.02)=2
        let result = ExtractiveSummarizer::summarize_ratio(SHORT_TEXT, 0.4, &[]);
        assert!(result.len() >= 1);
        assert!(result.len() <= 3);
    }

    #[test]
    fn join_preserves_order() {
        let result = ExtractiveSummarizer::summarize(SHORT_TEXT, 3, &[]);
        for (i, s) in result.windows(2).enumerate() {
            assert!(
                s[0].position < s[1].position,
                "out of order at window {i}: {} >= {}",
                s[0].position,
                s[1].position
            );
        }
    }

    #[test]
    fn to_text_joins_with_space() {
        let result = ExtractiveSummarizer::summarize(SHORT_TEXT, 2, &[]);
        let text = ExtractiveSummarizer::to_text(&result);
        assert!(!text.is_empty());
        // Must not start/end with a space.
        assert!(!text.starts_with(' '));
        assert!(!text.ends_with(' '));
    }

    #[test]
    fn keyword_extractor_returns_top_n() {
        let words = KeywordExtractor::extract(SHORT_TEXT, 3);
        assert_eq!(words.len(), 3);
    }
}
