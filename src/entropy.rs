//! # Prompt Entropy Analyzer
//!
//! Measures information entropy of LLM prompts and responses, detects
//! repetition loops, and tracks how entropy evolves across multi-turn
//! conversations.
//!
//! ## Key Types
//!
//! - [`ShannonEntropy`] - character- and token-level Shannon entropy calculator
//! - [`PerplexityEstimator`] - approximates perplexity using unigram statistics
//! - [`RepetitionDetector`] - detects repetitive n-gram loops in model output
//! - [`EntropyTimeline`] - tracks entropy per turn across a conversation
//! - [`PromptEntropy`] - unified facade over all of the above
//!
//! ## Alert behaviour
//!
//! [`EntropyTimeline::check_alert`] fires when the most recent turn's entropy
//! drops below the configured threshold, which is a strong signal that the
//! model is stuck in a repetition loop.

use std::collections::HashMap;

// ── ShannonEntropy ────────────────────────────────────────────────────────────

/// Computes Shannon entropy at character or token level.
///
/// Shannon entropy is defined as:
///
/// ```text
/// H(X) = -Σ p(x) * log2(p(x))
/// ```
///
/// A value of `0.0` means the input is perfectly uniform (single symbol
/// repeated); higher values indicate richer, more varied content.
#[derive(Debug, Clone, Default)]
pub struct ShannonEntropy;

impl ShannonEntropy {
    /// Create a new [`ShannonEntropy`] calculator.
    pub fn new() -> Self {
        Self
    }

    /// Compute Shannon entropy over the **characters** in `text`.
    ///
    /// Returns `0.0` for empty input.
    pub fn char_entropy(&self, text: &str) -> f64 {
        if text.is_empty() {
            return 0.0;
        }
        let mut counts: HashMap<char, usize> = HashMap::new();
        for ch in text.chars() {
            *counts.entry(ch).or_insert(0) += 1;
        }
        let total = text.chars().count() as f64;
        self.entropy_from_counts(counts.values().copied(), total)
    }

    /// Compute Shannon entropy over whitespace-delimited **tokens** (words) in `text`.
    ///
    /// Returns `0.0` for empty input.
    pub fn token_entropy(&self, text: &str) -> f64 {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        if tokens.is_empty() {
            return 0.0;
        }
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for tok in &tokens {
            *counts.entry(tok).or_insert(0) += 1;
        }
        let total = tokens.len() as f64;
        self.entropy_from_counts(counts.values().copied(), total)
    }

    /// Compute the entropy of a sequence of already-counted symbols.
    ///
    /// `total` must equal the sum of all counts.  Returns `0.0` when
    /// `total` is zero.
    pub fn entropy_from_counts(
        &self,
        counts: impl Iterator<Item = usize>,
        total: f64,
    ) -> f64 {
        if total <= 0.0 {
            return 0.0;
        }
        counts.fold(0.0, |acc, c| {
            if c == 0 {
                acc
            } else {
                let p = c as f64 / total;
                acc - p * p.log2()
            }
        })
    }
}

// ── PerplexityEstimator ───────────────────────────────────────────────────────

/// Approximates text perplexity using a unigram language model built from a
/// reference corpus.
///
/// Perplexity is the exponentiation of the average cross-entropy of the text
/// given the unigram model:
///
/// ```text
/// PP(text) = 2^( -1/N * Σ log2 P(w_i) )
/// ```
///
/// This is an approximation — a real perplexity would require a neural LM —
/// but unigram-based perplexity is fast, deterministic, and useful for
/// detecting out-of-distribution tokens.
#[derive(Debug, Clone)]
pub struct PerplexityEstimator {
    /// Unigram log-probabilities keyed by token.
    log_probs: HashMap<String, f64>,
    /// Smoothed log-probability for unseen tokens.
    unk_log_prob: f64,
    /// Total tokens in the reference corpus.
    total_tokens: usize,
}

impl PerplexityEstimator {
    /// Build an estimator from a reference corpus.
    ///
    /// `corpus` is any iterator over token strings (e.g. `text.split_whitespace()`).
    /// Applies +1 Laplace smoothing so that unseen tokens receive a small
    /// non-zero probability.
    pub fn from_corpus<I, S>(tokens: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut total = 0usize;
        for tok in tokens {
            *counts.entry(tok.as_ref().to_lowercase()).or_insert(0) += 1;
            total += 1;
        }

        // Laplace smoothing: add 1 to every count; vocabulary size grows by 1
        // for the implicit <UNK> token.
        let vocab_size = counts.len() + 1; // +1 for <UNK>
        let smoothed_total = (total + vocab_size) as f64;

        let log_probs = counts
            .iter()
            .map(|(tok, &c)| {
                let p = (c + 1) as f64 / smoothed_total;
                (tok.clone(), p.log2())
            })
            .collect();

        let unk_log_prob = (1.0_f64 / smoothed_total).log2();

        Self {
            log_probs,
            unk_log_prob,
            total_tokens: total,
        }
    }

    /// Estimate the perplexity of `text` against the reference unigram model.
    ///
    /// Returns `f64::INFINITY` for empty input (undefined perplexity).
    pub fn perplexity(&self, text: &str) -> f64 {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        if tokens.is_empty() {
            return f64::INFINITY;
        }
        let n = tokens.len() as f64;
        let sum_log_prob: f64 = tokens.iter().map(|t| {
            *self
                .log_probs
                .get(&t.to_lowercase())
                .unwrap_or(&self.unk_log_prob)
        }).sum();
        let avg_cross_entropy = -sum_log_prob / n;
        2_f64.powf(avg_cross_entropy)
    }

    /// The size of the vocabulary seen during corpus construction.
    pub fn vocab_size(&self) -> usize {
        self.log_probs.len()
    }

    /// Total tokens in the reference corpus.
    pub fn corpus_size(&self) -> usize {
        self.total_tokens
    }
}

// ── RepetitionDetector ────────────────────────────────────────────────────────

/// The result of a repetition analysis on a piece of text.
#[derive(Debug, Clone)]
pub struct RepetitionReport {
    /// The n-gram order used for the analysis.
    pub ngram_n: usize,
    /// Total number of n-grams in the input.
    pub total_ngrams: usize,
    /// Number of distinct n-grams.
    pub distinct_ngrams: usize,
    /// Repetition rate: `1.0 - (distinct / total)`.  `0.0` = no repetition.
    pub repetition_rate: f64,
    /// Whether the repetition rate exceeded the configured threshold.
    pub is_repetitive: bool,
    /// The most-repeated n-gram and its count (if any).
    pub top_repeated: Option<(Vec<String>, usize)>,
}

/// Detects repetitive loops in model output using n-gram overlap statistics.
///
/// A high repetition rate (many duplicate n-grams) is a reliable signal that
/// the model has entered a degenerate repetition loop.
#[derive(Debug, Clone)]
pub struct RepetitionDetector {
    /// N-gram order (default: 3).
    n: usize,
    /// Repetition rate above which [`RepetitionReport::is_repetitive`] is set.
    threshold: f64,
}

impl Default for RepetitionDetector {
    fn default() -> Self {
        Self::new(3, 0.3)
    }
}

impl RepetitionDetector {
    /// Create a detector with the given n-gram order and threshold.
    ///
    /// `threshold` should be in `[0.0, 1.0]`.  A value of `0.3` means
    /// "flag as repetitive when 30 % or more of n-grams are duplicates".
    pub fn new(n: usize, threshold: f64) -> Self {
        let n = n.max(1);
        let threshold = threshold.clamp(0.0, 1.0);
        Self { n, threshold }
    }

    /// Analyse `text` for n-gram repetition.
    ///
    /// Tokenises by whitespace.  Returns a [`RepetitionReport`] regardless of
    /// text length; very short inputs will have `total_ngrams = 0`.
    pub fn analyse(&self, text: &str) -> RepetitionReport {
        let tokens: Vec<String> = text
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();

        if tokens.len() < self.n {
            return RepetitionReport {
                ngram_n: self.n,
                total_ngrams: 0,
                distinct_ngrams: 0,
                repetition_rate: 0.0,
                is_repetitive: false,
                top_repeated: None,
            };
        }

        let ngrams: Vec<Vec<String>> = tokens
            .windows(self.n)
            .map(|w| w.to_vec())
            .collect();

        let total = ngrams.len();
        let mut counts: HashMap<Vec<String>, usize> = HashMap::new();
        for ng in &ngrams {
            *counts.entry(ng.clone()).or_insert(0) += 1;
        }

        let distinct = counts.len();
        let repetition_rate = if total > 0 {
            1.0 - distinct as f64 / total as f64
        } else {
            0.0
        };

        let top_repeated = counts
            .iter()
            .filter(|(_, &c)| c > 1)
            .max_by_key(|(_, &c)| c)
            .map(|(ng, &c)| (ng.clone(), c));

        RepetitionReport {
            ngram_n: self.n,
            total_ngrams: total,
            distinct_ngrams: distinct,
            repetition_rate,
            is_repetitive: repetition_rate >= self.threshold,
            top_repeated,
        }
    }
}

// ── EntropyTimeline ───────────────────────────────────────────────────────────

/// A single turn in a multi-turn conversation with its associated entropy.
#[derive(Debug, Clone)]
pub struct TurnEntropy {
    /// Zero-based turn index.
    pub turn: usize,
    /// Whether this turn was produced by the user (`true`) or the model (`false`).
    pub is_user: bool,
    /// Token-level Shannon entropy of this turn's text.
    pub token_entropy: f64,
    /// Character-level Shannon entropy of this turn's text.
    pub char_entropy: f64,
    /// N-gram repetition rate for this turn (using the default 3-gram detector).
    pub repetition_rate: f64,
}

/// Tracks how entropy evolves across each turn of a multi-turn conversation.
///
/// Fires an alert when entropy drops below a configured minimum, indicating
/// the model may be producing degenerate repetitive output.
///
/// # Example
///
/// ```rust
/// use every_other_token::entropy::EntropyTimeline;
///
/// let mut timeline = EntropyTimeline::new(1.5);
/// timeline.push("Hello, how are you?", true);
/// timeline.push("I am doing well, thank you for asking.", false);
///
/// let alerts = timeline.check_alerts();
/// // No alerts — both turns have healthy entropy.
/// assert!(alerts.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct EntropyTimeline {
    turns: Vec<TurnEntropy>,
    /// Entropy below this value triggers an alert.
    alert_threshold: f64,
    entropy_calc: ShannonEntropy,
    repetition: RepetitionDetector,
}

impl EntropyTimeline {
    /// Create a timeline with the given entropy alert threshold.
    ///
    /// A threshold of `1.5` bits is a reasonable default for token-level
    /// entropy — values below that suggest high repetition.
    pub fn new(alert_threshold: f64) -> Self {
        Self {
            turns: Vec::new(),
            alert_threshold,
            entropy_calc: ShannonEntropy::new(),
            repetition: RepetitionDetector::default(),
        }
    }

    /// Append a turn to the timeline.
    ///
    /// - `text` — the raw text of this turn.
    /// - `is_user` — `true` if the turn was produced by the human, `false` if
    ///   produced by the model.
    pub fn push(&mut self, text: &str, is_user: bool) {
        let turn = self.turns.len();
        let te = self.entropy_calc.token_entropy(text);
        let ce = self.entropy_calc.char_entropy(text);
        let rep = self.repetition.analyse(text).repetition_rate;
        self.turns.push(TurnEntropy {
            turn,
            is_user,
            token_entropy: te,
            char_entropy: ce,
            repetition_rate: rep,
        });
    }

    /// Return references to all recorded turns.
    pub fn turns(&self) -> &[TurnEntropy] {
        &self.turns
    }

    /// Return the number of turns recorded so far.
    pub fn len(&self) -> usize {
        self.turns.len()
    }

    /// `true` when no turns have been recorded.
    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    /// Return alerts for any turns whose token entropy is below the threshold.
    ///
    /// Only model turns (`is_user == false`) are checked; user turns are
    /// typically short and may legitimately have low entropy.
    pub fn check_alerts(&self) -> Vec<EntropyAlert> {
        self.turns
            .iter()
            .filter(|t| !t.is_user && t.token_entropy < self.alert_threshold)
            .map(|t| EntropyAlert {
                turn: t.turn,
                token_entropy: t.token_entropy,
                threshold: self.alert_threshold,
                repetition_rate: t.repetition_rate,
            })
            .collect()
    }

    /// Moving average of token entropy over the last `window` model turns.
    ///
    /// Returns `None` when fewer than one model turn has been recorded.
    pub fn rolling_entropy(&self, window: usize) -> Option<f64> {
        let model_turns: Vec<f64> = self
            .turns
            .iter()
            .filter(|t| !t.is_user)
            .map(|t| t.token_entropy)
            .collect();
        if model_turns.is_empty() {
            return None;
        }
        let window = window.min(model_turns.len());
        let slice = &model_turns[model_turns.len() - window..];
        Some(slice.iter().sum::<f64>() / slice.len() as f64)
    }
}

/// An alert fired when a model turn's entropy drops below the threshold.
#[derive(Debug, Clone)]
pub struct EntropyAlert {
    /// Turn index that triggered the alert.
    pub turn: usize,
    /// Observed token-level entropy (bits).
    pub token_entropy: f64,
    /// The threshold that was breached.
    pub threshold: f64,
    /// N-gram repetition rate of the flagged turn.
    pub repetition_rate: f64,
}

impl std::fmt::Display for EntropyAlert {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "EntropyAlert[turn={}]: entropy={:.3} bits < threshold={:.3} bits, repetition={:.1}%",
            self.turn,
            self.token_entropy,
            self.threshold,
            self.repetition_rate * 100.0,
        )
    }
}

// ── PromptEntropy ─────────────────────────────────────────────────────────────

/// Unified facade for analysing the information entropy of LLM prompts and
/// responses.
///
/// Bundles [`ShannonEntropy`], [`PerplexityEstimator`], [`RepetitionDetector`],
/// and [`EntropyTimeline`] behind a single API.
///
/// # Example
///
/// ```rust
/// use every_other_token::entropy::PromptEntropy;
///
/// let corpus = "the quick brown fox jumps over the lazy dog".split_whitespace();
/// let mut pe = PromptEntropy::new(corpus, 1.5);
///
/// let stats = pe.analyse("the the the the the");
/// assert!(stats.repetition.is_repetitive);
/// ```
pub struct PromptEntropy {
    /// Shannon entropy calculator.
    pub shannon: ShannonEntropy,
    /// Unigram perplexity estimator.
    pub perplexity: PerplexityEstimator,
    /// Repetition detector (default: 3-gram, 30 % threshold).
    pub repetition_detector: RepetitionDetector,
    /// Running conversation timeline.
    pub timeline: EntropyTimeline,
}

/// Statistics produced by [`PromptEntropy::analyse`].
#[derive(Debug, Clone)]
pub struct EntropyStats {
    /// Token-level Shannon entropy (bits).
    pub token_entropy: f64,
    /// Character-level Shannon entropy (bits).
    pub char_entropy: f64,
    /// Estimated perplexity against the reference corpus.
    pub perplexity: f64,
    /// N-gram repetition analysis.
    pub repetition: RepetitionReport,
}

impl PromptEntropy {
    /// Create a new analyser.
    ///
    /// - `corpus` — reference token iterator used to build the unigram model.
    /// - `alert_threshold` — entropy level (bits) below which alerts fire.
    pub fn new<I, S>(corpus: I, alert_threshold: f64) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self {
            shannon: ShannonEntropy::new(),
            perplexity: PerplexityEstimator::from_corpus(corpus),
            repetition_detector: RepetitionDetector::default(),
            timeline: EntropyTimeline::new(alert_threshold),
        }
    }

    /// Analyse `text` and return a full [`EntropyStats`] snapshot.
    ///
    /// Does **not** append to the timeline; use [`PromptEntropy::push_turn`]
    /// for that.
    pub fn analyse(&self, text: &str) -> EntropyStats {
        EntropyStats {
            token_entropy: self.shannon.token_entropy(text),
            char_entropy: self.shannon.char_entropy(text),
            perplexity: self.perplexity.perplexity(text),
            repetition: self.repetition_detector.analyse(text),
        }
    }

    /// Append `text` as a conversation turn and return its stats.
    ///
    /// - `is_user` — `true` for human turns, `false` for model turns.
    pub fn push_turn(&mut self, text: &str, is_user: bool) -> EntropyStats {
        self.timeline.push(text, is_user);
        self.analyse(text)
    }

    /// Return all pending entropy alerts from the timeline.
    pub fn alerts(&self) -> Vec<EntropyAlert> {
        self.timeline.check_alerts()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ShannonEntropy ──────────────────────────────────────────────────────

    #[test]
    fn test_char_entropy_empty() {
        let se = ShannonEntropy::new();
        assert_eq!(se.char_entropy(""), 0.0);
    }

    #[test]
    fn test_char_entropy_uniform() {
        let se = ShannonEntropy::new();
        // All same characters → entropy = 0
        assert_eq!(se.char_entropy("aaaa"), 0.0);
    }

    #[test]
    fn test_char_entropy_two_equal_symbols() {
        let se = ShannonEntropy::new();
        // Two equally probable symbols → entropy = 1 bit
        let h = se.char_entropy("ab");
        assert!((h - 1.0).abs() < 1e-9, "expected 1.0 bit, got {h}");
    }

    #[test]
    fn test_token_entropy_empty() {
        let se = ShannonEntropy::new();
        assert_eq!(se.token_entropy(""), 0.0);
    }

    #[test]
    fn test_token_entropy_single() {
        let se = ShannonEntropy::new();
        assert_eq!(se.token_entropy("hello hello hello"), 0.0);
    }

    #[test]
    fn test_token_entropy_increases_with_variety() {
        let se = ShannonEntropy::new();
        let low = se.token_entropy("a a a a b");
        let high = se.token_entropy("a b c d e f g h i j");
        assert!(high > low, "high={high} should exceed low={low}");
    }

    // ── PerplexityEstimator ─────────────────────────────────────────────────

    #[test]
    fn test_perplexity_empty_text() {
        let corpus = "the quick brown fox".split_whitespace();
        let est = PerplexityEstimator::from_corpus(corpus);
        assert!(est.perplexity("").is_infinite());
    }

    #[test]
    fn test_perplexity_known_vs_unknown() {
        let corpus = "hello world hello world hello".split_whitespace();
        let est = PerplexityEstimator::from_corpus(corpus);
        let low = est.perplexity("hello world");
        let high = est.perplexity("xyzzy frobnicate");
        assert!(
            high > low,
            "unseen tokens should have higher perplexity: low={low}, high={high}"
        );
    }

    #[test]
    fn test_perplexity_estimator_vocab_size() {
        let corpus = "a b c d e".split_whitespace();
        let est = PerplexityEstimator::from_corpus(corpus);
        assert_eq!(est.vocab_size(), 5);
        assert_eq!(est.corpus_size(), 5);
    }

    // ── RepetitionDetector ──────────────────────────────────────────────────

    #[test]
    fn test_repetition_no_repetition() {
        let det = RepetitionDetector::default();
        let report = det.analyse("the quick brown fox jumps over the lazy dog");
        assert!(!report.is_repetitive, "diverse text should not be flagged");
    }

    #[test]
    fn test_repetition_pure_loop() {
        let det = RepetitionDetector::new(2, 0.5);
        // "a b" repeated many times → near-100 % repetition rate
        let text = "a b a b a b a b a b a b a b a b";
        let report = det.analyse(text);
        assert!(report.is_repetitive, "repeated bigrams should be flagged");
        assert!(report.repetition_rate > 0.5);
    }

    #[test]
    fn test_repetition_short_input() {
        let det = RepetitionDetector::new(3, 0.3);
        let report = det.analyse("hello");
        assert_eq!(report.total_ngrams, 0);
        assert!(!report.is_repetitive);
    }

    #[test]
    fn test_repetition_top_repeated_present() {
        let det = RepetitionDetector::new(2, 0.1);
        let report = det.analyse("cat dog cat dog cat dog cat");
        assert!(report.top_repeated.is_some());
        let (ngram, count) = report.top_repeated.unwrap();
        assert!(count >= 3);
        assert_eq!(ngram.len(), 2);
    }

    // ── EntropyTimeline ─────────────────────────────────────────────────────

    #[test]
    fn test_timeline_len() {
        let mut tl = EntropyTimeline::new(1.5);
        assert!(tl.is_empty());
        tl.push("Hello there", true);
        tl.push("Hi, how can I help?", false);
        assert_eq!(tl.len(), 2);
    }

    #[test]
    fn test_timeline_no_alert_for_healthy_model_turn() {
        let mut tl = EntropyTimeline::new(0.5);
        // Rich, diverse model response — should not trigger alert
        tl.push("I can help you with many things including coding writing and analysis", false);
        assert!(
            tl.check_alerts().is_empty(),
            "diverse model turn should not alert"
        );
    }

    #[test]
    fn test_timeline_alert_for_repetitive_model_turn() {
        let mut tl = EntropyTimeline::new(2.0);
        // Repetitive model response: single word repeated → entropy = 0 < threshold
        tl.push("sorry sorry sorry sorry sorry sorry sorry sorry", false);
        let alerts = tl.check_alerts();
        assert!(
            !alerts.is_empty(),
            "repetitive model turn should fire alert"
        );
    }

    #[test]
    fn test_timeline_user_turns_never_alert() {
        let mut tl = EntropyTimeline::new(10.0); // extremely high threshold
        tl.push("ok", true); // low entropy but user turn
        assert!(
            tl.check_alerts().is_empty(),
            "user turns should never trigger alerts"
        );
    }

    #[test]
    fn test_timeline_rolling_entropy_none_when_empty() {
        let tl = EntropyTimeline::new(1.5);
        assert!(tl.rolling_entropy(3).is_none());
    }

    #[test]
    fn test_timeline_rolling_entropy_value() {
        let mut tl = EntropyTimeline::new(1.5);
        tl.push("the quick brown fox", false);
        tl.push("hello world goodbye", false);
        let avg = tl.rolling_entropy(2);
        assert!(avg.is_some());
        assert!(avg.unwrap() > 0.0);
    }

    // ── PromptEntropy (facade) ──────────────────────────────────────────────

    #[test]
    fn test_prompt_entropy_analyse() {
        let corpus = "the quick brown fox jumps over the lazy dog".split_whitespace();
        let pe = PromptEntropy::new(corpus, 1.5);
        let stats = pe.analyse("the quick brown fox");
        assert!(stats.token_entropy >= 0.0);
        assert!(stats.char_entropy >= 0.0);
        assert!(stats.perplexity > 0.0);
    }

    #[test]
    fn test_prompt_entropy_push_turn_adds_to_timeline() {
        let corpus = "a b c d e".split_whitespace();
        let mut pe = PromptEntropy::new(corpus, 1.5);
        pe.push_turn("hello world", true);
        pe.push_turn("how are you today", false);
        assert_eq!(pe.timeline.len(), 2);
    }

    #[test]
    fn test_prompt_entropy_repetitive_text_flagged() {
        let corpus = "the the the".split_whitespace();
        let pe = PromptEntropy::new(corpus, 1.5);
        let stats = pe.analyse("the the the the the the the the the the");
        assert!(
            stats.repetition.is_repetitive,
            "all-same-token text must be flagged"
        );
    }

    #[test]
    fn test_entropy_alert_display() {
        let alert = EntropyAlert {
            turn: 3,
            token_entropy: 0.8,
            threshold: 1.5,
            repetition_rate: 0.72,
        };
        let s = alert.to_string();
        assert!(s.contains("turn=3"));
        assert!(s.contains("0.800"));
    }
}
