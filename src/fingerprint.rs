//! # Model Fingerprinting
//!
//! Builds statistical signatures ("fingerprints") of model output distributions
//! and uses them to identify which known model produced an unknown token stream.
//!
//! ## Key Types
//!
//! - [`VocabBias`] - tracks which tokens a model overuses relative to a flat baseline
//! - [`StyleSignature`] - captures sentence length, punctuation frequency, and
//!   capitalisation patterns
//! - [`ModelFingerprint`] - combines [`VocabBias`] and [`StyleSignature`] into a
//!   single statistical signature for one model
//! - [`FingerprintDatabase`] - stores named model fingerprints
//! - [`FingerprintMatcher`] - given an unknown stream, identifies the closest
//!   known model by cosine similarity / L1 distance
//!
//! ## Blind A/B testing
//!
//! The matcher enables blind A/B tests: after collecting responses from two
//! unlabelled models, [`FingerprintMatcher::identify`] returns a scored ranking
//! so you can discover whether users can distinguish the models without knowing
//! which is which.

use std::collections::HashMap;

// ── VocabBias ─────────────────────────────────────────────────────────────────

/// Token over-use bias for a single model.
///
/// For each token the bias is:
///
/// ```text
/// bias(t) = observed_frequency(t) - expected_frequency(t)
/// ```
///
/// where `expected_frequency` is the frequency in a reference corpus (or a
/// uniform `1 / V` baseline when no reference is provided).
///
/// Positive values mean the model uses that token more than expected; negative
/// values mean it avoids it.
#[derive(Debug, Clone, Default)]
pub struct VocabBias {
    /// Raw token frequencies observed in the model's output (count per token).
    pub observed_counts: HashMap<String, usize>,
    /// Total tokens observed.
    pub total_observed: usize,
    /// Reference frequencies (count per token).  When empty, a uniform
    /// baseline is used.
    pub reference_counts: HashMap<String, usize>,
    /// Total tokens in the reference corpus.
    pub total_reference: usize,
}

impl VocabBias {
    /// Create an empty bias tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record tokens from a piece of model output.
    pub fn observe(&mut self, text: &str) {
        for tok in text.split_whitespace() {
            *self.observed_counts.entry(tok.to_lowercase()).or_insert(0) += 1;
            self.total_observed += 1;
        }
    }

    /// Set the reference corpus from a token iterator.
    pub fn set_reference<I, S>(&mut self, tokens: I)
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.reference_counts.clear();
        self.total_reference = 0;
        for tok in tokens {
            *self
                .reference_counts
                .entry(tok.as_ref().to_lowercase())
                .or_insert(0) += 1;
            self.total_reference += 1;
        }
    }

    /// Compute the signed bias for a specific token.
    ///
    /// Returns `0.0` when `total_observed` is zero.
    pub fn bias(&self, token: &str) -> f64 {
        if self.total_observed == 0 {
            return 0.0;
        }
        let obs_freq = *self.observed_counts.get(token).unwrap_or(&0) as f64
            / self.total_observed as f64;
        let ref_freq = if self.total_reference > 0 {
            *self.reference_counts.get(token).unwrap_or(&0) as f64
                / self.total_reference as f64
        } else {
            // Uniform baseline: 1 / vocab_size (or 0 if vocab is empty)
            let v = self.observed_counts.len();
            if v > 0 { 1.0 / v as f64 } else { 0.0 }
        };
        obs_freq - ref_freq
    }

    /// Return the top `n` over-used tokens sorted by descending bias.
    pub fn top_overused(&self, n: usize) -> Vec<(String, f64)> {
        let mut biases: Vec<(String, f64)> = self
            .observed_counts
            .keys()
            .map(|tok| (tok.clone(), self.bias(tok)))
            .filter(|(_, b)| *b > 0.0)
            .collect();
        biases.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        biases.truncate(n);
        biases
    }

    /// Produce a normalised frequency vector for all observed tokens.
    ///
    /// The order follows a sorted key iteration for reproducibility.
    pub fn frequency_vector(&self) -> Vec<f64> {
        if self.total_observed == 0 {
            return Vec::new();
        }
        let total = self.total_observed as f64;
        let mut keys: Vec<&String> = self.observed_counts.keys().collect();
        keys.sort();
        keys.iter()
            .map(|k| *self.observed_counts.get(*k).unwrap_or(&0) as f64 / total)
            .collect()
    }
}

// ── StyleSignature ────────────────────────────────────────────────────────────

/// Stylometric signature capturing surface-level writing patterns.
///
/// These features are model-discriminative even when the content varies: a
/// model that tends to write short sentences, avoid exclamation marks, and
/// capitalise proper nouns consistently will have a distinctive [`StyleSignature`].
#[derive(Debug, Clone, Default)]
pub struct StyleSignature {
    /// Mean sentence length in whitespace-delimited tokens.
    pub mean_sentence_len: f64,
    /// Standard deviation of sentence lengths.
    pub stddev_sentence_len: f64,
    /// Fraction of characters that are punctuation (`.`, `,`, `!`, `?`, `;`, `:`).
    pub punctuation_density: f64,
    /// Fraction of tokens that start with an upper-case letter.
    pub capitalisation_rate: f64,
    /// Fraction of sentences that end with `?`.
    pub question_rate: f64,
    /// Fraction of sentences that end with `!`.
    pub exclamation_rate: f64,
    /// Average word length in characters.
    pub mean_word_len: f64,
    /// Total characters processed.
    pub total_chars: usize,
    /// Total sentences processed.
    pub total_sentences: usize,
}

impl StyleSignature {
    /// Build a [`StyleSignature`] from a body of text.
    pub fn from_text(text: &str) -> Self {
        if text.is_empty() {
            return Self::default();
        }

        // Split into sentences on `.`, `!`, `?` terminators.
        let sentences: Vec<&str> = text
            .split(|c| c == '.' || c == '!' || c == '?')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        let total_sentences = sentences.len();

        // Sentence lengths in tokens.
        let lengths: Vec<usize> = sentences
            .iter()
            .map(|s| s.split_whitespace().count())
            .collect();

        let mean_sentence_len = if total_sentences > 0 {
            lengths.iter().sum::<usize>() as f64 / total_sentences as f64
        } else {
            0.0
        };

        let stddev_sentence_len = if total_sentences > 1 {
            let variance = lengths.iter().map(|&l| {
                let d = l as f64 - mean_sentence_len;
                d * d
            }).sum::<f64>() / total_sentences as f64;
            variance.sqrt()
        } else {
            0.0
        };

        // Punctuation density.
        let total_chars = text.len();
        let punct_count = text
            .chars()
            .filter(|c| matches!(c, '.' | ',' | '!' | '?' | ';' | ':'))
            .count();
        let punctuation_density = if total_chars > 0 {
            punct_count as f64 / total_chars as f64
        } else {
            0.0
        };

        // Capitalisation rate.
        let tokens: Vec<&str> = text.split_whitespace().collect();
        let total_tokens = tokens.len();
        let cap_count = tokens
            .iter()
            .filter(|t| t.chars().next().map(|c| c.is_uppercase()).unwrap_or(false))
            .count();
        let capitalisation_rate = if total_tokens > 0 {
            cap_count as f64 / total_tokens as f64
        } else {
            0.0
        };

        // Question / exclamation rates (based on the original text's sentence boundaries).
        let question_count = text.matches('?').count();
        let exclamation_count = text.matches('!').count();
        let question_rate = if total_sentences > 0 {
            question_count as f64 / total_sentences as f64
        } else {
            0.0
        };
        let exclamation_rate = if total_sentences > 0 {
            exclamation_count as f64 / total_sentences as f64
        } else {
            0.0
        };

        // Mean word length.
        let word_len_sum: usize = tokens.iter().map(|t| t.chars().count()).sum();
        let mean_word_len = if total_tokens > 0 {
            word_len_sum as f64 / total_tokens as f64
        } else {
            0.0
        };

        Self {
            mean_sentence_len,
            stddev_sentence_len,
            punctuation_density,
            capitalisation_rate,
            question_rate,
            exclamation_rate,
            mean_word_len,
            total_chars,
            total_sentences,
        }
    }

    /// Return the style features as a fixed-length numeric vector.
    ///
    /// Vector layout (7 elements):
    /// `[mean_sentence_len, stddev_sentence_len, punctuation_density,
    ///   capitalisation_rate, question_rate, exclamation_rate, mean_word_len]`
    pub fn feature_vector(&self) -> [f64; 7] {
        [
            self.mean_sentence_len,
            self.stddev_sentence_len,
            self.punctuation_density,
            self.capitalisation_rate,
            self.question_rate,
            self.exclamation_rate,
            self.mean_word_len,
        ]
    }

    /// L1 distance between two style signatures.
    pub fn l1_distance(&self, other: &Self) -> f64 {
        let a = self.feature_vector();
        let b = other.feature_vector();
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
    }
}

// ── ModelFingerprint ──────────────────────────────────────────────────────────

/// A statistical fingerprint for a single LLM.
///
/// Built from multiple output samples.  The fingerprint captures both
/// *what* the model says ([`VocabBias`]) and *how* it says it
/// ([`StyleSignature`]).
#[derive(Debug, Clone)]
pub struct ModelFingerprint {
    /// Human-readable model name (e.g. `"gpt-4o"`, `"claude-sonnet-4-6"`).
    pub model_name: String,
    /// Token over-use bias derived from the model's outputs.
    pub vocab_bias: VocabBias,
    /// Style / surface-form signature.
    pub style: StyleSignature,
    /// Total text samples incorporated into this fingerprint.
    pub sample_count: usize,
}

impl ModelFingerprint {
    /// Create an empty fingerprint for a named model.
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            vocab_bias: VocabBias::new(),
            style: StyleSignature::default(),
            sample_count: 0,
        }
    }

    /// Incorporate a text sample into the fingerprint.
    ///
    /// Both the vocabulary bias and style signature are updated.  The style
    /// signature is recomputed from scratch each call using an accumulated
    /// buffer — callers that produce many samples should consider using
    /// [`ModelFingerprint::incorporate_many`] instead.
    pub fn incorporate(&mut self, text: &str) {
        self.vocab_bias.observe(text);
        // Update style as a running average using Welford-style online update.
        let new_style = StyleSignature::from_text(text);
        let n = self.sample_count as f64;
        let n1 = n + 1.0;
        macro_rules! update_field {
            ($field:ident) => {
                self.style.$field = (self.style.$field * n + new_style.$field) / n1;
            };
        }
        update_field!(mean_sentence_len);
        update_field!(stddev_sentence_len);
        update_field!(punctuation_density);
        update_field!(capitalisation_rate);
        update_field!(question_rate);
        update_field!(exclamation_rate);
        update_field!(mean_word_len);
        self.style.total_chars += new_style.total_chars;
        self.style.total_sentences += new_style.total_sentences;
        self.sample_count += 1;
    }

    /// Incorporate multiple text samples at once.
    pub fn incorporate_many<'a>(&mut self, texts: impl IntoIterator<Item = &'a str>) {
        for text in texts {
            self.incorporate(text);
        }
    }
}

// ── FingerprintDatabase ───────────────────────────────────────────────────────

/// Stores named model fingerprints and serves them to the matcher.
#[derive(Debug, Default)]
pub struct FingerprintDatabase {
    fingerprints: HashMap<String, ModelFingerprint>,
}

impl FingerprintDatabase {
    /// Create an empty database.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register or replace a model fingerprint.
    pub fn insert(&mut self, fingerprint: ModelFingerprint) {
        self.fingerprints
            .insert(fingerprint.model_name.clone(), fingerprint);
    }

    /// Look up a model fingerprint by name.
    pub fn get(&self, model_name: &str) -> Option<&ModelFingerprint> {
        self.fingerprints.get(model_name)
    }

    /// All model names registered in the database.
    pub fn model_names(&self) -> Vec<&str> {
        self.fingerprints.keys().map(|s| s.as_str()).collect()
    }

    /// Number of models in the database.
    pub fn len(&self) -> usize {
        self.fingerprints.len()
    }

    /// `true` when no models have been registered.
    pub fn is_empty(&self) -> bool {
        self.fingerprints.is_empty()
    }

    /// Iterate over all stored fingerprints.
    pub fn iter(&self) -> impl Iterator<Item = &ModelFingerprint> {
        self.fingerprints.values()
    }
}

// ── FingerprintMatcher ────────────────────────────────────────────────────────

/// A match candidate returned by [`FingerprintMatcher::identify`].
#[derive(Debug, Clone)]
pub struct MatchCandidate {
    /// Name of the matched model.
    pub model_name: String,
    /// Style similarity score in `[0.0, 1.0]` (higher = more similar).
    pub style_score: f64,
    /// Combined similarity score (higher = more similar).
    pub combined_score: f64,
}

/// Identifies which known model most likely produced an unknown text stream.
///
/// The matcher scores candidates using a combination of:
/// - **Style similarity**: `1 / (1 + l1_distance)` on the [`StyleSignature`]
///   feature vector (7 dimensions).
/// - **Vocab overlap**: cosine similarity of the top-50 token frequencies.
///
/// Both signals are averaged with equal weight to form the `combined_score`.
///
/// # Example
///
/// ```rust
/// use every_other_token::fingerprint::{
///     FingerprintDatabase, FingerprintMatcher, ModelFingerprint,
/// };
///
/// let mut db = FingerprintDatabase::new();
///
/// let mut fp_a = ModelFingerprint::new("model-a");
/// fp_a.incorporate("Hello! How are you doing today? I hope everything is great!");
/// db.insert(fp_a);
///
/// let mut fp_b = ModelFingerprint::new("model-b");
/// fp_b.incorporate("the quick brown fox. the lazy dog. the cat sat on a mat.");
/// db.insert(fp_b);
///
/// let matcher = FingerprintMatcher::new(db);
/// let results = matcher.identify("Hello! How are you? Hope you are well!");
/// assert!(!results.is_empty());
/// assert_eq!(results[0].model_name, "model-a");
/// ```
pub struct FingerprintMatcher {
    db: FingerprintDatabase,
}

impl FingerprintMatcher {
    /// Create a matcher backed by the given database.
    pub fn new(db: FingerprintDatabase) -> Self {
        Self { db }
    }

    /// Identify which known model most likely produced `text`.
    ///
    /// Returns candidates sorted by `combined_score` descending (best match
    /// first).  Returns an empty `Vec` when the database is empty.
    pub fn identify(&self, text: &str) -> Vec<MatchCandidate> {
        if self.db.is_empty() {
            return Vec::new();
        }

        let probe_style = StyleSignature::from_text(text);
        let probe_vocab = {
            let mut vb = VocabBias::new();
            vb.observe(text);
            vb
        };

        let mut candidates: Vec<MatchCandidate> = self
            .db
            .iter()
            .map(|fp| {
                let style_score = style_similarity(&probe_style, &fp.style);
                let vocab_score = vocab_cosine_similarity(&probe_vocab, &fp.vocab_bias);
                let combined_score = (style_score + vocab_score) / 2.0;
                MatchCandidate {
                    model_name: fp.model_name.clone(),
                    style_score,
                    combined_score,
                }
            })
            .collect();

        candidates.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates
    }

    /// Return a reference to the underlying database.
    pub fn database(&self) -> &FingerprintDatabase {
        &self.db
    }
}

// ── Similarity helpers ────────────────────────────────────────────────────────

/// Style similarity: `1 / (1 + l1_distance(a, b))`.
fn style_similarity(a: &StyleSignature, b: &StyleSignature) -> f64 {
    1.0 / (1.0 + a.l1_distance(b))
}

/// Cosine similarity between two vocabulary frequency distributions.
///
/// Only tokens present in either distribution are considered.
fn vocab_cosine_similarity(a: &VocabBias, b: &VocabBias) -> f64 {
    if a.total_observed == 0 || b.total_observed == 0 {
        return 0.0;
    }

    // Collect all tokens from both distributions.
    let keys: std::collections::HashSet<&String> = a
        .observed_counts
        .keys()
        .chain(b.observed_counts.keys())
        .collect();

    let total_a = a.total_observed as f64;
    let total_b = b.total_observed as f64;

    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;

    for tok in keys {
        let fa = *a.observed_counts.get(tok).unwrap_or(&0) as f64 / total_a;
        let fb = *b.observed_counts.get(tok).unwrap_or(&0) as f64 / total_b;
        dot += fa * fb;
        norm_a += fa * fa;
        norm_b += fb * fb;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom <= 0.0 { 0.0 } else { (dot / denom).clamp(0.0, 1.0) }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── VocabBias ───────────────────────────────────────────────────────────

    #[test]
    fn test_vocab_bias_observe_accumulates() {
        let mut vb = VocabBias::new();
        vb.observe("hello world hello");
        assert_eq!(*vb.observed_counts.get("hello").unwrap_or(&0), 2);
        assert_eq!(vb.total_observed, 3);
    }

    #[test]
    fn test_vocab_bias_empty() {
        let vb = VocabBias::new();
        assert_eq!(vb.bias("anything"), 0.0);
    }

    #[test]
    fn test_vocab_bias_positive_for_frequent_token() {
        let mut vb = VocabBias::new();
        vb.observe("cat cat cat dog");
        // "cat" appears 75 % of the time; uniform baseline is ~50 % for 2-word vocab
        let b = vb.bias("cat");
        assert!(b > 0.0, "frequent token should have positive bias: {b}");
    }

    #[test]
    fn test_vocab_bias_top_overused() {
        let mut vb = VocabBias::new();
        vb.observe("a a a a b c");
        let top = vb.top_overused(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].0, "a");
    }

    #[test]
    fn test_vocab_frequency_vector_non_empty() {
        let mut vb = VocabBias::new();
        vb.observe("foo bar baz");
        let fv = vb.frequency_vector();
        assert_eq!(fv.len(), 3);
        let sum: f64 = fv.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "frequencies should sum to 1: {sum}");
    }

    // ── StyleSignature ──────────────────────────────────────────────────────

    #[test]
    fn test_style_signature_empty() {
        let s = StyleSignature::from_text("");
        assert_eq!(s.total_chars, 0);
        assert_eq!(s.total_sentences, 0);
    }

    #[test]
    fn test_style_signature_single_sentence() {
        let s = StyleSignature::from_text("Hello world how are you");
        // No sentence terminators → treated as one sentence
        assert_eq!(s.total_sentences, 1);
        assert_eq!(s.mean_sentence_len, 5.0);
    }

    #[test]
    fn test_style_signature_punctuation_density() {
        let s = StyleSignature::from_text("Hello, world.");
        assert!(s.punctuation_density > 0.0);
    }

    #[test]
    fn test_style_signature_question_rate() {
        let s = StyleSignature::from_text("How are you? I am fine.");
        assert!(s.question_rate > 0.0);
        assert_eq!(s.question_rate, 0.5); // 1 question out of 2 sentences
    }

    #[test]
    fn test_style_feature_vector_length() {
        let s = StyleSignature::from_text("Some text here.");
        assert_eq!(s.feature_vector().len(), 7);
    }

    #[test]
    fn test_style_l1_distance_self_zero() {
        let s = StyleSignature::from_text("The quick brown fox jumps.");
        assert_eq!(s.l1_distance(&s), 0.0);
    }

    #[test]
    fn test_style_l1_distance_distinct() {
        let a = StyleSignature::from_text("hello world");
        let b = StyleSignature::from_text("Hello! How are you doing today? Great!");
        assert!(a.l1_distance(&b) > 0.0);
    }

    // ── ModelFingerprint ────────────────────────────────────────────────────

    #[test]
    fn test_fingerprint_incorporate_updates_counts() {
        let mut fp = ModelFingerprint::new("test-model");
        fp.incorporate("hello world goodbye");
        assert_eq!(fp.sample_count, 1);
        assert_eq!(fp.vocab_bias.total_observed, 3);
    }

    #[test]
    fn test_fingerprint_incorporate_many() {
        let mut fp = ModelFingerprint::new("test-model");
        fp.incorporate_many(["foo bar", "baz qux", "hello world"]);
        assert_eq!(fp.sample_count, 3);
    }

    #[test]
    fn test_fingerprint_style_updated() {
        let mut fp = ModelFingerprint::new("test-model");
        fp.incorporate("Hello world how are you");
        assert!(fp.style.mean_sentence_len > 0.0);
    }

    // ── FingerprintDatabase ─────────────────────────────────────────────────

    #[test]
    fn test_database_insert_and_get() {
        let mut db = FingerprintDatabase::new();
        assert!(db.is_empty());
        let fp = ModelFingerprint::new("model-x");
        db.insert(fp);
        assert_eq!(db.len(), 1);
        assert!(db.get("model-x").is_some());
        assert!(db.get("unknown").is_none());
    }

    #[test]
    fn test_database_model_names() {
        let mut db = FingerprintDatabase::new();
        db.insert(ModelFingerprint::new("a"));
        db.insert(ModelFingerprint::new("b"));
        let mut names = db.model_names();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
    }

    // ── FingerprintMatcher ──────────────────────────────────────────────────

    #[test]
    fn test_matcher_empty_db_returns_empty() {
        let db = FingerprintDatabase::new();
        let matcher = FingerprintMatcher::new(db);
        assert!(matcher.identify("some text").is_empty());
    }

    #[test]
    fn test_matcher_returns_all_candidates() {
        let mut db = FingerprintDatabase::new();
        db.insert(ModelFingerprint::new("m1"));
        db.insert(ModelFingerprint::new("m2"));
        let matcher = FingerprintMatcher::new(db);
        let results = matcher.identify("hello world");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_matcher_prefers_matching_style() {
        let mut db = FingerprintDatabase::new();

        // Model A: question-heavy, exclamatory
        let mut fp_a = ModelFingerprint::new("enthusiastic");
        fp_a.incorporate_many([
            "Wow, that is amazing! How exciting is that?",
            "Really? You don't say! What a surprise!",
            "Incredible! Can you believe it? What a day!",
        ]);
        db.insert(fp_a);

        // Model B: plain declarative sentences
        let mut fp_b = ModelFingerprint::new("plain");
        fp_b.incorporate_many([
            "The sun rises in the east. Water is wet. Grass is green.",
            "The cat sat on the mat. The dog ran fast. The bird flew away.",
            "It is a fact. This is true. That is correct.",
        ]);
        db.insert(fp_b);

        let matcher = FingerprintMatcher::new(db);

        // Probe matches Model A's style (questions + exclamations)
        let results = matcher.identify("Wow! Is this real? Amazing!");
        assert!(!results.is_empty());
        assert_eq!(results[0].model_name, "enthusiastic");
    }

    #[test]
    fn test_matcher_scores_in_range() {
        let mut db = FingerprintDatabase::new();
        let mut fp = ModelFingerprint::new("test");
        fp.incorporate("hello world this is a test sentence for scoring");
        db.insert(fp);
        let matcher = FingerprintMatcher::new(db);
        let results = matcher.identify("hello world test");
        assert_eq!(results.len(), 1);
        let score = results[0].combined_score;
        assert!(score >= 0.0 && score <= 1.0, "score={score} out of [0,1]");
    }

    // ── Similarity helpers ──────────────────────────────────────────────────

    #[test]
    fn test_style_similarity_self_is_one() {
        let s = StyleSignature::from_text("The quick brown fox jumps over the lazy dog.");
        let sim = style_similarity(&s, &s);
        assert!((sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_vocab_cosine_similarity_identical() {
        let mut a = VocabBias::new();
        a.observe("hello world hello");
        let b = a.clone();
        let sim = vocab_cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-9, "identical vocab should have cosine=1: {sim}");
    }

    #[test]
    fn test_vocab_cosine_similarity_empty() {
        let a = VocabBias::new();
        let b = VocabBias::new();
        assert_eq!(vocab_cosine_similarity(&a, &b), 0.0);
    }
}
