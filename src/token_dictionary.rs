//! Automated token sentiment dictionary with persistence and interactive feedback.
//!
//! [`TokenDictionary`] learns per-token sentiment scores incrementally from
//! context-level feedback.  Scores are updated using an exponentially-weighted
//! running average so recent feedback has more influence than older feedback.
//!
//! [`FeedbackCollector`] allows callers to mark entire token stream segments as
//! "good" or "bad" and propagates that signal to every token in the segment.
//!
//! Both types serialise to / deserialise from JSON (via `serde_json`) and are
//! stored at `~/.every-other-token/token_dict.json` by default.
//!
//! ## Example
//!
//! ```rust
//! use every_other_token::token_dictionary::TokenDictionary;
//!
//! let mut dict = TokenDictionary::new();
//! dict.record_token("excellent", 0.9);
//! dict.record_token("terrible", -0.8);
//! assert!(dict.get_sentiment("excellent") > 0.0);
//! assert!(dict.get_sentiment("terrible") < 0.0);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// TokenEntry
// ---------------------------------------------------------------------------

/// A dictionary entry for a single token, capturing its learned sentiment and usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenEntry {
    /// The token text (lowercased).
    pub token_text: String,
    /// Learned sentiment score in `[-1.0, 1.0]`.  Positive = positive sentiment;
    /// negative = negative sentiment.
    pub sentiment_score: f64,
    /// Importance score in `[0.0, 1.0]` — driven by observation frequency.
    pub importance_score: f64,
    /// Total number of times this token has been observed.
    pub frequency: u64,
    /// Number of times a user has given explicit feedback about this token.
    pub user_feedback_count: u64,
}

impl TokenEntry {
    /// Create a new entry with the given initial sentiment score.
    fn new(token_text: impl Into<String>, sentiment_score: f64) -> Self {
        Self {
            token_text: token_text.into(),
            sentiment_score: sentiment_score.max(-1.0).min(1.0),
            importance_score: 0.0,
            frequency: 1,
            user_feedback_count: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// TokenDictionary
// ---------------------------------------------------------------------------

/// Exponential moving-average weight for new observations (α in EMA formula).
/// Values closer to 1.0 make the score respond faster to new data.
const EMA_ALPHA: f64 = 0.1;

/// Incrementally learns per-token sentiment scores from context-level feedback.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenDictionary {
    entries: HashMap<String, TokenEntry>,
}

impl TokenDictionary {
    /// Create an empty dictionary.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an observation of `token` in a context with `context_sentiment`
    /// (which should be in `[-1.0, 1.0]`).
    ///
    /// If the token has not been seen before it is inserted with the given
    /// sentiment as the initial score.  Otherwise the score is updated with an
    /// exponentially-weighted running average.
    pub fn record_token(&mut self, token: &str, context_sentiment: f64) {
        let sentiment = context_sentiment.max(-1.0).min(1.0);
        let key = token.to_lowercase();

        match self.entries.get_mut(&key) {
            Some(entry) => {
                // EMA update: score = alpha * new + (1 - alpha) * old
                entry.sentiment_score = EMA_ALPHA * sentiment + (1.0 - EMA_ALPHA) * entry.sentiment_score;
                entry.sentiment_score = entry.sentiment_score.max(-1.0).min(1.0);
                entry.frequency += 1;
                // Importance grows with log of frequency, capped at 1.0.
                entry.importance_score = ((entry.frequency as f64).ln() / 10.0).min(1.0);
            }
            None => {
                self.entries.insert(key.clone(), TokenEntry::new(key, sentiment));
            }
        }
    }

    /// Return the learned sentiment score for `token`, or `0.0` if the token
    /// has never been observed.
    pub fn get_sentiment(&self, token: &str) -> f64 {
        self.entries
            .get(&token.to_lowercase())
            .map(|e| e.sentiment_score)
            .unwrap_or(0.0)
    }

    /// Return `n` tokens with the highest sentiment scores.
    pub fn top_positive(&self, n: usize) -> Vec<TokenEntry> {
        let mut entries: Vec<TokenEntry> = self.entries.values().cloned().collect();
        entries.sort_by(|a, b| {
            b.sentiment_score
                .partial_cmp(&a.sentiment_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        entries.truncate(n);
        entries
    }

    /// Return `n` tokens with the lowest (most negative) sentiment scores.
    pub fn top_negative(&self, n: usize) -> Vec<TokenEntry> {
        let mut entries: Vec<TokenEntry> = self.entries.values().cloned().collect();
        entries.sort_by(|a, b| {
            a.sentiment_score
                .partial_cmp(&b.sentiment_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        entries.truncate(n);
        entries
    }

    /// Total number of distinct tokens in the dictionary.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the dictionary is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Persist the dictionary to `path` as JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created or if serialisation fails.
    pub fn persist_to_file(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load a dictionary from `path`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or if the JSON is malformed.
    pub fn load_from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        let dict: Self = serde_json::from_str(&data)?;
        Ok(dict)
    }
}

// ---------------------------------------------------------------------------
// FeedbackCollector
// ---------------------------------------------------------------------------

/// Collects user feedback about token stream segments and propagates it to the
/// underlying [`TokenDictionary`].
///
/// The default persistence path is `~/.every-other-token/token_dict.json`.
pub struct FeedbackCollector {
    dict: TokenDictionary,
    persist_path: PathBuf,
}

impl FeedbackCollector {
    /// Create a new collector backed by the default path
    /// (`~/.every-other-token/token_dict.json`).
    ///
    /// If the file already exists it is loaded; otherwise an empty dictionary
    /// is used.
    pub fn new() -> Self {
        let path = default_dict_path();
        let dict = TokenDictionary::load_from_file(&path).unwrap_or_default();
        Self { dict, persist_path: path }
    }

    /// Create a collector with an explicit persistence path.
    pub fn with_path(path: PathBuf) -> Self {
        let dict = TokenDictionary::load_from_file(&path).unwrap_or_default();
        Self { dict, persist_path: path }
    }

    /// Return a reference to the underlying [`TokenDictionary`].
    pub fn dictionary(&self) -> &TokenDictionary {
        &self.dict
    }

    /// Mark all tokens in `segment` as "good" (`sentiment = +1.0`).
    ///
    /// Writes through to disk after updating.
    pub fn mark_good(&mut self, segment: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
        self.apply_feedback(segment, 1.0)
    }

    /// Mark all tokens in `segment` as "bad" (`sentiment = -1.0`).
    ///
    /// Writes through to disk after updating.
    pub fn mark_bad(&mut self, segment: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
        self.apply_feedback(segment, -1.0)
    }

    /// Apply `sentiment` to every token in `segment`, increment each token's
    /// `user_feedback_count`, and persist.
    fn apply_feedback(
        &mut self,
        segment: &[&str],
        sentiment: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for &token in segment {
            self.dict.record_token(token, sentiment);
            // Increment user_feedback_count.
            if let Some(entry) = self.dict.entries.get_mut(&token.to_lowercase()) {
                entry.user_feedback_count += 1;
            }
        }
        self.dict.persist_to_file(&self.persist_path)
    }

    /// Apply a custom sentiment value in `[-1.0, 1.0]` to every token in `segment`.
    pub fn mark_with_sentiment(
        &mut self,
        segment: &[&str],
        sentiment: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.apply_feedback(segment, sentiment)
    }
}

impl Default for FeedbackCollector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

/// Returns the default dictionary path: `~/.every-other-token/token_dict.json`.
pub fn default_dict_path() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home)
        .join(".every-other-token")
        .join("token_dict.json")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_get_sentiment_positive() {
        let mut dict = TokenDictionary::new();
        dict.record_token("excellent", 0.9);
        let s = dict.get_sentiment("excellent");
        assert!(s > 0.0, "Expected positive sentiment, got {}", s);
    }

    #[test]
    fn test_record_and_get_sentiment_negative() {
        let mut dict = TokenDictionary::new();
        dict.record_token("terrible", -0.8);
        let s = dict.get_sentiment("terrible");
        assert!(s < 0.0, "Expected negative sentiment, got {}", s);
    }

    #[test]
    fn test_get_sentiment_unknown_token() {
        let dict = TokenDictionary::new();
        assert_eq!(dict.get_sentiment("unknown_xyz"), 0.0);
    }

    #[test]
    fn test_case_insensitive_lookup() {
        let mut dict = TokenDictionary::new();
        dict.record_token("Good", 0.5);
        assert_eq!(dict.get_sentiment("good"), dict.get_sentiment("GOOD"));
        assert_ne!(dict.get_sentiment("good"), 0.0);
    }

    #[test]
    fn test_ema_converges_toward_new_sentiment() {
        let mut dict = TokenDictionary::new();
        // Prime the entry with a neutral value.
        for _ in 0..50 {
            dict.record_token("word", 0.0);
        }
        let before = dict.get_sentiment("word");
        // Push strongly positive.
        for _ in 0..100 {
            dict.record_token("word", 1.0);
        }
        let after = dict.get_sentiment("word");
        assert!(after > before, "EMA should converge towards 1.0");
    }

    #[test]
    fn test_sentiment_clamped_to_range() {
        let mut dict = TokenDictionary::new();
        dict.record_token("over", 2.0);
        let s = dict.get_sentiment("over");
        assert!(s <= 1.0 && s >= -1.0, "Sentiment must stay in [-1,1], got {}", s);

        dict.record_token("under", -2.0);
        let s2 = dict.get_sentiment("under");
        assert!(s2 <= 1.0 && s2 >= -1.0);
    }

    #[test]
    fn test_top_positive_returns_sorted() {
        let mut dict = TokenDictionary::new();
        dict.record_token("great", 0.9);
        dict.record_token("good", 0.5);
        dict.record_token("okay", 0.2);
        dict.record_token("bad", -0.8);
        let top = dict.top_positive(2);
        assert_eq!(top.len(), 2);
        assert!(
            top[0].sentiment_score >= top[1].sentiment_score,
            "top_positive should be sorted descending"
        );
    }

    #[test]
    fn test_top_negative_returns_sorted() {
        let mut dict = TokenDictionary::new();
        dict.record_token("awful", -0.9);
        dict.record_token("bad", -0.5);
        dict.record_token("okay", 0.2);
        let bot = dict.top_negative(2);
        assert_eq!(bot.len(), 2);
        assert!(
            bot[0].sentiment_score <= bot[1].sentiment_score,
            "top_negative should be sorted ascending"
        );
    }

    #[test]
    fn test_top_positive_n_exceeds_dict_size() {
        let mut dict = TokenDictionary::new();
        dict.record_token("a", 0.5);
        let top = dict.top_positive(100);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut dict = TokenDictionary::new();
        assert!(dict.is_empty());
        dict.record_token("word", 0.3);
        assert_eq!(dict.len(), 1);
        assert!(!dict.is_empty());
    }

    #[test]
    fn test_frequency_increments() {
        let mut dict = TokenDictionary::new();
        for _ in 0..5 {
            dict.record_token("repeat", 0.1);
        }
        let entry = dict.entries.get("repeat").expect("entry present");
        assert_eq!(entry.frequency, 5);
    }

    #[test]
    fn test_persist_and_load() {
        let mut dict = TokenDictionary::new();
        dict.record_token("hello", 0.7);
        dict.record_token("world", -0.3);

        let tmp = std::env::temp_dir().join("token_dict_test.json");
        dict.persist_to_file(&tmp).expect("persist");

        let loaded = TokenDictionary::load_from_file(&tmp).expect("load");
        assert!((loaded.get_sentiment("hello") - dict.get_sentiment("hello")).abs() < 1e-9);
        assert!((loaded.get_sentiment("world") - dict.get_sentiment("world")).abs() < 1e-9);

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_load_from_nonexistent_file() {
        let result = TokenDictionary::load_from_file(Path::new("/nonexistent/path/dict.json"));
        assert!(result.is_err());
    }

    #[test]
    fn test_feedback_collector_mark_good() {
        let tmp_dir = std::env::temp_dir().join("eot_feedback_test_good");
        std::fs::create_dir_all(&tmp_dir).ok();
        let path = tmp_dir.join("dict.json");

        let mut collector = FeedbackCollector::with_path(path.clone());
        collector.mark_good(&["fantastic", "amazing"]).expect("mark good");

        let s = collector.dictionary().get_sentiment("fantastic");
        assert!(s > 0.0, "Expected positive sentiment after mark_good, got {}", s);

        std::fs::remove_dir_all(&tmp_dir).ok();
    }

    #[test]
    fn test_feedback_collector_mark_bad() {
        let tmp_dir = std::env::temp_dir().join("eot_feedback_test_bad");
        std::fs::create_dir_all(&tmp_dir).ok();
        let path = tmp_dir.join("dict.json");

        let mut collector = FeedbackCollector::with_path(path.clone());
        collector.mark_bad(&["horrible", "dreadful"]).expect("mark bad");

        let s = collector.dictionary().get_sentiment("horrible");
        assert!(s < 0.0, "Expected negative sentiment after mark_bad, got {}", s);

        std::fs::remove_dir_all(&tmp_dir).ok();
    }

    #[test]
    fn test_feedback_collector_user_feedback_count() {
        let tmp_dir = std::env::temp_dir().join("eot_feedback_count");
        std::fs::create_dir_all(&tmp_dir).ok();
        let path = tmp_dir.join("dict.json");

        let mut collector = FeedbackCollector::with_path(path.clone());
        collector.mark_good(&["nice"]).expect("mark good");
        collector.mark_good(&["nice"]).expect("mark good again");

        let entry = collector
            .dictionary()
            .entries
            .get("nice")
            .expect("entry");
        assert_eq!(entry.user_feedback_count, 2);

        std::fs::remove_dir_all(&tmp_dir).ok();
    }

    #[test]
    fn test_feedback_collector_persists_on_mark() {
        let tmp_dir = std::env::temp_dir().join("eot_feedback_persist");
        std::fs::create_dir_all(&tmp_dir).ok();
        let path = tmp_dir.join("dict.json");

        let mut collector = FeedbackCollector::with_path(path.clone());
        collector.mark_good(&["persist_me"]).expect("mark good");

        // Load fresh from disk.
        let loaded = TokenDictionary::load_from_file(&path).expect("load");
        assert!(loaded.get_sentiment("persist_me") > 0.0);

        std::fs::remove_dir_all(&tmp_dir).ok();
    }

    #[test]
    fn test_default_dict_path_not_empty() {
        let path = default_dict_path();
        assert!(path.to_str().map(|s| !s.is_empty()).unwrap_or(false));
    }
}
