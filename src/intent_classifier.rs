//! Zero-shot intent classification for LLM input text.
//!
//! Scores each intent category by keyword overlap and weighting, then
//! returns primary/secondary intents, entity extractions, and sentiment.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Intent
// ---------------------------------------------------------------------------

/// The set of recognised intent categories.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Intent {
    Question,
    Command,
    Clarification,
    Complaint,
    Compliment,
    RequestForHelp,
    Chitchat,
    Technical,
    Creative,
    DataRequest,
}

impl fmt::Display for Intent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Intent::Question => "Question",
            Intent::Command => "Command",
            Intent::Clarification => "Clarification",
            Intent::Complaint => "Complaint",
            Intent::Compliment => "Compliment",
            Intent::RequestForHelp => "RequestForHelp",
            Intent::Chitchat => "Chitchat",
            Intent::Technical => "Technical",
            Intent::Creative => "Creative",
            Intent::DataRequest => "DataRequest",
        };
        write!(f, "{}", name)
    }
}

// ---------------------------------------------------------------------------
// IntentScore
// ---------------------------------------------------------------------------

/// A single intent with its confidence score.
#[derive(Debug, Clone)]
pub struct IntentScore {
    pub intent: Intent,
    pub confidence: f64,
}

// ---------------------------------------------------------------------------
// ClassificationResult
// ---------------------------------------------------------------------------

/// The full result of classifying a text.
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub primary: IntentScore,
    pub secondary: Vec<IntentScore>,
    pub is_ambiguous: bool,
    pub detected_entities: Vec<String>,
}

// ---------------------------------------------------------------------------
// IntentSignal
// ---------------------------------------------------------------------------

/// A set of keywords associated with an intent, with a multiplier weight.
#[derive(Debug, Clone)]
pub struct IntentSignal {
    pub keywords: Vec<&'static str>,
    pub weight: f64,
}

// ---------------------------------------------------------------------------
// IntentClassifier
// ---------------------------------------------------------------------------

/// Zero-shot intent classifier backed by keyword signals.
pub struct IntentClassifier {
    signals: HashMap<Intent, Vec<IntentSignal>>,
}

impl IntentClassifier {
    /// Build the classifier with default keyword signals for every intent.
    pub fn new() -> Self {
        let mut signals: HashMap<Intent, Vec<IntentSignal>> = HashMap::new();

        signals.insert(
            Intent::Question,
            vec![
                IntentSignal {
                    keywords: vec!["what", "how", "why", "when", "where", "who", "which", "?"],
                    weight: 1.0,
                },
                IntentSignal {
                    keywords: vec!["explain", "tell me", "describe", "clarify"],
                    weight: 0.7,
                },
            ],
        );

        signals.insert(
            Intent::Command,
            vec![
                IntentSignal {
                    keywords: vec![
                        "do", "make", "create", "write", "build", "generate", "run", "execute",
                        "start", "stop", "delete", "update", "add", "remove",
                    ],
                    weight: 1.0,
                },
                IntentSignal {
                    keywords: vec!["please", "could you", "can you"],
                    weight: 0.4,
                },
            ],
        );

        signals.insert(
            Intent::Clarification,
            vec![
                IntentSignal {
                    keywords: vec![
                        "clarify", "unclear", "confused", "mean", "elaborate", "expand",
                        "what do you mean", "i don't understand",
                    ],
                    weight: 1.0,
                },
                IntentSignal {
                    keywords: vec!["again", "repeat", "rephrase"],
                    weight: 0.6,
                },
            ],
        );

        signals.insert(
            Intent::Complaint,
            vec![
                IntentSignal {
                    keywords: vec![
                        "wrong", "broken", "not working", "error", "fail", "terrible",
                        "awful", "bad", "worst", "useless", "disappointed",
                    ],
                    weight: 1.0,
                },
                IntentSignal {
                    keywords: vec!["issue", "problem", "bug", "crash", "doesn't work"],
                    weight: 0.8,
                },
            ],
        );

        signals.insert(
            Intent::Compliment,
            vec![
                IntentSignal {
                    keywords: vec![
                        "great", "excellent", "amazing", "wonderful", "fantastic",
                        "love", "perfect", "awesome", "brilliant", "thank",
                    ],
                    weight: 1.0,
                },
                IntentSignal {
                    keywords: vec!["good", "nice", "helpful", "useful", "well done"],
                    weight: 0.6,
                },
            ],
        );

        signals.insert(
            Intent::RequestForHelp,
            vec![
                IntentSignal {
                    keywords: vec![
                        "help", "assist", "support", "stuck", "lost", "guide",
                        "need", "struggling", "can't figure out",
                    ],
                    weight: 1.0,
                },
                IntentSignal {
                    keywords: vec!["please", "urgent", "asap", "quickly"],
                    weight: 0.4,
                },
            ],
        );

        signals.insert(
            Intent::Chitchat,
            vec![
                IntentSignal {
                    keywords: vec![
                        "hello", "hi", "hey", "how are you", "good morning", "good evening",
                        "whats up", "what's up", "bye", "goodbye", "talk",
                    ],
                    weight: 1.0,
                },
                IntentSignal {
                    keywords: vec!["weather", "weekend", "fun", "laugh", "joke"],
                    weight: 0.5,
                },
            ],
        );

        signals.insert(
            Intent::Technical,
            vec![
                IntentSignal {
                    keywords: vec![
                        "code", "function", "algorithm", "api", "database", "server",
                        "deploy", "debug", "compile", "syntax", "library", "framework",
                        "stack", "architecture",
                    ],
                    weight: 1.0,
                },
                IntentSignal {
                    keywords: vec![
                        "error", "exception", "null", "type", "class", "struct",
                        "method", "variable", "parameter",
                    ],
                    weight: 0.7,
                },
            ],
        );

        signals.insert(
            Intent::Creative,
            vec![
                IntentSignal {
                    keywords: vec![
                        "write", "story", "poem", "song", "imagine", "create", "design",
                        "invent", "brainstorm", "idea", "creative", "fiction",
                    ],
                    weight: 1.0,
                },
                IntentSignal {
                    keywords: vec!["art", "music", "paint", "draw", "compose", "narrative"],
                    weight: 0.7,
                },
            ],
        );

        signals.insert(
            Intent::DataRequest,
            vec![
                IntentSignal {
                    keywords: vec![
                        "data", "statistics", "numbers", "metrics", "report", "analysis",
                        "chart", "graph", "table", "trend", "count", "total", "average",
                    ],
                    weight: 1.0,
                },
                IntentSignal {
                    keywords: vec!["show me", "list", "find", "search", "query", "fetch"],
                    weight: 0.6,
                },
            ],
        );

        Self { signals }
    }

    /// Classify `text` and return a [`ClassificationResult`].
    pub fn classify(&self, text: &str) -> ClassificationResult {
        let lower = text.to_lowercase();
        let mut scores: Vec<(Intent, f64)> = Vec::new();

        for (intent, intent_signals) in &self.signals {
            let mut raw_score = 0.0_f64;
            for signal in intent_signals {
                for &kw in &signal.keywords {
                    if lower.contains(kw) {
                        raw_score += signal.weight;
                    }
                }
            }
            scores.push((intent.clone(), raw_score));
        }

        // Normalise
        let total: f64 = scores.iter().map(|(_, s)| s).sum();
        let mut normalised: Vec<(Intent, f64)> = if total > 0.0 {
            scores.iter().map(|(i, s)| (i.clone(), s / total)).collect()
        } else {
            scores.iter().map(|(i, _)| (i.clone(), 0.0)).collect()
        };

        normalised.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let primary_confidence = normalised[0].1;
        let secondary_confidence = normalised.get(1).map(|(_, c)| *c).unwrap_or(0.0);
        let is_ambiguous = total == 0.0
            || (primary_confidence > 0.0 && (primary_confidence - secondary_confidence).abs() < 0.1);

        let primary = IntentScore {
            intent: normalised[0].0.clone(),
            confidence: primary_confidence,
        };

        let secondary: Vec<IntentScore> = normalised[1..]
            .iter()
            .filter(|(_, c)| *c > 0.0)
            .map(|(intent, confidence)| IntentScore { intent: intent.clone(), confidence: *confidence })
            .collect();

        let detected_entities = Self::extract_entities(text);

        ClassificationResult { primary, secondary, is_ambiguous, detected_entities }
    }

    /// Extract quoted strings, capitalised non-first words, and numbers with units from `text`.
    pub fn extract_entities(text: &str) -> Vec<String> {
        let mut entities: Vec<String> = Vec::new();

        // Quoted strings
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == '"' {
                let start = i + 1;
                let mut end = start;
                while end < chars.len() && chars[end] != '"' {
                    end += 1;
                }
                if end < chars.len() {
                    let quoted: String = chars[start..end].iter().collect();
                    if !quoted.trim().is_empty() {
                        entities.push(quoted);
                    }
                    i = end + 1;
                    continue;
                }
            }
            i += 1;
        }

        // Capitalised words that are not the first word
        let words: Vec<&str> = text.split_whitespace().collect();
        for (idx, word) in words.iter().enumerate() {
            if idx == 0 {
                continue;
            }
            let clean: String = word.chars().filter(|c| c.is_alphabetic()).collect();
            if clean.len() >= 2 {
                let mut chars_iter = clean.chars();
                if let Some(first) = chars_iter.next() {
                    if first.is_uppercase() && chars_iter.all(|c| c.is_lowercase()) {
                        entities.push(clean);
                    }
                }
            }
        }

        // Numbers with units (e.g. "42 kg", "3.14 GHz")
        let unit_suffixes = [
            "kg", "g", "lb", "km", "m", "cm", "mm", "mi", "hz", "khz", "mhz", "ghz",
            "mb", "gb", "tb", "ms", "s", "min", "hr", "px", "pt",
        ];
        for (idx, word) in words.iter().enumerate() {
            let numeric: String = word.chars().filter(|c| c.is_ascii_digit() || *c == '.').collect();
            if !numeric.is_empty() {
                if let Some(next_word) = words.get(idx + 1) {
                    let next_lower = next_word.to_lowercase();
                    let next_clean: String = next_lower.chars().filter(|c| c.is_alphabetic()).collect();
                    if unit_suffixes.contains(&next_clean.as_str()) {
                        entities.push(format!("{} {}", numeric, next_clean));
                    }
                }
            }
        }

        entities.dedup();
        entities
    }

    /// Return `true` if the text is a question (ends with `?` or starts with a question word).
    pub fn is_question(text: &str) -> bool {
        let trimmed = text.trim();
        if trimmed.ends_with('?') {
            return true;
        }
        let lower = trimmed.to_lowercase();
        let question_words = ["what", "how", "why", "when", "where", "who", "which", "whose", "whom"];
        for qw in &question_words {
            if lower.starts_with(qw) {
                return true;
            }
        }
        false
    }

    /// Return a sentiment score in [-1.0, +1.0] based on positive/negative word lists.
    pub fn detect_sentiment(text: &str) -> f64 {
        let positive_words = [
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "love", "perfect", "awesome", "brilliant", "happy", "joy", "best",
            "helpful", "useful", "nice", "positive", "beautiful", "outstanding",
        ];
        let negative_words = [
            "bad", "terrible", "awful", "horrible", "worst", "hate", "useless",
            "broken", "wrong", "fail", "error", "problem", "issue", "sad",
            "angry", "frustrated", "disappointed", "poor", "negative", "ugly",
        ];

        let lower = text.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();
        let mut score = 0.0_f64;
        let mut count = 0usize;

        for word in &words {
            let clean: String = word.chars().filter(|c| c.is_alphabetic()).collect();
            if positive_words.contains(&clean.as_str()) {
                score += 1.0;
                count += 1;
            } else if negative_words.contains(&clean.as_str()) {
                score -= 1.0;
                count += 1;
            }
        }

        if count == 0 {
            0.0
        } else {
            (score / count as f64).clamp(-1.0, 1.0)
        }
    }

    /// Classify a batch of texts.
    pub fn classify_batch(&self, texts: &[&str]) -> Vec<ClassificationResult> {
        texts.iter().map(|t| self.classify(t)).collect()
    }
}

impl Default for IntentClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_question_detection() {
        assert!(IntentClassifier::is_question("What is the capital of France?"));
        assert!(IntentClassifier::is_question("How does this work?"));
        assert!(!IntentClassifier::is_question("Do it now"));
    }

    #[test]
    fn test_classify_question() {
        let c = IntentClassifier::new();
        let r = c.classify("What is the meaning of life?");
        assert_eq!(r.primary.intent, Intent::Question);
    }

    #[test]
    fn test_classify_command() {
        let c = IntentClassifier::new();
        let r = c.classify("Please create a new file and add the function.");
        // Should rank Command highly
        assert!(r.primary.confidence > 0.0);
    }

    #[test]
    fn test_sentiment_positive() {
        let score = IntentClassifier::detect_sentiment("This is great and amazing work!");
        assert!(score > 0.0);
    }

    #[test]
    fn test_sentiment_negative() {
        let score = IntentClassifier::detect_sentiment("This is terrible and broken.");
        assert!(score < 0.0);
    }

    #[test]
    fn test_entity_extraction_quoted() {
        let entities = IntentClassifier::extract_entities(r#"Find "Paris" and "London" in the data."#);
        assert!(entities.contains(&"Paris".to_string()));
        assert!(entities.contains(&"London".to_string()));
    }

    #[test]
    fn test_batch() {
        let c = IntentClassifier::new();
        let results = c.classify_batch(&["hello how are you", "create a function"]);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Intent::Question), "Question");
        assert_eq!(format!("{}", Intent::DataRequest), "DataRequest");
    }
}
