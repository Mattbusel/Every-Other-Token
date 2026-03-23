//! Score and suggest improvements for response clarity.

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ReadabilityMetrics {
    pub avg_sentence_length: f64,
    pub avg_word_length: f64,
    pub flesch_score: f64,
    pub grade_level: f64,
    pub vocabulary_diversity: f64,
}

#[derive(Debug, Clone)]
pub enum ClarityIssue {
    TooLong,
    TooShort,
    PoorVocabularyDiversity,
    LowReadability,
    HighGradeLevel,
    MissingStructure,
    ExcessiveJargon(Vec<String>),
}

#[derive(Debug, Clone)]
pub struct EnhancementSuggestion {
    pub issue: ClarityIssue,
    pub description: String,
    pub priority: u8,
}

#[derive(Debug, Clone)]
pub struct ResponseScore {
    pub clarity: f64,
    pub conciseness: f64,
    pub completeness: f64,
    pub overall: f64,
}

#[derive(Debug, Clone)]
pub struct ResponseEnhancer {
    pub jargon_list: Vec<String>,
    pub min_length: usize,
    pub max_length: usize,
    pub target_grade_level: f64,
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl ResponseEnhancer {
    pub fn new() -> Self {
        Self {
            jargon_list: vec![
                "utilize".to_string(),
                "leverage".to_string(),
                "synergy".to_string(),
                "paradigm".to_string(),
                "holistic".to_string(),
                "robust".to_string(),
                "scalable".to_string(),
                "ecosystem".to_string(),
                "bandwidth".to_string(),
                "iterate".to_string(),
                "granular".to_string(),
                "streamline".to_string(),
                "best-in-class".to_string(),
                "deep dive".to_string(),
                "circle back".to_string(),
            ],
            min_length: 50,
            max_length: 2000,
            target_grade_level: 10.0,
        }
    }

    /// Count sentences by splitting on .!? followed by whitespace or end of string.
    pub fn count_sentences(text: &str) -> usize {
        if text.trim().is_empty() {
            return 0;
        }
        let mut count = 0usize;
        let chars: Vec<char> = text.chars().collect();
        let len = chars.len();
        for i in 0..len {
            let c = chars[i];
            if c == '.' || c == '!' || c == '?' {
                // Check if followed by whitespace or end of string
                if i + 1 >= len || chars[i + 1].is_whitespace() {
                    count += 1;
                }
            }
        }
        // If no sentence-ending punctuation found, count as one sentence
        if count == 0 && !text.trim().is_empty() {
            count = 1;
        }
        count
    }

    /// Count words by splitting on whitespace.
    pub fn count_words(text: &str) -> usize {
        text.split_whitespace().count()
    }

    /// Count syllables in a word by counting vowel groups.
    pub fn count_syllables(word: &str) -> usize {
        let lower = word.to_lowercase();
        let vowels = ['a', 'e', 'i', 'o', 'u'];
        let chars: Vec<char> = lower.chars().collect();
        let mut count = 0usize;
        let mut prev_vowel = false;
        for &c in &chars {
            let is_vowel = vowels.contains(&c);
            if is_vowel && !prev_vowel {
                count += 1;
            }
            prev_vowel = is_vowel;
        }
        // Every word has at least one syllable
        count.max(1)
    }

    /// Flesch Reading Ease formula.
    pub fn flesch_reading_ease(asl: f64, asw: f64) -> f64 {
        206.835 - 1.015 * asl - 84.6 * asw
    }

    /// Flesch-Kincaid Grade Level formula.
    pub fn flesch_kincaid_grade(asl: f64, asw: f64) -> f64 {
        0.39 * asl + 11.8 * asw - 15.59
    }

    /// Compute all readability metrics for a block of text.
    pub fn compute_readability(&self, text: &str) -> ReadabilityMetrics {
        let sentence_count = Self::count_sentences(text).max(1);
        let word_count = Self::count_words(text).max(1);

        let asl = word_count as f64 / sentence_count as f64;

        let words: Vec<&str> = text.split_whitespace().collect();
        let total_syllables: usize = words
            .iter()
            .map(|w| {
                // Strip punctuation from word ends for syllable counting
                let clean: String = w.chars().filter(|c| c.is_alphabetic()).collect();
                Self::count_syllables(&clean)
            })
            .sum();
        let asw = total_syllables as f64 / word_count as f64;

        let flesch_score = Self::flesch_reading_ease(asl, asw);
        let grade_level = Self::flesch_kincaid_grade(asl, asw);

        // Vocabulary diversity: unique words / total words
        let unique_words: std::collections::HashSet<String> = words
            .iter()
            .map(|w| w.to_lowercase())
            .collect();
        let vocabulary_diversity = unique_words.len() as f64 / word_count as f64;

        // Average word length in characters
        let total_chars: usize = words.iter().map(|w| w.len()).sum();
        let avg_word_length = total_chars as f64 / word_count as f64;

        ReadabilityMetrics {
            avg_sentence_length: asl,
            avg_word_length,
            flesch_score,
            grade_level,
            vocabulary_diversity,
        }
    }

    /// Detect jargon terms in text (case-insensitive).
    pub fn detect_jargon(&self, text: &str) -> Vec<String> {
        let lower = text.to_lowercase();
        self.jargon_list
            .iter()
            .filter(|term| lower.contains(term.to_lowercase().as_str()))
            .cloned()
            .collect()
    }

    /// Compute a multi-dimensional response score.
    pub fn score_response(&self, text: &str) -> ResponseScore {
        let word_count = Self::count_words(text);
        let metrics = self.compute_readability(text);
        let jargon = self.detect_jargon(text);

        // Clarity: based on Flesch score (0..100 → 0..1), penalize jargon
        let flesch_norm = (metrics.flesch_score / 100.0).clamp(0.0, 1.0);
        let jargon_penalty = (jargon.len() as f64 * 0.05).min(0.5);
        let clarity = (flesch_norm - jargon_penalty).clamp(0.0, 1.0);

        // Conciseness: penalize if too long or too short
        let conciseness = if word_count < self.min_length {
            word_count as f64 / self.min_length as f64
        } else if word_count > self.max_length {
            self.max_length as f64 / word_count as f64
        } else {
            1.0
        };

        // Completeness: proxy via sentence count and length ratio
        let sentence_count = Self::count_sentences(text);
        let completeness = if sentence_count == 0 {
            0.0
        } else {
            (sentence_count as f64 / 5.0).min(1.0)
        };

        let overall = (clarity * 0.4 + conciseness * 0.3 + completeness * 0.3).clamp(0.0, 1.0);

        ResponseScore {
            clarity,
            conciseness,
            completeness,
            overall,
        }
    }

    /// Generate improvement suggestions for a response.
    pub fn suggest_enhancements(&self, text: &str) -> Vec<EnhancementSuggestion> {
        let mut suggestions = Vec::new();
        let word_count = Self::count_words(text);
        let metrics = self.compute_readability(text);
        let jargon = self.detect_jargon(text);

        if word_count < self.min_length {
            suggestions.push(EnhancementSuggestion {
                issue: ClarityIssue::TooShort,
                description: format!(
                    "Response is too short ({} words). Aim for at least {} words.",
                    word_count, self.min_length
                ),
                priority: 9,
            });
        }

        if word_count > self.max_length {
            suggestions.push(EnhancementSuggestion {
                issue: ClarityIssue::TooLong,
                description: format!(
                    "Response is too long ({} words). Consider trimming to {} words.",
                    word_count, self.max_length
                ),
                priority: 7,
            });
        }

        if metrics.vocabulary_diversity < 0.5 {
            suggestions.push(EnhancementSuggestion {
                issue: ClarityIssue::PoorVocabularyDiversity,
                description: format!(
                    "Vocabulary diversity is low ({:.0}%). Use more varied vocabulary.",
                    metrics.vocabulary_diversity * 100.0
                ),
                priority: 5,
            });
        }

        if metrics.flesch_score < 30.0 {
            suggestions.push(EnhancementSuggestion {
                issue: ClarityIssue::LowReadability,
                description: format!(
                    "Flesch score is very low ({:.1}). Simplify sentence structure.",
                    metrics.flesch_score
                ),
                priority: 8,
            });
        }

        if metrics.grade_level > self.target_grade_level {
            suggestions.push(EnhancementSuggestion {
                issue: ClarityIssue::HighGradeLevel,
                description: format!(
                    "Grade level ({:.1}) is above target ({:.1}). Simplify language.",
                    metrics.grade_level, self.target_grade_level
                ),
                priority: 6,
            });
        }

        // Structure: check for bullet points or numbered lists
        let has_structure = text.contains('\n')
            || text.contains("1.")
            || text.contains("- ")
            || text.contains("* ");
        if !has_structure && word_count > 100 {
            suggestions.push(EnhancementSuggestion {
                issue: ClarityIssue::MissingStructure,
                description: "Long response lacks structure. Consider using bullet points or numbered lists.".to_string(),
                priority: 4,
            });
        }

        if !jargon.is_empty() {
            suggestions.push(EnhancementSuggestion {
                issue: ClarityIssue::ExcessiveJargon(jargon.clone()),
                description: format!(
                    "Response contains jargon terms: {}. Use plain language instead.",
                    jargon.join(", ")
                ),
                priority: 6,
            });
        }

        suggestions
    }
}

impl Default for ResponseEnhancer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_syllables() {
        assert_eq!(ResponseEnhancer::count_syllables("cat"), 1);
        assert_eq!(ResponseEnhancer::count_syllables("hello"), 2);
        assert_eq!(ResponseEnhancer::count_syllables("beautiful"), 4);
        assert_eq!(ResponseEnhancer::count_syllables("a"), 1);
    }

    #[test]
    fn test_count_sentences() {
        assert_eq!(
            ResponseEnhancer::count_sentences("Hello. World. How are you?"),
            3
        );
        assert_eq!(
            ResponseEnhancer::count_sentences("No punctuation here"),
            1
        );
        assert_eq!(ResponseEnhancer::count_sentences(""), 0);
    }

    #[test]
    fn test_flesch_score_formula() {
        // Known values: asl=10, asw=1.5
        let score = ResponseEnhancer::flesch_reading_ease(10.0, 1.5);
        let expected = 206.835 - 1.015 * 10.0 - 84.6 * 1.5;
        assert!((score - expected).abs() < 1e-6, "Flesch formula mismatch");
    }

    #[test]
    fn test_jargon_detection() {
        let enhancer = ResponseEnhancer::new();
        let text = "We need to leverage synergy and utilize our bandwidth effectively.";
        let jargon = enhancer.detect_jargon(text);
        assert!(jargon.contains(&"leverage".to_string()));
        assert!(jargon.contains(&"synergy".to_string()));
        assert!(jargon.contains(&"utilize".to_string()));
        assert!(jargon.contains(&"bandwidth".to_string()));
    }

    #[test]
    fn test_short_text_triggers_too_short() {
        let enhancer = ResponseEnhancer::new();
        let text = "Short text.";
        let suggestions = enhancer.suggest_enhancements(text);
        let has_too_short = suggestions
            .iter()
            .any(|s| matches!(s.issue, ClarityIssue::TooShort));
        assert!(has_too_short, "Expected TooShort suggestion for short text");
    }

    #[test]
    fn test_jargon_list_has_15_terms() {
        let enhancer = ResponseEnhancer::new();
        assert!(
            enhancer.jargon_list.len() >= 15,
            "Expected at least 15 jargon terms"
        );
    }

    #[test]
    fn test_score_overall_in_range() {
        let enhancer = ResponseEnhancer::new();
        let text = "This is a well-written response. It covers the topic clearly. \
                    The language is simple and easy to understand. \
                    Each idea is explained concisely.";
        let score = enhancer.score_response(text);
        assert!(score.overall >= 0.0 && score.overall <= 1.0);
    }

    #[test]
    fn test_count_words() {
        assert_eq!(ResponseEnhancer::count_words("hello world foo"), 3);
        assert_eq!(ResponseEnhancer::count_words(""), 0);
    }
}
