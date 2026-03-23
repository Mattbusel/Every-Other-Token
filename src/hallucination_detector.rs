//! Hallucination detection via self-consistency and grounding.

use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Core data structures
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GroundingSource {
    pub id: String,
    pub content: String,
    pub credibility: f64, // 0..1
}

#[derive(Debug, Clone)]
pub struct ConsistencyCheck {
    pub statements: Vec<String>,
    pub agreement_threshold: f64,
}

#[derive(Debug, Clone)]
pub enum HallucinationIndicator {
    Uncertainty(f64),
    Contradiction { stmt_a: String, stmt_b: String },
    UngroundedClaim(String),
    ExcessiveHedging,
    ConfidenceCalibrationIssue,
}

#[derive(Debug, Clone)]
pub struct HallucinationReport {
    pub text: String,
    pub indicators: Vec<HallucinationIndicator>,
    pub risk_score: f64,
    pub grounded_fraction: f64,
}

// ---------------------------------------------------------------------------
// Detector
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct HallucinationDetector {
    pub sources: Vec<GroundingSource>,
    pub uncertainty_phrases: Vec<String>,
    pub contradiction_pairs: Vec<(String, String)>,
}

impl HallucinationDetector {
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            uncertainty_phrases: vec![
                "I think".to_string(),
                "I believe".to_string(),
                "possibly".to_string(),
                "might".to_string(),
                "could be".to_string(),
                "I'm not sure".to_string(),
                "approximately".to_string(),
                "roughly".to_string(),
            ],
            contradiction_pairs: vec![
                ("always".to_string(), "never".to_string()),
                ("all".to_string(), "none".to_string()),
                ("every".to_string(), "no".to_string()),
                ("definitely".to_string(), "possibly".to_string()),
            ],
        }
    }

    /// Count phrase hits / total words, clamped [0, 1].
    pub fn detect_uncertainty(text: &str, phrases: &[String]) -> f64 {
        if text.is_empty() {
            return 0.0;
        }
        let lower = text.to_lowercase();
        let total_words = text.split_whitespace().count();
        if total_words == 0 {
            return 0.0;
        }
        let mut hits = 0usize;
        for phrase in phrases {
            let phrase_lower = phrase.to_lowercase();
            let mut start = 0;
            while let Some(pos) = lower[start..].find(phrase_lower.as_str()) {
                hits += 1;
                start += pos + phrase_lower.len().max(1);
            }
        }
        (hits as f64 / total_words as f64).min(1.0)
    }

    /// Detect contradictory word pairs within the text.
    pub fn detect_contradictions(
        text: &str,
        pairs: &[(&str, &str)],
    ) -> Vec<HallucinationIndicator> {
        let lower = text.to_lowercase();
        let words: HashSet<&str> = lower.split_whitespace().collect();
        let mut indicators = Vec::new();
        for (a, b) in pairs {
            let a_lower = a.to_lowercase();
            let b_lower = b.to_lowercase();
            if words.contains(a_lower.as_str()) && words.contains(b_lower.as_str()) {
                indicators.push(HallucinationIndicator::Contradiction {
                    stmt_a: a.to_string(),
                    stmt_b: b.to_string(),
                });
            }
        }
        indicators
    }

    /// Ground a statement against sources.
    /// Returns max(credibility * word_overlap) across all sources.
    pub fn ground_statement(statement: &str, sources: &[GroundingSource]) -> f64 {
        if sources.is_empty() || statement.is_empty() {
            return 0.0;
        }
        let stmt_words: HashSet<String> = statement
            .to_lowercase()
            .split_whitespace()
            .map(|w| w.to_string())
            .collect();

        let mut best = 0.0f64;
        for src in sources {
            let src_words: HashSet<String> = src
                .content
                .to_lowercase()
                .split_whitespace()
                .map(|w| w.to_string())
                .collect();

            let intersection = stmt_words.intersection(&src_words).count();
            let union = stmt_words.union(&src_words).count();
            let overlap = if union == 0 {
                0.0
            } else {
                intersection as f64 / union as f64
            };
            let score = src.credibility * overlap;
            if score > best {
                best = score;
            }
        }
        best
    }

    /// Run all detectors and produce a full report.
    pub fn analyze(&self, text: &str) -> HallucinationReport {
        let mut indicators: Vec<HallucinationIndicator> = Vec::new();

        // Uncertainty
        let uncertainty_score = Self::detect_uncertainty(text, &self.uncertainty_phrases);
        if uncertainty_score > 0.0 {
            indicators.push(HallucinationIndicator::Uncertainty(uncertainty_score));
        }

        // Excessive hedging (uncertainty very high)
        if uncertainty_score > 0.3 {
            indicators.push(HallucinationIndicator::ExcessiveHedging);
        }

        // Contradictions
        let pairs_ref: Vec<(&str, &str)> = self
            .contradiction_pairs
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        let contradictions = Self::detect_contradictions(text, &pairs_ref);
        indicators.extend(contradictions);

        // Grounding per sentence
        let sentences: Vec<&str> = text
            .split(|c| c == '.' || c == '!' || c == '?')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        let total_sentences = sentences.len().max(1);
        let mut grounded_count = 0usize;
        for sentence in &sentences {
            let g = Self::ground_statement(sentence, &self.sources);
            if g < 0.05 && !self.sources.is_empty() {
                indicators.push(HallucinationIndicator::UngroundedClaim(
                    sentence.to_string(),
                ));
            } else {
                grounded_count += 1;
            }
        }

        let grounded_fraction = if self.sources.is_empty() {
            1.0 // no sources provided — cannot assess grounding
        } else {
            grounded_count as f64 / total_sentences as f64
        };

        // Confidence calibration issue: low grounding but no uncertainty phrases
        if grounded_fraction < 0.5 && uncertainty_score < 0.05 && !self.sources.is_empty() {
            indicators.push(HallucinationIndicator::ConfidenceCalibrationIssue);
        }

        // Risk score: weighted combination
        let contradiction_weight = indicators
            .iter()
            .filter(|i| matches!(i, HallucinationIndicator::Contradiction { .. }))
            .count() as f64
            * 0.3;
        let ungrounded_weight = if self.sources.is_empty() {
            0.0
        } else {
            (1.0 - grounded_fraction) * 0.4
        };
        let uncertainty_weight = uncertainty_score * 0.2;
        let calibration_weight = if indicators
            .iter()
            .any(|i| matches!(i, HallucinationIndicator::ConfidenceCalibrationIssue))
        {
            0.2
        } else {
            0.0
        };

        let risk_score =
            (contradiction_weight + ungrounded_weight + uncertainty_weight + calibration_weight)
                .min(1.0);

        HallucinationReport {
            text: text.to_string(),
            indicators,
            risk_score,
            grounded_fraction,
        }
    }

    /// Classify risk score into a human-readable level.
    pub fn risk_level(score: f64) -> &'static str {
        if score < 0.3 {
            "low"
        } else if score < 0.6 {
            "medium"
        } else {
            "high"
        }
    }
}

impl Default for HallucinationDetector {
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
    fn test_uncertainty_phrase_detection() {
        let detector = HallucinationDetector::new();
        let text = "I think this might be correct, possibly even roughly accurate.";
        let score = HallucinationDetector::detect_uncertainty(text, &detector.uncertainty_phrases);
        assert!(score > 0.0, "Expected nonzero uncertainty score");
        assert!(score <= 1.0, "Score must be clamped to 1.0");
    }

    #[test]
    fn test_uncertainty_empty_text() {
        let detector = HallucinationDetector::new();
        let score = HallucinationDetector::detect_uncertainty("", &detector.uncertainty_phrases);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_contradiction_detection() {
        let pairs: Vec<(&str, &str)> = vec![("always", "never"), ("all", "none")];
        let text = "It always works but sometimes never does.";
        let indicators = HallucinationDetector::detect_contradictions(text, &pairs);
        assert!(
            indicators.len() >= 1,
            "Expected at least one contradiction indicator"
        );
        let has_always_never = indicators.iter().any(|i| {
            matches!(i, HallucinationIndicator::Contradiction { stmt_a, stmt_b }
                if stmt_a == "always" && stmt_b == "never")
        });
        assert!(has_always_never, "Expected always/never contradiction");
    }

    #[test]
    fn test_no_contradiction_when_absent() {
        let pairs: Vec<(&str, &str)> = vec![("always", "never")];
        let text = "It always works well.";
        let indicators = HallucinationDetector::detect_contradictions(text, &pairs);
        assert!(indicators.is_empty(), "No contradictions expected");
    }

    #[test]
    fn test_grounding_with_matching_source() {
        let sources = vec![GroundingSource {
            id: "src1".to_string(),
            content: "The sky is blue and clouds are white".to_string(),
            credibility: 1.0,
        }];
        let score = HallucinationDetector::ground_statement("The sky is blue", &sources);
        assert!(score > 0.0, "Expected positive grounding for matching source");
    }

    #[test]
    fn test_grounding_with_non_matching_source() {
        let sources = vec![GroundingSource {
            id: "src1".to_string(),
            content: "quantum mechanics describes subatomic particles".to_string(),
            credibility: 1.0,
        }];
        let score = HallucinationDetector::ground_statement("The sky is blue", &sources);
        assert!(score < 0.3, "Expected low grounding for non-matching source");
    }

    #[test]
    fn test_risk_score_bounds() {
        let mut detector = HallucinationDetector::new();
        detector.sources.push(GroundingSource {
            id: "s".to_string(),
            content: "some content about facts".to_string(),
            credibility: 0.8,
        });
        let report = detector.analyze(
            "This definitely always works and never fails in all cases every time.",
        );
        assert!(report.risk_score >= 0.0, "Risk score must be >= 0");
        assert!(report.risk_score <= 1.0, "Risk score must be <= 1");
    }

    #[test]
    fn test_empty_text_analysis() {
        let detector = HallucinationDetector::new();
        let report = detector.analyze("");
        assert_eq!(report.risk_score, 0.0);
        assert!(report.indicators.is_empty());
    }

    #[test]
    fn test_risk_level() {
        assert_eq!(HallucinationDetector::risk_level(0.1), "low");
        assert_eq!(HallucinationDetector::risk_level(0.45), "medium");
        assert_eq!(HallucinationDetector::risk_level(0.75), "high");
    }
}
