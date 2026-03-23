//! Claim extraction and source-based fact verification.

/// The type of a factual claim.
#[derive(Debug, Clone, PartialEq)]
pub enum ClaimType {
    Factual,
    Statistical,
    Causal,
    Comparative,
    Temporal,
}

/// A claim extracted from text.
#[derive(Debug, Clone)]
pub struct Claim {
    pub text: String,
    pub claim_type: ClaimType,
    pub entities: Vec<String>,
    pub confidence: f64,
}

/// The result of verifying a claim against sources.
#[derive(Debug, Clone)]
pub enum VerificationResult {
    Supported(f64),
    Contradicted(f64),
    Unverifiable,
    Partial(f64),
}

/// A source of factual information used for verification.
#[derive(Debug, Clone)]
pub struct FactSource {
    pub id: String,
    pub content: String,
    pub credibility: f64,
    pub domain: String,
}

/// A report on a single claim's verification.
#[derive(Debug, Clone)]
pub struct FactCheckReport {
    pub claim: Claim,
    pub result: VerificationResult,
    pub supporting_sources: Vec<String>,
    pub explanation: String,
}

/// Extracts claims from raw text.
pub struct ClaimExtractor;

impl ClaimExtractor {
    pub fn extract(text: &str) -> Vec<Claim> {
        let sentences: Vec<&str> = text
            .split(|c| c == '.' || c == '!' || c == '?')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();

        let mut claims = Vec::new();

        for sentence in sentences {
            let lower = sentence.to_lowercase();

            // Detect claim type
            let claim_type = if has_statistical_markers(&lower) {
                ClaimType::Statistical
            } else if lower.contains("because")
                || lower.contains("causes")
                || lower.contains("leads to")
                || lower.contains("results in")
            {
                ClaimType::Causal
            } else if lower.contains("more than")
                || lower.contains("less than")
                || lower.contains("better than")
                || lower.contains("worse than")
            {
                ClaimType::Comparative
            } else if has_temporal_markers(&lower) {
                ClaimType::Temporal
            } else if is_factual(sentence) {
                ClaimType::Factual
            } else {
                continue;
            };

            let entities = extract_entities(sentence);

            claims.push(Claim {
                text: sentence.to_string(),
                claim_type,
                entities,
                confidence: 0.7,
            });
        }

        claims
    }
}

fn has_statistical_markers(lower: &str) -> bool {
    if lower.contains('%') || lower.contains("percent") {
        return true;
    }
    // digit followed by a unit-like word
    let words: Vec<&str> = lower.split_whitespace().collect();
    for (i, w) in words.iter().enumerate() {
        if w.chars().any(|c| c.is_ascii_digit()) {
            if i + 1 < words.len() {
                let next = words[i + 1];
                if matches!(next, "million" | "billion" | "thousand" | "km" | "kg" | "mb" | "gb") {
                    return true;
                }
            }
        }
    }
    false
}

fn has_temporal_markers(lower: &str) -> bool {
    if lower.contains("since") || lower.contains("until") || lower.contains("by 20") {
        return true;
    }
    // 4-digit year pattern
    let words: Vec<&str> = lower.split_whitespace().collect();
    for w in words {
        let digits: String = w.chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.len() == 4 {
            if let Ok(year) = digits.parse::<u32>() {
                if year >= 1000 && year <= 9999 {
                    return true;
                }
            }
        }
    }
    false
}

fn is_factual(sentence: &str) -> bool {
    let words: Vec<&str> = sentence.split_whitespace().collect();
    if words.is_empty() {
        return false;
    }
    let first = words[0];
    // Starts with a capital letter word
    if !first.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
        return false;
    }
    let lower = sentence.to_lowercase();
    lower.contains(" is ")
        || lower.contains(" are ")
        || lower.contains(" was ")
        || lower.contains(" were ")
}

fn extract_entities(sentence: &str) -> Vec<String> {
    sentence
        .split_whitespace()
        .filter(|w| {
            let clean: String = w.chars().filter(|c| c.is_alphabetic()).collect();
            clean.len() >= 4
                && clean.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
        })
        .map(|w| w.chars().filter(|c| c.is_alphabetic()).collect::<String>())
        .collect()
}

fn token_overlap(a: &str, b: &str) -> f64 {
    let a_tokens: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let b_tokens: std::collections::HashSet<&str> = b.split_whitespace().collect();
    if a_tokens.is_empty() {
        return 0.0;
    }
    let intersection = a_tokens.intersection(&b_tokens).count();
    intersection as f64 / a_tokens.len() as f64
}

/// Verifies claims against a set of fact sources.
pub struct FactChecker {
    pub sources: Vec<FactSource>,
}

impl FactChecker {
    pub fn new(sources: Vec<FactSource>) -> Self {
        Self { sources }
    }

    pub fn verify_claim(&self, claim: &Claim) -> FactCheckReport {
        let mut best_score = 0.0_f64;
        let mut best_source_id = None;
        let mut supporting_sources = Vec::new();

        for source in &self.sources {
            let overlap = token_overlap(&claim.text.to_lowercase(), &source.content.to_lowercase());
            let score = source.credibility * overlap;
            if score > 0.2 {
                supporting_sources.push(source.id.clone());
            }
            if score > best_score {
                best_score = score;
                best_source_id = Some(source.id.clone());
            }
        }

        let result = if best_score > 0.5 {
            VerificationResult::Supported(best_score)
        } else if best_score > 0.2 {
            VerificationResult::Partial(best_score)
        } else {
            VerificationResult::Unverifiable
        };

        let explanation = match &best_source_id {
            Some(id) => format!(
                "Best matching source: {} (score: {:.2})",
                id, best_score
            ),
            None => "No matching sources found.".to_string(),
        };

        FactCheckReport {
            claim: claim.clone(),
            result,
            supporting_sources,
            explanation,
        }
    }

    pub fn check_text(&self, text: &str) -> Vec<FactCheckReport> {
        let claims = ClaimExtractor::extract(text);
        claims.iter().map(|c| self.verify_claim(c)).collect()
    }
}

/// Computes the overall credibility of a set of fact-check reports.
pub fn overall_credibility(reports: &[FactCheckReport]) -> f64 {
    if reports.is_empty() {
        return 0.0;
    }
    let supported = reports
        .iter()
        .filter(|r| {
            matches!(
                r.result,
                VerificationResult::Supported(_) | VerificationResult::Partial(_)
            )
        })
        .count();
    supported as f64 / reports.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claim_extraction_factual() {
        let claims = ClaimExtractor::extract("Water is a molecule. The sky is blue.");
        let types: Vec<&ClaimType> = claims.iter().map(|c| &c.claim_type).collect();
        assert!(types.contains(&&ClaimType::Factual));
    }

    #[test]
    fn claim_extraction_statistical() {
        let claims = ClaimExtractor::extract("The rate increased by 42%.");
        assert!(!claims.is_empty());
        assert_eq!(claims[0].claim_type, ClaimType::Statistical);
    }

    #[test]
    fn claim_extraction_causal() {
        let claims = ClaimExtractor::extract("Smoking causes cancer in humans.");
        assert!(!claims.is_empty());
        assert_eq!(claims[0].claim_type, ClaimType::Causal);
    }

    #[test]
    fn claim_extraction_temporal() {
        let claims = ClaimExtractor::extract("This happened since 2015 in Europe.");
        assert!(!claims.is_empty());
        assert_eq!(claims[0].claim_type, ClaimType::Temporal);
    }

    #[test]
    fn entity_extraction() {
        let entities = extract_entities("Albert Einstein discovered relativity in Germany.");
        assert!(entities.contains(&"Albert".to_string()) || entities.contains(&"Einstein".to_string()));
    }

    #[test]
    fn fact_check_supported() {
        let source = FactSource {
            id: "src1".to_string(),
            content: "Water is a molecule composed of hydrogen and oxygen.".to_string(),
            credibility: 1.0,
            domain: "science".to_string(),
        };
        let checker = FactChecker::new(vec![source]);
        let claim = Claim {
            text: "Water is a molecule".to_string(),
            claim_type: ClaimType::Factual,
            entities: vec!["Water".to_string()],
            confidence: 0.8,
        };
        let report = checker.verify_claim(&claim);
        assert!(matches!(report.result, VerificationResult::Supported(_) | VerificationResult::Partial(_)));
    }

    #[test]
    fn fact_check_unverifiable() {
        let source = FactSource {
            id: "src1".to_string(),
            content: "The quick brown fox jumps over the lazy dog.".to_string(),
            credibility: 1.0,
            domain: "literature".to_string(),
        };
        let checker = FactChecker::new(vec![source]);
        let claim = Claim {
            text: "Quantum computers will replace classical computers by 2040".to_string(),
            claim_type: ClaimType::Temporal,
            entities: vec![],
            confidence: 0.5,
        };
        let report = checker.verify_claim(&claim);
        assert!(matches!(report.result, VerificationResult::Unverifiable | VerificationResult::Partial(_)));
    }

    #[test]
    fn credibility_score() {
        let reports = vec![
            FactCheckReport {
                claim: Claim {
                    text: "test".to_string(),
                    claim_type: ClaimType::Factual,
                    entities: vec![],
                    confidence: 0.8,
                },
                result: VerificationResult::Supported(0.9),
                supporting_sources: vec![],
                explanation: "test".to_string(),
            },
            FactCheckReport {
                claim: Claim {
                    text: "test2".to_string(),
                    claim_type: ClaimType::Factual,
                    entities: vec![],
                    confidence: 0.5,
                },
                result: VerificationResult::Unverifiable,
                supporting_sources: vec![],
                explanation: "no match".to_string(),
            },
        ];
        let score = overall_credibility(&reports);
        assert!((score - 0.5).abs() < 0.001);
    }
}
