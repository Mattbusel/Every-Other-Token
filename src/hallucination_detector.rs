//! Fact consistency checking for LLM outputs.
//!
//! Provides [`HallucinationDetector`] which can optionally be seeded with a
//! source document.  It extracts entity mentions and claims from LLM output,
//! checks them for grounding in the source (if provided), and measures
//! internal consistency to produce a [`HallucinationReport`].

// ── EntityMention ─────────────────────────────────────────────────────────────

/// A named entity or numeric pattern found in text.
#[derive(Debug, Clone, PartialEq)]
pub struct EntityMention {
    /// The surface form of the entity as it appeared in the text.
    pub text: String,
    /// Byte offset of the start of the mention (inclusive).
    pub start: usize,
    /// Byte offset of the end of the mention (exclusive).
    pub end: usize,
    /// Coarse entity type (`"PROPER_NOUN"`, `"DATE"`, `"PERCENT"`,
    /// `"QUANTITY"`, or `"NUMBER"`).
    pub entity_type: String,
}

// ── Claim extraction ──────────────────────────────────────────────────────────

/// Split `text` into sentences and return those that contain numbers or
/// probable proper nouns (words starting with an uppercase letter that are
/// not the first word of the sentence).
pub fn extract_claims(text: &str) -> Vec<String> {
    let sentences: Vec<&str> = text
        .split(|c| c == '.' || c == '!' || c == '?')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();

    sentences
        .into_iter()
        .filter(|s| {
            // Keep sentences that have a digit or a capitalised mid-sentence word
            let has_digit = s.chars().any(|c| c.is_ascii_digit());
            let has_proper = s.split_whitespace().skip(1).any(|w| {
                w.chars()
                    .next()
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false)
            });
            has_digit || has_proper
        })
        .map(str::to_owned)
        .collect()
}

// ── Entity extraction ─────────────────────────────────────────────────────────

/// Extract entity mentions from `text` using simple heuristics:
///
/// * Capitalised multi-word sequences (not at sentence start) → `PROPER_NOUN`
/// * `DD Month YYYY` or `YYYY-MM-DD` patterns → `DATE`
/// * Numbers followed by `%` → `PERCENT`
/// * Numbers followed by a unit word (km, kg, mi, lb, …) → `QUANTITY`
/// * Standalone integers / decimals → `NUMBER`
pub fn extract_entities(text: &str) -> Vec<EntityMention> {
    let mut entities = Vec::new();
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        // Skip whitespace
        if bytes[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }

        // Numeric patterns
        if bytes[i].is_ascii_digit() || (bytes[i] == b'-' && i + 1 < len && bytes[i + 1].is_ascii_digit()) {
            let start = i;
            while i < len && (bytes[i].is_ascii_digit() || bytes[i] == b'.' || bytes[i] == b'-') {
                i += 1;
            }
            // Peek at what follows
            let after = &text[i..];
            let after_trimmed = after.trim_start();
            let entity_type = if after_trimmed.starts_with('%') {
                i += 1 + (after.len() - after_trimmed.len()); // consume '%'
                "PERCENT"
            } else if after_trimmed
                .split_whitespace()
                .next()
                .map(|w| matches!(w, "km" | "kg" | "mi" | "lb" | "ms" | "s" | "m" | "ft" | "l"))
                .unwrap_or(false)
            {
                "QUANTITY"
            } else {
                "NUMBER"
            };
            entities.push(EntityMention {
                text: text[start..i].to_owned(),
                start,
                end: i,
                entity_type: entity_type.to_owned(),
            });
            continue;
        }

        // Capitalised word (potential proper noun or date)
        if bytes[i].is_ascii_uppercase() {
            let start = i;
            // Consume the first word
            while i < len && !bytes[i].is_ascii_whitespace() && bytes[i] != b'.' {
                i += 1;
            }
            let first_word = &text[start..i];

            // Check for month name (date pattern)
            let months = [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December",
            ];
            if months.contains(&first_word) {
                // Consume rest of date-like sequence
                let date_end_start = i;
                while i < len && (bytes[i].is_ascii_digit() || bytes[i].is_ascii_whitespace() || bytes[i] == b',') {
                    i += 1;
                }
                if i > date_end_start {
                    entities.push(EntityMention {
                        text: text[start..i].trim().to_owned(),
                        start,
                        end: i,
                        entity_type: "DATE".to_owned(),
                    });
                    continue;
                }
                i = date_end_start; // reset if nothing followed
            }

            // Try to accumulate a multi-word proper noun
            let mut end = i;
            loop {
                // Skip whitespace
                let ws_start = end;
                while end < len && bytes[end] == b' ' {
                    end += 1;
                }
                // Next word starts with uppercase?
                if end < len && bytes[end].is_ascii_uppercase() {
                    while end < len && !bytes[end].is_ascii_whitespace() && bytes[end] != b'.' {
                        end += 1;
                    }
                } else {
                    end = ws_start; // revert whitespace skip
                    break;
                }
            }
            i = end;
            entities.push(EntityMention {
                text: text[start..i].trim().to_owned(),
                start,
                end: i,
                entity_type: "PROPER_NOUN".to_owned(),
            });
            continue;
        }

        i += 1;
    }

    entities
}

// ── GroundingChecker ──────────────────────────────────────────────────────────

/// Checks whether claims can be grounded in a source document.
pub struct GroundingChecker {
    /// The source text used as the ground truth.
    pub source_text: String,
    /// Entities extracted from `source_text`.
    pub source_entities: Vec<EntityMention>,
}

impl GroundingChecker {
    /// Create a new grounding checker seeded with `source`.
    pub fn new(source: &str) -> Self {
        let source_entities = extract_entities(source);
        Self {
            source_text: source.to_owned(),
            source_entities,
        }
    }

    /// Return a confidence score in `[0.0, 1.0]` that `claim` is grounded in
    /// the source document.
    ///
    /// Heuristic: fraction of words in `claim` that also appear in
    /// `source_text` (case-insensitive), boosted when numeric entities in the
    /// claim are also present verbatim in the source.
    pub fn check_claim(&self, claim: &str) -> f64 {
        let claim_words: Vec<&str> = claim.split_whitespace().collect();
        if claim_words.is_empty() {
            return 1.0;
        }
        let source_lower = self.source_text.to_lowercase();
        let matched = claim_words
            .iter()
            .filter(|w| source_lower.contains(&w.to_lowercase()))
            .count();
        let word_score = matched as f64 / claim_words.len() as f64;

        // Entity grounding boost
        let claim_entities = extract_entities(claim);
        let entity_score = if claim_entities.is_empty() {
            1.0
        } else {
            let grounded = claim_entities
                .iter()
                .filter(|e| self.source_text.contains(&e.text))
                .count();
            grounded as f64 / claim_entities.len() as f64
        };

        (word_score + entity_score) / 2.0
    }
}

// ── ConsistencyChecker ────────────────────────────────────────────────────────

/// Checks a single text for internal consistency.
pub struct ConsistencyChecker;

impl ConsistencyChecker {
    /// Create a new consistency checker.
    pub fn new() -> Self {
        Self
    }

    /// Return a consistency score in `[0.0, 1.0]` for `text`.
    ///
    /// Penalises:
    /// * The same entity appearing with conflicting numeric values.
    /// * Negation words (`not`, `never`, `no`) in close proximity to
    ///   repeated positive assertions.
    pub fn check_internal_consistency(&self, text: &str) -> f64 {
        let entities = extract_entities(text);
        let mut contradictions = 0usize;
        let mut checks = 0usize;

        // Look for numeric entities with the same preceding context word but
        // different values.
        let sentences: Vec<&str> = text
            .split(|c| c == '.' || c == '!' || c == '?')
            .collect();

        let mut entity_values: std::collections::HashMap<String, Vec<f64>> =
            std::collections::HashMap::new();

        for entity in entities.iter().filter(|e| {
            matches!(e.entity_type.as_str(), "NUMBER" | "PERCENT" | "QUANTITY")
        }) {
            if let Ok(v) = entity.text.trim_end_matches('%').parse::<f64>() {
                // Use the word immediately before the entity as a "key"
                let before = &text[..entity.start];
                let key = before
                    .split_whitespace()
                    .last()
                    .unwrap_or("_")
                    .to_lowercase();
                entity_values.entry(key).or_default().push(v);
            }
        }

        for (_key, vals) in &entity_values {
            if vals.len() > 1 {
                checks += 1;
                let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                // Flag if the ratio differs by more than 20 %
                if min > 0.0 && (max - min) / min > 0.20 {
                    contradictions += 1;
                }
            }
        }

        // Check for negation proximity contradiction
        let negation_words = ["not", "never", "no", "cannot", "doesn't", "isn't", "wasn't"];
        for (i, sent) in sentences.iter().enumerate() {
            let has_neg = negation_words.iter().any(|n| sent.to_lowercase().contains(n));
            if has_neg && i + 1 < sentences.len() {
                let next = sentences[i + 1].to_lowercase();
                let words_i: std::collections::HashSet<&str> =
                    sent.split_whitespace().collect();
                let shared = next
                    .split_whitespace()
                    .filter(|w| words_i.contains(*w))
                    .count();
                if shared > 2 {
                    checks += 1;
                    contradictions += 1;
                }
            }
        }

        if checks == 0 {
            1.0
        } else {
            1.0 - (contradictions as f64 / checks as f64)
        }
    }
}

impl Default for ConsistencyChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ── HallucinationReport ───────────────────────────────────────────────────────

/// Summary of a hallucination analysis pass.
#[derive(Debug, Clone)]
pub struct HallucinationReport {
    /// Weighted average of grounding and consistency scores.
    pub overall_score: f64,
    /// Fraction of claims grounded in the source (`1.0` if no source).
    pub grounding_score: f64,
    /// Internal consistency score of the output.
    pub consistency_score: f64,
    /// Sentences/claims that appear poorly grounded.
    pub suspicious_claims: Vec<String>,
    /// `(claim_entity, nearest_source_entity)` pairs where values differ.
    pub entity_mismatches: Vec<(String, String)>,
}

impl HallucinationReport {
    /// Returns `true` if the overall score is below 0.5 (likely hallucinated).
    pub fn is_likely_hallucinated(&self) -> bool {
        self.overall_score < 0.5
    }
}

// ── HallucinationDetector ─────────────────────────────────────────────────────

/// Top-level hallucination detector.
pub struct HallucinationDetector {
    /// Optional grounding checker seeded from a source document.
    pub grounding_checker: Option<GroundingChecker>,
    /// Internal consistency checker.
    pub consistency_checker: ConsistencyChecker,
}

impl HallucinationDetector {
    /// Create a new detector, optionally seeded with a source document.
    pub fn new(source: Option<&str>) -> Self {
        Self {
            grounding_checker: source.map(GroundingChecker::new),
            consistency_checker: ConsistencyChecker::new(),
        }
    }

    /// Analyse `output` and return a [`HallucinationReport`].
    pub fn analyze(&self, output: &str) -> HallucinationReport {
        let consistency_score = self.consistency_checker.check_internal_consistency(output);

        let (grounding_score, suspicious_claims, entity_mismatches) =
            if let Some(gc) = &self.grounding_checker {
                let claims = extract_claims(output);
                let mut suspicious = Vec::new();
                let mut mismatches: Vec<(String, String)> = Vec::new();

                let mut total_score = 0.0f64;
                let n = claims.len();

                for claim in &claims {
                    let score = gc.check_claim(claim);
                    total_score += score;
                    if score < 0.5 {
                        suspicious.push(claim.clone());
                    }
                }

                // Entity mismatch: numeric entities in output not found in source
                let out_entities = extract_entities(output);
                for oe in &out_entities {
                    if matches!(oe.entity_type.as_str(), "NUMBER" | "PERCENT" | "QUANTITY") {
                        if !gc.source_text.contains(&oe.text) {
                            // Find the closest numeric entity in source by value
                            let nearest = gc
                                .source_entities
                                .iter()
                                .filter(|se| {
                                    matches!(
                                        se.entity_type.as_str(),
                                        "NUMBER" | "PERCENT" | "QUANTITY"
                                    )
                                })
                                .min_by_key(|se| {
                                    let a: f64 =
                                        oe.text.parse().unwrap_or(0.0);
                                    let b: f64 = se.text.parse().unwrap_or(0.0);
                                    ((a - b).abs() as u64)
                                });
                            let nearest_text = nearest
                                .map(|se| se.text.clone())
                                .unwrap_or_else(|| "(none)".to_owned());
                            mismatches.push((oe.text.clone(), nearest_text));
                        }
                    }
                }

                let gs = if n > 0 { total_score / n as f64 } else { 1.0 };
                (gs, suspicious, mismatches)
            } else {
                (1.0, Vec::new(), Vec::new())
            };

        let overall_score = (grounding_score + consistency_score) / 2.0;

        HallucinationReport {
            overall_score,
            grounding_score,
            consistency_score,
            suspicious_claims,
            entity_mismatches,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_claims_finds_numeric_sentences() {
        let text = "The sky is blue. The population is 8 billion people. Hello world.";
        let claims = extract_claims(text);
        assert!(claims.iter().any(|c| c.contains("8 billion")));
    }

    #[test]
    fn extract_entities_finds_number() {
        let entities = extract_entities("The temperature is 42 degrees.");
        assert!(entities.iter().any(|e| e.text == "42" && e.entity_type == "NUMBER"));
    }

    #[test]
    fn grounding_checker_high_score_on_match() {
        let source = "The GDP of France is 2.7 trillion USD in 2023.";
        let gc = GroundingChecker::new(source);
        let score = gc.check_claim("The GDP of France is 2.7 trillion USD.");
        assert!(score > 0.5, "score={score}");
    }

    #[test]
    fn hallucination_detector_no_source() {
        let det = HallucinationDetector::new(None);
        let report = det.analyze("The capital of France is Paris.");
        assert!(!report.is_likely_hallucinated());
    }

    #[test]
    fn hallucination_detector_with_source() {
        let source = "Paris is the capital of France. The population is 2 million.";
        let det = HallucinationDetector::new(Some(source));
        let report = det.analyze("Paris is the capital of France.");
        assert!(!report.is_likely_hallucinated());
    }
}
