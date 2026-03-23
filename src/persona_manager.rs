//! System persona management and style enforcement.
//!
//! Defines named personas with tone, vocabulary level, and forbidden/required
//! phrase lists.  The [`StyleEnforcer`] validates model responses against the
//! active persona and returns actionable [`StyleViolation`]s.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Tone enum
// ---------------------------------------------------------------------------

/// The conversational register of a persona.
#[derive(Debug, Clone, PartialEq)]
pub enum PersonaTone {
    Formal,
    Casual,
    Technical,
    Friendly,
    Academic,
    Professional,
    Concise,
    Verbose,
}

impl fmt::Display for PersonaTone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PersonaTone::Formal       => write!(f, "Formal"),
            PersonaTone::Casual       => write!(f, "Casual"),
            PersonaTone::Technical    => write!(f, "Technical"),
            PersonaTone::Friendly     => write!(f, "Friendly"),
            PersonaTone::Academic     => write!(f, "Academic"),
            PersonaTone::Professional => write!(f, "Professional"),
            PersonaTone::Concise      => write!(f, "Concise"),
            PersonaTone::Verbose      => write!(f, "Verbose"),
        }
    }
}

// ---------------------------------------------------------------------------
// Style structs
// ---------------------------------------------------------------------------

/// Numerical and boolean style parameters for a persona.
#[derive(Debug, Clone)]
pub struct PersonaStyle {
    pub tone: PersonaTone,
    /// Vocabulary complexity: 1 = simple, 10 = expert.
    pub vocabulary_level: u8,
    pub uses_jargon: bool,
    pub uses_humor: bool,
    /// Soft cap on sentence length in words.
    pub max_sentence_length: usize,
    pub prefers_bullets: bool,
}

/// A named persona with its system prompt and style constraints.
#[derive(Debug, Clone)]
pub struct Persona {
    pub id: String,
    pub name: String,
    pub description: String,
    pub system_prompt: String,
    pub style: PersonaStyle,
    pub forbidden_phrases: Vec<String>,
    pub required_phrases: Vec<String>,
    pub example_responses: Vec<String>,
}

// ---------------------------------------------------------------------------
// Violations
// ---------------------------------------------------------------------------

/// A single style rule violation with remediation advice.
#[derive(Debug, Clone)]
pub struct StyleViolation {
    pub rule_name: String,
    pub description: String,
    /// 0.0–1.0, where 1.0 is most severe.
    pub severity: f64,
    pub suggestion: String,
}

// ---------------------------------------------------------------------------
// Informal / jargon word lists
// ---------------------------------------------------------------------------

const INFORMAL_WORDS: &[&str] = &[
    "gonna", "wanna", "gotta", "yeah", "nope", "yep", "ok", "okay",
    "kinda", "sorta", "dunno", "cuz", "cos", "tbh", "lol", "omg",
    "btw", "fyi", "imo", "imho", "stuff", "things", "guys",
];

const TECHNICAL_JARGON: &[&str] = &[
    "algorithm", "heuristic", "latency", "throughput", "bandwidth",
    "polymorphism", "encapsulation", "refactoring", "idempotent",
    "asynchronous", "paradigm", "recursion", "instantiation",
    "compilation", "tokenization", "embedding", "inference",
    "gradient", "hyperparameter", "epoch", "backpropagation",
];

const COMPLEX_WORDS: &[&str] = &[
    "ubiquitous", "ephemeral", "paradigm", "methodology", "utilise",
    "leverage", "synergize", "proactive", "holistic", "robust",
    "scalable", "granular", "iterate", "facilitate", "exacerbate",
    "ameliorate", "disambiguation", "concatenation",
];

// ---------------------------------------------------------------------------
// StyleEnforcer
// ---------------------------------------------------------------------------

/// Checks model text for style-rule compliance against a [`Persona`].
#[derive(Debug, Default)]
pub struct StyleEnforcer;

impl StyleEnforcer {
    pub fn new() -> Self {
        StyleEnforcer
    }

    /// Check for tone mismatches (e.g., informal words in a formal persona).
    pub fn check_tone(text: &str, expected: &PersonaTone) -> Vec<StyleViolation> {
        let lower = text.to_lowercase();
        let mut violations = Vec::new();

        match expected {
            PersonaTone::Formal | PersonaTone::Academic | PersonaTone::Professional => {
                for word in INFORMAL_WORDS {
                    if lower.split_whitespace().any(|w| w.trim_matches(|c: char| !c.is_alphabetic()) == *word) {
                        violations.push(StyleViolation {
                            rule_name: "informal_word".to_string(),
                            description: format!("Informal word '{}' used in {} context.", word, expected),
                            severity: 0.6,
                            suggestion: format!("Replace '{}' with a more formal equivalent.", word),
                        });
                    }
                }
            }
            PersonaTone::Concise => {
                let word_count = text.split_whitespace().count();
                if word_count > 100 {
                    violations.push(StyleViolation {
                        rule_name: "too_verbose".to_string(),
                        description: format!("Response has {} words; concise tone expects ≤ 100.", word_count),
                        severity: 0.5,
                        suggestion: "Shorten the response significantly.".to_string(),
                    });
                }
            }
            PersonaTone::Verbose => {
                let word_count = text.split_whitespace().count();
                if word_count < 30 {
                    violations.push(StyleViolation {
                        rule_name: "too_brief".to_string(),
                        description: format!("Response has only {} words; verbose tone expects more detail.", word_count),
                        severity: 0.4,
                        suggestion: "Expand the response with more detail and examples.".to_string(),
                    });
                }
            }
            _ => {}
        }
        violations
    }

    /// Flag complex vocabulary when a low vocabulary level is expected.
    pub fn check_vocabulary(text: &str, level: u8) -> Vec<StyleViolation> {
        let mut violations = Vec::new();
        if level > 5 {
            return violations;
        }
        let lower = text.to_lowercase();
        for word in COMPLEX_WORDS {
            if lower.contains(word) {
                violations.push(StyleViolation {
                    rule_name: "complex_vocabulary".to_string(),
                    description: format!("Word '{}' may be too complex for vocabulary level {}.", word, level),
                    severity: 0.3,
                    suggestion: format!("Simplify '{}' to a more common word.", word),
                });
            }
        }
        violations
    }

    /// Flag sentences that exceed `max_len` words.
    pub fn check_sentence_length(text: &str, max_len: usize) -> Vec<StyleViolation> {
        let mut violations = Vec::new();
        for sentence in text.split('.') {
            let wc = sentence.split_whitespace().count();
            if wc > max_len {
                violations.push(StyleViolation {
                    rule_name: "sentence_too_long".to_string(),
                    description: format!(
                        "Sentence has {} words (limit: {}).",
                        wc, max_len
                    ),
                    severity: 0.4,
                    suggestion: "Split this sentence into shorter ones.".to_string(),
                });
            }
        }
        violations
    }

    /// Flag forbidden phrases.
    pub fn check_forbidden(text: &str, forbidden: &[String]) -> Vec<StyleViolation> {
        let lower = text.to_lowercase();
        forbidden
            .iter()
            .filter(|p| lower.contains(p.as_str()))
            .map(|p| StyleViolation {
                rule_name: "forbidden_phrase".to_string(),
                description: format!("Forbidden phrase '{}' found.", p),
                severity: 0.8,
                suggestion: format!("Remove or replace the phrase '{}'.", p),
            })
            .collect()
    }

    /// Flag presence/absence of technical jargon based on persona expectation.
    pub fn check_jargon(text: &str, expects_jargon: bool) -> Vec<StyleViolation> {
        let lower = text.to_lowercase();
        let jargon_found = TECHNICAL_JARGON
            .iter()
            .filter(|&&w| lower.contains(w))
            .count();

        let mut violations = Vec::new();
        if expects_jargon && jargon_found == 0 {
            violations.push(StyleViolation {
                rule_name: "missing_jargon".to_string(),
                description: "Technical persona expects domain-specific terminology, but none found.".to_string(),
                severity: 0.3,
                suggestion: "Add relevant technical terms to match the persona's expertise.".to_string(),
            });
        } else if !expects_jargon && jargon_found > 2 {
            violations.push(StyleViolation {
                rule_name: "excessive_jargon".to_string(),
                description: format!("{} technical terms found; non-technical persona should avoid jargon.", jargon_found),
                severity: 0.4,
                suggestion: "Replace technical jargon with plain-language equivalents.".to_string(),
            });
        }
        violations
    }

    /// Run all checks against a persona and return all violations.
    pub fn enforce(&self, text: &str, persona: &Persona) -> Vec<StyleViolation> {
        let mut all = Vec::new();
        all.extend(Self::check_tone(text, &persona.style.tone));
        all.extend(Self::check_vocabulary(text, persona.style.vocabulary_level));
        all.extend(Self::check_sentence_length(text, persona.style.max_sentence_length));
        all.extend(Self::check_forbidden(text, &persona.forbidden_phrases));
        all.extend(Self::check_jargon(text, persona.style.uses_jargon));
        all
    }

    /// Compute a style score in [0, 1] from a list of violations.
    pub fn style_score(violations: &[StyleViolation]) -> f64 {
        let penalty: f64 = violations.iter().map(|v| v.severity).sum();
        (1.0 - penalty).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// PersonaManager
// ---------------------------------------------------------------------------

/// Registry of named personas with response validation helpers.
#[derive(Debug, Default)]
pub struct PersonaManager {
    personas: HashMap<String, Persona>,
    enforcer: StyleEnforcer,
}

impl PersonaManager {
    pub fn new() -> Self {
        PersonaManager {
            personas: HashMap::new(),
            enforcer: StyleEnforcer::new(),
        }
    }

    /// Register a persona.
    pub fn register(&mut self, persona: Persona) {
        self.personas.insert(persona.id.clone(), persona);
    }

    /// Look up a persona by id.
    pub fn get(&self, id: &str) -> Option<&Persona> {
        self.personas.get(id)
    }

    /// Validate a response against the named persona.
    pub fn validate_response(&self, persona_id: &str, response: &str) -> Vec<StyleViolation> {
        match self.personas.get(persona_id) {
            Some(persona) => self.enforcer.enforce(response, persona),
            None => vec![StyleViolation {
                rule_name: "unknown_persona".to_string(),
                description: format!("Persona '{}' not found.", persona_id),
                severity: 1.0,
                suggestion: "Register the persona before validating responses.".to_string(),
            }],
        }
    }

    /// Prepend the persona's system prompt to `prompt`.
    pub fn apply_persona_prefix(&self, persona_id: &str, prompt: &str) -> String {
        match self.personas.get(persona_id) {
            Some(persona) => format!("{}\n\n{}", persona.system_prompt, prompt),
            None => prompt.to_string(),
        }
    }

    /// Create a manager pre-loaded with five built-in personas.
    pub fn built_in_personas() -> Self {
        let mut mgr = Self::new();

        mgr.register(Persona {
            id: "assistant".to_string(),
            name: "General Assistant".to_string(),
            description: "A helpful, neutral assistant.".to_string(),
            system_prompt: "You are a helpful assistant. Answer clearly and concisely.".to_string(),
            style: PersonaStyle {
                tone: PersonaTone::Friendly,
                vocabulary_level: 5,
                uses_jargon: false,
                uses_humor: false,
                max_sentence_length: 25,
                prefers_bullets: false,
            },
            forbidden_phrases: vec!["I cannot".to_string(), "As an AI".to_string()],
            required_phrases: vec![],
            example_responses: vec![
                "Sure! Here's what you need to know: …".to_string(),
            ],
        });

        mgr.register(Persona {
            id: "tutor".to_string(),
            name: "Patient Tutor".to_string(),
            description: "Explains concepts step by step for learners.".to_string(),
            system_prompt: "You are a patient tutor. Break down concepts clearly with examples.".to_string(),
            style: PersonaStyle {
                tone: PersonaTone::Friendly,
                vocabulary_level: 3,
                uses_jargon: false,
                uses_humor: true,
                max_sentence_length: 20,
                prefers_bullets: true,
            },
            forbidden_phrases: vec!["obviously".to_string(), "trivially".to_string()],
            required_phrases: vec![],
            example_responses: vec![
                "Great question! Let's break this down step by step.".to_string(),
            ],
        });

        mgr.register(Persona {
            id: "coder".to_string(),
            name: "Expert Coder".to_string(),
            description: "Provides technical coding help with precision.".to_string(),
            system_prompt: "You are an expert software engineer. Provide precise, idiomatic code and explanations.".to_string(),
            style: PersonaStyle {
                tone: PersonaTone::Technical,
                vocabulary_level: 8,
                uses_jargon: true,
                uses_humor: false,
                max_sentence_length: 30,
                prefers_bullets: true,
            },
            forbidden_phrases: vec![],
            required_phrases: vec![],
            example_responses: vec![
                "Here's an idiomatic implementation using iterators: …".to_string(),
            ],
        });

        mgr.register(Persona {
            id: "analyst".to_string(),
            name: "Data Analyst".to_string(),
            description: "Interprets data and provides statistical insights.".to_string(),
            system_prompt: "You are a rigorous data analyst. Support all claims with data and statistics.".to_string(),
            style: PersonaStyle {
                tone: PersonaTone::Academic,
                vocabulary_level: 7,
                uses_jargon: true,
                uses_humor: false,
                max_sentence_length: 35,
                prefers_bullets: false,
            },
            forbidden_phrases: vec!["I think".to_string(), "I feel".to_string()],
            required_phrases: vec![],
            example_responses: vec![
                "Based on the available data, the correlation coefficient is …".to_string(),
            ],
        });

        mgr.register(Persona {
            id: "creative_writer".to_string(),
            name: "Creative Writer".to_string(),
            description: "Crafts engaging narratives with vivid language.".to_string(),
            system_prompt: "You are a creative writer. Use vivid imagery, varied sentence rhythm, and narrative flair.".to_string(),
            style: PersonaStyle {
                tone: PersonaTone::Verbose,
                vocabulary_level: 7,
                uses_jargon: false,
                uses_humor: true,
                max_sentence_length: 40,
                prefers_bullets: false,
            },
            forbidden_phrases: vec!["in conclusion".to_string(), "to summarize".to_string()],
            required_phrases: vec![],
            example_responses: vec![
                "The sun dipped below the horizon like a coin vanishing into still water …".to_string(),
            ],
        });

        mgr
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_formal_persona() -> Persona {
        Persona {
            id: "formal".to_string(),
            name: "Formal Bot".to_string(),
            description: "Strictly formal".to_string(),
            system_prompt: "Be formal.".to_string(),
            style: PersonaStyle {
                tone: PersonaTone::Formal,
                vocabulary_level: 6,
                uses_jargon: false,
                uses_humor: false,
                max_sentence_length: 20,
                prefers_bullets: false,
            },
            forbidden_phrases: vec!["gonna".to_string()],
            required_phrases: vec![],
            example_responses: vec![],
        }
    }

    #[test]
    fn test_informal_in_formal_context() {
        let v = StyleEnforcer::check_tone("I'm gonna do that.", &PersonaTone::Formal);
        assert!(!v.is_empty());
    }

    #[test]
    fn test_forbidden_phrase() {
        let p = make_formal_persona();
        let e = StyleEnforcer::new();
        let v = e.enforce("I'm gonna help.", &p);
        assert!(v.iter().any(|x| x.rule_name == "forbidden_phrase" || x.rule_name == "informal_word"));
    }

    #[test]
    fn test_style_score_no_violations() {
        let score = StyleEnforcer::style_score(&[]);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_built_in_personas() {
        let mgr = PersonaManager::built_in_personas();
        assert!(mgr.get("assistant").is_some());
        assert!(mgr.get("coder").is_some());
        assert!(mgr.get("tutor").is_some());
        assert!(mgr.get("analyst").is_some());
        assert!(mgr.get("creative_writer").is_some());
    }

    #[test]
    fn test_apply_persona_prefix() {
        let mgr = PersonaManager::built_in_personas();
        let result = mgr.apply_persona_prefix("assistant", "Hello?");
        assert!(result.contains("Hello?"));
        assert!(result.contains("helpful assistant"));
    }

    #[test]
    fn test_validate_response_unknown_persona() {
        let mgr = PersonaManager::new();
        let v = mgr.validate_response("nonexistent", "some text");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].rule_name, "unknown_persona");
    }
}
