//! Systematic prompt mutation for exploring the prompt space.
//!
//! Provides a [`PromptMutator`] that applies a variety of textual mutations to
//! prompts and can generate populations of mutations via different strategies.

use uuid::Uuid;

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

/// The available output formats a prompt can request.
#[derive(Debug, Clone, PartialEq)]
pub enum OutputFormat {
    Bullet,
    Numbered,
    Paragraph,
    Json,
    Table,
    Code,
}

impl OutputFormat {
    fn label(&self) -> &'static str {
        match self {
            OutputFormat::Bullet => "bullet-point list",
            OutputFormat::Numbered => "numbered list",
            OutputFormat::Paragraph => "prose paragraphs",
            OutputFormat::Json => "JSON",
            OutputFormat::Table => "markdown table",
            OutputFormat::Code => "code block",
        }
    }
}

/// Rhetorical tone descriptors.
#[derive(Debug, Clone, PartialEq)]
pub enum Tone {
    Formal,
    Informal,
    Technical,
    Simple,
    Assertive,
    Tentative,
}

/// Every supported mutation operation.
#[derive(Debug, Clone)]
pub enum MutationType {
    Paraphrase,
    AddContext(String),
    ChangeFormat(OutputFormat),
    AddConstraint(String),
    RemoveSentence(usize),
    Synonym(String, String),
    ToneShift(Tone, Tone),
}

// ---------------------------------------------------------------------------
// PromptMutation
// ---------------------------------------------------------------------------

/// The result of applying a single mutation.
#[derive(Debug, Clone)]
pub struct PromptMutation {
    pub original: String,
    pub mutated: String,
    pub mutation_type: MutationType,
    pub mutation_id: String,
}

// ---------------------------------------------------------------------------
// Strategy
// ---------------------------------------------------------------------------

/// How mutations are generated when producing a population.
#[derive(Debug, Clone)]
pub enum MutationStrategy {
    Random { seed: u64 },
    Systematic,
    Targeted { focus_keyword: String },
}

// ---------------------------------------------------------------------------
// Synonym table
// ---------------------------------------------------------------------------

const SYNONYMS: &[(&str, &str)] = &[
    ("good", "excellent"),
    ("bad", "poor"),
    ("big", "large"),
    ("small", "tiny"),
    ("fast", "rapid"),
    ("slow", "sluggish"),
    ("happy", "joyful"),
    ("sad", "sorrowful"),
    ("smart", "intelligent"),
    ("dumb", "foolish"),
    ("help", "assist"),
    ("show", "display"),
    ("tell", "inform"),
    ("make", "create"),
    ("use", "utilize"),
    ("get", "obtain"),
    ("find", "locate"),
    ("start", "commence"),
    ("end", "conclude"),
    ("change", "modify"),
];

// ---------------------------------------------------------------------------
// Tone marker tables
// ---------------------------------------------------------------------------

const FORMAL_TO_INFORMAL: &[(&str, &str)] = &[
    ("please provide", "can you give me"),
    ("kindly", ""),
    ("therefore", "so"),
    ("however", "but"),
    ("furthermore", "also"),
    ("I would like", "I want"),
    ("regarding", "about"),
    ("concerning", "about"),
    ("utilize", "use"),
    ("commence", "start"),
];

const INFORMAL_TO_FORMAL: &[(&str, &str)] = &[
    ("can you give me", "please provide"),
    ("so", "therefore"),
    ("but", "however"),
    ("also", "furthermore"),
    ("I want", "I would like"),
    ("about", "regarding"),
    ("use", "utilize"),
    ("start", "commence"),
    ("need", "require"),
    ("think", "consider"),
];

const TECHNICAL_TO_SIMPLE: &[(&str, &str)] = &[
    ("algorithm", "method"),
    ("implement", "build"),
    ("parameter", "setting"),
    ("iterate", "repeat"),
    ("execute", "run"),
    ("instantiate", "create"),
    ("initialize", "set up"),
    ("traverse", "go through"),
    ("concatenate", "join"),
    ("deprecated", "outdated"),
];

const SIMPLE_TO_TECHNICAL: &[(&str, &str)] = &[
    ("method", "algorithm"),
    ("build", "implement"),
    ("setting", "parameter"),
    ("repeat", "iterate"),
    ("run", "execute"),
    ("create", "instantiate"),
    ("set up", "initialize"),
    ("go through", "traverse"),
    ("join", "concatenate"),
    ("outdated", "deprecated"),
];

const ASSERTIVE_TO_TENTATIVE: &[(&str, &str)] = &[
    ("is", "may be"),
    ("will", "might"),
    ("always", "often"),
    ("never", "rarely"),
    ("must", "should"),
    ("clearly", "arguably"),
    ("obviously", "apparently"),
    ("definitely", "possibly"),
    ("certainly", "perhaps"),
    ("prove", "suggest"),
];

const TENTATIVE_TO_ASSERTIVE: &[(&str, &str)] = &[
    ("may be", "is"),
    ("might", "will"),
    ("often", "always"),
    ("rarely", "never"),
    ("should", "must"),
    ("arguably", "clearly"),
    ("apparently", "obviously"),
    ("possibly", "definitely"),
    ("perhaps", "certainly"),
    ("suggest", "prove"),
];

// ---------------------------------------------------------------------------
// PromptMutator
// ---------------------------------------------------------------------------

/// Applies textual mutations to prompts.
pub struct PromptMutator;

impl PromptMutator {
    pub fn new() -> Self {
        Self
    }

    /// Apply a single mutation to `prompt`.
    pub fn mutate(&self, prompt: &str, mutation: &MutationType) -> PromptMutation {
        let mutated = match mutation {
            MutationType::Paraphrase => self.apply_paraphrase(prompt, 42),
            MutationType::AddContext(ctx) => format!("Context: {}\n\n{}", ctx, prompt),
            MutationType::ChangeFormat(fmt) => self.apply_format_change(prompt, fmt),
            MutationType::AddConstraint(c) => format!("{}\n\nConstraint: {}", prompt, c),
            MutationType::RemoveSentence(idx) => self.remove_sentence(prompt, *idx),
            MutationType::Synonym(word, replacement) => {
                prompt.replacen(word.as_str(), replacement.as_str(), 1)
            }
            MutationType::ToneShift(from, to) => self.apply_tone_shift(prompt, from, to),
        };

        PromptMutation {
            original: prompt.to_string(),
            mutated,
            mutation_type: mutation.clone(),
            mutation_id: Uuid::new_v4().to_string(),
        }
    }

    /// Generate `n` mutations using the given strategy.
    pub fn generate_mutations(
        &self,
        prompt: &str,
        strategy: &MutationStrategy,
        n: usize,
    ) -> Vec<PromptMutation> {
        let all_mutations = self.strategy_mutations(prompt, strategy);
        all_mutations
            .into_iter()
            .take(n)
            .map(|m| self.mutate(prompt, &m))
            .collect()
    }

    /// Paraphrase by substituting synonyms according to a seeded PRNG.
    pub fn apply_paraphrase(&self, prompt: &str, seed: u64) -> String {
        // Simple deterministic synonym substitution using seed-based cycling.
        let mut result = prompt.to_string();
        let mut rng_state = seed;
        let mut count = 0usize;
        for (word, replacement) in SYNONYMS {
            if result.to_lowercase().contains(*word) {
                // Use seed to decide whether to apply each substitution
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                if rng_state % 2 == 0 || count < 3 {
                    // Case-preserving replacement of first occurrence
                    if let Some(pos) = result.to_lowercase().find(*word) {
                        let original_segment = &result[pos..pos + word.len()];
                        let rep = if original_segment.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                            let mut r = replacement.to_string();
                            if let Some(fc) = r.get_mut(0..1) {
                                fc.make_ascii_uppercase();
                            }
                            r
                        } else {
                            replacement.to_string()
                        };
                        result = format!("{}{}{}", &result[..pos], rep, &result[pos + word.len()..]);
                        count += 1;
                    }
                }
            }
        }
        result
    }

    /// Prepend a format directive to the prompt.
    pub fn apply_format_change(&self, prompt: &str, format: &OutputFormat) -> String {
        format!("Respond in {} format:\n\n{}", format.label(), prompt)
    }

    /// Replace tone-bearing words based on from/to tone.
    pub fn apply_tone_shift(&self, prompt: &str, from: &Tone, to: &Tone) -> String {
        let table: &[(&str, &str)] = match (from, to) {
            (Tone::Formal, Tone::Informal) => FORMAL_TO_INFORMAL,
            (Tone::Informal, Tone::Formal) => INFORMAL_TO_FORMAL,
            (Tone::Technical, Tone::Simple) => TECHNICAL_TO_SIMPLE,
            (Tone::Simple, Tone::Technical) => SIMPLE_TO_TECHNICAL,
            (Tone::Assertive, Tone::Tentative) => ASSERTIVE_TO_TENTATIVE,
            (Tone::Tentative, Tone::Assertive) => TENTATIVE_TO_ASSERTIVE,
            _ => return prompt.to_string(),
        };
        let mut result = prompt.to_string();
        for (src, dst) in table {
            if !src.is_empty() {
                result = result.replace(*src, *dst);
            }
        }
        result
    }

    /// Compute normalized Levenshtein distance between the mutated strings of two mutations.
    pub fn mutation_distance(a: &PromptMutation, b: &PromptMutation) -> f64 {
        let dist = levenshtein(&a.mutated, &b.mutated);
        let max_len = a.mutated.len().max(b.mutated.len());
        if max_len == 0 {
            0.0
        } else {
            dist as f64 / max_len as f64
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn remove_sentence(&self, prompt: &str, index: usize) -> String {
        let sentences: Vec<&str> = prompt.split(". ").collect();
        if sentences.len() <= 1 || index >= sentences.len() {
            return prompt.to_string();
        }
        sentences
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != index)
            .map(|(_, s)| *s)
            .collect::<Vec<_>>()
            .join(". ")
    }

    fn strategy_mutations(&self, prompt: &str, strategy: &MutationStrategy) -> Vec<MutationType> {
        match strategy {
            MutationStrategy::Systematic => vec![
                MutationType::Paraphrase,
                MutationType::AddContext("additional background information".to_string()),
                MutationType::ChangeFormat(OutputFormat::Bullet),
                MutationType::ChangeFormat(OutputFormat::Numbered),
                MutationType::ChangeFormat(OutputFormat::Json),
                MutationType::ChangeFormat(OutputFormat::Table),
                MutationType::ChangeFormat(OutputFormat::Code),
                MutationType::AddConstraint("Keep the answer under 100 words".to_string()),
                MutationType::AddConstraint("Use only simple vocabulary".to_string()),
                MutationType::RemoveSentence(0),
                MutationType::ToneShift(Tone::Formal, Tone::Informal),
                MutationType::ToneShift(Tone::Informal, Tone::Formal),
                MutationType::ToneShift(Tone::Technical, Tone::Simple),
                MutationType::ToneShift(Tone::Assertive, Tone::Tentative),
            ],
            MutationStrategy::Random { seed } => {
                // Generate a deterministic pseudo-random selection
                let mut rng = *seed;
                let base = self.strategy_mutations(prompt, &MutationStrategy::Systematic);
                let mut selected = Vec::new();
                for item in base {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    if rng % 2 == 0 {
                        selected.push(item);
                    }
                }
                if selected.is_empty() {
                    selected.push(MutationType::Paraphrase);
                }
                selected
            }
            MutationStrategy::Targeted { focus_keyword } => vec![
                MutationType::AddContext(format!("Focus on: {}", focus_keyword)),
                MutationType::AddConstraint(format!("Specifically address '{}'", focus_keyword)),
                MutationType::Paraphrase,
                MutationType::ToneShift(Tone::Technical, Tone::Simple),
            ],
        }
    }
}

impl Default for PromptMutator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Levenshtein distance (character-level)
// ---------------------------------------------------------------------------

fn levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();
    if m == 0 { return n; }
    if n == 0 { return m; }

    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 0..=m { dp[i][0] = i; }
    for j in 0..=n { dp[0][j] = j; }

    for i in 1..=m {
        for j in 1..=n {
            dp[i][j] = if a_chars[i - 1] == b_chars[j - 1] {
                dp[i - 1][j - 1]
            } else {
                1 + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1])
            };
        }
    }
    dp[m][n]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn mutator() -> PromptMutator {
        PromptMutator::new()
    }

    #[test]
    fn paraphrase_changes_text() {
        let m = mutator();
        let orig = "This is a good and fast way to help you.";
        let result = m.apply_paraphrase(orig, 42);
        // Should differ from original (at least one synonym substituted)
        assert_ne!(result, orig);
    }

    #[test]
    fn format_change_bullet() {
        let m = mutator();
        let orig = "Explain sorting.";
        let result = m.apply_format_change(orig, &OutputFormat::Bullet);
        assert!(result.contains("bullet-point list"));
        assert!(result.contains("Explain sorting."));
    }

    #[test]
    fn format_change_json() {
        let m = mutator();
        let result = m.apply_format_change("Describe yourself.", &OutputFormat::Json);
        assert!(result.contains("JSON"));
    }

    #[test]
    fn add_context_prepends() {
        let m = mutator();
        let pm = m.mutate("Tell me about Rust.", &MutationType::AddContext("Systems programming".to_string()));
        assert!(pm.mutated.starts_with("Context: Systems programming"));
    }

    #[test]
    fn add_constraint_appends() {
        let m = mutator();
        let pm = m.mutate("What is AI?", &MutationType::AddConstraint("Max 50 words".to_string()));
        assert!(pm.mutated.contains("Constraint: Max 50 words"));
    }

    #[test]
    fn remove_sentence_removes_index_0() {
        let m = mutator();
        let orig = "First sentence. Second sentence. Third sentence.";
        let result = m.remove_sentence(orig, 0);
        assert!(!result.starts_with("First sentence"));
        assert!(result.contains("Second sentence"));
    }

    #[test]
    fn synonym_mutation_replaces_word() {
        let m = mutator();
        let pm = m.mutate("This is a good idea.", &MutationType::Synonym("good".to_string(), "great".to_string()));
        assert!(pm.mutated.contains("great"));
    }

    #[test]
    fn tone_shift_formal_to_informal() {
        let m = mutator();
        let orig = "Please provide details regarding this matter.";
        let result = m.apply_tone_shift(orig, &Tone::Formal, &Tone::Informal);
        // "Please provide" -> "can you give me", "regarding" -> "about"
        assert!(result.contains("can you give me") || result.contains("about"));
    }

    #[test]
    fn tone_shift_technical_to_simple() {
        let m = mutator();
        let orig = "Please implement the algorithm and execute it.";
        let result = m.apply_tone_shift(orig, &Tone::Technical, &Tone::Simple);
        assert!(result.contains("build") || result.contains("method") || result.contains("run"));
    }

    #[test]
    fn tone_shift_assertive_to_tentative() {
        let m = mutator();
        let orig = "This will always work and must be used.";
        let result = m.apply_tone_shift(orig, &Tone::Assertive, &Tone::Tentative);
        assert!(result.contains("might") || result.contains("often") || result.contains("should"));
    }

    #[test]
    fn mutation_distance_identical_is_zero() {
        let m = mutator();
        let pm1 = m.mutate("hello world", &MutationType::Paraphrase);
        let pm2 = PromptMutation {
            original: pm1.original.clone(),
            mutated: pm1.mutated.clone(),
            mutation_type: MutationType::Paraphrase,
            mutation_id: "x".to_string(),
        };
        let dist = PromptMutator::mutation_distance(&pm1, &pm2);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn mutation_distance_different_texts() {
        let m = mutator();
        let pm1 = m.mutate("The quick brown fox", &MutationType::Paraphrase);
        let pm2 = m.mutate("Completely different text here", &MutationType::Paraphrase);
        let dist = PromptMutator::mutation_distance(&pm1, &pm2);
        assert!(dist > 0.0);
        assert!(dist <= 1.0);
    }

    #[test]
    fn generate_mutations_systematic_count() {
        let m = mutator();
        let mutations = m.generate_mutations("Explain AI.", &MutationStrategy::Systematic, 5);
        assert_eq!(mutations.len(), 5);
    }

    #[test]
    fn generate_mutations_random_nonzero() {
        let m = mutator();
        let mutations = m.generate_mutations(
            "What is machine learning?",
            &MutationStrategy::Random { seed: 12345 },
            3,
        );
        assert!(!mutations.is_empty());
    }

    #[test]
    fn generate_mutations_targeted() {
        let m = mutator();
        let mutations = m.generate_mutations(
            "Describe neural networks.",
            &MutationStrategy::Targeted { focus_keyword: "backpropagation".to_string() },
            4,
        );
        assert!(!mutations.is_empty());
        // At least one should mention the focus keyword in the mutated text
        let has_focus = mutations.iter().any(|mu| mu.mutated.contains("backpropagation"));
        assert!(has_focus);
    }

    #[test]
    fn mutation_id_is_unique() {
        let m = mutator();
        let pm1 = m.mutate("test", &MutationType::Paraphrase);
        let pm2 = m.mutate("test", &MutationType::Paraphrase);
        assert_ne!(pm1.mutation_id, pm2.mutation_id);
    }

    #[test]
    fn levenshtein_known_values() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("abc", "abc"), 0);
    }

    #[test]
    fn synonym_table_has_at_least_15_pairs() {
        assert!(SYNONYMS.len() >= 15);
    }
}
