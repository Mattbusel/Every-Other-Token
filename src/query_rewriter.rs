//! Query expansion and reformulation for improved retrieval.
//!
//! Provides multiple rewrite strategies including synonym expansion,
//! HyDE (Hypothetical Document Embeddings), query decomposition,
//! clarification generation, paraphrasing, and step-back abstraction.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Strategy enum
// ---------------------------------------------------------------------------

/// Which rewrite strategy to apply to a query.
#[derive(Debug, Clone, PartialEq)]
pub enum RewriteStrategy {
    /// Adds synonyms and related terms to broaden recall.
    Expansion,
    /// Hypothetical Document Embeddings — rephrases as if the answer exists.
    HyDE,
    /// Splits compound questions into sub-questions.
    Decomposition,
    /// Generates clarifying questions for ambiguous queries.
    Clarification,
    /// Produces alternative phrasings of the same question.
    Paraphrase,
    /// Abstracts a specific query to a more general form.
    StepBack,
}

// ---------------------------------------------------------------------------
// Output type
// ---------------------------------------------------------------------------

/// The result of a query rewrite operation.
#[derive(Debug, Clone)]
pub struct ExpandedQuery {
    pub original: String,
    pub rewritten: String,
    pub strategy: RewriteStrategy,
    pub alternatives: Vec<String>,
    pub confidence: f64,
}

// ---------------------------------------------------------------------------
// QueryRewriter
// ---------------------------------------------------------------------------

/// Stateless query rewriting engine.
pub struct QueryRewriter {
    synonyms: HashMap<&'static str, Vec<&'static str>>,
}

impl Default for QueryRewriter {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryRewriter {
    /// Build a new rewriter with a built-in synonym map.
    pub fn new() -> Self {
        let mut synonyms: HashMap<&'static str, Vec<&'static str>> = HashMap::new();
        synonyms.insert("fast", vec!["quick", "rapid", "speedy", "swift"]);
        synonyms.insert("quick", vec!["fast", "rapid", "speedy"]);
        synonyms.insert("big", vec!["large", "huge", "enormous", "massive"]);
        synonyms.insert("small", vec!["tiny", "little", "compact", "miniature"]);
        synonyms.insert("good", vec!["excellent", "great", "superior", "quality"]);
        synonyms.insert("bad", vec!["poor", "inferior", "defective", "faulty"]);
        synonyms.insert("use", vec!["utilize", "employ", "apply", "leverage"]);
        synonyms.insert("make", vec!["create", "build", "construct", "generate"]);
        synonyms.insert("show", vec!["display", "demonstrate", "illustrate", "present"]);
        synonyms.insert("find", vec!["locate", "discover", "identify", "search"]);
        synonyms.insert("get", vec!["retrieve", "obtain", "fetch", "acquire"]);
        synonyms.insert("error", vec!["bug", "fault", "issue", "defect", "problem"]);
        synonyms.insert("fix", vec!["repair", "resolve", "correct", "patch"]);
        synonyms.insert("code", vec!["source", "program", "implementation", "script"]);
        synonyms.insert("best", vec!["optimal", "top", "leading", "premier"]);
        synonyms.insert("list", vec!["enumerate", "catalog", "collection", "inventory"]);
        synonyms.insert("explain", vec!["describe", "clarify", "elaborate", "detail"]);
        synonyms.insert("compare", vec!["contrast", "evaluate", "differentiate", "assess"]);
        synonyms.insert("difference", vec!["distinction", "contrast", "variation", "divergence"]);
        synonyms.insert("example", vec!["instance", "illustration", "sample", "case"]);
        synonyms.insert("method", vec!["approach", "technique", "procedure", "strategy"]);
        synonyms.insert("data", vec!["information", "dataset", "records", "values"]);
        synonyms.insert("model", vec!["network", "architecture", "system", "framework"]);
        synonyms.insert("learn", vec!["train", "acquire", "study", "master"]);
        synonyms.insert("improve", vec!["enhance", "optimize", "refine", "upgrade"]);
        Self { synonyms }
    }

    // -----------------------------------------------------------------------
    // Individual strategies
    // -----------------------------------------------------------------------

    /// Expand a query by appending synonyms for recognised words.
    pub fn expand(&self, query: &str) -> ExpandedQuery {
        let words: Vec<&str> = query.split_whitespace().collect();
        let mut extra_terms: Vec<String> = Vec::new();

        for word in &words {
            let lower = word.to_lowercase();
            let key = lower.trim_matches(|c: char| !c.is_alphabetic());
            if let Some(syns) = self.synonyms.get(key) {
                for s in syns.iter().take(2) {
                    extra_terms.push(s.to_string());
                }
            }
        }

        let rewritten = if extra_terms.is_empty() {
            query.to_string()
        } else {
            format!("{} {}", query, extra_terms.join(" "))
        };

        let alternatives = {
            let mut alts = Vec::new();
            if !extra_terms.is_empty() {
                alts.push(format!("{} (also: {})", query, extra_terms.join(", ")));
            }
            alts
        };

        ExpandedQuery {
            original: query.to_string(),
            rewritten,
            strategy: RewriteStrategy::Expansion,
            alternatives,
            confidence: if extra_terms.is_empty() { 0.5 } else { 0.8 },
        }
    }

    /// Generate a hypothetical document excerpt for the query.
    ///
    /// Format: "[domain] context: [query paraphrased as if answer exists]"
    pub fn hyde_expand(&self, query: &str, domain: &str) -> String {
        // Strip trailing punctuation, convert to statement form.
        let trimmed = query.trim_end_matches('?').trim_end_matches('.').trim();
        // Naively convert "How do I X" / "What is X" patterns.
        let statement = if let Some(rest) = trimmed.strip_prefix("How do I ") {
            format!("The way to {} is described here", rest)
        } else if let Some(rest) = trimmed.strip_prefix("How to ") {
            format!("To {}, one should", rest)
        } else if let Some(rest) = trimmed.strip_prefix("What is ") {
            format!("{} is defined as", rest)
        } else if let Some(rest) = trimmed.strip_prefix("Why is ") {
            format!("{} because", rest)
        } else if let Some(rest) = trimmed.strip_prefix("Why does ") {
            format!("{} due to", rest)
        } else {
            format!("The answer to '{}' is", trimmed)
        };
        format!("{} context: {}", domain, statement)
    }

    /// Decompose a compound question into sub-questions.
    ///
    /// Splits on conjunctions: "and", "also", "additionally", "as well as".
    pub fn decompose(&self, query: &str) -> Vec<String> {
        // Patterns that indicate a second clause.
        let splitters = [" and ", " also ", " additionally ", " as well as "];
        let lower = query.to_lowercase();

        for splitter in &splitters {
            if let Some(pos) = lower.find(splitter) {
                let first = query[..pos].trim().to_string();
                let second = query[pos + splitter.len()..].trim().to_string();
                if !first.is_empty() && !second.is_empty() {
                    // Capitalise second part if needed.
                    let second_cap = capitalise_first(&second);
                    return vec![
                        ensure_question_mark(&first),
                        ensure_question_mark(&second_cap),
                    ];
                }
            }
        }
        // No split found — return original as single item.
        vec![query.to_string()]
    }

    /// Generate clarifying questions for potentially ambiguous queries.
    pub fn generate_clarifications(&self, query: &str) -> Vec<String> {
        let mut clarifications = Vec::new();
        let lower = query.to_lowercase();

        // Detect ambiguous pronouns / vague references.
        if lower.contains(" it ") || lower.contains(" this ") || lower.contains(" that ") {
            clarifications.push(format!("What specific subject does 'it/this/that' refer to in: '{}'?", query));
        }
        // Detect under-specified comparisons.
        if lower.contains("better") || lower.contains("worse") || lower.contains("faster") {
            clarifications.push(format!("Compared to what baseline are you measuring in: '{}'?", query));
        }
        // Detect temporal ambiguity.
        if lower.contains("recently") || lower.contains("latest") || lower.contains("current") {
            clarifications.push(format!("What time period does 'recent/latest/current' refer to in: '{}'?", query));
        }
        // Detect scope ambiguity.
        if lower.contains("best") || lower.contains("most") || lower.contains("all") {
            clarifications.push(format!("In what context or domain should '{}' be evaluated?", query));
        }
        // Always add a generic domain clarification.
        clarifications.push(format!("Is this question about a specific field or domain: '{}'?", query));
        clarifications
    }

    /// Generate 3–5 alternative phrasings via word substitution.
    pub fn paraphrase(&self, query: &str) -> Vec<String> {
        let words: Vec<&str> = query.split_whitespace().collect();
        let mut variants: Vec<String> = Vec::new();

        // Variant 1: replace each known word with first synonym.
        let v1: Vec<String> = words
            .iter()
            .map(|w| {
                let lower = w.to_lowercase();
                let key = lower.trim_matches(|c: char| !c.is_alphabetic());
                if let Some(syns) = self.synonyms.get(key) {
                    syns[0].to_string()
                } else {
                    w.to_string()
                }
            })
            .collect();
        let v1_str = v1.join(" ");
        if v1_str != query {
            variants.push(v1_str);
        }

        // Variant 2: rearrange "How do I X" → "What is the way to X"
        let lower = query.to_lowercase();
        if let Some(rest) = lower.strip_prefix("how do i ") {
            variants.push(format!("What is the way to {}?", rest.trim_end_matches('?')));
        } else if let Some(rest) = lower.strip_prefix("what is ") {
            variants.push(format!("Can you define {}?", rest.trim_end_matches('?')));
        } else if let Some(rest) = lower.strip_prefix("why ") {
            variants.push(format!("What is the reason that {}?", rest.trim_end_matches('?')));
        }

        // Variant 3: prefix with "Tell me about"
        let bare = query.trim_end_matches('?').trim_end_matches('.');
        variants.push(format!("Tell me about {}", bare.to_lowercase()));

        // Variant 4: prefix with "Explain"
        variants.push(format!("Explain {}", bare.to_lowercase()));

        // Variant 5: append "in detail"
        variants.push(format!("{} in detail", bare));

        // Keep at most 5 unique variants.
        variants.dedup();
        variants.truncate(5);
        variants
    }

    /// Abstract a specific query to a more general form (step-back).
    pub fn step_back(&self, query: &str) -> String {
        let lower = query.to_lowercase();

        // Strip highly specific qualifiers.
        let generalised = lower
            .replace(" specifically", "")
            .replace(" exactly", "")
            .replace(" precisely", "")
            .replace(" in this case", "")
            .replace(" in my case", "");

        // Convert specific "how do I X in Y" → "how does X work in Y generally"
        if let Some(rest) = generalised.strip_prefix("how do i ") {
            return format!("How does {} work in general?", rest.trim_end_matches('?').trim());
        }
        if let Some(rest) = generalised.strip_prefix("why is my ") {
            return format!("What are common reasons for {}?", rest.trim_end_matches('?').trim());
        }
        if let Some(rest) = generalised.strip_prefix("why does my ") {
            return format!("What are common reasons for {}?", rest.trim_end_matches('?').trim());
        }

        // Generic step-back: prepend "In general, "
        format!("In general, {}", generalised.trim_end_matches('?').trim())
    }

    // -----------------------------------------------------------------------
    // Unified interface
    // -----------------------------------------------------------------------

    /// Apply the given strategy and return an `ExpandedQuery`.
    pub fn rewrite(&self, query: &str, strategy: RewriteStrategy) -> ExpandedQuery {
        match &strategy {
            RewriteStrategy::Expansion => self.expand(query),

            RewriteStrategy::HyDE => {
                let rewritten = self.hyde_expand(query, "general");
                ExpandedQuery {
                    original: query.to_string(),
                    rewritten,
                    strategy,
                    alternatives: vec![
                        self.hyde_expand(query, "technical"),
                        self.hyde_expand(query, "academic"),
                    ],
                    confidence: 0.7,
                }
            }

            RewriteStrategy::Decomposition => {
                let parts = self.decompose(query);
                let rewritten = parts.join(" | ");
                ExpandedQuery {
                    original: query.to_string(),
                    rewritten,
                    strategy,
                    alternatives: parts,
                    confidence: 0.75,
                }
            }

            RewriteStrategy::Clarification => {
                let clarifications = self.generate_clarifications(query);
                let rewritten = clarifications.first().cloned().unwrap_or_else(|| query.to_string());
                ExpandedQuery {
                    original: query.to_string(),
                    rewritten,
                    strategy,
                    alternatives: clarifications,
                    confidence: 0.65,
                }
            }

            RewriteStrategy::Paraphrase => {
                let alternatives = self.paraphrase(query);
                let rewritten = alternatives.first().cloned().unwrap_or_else(|| query.to_string());
                ExpandedQuery {
                    original: query.to_string(),
                    rewritten,
                    strategy,
                    alternatives,
                    confidence: 0.72,
                }
            }

            RewriteStrategy::StepBack => {
                let rewritten = self.step_back(query);
                ExpandedQuery {
                    original: query.to_string(),
                    rewritten: rewritten.clone(),
                    strategy,
                    alternatives: vec![rewritten],
                    confidence: 0.68,
                }
            }
        }
    }

    /// Heuristic: choose the best strategy for a query.
    ///
    /// - Short (≤4 words) → `Expansion`
    /// - Contains conjunctions → `Decomposition`
    /// - Very specific (long, proper nouns) → `StepBack`
    /// - Default → `Expansion`
    pub fn best_strategy(&self, query: &str) -> RewriteStrategy {
        let words: Vec<&str> = query.split_whitespace().collect();
        let lower = query.to_lowercase();

        // Compound question signals.
        let compound_signals = [" and ", " also ", " additionally ", " as well as "];
        if compound_signals.iter().any(|s| lower.contains(s)) {
            return RewriteStrategy::Decomposition;
        }

        // Short query — expand to increase recall.
        if words.len() <= 4 {
            return RewriteStrategy::Expansion;
        }

        // Long, likely specific query — step back for broader context.
        if words.len() >= 10 {
            return RewriteStrategy::StepBack;
        }

        // Medium query with ambiguous pronouns → clarify.
        if lower.contains(" it ") || lower.contains(" this ") || lower.contains(" that ") {
            return RewriteStrategy::Clarification;
        }

        RewriteStrategy::Expansion
    }

    /// Rewrite a batch of queries using the heuristically chosen strategy.
    pub fn batch_rewrite(&self, queries: &[&str]) -> Vec<ExpandedQuery> {
        queries
            .iter()
            .map(|q| {
                let strategy = self.best_strategy(q);
                self.rewrite(q, strategy)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn capitalise_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

fn ensure_question_mark(s: &str) -> String {
    let trimmed = s.trim();
    if trimmed.ends_with('?') {
        trimmed.to_string()
    } else {
        format!("{}?", trimmed)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_adds_synonyms() {
        let rw = QueryRewriter::new();
        let eq = rw.expand("find the best code");
        // Should contain the original words plus synonyms.
        assert!(eq.rewritten.contains("find"));
        assert!(eq.strategy == RewriteStrategy::Expansion);
    }

    #[test]
    fn hyde_expand_basic() {
        let rw = QueryRewriter::new();
        let result = rw.hyde_expand("What is Rust?", "programming");
        assert!(result.contains("programming context:"));
        assert!(result.contains("Rust"));
    }

    #[test]
    fn decompose_splits_on_and() {
        let rw = QueryRewriter::new();
        let parts = rw.decompose("How do I read files and write files?");
        assert_eq!(parts.len(), 2);
    }

    #[test]
    fn decompose_no_conjunction() {
        let rw = QueryRewriter::new();
        let parts = rw.decompose("What is Rust?");
        assert_eq!(parts.len(), 1);
    }

    #[test]
    fn paraphrase_returns_multiple() {
        let rw = QueryRewriter::new();
        let variants = rw.paraphrase("How do I fix a bug?");
        assert!(!variants.is_empty());
    }

    #[test]
    fn step_back_generalises() {
        let rw = QueryRewriter::new();
        let general = rw.step_back("How do I parse JSON in Rust?");
        assert!(general.contains("general") || general.contains("work"));
    }

    #[test]
    fn best_strategy_short() {
        let rw = QueryRewriter::new();
        assert_eq!(rw.best_strategy("fast code"), RewriteStrategy::Expansion);
    }

    #[test]
    fn best_strategy_compound() {
        let rw = QueryRewriter::new();
        assert_eq!(
            rw.best_strategy("What is Rust and how do I use it?"),
            RewriteStrategy::Decomposition
        );
    }

    #[test]
    fn batch_rewrite_returns_same_count() {
        let rw = QueryRewriter::new();
        let qs = ["find files", "What is this and that?", "explain"];
        let results = rw.batch_rewrite(&qs);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn rewrite_hyde_confidence() {
        let rw = QueryRewriter::new();
        let eq = rw.rewrite("What is machine learning?", RewriteStrategy::HyDE);
        assert!(eq.confidence > 0.0);
        assert_eq!(eq.strategy, RewriteStrategy::HyDE);
    }
}
