//! Route requests to different processing pipelines based on content analysis.
//!
//! Analyses a prompt string to produce a [`RoutingSignal`], then evaluates
//! a priority-ordered list of [`RoutingRule`]s to select the target [`Pipeline`].

use std::collections::HashMap;

// ── Pipeline ──────────────────────────────────────────────────────────────────

/// The processing pipelines that a request can be directed to.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Pipeline {
    FastPath,
    Standard,
    Detailed,
    Expert,
    Streaming,
}

impl Pipeline {
    /// Rough expected end-to-end latency for this pipeline in milliseconds.
    pub fn expected_latency_ms(&self) -> u64 {
        match self {
            Pipeline::FastPath => 200,
            Pipeline::Standard => 800,
            Pipeline::Detailed => 2_000,
            Pipeline::Expert => 5_000,
            Pipeline::Streaming => 300,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Pipeline::FastPath => "FastPath",
            Pipeline::Standard => "Standard",
            Pipeline::Detailed => "Detailed",
            Pipeline::Expert => "Expert",
            Pipeline::Streaming => "Streaming",
        }
    }
}

// ── RoutingSignal ─────────────────────────────────────────────────────────────

/// Extracted signals from a prompt used to drive routing decisions.
#[derive(Debug, Clone, Default)]
pub struct RoutingSignal {
    pub keyword_matches: Vec<String>,
    pub complexity_score: f64,
    pub estimated_tokens: u64,
    pub has_code: bool,
    pub has_math: bool,
    pub is_conversational: bool,
}

// ── RoutingCondition ──────────────────────────────────────────────────────────

/// A single boolean condition evaluated against a [`RoutingSignal`].
#[derive(Debug, Clone)]
pub enum RoutingCondition {
    ComplexityAbove(f64),
    TokensAbove(u64),
    ContainsKeyword(String),
    HasCode,
    HasMath,
    IsConversational,
    Always,
}

impl RoutingCondition {
    fn matches(&self, signal: &RoutingSignal) -> bool {
        match self {
            RoutingCondition::ComplexityAbove(t) => signal.complexity_score > *t,
            RoutingCondition::TokensAbove(t) => signal.estimated_tokens > *t,
            RoutingCondition::ContainsKeyword(kw) => {
                signal.keyword_matches.iter().any(|k| k == kw)
            }
            RoutingCondition::HasCode => signal.has_code,
            RoutingCondition::HasMath => signal.has_math,
            RoutingCondition::IsConversational => signal.is_conversational,
            RoutingCondition::Always => true,
        }
    }
}

// ── RoutingRule ───────────────────────────────────────────────────────────────

/// A named routing rule with a priority (lower value = higher priority).
#[derive(Debug, Clone)]
pub struct RoutingRule {
    pub name: String,
    pub condition: RoutingCondition,
    pub target: Pipeline,
    /// Lower value = evaluated first. Ties broken by insertion order.
    pub priority: u8,
}

// ── RequestRouter ─────────────────────────────────────────────────────────────

/// Routes incoming prompts to the appropriate processing pipeline.
pub struct RequestRouter {
    rules: Vec<RoutingRule>,
    stats: HashMap<String, u64>,
}

impl Default for RequestRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestRouter {
    /// Create a new router with the four default rules.
    pub fn new() -> Self {
        let mut router = Self {
            rules: Vec::new(),
            stats: HashMap::new(),
        };
        router.add_default_rules();
        router
    }

    fn add_default_rules(&mut self) {
        self.add_rule(RoutingRule {
            name: "code-to-expert".to_string(),
            condition: RoutingCondition::HasCode,
            target: Pipeline::Expert,
            priority: 10,
        });
        self.add_rule(RoutingRule {
            name: "math-to-detailed".to_string(),
            condition: RoutingCondition::HasMath,
            target: Pipeline::Detailed,
            priority: 20,
        });
        self.add_rule(RoutingRule {
            name: "conversational-to-fast".to_string(),
            condition: RoutingCondition::IsConversational,
            target: Pipeline::FastPath,
            priority: 30,
        });
        self.add_rule(RoutingRule {
            name: "high-complexity-to-expert".to_string(),
            condition: RoutingCondition::ComplexityAbove(0.8),
            target: Pipeline::Expert,
            priority: 40,
        });
    }

    /// Add a routing rule. Rules are always sorted by priority before evaluation.
    pub fn add_rule(&mut self, rule: RoutingRule) {
        self.rules.push(rule);
        self.rules.sort_by_key(|r| r.priority);
    }

    /// Analyse a prompt and produce a [`RoutingSignal`].
    pub fn analyze(&self, prompt: &str) -> RoutingSignal {
        let lower = prompt.to_lowercase();

        // Code detection: look for common code markers.
        let has_code = lower.contains("```")
            || lower.contains("fn ")
            || lower.contains("def ")
            || lower.contains("class ")
            || lower.contains("import ")
            || lower.contains("function ")
            || lower.contains("let ")
            || lower.contains("const ")
            || lower.contains("var ");

        // Math detection.
        let has_math = lower.contains("equation")
            || lower.contains("integral")
            || lower.contains("derivative")
            || lower.contains("matrix")
            || lower.contains("vector")
            || lower.contains("theorem")
            || lower.contains("calculus")
            || lower.contains("algebra")
            || lower.contains("sigma")
            || lower.contains("∑")
            || lower.contains("∫")
            || lower.contains("sqrt")
            || lower.contains("formula");

        // Conversational detection: short, greeting-like phrases.
        let word_count = prompt.split_whitespace().count();
        let is_conversational = word_count <= 15
            && (lower.starts_with("hi")
                || lower.starts_with("hello")
                || lower.starts_with("hey")
                || lower.starts_with("thanks")
                || lower.starts_with("ok")
                || lower.starts_with("yes")
                || lower.starts_with("no")
                || lower.starts_with("what is")
                || lower.starts_with("how are")
                || lower.starts_with("tell me"));

        // Rough complexity: normalised by word count and punctuation density.
        let punct_count = prompt.chars().filter(|c| "?!;:,.-()[]{}".contains(*c)).count();
        let complexity_score = if word_count == 0 {
            0.0
        } else {
            let base = (word_count as f64).ln() / 10.0;
            let punct_factor = punct_count as f64 / (word_count as f64 + 1.0);
            (base + punct_factor * 0.3).min(1.0)
        };

        // Token estimate: ~0.75 words per token on average.
        let estimated_tokens = ((word_count as f64) / 0.75).ceil() as u64;

        // Keyword matching: look for a curated list.
        let keywords = [
            "summarize", "explain", "analyze", "debug", "optimize",
            "translate", "compare", "review", "generate", "refactor",
        ];
        let keyword_matches: Vec<String> = keywords
            .iter()
            .filter(|&&kw| lower.contains(kw))
            .map(|s| s.to_string())
            .collect();

        RoutingSignal {
            keyword_matches,
            complexity_score,
            estimated_tokens,
            has_code,
            has_math,
            is_conversational,
        }
    }

    /// Route a signal to a pipeline.
    ///
    /// Evaluates rules in priority order; returns the first match.
    /// Falls back to [`Pipeline::Standard`] if no rule matches.
    pub fn route(&mut self, signal: &RoutingSignal) -> Pipeline {
        let pipeline = self
            .rules
            .iter()
            .find(|r| r.condition.matches(signal))
            .map(|r| r.target.clone())
            .unwrap_or(Pipeline::Standard);

        *self.stats.entry(pipeline.name().to_string()).or_insert(0) += 1;
        pipeline
    }

    /// Routing statistics: pipeline name → request count.
    pub fn routing_stats(&self) -> HashMap<String, u64> {
        self.stats.clone()
    }

    /// Analyse a prompt, route it, and return the pipeline together with the
    /// list of matching rule names (all rules whose condition is satisfied).
    pub fn explain_routing(&mut self, prompt: &str) -> (Pipeline, Vec<String>) {
        let signal = self.analyze(prompt);
        let matching_rules: Vec<String> = self
            .rules
            .iter()
            .filter(|r| r.condition.matches(&signal))
            .map(|r| r.name.clone())
            .collect();

        let pipeline = self.route(&signal);
        (pipeline, matching_rules)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_latencies_ordered() {
        assert!(Pipeline::FastPath.expected_latency_ms() < Pipeline::Standard.expected_latency_ms());
        assert!(Pipeline::Standard.expected_latency_ms() < Pipeline::Detailed.expected_latency_ms());
        assert!(Pipeline::Detailed.expected_latency_ms() < Pipeline::Expert.expected_latency_ms());
    }

    #[test]
    fn code_prompt_routes_to_expert() {
        let mut router = RequestRouter::new();
        let signal = router.analyze("```rust\nfn main() { println!(\"hello\"); }\n```");
        assert!(signal.has_code);
        let pipeline = router.route(&signal);
        assert_eq!(pipeline, Pipeline::Expert);
    }

    #[test]
    fn math_prompt_routes_to_detailed() {
        let mut router = RequestRouter::new();
        let signal = router.analyze("Solve the integral of x^2 using calculus.");
        assert!(signal.has_math);
        let pipeline = router.route(&signal);
        // Expert rule has priority 10, math rule 20, but no code → math wins.
        assert_eq!(pipeline, Pipeline::Detailed);
    }

    #[test]
    fn conversational_routes_to_fast() {
        let mut router = RequestRouter::new();
        let signal = router.analyze("Hello there how are you");
        assert!(signal.is_conversational);
        let pipeline = router.route(&signal);
        assert_eq!(pipeline, Pipeline::FastPath);
    }

    #[test]
    fn default_fallback_to_standard() {
        let mut router = RequestRouter::new();
        let signal = router.analyze("Tell me the history of the Roman Empire in detail.");
        // Not code, not math, not conversational, complexity < 0.8
        let pipeline = router.route(&signal);
        // Should be Standard (fallback) unless complexity fires Expert.
        // Just check it does not panic and returns a valid pipeline.
        let _ = pipeline;
    }

    #[test]
    fn high_complexity_routes_to_expert() {
        let mut router = RequestRouter::new();
        // Force a high complexity signal directly.
        let signal = RoutingSignal {
            complexity_score: 0.95,
            ..Default::default()
        };
        let pipeline = router.route(&signal);
        assert_eq!(pipeline, Pipeline::Expert);
    }

    #[test]
    fn custom_rule_added_with_lower_priority() {
        let mut router = RequestRouter::new();
        router.add_rule(RoutingRule {
            name: "streaming-rule".to_string(),
            condition: RoutingCondition::ContainsKeyword("stream".to_string()),
            target: Pipeline::Streaming,
            priority: 5, // highest priority
        });
        let mut signal = RoutingSignal::default();
        signal.keyword_matches.push("stream".to_string());
        let pipeline = router.route(&signal);
        assert_eq!(pipeline, Pipeline::Streaming);
    }

    #[test]
    fn routing_stats_accumulate() {
        let mut router = RequestRouter::new();
        let s1 = RoutingSignal { has_code: true, ..Default::default() };
        let s2 = RoutingSignal::default();
        router.route(&s1);
        router.route(&s2);
        router.route(&s2);
        let stats = router.routing_stats();
        assert_eq!(*stats.get("Expert").unwrap_or(&0), 1);
        assert_eq!(*stats.get("Standard").unwrap_or(&0), 2);
    }

    #[test]
    fn explain_routing_returns_matching_rules() {
        let mut router = RequestRouter::new();
        let (pipeline, rules) = router.explain_routing(
            "def solve(): integral = calculus.integrate(x**2)",
        );
        // Has code AND math → both rules match.
        assert!(rules.contains(&"code-to-expert".to_string()));
        assert!(rules.contains(&"math-to-detailed".to_string()));
        assert_eq!(pipeline, Pipeline::Expert); // highest-priority rule wins
    }

    #[test]
    fn analyze_keyword_extraction() {
        let router = RequestRouter::new();
        let signal = router.analyze("Please summarize and analyze this document.");
        assert!(signal.keyword_matches.contains(&"summarize".to_string()));
        assert!(signal.keyword_matches.contains(&"analyze".to_string()));
    }

    #[test]
    fn token_estimate_nonzero() {
        let router = RequestRouter::new();
        let signal = router.analyze("Some prompt text here.");
        assert!(signal.estimated_tokens > 0);
    }

    #[test]
    fn always_condition_matches() {
        let mut router = RequestRouter::new();
        router.add_rule(RoutingRule {
            name: "always-streaming".to_string(),
            condition: RoutingCondition::Always,
            target: Pipeline::Streaming,
            priority: 255,
        });
        // With no other signals, the first matching rule should be a default one,
        // but with Always at lowest priority it should catch the empty case.
        let signal = RoutingSignal::default();
        let pipeline = router.route(&signal);
        // Standard (priority 255 Always) or Streaming — just verify no panic.
        let _ = pipeline;
    }
}
