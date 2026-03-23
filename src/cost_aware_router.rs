//! Cost-aware request router.
//!
//! Routes LLM requests to the most appropriate model tier based on text
//! complexity signals and a caller-supplied routing policy.

// ---------------------------------------------------------------------------
// ModelTier
// ---------------------------------------------------------------------------

/// The four model tiers ordered by increasing capability and cost.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ModelTier {
    Economy,
    Standard,
    Premium,
    UltraPremium,
}

impl ModelTier {
    /// Approximate cost per token in USD (input + output blended).
    pub fn cost_per_token(&self) -> f64 {
        match self {
            ModelTier::Economy => 0.000_000_5,   // ~$0.50 / 1M tokens
            ModelTier::Standard => 0.000_002_0,  // ~$2.00 / 1M tokens
            ModelTier::Premium => 0.000_010_0,   // ~$10.00 / 1M tokens
            ModelTier::UltraPremium => 0.000_060_0, // ~$60.00 / 1M tokens
        }
    }

    /// Normalised quality score in [0.0, 1.0].
    pub fn quality_score(&self) -> f64 {
        match self {
            ModelTier::Economy => 0.55,
            ModelTier::Standard => 0.72,
            ModelTier::Premium => 0.88,
            ModelTier::UltraPremium => 1.00,
        }
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            ModelTier::Economy => "Economy",
            ModelTier::Standard => "Standard",
            ModelTier::Premium => "Premium",
            ModelTier::UltraPremium => "UltraPremium",
        }
    }

    /// All tiers in ascending cost order.
    fn all() -> [ModelTier; 4] {
        [
            ModelTier::Economy,
            ModelTier::Standard,
            ModelTier::Premium,
            ModelTier::UltraPremium,
        ]
    }
}

// ---------------------------------------------------------------------------
// ComplexitySignal
// ---------------------------------------------------------------------------

/// Estimated complexity of an input text.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ComplexitySignal {
    Simple,
    Moderate,
    Complex,
    ExpertLevel,
}

impl ComplexitySignal {
    /// Estimate complexity from raw text using lightweight heuristics.
    ///
    /// Heuristics applied (each can contribute to an integer score):
    /// * Average sentence length > 20 words → +1
    /// * Rare-word proxy (words with length > 8) > 15 % → +1
    /// * More than two question marks → +1
    /// * At least one fenced code block (```) → +2
    pub fn estimate(text: &str) -> ComplexitySignal {
        let mut score: u32 = 0;

        // --- average sentence length ---
        let sentences: Vec<&str> = text
            .split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .collect();
        let sentence_count = sentences.len().max(1);
        let word_count = text.split_whitespace().count();
        let avg_sentence_len = word_count as f64 / sentence_count as f64;
        if avg_sentence_len > 20.0 {
            score += 1;
        }

        // --- rare-word proxy (length > 8) ---
        let words: Vec<&str> = text.split_whitespace().collect();
        let total_words = words.len().max(1);
        let long_words = words.iter().filter(|w| w.len() > 8).count();
        let long_word_ratio = long_words as f64 / total_words as f64;
        if long_word_ratio > 0.15 {
            score += 1;
        }

        // --- question count ---
        let question_count = text.chars().filter(|&c| c == '?').count();
        if question_count > 2 {
            score += 1;
        }

        // --- code block presence ---
        if text.contains("```") {
            score += 2;
        }

        match score {
            0 => ComplexitySignal::Simple,
            1 => ComplexitySignal::Moderate,
            2 | 3 => ComplexitySignal::Complex,
            _ => ComplexitySignal::ExpertLevel,
        }
    }
}

// ---------------------------------------------------------------------------
// RoutingPolicy
// ---------------------------------------------------------------------------

/// Governs how `CostAwareRouter` selects a tier for a given request.
#[derive(Debug, Clone)]
pub enum RoutingPolicy {
    /// Pick the highest-quality tier whose estimated cost fits within the budget.
    MaxQualityWithinBudget { budget_usd: f64, token_estimate: u64 },
    /// Pick the cheapest tier whose quality score meets the minimum threshold.
    MinCostAboveQuality { min_quality: f64, token_estimate: u64 },
    /// Always use the specified tier, ignoring complexity.
    FixedTier(ModelTier),
}

// ---------------------------------------------------------------------------
// CostAwareRouter
// ---------------------------------------------------------------------------

/// Routes requests to model tiers based on complexity and policy.
pub struct CostAwareRouter;

impl CostAwareRouter {
    pub fn new() -> Self {
        CostAwareRouter
    }

    /// Estimated cost in USD for the given tier and token count.
    pub fn estimated_cost(tier: &ModelTier, tokens: u64) -> f64 {
        tier.cost_per_token() * tokens as f64
    }

    /// Route `text` under the given policy, returning the recommended tier.
    pub fn route(&self, text: &str, policy: &RoutingPolicy) -> ModelTier {
        match policy {
            RoutingPolicy::FixedTier(tier) => tier.clone(),

            RoutingPolicy::MaxQualityWithinBudget {
                budget_usd,
                token_estimate,
            } => {
                // Walk tiers from most to least expensive; pick the highest
                // that fits the budget.
                let mut chosen = ModelTier::Economy;
                for tier in ModelTier::all().iter().rev() {
                    let cost = Self::estimated_cost(tier, *token_estimate);
                    if cost <= *budget_usd {
                        chosen = tier.clone();
                        break;
                    }
                }
                // Downgrade one step if the text is very simple.
                let signal = ComplexitySignal::estimate(text);
                if signal == ComplexitySignal::Simple && chosen != ModelTier::Economy {
                    match chosen {
                        ModelTier::UltraPremium => ModelTier::Premium,
                        ModelTier::Premium => ModelTier::Standard,
                        ModelTier::Standard => ModelTier::Economy,
                        ModelTier::Economy => ModelTier::Economy,
                    }
                } else {
                    chosen
                }
            }

            RoutingPolicy::MinCostAboveQuality {
                min_quality,
                token_estimate: _,
            } => {
                // Also factor in complexity: if ExpertLevel, add one extra step.
                let signal = ComplexitySignal::estimate(text);
                let bump = signal == ComplexitySignal::ExpertLevel;
                let mut chosen = ModelTier::UltraPremium; // fallback
                for tier in ModelTier::all().iter() {
                    if tier.quality_score() >= *min_quality {
                        chosen = tier.clone();
                        break;
                    }
                }
                if bump {
                    // Upgrade one tier if needed.
                    chosen = match chosen {
                        ModelTier::Economy => ModelTier::Standard,
                        ModelTier::Standard => ModelTier::Premium,
                        ModelTier::Premium => ModelTier::UltraPremium,
                        ModelTier::UltraPremium => ModelTier::UltraPremium,
                    };
                }
                chosen
            }
        }
    }

    /// Returns a human-readable explanation of the routing decision.
    pub fn routing_explanation(&self, text: &str, policy: &RoutingPolicy) -> String {
        let signal = ComplexitySignal::estimate(text);
        let tier = self.route(text, policy);

        let signal_label = match &signal {
            ComplexitySignal::Simple => "Simple",
            ComplexitySignal::Moderate => "Moderate",
            ComplexitySignal::Complex => "Complex",
            ComplexitySignal::ExpertLevel => "ExpertLevel",
        };

        let policy_label = match policy {
            RoutingPolicy::MaxQualityWithinBudget { budget_usd, token_estimate } => {
                format!(
                    "MaxQualityWithinBudget(budget=${:.4}, ~{} tokens)",
                    budget_usd, token_estimate
                )
            }
            RoutingPolicy::MinCostAboveQuality { min_quality, token_estimate } => {
                format!(
                    "MinCostAboveQuality(min_quality={:.2}, ~{} tokens)",
                    min_quality, token_estimate
                )
            }
            RoutingPolicy::FixedTier(t) => format!("FixedTier({})", t.label()),
        };

        format!(
            "Routing decision: {} tier\n\
             Policy: {}\n\
             Complexity signal: {} (words={}, code_blocks={})\n\
             Tier quality: {:.2}, cost/token: ${:.7}",
            tier.label(),
            policy_label,
            signal_label,
            text.split_whitespace().count(),
            text.matches("```").count() / 2,
            tier.quality_score(),
            tier.cost_per_token(),
        )
    }
}

impl Default for CostAwareRouter {
    fn default() -> Self {
        CostAwareRouter::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_tier_ordering() {
        assert!(ModelTier::Economy < ModelTier::Standard);
        assert!(ModelTier::Standard < ModelTier::Premium);
        assert!(ModelTier::Premium < ModelTier::UltraPremium);
    }

    #[test]
    fn cost_per_token_increases_with_tier() {
        assert!(ModelTier::Economy.cost_per_token() < ModelTier::Standard.cost_per_token());
        assert!(ModelTier::Standard.cost_per_token() < ModelTier::Premium.cost_per_token());
        assert!(ModelTier::Premium.cost_per_token() < ModelTier::UltraPremium.cost_per_token());
    }

    #[test]
    fn quality_score_increases_with_tier() {
        assert!(ModelTier::Economy.quality_score() < ModelTier::Standard.quality_score());
        assert!(ModelTier::UltraPremium.quality_score() == 1.0);
    }

    #[test]
    fn complexity_simple_short_text() {
        let signal = ComplexitySignal::estimate("Hello world. How are you?");
        assert_eq!(signal, ComplexitySignal::Simple);
    }

    #[test]
    fn complexity_expert_with_code() {
        let text = "Explain the intricacies of monomorphisation in Rust generics. \
                    How does the borrow checker interact with lifetime elision? \
                    Please provide examples? Are there edge cases? \
                    ```rust\nfn foo<T: Clone>(x: T) -> T { x.clone() }\n```";
        let signal = ComplexitySignal::estimate(text);
        assert_eq!(signal, ComplexitySignal::ExpertLevel);
    }

    #[test]
    fn complexity_moderate() {
        // Sentence with many long words but no code and few questions.
        let text = "Understanding cryptographic authentication mechanisms requires \
                    considerable expertise in distributed systems architecture.";
        let signal = ComplexitySignal::estimate(text);
        assert!(signal >= ComplexitySignal::Moderate);
    }

    #[test]
    fn estimated_cost_scales_linearly() {
        let cost_100 = CostAwareRouter::estimated_cost(&ModelTier::Standard, 100);
        let cost_200 = CostAwareRouter::estimated_cost(&ModelTier::Standard, 200);
        assert!((cost_200 - 2.0 * cost_100).abs() < 1e-12);
    }

    #[test]
    fn route_fixed_tier() {
        let router = CostAwareRouter::new();
        let policy = RoutingPolicy::FixedTier(ModelTier::Premium);
        assert_eq!(router.route("anything", &policy), ModelTier::Premium);
    }

    #[test]
    fn route_max_quality_within_budget_tight() {
        let router = CostAwareRouter::new();
        // Budget only covers Economy (0.5 * 1000 = $0.0005 budget, economy costs $0.0005)
        let policy = RoutingPolicy::MaxQualityWithinBudget {
            budget_usd: 0.000_5,
            token_estimate: 1000,
        };
        let tier = router.route("Hello", &policy);
        assert_eq!(tier, ModelTier::Economy);
    }

    #[test]
    fn route_min_cost_above_quality() {
        let router = CostAwareRouter::new();
        // min_quality=0.70 → Standard (0.72) satisfies it
        let policy = RoutingPolicy::MinCostAboveQuality {
            min_quality: 0.70,
            token_estimate: 500,
        };
        let tier = router.route("Simple text.", &policy);
        assert_eq!(tier, ModelTier::Standard);
    }

    #[test]
    fn routing_explanation_is_non_empty() {
        let router = CostAwareRouter::new();
        let policy = RoutingPolicy::FixedTier(ModelTier::Economy);
        let expl = router.routing_explanation("test prompt", &policy);
        assert!(!expl.is_empty());
        assert!(expl.contains("Economy"));
    }
}
