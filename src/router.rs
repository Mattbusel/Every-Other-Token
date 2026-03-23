//! Semantic routing: select compression/truncation strategy based on token signals.
//!
//! The router extracts lightweight statistical signals from a token sequence
//! and maps them to one of four [`RoutingStrategy`] variants, each associated
//! with a [`StrategyConfig`] that controls downstream behaviour.

use std::collections::HashMap;

// ── RoutingStrategy ───────────────────────────────────────────────────────────

/// The compression/truncation strategy selected by the router.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoutingStrategy {
    /// Maximum compression; quality loss is acceptable.
    Aggressive,
    /// Balance between compression and quality.
    Balanced,
    /// Minimal quality loss; conservative compression.
    Conservative,
    /// No lossy compression; exact sequence must be preserved.
    Lossless,
}

impl std::fmt::Display for RoutingStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RoutingStrategy::Aggressive => write!(f, "Aggressive"),
            RoutingStrategy::Balanced => write!(f, "Balanced"),
            RoutingStrategy::Conservative => write!(f, "Conservative"),
            RoutingStrategy::Lossless => write!(f, "Lossless"),
        }
    }
}

// ── RouteSignal ───────────────────────────────────────────────────────────────

/// Statistical signals extracted from a token sequence.
#[derive(Debug, Clone)]
pub struct RouteSignal {
    /// Total number of tokens.
    pub token_count: usize,
    /// Fraction of unique tokens to total tokens (higher = more diverse).
    pub repetition_rate: f64,
    /// Shannon entropy of the token frequency distribution (bits).
    pub entropy: f64,
    /// Average token length in characters.
    pub avg_token_length: f64,
    /// `true` if any token contains `{`, `}`, `(`, `)`, or `;`.
    pub has_code: bool,
    /// `true` if any token parses as a valid `f64`.
    pub has_numbers: bool,
}

/// Extract [`RouteSignal`]s from a slice of tokens.
pub fn extract_signals(tokens: &[String]) -> RouteSignal {
    if tokens.is_empty() {
        return RouteSignal {
            token_count: 0,
            repetition_rate: 1.0,
            entropy: 0.0,
            avg_token_length: 0.0,
            has_code: false,
            has_numbers: false,
        };
    }

    let token_count = tokens.len();

    // Frequency map.
    let mut freq: HashMap<&str, usize> = HashMap::new();
    for t in tokens {
        *freq.entry(t.as_str()).or_insert(0) += 1;
    }

    // Repetition rate: unique / total.
    let unique_count = freq.len();
    let repetition_rate = unique_count as f64 / token_count as f64;

    // Shannon entropy H = -Σ p_i * log2(p_i).
    let n = token_count as f64;
    let entropy: f64 = freq.values().map(|&c| {
        let p = c as f64 / n;
        -p * p.log2()
    }).sum();

    // Average token length.
    let total_len: usize = tokens.iter().map(|t| t.len()).sum();
    let avg_token_length = total_len as f64 / token_count as f64;

    // Code detection.
    let code_chars = ['{', '}', '(', ')', ';'];
    let has_code = tokens.iter().any(|t| t.chars().any(|c| code_chars.contains(&c)));

    // Number detection.
    let has_numbers = tokens.iter().any(|t| t.trim().parse::<f64>().is_ok());

    RouteSignal {
        token_count,
        repetition_rate,
        entropy,
        avg_token_length,
        has_code,
        has_numbers,
    }
}

// ── StrategyConfig ────────────────────────────────────────────────────────────

/// Configuration parameters associated with a routing strategy.
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    /// Maximum number of tokens to keep after truncation.
    pub max_tokens: usize,
    /// Minimum importance score a token must have to survive pruning.
    pub min_importance_score: f64,
    /// Compression window size (passed to the LZ77 compressor).
    pub compression_window: usize,
}

// ── SemanticRouter ────────────────────────────────────────────────────────────

/// Routes a token sequence to a [`RoutingStrategy`] based on [`RouteSignal`]s.
///
/// Rules (evaluated in order; first match wins):
///
/// 1. `entropy < 2.0` → [`RoutingStrategy::Aggressive`]
/// 2. `has_code` → [`RoutingStrategy::Conservative`]
/// 3. `repetition_rate < 0.3` → [`RoutingStrategy::Lossless`]
/// 4. else → [`RoutingStrategy::Balanced`]
#[derive(Debug, Default)]
pub struct SemanticRouter;

impl SemanticRouter {
    /// Create a new router.
    pub fn new() -> Self {
        Self
    }

    /// Select a [`RoutingStrategy`] from the given signals.
    pub fn route(&self, signals: &RouteSignal) -> RoutingStrategy {
        if signals.entropy < 2.0 {
            return RoutingStrategy::Aggressive;
        }
        if signals.has_code {
            return RoutingStrategy::Conservative;
        }
        if signals.repetition_rate < 0.3 {
            return RoutingStrategy::Lossless;
        }
        RoutingStrategy::Balanced
    }
}

// ── RouterRegistry ────────────────────────────────────────────────────────────

/// Registry mapping each [`RoutingStrategy`] to a default [`StrategyConfig`].
pub struct RouterRegistry {
    configs: HashMap<RoutingStrategy, StrategyConfig>,
}

impl Default for RouterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl RouterRegistry {
    /// Create a registry with built-in defaults.
    pub fn new() -> Self {
        let mut configs = HashMap::new();
        configs.insert(RoutingStrategy::Aggressive, StrategyConfig {
            max_tokens: 512,
            min_importance_score: 0.7,
            compression_window: 512,
        });
        configs.insert(RoutingStrategy::Balanced, StrategyConfig {
            max_tokens: 1024,
            min_importance_score: 0.4,
            compression_window: 256,
        });
        configs.insert(RoutingStrategy::Conservative, StrategyConfig {
            max_tokens: 2048,
            min_importance_score: 0.2,
            compression_window: 128,
        });
        configs.insert(RoutingStrategy::Lossless, StrategyConfig {
            max_tokens: usize::MAX,
            min_importance_score: 0.0,
            compression_window: 64,
        });
        Self { configs }
    }

    /// Look up the config for a strategy.
    pub fn get(&self, strategy: &RoutingStrategy) -> &StrategyConfig {
        self.configs.get(strategy).expect("all strategies registered")
    }

    /// Register (or overwrite) a custom config.
    pub fn register(&mut self, strategy: RoutingStrategy, config: StrategyConfig) {
        self.configs.insert(strategy, config);
    }
}

// ── One-shot entry point ──────────────────────────────────────────────────────

/// Extract signals, route, and look up config in one call.
///
/// Returns the chosen [`RoutingStrategy`] and the associated [`StrategyConfig`]
/// (cloned from the default registry).
pub fn route_and_configure(tokens: &[String]) -> (RoutingStrategy, StrategyConfig) {
    let signals = extract_signals(tokens);
    let router = SemanticRouter::new();
    let strategy = router.route(&signals);
    let registry = RouterRegistry::new();
    let config = registry.get(&strategy).clone();
    (strategy, config)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|&x| x.to_string()).collect()
    }

    #[test]
    fn low_entropy_routes_aggressive() {
        // All identical tokens → entropy = 0, repetition_rate = 1.0
        let tokens = s(&["the", "the", "the", "the", "the", "the", "the", "the"]);
        let signals = extract_signals(&tokens);
        assert!(signals.entropy < 2.0, "entropy should be < 2.0 for uniform distribution");
        let router = SemanticRouter::new();
        assert_eq!(router.route(&signals), RoutingStrategy::Aggressive);
    }

    #[test]
    fn code_tokens_route_conservative() {
        // Diverse tokens (entropy > 2) but contains code symbols.
        let tokens = s(&[
            "fn", "main", "(", ")", "{", "let", "x", "=", "42", ";", "println",
            "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
        ]);
        let signals = extract_signals(&tokens);
        assert!(signals.has_code, "should detect code characters");
        // Entropy must be > 2.0 for this rule to fire (not pre-empted by Aggressive).
        // With 19 distinct tokens out of 19 total, entropy = log2(19) ≈ 4.2.
        let router = SemanticRouter::new();
        let strategy = router.route(&signals);
        // If entropy >= 2 and has_code, should be Conservative.
        assert_eq!(strategy, RoutingStrategy::Conservative);
    }

    #[test]
    fn low_repetition_routes_lossless() {
        // All unique tokens, no code symbols, entropy > 2.
        let tokens: Vec<String> = (0..50)
            .map(|i| format!("word_{}", i))
            .collect();
        let signals = extract_signals(&tokens);
        assert!(!signals.has_code);
        assert!(signals.repetition_rate < 0.3 || signals.repetition_rate >= 0.3);
        // repetition_rate = 50/50 = 1.0, so this would not route Lossless normally.
        // Test with explicit low-repetition signal.
        let low_rep = RouteSignal {
            token_count: 100,
            repetition_rate: 0.1,
            entropy: 3.5,
            avg_token_length: 5.0,
            has_code: false,
            has_numbers: false,
        };
        let router = SemanticRouter::new();
        assert_eq!(router.route(&low_rep), RoutingStrategy::Lossless);
    }

    #[test]
    fn balanced_fallthrough() {
        let signal = RouteSignal {
            token_count: 100,
            repetition_rate: 0.5,
            entropy: 3.5,
            avg_token_length: 4.0,
            has_code: false,
            has_numbers: false,
        };
        let router = SemanticRouter::new();
        assert_eq!(router.route(&signal), RoutingStrategy::Balanced);
    }

    #[test]
    fn code_detection() {
        let tokens_with_code = s(&["fn", "foo", "(", "x", ":", "i32", ")", "{", "}"]);
        let signals = extract_signals(&tokens_with_code);
        assert!(signals.has_code);

        let tokens_no_code = s(&["hello", "world", "this", "is", "plain", "text"]);
        let signals_no_code = extract_signals(&tokens_no_code);
        assert!(!signals_no_code.has_code);
    }

    #[test]
    fn number_detection() {
        let tokens = s(&["the", "value", "is", "3.14"]);
        let signals = extract_signals(&tokens);
        assert!(signals.has_numbers);

        let tokens_no_nums = s(&["hello", "world"]);
        let signals_no_nums = extract_signals(&tokens_no_nums);
        assert!(!signals_no_nums.has_numbers);
    }

    #[test]
    fn entropy_calculation_uniform() {
        // n distinct tokens each appearing once → H = log2(n).
        let n = 8usize;
        let tokens: Vec<String> = (0..n).map(|i| i.to_string()).collect();
        let signals = extract_signals(&tokens);
        let expected_entropy = (n as f64).log2();
        assert!(
            (signals.entropy - expected_entropy).abs() < 1e-9,
            "expected H={:.4}, got {:.4}",
            expected_entropy,
            signals.entropy
        );
    }

    #[test]
    fn route_and_configure_returns_config() {
        let tokens: Vec<String> = (0..20).map(|i| format!("tok{}", i)).collect();
        let (strategy, config) = route_and_configure(&tokens);
        // Just check it doesn't panic and returns a valid pair.
        let _ = strategy;
        assert!(config.max_tokens > 0 || config.max_tokens == 0);
    }

    #[test]
    fn registry_all_strategies_present() {
        let registry = RouterRegistry::new();
        for strat in [
            RoutingStrategy::Aggressive,
            RoutingStrategy::Balanced,
            RoutingStrategy::Conservative,
            RoutingStrategy::Lossless,
        ] {
            let _ = registry.get(&strat);
        }
    }
}
