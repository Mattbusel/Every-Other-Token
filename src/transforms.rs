use colored::*;
use rand::Rng;

#[derive(Debug, Clone)]
pub enum Transform {
    Reverse,
    Uppercase,
    Mock,
    Noise,
    Chaos,
}

impl Transform {
    pub fn from_str_loose(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "reverse" => Ok(Transform::Reverse),
            "uppercase" => Ok(Transform::Uppercase),
            "mock" => Ok(Transform::Mock),
            "noise" => Ok(Transform::Noise),
            "chaos" => Ok(Transform::Chaos),
            _ => Err(format!("Unknown transform: {}", s)),
        }
    }

    /// Apply the transform and return `(result, label)` where `label` names the
    /// sub-transform actually applied.  For Chaos, the sub-transform is chosen
    /// randomly at call time; for others the label equals the transform name.
    pub fn apply_with_label(&self, token: &str) -> (String, &'static str) {
        match self {
            Transform::Reverse => (token.chars().rev().collect(), "reverse"),
            Transform::Uppercase => (token.to_uppercase(), "uppercase"),
            Transform::Mock => {
                let result = token
                    .chars()
                    .enumerate()
                    .map(|(i, c)| {
                        if i % 2 == 0 {
                            c.to_lowercase().next().unwrap_or(c)
                        } else {
                            c.to_uppercase().next().unwrap_or(c)
                        }
                    })
                    .collect();
                (result, "mock")
            }
            Transform::Noise => {
                let mut rng = rand::thread_rng();
                let noise_chars = ['*', '+', '~', '@', '#', '$', '%'];
                let noise_char = noise_chars[rng.gen_range(0..noise_chars.len())];
                (format!("{}{}", token, noise_char), "noise")
            }
            Transform::Chaos => {
                let mut rng = rand::thread_rng();
                match rng.gen_range(0u8..4) {
                    0 => (token.chars().rev().collect(), "reverse"),
                    1 => (token.to_uppercase(), "uppercase"),
                    2 => {
                        let result = token
                            .chars()
                            .enumerate()
                            .map(|(i, c)| {
                                if i % 2 == 0 {
                                    c.to_lowercase().next().unwrap_or(c)
                                } else {
                                    c.to_uppercase().next().unwrap_or(c)
                                }
                            })
                            .collect();
                        (result, "mock")
                    }
                    _ => {
                        let noise_chars = ['*', '+', '~', '@', '#', '$', '%'];
                        let noise_char = noise_chars[rng.gen_range(0..noise_chars.len())];
                        (format!("{}{}", token, noise_char), "noise")
                    }
                }
            }
        }
    }

    pub fn apply(&self, token: &str) -> String {
        self.apply_with_label(token).0
    }
}

/// Split text into tokens (words, punctuation, whitespace).
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current_token = String::new();

    for ch in text.chars() {
        if ch.is_whitespace() || ch.is_ascii_punctuation() {
            if !current_token.is_empty() {
                tokens.push(current_token.clone());
                current_token.clear();
            }
            if !ch.is_whitespace() {
                tokens.push(ch.to_string());
            }
            if ch.is_whitespace() {
                tokens.push(ch.to_string());
            }
        } else {
            current_token.push(ch);
        }
    }

    if !current_token.is_empty() {
        tokens.push(current_token);
    }

    tokens
}

/// Calculate simulated token importance (0.0 to 1.0) based on length,
/// position, content type, and random jitter.
pub fn calculate_token_importance(token: &str, position: usize) -> f64 {
    let mut importance = 0.0;

    importance += (token.len() as f64 / 20.0).min(0.3);

    let position_factor = if position < 5 || position > 50 {
        0.3
    } else {
        0.1
    };
    importance += position_factor;

    if token.chars().any(|c| c.is_uppercase()) {
        importance += 0.2;
    }

    let important_patterns = [
        "the",
        "and",
        "or",
        "but",
        "if",
        "when",
        "where",
        "how",
        "why",
        "what",
        "robot",
        "AI",
        "technology",
        "system",
        "data",
        "algorithm",
        "model",
        "create",
        "build",
        "develop",
        "analyze",
        "process",
        "generate",
    ];

    let lower_token = token.to_lowercase();
    if important_patterns
        .iter()
        .any(|&pattern| lower_token.contains(pattern))
    {
        importance += 0.3;
    }

    if token.chars().all(|c| c.is_ascii_punctuation()) {
        importance *= 0.1;
    }

    let mut rng = rand::thread_rng();
    importance += rng.gen_range(-0.1..0.1);

    importance.clamp(0.0, 1.0)
}

/// Map an importance score to a terminal heatmap color.
pub fn apply_heatmap_color(token: &str, importance: f64) -> String {
    match importance {
        i if i >= 0.8 => token.on_bright_red().bright_white().to_string(),
        i if i >= 0.6 => token.on_red().bright_white().to_string(),
        i if i >= 0.4 => token.on_yellow().black().to_string(),
        i if i >= 0.2 => token.on_blue().bright_white().to_string(),
        _ => token.normal().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Transform apply tests --

    #[test]
    fn test_transform_reverse() {
        assert_eq!(Transform::Reverse.apply("hello"), "olleh");
        assert_eq!(Transform::Reverse.apply("world"), "dlrow");
    }

    #[test]
    fn test_transform_uppercase() {
        assert_eq!(Transform::Uppercase.apply("hello"), "HELLO");
        assert_eq!(Transform::Uppercase.apply("world"), "WORLD");
    }

    #[test]
    fn test_transform_mock() {
        assert_eq!(Transform::Mock.apply("hello"), "hElLo");
        assert_eq!(Transform::Mock.apply("world"), "wOrLd");
    }

    #[test]
    fn test_transform_noise() {
        let result = Transform::Noise.apply("hello");
        assert!(result.starts_with("hello"));
        assert!(result.len() > 5);
    }

    #[test]
    fn test_transform_from_str_valid() {
        assert!(matches!(
            Transform::from_str_loose("reverse"),
            Ok(Transform::Reverse)
        ));
        assert!(matches!(
            Transform::from_str_loose("uppercase"),
            Ok(Transform::Uppercase)
        ));
        assert!(matches!(
            Transform::from_str_loose("mock"),
            Ok(Transform::Mock)
        ));
        assert!(matches!(
            Transform::from_str_loose("noise"),
            Ok(Transform::Noise)
        ));
    }

    #[test]
    fn test_transform_from_str_invalid() {
        assert!(Transform::from_str_loose("invalid").is_err());
        assert!(Transform::from_str_loose("").is_err());
        assert!(Transform::from_str_loose("foo").is_err());
        assert!(Transform::from_str_loose("REVERSED").is_err());
    }

    #[test]
    fn test_transform_from_str_case_insensitive() {
        assert!(matches!(
            Transform::from_str_loose("REVERSE"),
            Ok(Transform::Reverse)
        ));
        assert!(matches!(
            Transform::from_str_loose("Uppercase"),
            Ok(Transform::Uppercase)
        ));
        assert!(matches!(
            Transform::from_str_loose("MoCk"),
            Ok(Transform::Mock)
        ));
    }

    #[test]
    fn test_transform_empty_inputs() {
        assert_eq!(Transform::Reverse.apply(""), "");
        assert_eq!(Transform::Uppercase.apply(""), "");
        assert_eq!(Transform::Mock.apply(""), "");
        assert_eq!(Transform::Noise.apply("").len(), 1);
    }

    #[test]
    fn test_transform_single_char() {
        assert_eq!(Transform::Reverse.apply("a"), "a");
        assert_eq!(Transform::Mock.apply("a"), "a");
        assert_eq!(Transform::Mock.apply("A"), "a");
    }

    #[test]
    fn test_transform_mock_two_chars() {
        assert_eq!(Transform::Mock.apply("ab"), "aB");
        assert_eq!(Transform::Mock.apply("AB"), "aB");
    }

    #[test]
    fn test_transform_preserves_length() {
        let inputs = ["hello", "a", "ab", "abcdefghij", ""];
        for input in &inputs {
            assert_eq!(Transform::Reverse.apply(input).len(), input.len());
            assert_eq!(Transform::Uppercase.apply(input).len(), input.len());
            assert_eq!(Transform::Mock.apply(input).len(), input.len());
        }
    }

    #[test]
    fn test_transform_noise_appends_one_char() {
        for _ in 0..20 {
            let result = Transform::Noise.apply("test");
            assert_eq!(result.len(), 5);
            assert!(result.starts_with("test"));
        }
    }

    #[test]
    fn test_transform_noise_char_from_set() {
        let noise_set = ['*', '+', '~', '@', '#', '$', '%'];
        for _ in 0..50 {
            let result = Transform::Noise.apply("x");
            let noise_char = result.chars().last().expect("should have noise char");
            assert!(
                noise_set.contains(&noise_char),
                "unexpected: {}",
                noise_char
            );
        }
    }

    #[test]
    fn test_reverse_is_involution() {
        let token = "hello";
        assert_eq!(
            Transform::Reverse.apply(&Transform::Reverse.apply(token)),
            token
        );
    }

    #[test]
    fn test_uppercase_is_idempotent() {
        let once = Transform::Uppercase.apply("hello");
        assert_eq!(Transform::Uppercase.apply(&once), once);
    }

    #[test]
    fn test_noise_length_always_plus_one() {
        for token in &["a", "hello", "test123", ""] {
            assert_eq!(Transform::Noise.apply(token).len(), token.len() + 1);
        }
    }

    #[test]
    fn test_all_transforms_produce_different_results() {
        let results: Vec<String> = [Transform::Reverse, Transform::Uppercase, Transform::Mock]
            .iter()
            .map(|t| t.apply("hello"))
            .collect();
        assert_ne!(results[0], results[1]);
        assert_ne!(results[1], results[2]);
        assert_ne!(results[0], results[2]);
    }

    #[test]
    fn test_uppercase_already_upper() {
        assert_eq!(Transform::Uppercase.apply("HELLO"), "HELLO");
    }

    #[test]
    fn test_uppercase_with_numbers() {
        assert_eq!(Transform::Uppercase.apply("test123"), "TEST123");
    }

    #[test]
    fn test_reverse_with_numbers() {
        assert_eq!(Transform::Reverse.apply("abc123"), "321cba");
    }

    #[test]
    fn test_mock_longer_string() {
        assert_eq!(Transform::Mock.apply("abcdef"), "aBcDeF");
    }

    // -- Tokenizer tests --

    #[test]
    fn test_tokenize_simple_sentence() {
        let tokens = tokenize("hello world");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
    }

    #[test]
    fn test_tokenize_with_punctuation() {
        let tokens = tokenize("hello, world!");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&",".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"!".to_string()));
    }

    #[test]
    fn test_tokenize_empty() {
        assert!(tokenize("").is_empty());
    }

    #[test]
    fn test_tokenize_single_word() {
        assert_eq!(tokenize("hello"), vec!["hello"]);
    }

    #[test]
    fn test_tokenize_only_whitespace() {
        assert!(tokenize("   ").iter().all(|t| t.trim().is_empty()));
    }

    #[test]
    fn test_tokenize_only_punctuation() {
        assert_eq!(tokenize("..."), vec![".", ".", "."]);
    }

    #[test]
    fn test_tokenize_mixed() {
        let tokens = tokenize("hello,world");
        assert_eq!(tokens, vec!["hello", ",", "world"]);
    }

    #[test]
    fn test_tokenize_preserves_all_chars() {
        let input = "hello, world! foo";
        assert_eq!(tokenize(input).join(""), input);
    }

    #[test]
    fn test_tokenize_multiple_spaces() {
        let tokens = tokenize("a  b");
        assert!(tokens.contains(&"a".to_string()));
        assert!(tokens.contains(&"b".to_string()));
    }

    #[test]
    fn test_tokenize_leading_trailing_space() {
        assert!(tokenize(" hello ").iter().any(|t| t == "hello"));
    }

    #[test]
    fn test_tokenize_numbers() {
        let tokens = tokenize("42 is the answer");
        assert!(tokens.contains(&"42".to_string()));
    }

    // -- Importance scoring tests --

    #[test]
    fn test_importance_clamped() {
        for pos in 0..100 {
            let imp = calculate_token_importance("test", pos);
            assert!(imp >= 0.0 && imp <= 1.0);
        }
    }

    #[test]
    fn test_punctuation_low_importance() {
        let mut total = 0.0;
        for _ in 0..100 {
            total += calculate_token_importance(".", 25);
        }
        assert!(total / 100.0 < 0.3);
    }

    #[test]
    fn test_importance_early_position_boost() {
        let early: f64 = (0..5)
            .map(|p| calculate_token_importance("word", p))
            .sum::<f64>()
            / 5.0;
        let mid: f64 = (10..15)
            .map(|p| calculate_token_importance("word", p))
            .sum::<f64>()
            / 5.0;
        assert!(early > mid - 0.2);
    }

    #[test]
    fn test_importance_uppercase_boost() {
        let n = 200;
        let upper: f64 = (0..n)
            .map(|_| calculate_token_importance("AI", 25))
            .sum::<f64>()
            / n as f64;
        let lower: f64 = (0..n)
            .map(|_| calculate_token_importance("ai", 25))
            .sum::<f64>()
            / n as f64;
        assert!(upper > lower);
    }

    #[test]
    fn test_importance_keyword_boost() {
        let n = 200;
        let kw: f64 = (0..n)
            .map(|_| calculate_token_importance("algorithm", 25))
            .sum::<f64>()
            / n as f64;
        let plain: f64 = (0..n)
            .map(|_| calculate_token_importance("xyz", 25))
            .sum::<f64>()
            / n as f64;
        assert!(kw > plain);
    }

    #[test]
    fn test_importance_long_token_boost() {
        let n = 200;
        let long: f64 = (0..n)
            .map(|_| calculate_token_importance("supercalifragilistic", 25))
            .sum::<f64>()
            / n as f64;
        let short: f64 = (0..n)
            .map(|_| calculate_token_importance("a", 25))
            .sum::<f64>()
            / n as f64;
        assert!(long > short);
    }

    #[test]
    fn test_importance_all_tokens_in_range() {
        let tokens = [
            ".",
            ",",
            "!",
            "?",
            "a",
            "AI",
            "algorithm",
            "the",
            "superlongtoken",
        ];
        for token in &tokens {
            for pos in [0, 1, 5, 25, 50, 100] {
                let imp = calculate_token_importance(token, pos);
                assert!(imp >= 0.0 && imp <= 1.0);
            }
        }
    }

    // -- Heatmap color tests --

    #[test]
    fn test_heatmap_color_nonempty() {
        for level in [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0] {
            assert!(!apply_heatmap_color("test", level).is_empty());
        }
    }

    #[test]
    fn test_heatmap_color_contains_text() {
        assert!(apply_heatmap_color("mytoken", 0.5).contains("mytoken"));
    }

    // -- Chaos transform tests --

    #[test]
    fn test_transform_chaos_from_str() {
        assert!(matches!(
            Transform::from_str_loose("chaos"),
            Ok(Transform::Chaos)
        ));
    }

    #[test]
    fn test_transform_chaos_from_str_case_insensitive() {
        assert!(matches!(
            Transform::from_str_loose("CHAOS"),
            Ok(Transform::Chaos)
        ));
        assert!(matches!(
            Transform::from_str_loose("Chaos"),
            Ok(Transform::Chaos)
        ));
    }

    #[test]
    fn test_transform_chaos_apply_nonempty() {
        for _ in 0..20 {
            let result = Transform::Chaos.apply("hello");
            assert!(!result.is_empty());
        }
    }

    #[test]
    fn test_transform_chaos_apply_with_label_returns_known_label() {
        let known = ["reverse", "uppercase", "mock", "noise"];
        for _ in 0..50 {
            let (_text, label) = Transform::Chaos.apply_with_label("hello");
            assert!(known.contains(&label), "unexpected label: {}", label);
        }
    }

    #[test]
    fn test_transform_chaos_apply_with_label_text_nonempty() {
        for _ in 0..20 {
            let (text, _label) = Transform::Chaos.apply_with_label("world");
            assert!(!text.is_empty());
        }
    }

    #[test]
    fn test_transform_chaos_empty_input() {
        // Noise appends 1 char, others keep length 0; either way no panic
        let (_text, label) = Transform::Chaos.apply_with_label("");
        let known = ["reverse", "uppercase", "mock", "noise"];
        assert!(known.contains(&label));
    }

    #[test]
    fn test_apply_with_label_non_chaos_label_matches_name() {
        assert_eq!(Transform::Reverse.apply_with_label("hi").1, "reverse");
        assert_eq!(Transform::Uppercase.apply_with_label("hi").1, "uppercase");
        assert_eq!(Transform::Mock.apply_with_label("hi").1, "mock");
        assert_eq!(Transform::Noise.apply_with_label("hi").1, "noise");
    }

    #[test]
    fn test_apply_with_label_text_matches_apply() {
        let inputs = ["hello", "world", "test", ""];
        for input in &inputs {
            // Deterministic transforms only (not Chaos/Noise which are random)
            assert_eq!(
                Transform::Reverse.apply_with_label(input).0,
                Transform::Reverse.apply(input)
            );
            assert_eq!(
                Transform::Uppercase.apply_with_label(input).0,
                Transform::Uppercase.apply(input)
            );
            assert_eq!(
                Transform::Mock.apply_with_label(input).0,
                Transform::Mock.apply(input)
            );
        }
    }

    #[test]
    fn test_transform_chaos_produces_variety_over_many_calls() {
        // Over 100 calls, Chaos should produce at least 2 distinct results
        let mut results: std::collections::HashSet<String> = std::collections::HashSet::new();
        for _ in 0..100 {
            results.insert(Transform::Chaos.apply("hello"));
        }
        assert!(results.len() >= 2, "Chaos should produce varied results");
    }
}
