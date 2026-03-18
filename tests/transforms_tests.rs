//! External integration tests for the `transforms` module public API.

use every_other_token::transforms::Transform;

// ---------------------------------------------------------------------------
// from_str_loose — parsing
// ---------------------------------------------------------------------------

#[test]
fn test_from_str_reverse() {
    let t = Transform::from_str_loose("reverse").expect("parse");
    assert!(matches!(t, Transform::Reverse));
}

#[test]
fn test_from_str_uppercase() {
    let t = Transform::from_str_loose("uppercase").expect("parse");
    assert!(matches!(t, Transform::Uppercase));
}

#[test]
fn test_from_str_mock() {
    let t = Transform::from_str_loose("mock").expect("parse");
    assert!(matches!(t, Transform::Mock));
}

#[test]
fn test_from_str_noise() {
    let t = Transform::from_str_loose("noise").expect("parse");
    assert!(matches!(t, Transform::Noise));
}

#[test]
fn test_from_str_chaos() {
    let t = Transform::from_str_loose("chaos").expect("parse");
    assert!(matches!(t, Transform::Chaos));
}

#[test]
fn test_from_str_scramble() {
    let t = Transform::from_str_loose("scramble").expect("parse");
    assert!(matches!(t, Transform::Scramble));
}

#[test]
fn test_from_str_delete() {
    let t = Transform::from_str_loose("delete").expect("parse");
    assert!(matches!(t, Transform::Delete));
}

#[test]
fn test_from_str_synonym() {
    let t = Transform::from_str_loose("synonym").expect("parse");
    assert!(matches!(t, Transform::Synonym));
}

#[test]
fn test_from_str_delay() {
    let t = Transform::from_str_loose("delay:50").expect("parse");
    assert!(matches!(t, Transform::Delay(50)));
}

#[test]
fn test_from_str_chain() {
    let t = Transform::from_str_loose("reverse,uppercase").expect("parse");
    assert!(matches!(t, Transform::Chain(_)));
}

#[test]
fn test_from_str_unknown_returns_err() {
    assert!(Transform::from_str_loose("nonexistent").is_err());
}

#[test]
fn test_from_str_case_insensitive() {
    Transform::from_str_loose("REVERSE").expect("case insensitive");
    Transform::from_str_loose("Uppercase").expect("case insensitive mixed");
}

// ---------------------------------------------------------------------------
// apply — transform behaviour
// ---------------------------------------------------------------------------

#[test]
fn test_reverse_apply() {
    let t = Transform::Reverse;
    let (out, _) = t.apply_with_label("hello");
    assert_eq!(out, "olleh");
}

#[test]
fn test_uppercase_apply() {
    let t = Transform::Uppercase;
    let (out, _) = t.apply_with_label("hello");
    assert_eq!(out, "HELLO");
}

#[test]
fn test_delete_apply_returns_empty() {
    let t = Transform::Delete;
    let (out, _) = t.apply_with_label("any");
    assert!(out.is_empty());
}

#[test]
fn test_synonym_unknown_token_passthrough() {
    // A word not in the synonym table should pass through unchanged.
    let t = Transform::Synonym;
    let unusual = "qxyzplonk";
    let (out, _) = t.apply_with_label(unusual);
    assert_eq!(out, unusual);
}

#[test]
fn test_synonym_known_word_substituted() {
    let t = Transform::Synonym;
    let (out, _) = t.apply_with_label("good");
    assert_ne!(out, "good", "known synonym should be substituted");
}

#[test]
fn test_noise_append_symbol() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let t = Transform::Noise;
    let (out, _) = t.apply_with_label_rng("word", &mut rng);
    // Must have at least one extra character appended
    assert!(out.len() > "word".len());
    assert!(out.starts_with("word"));
}

#[test]
fn test_chain_applies_in_order() {
    // "reverse,uppercase": reverse first → "olleh", then uppercase → "OLLEH"
    let t = Transform::from_str_loose("reverse,uppercase").expect("parse");
    let (out, _) = t.apply_with_label("hello");
    assert_eq!(out, "OLLEH");
}

#[test]
fn test_apply_with_label_returns_label() {
    let t = Transform::Reverse;
    let (_, label) = t.apply_with_label("token");
    assert!(!label.is_empty());
}

// ---------------------------------------------------------------------------
// tokenize
// ---------------------------------------------------------------------------

#[test]
fn test_tokenize_splits_words() {
    use every_other_token::transforms::tokenize;
    let tokens = tokenize("hello world");
    assert!(tokens.contains(&"hello".to_string()) || tokens.iter().any(|t| t.contains("hello")));
}

#[test]
fn test_tokenize_empty_string() {
    use every_other_token::transforms::tokenize;
    let tokens = tokenize("");
    assert!(tokens.is_empty());
}

// ---------------------------------------------------------------------------
// calculate_token_importance
// ---------------------------------------------------------------------------

#[test]
fn test_calculate_token_importance_range() {
    use every_other_token::transforms::calculate_token_importance;
    let imp = calculate_token_importance("the", 0);
    assert!((0.0..=1.0).contains(&imp));
}

#[test]
fn test_calculate_token_importance_deterministic() {
    use every_other_token::transforms::calculate_token_importance;
    let a = calculate_token_importance("word", 5);
    let b = calculate_token_importance("word", 5);
    assert!((a - b).abs() < 1e-12);
}
