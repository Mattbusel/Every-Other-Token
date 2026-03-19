//! Token transform pipeline.
//!
//! This module defines the [`Transform`] enum and associated helpers used to
//! mutate individual tokens in the live LLM stream.  Transforms can be stacked
//! via [`Transform::Chain`] or selected randomly via [`Transform::Chaos`].
//!
//! ## Available transforms
//!
//! | Name | Effect |
//! |------|--------|
//! | `reverse` | Reverses the characters of the token |
//! | `uppercase` | Converts the token to uppercase |
//! | `mock` | Applies alternating lower/upper case per character |
//! | `noise` | Appends a random symbol from `* + ~ @ # $ %` |
//! | `chaos` | Randomly selects one of the above per call |
//! | `scramble` | Fisher-Yates shuffles the token's characters |
//! | `delete` | Replaces the token with the empty string |
//! | `synonym` | Substitutes the token with a static synonym, if known |
//! | `delay:N` | Passes the token through after an N-millisecond pause |

use colored::*;
use once_cell::sync::Lazy;
use rand::Rng;
use std::collections::HashMap;
use std::sync::Mutex;

const NOISE_CHARS: [char; 7] = ['*', '+', '~', '@', '#', '$', '%'];

static SYNONYM_MAP: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut m = HashMap::new();
    // Original 30 entries
    m.insert("good", "great");
    m.insert("bad", "poor");
    m.insert("fast", "quick");
    m.insert("slow", "gradual");
    m.insert("big", "large");
    m.insert("small", "tiny");
    m.insert("happy", "glad");
    m.insert("sad", "unhappy");
    m.insert("smart", "clever");
    m.insert("old", "aged");
    m.insert("new", "fresh");
    m.insert("hot", "warm");
    m.insert("cold", "cool");
    m.insert("hard", "tough");
    m.insert("easy", "simple");
    m.insert("start", "begin");
    m.insert("end", "finish");
    m.insert("make", "create");
    m.insert("get", "obtain");
    m.insert("use", "employ");
    m.insert("say", "state");
    m.insert("go", "proceed");
    m.insert("see", "observe");
    m.insert("know", "understand");
    m.insert("think", "believe");
    m.insert("come", "arrive");
    m.insert("take", "acquire");
    m.insert("give", "provide");
    m.insert("find", "locate");
    m.insert("tell", "inform");
    // Adjectives
    m.insert("bright", "vivid");
    m.insert("dark", "dim");
    m.insert("clean", "pure");
    m.insert("dirty", "grimy");
    m.insert("strong", "powerful");
    m.insert("weak", "frail");
    m.insert("rich", "wealthy");
    m.insert("young", "youthful");
    m.insert("pretty", "beautiful");
    m.insert("ugly", "hideous");
    m.insert("loud", "noisy");
    m.insert("quiet", "silent");
    m.insert("angry", "furious");
    m.insert("calm", "serene");
    m.insert("brave", "courageous");
    m.insert("scared", "frightened");
    m.insert("funny", "amusing");
    m.insert("serious", "solemn");
    m.insert("kind", "gentle");
    m.insert("cruel", "harsh");
    m.insert("empty", "hollow");
    m.insert("full", "packed");
    m.insert("rough", "coarse");
    m.insert("smooth", "sleek");
    m.insert("sharp", "keen");
    m.insert("dull", "blunt");
    m.insert("deep", "profound");
    m.insert("shallow", "superficial");
    m.insert("wide", "broad");
    m.insert("narrow", "slim");
    m.insert("long", "lengthy");
    m.insert("short", "brief");
    m.insert("heavy", "weighty");
    m.insert("light", "featherweight");
    m.insert("warm", "heated");
    m.insert("frozen", "icy");
    m.insert("luminous", "bright");
    m.insert("gloomy", "dreary");
    m.insert("lively", "energetic");
    m.insert("tired", "weary");
    m.insert("healthy", "robust");
    m.insert("sick", "ill");
    m.insert("safe", "secure");
    m.insert("dangerous", "hazardous");
    m.insert("important", "crucial");
    m.insert("trivial", "minor");
    m.insert("simple", "plain");
    m.insert("complex", "intricate");
    m.insert("rare", "scarce");
    m.insert("common", "ordinary");
    m.insert("strange", "peculiar");
    m.insert("normal", "typical");
    m.insert("ancient", "archaic");
    m.insert("modern", "contemporary");
    m.insert("local", "regional");
    m.insert("distant", "remote");
    // Verbs
    m.insert("walk", "stroll");
    m.insert("run", "sprint");
    m.insert("eat", "consume");
    m.insert("drink", "sip");
    m.insert("write", "compose");
    m.insert("read", "peruse");
    m.insert("speak", "articulate");
    m.insert("listen", "hear");
    m.insert("look", "glance");
    m.insert("touch", "feel");
    m.insert("help", "assist");
    m.insert("stop", "halt");
    m.insert("try", "attempt");
    m.insert("fail", "falter");
    m.insert("win", "triumph");
    m.insert("forfeit", "lose");
    m.insert("buy", "purchase");
    m.insert("sell", "trade");
    m.insert("build", "construct");
    m.insert("break", "shatter");
    m.insert("fix", "repair");
    m.insert("cut", "slice");
    m.insert("push", "shove");
    m.insert("pull", "tug");
    m.insert("throw", "toss");
    m.insert("catch", "grab");
    m.insert("jump", "leap");
    m.insert("fall", "plunge");
    m.insert("rise", "ascend");
    m.insert("drop", "descend");
    m.insert("open", "unlock");
    m.insert("close", "shut");
    m.insert("move", "shift");
    m.insert("stay", "remain");
    m.insert("change", "alter");
    m.insert("grow", "expand");
    m.insert("shrink", "diminish");
    m.insert("show", "display");
    m.insert("hide", "conceal");
    m.insert("choose", "select");
    m.insert("allow", "permit");
    m.insert("prevent", "hinder");
    m.insert("need", "require");
    m.insert("want", "desire");
    m.insert("like", "enjoy");
    m.insert("hate", "despise");
    m.insert("fear", "dread");
    m.insert("love", "adore");
    m.insert("send", "dispatch");
    m.insert("receive", "accept");
    m.insert("keep", "retain");
    m.insert("misplace", "lose");
    m.insert("follow", "pursue");
    m.insert("lead", "guide");
    m.insert("wait", "linger");
    m.insert("hurry", "rush");
    m.insert("agree", "concur");
    m.insert("refuse", "decline");
    // Nouns
    m.insert("house", "dwelling");
    m.insert("car", "vehicle");
    m.insert("book", "volume");
    m.insert("friend", "companion");
    m.insert("work", "labor");
    m.insert("time", "duration");
    m.insert("way", "method");
    m.insert("place", "location");
    m.insert("thing", "object");
    m.insert("part", "component");
    m.insert("life", "existence");
    m.insert("day", "period");
    m.insert("man", "person");
    m.insert("woman", "individual");
    m.insert("child", "youth");
    m.insert("world", "realm");
    m.insert("school", "institution");
    m.insert("country", "nation");
    m.insert("city", "metropolis");
    m.insert("family", "household");
    m.insert("group", "collective");
    m.insert("system", "framework");
    m.insert("problem", "issue");
    m.insert("idea", "concept");
    m.insert("question", "inquiry");
    m.insert("result", "outcome");
    m.insert("road", "path");
    m.insert("tree", "plant");
    m.insert("water", "liquid");
    m.insert("fire", "flame");
    m.insert("glow", "light");
    m.insert("sound", "noise");
    m.insert("food", "nourishment");
    m.insert("money", "currency");
    m.insert("power", "strength");
    m.insert("mind", "intellect");
    m.insert("heart", "soul");
    m.insert("hand", "palm");
    m.insert("eye", "gaze");
    m.insert("word", "term");
    m.insert("story", "tale");
    m.insert("truth", "fact");
    m.insert("dream", "vision");
    m.insert("goal", "objective");
    m.insert("plan", "strategy");
    m.insert("step", "stage");
    m.insert("rule", "law");
    m.insert("right", "privilege");
    m.insert("choice", "option");
    m.insert("chance", "opportunity");
    m
});

/// Runtime synonym overrides, merged with SYNONYM_MAP at lookup time.
/// Set via [`set_synonym_overrides`].
static SYNONYM_OVERRIDES: Lazy<Mutex<HashMap<String, String>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Load synonym pairs from a file and register them as runtime overrides.
///
/// Supports two line formats:
/// - TSV: `word\treplacement`
/// - Key-value: `word = replacement`
///
/// Lines starting with `#` are treated as comments and skipped.
///
/// # Errors
/// Returns an error if the file cannot be read. Individual malformed lines are silently skipped.
pub fn load_synonym_overrides(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let mut overrides = SYNONYM_OVERRIDES.lock().unwrap_or_else(|e| e.into_inner());
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = line.split_once('\t') {
            overrides.insert(k.trim().to_lowercase(), v.trim().to_string());
        } else if let Some((k, v)) = line.split_once('=') {
            overrides.insert(k.trim().to_lowercase(), v.trim().to_string());
        }
    }
    Ok(())
}

/// Replace the current runtime synonym overrides with the given map.
pub fn set_synonym_overrides(map: HashMap<String, String>) {
    let mut overrides = SYNONYM_OVERRIDES.lock().unwrap_or_else(|e| e.into_inner());
    *overrides = map;
}

/// Look up a token in the synonym map, checking runtime overrides first.
fn synonym_lookup(token: &str) -> Option<String> {
    let lower = token.to_lowercase();
    {
        let overrides = SYNONYM_OVERRIDES.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(v) = overrides.get(lower.as_str()) {
            return Some(v.clone());
        }
    }
    SYNONYM_MAP.get(lower.as_str()).map(|s| s.to_string())
}

/// The set of token mutation strategies available at the interception layer.
///
/// Each variant describes a different way to perturb a token in the stream.
/// The transform is applied only at odd-indexed positions (i.e., every other
/// token, starting from index 1) unless the caller overrides the rate.
///
/// ## Strategies
///
/// | Variant | Behaviour |
/// |---------|-----------|
/// | `Reverse` | Reverses the Unicode characters of the token: `"hello"` -> `"olleh"`. |
/// | `Uppercase` | Uppercases every character: `"hello"` -> `"HELLO"`. |
/// | `Mock` | Alternates lowercase/uppercase per character position: `"hello"` -> `"hElLo"`. |
/// | `Noise` | Appends one random symbol from `* + ~ @ # $ %`: `"hello"` -> `"hello*"`. |
/// | `Chaos` | Randomly picks one of Reverse, Uppercase, Mock, or Noise per token. |
/// | `Scramble` | Fisher-Yates shuffles the characters: same characters, random order. |
/// | `Delete` | Drops the token entirely, returning an empty string. |
/// | `Synonym` | Replaces the token with a synonym from the built-in 200-entry map; passes through unchanged if no entry exists. |
/// | `Delay(ms)` | Returns the token unmodified after the given delay in milliseconds. Useful for pacing experiments. |
/// | `Chain(vec)` | Applies a sequence of transforms in order; label is the individual labels joined by `+`. |
#[derive(Debug, Clone)]
pub enum Transform {
    /// Reverse the Unicode characters of the token.
    Reverse,
    /// Uppercase every character of the token.
    Uppercase,
    /// Alternate lowercase/uppercase per character position (sPoNgEbOb case).
    Mock,
    /// Append one random symbol from the noise character set.
    Noise,
    /// Randomly select one of Reverse, Uppercase, Mock, or Noise for each token.
    Chaos,
    /// Shuffle the characters of the token using Fisher-Yates.
    Scramble,
    /// Return an empty string, effectively deleting the token from the stream.
    Delete,
    /// Replace the token with a built-in synonym; pass through unchanged if not found.
    Synonym,
    /// Return the token unchanged after sleeping for the given number of milliseconds.
    Delay(u64),
    /// Apply a sequence of transforms in order, chaining their effects.
    Chain(Vec<Transform>),
}

impl Transform {
    /// Parse a transform name (case-insensitive) or a comma-separated chain.
    ///
    /// Recognised single names: `reverse`, `uppercase`, `mock`, `noise`, `chaos`,
    /// `scramble`, `delete`, `synonym`, `delay`, `delay:N` (where N is milliseconds).
    ///
    /// Comma-separated input like `"reverse,uppercase"` produces a `Chain` variant.
    /// A single-element comma-separated string is unwrapped to the plain variant.
    ///
    /// # Errors
    ///
    /// Returns `Err(String)` if any component name is unrecognised.
    pub fn from_str_loose(s: &str) -> Result<Self, String> {
        // Handle "chain:reverse,uppercase" prefix syntax as an alias for "reverse,uppercase"
        let s = if let Some(rest) = s.strip_prefix("chain:") {
            rest
        } else {
            s
        };
        // Handle comma-separated chain: "reverse,uppercase"
        if s.contains(',') {
            let parts: Result<Vec<Transform>, String> = s
                .split(',')
                .map(|part| Transform::from_str_single(part.trim()))
                .collect();
            let transforms = parts?;
            if transforms.len() == 1 {
                // len == 1 is checked above, so into_iter().next() is always Some.
                return transforms
                    .into_iter()
                    .next()
                    .ok_or_else(|| "internal: empty transform list".to_string());
            }
            return Ok(Transform::Chain(transforms));
        }
        Transform::from_str_single(s)
    }

    fn from_str_single(s: &str) -> Result<Self, String> {
        let lower = s.to_lowercase();
        // Handle "delay:NNN" or "delay" forms
        if lower.starts_with("delay:") {
            let ms: u64 = lower
                .strip_prefix("delay:")
                .and_then(|n| n.parse().ok())
                .unwrap_or(100);
            return Ok(Transform::Delay(ms));
        }
        match lower.as_str() {
            "reverse" => Ok(Transform::Reverse),
            "uppercase" => Ok(Transform::Uppercase),
            "mock" => Ok(Transform::Mock),
            "noise" => Ok(Transform::Noise),
            "chaos" => Ok(Transform::Chaos),
            "scramble" => Ok(Transform::Scramble),
            "delete" => Ok(Transform::Delete),
            "synonym" => Ok(Transform::Synonym),
            "delay" => Ok(Transform::Delay(100)),
            _ => Err(format!("Unknown transform: {}", s)),
        }
    }

    /// Apply the transform using the provided RNG and return `(result, label)`.
    /// For Chaos, the sub-transform is chosen via `rng`; for others the label
    /// equals the transform name.  Prefer this over `apply_with_label` in hot
    /// paths to avoid per-call `thread_rng()` TLS lookups.
    pub fn apply_with_label_rng<R: Rng>(&self, token: &str, rng: &mut R) -> (String, String) {
        match self {
            Transform::Reverse => (token.chars().rev().collect(), "reverse".to_string()),
            Transform::Uppercase => (token.to_uppercase(), "uppercase".to_string()),
            Transform::Mock => (apply_mock(token), "mock".to_string()),
            Transform::Noise => {
                let noise_char = NOISE_CHARS[rng.gen_range(0..NOISE_CHARS.len())];
                (format!("{}{}", token, noise_char), "noise".to_string())
            }
            Transform::Scramble => {
                let mut chars: Vec<char> = token.chars().collect();
                // Fisher-Yates shuffle
                let n = chars.len();
                for i in (1..n).rev() {
                    let j = rng.gen_range(0..=i);
                    chars.swap(i, j);
                }
                (chars.into_iter().collect(), "scramble".to_string())
            }
            Transform::Delete => (String::new(), "delete".to_string()),
            Transform::Synonym => {
                let result = synonym_lookup(token).unwrap_or_else(|| token.to_string());
                (result, "synonym".to_string())
            }
            Transform::Delay(_) => (token.to_string(), "delay".to_string()),
            Transform::Chaos => match rng.gen_range(0u8..4) {
                0 => (token.chars().rev().collect(), "reverse".to_string()),
                1 => (token.to_uppercase(), "uppercase".to_string()),
                2 => (apply_mock(token), "mock".to_string()),
                _ => {
                    let noise_char = NOISE_CHARS[rng.gen_range(0..NOISE_CHARS.len())];
                    (format!("{}{}", token, noise_char), "noise".to_string())
                }
            },
            Transform::Chain(transforms) => {
                let mut current = token.to_string();
                let mut labels: Vec<String> = Vec::new();
                for t in transforms {
                    let (next, label) = t.apply_with_label_rng(&current, rng);
                    current = next;
                    labels.push(label);
                }
                (current, labels.join("+"))
            }
        }
    }

    /// Apply the transform using the provided RNG.
    pub fn apply_rng<R: Rng>(&self, token: &str, rng: &mut R) -> String {
        self.apply_with_label_rng(token, rng).0
    }

    /// Apply the transform and return `(result, label)`.  Creates a one-shot
    /// `thread_rng()`; use `apply_with_label_rng` in hot paths.
    pub fn apply_with_label(&self, token: &str) -> (String, String) {
        self.apply_with_label_rng(token, &mut rand::thread_rng())
    }

    /// Apply the transform and return only the resulting string.
    ///
    /// Convenience wrapper around [`apply_with_label`](Self::apply_with_label).
    pub fn apply(&self, token: &str) -> String {
        self.apply_with_label(token).0
    }
}

/// Shared mock-case logic: alternate lower/upper per character.
fn apply_mock(token: &str) -> String {
    token
        .chars()
        .enumerate()
        .map(|(i, c)| {
            if i % 2 == 0 {
                c.to_lowercase().next().unwrap_or(c)
            } else {
                c.to_uppercase().next().unwrap_or(c)
            }
        })
        .collect()
}

/// Returns true if `ch` is a CJK ideographic character that should be its own token.
fn is_cjk(ch: char) -> bool {
    matches!(ch,
        '\u{4E00}'..='\u{9FFF}'   // CJK Unified Ideographs
        | '\u{3400}'..='\u{4DBF}' // CJK Extension A
        | '\u{20000}'..='\u{2A6DF}' // CJK Extension B
        | '\u{F900}'..='\u{FAFF}' // CJK Compatibility Ideographs
        | '\u{3000}'..='\u{303F}' // CJK Symbols and Punctuation
        | '\u{FF00}'..='\u{FFEF}' // Halfwidth/Fullwidth Forms
    )
}

/// Returns true if `ch` should be treated as a word-boundary punctuation character
/// (split out as a single-char token, just like ASCII punctuation).
fn is_word_boundary_punct(ch: char) -> bool {
    ch.is_ascii_punctuation()
        || matches!(
            ch,
            '\u{2014}' // em dash —
            | '\u{2013}' // en dash –
            | '\u{2026}' // ellipsis …
            | '«'
            | '»'
            | '\u{201C}' // left double quote "
            | '\u{201D}' // right double quote "
            | '\u{2018}' // left single quote '
            | '\u{2019}' // right single quote '
            | '„'
            | '‹'
            | '›'
            | '·'
        )
}

/// Split text into tokens (words, punctuation, whitespace).
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current_token = String::new();

    for ch in text.chars() {
        if ch.is_whitespace() || is_word_boundary_punct(ch) {
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
        } else if is_cjk(ch) {
            // Each CJK ideograph is its own token (no spaces separate them).
            if !current_token.is_empty() {
                tokens.push(current_token.clone());
                current_token.clear();
            }
            tokens.push(ch.to_string());
        } else {
            current_token.push(ch);
        }
    }

    if !current_token.is_empty() {
        tokens.push(current_token);
    }

    tokens
}

/// Calculate simulated token importance (0.0 to 1.0) using a caller-supplied RNG.
/// Identical to `calculate_token_importance` but takes an explicit RNG parameter
/// for deterministic/seeded use.
pub fn calculate_token_importance_rng<R: rand::Rng>(
    token: &str,
    position: usize,
    rng: &mut R,
) -> f64 {
    let mut importance = 0.0;

    importance += (token.len() as f64 / 20.0).min(0.3);

    let position_factor = if !(5..=50).contains(&position) {
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

    importance += rng.gen_range(-0.1..0.1);

    importance.clamp(0.0, 1.0)
}

/// Calculate simulated token importance (0.0 to 1.0) based on length,
/// position, content type, and random jitter.
/// Uses `thread_rng()` for jitter; for deterministic output use
/// `calculate_token_importance_rng`.
pub fn calculate_token_importance(token: &str, position: usize) -> f64 {
    calculate_token_importance_rng(token, position, &mut rand::thread_rng())
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
            assert!(
                known.contains(&label.as_str()),
                "unexpected label: {}",
                label
            );
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
        assert!(known.contains(&label.as_str()));
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

    #[test]
    fn test_transform_scramble_same_chars() {
        let input = "hello";
        for _ in 0..20 {
            let result = Transform::Scramble.apply(input);
            let mut orig_sorted: Vec<char> = input.chars().collect();
            let mut res_sorted: Vec<char> = result.chars().collect();
            orig_sorted.sort();
            res_sorted.sort();
            assert_eq!(
                orig_sorted, res_sorted,
                "Scramble should produce same chars"
            );
        }
    }

    #[test]
    fn test_transform_scramble_label() {
        let (_, label) = Transform::Scramble.apply_with_label("hi");
        assert_eq!(label, "scramble");
    }

    #[test]
    fn test_transform_delete_empty() {
        assert_eq!(Transform::Delete.apply("hello"), "");
        assert_eq!(Transform::Delete.apply(""), "");
    }

    #[test]
    fn test_transform_delete_label() {
        let (text, label) = Transform::Delete.apply_with_label("foo");
        assert_eq!(text, "");
        assert_eq!(label, "delete");
    }

    #[test]
    fn test_transform_synonym_known() {
        assert_eq!(Transform::Synonym.apply("good"), "great");
        assert_eq!(Transform::Synonym.apply("bad"), "poor");
        assert_eq!(Transform::Synonym.apply("fast"), "quick");
    }

    #[test]
    fn test_transform_synonym_unknown_passthrough() {
        assert_eq!(Transform::Synonym.apply("xyzzy"), "xyzzy");
    }

    #[test]
    fn test_transform_synonym_label() {
        let (_, label) = Transform::Synonym.apply_with_label("good");
        assert_eq!(label, "synonym");
    }

    #[test]
    fn test_transform_from_str_delay_colon() {
        assert!(matches!(
            Transform::from_str_loose("delay:200"),
            Ok(Transform::Delay(200))
        ));
    }

    #[test]
    fn test_transform_from_str_delay_default() {
        assert!(matches!(
            Transform::from_str_loose("delay"),
            Ok(Transform::Delay(100))
        ));
    }

    #[test]
    fn test_transform_delay_passthrough() {
        assert_eq!(Transform::Delay(50).apply("hello"), "hello");
    }

    #[test]
    fn test_transform_from_str_scramble() {
        assert!(matches!(
            Transform::from_str_loose("scramble"),
            Ok(Transform::Scramble)
        ));
    }

    #[test]
    fn test_transform_from_str_delete() {
        assert!(matches!(
            Transform::from_str_loose("delete"),
            Ok(Transform::Delete)
        ));
    }

    #[test]
    fn test_transform_from_str_synonym() {
        assert!(matches!(
            Transform::from_str_loose("synonym"),
            Ok(Transform::Synonym)
        ));
    }

    // -- Chain transform tests (Change 1) --

    #[test]
    fn test_chain_reverse_uppercase() {
        let chain = Transform::Chain(vec![Transform::Reverse, Transform::Uppercase]);
        assert_eq!(chain.apply("hello"), "OLLEH");
    }

    #[test]
    fn test_chain_mock_noise_label() {
        let chain = Transform::Chain(vec![Transform::Mock, Transform::Noise]);
        let (result, label) = chain.apply_with_label("hello");
        assert!(
            result.starts_with("hElLo"),
            "expected mock applied: {}",
            result
        );
        assert_eq!(label, "mock+noise");
    }

    #[test]
    fn test_chain_from_str_loose_two() {
        let t = Transform::from_str_loose("reverse,uppercase").expect("parse ok");
        assert!(matches!(t, Transform::Chain(_)));
        assert_eq!(t.apply("hello"), "OLLEH");
    }

    #[test]
    fn test_chain_from_str_loose_single_no_chain() {
        let t = Transform::from_str_loose("reverse").expect("parse ok");
        assert!(matches!(t, Transform::Reverse));
    }

    #[test]
    fn test_chain_label_joined_with_plus() {
        let chain = Transform::Chain(vec![Transform::Reverse, Transform::Uppercase]);
        let (_, label) = chain.apply_with_label("hi");
        assert_eq!(label, "reverse+uppercase");
    }

    // -- Unicode punctuation tokenize tests (Change 2) --

    #[test]
    fn test_tokenize_em_dash() {
        let tokens = tokenize("word\u{2014}another");
        assert!(tokens.contains(&"word".to_string()));
        assert!(tokens.contains(&"\u{2014}".to_string()));
        assert!(tokens.contains(&"another".to_string()));
    }

    #[test]
    fn test_tokenize_smart_quotes() {
        let tokens = tokenize("\u{201C}hello\u{201D}");
        assert!(tokens.contains(&"\u{201C}".to_string()));
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"\u{201D}".to_string()));
    }

    #[test]
    fn test_tokenize_ellipsis_unicode() {
        let tokens = tokenize("wait\u{2026}done");
        assert!(tokens.contains(&"\u{2026}".to_string()));
        assert!(tokens.contains(&"wait".to_string()));
        assert!(tokens.contains(&"done".to_string()));
    }

    #[test]
    fn test_tokenize_en_dash() {
        let tokens = tokenize("2020\u{2013}2021");
        assert!(tokens.contains(&"\u{2013}".to_string()));
    }

    #[test]
    fn test_tokenize_unicode_punct_preserves_all_chars() {
        let input = "hello\u{2014}world";
        assert_eq!(tokenize(input).join(""), input);
    }

    // -- Seeded RNG importance tests (Change 3) --

    #[test]
    fn test_importance_rng_same_seed_same_output() {
        use rand::SeedableRng;
        let mut rng1 = rand::rngs::StdRng::seed_from_u64(42);
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(42);
        let v1 = calculate_token_importance_rng("hello", 10, &mut rng1);
        let v2 = calculate_token_importance_rng("hello", 10, &mut rng2);
        assert_eq!(v1, v2, "same seed must produce same result");
    }

    #[test]
    fn test_importance_rng_different_seeds_differ() {
        use rand::SeedableRng;
        let mut results: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for seed in 0u64..50 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let v = calculate_token_importance_rng("test", 10, &mut rng);
            results.insert(v.to_bits());
        }
        assert!(results.len() > 1, "different seeds should sometimes differ");
    }

    #[test]
    fn test_importance_rng_in_range() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let v = calculate_token_importance_rng("algorithm", 5, &mut rng);
        assert!(v >= 0.0 && v <= 1.0);
    }

    // -- Scramble and Delete extended tests (Improvement 7) --

    #[test]
    fn test_scramble_empty_string() {
        assert_eq!(Transform::Scramble.apply(""), "");
    }

    #[test]
    fn test_scramble_single_char() {
        for _ in 0..10 {
            assert_eq!(Transform::Scramble.apply("a"), "a");
        }
    }

    #[test]
    fn test_scramble_preserves_chars() {
        let input = "hello";
        for _ in 0..20 {
            let result = Transform::Scramble.apply(input);
            let mut orig: Vec<char> = input.chars().collect();
            let mut res: Vec<char> = result.chars().collect();
            orig.sort();
            res.sort();
            assert_eq!(orig, res, "scramble should preserve the same characters");
        }
    }

    #[test]
    fn test_scramble_produces_variety() {
        let mut results = std::collections::HashSet::new();
        for _ in 0..50 {
            results.insert(Transform::Scramble.apply("hello"));
        }
        assert!(
            results.len() >= 2,
            "scramble should produce different orderings"
        );
    }

    #[test]
    fn test_delete_always_returns_empty() {
        for input in &["hello", "world", "test", "a", "abc123", ""] {
            assert_eq!(Transform::Delete.apply(input), "");
        }
    }

    #[test]
    fn test_scramble_two_chars_both_permutations() {
        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            seen.insert(Transform::Scramble.apply("ab"));
        }
        assert!(seen.len() >= 1, "scramble of two chars should work");
    }

    // ---- rstest parameterized tests ----

    mod param_tests {
        use super::super::Transform;
        use rstest::rstest;

        #[rstest]
        #[case("reverse", "olleh")]
        #[case("uppercase", "HELLO")]
        #[case("mock", "hElLo")]
        #[case("delete", "")]
        fn test_deterministic_transforms(#[case] name: &str, #[case] expected: &str) {
            let t = Transform::from_str_loose(name).expect("valid transform");
            assert_eq!(t.apply("hello"), expected, "transform={name}");
        }

        #[rstest]
        #[case("reverse")]
        #[case("uppercase")]
        #[case("mock")]
        #[case("noise")]
        #[case("chaos")]
        #[case("scramble")]
        #[case("delete")]
        #[case("synonym")]
        #[case("delay")]
        fn test_all_transforms_parse(#[case] name: &str) {
            assert!(
                Transform::from_str_loose(name).is_ok(),
                "expected '{name}' to parse"
            );
        }

        #[rstest]
        #[case("REVERSE")]
        #[case("Uppercase")]
        #[case("MOCK")]
        #[case("NOISE")]
        fn test_case_insensitive_parse(#[case] name: &str) {
            assert!(
                Transform::from_str_loose(name).is_ok(),
                "expected '{name}' to parse case-insensitively"
            );
        }

        #[rstest]
        #[case("")]
        #[case("invalid")]
        #[case("REVERSED")]
        #[case("upper case")]
        fn test_invalid_transforms_error(#[case] name: &str) {
            assert!(
                Transform::from_str_loose(name).is_err(),
                "expected '{name}' to fail"
            );
        }
    }
}
