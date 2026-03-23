//! Lossy and lossless prompt compression for token budget management.

/// Compression aggressiveness level.
#[derive(Debug, Clone, PartialEq)]
pub enum CompressionLevel {
    None,
    Light,
    Medium,
    Aggressive,
}

/// Result of a compression operation.
#[derive(Debug, Clone)]
pub struct CompressionResult {
    pub original_text: String,
    pub compressed_text: String,
    pub original_tokens: usize,
    pub compressed_tokens: usize,
    pub compression_ratio: f64,
    pub lossy: bool,
}

/// Configuration for the prompt compressor.
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    pub level: CompressionLevel,
    pub max_output_tokens: usize,
    pub preserve_code: bool,
    pub preserve_json: bool,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            level: CompressionLevel::Light,
            max_output_tokens: 4096,
            preserve_code: true,
            preserve_json: true,
        }
    }
}

/// Applies text compression pipelines based on configuration.
pub struct PromptCompressor {
    pub config: CompressionConfig,
}

/// Estimate token count: word_count * 1.3 + 1.
pub fn estimate_tokens(text: &str) -> usize {
    let word_count = text.split_whitespace().count();
    (word_count as f64 * 1.3 + 1.0) as usize
}

/// Returns true if text looks like a code block.
pub fn is_code_block(text: &str) -> bool {
    if text.trim_start().starts_with("```") {
        return true;
    }
    // Check for 4-space indent lines
    text.lines().any(|line| line.starts_with("    ") && !line.trim().is_empty())
}

/// Returns true if text looks like JSON.
pub fn is_json(text: &str) -> bool {
    let t = text.trim_start();
    t.starts_with('{') || t.starts_with('[')
}

/// Deduplicate consecutive identical sentences (split on .!?).
pub fn remove_redundancy(text: &str) -> String {
    let mut sentences: Vec<&str> = Vec::new();
    let mut start = 0;
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if matches!(chars[i], '.' | '!' | '?') {
            let end = i + 1;
            let sentence = &text[start..end];
            sentences.push(sentence);
            // skip whitespace
            i += 1;
            while i < chars.len() && chars[i] == ' ' {
                i += 1;
            }
            start = i;
        } else {
            i += 1;
        }
    }
    if start < text.len() {
        sentences.push(&text[start..]);
    }

    let mut result = String::new();
    let mut prev: Option<&str> = None;
    for s in &sentences {
        let trimmed = s.trim();
        if Some(trimmed) != prev.map(|p: &str| p.trim()) {
            if !result.is_empty() && !result.ends_with(' ') {
                result.push(' ');
            }
            result.push_str(s);
            prev = Some(trimmed);
        }
    }
    result
}

/// Replace large numbers with abbreviations: 1000000 -> 1M, 1000 -> 1K.
pub fn abbreviate_numbers(text: &str) -> String {
    let mut result = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_ascii_digit() {
            let start = i;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            let num_str = &text[start..i];
            if let Ok(n) = num_str.parse::<u64>() {
                if n >= 1_000_000 {
                    result.push_str(&format!("{}M", n / 1_000_000));
                } else if n >= 1_000 {
                    result.push_str(&format!("{}K", n / 1_000));
                } else {
                    result.push_str(num_str);
                }
            } else {
                result.push_str(num_str);
            }
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }
    result
}

const FILLER_WORDS: &[&str] = &[
    "basically",
    "essentially",
    "actually",
    "literally",
    "very",
    "really",
    "just",
    "quite",
    "rather",
    "simply",
];

/// Remove filler words from text.
pub fn strip_filler_words(text: &str) -> String {
    let mut result = String::new();
    let mut words = text.split_whitespace().peekable();
    while let Some(word) = words.next() {
        // Strip punctuation for comparison
        let lower = word.to_lowercase();
        let core = lower.trim_matches(|c: char| !c.is_alphanumeric());
        if !FILLER_WORDS.contains(&core) {
            if !result.is_empty() {
                result.push(' ');
            }
            result.push_str(word);
        }
    }
    result
}

/// Truncate sentences longer than max_words to max_words + "...".
pub fn shorten_sentences(text: &str, max_words: usize) -> String {
    let mut result = String::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if matches!(bytes[i], b'.' | b'!' | b'?') {
            let sentence = &text[start..i + 1];
            let shortened = shorten_one(sentence, max_words);
            if !result.is_empty() && !result.ends_with(' ') {
                result.push(' ');
            }
            result.push_str(&shortened);
            i += 1;
            while i < bytes.len() && bytes[i] == b' ' {
                i += 1;
            }
            start = i;
        } else {
            i += 1;
        }
    }
    if start < text.len() {
        let sentence = &text[start..];
        let shortened = shorten_one(sentence, max_words);
        if !result.is_empty() && !result.ends_with(' ') {
            result.push(' ');
        }
        result.push_str(&shortened);
    }
    result
}

fn shorten_one(sentence: &str, max_words: usize) -> String {
    let words: Vec<&str> = sentence.split_whitespace().collect();
    if words.len() <= max_words {
        sentence.to_string()
    } else {
        let mut s = words[..max_words].join(" ");
        s.push_str("...");
        s
    }
}

impl PromptCompressor {
    pub fn new(config: CompressionConfig) -> Self {
        Self { config }
    }

    /// Compress text according to the configured level.
    pub fn compress(&self, text: &str) -> CompressionResult {
        let original_text = text.to_string();
        let original_tokens = estimate_tokens(text);

        // Skip compression for code/json blocks if configured
        let skip = (self.config.preserve_code && is_code_block(text))
            || (self.config.preserve_json && is_json(text));

        let (compressed_text, lossy) = if skip {
            (text.to_string(), false)
        } else {
            match self.config.level {
                CompressionLevel::None => (text.to_string(), false),
                CompressionLevel::Light => {
                    let s = strip_filler_words(text);
                    let s = abbreviate_numbers(&s);
                    (s, false)
                }
                CompressionLevel::Medium => {
                    let s = strip_filler_words(text);
                    let s = abbreviate_numbers(&s);
                    let s = remove_redundancy(&s);
                    let s = shorten_sentences(&s, 50);
                    (s, true)
                }
                CompressionLevel::Aggressive => {
                    let s = strip_filler_words(text);
                    let s = abbreviate_numbers(&s);
                    let s = remove_redundancy(&s);
                    let s = shorten_sentences(&s, 30);
                    // Additional pass: strip filler again
                    let s = strip_filler_words(&s);
                    (s, true)
                }
            }
        };

        let compressed_tokens = estimate_tokens(&compressed_text);
        let compression_ratio = if original_tokens > 0 {
            compressed_tokens as f64 / original_tokens as f64
        } else {
            1.0
        };

        CompressionResult {
            original_text,
            compressed_text,
            original_tokens,
            compressed_tokens,
            compression_ratio,
            lossy,
        }
    }

    /// Iteratively apply increasing compression levels until within token budget.
    pub fn compress_to_budget(&self, text: &str, token_budget: usize) -> CompressionResult {
        let levels = [
            CompressionLevel::None,
            CompressionLevel::Light,
            CompressionLevel::Medium,
            CompressionLevel::Aggressive,
        ];

        for level in &levels {
            let config = CompressionConfig {
                level: level.clone(),
                ..self.config.clone()
            };
            let compressor = PromptCompressor::new(config);
            let result = compressor.compress(text);
            if result.compressed_tokens <= token_budget {
                return result;
            }
        }

        // If still over budget after Aggressive, return Aggressive result anyway
        let config = CompressionConfig {
            level: CompressionLevel::Aggressive,
            ..self.config.clone()
        };
        let compressor = PromptCompressor::new(config);
        compressor.compress(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_filler_removes_words() {
        let text = "This is basically a very simple test.";
        let result = strip_filler_words(text);
        assert!(!result.contains("basically"));
        assert!(!result.contains("very"));
        assert!(result.contains("simple"));
        assert!(result.contains("test"));
    }

    #[test]
    fn test_abbreviate_numbers_works() {
        let text = "There are 1000 items and 2000000 records.";
        let result = abbreviate_numbers(text);
        assert!(result.contains("1K"));
        assert!(result.contains("2M"));
        assert!(!result.contains("1000 "));
    }

    #[test]
    fn test_remove_redundancy_deduplicates() {
        let text = "Hello world. Hello world. This is different.";
        let result = remove_redundancy(text);
        // Should only contain one "Hello world."
        let count = result.matches("Hello world.").count();
        assert_eq!(count, 1);
        assert!(result.contains("This is different."));
    }

    #[test]
    fn test_light_compression_reduces_length() {
        let config = CompressionConfig {
            level: CompressionLevel::Light,
            ..Default::default()
        };
        let compressor = PromptCompressor::new(config);
        let text = "This is basically essentially a very really quite simple test that is just literally straightforward.";
        let result = compressor.compress(text);
        assert!(result.compressed_text.len() < result.original_text.len());
    }

    #[test]
    fn test_compress_to_budget_fits_in_budget() {
        let config = CompressionConfig::default();
        let compressor = PromptCompressor::new(config);
        let text = "This is basically essentially a very really quite simple test. \
                    This is basically essentially a very really quite simple test. \
                    Numbers like 1000000 and 2000000 are big. \
                    Numbers like 1000000 and 2000000 are big.";
        let budget = 20;
        let result = compressor.compress_to_budget(text, budget);
        assert!(result.compressed_tokens <= budget);
    }
}
