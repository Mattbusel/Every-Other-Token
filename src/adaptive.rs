//! Quality-adaptive compression for token sequences.
//!
//! This module measures compression quality along three dimensions and provides
//! an `AdaptiveCompressor` that performs binary search over aggressiveness levels
//! to find the best compression ratio that still meets a quality target.
//!
//! # Quality model
//!
//! ```text
//! overall_score = 0.4 * semantic_preservation
//!               + 0.4 * syntax_validity
//!               + 0.2 * (1 - ratio)
//! ```
//!
//! Higher `overall_score` is better. `ratio` is `compressed_len / original_len`,
//! so a ratio of 0 (empty output) contributes maximally to the last term, but
//! would score 0 on the other two — ensuring a balanced optimum.

// ── Quality types ─────────────────────────────────────────────────────────────

/// Measured quality of a compressed token sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct CompressionQuality {
    /// Fraction of content words (non-stopwords) preserved. Range 0..=1.
    pub semantic_preservation: f64,
    /// Heuristic syntax validity score. Range 0..=1.
    pub syntax_validity: f64,
    /// Compression ratio (`compressed_len / original_len`). Lower means more compressed.
    pub ratio: f64,
    /// Weighted overall score (higher is better).
    pub overall_score: f64,
}

impl CompressionQuality {
    /// Compute the weighted overall score from components.
    pub fn compute_overall(semantic_preservation: f64, syntax_validity: f64, ratio: f64) -> f64 {
        0.4 * semantic_preservation + 0.4 * syntax_validity + 0.2 * (1.0 - ratio)
    }
}

// ── Compression target ────────────────────────────────────────────────────────

/// Target parameters for adaptive compression.
#[derive(Debug, Clone)]
pub struct CompressionTarget {
    /// Minimum acceptable `overall_score`. Compression stops if it would go below this.
    pub min_quality: f64,
    /// Desired ratio (compressed_len / original_len). Lower = more aggressive.
    pub target_ratio: f64,
}

impl Default for CompressionTarget {
    fn default() -> Self {
        Self {
            min_quality: 0.5,
            target_ratio: 0.5,
        }
    }
}

// ── Stopword list ─────────────────────────────────────────────────────────────

/// Common English stopwords used to distinguish content words from function words.
static STOPWORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "shall", "can", "it", "its", "this", "that", "these", "those",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "our", "their", "not", "no", "nor", "so", "yet", "both",
    "either", "neither", "each", "every", "all", "any", "few", "more", "most",
    "such", "than", "then", "when", "where", "which", "who", "whom", "how",
    "if", "as", "up", "out", "about", "into", "through", "during", "before",
    "after", "above", "below", "between", "there", "here", "now",
];

/// Punctuation tokens that should not appear consecutively.
static PUNCTUATION_TOKENS: &[&str] = &[
    ".", ",", "!", "?", ";", ":", ")", "]", "}", "(", "[", "{", "\"", "'",
    "-", "--", "...", "…",
];

// ── Quality metric ────────────────────────────────────────────────────────────

/// Measures the quality of a compressed token sequence relative to the original.
pub struct QualityMetric;

impl QualityMetric {
    /// Measure [`CompressionQuality`] for a compressed sequence.
    ///
    /// # Parameters
    /// - `original`: the uncompressed token sequence
    /// - `compressed`: the compressed token sequence
    pub fn measure(original: &[String], compressed: &[String]) -> CompressionQuality {
        let ratio = if original.is_empty() {
            1.0
        } else {
            compressed.len() as f64 / original.len() as f64
        };

        let semantic_preservation = Self::semantic_preservation(original, compressed);
        let syntax_validity = Self::syntax_validity(compressed);
        let overall_score =
            CompressionQuality::compute_overall(semantic_preservation, syntax_validity, ratio);

        CompressionQuality {
            semantic_preservation,
            syntax_validity,
            ratio,
            overall_score,
        }
    }

    /// Fraction of content words (non-stopwords) in `original` that appear in `compressed`.
    fn semantic_preservation(original: &[String], compressed: &[String]) -> f64 {
        let content_words: Vec<&str> = original
            .iter()
            .map(|t| t.as_str())
            .filter(|t| !Self::is_stopword(t) && !Self::is_punctuation(t))
            .collect();

        if content_words.is_empty() {
            return 1.0; // nothing to preserve
        }

        let preserved = content_words
            .iter()
            .filter(|&&w| {
                compressed
                    .iter()
                    .any(|c| c.eq_ignore_ascii_case(w))
            })
            .count();

        preserved as f64 / content_words.len() as f64
    }

    /// Heuristic: penalise consecutive punctuation tokens in the compressed sequence.
    fn syntax_validity(compressed: &[String]) -> f64 {
        if compressed.is_empty() {
            return 1.0;
        }
        let mut consecutive_punct = 0usize;
        let mut prev_was_punct = false;
        let mut violations = 0usize;

        for token in compressed {
            let is_punct = Self::is_punctuation(token.as_str());
            if is_punct && prev_was_punct {
                violations += 1;
            }
            if is_punct {
                consecutive_punct += 1;
            }
            prev_was_punct = is_punct;
        }

        let _ = consecutive_punct;
        let penalty = violations as f64 * 0.1;
        (1.0 - penalty).max(0.0)
    }

    fn is_stopword(token: &str) -> bool {
        let lower = token.to_lowercase();
        STOPWORDS.contains(&lower.as_str())
    }

    fn is_punctuation(token: &str) -> bool {
        PUNCTUATION_TOKENS.contains(&token)
            || (token.len() == 1 && !token.chars().next().map_or(false, |c| c.is_alphanumeric()))
    }
}

// ── Aggressiveness levels ─────────────────────────────────────────────────────

/// Compress tokens at a given aggressiveness level (0.0 = no compression, 1.0 = maximum).
///
/// At level `a`, only tokens whose index satisfies `(i as f64) % (1.0 / a).round() != 0`
/// are dropped. Level 0.0 returns all tokens; level 1.0 returns only the first token.
fn compress_at_level(tokens: &[String], level: f64) -> Vec<String> {
    if tokens.is_empty() || level <= 0.0 {
        return tokens.to_vec();
    }
    if level >= 1.0 {
        return tokens.first().map(|t| vec![t.clone()]).unwrap_or_default();
    }

    // Keep every k-th token where k = round(1 / (1 - level))
    // level=0.5 → keep every 2nd → ratio 0.5
    // level=0.33 → keep every 1.5th ≈ 2 → ratio 0.5  (clamped)
    // level=0.75 → keep every 4th → ratio 0.25
    let keep_every = (1.0 / (1.0 - level)).round().max(1.0) as usize;
    tokens
        .iter()
        .enumerate()
        .filter(|(i, _)| i % keep_every == 0)
        .map(|(_, t)| t.clone())
        .collect()
}

// ── AdaptiveCompressor ────────────────────────────────────────────────────────

/// Adjusts compression aggressiveness based on quality feedback.
///
/// Uses binary search over aggressiveness levels to find the most aggressive
/// compression that still meets the target quality.
pub struct AdaptiveCompressor {
    /// Number of binary-search iterations (precision).
    pub search_iterations: usize,
}

impl AdaptiveCompressor {
    /// Create a new `AdaptiveCompressor`.
    pub fn new() -> Self {
        Self {
            search_iterations: 16,
        }
    }

    /// Create with custom binary-search precision.
    pub fn with_iterations(mut self, n: usize) -> Self {
        self.search_iterations = n.max(1);
        self
    }

    /// Compress `tokens` to meet `target.min_quality` while maximising compression.
    ///
    /// Returns the compressed sequence and its measured quality.
    pub fn compress(
        &self,
        tokens: &[String],
        target: &CompressionTarget,
    ) -> (Vec<String>, CompressionQuality) {
        if tokens.is_empty() {
            let quality = QualityMetric::measure(tokens, tokens);
            return (tokens.to_vec(), quality);
        }

        // Binary search: lo = low aggressiveness (keep most), hi = high aggressiveness
        let mut lo = 0.0_f64;
        let mut hi = 0.99_f64;
        let mut best_compressed = tokens.to_vec();
        let mut best_quality = QualityMetric::measure(tokens, tokens);

        for _ in 0..self.search_iterations {
            let mid = (lo + hi) / 2.0;
            let candidate = compress_at_level(tokens, mid);
            let quality = QualityMetric::measure(tokens, &candidate);

            if quality.overall_score >= target.min_quality {
                // This level is acceptable — try harder (more aggressive)
                best_compressed = candidate;
                best_quality = quality;
                lo = mid;
            } else {
                // Too aggressive — back off
                hi = mid;
            }
        }

        // If we couldn't compress at all without going below min_quality, return original
        if best_quality.overall_score < target.min_quality {
            best_compressed = tokens.to_vec();
            best_quality = QualityMetric::measure(tokens, tokens);
        }

        (best_compressed, best_quality)
    }
}

impl Default for AdaptiveCompressor {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(s: &str) -> Vec<String> {
        s.split_whitespace().map(|t| t.to_string()).collect()
    }

    #[test]
    fn test_measure_no_compression() {
        let original = toks("the quick brown fox");
        let q = QualityMetric::measure(&original, &original);
        assert!((q.ratio - 1.0).abs() < 1e-9);
        assert!((q.overall_score - CompressionQuality::compute_overall(
            q.semantic_preservation, q.syntax_validity, 1.0
        )).abs() < 1e-9);
    }

    #[test]
    fn test_measure_empty_original() {
        let q = QualityMetric::measure(&[], &[]);
        assert!((q.ratio - 1.0).abs() < 1e-9);
        assert_eq!(q.semantic_preservation, 1.0);
        assert_eq!(q.syntax_validity, 1.0);
    }

    #[test]
    fn test_semantic_preservation_all_stopwords() {
        let original = toks("the and is a");
        let compressed = toks("the");
        let q = QualityMetric::measure(&original, &compressed);
        // All stopwords → nothing to preserve → semantic_preservation = 1.0
        assert_eq!(q.semantic_preservation, 1.0);
    }

    #[test]
    fn test_semantic_preservation_content_words() {
        let original = toks("cat sat mat");
        let compressed = toks("cat mat");
        let q = QualityMetric::measure(&original, &compressed);
        // 2 of 3 content words preserved
        assert!((q.semantic_preservation - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_syntax_validity_no_consecutive_punct() {
        let compressed = toks("hello world");
        let q = QualityMetric::measure(&toks("hello world foo"), &compressed);
        assert_eq!(q.syntax_validity, 1.0);
    }

    #[test]
    fn test_syntax_validity_consecutive_punct_penalised() {
        // Manually construct consecutive punct
        let original = vec!["hello".to_string(), ".".to_string(), ",".to_string(), "world".to_string()];
        let compressed = vec![".".to_string(), ",".to_string()];
        let q = QualityMetric::measure(&original, &compressed);
        assert!(q.syntax_validity < 1.0);
    }

    #[test]
    fn test_overall_score_formula() {
        let sp = 0.8;
        let sv = 0.9;
        let ratio = 0.5;
        let expected = 0.4 * sp + 0.4 * sv + 0.2 * (1.0 - ratio);
        assert!((CompressionQuality::compute_overall(sp, sv, ratio) - expected).abs() < 1e-12);
    }

    #[test]
    fn test_compress_at_level_zero() {
        let tokens = toks("a b c d e f");
        let result = compress_at_level(&tokens, 0.0);
        assert_eq!(result, tokens);
    }

    #[test]
    fn test_compress_at_level_half() {
        let tokens = toks("a b c d");
        let result = compress_at_level(&tokens, 0.5);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_compress_at_level_max() {
        let tokens = toks("a b c d");
        let result = compress_at_level(&tokens, 1.0);
        assert_eq!(result, vec!["a"]);
    }

    #[test]
    fn test_compress_at_level_empty() {
        let result = compress_at_level(&[], 0.5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_adaptive_compressor_no_compression_needed() {
        let ac = AdaptiveCompressor::new();
        let tokens = toks("the quick brown fox jumps over the lazy dog");
        let target = CompressionTarget {
            min_quality: 0.99, // very high bar — should return near-original
            target_ratio: 0.5,
        };
        let (compressed, quality) = ac.compress(&tokens, &target);
        // Should keep most tokens to meet quality
        assert!(compressed.len() >= tokens.len() / 2);
        assert!(quality.overall_score >= 0.0);
    }

    #[test]
    fn test_adaptive_compressor_empty_input() {
        let ac = AdaptiveCompressor::new();
        let (compressed, quality) = ac.compress(&[], &CompressionTarget::default());
        assert!(compressed.is_empty());
        assert_eq!(quality.ratio, 1.0);
    }

    #[test]
    fn test_adaptive_compressor_meets_quality() {
        let ac = AdaptiveCompressor::new();
        let tokens = toks("machine learning is transforming the field of computer science");
        let target = CompressionTarget {
            min_quality: 0.3,
            target_ratio: 0.5,
        };
        let (_, quality) = ac.compress(&tokens, &target);
        assert!(quality.overall_score >= target.min_quality);
    }

    #[test]
    fn test_adaptive_compressor_low_min_quality_allows_high_compression() {
        let ac = AdaptiveCompressor::new();
        let tokens = toks("a b c d e f g h i j k l m n o p");
        let target = CompressionTarget {
            min_quality: 0.01,
            target_ratio: 0.1,
        };
        let (compressed, _) = ac.compress(&tokens, &target);
        // Low quality bar means we can compress aggressively
        assert!(compressed.len() < tokens.len());
    }

    #[test]
    fn test_adaptive_compressor_default() {
        let ac = AdaptiveCompressor::default();
        assert_eq!(ac.search_iterations, 16);
    }

    #[test]
    fn test_adaptive_compressor_with_iterations() {
        let ac = AdaptiveCompressor::new().with_iterations(8);
        assert_eq!(ac.search_iterations, 8);
    }

    #[test]
    fn test_compression_target_default() {
        let t = CompressionTarget::default();
        assert_eq!(t.min_quality, 0.5);
        assert_eq!(t.target_ratio, 0.5);
    }

    #[test]
    fn test_is_stopword() {
        assert!(QualityMetric::is_stopword("the"));
        assert!(QualityMetric::is_stopword("THE"));
        assert!(!QualityMetric::is_stopword("machine"));
    }

    #[test]
    fn test_is_punctuation() {
        assert!(QualityMetric::is_punctuation("."));
        assert!(QualityMetric::is_punctuation(","));
        assert!(!QualityMetric::is_punctuation("hello"));
    }

    #[test]
    fn test_quality_metric_preserve_content() {
        // "science" and "art" are content words
        let original = toks("science and art");
        let compressed = toks("science art");
        let q = QualityMetric::measure(&original, &compressed);
        // Both content words preserved
        assert_eq!(q.semantic_preservation, 1.0);
    }

    #[test]
    fn test_overall_score_range() {
        let original = toks("hello world this is a test");
        let compressed = toks("hello world");
        let q = QualityMetric::measure(&original, &compressed);
        assert!((0.0..=1.0).contains(&q.overall_score));
        assert!((0.0..=1.0).contains(&q.semantic_preservation));
        assert!((0.0..=1.0).contains(&q.syntax_validity));
    }
}
