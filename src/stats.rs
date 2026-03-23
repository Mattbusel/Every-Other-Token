//! Token Statistics Analyzer
//!
//! Provides frequency analysis, sequence statistics, n-gram counting,
//! and repetition detection over token sequences.

use std::collections::HashMap;

// ── FrequencyMap ─────────────────────────────────────────────────────────────

/// Frequency map over token strings.
#[derive(Debug, Clone, Default)]
pub struct FrequencyMap {
    pub counts: HashMap<String, u64>,
    pub total: u64,
}

impl FrequencyMap {
    /// Build a frequency map from a token slice.
    pub fn from_tokens(tokens: &[String]) -> Self {
        let mut counts = HashMap::new();
        for t in tokens {
            *counts.entry(t.clone()).or_insert(0u64) += 1;
        }
        let total = tokens.len() as u64;
        Self { counts, total }
    }

    /// Return the top `n` tokens sorted by descending count (ties broken alphabetically).
    pub fn top_n(&self, n: usize) -> Vec<(String, u64)> {
        let mut pairs: Vec<(String, u64)> = self.counts.iter().map(|(k, v)| (k.clone(), *v)).collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        pairs.truncate(n);
        pairs
    }

    /// Shannon entropy: `-sum(p * log2(p))` over the distribution of tokens.
    pub fn entropy(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let total = self.total as f64;
        self.counts
            .values()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / total;
                -p * p.log2()
            })
            .sum()
    }
}

// ── SequenceStats ─────────────────────────────────────────────────────────────

/// Aggregate statistics for a token sequence.
#[derive(Debug, Clone)]
pub struct SequenceStats {
    /// Total number of tokens.
    pub length: usize,
    /// Number of distinct token types.
    pub unique_tokens: usize,
    /// `unique_tokens / length` — vocabulary richness in [0, 1].
    pub vocab_richness: f64,
    /// Mean character length of tokens.
    pub avg_token_len: f64,
    /// Type-token ratio: same as `vocab_richness`.
    pub type_token_ratio: f64,
    /// Number of tokens that appear exactly once (hapax legomena).
    pub hapax_legomena: usize,
}

impl SequenceStats {
    /// Compute statistics from a token slice.
    pub fn compute(tokens: &[String]) -> Self {
        let length = tokens.len();
        if length == 0 {
            return Self {
                length: 0,
                unique_tokens: 0,
                vocab_richness: 0.0,
                avg_token_len: 0.0,
                type_token_ratio: 0.0,
                hapax_legomena: 0,
            };
        }
        let freq = FrequencyMap::from_tokens(tokens);
        let unique_tokens = freq.counts.len();
        let vocab_richness = unique_tokens as f64 / length as f64;
        let avg_token_len =
            tokens.iter().map(|t| t.chars().count()).sum::<usize>() as f64 / length as f64;
        let type_token_ratio = vocab_richness;
        let hapax_legomena = freq.counts.values().filter(|&&c| c == 1).count();
        Self {
            length,
            unique_tokens,
            vocab_richness,
            avg_token_len,
            type_token_ratio,
            hapax_legomena,
        }
    }
}

// ── NgramAnalyzer ─────────────────────────────────────────────────────────────

/// N-gram frequency analyzer.
pub struct NgramAnalyzer {
    pub n: usize,
}

impl NgramAnalyzer {
    pub fn new(n: usize) -> Self {
        Self { n }
    }

    /// Count all n-grams in `tokens`.
    pub fn compute(&self, tokens: &[String]) -> HashMap<Vec<String>, u64> {
        let mut counts: HashMap<Vec<String>, u64> = HashMap::new();
        if tokens.len() < self.n {
            return counts;
        }
        for window in tokens.windows(self.n) {
            *counts.entry(window.to_vec()).or_insert(0) += 1;
        }
        counts
    }

    /// Return top `n` most-frequent n-grams, sorted by descending count.
    pub fn top_ngrams(tokens: &[String], n: usize) -> Vec<(Vec<String>, u64)> {
        // Default window size 2; callers who want a different size should use `compute`.
        let analyzer = NgramAnalyzer::new(2);
        let counts = analyzer.compute(tokens);
        let mut pairs: Vec<(Vec<String>, u64)> =
            counts.into_iter().map(|(k, v)| (k, v)).collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        pairs.truncate(n);
        pairs
    }

    /// Top n-grams with configurable window size.
    pub fn top_ngrams_n(tokens: &[String], window: usize, n: usize) -> Vec<(Vec<String>, u64)> {
        let analyzer = NgramAnalyzer::new(window);
        let counts = analyzer.compute(tokens);
        let mut pairs: Vec<(Vec<String>, u64)> = counts.into_iter().collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        pairs.truncate(n);
        pairs
    }
}

// ── RepetitionDetector ────────────────────────────────────────────────────────

/// A repeated sub-sequence found in the token stream.
#[derive(Debug, Clone, PartialEq)]
pub struct RepetitionSpan {
    /// The repeated token pattern.
    pub pattern: Vec<String>,
    /// Start positions (indices into the original token slice) of each occurrence.
    pub positions: Vec<usize>,
    /// Number of occurrences.
    pub count: usize,
}

/// Finds repeated sub-sequences via a sliding window approach.
pub struct RepetitionDetector;

impl RepetitionDetector {
    /// Find all sub-sequences of length `min_len` or more that appear more than once.
    ///
    /// The search iterates window lengths from `min_len` upward until no new
    /// repetitions are found, then returns all discovered spans.
    pub fn find(tokens: &[String], min_len: usize) -> Vec<RepetitionSpan> {
        let min_len = min_len.max(1);
        let mut results: Vec<RepetitionSpan> = Vec::new();

        // Try all window sizes from min_len up to half the sequence length.
        let max_len = tokens.len() / 2;
        if max_len < min_len {
            return results;
        }

        for window_size in min_len..=max_len {
            let mut map: HashMap<Vec<String>, Vec<usize>> = HashMap::new();
            for (i, window) in tokens.windows(window_size).enumerate() {
                map.entry(window.to_vec()).or_default().push(i);
            }
            for (pattern, positions) in map {
                if positions.len() >= 2 {
                    let count = positions.len();
                    results.push(RepetitionSpan { pattern, positions, count });
                }
            }
        }

        // De-duplicate: if pattern P is fully contained in a longer repeated pattern Q
        // at the exact same positions, prefer Q. Simple approach: just sort by pattern
        // length descending so callers see the most informative spans first.
        results.sort_by(|a, b| b.pattern.len().cmp(&a.pattern.len()).then_with(|| b.count.cmp(&a.count)));
        results
    }
}

// ── TokenStats (top-level façade) ─────────────────────────────────────────────

/// Comprehensive statistics bundle for a token sequence.
pub struct TokenStats {
    pub sequence: SequenceStats,
    pub frequency: FrequencyMap,
}

impl TokenStats {
    pub fn compute(tokens: &[String]) -> Self {
        Self {
            sequence: SequenceStats::compute(tokens),
            frequency: FrequencyMap::from_tokens(tokens),
        }
    }

    /// Print a human-readable summary to stdout.
    pub fn print_summary(&self) {
        println!("=== Token Statistics ===");
        println!("  length          : {}", self.sequence.length);
        println!("  unique tokens   : {}", self.sequence.unique_tokens);
        println!("  vocab richness  : {:.4}", self.sequence.vocab_richness);
        println!("  avg token len   : {:.2}", self.sequence.avg_token_len);
        println!("  type-token ratio: {:.4}", self.sequence.type_token_ratio);
        println!("  hapax legomena  : {}", self.sequence.hapax_legomena);
        println!("  entropy (bits)  : {:.4}", self.frequency.entropy());
        println!("  top-5 tokens    :");
        for (tok, cnt) in self.frequency.top_n(5) {
            println!("    {:20} {}", tok, cnt);
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(s: &str) -> Vec<String> {
        s.split_whitespace().map(|t| t.to_string()).collect()
    }

    // FrequencyMap tests
    #[test]
    fn test_freq_empty() {
        let f = FrequencyMap::from_tokens(&[]);
        assert_eq!(f.total, 0);
        assert!(f.counts.is_empty());
    }

    #[test]
    fn test_freq_counts() {
        let tokens = toks("a b a c a b");
        let f = FrequencyMap::from_tokens(&tokens);
        assert_eq!(f.counts["a"], 3);
        assert_eq!(f.counts["b"], 2);
        assert_eq!(f.counts["c"], 1);
        assert_eq!(f.total, 6);
    }

    #[test]
    fn test_freq_top_n() {
        let tokens = toks("a b a c a b");
        let f = FrequencyMap::from_tokens(&tokens);
        let top = f.top_n(2);
        assert_eq!(top[0], ("a".to_string(), 3));
        assert_eq!(top[1], ("b".to_string(), 2));
    }

    #[test]
    fn test_freq_top_n_more_than_vocab() {
        let tokens = toks("x y");
        let f = FrequencyMap::from_tokens(&tokens);
        // Requesting 10 from a vocab of 2 should return 2
        assert_eq!(f.top_n(10).len(), 2);
    }

    #[test]
    fn test_entropy_uniform() {
        // 4 tokens each appearing once → entropy = log2(4) = 2.0
        let tokens = toks("a b c d");
        let f = FrequencyMap::from_tokens(&tokens);
        let e = f.entropy();
        assert!((e - 2.0).abs() < 1e-9, "entropy={}", e);
    }

    #[test]
    fn test_entropy_single_token() {
        let tokens: Vec<String> = vec!["a".to_string(); 10];
        let f = FrequencyMap::from_tokens(&tokens);
        assert_eq!(f.entropy(), 0.0);
    }

    #[test]
    fn test_entropy_empty() {
        let f = FrequencyMap::from_tokens(&[]);
        assert_eq!(f.entropy(), 0.0);
    }

    // SequenceStats tests
    #[test]
    fn test_seq_stats_empty() {
        let s = SequenceStats::compute(&[]);
        assert_eq!(s.length, 0);
        assert_eq!(s.unique_tokens, 0);
    }

    #[test]
    fn test_seq_stats_basic() {
        let tokens = toks("the quick brown fox the fox");
        let s = SequenceStats::compute(&tokens);
        assert_eq!(s.length, 6);
        assert_eq!(s.unique_tokens, 4);
        assert!((s.vocab_richness - 4.0 / 6.0).abs() < 1e-9);
        assert_eq!(s.hapax_legomena, 2); // quick, brown
    }

    #[test]
    fn test_seq_avg_token_len() {
        // "ab cd" → avg = (2+2)/2 = 2.0
        let tokens = toks("ab cd");
        let s = SequenceStats::compute(&tokens);
        assert!((s.avg_token_len - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_type_token_ratio() {
        let tokens = toks("a a a a");
        let s = SequenceStats::compute(&tokens);
        // 1 unique / 4 total = 0.25
        assert!((s.type_token_ratio - 0.25).abs() < 1e-9);
    }

    // NgramAnalyzer tests
    #[test]
    fn test_ngram_bigrams() {
        let tokens = toks("a b c a b");
        let analyzer = NgramAnalyzer::new(2);
        let counts = analyzer.compute(&tokens);
        assert_eq!(counts[&vec!["a".to_string(), "b".to_string()]], 2);
        assert_eq!(counts[&vec!["b".to_string(), "c".to_string()]], 1);
    }

    #[test]
    fn test_ngram_trigrams() {
        let tokens = toks("a b c a b c");
        let analyzer = NgramAnalyzer::new(3);
        let counts = analyzer.compute(&tokens);
        assert_eq!(
            counts[&vec!["a".to_string(), "b".to_string(), "c".to_string()]],
            2
        );
    }

    #[test]
    fn test_ngram_too_short() {
        let tokens = toks("a b");
        let analyzer = NgramAnalyzer::new(5);
        assert!(analyzer.compute(&tokens).is_empty());
    }

    #[test]
    fn test_top_ngrams() {
        let tokens = toks("a b a b c d");
        let top = NgramAnalyzer::top_ngrams(&tokens, 1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].1, 2);
    }

    // RepetitionDetector tests
    #[test]
    fn test_repetition_basic() {
        let tokens = toks("a b c a b c");
        let spans = RepetitionDetector::find(&tokens, 3);
        let found = spans.iter().any(|s| {
            s.pattern == vec!["a".to_string(), "b".to_string(), "c".to_string()]
                && s.count >= 2
        });
        assert!(found, "expected 'a b c' repetition");
    }

    #[test]
    fn test_repetition_no_repeat() {
        let tokens = toks("a b c d e f");
        let spans = RepetitionDetector::find(&tokens, 3);
        assert!(spans.is_empty());
    }

    #[test]
    fn test_repetition_min_len() {
        let tokens = toks("x y x y x y");
        // With min_len=2 we should find "x y" repeated 3 times
        let spans = RepetitionDetector::find(&tokens, 2);
        let found = spans.iter().any(|s| {
            s.pattern == vec!["x".to_string(), "y".to_string()] && s.count >= 2
        });
        assert!(found);
    }

    #[test]
    fn test_repetition_count() {
        let tokens = toks("a b a b a b");
        let spans = RepetitionDetector::find(&tokens, 2);
        let pair_span = spans
            .iter()
            .find(|s| s.pattern == vec!["a".to_string(), "b".to_string()]);
        assert!(pair_span.is_some());
        assert!(pair_span.unwrap().count >= 2);
    }

    #[test]
    fn test_repetition_empty() {
        assert!(RepetitionDetector::find(&[], 3).is_empty());
    }

    // TokenStats tests
    #[test]
    fn test_token_stats_smoke() {
        let tokens = toks("hello world hello rust world");
        let stats = TokenStats::compute(&tokens);
        assert_eq!(stats.sequence.length, 5);
        assert!(stats.frequency.entropy() > 0.0);
    }
}
