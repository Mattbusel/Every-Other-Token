//! Vocabulary Analyzer
//!
//! Analyzes vocabulary properties of token sequences — coverage, OOV rate,
//! frequency statistics, and Zipf's law fit.

use std::collections::HashMap;

// ── VocabStats ────────────────────────────────────────────────────────────────

/// Statistics about a token sequence measured against a reference vocabulary.
#[derive(Debug, Clone)]
pub struct VocabStats {
    /// Number of distinct tokens in the analyzed sequence.
    pub size: usize,
    /// Fraction of sequence tokens found in the reference vocabulary ([0.0, 1.0]).
    pub coverage_pct: f64,
    /// Fraction of sequence tokens NOT in the reference vocabulary ([0.0, 1.0]).
    pub oov_rate: f64,
    /// Mean frequency of reference-vocabulary tokens in this sequence.
    pub avg_frequency: f64,
    /// Median frequency of reference-vocabulary tokens in this sequence.
    pub median_frequency: f64,
}

// ── ZipfParams ────────────────────────────────────────────────────────────────

/// Parameters from fitting Zipf's law to a frequency distribution.
///
/// Zipf's law: `freq(r) ∝ 1 / r^exponent`.
/// For natural language, the exponent is typically close to 1.0.
#[derive(Debug, Clone)]
pub struct ZipfParams {
    /// The Zipf exponent (slope of the log-rank vs. log-frequency regression).
    pub exponent: f64,
    /// R² goodness-of-fit (0.0 = no fit, 1.0 = perfect fit).
    pub r_squared: f64,
}

// ── Zipf ─────────────────────────────────────────────────────────────────────

/// Utilities for fitting Zipf's law to a frequency map.
pub struct Zipf;

impl Zipf {
    /// Fit Zipf's law to a frequency map using log-log linear regression.
    ///
    /// Ranks tokens by descending frequency (rank 1 = most frequent).
    /// Performs OLS on `log(rank) ~ log(freq)` and extracts the negative slope
    /// as the Zipf exponent, plus R².
    pub fn fit(freq_map: &HashMap<String, u64>) -> ZipfParams {
        if freq_map.is_empty() {
            return ZipfParams { exponent: 0.0, r_squared: 0.0 };
        }

        // Sort by descending frequency to assign ranks.
        let mut freqs: Vec<u64> = freq_map.values().copied().collect();
        freqs.sort_unstable_by(|a, b| b.cmp(a));

        // Build (log_rank, log_freq) pairs for all entries with freq > 0.
        let points: Vec<(f64, f64)> = freqs
            .iter()
            .enumerate()
            .filter(|(_, &f)| f > 0)
            .map(|(i, &f)| ((i as f64 + 1.0).ln(), (f as f64).ln()))
            .collect();

        let n = points.len();
        if n < 2 {
            return ZipfParams { exponent: 0.0, r_squared: 0.0 };
        }

        // OLS: regress log_freq (y) on log_rank (x).
        // y = a + b * x  → b = Cov(x,y)/Var(x)
        let n_f = n as f64;
        let mean_x = points.iter().map(|(x, _)| x).sum::<f64>() / n_f;
        let mean_y = points.iter().map(|(_, y)| y).sum::<f64>() / n_f;

        let cov_xy: f64 = points.iter().map(|(x, y)| (x - mean_x) * (y - mean_y)).sum();
        let var_x: f64 = points.iter().map(|(x, _)| (x - mean_x).powi(2)).sum();

        if var_x.abs() < f64::EPSILON {
            return ZipfParams { exponent: 0.0, r_squared: 0.0 };
        }

        let slope = cov_xy / var_x;
        // Zipf exponent is the negative slope (freq decreases as rank increases).
        let exponent = -slope;

        // R² = 1 - SS_res / SS_tot
        let intercept = mean_y - slope * mean_x;
        let ss_res: f64 = points
            .iter()
            .map(|(x, y)| {
                let y_hat = intercept + slope * x;
                (y - y_hat).powi(2)
            })
            .sum();
        let ss_tot: f64 = points.iter().map(|(_, y)| (y - mean_y).powi(2)).sum();
        let r_squared = if ss_tot < f64::EPSILON {
            1.0
        } else {
            (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
        };

        ZipfParams { exponent, r_squared }
    }
}

// ── VocabularyAnalyzer ────────────────────────────────────────────────────────

/// Analyzes vocabulary properties of token sequences against a reference corpus.
pub struct VocabularyAnalyzer {
    /// Reference vocabulary with per-token frequencies.
    vocab: HashMap<String, u64>,
}

impl VocabularyAnalyzer {
    /// Build a `VocabularyAnalyzer` from a corpus of tokenized documents.
    ///
    /// Each document is a `Vec<String>` of tokens. Frequencies are summed
    /// across all documents.
    pub fn build_from_corpus(docs: &[Vec<String>]) -> Self {
        let mut vocab: HashMap<String, u64> = HashMap::new();
        for doc in docs {
            for token in doc {
                *vocab.entry(token.clone()).or_insert(0) += 1;
            }
        }
        Self { vocab }
    }

    /// Analyze a token sequence, measuring coverage against the reference vocabulary.
    pub fn analyze(&self, tokens: &[String]) -> VocabStats {
        if tokens.is_empty() {
            return VocabStats {
                size: 0,
                coverage_pct: 1.0,
                oov_rate: 0.0,
                avg_frequency: 0.0,
                median_frequency: 0.0,
            };
        }

        let total = tokens.len() as f64;
        let in_vocab_count = tokens.iter().filter(|t| self.vocab.contains_key(*t)).count();
        let coverage_pct = in_vocab_count as f64 / total;
        let oov_rate = 1.0 - coverage_pct;

        // Count distinct tokens in this sequence.
        let distinct: std::collections::HashSet<&str> = tokens.iter().map(|s| s.as_str()).collect();
        let size = distinct.len();

        // Frequency of each token in the sequence (not the reference vocab).
        let mut seq_freq: HashMap<&str, u64> = HashMap::new();
        for t in tokens {
            *seq_freq.entry(t.as_str()).or_insert(0) += 1;
        }

        // For in-vocab tokens, collect their reference frequencies.
        let in_vocab_freqs: Vec<f64> = tokens
            .iter()
            .filter_map(|t| self.vocab.get(t).map(|&f| f as f64))
            .collect();

        let avg_frequency = if in_vocab_freqs.is_empty() {
            0.0
        } else {
            in_vocab_freqs.iter().sum::<f64>() / in_vocab_freqs.len() as f64
        };

        let median_frequency = median_f64(&in_vocab_freqs);

        VocabStats {
            size,
            coverage_pct,
            oov_rate,
            avg_frequency,
            median_frequency,
        }
    }

    /// Return all tokens in `tokens` that are NOT in the reference vocabulary (Out-Of-Vocabulary).
    ///
    /// Preserves order; duplicates are preserved.
    pub fn oov_tokens<'a>(&self, tokens: &'a [String]) -> Vec<&'a String> {
        tokens
            .iter()
            .filter(|t| !self.vocab.contains_key(*t))
            .collect()
    }

    /// Return OOV tokens as owned Strings.
    pub fn oov_tokens_owned(&self, tokens: &[String]) -> Vec<String> {
        tokens
            .iter()
            .filter(|t| !self.vocab.contains_key(*t))
            .cloned()
            .collect()
    }

    /// Compute the Zipf score for the reference vocabulary.
    ///
    /// Returns a value in [0.0, 1.0]; 1.0 means the distribution has a Zipf
    /// exponent of exactly 1.0 (perfect natural-language distribution).
    pub fn zipf_score(&self) -> f64 {
        let params = Zipf::fit(&self.vocab);
        (1.0 - (params.exponent - 1.0).abs()).max(0.0).min(1.0)
    }

    /// Expose the reference vocabulary.
    pub fn vocab(&self) -> &HashMap<String, u64> {
        &self.vocab
    }

    /// Number of distinct token types in the reference vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn median_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(s: &[&str]) -> Vec<String> {
        s.iter().map(|t| t.to_string()).collect()
    }

    fn corpus(docs: &[&[&str]]) -> Vec<Vec<String>> {
        docs.iter().map(|d| toks(d)).collect()
    }

    // 1. build_from_corpus creates vocab with correct frequency
    #[test]
    fn test_build_corpus_frequency() {
        let docs = corpus(&[&["a", "b", "a"]]);
        let va = VocabularyAnalyzer::build_from_corpus(&docs);
        assert_eq!(*va.vocab().get("a").unwrap(), 2);
        assert_eq!(*va.vocab().get("b").unwrap(), 1);
    }

    // 2. vocab_size
    #[test]
    fn test_vocab_size() {
        let docs = corpus(&[&["a", "b", "c"]]);
        let va = VocabularyAnalyzer::build_from_corpus(&docs);
        assert_eq!(va.vocab_size(), 3);
    }

    // 3. analyze - full coverage
    #[test]
    fn test_analyze_full_coverage() {
        let docs = corpus(&[&["a", "b", "c"]]);
        let va = VocabularyAnalyzer::build_from_corpus(&docs);
        let stats = va.analyze(&toks(&["a", "b"]));
        assert!((stats.coverage_pct - 1.0).abs() < 1e-9);
        assert!((stats.oov_rate).abs() < 1e-9);
    }

    // 4. analyze - zero coverage
    #[test]
    fn test_analyze_zero_coverage() {
        let docs = corpus(&[&["a", "b"]]);
        let va = VocabularyAnalyzer::build_from_corpus(&docs);
        let stats = va.analyze(&toks(&["x", "y", "z"]));
        assert!((stats.coverage_pct).abs() < 1e-9);
        assert!((stats.oov_rate - 1.0).abs() < 1e-9);
    }

    // 5. analyze - partial coverage
    #[test]
    fn test_analyze_partial_coverage() {
        let docs = corpus(&[&["a", "b"]]);
        let va = VocabularyAnalyzer::build_from_corpus(&docs);
        let stats = va.analyze(&toks(&["a", "z"]));
        assert!((stats.coverage_pct - 0.5).abs() < 1e-9);
        assert!((stats.oov_rate - 0.5).abs() < 1e-9);
    }

    // 6. analyze empty tokens
    #[test]
    fn test_analyze_empty_tokens() {
        let docs = corpus(&[&["a"]]);
        let va = VocabularyAnalyzer::build_from_corpus(&docs);
        let stats = va.analyze(&[]);
        assert_eq!(stats.size, 0);
    }

    // 7. size counts distinct tokens
    #[test]
    fn test_analyze_size_distinct() {
        let docs = corpus(&[&["a", "b", "c"]]);
        let va = VocabularyAnalyzer::build_from_corpus(&docs);
        let stats = va.analyze(&toks(&["a", "a", "b"]));
        assert_eq!(stats.size, 2);
    }

    // 8. oov_tokens returns correct tokens
    #[test]
    fn test_oov_tokens() {
        let docs = corpus(&[&["a", "b"]]);
        let va = VocabularyAnalyzer::build_from_corpus(&docs);
        let tokens = toks(&["a", "x", "y"]);
        let oov = va.oov_tokens_owned(&tokens);
        assert_eq!(oov, vec!["x", "y"]);
    }

    // 9. oov_tokens empty when all in vocab
    #[test]
    fn test_oov_tokens_empty() {
        let docs = corpus(&[&["a", "b"]]);
        let va = VocabularyAnalyzer::build_from_corpus(&docs);
        let tokens = toks(&["a", "b"]);
        assert!(va.oov_tokens_owned(&tokens).is_empty());
    }

    // 10. oov preserves duplicates
    #[test]
    fn test_oov_preserves_duplicates() {
        let docs = corpus(&[&["a"]]);
        let va = VocabularyAnalyzer::build_from_corpus(&docs);
        let tokens = toks(&["z", "z"]);
        assert_eq!(va.oov_tokens_owned(&tokens).len(), 2);
    }

    // 11. avg_frequency for single in-vocab token
    #[test]
    fn test_avg_frequency_single() {
        let docs = corpus(&[&["a", "a", "a"]]);
        let va = VocabularyAnalyzer::build_from_corpus(&docs);
        let stats = va.analyze(&toks(&["a"]));
        assert!((stats.avg_frequency - 3.0).abs() < 1e-9);
    }

    // 12. median_frequency for even count
    #[test]
    fn test_median_even() {
        let vals = vec![1.0, 3.0, 5.0, 7.0];
        assert!((median_f64(&vals) - 4.0).abs() < 1e-9);
    }

    // 13. median_frequency for odd count
    #[test]
    fn test_median_odd() {
        let vals = vec![1.0, 3.0, 5.0];
        assert!((median_f64(&vals) - 3.0).abs() < 1e-9);
    }

    // 14. Zipf fit on empty map returns 0
    #[test]
    fn test_zipf_empty() {
        let freq_map: HashMap<String, u64> = HashMap::new();
        let p = Zipf::fit(&freq_map);
        assert_eq!(p.exponent, 0.0);
        assert_eq!(p.r_squared, 0.0);
    }

    // 15. Zipf fit on single entry returns 0
    #[test]
    fn test_zipf_single_entry() {
        let mut m = HashMap::new();
        m.insert("a".to_string(), 10u64);
        let p = Zipf::fit(&m);
        assert_eq!(p.exponent, 0.0);
    }

    // 16. Zipf fit on perfect Zipf distribution exponent near 1.0
    #[test]
    fn test_zipf_fit_natural_language() {
        // Generate freq(r) = C / r for r=1..50 — perfect Zipf-1
        let mut freq_map: HashMap<String, u64> = HashMap::new();
        for r in 1u64..=50 {
            freq_map.insert(format!("tok{}", r), 10000 / r);
        }
        let p = Zipf::fit(&freq_map);
        // Exponent should be close to 1.0 for this distribution
        assert!((p.exponent - 1.0).abs() < 0.2, "exponent={}", p.exponent);
        assert!(p.r_squared > 0.95, "r_squared={}", p.r_squared);
    }

    // 17. zipf_score returns value in [0, 1]
    #[test]
    fn test_zipf_score_range() {
        let docs = corpus(&[&["a", "b", "b", "c", "c", "c"]]);
        let va = VocabularyAnalyzer::build_from_corpus(&docs);
        let score = va.zipf_score();
        assert!(score >= 0.0 && score <= 1.0);
    }

    // 18. zipf_score near 1 for good natural distribution
    #[test]
    fn test_zipf_score_near_one() {
        // Build a corpus with Zipf-like freq distribution
        let mut freq_map: HashMap<String, u64> = HashMap::new();
        for r in 1u64..=100 {
            freq_map.insert(format!("t{}", r), 10000 / r);
        }
        let va = VocabularyAnalyzer { vocab: freq_map };
        let score = va.zipf_score();
        assert!(score > 0.8, "score={}", score);
    }

    // 19. build from empty corpus gives empty vocab
    #[test]
    fn test_build_from_empty_corpus() {
        let va = VocabularyAnalyzer::build_from_corpus(&[]);
        assert_eq!(va.vocab_size(), 0);
    }

    // 20. multi-document corpus sums frequencies
    #[test]
    fn test_multi_doc_corpus() {
        let docs = corpus(&[&["a", "b"], &["a", "c"]]);
        let va = VocabularyAnalyzer::build_from_corpus(&docs);
        assert_eq!(*va.vocab().get("a").unwrap(), 2);
        assert_eq!(*va.vocab().get("b").unwrap(), 1);
        assert_eq!(*va.vocab().get("c").unwrap(), 1);
    }

    // 21. coverage_pct + oov_rate == 1.0
    #[test]
    fn test_coverage_plus_oov_is_one() {
        let docs = corpus(&[&["a", "b", "c"]]);
        let va = VocabularyAnalyzer::build_from_corpus(&docs);
        let stats = va.analyze(&toks(&["a", "x", "b", "y"]));
        assert!((stats.coverage_pct + stats.oov_rate - 1.0).abs() < 1e-9);
    }
}
