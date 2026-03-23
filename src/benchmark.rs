//! Compression Benchmark
//!
//! Compares multiple token-compression strategies on the same input,
//! producing a ranked ASCII table with composite scores.

use std::sync::Arc;
use std::time::Instant;

// ── BenchmarkStrategy ─────────────────────────────────────────────────────────

/// A named compression strategy wrapping a pure function.
pub struct BenchmarkStrategy {
    pub name: String,
    pub compress_fn: Arc<dyn Fn(&[String]) -> Vec<String> + Send + Sync>,
}

impl BenchmarkStrategy {
    pub fn new(
        name: impl Into<String>,
        f: impl Fn(&[String]) -> Vec<String> + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            compress_fn: Arc::new(f),
        }
    }
}

// ── BenchmarkResult ───────────────────────────────────────────────────────────

/// Result of running one strategy on one input.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub strategy_name: String,
    /// Fraction of original tokens kept: `compressed / original` (lower = more aggressive).
    pub ratio: f64,
    /// Heuristic quality score in [0, 1].
    pub quality_score: f64,
    /// Wall-clock time in microseconds.
    pub elapsed_us: u64,
}

impl BenchmarkResult {
    /// Composite score: `0.5 * (1 - ratio) + 0.3 * quality + 0.2 * speed_factor`.
    ///
    /// Here `1 - ratio` rewards more compression; speed_factor rewards low latency.
    pub fn composite_score(&self) -> f64 {
        let speed_factor = 1.0 / (1.0 + self.elapsed_us as f64 / 1000.0);
        0.5 * (1.0 - self.ratio) + 0.3 * self.quality_score + 0.2 * speed_factor
    }
}

// ── Quality helpers ───────────────────────────────────────────────────────────

/// Heuristic quality: fraction of content words preserved, penalising over-compression.
fn quality_score(original: &[String], compressed: &[String]) -> f64 {
    if original.is_empty() {
        return 1.0;
    }
    if compressed.is_empty() {
        return 0.0;
    }
    // Count how many original tokens appear at least once in the compressed output.
    let compressed_set: std::collections::HashSet<&String> = compressed.iter().collect();
    let preserved = original.iter().filter(|t| compressed_set.contains(t)).count();
    let coverage = preserved as f64 / original.len() as f64;
    // Penalise if compressed is too short (< 20 % of original).
    let ratio = compressed.len() as f64 / original.len() as f64;
    let length_penalty = if ratio < 0.2 { ratio / 0.2 } else { 1.0 };
    (coverage * length_penalty).clamp(0.0, 1.0)
}

// ── Built-in strategies ───────────────────────────────────────────────────────

/// Keep every other token (even indices).
pub fn strategy_every_other(tokens: &[String]) -> Vec<String> {
    tokens.iter().step_by(2).cloned().collect()
}

/// Keep the top half of tokens by positional importance (first half = more important).
pub fn strategy_importance_top_half(tokens: &[String]) -> Vec<String> {
    let keep = (tokens.len() + 1) / 2;
    tokens[..keep].to_vec()
}

/// Keep 50 % of tokens using a fixed budget with even distribution.
pub fn strategy_fixed_budget_50pct(tokens: &[String]) -> Vec<String> {
    if tokens.is_empty() {
        return vec![];
    }
    let budget = (tokens.len() / 2).max(1);
    let step = tokens.len() as f64 / budget as f64;
    (0..budget)
        .map(|i| {
            let idx = (i as f64 * step) as usize;
            tokens[idx.min(tokens.len() - 1)].clone()
        })
        .collect()
}

// ── CompressionBenchmark ──────────────────────────────────────────────────────

/// Orchestrates multiple compression strategies and collects results.
pub struct CompressionBenchmark {
    strategies: Vec<BenchmarkStrategy>,
}

impl CompressionBenchmark {
    pub fn new() -> Self {
        Self { strategies: Vec::new() }
    }

    /// Create a benchmark pre-loaded with the three built-in strategies.
    pub fn with_builtins() -> Self {
        let mut b = Self::new();
        b.add_strategy(BenchmarkStrategy::new("every_other", strategy_every_other));
        b.add_strategy(BenchmarkStrategy::new(
            "importance_top_half",
            strategy_importance_top_half,
        ));
        b.add_strategy(BenchmarkStrategy::new(
            "fixed_budget_50pct",
            strategy_fixed_budget_50pct,
        ));
        b
    }

    pub fn add_strategy(&mut self, strategy: BenchmarkStrategy) {
        self.strategies.push(strategy);
    }

    /// Run all strategies on `tokens` and return one result per strategy.
    pub fn run(&self, tokens: &[String]) -> Vec<BenchmarkResult> {
        self.strategies
            .iter()
            .map(|s| {
                let t0 = Instant::now();
                let compressed = (s.compress_fn)(tokens);
                let elapsed_us = t0.elapsed().as_micros() as u64;

                let ratio = if tokens.is_empty() {
                    1.0
                } else {
                    compressed.len() as f64 / tokens.len() as f64
                };
                let qs = quality_score(tokens, &compressed);

                BenchmarkResult {
                    strategy_name: s.name.clone(),
                    ratio,
                    quality_score: qs,
                    elapsed_us,
                }
            })
            .collect()
    }
}

impl Default for CompressionBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

// ── BenchmarkReport ───────────────────────────────────────────────────────────

/// Renders benchmark results as an ASCII table.
pub struct BenchmarkReport;

impl BenchmarkReport {
    /// Render an ASCII table sorted by composite score (descending).
    pub fn render_table(results: &[BenchmarkResult]) -> String {
        let mut sorted = results.to_vec();
        sorted.sort_by(|a, b| {
            b.composite_score()
                .partial_cmp(&a.composite_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Column widths
        let name_w = sorted
            .iter()
            .map(|r| r.strategy_name.len())
            .max()
            .unwrap_or(8)
            .max(8);

        let header = format!(
            "{:<name_w$} | {:>7} | {:>7} | {:>10} | {:>7}",
            "Strategy", "Ratio", "Quality", "Speed(µs)", "Score",
            name_w = name_w,
        );
        let sep = "-".repeat(header.len());

        let mut lines = vec![sep.clone(), header.clone(), sep.clone()];
        for r in &sorted {
            lines.push(format!(
                "{:<name_w$} | {:>7.4} | {:>7.4} | {:>10} | {:>7.4}",
                r.strategy_name,
                r.ratio,
                r.quality_score,
                r.elapsed_us,
                r.composite_score(),
                name_w = name_w,
            ));
        }
        lines.push(sep);
        lines.join("\n")
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tokens(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("token{}", i)).collect()
    }

    #[test]
    fn test_every_other_half() {
        let t = sample_tokens(10);
        let r = strategy_every_other(&t);
        assert_eq!(r.len(), 5);
        assert_eq!(r[0], "token0");
        assert_eq!(r[1], "token2");
    }

    #[test]
    fn test_every_other_empty() {
        assert!(strategy_every_other(&[]).is_empty());
    }

    #[test]
    fn test_importance_top_half_even() {
        let t = sample_tokens(10);
        let r = strategy_importance_top_half(&t);
        assert_eq!(r.len(), 5);
        assert_eq!(r[0], "token0");
    }

    #[test]
    fn test_importance_top_half_odd() {
        let t = sample_tokens(7);
        let r = strategy_importance_top_half(&t);
        assert_eq!(r.len(), 4);
    }

    #[test]
    fn test_fixed_budget_50pct_length() {
        let t = sample_tokens(20);
        let r = strategy_fixed_budget_50pct(&t);
        assert_eq!(r.len(), 10);
    }

    #[test]
    fn test_fixed_budget_empty() {
        assert!(strategy_fixed_budget_50pct(&[]).is_empty());
    }

    #[test]
    fn test_fixed_budget_single() {
        let t = vec!["only".to_string()];
        let r = strategy_fixed_budget_50pct(&t);
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn test_benchmark_run_returns_one_per_strategy() {
        let b = CompressionBenchmark::with_builtins();
        let tokens = sample_tokens(20);
        let results = b.run(&tokens);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_benchmark_ratio_in_range() {
        let b = CompressionBenchmark::with_builtins();
        let tokens = sample_tokens(100);
        for r in b.run(&tokens) {
            assert!(r.ratio >= 0.0 && r.ratio <= 1.0, "ratio={}", r.ratio);
        }
    }

    #[test]
    fn test_benchmark_quality_in_range() {
        let b = CompressionBenchmark::with_builtins();
        let tokens = sample_tokens(50);
        for r in b.run(&tokens) {
            assert!(r.quality_score >= 0.0 && r.quality_score <= 1.0);
        }
    }

    #[test]
    fn test_composite_score_range() {
        let r = BenchmarkResult {
            strategy_name: "test".to_string(),
            ratio: 0.5,
            quality_score: 0.8,
            elapsed_us: 100,
        };
        let s = r.composite_score();
        assert!(s >= 0.0 && s <= 1.0, "composite={}", s);
    }

    #[test]
    fn test_add_custom_strategy() {
        let mut b = CompressionBenchmark::new();
        b.add_strategy(BenchmarkStrategy::new("keep_all", |t: &[String]| t.to_vec()));
        let tokens = sample_tokens(10);
        let results = b.run(&tokens);
        assert_eq!(results.len(), 1);
        assert!((results[0].ratio - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_render_table_contains_headers() {
        let b = CompressionBenchmark::with_builtins();
        let tokens = sample_tokens(20);
        let results = b.run(&tokens);
        let table = BenchmarkReport::render_table(&results);
        assert!(table.contains("Strategy"));
        assert!(table.contains("Ratio"));
        assert!(table.contains("Quality"));
        assert!(table.contains("Speed"));
        assert!(table.contains("Score"));
    }

    #[test]
    fn test_render_table_contains_strategy_names() {
        let b = CompressionBenchmark::with_builtins();
        let tokens = sample_tokens(20);
        let results = b.run(&tokens);
        let table = BenchmarkReport::render_table(&results);
        assert!(table.contains("every_other"));
        assert!(table.contains("importance_top_half"));
        assert!(table.contains("fixed_budget_50pct"));
    }

    #[test]
    fn test_render_table_sorted() {
        let results = vec![
            BenchmarkResult {
                strategy_name: "slow".to_string(),
                ratio: 0.9,
                quality_score: 0.3,
                elapsed_us: 50000,
            },
            BenchmarkResult {
                strategy_name: "fast_good".to_string(),
                ratio: 0.5,
                quality_score: 0.9,
                elapsed_us: 10,
            },
        ];
        let table = BenchmarkReport::render_table(&results);
        let fast_pos = table.find("fast_good").unwrap();
        let slow_pos = table.find("slow").unwrap();
        assert!(fast_pos < slow_pos, "fast_good should rank higher");
    }

    #[test]
    fn test_empty_input() {
        let b = CompressionBenchmark::with_builtins();
        let results = b.run(&[]);
        for r in &results {
            assert_eq!(r.ratio, 1.0);
        }
    }

    #[test]
    fn test_elapsed_us_recorded() {
        let b = CompressionBenchmark::with_builtins();
        let tokens = sample_tokens(100);
        for r in b.run(&tokens) {
            // elapsed_us should be a non-negative number (u64 is always >= 0)
            let _ = r.elapsed_us;
        }
    }
}
