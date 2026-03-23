//! # Processing Pipeline Combinator
//!
//! Provides composable token transformation pipelines via the [`Stage`] and
//! [`Pipeline`] abstractions. Built-in stages cover filtering, mapping,
//! deduplication, normalisation, windowing, stopword removal, and more.
//!
//! ## Example
//!
//! ```rust
//! use every_other_token::pipeline::{Pipeline, FilterStage, NormalizeStage, LimitStage};
//!
//! let tokens = vec!["Hello".to_string(), "world".to_string(), "!".to_string()];
//! let result = Pipeline::new()
//!     .then(NormalizeStage)
//!     .then(FilterStage { predicate: Box::new(|t: &str| !t.is_empty()) })
//!     .then(LimitStage { max_tokens: 10 })
//!     .run(tokens);
//! ```

use std::collections::HashSet;

// ── Stage traits ──────────────────────────────────────────────────────────────

/// A synchronous processing stage that transforms an input into an output.
pub trait Stage<I, O>: Send + Sync {
    fn process(&self, input: I) -> O;
}

/// An asynchronous processing stage that transforms an input into an output.
pub trait AsyncStage<I, O>: Send + Sync {
    fn process<'a>(
        &'a self,
        input: I,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = O> + Send + 'a>>
    where
        I: 'a;
}

// ── Pipeline ──────────────────────────────────────────────────────────────────

type BoxedStage = Box<dyn Stage<Vec<String>, Vec<String>>>;

/// A sequential chain of [`Stage`]s that each operate on `Vec<String>` tokens.
pub struct Pipeline {
    stages: Vec<BoxedStage>,
}

impl Pipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Append a stage and return `self` for chaining.
    pub fn then<S: Stage<Vec<String>, Vec<String>> + 'static>(mut self, stage: S) -> Self {
        self.stages.push(Box::new(stage));
        self
    }

    /// Apply all stages in sequence and return the final token list.
    pub fn run(&self, tokens: Vec<String>) -> Vec<String> {
        self.stages.iter().fold(tokens, |acc, stage| stage.process(acc))
    }

    /// Apply all stages and return both the result and a [`PipelineStats`].
    pub fn run_with_stats(&self, tokens: Vec<String>) -> (Vec<String>, PipelineStats) {
        let input_count = tokens.len();
        let stage_count = self.stages.len();
        let output = self.run(tokens);
        let output_count = output.len();
        let reduction_ratio = if input_count == 0 {
            0.0
        } else {
            1.0 - (output_count as f64 / input_count as f64)
        };
        (
            output,
            PipelineStats { input_count, output_count, reduction_ratio, stage_count },
        )
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ── PipelineStats ─────────────────────────────────────────────────────────────

/// Statistics collected during a pipeline run.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineStats {
    pub input_count: usize,
    pub output_count: usize,
    /// Fraction of tokens removed (0.0 = nothing removed, 1.0 = all removed).
    pub reduction_ratio: f64,
    pub stage_count: usize,
}

// ── Built-in stages ───────────────────────────────────────────────────────────

/// Remove tokens that do not satisfy `predicate`.
pub struct FilterStage {
    pub predicate: Box<dyn Fn(&str) -> bool + Send + Sync>,
}

impl Stage<Vec<String>, Vec<String>> for FilterStage {
    fn process(&self, input: Vec<String>) -> Vec<String> {
        input.into_iter().filter(|t| (self.predicate)(t.as_str())).collect()
    }
}

/// Transform each token with `transform`.
pub struct MapStage {
    pub transform: Box<dyn Fn(String) -> String + Send + Sync>,
}

impl Stage<Vec<String>, Vec<String>> for MapStage {
    fn process(&self, input: Vec<String>) -> Vec<String> {
        input.into_iter().map(|t| (self.transform)(t)).collect()
    }
}

/// Remove *consecutive* duplicate tokens (analogous to Unix `uniq`).
pub struct DeduplicateStage;

impl Stage<Vec<String>, Vec<String>> for DeduplicateStage {
    fn process(&self, input: Vec<String>) -> Vec<String> {
        let mut out: Vec<String> = Vec::with_capacity(input.len());
        for token in input {
            if out.last().map_or(true, |last| last != &token) {
                out.push(token);
            }
        }
        out
    }
}

/// Lowercase every token and strip ASCII punctuation characters.
pub struct NormalizeStage;

impl Stage<Vec<String>, Vec<String>> for NormalizeStage {
    fn process(&self, input: Vec<String>) -> Vec<String> {
        input
            .into_iter()
            .map(|t| {
                t.to_lowercase()
                    .chars()
                    .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                    .collect()
            })
            .collect()
    }
}

/// Truncate the token list to at most `max_tokens` entries.
pub struct LimitStage {
    pub max_tokens: usize,
}

impl Stage<Vec<String>, Vec<String>> for LimitStage {
    fn process(&self, mut input: Vec<String>) -> Vec<String> {
        input.truncate(self.max_tokens);
        input
    }
}

/// Drop the first `skip` tokens.
pub struct OffsetStage {
    pub skip: usize,
}

impl Stage<Vec<String>, Vec<String>> for OffsetStage {
    fn process(&self, input: Vec<String>) -> Vec<String> {
        if self.skip >= input.len() {
            Vec::new()
        } else {
            input[self.skip..].to_vec()
        }
    }
}

/// Sliding-window grouping: collect windows of `size` tokens stepping by
/// `step`, join each window with a space, then split the result back on
/// whitespace so the output is still a flat `Vec<String>`.
pub struct WindowStage {
    pub size: usize,
    pub step: usize,
}

impl Stage<Vec<String>, Vec<String>> for WindowStage {
    fn process(&self, input: Vec<String>) -> Vec<String> {
        if self.size == 0 || self.step == 0 {
            return input;
        }
        let mut out = Vec::new();
        let mut i = 0;
        while i + self.size <= input.len() {
            let window = input[i..i + self.size].join(" ");
            // Re-emit each word from the joined window as its own token.
            out.extend(window.split_whitespace().map(|s| s.to_string()));
            i += self.step;
        }
        out
    }
}

/// Remove tokens that appear in `stopwords`.
pub struct StopwordStage {
    pub stopwords: HashSet<String>,
}

impl Stage<Vec<String>, Vec<String>> for StopwordStage {
    fn process(&self, input: Vec<String>) -> Vec<String> {
        input.into_iter().filter(|t| !self.stopwords.contains(t.as_str())).collect()
    }
}

// ── ParallelPipeline ──────────────────────────────────────────────────────────

/// Strategy used when merging the outputs of multiple independent pipelines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Append all outputs in pipeline order.
    Concatenate,
    /// Interleave outputs round-robin.
    Interleave,
    /// Union: concatenate then remove duplicates (preserving first occurrence).
    Union,
    /// Intersection: keep only tokens that appear in *every* pipeline's output.
    Intersection,
}

/// Run multiple independent [`Pipeline`]s on the same input and merge results.
pub struct ParallelPipeline {
    pipelines: Vec<Pipeline>,
    pub merge_strategy: MergeStrategy,
}

impl ParallelPipeline {
    pub fn new(merge_strategy: MergeStrategy) -> Self {
        Self { pipelines: Vec::new(), merge_strategy }
    }

    pub fn add_pipeline(mut self, pipeline: Pipeline) -> Self {
        self.pipelines.push(pipeline);
        self
    }

    pub fn run(&self, tokens: Vec<String>) -> Vec<String> {
        let results: Vec<Vec<String>> =
            self.pipelines.iter().map(|p| p.run(tokens.clone())).collect();

        match self.merge_strategy {
            MergeStrategy::Concatenate => results.into_iter().flatten().collect(),
            MergeStrategy::Interleave => {
                let max_len = results.iter().map(|r| r.len()).max().unwrap_or(0);
                let mut out = Vec::new();
                for i in 0..max_len {
                    for r in &results {
                        if let Some(t) = r.get(i) {
                            out.push(t.clone());
                        }
                    }
                }
                out
            }
            MergeStrategy::Union => {
                let mut seen = HashSet::new();
                let mut out = Vec::new();
                for token in results.into_iter().flatten() {
                    if seen.insert(token.clone()) {
                        out.push(token);
                    }
                }
                out
            }
            MergeStrategy::Intersection => {
                if results.is_empty() {
                    return Vec::new();
                }
                // Build multisets per pipeline, then keep tokens whose count
                // in every pipeline is > 0.
                let sets: Vec<HashSet<&String>> =
                    results.iter().map(|r| r.iter().collect()).collect();
                let first = &results[0];
                let mut seen_out = HashSet::new();
                let mut out = Vec::new();
                for token in first {
                    if seen_out.contains(token) {
                        continue;
                    }
                    if sets.iter().all(|s| s.contains(token)) {
                        seen_out.insert(token.clone());
                        out.push(token.clone());
                    }
                }
                out
            }
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn filter_removes_correct_tokens() {
        let stage = FilterStage { predicate: Box::new(|t: &str| t.len() > 3) };
        let input = tokens(&["hi", "hello", "world", "go"]);
        let out = stage.process(input);
        assert_eq!(out, tokens(&["hello", "world"]));
    }

    #[test]
    fn deduplicate_collapses_consecutive_runs() {
        let stage = DeduplicateStage;
        let input = tokens(&["a", "a", "b", "b", "b", "c", "a"]);
        // "a" reappears non-consecutively so it is kept both times.
        let out = stage.process(input);
        assert_eq!(out, tokens(&["a", "b", "c", "a"]));
    }

    #[test]
    fn pipeline_chain_applies_in_order() {
        let result = Pipeline::new()
            .then(NormalizeStage)
            .then(FilterStage { predicate: Box::new(|t: &str| !t.is_empty()) })
            .then(LimitStage { max_tokens: 2 })
            .run(tokens(&["Hello!", "WORLD", "foo", "bar"]));
        assert_eq!(result, tokens(&["hello", "world"]));
    }

    #[test]
    fn limit_truncates_to_max() {
        let stage = LimitStage { max_tokens: 3 };
        let input = tokens(&["a", "b", "c", "d", "e"]);
        let out = stage.process(input);
        assert_eq!(out, tokens(&["a", "b", "c"]));
    }

    #[test]
    fn offset_skips_tokens() {
        let stage = OffsetStage { skip: 2 };
        let input = tokens(&["a", "b", "c", "d"]);
        let out = stage.process(input);
        assert_eq!(out, tokens(&["c", "d"]));
    }

    #[test]
    fn parallel_union_deduplicates() {
        let p1 = Pipeline::new()
            .then(FilterStage { predicate: Box::new(|t: &str| t.starts_with('a')) });
        let p2 = Pipeline::new()
            .then(FilterStage { predicate: Box::new(|t: &str| t.starts_with('a') || t.starts_with('b')) });

        let pp = ParallelPipeline::new(MergeStrategy::Union)
            .add_pipeline(p1)
            .add_pipeline(p2);

        let input = tokens(&["apple", "banana", "avocado", "cherry"]);
        let out = pp.run(input);

        // Unique: apple, avocado from both + banana from p2 — no duplicates.
        assert!(out.contains(&"apple".to_string()));
        assert!(out.contains(&"avocado".to_string()));
        assert!(out.contains(&"banana".to_string()));
        // Check no duplicates.
        let unique: HashSet<_> = out.iter().collect();
        assert_eq!(unique.len(), out.len());
    }

    #[test]
    fn run_with_stats_reports_correct_counts() {
        let (out, stats) = Pipeline::new()
            .then(LimitStage { max_tokens: 3 })
            .run_with_stats(tokens(&["a", "b", "c", "d", "e"]));
        assert_eq!(stats.input_count, 5);
        assert_eq!(stats.output_count, 3);
        assert_eq!(stats.stage_count, 1);
        assert!((stats.reduction_ratio - 0.4).abs() < 1e-9);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn stopword_stage_removes_stopwords() {
        let stopwords: HashSet<String> =
            ["the", "a", "is"].iter().map(|s| s.to_string()).collect();
        let stage = StopwordStage { stopwords };
        let input = tokens(&["the", "cat", "is", "a", "bat"]);
        let out = stage.process(input);
        assert_eq!(out, tokens(&["cat", "bat"]));
    }

    #[test]
    fn window_stage_produces_grouped_tokens() {
        let stage = WindowStage { size: 2, step: 1 };
        let input = tokens(&["a", "b", "c"]);
        let out = stage.process(input);
        // Windows: [a b], [b c] → joined and split back
        assert_eq!(out, tokens(&["a", "b", "b", "c"]));
    }
}
