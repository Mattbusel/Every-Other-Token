//! # Module: research
//!
//! ## Responsibility
//! Headless research mode: run the token interceptor N times, collect
//! per-run statistics (confidence, perplexity, vocabulary diversity, …),
//! aggregate across runs, and emit a structured JSON report.
//!
//! ## Guarantees
//! - Deterministic output schema (all fields always present).
//! - Non-panicking: all fallible operations return Results.
//! - Writes to the file at `args.output` or to stdout when absent.
//!
//! ## NOT Responsible For
//! - Live web-UI streaming (see: web module).
//! - Provider authentication (delegated to TokenInterceptor).

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tokio::sync::mpsc;

use crate::cli::{resolve_model, Args};
use crate::transforms::Transform;
use crate::TokenEvent;
use crate::TokenInterceptor;

// ---------------------------------------------------------------------------
// Per-run structures
// ---------------------------------------------------------------------------

/// Statistics computed from one research run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunStats {
    /// Total number of tokens produced in this run.
    pub total_tokens: usize,
    /// Number of tokens that were transformed (odd-indexed tokens).
    pub transformed_count: usize,
    /// Mean model confidence across tokens that have a confidence value.
    /// -1.0 when no tokens reported confidence (e.g. Anthropic provider).
    pub avg_confidence: f64,
    /// Mean perplexity across tokens that have a perplexity value.
    /// -1.0 when no tokens reported perplexity.
    pub avg_perplexity: f64,
    /// Vocabulary diversity: unique original token strings / total tokens.
    /// 0.0 for an empty run.
    pub vocabulary_diversity: f64,
    /// Mean character length of original tokens.
    pub avg_token_length: f64,
    /// Estimated cost in USD based on a rough token-count heuristic.
    pub cost_estimate_usd: f64,
}

/// All data produced by a single research run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    /// Zero-based index of this run.
    pub run_index: u32,
    /// The prompt used for this run.
    pub prompt: String,
    /// All token events collected during the run.
    pub tokens: Vec<TokenEvent>,
    /// Derived statistics for this run.
    pub stats: RunStats,
}

// ---------------------------------------------------------------------------
// Aggregate structures
// ---------------------------------------------------------------------------

/// Statistics aggregated across all runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateStats {
    /// Sum of all tokens across every run.
    pub total_tokens: usize,
    /// Mean of per-run avg_confidence values (excluding -1.0 sentinel).
    pub avg_confidence_mean: f64,
    /// Mean of per-run avg_perplexity values (excluding -1.0 sentinel).
    pub avg_perplexity_mean: f64,
    /// Mean of per-run vocabulary_diversity values.
    pub vocab_diversity_mean: f64,
}

/// Top-level JSON envelope written as the research output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchOutput {
    /// The prompt that was used across all runs.
    pub prompt: String,
    /// Provider name ("openai" or "anthropic").
    pub provider: String,
    /// Model identifier.
    pub model: String,
    /// Transform name.
    pub transform: String,
    /// Number of runs completed.
    pub runs: u32,
    /// Per-run detailed results.
    pub runs_data: Vec<RunResult>,
    /// Aggregated statistics across all runs.
    pub aggregate_stats: AggregateStats,
    /// BibTeX citation for the output.
    pub citation: String,
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run headless research mode: collect token events, compute stats, emit JSON.
///
/// # Arguments
/// * `args` — Parsed CLI arguments.
///
/// # Returns
/// `Ok(())` on success; propagates I/O or provider errors.
///
/// # Panics
/// This function never panics.
pub async fn run_research(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let transform = Transform::from_str_loose(&args.transform)
        .map_err(|e| format!("Invalid transform: {}", e))?;

    let model = resolve_model(&args.provider, &args.model);
    let provider_str = args.provider.to_string();

    let mut runs_data: Vec<RunResult> = Vec::with_capacity(args.runs as usize);

    for run_index in 0..args.runs {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();

        let mut interceptor = TokenInterceptor::new(
            args.provider.clone(),
            transform.clone(),
            model.clone(),
            false, // visual_mode
            false, // heatmap_mode
            false, // orchestrator
        )?;
        interceptor.web_tx = Some(tx);
        interceptor.system_prompt = args.system_prompt.clone();

        // Drive the stream; ignore errors so we still write partial results.
        let _ = interceptor.intercept_stream(&args.prompt).await;

        // Drain the channel.
        let mut tokens: Vec<TokenEvent> = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            tokens.push(ev);
        }

        let stats = compute_stats(&tokens, &model);
        runs_data.push(RunResult {
            run_index,
            prompt: args.prompt.clone(),
            tokens,
            stats,
        });
    }

    let aggregate_stats = compute_aggregate(&runs_data);
    let citation = build_citation(
        &args.prompt,
        &provider_str,
        &model,
        &args.transform,
        args.runs,
    );

    let output = ResearchOutput {
        prompt: args.prompt.clone(),
        provider: provider_str,
        model,
        transform: args.transform.clone(),
        runs: args.runs,
        runs_data,
        aggregate_stats,
        citation,
    };

    let json = serde_json::to_string_pretty(&output)?;

    match &args.output {
        Some(path) => std::fs::write(path, &json)?,
        None => println!("{}", json),
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------

/// Compute per-run statistics from a slice of token events.
///
/// # Arguments
/// * `tokens` — Token events from one run.
/// * `model`  — Model name, used to estimate cost.
///
/// # Returns
/// A populated `RunStats`.
///
/// # Panics
/// This function never panics.
pub fn compute_stats(tokens: &[TokenEvent], model: &str) -> RunStats {
    let total_tokens = tokens.len();

    if total_tokens == 0 {
        return RunStats {
            total_tokens: 0,
            transformed_count: 0,
            avg_confidence: -1.0,
            avg_perplexity: -1.0,
            vocabulary_diversity: 0.0,
            avg_token_length: 0.0,
            cost_estimate_usd: 0.0,
        };
    }

    let transformed_count = tokens.iter().filter(|t| t.transformed).count();

    // Confidence: average over tokens that have Some(confidence)
    let conf_values: Vec<f64> = tokens
        .iter()
        .filter_map(|t| t.confidence.map(|c| c as f64))
        .collect();
    let avg_confidence = if conf_values.is_empty() {
        -1.0
    } else {
        conf_values.iter().sum::<f64>() / conf_values.len() as f64
    };

    // Perplexity
    let perp_values: Vec<f64> = tokens
        .iter()
        .filter_map(|t| t.perplexity.map(|p| p as f64))
        .collect();
    let avg_perplexity = if perp_values.is_empty() {
        -1.0
    } else {
        perp_values.iter().sum::<f64>() / perp_values.len() as f64
    };

    // Vocabulary diversity
    let unique_originals: HashSet<&str> =
        tokens.iter().map(|t| t.original.as_str()).collect();
    let vocabulary_diversity = unique_originals.len() as f64 / total_tokens as f64;

    // Average token length (in characters)
    let avg_token_length =
        tokens.iter().map(|t| t.original.len() as f64).sum::<f64>() / total_tokens as f64;

    // Cost estimate: rough heuristic — 1 token ≈ 0.75 words ≈ $0.000002 for GPT-4-class
    let cost_per_token: f64 = if model.contains("gpt-4") {
        0.000_030 // ~$0.03 per 1k tokens output
    } else if model.contains("claude") {
        0.000_015
    } else {
        0.000_002
    };
    let cost_estimate_usd = total_tokens as f64 * cost_per_token;

    RunStats {
        total_tokens,
        transformed_count,
        avg_confidence,
        avg_perplexity,
        vocabulary_diversity,
        avg_token_length,
        cost_estimate_usd,
    }
}

/// Aggregate statistics across multiple runs.
///
/// # Arguments
/// * `runs` — Completed run results.
///
/// # Returns
/// A populated `AggregateStats`.
///
/// # Panics
/// This function never panics.
pub fn compute_aggregate(runs: &[RunResult]) -> AggregateStats {
    let total_tokens = runs.iter().map(|r| r.stats.total_tokens).sum();

    let valid_conf: Vec<f64> = runs
        .iter()
        .map(|r| r.stats.avg_confidence)
        .filter(|&c| c >= 0.0)
        .collect();
    let avg_confidence_mean = if valid_conf.is_empty() {
        -1.0
    } else {
        valid_conf.iter().sum::<f64>() / valid_conf.len() as f64
    };

    let valid_perp: Vec<f64> = runs
        .iter()
        .map(|r| r.stats.avg_perplexity)
        .filter(|&p| p >= 0.0)
        .collect();
    let avg_perplexity_mean = if valid_perp.is_empty() {
        -1.0
    } else {
        valid_perp.iter().sum::<f64>() / valid_perp.len() as f64
    };

    let diversity_values: Vec<f64> =
        runs.iter().map(|r| r.stats.vocabulary_diversity).collect();
    let vocab_diversity_mean = if diversity_values.is_empty() {
        0.0
    } else {
        diversity_values.iter().sum::<f64>() / diversity_values.len() as f64
    };

    AggregateStats {
        total_tokens,
        avg_confidence_mean,
        avg_perplexity_mean,
        vocab_diversity_mean,
    }
}

/// Build a BibTeX citation string for this research run.
///
/// # Arguments
/// * `prompt`    — The prompt used.
/// * `provider`  — Provider name.
/// * `model`     — Model identifier.
/// * `transform` — Transform name.
/// * `runs`      — Number of runs.
///
/// # Returns
/// A BibTeX `@misc` entry as a `String`.
///
/// # Panics
/// This function never panics.
pub fn build_citation(
    prompt: &str,
    provider: &str,
    model: &str,
    transform: &str,
    runs: u32,
) -> String {
    // Truncate prompt to first 40 chars for the title
    let short_prompt: String = prompt.chars().take(40).collect();
    format!(
        "@misc{{every_other_token_{provider}_{model},\n  title = {{Every-Other-Token Research: {short_prompt}}},\n  author = {{Every-Other-Token}},\n  year = {{2026}},\n  note = {{provider={provider}, model={model}, transform={transform}, runs={runs}}},\n  howpublished = {{\\url{{https://github.com/example/every-other-token}}}}\n}}",
        provider = provider,
        model = model,
        short_prompt = short_prompt,
        transform = transform,
        runs = runs,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TokenAlternative;

    fn make_token(
        index: usize,
        transformed: bool,
        original: &str,
        confidence: Option<f32>,
        perplexity: Option<f32>,
    ) -> TokenEvent {
        TokenEvent {
            text: original.to_string(),
            original: original.to_string(),
            index,
            transformed,
            importance: 0.5,
            chaos_label: None,
            provider: None,
            confidence,
            perplexity,
            alternatives: vec![],
        }
    }

    // -- compute_stats --

    #[test]
    fn test_compute_stats_empty_tokens() {
        let stats = compute_stats(&[], "gpt-4");
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.transformed_count, 0);
        assert_eq!(stats.avg_confidence, -1.0);
        assert_eq!(stats.avg_perplexity, -1.0);
        assert_eq!(stats.vocabulary_diversity, 0.0);
        assert_eq!(stats.avg_token_length, 0.0);
        assert_eq!(stats.cost_estimate_usd, 0.0);
    }

    #[test]
    fn test_compute_stats_total_tokens() {
        let tokens = vec![
            make_token(0, false, "hello", None, None),
            make_token(1, true, "world", None, None),
            make_token(2, false, "foo", None, None),
        ];
        let stats = compute_stats(&tokens, "gpt-3.5-turbo");
        assert_eq!(stats.total_tokens, 3);
    }

    #[test]
    fn test_compute_stats_transformed_count() {
        let tokens = vec![
            make_token(0, false, "a", None, None),
            make_token(1, true, "b", None, None),
            make_token(2, false, "c", None, None),
            make_token(3, true, "d", None, None),
        ];
        let stats = compute_stats(&tokens, "gpt-3.5-turbo");
        assert_eq!(stats.transformed_count, 2);
    }

    #[test]
    fn test_compute_stats_avg_confidence_with_values() {
        let tokens = vec![
            make_token(0, false, "a", Some(0.8), None),
            make_token(1, true, "b", Some(0.6), None),
        ];
        let stats = compute_stats(&tokens, "gpt-4");
        // (0.8 + 0.6) / 2 = 0.7
        assert!((stats.avg_confidence - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_compute_stats_avg_confidence_none_sentinel() {
        let tokens = vec![
            make_token(0, false, "a", None, None),
            make_token(1, true, "b", None, None),
        ];
        let stats = compute_stats(&tokens, "gpt-4");
        assert_eq!(stats.avg_confidence, -1.0);
    }

    #[test]
    fn test_compute_stats_avg_perplexity_with_values() {
        let tokens = vec![
            make_token(0, false, "x", None, Some(2.0)),
            make_token(1, true, "y", None, Some(4.0)),
        ];
        let stats = compute_stats(&tokens, "gpt-4");
        assert!((stats.avg_perplexity - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_stats_vocabulary_diversity_all_unique() {
        let tokens = vec![
            make_token(0, false, "alpha", None, None),
            make_token(1, true, "beta", None, None),
            make_token(2, false, "gamma", None, None),
        ];
        let stats = compute_stats(&tokens, "gpt-3.5-turbo");
        assert!((stats.vocabulary_diversity - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_stats_vocabulary_diversity_all_same() {
        let tokens = vec![
            make_token(0, false, "the", None, None),
            make_token(1, true, "the", None, None),
            make_token(2, false, "the", None, None),
        ];
        let stats = compute_stats(&tokens, "gpt-3.5-turbo");
        // 1 unique / 3 total = 0.333...
        assert!(stats.vocabulary_diversity < 0.5);
        assert!(stats.vocabulary_diversity > 0.0);
    }

    #[test]
    fn test_compute_stats_avg_token_length() {
        // "ab" = 2, "cde" = 3 → avg = 2.5
        let tokens = vec![
            make_token(0, false, "ab", None, None),
            make_token(1, true, "cde", None, None),
        ];
        let stats = compute_stats(&tokens, "gpt-3.5-turbo");
        assert!((stats.avg_token_length - 2.5).abs() < 1e-5);
    }

    #[test]
    fn test_compute_stats_cost_estimate_gpt4() {
        let tokens: Vec<TokenEvent> =
            (0..100).map(|i| make_token(i, false, "tok", None, None)).collect();
        let stats = compute_stats(&tokens, "gpt-4");
        // 100 * 0.000_030 = 0.003
        assert!((stats.cost_estimate_usd - 0.003).abs() < 1e-9);
    }

    #[test]
    fn test_compute_stats_cost_estimate_claude() {
        let tokens: Vec<TokenEvent> =
            (0..100).map(|i| make_token(i, false, "tok", None, None)).collect();
        let stats = compute_stats(&tokens, "claude-sonnet-4-20250514");
        // 100 * 0.000_015 = 0.0015
        assert!((stats.cost_estimate_usd - 0.0015).abs() < 1e-9);
    }

    // -- compute_aggregate --

    #[test]
    fn test_compute_aggregate_empty_runs() {
        let agg = compute_aggregate(&[]);
        assert_eq!(agg.total_tokens, 0);
        assert_eq!(agg.avg_confidence_mean, -1.0);
        assert_eq!(agg.avg_perplexity_mean, -1.0);
        assert_eq!(agg.vocab_diversity_mean, 0.0);
    }

    #[test]
    fn test_compute_aggregate_total_tokens() {
        let runs = vec![
            RunResult {
                run_index: 0,
                prompt: "p".to_string(),
                tokens: vec![],
                stats: RunStats {
                    total_tokens: 10,
                    transformed_count: 5,
                    avg_confidence: 0.8,
                    avg_perplexity: 2.0,
                    vocabulary_diversity: 1.0,
                    avg_token_length: 4.0,
                    cost_estimate_usd: 0.001,
                },
            },
            RunResult {
                run_index: 1,
                prompt: "p".to_string(),
                tokens: vec![],
                stats: RunStats {
                    total_tokens: 20,
                    transformed_count: 10,
                    avg_confidence: 0.6,
                    avg_perplexity: 3.0,
                    vocabulary_diversity: 0.5,
                    avg_token_length: 3.0,
                    cost_estimate_usd: 0.002,
                },
            },
        ];
        let agg = compute_aggregate(&runs);
        assert_eq!(agg.total_tokens, 30);
        assert!((agg.avg_confidence_mean - 0.7).abs() < 1e-5);
        assert!((agg.avg_perplexity_mean - 2.5).abs() < 1e-5);
        assert!((agg.vocab_diversity_mean - 0.75).abs() < 1e-5);
    }

    #[test]
    fn test_compute_aggregate_excludes_negative_confidence_sentinel() {
        let make_run = |conf: f64, idx: u32| RunResult {
            run_index: idx,
            prompt: "p".to_string(),
            tokens: vec![],
            stats: RunStats {
                total_tokens: 5,
                transformed_count: 2,
                avg_confidence: conf,
                avg_perplexity: -1.0,
                vocabulary_diversity: 1.0,
                avg_token_length: 3.0,
                cost_estimate_usd: 0.0,
            },
        };
        let runs = vec![make_run(-1.0, 0), make_run(0.9, 1)];
        let agg = compute_aggregate(&runs);
        // Only the 0.9 value is valid; sentinel -1.0 excluded
        assert!((agg.avg_confidence_mean - 0.9).abs() < 1e-5);
    }

    // -- build_citation --

    #[test]
    fn test_build_citation_contains_provider() {
        let c = build_citation("hello", "openai", "gpt-4", "reverse", 3);
        assert!(c.contains("openai"));
    }

    #[test]
    fn test_build_citation_contains_model() {
        let c = build_citation("hello", "openai", "gpt-4", "reverse", 3);
        assert!(c.contains("gpt-4"));
    }

    #[test]
    fn test_build_citation_is_bibtex() {
        let c = build_citation("test prompt", "anthropic", "claude-3", "uppercase", 1);
        assert!(c.starts_with("@misc{"));
    }

    #[test]
    fn test_build_citation_contains_runs() {
        let c = build_citation("test", "openai", "gpt-4", "reverse", 5);
        assert!(c.contains("runs=5"));
    }

    #[test]
    fn test_build_citation_prompt_truncated_at_40() {
        let long_prompt = "a".repeat(100);
        let c = build_citation(&long_prompt, "openai", "gpt-4", "reverse", 1);
        // The truncated prompt in the title should be exactly 40 'a' chars
        assert!(c.contains(&"a".repeat(40)));
        assert!(!c.contains(&"a".repeat(41)));
    }

    // -- RunStats serialization --

    #[test]
    fn test_run_stats_serializes_and_deserializes() {
        let stats = RunStats {
            total_tokens: 42,
            transformed_count: 21,
            avg_confidence: 0.75,
            avg_perplexity: 2.5,
            vocabulary_diversity: 0.9,
            avg_token_length: 4.2,
            cost_estimate_usd: 0.00126,
        };
        let json = serde_json::to_string(&stats).expect("serialize");
        let back: RunStats = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.total_tokens, 42);
        assert!((back.avg_confidence - 0.75).abs() < 1e-5);
    }

    #[test]
    fn test_research_output_serializes() {
        let output = ResearchOutput {
            prompt: "test".to_string(),
            provider: "openai".to_string(),
            model: "gpt-4".to_string(),
            transform: "reverse".to_string(),
            runs: 1,
            runs_data: vec![],
            aggregate_stats: AggregateStats {
                total_tokens: 0,
                avg_confidence_mean: -1.0,
                avg_perplexity_mean: -1.0,
                vocab_diversity_mean: 0.0,
            },
            citation: "@misc{}".to_string(),
        };
        let json = serde_json::to_string(&output).expect("serialize");
        assert!(json.contains("\"provider\":\"openai\""));
        assert!(json.contains("\"model\":\"gpt-4\""));
    }

    #[test]
    fn test_token_alternative_used_in_token_event() {
        let alt = TokenAlternative { token: "hi".to_string(), probability: 0.5 };
        let event = TokenEvent {
            text: "hello".to_string(),
            original: "hello".to_string(),
            index: 0,
            transformed: false,
            importance: 0.5,
            chaos_label: None,
            provider: None,
            confidence: Some(0.9),
            perplexity: Some(1.1),
            alternatives: vec![alt],
        };
        assert_eq!(event.alternatives.len(), 1);
        assert_eq!(event.alternatives[0].token, "hi");
    }
}
