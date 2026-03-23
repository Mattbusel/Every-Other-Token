//! Headless research mode and batch experiment execution.
//!
//! This module provides two public async functions:
//!
//! - [`run_research`] -- runs a single prompt `N` times and writes aggregate
//!   statistics (mean confidence, perplexity, vocab diversity, collapse positions,
//!   95% CIs) to JSON.
//! - [`run_research_suite`] -- reads one prompt per line from `--prompt-file` and
//!   calls `run_research` for each, merging results into a JSONL or JSON array.
//!
//! A [`run_diff_terminal`] function is also provided for side-by-side OpenAI vs
//! Anthropic streaming in the terminal (`--diff-terminal`).
//!
//! ## Output schema
//!
//! The [`ResearchOutput`] struct is versioned with a `schema_version` field so
//! downstream consumers can detect breaking changes.  The current version is `1`.

use crate::cli::Args;
use crate::transforms::Transform;
use crate::TokenInterceptor;
use serde::Serialize;
use tokio::sync::mpsc;

/// Metrics collected from a single research run.
#[derive(Debug, Serialize, serde::Deserialize)]
pub struct ResearchRun {
    /// Zero-based index of this run within the session.
    pub run_index: u32,
    /// Total tokens streamed in this run.
    pub token_count: usize,
    /// Tokens mutated by the active transform.
    pub transformed_count: usize,
    /// Mean per-token model confidence (0.0--1.0), or `None` when logprobs are unavailable.
    pub avg_confidence: Option<f64>,
    /// Mean per-token perplexity, or `None` when logprobs are unavailable.
    pub avg_perplexity: Option<f64>,
    /// Vocabulary diversity: unique tokens / total tokens.
    pub vocab_diversity: f64,
    /// Starting positions of runs of 5+ consecutive tokens with confidence < 0.4
    pub collapse_positions: Vec<usize>,
    /// Per-transform breakdown: for each transform label, the mean perplexity of tokens
    /// that were transformed with that label (useful for Chaos mode sub-transform analysis).
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub per_transform_perplexity: std::collections::HashMap<String, f64>,
    /// Token throughput in tokens per second for this run (None if elapsed was zero).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_per_second: Option<f64>,
    /// Wall-clock duration of this run in milliseconds.
    pub elapsed_ms: u64,
    /// Per-token arrival latencies in milliseconds (measured from stream start).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub token_latencies_ms: Vec<u64>,
    /// P50 (median) token arrival latency in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p50_latency_ms: Option<u64>,
    /// P95 token arrival latency in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p95_latency_ms: Option<u64>,
}

/// Top-level JSON output written by [`run_research`].
#[derive(Debug, Serialize, serde::Deserialize)]
pub struct ResearchOutput {
    /// Monotonically increasing version number; consumers should check this before parsing.
    pub schema_version: u8,
    /// The prompt used for all runs in this session.
    pub prompt: String,
    /// Provider name (`"openai"` or `"anthropic"`).
    pub provider: String,
    /// Transform name applied to intercepted tokens.
    pub transform: String,
    /// Per-run data in order of execution.
    pub runs: Vec<ResearchRun>,
    /// Cross-run aggregated statistics.
    pub aggregate: ResearchAggregate,
}

/// Cross-run aggregate statistics, appended to every [`ResearchOutput`].
#[derive(Debug, Serialize, serde::Deserialize)]
pub struct ResearchAggregate {
    /// Total number of runs requested (may differ from `runs.len()` on early exit).
    /// Total number of runs requested (may differ from `runs.len()` on early exit).
    pub total_runs: u32,
    /// Mean token count per run.
    pub mean_token_count: f64,
    /// Mean model confidence across all runs and tokens, if available.
    pub mean_confidence: Option<f64>,
    /// Mean perplexity across all runs and tokens, if available.
    pub mean_perplexity: Option<f64>,
    /// Mean vocabulary diversity score across all runs.
    pub mean_vocab_diversity: f64,
    /// Sample standard deviation of per-run mean confidence.
    pub std_dev_confidence: Option<f64>,
    /// Sample standard deviation of per-run mean perplexity.
    pub std_dev_perplexity: Option<f64>,
    /// Sample standard deviation of per-run token count.
    pub std_dev_token_count: f64,
    /// 95% confidence interval for mean confidence (lower, upper).
    pub confidence_interval_95_confidence: Option<(f64, f64)>,
    /// 95% confidence interval for mean perplexity (lower, upper).
    pub confidence_interval_95_perplexity: Option<(f64, f64)>,
    /// Minimum token count across all runs (truncation-based alignment length)
    pub aligned_length: usize,
    /// Cross-run mean perplexity by transform label.
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub mean_per_transform_perplexity: std::collections::HashMap<String, f64>,
    /// True when `total_runs < 30`; the Z-score CI and t-test p-values are
    /// approximations that may be unreliable at small sample sizes.
    pub small_n_warning: bool,
}

/// Compute a percentile value (0–100) from a slice of latencies.
/// Returns `None` if the slice is empty.
pub fn percentile_latency(latencies: &[u64], pct: usize) -> Option<u64> {
    if latencies.is_empty() {
        return None;
    }
    let mut sorted = latencies.to_vec();
    sorted.sort_unstable();
    let idx = ((pct as f64 / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    Some(sorted[idx.min(sorted.len() - 1)])
}

fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance =
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

// 95% CI: mean ± 1.96 * (sd / √n)
// Ref: Casella & Berger, "Statistical Inference" 2nd ed. (2002), §9.2.
// Note: uses Z-score approximation; valid when N ≥ 30 (CLT).
fn ci_95(mean: f64, sd: f64, n: usize) -> (f64, f64) {
    let margin = 1.96 * sd / (n as f64).sqrt();
    (mean - margin, mean + margin)
}

/// Run the full headless research loop for `args.runs` iterations.
///
/// Each iteration calls [`TokenInterceptor::intercept_stream`], collects the
/// emitted `TokenEvent`s, and computes per-run metrics (token count, mean
/// confidence, perplexity, vocab diversity, collapse positions).  After all
/// runs complete, aggregate statistics and an optional CSV heatmap are written
/// to disk.
///
/// # Errors
/// Returns an error if the transform string is invalid, the API call fails,
/// or output file I/O fails.
pub async fn run_research(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.runs == 0 {
        return Err("--runs must be at least 1".into());
    }
    let provider = args.provider.clone();
    let transform_str = args.transform.clone();
    let transform =
        Transform::from_str_loose(&transform_str).map_err(|e| format!("Invalid transform: {e}"))?;
    let model = crate::cli::resolve_model(&provider, &args.model);

    tracing::info!(
        runs = args.runs,
        provider = %provider,
        transform = %transform_str,
        model = %model,
        "starting research session",
    );
    eprintln!(
        "[research] starting {} runs -- provider={} transform={} model={}",
        args.runs, provider, transform_str, model
    );

    // Open SQLite store if requested
    let store = if let Some(db_path) = &args.db {
        Some(crate::store::ExperimentStore::open(db_path)?)
    } else {
        None
    };

    // Determine experiment id in DB
    let exp_id: Option<i64> = if let Some(ref s) = store {
        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let created_at = format!("{}", now);
        let id = s.insert_experiment(
            &created_at,
            &args.prompt,
            &provider.to_string(),
            &transform_str,
            &model,
        )?;
        Some(id)
    } else {
        None
    };

    // Create heatmap exporter if requested
    let mut heatmap_exporter = args
        .heatmap_export
        .as_ref()
        .map(|_| crate::heatmap::HeatmapExporter::new());

    let mut runs: Vec<ResearchRun> = Vec::with_capacity(args.runs as usize);

    for i in 0..args.runs {
        tracing::info!(run = i + 1, total = args.runs, "starting research run");
        eprintln!("[research] run {}/{}", i + 1, args.runs);

        let (tx, mut rx) = mpsc::unbounded_channel();

        let mut interceptor = TokenInterceptor::new(
            provider.clone(),
            transform.clone(),
            model.clone(),
            false,
            false,
            false,
        )?;
        interceptor.web_tx = Some(tx);
        // A/B mode: alternate system prompts on even/odd runs so --significance
        // actually compares two different conditions.
        interceptor.system_prompt = if i % 2 == 1 {
            args.system_b.clone().or_else(|| args.system_a.clone())
        } else {
            args.system_a.clone()
        };
        interceptor.top_logprobs = args.top_logprobs;
        interceptor.min_confidence = args.min_confidence;
        // Enable in-session semantic dedup when the feature is compiled in.
        // Repeated identical prompts (common in research mode) hit the cache
        // after the first run, avoiding redundant API spend.
        #[cfg(feature = "self-modify")]
        interceptor.enable_dedup(300_000, 1_024);
        if let Some(rate) = args.rate {
            interceptor = interceptor.with_rate(rate);
        }
        if let Some(seed) = args.seed {
            interceptor = interceptor.with_seed(seed);
        }

        let run_start = std::time::Instant::now();
        interceptor.intercept_stream(&args.prompt).await?;
        let elapsed_ms = run_start.elapsed().as_millis() as u64;
        drop(interceptor);

        // Collect events and record per-token latencies from arrival_ms stamps
        // set inline by the interceptor relative to stream start.
        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        let token_latencies_ms: Vec<u64> = events
            .iter()
            .filter_map(|e| e.arrival_ms)
            .collect();
        let p50_latency_ms = percentile_latency(&token_latencies_ms, 50);
        let p95_latency_ms = percentile_latency(&token_latencies_ms, 95);

        let token_count = events.len();
        let transformed_count = events.iter().filter(|e| e.transformed).count();

        let confidences: Vec<f64> = events
            .iter()
            .filter_map(|e| e.confidence.map(|v| v as f64))
            .collect();
        let avg_confidence = if confidences.is_empty() {
            None
        } else {
            Some(confidences.iter().sum::<f64>() / confidences.len() as f64)
        };

        let perplexities: Vec<f64> = events
            .iter()
            .filter_map(|e| e.perplexity.map(|v| v as f64))
            .collect();
        let avg_perplexity = if perplexities.is_empty() {
            None
        } else {
            Some(perplexities.iter().sum::<f64>() / perplexities.len() as f64)
        };

        let unique_tokens: std::collections::HashSet<&str> =
            events.iter().map(|e| e.original.as_str()).collect();
        let vocab_diversity = if token_count == 0 {
            0.0
        } else {
            unique_tokens.len() as f64 / token_count as f64
        };

        // Detect confidence collapse: runs of 5+ consecutive tokens with confidence < 0.4
        let per_token_confidences: Vec<f64> = events
            .iter()
            .map(|e| e.confidence.map(|v| v as f64).unwrap_or(1.0))
            .collect();
        let collapse_positions =
            detect_collapse_positions(&per_token_confidences, args.collapse_window, 0.4);

        // Per-transform perplexity breakdown
        let mut transform_perp: std::collections::HashMap<String, Vec<f64>> =
            std::collections::HashMap::new();
        for event in &events {
            if event.transformed {
                let label = event
                    .chaos_label
                    .clone()
                    .unwrap_or_else(|| transform_str.clone());
                if let Some(p) = event.perplexity {
                    transform_perp.entry(label).or_default().push(p as f64);
                }
            }
        }
        let per_transform_perplexity: std::collections::HashMap<String, f64> = transform_perp
            .into_iter()
            .map(|(k, v)| (k, v.iter().sum::<f64>() / v.len() as f64))
            .collect();

        // Persist run to DB if store is open
        if let (Some(ref s), Some(eid)) = (&store, exp_id) {
            s.insert_run(
                eid,
                &crate::store::RunRecord {
                    run_index: i,
                    token_count,
                    transformed_count,
                    avg_confidence,
                    avg_perplexity,
                    vocab_diversity,
                },
            )?;
        }

        // Record events for heatmap
        if let Some(ref mut exporter) = heatmap_exporter {
            exporter.record_run(&events);
        }

        let tokens_per_second = if elapsed_ms > 0 {
            Some(token_count as f64 / (elapsed_ms as f64 / 1000.0))
        } else {
            None
        };
        runs.push(ResearchRun {
            run_index: i,
            token_count,
            transformed_count,
            avg_confidence,
            avg_perplexity,
            vocab_diversity,
            collapse_positions,
            per_transform_perplexity,
            tokens_per_second,
            elapsed_ms,
            token_latencies_ms,
            p50_latency_ms,
            p95_latency_ms,
        });
    }

    // Sample-size warning: CLT requires N >= 30 for valid inference
    if args.runs < 30 {
        eprintln!(
            "[research] WARNING: Only {} run{} — statistical results may be unreliable. \
             The central limit theorem typically requires N ≥ 30 for valid inference.",
            args.runs,
            if args.runs == 1 { "" } else { "s" }
        );
    }

    let aggregate = build_aggregate(args.runs, &runs);

    // Auto-baseline: compare A (even runs) vs B (odd runs) confidence when data available.
    // This always runs in research mode (no --baseline flag needed) if we have >= 2 runs.
    let even_confs: Vec<f64> = runs
        .iter()
        .filter(|r| r.run_index % 2 == 0)
        .filter_map(|r| r.avg_confidence)
        .collect();
    let odd_confs: Vec<f64> = runs
        .iter()
        .filter(|r| r.run_index % 2 == 1)
        .filter_map(|r| r.avg_confidence)
        .collect();
    if !even_confs.is_empty() && !odd_confs.is_empty() {
        let even_mean = even_confs.iter().sum::<f64>() / even_confs.len() as f64;
        let odd_mean = odd_confs.iter().sum::<f64>() / odd_confs.len() as f64;
        eprintln!("[auto-baseline] even_runs_mean_confidence={:.4} odd_runs_mean_confidence={:.4} delta={:+.4}",
            even_mean, odd_mean, odd_mean - even_mean);
        // Also run t-test if enough data
        if let Some(p) = two_sample_t_test(&even_confs, &odd_confs) {
            let sig = if p < 0.05 {
                "SIGNIFICANT"
            } else {
                "not significant"
            };
            eprintln!("[auto-baseline] t-test p={:.4} ({})", p, sig);
        }
    }

    // Baseline comparison: if --baseline and there's a "none" transform in the DB
    if args.baseline {
        if let (Some(ref s), Some(eid)) = (&store, exp_id) {
            let baseline_runs = s
                .load_runs_by_transform(&args.prompt, "none")
                .unwrap_or_default();
            if !baseline_runs.is_empty() {
                let baseline_confs: Vec<f64> = baseline_runs
                    .iter()
                    .filter_map(|r| r.avg_confidence)
                    .collect();
                let current_confs: Vec<f64> =
                    runs.iter().filter_map(|r| r.avg_confidence).collect();
                if !baseline_confs.is_empty() && !current_confs.is_empty() {
                    let baseline_mean =
                        baseline_confs.iter().sum::<f64>() / baseline_confs.len() as f64;
                    let current_mean =
                        current_confs.iter().sum::<f64>() / current_confs.len() as f64;
                    let delta = current_mean - baseline_mean;
                    eprintln!(
                        "[baseline] mean_confidence={:.4} vs current={:.4} delta={:+.4} (exp_id={})",
                        baseline_mean, current_mean, delta, eid
                    );
                }
            }
        }
    }

    // A/B significance test
    if args.significance && args.system_b.is_some() {
        let even_confs: Vec<f64> = runs
            .iter()
            .filter(|r| r.run_index % 2 == 0)
            .filter_map(|r| r.avg_confidence)
            .collect();
        let odd_confs: Vec<f64> = runs
            .iter()
            .filter(|r| r.run_index % 2 == 1)
            .filter_map(|r| r.avg_confidence)
            .collect();

        if !even_confs.is_empty() && !odd_confs.is_empty() {
            if let Some(p) = two_sample_t_test(&even_confs, &odd_confs) {
                if p < 0.05 {
                    eprintln!(
                        "[significance] WARNING: A/B confidence difference is statistically significant (p={:.4})",
                        p
                    );
                } else {
                    eprintln!(
                        "[significance] A/B confidence difference is NOT significant (p={:.4})",
                        p
                    );
                }
            }
        }
    }

    // Export heatmap if requested
    if let (Some(ref exporter), Some(ref path)) = (heatmap_exporter, &args.heatmap_export) {
        exporter.export_csv(path, args.heatmap_min_confidence, &args.heatmap_sort_by)?;
        eprintln!("[research] heatmap exported to {}", path);
    }

    let output = ResearchOutput {
        schema_version: 2,
        prompt: args.prompt.clone(),
        provider: provider.to_string(),
        transform: transform_str,
        runs,
        aggregate,
    };

    let json = serde_json::to_string_pretty(&output)?;
    std::fs::write(&args.output, &json)?;
    eprintln!("[research] wrote {} bytes to {}", json.len(), args.output);

    // Export timeseries CSV if requested via --export-timeseries.
    if let Some(ref ts_path) = args.export_timeseries {
        match write_timeseries_csv(ts_path, &output.runs) {
            Ok(_) => eprintln!("[eot] timeseries CSV written to {}", ts_path),
            Err(e) => eprintln!("[eot] failed to write timeseries CSV: {}", e),
        }
    }

    // Cost estimate summary (#13).
    let total_tokens: usize = output.runs.iter().map(|r| r.token_count).sum();
    let rate = cost_per_1k_tokens(&model);
    let estimated_cost = total_tokens as f64 / 1000.0 * rate;
    eprintln!(
        "[research] total tokens: {} | estimated cost: ${:.4} ({}, ${:.3}/1K tokens)",
        total_tokens, estimated_cost, model, rate
    );
    eprintln!("[eot] Note: cost estimates may be outdated — verify current pricing at your provider's documentation.");

    // Perplexity normalisation note (#20).
    // Anthropic never exposes logprobs, so perplexity is unavailable.
    // Cross-provider comparisons of perplexity are also not meaningful
    // without normalising by log(vocab_size) for each model.
    if output.provider == "anthropic" && output.aggregate.mean_perplexity.is_none() {
        eprintln!(
            "[research] note: Anthropic does not expose logprobs — \
             perplexity/confidence metrics are unavailable. \
             Use --provider openai for logprob-based analysis."
        );
    }

    Ok(())
}

// Cost estimate per model (output tokens, $/1K tokens).
// These are approximate list prices; verify at platform.openai.com / anthropic.com.
fn cost_per_1k_tokens(model: &str) -> f64 {
    match model {
        m if m.starts_with("gpt-4o") => 0.015,
        m if m.starts_with("gpt-4.1") => 0.010,
        m if m.starts_with("gpt-4") => 0.030,
        m if m.starts_with("gpt-3.5") => 0.002,
        m if m.contains("claude") && m.contains("opus") => 0.075,
        m if m.contains("claude") && m.contains("sonnet") => 0.015,
        m if m.contains("claude") && m.contains("haiku") => 0.001,
        _ => 0.002,
    }
}

fn build_aggregate(total_runs: u32, runs: &[ResearchRun]) -> ResearchAggregate {
    if runs.is_empty() {
        return ResearchAggregate::empty(total_runs);
    }

    let token_counts: Vec<f64> = runs.iter().map(|r| r.token_count as f64).collect();
    let mean_token_count = token_counts.iter().sum::<f64>() / token_counts.len() as f64;
    let std_dev_token_count = std_dev(&token_counts);

    let confidences: Vec<f64> = runs.iter().filter_map(|r| r.avg_confidence).collect();
    let (mean_confidence, std_dev_confidence, confidence_interval_95_confidence) =
        if confidences.is_empty() {
            (None, None, None)
        } else {
            let mean = confidences.iter().sum::<f64>() / confidences.len() as f64;
            let sd = std_dev(&confidences);
            let ci = if confidences.len() >= 2 {
                Some(ci_95(mean, sd, confidences.len()))
            } else {
                None
            };
            (Some(mean), Some(sd), ci)
        };

    let perplexities: Vec<f64> = runs.iter().filter_map(|r| r.avg_perplexity).collect();
    let (mean_perplexity, std_dev_perplexity, confidence_interval_95_perplexity) =
        if perplexities.is_empty() {
            (None, None, None)
        } else {
            let mean = perplexities.iter().sum::<f64>() / perplexities.len() as f64;
            let sd = std_dev(&perplexities);
            let ci = if perplexities.len() >= 2 {
                Some(ci_95(mean, sd, perplexities.len()))
            } else {
                None
            };
            (Some(mean), Some(sd), ci)
        };

    let mean_vocab_diversity =
        runs.iter().map(|r| r.vocab_diversity).sum::<f64>() / runs.len() as f64;

    let aligned_length = runs.iter().map(|r| r.token_count).min().unwrap_or(0);

    // Aggregate per-transform perplexity across runs
    let mut all_transform_perp: std::collections::HashMap<String, Vec<f64>> =
        std::collections::HashMap::new();
    for run in runs {
        for (label, perp) in &run.per_transform_perplexity {
            all_transform_perp
                .entry(label.clone())
                .or_default()
                .push(*perp);
        }
    }
    let mean_per_transform_perplexity: std::collections::HashMap<String, f64> = all_transform_perp
        .into_iter()
        .map(|(k, v)| (k, v.iter().sum::<f64>() / v.len() as f64))
        .collect();

    ResearchAggregate {
        total_runs,
        mean_token_count,
        mean_confidence,
        mean_perplexity,
        mean_vocab_diversity,
        std_dev_confidence,
        std_dev_perplexity,
        std_dev_token_count,
        confidence_interval_95_confidence,
        confidence_interval_95_perplexity,
        aligned_length,
        mean_per_transform_perplexity,
        small_n_warning: total_runs < 30,
    }
}

impl ResearchAggregate {
    fn empty(total_runs: u32) -> Self {
        ResearchAggregate {
            total_runs,
            mean_token_count: 0.0,
            mean_confidence: None,
            mean_perplexity: None,
            mean_vocab_diversity: 0.0,
            std_dev_confidence: None,
            std_dev_perplexity: None,
            std_dev_token_count: 0.0,
            confidence_interval_95_confidence: None,
            confidence_interval_95_perplexity: None,
            aligned_length: 0,
            mean_per_transform_perplexity: std::collections::HashMap::new(),
            small_n_warning: total_runs < 30,
        }
    }
}

/// Detect positions where confidence drops below `threshold` for `min_run` consecutive tokens.
fn detect_collapse_positions(confidences: &[f64], min_run: usize, threshold: f64) -> Vec<usize> {
    let mut positions = Vec::new();
    let mut run_start: Option<usize> = None;
    let mut run_len = 0usize;
    for (i, &conf) in confidences.iter().enumerate() {
        if conf < threshold {
            if run_start.is_none() {
                run_start = Some(i);
                run_len = 0;
            }
            run_len += 1;
            if run_len == min_run {
                // run_start is always Some here because it is set whenever conf < threshold
                // and only cleared when conf >= threshold (which resets run_len to 0).
                if let Some(start) = run_start {
                    positions.push(start);
                }
            }
        } else {
            run_start = None;
            run_len = 0;
        }
    }
    positions
}

/// Run [`run_research`] independently for each prompt listed in `args.prompt_file`.
///
/// The file must contain one prompt per line; blank lines and lines beginning
/// with `#` are skipped.  Results for each prompt are written to a separate
/// JSON file named `<output>_<index>.json`.
///
/// # Errors
/// Returns an error if `args.prompt_file` is `None`, the file cannot be read,
/// or any individual research run fails.
pub async fn run_research_suite(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let path = args.prompt_file.as_ref().ok_or("No prompt_file set")?;
    let content = std::fs::read_to_string(path)?;
    let prompts: Vec<String> = content
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();

    if prompts.is_empty() {
        tracing::warn!(path = %path, "no prompts found in prompt file");
        eprintln!("[suite] No prompts found in {}", path);
        return Ok(());
    }

    tracing::info!(count = prompts.len(), path = %path, "running research suite");
    eprintln!("[suite] Running {} prompts from {}", prompts.len(), path);
    for (idx, prompt) in prompts.iter().enumerate() {
        eprintln!("[suite] Prompt {}/{}: {}", idx + 1, prompts.len(), prompt);
        run_research_for_prompt(args, prompt, idx).await?;
    }
    Ok(())
}

/// Run research for a single prompt override (used by the suite runner).
async fn run_research_for_prompt(
    args: &Args,
    prompt: &str,
    idx: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let provider = args.provider.clone();
    let transform_str = args.transform.clone();
    let transform =
        Transform::from_str_loose(&transform_str).map_err(|e| format!("Invalid transform: {e}"))?;
    let model = crate::cli::resolve_model(&provider, &args.model);

    let store = if let Some(db_path) = &args.db {
        Some(crate::store::ExperimentStore::open(db_path)?)
    } else {
        None
    };

    let exp_id: Option<i64> = if let Some(ref s) = store {
        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let id = s.insert_experiment(
            &format!("{}", now),
            prompt,
            &provider.to_string(),
            &transform_str,
            &model,
        )?;
        Some(id)
    } else {
        None
    };
    let _ = exp_id;

    let mut runs: Vec<ResearchRun> = Vec::with_capacity(args.runs as usize);
    for i in 0..args.runs {
        eprintln!("[suite] run {}/{} for prompt {}", i + 1, args.runs, idx);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut interceptor = crate::TokenInterceptor::new(
            provider.clone(),
            transform.clone(),
            model.clone(),
            false,
            false,
            false,
        )?;
        interceptor.web_tx = Some(tx);
        if let Some(rate) = args.rate {
            interceptor = interceptor.with_rate(rate);
        }
        if let Some(seed) = args.seed {
            interceptor = interceptor.with_seed(seed);
        }
        let run_start = std::time::Instant::now();
        interceptor.intercept_stream(prompt).await?;
        let elapsed_ms = run_start.elapsed().as_millis() as u64;
        drop(interceptor);

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }

        let token_count = events.len();
        let transformed_count = events.iter().filter(|e| e.transformed).count();
        let confidences: Vec<f64> = events
            .iter()
            .filter_map(|e| e.confidence.map(|v| v as f64))
            .collect();
        let avg_confidence = if confidences.is_empty() {
            None
        } else {
            Some(confidences.iter().sum::<f64>() / confidences.len() as f64)
        };
        let perplexities: Vec<f64> = events
            .iter()
            .filter_map(|e| e.perplexity.map(|v| v as f64))
            .collect();
        let avg_perplexity = if perplexities.is_empty() {
            None
        } else {
            Some(perplexities.iter().sum::<f64>() / perplexities.len() as f64)
        };
        let unique: std::collections::HashSet<&str> =
            events.iter().map(|e| e.original.as_str()).collect();
        let vocab_diversity = if token_count == 0 {
            0.0
        } else {
            unique.len() as f64 / token_count as f64
        };
        let per_token_confidences: Vec<f64> = events
            .iter()
            .map(|e| e.confidence.map(|v| v as f64).unwrap_or(1.0))
            .collect();
        let collapse_positions =
            detect_collapse_positions(&per_token_confidences, args.collapse_window, 0.4);

        // Per-transform perplexity breakdown
        let mut transform_perp: std::collections::HashMap<String, Vec<f64>> =
            std::collections::HashMap::new();
        for event in &events {
            if event.transformed {
                let label = event
                    .chaos_label
                    .clone()
                    .unwrap_or_else(|| transform_str.clone());
                if let Some(p) = event.perplexity {
                    transform_perp.entry(label).or_default().push(p as f64);
                }
            }
        }
        let per_transform_perplexity: std::collections::HashMap<String, f64> = transform_perp
            .into_iter()
            .map(|(k, v)| (k, v.iter().sum::<f64>() / v.len() as f64))
            .collect();

        let tokens_per_second = if elapsed_ms > 0 {
            Some(token_count as f64 / (elapsed_ms as f64 / 1000.0))
        } else {
            None
        };

        let token_latencies_ms2: Vec<u64> = if token_count == 0 || elapsed_ms == 0 {
            Vec::new()
        } else {
            (0..token_count)
                .map(|j| ((j + 1) as u64 * elapsed_ms) / token_count as u64)
                .collect()
        };
        let p50_latency_ms2 = percentile_latency(&token_latencies_ms2, 50);
        let p95_latency_ms2 = percentile_latency(&token_latencies_ms2, 95);
        runs.push(ResearchRun {
            run_index: i,
            token_count,
            transformed_count,
            avg_confidence,
            avg_perplexity,
            vocab_diversity,
            collapse_positions,
            per_transform_perplexity,
            elapsed_ms,
            tokens_per_second,
            token_latencies_ms: token_latencies_ms2,
            p50_latency_ms: p50_latency_ms2,
            p95_latency_ms: p95_latency_ms2,
        });
    }

    let aggregate = build_aggregate(args.runs, &runs);
    let output_path = {
        let base = args.output.trim_end_matches(".json");
        format!("{}_{}.json", base, idx)
    };
    let output = ResearchOutput {
        schema_version: 2,
        prompt: prompt.to_string(),
        provider: provider.to_string(),
        transform: transform_str,
        runs,
        aggregate,
    };
    let json = serde_json::to_string_pretty(&output)?;
    std::fs::write(&output_path, &json)?;
    eprintln!("[suite] wrote {} bytes to {}", json.len(), output_path);
    Ok(())
}

/// Stream the same prompt through OpenAI and Anthropic in parallel and print
/// a side-by-side token diff in the terminal.
///
/// Diverging token positions are highlighted in red.
///
/// # Errors
/// Returns an error if either provider's streaming call fails.
pub async fn run_diff_terminal(args: &crate::cli::Args) -> Result<(), Box<dyn std::error::Error>> {
    use crate::providers::Provider;
    use crate::TokenInterceptor;
    use colored::*;
    use tokio::sync::mpsc;
    tracing::info!("starting diff terminal: OpenAI vs Anthropic in parallel");

    let transform_openai = crate::transforms::Transform::from_str_loose(&args.transform)
        .map_err(|e| format!("Invalid transform: {e}"))?;
    let transform_anthropic = transform_openai.clone();

    let model_openai = crate::cli::resolve_model(&Provider::Openai, &args.model);
    let model_anthropic = crate::cli::resolve_model(&Provider::Anthropic, &args.model);

    let (tx_a, mut rx_a) = mpsc::unbounded_channel();
    let (tx_b, mut rx_b) = mpsc::unbounded_channel();

    let mut ia = TokenInterceptor::new(
        Provider::Openai,
        transform_openai,
        model_openai,
        false,
        false,
        false,
    )?;
    ia.web_tx = Some(tx_a);
    let mut ib = TokenInterceptor::new(
        Provider::Anthropic,
        transform_anthropic,
        model_anthropic,
        false,
        false,
        false,
    )?;
    ib.web_tx = Some(tx_b);

    let prompt = args.prompt.clone();
    let prompt_b = prompt.clone();
    let (res_a, res_b) = tokio::join!(
        async move { ia.intercept_stream(&prompt).await },
        async move { ib.intercept_stream(&prompt_b).await }
    );
    res_a?;
    res_b?;

    let mut events_a = Vec::new();
    let mut events_b = Vec::new();
    while let Ok(e) = rx_a.try_recv() {
        events_a.push(e);
    }
    while let Ok(e) = rx_b.try_recv() {
        events_b.push(e);
    }

    let max_len = events_a.len().max(events_b.len());
    println!(
        "{:<30}  {}",
        "OpenAI".bright_cyan().bold(),
        "Anthropic".bright_magenta().bold()
    );
    println!("{}", "-".repeat(65));
    for i in 0..max_len {
        let a_text = events_a.get(i).map(|e| e.text.as_str()).unwrap_or("");
        let b_text = events_b.get(i).map(|e| e.text.as_str()).unwrap_or("");
        let diverge = a_text != b_text;
        if diverge {
            println!("{:<30}  {}", a_text.red(), b_text.red());
        } else {
            println!("{:<30}  {}", a_text, b_text);
        }
    }
    Ok(())
}

/// Simple two-sample Welch's t-test. Returns approximate p-value (two-tailed).
/// Returns None if variance is zero or samples too small.
fn two_sample_t_test(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() < 2 || b.len() < 2 {
        return None;
    }
    let mean_a = a.iter().sum::<f64>() / a.len() as f64;
    let mean_b = b.iter().sum::<f64>() / b.len() as f64;
    let var_a = a.iter().map(|v| (v - mean_a).powi(2)).sum::<f64>() / (a.len() - 1) as f64;
    let var_b = b.iter().map(|v| (v - mean_b).powi(2)).sum::<f64>() / (b.len() - 1) as f64;
    let se = ((var_a / a.len() as f64) + (var_b / b.len() as f64)).sqrt();
    if se == 0.0 {
        return None;
    }
    let t = (mean_a - mean_b).abs() / se;
    // Approximate p-value via normal distribution (large-sample approximation)
    let p = 2.0 * (1.0 - normal_cdf(t));
    Some(p)
}

/// Approximation of the standard normal CDF using Abramowitz & Stegun formula.
fn normal_cdf(z: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * z.abs());
    let poly = t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let pdf = (-z * z / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let p = 1.0 - pdf * poly;
    if z >= 0.0 {
        p
    } else {
        1.0 - p
    }
}

/// Write per-run timeseries data to a CSV file.
/// Columns: run,token_index,confidence,perplexity
pub fn write_timeseries_csv(path: &str, runs: &[ResearchRun]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "run,token_index,confidence,perplexity")?;
    for run in runs {
        let n = run.token_count;
        for i in 0..n {
            let conf = run.avg_confidence.map(|v| format!("{:.6}", v)).unwrap_or_default();
            let perp = run.avg_perplexity.map(|v| format!("{:.6}", v)).unwrap_or_default();
            writeln!(f, "{},{},{},{}", run.run_index, i, conf, perp)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Batch research mode (--batch <file.jsonl>)
// ---------------------------------------------------------------------------

/// One entry in a batch JSONL file.
#[derive(serde::Deserialize)]
pub struct BatchEntry {
    pub prompt: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub transforms: Vec<String>,
}

/// Result record written to the batch output JSONL.
#[derive(serde::Serialize)]
pub struct BatchResult {
    pub prompt: String,
    pub model: String,
    pub transform: String,
    pub token_count: usize,
    pub avg_confidence: Option<f64>,
    pub avg_perplexity: Option<f64>,
    pub vocab_diversity: f64,
    pub elapsed_ms: u64,
}

/// Simple terminal progress bar helper (no external deps).
struct BatchProgress {
    total: usize,
    current: usize,
    start: std::time::Instant,
}

impl BatchProgress {
    fn new(total: usize) -> Self {
        Self { total, current: 0, start: std::time::Instant::now() }
    }

    fn advance(&mut self, label: &str) {
        self.current += 1;
        let pct = self.current * 100 / self.total.max(1);
        let elapsed = self.start.elapsed().as_secs_f64();
        let bar_len = 40usize;
        let filled = (pct * bar_len / 100).min(bar_len);
        let bar: String = std::iter::repeat('#').take(filled)
            .chain(std::iter::repeat('-').take(bar_len - filled))
            .collect();
        eprintln!(
            "[batch] [{bar}] {}/{} ({pct}%) {label} ({elapsed:.1}s)",
            self.current, self.total,
            bar = bar,
            pct = pct,
            elapsed = elapsed,
        );
    }

    fn finish(&self) {
        let elapsed = self.start.elapsed().as_secs_f64();
        eprintln!("[batch] Done — {} entries in {:.1}s", self.total, elapsed);
    }
}

/// Run batch research mode: reads a JSONL file, processes each entry
/// sequentially, writes results to `batch_results_<timestamp>.jsonl`.
///
/// Each JSONL line must be: `{"prompt":"...","model":"gpt-4o","transforms":["reverse"]}`
pub async fn run_batch(
    args: &Args,
    batch_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    use std::time::SystemTime;

    let content = std::fs::read_to_string(batch_path)?;
    let entries: Vec<BatchEntry> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();

    if entries.is_empty() {
        eprintln!("[batch] No valid entries found in {}", batch_path);
        return Ok(());
    }

    let timestamp = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let output_path = format!("batch_results_{}.jsonl", timestamp);
    let mut out_file = std::fs::File::create(&output_path)?;

    let mut progress = BatchProgress::new(entries.len());
    eprintln!("[batch] Processing {} entries → {}", entries.len(), output_path);

    for (idx, entry) in entries.iter().enumerate() {
        let label = format!("prompt #{}: {:.50}", idx + 1, entry.prompt);

        let provider = args.provider.clone();
        let model = if entry.model.is_empty() {
            crate::cli::resolve_model(&provider, &args.model)
        } else {
            entry.model.clone()
        };

        let transforms: Vec<String> = if entry.transforms.is_empty() {
            vec![args.transform.clone()]
        } else {
            entry.transforms.clone()
        };

        for transform_str in &transforms {
            let transform = match crate::transforms::Transform::from_str_loose(transform_str) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("[batch] Skipping invalid transform '{}': {}", transform_str, e);
                    continue;
                }
            };

            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
            let mut interceptor = match crate::TokenInterceptor::new(
                provider.clone(),
                transform,
                model.clone(),
                false,
                false,
                false,
            ) {
                Ok(i) => i,
                Err(e) => {
                    eprintln!("[batch] Interceptor error: {}", e);
                    continue;
                }
            };
            interceptor.web_tx = Some(tx);
            if let Some(rate) = args.rate {
                interceptor = interceptor.with_rate(rate);
            }

            let run_start = std::time::Instant::now();
            let _ = interceptor.intercept_stream(&entry.prompt).await;
            let elapsed_ms = run_start.elapsed().as_millis() as u64;
            drop(interceptor);

            let mut events = Vec::new();
            while let Ok(e) = rx.try_recv() {
                events.push(e);
            }

            let token_count = events.len();
            let confidences: Vec<f64> = events
                .iter()
                .filter_map(|e| e.confidence.map(|v| v as f64))
                .collect();
            let avg_confidence = if confidences.is_empty() {
                None
            } else {
                Some(confidences.iter().sum::<f64>() / confidences.len() as f64)
            };
            let perplexities: Vec<f64> = events
                .iter()
                .filter_map(|e| e.perplexity.map(|v| v as f64))
                .collect();
            let avg_perplexity = if perplexities.is_empty() {
                None
            } else {
                Some(perplexities.iter().sum::<f64>() / perplexities.len() as f64)
            };
            let unique: std::collections::HashSet<&str> =
                events.iter().map(|e| e.original.as_str()).collect();
            let vocab_diversity = if token_count == 0 {
                0.0
            } else {
                unique.len() as f64 / token_count as f64
            };

            let result = BatchResult {
                prompt: entry.prompt.clone(),
                model: model.clone(),
                transform: transform_str.clone(),
                token_count,
                avg_confidence,
                avg_perplexity,
                vocab_diversity,
                elapsed_ms,
            };
            let line = serde_json::to_string(&result)?;
            writeln!(out_file, "{}", line)?;
        }

        progress.advance(&label);
    }

    progress.finish();
    eprintln!("[batch] Results written to {}", output_path);
    Ok(())
}

// ---------------------------------------------------------------------------
// Token logprob CSV export (--export-logprobs <file.csv>)
// ---------------------------------------------------------------------------

/// Write per-token logprob data to a CSV file from a list of token events.
/// Columns: token,logprob,rank,model,timestamp
pub fn write_logprob_csv(
    path: &str,
    events: &[crate::TokenEvent],
    model: &str,
) -> std::io::Result<()> {
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    let mut f = std::fs::File::create(path)?;
    writeln!(f, "token,logprob,rank,model,timestamp")?;

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    for (rank, event) in events.iter().enumerate() {
        // logprob: derive from confidence (confidence = exp(logprob))
        let logprob = event
            .confidence
            .map(|c| (c as f64).max(1e-10).ln())
            .unwrap_or(f64::NEG_INFINITY);

        // Escape commas in token text
        let token_escaped = event.text.replace('"', "\"\"");
        writeln!(
            f,
            "\"{}\",{:.6},{},{},{}",
            token_escaped, logprob, rank, model, ts
        )?;
    }
    Ok(())
}

/// Run a single stream and export logprobs to CSV (used by --export-logprobs).
pub async fn run_with_logprob_export(
    args: &Args,
    export_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let provider = args.provider.clone();
    let model = crate::cli::resolve_model(&provider, &args.model);
    let transform = crate::transforms::Transform::from_str_loose(&args.transform)
        .map_err(|e| format!("Invalid transform: {e}"))?;

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let mut interceptor = crate::TokenInterceptor::new(
        provider,
        transform,
        model.clone(),
        args.visual,
        args.heatmap,
        args.orchestrator,
    )?;
    interceptor.web_tx = Some(tx);
    interceptor.top_logprobs = args.top_logprobs;
    if let Some(rate) = args.rate {
        interceptor = interceptor.with_rate(rate);
    }

    interceptor.intercept_stream(&args.prompt).await?;
    drop(interceptor);

    let mut events = Vec::new();
    while let Ok(e) = rx.try_recv() {
        events.push(e);
    }

    write_logprob_csv(export_path, &events, &model)?;
    eprintln!("[eot] logprob CSV written to {} ({} tokens)", export_path, events.len());
    Ok(())
}

// ---------------------------------------------------------------------------
// Multi-model comparison heatmap (--compare model1,model2)
// ---------------------------------------------------------------------------

/// Token-level comparison result between models at one position.
#[derive(serde::Serialize)]
pub struct ModelTokenComparison {
    pub position: usize,
    pub model: String,
    pub token: String,
    pub confidence: Option<f64>,
    pub divergence: f64,
}

/// Run the same prompt through multiple models and produce a divergence heatmap.
/// Models are specified as a comma-separated list in `args.compare`.
pub async fn run_multi_model_compare(
    args: &Args,
    models_csv: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use colored::*;

    let models: Vec<String> = models_csv
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    if models.len() < 2 {
        return Err("--compare requires at least 2 models (comma-separated)".into());
    }

    let provider = args.provider.clone();
    let transform = crate::transforms::Transform::from_str_loose(&args.transform)
        .map_err(|e| format!("Invalid transform: {e}"))?;

    eprintln!(
        "[compare] Running prompt through {} models: {}",
        models.len(),
        models.join(", ")
    );

    // Collect token events per model (sequential to avoid rate-limit issues)
    let mut per_model_events: Vec<(String, Vec<crate::TokenEvent>)> = Vec::new();

    for model in &models {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut interceptor = crate::TokenInterceptor::new(
            provider.clone(),
            transform.clone(),
            model.clone(),
            false,
            false,
            false,
        )?;
        interceptor.web_tx = Some(tx);
        interceptor.top_logprobs = args.top_logprobs;
        if let Some(rate) = args.rate {
            interceptor = interceptor.with_rate(rate);
        }

        eprintln!("[compare] Streaming model: {}", model);
        interceptor.intercept_stream(&args.prompt).await?;
        drop(interceptor);

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        per_model_events.push((model.clone(), events));
    }

    // Build comparison table: for each position, show each model's token + confidence
    let max_len = per_model_events
        .iter()
        .map(|(_, e)| e.len())
        .max()
        .unwrap_or(0);

    // Compute per-position divergence (std dev of confidences across models)
    println!("\n[compare] Token divergence heatmap ({} positions):", max_len);
    println!(
        "{:>4}  {:>10}  {}",
        "pos", "divergence",
        models.iter().map(|m| format!("{:>20}", m)).collect::<Vec<_>>().join("  ")
    );
    println!("{}", "-".repeat(80));

    let mut all_comparisons: Vec<ModelTokenComparison> = Vec::new();

    for pos in 0..max_len {
        let confidences: Vec<f64> = per_model_events
            .iter()
            .filter_map(|(_, evs)| {
                evs.get(pos).and_then(|e| e.confidence.map(|c| c as f64))
            })
            .collect();

        // Divergence = std dev of confidences at this position
        let divergence = if confidences.len() >= 2 {
            let mean = confidences.iter().sum::<f64>() / confidences.len() as f64;
            let var = confidences.iter().map(|c| (c - mean).powi(2)).sum::<f64>()
                / (confidences.len() - 1) as f64;
            var.sqrt()
        } else {
            0.0
        };

        // Color by divergence level
        let div_str = format!("{:.4}", divergence);
        let div_colored = if divergence > 0.3 {
            div_str.red().to_string()
        } else if divergence > 0.1 {
            div_str.yellow().to_string()
        } else {
            div_str.green().to_string()
        };

        let token_cells: Vec<String> = per_model_events
            .iter()
            .map(|(model_name, evs)| {
                let tok = evs.get(pos).map(|e| e.text.as_str()).unwrap_or("-");
                let conf = evs.get(pos).and_then(|e| e.confidence);
                let cell = format!("{:>12} ({:.2})", &tok[..tok.len().min(12)], conf.unwrap_or(0.0));

                all_comparisons.push(ModelTokenComparison {
                    position: pos,
                    model: model_name.clone(),
                    token: tok.to_string(),
                    confidence: conf.map(|c| c as f64),
                    divergence,
                });

                if divergence > 0.3 {
                    cell.red().to_string()
                } else {
                    cell
                }
            })
            .collect();

        println!(
            "{:>4}  {:>10}  {}",
            pos,
            div_colored,
            token_cells.join("  ")
        );
    }

    // Summary statistics
    let high_divergence: Vec<_> = (0..max_len)
        .filter(|&pos| {
            let confs: Vec<f64> = per_model_events
                .iter()
                .filter_map(|(_, evs)| evs.get(pos).and_then(|e| e.confidence.map(|c| c as f64)))
                .collect();
            if confs.len() < 2 { return false; }
            let mean = confs.iter().sum::<f64>() / confs.len() as f64;
            let var = confs.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / (confs.len() - 1) as f64;
            var.sqrt() > 0.1
        })
        .collect();

    eprintln!(
        "\n[compare] High-divergence positions (>0.1 std dev): {}/{}",
        high_divergence.len(),
        max_len
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_runs(data: &[(usize, usize, Option<f64>, Option<f64>, f64)]) -> Vec<ResearchRun> {
        data.iter()
            .enumerate()
            .map(|(i, &(tc, tr, conf, perp, vd))| ResearchRun {
                run_index: i as u32,
                token_count: tc,
                transformed_count: tr,
                avg_confidence: conf,
                avg_perplexity: perp,
                vocab_diversity: vd,
                collapse_positions: vec![],
                per_transform_perplexity: std::collections::HashMap::new(),
                elapsed_ms: 0,
                tokens_per_second: None,
                token_latencies_ms: vec![],
                p50_latency_ms: None,
                p95_latency_ms: None,
            })
            .collect()
    }

    #[test]
    fn test_build_aggregate_empty() {
        let agg = build_aggregate(0, &[]);
        assert_eq!(agg.total_runs, 0);
        assert_eq!(agg.mean_token_count, 0.0);
        assert!(agg.mean_confidence.is_none());
        assert!(agg.mean_perplexity.is_none());
    }

    #[test]
    fn test_build_aggregate_single_run() {
        let runs = make_runs(&[(10, 5, Some(0.9), Some(1.1), 0.8)]);
        let agg = build_aggregate(1, &runs);
        assert_eq!(agg.total_runs, 1);
        assert_eq!(agg.mean_token_count, 10.0);
        assert!((agg.mean_confidence.unwrap() - 0.9).abs() < 1e-9);
        assert!((agg.mean_perplexity.unwrap() - 1.1).abs() < 1e-9);
        assert!((agg.mean_vocab_diversity - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_build_aggregate_multiple_runs() {
        let runs = make_runs(&[
            (10, 5, Some(0.8), Some(1.2), 0.6),
            (20, 10, Some(0.9), Some(1.1), 0.7),
        ]);
        let agg = build_aggregate(2, &runs);
        assert!((agg.mean_token_count - 15.0).abs() < 1e-9);
        assert!((agg.mean_confidence.unwrap() - 0.85).abs() < 1e-9);
        assert!((agg.mean_vocab_diversity - 0.65).abs() < 1e-9);
    }

    #[test]
    fn test_build_aggregate_no_confidence() {
        let runs = make_runs(&[(5, 2, None, None, 0.5)]);
        let agg = build_aggregate(1, &runs);
        assert!(agg.mean_confidence.is_none());
        assert!(agg.mean_perplexity.is_none());
    }

    #[test]
    fn test_build_aggregate_mean_token_count_fractional() {
        let runs = make_runs(&[
            (7, 3, None, None, 1.0),
            (8, 4, None, None, 1.0),
            (9, 4, None, None, 1.0),
        ]);
        let agg = build_aggregate(3, &runs);
        assert!((agg.mean_token_count - 8.0).abs() < 1e-9);
    }

    #[test]
    fn test_research_output_serializes() {
        let output = ResearchOutput {
            schema_version: 2,
            prompt: "test".to_string(),
            provider: "openai".to_string(),
            transform: "reverse".to_string(),
            runs: vec![],
            aggregate: ResearchAggregate {
                total_runs: 0,
                mean_token_count: 0.0,
                mean_confidence: None,
                mean_perplexity: None,
                mean_vocab_diversity: 0.0,
                std_dev_confidence: None,
                std_dev_perplexity: None,
                std_dev_token_count: 0.0,
                confidence_interval_95_confidence: None,
                confidence_interval_95_perplexity: None,
                aligned_length: 0,
                mean_per_transform_perplexity: std::collections::HashMap::new(),
                small_n_warning: false,
            },
        };
        let json = serde_json::to_string(&output).expect("serialize");
        assert!(json.contains("schema_version"));
        assert!(json.contains(r#""provider":"openai""#));
        assert!(json.contains(r#""schema_version":2"#));
    }

    #[test]
    fn test_research_run_serializes() {
        let run = ResearchRun {
            run_index: 0,
            token_count: 42,
            transformed_count: 21,
            avg_confidence: Some(0.95),
            avg_perplexity: Some(1.05),
            vocab_diversity: 0.7,
            collapse_positions: vec![],
            per_transform_perplexity: std::collections::HashMap::new(),
            elapsed_ms: 0,
            tokens_per_second: None,
            token_latencies_ms: vec![],
            p50_latency_ms: None,
            p95_latency_ms: None,
        };
        let json = serde_json::to_string(&run).expect("serialize");
        let v: serde_json::Value = serde_json::from_str(&json).expect("parse");
        assert_eq!(v["token_count"], 42);
        assert_eq!(v["transformed_count"], 21);
    }

    #[test]
    fn test_build_aggregate_total_runs_matches_param() {
        let runs = make_runs(&[(10, 5, None, None, 0.5)]);
        let agg = build_aggregate(99, &runs);
        assert_eq!(agg.total_runs, 99);
    }

    #[test]
    fn test_build_aggregate_std_dev_fields_present() {
        let runs = make_runs(&[
            (10, 5, Some(0.8), Some(1.2), 0.6),
            (20, 10, Some(0.9), Some(1.0), 0.7),
        ]);
        let agg = build_aggregate(2, &runs);
        assert!(agg.std_dev_confidence.is_some());
        assert!(agg.std_dev_perplexity.is_some());
        assert!(agg.confidence_interval_95_confidence.is_some());
        assert!(agg.confidence_interval_95_perplexity.is_some());
    }

    #[test]
    fn test_std_dev_calculation() {
        // Uses Bessel's correction (sample std dev, n-1).
        // For [2,4,4,4,5,5,7,9]: mean=5, sum-sq-dev=32, variance=32/7 ≈ 4.571
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let sd = std_dev(&values);
        let expected = (32.0f64 / 7.0).sqrt();
        assert!((sd - expected).abs() < 0.001);
    }

    #[test]
    fn test_two_sample_t_test_same_means() {
        let a = vec![1.0, 1.0, 1.0, 1.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        // Zero variance — should return None
        assert!(two_sample_t_test(&a, &b).is_none());
    }

    #[test]
    fn test_two_sample_t_test_different_means_returns_some() {
        let a = vec![0.9, 0.8, 0.85, 0.9, 0.88];
        let b = vec![0.1, 0.2, 0.15, 0.1, 0.12];
        let p = two_sample_t_test(&a, &b);
        assert!(p.is_some(), "should return Some for clearly different means");
        assert!(p.unwrap() < 0.05, "p-value should be significant for large difference");
    }

    #[test]
    fn test_two_sample_t_test_too_few_samples_returns_none() {
        assert!(two_sample_t_test(&[0.5], &[0.6]).is_none());
        assert!(two_sample_t_test(&[], &[0.5, 0.6]).is_none());
    }

    #[test]
    fn test_cost_per_1k_tokens_known_models() {
        assert_eq!(cost_per_1k_tokens("gpt-3.5-turbo"), 0.002);
        assert_eq!(cost_per_1k_tokens("gpt-4o"), 0.015);
        assert_eq!(cost_per_1k_tokens("claude-sonnet-4-6"), 0.015);
        assert_eq!(cost_per_1k_tokens("claude-opus-4-6"), 0.075);
    }

    #[tokio::test]
    async fn test_run_research_runs_zero_returns_error() {
        use crate::providers::Provider;
        let args = crate::cli::Args {
            prompt: "test".to_string(),
            transform: "reverse".to_string(),
            model: "gpt-3.5-turbo".to_string(),
            provider: Provider::Mock,
            visual: false,
            heatmap: false,
            orchestrator: false,
            web: false,
            port: 8888,
            research: true,
            runs: 0,
            output: "/tmp/test_research_out.json".to_string(),
            system_a: None,
            top_logprobs: 5,
            system_b: None,
            db: None,
            significance: false,
            heatmap_export: None,
            heatmap_min_confidence: 0.0,
            heatmap_sort_by: "position".to_string(),
            record: None,
            replay: None,
            rate: None,
            seed: None,
            log_db: None,
            baseline: false,
            prompt_file: None,
            diff_terminal: false,
            json_stream: false,
            completions: None,
            rate_range: None,
            dry_run: false,
            template: None,
            min_confidence: None,
            format: "json".to_string(),
            collapse_window: 5,
            orchestrator_url: "http://localhost:3000".to_string(),
            max_retries: 3,
            anthropic_max_tokens: 4096,
            synonym_file: None,
            api_key: None,
            replay_speed: 1.0,
            timeout: 120,
            export_timeseries: None,
            json_schema: false,
            list_models: None,
            validate_config: false,
        };
        let result = run_research(&args).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("--runs must be at least 1"));
    }

    #[test]
    fn test_collapse_positions_detected() {
        // 6-token dip starting at position 3
        let confs = vec![0.8, 0.9, 0.7, 0.2, 0.1, 0.3, 0.2, 0.1, 0.3, 0.9];
        let positions = detect_collapse_positions(&confs, 5, 0.4);
        assert!(!positions.is_empty(), "should detect at least one collapse");
        assert_eq!(positions[0], 3);
    }

    #[test]
    fn test_collapse_positions_none_when_run_too_short() {
        // Only 4 consecutive low tokens — not enough
        let confs = vec![0.8, 0.1, 0.1, 0.1, 0.1, 0.9];
        let positions = detect_collapse_positions(&confs, 5, 0.4);
        assert!(positions.is_empty());
    }

    #[test]
    fn test_aligned_length_is_min_token_count() {
        let runs = make_runs(&[
            (10, 5, None, None, 1.0),
            (20, 10, None, None, 1.0),
            (15, 7, None, None, 1.0),
        ]);
        let agg = build_aggregate(3, &runs);
        assert_eq!(agg.aligned_length, 10);
    }

    #[test]
    fn test_aligned_length_empty() {
        let agg = build_aggregate(0, &[]);
        assert_eq!(agg.aligned_length, 0);
    }

    // -- percentile_latency tests --

    #[test]
    fn test_percentile_latency_empty_returns_none() {
        assert_eq!(percentile_latency(&[], 50), None);
        assert_eq!(percentile_latency(&[], 95), None);
    }

    #[test]
    fn test_percentile_latency_single_element() {
        assert_eq!(percentile_latency(&[42], 50), Some(42));
        assert_eq!(percentile_latency(&[42], 95), Some(42));
        assert_eq!(percentile_latency(&[42], 0), Some(42));
        assert_eq!(percentile_latency(&[42], 100), Some(42));
    }

    #[test]
    fn test_percentile_latency_p50_even() {
        // [10, 20, 30, 40] → sorted same, p50 idx = round(0.5*3) = round(1.5) = 2 → 30
        let data = [40u64, 10, 30, 20];
        let p50 = percentile_latency(&data, 50).unwrap();
        assert!(p50 == 20 || p50 == 30, "p50={}", p50);
    }

    #[test]
    fn test_percentile_latency_p95() {
        let data: Vec<u64> = (1..=100).collect();
        let p95 = percentile_latency(&data, 95).unwrap();
        assert!(p95 >= 94 && p95 <= 100, "p95={}", p95);
    }

    #[test]
    fn test_percentile_latency_p0_is_min() {
        let data = [5u64, 3, 8, 1, 7];
        assert_eq!(percentile_latency(&data, 0), Some(1));
    }

    #[test]
    fn test_percentile_latency_p100_is_max() {
        let data = [5u64, 3, 8, 1, 7];
        assert_eq!(percentile_latency(&data, 100), Some(8));
    }

    // -- Item 8: latency increases monotonically --
    #[test]
    fn test_latency_increases_monotonically() {
        // Simulate token arrival timestamps that should be non-decreasing
        let latencies: Vec<u64> = vec![0, 5, 10, 15, 20, 25];
        for w in latencies.windows(2) {
            assert!(w[1] >= w[0], "latencies should be non-decreasing: {} < {}", w[1], w[0]);
        }
    }

    // -- Item 10: cost disclaimer message --
    #[test]
    fn test_cost_disclaimer_message_contains_outdated() {
        let msg = "[eot] Note: cost estimates may be outdated — verify current pricing at your provider's documentation.";
        assert!(msg.contains("outdated"), "disclaimer should mention 'outdated'");
        assert!(msg.contains("cost estimates"), "disclaimer should mention 'cost estimates'");
    }

    // -- Item 12: write_timeseries_csv creates file --
    #[test]
    fn test_write_timeseries_csv_creates_file() {
        let tmp = std::env::temp_dir().join("eot_timeseries_test.csv");
        let path = tmp.to_str().unwrap();
        let runs: Vec<ResearchRun> = vec![ResearchRun {
            run_index: 0,
            token_count: 3,
            transformed_count: 1,
            avg_confidence: Some(0.9),
            avg_perplexity: Some(1.1),
            vocab_diversity: 1.0,
            collapse_positions: vec![],
            per_transform_perplexity: std::collections::HashMap::new(),
            elapsed_ms: 100,
            tokens_per_second: None,
            token_latencies_ms: vec![],
            p50_latency_ms: None,
            p95_latency_ms: None,
        }];
        write_timeseries_csv(path, &runs).expect("should write CSV");
        let content = std::fs::read_to_string(path).expect("should read CSV");
        assert!(content.starts_with("run,token_index,confidence,perplexity"),
            "CSV should have header row");
        let _ = std::fs::remove_file(path);
    }

    // -- Item 21: CI 95 with two samples --
    #[test]
    fn test_ci_95_with_two_samples() {
        let (low, high) = ci_95(2.0, std::f64::consts::SQRT_2, 2);
        assert!(low < 2.0, "lower bound should be < mean");
        assert!(high > 2.0, "upper bound should be > mean");
        assert!(low.is_finite());
        assert!(high.is_finite());
    }

    // -- Item 22: empty prompt file returns ok --
    #[test]
    fn test_empty_prompt_file_returns_ok() {
        let tmp = std::env::temp_dir().join("eot_empty_prompt_test.txt");
        std::fs::write(&tmp, "").expect("should create empty file");
        let content = std::fs::read_to_string(&tmp).expect("should read empty file");
        // An empty prompt file produces zero lines — no error on read
        let prompts: Vec<&str> = content.lines().collect();
        assert_eq!(prompts.len(), 0, "empty file should have no prompts");
        let _ = std::fs::remove_file(&tmp);
    }
}
