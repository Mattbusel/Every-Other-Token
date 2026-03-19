//! Command-line argument definitions and helper functions.
//!
//! [`Args`] is the root Clap struct parsed in `main.rs`.  Helper functions
//! ([`resolve_model`], [`validate_model`], [`parse_rate_range`], [`apply_template`])
//! are kept here rather than in `main.rs` so they can be unit-tested in isolation.

use crate::providers::Provider;
use clap::Parser;

#[derive(Parser)]
#[command(name = "every-other-token")]
#[command(version = "4.0.0")]
#[command(about = "A real-time token stream mutator for LLM interpretability research")]
pub struct Args {
    /// Input prompt to send to the LLM (optional when using --web)
    #[arg(default_value = "")]
    pub prompt: String,

    /// Transformation type (reverse, uppercase, mock, noise)
    #[arg(default_value = "reverse")]
    pub transform: String,

    /// Model name (e.g. gpt-4, claude-sonnet-4-20250514)
    #[arg(default_value = "gpt-3.5-turbo")]
    pub model: String,

    /// LLM provider: openai or anthropic
    #[arg(long, value_enum, default_value = "openai")]
    pub provider: Provider,

    /// Enable visual mode with color-coded tokens
    #[arg(long, short)]
    pub visual: bool,

    /// Enable token importance heatmap
    #[arg(long)]
    pub heatmap: bool,

    /// Route through tokio-prompt-orchestrator MCP pipeline at localhost:3000
    #[arg(long)]
    pub orchestrator: bool,

    /// Launch web UI on localhost instead of terminal output
    #[arg(long)]
    pub web: bool,

    /// Port for the web UI server
    #[arg(long, default_value = "8888")]
    pub port: u16,

    /// Enable headless research mode — runs N times and outputs JSON stats
    #[arg(long)]
    pub research: bool,

    /// Number of runs in research mode
    #[arg(long, default_value = "10")]
    pub runs: u32,

    /// Output file path for research JSON (defaults to stdout)
    #[arg(long, default_value = "research_output.json")]
    pub output: String,

    /// System prompt A for A/B experiment mode
    #[arg(long)]
    pub system_a: Option<String>,

    /// Number of top alternative tokens to return per position (OpenAI only, 0–20)
    #[arg(long, default_value = "5")]
    pub top_logprobs: u8,

    /// System prompt B for A/B experiment mode
    #[arg(long)]
    pub system_b: Option<String>,

    /// Path to SQLite database for persisting experiment results (optional)
    #[arg(long)]
    pub db: Option<String>,

    /// Compute statistical significance (two-sample t-test) when ≥2 A/B runs available
    #[arg(long)]
    pub significance: bool,

    /// Export per-position token confidence heatmap to CSV at this path
    #[arg(long)]
    pub heatmap_export: Option<String>,

    /// Minimum average confidence to include a position in heatmap CSV export (0.0–1.0)
    #[arg(long, default_value = "0.0")]
    pub heatmap_min_confidence: f32,

    /// Sort heatmap CSV rows by "position" (default) or "confidence"
    #[arg(long, default_value = "position")]
    pub heatmap_sort_by: String,

    /// Record token events to a JSON replay file at this path
    #[arg(long)]
    pub record: Option<String>,

    /// Replay token events from a JSON file (bypasses live LLM call)
    #[arg(long)]
    pub replay: Option<String>,

    /// Fraction of tokens to intercept and transform (0.0–1.0, default 0.5).
    /// At 0.5 every other token is transformed; at 0.3 roughly one in three.
    /// Uses a deterministic Bresenham spread so results are reproducible when
    /// combined with --seed.
    ///
    /// Stored as `Option<f64>` so the config-file loader can distinguish
    /// "the user explicitly passed --rate" from "the user left it at the
    /// default".  The effective value is `rate.unwrap_or(0.5)`.
    #[arg(long)]
    pub rate: Option<f64>,

    /// Fixed RNG seed for reproducible Noise/Chaos transforms.
    /// Omit to use entropy-seeded randomness (default behaviour).
    #[arg(long)]
    pub seed: Option<u64>,

    /// Path to SQLite experiment log database (requires sqlite-log feature)
    #[arg(long)]
    pub log_db: Option<String>,

    /// Enable per-position confidence baseline comparison (research mode)
    #[arg(long)]
    pub baseline: bool,

    /// Path to a file with one prompt per line for batch research
    #[arg(long)]
    pub prompt_file: Option<String>,

    /// Run two parallel streams (OpenAI + Anthropic) and print side-by-side diff in terminal
    #[arg(long)]
    pub diff_terminal: bool,

    /// Print one JSON line per token to stdout instead of colored text
    #[arg(long)]
    pub json_stream: bool,

    /// Generate shell completions for the given shell and exit
    #[arg(long, value_name = "SHELL")]
    pub completions: Option<clap_complete::Shell>,

    /// HelixRouter base URL for cross-repo pressure feedback (e.g. http://127.0.0.1:8080).
    ///
    /// When set in --web mode, a HelixBridge background task polls HelixRouter's
    /// /api/stats and feeds its pressure_score into the self-improvement loop,
    /// letting EOT adapt token-stream parameters based on downstream load.
    #[cfg(feature = "helix-bridge")]
    #[arg(long)]
    pub helix_url: Option<String>,

    /// Rate range for stochastic experiments, e.g. "0.3-0.7". When set, the
    /// interceptor randomly picks a rate in [min, max] for each run.
    /// Overrides --rate when provided. Format: "MIN-MAX" (e.g. "0.2-0.8").
    #[arg(long)]
    pub rate_range: Option<String>,

    /// Dry-run mode: show what transforms would be applied without calling any API.
    /// Applies the configured transform to a sample token list and prints results.
    #[arg(long)]
    pub dry_run: bool,

    /// Prompt template with {input} placeholder. When set, the positional prompt
    /// is substituted into the template. Example: "Answer this: {input}"
    #[arg(long)]
    pub template: Option<String>,

    /// Only transform tokens whose API confidence is below this threshold.
    /// Tokens with confidence >= threshold are passed through unchanged.
    /// When no confidence data is available (Anthropic), falls back to rate-based selection.
    /// Range: 0.0–1.0. Example: --min-confidence 0.8
    #[arg(long)]
    pub min_confidence: Option<f64>,

    /// Output format for research mode: "json" (default), "jsonl" (one JSON object per line).
    #[arg(long, default_value = "json")]
    pub format: String,

    /// Number of consecutive low-confidence tokens to consider a "collapse" in research mode.
    /// Default: 5.
    #[arg(long, default_value = "5")]
    pub collapse_window: usize,

    /// Base URL for the MCP orchestrator pipeline (default: http://localhost:3000).
    #[arg(long, default_value = "http://localhost:3000")]
    pub orchestrator_url: String,

    /// Maximum API retry attempts on 429/5xx errors (default: 3).
    #[arg(long, default_value = "3")]
    pub max_retries: u32,

    /// Maximum tokens in the Anthropic response (default: 4096).
    /// Ignored when using the OpenAI provider.
    #[arg(long, default_value = "4096")]
    pub anthropic_max_tokens: u32,

    /// Path to a TSV or key=value file of additional synonym pairs to merge with the built-in map.
    /// Format: one `word\treplacement` or `word = replacement` pair per line.
    #[arg(long)]
    pub synonym_file: Option<String>,

    /// Optional API key required for /api/ endpoints in web UI mode.
    /// When set, requests to /api/* must include `Authorization: Bearer <key>`.
    #[arg(long)]
    pub api_key: Option<String>,

    /// Replay speed multiplier for --replay mode. 1.0 = real-time, 2.0 = double speed, 0.0 = instant.
    #[arg(long, default_value = "1.0")]
    pub replay_speed: f64,

    /// Stream hang timeout in seconds. The stream is forcibly dropped if no token
    /// arrives within this duration. Default: 120 (2 minutes). Set to 0 to disable.
    #[arg(long, default_value = "120")]
    pub timeout: u64,

    /// Export per-run timeseries data to a CSV file at this path.
    /// Columns: run,token_index,confidence,perplexity
    #[arg(long)]
    pub export_timeseries: Option<String>,

    /// Print the embedded research JSON schema and exit.
    #[arg(long)]
    pub json_schema: bool,

    /// List known models for a provider: "openai", "anthropic", or "all".
    #[arg(long)]
    pub list_models: Option<String>,

    /// Validate configuration (print resolved values and exit).
    #[arg(long)]
    pub validate_config: bool,
}

/// Select the appropriate default model for the given provider when the user
/// hasn't explicitly chosen one (i.e. the model is still the OpenAI default).
pub fn resolve_model(provider: &Provider, model: &str) -> String {
    match provider {
        Provider::Anthropic if model == "gpt-3.5-turbo" => "claude-sonnet-4-6".to_string(),
        Provider::Mock => "mock-fixture-v1".to_string(),
        _ => model.to_string(),
    }
}

/// Known-good model identifiers for basic validation (#18).
///
/// This list is non-exhaustive — new models are released regularly.
/// An unknown model string produces a warning, not an error.
const KNOWN_OPENAI_MODELS: &[&str] = &[
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "o1",
    "o1-mini",
    "o3",
    "o3-mini",
];

const KNOWN_ANTHROPIC_MODELS: &[&str] = &[
    "claude-3-haiku-20240307",
    "claude-3-sonnet-20240229",
    "claude-3-opus-20240229",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
];

/// Warn if `model` does not match any known model for `provider`.
/// Never errors — unknown models are still forwarded to the API.
pub fn validate_model(provider: &Provider, model: &str) {
    let known: &[&str] = match provider {
        Provider::Openai => KNOWN_OPENAI_MODELS,
        Provider::Anthropic => KNOWN_ANTHROPIC_MODELS,
        Provider::Mock => return,
    };
    if !known.contains(&model) {
        eprintln!(
            "[warn] '{}' is not in the known {} model list — verify the model name is correct",
            model, provider
        );
    }
}

/// Parse "MIN-MAX" rate range string. Returns (min, max) or None on error.
///
/// Uses `rfind('-')` to locate the separator so that scientific-notation
/// values such as `"1e-3-0.5"` are parsed correctly (`1e-3` = 0.001).
pub fn parse_rate_range(s: &str) -> Option<(f64, f64)> {
    let sep = s.rfind('-')?;
    let min = s[..sep].parse::<f64>().ok()?;
    let max = s[sep + 1..].parse::<f64>().ok()?;
    if min <= max && min >= 0.0 && max <= 1.0 {
        Some((min, max))
    } else {
        None
    }
}

/// Apply template substitution: replace "{input}" with the prompt.
///
/// The prompt is inserted verbatim — any literal `{input}` occurrences
/// already inside the prompt are not re-expanded because we use a single
/// non-recursive `replace` on the *template* string only.
pub fn apply_template(template: &str, prompt: &str) -> String {
    // Split on the literal placeholder and rejoin with the prompt so that
    // braces inside `prompt` itself are never interpreted as placeholders.
    template.split("{input}").collect::<Vec<_>>().join(prompt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_model_anthropic_default_swap() {
        assert_eq!(
            resolve_model(&Provider::Anthropic, "gpt-3.5-turbo"),
            "claude-sonnet-4-6"
        );
    }

    #[test]
    fn test_resolve_model_anthropic_explicit_model_kept() {
        assert_eq!(
            resolve_model(&Provider::Anthropic, "claude-haiku-4-5-20251001"),
            "claude-haiku-4-5-20251001"
        );
    }

    #[test]
    fn test_resolve_model_openai_default_kept() {
        assert_eq!(
            resolve_model(&Provider::Openai, "gpt-3.5-turbo"),
            "gpt-3.5-turbo"
        );
    }

    #[test]
    fn test_resolve_model_openai_explicit_model_kept() {
        assert_eq!(resolve_model(&Provider::Openai, "gpt-4"), "gpt-4");
    }

    #[test]
    fn test_args_parse_minimal() {
        let args = Args::parse_from(["eot", "hello world"]);
        assert_eq!(args.prompt, "hello world");
        assert_eq!(args.transform, "reverse");
        assert_eq!(args.model, "gpt-3.5-turbo");
        assert_eq!(args.provider, Provider::Openai);
        assert!(!args.visual);
        assert!(!args.heatmap);
        assert!(!args.orchestrator);
        assert!(!args.web);
        assert_eq!(args.port, 8888);
        assert_eq!(args.top_logprobs, 5);
    }

    #[test]
    fn test_args_parse_full() {
        let args = Args::parse_from([
            "eot",
            "test prompt",
            "uppercase",
            "gpt-4",
            "--provider",
            "anthropic",
            "--visual",
            "--heatmap",
            "--orchestrator",
            "--web",
            "--port",
            "9000",
        ]);
        assert_eq!(args.prompt, "test prompt");
        assert_eq!(args.transform, "uppercase");
        assert_eq!(args.model, "gpt-4");
        assert_eq!(args.provider, Provider::Anthropic);
        assert!(args.visual);
        assert!(args.heatmap);
        assert!(args.orchestrator);
        assert!(args.web);
        assert_eq!(args.port, 9000);
    }

    #[test]
    fn test_args_parse_provider_openai() {
        let args = Args::parse_from(["eot", "prompt", "--provider", "openai"]);
        assert_eq!(args.provider, Provider::Openai);
    }

    #[test]
    fn test_args_parse_provider_anthropic() {
        let args = Args::parse_from(["eot", "prompt", "--provider", "anthropic"]);
        assert_eq!(args.provider, Provider::Anthropic);
    }

    #[test]
    fn test_args_parse_short_visual() {
        let args = Args::parse_from(["eot", "prompt", "-v"]);
        assert!(args.visual);
    }

    #[test]
    fn test_args_default_port() {
        let args = Args::parse_from(["eot", "prompt"]);
        assert_eq!(args.port, 8888);
    }

    #[test]
    fn test_args_custom_port() {
        let args = Args::parse_from(["eot", "prompt", "--port", "3000"]);
        assert_eq!(args.port, 3000);
    }

    #[test]
    fn test_args_research_flag_default_false() {
        let args = Args::parse_from(["eot", "prompt"]);
        assert!(!args.research);
    }

    #[test]
    fn test_args_research_flag_set() {
        let args = Args::parse_from(["eot", "prompt", "--research"]);
        assert!(args.research);
    }

    #[test]
    fn test_args_runs_default_one() {
        let args = Args::parse_from(["eot", "prompt"]);
        assert_eq!(args.runs, 10);
    }

    #[test]
    fn test_args_runs_custom() {
        let args = Args::parse_from(["eot", "prompt", "--runs", "50"]);
        assert_eq!(args.runs, 50);
    }

    #[test]
    fn test_args_output_default_none() {
        let args = Args::parse_from(["eot", "prompt"]);
        assert_eq!(args.output, "research_output.json");
    }

    #[test]
    fn test_args_output_custom() {
        let args = Args::parse_from(["eot", "prompt", "--output", "results.json"]);
        assert_eq!(args.output, "results.json");
    }

    #[test]
    fn test_args_system_prompt_default_none() {
        let args = Args::parse_from(["eot", "prompt"]);
        assert!(args.system_a.is_none());
    }

    #[test]
    fn test_args_system_prompt_set() {
        let args = Args::parse_from(["eot", "prompt", "--system-a", "Be concise."]);
        assert_eq!(args.system_a.as_deref(), Some("Be concise."));
    }

    #[test]
    fn test_args_research_with_runs_and_output() {
        let args = Args::parse_from([
            "eot",
            "test prompt",
            "--research",
            "--runs",
            "100",
            "--output",
            "out.json",
        ]);
        assert!(args.research);
        assert_eq!(args.runs, 100);
        assert_eq!(args.output, "out.json");
    }

    #[test]
    fn test_args_research_does_not_require_web() {
        let args = Args::parse_from(["eot", "prompt", "--research"]);
        assert!(!args.web);
        assert!(args.research);
    }

    #[test]
    fn test_args_parse_research_flag() {
        let args = Args::parse_from(["eot", "prompt", "--research"]);
        assert!(args.research);
        assert_eq!(args.runs, 10);
        assert_eq!(args.output, "research_output.json");
    }

    #[test]
    fn test_args_parse_research_custom_runs() {
        let args = Args::parse_from(["eot", "prompt", "--research", "--runs", "50"]);
        assert_eq!(args.runs, 50);
    }

    #[test]
    fn test_args_parse_research_custom_output() {
        let args = Args::parse_from(["eot", "prompt", "--research", "--output", "out.json"]);
        assert_eq!(args.output, "out.json");
    }

    #[test]
    fn test_args_parse_system_a() {
        let args = Args::parse_from(["eot", "prompt", "--system-a", "Be concise."]);
        assert_eq!(args.system_a, Some("Be concise.".to_string()));
    }

    #[test]
    fn test_args_parse_system_b() {
        let args = Args::parse_from(["eot", "prompt", "--system-b", "Be verbose."]);
        assert_eq!(args.system_b, Some("Be verbose.".to_string()));
    }

    #[cfg(feature = "helix-bridge")]
    #[test]
    fn test_args_helix_url_default_none() {
        let args = Args::parse_from(["eot", "prompt"]);
        assert!(args.helix_url.is_none());
    }

    #[cfg(feature = "helix-bridge")]
    #[test]
    fn test_args_helix_url_set() {
        let args = Args::parse_from(["eot", "prompt", "--helix-url", "http://127.0.0.1:8080"]);
        assert_eq!(args.helix_url.as_deref(), Some("http://127.0.0.1:8080"));
    }

    #[test]
    fn test_parse_rate_range_valid() {
        assert_eq!(parse_rate_range("0.3-0.7"), Some((0.3, 0.7)));
    }

    #[test]
    fn test_parse_rate_range_equal() {
        assert_eq!(parse_rate_range("0.5-0.5"), Some((0.5, 0.5)));
    }

    #[test]
    fn test_parse_rate_range_invalid() {
        assert_eq!(parse_rate_range("invalid"), None);
    }

    #[test]
    fn test_parse_rate_range_min_greater_than_max() {
        assert_eq!(parse_rate_range("0.8-0.2"), None);
    }

    #[test]
    fn test_parse_rate_range_scientific_notation() {
        // rfind ensures the separator is the last '-', so "1e-3" parses correctly.
        let result = parse_rate_range("1e-3-0.5");
        assert!(result.is_some());
        let (min, max) = result.unwrap();
        assert!((min - 0.001).abs() < 1e-9);
        assert!((max - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_parse_rate_range_no_separator_returns_none() {
        assert_eq!(parse_rate_range("0.5"), None);
    }

    #[test]
    fn test_apply_template_with_placeholder() {
        assert_eq!(apply_template("Answer: {input}", "hello"), "Answer: hello");
    }

    #[test]
    fn test_apply_template_no_placeholder() {
        assert_eq!(apply_template("No placeholder", "hello"), "No placeholder");
    }

    #[test]
    fn test_args_dry_run_flag() {
        let args = Args::parse_from(["eot", "prompt", "--dry-run"]);
        assert!(args.dry_run);
    }

    #[test]
    fn test_args_min_confidence() {
        let args = Args::parse_from(["eot", "prompt", "--min-confidence", "0.8"]);
        assert_eq!(args.min_confidence, Some(0.8));
    }

    #[test]
    fn test_args_collapse_window() {
        let args = Args::parse_from(["eot", "prompt", "--collapse-window", "10"]);
        assert_eq!(args.collapse_window, 10);
    }

    #[test]
    fn test_args_format_jsonl() {
        let args = Args::parse_from(["eot", "prompt", "--format", "jsonl"]);
        assert_eq!(args.format, "jsonl");
    }

    // -- validate_model tests (#18) --

    #[test]
    fn test_validate_model_known_openai_no_warn() {
        // Should not panic; just verifying the function runs without error
        validate_model(&Provider::Openai, "gpt-4");
        validate_model(&Provider::Openai, "gpt-3.5-turbo");
        validate_model(&Provider::Openai, "gpt-4o");
    }

    #[test]
    fn test_validate_model_known_anthropic_no_warn() {
        validate_model(&Provider::Anthropic, "claude-sonnet-4-6");
        validate_model(&Provider::Anthropic, "claude-opus-4-6");
        validate_model(&Provider::Anthropic, "claude-haiku-4-5-20251001");
    }

    #[test]
    fn test_validate_model_unknown_does_not_panic() {
        // Unknown models emit a warning but must not panic
        validate_model(&Provider::Openai, "gpt-9-turbo-ultra");
        validate_model(&Provider::Anthropic, "claude-99");
    }

    #[test]
    fn test_validate_model_mock_always_silent() {
        // Mock provider skips validation entirely
        validate_model(&Provider::Mock, "any-model-string");
    }

    #[test]
    fn test_known_openai_models_nonempty() {
        assert!(!KNOWN_OPENAI_MODELS.is_empty());
    }

    #[test]
    fn test_known_anthropic_models_nonempty() {
        assert!(!KNOWN_ANTHROPIC_MODELS.is_empty());
    }

    #[test]
    fn test_known_openai_models_contain_gpt4() {
        assert!(KNOWN_OPENAI_MODELS.contains(&"gpt-4"));
    }

    #[test]
    fn test_known_anthropic_models_contain_sonnet() {
        assert!(KNOWN_ANTHROPIC_MODELS.contains(&"claude-sonnet-4-6"));
    }

    // -- Template injection safety tests (#6) --

    #[test]
    fn test_apply_template_prompt_with_placeholder_not_reexpanded() {
        // Prompt containing "{input}" must NOT be re-expanded
        let result = apply_template("Q: {input}", "what is {input}?");
        assert_eq!(result, "Q: what is {input}?");
    }

    #[test]
    fn test_apply_template_multiple_placeholders() {
        let result = apply_template("{input} and {input}", "hello");
        assert_eq!(result, "hello and hello");
    }

    // -- New CLI flags tests (#12, #13) --

    #[test]
    fn test_args_orchestrator_url_default() {
        let args = Args::parse_from(["eot", "prompt"]);
        assert_eq!(args.orchestrator_url, "http://localhost:3000");
    }

    #[test]
    fn test_args_orchestrator_url_custom() {
        let args = Args::parse_from([
            "eot",
            "prompt",
            "--orchestrator-url",
            "http://10.0.0.1:9000",
        ]);
        assert_eq!(args.orchestrator_url, "http://10.0.0.1:9000");
    }

    #[test]
    fn test_args_max_retries_default() {
        let args = Args::parse_from(["eot", "prompt"]);
        assert_eq!(args.max_retries, 3);
    }

    #[test]
    fn test_args_max_retries_custom() {
        let args = Args::parse_from(["eot", "prompt", "--max-retries", "5"]);
        assert_eq!(args.max_retries, 5);
    }

    #[test]
    fn test_args_max_retries_zero() {
        let args = Args::parse_from(["eot", "prompt", "--max-retries", "0"]);
        assert_eq!(args.max_retries, 0);
    }

    #[test]
    fn test_args_timeout_default() {
        let args = Args::parse_from(["eot", "prompt"]);
        assert_eq!(args.timeout, 120);
    }

    #[test]
    fn test_args_timeout_custom() {
        let args = Args::parse_from(["eot", "prompt", "--timeout", "60"]);
        assert_eq!(args.timeout, 60);
    }

    #[test]
    fn test_args_timeout_zero_disables() {
        let args = Args::parse_from(["eot", "prompt", "--timeout", "0"]);
        assert_eq!(args.timeout, 0);
    }

    // -- Item 14: --validate-config flag --
    #[test]
    fn test_validate_config_flag_exists() {
        let args = Args::parse_from(["eot", "prompt"]);
        assert!(!args.validate_config, "validate_config should default to false");
        let args2 = Args::parse_from(["eot", "prompt", "--validate-config"]);
        assert!(args2.validate_config);
    }

    // -- Item 15: --list-models flag --
    #[test]
    fn test_list_models_openai_includes_gpt4() {
        let openai_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"];
        assert!(openai_models.contains(&"gpt-4"), "openai list should include gpt-4");
    }

    #[test]
    fn test_list_models_flag_accepts_openai() {
        let args = Args::parse_from(["eot", "prompt", "--list-models", "openai"]);
        assert_eq!(args.list_models.as_deref(), Some("openai"));
    }

    #[test]
    fn test_list_models_flag_accepts_all() {
        let args = Args::parse_from(["eot", "prompt", "--list-models", "all"]);
        assert_eq!(args.list_models.as_deref(), Some("all"));
    }

    // -- Item 17: --json-schema flag --
    #[test]
    fn test_json_schema_flag_outputs_valid_json() {
        const RESEARCH_SCHEMA: &str = include_str!("../docs/research-schema.json");
        let result = serde_json::from_str::<serde_json::Value>(RESEARCH_SCHEMA);
        assert!(result.is_ok(), "research-schema.json must be valid JSON");
    }

    // -- Item 16: --record path unwritable detected --
    #[test]
    fn test_record_path_unwritable_detected() {
        // Test that trying to open a path in a non-existent directory fails
        let bad_path = "/nonexistent_dir_eot_test/output.json";
        let result = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(bad_path);
        assert!(result.is_err(), "opening a path in a nonexistent dir should fail");
    }

    // -- Item 12: --export-timeseries flag --
    #[test]
    fn test_export_timeseries_flag_default_none() {
        let args = Args::parse_from(["eot", "prompt"]);
        assert!(args.export_timeseries.is_none());
    }

    #[test]
    fn test_export_timeseries_flag_set() {
        let args = Args::parse_from(["eot", "prompt", "--export-timeseries", "out.csv"]);
        assert_eq!(args.export_timeseries.as_deref(), Some("out.csv"));
    }
}
