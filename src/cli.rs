use crate::providers::Provider;
use clap::Parser;

#[derive(Parser)]
#[command(name = "every-other-token")]
#[command(version = "4.0.0")]
#[command(about = "A real-time token stream mutator for LLM interpretability research")]
pub struct Args {
    /// Input prompt to send to the LLM
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

    /// System prompt B for A/B experiment mode
    #[arg(long)]
    pub system_b: Option<String>,
}

/// Select the appropriate default model for the given provider when the user
/// hasn't explicitly chosen one (i.e. the model is still the OpenAI default).
pub fn resolve_model(provider: &Provider, model: &str) -> String {
    if *provider == Provider::Anthropic && model == "gpt-3.5-turbo" {
        "claude-sonnet-4-20250514".to_string()
    } else {
        model.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_model_anthropic_default_swap() {
        assert_eq!(
            resolve_model(&Provider::Anthropic, "gpt-3.5-turbo"),
            "claude-sonnet-4-20250514"
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
}
