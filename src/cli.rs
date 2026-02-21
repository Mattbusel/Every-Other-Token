use clap::Parser;
use crate::providers::Provider;

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
        assert_eq!(
            resolve_model(&Provider::Openai, "gpt-4"),
            "gpt-4"
        );
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
            "eot", "test prompt", "uppercase", "gpt-4",
            "--provider", "anthropic", "--visual", "--heatmap",
            "--orchestrator", "--web", "--port", "9000",
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
}
