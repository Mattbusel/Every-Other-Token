use clap::Parser;
use every_other_token::cli::Args;
use every_other_token::providers::Provider;
use every_other_token::transforms::Transform;
use every_other_token::TokenInterceptor;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Web UI mode
    if args.web {
        every_other_token::web::serve(args.port, &args).await?;
        return Ok(());
    }

    let transform = Transform::from_str_loose(&args.transform)
        .map_err(|e| format!("Invalid transform: {}", e))?;

    // Auto-select a sensible default model when switching providers
    let model = if args.provider == Provider::Anthropic && args.model == "gpt-3.5-turbo" {
        "claude-sonnet-4-20250514".to_string()
    } else {
        args.model
    };

    let mut interceptor = TokenInterceptor::new(
        args.provider,
        transform,
        model,
        args.visual,
        args.heatmap,
        args.orchestrator,
    )?;

    interceptor.intercept_stream(&args.prompt).await?;

    Ok(())
}
