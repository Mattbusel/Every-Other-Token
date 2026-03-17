use clap::Parser;
use every_other_token::cli::Args;
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

    // Research mode: run N iterations, collect aggregate stats, write JSON
    if args.research {
        every_other_token::research::run_research(&args).await?;
        return Ok(());
    }

    let transform = Transform::from_str_loose(&args.transform)
        .map_err(|e| format!("Invalid transform: {}", e))?;

    // Auto-select a sensible default model when switching providers
    let model = every_other_token::cli::resolve_model(&args.provider, &args.model);

    let mut interceptor = {
        let mut i = TokenInterceptor::new(
            args.provider,
            transform,
            model,
            args.visual,
            args.heatmap,
            args.orchestrator,
        )?
        .with_rate(args.rate);
        if let Some(seed) = args.seed {
            i = i.with_seed(seed);
        }
        i
    };

    interceptor.top_logprobs = args.top_logprobs;
    interceptor.intercept_stream(&args.prompt).await?;

    Ok(())
}
