use clap::Parser;
use every_other_token::cli::Args;
use every_other_token::providers::Provider;
use every_other_token::transforms::Transform;
use every_other_token::{run_research_headless, TokenInterceptor};

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

    let model = if args.provider == Provider::Anthropic && args.model == "gpt-3.5-turbo" {
        "claude-sonnet-4-20250514".to_string()
    } else {
        args.model.clone()
    };

    // Headless research mode
    if args.research {
        let session = run_research_headless(
            &args.prompt,
            args.provider,
            transform,
            model,
            args.runs,
        )
        .await?;
        let json = serde_json::to_string_pretty(&session)?;
        if let Some(path) = &args.output {
            std::fs::write(path, &json)?;
            eprintln!("Research results written to {}", path);
        } else {
            println!("{}", json);
        }
        return Ok(());
    }

    let mut interceptor = TokenInterceptor::new(
        args.provider,
        transform,
        model,
        args.visual,
        args.heatmap,
        args.orchestrator,
    )?;

    if let Some(sys) = args.system_prompt {
        interceptor.system_prompt = Some(sys);
    }

    interceptor.intercept_stream(&args.prompt).await?;

    Ok(())
}
