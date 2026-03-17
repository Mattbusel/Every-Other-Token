use clap::CommandFactory;
use clap::Parser;
use every_other_token::cli::Args;
use every_other_token::transforms::Transform;
use every_other_token::TokenInterceptor;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Shell completion generation
    if let Some(shell) = args.completions {
        clap_complete::generate(
            shell,
            &mut Args::command(),
            "every-other-token",
            &mut std::io::stdout(),
        );
        return Ok(());
    }

    // Web UI mode
    if args.web {
        tokio::select! {
            result = every_other_token::web::serve(args.port, &args) => {
                result?;
            }
            _ = tokio::signal::ctrl_c() => {
                eprintln!("\n[eot] shutting down gracefully");
            }
        }
        return Ok(());
    }

    // Research mode: run N iterations, collect aggregate stats, write JSON
    if args.research {
        if args.prompt_file.is_some() {
            tokio::select! {
                result = every_other_token::research::run_research_suite(&args) => {
                    result?;
                }
                _ = tokio::signal::ctrl_c() => {
                    eprintln!("\n[eot] interrupted");
                }
            }
        } else {
            tokio::select! {
                result = every_other_token::research::run_research(&args) => {
                    result?;
                }
                _ = tokio::signal::ctrl_c() => {
                    eprintln!("\n[eot] interrupted — partial results may not have been written");
                }
            }
        }
        return Ok(());
    }

    // Diff terminal mode
    if args.diff_terminal {
        every_other_token::research::run_diff_terminal(&args).await?;
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
    interceptor.json_stream = args.json_stream;

    tokio::select! {
        result = interceptor.intercept_stream(&args.prompt) => {
            result?;
        }
        _ = tokio::signal::ctrl_c() => {
            eprintln!("\n[eot] shutting down gracefully");
        }
    }

    Ok(())
}
