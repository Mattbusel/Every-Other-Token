use clap::CommandFactory;
use clap::Parser;
use every_other_token::cli::Args;
use every_other_token::transforms::Transform;
use every_other_token::TokenInterceptor;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Structured logging (#10): honours RUST_LOG env var; defaults to warn.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    let mut args = Args::parse();

    // Config file support (#16): apply .eot.toml / ~/.eot.toml defaults only
    // when the user has not explicitly overridden the corresponding CLI flag
    // (identified by comparing to its default value).
    {
        use every_other_token::config::EotConfig;
        use every_other_token::providers::Provider;
        let cfg = EotConfig::load();
        if args.provider == Provider::Openai {
            if let Some(ref p) = cfg.provider {
                if let Ok(prov) = <Provider as std::str::FromStr>::from_str(p) {
                    args.provider = prov;
                }
            }
        }
        if args.model == "gpt-3.5-turbo" {
            if let Some(m) = cfg.model {
                args.model = m;
            }
        }
        if args.transform == "reverse" {
            if let Some(t) = cfg.transform {
                args.transform = t;
            }
        }
        if args.rate.is_none() {
            if let Some(r) = cfg.rate {
                args.rate = Some(r);
            }
        }
        if args.port == 8888 {
            if let Some(p) = cfg.port {
                args.port = p;
            }
        }
        if args.top_logprobs == 5 {
            if let Some(t) = cfg.top_logprobs {
                args.top_logprobs = t;
            }
        }
        if args.system_a.is_none() {
            args.system_a = cfg.system_a;
        }
        if args.anthropic_max_tokens == 4096 {
            if let Some(t) = cfg.anthropic_max_tokens {
                args.anthropic_max_tokens = t;
            }
        }
        if args.api_key.is_none() {
            args.api_key = cfg.api_key;
        }
    }

    // Stdin support (#17): if prompt is "-", read from stdin.
    if args.prompt == "-" {
        use std::io::Read;
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        args.prompt = buf.trim().to_string();
    }

    // Model validation (#18): warn early about unknown model names.
    {
        let model = every_other_token::cli::resolve_model(&args.provider, &args.model);
        every_other_token::cli::validate_model(&args.provider, &model);
    }

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

    // Dry-run mode: validate transform and show sample token transformations
    if args.dry_run {
        let transform = every_other_token::transforms::Transform::from_str_loose(&args.transform)
            .map_err(|e| format!("Invalid transform: {}", e))?;
        println!("[dry-run] Transform: {:?}", transform);
        println!("[dry-run] Rate: {}", args.rate.unwrap_or(0.5));
        println!("[dry-run] Sample token transformations:");
        let sample_tokens = [
            "The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog",
        ];
        for (i, token) in sample_tokens.iter().enumerate() {
            let (result, label) = transform.apply_with_label(token);
            let marker = if (i % 2) == 1 {
                "→ TRANSFORMED"
            } else {
                "  (pass-through)"
            };
            println!("  [{}] {:15} {} {:?} ({})", i, token, marker, result, label);
        }
        return Ok(());
    }

    // Apply prompt template substitution if --template is set
    if let Some(ref tmpl) = args.template.clone() {
        args.prompt = every_other_token::cli::apply_template(tmpl, &args.prompt);
    }

    // Apply rate range: pick a random rate in [min, max] if --rate-range is set
    if let Some(ref range_str) = args.rate_range.clone() {
        if let Some((min, max)) = every_other_token::cli::parse_rate_range(range_str) {
            use rand::Rng;
            let rate = rand::thread_rng().gen_range(min..=max);
            args.rate = Some(rate);
            eprintln!(
                "[rate-range] Selected rate: {:.4} from range [{}, {}]",
                rate, min, max
            );
        } else {
            eprintln!(
                "[rate-range] Warning: could not parse rate range '{}', using --rate value",
                range_str
            );
        }
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

    // Load synonym overrides from file if provided
    if let Some(ref path) = args.synonym_file {
        every_other_token::transforms::load_synonym_overrides(path)
            .map_err(|e| format!("Failed to load synonym file '{}': {}", path, e))?;
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
        .with_rate(args.rate.unwrap_or(0.5));
        if let Some(seed) = args.seed {
            i = i.with_seed(seed);
        }
        i
    };

    interceptor.top_logprobs = args.top_logprobs;
    interceptor.json_stream = args.json_stream;
    interceptor.orchestrator_url = args.orchestrator_url.clone();
    interceptor.max_retries = args.max_retries;
    interceptor.min_confidence = args.min_confidence;
    interceptor.anthropic_max_tokens = args.anthropic_max_tokens;

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
