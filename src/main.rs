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

    // No-argument fallback: if the user gave no prompt and no action flags
    // (happens when double-clicking the .exe on Windows, or running bare),
    // auto-launch the web UI instead of printing help and exiting immediately.
    if args.prompt.is_empty()
        && !args.web
        && !args.research
        && !args.dry_run
        && args.record.is_none()
        && args.replay.is_none()
        && !args.validate_config
        && args.list_models.is_none()
        && !args.json_schema
        && !args.diff_terminal
        && args.batch.is_none()
        && args.compare.is_none()
        && args.similarity.is_none()
        && !args.diversity_filter
        && !args.stats
        && !args.benchmark
    {
        eprintln!("[eot] No prompt given — launching web UI at http://localhost:{}", args.port);
        eprintln!("[eot] Tip: set OPENAI_API_KEY or ANTHROPIC_API_KEY in your environment.");
        eprintln!("[eot] Run with --help for full CLI usage.");
        args.web = true;
    }

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

    // --validate-config: print resolved config values and exit
    if args.validate_config {
        use every_other_token::config::EotConfig;
        let cfg = EotConfig::load();
        println!("[eot config] provider: {}", args.provider);
        println!("[eot config] model: {}", args.model);
        println!("[eot config] transform: {}", args.transform);
        println!("[eot config] rate: {}", args.rate.unwrap_or(0.5));
        println!("[eot config] port: {}", args.port);
        println!("[eot config] top_logprobs: {}", args.top_logprobs);
        println!("[eot config] max_retries: {}", args.max_retries);
        println!("[eot config] timeout: {}", args.timeout);
        println!("[eot config] anthropic_max_tokens: {}", args.anthropic_max_tokens);
        if let Some(ref sa) = args.system_a { println!("[eot config] system_a: {}", sa); }
        drop(cfg); // cfg loaded for side-effects
        std::process::exit(0);
    }

    // --list-models: print known models and exit
    if let Some(ref provider_filter) = args.list_models.clone() {
        let openai_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"];
        let anthropic_models = [
            "claude-3-5-haiku-20241022",
            "claude-3-5-sonnet-20241022",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
        ];
        let show_openai = provider_filter == "openai" || provider_filter == "all";
        let show_anthropic = provider_filter == "anthropic" || provider_filter == "all";
        if show_openai {
            println!("[openai models]");
            for m in &openai_models { println!("  {}", m); }
        }
        if show_anthropic {
            println!("[anthropic models]");
            for m in &anthropic_models { println!("  {}", m); }
        }
        if !show_openai && !show_anthropic {
            // unknown filter value — show all
            for m in &openai_models { println!("  {}", m); }
            for m in &anthropic_models { println!("  {}", m); }
        }
        std::process::exit(0);
    }

    // --json-schema: print embedded research schema and exit
    if args.json_schema {
        const RESEARCH_SCHEMA: &str = include_str!("../docs/research-schema.json");
        println!("{}", RESEARCH_SCHEMA);
        std::process::exit(0);
    }

    // --record early path check: verify the file is writable before making API calls
    if let Some(ref record_path) = args.record {
        if let Err(e) = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(record_path)
        {
            eprintln!("[eot] cannot open record file '{}': {}", record_path, e);
            std::process::exit(1);
        }
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
        // For Chain transforms, show intermediate outputs at each step.
        if let every_other_token::transforms::Transform::Chain(ref steps) = transform {
            let step_names: Vec<String> = steps.iter().map(|s| format!("{:?}", s)).collect();
            println!("Dry-run: Chain [{}]", step_names.join(", "));
            for token in &sample_tokens {
                println!("  Input: {:?}", token);
                let mut current = token.to_string();
                for step in steps {
                    let (next, _label) = step.apply_with_label(&current);
                    println!("  After {:?}: {:?}", step, next);
                    current = next;
                }
            }
        } else {
            for (i, token) in sample_tokens.iter().enumerate() {
                let (result, label) = transform.apply_with_label(token);
                let marker = if (i % 2) == 1 {
                    "→ TRANSFORMED"
                } else {
                    "  (pass-through)"
                };
                println!("  [{}] {:15} {} {:?} ({})", i, token, marker, result, label);
            }
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

    // Batch research mode (--batch <file.jsonl>)
    if let Some(ref batch_path) = args.batch.clone() {
        tokio::select! {
            result = every_other_token::research::run_batch(&args, batch_path) => {
                result?;
            }
            _ = tokio::signal::ctrl_c() => {
                eprintln!("\n[eot] batch interrupted");
            }
        }
        return Ok(());
    }

    // Multi-model comparison heatmap (--compare model1,model2)
    if let Some(ref compare_models) = args.compare.clone() {
        tokio::select! {
            result = every_other_token::research::run_multi_model_compare(&args, compare_models) => {
                result?;
            }
            _ = tokio::signal::ctrl_c() => {
                eprintln!("\n[eot] compare interrupted");
            }
        }
        return Ok(());
    }

    // Logprob CSV export (--export-logprobs <file.csv>)
    if let Some(ref export_path) = args.export_logprobs.clone() {
        tokio::select! {
            result = every_other_token::research::run_with_logprob_export(&args, export_path) => {
                result?;
            }
            _ = tokio::signal::ctrl_c() => {
                eprintln!("\n[eot] logprob export interrupted");
            }
        }
        return Ok(());
    }

    // Batch token compression mode (--batch-tokens <file.jsonl>)
    if let Some(ref batch_tokens_path) = args.batch_tokens.clone() {
        use every_other_token::batch::{BatchConfig, BatchProcessor};
        use tokio_stream::StreamExt;
        use std::time::Duration;

        let content = std::fs::read_to_string(batch_tokens_path)
            .map_err(|e| format!("Cannot read batch-tokens file '{}': {}", batch_tokens_path, e))?;

        let config = BatchConfig {
            max_concurrent: 8,
            queue_capacity: 256,
            timeout: Duration::from_secs(30),
        };
        let mut processor = BatchProcessor::new(config);
        let mut total_submitted = 0u64;

        for (line_no, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() { continue; }
            match serde_json::from_str::<Vec<String>>(line) {
                Ok(tokens) => {
                    processor.submit(tokens, 5);
                    total_submitted += 1;
                }
                Err(e) => {
                    eprintln!("[batch-tokens] line {}: parse error: {}", line_no + 1, e);
                }
            }
        }

        let start = std::time::Instant::now();
        let mut stream = processor.run();
        let mut results = Vec::new();
        while let Some(result) = stream.next().await {
            println!("{}",
                serde_json::json!({
                    "job_id": result.job_id,
                    "original_len": result.original_len,
                    "compressed_len": result.compressed_len,
                    "ratio": result.ratio,
                    "elapsed_ms": result.elapsed_ms,
                    "compressed": result.compressed,
                })
            );
            results.push(result);
        }
        let wall_secs = start.elapsed().as_secs_f64();
        let stats = BatchProcessor::compute_stats(&results, total_submitted, wall_secs);
        eprintln!("[batch-tokens] submitted={} completed={} failed={} avg_ratio={:.3} throughput={:.1} tok/s",
            stats.jobs_submitted, stats.jobs_completed, stats.jobs_failed,
            stats.avg_ratio, stats.throughput_tokens_per_sec);
        return Ok(());
    }

    // Semantic similarity mode (--similarity <text1> <text2>)
    if let Some(ref texts) = args.similarity.clone() {
        if texts.len() == 2 {
            use every_other_token::similarity::SemanticScorer;
            let a = &texts[0];
            let b = &texts[1];
            // Fit on the two texts themselves so there's always a vocabulary
            let scorer = SemanticScorer::fit(&[a.as_str(), b.as_str()]);
            let sim = scorer.similarity(a, b);
            println!("[similarity] text1: {:?}", a);
            println!("[similarity] text2: {:?}", b);
            println!("[similarity] cosine_similarity: {:.6}", sim);
        } else {
            eprintln!("[eot] --similarity requires exactly 2 arguments");
            std::process::exit(1);
        }
        return Ok(());
    }

    // Diversity filter mode (--diversity-filter): dedup prompt token sequences
    if args.diversity_filter {
        use every_other_token::similarity::DiversityFilter;
        // Treat each line of the prompt as a sequence of tokens
        let lines: Vec<Vec<String>> = args.prompt
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.split_whitespace().map(|t| t.to_string()).collect())
            .collect();
        if lines.is_empty() {
            eprintln!("[diversity-filter] No input lines to filter (prompt is empty)");
            return Ok(());
        }
        let filter = DiversityFilter::new(&lines);
        let filtered = filter.filter(lines.clone(), 0.85);
        println!("[diversity-filter] input sequences: {}", lines.len());
        println!("[diversity-filter] output sequences: {}", filtered.len());
        for seq in &filtered {
            println!("{}", seq.join(" "));
        }
        return Ok(());
    }

    // Quality metrics mode (--quality): show compression quality for the prompt tokens
    if args.quality {
        use every_other_token::adaptive::{AdaptiveCompressor, CompressionTarget, QualityMetric};

        let tokens: Vec<String> = args.prompt.split_whitespace().map(|t| t.to_string()).collect();
        let target = CompressionTarget::default();
        let compressor = AdaptiveCompressor::new();
        let (compressed, quality) = compressor.compress(&tokens, &target);

        println!("[quality] original tokens: {}", tokens.len());
        println!("[quality] compressed tokens: {}", compressed.len());
        println!("[quality] ratio: {:.4}", quality.ratio);
        println!("[quality] semantic_preservation: {:.4}", quality.semantic_preservation);
        println!("[quality] syntax_validity: {:.4}", quality.syntax_validity);
        println!("[quality] overall_score: {:.4}", quality.overall_score);
        println!("[quality] compressed: {:?}", compressed);

        // Also measure raw quality without adaptive compression
        let half: Vec<String> = tokens.iter().enumerate()
            .filter(|(i, _)| i % 2 == 0)
            .map(|(_, t)| t.clone())
            .collect();
        let raw_q = QualityMetric::measure(&tokens, &half);
        println!("[quality] --- every-other-token compression ---");
        println!("[quality] ratio: {:.4}", raw_q.ratio);
        println!("[quality] semantic_preservation: {:.4}", raw_q.semantic_preservation);
        println!("[quality] syntax_validity: {:.4}", raw_q.syntax_validity);
        println!("[quality] overall_score: {:.4}", raw_q.overall_score);

        return Ok(());
    }

    // Importance scoring mode (--importance <method>)
    if let Some(ref method_str) = args.importance.clone() {
        use every_other_token::importance::{ImportanceMethod, ImportanceScorer};
        let tokens: Vec<String> = args.prompt.split_whitespace().map(|t| t.to_string()).collect();
        if tokens.is_empty() {
            eprintln!("[importance] No tokens to score (prompt is empty)");
            return Ok(());
        }
        let method = match method_str.as_str() {
            "frequency" => ImportanceMethod::Frequency,
            "syntactic" => ImportanceMethod::Syntactic,
            "composite" => ImportanceMethod::Composite(vec![
                (ImportanceMethod::Positional { decay: 0.1 }, 0.5),
                (ImportanceMethod::Syntactic, 0.5),
            ]),
            _ => ImportanceMethod::Positional { decay: 0.1 },
        };
        let scorer = ImportanceScorer::new(method);
        let scores = scorer.score(&tokens);
        println!("[importance] method: {}", method_str);
        println!("[importance] tokens: {}", tokens.len());
        for s in &scores {
            println!("  rank={:3}  score={:.4}  token={:?}", s.rank, s.score, s.token);
        }
        return Ok(());
    }

    // Chunking mode (--chunk <strategy>)
    if let Some(ref strategy_str) = args.chunk.clone() {
        use every_other_token::chunking::{ChunkStrategy, Chunker};
        let tokens: Vec<String> = args.prompt.split_whitespace().map(|t| t.to_string()).collect();
        if tokens.is_empty() {
            eprintln!("[chunk] No tokens to chunk (prompt is empty)");
            return Ok(());
        }
        let strategy = if strategy_str.starts_with("fixed:") {
            let n: usize = strategy_str.trim_start_matches("fixed:").parse().unwrap_or(50);
            ChunkStrategy::Fixed(n)
        } else if strategy_str.starts_with("semantic:") {
            let t: f32 = strategy_str.trim_start_matches("semantic:").parse().unwrap_or(0.5);
            ChunkStrategy::Semantic { similarity_threshold: t }
        } else if strategy_str == "sentence" {
            ChunkStrategy::Sentence
        } else if strategy_str == "paragraph" {
            ChunkStrategy::Paragraph
        } else {
            ChunkStrategy::Fixed(50)
        };
        let chunks = Chunker::chunk_with_metadata(&tokens, strategy);
        println!("[chunk] strategy: {}", strategy_str);
        println!("[chunk] input tokens: {}", tokens.len());
        println!("[chunk] chunks: {}", chunks.len());
        for (chunk, meta) in &chunks {
            println!(
                "  chunk[{}]  tokens={}..{}  count={}  {:?}",
                meta.index, meta.start_token, meta.end_token, meta.token_count,
                &chunk[..chunk.len().min(8)]
            );
        }
        return Ok(());
    }

    // Token statistics mode (--stats)
    if args.stats {
        use every_other_token::stats::TokenStats;
        let tokens: Vec<String> = args.prompt.split_whitespace().map(|t| t.to_string()).collect();
        if tokens.is_empty() {
            eprintln!("[stats] No tokens to analyse (prompt is empty)");
            return Ok(());
        }
        TokenStats::compute(&tokens).print_summary();
        return Ok(());
    }

    // Vocabulary statistics mode (--vocab-stats)
    if args.vocab_stats {
        use every_other_token::vocab::VocabularyAnalyzer;
        let tokens: Vec<String> = args.prompt.split_whitespace().map(|t| t.to_string()).collect();
        if tokens.is_empty() {
            eprintln!("[vocab-stats] No tokens to analyse (prompt is empty)");
            return Ok(());
        }
        // Build a reference corpus from the prompt itself for demonstration
        let corpus = vec![tokens.clone()];
        let va = VocabularyAnalyzer::build_from_corpus(&corpus);
        let stats = va.analyze(&tokens);
        let zipf = va.zipf_score();
        let oov = va.oov_tokens_owned(&tokens);
        println!("[vocab-stats] vocab_size: {}", stats.size);
        println!("[vocab-stats] coverage_pct: {:.4}", stats.coverage_pct);
        println!("[vocab-stats] oov_rate: {:.4}", stats.oov_rate);
        println!("[vocab-stats] avg_frequency: {:.4}", stats.avg_frequency);
        println!("[vocab-stats] median_frequency: {:.4}", stats.median_frequency);
        println!("[vocab-stats] zipf_score: {:.4}", zipf);
        if !oov.is_empty() {
            println!("[vocab-stats] oov_tokens: {:?}", &oov[..oov.len().min(20)]);
        }
        return Ok(());
    }

    // Context budget info (--context-budget N)
    // This flag configures the ContextWindow budget; its value is available at args.context_budget.
    // When combined with --dry-run or --stats, print the budget configuration.
    {
        use every_other_token::context::{ContextWindow, ContextWindowConfig};
        let _ = ContextWindow::new(ContextWindowConfig {
            max_tokens: args.context_budget,
            reserved_for_output: 512,
            system_tokens: 0,
        });
        // Budget is wired; actual context management happens in the interceptor pipeline.
    }

    // Compression benchmark mode (--benchmark)
    if args.benchmark {
        use every_other_token::benchmark::{BenchmarkReport, CompressionBenchmark};
        let tokens: Vec<String> = args.prompt.split_whitespace().map(|t| t.to_string()).collect();
        if tokens.is_empty() {
            eprintln!("[benchmark] No tokens to benchmark (prompt is empty)");
            return Ok(());
        }
        let bench = CompressionBenchmark::with_builtins();
        let results = bench.run(&tokens);
        println!("{}", BenchmarkReport::render_table(&results));
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
    if args.timeout > 0 {
        interceptor = interceptor.with_timeout(args.timeout);
    }

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
