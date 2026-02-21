use crate::cli::Args;
use crate::transforms::Transform;
use crate::TokenInterceptor;
use serde::Serialize;
use tokio::sync::mpsc;

#[derive(Debug, Serialize)]
pub struct ResearchRun {
    pub run_index: u32,
    pub token_count: usize,
    pub transformed_count: usize,
    pub avg_confidence: Option<f64>,
    pub avg_perplexity: Option<f64>,
    pub vocab_diversity: f64,
}

#[derive(Debug, Serialize)]
pub struct ResearchOutput {
    pub schema_version: u8,
    pub prompt: String,
    pub provider: String,
    pub transform: String,
    pub runs: Vec<ResearchRun>,
    pub aggregate: ResearchAggregate,
}

#[derive(Debug, Serialize)]
pub struct ResearchAggregate {
    pub total_runs: u32,
    pub mean_token_count: f64,
    pub mean_confidence: Option<f64>,
    pub mean_perplexity: Option<f64>,
    pub mean_vocab_diversity: f64,
}

pub async fn run_research(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let provider = args.provider.clone();
    let transform_str = args.transform.clone();
    let transform =
        Transform::from_str_loose(&transform_str).map_err(|e| format!("Invalid transform: {e}"))?;
    let model = crate::cli::resolve_model(&provider, &args.model);

    eprintln!(
        "[research] starting {} runs -- provider={} transform={} model={}",
        args.runs, provider, transform_str, model
    );

    let mut runs: Vec<ResearchRun> = Vec::with_capacity(args.runs as usize);

    for i in 0..args.runs {
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
        interceptor.system_prompt = args.system_a.clone();

        interceptor.intercept_stream(&args.prompt).await?;
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

        let unique_tokens: std::collections::HashSet<&str> =
            events.iter().map(|e| e.original.as_str()).collect();
        let vocab_diversity = if token_count == 0 {
            0.0
        } else {
            unique_tokens.len() as f64 / token_count as f64
        };

        runs.push(ResearchRun {
            run_index: i,
            token_count,
            transformed_count,
            avg_confidence,
            avg_perplexity,
            vocab_diversity,
        });
    }

    let aggregate = build_aggregate(args.runs, &runs);

    let output = ResearchOutput {
        schema_version: 1,
        prompt: args.prompt.clone(),
        provider: provider.to_string(),
        transform: transform_str,
        runs,
        aggregate,
    };

    let json = serde_json::to_string_pretty(&output)?;
    std::fs::write(&args.output, &json)?;
    eprintln!("[research] wrote {} bytes to {}", json.len(), args.output);

    Ok(())
}

fn build_aggregate(total_runs: u32, runs: &[ResearchRun]) -> ResearchAggregate {
    if runs.is_empty() {
        return ResearchAggregate {
            total_runs,
            mean_token_count: 0.0,
            mean_confidence: None,
            mean_perplexity: None,
            mean_vocab_diversity: 0.0,
        };
    }

    let mean_token_count =
        runs.iter().map(|r| r.token_count as f64).sum::<f64>() / runs.len() as f64;

    let confidences: Vec<f64> = runs.iter().filter_map(|r| r.avg_confidence).collect();
    let mean_confidence = if confidences.is_empty() {
        None
    } else {
        Some(confidences.iter().sum::<f64>() / confidences.len() as f64)
    };

    let perplexities: Vec<f64> = runs.iter().filter_map(|r| r.avg_perplexity).collect();
    let mean_perplexity = if perplexities.is_empty() {
        None
    } else {
        Some(perplexities.iter().sum::<f64>() / perplexities.len() as f64)
    };

    let mean_vocab_diversity =
        runs.iter().map(|r| r.vocab_diversity).sum::<f64>() / runs.len() as f64;

    ResearchAggregate {
        total_runs,
        mean_token_count,
        mean_confidence,
        mean_perplexity,
        mean_vocab_diversity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_runs(
        data: &[(usize, usize, Option<f64>, Option<f64>, f64)],
    ) -> Vec<ResearchRun> {
        data.iter()
            .enumerate()
            .map(|(i, &(tc, tr, conf, perp, vd))| ResearchRun {
                run_index: i as u32,
                token_count: tc,
                transformed_count: tr,
                avg_confidence: conf,
                avg_perplexity: perp,
                vocab_diversity: vd,
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
            schema_version: 1,
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
            },
        };
        let json = serde_json::to_string(&output).expect("serialize");
        assert!(json.contains("schema_version"));
        assert!(json.contains(r#""provider":"openai""#));
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
}
