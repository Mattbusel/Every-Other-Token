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
    pub std_dev_confidence: Option<f64>,
    pub std_dev_perplexity: Option<f64>,
    pub std_dev_token_count: f64,
    pub confidence_interval_95_confidence: Option<(f64, f64)>,
    pub confidence_interval_95_perplexity: Option<(f64, f64)>,
}

fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance =
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

fn ci_95(mean: f64, sd: f64, n: usize) -> (f64, f64) {
    let margin = 1.96 * sd / (n as f64).sqrt();
    (mean - margin, mean + margin)
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

    // Open SQLite store if requested
    let store = if let Some(db_path) = &args.db {
        Some(crate::store::ExperimentStore::open(db_path)?)
    } else {
        None
    };

    // Determine experiment id in DB
    let exp_id: Option<i64> = if let Some(ref s) = store {
        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let created_at = format!("{}", now);
        let id = s.insert_experiment(
            &created_at,
            &args.prompt,
            &provider.to_string(),
            &transform_str,
            &model,
        )?;
        Some(id)
    } else {
        None
    };

    // Create heatmap exporter if requested
    let mut heatmap_exporter = args
        .heatmap_export
        .as_ref()
        .map(|_| crate::heatmap::HeatmapExporter::new());

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

        // Persist run to DB if store is open
        if let (Some(ref s), Some(eid)) = (&store, exp_id) {
            s.insert_run(
                eid,
                i,
                token_count,
                transformed_count,
                avg_confidence,
                avg_perplexity,
                vocab_diversity,
            )?;
        }

        // Record events for heatmap
        if let Some(ref mut exporter) = heatmap_exporter {
            exporter.record_run(&events);
        }

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

    // A/B significance test
    if args.significance && args.system_b.is_some() {
        let even_confs: Vec<f64> = runs
            .iter()
            .filter(|r| r.run_index % 2 == 0)
            .filter_map(|r| r.avg_confidence)
            .collect();
        let odd_confs: Vec<f64> = runs
            .iter()
            .filter(|r| r.run_index % 2 == 1)
            .filter_map(|r| r.avg_confidence)
            .collect();

        if !even_confs.is_empty() && !odd_confs.is_empty() {
            if let Some(p) = two_sample_t_test(&even_confs, &odd_confs) {
                if p < 0.05 {
                    eprintln!(
                        "[significance] WARNING: A/B confidence difference is statistically significant (p={:.4})",
                        p
                    );
                } else {
                    eprintln!(
                        "[significance] A/B confidence difference is NOT significant (p={:.4})",
                        p
                    );
                }
            }
        }
    }

    // Export heatmap if requested
    if let (Some(ref exporter), Some(ref path)) = (heatmap_exporter, &args.heatmap_export) {
        exporter.export_csv(path)?;
        eprintln!("[research] heatmap exported to {}", path);
    }

    let output = ResearchOutput {
        schema_version: 2,
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
            std_dev_confidence: None,
            std_dev_perplexity: None,
            std_dev_token_count: 0.0,
            confidence_interval_95_confidence: None,
            confidence_interval_95_perplexity: None,
        };
    }

    let token_counts: Vec<f64> = runs.iter().map(|r| r.token_count as f64).collect();
    let mean_token_count = token_counts.iter().sum::<f64>() / token_counts.len() as f64;
    let std_dev_token_count = std_dev(&token_counts);

    let confidences: Vec<f64> = runs.iter().filter_map(|r| r.avg_confidence).collect();
    let (mean_confidence, std_dev_confidence, confidence_interval_95_confidence) =
        if confidences.is_empty() {
            (None, None, None)
        } else {
            let mean = confidences.iter().sum::<f64>() / confidences.len() as f64;
            let sd = std_dev(&confidences);
            let ci = if confidences.len() >= 2 {
                Some(ci_95(mean, sd, confidences.len()))
            } else {
                None
            };
            (Some(mean), Some(sd), ci)
        };

    let perplexities: Vec<f64> = runs.iter().filter_map(|r| r.avg_perplexity).collect();
    let (mean_perplexity, std_dev_perplexity, confidence_interval_95_perplexity) =
        if perplexities.is_empty() {
            (None, None, None)
        } else {
            let mean = perplexities.iter().sum::<f64>() / perplexities.len() as f64;
            let sd = std_dev(&perplexities);
            let ci = if perplexities.len() >= 2 {
                Some(ci_95(mean, sd, perplexities.len()))
            } else {
                None
            };
            (Some(mean), Some(sd), ci)
        };

    let mean_vocab_diversity =
        runs.iter().map(|r| r.vocab_diversity).sum::<f64>() / runs.len() as f64;

    ResearchAggregate {
        total_runs,
        mean_token_count,
        mean_confidence,
        mean_perplexity,
        mean_vocab_diversity,
        std_dev_confidence,
        std_dev_perplexity,
        std_dev_token_count,
        confidence_interval_95_confidence,
        confidence_interval_95_perplexity,
    }
}

/// Simple two-sample Welch's t-test. Returns approximate p-value (two-tailed).
/// Returns None if variance is zero or samples too small.
fn two_sample_t_test(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() < 2 || b.len() < 2 {
        return None;
    }
    let mean_a = a.iter().sum::<f64>() / a.len() as f64;
    let mean_b = b.iter().sum::<f64>() / b.len() as f64;
    let var_a = a.iter().map(|v| (v - mean_a).powi(2)).sum::<f64>() / (a.len() - 1) as f64;
    let var_b = b.iter().map(|v| (v - mean_b).powi(2)).sum::<f64>() / (b.len() - 1) as f64;
    let se = ((var_a / a.len() as f64) + (var_b / b.len() as f64)).sqrt();
    if se == 0.0 {
        return None;
    }
    let t = (mean_a - mean_b).abs() / se;
    // Approximate p-value via normal distribution (large-sample approximation)
    let p = 2.0 * (1.0 - normal_cdf(t));
    Some(p)
}

/// Approximation of the standard normal CDF using Abramowitz & Stegun formula.
fn normal_cdf(z: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * z.abs());
    let poly = t
        * (0.319381530
            + t * (-0.356563782
                + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let pdf = (-z * z / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let p = 1.0 - pdf * poly;
    if z >= 0.0 {
        p
    } else {
        1.0 - p
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
            schema_version: 2,
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
                std_dev_confidence: None,
                std_dev_perplexity: None,
                std_dev_token_count: 0.0,
                confidence_interval_95_confidence: None,
                confidence_interval_95_perplexity: None,
            },
        };
        let json = serde_json::to_string(&output).expect("serialize");
        assert!(json.contains("schema_version"));
        assert!(json.contains(r#""provider":"openai""#));
        assert!(json.contains(r#""schema_version":2"#));
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

    #[test]
    fn test_build_aggregate_std_dev_fields_present() {
        let runs = make_runs(&[
            (10, 5, Some(0.8), Some(1.2), 0.6),
            (20, 10, Some(0.9), Some(1.0), 0.7),
        ]);
        let agg = build_aggregate(2, &runs);
        assert!(agg.std_dev_confidence.is_some());
        assert!(agg.std_dev_perplexity.is_some());
        assert!(agg.confidence_interval_95_confidence.is_some());
        assert!(agg.confidence_interval_95_perplexity.is_some());
    }

    #[test]
    fn test_std_dev_calculation() {
        // Uses Bessel's correction (sample std dev, n-1).
        // For [2,4,4,4,5,5,7,9]: mean=5, sum-sq-dev=32, variance=32/7 ≈ 4.571
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let sd = std_dev(&values);
        let expected = (32.0f64 / 7.0).sqrt();
        assert!((sd - expected).abs() < 0.001);
    }

    #[test]
    fn test_two_sample_t_test_same_means() {
        let a = vec![1.0, 1.0, 1.0, 1.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        // Zero variance — should return None
        assert!(two_sample_t_test(&a, &b).is_none());
    }
}
