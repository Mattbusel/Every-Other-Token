//! Token Attribution Exporter
//!
//! Exports per-token confidence and attribution data to formats compatible
//! with interpretability tools (LIME, Captum, custom dashboards, etc.).
//!
//! Three output formats are supported:
//!
//! * **JSONL** — one [`SequenceAttribution`] record per line, suitable for
//!   streaming ingestion by Python interpretability pipelines.
//! * **CSV** — flat table with one row per token, importable into spreadsheets
//!   or pandas DataFrames.
//! * **HTML heatmap** — self-contained single-file report; no external
//!   stylesheets or JavaScript libraries required.
//!
//! # Example
//!
//! ```
//! use every_other_token::attribution::{
//!     AttributionExporter, SequenceAttribution, TokenAttribution,
//! };
//!
//! let seq = SequenceAttribution {
//!     prompt: "What is recursion?".into(),
//!     model: "gpt-4o".into(),
//!     timestamp: "2026-03-22T00:00:00Z".into(),
//!     tokens: vec![
//!         TokenAttribution {
//!             token: "Recursion".into(),
//!             position: 0,
//!             confidence: 0.92,
//!             perplexity: 1.09,
//!             attribution_score: 0.45,
//!             top_alternatives: vec![("Iteration".into(), 0.05)],
//!         },
//!     ],
//!     aggregate_confidence: 0.92,
//!     aggregate_perplexity: 1.09,
//!     semantic_drift: 0.0,
//! };
//!
//! let exporter = AttributionExporter::new();
//! let csv = exporter.to_csv(&seq);
//! assert!(csv.contains("Recursion"));
//! ```

use serde::{Deserialize, Serialize};

// ── Data types ────────────────────────────────────────────────────────────────

/// Per-token attribution record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenAttribution {
    /// The raw token text as returned by the model.
    pub token: String,
    /// Zero-based position in the output sequence.
    pub position: usize,
    /// Model confidence at this position: `exp(logprob)` ∈ \[0.0, 1.0\].
    pub confidence: f32,
    /// Per-token perplexity: `exp(-logprob)` ≥ 1.0.
    pub perplexity: f32,
    /// Contribution of this token to the final output quality estimate.
    /// Computed externally (e.g. by attention rollout or occlusion).
    pub attribution_score: f32,
    /// Up to 5 alternative tokens with their logprob-derived probabilities.
    pub top_alternatives: Vec<(String, f32)>,
}

/// Full sequence attribution: one complete prompt/response pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceAttribution {
    /// The original input prompt.
    pub prompt: String,
    /// Model identifier, e.g. `"gpt-4o"` or `"claude-sonnet-4-6"`.
    pub model: String,
    /// ISO 8601 timestamp string.
    pub timestamp: String,
    /// Per-token attribution records in generation order.
    pub tokens: Vec<TokenAttribution>,
    /// Mean confidence across all tokens.
    pub aggregate_confidence: f32,
    /// Mean perplexity across all tokens.
    pub aggregate_perplexity: f32,
    /// Semantic drift: confidence decay from start to end of sequence.
    /// Positive values indicate declining confidence; negative values
    /// indicate increasing confidence.
    pub semantic_drift: f32,
}

impl SequenceAttribution {
    /// Recalculate `aggregate_confidence`, `aggregate_perplexity`, and
    /// `semantic_drift` from the current token list in place.
    pub fn recompute_aggregates(&mut self) {
        let exporter = AttributionExporter::new();
        self.semantic_drift = exporter.detect_drift(self);
        if self.tokens.is_empty() {
            self.aggregate_confidence = 0.0;
            self.aggregate_perplexity = 0.0;
        } else {
            let n = self.tokens.len() as f32;
            self.aggregate_confidence = self.tokens.iter().map(|t| t.confidence).sum::<f32>() / n;
            self.aggregate_perplexity = self.tokens.iter().map(|t| t.perplexity).sum::<f32>() / n;
        }
    }
}

// ── Exporter ──────────────────────────────────────────────────────────────────

/// Exports [`SequenceAttribution`] data to JSONL, CSV, or HTML formats.
pub struct AttributionExporter;

impl Default for AttributionExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl AttributionExporter {
    /// Create a new exporter.
    pub fn new() -> Self {
        AttributionExporter
    }

    // ── JSONL ──────────────────────────────────────────────────────────────

    /// Serialize a slice of sequences to JSONL (one JSON object per line).
    ///
    /// The output is compatible with Python's `json.loads` line-by-line and
    /// can be fed directly into LIME or Captum pipelines via a thin adapter.
    pub fn to_jsonl(&self, sequences: &[SequenceAttribution]) -> String {
        sequences
            .iter()
            .filter_map(|seq| serde_json::to_string(seq).ok())
            .collect::<Vec<_>>()
            .join("\n")
    }

    // ── CSV ────────────────────────────────────────────────────────────────

    /// Export a single sequence to CSV with the following columns:
    ///
    /// `position,token,confidence,perplexity,attribution_score,top_alt_1,prob_1,...,top_alt_5,prob_5`
    ///
    /// Missing alternatives are represented as empty cells.
    pub fn to_csv(&self, seq: &SequenceAttribution) -> String {
        let mut out = String::from(
            "position,token,confidence,perplexity,attribution_score,\
             alt_1,alt_1_prob,alt_2,alt_2_prob,alt_3,alt_3_prob,\
             alt_4,alt_4_prob,alt_5,alt_5_prob\n",
        );
        for t in &seq.tokens {
            let token_escaped = csv_escape(&t.token);
            out.push_str(&format!(
                "{},{},{:.6},{:.6},{:.6}",
                t.position, token_escaped, t.confidence, t.perplexity, t.attribution_score,
            ));
            for i in 0..5 {
                if let Some((alt_tok, alt_prob)) = t.top_alternatives.get(i) {
                    out.push_str(&format!(",{},{:.6}", csv_escape(alt_tok), alt_prob));
                } else {
                    out.push_str(",,");
                }
            }
            out.push('\n');
        }
        out
    }

    // ── HTML heatmap ───────────────────────────────────────────────────────

    /// Produce a self-contained HTML heatmap with no external dependencies.
    ///
    /// Each token is rendered as a `<span>` with a background colour interpolated
    /// between red (low confidence) and green (high confidence).  Hovering over a
    /// token shows its confidence, perplexity, attribution score, and top
    /// alternatives in a tooltip.
    pub fn to_html_heatmap(&self, seq: &SequenceAttribution) -> String {
        let token_spans: String = seq
            .tokens
            .iter()
            .map(|t| {
                let hue = confidence_to_hue(t.confidence);
                let bg = format!("hsl({hue},80%,35%)");
                let alts = t
                    .top_alternatives
                    .iter()
                    .map(|(tok, p)| format!("{tok}: {p:.3}"))
                    .collect::<Vec<_>>()
                    .join("&#10;");
                let tooltip = format!(
                    "pos:{pos}  conf:{conf:.3}  ppl:{ppl:.3}  attr:{attr:.3}&#10;alts:{alts}",
                    pos = t.position,
                    conf = t.confidence,
                    ppl = t.perplexity,
                    attr = t.attribution_score,
                    alts = alts,
                );
                let display_tok = html_escape(&t.token);
                format!(
                    r#"<span class="tok" style="background:{bg}" title="{tooltip}">{display_tok}</span>"#
                )
            })
            .collect();

        let high_conf: Vec<usize>;
        let low_conf: Vec<usize>;
        (high_conf, low_conf) = self.confidence_extremes(seq);

        let high_list = high_conf
            .iter()
            .map(|&i| {
                seq.tokens
                    .get(i)
                    .map(|t| format!("#{i} &ldquo;{}&rdquo; ({:.3})", html_escape(&t.token), t.confidence))
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>()
            .join(", ");
        let low_list = low_conf
            .iter()
            .map(|&i| {
                seq.tokens
                    .get(i)
                    .map(|t| format!("#{i} &ldquo;{}&rdquo; ({:.3})", html_escape(&t.token), t.confidence))
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Token Attribution Heatmap — {model}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0d0d0d; color: #e8e8e8; padding: 1.5rem; }}
  h1 {{ font-size: 1.2rem; margin-bottom: 0.4rem; color: #aaffaa; }}
  .meta {{ font-size: 0.8rem; color: #888; margin-bottom: 1rem; }}
  .prompt {{ background: #1a1a2e; border-left: 3px solid #4488ff; padding: 0.6rem 0.8rem;
             font-style: italic; border-radius: 3px; margin-bottom: 1rem; white-space: pre-wrap; }}
  .heatmap {{ line-height: 2.2; word-spacing: 2px; }}
  .tok {{ display: inline-block; padding: 2px 4px; border-radius: 3px; margin: 1px;
           cursor: default; font-family: 'Courier New', monospace; font-size: 0.9rem;
           white-space: pre; color: #fff; transition: filter 0.1s; }}
  .tok:hover {{ filter: brightness(1.4); }}
  .stats {{ margin-top: 1.2rem; font-size: 0.85rem; color: #ccc; }}
  .stats table {{ border-collapse: collapse; }}
  .stats td {{ padding: 0.2rem 0.8rem 0.2rem 0; }}
  .stats td:first-child {{ color: #888; }}
  .extremes {{ margin-top: 0.8rem; font-size: 0.8rem; color: #aaa; }}
  .extremes b {{ color: #88ddff; }}
  .legend {{ margin-top: 0.8rem; display: flex; gap: 6px; align-items: center; font-size: 0.78rem; color: #888; }}
  .swatch {{ width: 60px; height: 14px; border-radius: 2px; background: linear-gradient(to right, hsl(0,80%,35%), hsl(120,80%,35%)); }}
</style>
</head>
<body>
<h1>Token Attribution Heatmap</h1>
<div class="meta">Model: <b>{model}</b> &nbsp;|&nbsp; Generated: {timestamp}</div>
<div class="prompt">{prompt_display}</div>
<div class="heatmap">{token_spans}</div>
<div class="stats">
  <table>
    <tr><td>Tokens</td><td>{token_count}</td></tr>
    <tr><td>Aggregate confidence</td><td>{agg_conf:.4}</td></tr>
    <tr><td>Aggregate perplexity</td><td>{agg_ppl:.4}</td></tr>
    <tr><td>Semantic drift</td><td>{drift:.4}</td></tr>
  </table>
</div>
<div class="extremes">
  <b>Most confident:</b> {high_list}<br>
  <b>Least confident:</b> {low_list}
</div>
<div class="legend">
  <div class="swatch"></div>
  <span>low confidence → high confidence</span>
</div>
</body>
</html>
"#,
            model = html_escape(&seq.model),
            timestamp = html_escape(&seq.timestamp),
            prompt_display = html_escape(&seq.prompt),
            token_count = seq.tokens.len(),
            agg_conf = seq.aggregate_confidence,
            agg_ppl = seq.aggregate_perplexity,
            drift = seq.semantic_drift,
            token_spans = token_spans,
            high_list = high_list,
            low_list = low_list,
        )
    }

    // ── Analysis helpers ───────────────────────────────────────────────────

    /// Measure semantic drift: the signed confidence decay from the first
    /// half of the sequence to the second half.
    ///
    /// A positive value means confidence declined toward the end.
    /// Returns `0.0` for sequences with fewer than 2 tokens.
    pub fn detect_drift(&self, seq: &SequenceAttribution) -> f32 {
        let tokens = &seq.tokens;
        if tokens.len() < 2 {
            return 0.0;
        }
        let mid = tokens.len() / 2;
        let first_half_mean = tokens[..mid]
            .iter()
            .map(|t| t.confidence)
            .sum::<f32>()
            / mid as f32;
        let second_half_mean = tokens[mid..]
            .iter()
            .map(|t| t.confidence)
            .sum::<f32>()
            / (tokens.len() - mid) as f32;
        first_half_mean - second_half_mean
    }

    /// Return the indices of the top-5 most confident and top-5 least
    /// confident tokens in the sequence.
    ///
    /// Returns `(most_confident_indices, least_confident_indices)`.
    /// Both vectors contain at most 5 entries.
    pub fn confidence_extremes(
        &self,
        seq: &SequenceAttribution,
    ) -> (Vec<usize>, Vec<usize>) {
        if seq.tokens.is_empty() {
            return (vec![], vec![]);
        }
        let mut indexed: Vec<(usize, f32)> = seq
            .tokens
            .iter()
            .map(|t| (t.position, t.confidence))
            .collect();

        // Sort descending by confidence.
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_n = indexed.len().min(5);
        let most_confident: Vec<usize> = indexed[..top_n].iter().map(|(i, _)| *i).collect();
        let least_confident: Vec<usize> = indexed[indexed.len() - top_n..]
            .iter()
            .rev()
            .map(|(i, _)| *i)
            .collect();

        (most_confident, least_confident)
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Map a confidence value in `[0.0, 1.0]` to a CSS hue in `[0, 120]`
/// (red=0, green=120).
fn confidence_to_hue(confidence: f32) -> u32 {
    (confidence.clamp(0.0, 1.0) * 120.0).round() as u32
}

/// Escape a string for embedding as a CSV field.
/// Wraps the value in double-quotes if it contains a comma, double-quote, or newline.
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Escape characters that have special meaning in HTML.
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_seq() -> SequenceAttribution {
        SequenceAttribution {
            prompt: "What is recursion?".into(),
            model: "gpt-4o".into(),
            timestamp: "2026-01-01T00:00:00Z".into(),
            tokens: vec![
                TokenAttribution {
                    token: "Recursion".into(),
                    position: 0,
                    confidence: 0.95,
                    perplexity: 1.05,
                    attribution_score: 0.4,
                    top_alternatives: vec![
                        ("Iteration".into(), 0.03),
                        ("Repetition".into(), 0.01),
                    ],
                },
                TokenAttribution {
                    token: "is".into(),
                    position: 1,
                    confidence: 0.80,
                    perplexity: 1.25,
                    attribution_score: 0.1,
                    top_alternatives: vec![("was".into(), 0.10)],
                },
                TokenAttribution {
                    token: "a".into(),
                    position: 2,
                    confidence: 0.50,
                    perplexity: 2.00,
                    attribution_score: 0.05,
                    top_alternatives: vec![],
                },
                TokenAttribution {
                    token: "technique".into(),
                    position: 3,
                    confidence: 0.40,
                    perplexity: 2.50,
                    attribution_score: 0.2,
                    top_alternatives: vec![("method".into(), 0.30)],
                },
            ],
            aggregate_confidence: 0.6625,
            aggregate_perplexity: 1.7,
            semantic_drift: 0.275,
        }
    }

    #[test]
    fn to_jsonl_produces_valid_json_lines() {
        let exporter = AttributionExporter::new();
        let seq = make_seq();
        let output = exporter.to_jsonl(&[seq.clone(), seq]);
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 2);
        for line in lines {
            let parsed: serde_json::Value = serde_json::from_str(line).unwrap();
            assert!(parsed.get("prompt").is_some());
            assert!(parsed.get("tokens").is_some());
        }
    }

    #[test]
    fn to_jsonl_empty_slice() {
        let exporter = AttributionExporter::new();
        let out = exporter.to_jsonl(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn to_csv_header_and_rows() {
        let exporter = AttributionExporter::new();
        let seq = make_seq();
        let csv = exporter.to_csv(&seq);
        let lines: Vec<&str> = csv.lines().collect();
        // Header + 4 token rows.
        assert_eq!(lines.len(), 5);
        assert!(lines[0].starts_with("position,token,confidence"));
        assert!(lines[1].contains("Recursion"));
    }

    #[test]
    fn to_csv_escapes_commas() {
        let exporter = AttributionExporter::new();
        let mut seq = make_seq();
        seq.tokens[0].token = "hello, world".into();
        let csv = exporter.to_csv(&seq);
        assert!(csv.contains("\"hello, world\""));
    }

    #[test]
    fn to_html_heatmap_is_valid_html() {
        let exporter = AttributionExporter::new();
        let seq = make_seq();
        let html = exporter.to_html_heatmap(&seq);
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Recursion"));
        assert!(html.contains("hsl("));
        // Self-contained: no external script src.
        assert!(!html.contains("src=\"http"));
    }

    #[test]
    fn detect_drift_positive_for_declining_confidence() {
        let exporter = AttributionExporter::new();
        let seq = make_seq();
        // First half (pos 0,1) mean conf = (0.95+0.80)/2 = 0.875
        // Second half (pos 2,3) mean conf = (0.50+0.40)/2 = 0.45
        // drift = 0.875 - 0.45 = 0.425
        let drift = exporter.detect_drift(&seq);
        assert!(drift > 0.0, "Expected positive drift, got {drift}");
    }

    #[test]
    fn detect_drift_zero_for_single_token() {
        let exporter = AttributionExporter::new();
        let mut seq = make_seq();
        seq.tokens.truncate(1);
        assert_eq!(exporter.detect_drift(&seq), 0.0);
    }

    #[test]
    fn confidence_extremes_returns_sorted_indices() {
        let exporter = AttributionExporter::new();
        let seq = make_seq();
        let (high, low) = exporter.confidence_extremes(&seq);
        assert!(!high.is_empty());
        assert!(!low.is_empty());
        // Highest confidence is position 0 (0.95).
        assert_eq!(high[0], 0);
        // Lowest confidence is position 3 (0.40).
        assert_eq!(low[0], 3);
    }

    #[test]
    fn confidence_extremes_empty_sequence() {
        let exporter = AttributionExporter::new();
        let mut seq = make_seq();
        seq.tokens.clear();
        let (high, low) = exporter.confidence_extremes(&seq);
        assert!(high.is_empty());
        assert!(low.is_empty());
    }

    #[test]
    fn recompute_aggregates_matches_manual() {
        let mut seq = make_seq();
        seq.recompute_aggregates();
        let expected_mean = (0.95 + 0.80 + 0.50 + 0.40) / 4.0;
        assert!((seq.aggregate_confidence - expected_mean).abs() < 1e-4);
    }

    #[test]
    fn html_escape_handles_specials() {
        assert_eq!(html_escape("<b>"), "&lt;b&gt;");
        assert_eq!(html_escape("\"hello\""), "&quot;hello&quot;");
        assert_eq!(html_escape("AT&T"), "AT&amp;T");
    }

    #[test]
    fn csv_escape_handles_embedded_quotes() {
        assert_eq!(csv_escape("say \"hello\""), "\"say \"\"hello\"\"\"");
    }
}

// ── AttributionMap — causal leave-one-out attribution ─────────────────────────

/// Per-output-token causal attribution record.
///
/// For each output token at position `token_idx`, `attribution_scores` holds
/// a ranked list of `(input_token_idx, score)` pairs. Scores are derived
/// from a leave-one-out (LOO) proxy: the score for input token *i* is the
/// drop in confidence of the output token when input token *i* is masked.
/// Positive scores indicate that the input token increases output confidence;
/// negative scores indicate the opposite.
///
/// This is an approximate (not exact) attribution method — full attention
/// weights are not accessible through the standard chat-completion API.
/// The LOO proxy is a standard interpretability technique used by LIME and
/// similar toolkits.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CausalAttribution {
    /// Zero-based index of the output token being explained.
    pub token_idx: usize,
    /// The text of the output token being explained.
    pub token_text: String,
    /// Ranked influence scores: `(input_token_index, delta_confidence)`.
    ///
    /// A positive delta means that removing the input token *reduced*
    /// output confidence (i.e. the input was helping). A negative delta
    /// means removing the input token *increased* output confidence.
    ///
    /// Sorted in descending order of absolute influence.
    pub attribution_scores: Vec<(usize, f64)>,
}

/// A causal attribution map for a complete prompt–response pair.
///
/// `AttributionMap` accumulates `CausalAttribution` records for every output
/// token and can render them as a terminal heatmap or serialize them to JSON.
///
/// # Computation model
///
/// The leave-one-out proxy works as follows:
///
/// 1. Obtain the baseline confidence for every output token from a full,
///    unmasked inference (using the `logprobs` field from the provider API).
/// 2. For each input token *i*, re-run inference with token *i* replaced by a
///    neutral mask (e.g. `"[MASK]"` or a silent space). Record the new output
///    confidences.
/// 3. The LOO score for input token *i* on output token *j* is:
///    `score(i, j) = baseline_confidence(j) - masked_confidence(j)`.
///
/// Because this requires N+1 inferences for N input tokens, `AttributionMap`
/// supports a `max_input_tokens` budget to limit cost.
///
/// # Example
///
/// ```rust
/// use every_other_token::attribution::{AttributionMap, CausalAttribution};
///
/// let input_tokens = vec!["What".to_string(), "is".to_string(), "gravity".to_string()];
/// let baseline = vec![0.9_f64, 0.85, 0.7, 0.6]; // one per output token
///
/// let mut map = AttributionMap::new(input_tokens.clone(), baseline.clone());
///
/// // Simulate masking "gravity" (index 2): all output confidences drop.
/// let masked = vec![0.5_f64, 0.5, 0.3, 0.2];
/// map.add_masked_run(2, masked);
///
/// let attrs = map.compute();
/// assert_eq!(attrs.len(), baseline.len());
/// // Output token 0 should attribute positively to input token 2 ("gravity").
/// let scores_for_tok0 = &attrs[0].attribution_scores;
/// assert!(!scores_for_tok0.is_empty());
/// ```
#[derive(Debug)]
pub struct AttributionMap {
    /// Input token texts in order.
    pub input_tokens: Vec<String>,
    /// Baseline output confidences — one per output token.
    pub baseline_confidences: Vec<f64>,
    /// Results of masked inference runs.
    /// Each entry is `(masked_input_index, output_confidences_under_mask)`.
    masked_runs: Vec<(usize, Vec<f64>)>,
}

impl AttributionMap {
    /// Construct a new attribution map for the given input tokens and baseline.
    ///
    /// # Arguments
    ///
    /// * `input_tokens` — tokenized prompt tokens, in order.
    /// * `baseline_confidences` — output token confidences from a full,
    ///   unmasked forward pass. One value per output token.
    pub fn new(input_tokens: Vec<String>, baseline_confidences: Vec<f64>) -> Self {
        Self {
            input_tokens,
            baseline_confidences,
            masked_runs: Vec::new(),
        }
    }

    /// Record the output confidences observed when input token at
    /// `masked_input_idx` was replaced by a neutral mask token.
    ///
    /// `masked_confidences` must have the same length as `baseline_confidences`.
    /// If lengths differ the run is silently ignored.
    pub fn add_masked_run(&mut self, masked_input_idx: usize, masked_confidences: Vec<f64>) {
        if masked_confidences.len() == self.baseline_confidences.len() {
            self.masked_runs.push((masked_input_idx, masked_confidences));
        }
    }

    /// Compute per-output-token attribution scores from the accumulated masked
    /// runs.
    ///
    /// Returns one [`CausalAttribution`] per output token. Input tokens for
    /// which no masked run was recorded are not included in `attribution_scores`.
    ///
    /// Scores within each `CausalAttribution` are sorted in descending order
    /// of **absolute** influence, so the most impactful input token appears first.
    pub fn compute(&self) -> Vec<CausalAttribution> {
        let n_output = self.baseline_confidences.len();
        let mut attributions: Vec<CausalAttribution> = (0..n_output)
            .map(|j| CausalAttribution {
                token_idx: j,
                token_text: format!("<output@{j}>"),
                attribution_scores: Vec::new(),
            })
            .collect();

        for (input_idx, masked_confs) in &self.masked_runs {
            for (j, attr) in attributions.iter_mut().enumerate() {
                let baseline = self.baseline_confidences[j];
                let masked = masked_confs[j];
                // Positive score = removing input token *reduced* confidence
                // (i.e. input was helpful to this output position).
                let score = baseline - masked;
                attr.attribution_scores.push((*input_idx, score));
            }
        }

        // Sort each output token's scores by descending absolute value.
        for attr in &mut attributions {
            attr.attribution_scores.sort_by(|(_, a), (_, b)| {
                b.abs().partial_cmp(&a.abs()).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        attributions
    }

    /// Serialize the computed attributions to a pretty-printed JSON string.
    ///
    /// Suitable for writing to disk for downstream analysis in Python or other
    /// environments.
    ///
    /// # Errors
    ///
    /// Returns a `serde_json::Error` if serialization fails (should never
    /// happen for valid attribution data).
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        #[derive(serde::Serialize)]
        struct MapExport<'a> {
            input_tokens: &'a [String],
            baseline_confidences: &'a [f64],
            attributions: Vec<CausalAttribution>,
        }
        let export = MapExport {
            input_tokens: &self.input_tokens,
            baseline_confidences: &self.baseline_confidences,
            attributions: self.compute(),
        };
        serde_json::to_string_pretty(&export)
    }
}

// ── AttributionRenderer — terminal heatmap ────────────────────────────────────

/// Renders a [`CausalAttribution`] or a set of attributions as a colored
/// heatmap in the terminal.
///
/// Uses the same color bands as the existing token importance visualizer:
/// - Score > 0.3  → bright green  (strong positive influence)
/// - Score > 0.1  → yellow        (moderate positive influence)
/// - Score ≈ 0.0  → white         (neutral)
/// - Score < -0.1 → red           (negative influence)
///
/// # Example
///
/// ```rust
/// use every_other_token::attribution::{AttributionMap, AttributionRenderer};
///
/// let input_tokens = vec!["What".to_string(), "is".to_string(), "gravity".to_string()];
/// let baseline = vec![0.9_f64];
/// let mut map = AttributionMap::new(input_tokens, baseline);
/// map.add_masked_run(2, vec![0.4_f64]);
///
/// let renderer = AttributionRenderer::new();
/// let attrs = map.compute();
/// // renderer.render(&attrs, &map.input_tokens);  // prints to stdout
/// ```
pub struct AttributionRenderer {
    /// Maximum number of input-token bars to print per output token.
    pub top_k: usize,
}

impl Default for AttributionRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl AttributionRenderer {
    /// Create a renderer showing the top-5 most influential input tokens per
    /// output token.
    pub fn new() -> Self {
        Self { top_k: 5 }
    }

    /// Set how many input-token bars to display per output token.
    #[must_use]
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Render attributions to a `String` suitable for terminal display.
    ///
    /// Uses ANSI escape codes for color. If the terminal does not support ANSI
    /// colors the output is still readable; the color codes appear as literal
    /// escape sequences.
    ///
    /// `input_tokens` must be the same slice used to build the [`AttributionMap`].
    pub fn render_to_string(
        &self,
        attributions: &[CausalAttribution],
        input_tokens: &[String],
    ) -> String {
        let mut out = String::new();
        for attr in attributions {
            out.push_str(&format!(
                "\n  Output token #{}: \"{}\"\n",
                attr.token_idx, attr.token_text
            ));
            if attr.attribution_scores.is_empty() {
                out.push_str("    (no masked runs recorded)\n");
                continue;
            }
            for (input_idx, score) in attr.attribution_scores.iter().take(self.top_k) {
                let label = input_tokens
                    .get(*input_idx)
                    .map(|s| s.as_str())
                    .unwrap_or("?");
                let bar = self.score_bar(*score);
                let color = self.ansi_color(*score);
                let reset = "\x1b[0m";
                out.push_str(&format!(
                    "    [{:>2}] {:>12}  {} {:.4}{}\n",
                    input_idx,
                    label,
                    color,
                    score,
                    reset
                ));
                out.push_str(&format!("           {}{}{}\n", color, bar, reset));
            }
        }
        out
    }

    /// Print the rendered heatmap to stdout.
    pub fn render(&self, attributions: &[CausalAttribution], input_tokens: &[String]) {
        print!("{}", self.render_to_string(attributions, input_tokens));
    }

    // --- private helpers ---------------------------------------------------

    fn score_bar(&self, score: f64) -> String {
        let width = (score.abs() * 40.0).round() as usize;
        let ch = if score >= 0.0 { '+' } else { '-' };
        std::iter::repeat(ch).take(width.min(40)).collect()
    }

    fn ansi_color(&self, score: f64) -> &'static str {
        if score > 0.3 {
            "\x1b[92m" // bright green
        } else if score > 0.1 {
            "\x1b[93m" // yellow
        } else if score > -0.1 {
            "\x1b[97m" // white
        } else {
            "\x1b[91m" // bright red
        }
    }
}

#[cfg(test)]
mod attribution_map_tests {
    use super::*;

    #[test]
    fn attribution_map_basic_loo() {
        let input_tokens = vec!["What".to_string(), "is".to_string(), "gravity".to_string()];
        let baseline = vec![0.9, 0.85, 0.7];

        let mut map = AttributionMap::new(input_tokens.clone(), baseline.clone());

        // Masking "gravity" (index 2) causes output confidence to drop.
        map.add_masked_run(2, vec![0.5, 0.4, 0.3]);
        // Masking "What" (index 0) causes minor drop.
        map.add_masked_run(0, vec![0.85, 0.80, 0.65]);

        let attrs = map.compute();
        assert_eq!(attrs.len(), 3, "one attribution per output token");

        // For output token 0: gravity mask gave score 0.9-0.5=0.4, What mask gave 0.9-0.85=0.05.
        // "gravity" should rank first (higher absolute score).
        let scores_0 = &attrs[0].attribution_scores;
        assert!(!scores_0.is_empty());
        let (top_idx, top_score) = scores_0[0];
        assert_eq!(top_idx, 2, "gravity should be most influential");
        assert!((top_score - 0.4).abs() < 1e-9);
    }

    #[test]
    fn attribution_map_mismatched_run_is_ignored() {
        let input_tokens = vec!["A".to_string(), "B".to_string()];
        let baseline = vec![0.8, 0.6];
        let mut map = AttributionMap::new(input_tokens, baseline);

        // Wrong length — should be silently ignored.
        map.add_masked_run(0, vec![0.5]); // length 1, not 2

        let attrs = map.compute();
        // No runs recorded → all score lists are empty.
        for attr in &attrs {
            assert!(attr.attribution_scores.is_empty());
        }
    }

    #[test]
    fn attribution_map_to_json_is_valid() {
        let input_tokens = vec!["Hello".to_string()];
        let baseline = vec![0.9];
        let mut map = AttributionMap::new(input_tokens, baseline);
        map.add_masked_run(0, vec![0.5]);

        let json = map.to_json().expect("JSON serialization should succeed");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert!(parsed.get("attributions").is_some());
        assert!(parsed.get("input_tokens").is_some());
        assert!(parsed.get("baseline_confidences").is_some());
    }

    #[test]
    fn renderer_produces_colored_output() {
        let input_tokens = vec!["token0".to_string(), "token1".to_string()];
        let baseline = vec![0.8];
        let mut map = AttributionMap::new(input_tokens.clone(), baseline);
        map.add_masked_run(0, vec![0.3]);

        let renderer = AttributionRenderer::new();
        let attrs = map.compute();
        let output = renderer.render_to_string(&attrs, &input_tokens);

        assert!(output.contains("Output token #0"));
        // ANSI green for high positive score.
        assert!(output.contains("\x1b[92m"), "expected green ANSI color for high positive score");
    }

    #[test]
    fn renderer_top_k_limits_output() {
        let input_tokens: Vec<String> = (0..10).map(|i| format!("tok{i}")).collect();
        let baseline = vec![0.5; 1];
        let mut map = AttributionMap::new(input_tokens.clone(), baseline);
        for i in 0..10 {
            map.add_masked_run(i, vec![0.5 - (i as f64 * 0.01)]);
        }

        let renderer = AttributionRenderer::new().with_top_k(3);
        let attrs = map.compute();
        let output = renderer.render_to_string(&attrs, &input_tokens);

        // Count how many "[  N]" lines appear (one per displayed input token).
        let bar_lines = output.lines().filter(|l| l.contains('[') && l.contains(']')).count();
        assert!(bar_lines <= 3, "top_k=3 should limit to 3 bars, got {bar_lines}");
    }
}
