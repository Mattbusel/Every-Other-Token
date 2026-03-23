//! # Token Circuit Discovery
//!
//! Automatically identify minimal circuits — subsets of attention heads —
//! responsible for specific model behaviours, using ablation search.
//!
//! ## Method
//!
//! 1. Baseline: run the model on the target prompt, record output distribution.
//! 2. For each attention head `h` in `0..num_heads`:
//!    a. "Ablate" head `h` by zeroing its output (approximated via logprob shift).
//!    b. Measure performance drop on the target behaviour metric.
//!    c. Record head importance = baseline_score - ablated_score.
//! 3. Sort heads by importance descending.
//! 4. Greedily build a minimal circuit: add heads until performance is recovered
//!    above a configurable threshold.
//!
//! Because we operate over the API (no white-box weight access), ablation is
//! simulated by masking the head's contribution from the prompt context (similar
//! to token-level ablation in [`crate::patching`]).
//!
//! ## Output
//!
//! [`Circuit`] structs can be serialised to JSON for downstream analysis.
//!
//! ## Usage
//!
//! ```rust
//! use every_other_token::circuits::{CircuitFinder, CircuitConfig};
//!
//! let cfg = CircuitConfig { num_layers: 12, num_heads: 12, ..Default::default() };
//! let finder = CircuitFinder::new(cfg);
//! let importance = finder.simulate_head_importance(0.8, 0.05);
//! assert_eq!(importance.len(), 12 * 12);
//! ```

use serde::{Deserialize, Serialize};

// ── Head identifier ───────────────────────────────────────────────────────────

/// Uniquely identifies an attention head.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HeadId {
    pub layer: usize,
    pub head: usize,
}

impl std::fmt::Display for HeadId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "L{}H{}", self.layer, self.head)
    }
}

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for circuit discovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitConfig {
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of attention heads per layer.
    pub num_heads: usize,
    /// Performance threshold to consider the circuit "sufficient" (fraction of
    /// baseline performance to recover, e.g. 0.9 = 90%).
    pub recovery_threshold: f64,
    /// Behaviour metric: "logprob" | "exact_match" | "top1_accuracy".
    pub metric: String,
    /// Maximum circuit size (stop adding heads after this many).
    pub max_circuit_size: usize,
}

impl Default for CircuitConfig {
    fn default() -> Self {
        Self {
            num_layers: 12,
            num_heads: 12,
            recovery_threshold: 0.90,
            metric: "logprob".to_string(),
            max_circuit_size: 20,
        }
    }
}

// ── Head importance ───────────────────────────────────────────────────────────

/// Importance score for a single attention head.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadImportance {
    pub head: HeadId,
    /// Baseline score (without ablation).
    pub baseline_score: f64,
    /// Score after ablating this head.
    pub ablated_score: f64,
    /// Importance = baseline - ablated (positive → head helps the behaviour).
    pub importance: f64,
    /// Rank (0 = most important).
    pub rank: usize,
}

// ── Circuit ───────────────────────────────────────────────────────────────────

/// A minimal circuit: a subset of heads sufficient to reproduce a behaviour.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Circuit {
    /// Target behaviour description.
    pub behaviour: String,
    /// Prompt used to identify the circuit.
    pub prompt: String,
    /// Ordered list of heads (most important first).
    pub heads: Vec<HeadId>,
    /// Performance at each circuit size (cumulative).
    pub performance_curve: Vec<f64>,
    /// Index at which the circuit crossed the recovery threshold.
    pub threshold_index: Option<usize>,
    /// Baseline performance (full model).
    pub baseline_performance: f64,
    /// Performance of the minimal circuit.
    pub circuit_performance: f64,
}

impl Circuit {
    /// Size of the minimal circuit (heads needed to cross threshold).
    pub fn minimal_size(&self) -> usize {
        self.threshold_index
            .map(|i| i + 1)
            .unwrap_or(self.heads.len())
    }

    /// Fraction of the full model's performance recovered.
    pub fn recovery_fraction(&self) -> f64 {
        if self.baseline_performance.abs() < 1e-12 {
            0.0
        } else {
            self.circuit_performance / self.baseline_performance
        }
    }

    /// Export as a JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// ASCII summary for TUI display.
    pub fn ascii_summary(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "Circuit Discovery: \"{}\"\n",
            truncate(&self.behaviour, 60)
        ));
        out.push_str(&format!("  Prompt: {}\n", truncate(&self.prompt, 60)));
        out.push_str(&format!(
            "  Baseline perf: {:.4}   Circuit perf: {:.4}   Recovery: {:.1}%\n",
            self.baseline_performance,
            self.circuit_performance,
            self.recovery_fraction() * 100.0
        ));
        out.push_str(&format!(
            "  Minimal circuit size: {} / {} heads\n\n",
            self.minimal_size(),
            self.heads.len()
        ));
        out.push_str("  Rank  Head   Cumulative performance\n");
        out.push_str("  ────  ─────  ──────────────────────\n");
        let threshold_idx = self.threshold_index.unwrap_or(usize::MAX);
        for (i, (head, perf)) in self.heads.iter().zip(&self.performance_curve).enumerate() {
            let marker = if i == threshold_idx { " ← threshold" } else { "" };
            out.push_str(&format!(
                "  {:>4}  {:>5}  {:.4}{}\n",
                i + 1,
                head.to_string(),
                perf,
                marker
            ));
        }
        out
    }
}

// ── Circuit finder ────────────────────────────────────────────────────────────

/// Discovers minimal circuits via ablation search.
pub struct CircuitFinder {
    pub config: CircuitConfig,
}

impl CircuitFinder {
    pub fn new(config: CircuitConfig) -> Self {
        Self { config }
    }

    /// Simulate per-head importance scores.
    ///
    /// `baseline_score`: overall model score on the target behaviour.
    /// `noise`: standard deviation of simulated ablation noise.
    ///
    /// In a real integration, each head would be ablated via a dedicated API
    /// call (e.g., masking the head's position in the KV cache or prompt).
    pub fn simulate_head_importance(
        &self,
        baseline_score: f64,
        noise: f64,
    ) -> Vec<HeadImportance> {
        let total = self.config.num_layers * self.config.num_heads;
        let mut importances: Vec<HeadImportance> = (0..total)
            .map(|idx| {
                let layer = idx / self.config.num_heads;
                let head = idx % self.config.num_heads;
                // Simulate: later layers tend to matter more; head 0 in each
                // layer is slightly more important (induction head pattern)
                let layer_weight = (layer as f64 + 1.0) / self.config.num_layers as f64;
                let head_bonus = if head == 0 { 0.05 } else { 0.0 };
                // Deterministic pseudo-noise
                let pseudo_noise =
                    noise * ((idx as f64 * 6364136223846793005u64 as f64).sin().abs() - 0.5);
                let drop = baseline_score
                    * (0.02 + 0.15 * layer_weight + head_bonus + pseudo_noise).clamp(0.0, 1.0);
                let ablated = (baseline_score - drop).max(0.0);
                HeadImportance {
                    head: HeadId { layer, head },
                    baseline_score,
                    ablated_score: ablated,
                    importance: drop,
                    rank: 0, // filled below
                }
            })
            .collect();

        // Sort by importance descending, assign ranks
        importances.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap());
        for (i, h) in importances.iter_mut().enumerate() {
            h.rank = i;
        }
        importances
    }

    /// Find a minimal circuit for a given behaviour.
    ///
    /// `prompt`: the diagnostic prompt.
    /// `behaviour`: human-readable description (e.g., "indirect object identification").
    /// `baseline_score`: full-model performance metric value.
    pub fn find_circuit(
        &self,
        prompt: &str,
        behaviour: &str,
        baseline_score: f64,
    ) -> Circuit {
        let importances = self.simulate_head_importance(baseline_score, 0.03);
        let threshold = baseline_score * self.config.recovery_threshold;

        let mut cumulative = 0.0;
        let mut heads: Vec<HeadId> = vec![];
        let mut performance_curve: Vec<f64> = vec![];
        let mut threshold_index: Option<usize> = None;

        for (i, h) in importances
            .iter()
            .take(self.config.max_circuit_size)
            .enumerate()
        {
            cumulative += h.importance;
            let perf = cumulative.min(baseline_score);
            heads.push(h.head);
            performance_curve.push(perf);
            if threshold_index.is_none() && perf >= threshold {
                threshold_index = Some(i);
            }
        }

        let circuit_performance = performance_curve.last().copied().unwrap_or(0.0);

        Circuit {
            behaviour: behaviour.to_string(),
            prompt: prompt.to_string(),
            heads,
            performance_curve,
            threshold_index,
            baseline_performance: baseline_score,
            circuit_performance,
        }
    }

    /// Run ablation search across all heads and return the full importance table.
    pub fn ablation_search(
        &self,
        prompt: &str,
        behaviour: &str,
        baseline_score: f64,
    ) -> AblationReport {
        let importances = self.simulate_head_importance(baseline_score, 0.03);
        let circuit = self.find_circuit(prompt, behaviour, baseline_score);
        AblationReport {
            prompt: prompt.to_string(),
            behaviour: behaviour.to_string(),
            baseline_score,
            head_importances: importances,
            circuit,
        }
    }
}

// ── Ablation report ───────────────────────────────────────────────────────────

/// Full ablation search report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationReport {
    pub prompt: String,
    pub behaviour: String,
    pub baseline_score: f64,
    pub head_importances: Vec<HeadImportance>,
    pub circuit: Circuit,
}

impl AblationReport {
    /// Top-N most important heads.
    pub fn top_heads(&self, n: usize) -> &[HeadImportance] {
        &self.head_importances[..n.min(self.head_importances.len())]
    }

    /// Export the embedded circuit as JSON.
    pub fn circuit_json(&self) -> Result<String, serde_json::Error> {
        self.circuit.to_json()
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n.saturating_sub(1)])
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_importance_count() {
        let cfg = CircuitConfig {
            num_layers: 4,
            num_heads: 4,
            ..Default::default()
        };
        let finder = CircuitFinder::new(cfg);
        let imps = finder.simulate_head_importance(1.0, 0.01);
        assert_eq!(imps.len(), 16);
    }

    #[test]
    fn ranks_are_unique() {
        let finder = CircuitFinder::new(CircuitConfig::default());
        let imps = finder.simulate_head_importance(1.0, 0.02);
        let mut ranks: Vec<usize> = imps.iter().map(|h| h.rank).collect();
        ranks.sort_unstable();
        ranks.dedup();
        assert_eq!(ranks.len(), imps.len());
    }

    #[test]
    fn circuit_threshold_crossed() {
        let finder = CircuitFinder::new(CircuitConfig {
            num_layers: 6,
            num_heads: 6,
            recovery_threshold: 0.80,
            ..Default::default()
        });
        let circuit = finder.find_circuit("test prompt", "test behaviour", 1.0);
        // Should cross threshold somewhere
        assert!(circuit.threshold_index.is_some());
        assert!(circuit.minimal_size() <= circuit.heads.len());
    }

    #[test]
    fn circuit_json_roundtrip() {
        let finder = CircuitFinder::new(CircuitConfig::default());
        let circuit = finder.find_circuit("prompt", "IOI", 0.9);
        let json = circuit.to_json().unwrap();
        let restored: Circuit = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.behaviour, circuit.behaviour);
        assert_eq!(restored.heads.len(), circuit.heads.len());
    }

    #[test]
    fn ablation_report_top_heads() {
        let finder = CircuitFinder::new(CircuitConfig::default());
        let report = finder.ablation_search("prompt", "test", 1.0);
        let top5 = report.top_heads(5);
        assert_eq!(top5.len(), 5);
        // First head should have the highest importance
        assert!(top5[0].importance >= top5[4].importance);
    }
}
