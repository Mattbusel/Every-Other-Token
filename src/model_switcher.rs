//! Runtime model switching based on task complexity and budget.
//!
//! Evaluates a set of [`SwitchTrigger`] conditions against live [`RunMetrics`]
//! and returns a [`SwitchDecision`] when switching is warranted.

use std::time::{SystemTime, UNIX_EPOCH};

// ── ModelProfile ─────────────────────────────────────────────────────────────

/// Descriptor for a single model available in the pool.
#[derive(Debug, Clone)]
pub struct ModelProfile {
    pub model_id: String,
    pub cost_per_token: f64,
    pub max_context: u64,
    pub avg_latency_ms: u64,
    pub specializations: Vec<String>,
}

// ── SwitchTrigger ─────────────────────────────────────────────────────────────

/// Conditions that may trigger a model switch.
#[derive(Debug, Clone)]
pub enum SwitchTrigger {
    /// Switch when cumulative cost exceeds the threshold in USD.
    CostThreshold { usd: f64 },
    /// Switch when the last call latency exceeded the threshold in ms.
    LatencyThreshold { ms: u64 },
    /// Switch when token usage approaches the context limit.
    ContextLimit { tokens: u64 },
    /// Switch when the quality score drops below the threshold.
    QualityDrop { threshold: f64 },
    /// Switch to a model specialised for a given task type.
    Specialization { task_type: String },
}

// ── RunMetrics ────────────────────────────────────────────────────────────────

/// Live metrics collected during a model run.
#[derive(Debug, Clone, Default)]
pub struct RunMetrics {
    pub tokens_used: u64,
    pub cost_so_far_usd: f64,
    pub latency_ms: u64,
    pub quality_score: Option<f64>,
    pub task_type: String,
}

// ── SwitchDecision ────────────────────────────────────────────────────────────

/// Record of a model-switch event.
#[derive(Debug, Clone)]
pub struct SwitchDecision {
    pub from_model: String,
    pub to_model: String,
    pub trigger: SwitchTrigger,
    pub reason: String,
    pub timestamp: u64,
}

// ── ModelSwitcher ─────────────────────────────────────────────────────────────

/// Registry and decision engine for runtime model switching.
#[derive(Debug, Default)]
pub struct ModelSwitcher {
    profiles: Vec<ModelProfile>,
    history: Vec<SwitchDecision>,
    call_count: usize,
}

impl ModelSwitcher {
    /// Create a new, empty switcher.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a model profile.
    pub fn register(&mut self, profile: ModelProfile) {
        self.profiles.push(profile);
    }

    /// Evaluate all triggers against the current metrics.
    ///
    /// Returns the first matching [`SwitchDecision`], or `None` if no trigger
    /// fires. The decision is recorded in the switch history.
    pub fn evaluate_switch(
        &mut self,
        current_model: &str,
        metrics: &RunMetrics,
        triggers: &[SwitchTrigger],
    ) -> Option<SwitchDecision> {
        self.call_count += 1;

        for trigger in triggers {
            let fired = match trigger {
                SwitchTrigger::CostThreshold { usd } => metrics.cost_so_far_usd > *usd,
                SwitchTrigger::LatencyThreshold { ms } => metrics.latency_ms > *ms,
                SwitchTrigger::ContextLimit { tokens } => metrics.tokens_used > *tokens,
                SwitchTrigger::QualityDrop { threshold } => metrics
                    .quality_score
                    .map(|q| q < *threshold)
                    .unwrap_or(false),
                SwitchTrigger::Specialization { task_type } => {
                    metrics.task_type == *task_type
                }
            };

            if fired {
                // Find an alternative model (first that is not current_model).
                let candidate = self
                    .profiles
                    .iter()
                    .find(|p| p.model_id != current_model)
                    .map(|p| p.model_id.clone())
                    .unwrap_or_else(|| current_model.to_string());

                let reason = format!("{:?} triggered switch from {current_model} → {candidate}", trigger);
                let ts = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);

                let decision = SwitchDecision {
                    from_model: current_model.to_string(),
                    to_model: candidate,
                    trigger: trigger.clone(),
                    reason,
                    timestamp: ts,
                };
                self.history.push(decision.clone());
                return Some(decision);
            }
        }
        None
    }

    /// Return the model best suited for a given task type within a per-token budget.
    pub fn best_for_task(&self, task_type: &str, budget_per_token: f64) -> Option<&ModelProfile> {
        // Prefer models that specialise in the task and fit within budget.
        let mut candidates: Vec<&ModelProfile> = self
            .profiles
            .iter()
            .filter(|p| {
                p.cost_per_token <= budget_per_token
                    && p.specializations.iter().any(|s| s == task_type)
            })
            .collect();

        if candidates.is_empty() {
            // Fall back to any affordable model.
            candidates = self
                .profiles
                .iter()
                .filter(|p| p.cost_per_token <= budget_per_token)
                .collect();
        }

        // Among candidates, pick lowest cost then lowest latency.
        candidates.into_iter().min_by(|a, b| {
            a.cost_per_token
                .partial_cmp(&b.cost_per_token)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.avg_latency_ms.cmp(&b.avg_latency_ms))
        })
    }

    /// Full switch history (oldest first).
    pub fn switch_history(&self) -> &[SwitchDecision] {
        &self.history
    }

    /// Total number of switches recorded.
    pub fn total_switches(&self) -> usize {
        self.history.len()
    }

    /// Switches per 100 `evaluate_switch` calls.
    pub fn switch_rate_per_100_calls(&self) -> f64 {
        if self.call_count == 0 {
            return 0.0;
        }
        (self.history.len() as f64 / self.call_count as f64) * 100.0
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_profile(id: &str, cost: f64, latency: u64, specs: &[&str]) -> ModelProfile {
        ModelProfile {
            model_id: id.to_string(),
            cost_per_token: cost,
            max_context: 8_192,
            avg_latency_ms: latency,
            specializations: specs.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn make_switcher() -> ModelSwitcher {
        let mut sw = ModelSwitcher::new();
        sw.register(make_profile("gpt-4o-mini", 0.00001, 300, &["chat", "summarize"]));
        sw.register(make_profile("gpt-4o", 0.00005, 800, &["code", "reasoning"]));
        sw.register(make_profile("claude-3-haiku", 0.000025, 400, &["fast", "chat"]));
        sw
    }

    #[test]
    fn cost_threshold_fires() {
        let mut sw = make_switcher();
        let metrics = RunMetrics {
            cost_so_far_usd: 1.5,
            ..Default::default()
        };
        let triggers = vec![SwitchTrigger::CostThreshold { usd: 1.0 }];
        let dec = sw.evaluate_switch("gpt-4o", &metrics, &triggers);
        assert!(dec.is_some());
        let dec = dec.unwrap();
        assert_eq!(dec.from_model, "gpt-4o");
        assert!(sw.total_switches() == 1);
    }

    #[test]
    fn cost_threshold_no_fire() {
        let mut sw = make_switcher();
        let metrics = RunMetrics {
            cost_so_far_usd: 0.5,
            ..Default::default()
        };
        let triggers = vec![SwitchTrigger::CostThreshold { usd: 1.0 }];
        assert!(sw.evaluate_switch("gpt-4o", &metrics, &triggers).is_none());
    }

    #[test]
    fn latency_threshold_fires() {
        let mut sw = make_switcher();
        let metrics = RunMetrics {
            latency_ms: 2000,
            ..Default::default()
        };
        let triggers = vec![SwitchTrigger::LatencyThreshold { ms: 1000 }];
        assert!(sw.evaluate_switch("gpt-4o", &metrics, &triggers).is_some());
    }

    #[test]
    fn context_limit_fires() {
        let mut sw = make_switcher();
        let metrics = RunMetrics {
            tokens_used: 10_000,
            ..Default::default()
        };
        let triggers = vec![SwitchTrigger::ContextLimit { tokens: 8_000 }];
        assert!(sw.evaluate_switch("gpt-4o-mini", &metrics, &triggers).is_some());
    }

    #[test]
    fn quality_drop_fires() {
        let mut sw = make_switcher();
        let metrics = RunMetrics {
            quality_score: Some(0.3),
            ..Default::default()
        };
        let triggers = vec![SwitchTrigger::QualityDrop { threshold: 0.5 }];
        assert!(sw.evaluate_switch("gpt-4o-mini", &metrics, &triggers).is_some());
    }

    #[test]
    fn quality_drop_no_score_does_not_fire() {
        let mut sw = make_switcher();
        let metrics = RunMetrics {
            quality_score: None,
            ..Default::default()
        };
        let triggers = vec![SwitchTrigger::QualityDrop { threshold: 0.5 }];
        assert!(sw.evaluate_switch("gpt-4o-mini", &metrics, &triggers).is_none());
    }

    #[test]
    fn specialization_trigger_fires() {
        let mut sw = make_switcher();
        let metrics = RunMetrics {
            task_type: "code".to_string(),
            ..Default::default()
        };
        let triggers = vec![SwitchTrigger::Specialization { task_type: "code".to_string() }];
        assert!(sw.evaluate_switch("gpt-4o-mini", &metrics, &triggers).is_some());
    }

    #[test]
    fn best_for_task_within_budget() {
        let sw = make_switcher();
        let profile = sw.best_for_task("chat", 0.00003);
        assert!(profile.is_some());
        assert!(profile.unwrap().cost_per_token <= 0.00003);
    }

    #[test]
    fn best_for_task_exceeds_budget_returns_none() {
        let sw = make_switcher();
        let profile = sw.best_for_task("reasoning", 0.000001);
        assert!(profile.is_none());
    }

    #[test]
    fn switch_rate_zero_calls() {
        let sw = ModelSwitcher::new();
        assert_eq!(sw.switch_rate_per_100_calls(), 0.0);
    }

    #[test]
    fn switch_rate_calculated() {
        let mut sw = make_switcher();
        let triggers = vec![SwitchTrigger::CostThreshold { usd: 0.0 }];
        let m = RunMetrics { cost_so_far_usd: 1.0, ..Default::default() };
        sw.evaluate_switch("gpt-4o", &m, &triggers);
        sw.evaluate_switch("gpt-4o", &m, &triggers);
        sw.evaluate_switch("gpt-4o", &m, &[]);
        // 2 switches out of 3 calls → 66.67
        let rate = sw.switch_rate_per_100_calls();
        assert!((rate - 66.666_666_666).abs() < 0.001);
    }

    #[test]
    fn history_is_ordered() {
        let mut sw = make_switcher();
        let triggers = vec![SwitchTrigger::LatencyThreshold { ms: 0 }];
        let m = RunMetrics { latency_ms: 999, ..Default::default() };
        sw.evaluate_switch("gpt-4o", &m, &triggers);
        sw.evaluate_switch("gpt-4o", &m, &triggers);
        assert_eq!(sw.switch_history().len(), 2);
    }
}
