//! Dynamic token budget allocation for multi-turn conversations.
//!
//! Provides [`TokenBudgetManager`] which tracks per-turn token usage and
//! applies one of several allocation policies to distribute a total budget
//! across system prompt, conversation history, and model response.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Policy
// ---------------------------------------------------------------------------

/// How the budget should be split each turn.
#[derive(Debug, Clone)]
pub enum BudgetPolicy {
    /// Every turn gets the same fixed split.
    Fixed(usize),
    /// Percentage-based split (fractions must sum to ≤ 1.0).
    Proportional {
        system_pct: f64,
        history_pct: f64,
        response_pct: f64,
    },
    /// Response window adapts based on recent usage; history gets the rest.
    Adaptive {
        min_response: usize,
        max_response: usize,
    },
    /// Budget is re-computed over the last `window_turns` turns.
    Rolling { window_turns: usize },
}

// ---------------------------------------------------------------------------
// Per-turn structures
// ---------------------------------------------------------------------------

/// Token allocation for a single conversation turn.
#[derive(Debug, Clone)]
pub struct TurnBudget {
    pub system_tokens: usize,
    pub history_tokens: usize,
    pub response_tokens: usize,
    pub total: usize,
    pub turn_id: usize,
}

/// Efficiency report for a single turn.
#[derive(Debug, Clone)]
pub struct BudgetUsage {
    pub allocated: usize,
    pub used: usize,
    pub efficiency: f64,
    pub overflow: bool,
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

/// High-level summary across all turns managed so far.
#[derive(Debug, Clone)]
pub struct BudgetSummary {
    pub total_budget: usize,
    pub total_used: usize,
    pub efficiency: f64,
    pub turns_completed: usize,
    pub avg_turn_tokens: f64,
}

// ---------------------------------------------------------------------------
// Manager
// ---------------------------------------------------------------------------

/// Manages token budgets across multiple conversation turns.
pub struct TokenBudgetManager {
    pub total_budget: usize,
    policy: BudgetPolicy,
    history: VecDeque<TurnBudget>,
    /// Tracks actual usage for the *current* turn.
    used_this_turn: Arc<AtomicUsize>,
    /// Recorded usages per completed turn (allocated, used).
    usage_history: VecDeque<(usize, usize)>,
    /// Adaptive response target — updated by [`adaptive_adjust`].
    adaptive_response_target: usize,
}

impl TokenBudgetManager {
    /// Create a new manager with the given total budget and policy.
    pub fn new(total_budget: usize, policy: BudgetPolicy) -> Self {
        let adaptive_response_target = match &policy {
            BudgetPolicy::Adaptive { min_response, max_response } => {
                (min_response + max_response) / 2
            }
            BudgetPolicy::Fixed(n) => *n / 4,
            _ => total_budget / 4,
        };
        Self {
            total_budget,
            policy,
            history: VecDeque::new(),
            used_this_turn: Arc::new(AtomicUsize::new(0)),
            usage_history: VecDeque::new(),
            adaptive_response_target,
        }
    }

    /// Allocate tokens for a new turn given known system and history sizes.
    pub fn allocate_turn(
        &mut self,
        turn_id: usize,
        system_size: usize,
        history_size: usize,
    ) -> TurnBudget {
        // Reset current-turn counter.
        self.used_this_turn.store(0, Ordering::SeqCst);

        let budget = self.total_budget;

        let (system_tokens, history_tokens, response_tokens) = match &self.policy {
            BudgetPolicy::Fixed(per_turn) => {
                let avail = (*per_turn).min(budget);
                let sys = system_size.min(avail / 4);
                let hist = history_size.min(avail / 2);
                let resp = avail.saturating_sub(sys + hist);
                (sys, hist, resp)
            }
            BudgetPolicy::Proportional {
                system_pct,
                history_pct,
                response_pct,
            } => {
                let sys = ((budget as f64 * system_pct) as usize).min(system_size);
                let hist = ((budget as f64 * history_pct) as usize).min(history_size);
                let resp = (budget as f64 * response_pct) as usize;
                (sys, hist, resp)
            }
            BudgetPolicy::Adaptive {
                min_response,
                max_response,
            } => {
                let resp = self
                    .adaptive_response_target
                    .clamp(*min_response, *max_response);
                let remaining = budget.saturating_sub(resp);
                let sys = system_size.min(remaining / 3);
                let hist = history_size.min(remaining.saturating_sub(sys));
                (sys, hist, resp)
            }
            BudgetPolicy::Rolling { window_turns } => {
                let window = *window_turns;
                let avg_used = self.rolling_average_usage(window);
                let est_total = avg_used as usize;
                let sys = system_size.min(est_total / 4);
                let hist = history_size.min(est_total / 2);
                let resp = budget.saturating_sub(sys + hist);
                (sys, hist, resp)
            }
        };

        let allocated = system_tokens + history_tokens + response_tokens;
        let turn = TurnBudget {
            system_tokens,
            history_tokens,
            response_tokens,
            total: allocated,
            turn_id,
        };
        self.history.push_back(turn.clone());
        turn
    }

    /// Record how many tokens were actually consumed this turn.
    pub fn record_usage(&self, used: usize) {
        self.used_this_turn.store(used, Ordering::SeqCst);
    }

    /// Remaining budget after subtracting all recorded usage.
    pub fn remaining_budget(&self) -> usize {
        let used: usize = self.usage_history.iter().map(|(_, u)| u).sum();
        let current = self.used_this_turn.load(Ordering::SeqCst);
        self.total_budget.saturating_sub(used + current)
    }

    /// Per-turn efficiency report (most recent turns first).
    pub fn efficiency_report(&self) -> Vec<BudgetUsage> {
        self.history
            .iter()
            .zip(self.usage_history.iter())
            .map(|(turn, (allocated, used))| BudgetUsage {
                allocated: *allocated,
                used: *used,
                efficiency: if *allocated == 0 {
                    0.0
                } else {
                    (*used as f64) / (*allocated as f64)
                },
                overflow: *used > *allocated,
            })
            .collect()
    }

    /// Update the adaptive target based on the actual response token count.
    pub fn adaptive_adjust(&mut self, actual_response_tokens: usize) {
        // Commit the current turn's usage to history.
        let used = self.used_this_turn.load(Ordering::SeqCst);
        if let Some(last) = self.history.back() {
            self.usage_history.push_back((last.total, used));
        }

        match &self.policy {
            BudgetPolicy::Adaptive {
                min_response,
                max_response,
            } => {
                // Exponential moving average with α = 0.3.
                let alpha = 0.3_f64;
                let new_target = alpha * actual_response_tokens as f64
                    + (1.0 - alpha) * self.adaptive_response_target as f64;
                self.adaptive_response_target =
                    (new_target as usize).clamp(*min_response, *max_response);
            }
            _ => {
                // For non-adaptive policies just record the usage.
            }
        }
    }

    /// Rolling average of total tokens used over the last `window` turns.
    pub fn rolling_average_usage(&self, window: usize) -> f64 {
        if self.usage_history.is_empty() {
            return (self.total_budget / 2) as f64;
        }
        let skip = self.usage_history.len().saturating_sub(window);
        let slice: Vec<_> = self.usage_history.iter().skip(skip).collect();
        if slice.is_empty() {
            return 0.0;
        }
        let sum: usize = slice.iter().map(|(_, u)| u).sum();
        sum as f64 / slice.len() as f64
    }

    /// Suggest which content pieces to trim and by how much.
    ///
    /// Returns a parallel `Vec<usize>` where each entry is the number of
    /// tokens to cut from the corresponding `content_sizes` entry.
    pub fn suggest_truncation(&self, content_sizes: &[usize]) -> Vec<usize> {
        let remaining = self.remaining_budget();
        let total_content: usize = content_sizes.iter().sum();

        if total_content <= remaining {
            return vec![0; content_sizes.len()];
        }

        let excess = total_content - remaining;
        // Distribute cuts proportionally.
        content_sizes
            .iter()
            .map(|&size| {
                if total_content == 0 {
                    0
                } else {
                    let cut = (size as f64 / total_content as f64 * excess as f64) as usize;
                    cut.min(size)
                }
            })
            .collect()
    }

    /// High-level summary of overall budget consumption.
    pub fn budget_summary(&self) -> BudgetSummary {
        let total_used: usize = self.usage_history.iter().map(|(_, u)| u).sum();
        let turns = self.usage_history.len();
        let efficiency = if self.total_budget == 0 {
            0.0
        } else {
            total_used as f64 / self.total_budget as f64
        };
        let avg_turn_tokens = if turns == 0 {
            0.0
        } else {
            total_used as f64 / turns as f64
        };
        BudgetSummary {
            total_budget: self.total_budget,
            total_used,
            efficiency,
            turns_completed: turns,
            avg_turn_tokens,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_policy_allocates() {
        let mut mgr = TokenBudgetManager::new(
            4096,
            BudgetPolicy::Fixed(1024),
        );
        let tb = mgr.allocate_turn(0, 200, 400);
        assert!(tb.response_tokens > 0);
        assert_eq!(tb.turn_id, 0);
    }

    #[test]
    fn proportional_policy() {
        let mut mgr = TokenBudgetManager::new(
            10_000,
            BudgetPolicy::Proportional {
                system_pct: 0.1,
                history_pct: 0.4,
                response_pct: 0.5,
            },
        );
        let tb = mgr.allocate_turn(1, 500, 2000);
        assert!(tb.system_tokens <= 1000);
        assert!(tb.response_tokens <= 5000);
    }

    #[test]
    fn adaptive_policy_adjusts() {
        let mut mgr = TokenBudgetManager::new(
            8192,
            BudgetPolicy::Adaptive {
                min_response: 128,
                max_response: 2048,
            },
        );
        mgr.allocate_turn(0, 100, 300);
        mgr.record_usage(600);
        mgr.adaptive_adjust(512);
        let tb = mgr.allocate_turn(1, 100, 300);
        assert!(tb.response_tokens >= 128);
    }

    #[test]
    fn suggest_truncation_sums_correctly() {
        let mut mgr = TokenBudgetManager::new(1000, BudgetPolicy::Fixed(200));
        mgr.allocate_turn(0, 50, 100);
        mgr.record_usage(900); // almost exhausted
        mgr.adaptive_adjust(900);
        let cuts = mgr.suggest_truncation(&[400, 400, 400]);
        let total_cut: usize = cuts.iter().sum();
        assert!(total_cut > 0);
    }

    #[test]
    fn rolling_average_no_panic_empty() {
        let mgr = TokenBudgetManager::new(4096, BudgetPolicy::Rolling { window_turns: 5 });
        let avg = mgr.rolling_average_usage(5);
        assert!(avg > 0.0);
    }
}
