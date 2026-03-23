//! Token budget manager with per-request and per-session tracking.

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BudgetScope {
    Request(String),
    Session(String),
    User(String),
    Global,
}

impl BudgetScope {
    fn key(&self) -> String {
        match self {
            BudgetScope::Request(k) => format!("request:{k}"),
            BudgetScope::Session(k) => format!("session:{k}"),
            BudgetScope::User(k) => format!("user:{k}"),
            BudgetScope::Global => "global".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TokenBudget {
    pub input_limit: u64,
    pub output_limit: u64,
    pub total_limit: u64,
    pub cost_limit_usd: f64,
    pub cost_per_input_token: f64,
    pub cost_per_output_token: f64,
}

impl Default for TokenBudget {
    fn default() -> Self {
        Self {
            input_limit: u64::MAX,
            output_limit: u64::MAX,
            total_limit: u64::MAX,
            cost_limit_usd: f64::MAX,
            cost_per_input_token: 0.0,
            cost_per_output_token: 0.0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct BudgetUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub cost_usd: f64,
}

#[derive(Debug, Clone)]
pub struct BudgetStatus {
    pub ok: bool,
    pub warnings: Vec<String>,
    pub blocked_reason: Option<String>,
}

impl BudgetStatus {
    fn allowed() -> Self {
        Self { ok: true, warnings: vec![], blocked_reason: None }
    }
    fn blocked(reason: impl Into<String>) -> Self {
        let r = reason.into();
        Self { ok: false, warnings: vec![], blocked_reason: Some(r) }
    }
}

#[derive(Debug, Default)]
pub struct TokenBudgetManager {
    budgets: HashMap<String, TokenBudget>,
    usage: HashMap<String, BudgetUsage>,
}

impl TokenBudgetManager {
    pub fn new() -> Self { Self::default() }

    pub fn set_budget(&mut self, scope: BudgetScope, budget: TokenBudget) {
        let key = scope.key();
        self.budgets.insert(key.clone(), budget);
        self.usage.entry(key).or_default();
    }

    pub fn check(&self, scope: &BudgetScope, input_estimate: u64, output_estimate: u64) -> BudgetStatus {
        let key = scope.key();
        let budget = match self.budgets.get(&key) {
            Some(b) => b,
            None => return BudgetStatus::allowed(),
        };
        let usage = self.usage.get(&key).cloned().unwrap_or_default();
        let new_input = usage.input_tokens + input_estimate;
        let new_output = usage.output_tokens + output_estimate;
        let new_total = usage.total_tokens + input_estimate + output_estimate;
        let new_cost = usage.cost_usd
            + input_estimate as f64 * budget.cost_per_input_token
            + output_estimate as f64 * budget.cost_per_output_token;
        if new_input > budget.input_limit {
            return BudgetStatus::blocked(format!("Input token limit exceeded: {} / {}", new_input, budget.input_limit));
        }
        if new_output > budget.output_limit {
            return BudgetStatus::blocked(format!("Output token limit exceeded: {} / {}", new_output, budget.output_limit));
        }
        if new_total > budget.total_limit {
            return BudgetStatus::blocked(format!("Total token limit exceeded: {} / {}", new_total, budget.total_limit));
        }
        if new_cost > budget.cost_limit_usd {
            return BudgetStatus::blocked(format!("Cost limit exceeded: ${:.6} / ${:.6}", new_cost, budget.cost_limit_usd));
        }
        let mut warnings = vec![];
        if budget.total_limit < u64::MAX {
            let pct = new_total as f64 / budget.total_limit as f64;
            if pct > 0.9 { warnings.push(format!("Total tokens at {:.0}% of budget", pct * 100.0)); }
        }
        if budget.cost_limit_usd < 1e300 {
            let pct = new_cost / budget.cost_limit_usd;
            if pct > 0.9 { warnings.push(format!("Cost at {:.0}% of budget", pct * 100.0)); }
        }
        BudgetStatus { ok: true, warnings, blocked_reason: None }
    }

    pub fn consume(&mut self, scope: &BudgetScope, input: u64, output: u64) -> BudgetStatus {
        let status = self.check(scope, input, output);
        if !status.ok { return status; }
        let key = scope.key();
        let budget = self.budgets.get(&key).cloned();
        let u = self.usage.entry(key).or_default();
        u.input_tokens += input;
        u.output_tokens += output;
        u.total_tokens += input + output;
        if let Some(b) = budget {
            u.cost_usd += input as f64 * b.cost_per_input_token + output as f64 * b.cost_per_output_token;
        }
        status
    }

    pub fn usage(&self, scope: &BudgetScope) -> Option<BudgetUsage> {
        self.usage.get(&scope.key()).cloned()
    }

    pub fn reset(&mut self, scope: &BudgetScope) {
        self.usage.insert(scope.key(), BudgetUsage::default());
    }

    pub fn overage_report(&self) -> Vec<(String, BudgetUsage)> {
        let mut report = vec![];
        for (key, usage) in &self.usage {
            if let Some(budget) = self.budgets.get(key) {
                if budget.total_limit < u64::MAX {
                    let pct = usage.total_tokens as f64 / budget.total_limit as f64;
                    if pct > 0.9 { report.push((key.clone(), usage.clone())); }
                }
            }
        }
        report.sort_by(|a, b| a.0.cmp(&b.0));
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_budget(total: u64) -> TokenBudget {
        TokenBudget {
            input_limit: total, output_limit: total, total_limit: total,
            cost_limit_usd: 1.0, cost_per_input_token: 0.000001, cost_per_output_token: 0.000002,
        }
    }

    #[test]
    fn test_consume_within_budget() {
        let mut mgr = TokenBudgetManager::new();
        let scope = BudgetScope::Request("r1".to_string());
        mgr.set_budget(scope.clone(), simple_budget(1000));
        let status = mgr.consume(&scope, 100, 200);
        assert!(status.ok);
        let u = mgr.usage(&scope).unwrap();
        assert_eq!(u.input_tokens, 100);
        assert_eq!(u.output_tokens, 200);
        assert_eq!(u.total_tokens, 300);
    }

    #[test]
    fn test_blocked_over_total() {
        let mut mgr = TokenBudgetManager::new();
        let scope = BudgetScope::Session("s1".to_string());
        mgr.set_budget(scope.clone(), simple_budget(100));
        let status = mgr.consume(&scope, 60, 60);
        assert!(!status.ok);
        assert_eq!(mgr.usage(&scope).unwrap().total_tokens, 0);
    }

    #[test]
    fn test_check_no_mutate() {
        let mut mgr = TokenBudgetManager::new();
        let scope = BudgetScope::User("u1".to_string());
        mgr.set_budget(scope.clone(), simple_budget(1000));
        mgr.check(&scope, 100, 100);
        assert_eq!(mgr.usage(&scope).unwrap().total_tokens, 0);
    }

    #[test]
    fn test_reset() {
        let mut mgr = TokenBudgetManager::new();
        let scope = BudgetScope::Global;
        mgr.set_budget(scope.clone(), simple_budget(5000));
        mgr.consume(&scope, 500, 500);
        mgr.reset(&scope);
        assert_eq!(mgr.usage(&scope).unwrap().total_tokens, 0);
    }

    #[test]
    fn test_warnings_near_limit() {
        let mut mgr = TokenBudgetManager::new();
        let scope = BudgetScope::Request("r2".to_string());
        mgr.set_budget(scope.clone(), simple_budget(1000));
        mgr.consume(&scope, 500, 410);
        let status = mgr.check(&scope, 5, 5);
        assert!(status.ok);
        assert!(!status.warnings.is_empty());
    }

    #[test]
    fn test_overage_report() {
        let mut mgr = TokenBudgetManager::new();
        let s1 = BudgetScope::Session("a".to_string());
        let s2 = BudgetScope::Session("b".to_string());
        mgr.set_budget(s1.clone(), simple_budget(100));
        mgr.set_budget(s2.clone(), simple_budget(100));
        mgr.consume(&s1, 50, 45);
        mgr.consume(&s2, 20, 20);
        let report = mgr.overage_report();
        assert_eq!(report.len(), 1);
        assert!(report[0].0.contains("session:a"));
    }

    #[test]
    fn test_unlimited() {
        let mut mgr = TokenBudgetManager::new();
        let scope = BudgetScope::Request("x".to_string());
        assert!(mgr.consume(&scope, 999999, 999999).ok);
    }

    #[test]
    fn test_cost_tracking() {
        let mut mgr = TokenBudgetManager::new();
        let scope = BudgetScope::User("u2".to_string());
        mgr.set_budget(scope.clone(), TokenBudget {
            input_limit: 10_000, output_limit: 10_000, total_limit: 20_000,
            cost_limit_usd: 1.0, cost_per_input_token: 0.001, cost_per_output_token: 0.002,
        });
        mgr.consume(&scope, 100, 50);
        let u = mgr.usage(&scope).unwrap();
        assert!((u.cost_usd - (0.1 + 0.1)).abs() < 1e-9);
    }

    #[test]
    fn test_blocked_by_cost() {
        let mut mgr = TokenBudgetManager::new();
        let scope = BudgetScope::User("u3".to_string());
        mgr.set_budget(scope.clone(), TokenBudget {
            input_limit: 1_000_000, output_limit: 1_000_000, total_limit: 2_000_000,
            cost_limit_usd: 0.01, cost_per_input_token: 0.001, cost_per_output_token: 0.001,
        });
        let status = mgr.consume(&scope, 100, 100);
        assert!(!status.ok);
        assert!(status.blocked_reason.unwrap().contains("Cost"));
    }
}