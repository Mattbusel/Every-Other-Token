//! # Stage: Cost Optimizer
//!
//! ## Responsibility
//! Real-time cost tracking per model backend per request. Maintains a cost model
//! (tokens × price-per-token) with actual costs reconciled against API billing
//! responses. Implements budget-aware routing: when spend approaches a configurable
//! ceiling, traffic is shifted toward cheaper backends.
//!
//! Tracks cost-per-quality by correlating routing decisions with downstream
//! success metrics, and publishes Pareto-optimal cost/quality curves for the
//! dashboard.
//!
//! ## Guarantees
//! - Bounded: all history buffers have configurable capacity
//! - Non-panicking: division-by-zero and empty-collection cases are all guarded
//! - Thread-safe: designed to be wrapped in Arc<Mutex> by callers
//!
//! ## NOT Responsible For
//! - Actual HTTP routing decisions (that is the router)
//! - Storing cost history in Redis (snapshotter, task 1.6)

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Backend identity
// ---------------------------------------------------------------------------

/// A named model backend.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Backend(pub String);

impl Backend {
    pub fn new(name: impl Into<String>) -> Self { Self(name.into()) }
    pub fn name(&self) -> &str { &self.0 }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// TokenPrice — cost model for one backend
// ---------------------------------------------------------------------------

/// Price model for a backend, in USD per token.
#[derive(Debug, Clone)]
pub struct TokenPrice {
    /// Cost per input (prompt) token in USD.
    pub input_per_token: f64,
    /// Cost per output (completion) token in USD.
    pub output_per_token: f64,
}

impl TokenPrice {
    pub fn new(input_per_token: f64, output_per_token: f64) -> Self {
        Self { input_per_token, output_per_token }
    }

    /// Estimate cost in USD for given token counts.
    pub fn estimate(&self, input_tokens: u64, output_tokens: u64) -> f64 {
        self.input_per_token * input_tokens as f64
            + self.output_per_token * output_tokens as f64
    }
}

// ---------------------------------------------------------------------------
// RequestCost — one billed request
// ---------------------------------------------------------------------------

/// Cost record for a single completed request.
#[derive(Debug, Clone)]
pub struct RequestCost {
    pub backend: Backend,
    /// Estimated cost at time of routing (before billing response).
    pub estimated_usd: f64,
    /// Actual cost from billing API response (`None` until reconciled).
    pub actual_usd: Option<f64>,
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Quality score in [-1, 1] provided by the quality estimator (task 3.4).
    /// `None` until the quality signal arrives.
    pub quality_score: Option<f64>,
}

impl RequestCost {
    /// The cost used for budget accounting: actual if available, else estimated.
    pub fn effective_usd(&self) -> f64 {
        self.actual_usd.unwrap_or(self.estimated_usd)
    }
}

// ---------------------------------------------------------------------------
// BudgetConfig
// ---------------------------------------------------------------------------

/// Budget configuration for the cost optimizer.
#[derive(Debug, Clone)]
pub struct BudgetConfig {
    /// Total spend ceiling in USD. Routing shifts to cheaper backends above
    /// `warn_fraction * ceiling`.
    pub ceiling_usd: f64,
    /// Fraction of `ceiling_usd` at which to start shifting traffic (e.g. 0.8 = 80%).
    pub warn_fraction: f64,
    /// Fraction at which to hard-shift all traffic to the cheapest backend.
    pub critical_fraction: f64,
    /// Maximum number of `RequestCost` records kept in the rolling history.
    pub history_cap: usize,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            ceiling_usd: 10.0,
            warn_fraction: 0.80,
            critical_fraction: 0.95,
            history_cap: 10_000,
        }
    }
}

// ---------------------------------------------------------------------------
// BudgetPressure
// ---------------------------------------------------------------------------

/// Current budget pressure level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetPressure {
    /// Spend is well below the ceiling.
    Normal,
    /// Spend has crossed `warn_fraction` — prefer cheaper backends.
    Warn,
    /// Spend has crossed `critical_fraction` — hard-shift to cheapest backend.
    Critical,
}

impl std::fmt::Display for BudgetPressure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BudgetPressure::Normal   => write!(f, "normal"),
            BudgetPressure::Warn     => write!(f, "warn"),
            BudgetPressure::Critical => write!(f, "critical"),
        }
    }
}

// ---------------------------------------------------------------------------
// ParetoPoint — one point on the cost/quality Pareto frontier
// ---------------------------------------------------------------------------

/// A backend's position on the cost/quality Pareto frontier.
#[derive(Debug, Clone)]
pub struct ParetoPoint {
    pub backend: Backend,
    /// Average cost per request in USD.
    pub avg_cost_usd: f64,
    /// Average quality score in [-1, 1].
    pub avg_quality: f64,
    /// Number of requests contributing to this point.
    pub request_count: u64,
}

// ---------------------------------------------------------------------------
// BackendStats — per-backend rolling statistics
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct BackendStats {
    total_spent_usd: f64,
    request_count: u64,
    quality_sum: f64,
    quality_count: u64,
    /// EMA of cost per request.
    cost_ema: f64,
    /// EMA of quality score.
    quality_ema: f64,
    ema_alpha: f64,
}

impl BackendStats {
    fn new(ema_alpha: f64) -> Self {
        Self { ema_alpha, ..Default::default() }
    }

    fn record_cost(&mut self, cost_usd: f64) {
        self.total_spent_usd += cost_usd;
        self.request_count += 1;
        let alpha = if self.ema_alpha > 0.0 { self.ema_alpha } else { 0.1 };
        self.cost_ema = if self.request_count == 1 {
            cost_usd
        } else {
            alpha * cost_usd + (1.0 - alpha) * self.cost_ema
        };
    }

    fn record_quality(&mut self, score: f64) {
        self.quality_sum += score;
        self.quality_count += 1;
        let alpha = if self.ema_alpha > 0.0 { self.ema_alpha } else { 0.1 };
        self.quality_ema = if self.quality_count == 1 {
            score
        } else {
            alpha * score + (1.0 - alpha) * self.quality_ema
        };
    }

    fn avg_cost(&self) -> f64 {
        if self.request_count == 0 { 0.0 }
        else { self.total_spent_usd / self.request_count as f64 }
    }

    fn avg_quality(&self) -> f64 {
        if self.quality_count == 0 { 0.0 }
        else { self.quality_sum / self.quality_count as f64 }
    }
}

// ---------------------------------------------------------------------------
// CostOptimizer — the main type
// ---------------------------------------------------------------------------

/// Real-time cost tracker and budget-aware routing advisor.
///
/// # Usage
/// ```ignore
/// let mut opt = CostOptimizer::new(BudgetConfig::default(), 0.1);
/// opt.set_price(Backend::new("gpt-4"), TokenPrice::new(0.00003, 0.00006));
/// opt.record_request(req_cost);
/// let preferred = opt.preferred_backends();
/// ```
pub struct CostOptimizer {
    cfg: BudgetConfig,
    prices: HashMap<Backend, TokenPrice>,
    stats: HashMap<Backend, BackendStats>,
    history: Vec<RequestCost>,
    ema_alpha: f64,
}

impl CostOptimizer {
    pub fn new(cfg: BudgetConfig, ema_alpha: f64) -> Self {
        Self {
            cfg,
            prices: HashMap::new(),
            stats: HashMap::new(),
            history: Vec::new(),
            ema_alpha,
        }
    }

    // --- Configuration ---

    /// Register or update the price model for a backend.
    pub fn set_price(&mut self, backend: Backend, price: TokenPrice) {
        self.prices.insert(backend, price);
    }

    /// Estimate the cost of a request to a backend before sending it.
    pub fn estimate(&self, backend: &Backend, input_tokens: u64, output_tokens: u64) -> f64 {
        self.prices
            .get(backend)
            .map(|p| p.estimate(input_tokens, output_tokens))
            .unwrap_or(0.0)
    }

    // --- Recording ---

    /// Record a completed request cost (estimated or reconciled).
    pub fn record_request(&mut self, req: RequestCost) {
        let cost = req.effective_usd();
        let stats = self.stats.entry(req.backend.clone())
            .or_insert_with(|| BackendStats::new(self.ema_alpha));
        stats.record_cost(cost);
        if let Some(q) = req.quality_score {
            stats.record_quality(q);
        }

        if self.history.len() >= self.cfg.history_cap {
            self.history.remove(0);
        }
        self.history.push(req);
    }

    /// Reconcile an estimated cost with the actual billing amount.
    ///
    /// Finds the most recent unreconciled request for the backend and updates it.
    pub fn reconcile(&mut self, backend: &Backend, actual_usd: f64) {
        if let Some(req) = self.history.iter_mut().rev()
            .find(|r| &r.backend == backend && r.actual_usd.is_none())
        {
            let old_cost = req.estimated_usd;
            req.actual_usd = Some(actual_usd);
            // Adjust stats: remove estimated, add actual
            if let Some(stats) = self.stats.get_mut(backend) {
                stats.total_spent_usd += actual_usd - old_cost;
            }
        }
    }

    // --- Budget tracking ---

    /// Total USD spent across all backends (using effective cost).
    pub fn total_spent_usd(&self) -> f64 {
        self.stats.values().map(|s| s.total_spent_usd).sum()
    }

    /// Fraction of budget ceiling consumed [0, ∞).
    pub fn budget_fraction(&self) -> f64 {
        if self.cfg.ceiling_usd <= 0.0 { return 0.0; }
        self.total_spent_usd() / self.cfg.ceiling_usd
    }

    /// Current budget pressure level.
    pub fn pressure(&self) -> BudgetPressure {
        let frac = self.budget_fraction();
        if frac >= self.cfg.critical_fraction {
            BudgetPressure::Critical
        } else if frac >= self.cfg.warn_fraction {
            BudgetPressure::Warn
        } else {
            BudgetPressure::Normal
        }
    }

    /// Remaining budget in USD (`ceiling - spent`), clamped to ≥ 0.
    pub fn remaining_usd(&self) -> f64 {
        (self.cfg.ceiling_usd - self.total_spent_usd()).max(0.0)
    }

    // --- Routing advice ---

    /// All known backends sorted by average cost per request (cheapest first).
    pub fn backends_by_cost(&self) -> Vec<Backend> {
        let mut v: Vec<_> = self.stats.iter()
            .map(|(b, s)| (b.clone(), s.avg_cost()))
            .collect();
        v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        v.into_iter().map(|(b, _)| b).collect()
    }

    /// Preferred backends given current budget pressure.
    ///
    /// - `Normal`: all backends (in cost order, cheapest first)
    /// - `Warn`: cheapest 50% of backends
    /// - `Critical`: cheapest backend only
    pub fn preferred_backends(&self) -> Vec<Backend> {
        let by_cost = self.backends_by_cost();
        match self.pressure() {
            BudgetPressure::Normal => by_cost,
            BudgetPressure::Warn => {
                let keep = (by_cost.len() / 2).max(1);
                by_cost.into_iter().take(keep).collect()
            }
            BudgetPressure::Critical => {
                by_cost.into_iter().take(1).collect()
            }
        }
    }

    /// The single cheapest backend, or `None` if no backends have been seen.
    pub fn cheapest_backend(&self) -> Option<Backend> {
        self.backends_by_cost().into_iter().next()
    }

    // --- Cost-per-quality / Pareto ---

    /// Build the Pareto frontier: non-dominated (backend, cost, quality) points.
    ///
    /// A point dominates another if it has both lower cost AND higher quality.
    /// The returned vec contains only non-dominated points, sorted by cost ascending.
    pub fn pareto_frontier(&self) -> Vec<ParetoPoint> {
        let mut points: Vec<ParetoPoint> = self.stats.iter()
            .filter(|(_, s)| s.request_count > 0 && s.quality_count > 0)
            .map(|(b, s)| ParetoPoint {
                backend: b.clone(),
                avg_cost_usd: s.avg_cost(),
                avg_quality: s.avg_quality(),
                request_count: s.request_count,
            })
            .collect();

        // Sort by cost ascending
        points.sort_by(|a, b| a.avg_cost_usd.partial_cmp(&b.avg_cost_usd)
            .unwrap_or(std::cmp::Ordering::Equal));

        // Keep non-dominated points (sweep: a point is dominated if another
        // has lower/equal cost AND strictly higher quality)
        let mut frontier: Vec<ParetoPoint> = Vec::new();
        let mut best_quality = f64::NEG_INFINITY;
        for p in points {
            if p.avg_quality > best_quality {
                best_quality = p.avg_quality;
                frontier.push(p);
            }
        }
        frontier
    }

    // --- Reporting ---

    /// Per-backend cost and quality summary.
    pub fn backend_report(&self) -> Vec<BackendReport> {
        let mut v: Vec<BackendReport> = self.stats.iter().map(|(b, s)| BackendReport {
            backend: b.clone(),
            total_spent_usd: s.total_spent_usd,
            request_count: s.request_count,
            avg_cost_usd: s.avg_cost(),
            cost_ema_usd: s.cost_ema,
            avg_quality: s.avg_quality(),
            quality_ema: s.quality_ema,
        }).collect();
        v.sort_by(|a, b| a.backend.0.cmp(&b.backend.0));
        v
    }

    pub fn history_len(&self) -> usize { self.history.len() }
    pub fn known_backend_count(&self) -> usize { self.stats.len() }
}

/// Summary row for one backend.
#[derive(Debug, Clone)]
pub struct BackendReport {
    pub backend: Backend,
    pub total_spent_usd: f64,
    pub request_count: u64,
    pub avg_cost_usd: f64,
    pub cost_ema_usd: f64,
    pub avg_quality: f64,
    pub quality_ema: f64,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn b(name: &str) -> Backend { Backend::new(name) }

    fn cheap_price() -> TokenPrice { TokenPrice::new(0.000001, 0.000002) }

    fn req(backend: &str, cost: f64, quality: Option<f64>) -> RequestCost {
        RequestCost {
            backend: b(backend),
            estimated_usd: cost,
            actual_usd: None,
            input_tokens: 100,
            output_tokens: 50,
            quality_score: quality,
        }
    }

    fn default_optimizer() -> CostOptimizer {
        CostOptimizer::new(BudgetConfig::default(), 0.1)
    }

    // ===== Backend =====

    #[test]
    fn test_backend_display() {
        assert_eq!(b("gpt-4").to_string(), "gpt-4");
    }

    #[test]
    fn test_backend_eq() {
        assert_eq!(b("a"), b("a"));
        assert_ne!(b("a"), b("b"));
    }

    // ===== TokenPrice =====

    #[test]
    fn test_token_price_estimate_zero_tokens() {
        let p = TokenPrice::new(0.001, 0.002);
        assert_eq!(p.estimate(0, 0), 0.0);
    }

    #[test]
    fn test_token_price_estimate_input_only() {
        let p = TokenPrice::new(0.001, 0.0);
        assert!((p.estimate(1000, 0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_token_price_estimate_output_only() {
        let p = TokenPrice::new(0.0, 0.002);
        assert!((p.estimate(0, 500) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_token_price_estimate_combined() {
        let p = TokenPrice::new(0.001, 0.002);
        // 100 input + 50 output = 0.1 + 0.1 = 0.2
        assert!((p.estimate(100, 50) - 0.2).abs() < 1e-9);
    }

    // ===== RequestCost =====

    #[test]
    fn test_request_cost_effective_usd_no_actual() {
        let r = req("gpt-4", 0.05, None);
        assert_eq!(r.effective_usd(), 0.05);
    }

    #[test]
    fn test_request_cost_effective_usd_with_actual() {
        let mut r = req("gpt-4", 0.05, None);
        r.actual_usd = Some(0.07);
        assert_eq!(r.effective_usd(), 0.07);
    }

    // ===== BudgetPressure display =====

    #[test]
    fn test_budget_pressure_display() {
        assert_eq!(BudgetPressure::Normal.to_string(),   "normal");
        assert_eq!(BudgetPressure::Warn.to_string(),     "warn");
        assert_eq!(BudgetPressure::Critical.to_string(), "critical");
    }

    // ===== CostOptimizer — basics =====

    #[test]
    fn test_optimizer_new_zero_spent() {
        let opt = default_optimizer();
        assert_eq!(opt.total_spent_usd(), 0.0);
    }

    #[test]
    fn test_optimizer_estimate_unknown_backend_is_zero() {
        let opt = default_optimizer();
        assert_eq!(opt.estimate(&b("unknown"), 1000, 500), 0.0);
    }

    #[test]
    fn test_optimizer_estimate_known_backend() {
        let mut opt = default_optimizer();
        opt.set_price(b("cheap"), cheap_price());
        let e = opt.estimate(&b("cheap"), 1000, 500);
        assert!(e > 0.0);
    }

    #[test]
    fn test_optimizer_record_request_increments_spent() {
        let mut opt = default_optimizer();
        opt.record_request(req("gpt-4", 0.05, None));
        assert!((opt.total_spent_usd() - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_optimizer_record_multiple_requests_sums_correctly() {
        let mut opt = default_optimizer();
        opt.record_request(req("a", 0.10, None));
        opt.record_request(req("b", 0.20, None));
        opt.record_request(req("a", 0.05, None));
        assert!((opt.total_spent_usd() - 0.35).abs() < 1e-9);
    }

    #[test]
    fn test_optimizer_known_backend_count() {
        let mut opt = default_optimizer();
        opt.record_request(req("x", 0.01, None));
        opt.record_request(req("y", 0.02, None));
        assert_eq!(opt.known_backend_count(), 2);
    }

    #[test]
    fn test_optimizer_history_bounded() {
        let mut opt = CostOptimizer::new(BudgetConfig { history_cap: 5, ..Default::default() }, 0.1);
        for _ in 0..10 { opt.record_request(req("a", 0.01, None)); }
        assert_eq!(opt.history_len(), 5);
    }

    // ===== Budget pressure =====

    #[test]
    fn test_optimizer_pressure_normal_when_under_warn() {
        let mut opt = CostOptimizer::new(BudgetConfig { ceiling_usd: 10.0, warn_fraction: 0.8, ..Default::default() }, 0.1);
        opt.record_request(req("a", 5.0, None)); // 50% of ceiling
        assert_eq!(opt.pressure(), BudgetPressure::Normal);
    }

    #[test]
    fn test_optimizer_pressure_warn_when_above_warn_fraction() {
        let mut opt = CostOptimizer::new(BudgetConfig { ceiling_usd: 10.0, warn_fraction: 0.8, critical_fraction: 0.95, ..Default::default() }, 0.1);
        opt.record_request(req("a", 8.5, None)); // 85%
        assert_eq!(opt.pressure(), BudgetPressure::Warn);
    }

    #[test]
    fn test_optimizer_pressure_critical_when_above_critical_fraction() {
        let mut opt = CostOptimizer::new(BudgetConfig { ceiling_usd: 10.0, warn_fraction: 0.8, critical_fraction: 0.95, ..Default::default() }, 0.1);
        opt.record_request(req("a", 9.6, None)); // 96%
        assert_eq!(opt.pressure(), BudgetPressure::Critical);
    }

    #[test]
    fn test_optimizer_remaining_usd_decreases_with_spend() {
        let mut opt = CostOptimizer::new(BudgetConfig { ceiling_usd: 10.0, ..Default::default() }, 0.1);
        opt.record_request(req("a", 3.0, None));
        assert!((opt.remaining_usd() - 7.0).abs() < 1e-9);
    }

    #[test]
    fn test_optimizer_remaining_usd_clamped_to_zero() {
        let mut opt = CostOptimizer::new(BudgetConfig { ceiling_usd: 1.0, ..Default::default() }, 0.1);
        opt.record_request(req("a", 5.0, None));
        assert_eq!(opt.remaining_usd(), 0.0);
    }

    #[test]
    fn test_optimizer_budget_fraction_zero_ceiling_is_zero() {
        let opt = CostOptimizer::new(BudgetConfig { ceiling_usd: 0.0, ..Default::default() }, 0.1);
        assert_eq!(opt.budget_fraction(), 0.0);
    }

    // ===== Reconciliation =====

    #[test]
    fn test_reconcile_updates_actual_and_stats() {
        let mut opt = default_optimizer();
        opt.record_request(req("a", 0.05, None));
        opt.reconcile(&b("a"), 0.07);
        // Total should now reflect 0.07 instead of 0.05 (+0.02 delta)
        assert!((opt.total_spent_usd() - 0.07).abs() < 1e-6);
    }

    #[test]
    fn test_reconcile_no_op_when_no_matching_request() {
        let mut opt = default_optimizer();
        opt.record_request(req("a", 0.05, None));
        opt.reconcile(&b("b"), 0.10); // no "b" request
        assert!((opt.total_spent_usd() - 0.05).abs() < 1e-9);
    }

    // ===== Routing advice =====

    #[test]
    fn test_backends_by_cost_sorted_cheapest_first() {
        let mut opt = default_optimizer();
        opt.record_request(req("expensive", 1.00, None));
        opt.record_request(req("medium",    0.50, None));
        opt.record_request(req("cheap",     0.10, None));
        let order = opt.backends_by_cost();
        assert_eq!(order[0], b("cheap"));
        assert_eq!(order[2], b("expensive"));
    }

    #[test]
    fn test_preferred_backends_normal_returns_all() {
        let mut opt = CostOptimizer::new(BudgetConfig { ceiling_usd: 100.0, ..Default::default() }, 0.1);
        opt.record_request(req("a", 0.01, None));
        opt.record_request(req("b", 0.02, None));
        let preferred = opt.preferred_backends();
        assert_eq!(preferred.len(), 2);
    }

    #[test]
    fn test_preferred_backends_critical_returns_one() {
        let mut opt = CostOptimizer::new(BudgetConfig { ceiling_usd: 1.0, critical_fraction: 0.5, warn_fraction: 0.3, ..Default::default() }, 0.1);
        opt.record_request(req("a", 0.01, Some(0.8)));
        opt.record_request(req("b", 0.02, Some(0.9)));
        opt.record_request(req("c", 0.03, Some(0.7)));
        opt.record_request(req("x", 0.9,  None));   // push past critical
        let preferred = opt.preferred_backends();
        assert_eq!(preferred.len(), 1);
    }

    #[test]
    fn test_cheapest_backend_returns_lowest_avg_cost() {
        let mut opt = default_optimizer();
        opt.record_request(req("expensive", 1.0, None));
        opt.record_request(req("cheap", 0.01, None));
        assert_eq!(opt.cheapest_backend(), Some(b("cheap")));
    }

    #[test]
    fn test_cheapest_backend_none_when_empty() {
        let opt = default_optimizer();
        assert_eq!(opt.cheapest_backend(), None);
    }

    // ===== Pareto frontier =====

    #[test]
    fn test_pareto_frontier_empty_when_no_quality_data() {
        let mut opt = default_optimizer();
        opt.record_request(req("a", 0.01, None));
        assert!(opt.pareto_frontier().is_empty());
    }

    #[test]
    fn test_pareto_frontier_single_point() {
        let mut opt = default_optimizer();
        opt.record_request(req("a", 0.01, Some(0.8)));
        let f = opt.pareto_frontier();
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].backend, b("a"));
    }

    #[test]
    fn test_pareto_frontier_dominated_point_excluded() {
        let mut opt = default_optimizer();
        // "cheap" costs less AND has higher quality — "expensive" is dominated
        opt.record_request(req("cheap",     0.01, Some(0.9)));
        opt.record_request(req("expensive", 0.10, Some(0.5)));
        let f = opt.pareto_frontier();
        // Only "cheap" is non-dominated
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].backend, b("cheap"));
    }

    #[test]
    fn test_pareto_frontier_tradeoff_keeps_both() {
        let mut opt = default_optimizer();
        // "cheap" costs less but lower quality; "premium" costs more but higher quality
        opt.record_request(req("cheap",   0.01, Some(0.3)));
        opt.record_request(req("premium", 0.10, Some(0.9)));
        let f = opt.pareto_frontier();
        assert_eq!(f.len(), 2);
    }

    #[test]
    fn test_pareto_frontier_sorted_by_cost() {
        let mut opt = default_optimizer();
        opt.record_request(req("z_expensive", 1.00, Some(0.9)));
        opt.record_request(req("a_cheap",     0.01, Some(0.1)));
        let f = opt.pareto_frontier();
        if f.len() >= 2 {
            assert!(f[0].avg_cost_usd <= f[1].avg_cost_usd);
        }
    }

    // ===== Backend report =====

    #[test]
    fn test_backend_report_sorted_alphabetically() {
        let mut opt = default_optimizer();
        opt.record_request(req("z", 0.1, Some(0.5)));
        opt.record_request(req("a", 0.2, Some(0.7)));
        let report = opt.backend_report();
        assert_eq!(report[0].backend, b("a"));
        assert_eq!(report[1].backend, b("z"));
    }

    #[test]
    fn test_backend_report_correct_total_spend() {
        let mut opt = default_optimizer();
        opt.record_request(req("a", 0.1, None));
        opt.record_request(req("a", 0.3, None));
        let report = opt.backend_report();
        let a = report.iter().find(|r| r.backend == b("a")).unwrap();
        assert!((a.total_spent_usd - 0.4).abs() < 1e-9);
    }

    #[test]
    fn test_backend_report_avg_cost_correct() {
        let mut opt = default_optimizer();
        opt.record_request(req("a", 0.10, None));
        opt.record_request(req("a", 0.30, None));
        let report = opt.backend_report();
        let a = report.iter().find(|r| r.backend == b("a")).unwrap();
        assert!((a.avg_cost_usd - 0.20).abs() < 1e-9);
    }

    #[test]
    fn test_backend_report_quality_tracked() {
        let mut opt = default_optimizer();
        opt.record_request(req("a", 0.01, Some(0.6)));
        opt.record_request(req("a", 0.01, Some(0.8)));
        let report = opt.backend_report();
        let a = report.iter().find(|r| r.backend == b("a")).unwrap();
        assert!((a.avg_quality - 0.7).abs() < 1e-9);
    }

    // ===== EMA tracking =====

    #[test]
    fn test_cost_ema_first_value_equals_cost() {
        let mut opt = default_optimizer();
        opt.record_request(req("a", 0.05, None));
        let report = opt.backend_report();
        assert!((report[0].cost_ema_usd - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_quality_ema_first_value_equals_score() {
        let mut opt = default_optimizer();
        opt.record_request(req("a", 0.01, Some(0.75)));
        let report = opt.backend_report();
        assert!((report[0].quality_ema - 0.75).abs() < 1e-9);
    }
}
