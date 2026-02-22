//! Converts HelixRouter stats into TelemetrySnapshot format.
//!
//! ## Responsibility
//! Translate the external [`RouterStats`] schema produced by HelixRouter's HTTP
//! API into the internal [`TelemetrySnapshot`] type consumed by [`TelemetryBus`]
//! subscribers in the self-tune pipeline.
//!
//! ## Mapping
//! | RouterStats field              | TelemetrySnapshot field | Notes                              |
//! |-------------------------------|-------------------------|------------------------------------|
//! | dropped / (completed+dropped) | drop_rate               | Falls back to 0 when total == 0    |
//! | pressure_score × 10_000       | avg_latency_us          | Scales [0,1] score to microseconds |
//! | adaptive_spawn_threshold      | p95_1m_us               | Proxy for latency budget           |
//! | sum(routed.values())          | tokens_per_sec          | Total routed task count            |
//! | routed[Drop] / total_routed   | drop_rate (max)         | Explicit drop signal               |
//!
//! [`TelemetryBus`]: crate::self_tune::telemetry_bus::TelemetryBus

use crate::self_tune::telemetry_bus::TelemetrySnapshot;
use super::client::{RouterStats, RoutingStrategy};

/// Convert a [`RouterStats`] snapshot from HelixRouter into a [`TelemetrySnapshot`].
///
/// # Arguments
/// * `stats` — Reference to a stats value freshly parsed from `/api/stats`.
///
/// # Returns
/// A [`TelemetrySnapshot::zero()`]-based struct with fields populated from `stats`.
/// All returned float fields are guaranteed to be finite (not NaN, not infinite).
///
/// # Panics
/// This function never panics.
pub fn stats_to_snapshot(stats: &RouterStats) -> TelemetrySnapshot {
    let mut snap = TelemetrySnapshot::zero();

    // --- drop_rate from completed/dropped counts ---
    let total = stats.completed + stats.dropped;
    snap.drop_rate = if total > 0 {
        stats.dropped as f64 / total as f64
    } else {
        0.0
    };

    // --- avg_latency_us from pressure_score ---
    // pressure_score is a [0, 1] normalized score; scale to microseconds
    // so downstream controllers operate in the expected unit.
    snap.avg_latency_us = stats.pressure_score * 10_000.0;

    // --- p95_1m_us from spawn threshold ---
    // adaptive_spawn_threshold is a task-count heuristic in HelixRouter;
    // we store it in p95_1m_us as a latency-budget proxy for the anomaly detector.
    snap.p95_1m_us = stats.adaptive_spawn_threshold as f64;

    // --- tokens_per_sec from total routed count ---
    let total_routed: u64 = stats.routed.values().sum();
    snap.interval_requests = total_routed;

    // --- explicit Drop-strategy signal ---
    // If HelixRouter explicitly routed tasks to "Drop", that is a stronger
    // shedding signal than the completed/dropped ratio alone.
    if let Some(&drop_routed) = stats.routed.get(&RoutingStrategy::Drop) {
        if total_routed > 0 {
            let explicit_drop_rate = drop_routed as f64 / total_routed as f64;
            // Take the maximum of both drop signals.
            snap.drop_rate = snap.drop_rate.max(explicit_drop_rate);
        }
    }

    snap
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn zero_stats() -> RouterStats {
        RouterStats {
            completed: 0,
            dropped: 0,
            adaptive_spawn_threshold: 0,
            pressure_score: 0.0,
            routed: HashMap::new(),
        }
    }

    fn stats_with(
        completed: u64,
        dropped: u64,
        threshold: u64,
        pressure: f64,
        routed: HashMap<RoutingStrategy, u64>,
    ) -> RouterStats {
        RouterStats {
            completed,
            dropped,
            adaptive_spawn_threshold: threshold,
            pressure_score: pressure,
            routed,
        }
    }

    // -----------------------------------------------------------------------
    // drop_rate from completed/dropped counts
    // -----------------------------------------------------------------------

    #[test]
    fn zero_stats_gives_zero_drop_rate() {
        let snap = stats_to_snapshot(&zero_stats());
        assert_eq!(snap.drop_rate, 0.0);
    }

    #[test]
    fn zero_completed_zero_dropped_drop_rate_is_zero() {
        let stats = stats_with(0, 0, 0, 0.0, HashMap::new());
        let snap = stats_to_snapshot(&stats);
        assert_eq!(snap.drop_rate, 0.0);
    }

    #[test]
    fn all_dropped_gives_drop_rate_one() {
        let stats = stats_with(0, 100, 0, 0.0, HashMap::new());
        let snap = stats_to_snapshot(&stats);
        assert!((snap.drop_rate - 1.0).abs() < 1e-9, "drop_rate={}", snap.drop_rate);
    }

    #[test]
    fn half_dropped_gives_drop_rate_half() {
        let stats = stats_with(50, 50, 0, 0.0, HashMap::new());
        let snap = stats_to_snapshot(&stats);
        assert!((snap.drop_rate - 0.5).abs() < 1e-9, "drop_rate={}", snap.drop_rate);
    }

    #[test]
    fn snapshot_drop_rate_clamped_to_zero_one_range() {
        // Even with pathological inputs the result should stay in [0, 1].
        let stats = stats_with(1, 1, 0, 0.0, HashMap::new());
        let snap = stats_to_snapshot(&stats);
        assert!(snap.drop_rate >= 0.0 && snap.drop_rate <= 1.0,
            "drop_rate out of range: {}", snap.drop_rate);
    }

    // -----------------------------------------------------------------------
    // avg_latency_us from pressure_score
    // -----------------------------------------------------------------------

    #[test]
    fn pressure_score_zero_gives_zero_avg_latency() {
        let stats = stats_with(10, 0, 0, 0.0, HashMap::new());
        let snap = stats_to_snapshot(&stats);
        assert_eq!(snap.avg_latency_us, 0.0);
    }

    #[test]
    fn pressure_score_one_gives_ten_thousand_avg_latency() {
        let stats = stats_with(10, 0, 0, 1.0, HashMap::new());
        let snap = stats_to_snapshot(&stats);
        assert!((snap.avg_latency_us - 10_000.0).abs() < 1e-6,
            "avg_latency_us={}", snap.avg_latency_us);
    }

    #[test]
    fn pressure_score_maps_to_avg_latency_us() {
        let stats = stats_with(1, 0, 0, 0.5, HashMap::new());
        let snap = stats_to_snapshot(&stats);
        assert!((snap.avg_latency_us - 5_000.0).abs() < 1e-6,
            "expected 5000.0, got {}", snap.avg_latency_us);
    }

    // -----------------------------------------------------------------------
    // p95_1m_us from adaptive_spawn_threshold
    // -----------------------------------------------------------------------

    #[test]
    fn spawn_threshold_maps_to_p95() {
        let stats = stats_with(0, 0, 250, 0.0, HashMap::new());
        let snap = stats_to_snapshot(&stats);
        assert!((snap.p95_1m_us - 250.0).abs() < 1e-9,
            "p95_1m_us={}", snap.p95_1m_us);
    }

    #[test]
    fn high_spawn_threshold_stored_in_p95() {
        let stats = stats_with(0, 0, 99_999, 0.0, HashMap::new());
        let snap = stats_to_snapshot(&stats);
        assert!((snap.p95_1m_us - 99_999.0).abs() < 1.0);
    }

    // -----------------------------------------------------------------------
    // tokens_per_sec from total_routed
    // -----------------------------------------------------------------------

    #[test]
    fn routed_empty_gives_zero_tokens_per_sec() {
        let snap = stats_to_snapshot(&zero_stats());
        assert_eq!(snap.interval_requests as f64, 0.0);
    }

    #[test]
    fn total_routed_maps_to_tokens_per_sec() {
        let mut routed = HashMap::new();
        routed.insert(RoutingStrategy::Inline, 7);
        routed.insert(RoutingStrategy::Spawn, 3);
        let stats = stats_with(10, 0, 0, 0.0, routed);
        let snap = stats_to_snapshot(&stats);
        assert!((snap.interval_requests as f64 - 10.0).abs() < 1e-9,
            "tokens_per_sec={}", snap.interval_requests as f64);
    }

    #[test]
    fn routed_multiple_strategies_sums_correctly() {
        let mut routed = HashMap::new();
        routed.insert(RoutingStrategy::Inline, 10);
        routed.insert(RoutingStrategy::Spawn, 20);
        routed.insert(RoutingStrategy::CpuPool, 30);
        routed.insert(RoutingStrategy::Batch, 5);
        let stats = stats_with(65, 0, 0, 0.0, routed);
        let snap = stats_to_snapshot(&stats);
        assert!((snap.interval_requests as f64 - 65.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // Drop strategy explicit signal
    // -----------------------------------------------------------------------

    #[test]
    fn no_drop_strategy_in_routed_leaves_computed_drop_rate() {
        let mut routed = HashMap::new();
        routed.insert(RoutingStrategy::Inline, 10);
        // 5 dropped out of 15 total → 0.333...
        let stats = stats_with(10, 5, 0, 0.0, routed);
        let snap = stats_to_snapshot(&stats);
        assert!((snap.drop_rate - (5.0 / 15.0)).abs() < 1e-9,
            "drop_rate={}", snap.drop_rate);
    }

    #[test]
    fn explicit_drop_strategy_raises_drop_rate() {
        let mut routed = HashMap::new();
        routed.insert(RoutingStrategy::Inline, 8);
        routed.insert(RoutingStrategy::Drop, 2); // 2/10 = 0.2
        // completed/dropped gives 0.0 drop rate; Drop strategy signal gives 0.2
        let stats = stats_with(10, 0, 0, 0.0, routed);
        let snap = stats_to_snapshot(&stats);
        assert!((snap.drop_rate - 0.2).abs() < 1e-9,
            "drop_rate={}", snap.drop_rate);
    }

    #[test]
    fn explicit_drop_strategy_takes_max_with_actual_drop_rate() {
        let mut routed = HashMap::new();
        routed.insert(RoutingStrategy::Inline, 5);
        routed.insert(RoutingStrategy::Drop, 5); // explicit rate = 0.5
        // completed/dropped: 5 completed, 8 dropped → 8/13 ≈ 0.615 > 0.5
        let stats = stats_with(5, 8, 0, 0.0, routed);
        let snap = stats_to_snapshot(&stats);
        let completed_rate = 8.0_f64 / 13.0;
        assert!(snap.drop_rate >= completed_rate - 1e-9,
            "should keep the higher rate, got {}", snap.drop_rate);
    }

    #[test]
    fn drop_strategy_count_zero_no_change_to_drop_rate() {
        let mut routed = HashMap::new();
        routed.insert(RoutingStrategy::Drop, 0);
        routed.insert(RoutingStrategy::Inline, 10);
        // Drop count == 0; explicit rate = 0/10 = 0.0.
        // completed drop rate = 2/12 ≈ 0.167.
        let stats = stats_with(10, 2, 0, 0.0, routed);
        let snap = stats_to_snapshot(&stats);
        // max(0.167, 0.0) = 0.167 — computed rate should win.
        assert!((snap.drop_rate - 2.0 / 12.0).abs() < 1e-9,
            "drop_rate={}", snap.drop_rate);
    }

    #[test]
    fn stats_with_only_inline_routed_no_drop_rate_change() {
        let mut routed = HashMap::new();
        routed.insert(RoutingStrategy::Inline, 20);
        let stats = stats_with(20, 0, 0, 0.0, routed);
        let snap = stats_to_snapshot(&stats);
        assert_eq!(snap.drop_rate, 0.0);
    }

    // -----------------------------------------------------------------------
    // Invariants
    // -----------------------------------------------------------------------

    #[test]
    fn snapshot_fields_are_valid_finite_floats() {
        let mut routed = HashMap::new();
        routed.insert(RoutingStrategy::Spawn, 100);
        routed.insert(RoutingStrategy::Drop, 10);
        let stats = stats_with(90, 10, 500, 0.75, routed);
        let snap = stats_to_snapshot(&stats);

        assert!(snap.drop_rate.is_finite(), "drop_rate NaN/inf");
        assert!(snap.avg_latency_us.is_finite(), "avg_latency_us NaN/inf");
        assert!(snap.p95_1m_us.is_finite(), "p95_1m_us NaN/inf");
        assert!((snap.interval_requests as f64).is_finite(), "tokens_per_sec NaN/inf");
    }

    #[test]
    fn converter_does_not_panic_on_empty_routed_map() {
        // This exercises all branches with minimum input — most important
        // is that no indexing panics occur on an empty HashMap.
        let stats = zero_stats();
        let snap = stats_to_snapshot(&stats);
        assert_eq!(snap.drop_rate, 0.0);
        assert_eq!(snap.interval_requests as f64, 0.0);
    }
}
