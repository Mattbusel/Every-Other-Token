//! Converts HelixRouter stats into TelemetrySnapshot format.
//!
//! ## Responsibility
//! Translate the external [`RouterStats`] schema produced by HelixRouter's HTTP
//! API into the internal [`TelemetrySnapshot`] type consumed by [`TelemetryBus`]
//! subscribers in the self-tune pipeline.
//!
//! ## Mapping
//! | RouterStats field                          | TelemetrySnapshot field | Notes                                        |
//! |--------------------------------------------|-------------------------|----------------------------------------------|
//! | dropped / (completed+dropped)              | drop_rate               | Falls back to 0 when total == 0              |
//! | weighted avg of latency_by_strategy.avg_ms | avg_latency_us          | Falls back to pressure_score×10_000 if empty |
//! | max(latency_by_strategy.p95_ms) × 1000     | p95_1m_us               | Real p95 in microseconds; 0 if no data       |
//! | sum(routed_by_strategy[*].count)           | interval_requests       | Total routed task count                      |
//! | routed[Drop] / total_routed                | drop_rate (max)         | Explicit drop signal                         |
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

    // --- avg_latency_us: use actual weighted latency when available ---
    // Compute weighted average of avg_ms across all strategies, weighted by count.
    // Fall back to scaling pressure_score when no latency rows are present.
    let (total_lat_count, weighted_ms_sum) = stats
        .latency_by_strategy
        .iter()
        .fold((0u64, 0.0f64), |(tc, wms), s| {
            (tc + s.count, wms + s.avg_ms * s.count as f64)
        });

    snap.avg_latency_us = if total_lat_count > 0 {
        // avg_ms → avg_us (× 1000)
        (weighted_ms_sum / total_lat_count as f64) * 1_000.0
    } else {
        // Fallback: pressure_score [0,1] scaled to microseconds
        stats.pressure_score * 10_000.0
    };

    // --- p95_1m_us: real p95 from the highest-latency strategy ---
    let max_p95_ms = stats
        .latency_by_strategy
        .iter()
        .map(|s| s.p95_ms)
        .max()
        .unwrap_or(0);
    snap.p95_1m_us = max_p95_ms as f64 * 1_000.0; // ms → us

    // --- interval_requests from total routed count ---
    let total_routed: u64 = stats.routed_by_strategy.iter().map(|r| r.count).sum();
    snap.interval_requests = total_routed;

    // --- explicit Drop-strategy signal ---
    // If HelixRouter explicitly routed tasks to "Drop", that is a stronger
    // shedding signal than the completed/dropped ratio alone.
    let drop_routed = stats
        .routed_by_strategy
        .iter()
        .find(|r| r.strategy == RoutingStrategy::Drop)
        .map(|r| r.count)
        .unwrap_or(0);

    if total_routed > 0 && drop_routed > 0 {
        let explicit_drop_rate = drop_routed as f64 / total_routed as f64;
        snap.drop_rate = snap.drop_rate.max(explicit_drop_rate);
    }

    snap
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::client::{LatencySummary, RoutedStrategyCount, RouterStats};

    fn zero_stats() -> RouterStats {
        RouterStats {
            completed: 0,
            dropped: 0,
            adaptive_spawn_threshold: 0,
            pressure_score: 0.0,
            routed_by_strategy: vec![],
            latency_by_strategy: vec![],
        }
    }

    fn stats_with(
        completed: u64,
        dropped: u64,
        threshold: u64,
        pressure: f64,
        routed: Vec<(RoutingStrategy, u64)>,
    ) -> RouterStats {
        RouterStats {
            completed,
            dropped,
            adaptive_spawn_threshold: threshold,
            pressure_score: pressure,
            routed_by_strategy: routed
                .into_iter()
                .map(|(strategy, count)| RoutedStrategyCount { strategy, count })
                .collect(),
            latency_by_strategy: vec![],
        }
    }

    fn stats_with_latency(
        completed: u64,
        dropped: u64,
        threshold: u64,
        pressure: f64,
        routed: Vec<(RoutingStrategy, u64)>,
        latency: Vec<(RoutingStrategy, u64, f64, f64, u64)>,
    ) -> RouterStats {
        RouterStats {
            completed,
            dropped,
            adaptive_spawn_threshold: threshold,
            pressure_score: pressure,
            routed_by_strategy: routed
                .into_iter()
                .map(|(strategy, count)| RoutedStrategyCount { strategy, count })
                .collect(),
            latency_by_strategy: latency
                .into_iter()
                .map(|(strategy, count, avg_ms, ema_ms, p95_ms)| LatencySummary {
                    strategy,
                    count,
                    avg_ms,
                    ema_ms,
                    p95_ms,
                })
                .collect(),
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
        let stats = stats_with(0, 0, 0, 0.0, vec![]);
        let snap = stats_to_snapshot(&stats);
        assert_eq!(snap.drop_rate, 0.0);
    }

    #[test]
    fn all_dropped_gives_drop_rate_one() {
        let stats = stats_with(0, 100, 0, 0.0, vec![]);
        let snap = stats_to_snapshot(&stats);
        assert!((snap.drop_rate - 1.0).abs() < 1e-9, "drop_rate={}", snap.drop_rate);
    }

    #[test]
    fn half_dropped_gives_drop_rate_half() {
        let stats = stats_with(50, 50, 0, 0.0, vec![]);
        let snap = stats_to_snapshot(&stats);
        assert!((snap.drop_rate - 0.5).abs() < 1e-9, "drop_rate={}", snap.drop_rate);
    }

    #[test]
    fn snapshot_drop_rate_in_zero_one_range() {
        let stats = stats_with(1, 1, 0, 0.0, vec![]);
        let snap = stats_to_snapshot(&stats);
        assert!(snap.drop_rate >= 0.0 && snap.drop_rate <= 1.0,
            "drop_rate out of range: {}", snap.drop_rate);
    }

    // -----------------------------------------------------------------------
    // avg_latency_us: real latency data takes priority over pressure_score
    // -----------------------------------------------------------------------

    #[test]
    fn pressure_score_zero_no_latency_rows_gives_zero_avg_latency() {
        let stats = stats_with(10, 0, 0, 0.0, vec![]);
        let snap = stats_to_snapshot(&stats);
        assert_eq!(snap.avg_latency_us, 0.0);
    }

    #[test]
    fn pressure_score_fallback_when_no_latency_rows() {
        let stats = stats_with(10, 0, 0, 1.0, vec![]);
        let snap = stats_to_snapshot(&stats);
        assert!((snap.avg_latency_us - 10_000.0).abs() < 1e-6,
            "avg_latency_us={}", snap.avg_latency_us);
    }

    #[test]
    fn pressure_score_half_fallback() {
        let stats = stats_with(1, 0, 0, 0.5, vec![]);
        let snap = stats_to_snapshot(&stats);
        assert!((snap.avg_latency_us - 5_000.0).abs() < 1e-6,
            "expected 5000.0, got {}", snap.avg_latency_us);
    }

    #[test]
    fn actual_latency_data_overrides_pressure_score() {
        // 10 Inline jobs at 2.0ms avg — should produce 2000 us, not pressure_score * 10_000
        let stats = stats_with_latency(
            10, 0, 0, 0.99,
            vec![(RoutingStrategy::Inline, 10)],
            vec![(RoutingStrategy::Inline, 10, 2.0, 1.9, 5)],
        );
        let snap = stats_to_snapshot(&stats);
        assert!((snap.avg_latency_us - 2_000.0).abs() < 1e-3,
            "expected 2000 us from real data, got {}", snap.avg_latency_us);
    }

    #[test]
    fn weighted_avg_latency_two_strategies() {
        // 4 Inline at 1ms, 6 Spawn at 4ms → weighted avg = (4*1 + 6*4)/10 = 2.8ms = 2800us
        let stats = stats_with_latency(
            10, 0, 0, 0.0,
            vec![(RoutingStrategy::Inline, 4), (RoutingStrategy::Spawn, 6)],
            vec![
                (RoutingStrategy::Inline, 4, 1.0, 1.0, 2),
                (RoutingStrategy::Spawn,  6, 4.0, 3.8, 9),
            ],
        );
        let snap = stats_to_snapshot(&stats);
        assert!((snap.avg_latency_us - 2_800.0).abs() < 1.0,
            "expected 2800 us, got {}", snap.avg_latency_us);
    }

    // -----------------------------------------------------------------------
    // p95_1m_us: real p95 from latency_by_strategy
    // -----------------------------------------------------------------------

    #[test]
    fn no_latency_rows_gives_zero_p95() {
        let stats = stats_with(0, 0, 250, 0.0, vec![]);
        let snap = stats_to_snapshot(&stats);
        assert_eq!(snap.p95_1m_us, 0.0);
    }

    #[test]
    fn p95_converted_from_ms_to_us() {
        // p95 of 10ms should become 10_000 us
        let stats = stats_with_latency(
            5, 0, 0, 0.0,
            vec![(RoutingStrategy::Inline, 5)],
            vec![(RoutingStrategy::Inline, 5, 2.0, 1.9, 10)],
        );
        let snap = stats_to_snapshot(&stats);
        assert!((snap.p95_1m_us - 10_000.0).abs() < 1e-6,
            "p95_1m_us={}", snap.p95_1m_us);
    }

    #[test]
    fn p95_takes_max_across_strategies() {
        // Two strategies, p95 of 5ms and 20ms → should use 20ms = 20_000 us
        let stats = stats_with_latency(
            20, 0, 0, 0.0,
            vec![(RoutingStrategy::Inline, 10), (RoutingStrategy::CpuPool, 10)],
            vec![
                (RoutingStrategy::Inline,  10, 1.0, 1.0, 5),
                (RoutingStrategy::CpuPool, 10, 8.0, 7.5, 20),
            ],
        );
        let snap = stats_to_snapshot(&stats);
        assert!((snap.p95_1m_us - 20_000.0).abs() < 1e-6,
            "expected 20_000 us, got {}", snap.p95_1m_us);
    }

    // -----------------------------------------------------------------------
    // interval_requests from total_routed
    // -----------------------------------------------------------------------

    #[test]
    fn routed_empty_gives_zero_interval_requests() {
        let snap = stats_to_snapshot(&zero_stats());
        assert_eq!(snap.interval_requests, 0);
    }

    #[test]
    fn total_routed_maps_to_interval_requests() {
        let stats = stats_with(10, 0, 0, 0.0,
            vec![(RoutingStrategy::Inline, 7), (RoutingStrategy::Spawn, 3)]);
        let snap = stats_to_snapshot(&stats);
        assert_eq!(snap.interval_requests, 10,
            "interval_requests={}", snap.interval_requests);
    }

    #[test]
    fn routed_multiple_strategies_sums_correctly() {
        let stats = stats_with(65, 0, 0, 0.0, vec![
            (RoutingStrategy::Inline,  10),
            (RoutingStrategy::Spawn,   20),
            (RoutingStrategy::CpuPool, 30),
            (RoutingStrategy::Batch,    5),
        ]);
        let snap = stats_to_snapshot(&stats);
        assert_eq!(snap.interval_requests, 65);
    }

    // -----------------------------------------------------------------------
    // Drop strategy explicit signal
    // -----------------------------------------------------------------------

    #[test]
    fn no_drop_strategy_in_routed_leaves_computed_drop_rate() {
        // 5 dropped out of 15 total → 0.333...
        let stats = stats_with(10, 5, 0, 0.0,
            vec![(RoutingStrategy::Inline, 10)]);
        let snap = stats_to_snapshot(&stats);
        assert!((snap.drop_rate - (5.0 / 15.0)).abs() < 1e-9,
            "drop_rate={}", snap.drop_rate);
    }

    #[test]
    fn explicit_drop_strategy_raises_drop_rate() {
        // completed/dropped gives 0.0 drop rate; Drop strategy gives 2/10 = 0.2
        let stats = stats_with(10, 0, 0, 0.0, vec![
            (RoutingStrategy::Inline, 8),
            (RoutingStrategy::Drop,   2),
        ]);
        let snap = stats_to_snapshot(&stats);
        assert!((snap.drop_rate - 0.2).abs() < 1e-9,
            "drop_rate={}", snap.drop_rate);
    }

    #[test]
    fn explicit_drop_strategy_takes_max_with_actual_drop_rate() {
        // explicit rate = 5/10 = 0.5; completed/dropped = 8/13 ≈ 0.615 > 0.5
        let stats = stats_with(5, 8, 0, 0.0, vec![
            (RoutingStrategy::Inline, 5),
            (RoutingStrategy::Drop,   5),
        ]);
        let snap = stats_to_snapshot(&stats);
        let completed_rate = 8.0_f64 / 13.0;
        assert!(snap.drop_rate >= completed_rate - 1e-9,
            "should keep the higher rate, got {}", snap.drop_rate);
    }

    #[test]
    fn drop_strategy_count_zero_no_change_to_drop_rate() {
        // Drop count == 0 so explicit rate branch is skipped.
        // completed drop rate = 2/12 ≈ 0.167 should win.
        let stats = stats_with(10, 2, 0, 0.0, vec![
            (RoutingStrategy::Drop,   0),
            (RoutingStrategy::Inline, 10),
        ]);
        let snap = stats_to_snapshot(&stats);
        assert!((snap.drop_rate - 2.0 / 12.0).abs() < 1e-9,
            "drop_rate={}", snap.drop_rate);
    }

    #[test]
    fn stats_with_only_inline_routed_no_drop_rate_change() {
        let stats = stats_with(20, 0, 0, 0.0,
            vec![(RoutingStrategy::Inline, 20)]);
        let snap = stats_to_snapshot(&stats);
        assert_eq!(snap.drop_rate, 0.0);
    }

    // -----------------------------------------------------------------------
    // Invariants
    // -----------------------------------------------------------------------

    #[test]
    fn snapshot_fields_are_valid_finite_floats() {
        let stats = stats_with_latency(90, 10, 500, 0.75,
            vec![
                (RoutingStrategy::Spawn, 90),
                (RoutingStrategy::Drop,  10),
            ],
            vec![
                (RoutingStrategy::Spawn, 90, 5.0, 4.8, 12),
                (RoutingStrategy::Drop,  10, 0.0, 0.0, 0),
            ],
        );
        let snap = stats_to_snapshot(&stats);

        assert!(snap.drop_rate.is_finite(),    "drop_rate NaN/inf");
        assert!(snap.avg_latency_us.is_finite(), "avg_latency_us NaN/inf");
        assert!(snap.p95_1m_us.is_finite(),    "p95_1m_us NaN/inf");
        assert!((snap.interval_requests as f64).is_finite(), "interval_requests NaN/inf");
    }

    #[test]
    fn converter_does_not_panic_on_empty_vecs() {
        let stats = zero_stats();
        let snap = stats_to_snapshot(&stats);
        assert_eq!(snap.drop_rate, 0.0);
        assert_eq!(snap.interval_requests, 0);
    }
}
