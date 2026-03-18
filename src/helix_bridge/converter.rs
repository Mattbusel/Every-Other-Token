use std::time::Instant;

use super::client::RouterStats;
use crate::self_tune::telemetry_bus::TelemetrySnapshot;

/// Convert a [`RouterStats`] polled from HelixRouter into a [`TelemetrySnapshot`]
/// suitable for publishing on the local [`crate::self_tune::telemetry_bus::TelemetryBus`].
///
/// Fields that have no direct equivalent in `RouterStats` are left at their zero
/// values so callers can merge or overlay additional data as needed.
pub fn stats_to_snapshot(stats: &RouterStats) -> TelemetrySnapshot {
    // Derive average latency from the per-strategy latency breakdown when available,
    // falling back to pressure_score as a rough proxy (scaled to microseconds).
    let avg_latency_us = if stats.latency_by_strategy.is_empty() {
        stats.pressure_score * 1_000.0
    } else {
        let total_count: u64 = stats.latency_by_strategy.iter().map(|l| l.count).sum();
        if total_count == 0 {
            0.0
        } else {
            let weighted: f64 = stats
                .latency_by_strategy
                .iter()
                .map(|l| l.avg_ms * l.count as f64)
                .sum();
            (weighted / total_count as f64) * 1_000.0
        }
    };

    let p95_us = stats
        .latency_by_strategy
        .iter()
        .map(|l| l.p95_ms as f64 * 1_000.0)
        .fold(0.0_f64, f64::max);

    let total = stats.completed + stats.dropped;
    let drop_rate = if total == 0 {
        0.0
    } else {
        stats.dropped as f64 / total as f64
    };

    TelemetrySnapshot {
        captured_at: Instant::now(),
        total_requests: stats.completed,
        total_dropped: stats.dropped,
        total_errors: 0,
        total_cache_hits: 0,
        total_dedup_hits: 0,
        interval_requests: 0,
        interval_errors: 0,
        drop_rate,
        cache_hit_rate: 0.0,
        avg_latency_us,
        p95_1m_us: p95_us,
        p95_5m_us: p95_us,
        p95_15m_us: p95_us,
        circuit_open: false,
        circuit_trips: 0,
        queue_depth: 0,
        queue_fill_frac: 0.0,
    }
}
