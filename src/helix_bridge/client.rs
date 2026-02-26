//! HTTP client and bridge runner for HelixRouter integration.

use std::sync::Arc;
use std::time::Duration;
use serde::{Deserialize, Serialize};

use tracing::{warn, error};

use crate::self_tune::telemetry_bus::TelemetryBus;
use super::converter::stats_to_snapshot;

// --- HelixRouter API types (mirror what HelixRouter exposes) ---

/// Strategy variants from HelixRouter. Mirrors helix_router::types::Strategy.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingStrategy {
    Inline,
    Spawn,
    CpuPool,
    Batch,
    Drop,
}

impl std::fmt::Display for RoutingStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RoutingStrategy::Inline  => write!(f, "Inline"),
            RoutingStrategy::Spawn   => write!(f, "Spawn"),
            RoutingStrategy::CpuPool => write!(f, "CpuPool"),
            RoutingStrategy::Batch   => write!(f, "Batch"),
            RoutingStrategy::Drop    => write!(f, "Drop"),
        }
    }
}

/// Per-strategy routing count row from HelixRouter's `/api/stats`.
/// Mirrors `CountRow` in helixrouter::web.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutedStrategyCount {
    pub strategy: RoutingStrategy,
    pub count: u64,
}

/// Mirrors HelixRouter's `/api/stats` JSON response exactly.
///
/// Field names match HelixRouter's `StatsResponse`:
/// - `routed_by_strategy` — per-strategy count vec (was previously a HashMap under `routed`)
/// - `latency_by_strategy` — per-strategy latency breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterStats {
    pub completed: u64,
    pub dropped: u64,
    pub adaptive_spawn_threshold: u64,
    pub pressure_score: f64,
    /// Per-strategy routing counts. Matches HelixRouter's `routed_by_strategy` field.
    #[serde(default)]
    pub routed_by_strategy: Vec<RoutedStrategyCount>,
    /// Per-strategy latency breakdown. Matches HelixRouter's `latency_by_strategy` field.
    #[serde(default)]
    pub latency_by_strategy: Vec<LatencySummary>,
}

/// Latency summary from HelixRouter (one entry per strategy).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencySummary {
    pub strategy: RoutingStrategy,
    pub count: u64,
    pub avg_ms: f64,
    pub ema_ms: f64,
    pub p95_ms: u64,
}

/// Subset of HelixRouter's RouterConfig that we may want to patch.
///
/// All fields are optional — only present fields are sent to the router.
/// Absent fields are omitted from the serialized JSON body.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RouterConfigPatch {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inline_threshold: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spawn_threshold: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_queue_cap: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_parallelism: Option<usize>,
    /// Override backpressure_busy_threshold — number of busy CPU workers above
    /// which Batch/Drop strategies are forced regardless of compute cost.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backpressure_busy_threshold: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_max_size: Option<usize>,
    /// Override batch_max_delay_ms — maximum time a batch waits before flushing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_max_delay_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ema_alpha: Option<f64>,
    /// Override adaptive_step — how aggressively the spawn threshold adapts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adaptive_step: Option<f64>,
    /// Override cpu_p95_budget_ms — p95 latency budget before CPU pool is considered overloaded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_p95_budget_ms: Option<u64>,
    /// Override adaptive_p95_threshold_factor — multiplier above which adaptive
    /// threshold raising triggers (e.g. 1.5 × cpu_p95_budget_ms).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adaptive_p95_threshold_factor: Option<f64>,
}

/// Neural router state snapshot from HelixRouter's `GET /api/neural`.
///
/// Mirrors HelixRouter's `NeuralSnapshot` struct.  When the neural router is
/// warmed up and `avg_reward` is positive, HelixRouter is already routing well
/// and aggressive config patches may destabilise learned behaviour.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralRouterState {
    /// Total outcomes the neural router has observed.
    pub sample_count: u64,
    /// Average reward across all observed outcomes. Positive → net good routing.
    pub avg_reward: f64,
    /// `true` once the neural router has passed its warm-up threshold.
    pub is_warmed_up: bool,
    /// Full 5×7 weight matrix `[strategy][feature]`.
    pub weights: Vec<Vec<f64>>,
}

/// Errors that can occur during bridge operations.
///
/// Each variant carries enough context to diagnose the failure without
/// needing to inspect the originating error directly.
#[derive(Debug)]
pub enum HelixBridgeError {
    /// The remote server replied with a non-2xx HTTP status code.
    Http { status: u16, url: String },
    /// Response body could not be parsed as the expected JSON structure.
    Json { field: String, detail: String },
    /// A TCP-level connection could not be established.
    Connect { url: String, detail: String },
}

impl std::fmt::Display for HelixBridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HelixBridgeError::Http { status, url } => {
                write!(f, "HTTP {status} from {url}")
            }
            HelixBridgeError::Json { field, detail } => {
                write!(f, "JSON parse error on field '{field}': {detail}")
            }
            HelixBridgeError::Connect { url, detail } => {
                write!(f, "Connection failed to {url}: {detail}")
            }
        }
    }
}

impl std::error::Error for HelixBridgeError {}

/// Configuration for the HelixBridge runtime.
#[derive(Debug, Clone)]
pub struct HelixBridgeConfig {
    /// Base URL of the HelixRouter HTTP API (e.g. `http://127.0.0.1:3000`).
    pub base_url: String,
    /// How often to poll `/api/stats`.
    pub poll_interval: Duration,
    /// TCP connection timeout.
    pub connect_timeout: Duration,
    /// Per-request read timeout.
    pub request_timeout: Duration,
    /// Pressure score threshold above which a tightening config patch is pushed.
    /// Set to `1.0` or higher to disable. Default: `0.8`.
    pub pressure_high_threshold: f64,
    /// Pressure score threshold below which a relaxing config patch is pushed.
    /// Set to `0.0` or lower to disable. Default: `0.3`.
    pub pressure_low_threshold: f64,
    /// Whether to push `RouterConfigPatch` updates back to HelixRouter.
    /// When `false`, the bridge is read-only (poll-only). Default: `true`.
    pub enable_config_push: bool,
}

impl HelixBridgeConfig {
    /// Create a config with sensible defaults.
    ///
    /// - poll_interval: 5 s
    /// - connect_timeout: 3 s
    /// - request_timeout: 10 s
    /// - pressure_high_threshold: 0.8
    /// - pressure_low_threshold: 0.3
    /// - enable_config_push: true
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            poll_interval: Duration::from_secs(5),
            connect_timeout: Duration::from_secs(3),
            request_timeout: Duration::from_secs(10),
            pressure_high_threshold: 0.8,
            pressure_low_threshold: 0.3,
            enable_config_push: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Pressure-reactive config suggestion (pure function, no I/O)
// ---------------------------------------------------------------------------

/// Suggest a [`RouterConfigPatch`] based on the current HelixRouter stats.
///
/// Returns `Some(patch)` when the observed `pressure_score` is outside the
/// healthy range `[low_threshold, high_threshold]`, otherwise `None`.
///
/// ## Strategy
/// - **High pressure** (`pressure_score > high_threshold`):
///   Increase `adaptive_step` so the spawn threshold climbs faster, and raise
///   `ema_alpha` so the latency EMA reacts faster to the increased load.
/// - **Low pressure** (`pressure_score < low_threshold`):
///   Decrease `adaptive_step` to prevent threshold overshoot, and lower
///   `ema_alpha` for smoother latency estimates.
/// - **Normal range**: returns `None` — no patch required.
///
/// All output values are clamped to known-safe ranges used by HelixRouter:
/// - `adaptive_step`: `[0.02, 0.50]`
/// - `ema_alpha`:     `[0.05, 0.90]`
///
/// # Panics
/// This function never panics.
pub fn suggest_config_patch(
    stats: &RouterStats,
    low_threshold: f64,
    high_threshold: f64,
) -> Option<RouterConfigPatch> {
    let pressure = stats.pressure_score;

    if pressure > high_threshold {
        // Under heavy load — push adaptive parameters higher so HelixRouter
        // reacts more aggressively to the pressure it is already observing.
        // Absolute target values; safe regardless of current HelixRouter state.
        // adaptive_step 0.20 = 2× the default (0.10) — moves spawn threshold faster.
        // ema_alpha    0.30 = 2× the default (0.15) — latency EMA tracks spikes faster.
        // backpressure_busy_threshold 5 (< default 7) — enter backpressure earlier
        //   under heavy load to shed work sooner.
        // adaptive_p95_threshold_factor 1.2 (< default 1.5) — trigger adaptive
        //   threshold raises sooner when p95 exceeds 1.2× budget.
        Some(RouterConfigPatch {
            adaptive_step: Some(0.20),
            ema_alpha: Some(0.30),
            backpressure_busy_threshold: Some(5),
            adaptive_p95_threshold_factor: Some(1.2),
            ..RouterConfigPatch::default()
        })
    } else if pressure < low_threshold {
        // Under light load — relax adaptive parameters for stability.
        // adaptive_step 0.05 = ½ the default — prevents threshold oscillation.
        // ema_alpha    0.10 = ⅔ the default  — smoother latency estimates.
        // backpressure_busy_threshold 9 (> default 7) — allow more parallel
        //   workers before entering backpressure during low-pressure periods.
        // adaptive_p95_threshold_factor 1.8 (> default 1.5) — wait for a larger
        //   p95 overshoot before raising adaptive threshold during quiet periods.
        Some(RouterConfigPatch {
            adaptive_step: Some(0.05),
            ema_alpha: Some(0.10),
            backpressure_busy_threshold: Some(9),
            adaptive_p95_threshold_factor: Some(1.8),
            ..RouterConfigPatch::default()
        })
    } else {
        None
    }
}

/// The bridge runner.
///
/// Polls HelixRouter at `config.poll_interval` and feeds converted stats
/// into the [`TelemetryBus`]. Use [`HelixBridgeBuilder`] for construction.
pub struct HelixBridge {
    config: HelixBridgeConfig,
    bus: Arc<TelemetryBus>,
    client: reqwest::Client,
}

impl HelixBridge {
    /// Start building a bridge aimed at `base_url`.
    pub fn builder(base_url: impl Into<String>) -> HelixBridgeBuilder {
        HelixBridgeBuilder::new(base_url)
    }

    /// Fetch the current stats snapshot from HelixRouter's `/api/stats`.
    ///
    /// Accepts both the direct `RouterStats` shape and a `{ "stats": RouterStats }`
    /// wrapper so that the bridge is forward-compatible with HelixRouter's response
    /// envelope changes.
    ///
    /// # Returns
    /// - `Ok(RouterStats)` — on a successful 2xx response with parseable JSON.
    /// - `Err(HelixBridgeError::Connect)` — when the TCP connection fails.
    /// - `Err(HelixBridgeError::Http)` — when the server replies with a non-2xx code.
    /// - `Err(HelixBridgeError::Json)` — when the body cannot be parsed.
    ///
    /// # Panics
    /// This function never panics.
    pub async fn fetch_stats(&self) -> Result<RouterStats, HelixBridgeError> {
        let url = format!("{}/api/stats", self.config.base_url);
        let resp = self.client.get(&url).send().await.map_err(|e| {
            HelixBridgeError::Connect {
                url: url.clone(),
                detail: e.to_string(),
            }
        })?;

        if !resp.status().is_success() {
            return Err(HelixBridgeError::Http {
                status: resp.status().as_u16(),
                url,
            });
        }

        let bytes = resp.bytes().await.map_err(|e| HelixBridgeError::Json {
            field: "body".into(),
            detail: e.to_string(),
        })?;

        // Try direct RouterStats parse first.
        if let Ok(stats) = serde_json::from_slice::<RouterStats>(&bytes) {
            return Ok(stats);
        }

        // Fall back to a wrapped shape: { "stats": { ... } }.
        #[derive(Deserialize)]
        struct Wrapped {
            stats: RouterStats,
        }

        serde_json::from_slice::<Wrapped>(&bytes)
            .map(|w| w.stats)
            .map_err(|e| HelixBridgeError::Json {
                field: "stats".into(),
                detail: e.to_string(),
            })
    }

    /// Push a partial config update to HelixRouter's `/api/config`.
    ///
    /// Only fields present in `patch` are serialized into the JSON body;
    /// absent fields are omitted entirely (not set to `null`).
    ///
    /// # Returns
    /// - `Ok(())` — on a 2xx response.
    /// - `Err(HelixBridgeError::Connect)` — on TCP-level failure.
    /// - `Err(HelixBridgeError::Http)` — on a non-2xx response.
    ///
    /// # Panics
    /// This function never panics.
    pub async fn push_config(&self, patch: &RouterConfigPatch) -> Result<(), HelixBridgeError> {
        let url = format!("{}/api/config", self.config.base_url);
        let resp = self
            .client
            .patch(&url)
            .json(patch)
            .send()
            .await
            .map_err(|e| HelixBridgeError::Connect {
                url: url.clone(),
                detail: e.to_string(),
            })?;

        if !resp.status().is_success() {
            return Err(HelixBridgeError::Http {
                status: resp.status().as_u16(),
                url,
            });
        }

        Ok(())
    }

    /// Fetch the current neural router state from HelixRouter's `/api/neural`.
    ///
    /// Returns `Ok(None)` when the endpoint returns a non-2xx status (e.g. if
    /// HelixRouter is an older version without the neural endpoint). Returns
    /// `Err` only on transport-level failures.
    ///
    /// # Panics
    /// This function never panics.
    pub async fn fetch_neural_state(
        &self,
    ) -> Result<Option<NeuralRouterState>, HelixBridgeError> {
        let url = format!("{}/api/neural", self.config.base_url);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| HelixBridgeError::Connect {
                url: url.clone(),
                detail: e.to_string(),
            })?;

        if !resp.status().is_success() {
            // Older HelixRouter without /api/neural — treat as absent, not an error.
            return Ok(None);
        }

        let state = resp
            .json::<NeuralRouterState>()
            .await
            .map_err(|e| HelixBridgeError::Json {
                field: "neural".into(),
                detail: e.to_string(),
            })?;

        Ok(Some(state))
    }

    /// Run the polling loop indefinitely.
    ///
    /// On each tick, fetches `/api/stats`, records the converted snapshot
    /// into the `TelemetryBus`, and — when [`HelixBridgeConfig::enable_config_push`]
    /// is `true` — pushes a [`RouterConfigPatch`] back to HelixRouter whenever
    /// the observed pressure is outside the healthy range defined by
    /// `pressure_low_threshold` / `pressure_high_threshold`.
    ///
    /// Connection failures are soft-errors — the loop skips the tick and retries.
    ///
    /// Cancel the task (drop the `JoinHandle`) to stop the loop cleanly.
    ///
    /// # Panics
    /// This function never panics.
    pub async fn run(self) {
        let mut ticker = tokio::time::interval(self.config.poll_interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        let mut consecutive_failures: u32 = 0;

        loop {
            ticker.tick().await;

            match self.fetch_stats().await {
                Ok(stats) => {
                    consecutive_failures = 0;

                    let snap = stats_to_snapshot(&stats);

                    // Feed avg latency signal into the bus.
                    // snap.avg_latency_us is derived from actual weighted latency data
                    // (or pressure_score fallback when no latency rows are present).
                    self.bus.record_latency(
                        crate::self_tune::telemetry_bus::PipelineStage::Inference,
                        (snap.avg_latency_us as u64).max(1),
                    );

                    // Feed p95 latency as a high-percentile signal on the Cache stage
                    // (closest proxy for HelixRouter's spawn-threshold latency budget).
                    if snap.p95_1m_us > 0.0 {
                        self.bus.record_latency(
                            crate::self_tune::telemetry_bus::PipelineStage::Cache,
                            snap.p95_1m_us as u64,
                        );
                    }

                    // Feed drop pressure as an auxiliary signal on the Other stage.
                    if snap.drop_rate > 0.0 {
                        self.bus.record_latency(
                            crate::self_tune::telemetry_bus::PipelineStage::Other,
                            (snap.drop_rate * 1_000.0) as u64 + 1,
                        );
                    }

                    // ── Feedback: push config patch back to HelixRouter ───────
                    // Only when config push is enabled and pressure is out of range.
                    // Additionally, if the neural router is warmed up with positive
                    // avg_reward, it is already routing well — suppress redundant
                    // config patches to avoid destabilising learned behaviour.
                    if self.config.enable_config_push {
                        // Best-effort: fetch neural state to inform patch decision.
                        let neural_healthy = match self.fetch_neural_state().await {
                            Ok(Some(ns)) => ns.is_warmed_up && ns.avg_reward > 0.0,
                            _ => false,
                        };

                        if let Some(patch) = suggest_config_patch(
                            &stats,
                            self.config.pressure_low_threshold,
                            self.config.pressure_high_threshold,
                        ) {
                            // Skip the patch when neural routing is healthy AND
                            // pressure is only mildly elevated (within 10% of threshold).
                            let pressure_far_from_threshold = stats.pressure_score
                                > self.config.pressure_high_threshold * 1.1
                                || stats.pressure_score
                                    < self.config.pressure_low_threshold * 0.9;
                            let should_push = !neural_healthy || pressure_far_from_threshold;

                            if should_push {
                                if let Err(e) = self.push_config(&patch).await {
                                    warn!(
                                        error = %e,
                                        pressure = stats.pressure_score,
                                        neural_healthy,
                                        url = %self.config.base_url,
                                        "HelixBridge config push failed, continuing"
                                    );
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    consecutive_failures = consecutive_failures.saturating_add(1);

                    if consecutive_failures >= 5 {
                        error!(
                            error = %e,
                            url = %self.config.base_url,
                            consecutive_failures,
                            "HelixBridge poll failed repeatedly, will retry next tick"
                        );
                    } else {
                        warn!(
                            error = %e,
                            url = %self.config.base_url,
                            "HelixBridge poll failed, will retry next tick"
                        );
                    }
                }
            }
        }
    }
}

/// Builder for [`HelixBridge`].
///
/// # Example
/// ```rust,ignore
/// let bridge = HelixBridge::builder("http://127.0.0.1:3000")
///     .bus(Arc::clone(&bus))
///     .poll_interval(Duration::from_secs(10))
///     .build()
///     .expect("bus is required");
/// ```
pub struct HelixBridgeBuilder {
    config: HelixBridgeConfig,
    bus: Option<Arc<TelemetryBus>>,
}

impl HelixBridgeBuilder {
    /// Create a builder targeting `base_url`.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            config: HelixBridgeConfig::new(base_url),
            bus: None,
        }
    }

    /// Attach the [`TelemetryBus`] that converted stats will be fed into.
    ///
    /// This field is **required** — [`build`](Self::build) will return `Err`
    /// if it is not set.
    pub fn bus(mut self, bus: Arc<TelemetryBus>) -> Self {
        self.bus = Some(bus);
        self
    }

    /// Override the stats polling interval (default 5 s).
    pub fn poll_interval(mut self, interval: Duration) -> Self {
        self.config.poll_interval = interval;
        self
    }

    /// Override the TCP connect timeout (default 3 s).
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.config.connect_timeout = timeout;
        self
    }

    /// Override the per-request read timeout (default 10 s).
    pub fn request_timeout(mut self, timeout: Duration) -> Self {
        self.config.request_timeout = timeout;
        self
    }

    /// Consume the builder and construct a [`HelixBridge`].
    ///
    /// # Errors
    /// Returns `Err("bus is required")` when no bus was provided via
    /// [`bus`](Self::bus).
    ///
    /// # Panics
    /// This function never panics.
    pub fn build(self) -> Result<HelixBridge, &'static str> {
        let bus = self.bus.ok_or("bus is required")?;

        // reqwest::Client::builder() can fail in extreme environments, but
        // unwrap_or_default() falls back to a default client instead of panicking.
        let client = reqwest::Client::builder()
            .connect_timeout(self.config.connect_timeout)
            .timeout(self.config.request_timeout)
            .build()
            .unwrap_or_default();

        Ok(HelixBridge {
            config: self.config,
            bus,
            client,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::self_tune::telemetry_bus::{BusConfig, TelemetryBus};
    use std::sync::Arc;
    use std::time::Duration;

    fn make_bus() -> Arc<TelemetryBus> {
        Arc::new(TelemetryBus::new(BusConfig {
            emit_interval: Duration::from_secs(60),
            queue_capacity: 64,
        }))
    }

    fn make_stats() -> RouterStats {
        RouterStats {
            completed: 15,
            dropped: 0,
            adaptive_spawn_threshold: 100,
            pressure_score: 0.2,
            routed_by_strategy: vec![
                RoutedStrategyCount { strategy: RoutingStrategy::Inline, count: 10 },
                RoutedStrategyCount { strategy: RoutingStrategy::Spawn, count: 5 },
            ],
            latency_by_strategy: vec![
                LatencySummary { strategy: RoutingStrategy::Inline, count: 10, avg_ms: 1.2, ema_ms: 1.1, p95_ms: 4 },
                LatencySummary { strategy: RoutingStrategy::Spawn, count: 5, avg_ms: 8.5, ema_ms: 8.0, p95_ms: 18 },
            ],
        }
    }

    // -----------------------------------------------------------------------
    // Builder tests
    // -----------------------------------------------------------------------

    #[test]
    fn builder_requires_bus_returns_err_without_bus() {
        let result = HelixBridge::builder("http://localhost:3000").build();
        assert!(result.is_err());
        assert!(result.is_err());
        assert_eq!(result.err().unwrap(), "bus is required");
    }

    #[test]
    fn builder_with_bus_builds_ok() {
        let bus = make_bus();
        let result = HelixBridge::builder("http://localhost:3000")
            .bus(Arc::clone(&bus))
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn builder_poll_interval_set() {
        let bus = make_bus();
        let bridge = HelixBridge::builder("http://localhost:3000")
            .bus(Arc::clone(&bus))
            .poll_interval(Duration::from_secs(30))
            .build()
            .unwrap();
        assert_eq!(bridge.config.poll_interval, Duration::from_secs(30));
    }

    #[test]
    fn builder_connect_timeout_set() {
        let bus = make_bus();
        let bridge = HelixBridge::builder("http://localhost:3000")
            .bus(Arc::clone(&bus))
            .connect_timeout(Duration::from_secs(7))
            .build()
            .unwrap();
        assert_eq!(bridge.config.connect_timeout, Duration::from_secs(7));
    }

    #[test]
    fn builder_request_timeout_set() {
        let bus = make_bus();
        let bridge = HelixBridge::builder("http://localhost:3000")
            .bus(Arc::clone(&bus))
            .request_timeout(Duration::from_secs(20))
            .build()
            .unwrap();
        assert_eq!(bridge.config.request_timeout, Duration::from_secs(20));
    }

    #[test]
    fn builder_default_config_poll_interval_five_seconds() {
        let bus = make_bus();
        let bridge = HelixBridge::builder("http://localhost:3000")
            .bus(Arc::clone(&bus))
            .build()
            .unwrap();
        assert_eq!(bridge.config.poll_interval, Duration::from_secs(5));
    }

    #[test]
    fn builder_builds_with_all_options_set() {
        let bus = make_bus();
        let result = HelixBridge::builder("http://127.0.0.1:4000")
            .bus(Arc::clone(&bus))
            .poll_interval(Duration::from_secs(2))
            .connect_timeout(Duration::from_secs(1))
            .request_timeout(Duration::from_secs(5))
            .build();
        assert!(result.is_ok());
        let bridge = result.unwrap();
        assert_eq!(bridge.config.base_url, "http://127.0.0.1:4000");
        assert_eq!(bridge.config.poll_interval, Duration::from_secs(2));
        assert_eq!(bridge.config.connect_timeout, Duration::from_secs(1));
        assert_eq!(bridge.config.request_timeout, Duration::from_secs(5));
    }

    // -----------------------------------------------------------------------
    // HelixBridgeConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn config_new_has_default_poll_interval() {
        let cfg = HelixBridgeConfig::new("http://localhost:3000");
        assert_eq!(cfg.poll_interval, Duration::from_secs(5));
    }

    #[test]
    fn config_new_stores_base_url() {
        let cfg = HelixBridgeConfig::new("http://example.com:8080");
        assert_eq!(cfg.base_url, "http://example.com:8080");
    }

    #[test]
    fn config_new_has_default_pressure_high_threshold() {
        let cfg = HelixBridgeConfig::new("http://localhost:3000");
        assert!((cfg.pressure_high_threshold - 0.8).abs() < 1e-9);
    }

    #[test]
    fn config_new_has_default_pressure_low_threshold() {
        let cfg = HelixBridgeConfig::new("http://localhost:3000");
        assert!((cfg.pressure_low_threshold - 0.3).abs() < 1e-9);
    }

    #[test]
    fn config_new_has_config_push_enabled_by_default() {
        let cfg = HelixBridgeConfig::new("http://localhost:3000");
        assert!(cfg.enable_config_push);
    }

    // -----------------------------------------------------------------------
    // suggest_config_patch tests
    // -----------------------------------------------------------------------

    fn make_stats_with_pressure(pressure: f64) -> RouterStats {
        RouterStats {
            completed: 100,
            dropped: 0,
            adaptive_spawn_threshold: 60_000,
            pressure_score: pressure,
            routed_by_strategy: vec![],
            latency_by_strategy: vec![],
        }
    }

    #[test]
    fn suggest_patch_returns_none_in_normal_range() {
        let stats = make_stats_with_pressure(0.5);
        let patch = suggest_config_patch(&stats, 0.3, 0.8);
        assert!(patch.is_none(), "mid-range pressure should produce no patch");
    }

    #[test]
    fn suggest_patch_returns_none_exactly_at_low_threshold() {
        let stats = make_stats_with_pressure(0.3);
        let patch = suggest_config_patch(&stats, 0.3, 0.8);
        // pressure == threshold is not strictly less-than, so no patch.
        assert!(patch.is_none());
    }

    #[test]
    fn suggest_patch_returns_none_exactly_at_high_threshold() {
        let stats = make_stats_with_pressure(0.8);
        let patch = suggest_config_patch(&stats, 0.3, 0.8);
        // pressure == threshold is not strictly greater-than, so no patch.
        assert!(patch.is_none());
    }

    #[test]
    fn suggest_patch_high_pressure_returns_some() {
        let stats = make_stats_with_pressure(0.85);
        let patch = suggest_config_patch(&stats, 0.3, 0.8);
        assert!(patch.is_some(), "high pressure should produce a patch");
    }

    #[test]
    fn suggest_patch_high_pressure_increases_adaptive_step() {
        let stats = make_stats_with_pressure(0.9);
        let patch = suggest_config_patch(&stats, 0.3, 0.8).expect("should produce patch");
        let step = patch.adaptive_step.expect("adaptive_step should be set");
        // Under high pressure we want a value higher than the default (0.10).
        assert!(step > 0.10, "adaptive_step should be above default: {step}");
        assert!(step <= 0.50, "adaptive_step should not exceed 0.50: {step}");
    }

    #[test]
    fn suggest_patch_high_pressure_increases_ema_alpha() {
        let stats = make_stats_with_pressure(0.95);
        let patch = suggest_config_patch(&stats, 0.3, 0.8).expect("should produce patch");
        let alpha = patch.ema_alpha.expect("ema_alpha should be set");
        // Under high pressure we want a value higher than the default (0.15).
        assert!(alpha > 0.15, "ema_alpha should be above default: {alpha}");
        assert!(alpha <= 0.90, "ema_alpha should not exceed 0.90: {alpha}");
    }

    #[test]
    fn suggest_patch_low_pressure_returns_some() {
        let stats = make_stats_with_pressure(0.1);
        let patch = suggest_config_patch(&stats, 0.3, 0.8);
        assert!(patch.is_some(), "low pressure should produce a patch");
    }

    #[test]
    fn suggest_patch_low_pressure_reduces_adaptive_step() {
        let stats = make_stats_with_pressure(0.05);
        let patch = suggest_config_patch(&stats, 0.3, 0.8).expect("should produce patch");
        let step = patch.adaptive_step.expect("adaptive_step should be set");
        // Under low pressure we want a value lower than the default (0.10).
        assert!(step < 0.10, "adaptive_step should be below default: {step}");
        assert!(step >= 0.02, "adaptive_step should not go below 0.02: {step}");
    }

    #[test]
    fn suggest_patch_low_pressure_reduces_ema_alpha() {
        let stats = make_stats_with_pressure(0.0);
        let patch = suggest_config_patch(&stats, 0.3, 0.8).expect("should produce patch");
        let alpha = patch.ema_alpha.expect("ema_alpha should be set");
        // Under low pressure we want a value lower than the default (0.15).
        assert!(alpha < 0.15, "ema_alpha should be below default: {alpha}");
        assert!(alpha >= 0.05, "ema_alpha should not go below 0.05: {alpha}");
    }

    #[test]
    fn suggest_patch_high_pressure_only_sets_adaptive_and_ema() {
        // Other fields (inline_threshold, spawn_threshold, etc.) must be None.
        let stats = make_stats_with_pressure(0.99);
        let patch = suggest_config_patch(&stats, 0.3, 0.8).expect("should produce patch");
        assert!(patch.inline_threshold.is_none());
        assert!(patch.spawn_threshold.is_none());
        assert!(patch.cpu_queue_cap.is_none());
        assert!(patch.cpu_parallelism.is_none());
        assert!(patch.batch_max_size.is_none());
    }

    #[test]
    fn suggest_patch_low_pressure_only_sets_adaptive_and_ema() {
        let stats = make_stats_with_pressure(0.0);
        let patch = suggest_config_patch(&stats, 0.3, 0.8).expect("should produce patch");
        assert!(patch.inline_threshold.is_none());
        assert!(patch.spawn_threshold.is_none());
        assert!(patch.cpu_queue_cap.is_none());
        assert!(patch.cpu_parallelism.is_none());
        assert!(patch.batch_max_size.is_none());
        // Low pressure: loosen backpressure threshold and p95 factor
        assert_eq!(patch.backpressure_busy_threshold, Some(9));
        assert!(patch.adaptive_p95_threshold_factor.is_some());
    }

    #[test]
    fn suggest_patch_high_pressure_sets_backpressure_threshold() {
        let stats = make_stats_with_pressure(1.0);
        let patch = suggest_config_patch(&stats, 0.3, 0.8).expect("should produce patch");
        // High pressure: tighten backpressure threshold so HelixRouter sheds earlier
        assert_eq!(patch.backpressure_busy_threshold, Some(5));
    }

    #[test]
    fn suggest_patch_high_pressure_sets_adaptive_p95_factor() {
        let stats = make_stats_with_pressure(1.0);
        let patch = suggest_config_patch(&stats, 0.3, 0.8).expect("should produce patch");
        // High pressure: trigger adaptive raises sooner (1.2 < default 1.5)
        let factor = patch.adaptive_p95_threshold_factor.expect("should be set");
        assert!(factor < 1.5, "high pressure should tighten factor, got {factor}");
    }

    #[test]
    fn suggest_patch_low_pressure_backpressure_threshold_greater_than_high() {
        let high_stats = make_stats_with_pressure(1.0);
        let low_stats = make_stats_with_pressure(0.0);
        let high_patch = suggest_config_patch(&high_stats, 0.3, 0.8).unwrap();
        let low_patch = suggest_config_patch(&low_stats, 0.3, 0.8).unwrap();
        let high_thresh = high_patch.backpressure_busy_threshold.unwrap();
        let low_thresh = low_patch.backpressure_busy_threshold.unwrap();
        assert!(low_thresh > high_thresh, "low pressure should relax threshold: {low_thresh} > {high_thresh}");
    }

    #[test]
    fn suggest_patch_disabled_thresholds_never_fire() {
        // Setting thresholds to extreme values disables both branches.
        let high_stats = make_stats_with_pressure(1.0);
        let low_stats = make_stats_with_pressure(0.0);
        // Disable high-pressure branch: threshold above max pressure.
        assert!(suggest_config_patch(&high_stats, 0.0, 2.0).is_none());
        // Disable low-pressure branch: threshold below min pressure.
        assert!(suggest_config_patch(&low_stats, -1.0, 1.0).is_none());
    }

    // -----------------------------------------------------------------------
    // HelixBridgeError Display / std::error::Error
    // -----------------------------------------------------------------------

    #[test]
    fn helix_bridge_error_display_http() {
        let err = HelixBridgeError::Http {
            status: 503,
            url: "http://localhost:3000/api/stats".to_string(),
        };
        let s = err.to_string();
        assert!(s.contains("503"), "expected status in display: {s}");
        assert!(s.contains("http://localhost:3000/api/stats"), "expected url: {s}");
    }

    #[test]
    fn helix_bridge_error_display_json() {
        let err = HelixBridgeError::Json {
            field: "stats".to_string(),
            detail: "missing field `completed`".to_string(),
        };
        let s = err.to_string();
        assert!(s.contains("stats"), "field in display: {s}");
        assert!(s.contains("missing field"), "detail in display: {s}");
    }

    #[test]
    fn helix_bridge_error_display_connect() {
        let err = HelixBridgeError::Connect {
            url: "http://localhost:3000".to_string(),
            detail: "connection refused".to_string(),
        };
        let s = err.to_string();
        assert!(s.contains("http://localhost:3000"), "url in display: {s}");
        assert!(s.contains("connection refused"), "detail in display: {s}");
    }

    #[test]
    fn helix_bridge_error_is_std_error() {
        // Compile-time proof that HelixBridgeError implements std::error::Error.
        fn assert_error<E: std::error::Error>(_: &E) {}
        let err = HelixBridgeError::Http { status: 500, url: "x".to_string() };
        assert_error(&err);
    }

    #[test]
    fn helix_bridge_error_debug_formats() {
        let err = HelixBridgeError::Connect {
            url: "http://a".to_string(),
            detail: "refused".to_string(),
        };
        let dbg = format!("{:?}", err);
        assert!(dbg.contains("Connect"), "Debug should contain variant name: {dbg}");
    }

    // -----------------------------------------------------------------------
    // RouterStats serde
    // -----------------------------------------------------------------------

    #[test]
    fn router_stats_serde_roundtrip() {
        let stats = make_stats();
        let json = serde_json::to_string(&stats).unwrap();
        let back: RouterStats = serde_json::from_str(&json).unwrap();
        assert_eq!(back.completed, stats.completed);
        assert_eq!(back.dropped, stats.dropped);
        assert_eq!(back.adaptive_spawn_threshold, stats.adaptive_spawn_threshold);
        assert!((back.pressure_score - stats.pressure_score).abs() < 1e-9);
    }

    #[test]
    fn router_stats_zero_completed_is_valid() {
        let stats = RouterStats {
            completed: 0,
            dropped: 0,
            adaptive_spawn_threshold: 0,
            pressure_score: 0.0,
            routed_by_strategy: vec![],
            latency_by_strategy: vec![],
        };
        let json = serde_json::to_string(&stats).unwrap();
        let back: RouterStats = serde_json::from_str(&json).unwrap();
        assert_eq!(back.completed, 0);
    }

    #[test]
    fn router_stats_pressure_score_range() {
        let stats = make_stats();
        assert!(stats.pressure_score.is_finite(), "pressure_score must be finite");
        assert!(!stats.pressure_score.is_nan(), "pressure_score must not be NaN");
    }

    // -----------------------------------------------------------------------
    // RouterConfigPatch serde / skip_serializing_if
    // -----------------------------------------------------------------------

    #[test]
    fn router_config_patch_default_all_none() {
        let patch = RouterConfigPatch::default();
        assert!(patch.inline_threshold.is_none());
        assert!(patch.spawn_threshold.is_none());
        assert!(patch.cpu_queue_cap.is_none());
        assert!(patch.cpu_parallelism.is_none());
        assert!(patch.batch_max_size.is_none());
        assert!(patch.ema_alpha.is_none());
    }

    #[test]
    fn router_config_patch_serialize_skips_none_fields() {
        let patch = RouterConfigPatch::default();
        let json = serde_json::to_string(&patch).unwrap();
        // A fully-None patch must serialize to an empty object.
        assert_eq!(json.trim(), "{}");
    }

    #[test]
    fn router_config_patch_serialize_includes_set_fields() {
        let patch = RouterConfigPatch {
            inline_threshold: Some(100),
            ema_alpha: Some(0.3),
            ..Default::default()
        };
        let json = serde_json::to_string(&patch).unwrap();
        assert!(json.contains("inline_threshold"), "json: {json}");
        assert!(json.contains("ema_alpha"), "json: {json}");
        assert!(!json.contains("spawn_threshold"), "absent field should be omitted: {json}");
    }

    #[test]
    fn router_config_patch_partial_only_set_fields_in_json() {
        let patch = RouterConfigPatch {
            cpu_queue_cap: Some(512),
            ..Default::default()
        };
        let json = serde_json::to_string(&patch).unwrap();
        assert!(json.contains("cpu_queue_cap"));
        assert!(!json.contains("inline_threshold"));
        assert!(!json.contains("spawn_threshold"));
        assert!(!json.contains("ema_alpha"));
    }

    #[test]
    fn router_config_patch_inline_threshold_serialized() {
        let patch = RouterConfigPatch {
            inline_threshold: Some(42),
            ..Default::default()
        };
        let json = serde_json::to_string(&patch).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["inline_threshold"], 42);
    }

    #[test]
    fn router_config_patch_spawn_threshold_serialized() {
        let patch = RouterConfigPatch {
            spawn_threshold: Some(256),
            ..Default::default()
        };
        let json = serde_json::to_string(&patch).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["spawn_threshold"], 256);
    }

    // -----------------------------------------------------------------------
    // RoutingStrategy Display
    // -----------------------------------------------------------------------

    #[test]
    fn routing_strategy_display_inline() {
        assert_eq!(RoutingStrategy::Inline.to_string(), "Inline");
    }

    #[test]
    fn routing_strategy_display_spawn() {
        assert_eq!(RoutingStrategy::Spawn.to_string(), "Spawn");
    }

    #[test]
    fn routing_strategy_display_cpu_pool() {
        assert_eq!(RoutingStrategy::CpuPool.to_string(), "CpuPool");
    }

    #[test]
    fn routing_strategy_display_batch() {
        assert_eq!(RoutingStrategy::Batch.to_string(), "Batch");
    }

    #[test]
    fn routing_strategy_display_drop() {
        assert_eq!(RoutingStrategy::Drop.to_string(), "Drop");
    }

    #[test]
    fn routing_strategy_serde_roundtrip() {
        let strategies = [
            RoutingStrategy::Inline,
            RoutingStrategy::Spawn,
            RoutingStrategy::CpuPool,
            RoutingStrategy::Batch,
            RoutingStrategy::Drop,
        ];
        for s in &strategies {
            let json = serde_json::to_string(s).unwrap();
            let back: RoutingStrategy = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, s, "roundtrip failed for {:?}", s);
        }
    }

    // -----------------------------------------------------------------------
    // LatencySummary serde
    // -----------------------------------------------------------------------

    #[test]
    fn latency_summary_serde_roundtrip() {
        let summary = LatencySummary {
            strategy: RoutingStrategy::CpuPool,
            count: 42,
            avg_ms: 1.5,
            ema_ms: 1.4,
            p95_ms: 5,
        };
        let json = serde_json::to_string(&summary).unwrap();
        let back: LatencySummary = serde_json::from_str(&json).unwrap();
        assert_eq!(back.count, 42);
        assert_eq!(back.strategy, RoutingStrategy::CpuPool);
        assert!((back.avg_ms - 1.5).abs() < 1e-9);
        assert!((back.ema_ms - 1.4).abs() < 1e-9);
        assert_eq!(back.p95_ms, 5);
    }

    // -----------------------------------------------------------------------
    // NeuralRouterState serde
    // -----------------------------------------------------------------------

    #[test]
    fn neural_router_state_serde_roundtrip() {
        let state = NeuralRouterState {
            sample_count: 42,
            avg_reward: 0.75,
            is_warmed_up: true,
            weights: vec![vec![0.1, 0.2]; 5],
        };
        let json = serde_json::to_string(&state).unwrap();
        let back: NeuralRouterState = serde_json::from_str(&json).unwrap();
        assert_eq!(back.sample_count, 42);
        assert!((back.avg_reward - 0.75).abs() < 1e-9);
        assert!(back.is_warmed_up);
        assert_eq!(back.weights.len(), 5);
    }

    #[test]
    fn neural_router_state_cold_start_defaults() {
        let state = NeuralRouterState {
            sample_count: 0,
            avg_reward: 0.0,
            is_warmed_up: false,
            weights: vec![vec![0.0; 7]; 5],
        };
        assert!(!state.is_warmed_up);
        assert_eq!(state.sample_count, 0);
        assert!((state.avg_reward).abs() < 1e-9);
    }

    #[test]
    fn neural_router_state_warmed_up_positive_reward() {
        let state = NeuralRouterState {
            sample_count: 100,
            avg_reward: 0.42,
            is_warmed_up: true,
            weights: vec![vec![0.05; 7]; 5],
        };
        // is_warmed_up && avg_reward > 0 is the condition for suppressing patches
        assert!(state.is_warmed_up && state.avg_reward > 0.0);
    }

    #[test]
    fn neural_router_state_warmed_but_negative_reward_still_allows_patch() {
        let state = NeuralRouterState {
            sample_count: 50,
            avg_reward: -0.3,
            is_warmed_up: true,
            weights: vec![vec![-0.1; 7]; 5],
        };
        // Neural is warmed up but reward is negative: neural_healthy should be false
        let neural_healthy = state.is_warmed_up && state.avg_reward > 0.0;
        assert!(!neural_healthy, "negative avg_reward means routing is not healthy");
    }

    // -----------------------------------------------------------------------
    // Config patch suppression logic unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn patch_suppressed_when_neural_healthy_and_pressure_near_threshold() {
        // Simulate the gating logic from run():
        // neural_healthy=true, pressure only barely above threshold (within 10%)
        let neural_healthy = true;
        let pressure = 0.82_f64; // > 0.8 threshold but < 1.1 * 0.8 = 0.88
        let high_threshold = 0.8_f64;
        let low_threshold = 0.3_f64;

        let pressure_far_from_threshold =
            pressure > high_threshold * 1.1 || pressure < low_threshold * 0.9;
        let should_push = !neural_healthy || pressure_far_from_threshold;

        assert!(!should_push, "patch should be suppressed when neural healthy and pressure barely over threshold");
    }

    #[test]
    fn patch_pushed_when_neural_healthy_but_pressure_far_above_threshold() {
        let neural_healthy = true;
        let pressure = 0.95_f64; // >> 0.8 threshold, > 1.1 * 0.8 = 0.88
        let high_threshold = 0.8_f64;
        let low_threshold = 0.3_f64;

        let pressure_far_from_threshold =
            pressure > high_threshold * 1.1 || pressure < low_threshold * 0.9;
        let should_push = !neural_healthy || pressure_far_from_threshold;

        assert!(should_push, "patch should be pushed when pressure is far above threshold even if neural is healthy");
    }

    #[test]
    fn patch_pushed_when_neural_not_healthy() {
        let neural_healthy = false;
        let pressure = 0.82_f64;
        let high_threshold = 0.8_f64;
        let low_threshold = 0.3_f64;

        let pressure_far_from_threshold =
            pressure > high_threshold * 1.1 || pressure < low_threshold * 0.9;
        let should_push = !neural_healthy || pressure_far_from_threshold;

        assert!(should_push, "patch should always be pushed when neural routing is not healthy");
    }

    #[test]
    fn patch_pushed_when_pressure_far_below_threshold() {
        let neural_healthy = true;
        let pressure = 0.1_f64; // << 0.3 low_threshold, < 0.9 * 0.3 = 0.27
        let high_threshold = 0.8_f64;
        let low_threshold = 0.3_f64;

        let pressure_far_from_threshold =
            pressure > high_threshold * 1.1 || pressure < low_threshold * 0.9;
        let should_push = !neural_healthy || pressure_far_from_threshold;

        assert!(should_push, "patch should be pushed when pressure is far below low threshold");
    }

    // -----------------------------------------------------------------------
    // Mock-HTTP integration tests for push_config and fetch_neural_state
    // -----------------------------------------------------------------------
    //
    // These spin up a minimal tokio TCP listener that speaks just enough HTTP/1.1
    // to satisfy reqwest, exercising the actual network paths of push_config()
    // and fetch_neural_state().

    async fn bind_mock() -> (u16, tokio::net::TcpListener) {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let port = l.local_addr().expect("addr").port();
        (port, l)
    }

    async fn serve_once(listener: tokio::net::TcpListener, status: u16, body: &'static str) {
        use tokio::io::AsyncWriteExt;
        let (mut s, _) = listener.accept().await.expect("accept");
        let mut buf = [0u8; 4096];
        let _ = tokio::time::timeout(
            Duration::from_millis(200),
            tokio::io::AsyncReadExt::read(&mut s, &mut buf),
        ).await;
        let resp = format!(
            "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let _ = s.write_all(resp.as_bytes()).await;
        let _ = s.flush().await;
    }

    fn make_bridge_at(port: u16) -> HelixBridge {
        HelixBridge::builder(format!("http://127.0.0.1:{port}"))
            .bus(make_bus())
            .connect_timeout(Duration::from_secs(2))
            .request_timeout(Duration::from_secs(5))
            .build()
            .expect("build bridge")
    }

    // -- push_config mock-HTTP tests --

    #[tokio::test]
    async fn push_config_succeeds_on_http_200() {
        let (port, listener) = bind_mock().await;
        let body = r#"{"inline_threshold":8000,"spawn_threshold":54000,"cpu_queue_cap":512,"cpu_parallelism":8,"backpressure_busy_threshold":5,"batch_max_size":8,"batch_max_delay_ms":10,"ema_alpha":0.30,"adaptive_step":0.20,"cpu_p95_budget_ms":200,"adaptive_p95_threshold_factor":1.2}"#;
        tokio::spawn(serve_once(listener, 200, body));

        let bridge = make_bridge_at(port);
        let patch = RouterConfigPatch { adaptive_step: Some(0.20), ema_alpha: Some(0.30), ..Default::default() };
        let result = bridge.push_config(&patch).await;
        assert!(result.is_ok(), "push_config should succeed on HTTP 200: {result:?}");
    }

    #[tokio::test]
    async fn push_config_fails_on_http_400() {
        let (port, listener) = bind_mock().await;
        tokio::spawn(serve_once(listener, 400, r#"{"error":"bad request"}"#));

        let bridge = make_bridge_at(port);
        let patch = RouterConfigPatch::default();
        let result = bridge.push_config(&patch).await;
        assert!(result.is_err(), "push_config should fail on HTTP 400");
        let err = result.unwrap_err();
        assert!(matches!(err, HelixBridgeError::Http { status: 400, .. }), "expected Http 400 error: {err}");
    }

    #[tokio::test]
    async fn push_config_fails_on_http_500() {
        let (port, listener) = bind_mock().await;
        tokio::spawn(serve_once(listener, 500, r#"{"error":"internal"}"#));

        let bridge = make_bridge_at(port);
        let patch = RouterConfigPatch { spawn_threshold: Some(54_000), ..Default::default() };
        let result = bridge.push_config(&patch).await;
        assert!(result.is_err(), "push_config should fail on HTTP 500");
        let err = result.unwrap_err();
        assert!(matches!(err, HelixBridgeError::Http { status: 500, .. }), "expected Http 500 error: {err}");
    }

    #[tokio::test]
    async fn push_config_connection_refused_returns_connect_error() {
        // Port 1 is privileged and will connection-refuse immediately.
        let bridge = HelixBridge::builder("http://127.0.0.1:1")
            .bus(make_bus())
            .connect_timeout(Duration::from_millis(200))
            .request_timeout(Duration::from_millis(400))
            .build()
            .expect("build bridge");

        let patch = RouterConfigPatch::default();
        let result = bridge.push_config(&patch).await;
        assert!(result.is_err(), "should fail on connection refused");
        assert!(matches!(result.unwrap_err(), HelixBridgeError::Connect { .. }), "expected Connect error");
    }

    #[tokio::test]
    async fn push_config_high_pressure_patch_has_expected_fields() {
        // Verify the patch produced by suggest_config_patch for high pressure
        // contains all the fields we actually want HelixRouter to apply.
        let stats = RouterStats {
            completed: 100,
            dropped: 10,
            adaptive_spawn_threshold: 60_000,
            pressure_score: 0.95,
            routed_by_strategy: vec![],
            latency_by_strategy: vec![],
        };
        let patch = suggest_config_patch(&stats, 0.3, 0.8).expect("should produce patch for high pressure");
        assert!(patch.adaptive_step.is_some(), "high-pressure patch must set adaptive_step");
        assert!(patch.ema_alpha.is_some(), "high-pressure patch must set ema_alpha");
        assert!(patch.backpressure_busy_threshold.is_some(), "high-pressure patch must set backpressure_busy_threshold");
        // High-pressure adaptive_step should be larger than low-pressure
        assert!(patch.adaptive_step.unwrap() > 0.05, "high-pressure adaptive_step should be elevated");
    }

    #[tokio::test]
    async fn push_config_low_pressure_patch_relaxes_params() {
        let stats = RouterStats {
            completed: 100,
            dropped: 0,
            adaptive_spawn_threshold: 60_000,
            pressure_score: 0.1,
            routed_by_strategy: vec![],
            latency_by_strategy: vec![],
        };
        let patch = suggest_config_patch(&stats, 0.3, 0.8).expect("should produce patch for low pressure");
        assert!(patch.adaptive_step.is_some());
        assert!(patch.ema_alpha.is_some());
        // Low-pressure adaptive_step should be smaller than default (0.10)
        assert!(patch.adaptive_step.unwrap() < 0.10, "low-pressure adaptive_step should be reduced");
    }

    // -- fetch_neural_state mock-HTTP tests --

    #[tokio::test]
    async fn fetch_neural_state_returns_some_on_200() {
        let (port, listener) = bind_mock().await;
        // JSON matches NeuralRouterState: sample_count, avg_reward, is_warmed_up, weights.
        let body = r#"{"sample_count":150,"avg_reward":0.82,"is_warmed_up":true,"weights":[[0.1,0.2,0.3,0.4,0.5,0.6,0.7],[0.1,0.2,0.3,0.4,0.5,0.6,0.7],[0.1,0.2,0.3,0.4,0.5,0.6,0.7],[0.1,0.2,0.3,0.4,0.5,0.6,0.7],[0.1,0.2,0.3,0.4,0.5,0.6,0.7]]}"#;
        tokio::spawn(serve_once(listener, 200, body));

        let bridge = make_bridge_at(port);
        let result = bridge.fetch_neural_state().await;
        assert!(result.is_ok(), "fetch_neural_state should succeed: {result:?}");
        let state = result.unwrap().expect("should return Some on 200");
        assert!(state.is_warmed_up);
        assert!((state.avg_reward - 0.82).abs() < 1e-6);
    }

    #[tokio::test]
    async fn fetch_neural_state_returns_none_on_404() {
        let (port, listener) = bind_mock().await;
        tokio::spawn(serve_once(listener, 404, r#"{"error":"not found"}"#));

        let bridge = make_bridge_at(port);
        let result = bridge.fetch_neural_state().await;
        assert!(result.is_ok(), "404 should not be an error, just None");
        assert!(result.unwrap().is_none(), "404 should return None");
    }

    #[tokio::test]
    async fn fetch_neural_state_returns_none_on_501() {
        // 501 = older HelixRouter without /api/neural
        let (port, listener) = bind_mock().await;
        tokio::spawn(serve_once(listener, 501, ""));

        let bridge = make_bridge_at(port);
        let result = bridge.fetch_neural_state().await;
        assert!(result.is_ok(), "non-2xx on neural should return Ok(None)");
        assert!(result.unwrap().is_none());
    }

    #[tokio::test]
    async fn fetch_neural_state_error_on_bad_json() {
        let (port, listener) = bind_mock().await;
        tokio::spawn(serve_once(listener, 200, r#"not valid json"#));

        let bridge = make_bridge_at(port);
        let result = bridge.fetch_neural_state().await;
        assert!(result.is_err(), "malformed JSON should return Err");
        assert!(matches!(result.unwrap_err(), HelixBridgeError::Json { .. }));
    }

    #[tokio::test]
    async fn fetch_stats_then_push_config_round_trip() {
        // Serve /api/stats on first connection, /api/config on second.
        // Verifies the stats → suggest → push pipeline end-to-end.
        use tokio::io::AsyncWriteExt;

        let (port, listener) = bind_mock().await;
        let stats_body = r#"{"completed":50,"dropped":5,"adaptive_spawn_threshold":60000,"pressure_score":0.9,"routed_by_strategy":[],"latency_by_strategy":[]}"#;
        let config_body = r#"{"inline_threshold":8000,"spawn_threshold":54000,"cpu_queue_cap":512,"cpu_parallelism":8,"backpressure_busy_threshold":5,"batch_max_size":8,"batch_max_delay_ms":10,"ema_alpha":0.30,"adaptive_step":0.20,"cpu_p95_budget_ms":200,"adaptive_p95_threshold_factor":1.2}"#;

        // Spawn handler that serves two sequential connections.
        tokio::spawn(async move {
            for body in [stats_body, config_body] {
                if let Ok((mut s, _)) = listener.accept().await {
                    let mut buf = [0u8; 4096];
                    let _ = tokio::time::timeout(
                        Duration::from_millis(200),
                        tokio::io::AsyncReadExt::read(&mut s, &mut buf),
                    ).await;
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = s.write_all(resp.as_bytes()).await;
                    let _ = s.flush().await;
                }
            }
        });

        let bridge = make_bridge_at(port);

        // Step 1: fetch stats
        let stats = bridge.fetch_stats().await.expect("fetch_stats");
        assert!((stats.pressure_score - 0.9).abs() < 1e-6);

        // Step 2: derive patch
        let patch = suggest_config_patch(&stats, 0.3, 0.8)
            .expect("high pressure should produce a patch");
        assert!(patch.adaptive_step.is_some());

        // Step 3: push patch
        let push_result = bridge.push_config(&patch).await;
        assert!(push_result.is_ok(), "push_config should succeed: {push_result:?}");
    }
}
