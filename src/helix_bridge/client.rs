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

/// Mirrors HelixRouter's RouterStats JSON shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterStats {
    pub completed: u64,
    pub dropped: u64,
    pub adaptive_spawn_threshold: u64,
    pub pressure_score: f64,
    /// strategy → count map
    pub routed: std::collections::HashMap<RoutingStrategy, u64>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_max_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ema_alpha: Option<f64>,
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
}

impl HelixBridgeConfig {
    /// Create a config with sensible defaults.
    ///
    /// - poll_interval: 5 s
    /// - connect_timeout: 3 s
    /// - request_timeout: 10 s
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            poll_interval: Duration::from_secs(5),
            connect_timeout: Duration::from_secs(3),
            request_timeout: Duration::from_secs(10),
        }
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
            .post(&url)
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

    /// Run the polling loop indefinitely.
    ///
    /// On each tick, fetches `/api/stats` and records the converted snapshot
    /// into the `TelemetryBus`. Connection failures are soft-errors — the loop
    /// simply skips that tick and tries again at the next interval.
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
                    self.bus.record_latency(
                        crate::self_tune::telemetry_bus::PipelineStage::Inference,
                        (snap.avg_latency_us as u64).max(1),
                    );

                    // Feed drop pressure as an auxiliary signal on the Other stage.
                    if snap.drop_rate > 0.0 {
                        self.bus.record_latency(
                            crate::self_tune::telemetry_bus::PipelineStage::Other,
                            (snap.drop_rate * 1_000.0) as u64 + 1,
                        );
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
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;

    fn make_bus() -> Arc<TelemetryBus> {
        Arc::new(TelemetryBus::new(BusConfig {
            emit_interval: Duration::from_secs(60),
            queue_capacity: 64,
        }))
    }

    fn make_stats() -> RouterStats {
        let mut routed = HashMap::new();
        routed.insert(RoutingStrategy::Inline, 10);
        routed.insert(RoutingStrategy::Spawn, 5);
        RouterStats {
            completed: 15,
            dropped: 0,
            adaptive_spawn_threshold: 100,
            pressure_score: 0.2,
            routed,
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
            routed: HashMap::new(),
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
}
