//! # Self-Improvement Orchestrator
//!
//! The main event loop that closes the self-improvement feedback cycle:
//!
//! ```text
//! TelemetryBus ──► AnomalyDetector ──► TaskGenerator ──► AgentMemory
//!      ▲                                                      │
//!      │               Controller ◄──────────────────────────┘
//!      │                   │
//! TokenInterceptor ◄───────┘  (parameter feedback)
//! ```
//!
//! ## What It Does
//!
//! 1. Subscribes to the `TelemetryBus` for live pipeline snapshots.
//! 2. Runs every snapshot through the `AnomalyDetector`.
//! 3. Converts anomalies → `DegradationSignal`s → `GeneratedTask`s via `TaskGenerator`.
//! 4. Feeds each snapshot into the `Controller` so PID adjustments fire.
//! 5. Records generated tasks and controller decisions in `AgentMemory`.
//! 6. Exposes current state (anomalies, tasks, parameter values) via `OrchestratorStatus`.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use every_other_token::self_tune::orchestrator::{SelfImprovementOrchestrator, OrchestratorConfig};
//! use every_other_token::self_tune::telemetry_bus::{TelemetryBus, BusConfig};
//! use std::sync::Arc;
//!
//! let bus = Arc::new(TelemetryBus::new(BusConfig::default()));
//! bus.start_emitter();
//!
//! let orc = SelfImprovementOrchestrator::new(OrchestratorConfig::default(), Arc::clone(&bus));
//! tokio::spawn(async move { orc.run().await });
//! ```

use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::time::interval;

use crate::self_tune::{
    anomaly::{AnomalyDetector, DetectorConfig, Severity},
    controller::{Controller, ControllerConfig},
    telemetry_bus::{TelemetryBus, TelemetrySnapshot},
};
#[cfg(feature = "self-modify")]
use crate::self_modify::{
    memory::{AgentMemory, MemoryConfig, ModificationRecord, Outcome},
    task_gen::{AnomalySeverity, DegradationSignal, TaskGenConfig, TaskGenerator},
};

// ---------------------------------------------------------------------------
// OrchestratorConfig
// ---------------------------------------------------------------------------

/// Configuration for the self-improvement orchestration loop.
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// How often to poll the latest telemetry snapshot when no broadcast arrives.
    pub poll_interval: Duration,
    /// How many generated tasks to keep in the recent-tasks ring.
    pub recent_tasks_cap: usize,
    /// Minimum anomaly severity that triggers task generation.
    pub task_gen_severity_threshold: Severity,
    /// Whether the controller should apply parameter adjustments automatically.
    pub auto_adjust_params: bool,
    /// Configuration forwarded to the anomaly detector.
    pub detector: DetectorConfig,
    /// Configuration forwarded to the PID controller.
    pub controller: ControllerConfig,
    /// Configuration forwarded to the task generator.
    #[cfg(feature = "self-modify")]
    pub task_gen: TaskGenConfig,
    /// Configuration forwarded to agent memory.
    #[cfg(feature = "self-modify")]
    pub memory: MemoryConfig,
    /// Gate config for staged deployment pipeline. None = no pipeline.
    #[cfg(feature = "self-modify")]
    pub gate_config: Option<crate::self_modify::gate::GateConfig>,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_secs(5),
            recent_tasks_cap: 100,
            task_gen_severity_threshold: Severity::Warn,
            auto_adjust_params: true,
            detector: DetectorConfig::default(),
            controller: ControllerConfig::default(),
            #[cfg(feature = "self-modify")]
            task_gen: TaskGenConfig::default(),
            #[cfg(feature = "self-modify")]
            memory: MemoryConfig::default(),
            #[cfg(feature = "self-modify")]
            gate_config: None,
        }
    }
}

// ---------------------------------------------------------------------------
// OrchestratorStatus — observable state snapshot
// ---------------------------------------------------------------------------

/// Metrics about the orchestrator's current activity, readable from outside.
#[derive(Debug, Clone, Default)]
pub struct OrchestratorStatus {
    /// Total telemetry snapshots processed since start.
    pub snapshots_processed: u64,
    /// Total anomalies detected since start.
    pub anomalies_detected: u64,
    /// Total tasks generated since start.
    pub tasks_generated: u64,
    /// Total parameter adjustments made since start.
    pub param_adjustments: u64,
    /// Whether the orchestrator loop is currently running.
    pub running: bool,
    /// Most recently generated task names (capped at `recent_tasks_cap`).
    pub recent_task_names: Vec<String>,
}

// ---------------------------------------------------------------------------
// SelfImprovementOrchestrator
// ---------------------------------------------------------------------------

/// Ties together the self-tune pipeline into a single runnable loop.
pub struct SelfImprovementOrchestrator {
    config: OrchestratorConfig,
    bus: Arc<TelemetryBus>,
    /// Shared status, readable from outside the loop.
    status: Arc<Mutex<OrchestratorStatus>>,
    /// Anomaly detector (not shared — owned by the loop task).
    detector: AnomalyDetector,
    /// PID controller (not shared — owned by the loop task).
    controller: Controller,
    /// Task generator (not shared — owned by the loop task).
    #[cfg(feature = "self-modify")]
    task_gen: TaskGenerator,
    /// Agent memory (shared so callers can inspect history).
    #[cfg(feature = "self-modify")]
    memory: Arc<Mutex<AgentMemory>>,
    /// Staged deployment pipeline.
    #[cfg(feature = "self-modify")]
    deployment_pipeline: Option<crate::self_modify::deployment::StagedDeploymentPipeline>,
}

impl SelfImprovementOrchestrator {
    /// Create a new orchestrator. Call [`run`] to start the loop.
    pub fn new(config: OrchestratorConfig, bus: Arc<TelemetryBus>) -> Self {
        let detector = AnomalyDetector::new(config.detector.clone());
        let controller = Controller::new(config.controller.clone());
        #[cfg(feature = "self-modify")]
        let task_gen = TaskGenerator::new(config.task_gen.clone());
        #[cfg(feature = "self-modify")]
        let memory = Arc::new(Mutex::new(AgentMemory::new(config.memory.clone())));
        #[cfg(feature = "self-modify")]
        let deployment_pipeline = config.gate_config.as_ref().map(|gc| {
            crate::self_modify::deployment::StagedDeploymentPipeline::new(
                crate::self_modify::gate::ValidationGate::new(gc.clone())
            )
        });

        Self {
            config,
            bus,
            status: Arc::new(Mutex::new(OrchestratorStatus::default())),
            detector,
            controller,
            #[cfg(feature = "self-modify")]
            task_gen,
            #[cfg(feature = "self-modify")]
            memory,
            #[cfg(feature = "self-modify")]
            deployment_pipeline,
        }
    }

    /// Return a cloneable handle to the shared status (for dashboard / MCP tools).
    pub fn status_handle(&self) -> Arc<Mutex<OrchestratorStatus>> {
        Arc::clone(&self.status)
    }

    /// Return a cloneable handle to agent memory (for inspection / MCP tools).
    #[cfg(feature = "self-modify")]
    pub fn memory_handle(&self) -> Arc<Mutex<AgentMemory>> {
        Arc::clone(&self.memory)
    }

    /// Register a deployment target.
    #[cfg(feature = "self-modify")]
    pub fn add_deployment_target(&mut self, target: Box<dyn crate::self_modify::deployment::DeploymentTarget>) {
        if let Some(ref mut pipeline) = self.deployment_pipeline {
            pipeline.add_target(target);
        }
    }

    /// Run the orchestration loop.  This is `async` and runs until the task is
    /// cancelled or the bus drops.
    pub async fn run(mut self) {
        // Mark running
        if let Ok(mut s) = self.status.lock() {
            s.running = true;
        }

        let mut rx = self.bus.subscribe();
        let mut poll = interval(self.config.poll_interval);
        poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            // Wait for either a broadcast snapshot or a poll tick.
            let snap: TelemetrySnapshot = tokio::select! {
                Ok(s) = rx.recv() => s,
                _ = poll.tick() => {
                    self.bus.latest().await
                }
            };

            self.process_snapshot(snap);
        }
    }

    /// Process one telemetry snapshot through the full pipeline.
    ///
    /// This is `pub` so tests can drive it synchronously without spawning tasks.
    pub fn process_snapshot(&mut self, snap: TelemetrySnapshot) {
        // 1. Run through anomaly detector
        let anomalies = self.detector.observe(&snap);

        // 2. Run through PID controller (auto-adjust parameters)
        if self.config.auto_adjust_params {
            self.controller.observe(&snap);
        }

        // 3. Convert qualifying anomalies → signals → tasks
        #[cfg(feature = "self-modify")]
        let mut new_tasks: Vec<String> = Vec::new();
        #[cfg(not(feature = "self-modify"))]
        let new_tasks: Vec<String> = Vec::new();
        #[cfg(feature = "self-modify")]
        for anomaly in &anomalies {
            // Skip below threshold
            let passes_threshold = match self.config.task_gen_severity_threshold {
                Severity::Info => true,
                Severity::Warn => {
                    anomaly.severity == Severity::Warn
                        || anomaly.severity == Severity::Critical
                }
                Severity::Critical => anomaly.severity == Severity::Critical,
            };
            if !passes_threshold {
                continue;
            }

            let sev = match anomaly.severity {
                Severity::Info => AnomalySeverity::Info,
                Severity::Warn => AnomalySeverity::Warn,
                Severity::Critical => AnomalySeverity::Critical,
            };

            // Extract metric name from the anomaly message (format: "metric_name: ...")
            let metric_name = anomaly
                .message
                .split(':')
                .next()
                .unwrap_or("unknown")
                .trim()
                .to_string();

            let signal = DegradationSignal::Anomaly {
                metric: metric_name.clone(),
                severity: sev,
                observed: anomaly.metric_value,
                baseline: anomaly.score, // score acts as a relative baseline reference
            };

            let now_ms = snap.captured_at.elapsed().as_millis() as u64;
            if let Some(task) = self.task_gen.generate_at(signal, std::time::Instant::now(), now_ms) {
                new_tasks.push(task.name.clone());

                // Record in agent memory as a pending modification
                if let Ok(mut mem) = self.memory.lock() {
                    mem.record_modification(ModificationRecord {
                        id: task.id.clone(),
                        description: task.description.clone(),
                        affected_files: task.affected_files.clone(),
                        outcome: Outcome::Pending,
                        metric_deltas: std::collections::HashMap::new(),
                        notes: format!(
                            "Generated from anomaly: metric={} severity={:?}",
                            metric_name, anomaly.severity
                        ),
                        timestamp_ms: now_ms,
                    });
                }
            }
        }

        // If we have a deployment pipeline and params were adjusted, validate and apply.
        // Uses CargoCheckRunner (real cargo test/clippy) instead of PassAllCheckRunner
        // to prevent unchecked auto-deploys from bypassing validation gates.
        #[cfg(feature = "self-modify")]
        if let Some(ref pipeline) = self.deployment_pipeline {
            if !new_tasks.is_empty() {
                use crate::self_modify::deployment::{CargoCheckRunner, ParamChange};
                let runner = CargoCheckRunner::new(".");
                let changes: Vec<ParamChange> = new_tasks.iter().map(|name| ParamChange {
                    param_name: name.clone(),
                    old_value: 0.0,
                    new_value: 1.0,
                }).collect();
                let outcome = pipeline.deploy(
                    format!("auto-{}", new_tasks.len()),
                    &runner,
                    &changes,
                );
                // Log deployment outcome for observability.
                match &outcome {
                    crate::self_modify::deployment::DeploymentOutcome::Deployed { changes_applied, .. } => {
                        tracing::info!(
                            target: "self_tune::orchestrator",
                            changes = changes_applied,
                            "Auto-deployment succeeded"
                        );
                    }
                    crate::self_modify::deployment::DeploymentOutcome::Rejected { failed_checks } => {
                        tracing::warn!(
                            target: "self_tune::orchestrator",
                            checks = ?failed_checks,
                            "Auto-deployment rejected by validation gate"
                        );
                    }
                    crate::self_modify::deployment::DeploymentOutcome::AwaitingReview { change_id } => {
                        tracing::info!(
                            target: "self_tune::orchestrator",
                            change_id = %change_id,
                            "Auto-deployment awaiting human review"
                        );
                    }
                    other => {
                        tracing::debug!(
                            target: "self_tune::orchestrator",
                            outcome = ?other,
                            "Auto-deployment outcome"
                        );
                    }
                }
            }
        }

        // 4. Update shared status
        let adj_count = self.controller.audit_log().len() as u64;
        if let Ok(mut s) = self.status.lock() {
            s.snapshots_processed += 1;
            s.anomalies_detected += anomalies.len() as u64;
            s.tasks_generated += new_tasks.len() as u64;
            s.param_adjustments = adj_count;
            for name in new_tasks {
                if s.recent_task_names.len() >= self.config.recent_tasks_cap {
                    s.recent_task_names.remove(0);
                }
                s.recent_task_names.push(name);
            }
        }
    }

    /// Return the current parameter value for a given parameter name.
    pub fn get_param(&self, param: crate::self_tune::controller::Param) -> f64 {
        self.controller.get(param)
    }

    /// Return a snapshot of the current status without locking for long.
    pub fn status_snapshot(&self) -> OrchestratorStatus {
        self.status.lock().map(|s| s.clone()).unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::self_tune::telemetry_bus::{BusConfig, TelemetrySnapshot};

    fn make_bus() -> Arc<TelemetryBus> {
        Arc::new(TelemetryBus::new(BusConfig {
            emit_interval: Duration::from_secs(60), // won't auto-emit in tests
            queue_capacity: 100,
        }))
    }

    fn make_orc() -> SelfImprovementOrchestrator {
        SelfImprovementOrchestrator::new(OrchestratorConfig::default(), make_bus())
    }

    fn snap_with_latency(p95_us: u64) -> TelemetrySnapshot {
        let mut s = TelemetrySnapshot::zero();
        s.p95_1m_us = p95_us as f64;
        s.avg_latency_us = (p95_us / 2) as f64;
        s
    }


    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    #[test]
    fn test_new_orchestrator_starts_with_zero_status() {
        let orc = make_orc();
        let status = orc.status_snapshot();
        assert_eq!(status.snapshots_processed, 0);
        assert_eq!(status.anomalies_detected, 0);
        assert_eq!(status.tasks_generated, 0);
        assert!(!status.running);
    }

    #[test]
    fn test_status_handle_is_shared() {
        let orc = make_orc();
        let h1 = orc.status_handle();
        let h2 = orc.status_handle();
        assert!(Arc::ptr_eq(&h1, &h2));
    }

    #[cfg(feature = "self-modify")]
    #[test]
    fn test_memory_handle_is_shared() {
        let orc = make_orc();
        let h1 = orc.memory_handle();
        let h2 = orc.memory_handle();
        assert!(Arc::ptr_eq(&h1, &h2));
    }

    // -------------------------------------------------------------------
    // process_snapshot — basic counting
    // -------------------------------------------------------------------

    #[test]
    fn test_process_snapshot_increments_count() {
        let mut orc = make_orc();
        orc.process_snapshot(TelemetrySnapshot::zero());
        orc.process_snapshot(TelemetrySnapshot::zero());
        let s = orc.status_snapshot();
        assert_eq!(s.snapshots_processed, 2);
    }

    #[test]
    fn test_process_snapshot_counts_snapshots() {
        let mut orc = make_orc();
        for _ in 0..20 {
            orc.process_snapshot(snap_with_latency(5_000));
        }
        let s = orc.status_snapshot();
        assert_eq!(s.snapshots_processed, 20);
    }

    #[test]
    fn test_process_snapshot_spike_triggers_anomaly_detection() {
        let mut orc = SelfImprovementOrchestrator::new(
            OrchestratorConfig {
                task_gen_severity_threshold: Severity::Info,
                ..OrchestratorConfig::default()
            },
            make_bus(),
        );
        // Warm up with alternating values to establish non-zero variance
        for i in 0..30 {
            let v = if i % 2 == 0 { 4_000 } else { 6_000 };
            orc.process_snapshot(snap_with_latency(v));
        }
        // Inject a huge spike
        for _ in 0..5 {
            orc.process_snapshot(snap_with_latency(500_000)); // 500ms — far from mean
        }
        let s = orc.status_snapshot();
        assert!(s.anomalies_detected > 0, "spike should trigger anomaly detection");
    }

    #[test]
    fn test_process_snapshot_auto_adjust_records_adjustments() {
        let mut orc = make_orc();
        // Feed snapshots with high drop rate to trigger PID adjustment
        for _ in 0..5 {
            let mut snap = TelemetrySnapshot::zero();
            snap.drop_rate = 0.5; // 50% drop → PID should react
            orc.process_snapshot(snap);
        }
        let s = orc.status_snapshot();
        assert!(s.param_adjustments > 0); // PID should have fired at least once
    }

    #[test]
    fn test_process_snapshot_auto_adjust_disabled_no_controller_effect() {
        let mut orc = SelfImprovementOrchestrator::new(
            OrchestratorConfig { auto_adjust_params: false, ..OrchestratorConfig::default() },
            make_bus(),
        );
        for _ in 0..5 {
            let mut snap = TelemetrySnapshot::zero();
            snap.drop_rate = 0.9;
            orc.process_snapshot(snap);
        }
        // With auto_adjust disabled, controller audit log stays empty
        assert_eq!(orc.controller.audit_log().len(), 0);
    }

    // -------------------------------------------------------------------
    // Severity threshold gating
    // -------------------------------------------------------------------

    #[test]
    fn test_info_threshold_gates_task_gen_below_warn() {
        // With Info threshold, any anomaly triggers task gen.
        // Verify tasks_generated tracks with anomalies_detected.
        let mut orc = SelfImprovementOrchestrator::new(
            OrchestratorConfig {
                task_gen_severity_threshold: Severity::Info,
                ..OrchestratorConfig::default()
            },
            make_bus(),
        );
        for i in 0..30 {
            orc.process_snapshot(snap_with_latency(if i % 2 == 0 { 4_000 } else { 6_000 }));
        }
        for _ in 0..5 {
            orc.process_snapshot(snap_with_latency(500_000));
        }
        let s = orc.status_snapshot();
        // tasks_generated ≤ anomalies_detected (dedup/rate-limit may suppress some)
        assert!(s.tasks_generated <= s.anomalies_detected);
    }

    // -------------------------------------------------------------------
    // Deployment pipeline integration
    // -------------------------------------------------------------------

    #[cfg(feature = "self-modify")]
    #[test]
    fn test_deployment_pipeline_wired_when_gate_config_set() {
        use crate::self_modify::gate::GateConfig;
        let config = OrchestratorConfig {
            gate_config: Some(GateConfig::default()),
            ..OrchestratorConfig::default()
        };
        let orc = SelfImprovementOrchestrator::new(config, make_bus());
        assert!(orc.deployment_pipeline.is_some());
    }

    #[cfg(feature = "self-modify")]
    #[test]
    fn test_deployment_pipeline_absent_when_gate_config_none() {
        let orc = SelfImprovementOrchestrator::new(
            OrchestratorConfig { gate_config: None, ..OrchestratorConfig::default() },
            make_bus(),
        );
        assert!(orc.deployment_pipeline.is_none());
    }

    #[cfg(feature = "self-modify")]
    #[test]
    fn test_add_deployment_target_noop_when_pipeline_absent() {
        use crate::self_modify::deployment::InMemoryParamTarget;
        let mut orc = SelfImprovementOrchestrator::new(
            OrchestratorConfig { gate_config: None, ..OrchestratorConfig::default() },
            make_bus(),
        );
        orc.add_deployment_target(Box::new(InMemoryParamTarget::new("t")));
    }

    #[cfg(feature = "self-modify")]
    #[test]
    fn test_default_config_gate_config_is_none() {
        assert!(OrchestratorConfig::default().gate_config.is_none());
    }

    // -------------------------------------------------------------------
    // Memory recording
    // -------------------------------------------------------------------

    #[cfg(feature = "self-modify")]
    #[test]
    fn test_generated_tasks_recorded_in_memory() {
        let mut orc = SelfImprovementOrchestrator::new(
            OrchestratorConfig {
                task_gen_severity_threshold: Severity::Info,
                ..OrchestratorConfig::default()
            },
            make_bus(),
        );
        let mem_handle = orc.memory_handle();
        // Warm up
        for i in 0..30 {
            orc.process_snapshot(snap_with_latency(if i % 2 == 0 { 4_000 } else { 6_000 }));
        }
        // Spike
        for _ in 0..5 {
            orc.process_snapshot(snap_with_latency(500_000));
        }
        // If any tasks generated, memory should have pending modifications
        let s = orc.status_snapshot();
        if s.tasks_generated > 0 {
            let mem = mem_handle.lock().unwrap();
            let pending: Vec<_> = mem
                .modifications()
                .filter(|r| r.outcome == crate::self_modify::memory::Outcome::Pending)
                .collect();
            assert!(!pending.is_empty(), "generated tasks should be in memory as Pending");
        }
    }

    // -------------------------------------------------------------------
    // get_param
    // -------------------------------------------------------------------

    #[test]
    fn test_get_param_returns_default_value() {
        let orc = make_orc();
        let v = orc.get_param(crate::self_tune::controller::Param::DedupChannelBuf);
        assert!(v > 0.0, "default param value should be positive");
    }

    // -------------------------------------------------------------------
    // OrchestratorConfig defaults
    // -------------------------------------------------------------------

    #[test]
    fn test_default_config_auto_adjust_enabled() {
        assert!(OrchestratorConfig::default().auto_adjust_params);
    }

    #[test]
    fn test_default_config_threshold_is_warn() {
        assert_eq!(
            OrchestratorConfig::default().task_gen_severity_threshold,
            Severity::Warn
        );
    }

    #[test]
    fn test_default_config_poll_interval_is_5s() {
        assert_eq!(
            OrchestratorConfig::default().poll_interval,
            Duration::from_secs(5)
        );
    }
}
