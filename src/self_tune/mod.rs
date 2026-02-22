//! # Module: self_tune
//!
//! Self-tuning core — monitors the pipeline and adjusts its own parameters.
//!
//! All sub-modules are gated behind the `self-tune` feature flag.
//! The pipeline continues with static configuration when the feature is disabled.
//!
//! ## Sub-modules
//! - [`telemetry_bus`] — nervous system: aggregates metrics from all stages
//! - `controller` (task 1.2) — PID controllers for 12 tunable parameters
//! - `experiment` (task 1.3) — A/B testing framework
//! - `anomaly` (task 1.4) — statistical anomaly detection
//! - `cost` (task 1.5) — real-time cost tracking and budget-aware routing
//! - `snapshot` (task 1.6) — git-like configuration versioning

pub mod telemetry_bus;
pub mod controller;
pub mod experiment;
pub mod anomaly;
pub mod cost;
pub mod snapshot;
pub mod orchestrator;

#[cfg(feature = "redis-backing")]
pub mod redis_snapshot;
