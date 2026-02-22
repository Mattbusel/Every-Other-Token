//! # Module: self_modify
//!
//! Self-modifying agent loop — observes degradation signals and generates,
//! validates, and deploys corrective changes to the pipeline.
//!
//! All sub-modules are gated behind the `self-modify` feature flag.
//!
//! ## Sub-modules
//! - [`task_gen`] (2.1) — generates TOML tasks from anomaly/telemetry signals
//! - [`gate`]     (2.2) — validation gate: test, clippy, bench, smoke, staging
//! - [`memory`]   (2.3) — agent knowledge base: past outcomes, baselines, dead-ends
//! - [`deploy`]   (2.4) — blue-green canary deployment pipeline
//! - [`docs`]     (2.5) — auto-generated changelogs and architecture diagrams
//! - [`discover`] (2.6) — capability discovery: dead code, coverage gaps, hotspots

pub mod task_gen;
pub mod gate;
pub mod memory;
pub mod deployment;
