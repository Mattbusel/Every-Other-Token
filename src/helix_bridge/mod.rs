//! # HelixBridge
//!
//! Feature-gated integration between Every-Other-Token's self-improvement loop
//! and a running HelixRouter instance.
//!
//! ## What It Does
//!
//! 1. **Stats ingestion** — polls HelixRouter's `/api/stats` endpoint and converts
//!    `RouterStats` snapshots into `TelemetrySnapshot`s fed to the local `TelemetryBus`.
//! 2. **Config push** — accepts `RouterConfigPatch` structs and POSTs them to
//!    HelixRouter's `/api/config` endpoint to apply orchestrator-generated param changes.
//!
//! ## Usage
//!
//! ```rust,ignore
//! let bridge = HelixBridge::builder("http://127.0.0.1:3000")
//!     .bus(Arc::clone(&bus))
//!     .poll_interval(Duration::from_secs(5))
//!     .build();
//! tokio::spawn(bridge.run());
//! ```

pub mod client;
pub mod converter;

pub use client::{HelixBridge, HelixBridgeBuilder, HelixBridgeError, RouterConfigPatch, RouterStats};
pub use converter::stats_to_snapshot;
