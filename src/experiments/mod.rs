//! # Experiment Manager
//!
//! Tracks multiple research experiments, each with a name, hypothesis,
//! parameters, and accumulated results.  Experiments are auto-saved to JSON
//! files in an `experiments/` directory.
//!
//! ## CLI integration
//!
//! ```text
//! every-other-token "What is consciousness?" \
//!     --experiment my_exp \
//!     --hypothesis "dropping every other token reduces quality by 10%"
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use every_other_token::experiments::{ExperimentManager, ExperimentConfig};
//!
//! let config = ExperimentConfig {
//!     name: "token_drop_study".into(),
//!     hypothesis: "Dropping every other token reduces quality by 10%".into(),
//!     ..Default::default()
//! };
//!
//! let mut manager = ExperimentManager::in_memory(config);
//! manager.record_result("run_1", serde_json::json!({"quality": 0.87}));
//! let summary = manager.summary();
//! assert_eq!(summary.result_count, 1);
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for a single experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    /// Unique name for this experiment (used as the JSON filename stem).
    pub name: String,
    /// The hypothesis being tested.
    pub hypothesis: String,
    /// Free-form parameter map (model, rate, transform, etc.).
    #[serde(default)]
    pub parameters: HashMap<String, Value>,
    /// Optional human-readable description.
    #[serde(default)]
    pub description: String,
    /// Optional list of tags for filtering/grouping.
    #[serde(default)]
    pub tags: Vec<String>,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            name: "unnamed_experiment".into(),
            hypothesis: String::new(),
            parameters: HashMap::new(),
            description: String::new(),
            tags: Vec::new(),
        }
    }
}

// ── Result record ─────────────────────────────────────────────────────────────

/// A single recorded result within an experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultRecord {
    /// Caller-assigned key (e.g. `"run_1"`, `"baseline"`).
    pub key: String,
    /// Arbitrary JSON payload (metrics, outputs, timings, …).
    pub data: Value,
    /// RFC 3339 timestamp when this result was recorded.
    pub timestamp: String,
}

// ── Experiment ────────────────────────────────────────────────────────────────

/// A single research experiment with accumulated results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    /// Configuration declared at creation time.
    pub config: ExperimentConfig,
    /// Results in the order they were recorded.
    pub results: Vec<ResultRecord>,
    /// RFC 3339 timestamp when this experiment was created.
    pub created_at: String,
    /// RFC 3339 timestamp of the most recent update.
    pub updated_at: String,
}

impl Experiment {
    /// Create a new, empty experiment.
    pub fn new(config: ExperimentConfig) -> Self {
        let now = now_rfc3339();
        Self {
            config,
            results: Vec::new(),
            created_at: now.clone(),
            updated_at: now,
        }
    }

    /// Record a result.
    pub fn add_result(&mut self, key: impl Into<String>, data: Value) {
        self.results.push(ResultRecord {
            key: key.into(),
            data,
            timestamp: now_rfc3339(),
        });
        self.updated_at = now_rfc3339();
    }

    /// Number of results recorded.
    pub fn result_count(&self) -> usize {
        self.results.len()
    }

    /// Look up a result by key (returns the first match).
    pub fn get_result(&self, key: &str) -> Option<&ResultRecord> {
        self.results.iter().find(|r| r.key == key)
    }

    /// Compute a simple numeric summary for a specific data field across all runs.
    ///
    /// `field` is a JSON pointer (e.g. `"quality"` or `"/nested/value"`).
    pub fn field_stats(&self, field: &str) -> Option<FieldStats> {
        let values: Vec<f64> = self
            .results
            .iter()
            .filter_map(|r| r.data.get(field)?.as_f64())
            .collect();

        if values.is_empty() {
            return None;
        }

        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        Some(FieldStats {
            field: field.to_string(),
            count: values.len(),
            mean,
            min,
            max,
            std_dev,
        })
    }
}

/// Numeric summary of a single data field across all results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldStats {
    /// Field name.
    pub field: String,
    /// Number of observations.
    pub count: usize,
    /// Arithmetic mean.
    pub mean: f64,
    /// Minimum observed value.
    pub min: f64,
    /// Maximum observed value.
    pub max: f64,
    /// Population standard deviation.
    pub std_dev: f64,
}

// ── Summary ───────────────────────────────────────────────────────────────────

/// A lightweight summary of an experiment (for listing/reporting).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentSummary {
    /// Experiment name.
    pub name: String,
    /// The hypothesis.
    pub hypothesis: String,
    /// Total results recorded.
    pub result_count: usize,
    /// Creation timestamp.
    pub created_at: String,
    /// Last-update timestamp.
    pub updated_at: String,
}

// ── Manager ───────────────────────────────────────────────────────────────────

/// Manages the lifecycle of a single experiment, with optional disk persistence.
pub struct ExperimentManager {
    experiment: Experiment,
    /// Directory for auto-saving JSON files.  `None` = in-memory only.
    save_dir: Option<PathBuf>,
}

impl ExperimentManager {
    /// Create an in-memory manager (no disk I/O).
    pub fn in_memory(config: ExperimentConfig) -> Self {
        Self {
            experiment: Experiment::new(config),
            save_dir: None,
        }
    }

    /// Create a manager that auto-saves to `<save_dir>/<name>.json` on every
    /// [`record_result`] call.
    ///
    /// # Errors
    /// Returns an `std::io::Error` if the directory cannot be created.
    pub fn with_save_dir(
        config: ExperimentConfig,
        save_dir: impl AsRef<Path>,
    ) -> std::io::Result<Self> {
        let dir = save_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            experiment: Experiment::new(config),
            save_dir: Some(dir),
        })
    }

    /// Load an experiment from a JSON file, with the given save directory for
    /// future writes.
    ///
    /// # Errors
    /// Returns `std::io::Error` on read/parse failure.
    pub fn load(
        path: impl AsRef<Path>,
        save_dir: Option<PathBuf>,
    ) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let experiment: Experiment = serde_json::from_slice(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(Self {
            experiment,
            save_dir,
        })
    }

    /// Record a result and auto-save if a save directory is configured.
    ///
    /// # Errors
    /// Returns `std::io::Error` only if auto-save fails.
    pub fn record_result(
        &mut self,
        key: impl Into<String>,
        data: Value,
    ) -> std::io::Result<()> {
        self.experiment.add_result(key, data);
        self.auto_save()
    }

    /// Return a lightweight summary of the current experiment.
    pub fn summary(&self) -> ExperimentSummary {
        ExperimentSummary {
            name: self.experiment.config.name.clone(),
            hypothesis: self.experiment.config.hypothesis.clone(),
            result_count: self.experiment.result_count(),
            created_at: self.experiment.created_at.clone(),
            updated_at: self.experiment.updated_at.clone(),
        }
    }

    /// Borrow the underlying experiment.
    pub fn experiment(&self) -> &Experiment {
        &self.experiment
    }

    /// Compute field statistics across all results.
    pub fn field_stats(&self, field: &str) -> Option<FieldStats> {
        self.experiment.field_stats(field)
    }

    /// Save the experiment to disk as `<save_dir>/<name>.json`.
    ///
    /// # Errors
    /// Returns `std::io::Error` on serialisation or write failure.
    pub fn save(&self) -> std::io::Result<()> {
        let dir = match &self.save_dir {
            None => return Ok(()),
            Some(d) => d,
        };
        let filename = format!("{}.json", sanitise_filename(&self.experiment.config.name));
        let path = dir.join(filename);
        let json = serde_json::to_string_pretty(&self.experiment)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    /// Return the path this experiment would be saved to (if a save dir is set).
    pub fn save_path(&self) -> Option<PathBuf> {
        self.save_dir.as_ref().map(|dir| {
            let filename = format!("{}.json", sanitise_filename(&self.experiment.config.name));
            dir.join(filename)
        })
    }

    fn auto_save(&self) -> std::io::Result<()> {
        self.save()
    }
}

// ── Multi-experiment registry ─────────────────────────────────────────────────

/// A registry that manages multiple [`ExperimentManager`]s.
#[derive(Default)]
pub struct ExperimentRegistry {
    managers: HashMap<String, ExperimentManager>,
}

impl ExperimentRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an experiment manager.
    pub fn register(&mut self, manager: ExperimentManager) {
        let name = manager.experiment.config.name.clone();
        self.managers.insert(name, manager);
    }

    /// Look up a manager by experiment name.
    pub fn get(&self, name: &str) -> Option<&ExperimentManager> {
        self.managers.get(name)
    }

    /// Look up a manager mutably by experiment name.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut ExperimentManager> {
        self.managers.get_mut(name)
    }

    /// List summaries of all registered experiments.
    pub fn list(&self) -> Vec<ExperimentSummary> {
        let mut summaries: Vec<ExperimentSummary> =
            self.managers.values().map(|m| m.summary()).collect();
        summaries.sort_by(|a, b| a.name.cmp(&b.name));
        summaries
    }

    /// Number of registered experiments.
    pub fn len(&self) -> usize {
        self.managers.len()
    }

    /// Returns `true` when no experiments are registered.
    pub fn is_empty(&self) -> bool {
        self.managers.is_empty()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Return the current time as a naive RFC 3339-like string.
///
/// Uses [`std::time::SystemTime`] to avoid a `chrono` dependency.
fn now_rfc3339() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Format as YYYY-MM-DDTHH:MM:SSZ (UTC, no sub-second precision).
    let s = secs;
    let days_since_epoch = s / 86400;
    let time_of_day = s % 86400;
    let h = time_of_day / 3600;
    let m = (time_of_day % 3600) / 60;
    let sec = time_of_day % 60;
    // Simple Gregorian calendar conversion.
    let (year, month, day) = days_to_ymd(days_since_epoch);
    format!("{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{sec:02}Z")
}

fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    // Epoch = 1970-01-01.
    let mut year = 1970u64;
    loop {
        let leap = is_leap(year);
        let days_in_year = if leap { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }
    let leap = is_leap(year);
    let month_days: &[u64] = if leap {
        &[31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        &[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 1u64;
    for &md in month_days {
        if days < md {
            break;
        }
        days -= md;
        month += 1;
    }
    (year, month, days + 1)
}

fn is_leap(year: u64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Replace characters unsafe for filenames with underscores.
fn sanitise_filename(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_config(name: &str) -> ExperimentConfig {
        ExperimentConfig {
            name: name.into(),
            hypothesis: format!("Hypothesis for {name}"),
            ..Default::default()
        }
    }

    #[test]
    fn in_memory_manager_records_results() {
        let mut mgr = ExperimentManager::in_memory(make_config("test_exp"));
        mgr.record_result("run_1", json!({"quality": 0.9})).unwrap();
        mgr.record_result("run_2", json!({"quality": 0.8})).unwrap();
        assert_eq!(mgr.summary().result_count, 2);
    }

    #[test]
    fn field_stats_computed_correctly() {
        let mut mgr = ExperimentManager::in_memory(make_config("stats_test"));
        mgr.record_result("r1", json!({"score": 10.0})).unwrap();
        mgr.record_result("r2", json!({"score": 20.0})).unwrap();
        mgr.record_result("r3", json!({"score": 30.0})).unwrap();

        let stats = mgr.field_stats("score").unwrap();
        assert_eq!(stats.count, 3);
        assert!((stats.mean - 20.0).abs() < 1e-10);
        assert!((stats.min - 10.0).abs() < 1e-10);
        assert!((stats.max - 30.0).abs() < 1e-10);
    }

    #[test]
    fn field_stats_none_when_no_numeric_field() {
        let mgr = ExperimentManager::in_memory(make_config("no_field"));
        assert!(mgr.field_stats("nonexistent").is_none());
    }

    #[test]
    fn get_result_by_key() {
        let mut mgr = ExperimentManager::in_memory(make_config("key_test"));
        mgr.record_result("baseline", json!({"val": 1})).unwrap();
        mgr.record_result("variant_a", json!({"val": 2})).unwrap();

        let r = mgr.experiment().get_result("baseline").unwrap();
        assert_eq!(r.data["val"], 1);
        assert!(mgr.experiment().get_result("missing").is_none());
    }

    #[test]
    fn registry_list_and_lookup() {
        let mut registry = ExperimentRegistry::new();
        registry.register(ExperimentManager::in_memory(make_config("exp_a")));
        registry.register(ExperimentManager::in_memory(make_config("exp_b")));

        assert_eq!(registry.len(), 2);
        let list = registry.list();
        assert_eq!(list[0].name, "exp_a");
        assert_eq!(list[1].name, "exp_b");
    }

    #[test]
    fn save_dir_creates_json() {
        let tmp = std::env::temp_dir().join("eot_experiment_test");
        let mut mgr =
            ExperimentManager::with_save_dir(make_config("disk_test"), &tmp).unwrap();
        mgr.record_result("r1", json!({"x": 42})).unwrap();

        let path = mgr.save_path().unwrap();
        assert!(path.exists(), "JSON file should have been written");
        let content = std::fs::read_to_string(&path).unwrap();
        let loaded: Experiment = serde_json::from_str(&content).unwrap();
        assert_eq!(loaded.result_count(), 1);

        // Cleanup.
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_dir(tmp);
    }

    #[test]
    fn sanitise_filename_replaces_special_chars() {
        assert_eq!(sanitise_filename("my exp/test"), "my_exp_test");
        assert_eq!(sanitise_filename("valid-name_123"), "valid-name_123");
    }

    #[test]
    fn now_rfc3339_format() {
        let ts = now_rfc3339();
        // Should look like 2026-03-22T...
        assert!(ts.len() >= 19);
        assert!(ts.contains('T'));
        assert!(ts.ends_with('Z'));
    }
}
