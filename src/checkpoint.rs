//! # Experiment Checkpointing
//!
//! Saves the full state of a running research experiment to disk at configurable
//! intervals, enabling **resumable long runs**.
//!
//! ## Problem
//!
//! A 500-run experiment (50 prompts × 10 transforms) can take hours.  If the
//! process is killed halfway through, all progress is lost and the run must
//! restart from zero.
//!
//! ## Solution
//!
//! [`ExperimentCheckpointer`] serialises the experiment's accumulated
//! [`ResearchOutput`]s plus the [`ThompsonBandit`] posterior state to a
//! JSON file after every `checkpoint_interval` runs.  On startup, if a
//! checkpoint file exists, it is loaded and the run continues from where it
//! left off rather than restarting.
//!
//! ## File format
//!
//! Checkpoints are plain JSON files (`.eot_checkpoint.json` by default).
//! The schema includes a `version` field for forward-compatibility.
//!
//! ## Example
//!
//! ```no_run
//! use every_other_token::checkpoint::{ExperimentCheckpointer, CheckpointConfig};
//! use every_other_token::bayesian::ThompsonBandit;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = CheckpointConfig {
//!     path: std::path::PathBuf::from(".eot_checkpoint.json"),
//!     interval: 10,  // save every 10 completed runs
//! };
//!
//! let mut cp = ExperimentCheckpointer::new(config);
//!
//! // Resume from prior state if available.
//! let mut bandit = ThompsonBandit::new(0.5);
//! let start_run = cp.load_into(&mut bandit).await?;
//! println!("Resuming from run {start_run}");
//!
//! // After each completed run:
//! // cp.record_run(run_output).await;
//! // cp.maybe_save(&bandit, run_index).await?;
//! # Ok(())
//! # }
//! ```

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;
use tracing::{debug, info, warn};

use crate::bayesian::ThompsonBandit;
use crate::research::ResearchOutput;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`ExperimentCheckpointer`].
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// File path for the checkpoint JSON.
    ///
    /// Defaults to `.eot_checkpoint.json` in the current directory.
    pub path: PathBuf,
    /// Save a checkpoint every `interval` completed runs.
    ///
    /// Set to `1` to checkpoint after every run (safest; more I/O).
    /// Defaults to `10`.
    pub interval: usize,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from(".eot_checkpoint.json"),
            interval: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// Checkpoint file schema
// ---------------------------------------------------------------------------

/// The on-disk checkpoint schema.
#[derive(Debug, Serialize, Deserialize)]
pub struct CheckpointFile {
    /// Schema version for forward-compatibility.
    pub version: u32,
    /// Index of the next run to execute (i.e., the number of completed runs).
    pub completed_runs: usize,
    /// Serialised [`ThompsonBandit`] state (arms, posteriors, totals).
    pub bandit: ThompsonBandit,
    /// Accumulated research outputs from completed runs.
    pub outputs: Vec<ResearchOutput>,
}

impl CheckpointFile {
    const VERSION: u32 = 1;
}

// ---------------------------------------------------------------------------
// ExperimentCheckpointer
// ---------------------------------------------------------------------------

/// Manages save/load of experiment state for resumable runs.
///
/// See the [module documentation][self] for a full usage example.
pub struct ExperimentCheckpointer {
    config: CheckpointConfig,
    outputs: Vec<ResearchOutput>,
}

impl ExperimentCheckpointer {
    /// Create a new checkpointer with the given configuration.
    pub fn new(config: CheckpointConfig) -> Self {
        Self {
            config,
            outputs: Vec::new(),
        }
    }

    /// Try to load a prior checkpoint.
    ///
    /// On success, restores the bandit posteriors and accumulated outputs into
    /// `bandit` and returns the index of the next run to execute
    /// (`completed_runs`).  Returns `0` if no checkpoint exists (fresh start).
    ///
    /// # Errors
    ///
    /// Returns an error only on I/O or JSON parse failure when the file
    /// *exists* but cannot be read.
    pub async fn load_into(
        &mut self,
        bandit: &mut ThompsonBandit,
    ) -> Result<usize, CheckpointError> {
        if !self.config.path.exists() {
            debug!("no checkpoint found at {:?} — starting fresh", self.config.path);
            return Ok(0);
        }

        let bytes = tokio::fs::read(&self.config.path)
            .await
            .map_err(|e| CheckpointError::Io(e.to_string()))?;

        let file: CheckpointFile = serde_json::from_slice(&bytes)
            .map_err(|e| CheckpointError::Parse(e.to_string()))?;

        if file.version != CheckpointFile::VERSION {
            warn!(
                "checkpoint version {} != expected {}; ignoring",
                file.version,
                CheckpointFile::VERSION
            );
            return Ok(0);
        }

        *bandit = file.bandit;
        self.outputs = file.outputs;

        info!(
            completed = file.completed_runs,
            outputs = self.outputs.len(),
            "resumed from checkpoint"
        );

        Ok(file.completed_runs)
    }

    /// Record one completed run's output.
    pub fn record_run(&mut self, output: ResearchOutput) {
        self.outputs.push(output);
    }

    /// Save a checkpoint if `run_index` is a multiple of `config.interval`.
    ///
    /// The checkpoint is written atomically (write to `.tmp`, then rename).
    ///
    /// # Errors
    ///
    /// Returns an error on I/O failure.
    pub async fn maybe_save(
        &self,
        bandit: &ThompsonBandit,
        run_index: usize,
    ) -> Result<(), CheckpointError> {
        if run_index % self.config.interval == 0 && run_index > 0 {
            self.save(bandit, run_index).await?;
        }
        Ok(())
    }

    /// Force-save a checkpoint immediately.
    pub async fn save(
        &self,
        bandit: &ThompsonBandit,
        completed_runs: usize,
    ) -> Result<(), CheckpointError> {
        let file = CheckpointFile {
            version: CheckpointFile::VERSION,
            completed_runs,
            bandit: bandit.clone(),
            outputs: self.outputs.clone(),
        };

        let json = serde_json::to_vec_pretty(&file)
            .map_err(|e| CheckpointError::Serialize(e.to_string()))?;

        let tmp_path = self.config.path.with_extension("tmp");

        let mut f = tokio::fs::File::create(&tmp_path)
            .await
            .map_err(|e| CheckpointError::Io(e.to_string()))?;

        f.write_all(&json)
            .await
            .map_err(|e| CheckpointError::Io(e.to_string()))?;

        f.flush()
            .await
            .map_err(|e| CheckpointError::Io(e.to_string()))?;

        drop(f);

        tokio::fs::rename(&tmp_path, &self.config.path)
            .await
            .map_err(|e| CheckpointError::Io(e.to_string()))?;

        debug!(
            path = ?self.config.path,
            completed = completed_runs,
            outputs = self.outputs.len(),
            "checkpoint saved"
        );

        Ok(())
    }

    /// Delete the checkpoint file (e.g. after a successful run completes).
    pub async fn delete(&self) -> Result<(), CheckpointError> {
        if self.config.path.exists() {
            tokio::fs::remove_file(&self.config.path)
                .await
                .map_err(|e| CheckpointError::Io(e.to_string()))?;
            info!(path = ?self.config.path, "checkpoint deleted after successful run");
        }
        Ok(())
    }

    /// Return all accumulated outputs.
    pub fn outputs(&self) -> &[ResearchOutput] {
        &self.outputs
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by [`ExperimentCheckpointer`].
#[derive(Debug)]
pub enum CheckpointError {
    /// File I/O error.
    Io(String),
    /// JSON deserialisation error.
    Parse(String),
    /// JSON serialisation error.
    Serialize(String),
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckpointError::Io(e) => write!(f, "checkpoint I/O error: {e}"),
            CheckpointError::Parse(e) => write!(f, "checkpoint parse error: {e}"),
            CheckpointError::Serialize(e) => write!(f, "checkpoint serialize error: {e}"),
        }
    }
}

impl std::error::Error for CheckpointError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn empty_bandit() -> ThompsonBandit {
        let mut b = ThompsonBandit::new(0.5);
        b.add_arm("a");
        b.add_arm("b");
        b
    }

    #[tokio::test]
    async fn load_nonexistent_returns_zero() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().with_extension("nonexistent.json");
        let config = CheckpointConfig { path, interval: 5 };
        let mut cp = ExperimentCheckpointer::new(config);
        let mut bandit = empty_bandit();
        let start = cp.load_into(&mut bandit).await.unwrap();
        assert_eq!(start, 0);
    }

    #[tokio::test]
    async fn save_and_reload_restores_bandit() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let config = CheckpointConfig { path, interval: 1 };
        let mut cp = ExperimentCheckpointer::new(config.clone());

        let mut bandit = empty_bandit();
        bandit.update("a", 0.9);
        bandit.update("a", 0.8);
        bandit.update("b", 0.2);

        cp.save(&bandit, 5).await.unwrap();

        // Reload.
        let config2 = CheckpointConfig {
            path: config.path.clone(),
            interval: 1,
        };
        let mut cp2 = ExperimentCheckpointer::new(config2);
        let mut bandit2 = empty_bandit();
        let start = cp2.load_into(&mut bandit2).await.unwrap();

        assert_eq!(start, 5);
        assert_eq!(
            bandit2.best_arm().map(|a| a.name.as_str()),
            Some("a"),
            "best arm should be restored"
        );
    }

    #[tokio::test]
    async fn maybe_save_respects_interval() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().with_extension("maybe.json");
        let config = CheckpointConfig { path: path.clone(), interval: 5 };
        let cp = ExperimentCheckpointer::new(config);
        let bandit = empty_bandit();

        // Not a multiple of 5 — should not save.
        cp.maybe_save(&bandit, 3).await.unwrap();
        assert!(!path.exists(), "should not have saved at run 3");

        // Multiple of 5 — should save.
        cp.maybe_save(&bandit, 5).await.unwrap();
        assert!(path.exists(), "should have saved at run 5");
    }
}
