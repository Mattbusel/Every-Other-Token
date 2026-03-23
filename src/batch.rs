//! Batch processing pipeline for token sequences.
//!
//! This module provides a priority-queue based batch processor that can handle
//! multiple token compression jobs concurrently using Tokio's `JoinSet`.
//!
//! # Example
//! ```no_run
//! use every_other_token::batch::{BatchConfig, BatchProcessor};
//! use std::time::Duration;
//!
//! #[tokio::main]
//! async fn main() {
//!     let config = BatchConfig {
//!         max_concurrent: 4,
//!         queue_capacity: 100,
//!         timeout: Duration::from_secs(30),
//!     };
//!     let mut processor = BatchProcessor::new(config);
//!     let id = processor.submit(vec!["hello".into(), "world".into()], 5);
//!     println!("Submitted job {id}");
//! }
//! ```

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::{Duration, Instant};
use tokio::task::JoinSet;
use tokio_stream::wrappers::ReceiverStream;
use tokio::sync::mpsc;

// ── Job types ────────────────────────────────────────────────────────────────

/// A single batch compression job.
#[derive(Debug, Clone)]
pub struct BatchJob {
    /// Unique job identifier.
    pub id: u64,
    /// Token sequence to compress.
    pub tokens: Vec<String>,
    /// Priority (higher = processed first).
    pub priority: u8,
    /// When the job was submitted.
    pub submitted_at: Instant,
}

impl PartialEq for BatchJob {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.id == other.id
    }
}

impl Eq for BatchJob {}

impl PartialOrd for BatchJob {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BatchJob {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first; break ties by submission order (earlier = first)
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.id.cmp(&self.id))
    }
}

// ── Result types ─────────────────────────────────────────────────────────────

/// Result of processing a single batch job.
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// The job that produced this result.
    pub job_id: u64,
    /// Compressed token sequence.
    pub compressed: Vec<String>,
    /// Length of the original token sequence.
    pub original_len: usize,
    /// Length of the compressed token sequence.
    pub compressed_len: usize,
    /// Compression ratio (compressed_len / original_len). Lower is better.
    pub ratio: f64,
    /// Wall-clock time taken to process, in milliseconds.
    pub elapsed_ms: u64,
}

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for `BatchProcessor`.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of jobs running concurrently.
    pub max_concurrent: usize,
    /// Maximum number of queued jobs (channel capacity).
    pub queue_capacity: usize,
    /// Per-job timeout. Jobs that exceed this are cancelled.
    pub timeout: Duration,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 4,
            queue_capacity: 256,
            timeout: Duration::from_secs(60),
        }
    }
}

// ── Statistics ─────────────────────────────────────────────────────────────────

/// Aggregate statistics for a completed batch run.
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    /// Number of jobs submitted.
    pub jobs_submitted: u64,
    /// Number of jobs that completed successfully.
    pub jobs_completed: u64,
    /// Number of jobs that failed or timed out.
    pub jobs_failed: u64,
    /// Mean compression ratio across completed jobs.
    pub avg_ratio: f64,
    /// Throughput in tokens per second.
    pub throughput_tokens_per_sec: f64,
}

// ── BatchProcessor ─────────────────────────────────────────────────────────────

/// Processes multiple token sequences concurrently using a priority queue and
/// Tokio's `JoinSet`.
///
/// Jobs are sorted by priority (highest first). Submit jobs with [`submit`],
/// then call [`run`] to get a stream of [`BatchResult`]s as they complete.
pub struct BatchProcessor {
    config: BatchConfig,
    queue: BinaryHeap<BatchJob>,
    next_id: u64,
}

impl BatchProcessor {
    /// Create a new `BatchProcessor` with the given configuration.
    pub fn new(config: BatchConfig) -> Self {
        Self {
            config,
            queue: BinaryHeap::new(),
            next_id: 1,
        }
    }

    /// Submit a token sequence for compression.
    ///
    /// Returns the job ID that can be used to correlate results.
    pub fn submit(&mut self, tokens: Vec<String>, priority: u8) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.queue.push(BatchJob {
            id,
            tokens,
            priority,
            submitted_at: Instant::now(),
        });
        id
    }

    /// Drain the priority queue and process jobs concurrently.
    ///
    /// Returns a stream of `BatchResult`s in completion order.
    pub fn run(self) -> ReceiverStream<BatchResult> {
        let (tx, rx) = mpsc::channel::<BatchResult>(self.config.queue_capacity.max(1));
        let config = self.config.clone();
        let jobs: Vec<BatchJob> = self.queue.into_sorted_vec();
        // into_sorted_vec gives ascending order; we want descending (highest priority first)
        let jobs: Vec<BatchJob> = jobs.into_iter().rev().collect();

        tokio::spawn(async move {
            let mut set: JoinSet<Option<BatchResult>> = JoinSet::new();
            let mut job_iter = jobs.into_iter();
            let timeout = config.timeout;
            let max_concurrent = config.max_concurrent.max(1);

            // Seed the initial batch
            while set.len() < max_concurrent {
                if let Some(job) = job_iter.next() {
                    let job_clone = job.clone();
                    let timeout_dur = timeout;
                    set.spawn(async move {
                        process_job_with_timeout(job_clone, timeout_dur).await
                    });
                } else {
                    break;
                }
            }

            while let Some(result) = set.join_next().await {
                // Dispatch next queued job if any
                if let Some(job) = job_iter.next() {
                    let timeout_dur = timeout;
                    set.spawn(async move {
                        process_job_with_timeout(job, timeout_dur).await
                    });
                }

                match result {
                    Ok(Some(r)) => {
                        let _ = tx.send(r).await;
                    }
                    _ => {
                        // failed/timed-out/panicked — skip
                    }
                }
            }
        });

        ReceiverStream::new(rx)
    }

    /// Return aggregate statistics over a completed set of results.
    pub fn compute_stats(results: &[BatchResult], total_submitted: u64, wall_secs: f64) -> BatchStats {
        let jobs_completed = results.len() as u64;
        let jobs_failed = total_submitted.saturating_sub(jobs_completed);
        let avg_ratio = if jobs_completed > 0 {
            results.iter().map(|r| r.ratio).sum::<f64>() / jobs_completed as f64
        } else {
            0.0
        };
        let total_tokens: usize = results.iter().map(|r| r.original_len).sum();
        let throughput_tokens_per_sec = if wall_secs > 0.0 {
            total_tokens as f64 / wall_secs
        } else {
            0.0
        };
        BatchStats {
            jobs_submitted: total_submitted,
            jobs_completed,
            jobs_failed,
            avg_ratio,
            throughput_tokens_per_sec,
        }
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Compress a token sequence by dropping every-other token (the core EOT operation).
fn compress_tokens(tokens: &[String]) -> Vec<String> {
    tokens
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 2 == 0)
        .map(|(_, t)| t.clone())
        .collect()
}

/// Process a single job with a timeout.
async fn process_job_with_timeout(job: BatchJob, timeout: Duration) -> Option<BatchResult> {
    let start = Instant::now();
    let result = tokio::time::timeout(timeout, async move {
        // Simulate async work (yields once so other tasks can run)
        tokio::task::yield_now().await;
        let original_len = job.tokens.len();
        let compressed = compress_tokens(&job.tokens);
        let compressed_len = compressed.len();
        let ratio = if original_len > 0 {
            compressed_len as f64 / original_len as f64
        } else {
            1.0
        };
        let elapsed_ms = start.elapsed().as_millis() as u64;
        BatchResult {
            job_id: job.id,
            compressed,
            original_len,
            compressed_len,
            ratio,
            elapsed_ms,
        }
    })
    .await;
    result.ok()
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_stream::StreamExt;

    fn make_tokens(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("tok{i}")).collect()
    }

    #[test]
    fn test_batch_job_ordering() {
        let a = BatchJob {
            id: 1,
            tokens: vec![],
            priority: 5,
            submitted_at: Instant::now(),
        };
        let b = BatchJob {
            id: 2,
            tokens: vec![],
            priority: 10,
            submitted_at: Instant::now(),
        };
        assert!(b > a, "higher priority should be greater");
    }

    #[test]
    fn test_submit_increments_id() {
        let mut p = BatchProcessor::new(BatchConfig::default());
        let id1 = p.submit(make_tokens(4), 1);
        let id2 = p.submit(make_tokens(4), 2);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
    }

    #[test]
    fn test_compress_tokens_half() {
        let tokens = make_tokens(6);
        let c = compress_tokens(&tokens);
        assert_eq!(c.len(), 3);
        assert_eq!(c[0], "tok0");
        assert_eq!(c[1], "tok2");
        assert_eq!(c[2], "tok4");
    }

    #[test]
    fn test_compress_tokens_empty() {
        let c = compress_tokens(&[]);
        assert!(c.is_empty());
    }

    #[test]
    fn test_compress_tokens_single() {
        let c = compress_tokens(&["only".to_string()]);
        assert_eq!(c, vec!["only"]);
    }

    #[test]
    fn test_compress_tokens_two() {
        let c = compress_tokens(&["a".to_string(), "b".to_string()]);
        assert_eq!(c, vec!["a"]);
    }

    #[test]
    fn test_batch_result_ratio() {
        let original = make_tokens(10);
        let compressed = compress_tokens(&original);
        let ratio = compressed.len() as f64 / original.len() as f64;
        assert!((ratio - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_priority_queue_order() {
        let mut p = BatchProcessor::new(BatchConfig::default());
        p.submit(make_tokens(2), 1);
        p.submit(make_tokens(2), 10);
        p.submit(make_tokens(2), 5);
        // Peek: highest priority should be at the top
        let top = p.queue.peek().unwrap();
        assert_eq!(top.priority, 10);
    }

    #[test]
    fn test_compute_stats_empty() {
        let stats = BatchProcessor::compute_stats(&[], 5, 1.0);
        assert_eq!(stats.jobs_completed, 0);
        assert_eq!(stats.jobs_failed, 5);
        assert_eq!(stats.avg_ratio, 0.0);
    }

    #[test]
    fn test_compute_stats_basic() {
        let r = BatchResult {
            job_id: 1,
            compressed: vec![],
            original_len: 10,
            compressed_len: 5,
            ratio: 0.5,
            elapsed_ms: 10,
        };
        let stats = BatchProcessor::compute_stats(&[r], 1, 1.0);
        assert_eq!(stats.jobs_submitted, 1);
        assert_eq!(stats.jobs_completed, 1);
        assert_eq!(stats.jobs_failed, 0);
        assert!((stats.avg_ratio - 0.5).abs() < 1e-9);
        assert!((stats.throughput_tokens_per_sec - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_stats_throughput_zero_duration() {
        let r = BatchResult {
            job_id: 1,
            compressed: vec![],
            original_len: 10,
            compressed_len: 5,
            ratio: 0.5,
            elapsed_ms: 10,
        };
        let stats = BatchProcessor::compute_stats(&[r], 1, 0.0);
        assert_eq!(stats.throughput_tokens_per_sec, 0.0);
    }

    #[test]
    fn test_batch_config_default() {
        let cfg = BatchConfig::default();
        assert_eq!(cfg.max_concurrent, 4);
        assert_eq!(cfg.queue_capacity, 256);
    }

    #[tokio::test]
    async fn test_run_empty_queue() {
        let p = BatchProcessor::new(BatchConfig::default());
        let mut stream = p.run();
        // Should complete immediately with no results
        let result = tokio::time::timeout(Duration::from_secs(2), stream.next()).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_run_single_job() {
        let mut p = BatchProcessor::new(BatchConfig::default());
        p.submit(make_tokens(6), 1);
        let mut stream = p.run();
        let result = tokio::time::timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("timeout")
            .expect("expected a result");
        assert_eq!(result.job_id, 1);
        assert_eq!(result.original_len, 6);
        assert_eq!(result.compressed_len, 3);
        assert!((result.ratio - 0.5).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_run_multiple_jobs() {
        let mut p = BatchProcessor::new(BatchConfig::default());
        for i in 0..5u8 {
            p.submit(make_tokens(4), i);
        }
        let mut stream = p.run();
        let mut count = 0;
        while let Some(r) = tokio::time::timeout(Duration::from_secs(10), stream.next())
            .await
            .unwrap_or(None)
        {
            assert_eq!(r.original_len, 4);
            assert_eq!(r.compressed_len, 2);
            count += 1;
        }
        assert_eq!(count, 5);
    }

    #[tokio::test]
    async fn test_run_priority_respected() {
        // Submit low priority first, then high. High should complete (jobs are
        // processed in priority order, but stream yields in completion order).
        let mut p = BatchProcessor::new(BatchConfig {
            max_concurrent: 1,
            queue_capacity: 16,
            timeout: Duration::from_secs(10),
        });
        let _id_low = p.submit(make_tokens(2), 1);
        let _id_high = p.submit(make_tokens(2), 255);

        let mut stream = p.run();
        let first = tokio::time::timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("timeout")
            .expect("expected result");
        // With max_concurrent=1 and priority queue, the highest priority job
        // should start first and complete first.
        assert_eq!(first.job_id, 2); // id 2 is the high-priority one
    }

    #[tokio::test]
    async fn test_run_large_batch() {
        let mut p = BatchProcessor::new(BatchConfig {
            max_concurrent: 8,
            queue_capacity: 64,
            timeout: Duration::from_secs(30),
        });
        for i in 0..20u8 {
            p.submit(make_tokens(10), i % 5);
        }
        let stream = p.run();
        let results: Vec<BatchResult> = stream.collect().await;
        assert_eq!(results.len(), 20);
    }
}
