//! Streaming token compression with configurable backpressure.
//!
//! Provides a `StreamCompressor` that accepts token batches, applies
//! configurable backpressure strategies when the buffer fills up, and
//! drains a compressed output stream on demand.

use std::collections::VecDeque;

// ── BackpressureStrategy ──────────────────────────────────────────────────────

/// Strategy applied when the stream buffer is above the high-watermark.
#[derive(Debug, Clone, PartialEq)]
pub enum BackpressureStrategy {
    /// Drop new tokens when the buffer is full.
    Drop,
    /// Signal the caller to throttle until the buffer drains.
    Block,
    /// Aggressively compress incoming tokens when above the high-watermark.
    Compress,
}

// ── BackpressureConfig ────────────────────────────────────────────────────────

/// Configuration for a `StreamCompressor`.
#[derive(Debug, Clone)]
pub struct BackpressureConfig {
    /// Maximum number of tokens the buffer may hold.
    pub buffer_size: usize,
    /// Buffer level at which backpressure kicks in.
    pub high_watermark: usize,
    /// Buffer level below which backpressure is released.
    pub low_watermark: usize,
    /// Which backpressure strategy to apply.
    pub strategy: BackpressureStrategy,
}

impl Default for BackpressureConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1024,
            high_watermark: 768,
            low_watermark: 256,
            strategy: BackpressureStrategy::Drop,
        }
    }
}

// ── PushResult ────────────────────────────────────────────────────────────────

/// Result returned by `StreamCompressor::push`.
#[derive(Debug, Clone, PartialEq)]
pub enum PushResult {
    /// All tokens were accepted into the buffer.
    Accepted,
    /// `n` tokens were dropped because the buffer was full.
    Dropped(usize),
    /// The caller should throttle; tokens were held back.
    Throttled,
}

// ── StreamStats ───────────────────────────────────────────────────────────────

/// Snapshot of compressor statistics.
#[derive(Debug, Clone, Default)]
pub struct StreamStats {
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub drops: u64,
    pub throttles: u64,
    pub current_buffer: usize,
    pub ratio: f64,
}

impl StreamStats {
    fn compute_ratio(&mut self) {
        if self.tokens_in == 0 {
            self.ratio = 1.0;
        } else {
            self.ratio = self.tokens_out as f64 / self.tokens_in as f64;
        }
    }
}

// ── StreamCompressor ──────────────────────────────────────────────────────────

/// Compress a stream of tokens with configurable backpressure.
///
/// The compressor maintains an internal buffer. When `push` is called, tokens
/// are added to the buffer according to the configured backpressure strategy.
/// When `drain` is called, the buffer is flushed and the tokens are compressed
/// (by keeping one in every `1/compression_ratio` tokens).
pub struct StreamCompressor {
    config: BackpressureConfig,
    /// Target fraction of tokens to keep (0.0 – 1.0).
    compression_ratio: f64,
    buffer: VecDeque<String>,
    stats: StreamStats,
    /// Whether we are currently in a throttled state.
    throttled: bool,
}

impl StreamCompressor {
    /// Create a new compressor with the given config and compression ratio.
    ///
    /// `compression_ratio` is the fraction of tokens kept (e.g. 0.5 keeps half).
    /// Clamped to [0.0, 1.0].
    pub fn new(config: BackpressureConfig, compression_ratio: f64) -> Self {
        let compression_ratio = compression_ratio.clamp(0.0, 1.0);
        Self {
            config,
            compression_ratio,
            buffer: VecDeque::new(),
            stats: StreamStats::default(),
            throttled: false,
        }
    }

    /// Current buffer fill level.
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Read-only view of the current statistics.
    pub fn stats(&self) -> &StreamStats {
        &self.stats
    }

    /// Push a batch of tokens into the compressor.
    ///
    /// Returns a `PushResult` describing what happened.
    pub fn push(&mut self, tokens: Vec<String>) -> PushResult {
        self.stats.tokens_in += tokens.len() as u64;
        let buf_len = self.buffer.len();

        // If throttled, check whether we have drained below the low watermark.
        if self.throttled {
            if buf_len <= self.config.low_watermark {
                self.throttled = false;
            } else {
                self.stats.throttles += 1;
                self.stats.current_buffer = buf_len;
                return PushResult::Throttled;
            }
        }

        match self.config.strategy {
            BackpressureStrategy::Drop => {
                let space = self.config.buffer_size.saturating_sub(buf_len);
                if space == 0 {
                    let dropped = tokens.len();
                    self.stats.drops += dropped as u64;
                    self.stats.current_buffer = self.buffer.len();
                    return PushResult::Dropped(dropped);
                }
                let to_take = tokens.len().min(space);
                let dropped = tokens.len() - to_take;
                for tok in tokens.into_iter().take(to_take) {
                    self.buffer.push_back(tok);
                }
                self.stats.drops += dropped as u64;
                self.stats.current_buffer = self.buffer.len();
                if dropped > 0 {
                    PushResult::Dropped(dropped)
                } else {
                    PushResult::Accepted
                }
            }

            BackpressureStrategy::Block => {
                if buf_len >= self.config.high_watermark {
                    self.throttled = true;
                    self.stats.throttles += 1;
                    self.stats.current_buffer = buf_len;
                    return PushResult::Throttled;
                }
                for tok in tokens {
                    self.buffer.push_back(tok);
                }
                self.stats.current_buffer = self.buffer.len();
                PushResult::Accepted
            }

            BackpressureStrategy::Compress => {
                let incoming = if buf_len >= self.config.high_watermark {
                    // Aggressively compress: keep only every other token
                    tokens
                        .into_iter()
                        .step_by(2)
                        .collect::<Vec<_>>()
                } else {
                    tokens
                };
                let space = self.config.buffer_size.saturating_sub(self.buffer.len());
                let to_take = incoming.len().min(space);
                let dropped = incoming.len() - to_take;
                for tok in incoming.into_iter().take(to_take) {
                    self.buffer.push_back(tok);
                }
                self.stats.drops += dropped as u64;
                self.stats.current_buffer = self.buffer.len();
                if dropped > 0 {
                    PushResult::Dropped(dropped)
                } else {
                    PushResult::Accepted
                }
            }
        }
    }

    /// Drain the buffer and return compressed output.
    ///
    /// Compression is performed by keeping tokens at a stride determined by
    /// `compression_ratio`. A ratio of 1.0 keeps all tokens; 0.5 keeps every
    /// other token; 0.0 keeps no tokens.
    pub fn drain(&mut self) -> Vec<String> {
        let all: Vec<String> = self.buffer.drain(..).collect();
        self.stats.current_buffer = 0;

        if all.is_empty() {
            return Vec::new();
        }

        let compressed = self.compress_tokens(all);
        self.stats.tokens_out += compressed.len() as u64;
        self.stats.compute_ratio();
        compressed
    }

    /// Drain only up to `n` raw tokens from the buffer (before compression).
    pub fn drain_n(&mut self, n: usize) -> Vec<String> {
        let take = n.min(self.buffer.len());
        let chunk: Vec<String> = self.buffer.drain(..take).collect();
        self.stats.current_buffer = self.buffer.len();

        if chunk.is_empty() {
            return Vec::new();
        }

        let compressed = self.compress_tokens(chunk);
        self.stats.tokens_out += compressed.len() as u64;
        self.stats.compute_ratio();
        compressed
    }

    /// Apply compression to a token list using the configured ratio.
    fn compress_tokens(&self, tokens: Vec<String>) -> Vec<String> {
        if self.compression_ratio >= 1.0 {
            return tokens;
        }
        if self.compression_ratio <= 0.0 {
            return Vec::new();
        }

        let n = tokens.len();
        let keep = ((n as f64) * self.compression_ratio).round() as usize;
        if keep == 0 {
            return Vec::new();
        }

        // Uniformly sample `keep` tokens from the sequence.
        let stride = if keep >= n {
            1
        } else {
            (n as f64 / keep as f64).round() as usize
        };

        tokens.into_iter().step_by(stride.max(1)).take(keep).collect()
    }

    /// Reset the compressor state (clears buffer and stats).
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.stats = StreamStats::default();
        self.throttled = false;
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tokens(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("tok{}", i)).collect()
    }

    fn default_config() -> BackpressureConfig {
        BackpressureConfig {
            buffer_size: 10,
            high_watermark: 7,
            low_watermark: 3,
            strategy: BackpressureStrategy::Drop,
        }
    }

    // ── Basic push/drain tests ────────────────────────────────────────────

    #[test]
    fn test_push_accepted_within_capacity() {
        let mut sc = StreamCompressor::new(default_config(), 1.0);
        let result = sc.push(make_tokens(5));
        assert_eq!(result, PushResult::Accepted);
        assert_eq!(sc.buffer_len(), 5);
    }

    #[test]
    fn test_drain_returns_compressed() {
        let mut sc = StreamCompressor::new(default_config(), 0.5);
        sc.push(make_tokens(10)).ok();
        let out = sc.drain();
        assert!(out.len() <= 5, "expected ~50% of 10 = 5, got {}", out.len());
        assert!(!out.is_empty());
    }

    #[test]
    fn test_drain_clears_buffer() {
        let mut sc = StreamCompressor::new(default_config(), 1.0);
        sc.push(make_tokens(4)).ok();
        sc.drain();
        assert_eq!(sc.buffer_len(), 0);
    }

    #[test]
    fn test_drain_empty_buffer() {
        let mut sc = StreamCompressor::new(default_config(), 1.0);
        let out = sc.drain();
        assert!(out.is_empty());
    }

    // ── Drop strategy ─────────────────────────────────────────────────────

    #[test]
    fn test_drop_strategy_drops_when_full() {
        let mut sc = StreamCompressor::new(default_config(), 1.0);
        // Fill buffer to capacity
        sc.push(make_tokens(10)).ok();
        let result = sc.push(vec!["overflow".into()]);
        match result {
            PushResult::Dropped(n) => assert_eq!(n, 1),
            other => panic!("expected Dropped(1), got {:?}", other),
        }
    }

    #[test]
    fn test_drop_strategy_partial_drop() {
        let mut sc = StreamCompressor::new(default_config(), 1.0);
        // Fill to 8 (2 spaces left)
        sc.push(make_tokens(8)).ok();
        let result = sc.push(make_tokens(5));
        match result {
            PushResult::Dropped(n) => assert_eq!(n, 3),
            other => panic!("expected Dropped(3), got {:?}", other),
        }
        assert_eq!(sc.buffer_len(), 10);
    }

    // ── Block strategy ────────────────────────────────────────────────────

    #[test]
    fn test_block_strategy_throttles_above_high_watermark() {
        let config = BackpressureConfig {
            strategy: BackpressureStrategy::Block,
            ..default_config()
        };
        let mut sc = StreamCompressor::new(config, 1.0);
        // Fill past high watermark
        sc.push(make_tokens(7)).ok();
        let result = sc.push(vec!["extra".into()]);
        assert_eq!(result, PushResult::Throttled);
    }

    #[test]
    fn test_block_strategy_accepts_below_high_watermark() {
        let config = BackpressureConfig {
            strategy: BackpressureStrategy::Block,
            ..default_config()
        };
        let mut sc = StreamCompressor::new(config, 1.0);
        sc.push(make_tokens(3)).ok();
        let result = sc.push(make_tokens(2));
        assert_eq!(result, PushResult::Accepted);
    }

    // ── Compress strategy ─────────────────────────────────────────────────

    #[test]
    fn test_compress_strategy_reduces_tokens_above_watermark() {
        let config = BackpressureConfig {
            strategy: BackpressureStrategy::Compress,
            ..default_config()
        };
        let mut sc = StreamCompressor::new(config, 1.0);
        // Fill past high watermark
        sc.push(make_tokens(8)).ok();
        let buf_before = sc.buffer_len();
        sc.push(make_tokens(6)).ok(); // Should be halved before adding
        let added = sc.buffer_len() - buf_before;
        assert!(added <= 3, "above watermark should aggressively compress, added {}", added);
    }

    // ── Compression ratio tests ───────────────────────────────────────────

    #[test]
    fn test_ratio_100_percent_keeps_all() {
        let mut sc = StreamCompressor::new(default_config(), 1.0);
        sc.push(make_tokens(8)).ok();
        let out = sc.drain();
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_ratio_0_percent_keeps_none() {
        let mut sc = StreamCompressor::new(default_config(), 0.0);
        sc.push(make_tokens(8)).ok();
        let out = sc.drain();
        assert_eq!(out.len(), 0);
    }

    #[test]
    fn test_ratio_50_percent_keeps_half() {
        let config = BackpressureConfig {
            buffer_size: 100,
            high_watermark: 80,
            low_watermark: 20,
            strategy: BackpressureStrategy::Drop,
        };
        let mut sc = StreamCompressor::new(config, 0.5);
        sc.push(make_tokens(20)).ok();
        let out = sc.drain();
        assert!(
            out.len() >= 8 && out.len() <= 12,
            "expected ~10 tokens, got {}",
            out.len()
        );
    }

    // ── Stats tests ───────────────────────────────────────────────────────

    #[test]
    fn test_stats_tokens_in_tracked() {
        let mut sc = StreamCompressor::new(default_config(), 1.0);
        sc.push(make_tokens(5)).ok();
        sc.push(make_tokens(3)).ok();
        assert_eq!(sc.stats().tokens_in, 8);
    }

    #[test]
    fn test_stats_tokens_out_tracked_after_drain() {
        let mut sc = StreamCompressor::new(default_config(), 1.0);
        sc.push(make_tokens(5)).ok();
        sc.drain();
        assert_eq!(sc.stats().tokens_out, 5);
    }

    #[test]
    fn test_stats_drops_tracked() {
        let mut sc = StreamCompressor::new(default_config(), 1.0);
        sc.push(make_tokens(10)).ok();
        sc.push(make_tokens(3)).ok();
        assert!(sc.stats().drops > 0);
    }

    #[test]
    fn test_stats_ratio_after_drain() {
        let config = BackpressureConfig {
            buffer_size: 100,
            high_watermark: 80,
            low_watermark: 20,
            strategy: BackpressureStrategy::Drop,
        };
        let mut sc = StreamCompressor::new(config, 0.5);
        sc.push(make_tokens(10)).ok();
        sc.drain();
        let r = sc.stats().ratio;
        assert!(r > 0.0 && r <= 1.0, "ratio should be in (0, 1], got {}", r);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut sc = StreamCompressor::new(default_config(), 1.0);
        sc.push(make_tokens(5)).ok();
        sc.reset();
        assert_eq!(sc.buffer_len(), 0);
        assert_eq!(sc.stats().tokens_in, 0);
    }

    // ── Async-style integration tests ─────────────────────────────────────

    #[test]
    fn test_pipeline_multiple_push_drain_cycles() {
        let config = BackpressureConfig {
            buffer_size: 20,
            high_watermark: 15,
            low_watermark: 5,
            strategy: BackpressureStrategy::Drop,
        };
        let mut sc = StreamCompressor::new(config, 0.75);
        let mut total_out = 0usize;

        for _ in 0..5 {
            sc.push(make_tokens(6)).ok();
            let out = sc.drain();
            total_out += out.len();
        }

        assert!(total_out > 0, "should have produced output over multiple cycles");
        assert!(sc.stats().tokens_in > 0);
    }

    #[test]
    fn test_drain_n_partial() {
        let config = BackpressureConfig {
            buffer_size: 100,
            high_watermark: 80,
            low_watermark: 20,
            strategy: BackpressureStrategy::Drop,
        };
        let mut sc = StreamCompressor::new(config, 1.0);
        sc.push(make_tokens(10)).ok();
        let out = sc.drain_n(5);
        assert_eq!(out.len(), 5);
        assert_eq!(sc.buffer_len(), 5);
    }

    #[test]
    fn test_throttle_released_after_drain() {
        let config = BackpressureConfig {
            buffer_size: 10,
            high_watermark: 7,
            low_watermark: 3,
            strategy: BackpressureStrategy::Block,
        };
        let mut sc = StreamCompressor::new(config, 1.0);
        sc.push(make_tokens(7)).ok();
        let r1 = sc.push(make_tokens(1));
        assert_eq!(r1, PushResult::Throttled);

        // Drain below low watermark
        sc.drain_n(5);

        // Now push should work
        let r2 = sc.push(make_tokens(1));
        assert_eq!(r2, PushResult::Accepted);
    }

    #[test]
    fn test_push_empty_tokens() {
        let mut sc = StreamCompressor::new(default_config(), 1.0);
        let result = sc.push(vec![]);
        assert_eq!(result, PushResult::Accepted);
        assert_eq!(sc.buffer_len(), 0);
    }
}
