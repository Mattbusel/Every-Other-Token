//! LZ77-style token sequence deduplication.
//!
//! This module provides sliding-window compression for token sequences,
//! delta encoding between consecutive windows, and compression statistics.

// ── TokenRef ─────────────────────────────────────────────────────────────────

/// An element in the compressed token stream.
///
/// Either a raw literal token or a back-reference into the already-decoded
/// output (LZ77 style).
#[derive(Debug, Clone, PartialEq)]
pub enum TokenRef {
    /// A token that appears verbatim.
    Literal(String),
    /// A back-reference: `offset` tokens back, copy `length` tokens.
    Reference { offset: usize, length: usize },
}

// ── TokenCompressor ───────────────────────────────────────────────────────────

/// Sliding-window LZ77 compressor for token sequences.
#[derive(Debug, Clone)]
pub struct TokenCompressor {
    /// Number of tokens in the search window (history).
    pub window_size: usize,
    /// Number of tokens in the lookahead buffer.
    pub lookahead_size: usize,
}

impl Default for TokenCompressor {
    fn default() -> Self {
        Self {
            window_size: 256,
            lookahead_size: 32,
        }
    }
}

impl TokenCompressor {
    /// Create a new compressor with explicit parameters.
    pub fn new(window_size: usize, lookahead_size: usize) -> Self {
        Self { window_size, lookahead_size }
    }

    /// Compress `tokens` using a sliding-window LZ77 approach.
    ///
    /// For each position, the algorithm searches the preceding `window_size`
    /// tokens for the longest match with the current lookahead.  If the best
    /// match is at least `MIN_MATCH` tokens long, a [`TokenRef::Reference`]
    /// is emitted; otherwise a [`TokenRef::Literal`] is emitted and the
    /// position advances by one.
    pub fn compress(&self, tokens: &[String]) -> Vec<TokenRef> {
        const MIN_MATCH: usize = 3;

        let mut output = Vec::new();
        let mut pos = 0;

        while pos < tokens.len() {
            // Define search window bounds.
            let win_start = pos.saturating_sub(self.window_size);
            let lookahead_end = (pos + self.lookahead_size).min(tokens.len());

            let mut best_offset = 0usize;
            let mut best_length = 0usize;

            // Search the window O(n * w).
            for win_pos in win_start..pos {
                let mut length = 0;
                while pos + length < lookahead_end
                    && tokens[win_pos + length] == tokens[pos + length]
                {
                    length += 1;
                    // Prevent the match from running into the part of the
                    // window that is still ahead of win_pos's own length.
                    if win_pos + length >= pos && length >= pos - win_pos {
                        break;
                    }
                }
                if length > best_length {
                    best_length = length;
                    best_offset = pos - win_pos;
                }
            }

            if best_length >= MIN_MATCH {
                output.push(TokenRef::Reference {
                    offset: best_offset,
                    length: best_length,
                });
                pos += best_length;
            } else {
                output.push(TokenRef::Literal(tokens[pos].clone()));
                pos += 1;
            }
        }

        output
    }

    /// Decompress a sequence of [`TokenRef`]s back into the original tokens.
    pub fn decompress(&self, refs: &[TokenRef]) -> Vec<String> {
        let mut output: Vec<String> = Vec::new();

        for token_ref in refs {
            match token_ref {
                TokenRef::Literal(s) => output.push(s.clone()),
                TokenRef::Reference { offset, length } => {
                    let start = output.len().saturating_sub(*offset);
                    for i in 0..*length {
                        let idx = start + i;
                        if idx < output.len() {
                            let s = output[idx].clone();
                            output.push(s);
                        }
                    }
                }
            }
        }

        output
    }
}

// ── Top-level helpers ─────────────────────────────────────────────────────────

/// Compress using a default [`TokenCompressor`].
pub fn compress(tokens: &[String]) -> Vec<TokenRef> {
    TokenCompressor::default().compress(tokens)
}

/// Decompress using a default [`TokenCompressor`].
pub fn decompress(refs: &[TokenRef]) -> Vec<String> {
    TokenCompressor::default().decompress(refs)
}

// ── CompressionStats ──────────────────────────────────────────────────────────

/// Statistics describing the result of a compression run.
#[derive(Debug, Clone)]
pub struct CompressionStats {
    /// Number of tokens in the original sequence.
    pub original_count: usize,
    /// Number of entries in the compressed sequence.
    pub compressed_count: usize,
    /// Compression ratio: `original_count / compressed_count` (higher = better).
    pub ratio: f64,
    /// Number of [`TokenRef::Literal`] entries.
    pub literal_count: usize,
    /// Number of [`TokenRef::Reference`] entries.
    pub reference_count: usize,
}

/// Compute statistics for a compression run.
pub fn stats(original: &[String], compressed: &[TokenRef]) -> CompressionStats {
    let original_count = original.len();
    let compressed_count = compressed.len();
    let literal_count = compressed.iter().filter(|r| matches!(r, TokenRef::Literal(_))).count();
    let reference_count = compressed_count - literal_count;
    let ratio = if compressed_count == 0 {
        1.0
    } else {
        original_count as f64 / compressed_count as f64
    };

    CompressionStats {
        original_count,
        compressed_count,
        ratio,
        literal_count,
        reference_count,
    }
}

// ── DeltaEncoder ─────────────────────────────────────────────────────────────

/// A patch representing an edit list from one token window to the next.
#[derive(Debug, Clone, PartialEq)]
pub struct DeltaPatch {
    /// Positions (in the result) and the string to insert there.
    pub insertions: Vec<(usize, String)>,
    /// Positions (in the base) to delete (sorted ascending).
    pub deletions: Vec<usize>,
}

/// Streaming delta encoder: encodes the difference between consecutive token windows.
#[derive(Debug, Default)]
pub struct DeltaEncoder;

impl DeltaEncoder {
    /// Create a new `DeltaEncoder`.
    pub fn new() -> Self {
        Self
    }

    /// Encode the difference from `prev` to `curr` as a [`DeltaPatch`].
    ///
    /// Uses a simple longest-common-subsequence approach via a greedy
    /// two-pointer scan to find matching tokens, then classifies
    /// non-matching positions as deletions from `prev` or insertions into
    /// `curr`.
    pub fn encode_delta(&self, prev: &[String], curr: &[String]) -> DeltaPatch {
        // Build LCS via DP to get the edit script.
        let m = prev.len();
        let n = curr.len();

        // dp[i][j] = LCS length of prev[..i] vs curr[..j]
        let mut dp = vec![vec![0usize; n + 1]; m + 1];
        for i in 1..=m {
            for j in 1..=n {
                if prev[i - 1] == curr[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
                }
            }
        }

        // Backtrack to find which positions are in the LCS.
        let mut in_prev_lcs = vec![false; m];
        let mut in_curr_lcs = vec![false; n];
        let (mut i, mut j) = (m, n);
        while i > 0 && j > 0 {
            if prev[i - 1] == curr[j - 1] {
                in_prev_lcs[i - 1] = true;
                in_curr_lcs[j - 1] = true;
                i -= 1;
                j -= 1;
            } else if dp[i - 1][j] >= dp[i][j - 1] {
                i -= 1;
            } else {
                j -= 1;
            }
        }

        let deletions: Vec<usize> = in_prev_lcs
            .iter()
            .enumerate()
            .filter(|(_, &in_lcs)| !in_lcs)
            .map(|(idx, _)| idx)
            .collect();

        // Insertions: positions in curr that are not in LCS, with their
        // target index in the final sequence.
        let mut insertions = Vec::new();
        let mut result_pos = 0usize;
        let mut prev_lcs_iter = in_prev_lcs.iter().filter(|&&x| x);
        for (j_idx, &in_lcs) in in_curr_lcs.iter().enumerate() {
            if in_lcs {
                // This position is matched; advance result_pos.
                let _ = prev_lcs_iter.next();
                result_pos += 1;
            } else {
                insertions.push((result_pos, curr[j_idx].clone()));
                result_pos += 1;
            }
        }

        DeltaPatch { insertions, deletions }
    }

    /// Apply a [`DeltaPatch`] to `base`, producing the patched token sequence.
    pub fn apply_patch(&self, base: &[String], patch: &DeltaPatch) -> Vec<String> {
        // Remove deleted positions first.
        let del_set: std::collections::HashSet<usize> = patch.deletions.iter().cloned().collect();
        let mut result: Vec<String> = base
            .iter()
            .enumerate()
            .filter(|(i, _)| !del_set.contains(i))
            .map(|(_, s)| s.clone())
            .collect();

        // Apply insertions in order. Each insertion specifies an absolute
        // index into the *final* sequence.
        for (pos, token) in &patch.insertions {
            let idx = (*pos).min(result.len());
            result.insert(idx, token.clone());
        }

        result
    }
}

/// Convenience wrapper: encode delta.
pub fn encode_delta(prev: &[String], curr: &[String]) -> DeltaPatch {
    DeltaEncoder::new().encode_delta(prev, curr)
}

/// Convenience wrapper: apply patch.
pub fn apply_patch(base: &[String], patch: &DeltaPatch) -> Vec<String> {
    DeltaEncoder::new().apply_patch(base, patch)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|&x| x.to_string()).collect()
    }

    #[test]
    fn roundtrip_empty() {
        let tokens: Vec<String> = vec![];
        let compressed = compress(&tokens);
        let decompressed = decompress(&compressed);
        assert_eq!(tokens, decompressed);
    }

    #[test]
    fn roundtrip_no_repetition() {
        let tokens = s(&["the", "quick", "brown", "fox", "jumps"]);
        let compressed = compress(&tokens);
        let decompressed = decompress(&compressed);
        assert_eq!(tokens, decompressed);
    }

    #[test]
    fn roundtrip_repetitive() {
        // Highly repetitive: should compress well.
        let pattern = s(&["hello", "world", "foo"]);
        let tokens: Vec<String> = pattern.iter().cloned().cycle().take(30).collect();
        let compressed = compress(&tokens);
        let decompressed = decompress(&compressed);
        assert_eq!(tokens, decompressed);
    }

    #[test]
    fn compression_ratio_repetitive() {
        let pattern = s(&["alpha", "beta", "gamma"]);
        let tokens: Vec<String> = pattern.iter().cloned().cycle().take(60).collect();
        let compressed = compress(&tokens);
        let st = stats(&tokens, &compressed);
        // Repetitive input should achieve > 1.5x compression.
        assert!(
            st.ratio > 1.5,
            "expected ratio > 1.5, got {}",
            st.ratio
        );
    }

    #[test]
    fn delta_roundtrip_identical() {
        let base = s(&["a", "b", "c"]);
        let patch = encode_delta(&base, &base);
        let result = apply_patch(&base, &patch);
        assert_eq!(result, base);
    }

    #[test]
    fn delta_roundtrip_insertions() {
        let prev = s(&["a", "b", "c"]);
        let curr = s(&["a", "x", "b", "c"]);
        let patch = encode_delta(&prev, &curr);
        let result = apply_patch(&prev, &patch);
        assert_eq!(result, curr);
    }

    #[test]
    fn delta_roundtrip_deletions() {
        let prev = s(&["a", "b", "c", "d"]);
        let curr = s(&["a", "c", "d"]);
        let patch = encode_delta(&prev, &curr);
        let result = apply_patch(&prev, &patch);
        assert_eq!(result, curr);
    }

    #[test]
    fn stats_counts() {
        let tokens = s(&["a", "b", "a", "b", "a", "b", "a", "b"]);
        let compressed = compress(&tokens);
        let st = stats(&tokens, &compressed);
        assert_eq!(st.original_count, tokens.len());
        assert_eq!(st.compressed_count, compressed.len());
        assert_eq!(st.literal_count + st.reference_count, st.compressed_count);
    }
}
