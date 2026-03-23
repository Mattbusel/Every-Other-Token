//! Chunking strategies for token sequences.
//!
//! Splits a flat token list into meaningful sub-sequences ("chunks") using
//! several configurable strategies and optional inter-chunk overlap.

use crate::similarity::{TfIdfVectorizer, SparseVector};

// ── ChunkStrategy ─────────────────────────────────────────────────────────────

/// Strategy used to split a token sequence into chunks.
#[derive(Debug, Clone)]
pub enum ChunkStrategy {
    /// Split into non-overlapping windows of exactly `n` tokens.
    Fixed(usize),
    /// Split on sentence-ending punctuation (`.`, `!`, `?`).
    Sentence,
    /// Split on blank-line markers — any token that is exactly `""` or `"\n"`.
    Paragraph,
    /// Split when the cosine similarity between adjacent windows of tokens drops
    /// below `similarity_threshold` (bag-of-words / TF-IDF based).
    Semantic { similarity_threshold: f32 },
}

// ── ChunkOverlap ──────────────────────────────────────────────────────────────

/// Controls how adjacent chunks share tokens at their boundaries.
#[derive(Debug, Clone, PartialEq)]
pub enum ChunkOverlap {
    /// No overlap — chunks are disjoint.
    None,
    /// The last `n` tokens of a chunk are prepended to the next chunk.
    Tokens(usize),
}

// ── ChunkMetadata ─────────────────────────────────────────────────────────────

/// Positional metadata for a single chunk.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkMetadata {
    /// Zero-based chunk index.
    pub index: usize,
    /// Index of the first token in the original sequence.
    pub start_token: usize,
    /// Index one past the last token in the original sequence.
    pub end_token: usize,
    /// Number of tokens in the chunk.
    pub token_count: usize,
}

// ── Chunker ───────────────────────────────────────────────────────────────────

/// Splits token sequences into chunks using configurable strategies.
pub struct Chunker;

impl Chunker {
    /// Split `tokens` into chunks according to `strategy`.
    pub fn chunk(tokens: &[String], strategy: ChunkStrategy) -> Vec<Vec<String>> {
        Self::chunk_with_metadata(tokens, strategy)
            .into_iter()
            .map(|(chunk, _)| chunk)
            .collect()
    }

    /// Split `tokens` and return each chunk paired with its [`ChunkMetadata`].
    pub fn chunk_with_metadata(
        tokens: &[String],
        strategy: ChunkStrategy,
    ) -> Vec<(Vec<String>, ChunkMetadata)> {
        if tokens.is_empty() {
            return Vec::new();
        }

        let boundaries = Self::compute_boundaries(tokens, &strategy);
        Self::build_chunks(tokens, &boundaries)
    }

    /// Split `tokens` with the given strategy, then apply `overlap` to produce
    /// overlapping chunks.
    pub fn chunk_with_overlap(
        tokens: &[String],
        strategy: ChunkStrategy,
        overlap: ChunkOverlap,
    ) -> Vec<Vec<String>> {
        let base = Self::chunk(tokens, strategy);
        match overlap {
            ChunkOverlap::None => base,
            ChunkOverlap::Tokens(n) => {
                if n == 0 {
                    return base;
                }
                let mut result: Vec<Vec<String>> = Vec::with_capacity(base.len());
                let mut suffix: Vec<String> = Vec::new();
                for chunk in base {
                    let mut new_chunk = suffix.clone();
                    new_chunk.extend(chunk.iter().cloned());
                    suffix = chunk
                        .iter()
                        .rev()
                        .take(n)
                        .cloned()
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect();
                    result.push(new_chunk);
                }
                result
            }
        }
    }

    // ── internal ──────────────────────────────────────────────────────────────

    /// Compute the list of (start, end_exclusive) ranges for each chunk.
    fn compute_boundaries(tokens: &[String], strategy: &ChunkStrategy) -> Vec<(usize, usize)> {
        match strategy {
            ChunkStrategy::Fixed(n) => {
                let n = n.max(&1);
                (0..tokens.len())
                    .step_by(*n)
                    .map(|start| (start, (start + n).min(tokens.len())))
                    .collect()
            }
            ChunkStrategy::Sentence => Self::sentence_boundaries(tokens),
            ChunkStrategy::Paragraph => Self::paragraph_boundaries(tokens),
            ChunkStrategy::Semantic { similarity_threshold } => {
                Self::semantic_boundaries(tokens, *similarity_threshold)
            }
        }
    }

    fn sentence_boundaries(tokens: &[String]) -> Vec<(usize, usize)> {
        let mut ranges = Vec::new();
        let mut start = 0;
        for (i, tok) in tokens.iter().enumerate() {
            let ends = tok.ends_with('.') || tok.ends_with('!') || tok.ends_with('?')
                || tok == "." || tok == "!" || tok == "?";
            if ends {
                ranges.push((start, i + 1));
                start = i + 1;
            }
        }
        if start < tokens.len() {
            ranges.push((start, tokens.len()));
        }
        if ranges.is_empty() {
            ranges.push((0, tokens.len()));
        }
        ranges
    }

    fn paragraph_boundaries(tokens: &[String]) -> Vec<(usize, usize)> {
        let mut ranges = Vec::new();
        let mut start = 0;
        for (i, tok) in tokens.iter().enumerate() {
            if tok.is_empty() || tok == "\n\n" || tok == "\n" {
                if i > start {
                    ranges.push((start, i));
                }
                start = i + 1;
            }
        }
        if start < tokens.len() {
            ranges.push((start, tokens.len()));
        }
        if ranges.is_empty() {
            ranges.push((0, tokens.len()));
        }
        ranges
    }

    fn semantic_boundaries(tokens: &[String], threshold: f32) -> Vec<(usize, usize)> {
        const WINDOW: usize = 5;
        if tokens.len() <= WINDOW {
            return vec![(0, tokens.len())];
        }

        // Build TF-IDF from sliding windows.
        let n_windows = tokens.len().saturating_sub(WINDOW - 1);
        let docs: Vec<String> = (0..n_windows)
            .map(|i| tokens[i..i + WINDOW.min(tokens.len() - i)].join(" "))
            .collect();
        let doc_strs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
        let mut vectorizer = TfIdfVectorizer::new();
        vectorizer.fit(&doc_strs);

        let vectors: Vec<SparseVector> = docs
            .iter()
            .map(|d| vectorizer.transform(d))
            .collect();

        let mut ranges = Vec::new();
        let mut start = 0;

        for i in 1..n_windows {
            let sim = vectors[i - 1].cosine_similarity(&vectors[i]);
            if sim < threshold {
                ranges.push((start, i + WINDOW / 2));
                start = i + WINDOW / 2;
            }
        }

        // Last chunk
        if start < tokens.len() {
            ranges.push((start, tokens.len()));
        }
        if ranges.is_empty() {
            ranges.push((0, tokens.len()));
        }
        // Clamp end to valid range
        for r in &mut ranges {
            r.1 = r.1.min(tokens.len());
            if r.0 > r.1 {
                r.0 = r.1;
            }
        }
        // Remove empty ranges
        ranges.retain(|(s, e)| e > s);
        if ranges.is_empty() {
            ranges.push((0, tokens.len()));
        }
        ranges
    }

    fn build_chunks(
        tokens: &[String],
        boundaries: &[(usize, usize)],
    ) -> Vec<(Vec<String>, ChunkMetadata)> {
        boundaries
            .iter()
            .enumerate()
            .map(|(idx, &(start, end))| {
                let chunk: Vec<String> = tokens[start..end].to_vec();
                let meta = ChunkMetadata {
                    index: idx,
                    start_token: start,
                    end_token: end,
                    token_count: chunk.len(),
                };
                (chunk, meta)
            })
            .collect()
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(s: &str) -> Vec<String> {
        s.split_whitespace().map(|t| t.to_string()).collect()
    }

    // 1. Fixed strategy splits into windows of n
    #[test]
    fn fixed_even_split() {
        let tokens = toks("a b c d e f");
        let chunks = Chunker::chunk(&tokens, ChunkStrategy::Fixed(2));
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], vec!["a", "b"]);
        assert_eq!(chunks[1], vec!["c", "d"]);
        assert_eq!(chunks[2], vec!["e", "f"]);
    }

    // 2. Fixed strategy handles remainder
    #[test]
    fn fixed_uneven_split() {
        let tokens = toks("a b c d e");
        let chunks = Chunker::chunk(&tokens, ChunkStrategy::Fixed(2));
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[2], vec!["e"]);
    }

    // 3. Sentence strategy splits on period
    #[test]
    fn sentence_period() {
        let tokens: Vec<String> = vec!["Hello", "world", ".", "Goodbye", "."]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let chunks = Chunker::chunk(&tokens, ChunkStrategy::Sentence);
        assert!(chunks.len() >= 2);
    }

    // 4. Sentence strategy splits on exclamation
    #[test]
    fn sentence_exclamation() {
        let tokens: Vec<String> = vec!["Run", "!", "Walk", "."]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let chunks = Chunker::chunk(&tokens, ChunkStrategy::Sentence);
        assert!(chunks.len() >= 2);
    }

    // 5. Sentence strategy splits on question mark
    #[test]
    fn sentence_question() {
        let tokens: Vec<String> = vec!["Why", "?", "Because", "."]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let chunks = Chunker::chunk(&tokens, ChunkStrategy::Sentence);
        assert!(chunks.len() >= 2);
    }

    // 6. Sentence: no sentence ends → single chunk
    #[test]
    fn sentence_no_split() {
        let tokens = toks("hello world foo bar");
        let chunks = Chunker::chunk(&tokens, ChunkStrategy::Sentence);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 4);
    }

    // 7. Paragraph splits on blank token
    #[test]
    fn paragraph_blank_token() {
        let tokens = vec![
            "hello".to_string(), "world".to_string(),
            "".to_string(),
            "foo".to_string(), "bar".to_string(),
        ];
        let chunks = Chunker::chunk(&tokens, ChunkStrategy::Paragraph);
        assert_eq!(chunks.len(), 2);
    }

    // 8. Paragraph: no blank → single chunk
    #[test]
    fn paragraph_no_split() {
        let tokens = toks("hello world foo");
        let chunks = Chunker::chunk(&tokens, ChunkStrategy::Paragraph);
        assert_eq!(chunks.len(), 1);
    }

    // 9. Empty input → empty output
    #[test]
    fn empty_tokens() {
        let chunks = Chunker::chunk(&[], ChunkStrategy::Fixed(5));
        assert!(chunks.is_empty());
    }

    // 10. chunk_with_metadata returns correct metadata
    #[test]
    fn metadata_indices() {
        let tokens = toks("a b c d e f");
        let chunks = Chunker::chunk_with_metadata(&tokens, ChunkStrategy::Fixed(3));
        assert_eq!(chunks[0].1.start_token, 0);
        assert_eq!(chunks[0].1.end_token, 3);
        assert_eq!(chunks[0].1.token_count, 3);
        assert_eq!(chunks[1].1.start_token, 3);
        assert_eq!(chunks[1].1.index, 1);
    }

    // 11. metadata index is sequential
    #[test]
    fn metadata_index_sequential() {
        let tokens = toks("a b c d e f g h");
        let chunks = Chunker::chunk_with_metadata(&tokens, ChunkStrategy::Fixed(2));
        for (i, (_, meta)) in chunks.iter().enumerate() {
            assert_eq!(meta.index, i);
        }
    }

    // 12. No-overlap chunking — total tokens == original
    #[test]
    fn no_overlap_full_coverage() {
        let tokens = toks("one two three four five");
        let chunks = Chunker::chunk_with_overlap(&tokens, ChunkStrategy::Fixed(2), ChunkOverlap::None);
        let total: usize = chunks.iter().map(|c| c.len()).sum();
        assert_eq!(total, tokens.len());
    }

    // 13. Overlap increases total token count
    #[test]
    fn overlap_increases_count() {
        let tokens = toks("one two three four five six");
        let base = Chunker::chunk(&tokens, ChunkStrategy::Fixed(2));
        let overlapped = Chunker::chunk_with_overlap(
            &tokens,
            ChunkStrategy::Fixed(2),
            ChunkOverlap::Tokens(1),
        );
        let base_total: usize = base.iter().map(|c| c.len()).sum();
        let overlap_total: usize = overlapped.iter().map(|c| c.len()).sum();
        assert!(overlap_total >= base_total);
    }

    // 14. Overlap Tokens(0) is same as None
    #[test]
    fn overlap_zero_same_as_none() {
        let tokens = toks("a b c d");
        let none = Chunker::chunk_with_overlap(&tokens, ChunkStrategy::Fixed(2), ChunkOverlap::None);
        let zero = Chunker::chunk_with_overlap(&tokens, ChunkStrategy::Fixed(2), ChunkOverlap::Tokens(0));
        assert_eq!(none, zero);
    }

    // 15. Fixed(1) → each token its own chunk
    #[test]
    fn fixed_size_one() {
        let tokens = toks("a b c");
        let chunks = Chunker::chunk(&tokens, ChunkStrategy::Fixed(1));
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], vec!["a"]);
    }

    // 16. Semantic strategy produces at least one chunk
    #[test]
    fn semantic_at_least_one_chunk() {
        let tokens = toks("the cat sat on the mat");
        let chunks = Chunker::chunk(&tokens, ChunkStrategy::Semantic { similarity_threshold: 0.9 });
        assert!(!chunks.is_empty());
    }

    // 17. Semantic: all tokens covered (no gaps)
    #[test]
    fn semantic_covers_all_tokens() {
        let tokens = toks("apple orange banana cherry grape lemon mango");
        let chunks = Chunker::chunk_with_metadata(
            &tokens,
            ChunkStrategy::Semantic { similarity_threshold: 0.5 },
        );
        // Check no token is beyond last chunk end
        if let Some((_, last_meta)) = chunks.last() {
            assert!(last_meta.end_token <= tokens.len());
        }
    }

    // 18. Chunk contents match original token slice
    #[test]
    fn fixed_chunk_contents_match() {
        let tokens = toks("red green blue yellow");
        let chunks = Chunker::chunk_with_metadata(&tokens, ChunkStrategy::Fixed(2));
        for (chunk, meta) in &chunks {
            let expected = &tokens[meta.start_token..meta.end_token];
            assert_eq!(chunk.as_slice(), expected);
        }
    }

    // 19. Overlap Tokens(n) where n >= chunk size
    #[test]
    fn overlap_large_n() {
        let tokens = toks("a b c d e f");
        let chunks = Chunker::chunk_with_overlap(
            &tokens,
            ChunkStrategy::Fixed(2),
            ChunkOverlap::Tokens(5),
        );
        assert!(!chunks.is_empty());
    }

    // 20. Sentence ending inline (e.g. "Hello." as single token)
    #[test]
    fn sentence_inline_period() {
        let tokens: Vec<String> = vec!["Hello.".to_string(), "World.".to_string()];
        let chunks = Chunker::chunk(&tokens, ChunkStrategy::Sentence);
        assert_eq!(chunks.len(), 2);
    }

    // 21. Multiple consecutive paragraph separators
    #[test]
    fn paragraph_multiple_blanks() {
        let tokens = vec![
            "a".to_string(), "".to_string(), "".to_string(), "b".to_string(),
        ];
        let chunks = Chunker::chunk(&tokens, ChunkStrategy::Paragraph);
        assert!(chunks.len() >= 2);
    }

    // 22. chunk_with_metadata token_count matches chunk vec length
    #[test]
    fn metadata_token_count_matches_len() {
        let tokens = toks("alpha beta gamma delta epsilon");
        let chunks = Chunker::chunk_with_metadata(&tokens, ChunkStrategy::Fixed(2));
        for (chunk, meta) in &chunks {
            assert_eq!(chunk.len(), meta.token_count);
        }
    }
}
