//! # BPE Tokenizer
//!
//! Simple Byte-Pair Encoding tokenizer with training, encode, and decode.

use std::collections::HashMap;

// ── Type aliases ─────────────────────────────────────────────────────────────

/// Convenience alias for a pair of token strings.
pub type TokenPair = (String, String);

// ── BpeMerge ─────────────────────────────────────────────────────────────────

/// A single BPE merge rule: two adjacent tokens that were merged into one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BpeMerge {
    /// The pair of tokens that were merged.
    pub pair: (String, String),
    /// The resulting merged token.
    pub merged: String,
    /// Rank of this merge (lower = earlier / higher priority).
    pub rank: usize,
}

// ── BpeTokenizer ─────────────────────────────────────────────────────────────

/// Byte-Pair Encoding tokenizer.
///
/// Train with [`BpeTokenizer::train`], then call [`encode`](Self::encode) /
/// [`decode`](Self::decode).
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    /// Ordered list of merge rules (applied in rank order during encoding).
    pub merges: Vec<BpeMerge>,
    /// Vocabulary mapping token → id.
    pub vocab: HashMap<String, usize>,
    /// Whether to use byte-level initialisation (future extension; currently
    /// always character-level).
    pub byte_vocab: bool,
}

impl BpeTokenizer {
    // ── Training ─────────────────────────────────────────────────────────────

    /// Train a BPE tokenizer on `text` until the vocabulary reaches
    /// `vocab_size` or no more pairs can be merged.
    pub fn train(text: &str, vocab_size: usize) -> Self {
        // Step 1: initialise vocabulary with individual characters.
        let mut vocab: HashMap<String, usize> = HashMap::new();

        // Split text into words separated by whitespace, add a space marker
        // to every word except the first so decode can reconstruct spaces.
        let words: Vec<Vec<String>> = text
            .split_whitespace()
            .enumerate()
            .map(|(i, word)| {
                let prefix = if i == 0 { "" } else { "Ġ" };
                let mut chars: Vec<String> = word
                    .chars()
                    .map(|c| c.to_string())
                    .collect();
                if !chars.is_empty() && !prefix.is_empty() {
                    chars[0] = format!("{}{}", prefix, chars[0]);
                }
                chars
            })
            .collect();

        // Populate initial vocab from characters present in the corpus.
        let mut next_id = 0usize;
        for word in &words {
            for ch in word {
                vocab.entry(ch.clone()).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                });
            }
        }

        let mut corpus: Vec<Vec<String>> = words;
        let mut merges: Vec<BpeMerge> = Vec::new();
        let mut rank = 0usize;

        // Step 2 & 3: greedily merge most-frequent adjacent pairs.
        while vocab.len() < vocab_size {
            let freqs = pair_frequencies(&corpus);
            if freqs.is_empty() {
                break;
            }

            // Find the most-frequent pair (tie-break: lexicographic order).
            let best = freqs
                .iter()
                .max_by(|a, b| {
                    a.1.cmp(b.1)
                        .then_with(|| a.0.cmp(b.0).reverse())
                })
                .map(|(pair, _)| pair.clone());

            let Some((left, right)) = best else { break };

            let merged = format!("{}{}", left, right);
            let merge = BpeMerge {
                pair: (left.clone(), right.clone()),
                merged: merged.clone(),
                rank,
            };
            rank += 1;

            // Add merged token to vocab.
            let id = next_id;
            next_id += 1;
            vocab.insert(merged.clone(), id);

            // Apply the merge throughout the corpus.
            apply_merge(&mut corpus, &merge);
            merges.push(merge);
        }

        Self {
            merges,
            vocab,
            byte_vocab: false,
        }
    }

    // ── Encode ───────────────────────────────────────────────────────────────

    /// Encode `text` into a sequence of token strings using the learned merges.
    pub fn encode(&self, text: &str) -> Vec<String> {
        // Start from character split (with Ġ space markers).
        let mut corpus: Vec<Vec<String>> = text
            .split_whitespace()
            .enumerate()
            .map(|(i, word)| {
                let prefix = if i == 0 { "" } else { "Ġ" };
                let mut chars: Vec<String> = word
                    .chars()
                    .map(|c| c.to_string())
                    .collect();
                if !chars.is_empty() && !prefix.is_empty() {
                    chars[0] = format!("{}{}", prefix, chars[0]);
                }
                chars
            })
            .collect();

        // Apply each merge in rank order.
        for merge in &self.merges {
            apply_merge(&mut corpus, merge);
        }

        corpus.into_iter().flatten().collect()
    }

    // ── Decode ───────────────────────────────────────────────────────────────

    /// Decode a token sequence back to a string.
    ///
    /// The `Ġ` space marker (used internally to represent word boundaries) is
    /// replaced with a space character.
    pub fn decode(&self, tokens: &[String]) -> String {
        tokens
            .iter()
            .map(|t| t.replace('Ġ', " "))
            .collect::<String>()
            .trim_start()
            .to_string()
    }

    /// Number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

// ── Pair frequency counting ───────────────────────────────────────────────────

/// Count adjacent token-pair frequencies across all sequences in `corpus`.
pub fn pair_frequencies(corpus: &[Vec<String>]) -> HashMap<(String, String), usize> {
    let mut counts: HashMap<(String, String), usize> = HashMap::new();
    for seq in corpus {
        for window in seq.windows(2) {
            let pair = (window[0].clone(), window[1].clone());
            *counts.entry(pair).or_insert(0) += 1;
        }
    }
    counts
}

// ── Merge application ─────────────────────────────────────────────────────────

/// Replace every occurrence of `merge.pair` with `merge.merged` in `corpus`.
pub fn apply_merge(corpus: &mut Vec<Vec<String>>, merge: &BpeMerge) {
    for seq in corpus.iter_mut() {
        let mut new_seq: Vec<String> = Vec::with_capacity(seq.len());
        let mut i = 0;
        while i < seq.len() {
            if i + 1 < seq.len()
                && seq[i] == merge.pair.0
                && seq[i + 1] == merge.pair.1
            {
                new_seq.push(merge.merged.clone());
                i += 2;
            } else {
                new_seq.push(seq[i].clone());
                i += 1;
            }
        }
        *seq = new_seq;
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_produces_merges() {
        let text = "hello hello hello world world";
        let tok = BpeTokenizer::train(text, 30);
        // Should have merged some pairs — initial chars < final vocab.
        assert!(
            !tok.merges.is_empty(),
            "expected at least one merge from repeated text"
        );
    }

    #[test]
    fn vocab_grows_with_training() {
        let small = BpeTokenizer::train("ab ab", 10);
        let large = BpeTokenizer::train("ab ab", 20);
        // Larger target → at least as many merges tried.
        assert!(large.vocab_size() >= small.vocab_size());
    }

    #[test]
    fn encode_decode_roundtrip() {
        let text = "hello world";
        let tok = BpeTokenizer::train(text, 20);
        let tokens = tok.encode(text);
        let decoded = tok.decode(&tokens);
        assert_eq!(decoded, text);
    }

    #[test]
    fn encode_returns_nonempty() {
        let tok = BpeTokenizer::train("the cat sat on the mat", 30);
        let tokens = tok.encode("the cat");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn pair_frequencies_counts_correctly() {
        let corpus = vec![
            vec!["a".to_string(), "b".to_string(), "a".to_string()],
        ];
        let freqs = pair_frequencies(&corpus);
        assert_eq!(freqs[&("a".to_string(), "b".to_string())], 1);
        assert_eq!(freqs[&("b".to_string(), "a".to_string())], 1);
    }

    #[test]
    fn apply_merge_replaces_all_occurrences() {
        let merge = BpeMerge {
            pair: ("a".to_string(), "b".to_string()),
            merged: "ab".to_string(),
            rank: 0,
        };
        let mut corpus = vec![vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "a".to_string(),
            "b".to_string(),
        ]];
        apply_merge(&mut corpus, &merge);
        assert_eq!(corpus[0], vec!["ab", "c", "ab"]);
    }

    #[test]
    fn decode_handles_space_markers() {
        // Manually construct tokens with space markers.
        let tokens = vec!["hello".to_string(), "Ġworld".to_string()];
        let tok = BpeTokenizer {
            merges: vec![],
            vocab: HashMap::new(),
            byte_vocab: false,
        };
        assert_eq!(tok.decode(&tokens), "hello world");
    }
}
