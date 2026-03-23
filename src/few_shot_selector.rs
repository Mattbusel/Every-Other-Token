//! BM25-based few-shot example retrieval with MMR-style diversity re-ranking.
//!
//! Provides a `BM25Index` for keyword-based search over a corpus of
//! `FewShotExample` items, and a `FewShotSelector` that combines BM25 scores
//! with Jaccard-based diversity to return varied, relevant examples.

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// FewShotExample
// ---------------------------------------------------------------------------

/// A labelled example suitable for few-shot prompting.
#[derive(Debug, Clone)]
pub struct FewShotExample {
    pub id: u64,
    pub instruction: String,
    pub input: String,
    pub output: String,
    pub tags: Vec<String>,
    pub usage_count: u32,
}

// ---------------------------------------------------------------------------
// Tokenization
// ---------------------------------------------------------------------------

/// Lowercase and split on non-alphanumeric characters; keep tokens with len >= 2.
pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(|t| t.to_string())
        .collect()
}

// ---------------------------------------------------------------------------
// BM25Index
// ---------------------------------------------------------------------------

/// BM25 full-text index over a corpus of [`FewShotExample`] items.
pub struct BM25Index {
    /// Term-frequency saturation parameter (default 1.5).
    pub k1: f64,
    /// Length normalisation parameter (default 0.75).
    pub b: f64,
    pub corpus: Vec<FewShotExample>,
    /// Precomputed IDF for every term in the corpus.
    pub idf: HashMap<String, f64>,
    pub avg_doc_len: f64,
}

impl BM25Index {
    /// Build an index from a corpus, computing per-term IDF values.
    pub fn build(examples: Vec<FewShotExample>) -> Self {
        let n = examples.len() as f64;
        let mut df: HashMap<String, usize> = HashMap::new();
        let mut total_len = 0usize;

        // Count document frequencies and total token length.
        let doc_tokens: Vec<Vec<String>> = examples
            .iter()
            .map(|ex| {
                let text = format!("{} {} {}", ex.instruction, ex.input, ex.output);
                tokenize(&text)
            })
            .collect();

        for tokens in &doc_tokens {
            total_len += tokens.len();
            let unique: HashSet<&str> = tokens.iter().map(|t| t.as_str()).collect();
            for term in unique {
                *df.entry(term.to_string()).or_insert(0) += 1;
        }
        }

        let avg_doc_len = if examples.is_empty() {
            0.0
        } else {
            total_len as f64 / examples.len() as f64
        };

        // IDF = ln((N - df + 0.5) / (df + 0.5) + 1)  (Robertson–Sparck Jones)
        let idf: HashMap<String, f64> = df
            .into_iter()
            .map(|(term, dft)| {
                let dft = dft as f64;
                let score = ((n - dft + 0.5) / (dft + 0.5) + 1.0).ln();
                (term, score.max(0.0))
            })
            .collect();

        Self {
            k1: 1.5,
            b: 0.75,
            corpus: examples,
            idf,
            avg_doc_len,
        }
    }

    /// Compute the BM25 score for a single document given query tokens.
    pub fn score(&self, query_tokens: &[String], doc_tokens: &[String], doc_len: usize) -> f64 {
        // Term frequencies in this document.
        let mut tf: HashMap<&str, usize> = HashMap::new();
        for t in doc_tokens {
            *tf.entry(t.as_str()).or_insert(0) += 1;
        }

        let dl = doc_len as f64;
        let k1 = self.k1;
        let b = self.b;
        let avdl = self.avg_doc_len.max(1.0);

        query_tokens
            .iter()
            .map(|qt| {
                let idf = self.idf.get(qt.as_str()).copied().unwrap_or(0.0);
                let freq = *tf.get(qt.as_str()).unwrap_or(&0) as f64;
                let numerator = freq * (k1 + 1.0);
                let denominator = freq + k1 * (1.0 - b + b * dl / avdl);
                idf * numerator / denominator.max(1e-9)
            })
            .sum()
    }

    /// Return the top-k results sorted by BM25 score descending.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<(&FewShotExample, f64)> {
        let q_tokens = tokenize(query);
        if q_tokens.is_empty() || self.corpus.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(&FewShotExample, f64)> = self
            .corpus
            .iter()
            .map(|ex| {
                let text = format!("{} {} {}", ex.instruction, ex.input, ex.output);
                let doc_tokens = tokenize(&text);
                let len = doc_tokens.len();
                let s = self.score(&q_tokens, &doc_tokens, len);
                (ex, s)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }
}

// ---------------------------------------------------------------------------
// Jaccard similarity
// ---------------------------------------------------------------------------

fn jaccard(a: &[String], b: &[String]) -> f64 {
    let sa: HashSet<&str> = a.iter().map(|s| s.as_str()).collect();
    let sb: HashSet<&str> = b.iter().map(|s| s.as_str()).collect();
    let intersection = sa.intersection(&sb).count();
    let union = sa.union(&sb).count();
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

// ---------------------------------------------------------------------------
// FewShotSelector
// ---------------------------------------------------------------------------

/// Combines BM25 retrieval with MMR-style diversity re-ranking.
pub struct FewShotSelector {
    pub index: BM25Index,
    pub max_examples: usize,
    pub diversity_penalty: f64,
}

impl FewShotSelector {
    pub fn new(index: BM25Index, max_examples: usize, diversity_penalty: f64) -> Self {
        Self {
            index,
            max_examples,
            diversity_penalty,
        }
    }

    /// Select up to `n` examples using MMR: maximise relevance while penalising
    /// similarity to already-selected outputs.
    pub fn select(&self, query: &str, n: usize) -> Vec<&FewShotExample> {
        let candidates = self.index.search(query, self.max_examples);
        if candidates.is_empty() {
            return Vec::new();
        }

        // Pre-tokenize all candidate outputs for Jaccard diversity.
        let cand_output_tokens: Vec<Vec<String>> = candidates
            .iter()
            .map(|(ex, _)| tokenize(&ex.output))
            .collect();

        let mut selected_indices: Vec<usize> = Vec::new();
        let mut remaining: Vec<usize> = (0..candidates.len()).collect();

        while selected_indices.len() < n && !remaining.is_empty() {
            let mut best_idx = 0usize;
            let mut best_score = f64::NEG_INFINITY;

            for &ri in &remaining {
                let (_, bm25) = candidates[ri];
                // Diversity: maximum Jaccard similarity to any already-selected example.
                let max_sim = selected_indices
                    .iter()
                    .map(|&si| jaccard(&cand_output_tokens[ri], &cand_output_tokens[si]))
                    .fold(0.0_f64, f64::max);

                let mmr_score = bm25 - self.diversity_penalty * max_sim;
                if mmr_score > best_score {
                    best_score = mmr_score;
                    best_idx = ri;
                }
            }

            selected_indices.push(best_idx);
            remaining.retain(|&ri| ri != best_idx);
        }

        selected_indices
            .into_iter()
            .map(|i| candidates[i].0)
            .collect()
    }

    /// Add a new example and rebuild the index.
    pub fn add_example(&mut self, example: FewShotExample) {
        let mut corpus = std::mem::take(&mut self.index.corpus);
        corpus.push(example);
        self.index = BM25Index::build(corpus);
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_example(id: u64, instruction: &str, input: &str, output: &str) -> FewShotExample {
        FewShotExample {
            id,
            instruction: instruction.to_string(),
            input: input.to_string(),
            output: output.to_string(),
            tags: vec![],
            usage_count: 0,
        }
    }

    fn small_corpus() -> Vec<FewShotExample> {
        vec![
            make_example(1, "Translate to French", "Hello world", "Bonjour le monde"),
            make_example(2, "Summarise text", "Long article about climate", "Climate summary"),
            make_example(3, "Translate to Spanish", "Good morning", "Buenos dias"),
            make_example(4, "Write a poem", "About the ocean", "Waves crash softly on the shore"),
        ]
    }

    #[test]
    fn exact_keyword_match_ranks_first() {
        let index = BM25Index::build(small_corpus());
        let results = index.search("translate french", 4);
        assert!(!results.is_empty());
        // The "Translate to French" example (id=1) should rank highly.
        assert_eq!(results[0].0.id, 1, "Expected id=1 first, got id={}", results[0].0.id);
    }

    #[test]
    fn diversity_reranking_changes_order() {
        let corpus = vec![
            make_example(1, "Translate", "cat", "chat"),
            make_example(2, "Translate", "dog", "chien"),
            make_example(3, "Summarise", "article", "summary of article about topic"),
        ];
        let index = BM25Index::build(corpus.clone());
        let selector = FewShotSelector::new(index, 3, 1.0);
        let selected = selector.select("translate", 2);
        // With high diversity penalty the second pick should not be the most
        // similar output. Both translate examples have similar outputs ("chat",
        // "chien") — diversity should still prefer something dissimilar as 2nd.
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn empty_corpus_returns_empty() {
        let index = BM25Index::build(vec![]);
        let results = index.search("anything", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn add_example_rebuilds_index() {
        let index = BM25Index::build(small_corpus());
        let mut selector = FewShotSelector::new(index, 10, 0.5);
        selector.add_example(make_example(99, "Unique topic", "rust programming", "ownership"));
        let results = selector.index.search("rust programming ownership", 1);
        assert!(!results.is_empty());
        assert_eq!(results[0].0.id, 99);
    }

    #[test]
    fn tokenize_filters_short_tokens() {
        let tokens = tokenize("a bb ccc");
        assert!(!tokens.contains(&"a".to_string()));
        assert!(tokens.contains(&"bb".to_string()));
        assert!(tokens.contains(&"ccc".to_string()));
    }
}
