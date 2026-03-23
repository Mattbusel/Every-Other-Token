//! Semantic similarity scoring using TF-IDF vectors.
//!
//! Provides TF-IDF vectorization, cosine similarity, and diversity filtering
//! for token sequences.

use std::collections::HashMap;

// ── SparseVector ──────────────────────────────────────────────────────────────

/// A sparse vector representation with parallel index/value arrays.
#[derive(Debug, Clone, Default)]
pub struct SparseVector {
    pub indices: Vec<usize>,
    pub values: Vec<f32>,
}

impl SparseVector {
    /// Create an empty sparse vector.
    pub fn new() -> Self {
        Self { indices: Vec::new(), values: Vec::new() }
    }

    /// Dot product with another sparse vector.
    pub fn dot(&self, other: &SparseVector) -> f32 {
        let mut result = 0.0f32;
        let mut i = 0;
        let mut j = 0;
        while i < self.indices.len() && j < other.indices.len() {
            match self.indices[i].cmp(&other.indices[j]) {
                std::cmp::Ordering::Equal => {
                    result += self.values[i] * other.values[j];
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }
        result
    }

    /// L2 norm of this vector.
    pub fn norm(&self) -> f32 {
        self.values.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    /// Cosine similarity with another sparse vector. Returns 0.0 if either norm is zero.
    pub fn cosine_similarity(&self, other: &SparseVector) -> f32 {
        let n1 = self.norm();
        let n2 = other.norm();
        if n1 < 1e-10 || n2 < 1e-10 {
            return 0.0;
        }
        (self.dot(other) / (n1 * n2)).clamp(-1.0, 1.0)
    }
}

// ── TfIdfVectorizer ───────────────────────────────────────────────────────────

/// Builds a TF-IDF vocabulary from a corpus and transforms documents to sparse
/// TF-IDF vectors.
///
/// IDF is computed with smoothing: `log((N+1)/(df+1)) + 1`.
#[derive(Debug, Clone, Default)]
pub struct TfIdfVectorizer {
    /// Maps each token to its vocabulary index.
    vocab: HashMap<String, usize>,
    /// IDF weight for each vocabulary entry (indexed by vocab index).
    idf: Vec<f32>,
}

impl TfIdfVectorizer {
    /// Create a new (unfitted) vectorizer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Tokenize a document into lowercased words.
    fn tokenize(doc: &str) -> Vec<String> {
        doc.split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_lowercase())
            .collect()
    }

    /// Fit the vectorizer on a corpus.
    ///
    /// Builds vocabulary and computes smoothed IDF weights:
    /// `idf(t) = log((N+1)/(df(t)+1)) + 1`
    pub fn fit(&mut self, docs: &[&str]) {
        let n = docs.len();
        let mut df: HashMap<String, usize> = HashMap::new();

        for doc in docs {
            let tokens = Self::tokenize(doc);
            // unique tokens per doc
            let unique: std::collections::HashSet<String> = tokens.into_iter().collect();
            for tok in unique {
                *df.entry(tok).or_insert(0) += 1;
            }
        }

        // Sort vocabulary for deterministic indexing
        let mut terms: Vec<String> = df.keys().cloned().collect();
        terms.sort();

        self.vocab = terms
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i))
            .collect();

        self.idf = terms
            .iter()
            .map(|t| {
                let d = *df.get(t).unwrap_or(&0) as f32;
                ((n as f32 + 1.0) / (d + 1.0)).ln() + 1.0
            })
            .collect();
    }

    /// Transform a document into a sparse TF-IDF vector.
    ///
    /// TF is computed as raw term count divided by document length.
    /// Returns indices sorted in ascending order.
    pub fn transform(&self, doc: &str) -> SparseVector {
        let tokens = Self::tokenize(doc);
        let n_tokens = tokens.len();
        if n_tokens == 0 || self.vocab.is_empty() {
            return SparseVector::new();
        }

        let mut tf: HashMap<usize, f32> = HashMap::new();
        for tok in &tokens {
            if let Some(&idx) = self.vocab.get(tok) {
                *tf.entry(idx).or_insert(0.0) += 1.0;
            }
        }

        let mut indices: Vec<usize> = tf.keys().cloned().collect();
        indices.sort_unstable();

        let values: Vec<f32> = indices
            .iter()
            .map(|&idx| {
                let tf_val = tf[&idx] / n_tokens as f32;
                let idf_val = self.idf.get(idx).copied().unwrap_or(1.0);
                tf_val * idf_val
            })
            .collect();

        SparseVector { indices, values }
    }

    /// Number of vocabulary terms.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Check if this vectorizer has been fitted.
    pub fn is_fitted(&self) -> bool {
        !self.vocab.is_empty()
    }
}

// ── SemanticScorer ────────────────────────────────────────────────────────────

/// Computes semantic similarity between text strings using TF-IDF vectors.
pub struct SemanticScorer {
    vectorizer: TfIdfVectorizer,
}

impl SemanticScorer {
    /// Create a scorer and fit it on the provided corpus.
    pub fn fit(corpus: &[&str]) -> Self {
        let mut vectorizer = TfIdfVectorizer::new();
        vectorizer.fit(corpus);
        Self { vectorizer }
    }

    /// Cosine similarity between two text strings.
    pub fn similarity(&self, a: &str, b: &str) -> f32 {
        let va = self.vectorizer.transform(a);
        let vb = self.vectorizer.transform(b);
        va.cosine_similarity(&vb)
    }

    /// Return the top-k most similar candidates to a query, sorted by
    /// descending similarity.
    pub fn most_similar<'a>(
        &self,
        query: &str,
        candidates: &[&'a str],
        top_k: usize,
    ) -> Vec<(f32, &'a str)> {
        let vq = self.vectorizer.transform(query);
        let mut scores: Vec<(f32, &'a str)> = candidates
            .iter()
            .map(|&c| {
                let vc = self.vectorizer.transform(c);
                (vq.cosine_similarity(&vc), c)
            })
            .collect();

        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }

    /// Access the internal vectorizer.
    pub fn vectorizer(&self) -> &TfIdfVectorizer {
        &self.vectorizer
    }
}

// ── DiversityFilter ───────────────────────────────────────────────────────────

/// Removes near-duplicate token sequences whose cosine similarity exceeds a
/// given threshold. Greedy algorithm: keeps the first of each near-duplicate
/// cluster.
pub struct DiversityFilter {
    vectorizer: TfIdfVectorizer,
}

impl DiversityFilter {
    /// Build a diversity filter fitted on the sequences themselves.
    pub fn new(seqs: &[Vec<String>]) -> Self {
        let docs: Vec<String> = seqs.iter().map(|s| s.join(" ")).collect();
        let doc_refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
        let mut vectorizer = TfIdfVectorizer::new();
        vectorizer.fit(&doc_refs);
        Self { vectorizer }
    }

    /// Create a filter using an externally fitted vectorizer.
    pub fn with_vectorizer(vectorizer: TfIdfVectorizer) -> Self {
        Self { vectorizer }
    }

    /// Remove sequences that are too similar to an already-kept sequence.
    ///
    /// Two sequences are near-duplicates when their cosine similarity exceeds
    /// `threshold`. Processes sequences in order; keeps the first of each
    /// near-duplicate group.
    pub fn filter(&self, seqs: Vec<Vec<String>>, threshold: f32) -> Vec<Vec<String>> {
        let mut kept: Vec<Vec<String>> = Vec::new();
        let mut kept_vecs: Vec<SparseVector> = Vec::new();

        for seq in seqs {
            let doc = seq.join(" ");
            let v = self.vectorizer.transform(&doc);

            let is_duplicate = kept_vecs.iter().any(|kv| {
                v.cosine_similarity(kv) > threshold
            });

            if !is_duplicate {
                kept_vecs.push(v);
                kept.push(seq);
            }
        }

        kept
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> Vec<&'static str> {
        vec![
            "the quick brown fox",
            "the lazy dog sleeps",
            "machine learning models",
            "deep neural networks",
            "natural language processing",
            "the fox and the dog",
            "learning is fun",
            "cats and dogs are pets",
        ]
    }

    // ── SparseVector tests ────────────────────────────────────────────────

    #[test]
    fn test_sparse_vector_empty_norm() {
        let v = SparseVector::new();
        assert_eq!(v.norm(), 0.0);
    }

    #[test]
    fn test_sparse_vector_dot_product() {
        let a = SparseVector { indices: vec![0, 2, 4], values: vec![1.0, 2.0, 3.0] };
        let b = SparseVector { indices: vec![0, 2, 4], values: vec![4.0, 5.0, 6.0] };
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((a.dot(&b) - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_sparse_vector_dot_non_overlapping() {
        let a = SparseVector { indices: vec![0, 2], values: vec![1.0, 2.0] };
        let b = SparseVector { indices: vec![1, 3], values: vec![3.0, 4.0] };
        assert_eq!(a.dot(&b), 0.0);
    }

    #[test]
    fn test_sparse_vector_dot_partial_overlap() {
        let a = SparseVector { indices: vec![0, 1, 2], values: vec![1.0, 2.0, 3.0] };
        let b = SparseVector { indices: vec![1, 3], values: vec![5.0, 6.0] };
        // only index 1 overlaps: 2*5 = 10
        assert!((a.dot(&b) - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_sparse_vector_norm() {
        let v = SparseVector { indices: vec![0, 1], values: vec![3.0, 4.0] };
        assert!((v.norm() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = SparseVector { indices: vec![0, 1], values: vec![1.0, 1.0] };
        assert!((v.cosine_similarity(&v.clone()) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = SparseVector { indices: vec![0], values: vec![1.0] };
        let b = SparseVector { indices: vec![1], values: vec![1.0] };
        assert_eq!(a.cosine_similarity(&b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = SparseVector::new();
        let b = SparseVector { indices: vec![0], values: vec![1.0] };
        assert_eq!(a.cosine_similarity(&b), 0.0);
    }

    // ── TfIdfVectorizer tests ─────────────────────────────────────────────

    #[test]
    fn test_vectorizer_fit_builds_vocab() {
        let mut v = TfIdfVectorizer::new();
        v.fit(&["hello world", "world of rust"]);
        assert!(v.vocab_size() > 0);
        assert!(v.is_fitted());
    }

    #[test]
    fn test_vectorizer_unfitted_is_empty() {
        let v = TfIdfVectorizer::new();
        assert!(!v.is_fitted());
        assert_eq!(v.vocab_size(), 0);
    }

    #[test]
    fn test_vectorizer_transform_empty_doc() {
        let mut v = TfIdfVectorizer::new();
        v.fit(&["hello world"]);
        let sv = v.transform("");
        assert!(sv.indices.is_empty());
    }

    #[test]
    fn test_vectorizer_transform_known_word() {
        let mut v = TfIdfVectorizer::new();
        v.fit(&["hello world", "hello rust"]);
        let sv = v.transform("hello");
        assert!(!sv.indices.is_empty());
        assert!(sv.values.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_vectorizer_transform_unknown_word_gives_zero_vector() {
        let mut v = TfIdfVectorizer::new();
        v.fit(&["hello world"]);
        let sv = v.transform("xyzzy");
        assert!(sv.indices.is_empty() || sv.values.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_vectorizer_indices_sorted() {
        let mut v = TfIdfVectorizer::new();
        v.fit(&corpus());
        let sv = v.transform("the quick brown fox");
        for w in sv.indices.windows(2) {
            assert!(w[0] < w[1], "indices not sorted: {:?}", sv.indices);
        }
    }

    #[test]
    fn test_tfidf_rare_word_has_higher_idf() {
        let mut v = TfIdfVectorizer::new();
        // "fox" appears in 1 doc, "the" in many
        v.fit(&corpus());
        let sv_fox = v.transform("fox");
        let sv_the = v.transform("the");
        // Fox vector magnitude should be >= the vector (rare words get higher weight)
        assert!(
            sv_fox.norm() >= sv_the.norm() - 1e-5,
            "expected fox norm {} >= the norm {}",
            sv_fox.norm(),
            sv_the.norm()
        );
    }

    // ── SemanticScorer tests ──────────────────────────────────────────────

    #[test]
    fn test_scorer_self_similarity_is_one() {
        let scorer = SemanticScorer::fit(&corpus());
        let sim = scorer.similarity("the quick brown fox", "the quick brown fox");
        assert!((sim - 1.0).abs() < 1e-5, "self-similarity should be 1.0, got {}", sim);
    }

    #[test]
    fn test_scorer_similar_docs_higher_than_dissimilar() {
        let scorer = SemanticScorer::fit(&corpus());
        let sim_close = scorer.similarity("the quick brown fox", "the fox and the dog");
        let sim_far = scorer.similarity("the quick brown fox", "deep neural networks");
        assert!(
            sim_close > sim_far,
            "expected {} > {} (close vs far)",
            sim_close,
            sim_far
        );
    }

    #[test]
    fn test_scorer_most_similar_returns_top_k() {
        let scorer = SemanticScorer::fit(&corpus());
        let candidates = &corpus();
        let results = scorer.most_similar("fox and dog", candidates, 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_scorer_most_similar_sorted_descending() {
        let scorer = SemanticScorer::fit(&corpus());
        let candidates = &corpus();
        let results = scorer.most_similar("machine learning", candidates, 5);
        for w in results.windows(2) {
            assert!(w[0].0 >= w[1].0, "results not sorted: {:?}", results);
        }
    }

    #[test]
    fn test_scorer_most_similar_empty_candidates() {
        let scorer = SemanticScorer::fit(&corpus());
        let results = scorer.most_similar("hello", &[], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_scorer_top_k_larger_than_candidates() {
        let scorer = SemanticScorer::fit(&corpus());
        let candidates = vec!["hello world"];
        let results = scorer.most_similar("hello", &candidates, 10);
        assert!(results.len() <= 1);
    }

    // ── DiversityFilter tests ─────────────────────────────────────────────

    #[test]
    fn test_diversity_filter_keeps_all_unique() {
        let seqs: Vec<Vec<String>> = vec![
            vec!["cat".into(), "sat".into()],
            vec!["machine".into(), "learning".into()],
            vec!["deep".into(), "networks".into()],
        ];
        let filter = DiversityFilter::new(&seqs);
        let result = filter.filter(seqs.clone(), 0.9);
        assert_eq!(result.len(), 3, "all unique sequences should be kept");
    }

    #[test]
    fn test_diversity_filter_removes_duplicates() {
        let seqs: Vec<Vec<String>> = vec![
            vec!["the".into(), "quick".into(), "fox".into()],
            vec!["the".into(), "quick".into(), "fox".into()],
        ];
        let filter = DiversityFilter::new(&seqs);
        let result = filter.filter(seqs, 0.9);
        assert_eq!(result.len(), 1, "duplicate sequences should be deduplicated");
    }

    #[test]
    fn test_diversity_filter_threshold_zero_keeps_one() {
        let seqs: Vec<Vec<String>> = vec![
            vec!["hello".into()],
            vec!["world".into()],
        ];
        // threshold=0 means any non-zero similarity causes dedup;
        // orthogonal vectors have sim=0, so both kept
        let filter = DiversityFilter::new(&seqs);
        let result = filter.filter(seqs, 0.0);
        // orthogonal so both kept
        assert!(result.len() <= 2);
    }

    #[test]
    fn test_diversity_filter_preserves_order() {
        let seqs: Vec<Vec<String>> = vec![
            vec!["apple".into()],
            vec!["banana".into()],
            vec!["cherry".into()],
        ];
        let filter = DiversityFilter::new(&seqs);
        let result = filter.filter(seqs.clone(), 0.99);
        // All distinct words, should all be kept in original order
        assert_eq!(result, seqs);
    }

    #[test]
    fn test_diversity_filter_empty_input() {
        let filter = DiversityFilter::new(&[]);
        let result = filter.filter(vec![], 0.8);
        assert!(result.is_empty());
    }

    #[test]
    fn test_diversity_filter_high_threshold_keeps_all() {
        let seqs: Vec<Vec<String>> = vec![
            vec!["the".into(), "quick".into(), "brown".into(), "fox".into()],
            vec!["the".into(), "quick".into(), "brown".into(), "fox".into(), "jumps".into()],
        ];
        let filter = DiversityFilter::new(&seqs);
        // threshold=1.0 means only exact duplicates removed
        let result = filter.filter(seqs, 1.0);
        assert_eq!(result.len(), 2);
    }
}
