//! Semantic similarity computations without external ML dependencies.
//!
//! Provides token-set operations (Jaccard, Overlap), TF-IDF cosine similarity,
//! BM25 scoring, and a convenience [`SimilarityEngine`] that keeps a small
//! document corpus and computes IDF weights.

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// TokenSet
// ---------------------------------------------------------------------------

/// Unigram and bigram token set built from raw text.
#[derive(Debug, Clone, Default)]
pub struct TokenSet {
    pub tokens: HashSet<String>,
    pub bigrams: HashSet<(String, String)>,
}

impl TokenSet {
    /// Build from text: lowercase, strip punctuation, extract unigrams + bigrams.
    pub fn from_text(text: &str) -> Self {
        let words: Vec<String> = text
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '\'' { c } else { ' ' })
            .collect::<String>()
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .filter(|w| !w.is_empty())
            .collect();

        let tokens: HashSet<String> = words.iter().cloned().collect();
        let bigrams: HashSet<(String, String)> = words
            .windows(2)
            .map(|w| (w[0].clone(), w[1].clone()))
            .collect();

        Self { tokens, bigrams }
    }
}

// ---------------------------------------------------------------------------
// SimilarityMethod
// ---------------------------------------------------------------------------

/// Which similarity metric to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimilarityMethod {
    Jaccard,
    Overlap,
    Cosine,
    BM25,
    /// Arithmetic mean of Jaccard, Cosine, and BM25.
    Combined,
}

// ---------------------------------------------------------------------------
// TfIdfVector
// ---------------------------------------------------------------------------

/// A TF-IDF weighted term vector.
#[derive(Debug, Clone, Default)]
pub struct TfIdfVector {
    pub term_freq: HashMap<String, f64>,
    pub magnitude: f64,
}

impl TfIdfVector {
    fn from_map(map: HashMap<String, f64>) -> Self {
        let magnitude = map.values().map(|v| v * v).sum::<f64>().sqrt();
        Self {
            term_freq: map,
            magnitude,
        }
    }
}

// ---------------------------------------------------------------------------
// SimilarityEngine
// ---------------------------------------------------------------------------

/// Maintains a document corpus and associated IDF weights.
pub struct SimilarityEngine {
    corpus: Vec<String>,
    idf: HashMap<String, f64>,
}

impl SimilarityEngine {
    /// Create an empty engine.
    pub fn new() -> Self {
        Self {
            corpus: Vec::new(),
            idf: HashMap::new(),
        }
    }

    /// Add a document to the corpus and recompute IDF.
    pub fn add_document(&mut self, doc: &str) {
        self.corpus.push(doc.to_string());
        self.compute_idf();
    }

    /// Recompute ln(N / df) IDF scores for all terms in the corpus.
    pub fn compute_idf(&mut self) {
        let n = self.corpus.len() as f64;
        if n == 0.0 {
            return;
        }

        let mut df: HashMap<String, usize> = HashMap::new();
        for doc in &self.corpus {
            let ts = TokenSet::from_text(doc);
            for token in ts.tokens {
                *df.entry(token).or_insert(0) += 1;
            }
        }

        self.idf = df
            .into_iter()
            .map(|(term, count)| (term, (n / count as f64).ln()))
            .collect();
    }

    /// Compute TF-IDF vector for arbitrary text against the stored IDF.
    pub fn tfidf_vector(&self, text: &str) -> HashMap<String, f64> {
        let ts = TokenSet::from_text(text);
        let total = ts.tokens.len() as f64;
        if total == 0.0 {
            return HashMap::new();
        }

        // Count raw TF.
        let mut tf: HashMap<String, usize> = HashMap::new();
        let words: Vec<String> = text
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();
        for w in &words {
            *tf.entry(w.clone()).or_insert(0) += 1;
        }

        tf.into_iter()
            .filter_map(|(term, count)| {
                let idf = self.idf.get(&term).copied().unwrap_or(0.0);
                let score = (count as f64 / total) * idf;
                if score > 0.0 {
                    Some((term, score))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Cosine similarity between two TF-IDF maps.
    pub fn cosine_similarity(
        a: &HashMap<String, f64>,
        b: &HashMap<String, f64>,
    ) -> f64 {
        let dot: f64 = a
            .iter()
            .filter_map(|(k, va)| b.get(k).map(|vb| va * vb))
            .sum();
        let mag_a: f64 = a.values().map(|v| v * v).sum::<f64>().sqrt();
        let mag_b: f64 = b.values().map(|v| v * v).sum::<f64>().sqrt();
        if mag_a == 0.0 || mag_b == 0.0 {
            0.0
        } else {
            (dot / (mag_a * mag_b)).clamp(0.0, 1.0)
        }
    }

    /// Jaccard similarity: |A ∩ B| / |A ∪ B|.
    pub fn jaccard_similarity(a: &str, b: &str) -> f64 {
        let sa = TokenSet::from_text(a);
        let sb = TokenSet::from_text(b);
        let intersection = sa.tokens.intersection(&sb.tokens).count();
        let union = sa.tokens.union(&sb.tokens).count();
        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// Overlap coefficient: |A ∩ B| / min(|A|, |B|).
    pub fn overlap_coefficient(a: &str, b: &str) -> f64 {
        let sa = TokenSet::from_text(a);
        let sb = TokenSet::from_text(b);
        let intersection = sa.tokens.intersection(&sb.tokens).count();
        let min_size = sa.tokens.len().min(sb.tokens.len());
        if min_size == 0 {
            0.0
        } else {
            intersection as f64 / min_size as f64
        }
    }

    /// BM25 score of `query` against `document`.
    ///
    /// Uses standard parameters k1 and b.
    pub fn bm25_score(query: &str, document: &str, k1: f64, b: f64) -> f64 {
        let query_terms = TokenSet::from_text(query);
        let doc_words: Vec<String> = document
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();
        let doc_len = doc_words.len() as f64;
        // Average document length assumption: 100 tokens.
        let avg_dl = 100.0_f64;

        let mut tf_map: HashMap<String, usize> = HashMap::new();
        for w in &doc_words {
            *tf_map.entry(w.clone()).or_insert(0) += 1;
        }

        let mut score = 0.0_f64;
        for term in &query_terms.tokens {
            let tf = *tf_map.get(term).unwrap_or(&0) as f64;
            if tf == 0.0 {
                continue;
            }
            let numerator = tf * (k1 + 1.0);
            let denominator = tf + k1 * (1.0 - b + b * doc_len / avg_dl);
            score += numerator / denominator;
        }
        score
    }

    /// Compute similarity between two strings using the specified method.
    pub fn similarity(&self, a: &str, b: &str, method: SimilarityMethod) -> f64 {
        match method {
            SimilarityMethod::Jaccard => Self::jaccard_similarity(a, b),
            SimilarityMethod::Overlap => Self::overlap_coefficient(a, b),
            SimilarityMethod::Cosine => {
                let va = self.tfidf_vector(a);
                let vb = self.tfidf_vector(b);
                Self::cosine_similarity(&va, &vb)
            }
            SimilarityMethod::BM25 => {
                // Normalise BM25 to [0, 1] using a simple logistic transform.
                let raw = Self::bm25_score(a, b, 1.5, 0.75);
                1.0 / (1.0 + (-raw * 0.1).exp())
            }
            SimilarityMethod::Combined => {
                let j = Self::jaccard_similarity(a, b);
                let va = self.tfidf_vector(a);
                let vb = self.tfidf_vector(b);
                let c = Self::cosine_similarity(&va, &vb);
                let raw_bm25 = Self::bm25_score(a, b, 1.5, 0.75);
                let bm25_norm = 1.0 / (1.0 + (-raw_bm25 * 0.1).exp());
                (j + c + bm25_norm) / 3.0
            }
        }
    }

    /// Find all pairs of texts whose similarity exceeds `threshold`.
    pub fn find_duplicates(texts: &[String], threshold: f64) -> Vec<(usize, usize)> {
        let mut engine = SimilarityEngine::new();
        for t in texts {
            engine.add_document(t);
        }
        let mut pairs = Vec::new();
        for i in 0..texts.len() {
            for j in (i + 1)..texts.len() {
                let sim = engine.similarity(&texts[i], &texts[j], SimilarityMethod::Combined);
                if sim >= threshold {
                    pairs.push((i, j));
                }
            }
        }
        pairs
    }

    /// Rank corpus texts by similarity to `query`, descending.
    pub fn rank_by_similarity(
        query: &str,
        corpus: &[String],
        method: SimilarityMethod,
    ) -> Vec<(usize, f64)> {
        let mut engine = SimilarityEngine::new();
        for doc in corpus {
            engine.add_document(doc);
        }

        let mut ranked: Vec<(usize, f64)> = corpus
            .iter()
            .enumerate()
            .map(|(i, doc)| (i, engine.similarity(query, doc, method)))
            .collect();

        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }
}

impl Default for SimilarityEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_set_from_text() {
        let ts = TokenSet::from_text("Hello, World! Hello.");
        assert!(ts.tokens.contains("hello"));
        assert!(ts.tokens.contains("world"));
    }

    #[test]
    fn jaccard_identical() {
        let s = SimilarityEngine::jaccard_similarity("the cat sat", "the cat sat");
        assert!((s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn jaccard_disjoint() {
        let s = SimilarityEngine::jaccard_similarity("alpha beta", "gamma delta");
        assert_eq!(s, 0.0);
    }

    #[test]
    fn overlap_coefficient_subset() {
        let s = SimilarityEngine::overlap_coefficient("cat sat", "the cat sat on the mat");
        assert!(s > 0.0);
    }

    #[test]
    fn bm25_positive() {
        let score = SimilarityEngine::bm25_score("cat sat", "the cat sat on the mat", 1.5, 0.75);
        assert!(score > 0.0);
    }

    #[test]
    fn cosine_similarity_self() {
        let mut engine = SimilarityEngine::new();
        engine.add_document("rust programming language");
        let v = engine.tfidf_vector("rust programming language");
        let sim = SimilarityEngine::cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn find_duplicates_detects_similar() {
        let texts: Vec<String> = vec![
            "the quick brown fox".into(),
            "the quick brown fox jumps".into(),
            "completely different text about cats".into(),
        ];
        let dups = SimilarityEngine::find_duplicates(&texts, 0.3);
        assert!(!dups.is_empty());
    }

    #[test]
    fn rank_by_similarity_order() {
        let corpus: Vec<String> = vec![
            "rust programming".into(),
            "python scripting".into(),
            "rust ownership model".into(),
        ];
        let ranked = SimilarityEngine::rank_by_similarity("rust", &corpus, SimilarityMethod::Jaccard);
        // First result should mention rust.
        assert!(corpus[ranked[0].0].contains("rust"));
    }
}
