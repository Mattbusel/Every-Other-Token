//! BM25 retrieval from a document store.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// A document in the store.
#[derive(Debug, Clone)]
pub struct Document {
    pub id: u64,
    pub title: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
    /// Unused; reserved for future embedding-based retrieval.
    pub embedding_hint: f64,
}

/// Thread-safe document store.
pub struct DocumentStore {
    documents: Mutex<HashMap<u64, Document>>,
    next_id: AtomicU64,
}

impl DocumentStore {
    pub fn new() -> Self {
        Self {
            documents: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
        }
    }

    /// Add a document and return its assigned id.
    pub fn add(&self, title: String, content: String, metadata: HashMap<String, String>) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let doc = Document {
            id,
            title,
            content,
            metadata,
            embedding_hint: 0.0,
        };
        self.documents.lock().unwrap().insert(id, doc);
        id
    }

    pub fn get(&self, id: u64) -> Option<Document> {
        self.documents.lock().unwrap().get(&id).cloned()
    }

    pub fn all_ids(&self) -> Vec<u64> {
        self.documents.lock().unwrap().keys().cloned().collect()
    }
}

impl Default for DocumentStore {
    fn default() -> Self {
        Self::new()
    }
}

const STOP_WORDS: &[&str] = &[
    "the", "is", "in", "at", "of", "a", "an", "and", "or", "but", "to", "for", "with", "on",
    "by",
];

/// Tokenize text: lowercase, split on non-alphanumeric, len >= 2, filter stop words.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .map(|t| t.to_lowercase())
        .filter(|t| t.len() >= 2 && !STOP_WORDS.contains(&t.as_str()))
        .collect()
}

/// BM25 retriever over a DocumentStore.
pub struct BM25Retriever {
    pub store: Arc<DocumentStore>,
    pub k1: f64,
    pub b: f64,
    idf: HashMap<String, f64>,
    doc_lengths: HashMap<u64, usize>,
    avg_length: f64,
}

impl BM25Retriever {
    pub fn new(store: Arc<DocumentStore>) -> Self {
        let mut retriever = Self {
            store,
            k1: 1.5,
            b: 0.75,
            idf: HashMap::new(),
            doc_lengths: HashMap::new(),
            avg_length: 0.0,
        };
        retriever.build_index();
        retriever
    }

    /// Build the BM25 index: compute IDF for all terms.
    pub fn build_index(&mut self) {
        let ids = self.store.all_ids();
        let n = ids.len() as f64;

        // Compute doc lengths
        let mut term_doc_freq: HashMap<String, usize> = HashMap::new();
        let mut total_len = 0usize;

        for id in &ids {
            if let Some(doc) = self.store.get(*id) {
                let tokens = tokenize(&format!("{} {}", doc.title, doc.content));
                let len = tokens.len();
                self.doc_lengths.insert(*id, len);
                total_len += len;

                // Count unique terms per document
                let unique: std::collections::HashSet<String> = tokens.into_iter().collect();
                for term in unique {
                    *term_doc_freq.entry(term).or_insert(0) += 1;
                }
            }
        }

        self.avg_length = if ids.is_empty() {
            0.0
        } else {
            total_len as f64 / ids.len() as f64
        };

        // Compute IDF
        self.idf.clear();
        for (term, df) in &term_doc_freq {
            let df = *df as f64;
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            self.idf.insert(term.clone(), idf);
        }
    }

    /// Score a document against query tokens using BM25.
    pub fn score(&self, query_tokens: &[String], doc_id: u64) -> f64 {
        let doc = match self.store.get(doc_id) {
            Some(d) => d,
            None => return 0.0,
        };
        let tokens = tokenize(&format!("{} {}", doc.title, doc.content));
        let doc_len = *self.doc_lengths.get(&doc_id).unwrap_or(&tokens.len()) as f64;

        // Build term frequencies
        let mut tf: HashMap<&str, usize> = HashMap::new();
        for t in &tokens {
            *tf.entry(t.as_str()).or_insert(0) += 1;
        }

        let mut score = 0.0;
        for term in query_tokens {
            let idf = self.idf.get(term.as_str()).copied().unwrap_or(0.0);
            let freq = *tf.get(term.as_str()).unwrap_or(&0) as f64;
            let numerator = freq * (self.k1 + 1.0);
            let denominator =
                freq + self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_length.max(1.0));
            score += idf * numerator / denominator.max(1e-9);
        }
        score
    }

    /// Search and return top-k (doc_id, score) sorted descending.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<(u64, f64)> {
        let query_tokens = tokenize(query);
        let ids = self.store.all_ids();
        let mut scores: Vec<(u64, f64)> = ids
            .iter()
            .map(|id| (*id, self.score(&query_tokens, *id)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }

    /// Search with metadata filter: only include docs where all filter key-values match.
    pub fn search_with_filter(
        &self,
        query: &str,
        top_k: usize,
        filter: &HashMap<String, String>,
    ) -> Vec<(u64, f64)> {
        let query_tokens = tokenize(query);
        let ids = self.store.all_ids();
        let mut scores: Vec<(u64, f64)> = ids
            .iter()
            .filter_map(|id| {
                let doc = self.store.get(*id)?;
                // Check all filter conditions
                for (k, v) in filter {
                    if doc.metadata.get(k).map(|s| s.as_str()) != Some(v.as_str()) {
                        return None;
                    }
                }
                Some((*id, self.score(&query_tokens, *id)))
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }
}

/// High-level context retriever that formats results.
pub struct ContextRetriever {
    pub retriever: BM25Retriever,
    pub max_context_chars: usize,
}

impl ContextRetriever {
    pub fn new(retriever: BM25Retriever, max_context_chars: usize) -> Self {
        Self {
            retriever,
            max_context_chars,
        }
    }

    /// Retrieve top-n documents and format as context string.
    pub fn retrieve_context(&self, query: &str, n: usize) -> String {
        let results = self.retriever.search(query, n);
        let mut context = String::new();
        for (doc_id, _score) in results {
            if let Some(doc) = self.retriever.store.get(doc_id) {
                let section = format!("## Doc: {}\n{}\n\n", doc.title, doc.content);
                if context.len() + section.len() > self.max_context_chars {
                    let remaining = self.max_context_chars.saturating_sub(context.len());
                    if remaining > 0 {
                        context.push_str(&section[..remaining]);
                    }
                    break;
                }
                context.push_str(&section);
            }
        }
        context
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store() -> Arc<DocumentStore> {
        let store = Arc::new(DocumentStore::new());
        store.add(
            "Rust Programming".to_string(),
            "Rust is a systems programming language focused on safety and performance.".to_string(),
            HashMap::new(),
        );
        store.add(
            "Python Tutorial".to_string(),
            "Python is a high-level scripting language easy to learn.".to_string(),
            HashMap::new(),
        );
        store
    }

    #[test]
    fn test_add_and_search_finds_relevant_doc() {
        let store = make_store();
        let retriever = BM25Retriever::new(store.clone());
        let results = retriever.search("Rust systems programming", 1);
        assert!(!results.is_empty());
        let (doc_id, _) = results[0];
        let doc = store.get(doc_id).unwrap();
        assert!(doc.title.contains("Rust"));
    }

    #[test]
    fn test_stop_words_filtered() {
        let tokens = tokenize("the and is of a an");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_filter_by_metadata() {
        let store = Arc::new(DocumentStore::new());
        let mut meta1 = HashMap::new();
        meta1.insert("lang".to_string(), "rust".to_string());
        store.add(
            "Rust Doc".to_string(),
            "Rust systems language".to_string(),
            meta1,
        );
        let mut meta2 = HashMap::new();
        meta2.insert("lang".to_string(), "python".to_string());
        store.add(
            "Python Doc".to_string(),
            "Python scripting language".to_string(),
            meta2,
        );

        let retriever = BM25Retriever::new(store.clone());
        let mut filter = HashMap::new();
        filter.insert("lang".to_string(), "rust".to_string());
        let results = retriever.search_with_filter("language", 5, &filter);
        assert_eq!(results.len(), 1);
        let doc = store.get(results[0].0).unwrap();
        assert_eq!(doc.metadata["lang"], "rust");
    }

    #[test]
    fn test_context_formatted_correctly() {
        let store = make_store();
        let retriever = BM25Retriever::new(store);
        let ctx_retriever = ContextRetriever::new(retriever, 10000);
        let ctx = ctx_retriever.retrieve_context("Rust", 2);
        assert!(ctx.contains("## Doc:"));
        assert!(ctx.contains("Rust"));
    }

    #[test]
    fn test_exact_match_scores_higher_than_unrelated() {
        let store = Arc::new(DocumentStore::new());
        store.add(
            "Exact Match".to_string(),
            "quantum entanglement physics experiment".to_string(),
            HashMap::new(),
        );
        store.add(
            "Unrelated".to_string(),
            "cooking recipes pasta sauce".to_string(),
            HashMap::new(),
        );
        let retriever = BM25Retriever::new(store.clone());
        let results = retriever.search("quantum physics", 2);
        assert!(results.len() >= 1);
        // The physics doc should rank first
        let top_doc = store.get(results[0].0).unwrap();
        assert!(top_doc.title.contains("Exact"));
    }
}
