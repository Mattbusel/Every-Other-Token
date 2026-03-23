//! Semantic similarity heatmap — TF-IDF cosine similarity across token windows.
//!
//! [`SemanticDriftTracker`] groups a token sequence into fixed-size windows, builds
//! TF-IDF vectors for each window (no external API required), and exposes cosine
//! similarity between pairs of windows.  The full NxN similarity matrix can be
//! exported as an SVG colour heatmap (blue = similar, red = divergent) or as CSV.
//!
//! ## Example
//!
//! ```rust
//! use every_other_token::semantic_heatmap::{SemanticDriftTracker, HeatmapExporter};
//!
//! let mut tracker = SemanticDriftTracker::new(20);
//! tracker.add_token("the quick brown fox");
//! let heatmap = tracker.full_heatmap();
//! let svg = HeatmapExporter::to_svg(&heatmap);
//! let csv = HeatmapExporter::to_csv(&heatmap);
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Internal TF-IDF helpers
// ---------------------------------------------------------------------------

/// Tokenise a text string into lowercase words (splits on non-alphanumeric chars).
fn tokenise_text(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

/// Build a raw term-frequency map for a slice of word tokens.
fn term_frequency(words: &[String]) -> HashMap<String, f64> {
    let mut tf: HashMap<String, f64> = HashMap::new();
    for w in words {
        *tf.entry(w.clone()).or_insert(0.0) += 1.0;
    }
    // Normalise by document length.
    let total = words.len() as f64;
    if total > 0.0 {
        for v in tf.values_mut() {
            *v /= total;
        }
    }
    tf
}

/// Compute IDF weights from a corpus of term-frequency maps.
fn inverse_document_frequency(docs: &[HashMap<String, f64>]) -> HashMap<String, f64> {
    let n = docs.len() as f64;
    if n == 0.0 {
        return HashMap::new();
    }
    let mut df: HashMap<String, usize> = HashMap::new();
    for doc in docs {
        for term in doc.keys() {
            *df.entry(term.clone()).or_insert(0) += 1;
        }
    }
    df.into_iter()
        .map(|(term, count)| {
            let idf = ((n + 1.0) / (count as f64 + 1.0)).ln() + 1.0; // smoothed IDF
            (term, idf)
        })
        .collect()
}

/// Multiply TF by IDF to produce a TF-IDF vector.
fn tfidf_vector(tf: &HashMap<String, f64>, idf: &HashMap<String, f64>) -> HashMap<String, f64> {
    tf.iter()
        .filter_map(|(term, &tf_val)| idf.get(term).map(|&idf_val| (term.clone(), tf_val * idf_val)))
        .collect()
}

/// Cosine similarity between two sparse vectors represented as `HashMap<term, weight>`.
fn cosine_similarity(a: &HashMap<String, f64>, b: &HashMap<String, f64>) -> f64 {
    let dot: f64 = a
        .iter()
        .filter_map(|(term, &wa)| b.get(term).map(|&wb| wa * wb))
        .sum();
    let norm_a: f64 = a.values().map(|v| v * v).sum::<f64>().sqrt();
    let norm_b: f64 = b.values().map(|v| v * v).sum::<f64>().sqrt();
    let denom = norm_a * norm_b;
    if denom < f64::EPSILON { 0.0 } else { (dot / denom).max(0.0).min(1.0) }
}

// ---------------------------------------------------------------------------
// SemanticDriftTracker
// ---------------------------------------------------------------------------

/// Groups token text values into windows and computes pairwise cosine similarity
/// using TF-IDF vectors, with no external API or embedding service required.
pub struct SemanticDriftTracker {
    /// Number of tokens per semantic window.
    window_size: usize,
    /// Accumulated token text strings.
    tokens: Vec<String>,
}

impl SemanticDriftTracker {
    /// Create a new tracker with a given window size.
    ///
    /// Panics in debug mode if `window_size` is zero; in release it is clamped to 1.
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size: window_size.max(1),
            tokens: Vec::new(),
        }
    }

    /// Append a single token string (typically the `text` or `original` field of
    /// a `TokenEvent`).
    pub fn add_token(&mut self, token: &str) {
        self.tokens.push(token.to_string());
    }

    /// Append multiple token strings at once.
    pub fn add_tokens(&mut self, tokens: &[&str]) {
        for &t in tokens {
            self.tokens.push(t.to_string());
        }
    }

    /// Number of complete windows that have accumulated so far.
    pub fn window_count(&self) -> usize {
        if self.tokens.is_empty() {
            0
        } else {
            (self.tokens.len() + self.window_size - 1) / self.window_size
        }
    }

    /// Return the concatenated text for window index `w`.
    ///
    /// Returns `None` when `w` is out of range.
    fn window_text(&self, w: usize) -> Option<String> {
        let start = w * self.window_size;
        if start >= self.tokens.len() {
            return None;
        }
        let end = (start + self.window_size).min(self.tokens.len());
        Some(self.tokens[start..end].join(" "))
    }

    /// Build all window TF maps.
    fn build_tf_maps(&self) -> Vec<HashMap<String, f64>> {
        (0..self.window_count())
            .map(|w| {
                let text = self.window_text(w).unwrap_or_default();
                let words = tokenise_text(&text);
                term_frequency(&words)
            })
            .collect()
    }

    /// Compute the cosine similarity between two windows identified by index.
    ///
    /// Returns `0.0` when either index is out of range or both windows are empty.
    pub fn drift_score(&self, window_a: usize, window_b: usize) -> f64 {
        let tf_maps = self.build_tf_maps();
        let idf = inverse_document_frequency(&tf_maps);
        let get_vec = |w: usize| -> HashMap<String, f64> {
            tf_maps.get(w).map(|tf| tfidf_vector(tf, &idf)).unwrap_or_default()
        };
        let va = get_vec(window_a);
        let vb = get_vec(window_b);
        cosine_similarity(&va, &vb)
    }

    /// Compute the full NxN similarity matrix across all windows.
    ///
    /// Entry `[i][j]` is the cosine similarity between window `i` and window `j`.
    /// Returns an empty matrix when there are no windows.
    pub fn full_heatmap(&self) -> Vec<Vec<f64>> {
        let n = self.window_count();
        if n == 0 {
            return vec![];
        }
        let tf_maps = self.build_tf_maps();
        let idf = inverse_document_frequency(&tf_maps);
        let vecs: Vec<HashMap<String, f64>> =
            tf_maps.iter().map(|tf| tfidf_vector(tf, &idf)).collect();

        (0..n)
            .map(|i| (0..n).map(|j| cosine_similarity(&vecs[i], &vecs[j])).collect())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// HeatmapExporter
// ---------------------------------------------------------------------------

/// Exports a 2-D similarity matrix to SVG or CSV.
pub struct HeatmapExporter;

impl HeatmapExporter {
    /// Generate an SVG colour heatmap.
    ///
    /// Values close to `1.0` (similar) are rendered in blue; values close to `0.0`
    /// (divergent) are rendered in red.  Each cell is 40×40 px with a 1 px gap.
    pub fn to_svg(heatmap: &[Vec<f64>]) -> String {
        let n = heatmap.len();
        if n == 0 {
            return r#"<svg xmlns="http://www.w3.org/2000/svg" width="0" height="0"></svg>"#
                .to_string();
        }

        const CELL: usize = 40;
        const GAP: usize = 1;
        let size = n * CELL + (n + 1) * GAP;

        let mut svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">"#,
            size = size
        );

        // Light-grey background.
        svg.push_str(&format!(
            r##"<rect width="{s}" height="{s}" fill="#f0f0f0"/>"##,
            s = size
        ));

        for (i, row) in heatmap.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                let val = val.max(0.0).min(1.0);
                // Blue (similar) → Red (divergent)
                let r = ((1.0 - val) * 255.0) as u8;
                let g = 0u8;
                let b = (val * 255.0) as u8;
                let x = GAP + j * (CELL + GAP);
                let y = GAP + i * (CELL + GAP);
                svg.push_str(&format!(
                    r#"<rect x="{x}" y="{y}" width="{c}" height="{c}" fill="rgb({r},{g},{b})"/>"#,
                    x = x,
                    y = y,
                    c = CELL,
                    r = r,
                    g = g,
                    b = b,
                ));
                // Value label.
                let lx = x + CELL / 2;
                let ly = y + CELL / 2 + 4;
                let label = format!("{:.2}", val);
                svg.push_str(&format!(
                    r#"<text x="{lx}" y="{ly}" text-anchor="middle" font-size="10" fill="white">{label}</text>"#,
                    lx = lx,
                    ly = ly,
                    label = label,
                ));
            }
        }

        svg.push_str("</svg>");
        svg
    }

    /// Export the heatmap as a CSV string.
    ///
    /// First row is a header `w_0,w_1,...`; subsequent rows contain the similarity values.
    pub fn to_csv(heatmap: &[Vec<f64>]) -> String {
        let n = heatmap.len();
        if n == 0 {
            return String::new();
        }
        let header: Vec<String> = (0..n).map(|i| format!("w_{}", i)).collect();
        let mut out = header.join(",");
        out.push('\n');
        for row in heatmap {
            let cells: Vec<String> = row.iter().map(|v| format!("{:.6}", v)).collect();
            out.push_str(&cells.join(","));
            out.push('\n');
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tracker_with_tokens(tokens: &[&str], window_size: usize) -> SemanticDriftTracker {
        let mut t = SemanticDriftTracker::new(window_size);
        t.add_tokens(tokens);
        t
    }

    #[test]
    fn test_window_count_empty() {
        let tracker = SemanticDriftTracker::new(20);
        assert_eq!(tracker.window_count(), 0);
    }

    #[test]
    fn test_window_count_exact_multiple() {
        let tokens: Vec<&str> = (0..40).map(|_| "tok").collect();
        let tracker = tracker_with_tokens(&tokens, 20);
        assert_eq!(tracker.window_count(), 2);
    }

    #[test]
    fn test_window_count_partial_last_window() {
        let tokens: Vec<&str> = (0..25).map(|_| "tok").collect();
        let tracker = tracker_with_tokens(&tokens, 20);
        assert_eq!(tracker.window_count(), 2);
    }

    #[test]
    fn test_drift_score_identical_windows() {
        // Two identical windows should have similarity = 1.0.
        let tokens: Vec<&str> = (0..40).map(|_| "hello world").collect();
        let tracker = tracker_with_tokens(&tokens, 20);
        let score = tracker.drift_score(0, 0);
        assert!((score - 1.0).abs() < 1e-9, "Self-similarity should be 1.0, got {}", score);
    }

    #[test]
    fn test_drift_score_out_of_range() {
        let tracker = SemanticDriftTracker::new(20);
        // No tokens — should return 0.0 gracefully.
        assert_eq!(tracker.drift_score(0, 1), 0.0);
    }

    #[test]
    fn test_drift_score_in_range() {
        let tokens_a: Vec<&str> = vec!["apple", "banana", "cherry"];
        let tokens_b: Vec<&str> = vec!["dog", "cat", "fish"];
        let mut tracker = SemanticDriftTracker::new(3);
        tracker.add_tokens(&tokens_a);
        tracker.add_tokens(&tokens_b);
        let score = tracker.drift_score(0, 1);
        // Completely different vocabulary → should be near 0.
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_full_heatmap_dimensions() {
        let tokens: Vec<&str> = (0..60).map(|_| "the").collect();
        let tracker = tracker_with_tokens(&tokens, 20);
        let hm = tracker.full_heatmap();
        assert_eq!(hm.len(), 3);
        for row in &hm {
            assert_eq!(row.len(), 3);
        }
    }

    #[test]
    fn test_full_heatmap_diagonal_is_one() {
        let tokens: Vec<&str> = vec!["alpha", "beta", "gamma", "delta", "epsilon", "zeta"];
        let tracker = tracker_with_tokens(&tokens, 2);
        let hm = tracker.full_heatmap();
        for (i, row) in hm.iter().enumerate() {
            assert!((row[i] - 1.0).abs() < 1e-9, "Diagonal [{}][{}] should be 1.0", i, i);
        }
    }

    #[test]
    fn test_full_heatmap_empty() {
        let tracker = SemanticDriftTracker::new(20);
        assert!(tracker.full_heatmap().is_empty());
    }

    #[test]
    fn test_heatmap_exporter_svg_contains_svg_tag() {
        let hm = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        let svg = HeatmapExporter::to_svg(&hm);
        assert!(svg.contains("<svg"), "SVG output should contain <svg tag");
        assert!(svg.contains("</svg>"), "SVG output should be closed");
    }

    #[test]
    fn test_heatmap_exporter_svg_empty() {
        let svg = HeatmapExporter::to_svg(&[]);
        assert!(svg.contains("<svg"), "Empty SVG should still be valid XML");
    }

    #[test]
    fn test_heatmap_exporter_csv_header() {
        let hm = vec![vec![1.0, 0.7], vec![0.7, 1.0]];
        let csv = HeatmapExporter::to_csv(&hm);
        assert!(csv.starts_with("w_0,w_1"), "CSV should start with header");
    }

    #[test]
    fn test_heatmap_exporter_csv_empty() {
        let csv = HeatmapExporter::to_csv(&[]);
        assert!(csv.is_empty(), "Empty heatmap should produce empty CSV");
    }

    #[test]
    fn test_heatmap_exporter_csv_row_count() {
        let n = 4;
        let hm: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.5 }).collect())
            .collect();
        let csv = HeatmapExporter::to_csv(&hm);
        // 1 header + n data rows.
        assert_eq!(csv.lines().count(), n + 1);
    }

    #[test]
    fn test_tokenise_text_splits_on_non_alphanum() {
        let words = tokenise_text("hello, world! foo123");
        assert!(words.contains(&"hello".to_string()));
        assert!(words.contains(&"world".to_string()));
        assert!(words.contains(&"foo123".to_string()));
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let mut a = HashMap::new();
        a.insert("x".to_string(), 1.0);
        let mut b = HashMap::new();
        b.insert("y".to_string(), 1.0);
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let mut a = HashMap::new();
        a.insert("x".to_string(), 0.5);
        a.insert("y".to_string(), 0.5);
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-9);
    }
}
