//! Source tracking and citation management.
//!
//! Provides structured representations for academic and web citations,
//! a collection type with fuzzy search, and utilities for extracting
//! inline references from free text.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// CitationStyle
// ---------------------------------------------------------------------------

/// Formatting style for citations and bibliographies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CitationStyle {
    APA,
    MLA,
    Chicago,
    IEEE,
    Numeric,
    AuthorDate,
}

impl fmt::Display for CitationStyle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CitationStyle::APA => write!(f, "APA"),
            CitationStyle::MLA => write!(f, "MLA"),
            CitationStyle::Chicago => write!(f, "Chicago"),
            CitationStyle::IEEE => write!(f, "IEEE"),
            CitationStyle::Numeric => write!(f, "Numeric"),
            CitationStyle::AuthorDate => write!(f, "Author-Date"),
        }
    }
}

// ---------------------------------------------------------------------------
// SourceType
// ---------------------------------------------------------------------------

/// The type of source being cited.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceType {
    WebPage,
    Book,
    JournalArticle,
    ArxivPaper,
    WikipediaArticle,
    BlogPost,
    TechnicalReport,
}

impl fmt::Display for SourceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SourceType::WebPage => write!(f, "Web Page"),
            SourceType::Book => write!(f, "Book"),
            SourceType::JournalArticle => write!(f, "Journal Article"),
            SourceType::ArxivPaper => write!(f, "arXiv Paper"),
            SourceType::WikipediaArticle => write!(f, "Wikipedia Article"),
            SourceType::BlogPost => write!(f, "Blog Post"),
            SourceType::TechnicalReport => write!(f, "Technical Report"),
        }
    }
}

// ---------------------------------------------------------------------------
// Citation
// ---------------------------------------------------------------------------

static CITATION_COUNTER: AtomicU64 = AtomicU64::new(1);

/// A single source citation.
#[derive(Debug, Clone)]
pub struct Citation {
    pub id: u64,
    pub title: String,
    pub authors: Vec<String>,
    pub url: Option<String>,
    pub published_date: Option<String>,
    pub source_type: SourceType,
    pub accessed_date: String,
    pub doi: Option<String>,
    pub excerpt: Option<String>,
}

impl Citation {
    /// Create a new citation with an auto-assigned ID.
    pub fn new(
        title: impl Into<String>,
        authors: Vec<String>,
        source_type: SourceType,
        accessed_date: impl Into<String>,
    ) -> Self {
        Self {
            id: CITATION_COUNTER.fetch_add(1, Ordering::Relaxed),
            title: title.into(),
            authors,
            url: None,
            published_date: None,
            source_type,
            accessed_date: accessed_date.into(),
            doi: None,
            excerpt: None,
        }
    }

    /// Return the first author's last name or "Unknown".
    fn first_author_last(&self) -> &str {
        self.authors
            .first()
            .and_then(|a| a.split_whitespace().last())
            .unwrap_or("Unknown")
    }

    /// Year portion of `published_date`, e.g. "2023".
    fn year(&self) -> &str {
        self.published_date
            .as_deref()
            .and_then(|d| d.split('-').next())
            .unwrap_or("n.d.")
    }

    /// Format a fully-rendered citation string in the requested style.
    pub fn format(&self, style: &CitationStyle) -> String {
        let authors_str = if self.authors.is_empty() {
            "Unknown Author".to_string()
        } else {
            self.authors.join(", ")
        };

        let url_part = self.url.as_deref().unwrap_or("");
        let doi_part = self.doi.as_deref().unwrap_or("");

        match style {
            CitationStyle::APA => {
                let mut s = format!("{} ({}). {}.", authors_str, self.year(), self.title);
                if !url_part.is_empty() {
                    s.push_str(&format!(" Retrieved from {}", url_part));
                }
                if !doi_part.is_empty() {
                    s.push_str(&format!(" https://doi.org/{}", doi_part));
                }
                s
            }
            CitationStyle::MLA => {
                let mut s = format!("{}. \"{}\".", authors_str, self.title);
                if let Some(ref d) = self.published_date {
                    s.push_str(&format!(" {}.", d));
                }
                if !url_part.is_empty() {
                    s.push_str(&format!(" {}", url_part));
                }
                s
            }
            CitationStyle::Chicago => {
                let mut s = format!("{}. \"{}\".", authors_str, self.title);
                if let Some(ref d) = self.published_date {
                    s.push_str(&format!(" {}", d));
                }
                if !url_part.is_empty() {
                    s.push_str(&format!(" Accessed {}. {}", self.accessed_date, url_part));
                }
                s
            }
            CitationStyle::IEEE => {
                let author_abbrev = self.authors.iter().map(|a| {
                    let parts: Vec<_> = a.split_whitespace().collect();
                    if parts.len() >= 2 {
                        format!("{} {}", &parts[..parts.len()-1].iter().map(|p| format!("{}.", &p[..1])).collect::<Vec<_>>().join(" "), parts.last().unwrap())
                    } else {
                        a.clone()
                    }
                }).collect::<Vec<_>>().join(", ");
                let mut s = format!("{}, \"{},\"", author_abbrev, self.title);
                if let Some(ref d) = self.published_date {
                    s.push_str(&format!(" {}.", d));
                }
                if !doi_part.is_empty() {
                    s.push_str(&format!(" doi: {}", doi_part));
                } else if !url_part.is_empty() {
                    s.push_str(&format!(" [Online]. Available: {}", url_part));
                }
                s
            }
            CitationStyle::Numeric => {
                format!("[{}] {}. {}. {}.", self.id, authors_str, self.title, self.year())
            }
            CitationStyle::AuthorDate => {
                format!("{} {}. {}.", self.first_author_last(), self.year(), self.title)
            }
        }
    }

    /// Return a short inline reference: "(LastName, Year)" or "[N]".
    pub fn short_ref(&self) -> String {
        format!("({}, {})", self.first_author_last(), self.year())
    }
}

// ---------------------------------------------------------------------------
// CitationCollection
// ---------------------------------------------------------------------------

/// Manages a set of citations in a specific formatting style.
#[derive(Debug, Clone)]
pub struct CitationCollection {
    pub citations: Vec<Citation>,
    pub style: CitationStyle,
}

impl CitationCollection {
    pub fn new(style: CitationStyle) -> Self {
        Self { citations: Vec::new(), style }
    }

    /// Add a citation and return its ID.
    pub fn add(&mut self, citation: Citation) -> u64 {
        let id = citation.id;
        self.citations.push(citation);
        id
    }

    /// Look up a citation by ID.
    pub fn get(&self, id: u64) -> Option<&Citation> {
        self.citations.iter().find(|c| c.id == id)
    }

    /// Find a citation by exact URL.
    pub fn find_by_url(&self, url: &str) -> Option<&Citation> {
        self.citations.iter().find(|c| c.url.as_deref() == Some(url))
    }

    /// Fuzzy title search — returns citations whose title contains any word
    /// from the query (case-insensitive).
    pub fn find_by_title(&self, title: &str) -> Vec<&Citation> {
        let query_lower = title.to_lowercase();
        let query_words: Vec<_> = query_lower.split_whitespace().collect();
        self.citations
            .iter()
            .filter(|c| {
                let t = c.title.to_lowercase();
                query_words.iter().any(|w| t.contains(w))
            })
            .collect()
    }

    /// Generate a formatted bibliography (sorted by first author then year).
    pub fn bibliography(&self) -> String {
        let mut sorted = self.citations.clone();
        sorted.sort_by(|a, b| {
            a.authors
                .first()
                .cloned()
                .unwrap_or_default()
                .cmp(&b.authors.first().cloned().unwrap_or_default())
                .then(a.published_date.cmp(&b.published_date))
        });
        sorted
            .iter()
            .map(|c| c.format(&self.style))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Return the inline citation text for the given ID.
    pub fn inline_cite(&self, id: u64) -> String {
        match self.get(id) {
            Some(c) => match &self.style {
                CitationStyle::Numeric => format!("[{}]", c.id),
                _ => c.short_ref(),
            },
            None => format!("[citation {}]", id),
        }
    }

    /// Return all citations that are referenced in *text* by their ID
    /// (e.g. "[3]" or "citation 3").
    pub fn used_citations(&self, text: &str) -> Vec<&Citation> {
        self.citations
            .iter()
            .filter(|c| {
                let id_str = c.id.to_string();
                text.contains(&format!("[{}]", id_str))
                    || text.contains(&format!("citation {}", id_str))
                    || text.contains(&c.short_ref())
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// CitationExtractor
// ---------------------------------------------------------------------------

/// Utilities for extracting citation-related content from free text.
#[derive(Debug, Default)]
pub struct CitationExtractor;

impl CitationExtractor {
    pub fn new() -> Self {
        Self
    }

    /// Extract all HTTP/HTTPS URLs from text.
    pub fn extract_urls(&self, text: &str) -> Vec<String> {
        let mut urls = Vec::new();
        let mut remaining = text;
        while let Some(start) = remaining.find("http") {
            let slice = &remaining[start..];
            // Check it's http:// or https://
            if slice.starts_with("http://") || slice.starts_with("https://") {
                let end = slice
                    .find(|c: char| c.is_whitespace() || c == ')' || c == '>' || c == '"' || c == '\'')
                    .unwrap_or(slice.len());
                let url = &slice[..end];
                if url.len() > 8 {
                    urls.push(url.to_string());
                }
                remaining = &remaining[start + end..];
            } else {
                remaining = &remaining[start + 4..];
            }
        }
        urls
    }

    /// Extract inline citation markers: [1], (Smith, 2020), etc.
    pub fn extract_inline_refs(&self, text: &str) -> Vec<String> {
        let mut refs = Vec::new();

        // Numeric refs: [1], [12], etc.
        let mut chars = text.char_indices().peekable();
        while let Some((i, c)) = chars.next() {
            if c == '[' {
                let rest = &text[i..];
                if let Some(end) = rest.find(']') {
                    let inner = &rest[1..end];
                    if inner.chars().all(|ch| ch.is_ascii_digit()) && !inner.is_empty() {
                        refs.push(rest[..=end].to_string());
                    }
                }
            }
        }

        // Author-date refs: (Word, YYYY) pattern
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'(' {
                if let Some(end_rel) = text[i..].find(')') {
                    let inner = &text[i+1..i+end_rel];
                    // Check pattern: "Name, YYYY" or "Name YYYY"
                    let has_year = inner.split_whitespace().any(|w| {
                        w.len() == 4 && w.chars().all(|c| c.is_ascii_digit())
                            && w.starts_with(|c: char| c == '1' || c == '2')
                    });
                    if has_year && inner.contains(',') {
                        refs.push(format!("({})", inner));
                    }
                    i += end_rel + 1;
                    continue;
                }
            }
            i += 1;
        }

        refs.sort();
        refs.dedup();
        refs
    }

    /// Find sentences that contain factual indicators but no nearby citation.
    pub fn detect_uncited_claims(&self, text: &str, citations: &CitationCollection) -> Vec<String> {
        let factual_indicators = [
            "according to", "studies show", "research indicates", "data suggests",
            "it has been shown", "evidence shows", "experts say", "statistics",
            "percent", "%" , "found that", "demonstrated", "confirmed",
        ];

        let text_lower = text.to_lowercase();
        let sentences: Vec<&str> = text
            .split(|c| c == '.' || c == '!' || c == '?')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();

        sentences
            .into_iter()
            .filter(|sent| {
                let sl = sent.to_lowercase();
                let has_factual = factual_indicators.iter().any(|ind| sl.contains(ind));
                if !has_factual {
                    return false;
                }
                // Check if any citation appears nearby (in this sentence or adjacent text)
                let has_citation = citations.citations.iter().any(|c| {
                    let id_str = c.id.to_string();
                    sent.contains(&format!("[{}]", id_str))
                        || sent.contains(&c.short_ref())
                        || text_lower.contains(&c.title.to_lowercase())
                });
                !has_citation
            })
            .map(|s| s.to_string())
            .collect()
    }

    /// Replace citation markers in *text* with formatted references using the
    /// collection's style.
    pub fn auto_format(
        &self,
        text: &str,
        citations: &CitationCollection,
        style: &CitationStyle,
    ) -> String {
        let mut result = text.to_string();

        for citation in &citations.citations {
            let id_str = citation.id.to_string();
            let marker = format!("[{}]", id_str);
            let formatted = match style {
                CitationStyle::Numeric => format!("[{}]", citation.id),
                _ => citation.short_ref(),
            };
            result = result.replace(&marker, &formatted);
        }

        result
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_citation() -> Citation {
        let mut c = Citation::new(
            "Understanding Rust",
            vec!["Alice Smith".to_string(), "Bob Jones".to_string()],
            SourceType::Book,
            "2024-01-15",
        );
        c.published_date = Some("2023".to_string());
        c.url = Some("https://example.com/rust".to_string());
        c
    }

    #[test]
    fn test_citation_format_apa() {
        let c = make_citation();
        let s = c.format(&CitationStyle::APA);
        assert!(s.contains("Smith"), "APA should include author");
        assert!(s.contains("2023"), "APA should include year");
        assert!(s.contains("Understanding Rust"), "APA should include title");
    }

    #[test]
    fn test_citation_format_ieee() {
        let c = make_citation();
        let s = c.format(&CitationStyle::IEEE);
        assert!(s.contains("Understanding Rust"));
    }

    #[test]
    fn test_collection_add_get() {
        let mut coll = CitationCollection::new(CitationStyle::APA);
        let c = make_citation();
        let id = coll.add(c);
        assert!(coll.get(id).is_some());
    }

    #[test]
    fn test_find_by_title_fuzzy() {
        let mut coll = CitationCollection::new(CitationStyle::APA);
        coll.add(make_citation());
        let results = coll.find_by_title("Rust");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_bibliography() {
        let mut coll = CitationCollection::new(CitationStyle::MLA);
        coll.add(make_citation());
        let bib = coll.bibliography();
        assert!(bib.contains("Understanding Rust"));
    }

    #[test]
    fn test_extract_urls() {
        let extractor = CitationExtractor::new();
        let text = "See https://example.com/page and http://other.org/doc for details.";
        let urls = extractor.extract_urls(text);
        assert_eq!(urls.len(), 2);
        assert!(urls[0].contains("example.com"));
    }

    #[test]
    fn test_extract_inline_refs() {
        let extractor = CitationExtractor::new();
        let text = "As noted [1] and confirmed by (Smith, 2020) and also [3].";
        let refs = extractor.extract_inline_refs(text);
        assert!(refs.iter().any(|r| r == "[1]"));
        assert!(refs.iter().any(|r| r == "[3]"));
    }

    #[test]
    fn test_inline_cite_numeric() {
        let mut coll = CitationCollection::new(CitationStyle::Numeric);
        let c = make_citation();
        let id = coll.add(c);
        let inline = coll.inline_cite(id);
        assert!(inline.contains(&id.to_string()));
    }

    #[test]
    fn test_style_display() {
        assert_eq!(CitationStyle::APA.to_string(), "APA");
        assert_eq!(CitationStyle::Chicago.to_string(), "Chicago");
    }
}
