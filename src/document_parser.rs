//! Document text extraction and structure detection.

use std::collections::HashMap;

/// The detected format of a document.
#[derive(Debug, Clone, PartialEq)]
pub enum DocumentFormat {
    PlainText,
    Markdown,
    Html,
    Json,
    Csv,
    Code(String),
    Unknown,
}

/// A section of a parsed document.
#[derive(Debug, Clone)]
pub struct DocumentSection {
    pub title: Option<String>,
    pub content: String,
    /// Heading depth: 0 = body text, 1-6 = heading level.
    pub level: u8,
    pub line_start: usize,
    pub line_end: usize,
}

/// The result of parsing a document.
#[derive(Debug, Clone)]
pub struct ParsedDocument {
    pub format: DocumentFormat,
    pub sections: Vec<DocumentSection>,
    pub metadata: HashMap<String, String>,
    pub word_count: usize,
    pub char_count: usize,
    pub detected_language: Option<String>,
}

/// Stateless document parser.
pub struct DocumentParser;

impl DocumentParser {
    /// Heuristically detect the format of the given text.
    pub fn detect_format(text: &str) -> DocumentFormat {
        let trimmed = text.trim();

        // JSON: starts with { or [
        if (trimmed.starts_with('{') && trimmed.ends_with('}'))
            || (trimmed.starts_with('[') && trimmed.ends_with(']'))
        {
            return DocumentFormat::Json;
        }

        // HTML: contains common HTML tags
        if trimmed.contains("</") && (trimmed.contains("<html") || trimmed.contains("<div") || trimmed.contains("<p>") || trimmed.contains("<span")) {
            return DocumentFormat::Html;
        }

        // Code fence (Markdown code block at top level)
        if trimmed.starts_with("```") {
            let lang = trimmed
                .lines()
                .next()
                .unwrap_or("")
                .trim_start_matches('`')
                .trim()
                .to_string();
            return DocumentFormat::Code(lang);
        }

        // Markdown: has ATX headers or emphasis
        let has_md_header = text.lines().any(|l| l.starts_with('#'));
        let has_md_emphasis = text.contains("**") || text.contains("__");
        let has_code_fence = text.contains("```");
        if has_md_header || has_md_emphasis || has_code_fence {
            return DocumentFormat::Markdown;
        }

        // CSV: multiple lines with consistent comma counts
        let lines: Vec<&str> = text.lines().take(5).collect();
        if lines.len() >= 2 {
            let counts: Vec<usize> = lines.iter().map(|l| l.matches(',').count()).collect();
            let first = counts[0];
            if first > 0 && counts.iter().all(|&c| c == first) {
                return DocumentFormat::Csv;
            }
        }

        DocumentFormat::PlainText
    }

    /// Parse a Markdown document into sections.
    pub fn parse_markdown(text: &str) -> ParsedDocument {
        let mut sections: Vec<DocumentSection> = Vec::new();
        let mut metadata: HashMap<String, String> = HashMap::new();

        let lines: Vec<&str> = text.lines().collect();
        let mut i = 0;

        // YAML front matter
        if lines.first().copied() == Some("---") {
            i = 1;
            while i < lines.len() && lines[i] != "---" {
                let line = lines[i];
                if let Some(colon) = line.find(':') {
                    let key = line[..colon].trim().to_string();
                    let value = line[colon + 1..].trim().to_string();
                    metadata.insert(key, value);
                }
                i += 1;
            }
            if i < lines.len() {
                i += 1; // skip closing ---
            }
        }

        let mut current_title: Option<String> = None;
        let mut current_level: u8 = 0;
        let mut current_content = String::new();
        let mut section_start = i;

        while i < lines.len() {
            let line = lines[i];
            if let Some(header) = line.strip_prefix('#') {
                // Count '#' characters
                let hashes = line.chars().take_while(|&c| c == '#').count() as u8;
                let title = line.trim_start_matches('#').trim().to_string();

                // Save previous section
                if !current_content.trim().is_empty() || current_title.is_some() {
                    sections.push(DocumentSection {
                        title: current_title.clone(),
                        content: current_content.trim().to_string(),
                        level: current_level,
                        line_start: section_start,
                        line_end: i,
                    });
                }

                current_title = Some(title);
                current_level = hashes;
                current_content = String::new();
                section_start = i;
                let _ = header; // suppress unused warning
            } else {
                current_content.push_str(line);
                current_content.push('\n');
            }
            i += 1;
        }

        // Final section
        if !current_content.trim().is_empty() || current_title.is_some() {
            sections.push(DocumentSection {
                title: current_title,
                content: current_content.trim().to_string(),
                level: current_level,
                line_start: section_start,
                line_end: lines.len(),
            });
        }

        let wc = Self::word_count(text);
        let cc = text.chars().count();
        let lang = Self::detect_language(text);

        ParsedDocument {
            format: DocumentFormat::Markdown,
            sections,
            metadata,
            word_count: wc,
            char_count: cc,
            detected_language: lang,
        }
    }

    /// Parse an HTML document into sections.
    pub fn parse_html(text: &str) -> ParsedDocument {
        let mut sections: Vec<DocumentSection> = Vec::new();
        let mut metadata: HashMap<String, String> = HashMap::new();

        // Extract <title>
        if let Some(start) = text.find("<title>") {
            if let Some(end) = text.find("</title>") {
                let title = &text[start + 7..end];
                metadata.insert("title".to_string(), title.trim().to_string());
            }
        }

        // Extract headings h1-h6
        let plain = Self::strip_html_tags(text);
        let lines: Vec<&str> = text.lines().collect();

        for (level, tag) in &[
            (1u8, "h1"),
            (2, "h2"),
            (3, "h3"),
            (4, "h4"),
            (5, "h5"),
            (6, "h6"),
        ] {
            let open = format!("<{}>", tag);
            let close = format!("</{}>", tag);
            let mut search = text;
            let mut offset = 0;
            while let Some(start) = search.find(&open) {
                let abs_start = offset + start;
                let after_open = abs_start + open.len();
                if let Some(end_rel) = text[after_open..].find(&close) {
                    let heading_text = &text[after_open..after_open + end_rel];
                    let heading_text = Self::strip_html_tags(heading_text);
                    let line_start = text[..abs_start].matches('\n').count();
                    sections.push(DocumentSection {
                        title: Some(heading_text),
                        content: String::new(),
                        level: *level,
                        line_start,
                        line_end: line_start + 1,
                    });
                    offset = after_open + end_rel + close.len();
                    search = &text[offset..];
                } else {
                    break;
                }
            }
        }

        // Sort sections by line_start
        sections.sort_by_key(|s| s.line_start);

        // Add a body section with plain text
        if sections.is_empty() {
            sections.push(DocumentSection {
                title: None,
                content: plain.trim().to_string(),
                level: 0,
                line_start: 0,
                line_end: lines.len(),
            });
        }

        let wc = Self::word_count(&plain);
        let cc = plain.chars().count();
        let lang = Self::detect_language(&plain);

        ParsedDocument {
            format: DocumentFormat::Html,
            sections,
            metadata,
            word_count: wc,
            char_count: cc,
            detected_language: lang,
        }
    }

    /// Parse a JSON document, extracting string values and keys as content.
    pub fn parse_json(text: &str) -> ParsedDocument {
        let mut content = String::new();
        let mut in_string = false;
        let mut escape = false;
        let mut current_string = String::new();

        for ch in text.chars() {
            if escape {
                current_string.push(ch);
                escape = false;
                continue;
            }
            match ch {
                '\\' if in_string => escape = true,
                '"' => {
                    if in_string {
                        // end of string
                        content.push_str(&current_string);
                        content.push(' ');
                        current_string.clear();
                        in_string = false;
                    } else {
                        in_string = true;
                    }
                }
                _ if in_string => current_string.push(ch),
                _ => {}
            }
        }

        let wc = Self::word_count(&content);
        let cc = content.chars().count();
        let lang = Self::detect_language(&content);

        let section = DocumentSection {
            title: Some("JSON Content".to_string()),
            content: content.trim().to_string(),
            level: 0,
            line_start: 0,
            line_end: text.lines().count(),
        };

        ParsedDocument {
            format: DocumentFormat::Json,
            sections: vec![section],
            metadata: HashMap::new(),
            word_count: wc,
            char_count: cc,
            detected_language: lang,
        }
    }

    /// Parse a CSV document, treating the header row as section titles.
    pub fn parse_csv(text: &str) -> ParsedDocument {
        let mut sections: Vec<DocumentSection> = Vec::new();
        let mut lines = text.lines().enumerate();

        let headers: Vec<String> = if let Some((_, header_line)) = lines.next() {
            header_line.split(',').map(|s| s.trim().to_string()).collect()
        } else {
            return ParsedDocument {
                format: DocumentFormat::Csv,
                sections: vec![],
                metadata: HashMap::new(),
                word_count: 0,
                char_count: 0,
                detected_language: None,
            };
        };

        let mut rows: Vec<Vec<String>> = Vec::new();
        for (_, line) in lines {
            let row: Vec<String> = line.split(',').map(|s| s.trim().to_string()).collect();
            rows.push(row);
        }

        for (col_idx, header) in headers.iter().enumerate() {
            let col_content: Vec<String> = rows
                .iter()
                .filter_map(|r| r.get(col_idx).cloned())
                .filter(|s| !s.is_empty())
                .collect();
            let content = col_content.join(", ");
            sections.push(DocumentSection {
                title: Some(header.clone()),
                content,
                level: 1,
                line_start: 0,
                line_end: rows.len() + 1,
            });
        }

        let all_content: String = sections.iter().map(|s| s.content.as_str()).collect::<Vec<_>>().join(" ");
        let wc = Self::word_count(&all_content);
        let cc = text.chars().count();
        let lang = Self::detect_language(&all_content);

        ParsedDocument {
            format: DocumentFormat::Csv,
            sections,
            metadata: HashMap::new(),
            word_count: wc,
            char_count: cc,
            detected_language: lang,
        }
    }

    /// Auto-detect format and dispatch to the appropriate parser.
    pub fn parse(text: &str) -> ParsedDocument {
        match Self::detect_format(text) {
            DocumentFormat::Markdown => Self::parse_markdown(text),
            DocumentFormat::Html => Self::parse_html(text),
            DocumentFormat::Json => Self::parse_json(text),
            DocumentFormat::Csv => Self::parse_csv(text),
            _ => {
                // Plain text: one section
                let wc = Self::word_count(text);
                let cc = text.chars().count();
                let lang = Self::detect_language(text);
                let lines = text.lines().count();
                ParsedDocument {
                    format: DocumentFormat::PlainText,
                    sections: vec![DocumentSection {
                        title: None,
                        content: text.to_string(),
                        level: 0,
                        line_start: 0,
                        line_end: lines,
                    }],
                    metadata: HashMap::new(),
                    word_count: wc,
                    char_count: cc,
                    detected_language: lang,
                }
            }
        }
    }

    /// Extract all URLs (http/https) and Markdown-style links from text.
    pub fn extract_links(text: &str) -> Vec<String> {
        let mut links: Vec<String> = Vec::new();

        // Markdown links: [text](url)
        let mut rest = text;
        while let Some(open) = rest.find("](") {
            let after = &rest[open + 2..];
            if let Some(close) = after.find(')') {
                let url = &after[..close];
                if url.starts_with("http://") || url.starts_with("https://") || url.starts_with('/') {
                    links.push(url.to_string());
                }
                rest = &after[close + 1..];
            } else {
                break;
            }
        }

        // Bare URLs
        let words = text.split_whitespace();
        for word in words {
            let word = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '/' && c != ':' && c != '.' && c != '-' && c != '_' && c != '?' && c != '=' && c != '&' && c != '#' && c != '%');
            if word.starts_with("http://") || word.starts_with("https://") {
                if !links.contains(&word.to_string()) {
                    links.push(word.to_string());
                }
            }
        }

        links
    }

    /// Extract code blocks as (language, code) pairs.
    pub fn extract_code_blocks(text: &str) -> Vec<(String, String)> {
        let mut blocks: Vec<(String, String)> = Vec::new();
        let mut rest = text;

        while let Some(start) = rest.find("```") {
            let after_fence = &rest[start + 3..];
            // Language identifier is on the first line after ```
            let newline = after_fence.find('\n').unwrap_or(after_fence.len());
            let lang = after_fence[..newline].trim().to_string();
            let code_start = newline + 1;
            if code_start >= after_fence.len() {
                break;
            }
            let code_rest = &after_fence[code_start..];
            if let Some(end) = code_rest.find("```") {
                let code = code_rest[..end].to_string();
                blocks.push((lang, code));
                rest = &code_rest[end + 3..];
            } else {
                break;
            }
        }

        blocks
    }

    /// Simple heuristic language detection based on common words.
    pub fn detect_language(text: &str) -> Option<String> {
        let lower = text.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();

        let score = |list: &[&str]| -> usize {
            words.iter().filter(|w| list.contains(w)).count()
        };

        let english = ["the", "is", "are", "was", "were", "and", "or", "in", "of", "to", "a", "an", "it", "that", "this"];
        let french = ["le", "la", "les", "est", "sont", "et", "ou", "dans", "de", "du", "un", "une", "il", "elle", "que"];
        let spanish = ["el", "la", "los", "las", "es", "son", "y", "o", "en", "de", "un", "una", "que", "se", "no"];
        let german = ["der", "die", "das", "ist", "sind", "und", "oder", "in", "von", "zu", "ein", "eine", "er", "sie", "es"];
        // Chinese: look for CJK characters
        let chinese_chars = text.chars().filter(|&c| c >= '\u{4E00}' && c <= '\u{9FFF}').count();

        let scores = [
            ("English", score(&english)),
            ("French", score(&french)),
            ("Spanish", score(&spanish)),
            ("German", score(&german)),
        ];

        if chinese_chars > 5 {
            return Some("Chinese".to_string());
        }

        let best = scores.iter().max_by_key(|&&(_, s)| s);
        if let Some(&(lang, s)) = best {
            if s > 0 {
                return Some(lang.to_string());
            }
        }
        None
    }

    /// Count words in text.
    pub fn word_count(text: &str) -> usize {
        text.split_whitespace().count()
    }

    /// Strip HTML tags from a string.
    pub fn strip_html_tags(html: &str) -> String {
        let mut result = String::new();
        let mut in_tag = false;
        for ch in html.chars() {
            match ch {
                '<' => in_tag = true,
                '>' => {
                    in_tag = false;
                    result.push(' ');
                }
                _ if !in_tag => result.push(ch),
                _ => {}
            }
        }
        // Collapse multiple spaces
        let mut out = String::new();
        let mut prev_space = false;
        for ch in result.chars() {
            if ch.is_whitespace() {
                if !prev_space {
                    out.push(' ');
                }
                prev_space = true;
            } else {
                out.push(ch);
                prev_space = false;
            }
        }
        out
    }

    /// Split text into sentences.
    pub fn split_sentences(text: &str) -> Vec<String> {
        let mut sentences: Vec<String> = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            current.push(ch);
            if ch == '.' || ch == '!' || ch == '?' {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current.clear();
            }
        }

        let trimmed = current.trim().to_string();
        if !trimmed.is_empty() {
            sentences.push(trimmed);
        }

        sentences
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_format_markdown() {
        let md = "# Hello\n\nSome **bold** text.";
        assert_eq!(DocumentParser::detect_format(md), DocumentFormat::Markdown);
    }

    #[test]
    fn test_detect_format_json() {
        let json = r#"{"key": "value"}"#;
        assert_eq!(DocumentParser::detect_format(json), DocumentFormat::Json);
    }

    #[test]
    fn test_word_count() {
        assert_eq!(DocumentParser::word_count("hello world foo"), 3);
    }

    #[test]
    fn test_strip_html_tags() {
        let html = "<h1>Title</h1><p>Body text.</p>";
        let plain = DocumentParser::strip_html_tags(html);
        assert!(plain.contains("Title"));
        assert!(plain.contains("Body text."));
    }

    #[test]
    fn test_extract_links() {
        let text = "Visit [docs](https://example.com) for more info. Also see https://rust-lang.org";
        let links = DocumentParser::extract_links(text);
        assert!(!links.is_empty());
    }

    #[test]
    fn test_detect_language_english() {
        let text = "The quick brown fox is and the lazy dog was in the field";
        assert_eq!(DocumentParser::detect_language(text), Some("English".to_string()));
    }

    #[test]
    fn test_parse_markdown() {
        let md = "# Title\n\nBody text\n\n## Section\n\nMore content.";
        let doc = DocumentParser::parse_markdown(md);
        assert_eq!(doc.format, DocumentFormat::Markdown);
        assert!(!doc.sections.is_empty());
    }

    #[test]
    fn test_extract_code_blocks() {
        let md = "Some text\n```rust\nfn main() {}\n```\nMore text";
        let blocks = DocumentParser::extract_code_blocks(md);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].0, "rust");
    }

    #[test]
    fn test_split_sentences() {
        let text = "Hello world. How are you? I am fine!";
        let sents = DocumentParser::split_sentences(text);
        assert_eq!(sents.len(), 3);
    }
}
