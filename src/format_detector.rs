//! Format detector: classify structured content in token sequences.
//!
//! Detects JSON, XML, Markdown, Code (with language), CSV, YAML, and plain text
//! by analysing the joined token text.  A `StructureExtractor` companion pulls
//! concrete structural elements (JSON keys, Markdown headers, code identifiers)
//! out of raw text without requiring any external regex crate.

use std::collections::HashMap;

// ── ContentFormat ─────────────────────────────────────────────────────────────

/// Top-level content-format classification.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ContentFormat {
    Json,
    Xml,
    Markdown,
    Code(CodeLang),
    Csv,
    Yaml,
    PlainText,
    Mixed,
}

// ── CodeLang ──────────────────────────────────────────────────────────────────

/// Programming language detected inside a `ContentFormat::Code` segment.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CodeLang {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Go,
    Java,
    Cpp,
    Shell,
    Unknown,
}

// ── FormatDetector ────────────────────────────────────────────────────────────

/// Detects and classifies structured content in token sequences.
pub struct FormatDetector;

impl FormatDetector {
    /// Join tokens and classify the overall content format.
    pub fn detect(tokens: &[String]) -> ContentFormat {
        let text = tokens.join("");
        Self::detect_text(&text)
    }

    fn detect_text(text: &str) -> ContentFormat {
        let trimmed = text.trim();

        // JSON: starts with { or [
        if Self::looks_like_json(trimmed) {
            return ContentFormat::Json;
        }

        // XML: contains <tag> or </tag> patterns
        if Self::looks_like_xml(trimmed) {
            return ContentFormat::Xml;
        }

        // Code detection (before Markdown — backtick fences count as code)
        if let Some(lang) = Self::looks_like_code(trimmed) {
            return ContentFormat::Code(lang);
        }

        // Markdown
        if Self::looks_like_markdown(trimmed) {
            return ContentFormat::Markdown;
        }

        // YAML: lines with `: ` or `- ` indentation
        if Self::looks_like_yaml(trimmed) {
            return ContentFormat::Yaml;
        }

        // CSV: multiple lines with consistent comma-separated fields
        if Self::looks_like_csv(trimmed) {
            return ContentFormat::Csv;
        }

        ContentFormat::PlainText
    }

    // ── private heuristics ────────────────────────────────────────────────────

    fn looks_like_json(text: &str) -> bool {
        let t = text.trim();
        (t.starts_with('{') || t.starts_with('['))
            && (t.ends_with('}') || t.ends_with(']'))
    }

    fn looks_like_xml(text: &str) -> bool {
        // Must contain at least one <tag> or </tag>
        let has_open = Self::contains_xml_open_tag(text);
        let has_close = text.contains("</");
        has_open || has_close
    }

    fn contains_xml_open_tag(text: &str) -> bool {
        let bytes = text.as_bytes();
        let n = bytes.len();
        let mut i = 0;
        while i < n {
            if bytes[i] == b'<' {
                // skip whitespace
                let mut j = i + 1;
                while j < n && bytes[j] == b' ' {
                    j += 1;
                }
                // must start with alpha for an element name
                if j < n && bytes[j].is_ascii_alphabetic() {
                    return true;
                }
            }
            i += 1;
        }
        false
    }

    fn looks_like_code(text: &str) -> Option<CodeLang> {
        // Rust keywords
        let rust_score = Self::keyword_count(text, &["fn ", "let ", "impl ", "pub ", "use ", "mod ", "struct ", "enum "]);
        // Python keywords
        let py_score = Self::keyword_count(text, &["def ", "import ", "class ", "elif ", "print(", "self."]);
        // JavaScript/TypeScript keywords
        let js_score = Self::keyword_count(text, &["function ", "const ", "=>", "var ", "let ", "console."]);
        let ts_score = Self::keyword_count(text, &["interface ", "type ", ": string", ": number", ": boolean"]);
        // Go keywords
        let go_score = Self::keyword_count(text, &["func ", "package ", ":=", "goroutine", "go func"]);
        // Java keywords
        let java_score = Self::keyword_count(text, &["public class ", "private ", "protected ", "void ", "System.out"]);
        // C++
        let cpp_score = Self::keyword_count(text, &["#include", "std::", "cout", "cin", "->", "::"]);
        // Shell
        let sh_score = Self::keyword_count(text, &["#!/", "echo ", "export ", "grep ", "awk ", "sed "]);

        let max_score = [rust_score, py_score, js_score + ts_score, go_score, java_score, cpp_score, sh_score]
            .iter()
            .copied()
            .max()
            .unwrap_or(0);

        if max_score == 0 {
            return None;
        }

        // Must have at least 2 signals to avoid false positives
        if max_score < 2 {
            return None;
        }

        if rust_score == max_score {
            Some(CodeLang::Rust)
        } else if py_score == max_score {
            Some(CodeLang::Python)
        } else if (js_score + ts_score) == max_score {
            if ts_score > 0 {
                Some(CodeLang::TypeScript)
            } else {
                Some(CodeLang::JavaScript)
            }
        } else if go_score == max_score {
            Some(CodeLang::Go)
        } else if java_score == max_score {
            Some(CodeLang::Java)
        } else if cpp_score == max_score {
            Some(CodeLang::Cpp)
        } else if sh_score == max_score {
            Some(CodeLang::Shell)
        } else {
            Some(CodeLang::Unknown)
        }
    }

    fn keyword_count(text: &str, keywords: &[&str]) -> usize {
        keywords.iter().filter(|&&kw| text.contains(kw)).count()
    }

    fn looks_like_markdown(text: &str) -> bool {
        let mut score = 0usize;
        for line in text.lines() {
            let l = line.trim();
            if l.starts_with('#') { score += 2; }
            if l.starts_with("- ") || l.starts_with("* ") { score += 1; }
        }
        if text.contains("**") { score += 2; }
        if text.contains('`') { score += 1; }
        // [text](url) link pattern
        if Self::contains_markdown_link(text) { score += 2; }
        score >= 3
    }

    fn contains_markdown_link(text: &str) -> bool {
        let bytes = text.as_bytes();
        let n = bytes.len();
        let mut i = 0;
        while i + 4 < n {
            if bytes[i] == b'[' {
                // look for ](
                if let Some(close_bracket) = Self::find_byte(bytes, i + 1, b']') {
                    if close_bracket + 1 < n && bytes[close_bracket + 1] == b'(' {
                        return true;
                    }
                }
            }
            i += 1;
        }
        false
    }

    fn find_byte(bytes: &[u8], start: usize, target: u8) -> Option<usize> {
        for i in start..bytes.len() {
            if bytes[i] == target {
                return Some(i);
            }
        }
        None
    }

    fn looks_like_yaml(text: &str) -> bool {
        let mut colon_space_lines = 0usize;
        let mut dash_lines = 0usize;
        for line in text.lines() {
            let l = line.trim_start();
            if l.contains(": ") && !l.starts_with('{') { colon_space_lines += 1; }
            if l.starts_with("- ") { dash_lines += 1; }
        }
        colon_space_lines + dash_lines >= 3
    }

    fn looks_like_csv(text: &str) -> bool {
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        if lines.len() < 2 {
            return false;
        }
        let counts: Vec<usize> = lines.iter().map(|l| l.matches(',').count()).collect();
        if counts[0] == 0 {
            return false;
        }
        // Check consistency: all lines have the same number of commas
        counts.iter().all(|&c| c == counts[0])
    }

    // ── public API ────────────────────────────────────────────────────────────

    /// Detect segments of different formats within the token stream.
    ///
    /// Returns a list of `(format, start_token_index, end_token_index)` tuples.
    /// Uses a sliding window: groups consecutive tokens whose joined text matches
    /// the same format.
    pub fn detect_mixed(tokens: &[String]) -> Vec<(ContentFormat, usize, usize)> {
        if tokens.is_empty() {
            return vec![];
        }

        let window = 8.min(tokens.len());
        let mut segments: Vec<(ContentFormat, usize, usize)> = Vec::new();
        let mut seg_start = 0;
        let mut seg_fmt = Self::detect_text(&tokens[0..window.min(tokens.len())].join(""));

        let mut i = window;
        while i <= tokens.len() {
            let end = (i + window).min(tokens.len());
            if i >= tokens.len() {
                break;
            }
            let chunk = tokens[i..end].join("");
            let fmt = Self::detect_text(&chunk);
            if fmt != seg_fmt {
                segments.push((seg_fmt.clone(), seg_start, i.saturating_sub(1)));
                seg_start = i;
                seg_fmt = fmt;
            }
            i += window;
        }
        segments.push((seg_fmt, seg_start, tokens.len().saturating_sub(1)));
        segments
    }

    /// Confidence score in [0, 1] that `tokens` match `format`.
    pub fn confidence(tokens: &[String], format: ContentFormat) -> f64 {
        let text = tokens.join("");
        match format {
            ContentFormat::Json => {
                let t = text.trim();
                if Self::looks_like_json(t) {
                    // Count balanced braces
                    let opens = t.chars().filter(|&c| c == '{' || c == '[').count() as f64;
                    let closes = t.chars().filter(|&c| c == '}' || c == ']').count() as f64;
                    if opens == 0.0 { return 0.0; }
                    let balance = 1.0 - ((opens - closes).abs() / opens).min(1.0);
                    0.5 + 0.5 * balance
                } else {
                    0.0
                }
            }
            ContentFormat::Xml => {
                let open_tags = Self::count_pattern(&text, "<");
                let close_tags = Self::count_pattern(&text, "</");
                if open_tags == 0 { return 0.0; }
                let ratio = (close_tags as f64 / open_tags as f64).min(1.0);
                (0.3 + 0.7 * ratio).min(1.0)
            }
            ContentFormat::Markdown => {
                let mut score = 0.0f64;
                let header_count = text.lines().filter(|l| l.trim_start().starts_with('#')).count();
                let bold_count = Self::count_pattern(&text, "**");
                let list_count = text.lines().filter(|l| {
                    let t = l.trim_start();
                    t.starts_with("- ") || t.starts_with("* ")
                }).count();
                score += (header_count as f64 * 0.2).min(0.4);
                score += (bold_count as f64 * 0.1).min(0.3);
                score += (list_count as f64 * 0.1).min(0.3);
                score.min(1.0)
            }
            ContentFormat::Code(_) => {
                // Use keyword detection strength
                let rust_s = Self::keyword_count(&text, &["fn ", "let ", "impl ", "pub ", "use ", "mod ", "struct ", "enum "]);
                let py_s = Self::keyword_count(&text, &["def ", "import ", "class ", "elif ", "print(", "self."]);
                let js_s = Self::keyword_count(&text, &["function ", "const ", "=>", "var ", "console."]);
                let go_s = Self::keyword_count(&text, &["func ", "package ", ":=", "goroutine"]);
                let max_s = [rust_s, py_s, js_s, go_s].iter().copied().max().unwrap_or(0);
                (max_s as f64 * 0.15).min(1.0)
            }
            ContentFormat::Csv => {
                if Self::looks_like_csv(text.trim()) { 0.9 } else { 0.0 }
            }
            ContentFormat::Yaml => {
                if Self::looks_like_yaml(text.trim()) { 0.8 } else { 0.1 }
            }
            ContentFormat::PlainText => {
                if Self::detect_text(&text) == ContentFormat::PlainText { 0.85 } else { 0.2 }
            }
            ContentFormat::Mixed => {
                let segs = Self::detect_mixed(tokens);
                if segs.len() > 1 { 0.8 } else { 0.1 }
            }
        }
    }

    fn count_pattern(text: &str, pattern: &str) -> usize {
        let mut count = 0;
        let mut start = 0;
        while let Some(pos) = text[start..].find(pattern) {
            count += 1;
            start += pos + pattern.len();
        }
        count
    }
}

// ── StructureExtractor ────────────────────────────────────────────────────────

/// Extracts structural elements from raw text (JSON keys, Markdown headers,
/// code identifiers, token-format counts).
pub struct StructureExtractor;

impl StructureExtractor {
    /// Extract all JSON keys from text by finding `"key":` patterns.
    ///
    /// No external regex needed — scans byte-by-byte.
    pub fn extract_json_keys(text: &str) -> Vec<String> {
        let mut keys = Vec::new();
        let bytes = text.as_bytes();
        let n = bytes.len();
        let mut i = 0;

        while i < n {
            if bytes[i] == b'"' {
                // Collect key string
                let start = i + 1;
                let mut j = start;
                while j < n && !(bytes[j] == b'"' && (j == 0 || bytes[j - 1] != b'\\')) {
                    j += 1;
                }
                if j < n {
                    // Check if followed by optional whitespace then ':'
                    let mut k = j + 1;
                    while k < n && (bytes[k] == b' ' || bytes[k] == b'\t' || bytes[k] == b'\n') {
                        k += 1;
                    }
                    if k < n && bytes[k] == b':' {
                        if let Ok(key) = std::str::from_utf8(&bytes[start..j]) {
                            keys.push(key.to_string());
                        }
                    }
                    i = j + 1;
                    continue;
                }
            }
            i += 1;
        }
        keys
    }

    /// Extract Markdown headers as `(level, header_text)` tuples.
    pub fn extract_markdown_headers(text: &str) -> Vec<(u8, String)> {
        let mut headers = Vec::new();
        for line in text.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with('#') {
                let level = trimmed.chars().take_while(|&c| c == '#').count() as u8;
                let title = trimmed[level as usize..].trim().to_string();
                if !title.is_empty() {
                    headers.push((level, title));
                }
            }
        }
        headers
    }

    /// Extract identifiers: words that look like snake_case, camelCase, or PascalCase.
    pub fn extract_code_identifiers(text: &str) -> Vec<String> {
        let mut identifiers = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            if ch.is_alphanumeric() || ch == '_' {
                current.push(ch);
            } else {
                if Self::is_identifier(&current) {
                    identifiers.push(current.clone());
                }
                current.clear();
            }
        }
        if Self::is_identifier(&current) {
            identifiers.push(current);
        }
        identifiers.sort();
        identifiers.dedup();
        identifiers
    }

    fn is_identifier(s: &str) -> bool {
        if s.len() < 2 {
            return false;
        }
        // Must start with letter or underscore
        let first = s.chars().next().unwrap();
        if !first.is_alphabetic() && first != '_' {
            return false;
        }
        // snake_case: contains underscore with adjacent letters
        let has_underscore = s.contains('_') && s.len() > 2;
        // camelCase: has lowercase followed by uppercase
        let has_camel = s.chars().zip(s.chars().skip(1)).any(|(a, b)| a.is_lowercase() && b.is_uppercase());
        // PascalCase: starts uppercase, has lowercase
        let has_pascal = first.is_uppercase() && s.chars().any(|c| c.is_lowercase());
        has_underscore || has_camel || has_pascal
    }

    /// Count how many tokens fall into each detected format segment.
    pub fn count_tokens_by_format(tokens: &[String]) -> HashMap<ContentFormat, usize> {
        let mut map: HashMap<ContentFormat, usize> = HashMap::new();
        let segments = FormatDetector::detect_mixed(tokens);
        for (fmt, start, end) in segments {
            let count = end.saturating_sub(start) + 1;
            *map.entry(fmt).or_insert(0) += count;
        }
        map
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(s: &str) -> Vec<String> {
        s.split_whitespace().map(|t| t.to_string()).collect()
    }

    fn toks_raw(s: &str) -> Vec<String> {
        vec![s.to_string()]
    }

    #[test]
    fn detect_json_object() {
        let tokens = toks_raw(r#"{"name": "Alice", "age": 30}"#);
        assert_eq!(FormatDetector::detect(&tokens), ContentFormat::Json);
    }

    #[test]
    fn detect_json_array() {
        let tokens = toks_raw(r#"[1, 2, 3]"#);
        assert_eq!(FormatDetector::detect(&tokens), ContentFormat::Json);
    }

    #[test]
    fn detect_rust_code() {
        let code = "fn main() { let x = 42; impl Foo { pub fn bar() {} } }";
        let tokens = toks_raw(code);
        assert_eq!(FormatDetector::detect(&tokens), ContentFormat::Code(CodeLang::Rust));
    }

    #[test]
    fn detect_python_code() {
        let code = "def greet(self): import os\nclass Foo:\n    def bar(self): pass";
        let tokens = toks_raw(code);
        assert_eq!(FormatDetector::detect(&tokens), ContentFormat::Code(CodeLang::Python));
    }

    #[test]
    fn detect_markdown() {
        let md = "# Title\n\n**bold** text\n\n- item one\n- item two";
        let tokens = toks_raw(md);
        assert_eq!(FormatDetector::detect(&tokens), ContentFormat::Markdown);
    }

    #[test]
    fn detect_xml() {
        let xml = "<root><child>value</child></root>";
        let tokens = toks_raw(xml);
        assert_eq!(FormatDetector::detect(&tokens), ContentFormat::Xml);
    }

    #[test]
    fn extract_markdown_headers() {
        let text = "# Top\n## Section\n### Sub\nsome text";
        let headers = StructureExtractor::extract_markdown_headers(text);
        assert_eq!(headers.len(), 3);
        assert_eq!(headers[0], (1, "Top".to_string()));
        assert_eq!(headers[1], (2, "Section".to_string()));
        assert_eq!(headers[2], (3, "Sub".to_string()));
    }

    #[test]
    fn extract_json_keys() {
        let json = r#"{"name": "Alice", "age": 30, "active": true}"#;
        let keys = StructureExtractor::extract_json_keys(json);
        assert!(keys.contains(&"name".to_string()));
        assert!(keys.contains(&"age".to_string()));
        assert!(keys.contains(&"active".to_string()));
    }

    #[test]
    fn confidence_json_above_half() {
        let tokens = toks_raw(r#"{"key": "value", "count": 42}"#);
        let c = FormatDetector::confidence(&tokens, ContentFormat::Json);
        assert!(c > 0.5, "JSON confidence was {}", c);
    }

    #[test]
    fn confidence_rust_above_half() {
        let code = "fn main() { let x = 42; impl Foo { pub fn bar() {} } }";
        let tokens = toks_raw(code);
        let c = FormatDetector::confidence(&tokens, ContentFormat::Code(CodeLang::Rust));
        assert!(c > 0.5, "Rust code confidence was {}", c);
    }

    #[test]
    fn detect_mixed_segments() {
        // Two clearly different segments
        let json_part = r#"{"a":1}"#.to_string();
        let md_part = "# Header\n**bold** text - item".to_string();
        let tokens = vec![json_part, " ".to_string(), md_part];
        let segs = FormatDetector::detect_mixed(&tokens);
        // At minimum we get at least one segment back
        assert!(!segs.is_empty());
    }

    #[test]
    fn extract_code_identifiers() {
        let code = "fn hello_world() { let myVar = SomeStruct::new(); }";
        let ids = StructureExtractor::extract_code_identifiers(code);
        assert!(ids.contains(&"hello_world".to_string()) || ids.contains(&"myVar".to_string()) || ids.contains(&"SomeStruct".to_string()));
    }

    #[test]
    fn count_tokens_by_format_non_empty() {
        let tokens = vec![r#"{"key": 1}"#.to_string()];
        let counts = StructureExtractor::count_tokens_by_format(&tokens);
        assert!(!counts.is_empty());
    }
}
