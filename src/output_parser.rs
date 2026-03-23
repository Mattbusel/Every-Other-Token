//! Structured output parser for LLM responses.
//! Extracts JSON objects, code blocks, lists, and key-value pairs.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum ParsedOutput {
    JsonObject(serde_json::Value),
    CodeBlock { language: String, code: String },
    NumberedList(Vec<String>),
    BulletList(Vec<String>),
    KeyValuePairs(HashMap<String, String>),
    PlainText(String),
}

#[derive(Debug, Clone, Default)]
pub struct ParserStats {
    pub json_count: usize,
    pub code_block_count: usize,
    pub list_count: usize,
}

// ---------------------------------------------------------------------------
// OutputParser
// ---------------------------------------------------------------------------

pub struct OutputParser;

impl OutputParser {
    /// Extract all structured elements from an LLM response, in order.
    pub fn parse(response: &str) -> Vec<ParsedOutput> {
        let mut results = Vec::new();

        // We process the text by scanning through it and extracting elements.
        let mut remaining = response;
        let mut plain_acc = String::new();

        while !remaining.is_empty() {
            // Try to find a fenced code block starting with ```
            if let Some(pos) = remaining.find("```") {
                // Accumulate text before the block
                let before = &remaining[..pos];
                plain_acc.push_str(before);

                remaining = &remaining[pos..];

                // Try to parse a fenced block
                if let Some((lang, code, after)) = extract_fenced_block(remaining) {
                    // Flush plain text accumulator
                    let plain = drain_plain(&mut plain_acc, response);
                    results.extend(plain);

                    // Try JSON first if language is "json"
                    if lang.eq_ignore_ascii_case("json") {
                        if let Ok(v) = serde_json::from_str(code.trim()) {
                            results.push(ParsedOutput::JsonObject(v));
                        } else {
                            results.push(ParsedOutput::CodeBlock {
                                language: lang.to_string(),
                                code: code.to_string(),
                            });
                        }
                    } else {
                        results.push(ParsedOutput::CodeBlock {
                            language: lang.to_string(),
                            code: code.to_string(),
                        });
                    }
                    remaining = after;
                } else {
                    // Malformed fence — consume the ``` and continue
                    plain_acc.push_str("```");
                    remaining = &remaining[3..];
                }
            } else {
                // No more fenced blocks
                plain_acc.push_str(remaining);
                remaining = "";
            }
        }

        // Process remaining plain text for lists, key-values, bare JSON, plain text
        let plain_text = plain_acc;
        if !plain_text.trim().is_empty() {
            results.extend(parse_plain_text(&plain_text));
        }

        results
    }

    /// Find and parse the first JSON block (```json ... ``` or bare object/array).
    pub fn extract_json(text: &str) -> Option<serde_json::Value> {
        // Try fenced ```json ... ``` first
        let lower = text.to_lowercase();
        if let Some(start) = lower.find("```json") {
            let after_fence = &text[start + 7..];
            if let Some(end) = after_fence.find("```") {
                let json_str = after_fence[..end].trim();
                if let Ok(v) = serde_json::from_str(json_str) {
                    return Some(v);
                }
            }
        }

        // Try bare ``` blocks that contain valid JSON
        let mut search = text;
        while let Some(pos) = search.find("```") {
            let after = &search[pos + 3..];
            // Skip language tag if any (up to newline)
            let code_start = after.find('\n').map(|n| n + 1).unwrap_or(0);
            let code_region = &after[code_start..];
            if let Some(end) = code_region.find("```") {
                let candidate = code_region[..end].trim();
                if let Ok(v) = serde_json::from_str(candidate) {
                    return Some(v);
                }
            }
            if let Some(next) = after.find("```") {
                search = &after[next..];
            } else {
                break;
            }
        }

        // Scan for bare { or [ at start of a token boundary
        find_bare_json(text)
    }

    /// Extract all fenced code blocks as (language, code) pairs.
    pub fn extract_code_blocks(text: &str) -> Vec<(String, String)> {
        let mut results = Vec::new();
        let mut remaining = text;
        while let Some(pos) = remaining.find("```") {
            remaining = &remaining[pos..];
            if let Some((lang, code, after)) = extract_fenced_block(remaining) {
                results.push((lang.to_string(), code.to_string()));
                remaining = after;
            } else {
                // Skip this ``` and keep scanning
                remaining = &remaining[3..];
            }
        }
        results
    }

    /// Extract numbered or bulleted list items from text.
    pub fn extract_list(text: &str) -> Vec<String> {
        let mut items = Vec::new();
        for line in text.lines() {
            let trimmed = line.trim();
            // Numbered: "1. ", "2) ", etc.
            if let Some(item) = strip_numbered(trimmed) {
                items.push(item.trim().to_string());
            }
            // Bullet: "- ", "* ", "+ ", "• "
            else if let Some(item) = strip_bullet(trimmed) {
                items.push(item.trim().to_string());
            }
        }
        items
    }

    /// Extract "Key: value" or "key = value" patterns.
    pub fn extract_key_values(text: &str) -> HashMap<String, String> {
        let mut map = HashMap::new();
        for line in text.lines() {
            let trimmed = line.trim();
            // Try "Key: value"
            if let Some(idx) = trimmed.find(':') {
                let key = trimmed[..idx].trim();
                let val = trimmed[idx + 1..].trim();
                if !key.is_empty() && !val.is_empty() && !key.contains(' ') || key.len() <= 40 {
                    if !key.is_empty() && !val.is_empty() {
                        map.insert(key.to_string(), val.to_string());
                        continue;
                    }
                }
            }
            // Try "key = value"
            if let Some(idx) = trimmed.find('=') {
                let key = trimmed[..idx].trim();
                let val = trimmed[idx + 1..].trim();
                if !key.is_empty() && !val.is_empty() {
                    map.insert(key.to_string(), val.to_string());
                }
            }
        }
        map
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Try to parse a fenced block starting at `text` (which must start with ```).
/// Returns (language, code, rest_of_text) or None.
fn extract_fenced_block<'a>(text: &'a str) -> Option<(&'a str, &'a str, &'a str)> {
    if !text.starts_with("```") {
        return None;
    }
    let after_open = &text[3..];
    // Language tag is the rest of the first line
    let newline_pos = after_open.find('\n')?;
    let lang = after_open[..newline_pos].trim();
    let code_start = &after_open[newline_pos + 1..];
    // Find the closing ```
    let close_pos = code_start.find("```")?;
    let code = &code_start[..close_pos];
    let after = &code_start[close_pos + 3..];
    // Skip trailing newline after closing fence
    let after = after.strip_prefix('\n').unwrap_or(after);
    Some((lang, code, after))
}

/// Flush a plain text accumulator, extracting sub-elements.
fn drain_plain(acc: &mut String, _full: &str) -> Vec<ParsedOutput> {
    let text = std::mem::take(acc);
    if text.trim().is_empty() {
        return vec![];
    }
    parse_plain_text(&text)
}

/// Parse a block of plain text for lists, key-values, bare JSON, plain text.
fn parse_plain_text(text: &str) -> Vec<ParsedOutput> {
    let mut results = Vec::new();

    // Try bare JSON
    if let Some(v) = find_bare_json(text) {
        results.push(ParsedOutput::JsonObject(v));
        return results;
    }

    // Try numbered list
    let numbered: Vec<String> = text
        .lines()
        .filter_map(|l| strip_numbered(l.trim()).map(|s| s.trim().to_string()))
        .collect();
    if !numbered.is_empty() {
        results.push(ParsedOutput::NumberedList(numbered));
        return results;
    }

    // Try bullet list
    let bulleted: Vec<String> = text
        .lines()
        .filter_map(|l| strip_bullet(l.trim()).map(|s| s.trim().to_string()))
        .collect();
    if !bulleted.is_empty() {
        results.push(ParsedOutput::BulletList(bulleted));
        return results;
    }

    // Try key-value pairs
    let kv = OutputParser::extract_key_values(text);
    if kv.len() >= 2 {
        results.push(ParsedOutput::KeyValuePairs(kv));
        return results;
    }

    // Fall back to plain text
    let t = text.trim();
    if !t.is_empty() {
        results.push(ParsedOutput::PlainText(t.to_string()));
    }
    results
}

/// Strip a numbered list prefix like "1. " or "2) " and return the rest.
fn strip_numbered(s: &str) -> Option<&str> {
    let s = s.trim();
    let end = s.find(|c: char| !c.is_ascii_digit())?;
    if end == 0 {
        return None;
    }
    let rest = &s[end..];
    if rest.starts_with(". ") || rest.starts_with(") ") {
        Some(&rest[2..])
    } else {
        None
    }
}

/// Strip a bullet prefix and return the rest.
fn strip_bullet(s: &str) -> Option<&str> {
    let prefixes = ["- ", "* ", "+ ", "• "];
    for p in &prefixes {
        if s.starts_with(p) {
            return Some(&s[p.len()..]);
        }
    }
    None
}

/// Scan text for bare JSON object `{...}` or array `[...]`.
fn find_bare_json(text: &str) -> Option<serde_json::Value> {
    for (i, ch) in text.char_indices() {
        if ch == '{' || ch == '[' {
            // Try to parse from this position onwards with increasing length
            let candidate = &text[i..];
            // Find balanced close
            if let Some(v) = try_parse_json_prefix(candidate) {
                return Some(v);
            }
        }
    }
    None
}

/// Try to parse a JSON value from the start of `s`, trying from longest to shortest.
fn try_parse_json_prefix(s: &str) -> Option<serde_json::Value> {
    // Try full string first, then shrink
    let bytes = s.as_bytes();
    let (open, close) = if bytes[0] == b'{' { (b'{', b'}') } else { (b'[', b']') };
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate() {
        if escape {
            escape = false;
            continue;
        }
        if in_string {
            if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        if b == b'"' {
            in_string = true;
        } else if b == open {
            depth += 1;
        } else if b == close {
            depth -= 1;
            if depth == 0 {
                let candidate = &s[..=i];
                return serde_json::from_str(candidate).ok();
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_fenced() {
        let text = "Here is the data:\n```json\n{\"key\": \"value\", \"num\": 42}\n```\nDone.";
        let v = OutputParser::extract_json(text).unwrap();
        assert_eq!(v["key"], "value");
        assert_eq!(v["num"], 42);
    }

    #[test]
    fn test_extract_json_bare() {
        let text = "Result: {\"status\": \"ok\"}";
        let v = OutputParser::extract_json(text).unwrap();
        assert_eq!(v["status"], "ok");
    }

    #[test]
    fn test_extract_code_blocks() {
        let text = "```rust\nfn main() {}\n```\n```python\nprint('hi')\n```";
        let blocks = OutputParser::extract_code_blocks(text);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].0, "rust");
        assert!(blocks[0].1.contains("fn main"));
        assert_eq!(blocks[1].0, "python");
    }

    #[test]
    fn test_extract_list_numbered() {
        let text = "1. First item\n2. Second item\n3. Third item";
        let items = OutputParser::extract_list(text);
        assert_eq!(items, vec!["First item", "Second item", "Third item"]);
    }

    #[test]
    fn test_extract_list_bulleted() {
        let text = "- Apple\n- Banana\n* Cherry\n+ Date";
        let items = OutputParser::extract_list(text);
        assert_eq!(items, vec!["Apple", "Banana", "Cherry", "Date"]);
    }

    #[test]
    fn test_extract_key_values_colon() {
        let text = "Name: Alice\nAge: 30\nCity: Wonderland";
        let kv = OutputParser::extract_key_values(text);
        assert_eq!(kv.get("Name").map(String::as_str), Some("Alice"));
        assert_eq!(kv.get("Age").map(String::as_str), Some("30"));
    }

    #[test]
    fn test_extract_key_values_equals() {
        let text = "color = blue\nsize = large";
        let kv = OutputParser::extract_key_values(text);
        assert_eq!(kv.get("color").map(String::as_str), Some("blue"));
        assert_eq!(kv.get("size").map(String::as_str), Some("large"));
    }

    #[test]
    fn test_parse_mixed_response() {
        let text = "Here is code:\n```rust\nlet x = 1;\n```\nAnd a list:\n1. First\n2. Second";
        let outputs = OutputParser::parse(text);
        assert!(!outputs.is_empty());
        let has_code = outputs.iter().any(|o| matches!(o, ParsedOutput::CodeBlock { .. }));
        assert!(has_code, "Expected a code block in parsed output");
    }

    #[test]
    fn test_parse_json_block() {
        let text = "```json\n{\"result\": true}\n```";
        let outputs = OutputParser::parse(text);
        let has_json = outputs.iter().any(|o| matches!(o, ParsedOutput::JsonObject(_)));
        assert!(has_json);
    }

    #[test]
    fn test_parse_plain_text() {
        let text = "Just some plain text here.";
        let outputs = OutputParser::parse(text);
        assert!(outputs.iter().any(|o| matches!(o, ParsedOutput::PlainText(_))));
    }

    #[test]
    fn test_parse_empty() {
        let outputs = OutputParser::parse("");
        assert!(outputs.is_empty());
    }

    #[test]
    fn test_extract_json_array() {
        let text = "Result: [1, 2, 3]";
        let v = OutputParser::extract_json(text).unwrap();
        assert!(v.is_array());
        assert_eq!(v.as_array().unwrap().len(), 3);
    }

    #[test]
    fn test_parser_stats_counting() {
        let text = "```json\n{\"a\":1}\n```\n```rust\nfn f(){}\n```\n1. A\n2. B";
        let outputs = OutputParser::parse(text);
        let json_count = outputs.iter().filter(|o| matches!(o, ParsedOutput::JsonObject(_))).count();
        let code_count = outputs.iter().filter(|o| matches!(o, ParsedOutput::CodeBlock { .. })).count();
        assert_eq!(json_count, 1);
        assert_eq!(code_count, 1);
    }

    #[test]
    fn test_strip_numbered() {
        assert_eq!(strip_numbered("1. hello"), Some("hello"));
        assert_eq!(strip_numbered("42) world"), Some("world"));
        assert_eq!(strip_numbered("no prefix"), None);
    }

    #[test]
    fn test_strip_bullet() {
        assert_eq!(strip_bullet("- item"), Some("item"));
        assert_eq!(strip_bullet("* item"), Some("item"));
        assert_eq!(strip_bullet("plain"), None);
    }
}
