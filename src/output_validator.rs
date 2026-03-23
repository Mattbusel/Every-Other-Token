//! LLM output validation and schema checking.
//!
//! Provides rule-based validation of model outputs, including length checks,
//! keyword presence, JSON/code validity, and a weighted compliance score.

// ---------------------------------------------------------------------------
// ValidationRule
// ---------------------------------------------------------------------------

/// A single validation rule that can be applied to an output string.
#[derive(Debug, Clone)]
pub enum ValidationRule {
    MinLength(usize),
    MaxLength(usize),
    ContainsKeyword(String),
    NotContainsKeyword(String),
    MatchesRegexSimple(String),
    IsValidJson,
    IsValidCode { language: String },
    StartsWithCapital,
    EndsWithPunctuation,
    WordCount { min: usize, max: usize },
}

// ---------------------------------------------------------------------------
// ValidationResult
// ---------------------------------------------------------------------------

/// The result of validating an output against a schema.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether all rules passed.
    pub passed: bool,
    /// Descriptions of individual rule violations.
    pub violations: Vec<String>,
    /// Compliance score in [0.0, 1.0].
    pub score: f64,
    /// Suggested fixes for the violations found.
    pub suggestions: Vec<String>,
}

// ---------------------------------------------------------------------------
// OutputSchema
// ---------------------------------------------------------------------------

/// A schema describing the expected shape of an LLM output.
#[derive(Debug, Clone, Default)]
pub struct OutputSchema {
    pub rules: Vec<ValidationRule>,
    pub required_sections: Vec<String>,
    pub forbidden_phrases: Vec<String>,
}

// ---------------------------------------------------------------------------
// OutputValidator
// ---------------------------------------------------------------------------

/// Validates LLM outputs against an [`OutputSchema`].
#[derive(Debug, Default)]
pub struct OutputValidator;

impl OutputValidator {
    pub fn new() -> Self {
        Self
    }

    /// Validate `output` against `schema` and return a [`ValidationResult`].
    pub fn validate(&self, output: &str, schema: &OutputSchema) -> ValidationResult {
        let mut violations: Vec<String> = Vec::new();
        let mut suggestions: Vec<String> = Vec::new();
        let total_rules = schema.rules.len()
            + schema.required_sections.len()
            + schema.forbidden_phrases.len();

        // --- check ValidationRules ---
        for rule in &schema.rules {
            match rule {
                ValidationRule::MinLength(n) => {
                    if output.len() < *n {
                        violations.push(format!(
                            "Output length {} is below minimum {}",
                            output.len(),
                            n
                        ));
                        suggestions.push(format!("Expand the output to at least {} characters.", n));
                    }
                }
                ValidationRule::MaxLength(n) => {
                    if output.len() > *n {
                        violations.push(format!(
                            "Output length {} exceeds maximum {}",
                            output.len(),
                            n
                        ));
                        suggestions.push(format!("Shorten the output to at most {} characters.", n));
                    }
                }
                ValidationRule::ContainsKeyword(kw) => {
                    if !output.to_lowercase().contains(&kw.to_lowercase()) {
                        violations.push(format!("Missing required keyword: '{}'", kw));
                        suggestions.push(format!("Include the keyword '{}' in the output.", kw));
                    }
                }
                ValidationRule::NotContainsKeyword(kw) => {
                    if output.to_lowercase().contains(&kw.to_lowercase()) {
                        violations.push(format!("Forbidden keyword present: '{}'", kw));
                        suggestions.push(format!("Remove all occurrences of '{}'.", kw));
                    }
                }
                ValidationRule::MatchesRegexSimple(pattern) => {
                    if !Self::simple_regex_match(output, pattern) {
                        violations.push(format!("Output does not match pattern: '{}'", pattern));
                        suggestions.push(format!("Ensure output matches the pattern '{}'.", pattern));
                    }
                }
                ValidationRule::IsValidJson => {
                    if !Self::validate_json(output) {
                        violations.push("Output is not valid JSON (unbalanced braces/brackets).".into());
                        suggestions.push("Ensure the output is valid, balanced JSON.".into());
                    }
                }
                ValidationRule::IsValidCode { language } => {
                    if !Self::validate_code_block(output, language) {
                        violations.push(format!("Output does not appear to be valid {} code.", language));
                        suggestions.push(format!("Include recognisable {} code constructs.", language));
                    }
                }
                ValidationRule::StartsWithCapital => {
                    if !output.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                        violations.push("Output does not start with a capital letter.".into());
                        suggestions.push("Capitalise the first character of the output.".into());
                    }
                }
                ValidationRule::EndsWithPunctuation => {
                    let last = output.trim_end().chars().last();
                    let ok = last.map(|c| matches!(c, '.' | '!' | '?' | ':')).unwrap_or(false);
                    if !ok {
                        violations.push("Output does not end with punctuation (. ! ? :).".into());
                        suggestions.push("Ensure the output ends with a punctuation mark.".into());
                    }
                }
                ValidationRule::WordCount { min, max } => {
                    let wc = output.split_whitespace().count();
                    if wc < *min {
                        violations.push(format!("Word count {} below minimum {}.", wc, min));
                        suggestions.push(format!("Add more content to reach at least {} words.", min));
                    } else if wc > *max {
                        violations.push(format!("Word count {} exceeds maximum {}.", wc, max));
                        suggestions.push(format!("Trim content to at most {} words.", max));
                    }
                }
            }
        }

        // --- required sections ---
        let missing_sections = Self::check_required_sections(output, &schema.required_sections);
        for sec in &missing_sections {
            violations.push(format!("Required section missing: '{}'", sec));
            suggestions.push(format!("Add a section titled '{}'.", sec));
        }

        // --- forbidden phrases ---
        let found_phrases = Self::check_forbidden_phrases(output, &schema.forbidden_phrases);
        for phrase in &found_phrases {
            violations.push(format!("Forbidden phrase found: '{}'", phrase));
            suggestions.push(format!("Remove the phrase '{}'.", phrase));
        }

        let failed = violations.len();
        let score = if total_rules == 0 {
            1.0
        } else {
            let passed_count = total_rules.saturating_sub(failed);
            passed_count as f64 / total_rules as f64
        };

        ValidationResult {
            passed: violations.is_empty(),
            violations,
            score,
            suggestions,
        }
    }

    /// Check whether `text` has balanced `{}` and `[]` characters.
    pub fn validate_json(text: &str) -> bool {
        let mut brace_depth: i32 = 0;
        let mut bracket_depth: i32 = 0;
        let mut in_string = false;
        let mut escape_next = false;

        for ch in text.chars() {
            if escape_next {
                escape_next = false;
                continue;
            }
            if ch == '\\' && in_string {
                escape_next = true;
                continue;
            }
            if ch == '"' {
                in_string = !in_string;
                continue;
            }
            if in_string {
                continue;
            }
            match ch {
                '{' => brace_depth += 1,
                '}' => {
                    brace_depth -= 1;
                    if brace_depth < 0 {
                        return false;
                    }
                }
                '[' => bracket_depth += 1,
                ']' => {
                    bracket_depth -= 1;
                    if bracket_depth < 0 {
                        return false;
                    }
                }
                _ => {}
            }
        }
        brace_depth == 0 && bracket_depth == 0
    }

    /// Heuristically decide whether `text` looks like code in `language`.
    pub fn validate_code_block(text: &str, language: &str) -> bool {
        let lower_text = text.to_lowercase();
        let lower_lang = language.to_lowercase();
        match lower_lang.as_str() {
            "rust" => {
                lower_text.contains("fn ")
                    || lower_text.contains("let ")
                    || lower_text.contains("impl ")
                    || lower_text.contains("struct ")
                    || lower_text.contains("use ")
            }
            "python" => {
                lower_text.contains("def ")
                    || lower_text.contains("import ")
                    || lower_text.contains("class ")
                    || lower_text.contains("print(")
                    || lower_text.contains("if __name__")
            }
            "javascript" | "js" | "typescript" | "ts" => {
                lower_text.contains("function ")
                    || lower_text.contains("const ")
                    || lower_text.contains("let ")
                    || lower_text.contains("var ")
                    || lower_text.contains("=>")
                    || lower_text.contains("require(")
                    || lower_text.contains("import ")
            }
            "java" => {
                lower_text.contains("public ")
                    || lower_text.contains("class ")
                    || lower_text.contains("void ")
                    || lower_text.contains("import java")
            }
            "c" | "cpp" | "c++" => {
                lower_text.contains("#include")
                    || lower_text.contains("int main")
                    || lower_text.contains("void ")
                    || lower_text.contains("return ")
            }
            "sql" => {
                lower_text.contains("select ")
                    || lower_text.contains("insert ")
                    || lower_text.contains("update ")
                    || lower_text.contains("create table")
            }
            "html" => {
                lower_text.contains("<html")
                    || lower_text.contains("<div")
                    || lower_text.contains("<p>")
                    || lower_text.contains("<!doctype")
            }
            "json" => Self::validate_json(text),
            _ => {
                // Generic: look for common programming constructs
                lower_text.contains("(")
                    && lower_text.contains(")")
                    && (lower_text.contains("{") || lower_text.contains(":"))
            }
        }
    }

    /// Return sections from `sections` that are not present in `text`.
    pub fn check_required_sections(text: &str, sections: &[String]) -> Vec<String> {
        sections
            .iter()
            .filter(|sec| !text.to_lowercase().contains(&sec.to_lowercase()))
            .cloned()
            .collect()
    }

    /// Return forbidden phrases that are found in `text`.
    pub fn check_forbidden_phrases(text: &str, phrases: &[String]) -> Vec<String> {
        phrases
            .iter()
            .filter(|phrase| text.to_lowercase().contains(&phrase.to_lowercase()))
            .cloned()
            .collect()
    }

    /// Minimal glob-like pattern matching supporting `*`, `?`, `^` (start anchor), `$` (end anchor).
    pub fn simple_regex_match(text: &str, pattern: &str) -> bool {
        let (anchored_start, rest) = if pattern.starts_with('^') {
            (true, &pattern[1..])
        } else {
            (false, pattern)
        };
        let (anchored_end, pat) = if rest.ends_with('$') {
            (true, &rest[..rest.len() - 1])
        } else {
            (false, rest)
        };

        if anchored_start && anchored_end {
            return Self::glob_match(text, pat);
        }
        if anchored_start {
            return Self::glob_match_prefix(text, pat);
        }
        if anchored_end {
            return Self::glob_match_suffix(text, pat);
        }
        // Unanchored: try matching at every position
        Self::glob_match_anywhere(text, pat)
    }

    fn glob_match(text: &str, pattern: &str) -> bool {
        let t: Vec<char> = text.chars().collect();
        let p: Vec<char> = pattern.chars().collect();
        Self::glob_dp(&t, &p)
    }

    fn glob_match_prefix(text: &str, pattern: &str) -> bool {
        // Pattern must match from the start; append '*' to allow trailing text
        let mut p = pattern.to_string();
        p.push('*');
        Self::glob_match(text, &p)
    }

    fn glob_match_suffix(text: &str, pattern: &str) -> bool {
        let mut p = String::from("*");
        p.push_str(pattern);
        Self::glob_match(text, &p)
    }

    fn glob_match_anywhere(text: &str, pattern: &str) -> bool {
        let mut p = String::from("*");
        p.push_str(pattern);
        p.push('*');
        Self::glob_match(text, &p)
    }

    /// DP glob matching: `*` matches any sequence, `?` matches one character.
    fn glob_dp(text: &[char], pattern: &[char]) -> bool {
        let n = text.len();
        let m = pattern.len();
        // dp[i][j] = text[..i] matches pattern[..j]
        let mut dp = vec![vec![false; m + 1]; n + 1];
        dp[0][0] = true;
        for j in 1..=m {
            if pattern[j - 1] == '*' {
                dp[0][j] = dp[0][j - 1];
            }
        }
        for i in 1..=n {
            for j in 1..=m {
                match pattern[j - 1] {
                    '*' => dp[i][j] = dp[i - 1][j] || dp[i][j - 1],
                    '?' => dp[i][j] = dp[i - 1][j - 1],
                    c => dp[i][j] = dp[i - 1][j - 1] && text[i - 1] == c,
                }
            }
        }
        dp[n][m]
    }

    /// Compute a weighted compliance score for `text` against `schema`.
    pub fn score_output(text: &str, schema: &OutputSchema) -> f64 {
        let total = schema.rules.len()
            + schema.required_sections.len()
            + schema.forbidden_phrases.len();
        if total == 0 {
            return 1.0;
        }
        let validator = OutputValidator::new();
        let result = validator.validate(text, schema);
        result.score
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_schema(rules: Vec<ValidationRule>) -> OutputSchema {
        OutputSchema { rules, required_sections: vec![], forbidden_phrases: vec![] }
    }

    #[test]
    fn test_min_length_pass() {
        let v = OutputValidator::new();
        let schema = make_schema(vec![ValidationRule::MinLength(5)]);
        let r = v.validate("Hello world", &schema);
        assert!(r.passed);
    }

    #[test]
    fn test_min_length_fail() {
        let v = OutputValidator::new();
        let schema = make_schema(vec![ValidationRule::MinLength(100)]);
        let r = v.validate("Hi", &schema);
        assert!(!r.passed);
        assert!(!r.violations.is_empty());
    }

    #[test]
    fn test_json_valid() {
        assert!(OutputValidator::validate_json(r#"{"key": "value", "arr": [1,2,3]}"#));
    }

    #[test]
    fn test_json_invalid() {
        assert!(!OutputValidator::validate_json(r#"{"key": "value""#));
    }

    #[test]
    fn test_starts_with_capital() {
        let v = OutputValidator::new();
        let schema = make_schema(vec![ValidationRule::StartsWithCapital]);
        assert!(v.validate("Hello world", &schema).passed);
        assert!(!v.validate("hello world", &schema).passed);
    }

    #[test]
    fn test_ends_with_punctuation() {
        let v = OutputValidator::new();
        let schema = make_schema(vec![ValidationRule::EndsWithPunctuation]);
        assert!(v.validate("Hello world.", &schema).passed);
        assert!(!v.validate("Hello world", &schema).passed);
    }

    #[test]
    fn test_simple_regex_anchor() {
        assert!(OutputValidator::simple_regex_match("Hello world", "^Hello"));
        assert!(!OutputValidator::simple_regex_match("Say Hello", "^Hello"));
        assert!(OutputValidator::simple_regex_match("Hello world", "world$"));
        assert!(!OutputValidator::simple_regex_match("Hello world!", "world$"));
    }

    #[test]
    fn test_score_perfect() {
        let schema = OutputSchema::default();
        assert!((OutputValidator::score_output("anything", &schema) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_required_sections() {
        let schema = OutputSchema {
            rules: vec![],
            required_sections: vec!["Introduction".to_string(), "Conclusion".to_string()],
            forbidden_phrases: vec![],
        };
        let v = OutputValidator::new();
        let r = v.validate("Introduction\nSome text\nConclusion", &schema);
        assert!(r.passed);
        let r2 = v.validate("Introduction\nSome text", &schema);
        assert!(!r2.passed);
    }

    #[test]
    fn test_word_count() {
        let v = OutputValidator::new();
        let schema = make_schema(vec![ValidationRule::WordCount { min: 2, max: 5 }]);
        assert!(v.validate("one two three", &schema).passed);
        assert!(!v.validate("one", &schema).passed);
        assert!(!v.validate("one two three four five six", &schema).passed);
    }
}
