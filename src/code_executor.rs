//! Sandboxed code block extraction and simulation.
//!
//! Extracts fenced code blocks from Markdown text, simulates execution for
//! common patterns, and validates basic syntax properties.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Language
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Language {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Go,
    Bash,
    Unknown(String),
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Language::Rust => write!(f, "rust"),
            Language::Python => write!(f, "python"),
            Language::JavaScript => write!(f, "javascript"),
            Language::TypeScript => write!(f, "typescript"),
            Language::Go => write!(f, "go"),
            Language::Bash => write!(f, "bash"),
            Language::Unknown(s) => write!(f, "{}", s),
        }
    }
}

// ---------------------------------------------------------------------------
// CodeBlock
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CodeBlock {
    pub language: Language,
    pub code: String,
    pub start_line: usize,
    pub end_line: usize,
}

// ---------------------------------------------------------------------------
// ExecutionMode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionMode {
    Simulate,
    Extract,
    Validate,
}

// ---------------------------------------------------------------------------
// SimulationResult
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub block: CodeBlock,
    pub output: String,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub execution_ms: u64,
}

// ---------------------------------------------------------------------------
// CodeExtractor
// ---------------------------------------------------------------------------

pub struct CodeExtractor;

impl CodeExtractor {
    /// Extract all fenced code blocks from the given Markdown text.
    pub fn extract(text: &str) -> Vec<CodeBlock> {
        let mut blocks = Vec::new();
        let lines: Vec<&str> = text.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i].trim();
            if line.starts_with("```") {
                let tag = line.trim_start_matches('`').trim();
                let language = Self::detect_language(tag);
                let start_line = i + 1; // 1-based, line after the opening fence
                i += 1;

                let mut code_lines = Vec::new();
                while i < lines.len() {
                    let inner = lines[i];
                    if inner.trim() == "```" {
                        break;
                    }
                    code_lines.push(inner);
                    i += 1;
                }
                let end_line = i; // line of closing fence

                blocks.push(CodeBlock {
                    language,
                    code: code_lines.join("\n"),
                    start_line,
                    end_line,
                });
            }
            i += 1;
        }

        blocks
    }

    /// Detect programming language from a fence tag string.
    pub fn detect_language(tag: &str) -> Language {
        match tag.to_lowercase().trim() {
            "rust" | "rs" => Language::Rust,
            "python" | "py" => Language::Python,
            "javascript" | "js" => Language::JavaScript,
            "typescript" | "ts" => Language::TypeScript,
            "go" | "golang" => Language::Go,
            "bash" | "sh" | "shell" => Language::Bash,
            other => Language::Unknown(other.to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// CodeSimulator
// ---------------------------------------------------------------------------

pub struct CodeSimulator {
    pub mode: ExecutionMode,
}

impl CodeSimulator {
    pub fn new(mode: ExecutionMode) -> Self {
        Self { mode }
    }

    /// Simulate execution of a code block, producing plausible output.
    pub fn simulate(&self, block: &CodeBlock) -> SimulationResult {
        let start = std::time::Instant::now();
        let mut output = String::new();
        let mut errors = Vec::new();
        let warnings = Vec::new();

        match &block.language {
            Language::Rust => {
                if block.code.contains("panic!") {
                    errors.push("thread 'main' panicked".to_string());
                } else if block.code.contains("println!") {
                    output = Self::extract_rust_println(&block.code);
                } else {
                    output = "Execution simulated".to_string();
                }
            }
            Language::Python => {
                if block.code.contains("raise") {
                    errors.push("Traceback (most recent call last): RuntimeError".to_string());
                } else if block.code.contains("print(") {
                    output = Self::extract_python_print(&block.code);
                } else {
                    output = "Execution simulated".to_string();
                }
            }
            Language::Bash => {
                if block.code.contains("echo ") {
                    output = Self::extract_bash_echo(&block.code);
                } else if block.code.contains("ls") {
                    output = "[file list simulation]".to_string();
                } else {
                    output = "Execution simulated".to_string();
                }
            }
            _ => {
                output = "Execution simulated".to_string();
            }
        }

        SimulationResult {
            block: block.clone(),
            output,
            errors,
            warnings,
            execution_ms: start.elapsed().as_millis() as u64,
        }
    }

    fn extract_rust_println(code: &str) -> String {
        // Find println!("...") or println!("{}", ...) style
        let mut results = Vec::new();
        for line in code.lines() {
            let trimmed = line.trim();
            if let Some(start) = trimmed.find("println!(") {
                let after = &trimmed[start + 9..];
                // Extract the string literal
                if let Some(content) = Self::extract_string_arg(after) {
                    // Simplistic: just output the literal, stripping format specifiers
                    results.push(content);
                }
            }
        }
        results.join("\n")
    }

    fn extract_python_print(code: &str) -> String {
        let mut results = Vec::new();
        for line in code.lines() {
            let trimmed = line.trim();
            if let Some(start) = trimmed.find("print(") {
                let after = &trimmed[start + 6..];
                if let Some(content) = Self::extract_string_arg(after) {
                    results.push(content);
                }
            }
        }
        results.join("\n")
    }

    fn extract_bash_echo(code: &str) -> String {
        let mut results = Vec::new();
        for line in code.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("echo ") {
                let arg = trimmed[5..].trim().trim_matches('"').trim_matches('\'');
                results.push(arg.to_string());
            }
        }
        results.join("\n")
    }

    fn extract_string_arg(s: &str) -> Option<String> {
        // Find first " or ' and extract until the matching close
        let s = s.trim();
        if s.starts_with('"') {
            let inner = &s[1..];
            if let Some(end) = inner.find('"') {
                return Some(inner[..end].to_string());
            }
        } else if s.starts_with('\'') {
            let inner = &s[1..];
            if let Some(end) = inner.find('\'') {
                return Some(inner[..end].to_string());
            }
        }
        None
    }

    /// Validate basic syntax properties of a code block.
    pub fn validate_syntax(block: &CodeBlock) -> Vec<String> {
        let mut errors = Vec::new();
        let code = &block.code;

        // Check balanced braces
        let open_braces = code.chars().filter(|&c| c == '{').count();
        let close_braces = code.chars().filter(|&c| c == '}').count();
        if open_braces != close_braces {
            errors.push(format!(
                "Unbalanced braces: {} opening, {} closing",
                open_braces, close_braces
            ));
        }

        // Check balanced brackets
        let open_brackets = code.chars().filter(|&c| c == '[').count();
        let close_brackets = code.chars().filter(|&c| c == ']').count();
        if open_brackets != close_brackets {
            errors.push(format!(
                "Unbalanced brackets: {} opening, {} closing",
                open_brackets, close_brackets
            ));
        }

        // Check balanced parens
        let open_parens = code.chars().filter(|&c| c == '(').count();
        let close_parens = code.chars().filter(|&c| c == ')').count();
        if open_parens != close_parens {
            errors.push(format!(
                "Unbalanced parentheses: {} opening, {} closing",
                open_parens, close_parens
            ));
        }

        // Check for unclosed strings (odd number of unescaped double quotes)
        // Simple heuristic: count non-escaped "
        let quote_count = code.chars().filter(|&c| c == '"').count();
        if quote_count % 2 != 0 {
            errors.push("Possible unclosed string: odd number of double-quote characters".to_string());
        }

        errors
    }
}

// ---------------------------------------------------------------------------
// CodeReport
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CodeReport {
    pub total_blocks: usize,
    pub by_language: HashMap<String, usize>,
    pub total_lines: usize,
    pub syntax_errors: usize,
}

/// Analyze a slice of code blocks and produce a summary report.
pub fn analyze_blocks(blocks: &[CodeBlock]) -> CodeReport {
    let mut by_language: HashMap<String, usize> = HashMap::new();
    let mut total_lines = 0usize;
    let mut syntax_errors = 0usize;

    for block in blocks {
        let lang_key = block.language.to_string();
        *by_language.entry(lang_key).or_insert(0) += 1;

        let lines = block.code.lines().count();
        total_lines += lines;

        let errs = CodeSimulator::validate_syntax(block);
        syntax_errors += errs.len();
    }

    CodeReport {
        total_blocks: blocks.len(),
        by_language,
        total_lines,
        syntax_errors,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_MD: &str = r#"
Some text here.

```rust
fn main() {
    println!("Hello, world!");
}
```

Another paragraph.

```python
def greet():
    print("Hello from Python")
```

```bash
echo "Hello from bash"
ls
```
"#;

    #[test]
    fn test_extract_finds_code_blocks() {
        let blocks = CodeExtractor::extract(SAMPLE_MD);
        assert_eq!(blocks.len(), 3);
    }

    #[test]
    fn test_language_detection() {
        assert_eq!(CodeExtractor::detect_language("rust"), Language::Rust);
        assert_eq!(CodeExtractor::detect_language("py"), Language::Python);
        assert_eq!(CodeExtractor::detect_language("js"), Language::JavaScript);
        assert_eq!(CodeExtractor::detect_language("ts"), Language::TypeScript);
        assert_eq!(CodeExtractor::detect_language("go"), Language::Go);
        assert_eq!(CodeExtractor::detect_language("sh"), Language::Bash);
        assert_eq!(
            CodeExtractor::detect_language("haskell"),
            Language::Unknown("haskell".to_string())
        );
    }

    #[test]
    fn test_simulate_rust_println() {
        let block = CodeBlock {
            language: Language::Rust,
            code: r#"fn main() { println!("Hello, world!"); }"#.to_string(),
            start_line: 1,
            end_line: 1,
        };
        let sim = CodeSimulator::new(ExecutionMode::Simulate);
        let result = sim.simulate(&block);
        assert!(result.output.contains("Hello, world!"));
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_simulate_python_print() {
        let block = CodeBlock {
            language: Language::Python,
            code: "print(\"Hello from Python\")".to_string(),
            start_line: 1,
            end_line: 1,
        };
        let sim = CodeSimulator::new(ExecutionMode::Simulate);
        let result = sim.simulate(&block);
        assert!(result.output.contains("Hello from Python"));
    }

    #[test]
    fn test_simulate_bash_echo() {
        let block = CodeBlock {
            language: Language::Bash,
            code: "echo \"Hello from bash\"".to_string(),
            start_line: 1,
            end_line: 1,
        };
        let sim = CodeSimulator::new(ExecutionMode::Simulate);
        let result = sim.simulate(&block);
        assert!(result.output.contains("Hello from bash"));
    }

    #[test]
    fn test_validate_catches_unbalanced_braces() {
        let block = CodeBlock {
            language: Language::Rust,
            code: "fn main() { let x = 1;".to_string(),
            start_line: 1,
            end_line: 1,
        };
        let errors = CodeSimulator::validate_syntax(&block);
        assert!(!errors.is_empty());
        assert!(errors[0].contains("brace"));
    }

    #[test]
    fn test_validate_balanced_code() {
        let block = CodeBlock {
            language: Language::Rust,
            code: "fn main() { let x = 1; }".to_string(),
            start_line: 1,
            end_line: 1,
        };
        let errors = CodeSimulator::validate_syntax(&block);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_analyze_blocks() {
        let blocks = CodeExtractor::extract(SAMPLE_MD);
        let report = analyze_blocks(&blocks);
        assert_eq!(report.total_blocks, 3);
        assert!(report.by_language.contains_key("rust"));
        assert!(report.by_language.contains_key("python"));
        assert!(report.by_language.contains_key("bash"));
    }

    #[test]
    fn test_rust_panic_produces_error() {
        let block = CodeBlock {
            language: Language::Rust,
            code: r#"fn main() { panic!("something went wrong"); }"#.to_string(),
            start_line: 1,
            end_line: 1,
        };
        let sim = CodeSimulator::new(ExecutionMode::Simulate);
        let result = sim.simulate(&block);
        assert!(!result.errors.is_empty());
    }
}
