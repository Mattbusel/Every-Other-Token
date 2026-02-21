use clap::Parser;
use colored::*;
use rand::Rng;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::io::{self, Write};
use tokio_stream::StreamExt;

// ── Provider enum ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Provider {
    OpenAi,
    Anthropic,
}

impl Provider {
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "openai" => Ok(Provider::OpenAi),
            "anthropic" | "claude" => Ok(Provider::Anthropic),
            _ => Err(format!("Unknown provider: {}. Use 'openai' or 'anthropic'.", s)),
        }
    }

    fn env_var_name(&self) -> &'static str {
        match self {
            Provider::OpenAi => "OPENAI_API_KEY",
            Provider::Anthropic => "ANTHROPIC_API_KEY",
        }
    }

    fn default_model(&self) -> &'static str {
        match self {
            Provider::OpenAi => "gpt-3.5-turbo",
            Provider::Anthropic => "claude-sonnet-4-20250514",
        }
    }

    fn display_name(&self) -> &'static str {
        match self {
            Provider::OpenAi => "OpenAI",
            Provider::Anthropic => "Anthropic",
        }
    }
}

// ── Transform enum ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Transform {
    Reverse,
    Uppercase,
    Mock,
    Noise,
}

impl Transform {
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "reverse" => Ok(Transform::Reverse),
            "uppercase" => Ok(Transform::Uppercase),
            "mock" => Ok(Transform::Mock),
            "noise" => Ok(Transform::Noise),
            _ => Err(format!("Unknown transform: {}", s)),
        }
    }

    fn apply(&self, token: &str) -> String {
        match self {
            Transform::Reverse => token.chars().rev().collect(),
            Transform::Uppercase => token.to_uppercase(),
            Transform::Mock => {
                token
                    .chars()
                    .enumerate()
                    .map(|(i, c)| {
                        if i % 2 == 0 {
                            c.to_lowercase().next().unwrap_or(c)
                        } else {
                            c.to_uppercase().next().unwrap_or(c)
                        }
                    })
                    .collect()
            }
            Transform::Noise => {
                let mut rng = rand::thread_rng();
                let noise_chars = ['*', '+', '~', '@', '#', '$', '%'];
                let noise_char = noise_chars[rng.gen_range(0..noise_chars.len())];
                format!("{}{}", token, noise_char)
            }
        }
    }
}

// ── OpenAI SSE types ─────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    temperature: f32,
}

#[derive(Debug, Deserialize)]
struct OpenAiDelta {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    delta: OpenAiDelta,
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamChunk {
    choices: Vec<OpenAiChoice>,
}

// ── Anthropic SSE types ──────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<AnthropicMessage>,
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct AnthropicContentDelta {
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamEvent {
    #[serde(default)]
    delta: Option<AnthropicContentDelta>,
    #[serde(rename = "type")]
    event_type: Option<String>,
}

// ── Orchestrator types ───────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct OrchestratorInferRequest {
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    session_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OrchestratorInferResponse {
    #[allow(dead_code)]
    request_id: Option<String>,
    result: Option<String>,
    #[allow(dead_code)]
    error: Option<String>,
}

// ── TokenInterceptor ─────────────────────────────────────────────────────────

pub struct TokenInterceptor {
    client: Client,
    api_key: String,
    provider: Provider,
    transform: Transform,
    model: String,
    token_count: usize,
    transformed_count: usize,
    visual_mode: bool,
    heatmap_mode: bool,
    orchestrator: bool,
    orchestrator_url: String,
}

impl TokenInterceptor {
    pub fn new(
        provider: Provider,
        transform: Transform,
        model: String,
        visual_mode: bool,
        heatmap_mode: bool,
        orchestrator: bool,
        orchestrator_url: String,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let api_key = env::var(provider.env_var_name()).map_err(|_| {
            format!(
                "{} API key not found. Set {} environment variable.",
                provider.display_name(),
                provider.env_var_name()
            )
        })?;

        let client = Client::new();

        Ok(TokenInterceptor {
            client,
            api_key,
            provider,
            transform,
            model,
            token_count: 0,
            transformed_count: 0,
            visual_mode,
            heatmap_mode,
            orchestrator,
            orchestrator_url,
        })
    }

    pub async fn intercept_stream(
        &mut self,
        prompt: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.print_header(prompt);

        // If orchestrator is enabled, pre-process the prompt through the pipeline
        let effective_prompt = if self.orchestrator {
            self.route_through_orchestrator(prompt).await?
        } else {
            prompt.to_string()
        };

        match self.provider {
            Provider::OpenAi => self.stream_openai(&effective_prompt).await?,
            Provider::Anthropic => self.stream_anthropic(&effective_prompt).await?,
        }

        self.print_footer();
        Ok(())
    }

    // ── Orchestrator integration ─────────────────────────────────────────

    async fn route_through_orchestrator(
        &self,
        prompt: &str,
    ) -> Result<String, Box<dyn std::error::Error>> {
        eprintln!(
            "{}",
            format!(
                "  Routing through orchestrator at {} ...",
                self.orchestrator_url
            )
            .bright_blue()
        );

        let infer_url = format!("{}/api/v1/infer", self.orchestrator_url);

        let request_body = OrchestratorInferRequest {
            prompt: prompt.to_string(),
            session_id: None,
        };

        let response = self
            .client
            .post(&infer_url)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await;

        match response {
            Ok(resp) if resp.status().is_success() => {
                let body: OrchestratorInferResponse = resp.json().await.unwrap_or(
                    OrchestratorInferResponse {
                        request_id: None,
                        result: None,
                        error: None,
                    },
                );
                if let Some(result) = body.result {
                    eprintln!(
                        "{}",
                        "  Orchestrator: prompt enriched via pipeline."
                            .bright_green()
                    );
                    Ok(result)
                } else {
                    eprintln!(
                        "{}",
                        "  Orchestrator: no enrichment returned, using original prompt."
                            .bright_yellow()
                    );
                    Ok(prompt.to_string())
                }
            }
            Ok(resp) => {
                let status = resp.status();
                eprintln!(
                    "{}",
                    format!(
                        "  Orchestrator returned {}, falling back to direct provider.",
                        status
                    )
                    .bright_yellow()
                );
                Ok(prompt.to_string())
            }
            Err(e) => {
                eprintln!(
                    "{}",
                    format!(
                        "  Orchestrator unreachable ({}), falling back to direct provider.",
                        e
                    )
                    .bright_yellow()
                );
                Ok(prompt.to_string())
            }
        }
    }

    // ── OpenAI streaming ─────────────────────────────────────────────────

    async fn stream_openai(&mut self, prompt: &str) -> Result<(), Box<dyn std::error::Error>> {
        let request = OpenAiRequest {
            model: self.model.clone(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            stream: true,
            temperature: 0.7,
        };

        let response = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(format!("OpenAI API Error: {}", error_text).into());
        }

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            let chunk_str = String::from_utf8_lossy(&chunk);
            buffer.push_str(&chunk_str);

            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim().to_string();
                buffer.drain(..=line_end);

                if line.starts_with("data: ") && line != "data: [DONE]" {
                    let json_str = &line[6..];
                    if let Ok(parsed) = serde_json::from_str::<OpenAiStreamChunk>(json_str) {
                        if let Some(choice) = parsed.choices.first() {
                            if let Some(content) = &choice.delta.content {
                                self.process_content(content);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    // ── Anthropic streaming ──────────────────────────────────────────────

    async fn stream_anthropic(&mut self, prompt: &str) -> Result<(), Box<dyn std::error::Error>> {
        let request = AnthropicRequest {
            model: self.model.clone(),
            max_tokens: 4096,
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            stream: true,
        };

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(format!("Anthropic API Error: {}", error_text).into());
        }

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            let chunk_str = String::from_utf8_lossy(&chunk);
            buffer.push_str(&chunk_str);

            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim().to_string();
                buffer.drain(..=line_end);

                // Anthropic SSE: "event: content_block_delta" then "data: {...}"
                if line.starts_with("data: ") {
                    let json_str = &line[6..];
                    if let Ok(event) = serde_json::from_str::<AnthropicStreamEvent>(json_str) {
                        // content_block_delta events carry text in delta.text
                        let is_delta = event
                            .event_type
                            .as_deref()
                            .map_or(true, |t| t == "content_block_delta");
                        if is_delta {
                            if let Some(delta) = &event.delta {
                                if let Some(text) = &delta.text {
                                    self.process_content(text);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    // ── Token processing (shared by all providers) ───────────────────────

    fn process_content(&mut self, content: &str) {
        let tokens = self.tokenize(content);

        for token in tokens {
            if !token.trim().is_empty() {
                if self.token_count % 2 == 0 {
                    // Even tokens — pass through unchanged
                    if self.heatmap_mode {
                        let importance =
                            self.calculate_token_importance(&token, self.token_count);
                        print!("{}", self.apply_heatmap_color(&token, importance));
                    } else if self.visual_mode {
                        print!("{}", token.normal());
                    } else {
                        print!("{}", token);
                    }
                } else {
                    // Odd tokens — apply transformation
                    self.transformed_count += 1;
                    let transformed = self.transform.apply(&token);
                    if self.heatmap_mode {
                        let importance =
                            self.calculate_token_importance(&transformed, self.token_count);
                        print!("{}", self.apply_heatmap_color(&transformed, importance));
                    } else if self.visual_mode {
                        print!("{}", transformed.bright_cyan().bold());
                    } else {
                        print!("{}", transformed);
                    }
                };

                let _ = io::stdout().flush();
                self.token_count += 1;
            }
        }
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current_token = String::new();

        for ch in text.chars() {
            if ch.is_whitespace() || ch.is_ascii_punctuation() {
                if !current_token.is_empty() {
                    tokens.push(current_token.clone());
                    current_token.clear();
                }
                if !ch.is_whitespace() {
                    tokens.push(ch.to_string());
                }
                if ch.is_whitespace() {
                    tokens.push(ch.to_string());
                }
            } else {
                current_token.push(ch);
            }
        }

        if !current_token.is_empty() {
            tokens.push(current_token);
        }

        tokens
    }

    fn calculate_token_importance(&self, token: &str, position: usize) -> f64 {
        let mut importance = 0.0;

        // Token length importance
        importance += (token.len() as f64 / 20.0).min(0.3);

        // Position-based importance
        let position_factor = if position < 5 || position > 50 {
            0.3
        } else {
            0.1
        };
        importance += position_factor;

        // Content-based importance
        if token.chars().any(|c| c.is_uppercase()) {
            importance += 0.2;
        }

        let important_patterns = [
            "the", "and", "or", "but", "if", "when", "where", "how", "why", "what",
            "robot", "AI", "technology", "system", "data", "algorithm", "model",
            "create", "build", "develop", "analyze", "process", "generate",
        ];

        let lower_token = token.to_lowercase();
        if important_patterns
            .iter()
            .any(|&pattern| lower_token.contains(pattern))
        {
            importance += 0.3;
        }

        // Punctuation reduces importance
        if token.chars().all(|c| c.is_ascii_punctuation()) {
            importance *= 0.1;
        }

        // Simulated attention jitter
        let mut rng = rand::thread_rng();
        importance += rng.gen_range(-0.1..0.1);

        importance.max(0.0).min(1.0)
    }

    fn apply_heatmap_color(&self, token: &str, importance: f64) -> String {
        match importance {
            i if i >= 0.8 => token.on_bright_red().bright_white().to_string(),
            i if i >= 0.6 => token.on_red().bright_white().to_string(),
            i if i >= 0.4 => token.on_yellow().black().to_string(),
            i if i >= 0.2 => token.on_blue().bright_white().to_string(),
            _ => token.normal().to_string(),
        }
    }

    fn print_header(&self, prompt: &str) {
        println!(
            "{}",
            "EVERY OTHER TOKEN INTERCEPTOR".bright_cyan().bold()
        );
        println!(
            "{}: {}",
            "Provider".bright_yellow(),
            self.provider.display_name()
        );
        println!("{}: {:?}", "Transform".bright_yellow(), self.transform);
        println!("{}: {}", "Model".bright_yellow(), self.model);
        println!("{}: {}", "Prompt".bright_yellow(), prompt);
        if self.orchestrator {
            println!(
                "{}: {} ({})",
                "Orchestrator".bright_magenta(),
                "ON".bright_green(),
                self.orchestrator_url
            );
        }
        if self.visual_mode {
            println!(
                "{}: {}",
                "Visual Mode".bright_green(),
                "ON (even=normal, odd=cyan+bold)".bright_green()
            );
        }
        if self.heatmap_mode {
            println!(
                "{}: {}",
                "Heatmap Mode".bright_magenta(),
                "ON (color intensity = token importance)".bright_magenta()
            );
            println!(
                "{}: {} {} {} {}",
                "Legend".bright_white(),
                "Low".on_blue(),
                "Medium".on_yellow(),
                "High".on_red(),
                "Critical".on_bright_red().bright_white()
            );
        }
        println!("{}", "=".repeat(60).bright_blue());
        println!("{}", "Response (with transformations):".bright_green());
        println!();
    }

    fn print_footer(&self) {
        println!("\n{}", "=".repeat(60).bright_blue());
        println!(
            "Complete! Processed {} tokens ({} transformed).",
            self.token_count, self.transformed_count
        );
    }
}

// ── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "every-other-token")]
#[command(version = "2.0.0")]
#[command(about = "A real-time token stream mutator for LLM interpretability research")]
struct Args {
    /// Input prompt to send to the LLM
    prompt: String,

    /// Transformation type: reverse, uppercase, mock, noise
    #[arg(default_value = "reverse")]
    transform: String,

    /// Model name (provider-specific, e.g. gpt-4, claude-sonnet-4-20250514)
    #[arg(short, long)]
    model: Option<String>,

    /// LLM provider: openai or anthropic
    #[arg(long, default_value = "openai")]
    provider: String,

    /// Enable visual mode with color-coded tokens
    #[arg(long, short)]
    visual: bool,

    /// Enable token importance heatmap
    #[arg(long)]
    heatmap: bool,

    /// Route prompt through tokio-prompt-orchestrator before sending to LLM
    #[arg(long)]
    orchestrator: bool,

    /// Orchestrator base URL
    #[arg(long, default_value = "http://localhost:3000")]
    orchestrator_url: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let provider = Provider::from_str(&args.provider)
        .map_err(|e| format!("Invalid provider: {}", e))?;

    let model = args
        .model
        .unwrap_or_else(|| provider.default_model().to_string());

    let transform =
        Transform::from_str(&args.transform).map_err(|e| format!("Invalid transform: {}", e))?;

    let mut interceptor = TokenInterceptor::new(
        provider,
        transform,
        model,
        args.visual,
        args.heatmap,
        args.orchestrator,
        args.orchestrator_url,
    )?;

    interceptor.intercept_stream(&args.prompt).await?;

    Ok(())
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Transform tests ──────────────────────────────────────────────────

    #[test]
    fn test_transform_reverse() {
        let transform = Transform::Reverse;
        assert_eq!(transform.apply("hello"), "olleh");
        assert_eq!(transform.apply("world"), "dlrow");
    }

    #[test]
    fn test_transform_uppercase() {
        let transform = Transform::Uppercase;
        assert_eq!(transform.apply("hello"), "HELLO");
        assert_eq!(transform.apply("world"), "WORLD");
    }

    #[test]
    fn test_transform_mock() {
        let transform = Transform::Mock;
        assert_eq!(transform.apply("hello"), "hElLo");
        assert_eq!(transform.apply("world"), "wOrLd");
    }

    #[test]
    fn test_transform_noise() {
        let transform = Transform::Noise;
        let result = transform.apply("hello");
        assert!(result.starts_with("hello"));
        assert!(result.len() > 5);
    }

    #[test]
    fn test_transform_from_str() {
        assert!(matches!(Transform::from_str("reverse"), Ok(Transform::Reverse)));
        assert!(matches!(
            Transform::from_str("uppercase"),
            Ok(Transform::Uppercase)
        ));
        assert!(matches!(Transform::from_str("mock"), Ok(Transform::Mock)));
        assert!(matches!(Transform::from_str("noise"), Ok(Transform::Noise)));
        assert!(Transform::from_str("invalid").is_err());
    }

    #[test]
    fn test_transform_from_str_case_insensitive() {
        assert!(matches!(Transform::from_str("REVERSE"), Ok(Transform::Reverse)));
        assert!(matches!(
            Transform::from_str("Uppercase"),
            Ok(Transform::Uppercase)
        ));
        assert!(matches!(Transform::from_str("MoCk"), Ok(Transform::Mock)));
    }

    #[test]
    fn test_transform_reverse_empty() {
        let transform = Transform::Reverse;
        assert_eq!(transform.apply(""), "");
    }

    #[test]
    fn test_transform_reverse_single_char() {
        let transform = Transform::Reverse;
        assert_eq!(transform.apply("a"), "a");
    }

    #[test]
    fn test_transform_uppercase_already_upper() {
        let transform = Transform::Uppercase;
        assert_eq!(transform.apply("HELLO"), "HELLO");
    }

    #[test]
    fn test_transform_mock_single_char() {
        let transform = Transform::Mock;
        assert_eq!(transform.apply("a"), "a");
    }

    #[test]
    fn test_transform_noise_appends_one_char() {
        let transform = Transform::Noise;
        let result = transform.apply("test");
        assert_eq!(result.len(), 5); // "test" + 1 noise char
    }

    // ── Provider tests ───────────────────────────────────────────────────

    #[test]
    fn test_provider_from_str_openai() {
        assert!(matches!(Provider::from_str("openai"), Ok(Provider::OpenAi)));
    }

    #[test]
    fn test_provider_from_str_anthropic() {
        assert!(matches!(
            Provider::from_str("anthropic"),
            Ok(Provider::Anthropic)
        ));
    }

    #[test]
    fn test_provider_from_str_claude_alias() {
        assert!(matches!(
            Provider::from_str("claude"),
            Ok(Provider::Anthropic)
        ));
    }

    #[test]
    fn test_provider_from_str_invalid() {
        assert!(Provider::from_str("invalid").is_err());
    }

    #[test]
    fn test_provider_from_str_case_insensitive() {
        assert!(matches!(Provider::from_str("OPENAI"), Ok(Provider::OpenAi)));
        assert!(matches!(
            Provider::from_str("Anthropic"),
            Ok(Provider::Anthropic)
        ));
    }

    #[test]
    fn test_provider_env_var_name_openai() {
        assert_eq!(Provider::OpenAi.env_var_name(), "OPENAI_API_KEY");
    }

    #[test]
    fn test_provider_env_var_name_anthropic() {
        assert_eq!(Provider::Anthropic.env_var_name(), "ANTHROPIC_API_KEY");
    }

    #[test]
    fn test_provider_default_model_openai() {
        assert_eq!(Provider::OpenAi.default_model(), "gpt-3.5-turbo");
    }

    #[test]
    fn test_provider_default_model_anthropic() {
        let model = Provider::Anthropic.default_model();
        assert!(model.contains("claude"));
    }

    #[test]
    fn test_provider_display_name() {
        assert_eq!(Provider::OpenAi.display_name(), "OpenAI");
        assert_eq!(Provider::Anthropic.display_name(), "Anthropic");
    }

    #[test]
    fn test_provider_equality() {
        assert_eq!(Provider::OpenAi, Provider::OpenAi);
        assert_eq!(Provider::Anthropic, Provider::Anthropic);
        assert_ne!(Provider::OpenAi, Provider::Anthropic);
    }

    // ── Serialization tests ──────────────────────────────────────────────

    #[test]
    fn test_openai_request_serializes() {
        let req = OpenAiRequest {
            model: "gpt-4".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "test".to_string(),
            }],
            stream: true,
            temperature: 0.7,
        };
        let json = serde_json::to_string(&req);
        assert!(json.is_ok());
        let s = json.unwrap_or_default();
        assert!(s.contains("gpt-4"));
        assert!(s.contains("\"stream\":true"));
    }

    #[test]
    fn test_anthropic_request_serializes() {
        let req = AnthropicRequest {
            model: "claude-sonnet-4-20250514".to_string(),
            max_tokens: 4096,
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: "test".to_string(),
            }],
            stream: true,
        };
        let json = serde_json::to_string(&req);
        assert!(json.is_ok());
        let s = json.unwrap_or_default();
        assert!(s.contains("claude"));
        assert!(s.contains("max_tokens"));
    }

    #[test]
    fn test_openai_stream_chunk_deserializes() {
        let json = r#"{"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}"#;
        let parsed: Result<OpenAiStreamChunk, _> = serde_json::from_str(json);
        assert!(parsed.is_ok());
        let chunk = parsed.unwrap();
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(
            chunk.choices[0].delta.content.as_deref(),
            Some("hello")
        );
    }

    #[test]
    fn test_openai_stream_chunk_empty_delta() {
        let json = r#"{"choices":[{"delta":{},"finish_reason":null}]}"#;
        let parsed: Result<OpenAiStreamChunk, _> = serde_json::from_str(json);
        assert!(parsed.is_ok());
        let chunk = parsed.unwrap();
        assert!(chunk.choices[0].delta.content.is_none());
    }

    #[test]
    fn test_anthropic_stream_event_deserializes() {
        let json = r#"{"type":"content_block_delta","delta":{"type":"text_delta","text":"world"}}"#;
        let parsed: Result<AnthropicStreamEvent, _> = serde_json::from_str(json);
        assert!(parsed.is_ok());
        let event = parsed.unwrap();
        assert_eq!(
            event.event_type.as_deref(),
            Some("content_block_delta")
        );
        assert_eq!(
            event.delta.as_ref().and_then(|d| d.text.as_deref()),
            Some("world")
        );
    }

    #[test]
    fn test_anthropic_stream_event_no_delta() {
        let json = r#"{"type":"message_start"}"#;
        let parsed: Result<AnthropicStreamEvent, _> = serde_json::from_str(json);
        assert!(parsed.is_ok());
        let event = parsed.unwrap();
        assert!(event.delta.is_none());
    }

    // ── Orchestrator request serialization ───────────────────────────────

    #[test]
    fn test_orchestrator_request_serializes() {
        let req = OrchestratorInferRequest {
            prompt: "test prompt".to_string(),
            session_id: None,
        };
        let json = serde_json::to_string(&req);
        assert!(json.is_ok());
        let s = json.unwrap_or_default();
        assert!(s.contains("test prompt"));
        // session_id should be omitted when None
        assert!(!s.contains("session_id"));
    }

    #[test]
    fn test_orchestrator_request_with_session() {
        let req = OrchestratorInferRequest {
            prompt: "test".to_string(),
            session_id: Some("sess-123".to_string()),
        };
        let json = serde_json::to_string(&req);
        assert!(json.is_ok());
        let s = json.unwrap_or_default();
        assert!(s.contains("sess-123"));
    }

    #[test]
    fn test_orchestrator_response_deserializes() {
        let json = r#"{"request_id":"abc","result":"enriched prompt","error":null}"#;
        let parsed: Result<OrchestratorInferResponse, _> = serde_json::from_str(json);
        assert!(parsed.is_ok());
        let resp = parsed.unwrap();
        assert_eq!(resp.result.as_deref(), Some("enriched prompt"));
    }

    #[test]
    fn test_orchestrator_response_error_case() {
        let json = r#"{"request_id":null,"result":null,"error":"pipeline closed"}"#;
        let parsed: Result<OrchestratorInferResponse, _> = serde_json::from_str(json);
        assert!(parsed.is_ok());
        let resp = parsed.unwrap();
        assert!(resp.result.is_none());
        assert_eq!(resp.error.as_deref(), Some("pipeline closed"));
    }

    // ── Tokenizer tests ─────────────────────────────────────────────────

    #[test]
    fn test_tokenize_simple_sentence() {
        // Create a minimal interceptor just to test tokenize
        // We can't use new() without env vars, so test the logic directly
        let tokens = tokenize_standalone("hello world");
        let non_ws: Vec<&String> = tokens.iter().filter(|t| !t.trim().is_empty()).collect();
        assert_eq!(non_ws, &["hello", "world"]);
    }

    #[test]
    fn test_tokenize_with_punctuation() {
        let tokens = tokenize_standalone("hello, world!");
        let non_ws: Vec<&String> = tokens.iter().filter(|t| !t.trim().is_empty()).collect();
        assert_eq!(non_ws, &["hello", ",", "world", "!"]);
    }

    #[test]
    fn test_tokenize_empty_string() {
        let tokens = tokenize_standalone("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_tokenize_whitespace_only() {
        let tokens = tokenize_standalone("   ");
        let non_ws: Vec<&String> = tokens.iter().filter(|t| !t.trim().is_empty()).collect();
        assert!(non_ws.is_empty());
    }

    #[test]
    fn test_tokenize_single_word() {
        let tokens = tokenize_standalone("hello");
        assert_eq!(tokens, &["hello"]);
    }

    #[test]
    fn test_tokenize_preserves_whitespace_tokens() {
        let tokens = tokenize_standalone("a b");
        // Should produce ["a", " ", "b"]
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[1], " ");
    }

    /// Standalone tokenizer for testing without constructing a full TokenInterceptor.
    fn tokenize_standalone(text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current_token = String::new();

        for ch in text.chars() {
            if ch.is_whitespace() || ch.is_ascii_punctuation() {
                if !current_token.is_empty() {
                    tokens.push(current_token.clone());
                    current_token.clear();
                }
                if !ch.is_whitespace() {
                    tokens.push(ch.to_string());
                }
                if ch.is_whitespace() {
                    tokens.push(ch.to_string());
                }
            } else {
                current_token.push(ch);
            }
        }

        if !current_token.is_empty() {
            tokens.push(current_token);
        }

        tokens
    }

    // ── Heatmap importance scoring ───────────────────────────────────────

    #[test]
    fn test_importance_punctuation_low() {
        // Punctuation-only tokens should have very low importance
        let score = importance_standalone(".", 10);
        assert!(score < 0.3, "punctuation importance should be low, got {score}");
    }

    #[test]
    fn test_importance_keyword_high() {
        // Keywords like "algorithm" should score higher
        let scores: Vec<f64> = (0..20).map(|_| importance_standalone("algorithm", 3)).collect();
        let avg = scores.iter().sum::<f64>() / scores.len() as f64;
        assert!(avg > 0.4, "keyword importance should be high, avg was {avg}");
    }

    #[test]
    fn test_importance_clamped_0_to_1() {
        for i in 0..50 {
            let score = importance_standalone("test", i);
            assert!(score >= 0.0 && score <= 1.0, "score out of range: {score}");
        }
    }

    /// Standalone importance calculator for testing.
    fn importance_standalone(token: &str, position: usize) -> f64 {
        let mut importance = 0.0;
        importance += (token.len() as f64 / 20.0).min(0.3);
        let position_factor = if position < 5 || position > 50 {
            0.3
        } else {
            0.1
        };
        importance += position_factor;
        if token.chars().any(|c| c.is_uppercase()) {
            importance += 0.2;
        }
        let important_patterns = [
            "the", "and", "or", "but", "if", "when", "where", "how", "why", "what",
            "robot", "AI", "technology", "system", "data", "algorithm", "model",
            "create", "build", "develop", "analyze", "process", "generate",
        ];
        let lower_token = token.to_lowercase();
        if important_patterns.iter().any(|&p| lower_token.contains(p)) {
            importance += 0.3;
        }
        if token.chars().all(|c| c.is_ascii_punctuation()) {
            importance *= 0.1;
        }
        let mut rng = rand::thread_rng();
        importance += rng.gen_range(-0.1..0.1);
        importance.max(0.0).min(1.0)
    }
}
