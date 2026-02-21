pub mod cli;
pub mod providers;
pub mod transforms;
pub mod web;

use colored::*;
use reqwest::Client;
use serde::Serialize;
use std::env;
use std::io::{self, Write};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;

use providers::*;
use transforms::{apply_heatmap_color, calculate_token_importance, tokenize, Transform};

// ---------------------------------------------------------------------------
// Token event (for web UI streaming)
// ---------------------------------------------------------------------------

/// A single processed token, sent as an SSE event to the web UI.
#[derive(Debug, Clone, Serialize)]
pub struct TokenEvent {
    pub text: String,
    pub original: String,
    pub index: usize,
    pub transformed: bool,
    pub importance: f64,
    /// For Chaos transform: which sub-transform was applied. None for other transforms.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chaos_label: Option<String>,
    /// For diff mode: which provider produced this token ("openai" or "anthropic").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
}

// ---------------------------------------------------------------------------
// TokenInterceptor — multi-provider streaming engine
// ---------------------------------------------------------------------------

pub struct TokenInterceptor {
    client: Client,
    api_key: String,
    pub provider: Provider,
    pub transform: Transform,
    pub model: String,
    pub token_count: usize,
    pub transformed_count: usize,
    pub visual_mode: bool,
    pub heatmap_mode: bool,
    pub orchestrator: bool,
    pub orchestrator_url: String,
    /// When set, token events are sent here instead of printed to stdout.
    pub web_tx: Option<mpsc::UnboundedSender<TokenEvent>>,
    /// When set, each emitted TokenEvent carries this provider label (for diff mode).
    pub web_provider_label: Option<String>,
}

impl TokenInterceptor {
    pub fn new(
        provider: Provider,
        transform: Transform,
        model: String,
        visual_mode: bool,
        heatmap_mode: bool,
        orchestrator: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let api_key = match provider {
            Provider::Openai => env::var("OPENAI_API_KEY")
                .map_err(|_| "OPENAI_API_KEY not set. Export it or pass via environment.")?,
            Provider::Anthropic => env::var("ANTHROPIC_API_KEY")
                .map_err(|_| "ANTHROPIC_API_KEY not set. Export it or pass via environment.")?,
        };

        Ok(TokenInterceptor {
            client: Client::new(),
            api_key,
            provider,
            transform,
            model,
            token_count: 0,
            transformed_count: 0,
            visual_mode,
            heatmap_mode,
            orchestrator,
            orchestrator_url: "http://localhost:3000".to_string(),
            web_tx: None,
            web_provider_label: None,
        })
    }

    // -----------------------------------------------------------------------
    // Public entry point
    // -----------------------------------------------------------------------

    pub async fn intercept_stream(
        &mut self,
        prompt: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.web_tx.is_none() {
            self.print_header(prompt);
        }

        // If --orchestrator is active, pre-process through MCP pipeline
        let effective_prompt = if self.orchestrator {
            eprintln!(
                "{}",
                "[orchestrator] routing through MCP pipeline at localhost:3000".bright_magenta()
            );
            match self.orchestrator_infer(prompt).await {
                Ok(enriched) => enriched,
                Err(e) => {
                    eprintln!(
                        "{} {}",
                        "[orchestrator] pipeline unavailable, using raw prompt:".bright_red(),
                        e
                    );
                    prompt.to_string()
                }
            }
        } else {
            prompt.to_string()
        };

        match self.provider {
            Provider::Openai => self.stream_openai(&effective_prompt).await?,
            Provider::Anthropic => self.stream_anthropic(&effective_prompt).await?,
        }

        if self.web_tx.is_none() {
            self.print_footer();
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // OpenAI streaming
    // -----------------------------------------------------------------------

    async fn stream_openai(&mut self, prompt: &str) -> Result<(), Box<dyn std::error::Error>> {
        let request = OpenAIChatRequest {
            model: self.model.clone(),
            messages: vec![OpenAIChatMessage {
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
            return Err(format!("OpenAI API error: {}", error_text).into());
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
                    if let Ok(parsed) = serde_json::from_str::<OpenAIChunk>(json_str) {
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

    // -----------------------------------------------------------------------
    // Anthropic streaming
    // -----------------------------------------------------------------------

    async fn stream_anthropic(&mut self, prompt: &str) -> Result<(), Box<dyn std::error::Error>> {
        let request = AnthropicRequest {
            model: self.model.clone(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            max_tokens: 4096,
            stream: true,
            temperature: 0.7,
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
            return Err(format!("Anthropic API error: {}", error_text).into());
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

                if line.starts_with("data: ") {
                    let json_str = &line[6..];
                    if let Ok(event) = serde_json::from_str::<AnthropicStreamEvent>(json_str) {
                        if event.event_type == "content_block_delta" {
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

    // -----------------------------------------------------------------------
    // Orchestrator MCP infer call
    // -----------------------------------------------------------------------

    async fn orchestrator_infer(&self, prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
        let mcp_request = McpInferRequest {
            jsonrpc: "2.0".to_string(),
            method: "tools/call".to_string(),
            id: 1,
            params: McpInferParams {
                name: "infer".to_string(),
                arguments: McpInferArguments {
                    prompt: prompt.to_string(),
                    worker: "llama_cpp".to_string(),
                },
            },
        };

        let response = self
            .client
            .post(&self.orchestrator_url)
            .header("Content-Type", "application/json")
            .json(&mcp_request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(format!("Orchestrator returned HTTP {}", response.status()).into());
        }

        let mcp_resp: McpInferResponse = response.json().await?;

        if let Some(err) = mcp_resp.error {
            return Err(format!("Orchestrator MCP error: {}", err.message).into());
        }

        if let Some(result) = mcp_resp.result {
            if let Some(content) = result.content.first() {
                if let Some(text) = &content.text {
                    return Ok(text.clone());
                }
            }
        }

        Err("Orchestrator returned empty result".into())
    }

    // -----------------------------------------------------------------------
    // Token processing (shared by both providers)
    // -----------------------------------------------------------------------

    pub fn process_content(&mut self, content: &str) {
        let tokens = tokenize(content);

        for token in tokens {
            if !token.trim().is_empty() {
                let is_odd = self.token_count % 2 != 0;
                let importance = calculate_token_importance(&token, self.token_count);

                let (display_text, chaos_label) = if is_odd {
                    self.transformed_count += 1;
                    let (text, label) = self.transform.apply_with_label(&token);
                    let cl = if matches!(self.transform, Transform::Chaos) {
                        Some(label.to_string())
                    } else {
                        None
                    };
                    (text, cl)
                } else {
                    (token.clone(), None)
                };

                // Web mode: send event through channel
                if let Some(tx) = &self.web_tx {
                    let event = TokenEvent {
                        text: display_text,
                        original: token.clone(),
                        index: self.token_count,
                        transformed: is_odd,
                        importance,
                        chaos_label,
                        provider: self.web_provider_label.clone(),
                    };
                    let _ = tx.send(event);
                } else {
                    // Terminal mode: print with colors
                    if self.heatmap_mode {
                        print!("{}", apply_heatmap_color(&display_text, importance));
                    } else if self.visual_mode && is_odd {
                        print!("{}", display_text.bright_cyan().bold());
                    } else if self.visual_mode {
                        print!("{}", display_text.normal());
                    } else {
                        print!("{}", display_text);
                    }
                    let _ = io::stdout().flush();
                }

                self.token_count += 1;
            }
        }
    }

    pub fn print_header(&self, prompt: &str) {
        println!("{}", "EVERY OTHER TOKEN INTERCEPTOR".bright_cyan().bold());
        println!(
            "{}: {}",
            "Provider".bright_yellow(),
            self.provider.to_string().bright_white()
        );
        println!("{}: {:?}", "Transform".bright_yellow(), self.transform);
        println!("{}: {}", "Model".bright_yellow(), self.model);
        println!("{}: {}", "Prompt".bright_yellow(), prompt);
        if self.orchestrator {
            println!(
                "{}: {}",
                "Orchestrator".bright_magenta(),
                "ON (MCP pipeline at localhost:3000)".bright_magenta()
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
        println!("{}", "=".repeat(50).bright_blue());
        println!("{}", "Response (with transformations):".bright_green());
        println!();
    }

    pub fn print_footer(&self) {
        println!("\n{}", "=".repeat(50).bright_blue());
        println!("Complete! Processed {} tokens.", self.token_count);
        println!("Transform applied to {} tokens.", self.transformed_count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    fn make_test_interceptor() -> TokenInterceptor {
        TokenInterceptor {
            client: Client::new(),
            api_key: "test-key".to_string(),
            provider: Provider::Openai,
            transform: Transform::Reverse,
            model: "test-model".to_string(),
            token_count: 0,
            transformed_count: 0,
            visual_mode: false,
            heatmap_mode: false,
            orchestrator: false,
            orchestrator_url: "http://localhost:3000".to_string(),
            web_tx: None,
            web_provider_label: None,
        }
    }

    // -- TokenInterceptor construction --

    #[test]
    fn test_new_openai_requires_api_key() {
        std::env::remove_var("OPENAI_API_KEY");
        let result = TokenInterceptor::new(
            Provider::Openai,
            Transform::Reverse,
            "gpt-4".to_string(),
            false,
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_new_anthropic_requires_api_key() {
        std::env::remove_var("ANTHROPIC_API_KEY");
        let result = TokenInterceptor::new(
            Provider::Anthropic,
            Transform::Reverse,
            "claude".to_string(),
            false,
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_interceptor_initial_counts_zero() {
        let interceptor = make_test_interceptor();
        assert_eq!(interceptor.token_count, 0);
        assert_eq!(interceptor.transformed_count, 0);
    }

    #[test]
    fn test_interceptor_fields_match_construction() {
        let interceptor = make_test_interceptor();
        assert_eq!(interceptor.provider, Provider::Openai);
        assert_eq!(interceptor.model, "test-model");
        assert!(!interceptor.visual_mode);
        assert!(!interceptor.heatmap_mode);
        assert!(!interceptor.orchestrator);
        assert!(interceptor.web_tx.is_none());
    }

    // -- process_content tests --

    #[test]
    fn test_process_content_two_tokens() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("hello world");
        assert_eq!(interceptor.token_count, 2);

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].text, "hello");
        assert_eq!(events[0].original, "hello");
        assert!(!events[0].transformed);
        assert_eq!(events[0].index, 0);
        assert_eq!(events[1].original, "world");
        assert!(events[1].transformed);
        assert_eq!(events[1].index, 1);
    }

    #[test]
    fn test_process_content_transforms_odd_tokens() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("hello world");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        // "world" reversed = "dlrow"
        assert_eq!(events[1].text, "dlrow");
        assert_eq!(events[1].original, "world");
    }

    #[test]
    fn test_process_content_empty_string() {
        let (tx, _rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("");
        assert_eq!(interceptor.token_count, 0);
        assert_eq!(interceptor.transformed_count, 0);
    }

    #[test]
    fn test_process_content_whitespace_only() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("   ");
        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        assert!(events.is_empty());
    }

    #[test]
    fn test_process_content_single_token() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("hello");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].text, "hello");
        assert!(!events[0].transformed);
    }

    #[test]
    fn test_process_content_cross_call_continuity() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("hello");
        interceptor.process_content("world");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].index, 0);
        assert_eq!(events[1].index, 1);
        assert!(events[1].transformed);
    }

    #[test]
    fn test_process_content_increments_transformed_count() {
        let (tx, _rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("hello world foo bar");
        assert_eq!(interceptor.transformed_count, 2);
    }

    #[test]
    fn test_process_content_six_tokens_three_transformed() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("one two three four five six");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        assert_eq!(events.len(), 6);
        let xformed: Vec<_> = events.iter().filter(|e| e.transformed).collect();
        assert_eq!(xformed.len(), 3);
    }

    // -- original field preservation --

    #[test]
    fn test_original_field_preserved_for_all_tokens() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("the quick brown fox");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }

        for event in &events {
            assert!(!event.original.is_empty());
            if event.transformed {
                assert_ne!(event.text, event.original);
            } else {
                assert_eq!(event.text, event.original);
            }
        }
    }

    #[test]
    fn test_sidebyside_original_is_raw_token() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("quick brown fox");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }

        let originals: Vec<&str> = events.iter().map(|e| e.original.as_str()).collect();
        assert!(originals.contains(&"quick"));
        assert!(originals.contains(&"brown"));
        assert!(originals.contains(&"fox"));
    }

    // -- even/odd alternation for graph --

    #[test]
    fn test_even_odd_alternation() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("a b c d");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }

        for event in &events {
            if event.index % 2 == 0 {
                assert!(!event.transformed);
            } else {
                assert!(event.transformed);
            }
        }
    }

    #[test]
    fn test_graph_pairs_alternate() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("alpha beta gamma delta epsilon zeta");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }

        for pair in events.chunks(2) {
            assert!(!pair[0].transformed);
            if pair.len() > 1 {
                assert!(pair[1].transformed);
            }
        }
    }

    #[test]
    fn test_graph_indices_sequential() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("one two three four");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }

        for (i, event) in events.iter().enumerate() {
            assert_eq!(event.index, i);
        }
    }

    // -- export structure tests --

    #[test]
    fn test_export_array_sequential_indices() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("the quick brown fox jumps over");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }

        for (i, event) in events.iter().enumerate() {
            assert_eq!(event.index, i);
        }
    }

    #[test]
    fn test_export_all_tokens_have_valid_importance() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("testing export importance values");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }

        for event in &events {
            assert!(event.importance >= 0.0 && event.importance <= 1.0);
        }
    }

    #[test]
    fn test_export_large_set_serializes() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("the quick brown fox jumps over the lazy dog and runs around");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }

        let json = serde_json::to_string(&events).expect("serialize");
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).expect("parse");
        assert_eq!(parsed.len(), events.len());
        assert!(parsed.len() > 5);
    }

    #[test]
    fn test_multiple_tokens_form_valid_export_array() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("hello world foo bar");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }

        let json = serde_json::to_string(&events).expect("serialize");
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).expect("parse");
        assert_eq!(parsed.len(), events.len());
        for (i, entry) in parsed.iter().enumerate() {
            assert_eq!(entry["index"].as_u64().expect("index"), i as u64);
        }
    }

    // -- print header/footer (no crash) --

    #[test]
    fn test_print_header_all_modes() {
        let interceptor = make_test_interceptor();
        interceptor.print_header("test prompt");
    }

    #[test]
    fn test_print_header_with_orchestrator() {
        let mut interceptor = make_test_interceptor();
        interceptor.orchestrator = true;
        interceptor.print_header("test");
    }

    #[test]
    fn test_print_header_with_visual_mode() {
        let mut interceptor = make_test_interceptor();
        interceptor.visual_mode = true;
        interceptor.print_header("test");
    }

    #[test]
    fn test_print_header_with_heatmap_mode() {
        let mut interceptor = make_test_interceptor();
        interceptor.heatmap_mode = true;
        interceptor.print_header("test");
    }

    #[test]
    fn test_print_footer() {
        let interceptor = make_test_interceptor();
        interceptor.print_footer();
    }

    #[test]
    fn test_print_footer_after_processing() {
        let mut interceptor = make_test_interceptor();
        interceptor.token_count = 42;
        interceptor.transformed_count = 21;
        interceptor.print_footer();
    }

    // -- different transform types --

    #[test]
    fn test_process_content_uppercase_transform() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.transform = Transform::Uppercase;
        interceptor.web_tx = Some(tx);

        interceptor.process_content("hello world");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        assert_eq!(events[1].text, "WORLD");
        assert_eq!(events[1].original, "world");
    }

    #[test]
    fn test_process_content_mock_transform() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.transform = Transform::Mock;
        interceptor.web_tx = Some(tx);

        interceptor.process_content("hello world");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        assert_eq!(events[1].text, "wOrLd");
    }

    #[test]
    fn test_process_content_noise_transform() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.transform = Transform::Noise;
        interceptor.web_tx = Some(tx);

        interceptor.process_content("hello world");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        assert!(events[1].text.starts_with("world"));
        assert_eq!(events[1].text.len(), 6); // "world" + 1 noise char
    }

    // -- web_tx none falls back to terminal mode --

    #[test]
    fn test_process_content_terminal_mode_no_crash() {
        let mut interceptor = make_test_interceptor();
        // web_tx is None, so this prints to stdout (terminal mode)
        interceptor.process_content("hello world");
        assert_eq!(interceptor.token_count, 2);
    }

    #[test]
    fn test_process_content_visual_mode_no_crash() {
        let mut interceptor = make_test_interceptor();
        interceptor.visual_mode = true;
        interceptor.process_content("hello world");
        assert_eq!(interceptor.token_count, 2);
    }

    #[test]
    fn test_process_content_heatmap_mode_no_crash() {
        let mut interceptor = make_test_interceptor();
        interceptor.heatmap_mode = true;
        interceptor.process_content("hello world");
        assert_eq!(interceptor.token_count, 2);
    }

    // -- chaos_label field tests --

    #[test]
    fn test_chaos_label_set_for_chaos_transform() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.transform = Transform::Chaos;
        interceptor.web_tx = Some(tx);

        interceptor.process_content("hello world");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        // "world" is the odd token — should have chaos_label
        let known = ["reverse", "uppercase", "mock", "noise"];
        let odd = events.iter().find(|e| e.transformed).expect("should have odd token");
        let label = odd.chaos_label.as_ref().expect("chaos_label should be Some for Chaos transform");
        assert!(known.contains(&label.as_str()), "unexpected label: {}", label);
    }

    #[test]
    fn test_chaos_label_none_for_reverse_transform() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.transform = Transform::Reverse;
        interceptor.web_tx = Some(tx);

        interceptor.process_content("hello world");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        for event in &events {
            assert!(event.chaos_label.is_none(), "Reverse should not set chaos_label");
        }
    }

    #[test]
    fn test_chaos_label_none_for_even_tokens() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.transform = Transform::Chaos;
        interceptor.web_tx = Some(tx);

        interceptor.process_content("hello world foo bar");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        for event in events.iter().filter(|e| !e.transformed) {
            assert!(event.chaos_label.is_none(), "Even tokens should not have chaos_label");
        }
    }

    #[test]
    fn test_chaos_label_serialization() {
        let event = TokenEvent {
            text: "dlrow".to_string(),
            original: "world".to_string(),
            index: 1,
            transformed: true,
            importance: 0.5,
            chaos_label: Some("reverse".to_string()),
            provider: None,
        };
        let json = serde_json::to_string(&event).expect("serialize");
        assert!(json.contains("chaos_label"));
        assert!(json.contains("reverse"));
    }

    #[test]
    fn test_chaos_label_skipped_when_none() {
        let event = TokenEvent {
            text: "hello".to_string(),
            original: "hello".to_string(),
            index: 0,
            transformed: false,
            importance: 0.3,
            chaos_label: None,
            provider: None,
        };
        let json = serde_json::to_string(&event).expect("serialize");
        assert!(!json.contains("chaos_label"), "None chaos_label should be skipped in JSON");
    }

    // -- provider field tests --

    #[test]
    fn test_provider_label_none_by_default() {
        let interceptor = make_test_interceptor();
        assert!(interceptor.web_provider_label.is_none());
    }

    #[test]
    fn test_provider_label_propagates_to_event() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);
        interceptor.web_provider_label = Some("openai".to_string());

        interceptor.process_content("hello world");

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        for event in &events {
            assert_eq!(
                event.provider.as_deref(),
                Some("openai"),
                "provider label should propagate to all events"
            );
        }
    }

    #[test]
    fn test_provider_label_none_means_skipped_in_json() {
        let event = TokenEvent {
            text: "hello".to_string(),
            original: "hello".to_string(),
            index: 0,
            transformed: false,
            importance: 0.5,
            chaos_label: None,
            provider: None,
        };
        let json = serde_json::to_string(&event).expect("serialize");
        assert!(!json.contains("\"provider\""), "None provider should be skipped in JSON");
    }

    #[test]
    fn test_provider_label_some_appears_in_json() {
        let event = TokenEvent {
            text: "hello".to_string(),
            original: "hello".to_string(),
            index: 0,
            transformed: false,
            importance: 0.5,
            chaos_label: None,
            provider: Some("anthropic".to_string()),
        };
        let json = serde_json::to_string(&event).expect("serialize");
        assert!(json.contains("\"provider\""));
        assert!(json.contains("anthropic"));
    }

    #[test]
    fn test_provider_label_openai_and_anthropic_distinct() {
        let (tx1, mut rx1) = mpsc::unbounded_channel::<TokenEvent>();
        let mut openai_i = make_test_interceptor();
        openai_i.web_tx = Some(tx1);
        openai_i.web_provider_label = Some("openai".to_string());

        let (tx2, mut rx2) = mpsc::unbounded_channel::<TokenEvent>();
        let mut anthropic_i = make_test_interceptor();
        anthropic_i.web_tx = Some(tx2);
        anthropic_i.web_provider_label = Some("anthropic".to_string());

        openai_i.process_content("hello");
        anthropic_i.process_content("hello");

        let e1 = rx1.try_recv().expect("openai event");
        let e2 = rx2.try_recv().expect("anthropic event");
        assert_eq!(e1.provider.as_deref(), Some("openai"));
        assert_eq!(e2.provider.as_deref(), Some("anthropic"));
        assert_ne!(e1.provider, e2.provider);
    }
}
