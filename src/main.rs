use clap::{Parser, ValueEnum};
use colored::*;
use rand::Rng;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::io::{self, Write};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;

// ---------------------------------------------------------------------------
// Transform enum
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Transform {
    Reverse,
    Uppercase,
    Mock,
    Noise,
}

impl Transform {
    fn from_str_loose(s: &str) -> Result<Self, String> {
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
            Transform::Mock => token
                .chars()
                .enumerate()
                .map(|(i, c)| {
                    if i % 2 == 0 {
                        c.to_lowercase().next().unwrap_or(c)
                    } else {
                        c.to_uppercase().next().unwrap_or(c)
                    }
                })
                .collect(),
            Transform::Noise => {
                let mut rng = rand::thread_rng();
                let noise_chars = ['*', '+', '~', '@', '#', '$', '%'];
                let noise_char = noise_chars[rng.gen_range(0..noise_chars.len())];
                format!("{}{}", token, noise_char)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Provider selection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ValueEnum, PartialEq)]
pub enum Provider {
    Openai,
    Anthropic,
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Provider::Openai => write!(f, "openai"),
            Provider::Anthropic => write!(f, "anthropic"),
        }
    }
}

// ---------------------------------------------------------------------------
// OpenAI SSE types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct OpenAIChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct OpenAIChatRequest {
    model: String,
    messages: Vec<OpenAIChatMessage>,
    stream: bool,
    temperature: f32,
}

#[derive(Debug, Deserialize)]
struct OpenAIDelta {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    delta: OpenAIDelta,
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChunk {
    choices: Vec<OpenAIChoice>,
}

// ---------------------------------------------------------------------------
// Anthropic SSE types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: u32,
    stream: bool,
    temperature: f32,
}

#[derive(Debug, Deserialize)]
struct AnthropicContentDelta {
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    #[serde(default)]
    delta: Option<AnthropicContentDelta>,
}

// ---------------------------------------------------------------------------
// Orchestrator MCP types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct McpInferRequest {
    jsonrpc: String,
    method: String,
    id: u64,
    params: McpInferParams,
}

#[derive(Debug, Serialize)]
struct McpInferParams {
    name: String,
    arguments: McpInferArguments,
}

#[derive(Debug, Serialize)]
struct McpInferArguments {
    prompt: String,
    worker: String,
}

#[derive(Debug, Deserialize)]
struct McpInferResponse {
    #[allow(dead_code)]
    jsonrpc: Option<String>,
    result: Option<McpInferResult>,
    error: Option<McpError>,
}

#[derive(Debug, Deserialize)]
struct McpInferResult {
    content: Vec<McpContent>,
}

#[derive(Debug, Deserialize)]
struct McpContent {
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct McpError {
    message: String,
}

// ---------------------------------------------------------------------------
// Token event (for web UI streaming)
// ---------------------------------------------------------------------------

/// A single processed token, sent as an SSE event to the web UI.
#[derive(Debug, Clone, Serialize)]
pub struct TokenEvent {
    pub text: String,
    pub index: usize,
    pub transformed: bool,
    pub importance: f64,
}

// ---------------------------------------------------------------------------
// TokenInterceptor — multi-provider streaming engine
// ---------------------------------------------------------------------------

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
    /// When set, token events are sent here instead of printed to stdout.
    web_tx: Option<mpsc::UnboundedSender<TokenEvent>>,
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
            Provider::Openai => env::var("OPENAI_API_KEY").map_err(|_| {
                "OPENAI_API_KEY not set. Export it or pass via environment."
            })?,
            Provider::Anthropic => env::var("ANTHROPIC_API_KEY").map_err(|_| {
                "ANTHROPIC_API_KEY not set. Export it or pass via environment."
            })?,
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
                "[orchestrator] routing through MCP pipeline at localhost:3000"
                    .bright_magenta()
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
    // OpenAI streaming (existing, refactored)
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
    // Anthropic streaming (new)
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

                // Anthropic SSE: "event: content_block_delta" followed by "data: {...}"
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

    async fn orchestrator_infer(
        &self,
        prompt: &str,
    ) -> Result<String, Box<dyn std::error::Error>> {
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
            return Err(format!(
                "Orchestrator returned HTTP {}",
                response.status()
            )
            .into());
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

    fn process_content(&mut self, content: &str) {
        let tokens = self.tokenize(content);

        for token in tokens {
            if !token.trim().is_empty() {
                let is_odd = self.token_count % 2 != 0;
                let importance = self.calculate_token_importance(&token, self.token_count);

                let display_text = if is_odd {
                    self.transformed_count += 1;
                    self.transform.apply(&token)
                } else {
                    token.clone()
                };

                // Web mode: send event through channel
                if let Some(tx) = &self.web_tx {
                    let event = TokenEvent {
                        text: display_text,
                        index: self.token_count,
                        transformed: is_odd,
                        importance,
                    };
                    let _ = tx.send(event);
                } else {
                    // Terminal mode: print with colors
                    if self.heatmap_mode {
                        print!("{}", self.apply_heatmap_color(&display_text, importance));
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

        // Token length
        importance += (token.len() as f64 / 20.0).min(0.3);

        // Position (beginning/end more important)
        let position_factor = if position < 5 || position > 50 {
            0.3
        } else {
            0.1
        };
        importance += position_factor;

        // Uppercase / proper nouns
        if token.chars().any(|c| c.is_uppercase()) {
            importance += 0.2;
        }

        // Content-based importance
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

        // Random jitter
        let mut rng = rand::thread_rng();
        importance += rng.gen_range(-0.1..0.1);

        importance.clamp(0.0, 1.0)
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

    fn print_footer(&self) {
        println!("\n{}", "=".repeat(50).bright_blue());
        println!(
            "Complete! Processed {} tokens.",
            self.token_count
        );
        println!(
            "Transform applied to {} tokens.",
            self.transformed_count
        );
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "every-other-token")]
#[command(version = "3.0.0")]
#[command(about = "A real-time token stream mutator for LLM interpretability research")]
struct Args {
    /// Input prompt to send to the LLM
    prompt: String,

    /// Transformation type (reverse, uppercase, mock, noise)
    #[arg(default_value = "reverse")]
    transform: String,

    /// Model name (e.g. gpt-4, claude-sonnet-4-20250514)
    #[arg(default_value = "gpt-3.5-turbo")]
    model: String,

    /// LLM provider: openai or anthropic
    #[arg(long, value_enum, default_value = "openai")]
    provider: Provider,

    /// Enable visual mode with color-coded tokens
    #[arg(long, short)]
    visual: bool,

    /// Enable token importance heatmap
    #[arg(long)]
    heatmap: bool,

    /// Route through tokio-prompt-orchestrator MCP pipeline at localhost:3000
    #[arg(long)]
    orchestrator: bool,

    /// Launch web UI on localhost instead of terminal output
    #[arg(long)]
    web: bool,

    /// Port for the web UI server
    #[arg(long, default_value = "8888")]
    port: u16,
}

// ---------------------------------------------------------------------------
// Web UI server (zero external deps — raw tokio TCP + embedded HTML)
// ---------------------------------------------------------------------------

mod web {
    use super::*;
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;

    /// Embedded single-page HTML application.
    const INDEX_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Every Other Token</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font-family:'Cascadia Code','Fira Code',monospace;min-height:100vh;display:flex;flex-direction:column}
header{padding:24px 32px 16px;border-bottom:1px solid #21262d}
header h1{font-size:1.4rem;color:#58a6ff;margin-bottom:4px}
header p{font-size:.85rem;color:#8b949e}
.controls{display:flex;gap:12px;padding:16px 32px;flex-wrap:wrap;align-items:end;border-bottom:1px solid #21262d;background:#161b22}
.field{display:flex;flex-direction:column;gap:4px}
.field label{font-size:.75rem;color:#8b949e;text-transform:uppercase;letter-spacing:.5px}
.field input,.field select{background:#0d1117;border:1px solid #30363d;color:#c9d1d9;padding:8px 12px;border-radius:6px;font-family:inherit;font-size:.9rem}
.field input:focus,.field select:focus{outline:none;border-color:#58a6ff}
.field input[type=text]{min-width:320px}
button#start{background:#238636;color:#fff;border:none;padding:8px 20px;border-radius:6px;font-family:inherit;font-size:.9rem;cursor:pointer;align-self:end}
button#start:hover{background:#2ea043}
button#start:disabled{background:#21262d;color:#484f58;cursor:not-allowed}
.toggle{display:flex;align-items:center;gap:6px;font-size:.85rem;color:#8b949e;cursor:pointer;user-select:none}
.toggle input{accent-color:#58a6ff}
#output{flex:1;padding:24px 32px;line-height:1.8;font-size:1.05rem;overflow-y:auto;white-space:pre-wrap;word-wrap:break-word}
.token{display:inline;animation:fadeIn .15s ease-in}
.token.odd{color:#00d4ff;font-weight:bold}
.token.even{color:#c9d1d9}
/* heatmap backgrounds */
.heat-4{background:#da3633;color:#fff}
.heat-3{background:#b62324;color:#fff}
.heat-2{background:#9e6a03;color:#000}
.heat-1{background:#1f6feb;color:#fff}
.heat-0{background:transparent}
@keyframes fadeIn{from{opacity:0;transform:translateY(2px)}to{opacity:1;transform:translateY(0)}}
#stats{padding:12px 32px;border-top:1px solid #21262d;font-size:.8rem;color:#8b949e;background:#161b22}
</style>
</head>
<body>
<header>
  <h1>Every Other Token</h1>
  <p>Real-time token stream mutator for LLM interpretability research</p>
</header>
<div class="controls">
  <div class="field">
    <label>Prompt</label>
    <input type="text" id="prompt" value="Tell me a story about a robot" placeholder="Enter prompt...">
  </div>
  <div class="field">
    <label>Transform</label>
    <select id="transform">
      <option value="reverse">reverse</option>
      <option value="uppercase">uppercase</option>
      <option value="mock">mock</option>
      <option value="noise">noise</option>
    </select>
  </div>
  <div class="field">
    <label>Provider</label>
    <select id="provider">
      <option value="openai">OpenAI</option>
      <option value="anthropic">Anthropic</option>
    </select>
  </div>
  <div class="field">
    <label>Model</label>
    <input type="text" id="model" value="" placeholder="auto" style="min-width:180px">
  </div>
  <label class="toggle"><input type="checkbox" id="heatmap"> Heatmap</label>
  <button id="start">Stream</button>
</div>
<div id="output"></div>
<div id="stats"></div>
<script>
const $=s=>document.querySelector(s);
let es=null;
$('#start').onclick=()=>{
  if(es){es.close();es=null}
  const out=$('#output');out.innerHTML='';
  $('#stats').textContent='';
  const p=encodeURIComponent($('#prompt').value);
  const t=$('#transform').value;
  const prov=$('#provider').value;
  const m=encodeURIComponent($('#model').value);
  const hm=$('#heatmap').checked?'1':'0';
  const url='/stream?prompt='+p+'&transform='+t+'&provider='+prov+'&model='+m+'&heatmap='+hm;
  $('#start').disabled=true;$('#start').textContent='Streaming...';
  let count=0,xformed=0;
  es=new EventSource(url);
  es.onmessage=e=>{
    if(e.data==='[DONE]'){es.close();es=null;$('#start').disabled=false;$('#start').textContent='Stream';return}
    try{
      const tk=JSON.parse(e.data);
      const span=document.createElement('span');
      span.className='token '+(tk.transformed?'odd':'even');
      if($('#heatmap').checked){
        const h=tk.importance>=.8?4:tk.importance>=.6?3:tk.importance>=.4?2:tk.importance>=.2?1:0;
        span.classList.add('heat-'+h);
      }
      span.textContent=tk.text;
      out.appendChild(span);
      out.scrollTop=out.scrollHeight;
      count++;if(tk.transformed)xformed++;
      $('#stats').textContent='Tokens: '+count+' | Transformed: '+xformed;
    }catch(_){}
  };
  es.onerror=()=>{es.close();es=null;$('#start').disabled=false;$('#start').textContent='Stream'};
};
</script>
</body>
</html>"##;

    /// Simple percent-decoding for URL query parameters.
    pub fn url_decode(s: &str) -> String {
        let mut result = String::with_capacity(s.len());
        let mut chars = s.chars();
        while let Some(c) = chars.next() {
            match c {
                '+' => result.push(' '),
                '%' => {
                    let hex: String = chars.by_ref().take(2).collect();
                    if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                        result.push(byte as char);
                    }
                }
                _ => result.push(c),
            }
        }
        result
    }

    /// Parse query string into key-value pairs.
    pub fn parse_query(query: &str) -> std::collections::HashMap<String, String> {
        query
            .split('&')
            .filter_map(|pair| {
                let mut parts = pair.splitn(2, '=');
                let key = parts.next()?;
                let val = parts.next().unwrap_or("");
                Some((key.to_string(), url_decode(val)))
            })
            .collect()
    }

    /// Start the web UI server and open the browser.
    pub async fn serve(port: u16, default_args: &Args) -> Result<(), Box<dyn std::error::Error>> {
        let listener = TcpListener::bind(format!("127.0.0.1:{}", port)).await?;

        eprintln!(
            "{}",
            format!("  Web UI running at http://localhost:{}", port).bright_green()
        );
        eprintln!(
            "{}",
            "  Press Ctrl+C to stop.".bright_blue()
        );

        // Try to open the browser
        #[cfg(target_os = "windows")]
        {
            let _ = std::process::Command::new("cmd")
                .args(["/C", &format!("start http://localhost:{}", port)])
                .spawn();
        }
        #[cfg(target_os = "macos")]
        {
            let _ = std::process::Command::new("open")
                .arg(format!("http://localhost:{}", port))
                .spawn();
        }
        #[cfg(target_os = "linux")]
        {
            let _ = std::process::Command::new("xdg-open")
                .arg(format!("http://localhost:{}", port))
                .spawn();
        }

        let default_provider = default_args.provider.clone();
        let orchestrator = default_args.orchestrator;

        loop {
            let (stream, _addr) = listener.accept().await?;
            let provider = default_provider.clone();
            tokio::spawn(async move {
                if let Err(e) = handle_connection(stream, provider, orchestrator).await {
                    eprintln!("  connection error: {}", e);
                }
            });
        }
    }

    async fn handle_connection(
        mut stream: tokio::net::TcpStream,
        default_provider: Provider,
        orchestrator: bool,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use tokio::io::AsyncReadExt;

        let mut buf = vec![0u8; 8192];
        let n = stream.read(&mut buf).await?;
        let request = String::from_utf8_lossy(&buf[..n]);

        // Parse the request line: "GET /path?query HTTP/1.1"
        let first_line = request.lines().next().unwrap_or("");
        let parts: Vec<&str> = first_line.split_whitespace().collect();
        if parts.len() < 2 {
            return Ok(());
        }
        let path_and_query = parts[1];

        // Split path and query
        let (path, query_str) = if let Some(idx) = path_and_query.find('?') {
            (&path_and_query[..idx], &path_and_query[idx + 1..])
        } else {
            (path_and_query, "")
        };

        match path {
            "/" => {
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    INDEX_HTML.len(),
                    INDEX_HTML,
                );
                stream.write_all(response.as_bytes()).await?;
            }
            "/stream" => {
                let params = parse_query(query_str);
                let prompt = params.get("prompt").cloned().unwrap_or_default();
                let transform_str = params.get("transform").cloned().unwrap_or_else(|| "reverse".to_string());
                let provider_str = params.get("provider").cloned().unwrap_or_else(|| default_provider.to_string());
                let model_input = params.get("model").cloned().unwrap_or_default();
                let heatmap = params.get("heatmap").map_or(false, |v| v == "1");

                let provider = match provider_str.as_str() {
                    "anthropic" => Provider::Anthropic,
                    _ => Provider::Openai,
                };

                let model = if model_input.is_empty() {
                    match provider {
                        Provider::Openai => "gpt-3.5-turbo".to_string(),
                        Provider::Anthropic => "claude-sonnet-4-20250514".to_string(),
                    }
                } else {
                    model_input
                };

                let transform = Transform::from_str_loose(&transform_str)
                    .unwrap_or(Transform::Reverse);

                // SSE headers
                let headers = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
                stream.write_all(headers.as_bytes()).await?;

                // Create channel for token events
                let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();

                let interceptor_result = TokenInterceptor::new(
                    provider,
                    transform,
                    model,
                    true,     // visual mode always on for web
                    heatmap,
                    orchestrator,
                );

                // Convert result early — stringify the error before any await
                // to satisfy Send bounds on the spawned task.
                let interceptor_result = interceptor_result.map_err(|e| e.to_string());
                let mut interceptor = match interceptor_result {
                    Ok(mut i) => {
                        i.web_tx = Some(tx);
                        i
                    }
                    Err(msg) => {
                        let err_event = format!(
                            "data: {{\"error\": \"{}\"}}\n\ndata: [DONE]\n\n",
                            msg.replace('"', "'")
                        );
                        stream.write_all(err_event.as_bytes()).await?;
                        return Ok(());
                    }
                };

                // Spawn the LLM streaming in background
                let prompt_clone = prompt.clone();
                let stream_task = tokio::spawn(async move {
                    let _ = interceptor.intercept_stream(&prompt_clone).await;
                });

                // Forward token events as SSE
                while let Some(event) = rx.recv().await {
                    if let Ok(json) = serde_json::to_string(&event) {
                        let sse = format!("data: {}\n\n", json);
                        if stream.write_all(sse.as_bytes()).await.is_err() {
                            break;
                        }
                    }
                }

                let _ = stream_task.await;

                // Send done signal
                let _ = stream.write_all(b"data: [DONE]\n\n").await;
            }
            _ => {
                let response = "HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\nConnection: close\r\n\r\nNot Found";
                stream.write_all(response.as_bytes()).await?;
            }
        }

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Web UI mode
    if args.web {
        web::serve(args.port, &args).await?;
        return Ok(());
    }

    let transform = Transform::from_str_loose(&args.transform)
        .map_err(|e| format!("Invalid transform: {}", e))?;

    // Auto-select a sensible default model when switching providers
    let model = if args.provider == Provider::Anthropic && args.model == "gpt-3.5-turbo" {
        "claude-sonnet-4-20250514".to_string()
    } else {
        args.model
    };

    let mut interceptor = TokenInterceptor::new(
        args.provider,
        transform,
        model,
        args.visual,
        args.heatmap,
        args.orchestrator,
    )?;

    interceptor.intercept_stream(&args.prompt).await?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Transform unit tests -----------------------------------------------

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
        assert!(matches!(
            Transform::from_str_loose("reverse"),
            Ok(Transform::Reverse)
        ));
        assert!(matches!(
            Transform::from_str_loose("uppercase"),
            Ok(Transform::Uppercase)
        ));
        assert!(matches!(
            Transform::from_str_loose("mock"),
            Ok(Transform::Mock)
        ));
        assert!(matches!(
            Transform::from_str_loose("noise"),
            Ok(Transform::Noise)
        ));
        assert!(Transform::from_str_loose("invalid").is_err());
    }

    #[test]
    fn test_transform_from_str_case_insensitive() {
        assert!(matches!(
            Transform::from_str_loose("REVERSE"),
            Ok(Transform::Reverse)
        ));
        assert!(matches!(
            Transform::from_str_loose("Uppercase"),
            Ok(Transform::Uppercase)
        ));
        assert!(matches!(
            Transform::from_str_loose("MoCk"),
            Ok(Transform::Mock)
        ));
    }

    // -- Provider tests -----------------------------------------------------

    #[test]
    fn test_provider_display() {
        assert_eq!(Provider::Openai.to_string(), "openai");
        assert_eq!(Provider::Anthropic.to_string(), "anthropic");
    }

    #[test]
    fn test_provider_equality() {
        assert_eq!(Provider::Openai, Provider::Openai);
        assert_eq!(Provider::Anthropic, Provider::Anthropic);
        assert_ne!(Provider::Openai, Provider::Anthropic);
    }

    // -- Tokenizer tests ----------------------------------------------------

    #[test]
    fn test_tokenize_simple_sentence() {
        let interceptor = make_test_interceptor();
        let tokens = interceptor.tokenize("hello world");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
    }

    #[test]
    fn test_tokenize_with_punctuation() {
        let interceptor = make_test_interceptor();
        let tokens = interceptor.tokenize("hello, world!");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&",".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"!".to_string()));
    }

    #[test]
    fn test_tokenize_empty_string() {
        let interceptor = make_test_interceptor();
        let tokens = interceptor.tokenize("");
        assert!(tokens.is_empty());
    }

    // -- Heatmap importance tests -------------------------------------------

    #[test]
    fn test_importance_clamped_to_unit_interval() {
        let interceptor = make_test_interceptor();
        for pos in 0..100 {
            let imp = interceptor.calculate_token_importance("test", pos);
            assert!(imp >= 0.0 && imp <= 1.0, "importance out of range: {}", imp);
        }
    }

    #[test]
    fn test_punctuation_has_low_importance() {
        let interceptor = make_test_interceptor();
        let mut total = 0.0;
        let n = 100;
        for _ in 0..n {
            total += interceptor.calculate_token_importance(".", 25);
        }
        let avg = total / n as f64;
        assert!(avg < 0.3, "punctuation avg importance too high: {}", avg);
    }

    // -- MCP request serialization ------------------------------------------

    #[test]
    fn test_mcp_request_serializes() {
        let req = McpInferRequest {
            jsonrpc: "2.0".to_string(),
            method: "tools/call".to_string(),
            id: 1,
            params: McpInferParams {
                name: "infer".to_string(),
                arguments: McpInferArguments {
                    prompt: "hello".to_string(),
                    worker: "llama_cpp".to_string(),
                },
            },
        };
        let json = serde_json::to_string(&req).expect("serialization failed");
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"worker\":\"llama_cpp\""));
        assert!(json.contains("\"prompt\":\"hello\""));
    }

    #[test]
    fn test_mcp_response_deserializes_success() {
        let json = r#"{"jsonrpc":"2.0","result":{"content":[{"text":"enriched prompt"}]}}"#;
        let resp: McpInferResponse = serde_json::from_str(json).expect("deser failed");
        assert!(resp.error.is_none());
        let text = resp
            .result
            .as_ref()
            .and_then(|r| r.content.first())
            .and_then(|c| c.text.as_ref());
        assert_eq!(text, Some(&"enriched prompt".to_string()));
    }

    #[test]
    fn test_mcp_response_deserializes_error() {
        let json = r#"{"jsonrpc":"2.0","error":{"message":"pipeline down"}}"#;
        let resp: McpInferResponse = serde_json::from_str(json).expect("deser failed");
        assert!(resp.result.is_none());
        assert_eq!(
            resp.error.as_ref().map(|e| &e.message[..]),
            Some("pipeline down")
        );
    }

    // -- Anthropic SSE event parsing ----------------------------------------

    #[test]
    fn test_anthropic_content_block_delta_deserializes() {
        let json = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;
        let event: AnthropicStreamEvent = serde_json::from_str(json).expect("deser failed");
        assert_eq!(event.event_type, "content_block_delta");
        assert_eq!(
            event
                .delta
                .as_ref()
                .and_then(|d| d.text.as_ref())
                .map(|s| s.as_str()),
            Some("Hello")
        );
    }

    #[test]
    fn test_anthropic_message_start_deserializes() {
        let json = r#"{"type":"message_start","message":{"id":"msg_123"}}"#;
        let event: AnthropicStreamEvent = serde_json::from_str(json).expect("deser failed");
        assert_eq!(event.event_type, "message_start");
        assert!(event.delta.is_none());
    }

    // -- OpenAI SSE chunk parsing -------------------------------------------

    #[test]
    fn test_openai_chunk_deserializes() {
        let json = r#"{"id":"chatcmpl-abc","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}"#;
        let chunk: OpenAIChunk = serde_json::from_str(json).expect("deser failed");
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(
            chunk.choices[0].delta.content.as_ref().map(|s| s.as_str()),
            Some("Hi")
        );
    }

    #[test]
    fn test_openai_chunk_empty_delta() {
        let json = r#"{"id":"chatcmpl-abc","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#;
        let chunk: OpenAIChunk = serde_json::from_str(json).expect("deser failed");
        assert!(chunk.choices[0].delta.content.is_none());
    }

    // -- Auto model selection -----------------------------------------------

    #[test]
    fn test_anthropic_auto_model_selection() {
        let model = "gpt-3.5-turbo";
        let result = if Provider::Anthropic == Provider::Anthropic && model == "gpt-3.5-turbo" {
            "claude-sonnet-4-20250514"
        } else {
            model
        };
        assert_eq!(result, "claude-sonnet-4-20250514");
    }

    // -- Web mode: token events via channel -----------------------------------

    #[test]
    fn test_web_tx_sends_token_events() {
        let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
        let mut interceptor = make_test_interceptor();
        interceptor.web_tx = Some(tx);

        interceptor.process_content("hello world");

        let mut events: Vec<TokenEvent> = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }

        // "hello", " ", "world" — whitespace is skipped for non-empty check
        // "hello" (even, index 0), "world" (odd, index 1)
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].text, "hello");
        assert!(!events[0].transformed);
        assert_eq!(events[0].index, 0);
        // "world" reversed = "dlrow"
        assert_eq!(events[1].text, "dlrow");
        assert!(events[1].transformed);
        assert_eq!(events[1].index, 1);
    }

    #[test]
    fn test_web_tx_token_event_serializes() {
        let event = TokenEvent {
            text: "hello".to_string(),
            index: 0,
            transformed: false,
            importance: 0.5,
        };
        let json = serde_json::to_string(&event).expect("serialize failed");
        assert!(json.contains("\"text\":\"hello\""));
        assert!(json.contains("\"transformed\":false"));
        assert!(json.contains("\"importance\":0.5"));
    }

    #[test]
    fn test_url_decode() {
        assert_eq!(web::url_decode("hello+world"), "hello world");
        assert_eq!(web::url_decode("hello%20world"), "hello world");
        assert_eq!(web::url_decode("a%26b"), "a&b");
        assert_eq!(web::url_decode("plain"), "plain");
    }

    #[test]
    fn test_parse_query() {
        let params = web::parse_query("prompt=hello+world&transform=reverse&heatmap=1");
        assert_eq!(params.get("prompt").map(|s| s.as_str()), Some("hello world"));
        assert_eq!(params.get("transform").map(|s| s.as_str()), Some("reverse"));
        assert_eq!(params.get("heatmap").map(|s| s.as_str()), Some("1"));
    }

    #[test]
    fn test_parse_query_empty() {
        let params = web::parse_query("");
        assert!(params.is_empty() || params.get("").map_or(true, |v| v.is_empty()));
    }

    // -- Helper -------------------------------------------------------------

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
        }
    }
}
