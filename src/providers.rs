//! Provider plugin system and SSE wire types.
//!
//! Each supported LLM provider is represented by a zero-sized struct that
//! implements [`ProviderPlugin`].  The [`TokenInterceptor`](crate::TokenInterceptor)
//! selects the appropriate plugin at construction time and uses it to build
//! authenticated HTTP requests and parse streaming responses.
//!
//! ## Supported providers
//!
//! | Variant | Plugin | Endpoint |
//! |---------|--------|----------|
//! | `openai` | [`OpenAiPlugin`] | `https://api.openai.com/v1/chat/completions` |
//! | `anthropic` | [`AnthropicPlugin`] | `https://api.anthropic.com/v1/messages` |
//! | `mock` | (inline fixture) | n/a -- returns canned tokens for tests |

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Provider plugin trait
// ---------------------------------------------------------------------------

/// Trait implemented by each provider plug-in.
///
/// Concrete implementations ([`OpenAiPlugin`], [`AnthropicPlugin`]) supply the
/// provider-specific details (URL, authentication header, request shape) so
/// that the token interceptor can switch providers without branching.
pub trait ProviderPlugin: Send + Sync {
    /// Lowercase display name of the provider (e.g. `"openai"`, `"anthropic"`).
    fn name(&self) -> &str;
    /// Default model string to use when the user has not explicitly chosen one.
    fn default_model(&self) -> &str;
    /// Base URL for the provider's streaming chat completions endpoint.
    ///
    /// This centralises API endpoint knowledge so that callers building HTTP
    /// requests do not hard-code provider URLs and future providers only need
    /// to implement this single method.
    fn api_url(&self) -> &str;
    /// Build a JSON request body for the provider's streaming chat API.
    fn build_request(&self, prompt: &str, system: Option<&str>, model: &str) -> serde_json::Value;
}

/// Provider plug-in for the OpenAI Chat Completions API.
pub struct OpenAiPlugin;
/// Provider plug-in for the Anthropic Messages API.
pub struct AnthropicPlugin;

impl ProviderPlugin for OpenAiPlugin {
    fn name(&self) -> &str {
        "openai"
    }
    fn default_model(&self) -> &str {
        "gpt-3.5-turbo"
    }
    fn api_url(&self) -> &str {
        "https://api.openai.com/v1/chat/completions"
    }
    fn build_request(&self, prompt: &str, system: Option<&str>, model: &str) -> serde_json::Value {
        let mut messages = Vec::new();
        if let Some(sys) = system {
            messages.push(serde_json::json!({ "role": "system", "content": sys }));
        }
        messages.push(serde_json::json!({ "role": "user", "content": prompt }));
        serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": true,
            "temperature": 0.7,
            "logprobs": true,
            "top_logprobs": 5,
        })
    }
}

/// Anthropic API version header value. Update here when Anthropic releases a new stable version.
/// As of 2026-03: Anthropic has not published a newer stable version header. Revisit quarterly.
pub const ANTHROPIC_API_VERSION: &str = "2023-06-01";

impl ProviderPlugin for AnthropicPlugin {
    fn name(&self) -> &str {
        "anthropic"
    }
    fn default_model(&self) -> &str {
        "claude-sonnet-4-6"
    }
    fn api_url(&self) -> &str {
        "https://api.anthropic.com/v1/messages"
    }
    fn build_request(&self, prompt: &str, system: Option<&str>, model: &str) -> serde_json::Value {
        let mut req = serde_json::json!({
            "model": model,
            "messages": [{ "role": "user", "content": prompt }],
            "max_tokens": 1024,
            "stream": true,
            "temperature": 0.7,
        });
        if let Some(sys) = system {
            req["system"] = serde_json::Value::String(sys.to_string());
        }
        req
    }
}

// -- Token probability / logprob types --------------------------------------

/// One alternative token returned alongside a logprob entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAITopLogprob {
    pub token: String,
    pub logprob: f32,
}

/// Per-API-token logprob entry in a streaming chunk.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAILogprobContent {
    pub token: String,
    pub logprob: f32,
    #[serde(default)]
    pub top_logprobs: Vec<OpenAITopLogprob>,
}

/// The `logprobs` block on a streaming choice.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIChunkLogprobs {
    #[serde(default)]
    pub content: Vec<OpenAILogprobContent>,
}

/// Selectable LLM provider.
///
/// Used as a CLI argument (`--provider`) and throughout the codebase to branch
/// on provider-specific behaviour.
#[derive(Debug, Clone, ValueEnum, PartialEq)]
pub enum Provider {
    /// OpenAI Chat Completions API (GPT-3.5, GPT-4, etc.).
    Openai,
    /// Anthropic Messages API (Claude family).
    Anthropic,
    /// In-process mock provider for tests and dry-run mode.
    Mock,
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Provider::Openai => write!(f, "openai"),
            Provider::Anthropic => write!(f, "anthropic"),
            Provider::Mock => write!(f, "mock"),
        }
    }
}

impl std::str::FromStr for Provider {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "openai" => Ok(Provider::Openai),
            "anthropic" => Ok(Provider::Anthropic),
            "mock" => Ok(Provider::Mock),
            other => Err(format!(
                "unknown provider: '{}' (expected openai, anthropic, or mock)",
                other
            )),
        }
    }
}

// -- OpenAI SSE types -------------------------------------------------------

/// A single message in an OpenAI chat request (role + content pair).
#[derive(Debug, Serialize)]
pub struct OpenAIChatMessage {
    /// Role of the message author: `"system"`, `"user"`, or `"assistant"`.
    pub role: String,
    /// Text content of the message.
    pub content: String,
}

/// Full JSON body for an OpenAI streaming chat completions request.
#[derive(Debug, Serialize)]
pub struct OpenAIChatRequest {
    /// Model identifier (e.g. `"gpt-4"`).
    pub model: String,
    /// Conversation history including the current user turn.
    pub messages: Vec<OpenAIChatMessage>,
    /// Must be `true` to enable SSE streaming.
    pub stream: bool,
    /// Sampling temperature (0.0–2.0).
    pub temperature: f32,
    /// Whether to include per-token log probabilities in the response.
    pub logprobs: bool,
    /// Number of top alternative tokens per position (0–20).
    pub top_logprobs: u8,
}

/// Incremental content fragment within a streaming choice delta.
#[derive(Debug, Deserialize)]
pub struct OpenAIDelta {
    /// Text fragment, absent on the final chunk where `finish_reason` is set.
    pub content: Option<String>,
}

/// One streaming choice from an OpenAI chunk event.
#[derive(Debug, Deserialize)]
pub struct OpenAIChoice {
    /// Incremental content delta for this chunk.
    pub delta: OpenAIDelta,
    /// Populated on the final chunk (`"stop"`, `"length"`, etc.).
    #[allow(dead_code)]
    pub finish_reason: Option<String>,
    /// Log probability data, present when `logprobs=true` was requested.
    #[serde(default)]
    pub logprobs: Option<OpenAIChunkLogprobs>,
}

/// One server-sent event chunk from the OpenAI streaming API.
#[derive(Debug, Deserialize)]
pub struct OpenAIChunk {
    /// List of choice objects (typically one entry for non-parallel requests).
    pub choices: Vec<OpenAIChoice>,
}

// -- Anthropic SSE types ----------------------------------------------------

/// A single message in an Anthropic Messages API request.
#[derive(Debug, Serialize)]
pub struct AnthropicMessage {
    /// Role: `"user"` or `"assistant"`.
    pub role: String,
    /// Text content of the message.
    pub content: String,
}

/// Full JSON body for an Anthropic streaming messages request.
#[derive(Debug, Serialize)]
pub struct AnthropicRequest {
    /// Model identifier (e.g. `"claude-sonnet-4-6"`).
    pub model: String,
    /// Conversation messages.
    pub messages: Vec<AnthropicMessage>,
    /// Maximum tokens to generate.
    pub max_tokens: u32,
    /// Must be `true` to enable SSE streaming.
    pub stream: bool,
    /// Sampling temperature (0.0–1.0 for Anthropic).
    pub temperature: f32,
    /// Optional system prompt prepended before the conversation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
}

/// Incremental text delta from an Anthropic `content_block_delta` event.
#[derive(Debug, Deserialize)]
pub struct AnthropicContentDelta {
    /// New text fragment, present only on `text_delta` sub-events.
    #[serde(default)]
    pub text: Option<String>,
}

/// One server-sent event from the Anthropic streaming API.
#[derive(Debug, Deserialize)]
pub struct AnthropicStreamEvent {
    /// Event type: `"content_block_delta"`, `"message_start"`, `"ping"`, etc.
    #[serde(rename = "type")]
    pub event_type: String,
    /// Content delta, present only on `content_block_delta` events.
    #[serde(default)]
    pub delta: Option<AnthropicContentDelta>,
}

// -- Orchestrator MCP types -------------------------------------------------

/// JSON-RPC 2.0 request sent to the MCP orchestrator (`tools/call infer`).
#[derive(Debug, Serialize)]
pub struct McpInferRequest {
    /// Always `"2.0"`.
    pub jsonrpc: String,
    /// Always `"tools/call"`.
    pub method: String,
    /// Caller-assigned request identifier.
    pub id: u64,
    /// Typed parameters block.
    pub params: McpInferParams,
}

/// Parameters block for an MCP `tools/call` request.
#[derive(Debug, Serialize)]
pub struct McpInferParams {
    /// Tool name to invoke (e.g. `"infer"`).
    pub name: String,
    /// Arguments forwarded to the tool.
    pub arguments: McpInferArguments,
}

/// Arguments forwarded to the MCP `infer` tool.
#[derive(Debug, Serialize)]
pub struct McpInferArguments {
    /// Text prompt to enrich or process.
    pub prompt: String,
    /// Target worker backend (e.g. `"llama_cpp"`).
    pub worker: String,
}

/// JSON-RPC 2.0 response returned by the MCP orchestrator.
#[derive(Debug, Deserialize)]
pub struct McpInferResponse {
    /// Protocol version echo, always `"2.0"` when present.
    #[allow(dead_code)]
    pub jsonrpc: Option<String>,
    /// Successful result payload; mutually exclusive with `error`.
    pub result: Option<McpInferResult>,
    /// Error payload; mutually exclusive with `result`.
    pub error: Option<McpError>,
}

/// Successful result from an MCP `infer` call.
#[derive(Debug, Deserialize)]
pub struct McpInferResult {
    /// List of content items returned by the tool.
    pub content: Vec<McpContent>,
}

/// A single content item within an MCP tool result.
#[derive(Debug, Deserialize)]
pub struct McpContent {
    /// Text value of this content item, or `None` for non-text items.
    pub text: Option<String>,
}

/// Error payload from an MCP JSON-RPC response.
#[derive(Debug, Deserialize)]
pub struct McpError {
    /// Human-readable error message.
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_display() {
        assert_eq!(Provider::Openai.to_string(), "openai");
        assert_eq!(Provider::Anthropic.to_string(), "anthropic");
        assert_eq!(Provider::Mock.to_string(), "mock");
    }

    #[test]
    fn test_provider_equality() {
        assert_eq!(Provider::Openai, Provider::Openai);
        assert_eq!(Provider::Anthropic, Provider::Anthropic);
        assert_eq!(Provider::Mock, Provider::Mock);
        assert_ne!(Provider::Openai, Provider::Anthropic);
        assert_ne!(Provider::Openai, Provider::Mock);
        assert_ne!(Provider::Anthropic, Provider::Mock);
    }

    #[test]
    fn test_provider_openai_display_lowercase() {
        let s = format!("{}", Provider::Openai);
        assert_eq!(s, "openai");
        assert!(s.chars().all(|c| c.is_lowercase() || c.is_alphanumeric()));
    }

    #[test]
    fn test_provider_anthropic_display_lowercase() {
        assert_eq!(format!("{}", Provider::Anthropic), "anthropic");
    }

    #[test]
    fn test_provider_clone() {
        let p = Provider::Openai;
        let p2 = p.clone();
        assert_eq!(p, p2);
    }

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
        let json =
            r#"{"id":"chatcmpl-abc","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#;
        let chunk: OpenAIChunk = serde_json::from_str(json).expect("deser failed");
        assert!(chunk.choices[0].delta.content.is_none());
    }

    #[test]
    fn test_mcp_infer_request_contains_all_fields() {
        let req = McpInferRequest {
            jsonrpc: "2.0".to_string(),
            method: "tools/call".to_string(),
            id: 42,
            params: McpInferParams {
                name: "infer".to_string(),
                arguments: McpInferArguments {
                    prompt: "test prompt".to_string(),
                    worker: "llama_cpp".to_string(),
                },
            },
        };
        let json = serde_json::to_string(&req).expect("serialize");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("parse");
        assert_eq!(parsed["jsonrpc"], "2.0");
        assert_eq!(parsed["method"], "tools/call");
        assert_eq!(parsed["id"], 42);
        assert_eq!(parsed["params"]["arguments"]["prompt"], "test prompt");
    }

    #[test]
    fn test_mcp_response_empty_content() {
        let json = r#"{"jsonrpc":"2.0","result":{"content":[]}}"#;
        let resp: McpInferResponse = serde_json::from_str(json).expect("deser");
        assert!(resp.result.as_ref().expect("result").content.is_empty());
    }

    #[test]
    fn test_mcp_response_null_text() {
        let json = r#"{"jsonrpc":"2.0","result":{"content":[{"text":null}]}}"#;
        let resp: McpInferResponse = serde_json::from_str(json).expect("deser");
        assert!(resp.result.as_ref().expect("result").content[0]
            .text
            .is_none());
    }

    #[test]
    fn test_openai_chunk_multiple_choices() {
        let json = r#"{"id":"chatcmpl-x","choices":[{"index":0,"delta":{"content":"A"},"finish_reason":null},{"index":1,"delta":{"content":"B"},"finish_reason":null}]}"#;
        let chunk: OpenAIChunk = serde_json::from_str(json).expect("deser");
        assert_eq!(chunk.choices.len(), 2);
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("A"));
        assert_eq!(chunk.choices[1].delta.content.as_deref(), Some("B"));
    }

    #[test]
    fn test_openai_chunk_no_choices() {
        let json = r#"{"id":"chatcmpl-x","choices":[]}"#;
        let chunk: OpenAIChunk = serde_json::from_str(json).expect("deser");
        assert!(chunk.choices.is_empty());
    }

    #[test]
    fn test_anthropic_event_message_delta() {
        let json = r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#;
        let event: AnthropicStreamEvent = serde_json::from_str(json).expect("deser");
        assert_eq!(event.event_type, "message_delta");
        assert!(event.delta.as_ref().and_then(|d| d.text.as_ref()).is_none());
    }

    #[test]
    fn test_anthropic_event_ping() {
        let json = r#"{"type":"ping"}"#;
        let event: AnthropicStreamEvent = serde_json::from_str(json).expect("deser");
        assert_eq!(event.event_type, "ping");
        assert!(event.delta.is_none());
    }

    // -- Logprob type tests --

    #[test]
    fn test_openai_chat_request_has_logprobs_fields() {
        let req = OpenAIChatRequest {
            model: "gpt-4".to_string(),
            messages: vec![OpenAIChatMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
            }],
            stream: true,
            temperature: 0.7,
            logprobs: true,
            top_logprobs: 5,
        };
        let json = serde_json::to_string(&req).expect("serialize");
        assert!(json.contains("\"logprobs\":true"));
        assert!(json.contains("\"top_logprobs\":5"));
    }

    #[test]
    fn test_openai_top_logprob_deserializes() {
        let json = r#"{"token":"hello","logprob":-0.5}"#;
        let tlp: OpenAITopLogprob = serde_json::from_str(json).expect("deser");
        assert_eq!(tlp.token, "hello");
        assert!((tlp.logprob - (-0.5)).abs() < 1e-5);
    }

    #[test]
    fn test_openai_logprob_content_deserializes() {
        let json = r#"{"token":"world","logprob":-1.2,"top_logprobs":[{"token":"world","logprob":-1.2},{"token":"earth","logprob":-2.5}]}"#;
        let lc: OpenAILogprobContent = serde_json::from_str(json).expect("deser");
        assert_eq!(lc.token, "world");
        assert_eq!(lc.top_logprobs.len(), 2);
        assert_eq!(lc.top_logprobs[1].token, "earth");
    }

    #[test]
    fn test_openai_chunk_logprobs_empty_content() {
        let json = r#"{"content":[]}"#;
        let cl: OpenAIChunkLogprobs = serde_json::from_str(json).expect("deser");
        assert!(cl.content.is_empty());
    }

    #[test]
    fn test_openai_choice_with_logprobs_deserializes() {
        let json = r#"{"delta":{"content":"Hi"},"finish_reason":null,"logprobs":{"content":[{"token":"Hi","logprob":-0.1,"top_logprobs":[{"token":"Hi","logprob":-0.1},{"token":"Hey","logprob":-2.3}]}]}}"#;
        let choice: OpenAIChoice = serde_json::from_str(json).expect("deser");
        assert_eq!(choice.delta.content.as_deref(), Some("Hi"));
        let lp = choice.logprobs.as_ref().expect("logprobs present");
        assert_eq!(lp.content.len(), 1);
        assert!((lp.content[0].logprob - (-0.1)).abs() < 1e-5);
        assert_eq!(lp.content[0].top_logprobs[1].token, "Hey");
    }

    #[test]
    fn test_openai_choice_without_logprobs_is_none() {
        let json = r#"{"delta":{"content":"Hi"},"finish_reason":null}"#;
        let choice: OpenAIChoice = serde_json::from_str(json).expect("deser");
        assert!(choice.logprobs.is_none());
    }

    #[test]
    fn test_anthropic_request_with_system_serializes() {
        let req = AnthropicRequest {
            model: "claude-sonnet-4-20250514".to_string(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
            }],
            max_tokens: 1024,
            stream: true,
            temperature: 0.7,
            system: Some("You are a helpful assistant.".to_string()),
        };
        let json = serde_json::to_string(&req).expect("serialize");
        assert!(json.contains("\"system\":\"You are a helpful assistant.\""));
    }

    #[test]
    fn test_anthropic_request_without_system_omits_field() {
        let req = AnthropicRequest {
            model: "claude-sonnet-4-20250514".to_string(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
            }],
            max_tokens: 1024,
            stream: true,
            temperature: 0.7,
            system: None,
        };
        let json = serde_json::to_string(&req).expect("serialize");
        assert!(!json.contains("system"));
    }

    #[test]
    fn test_openai_top_logprob_clone() {
        let t = OpenAITopLogprob {
            token: "foo".to_string(),
            logprob: -1.0,
        };
        let t2 = t.clone();
        assert_eq!(t2.token, t.token);
        assert!((t2.logprob - t.logprob).abs() < 1e-6);
    }

    #[test]
    fn test_openai_logprob_content_no_top_logprobs() {
        let json = r#"{"token":"test","logprob":-0.8}"#;
        let lc: OpenAILogprobContent = serde_json::from_str(json).expect("deser");
        assert!(lc.top_logprobs.is_empty());
    }

    #[test]
    fn test_openai_top_logprob_serializes() {
        let t = OpenAITopLogprob {
            token: "bar".to_string(),
            logprob: -2.0,
        };
        let json = serde_json::to_string(&t).expect("serialize");
        assert!(json.contains("\"token\":\"bar\""));
        assert!(json.contains("\"logprob\":-2.0"));
    }

    // ---- ProviderPlugin api_url() tests ----

    #[test]
    fn test_openai_plugin_api_url_https() {
        assert!(OpenAiPlugin.api_url().starts_with("https://"));
    }

    #[test]
    fn test_anthropic_plugin_api_url_https() {
        assert!(AnthropicPlugin.api_url().starts_with("https://"));
    }

    #[test]
    fn test_openai_plugin_api_url_contains_openai() {
        assert!(OpenAiPlugin.api_url().contains("openai.com"));
    }

    #[test]
    fn test_anthropic_plugin_api_url_contains_anthropic() {
        assert!(AnthropicPlugin.api_url().contains("anthropic.com"));
    }

    #[test]
    fn test_openai_plugin_name_matches_display() {
        assert_eq!(OpenAiPlugin.name(), Provider::Openai.to_string());
    }

    #[test]
    fn test_anthropic_plugin_name_matches_display() {
        assert_eq!(AnthropicPlugin.name(), Provider::Anthropic.to_string());
    }

    #[test]
    fn test_openai_plugin_default_model_nonempty() {
        assert!(!OpenAiPlugin.default_model().is_empty());
    }

    #[test]
    fn test_anthropic_plugin_default_model_nonempty() {
        assert!(!AnthropicPlugin.default_model().is_empty());
    }

    // -- ANTHROPIC_API_VERSION constant tests (#14) --

    #[test]
    fn test_anthropic_api_version_nonempty() {
        assert!(!super::ANTHROPIC_API_VERSION.is_empty());
    }

    #[test]
    fn test_anthropic_api_version_format() {
        // Should be a date in YYYY-MM-DD format
        let v = super::ANTHROPIC_API_VERSION;
        assert_eq!(v.len(), 10, "version should be YYYY-MM-DD");
        assert!(v.chars().nth(4) == Some('-'), "4th char should be -");
        assert!(v.chars().nth(7) == Some('-'), "7th char should be -");
    }
}
