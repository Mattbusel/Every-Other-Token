use clap::ValueEnum;
use serde::{Deserialize, Serialize};

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

// -- OpenAI SSE types -------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct OpenAIChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct OpenAIChatRequest {
    pub model: String,
    pub messages: Vec<OpenAIChatMessage>,
    pub stream: bool,
    pub temperature: f32,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIDelta {
    pub content: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIChoice {
    pub delta: OpenAIDelta,
    #[allow(dead_code)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIChunk {
    pub choices: Vec<OpenAIChoice>,
}

// -- Anthropic SSE types ----------------------------------------------------

#[derive(Debug, Serialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct AnthropicRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: u32,
    pub stream: bool,
    pub temperature: f32,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicContentDelta {
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicStreamEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    #[serde(default)]
    pub delta: Option<AnthropicContentDelta>,
}

// -- Orchestrator MCP types -------------------------------------------------

#[derive(Debug, Serialize)]
pub struct McpInferRequest {
    pub jsonrpc: String,
    pub method: String,
    pub id: u64,
    pub params: McpInferParams,
}

#[derive(Debug, Serialize)]
pub struct McpInferParams {
    pub name: String,
    pub arguments: McpInferArguments,
}

#[derive(Debug, Serialize)]
pub struct McpInferArguments {
    pub prompt: String,
    pub worker: String,
}

#[derive(Debug, Deserialize)]
pub struct McpInferResponse {
    #[allow(dead_code)]
    pub jsonrpc: Option<String>,
    pub result: Option<McpInferResult>,
    pub error: Option<McpError>,
}

#[derive(Debug, Deserialize)]
pub struct McpInferResult {
    pub content: Vec<McpContent>,
}

#[derive(Debug, Deserialize)]
pub struct McpContent {
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct McpError {
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let text = resp.result.as_ref().and_then(|r| r.content.first()).and_then(|c| c.text.as_ref());
        assert_eq!(text, Some(&"enriched prompt".to_string()));
    }

    #[test]
    fn test_mcp_response_deserializes_error() {
        let json = r#"{"jsonrpc":"2.0","error":{"message":"pipeline down"}}"#;
        let resp: McpInferResponse = serde_json::from_str(json).expect("deser failed");
        assert!(resp.result.is_none());
        assert_eq!(resp.error.as_ref().map(|e| &e.message[..]), Some("pipeline down"));
    }

    #[test]
    fn test_anthropic_content_block_delta_deserializes() {
        let json = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;
        let event: AnthropicStreamEvent = serde_json::from_str(json).expect("deser failed");
        assert_eq!(event.event_type, "content_block_delta");
        assert_eq!(event.delta.as_ref().and_then(|d| d.text.as_ref()).map(|s| s.as_str()), Some("Hello"));
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
        assert_eq!(chunk.choices[0].delta.content.as_ref().map(|s| s.as_str()), Some("Hi"));
    }

    #[test]
    fn test_openai_chunk_empty_delta() {
        let json = r#"{"id":"chatcmpl-abc","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#;
        let chunk: OpenAIChunk = serde_json::from_str(json).expect("deser failed");
        assert!(chunk.choices[0].delta.content.is_none());
    }

    #[test]
    fn test_mcp_infer_request_contains_all_fields() {
        let req = McpInferRequest {
            jsonrpc: "2.0".to_string(), method: "tools/call".to_string(), id: 42,
            params: McpInferParams {
                name: "infer".to_string(),
                arguments: McpInferArguments { prompt: "test prompt".to_string(), worker: "llama_cpp".to_string() },
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
        assert!(resp.result.as_ref().expect("result").content[0].text.is_none());
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
}
