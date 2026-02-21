use clap::ValueEnum;
use serde::{Deserialize, Serialize};

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

// ---------------------------------------------------------------------------
// Anthropic SSE types
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Orchestrator MCP types
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// TokenEvent (shared between terminal and web modes)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct TokenEvent {
    pub text: String,
    pub original: String,
    pub index: usize,
    pub transformed: bool,
    pub importance: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Provider tests --

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
    fn test_provider_clone() {
        let p = Provider::Openai;
        let p2 = p.clone();
        assert_eq!(p, p2);
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

    // -- MCP request/response serialization --

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
    fn test_mcp_request_all_fields() {
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
        assert_eq!(parsed["params"]["name"], "infer");
        assert_eq!(parsed["params"]["arguments"]["prompt"], "test prompt");
        assert_eq!(parsed["params"]["arguments"]["worker"], "llama_cpp");
    }

    #[test]
    fn test_mcp_response_success() {
        let json = r#"{"jsonrpc":"2.0","result":{"content":[{"text":"enriched prompt"}]}}"#;
        let resp: McpInferResponse = serde_json::from_str(json).expect("deser failed");
        assert!(resp.error.is_none());
        let text = resp.result.as_ref()
            .and_then(|r| r.content.first())
            .and_then(|c| c.text.as_ref());
        assert_eq!(text, Some(&"enriched prompt".to_string()));
    }

    #[test]
    fn test_mcp_response_error() {
        let json = r#"{"jsonrpc":"2.0","error":{"message":"pipeline down"}}"#;
        let resp: McpInferResponse = serde_json::from_str(json).expect("deser failed");
        assert!(resp.result.is_none());
        assert_eq!(resp.error.as_ref().map(|e| &e.message[..]), Some("pipeline down"));
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
        let content = &resp.result.as_ref().expect("result").content[0];
        assert!(content.text.is_none());
    }

    // -- Anthropic SSE event parsing --

    #[test]
    fn test_anthropic_content_block_delta() {
        let json = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;
        let event: AnthropicStreamEvent = serde_json::from_str(json).expect("deser failed");
        assert_eq!(event.event_type, "content_block_delta");
        assert_eq!(event.delta.as_ref().and_then(|d| d.text.as_ref()).map(|s| s.as_str()), Some("Hello"));
    }

    #[test]
    fn test_anthropic_message_start() {
        let json = r#"{"type":"message_start","message":{"id":"msg_123"}}"#;
        let event: AnthropicStreamEvent = serde_json::from_str(json).expect("deser failed");
        assert_eq!(event.event_type, "message_start");
        assert!(event.delta.is_none());
    }

    #[test]
    fn test_anthropic_message_delta() {
        let json = r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#;
        let event: AnthropicStreamEvent = serde_json::from_str(json).expect("deser");
        assert_eq!(event.event_type, "message_delta");
        assert!(event.delta.as_ref().and_then(|d| d.text.as_ref()).is_none());
    }

    #[test]
    fn test_anthropic_ping() {
        let json = r#"{"type":"ping"}"#;
        let event: AnthropicStreamEvent = serde_json::from_str(json).expect("deser");
        assert_eq!(event.event_type, "ping");
        assert!(event.delta.is_none());
    }

    // -- OpenAI SSE chunk parsing --

    #[test]
    fn test_openai_chunk_deserializes() {
        let json = r#"{"id":"chatcmpl-abc","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}"#;
        let chunk: OpenAIChunk = serde_json::from_str(json).expect("deser failed");
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Hi"));
    }

    #[test]
    fn test_openai_chunk_empty_delta() {
        let json = r#"{"id":"chatcmpl-abc","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#;
        let chunk: OpenAIChunk = serde_json::from_str(json).expect("deser failed");
        assert!(chunk.choices[0].delta.content.is_none());
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

    // -- TokenEvent serialization --

    #[test]
    fn test_token_event_serializes() {
        let event = TokenEvent {
            text: "hello".to_string(),
            original: "hello".to_string(),
            index: 0,
            transformed: false,
            importance: 0.5,
        };
        let json = serde_json::to_string(&event).expect("serialize failed");
        assert!(json.contains("\"text\":\"hello\""));
        assert!(json.contains("\"original\":\"hello\""));
        assert!(json.contains("\"transformed\":false"));
        assert!(json.contains("\"importance\":0.5"));
    }

    #[test]
    fn test_token_event_export_structure() {
        let event = TokenEvent {
            text: "dlrow".to_string(),
            original: "world".to_string(),
            index: 1,
            transformed: true,
            importance: 0.75,
        };
        let json = serde_json::to_string(&event).expect("serialize");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("parse");
        assert_eq!(parsed["text"], "dlrow");
        assert_eq!(parsed["original"], "world");
        assert_eq!(parsed["index"], 1);
        assert_eq!(parsed["transformed"], true);
        assert!(parsed["importance"].as_f64().is_some());
    }

    #[test]
    fn test_token_event_roundtrip() {
        let event = TokenEvent {
            text: "dlrow".to_string(),
            original: "world".to_string(),
            index: 5,
            transformed: true,
            importance: 0.87,
        };
        let json = serde_json::to_string(&event).expect("serialize");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("parse");
        assert_eq!(parsed["text"], "dlrow");
        assert_eq!(parsed["index"], 5);
        let imp = parsed["importance"].as_f64().expect("float");
        assert!((imp - 0.87).abs() < 0.001);
    }

    #[test]
    fn test_token_event_zero_importance() {
        let event = TokenEvent {
            text: ".".to_string(),
            original: ".".to_string(),
            index: 0,
            transformed: false,
            importance: 0.0,
        };
        let json = serde_json::to_string(&event).expect("serialize");
        assert!(json.contains("\"importance\":0.0"));
    }

    #[test]
    fn test_token_event_max_importance() {
        let event = TokenEvent {
            text: "critical".to_string(),
            original: "critical".to_string(),
            index: 0,
            transformed: false,
            importance: 1.0,
        };
        let json = serde_json::to_string(&event).expect("serialize");
        assert!(json.contains("\"importance\":1.0"));
    }

    #[test]
    fn test_token_event_clone() {
        let event = TokenEvent {
            text: "test".to_string(),
            original: "test".to_string(),
            index: 0,
            transformed: false,
            importance: 0.5,
        };
        let cloned = event.clone();
        assert_eq!(cloned.text, event.text);
        assert_eq!(cloned.original, event.original);
        assert_eq!(cloned.index, event.index);
    }

    #[test]
    fn test_multiple_token_events_serialize_as_array() {
        let events = vec![
            TokenEvent { text: "hello".to_string(), original: "hello".to_string(), index: 0, transformed: false, importance: 0.5 },
            TokenEvent { text: "dlrow".to_string(), original: "world".to_string(), index: 1, transformed: true, importance: 0.7 },
        ];
        let json = serde_json::to_string(&events).expect("serialize");
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).expect("parse");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0]["index"], 0);
        assert_eq!(parsed[1]["index"], 1);
    }
}
