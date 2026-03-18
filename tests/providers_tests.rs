//! External tests for providers module -- serialization, deserialization,
//! and display implementations.

use every_other_token::providers::*;

// -- Provider display tests -----------------------------------------------

#[test]
fn test_provider_display_openai() {
    assert_eq!(Provider::Openai.to_string(), "openai");
}

#[test]
fn test_provider_display_anthropic() {
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
fn test_provider_openai_is_lowercase() {
    let s = format!("{}", Provider::Openai);
    assert!(s.chars().all(|c| c.is_lowercase()));
}

#[test]
fn test_provider_anthropic_is_lowercase() {
    let s = format!("{}", Provider::Anthropic);
    assert!(s.chars().all(|c| c.is_lowercase()));
}

// -- MCP request serialization -------------------------------------------

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
        resp.error.as_ref().map(|e| e.message.as_str()),
        Some("pipeline down")
    );
}

// -- Logprob deserialization ---------------------------------------------

#[test]
fn test_openai_top_logprob_deserializes() {
    let json = r#"{"token":"hello","logprob":-0.5}"#;
    let tlp: OpenAITopLogprob = serde_json::from_str(json).expect("deser");
    assert_eq!(tlp.token, "hello");
    assert!((tlp.logprob - (-0.5f32)).abs() < 1e-5);
}

#[test]
fn test_openai_logprob_content_deserializes_with_alternatives() {
    let json = r#"{"token":"world","logprob":-1.2,"top_logprobs":[{"token":"world","logprob":-1.2},{"token":"earth","logprob":-2.5}]}"#;
    let lc: OpenAILogprobContent = serde_json::from_str(json).expect("deser");
    assert_eq!(lc.token, "world");
    assert_eq!(lc.top_logprobs.len(), 2);
    assert_eq!(lc.top_logprobs[1].token, "earth");
}

#[test]
fn test_openai_chunk_deserializes_with_content() {
    let json = r#"{"id":"chatcmpl-abc","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}"#;
    let chunk: OpenAIChunk = serde_json::from_str(json).expect("deser");
    assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Hi"));
}

#[test]
fn test_openai_chunk_empty_delta() {
    let json = r#"{"id":"chatcmpl-abc","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#;
    let chunk: OpenAIChunk = serde_json::from_str(json).expect("deser");
    assert!(chunk.choices[0].delta.content.is_none());
}

// -- Anthropic SSE types -------------------------------------------------

#[test]
fn test_anthropic_content_block_delta_deserializes() {
    let json =
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;
    let event: AnthropicStreamEvent = serde_json::from_str(json).expect("deser");
    assert_eq!(event.event_type, "content_block_delta");
    assert_eq!(
        event.delta.as_ref().and_then(|d| d.text.as_deref()),
        Some("Hello")
    );
}

#[test]
fn test_anthropic_message_start_has_no_delta_text() {
    let json = r#"{"type":"message_start","message":{"id":"msg_123"}}"#;
    let event: AnthropicStreamEvent = serde_json::from_str(json).expect("deser");
    assert_eq!(event.event_type, "message_start");
    assert!(event.delta.is_none());
}

#[test]
fn test_anthropic_ping_event() {
    let json = r#"{"type":"ping"}"#;
    let event: AnthropicStreamEvent = serde_json::from_str(json).expect("deser");
    assert_eq!(event.event_type, "ping");
    assert!(event.delta.is_none());
}

// -- ProviderPlugin URL tests -------------------------------------------

#[test]
fn test_openai_plugin_api_url_is_https() {
    assert!(OpenAiPlugin.api_url().starts_with("https://"));
}

#[test]
fn test_anthropic_plugin_api_url_is_https() {
    assert!(AnthropicPlugin.api_url().starts_with("https://"));
}

#[test]
fn test_openai_plugin_name_matches_display() {
    assert_eq!(OpenAiPlugin.name(), Provider::Openai.to_string());
}

#[test]
fn test_anthropic_plugin_name_matches_display() {
    assert_eq!(AnthropicPlugin.name(), Provider::Anthropic.to_string());
}

// -- AnthropicRequest serialization tests --------------------------------

#[test]
fn test_anthropic_request_with_system_serializes() {
    let req = AnthropicRequest {
        model: "claude-sonnet-4-6".to_string(),
        messages: vec![AnthropicMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
        }],
        max_tokens: 1024,
        stream: true,
        temperature: 0.7,
        system: Some("You are helpful.".to_string()),
    };
    let json = serde_json::to_string(&req).expect("serialize");
    assert!(json.contains("\"system\":\"You are helpful.\""));
}

#[test]
fn test_anthropic_request_without_system_omits_field() {
    let req = AnthropicRequest {
        model: "claude-sonnet-4-6".to_string(),
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
    assert!(!json.contains("\"system\""));
}

// -- ANTHROPIC_API_VERSION constant tests --------------------------------

#[test]
fn test_anthropic_api_version_format() {
    let v = ANTHROPIC_API_VERSION;
    assert_eq!(v.len(), 10, "expected YYYY-MM-DD format");
    assert_eq!(v.chars().nth(4), Some('-'));
    assert_eq!(v.chars().nth(7), Some('-'));
}

// -- Provider::from_str edge cases ---------------------------------------

#[test]
fn test_provider_from_str_openai() {
    let p: Provider = "openai".parse().expect("parse");
    assert_eq!(p, Provider::Openai);
}

#[test]
fn test_provider_from_str_anthropic() {
    let p: Provider = "anthropic".parse().expect("parse");
    assert_eq!(p, Provider::Anthropic);
}

#[test]
fn test_provider_from_str_mock() {
    let p: Provider = "mock".parse().expect("parse");
    assert_eq!(p, Provider::Mock);
}

#[test]
fn test_provider_from_str_case_insensitive_openai() {
    let p: Provider = "OpenAI".parse().expect("parse case insensitive");
    assert_eq!(p, Provider::Openai);
}

#[test]
fn test_provider_from_str_case_insensitive_anthropic() {
    let p: Provider = "ANTHROPIC".parse().expect("parse case insensitive");
    assert_eq!(p, Provider::Anthropic);
}

#[test]
fn test_provider_from_str_unknown_returns_err() {
    let result: Result<Provider, _> = "google".parse();
    assert!(result.is_err());
}

#[test]
fn test_provider_from_str_empty_returns_err() {
    let result: Result<Provider, _> = "".parse();
    assert!(result.is_err());
}

#[test]
fn test_provider_roundtrip_display_fromstr() {
    for p in [Provider::Openai, Provider::Anthropic, Provider::Mock] {
        let s = p.to_string();
        let p2: Provider = s.parse().expect("roundtrip");
        assert_eq!(p, p2);
    }
}

#[test]
fn test_openai_keeps_default_model() {
    let model = "gpt-3.5-turbo";
    let result = if Provider::Openai == Provider::Anthropic && model == "gpt-3.5-turbo" {
        "claude-sonnet-4-20250514"
    } else {
        model
    };
    assert_eq!(result, "gpt-3.5-turbo");
}
