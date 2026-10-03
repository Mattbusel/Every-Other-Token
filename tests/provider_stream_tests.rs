//! End-to-end tests for the provider streaming paths.
//!
//! Each test starts a tiny HTTP server on 127.0.0.1 that plays back a canned
//! SSE response, points a `TokenInterceptor` at it with `--base-url`, and
//! checks both what was sent (path, headers, JSON body) and what came out
//! (the token events). No network access and no API keys are needed.

use every_other_token::providers::Provider;
use every_other_token::transforms::Transform;
use every_other_token::{TokenEvent, TokenInterceptor};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::mpsc;

/// One canned HTTP response: a status line plus headers, and a body sent as
/// separate writes so the client sees it split at exactly these points.
struct Canned {
    head: String,
    body_parts: Vec<Vec<u8>>,
}

fn sse_ok(parts: Vec<Vec<u8>>) -> Canned {
    Canned {
        head: "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"
            .to_string(),
        body_parts: parts,
    }
}

fn status(code: u16, reason: &str, extra_headers: &str, body: &str) -> Canned {
    Canned {
        head: format!(
            "HTTP/1.1 {code} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n{extra_headers}\r\n",
            body.len()
        ),
        body_parts: vec![body.as_bytes().to_vec()],
    }
}

/// Serve `responses` in order, one per connection. Returns the base URL
/// (`http://127.0.0.1:PORT/v1`) and the raw requests received.
async fn serve(responses: Vec<Canned>) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen_srv = Arc::clone(&seen);
    tokio::spawn(async move {
        for canned in responses {
            let (mut sock, _) = match listener.accept().await {
                Ok(c) => c,
                Err(_) => return,
            };
            let req = read_request(&mut sock).await;
            seen_srv.lock().unwrap().push(req);
            let _ = sock.write_all(canned.head.as_bytes()).await;
            for part in canned.body_parts {
                let _ = sock.write_all(&part).await;
                let _ = sock.flush().await;
                // Give the client time to read each part on its own.
                tokio::time::sleep(Duration::from_millis(15)).await;
            }
            let _ = sock.shutdown().await;
        }
    });
    (format!("http://127.0.0.1:{port}/v1"), seen)
}

/// Read one HTTP request (headers plus a Content-Length body).
async fn read_request(sock: &mut tokio::net::TcpStream) -> String {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        let n = sock.read(&mut tmp).await.unwrap_or(0);
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
        let text = String::from_utf8_lossy(&buf).to_string();
        if let Some(end) = text.find("\r\n\r\n") {
            let len = text[..end]
                .lines()
                .find_map(|l| {
                    let l = l.to_ascii_lowercase();
                    l.strip_prefix("content-length:")
                        .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                })
                .unwrap_or(0);
            if buf.len() >= end + 4 + len {
                break;
            }
        }
    }
    String::from_utf8_lossy(&buf).to_string()
}

fn body_json(req: &str) -> serde_json::Value {
    let body = req.split("\r\n\r\n").nth(1).unwrap_or("");
    serde_json::from_str(body).expect("request body is JSON")
}

fn header<'a>(req: &'a str, name: &str) -> Option<&'a str> {
    let name = name.to_ascii_lowercase();
    req.split("\r\n\r\n").next()?.lines().find_map(|l| {
        let (k, v) = l.split_once(':')?;
        (k.trim().to_ascii_lowercase() == name).then(|| v.trim())
    })
}

fn interceptor(provider: Provider, model: &str, base: &str) -> (TokenInterceptor, mpsc::UnboundedReceiver<TokenEvent>) {
    let (tx, rx) = mpsc::unbounded_channel();
    let i = TokenInterceptor::new_with_base_url(
        provider,
        Transform::Uppercase,
        model.to_string(),
        false,
        false,
        false,
        Some(base.to_string()),
    )
    .expect("construct")
    .with_web_tx(tx)
    .with_max_retries(3);
    (i, rx)
}

fn drain(rx: &mut mpsc::UnboundedReceiver<TokenEvent>) -> Vec<TokenEvent> {
    let mut out = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        out.push(ev);
    }
    out
}

fn joined_original(events: &[TokenEvent]) -> String {
    events.iter().map(|e| e.original.as_str()).collect()
}

fn openai_chunk(content: &str, logprob: Option<f32>) -> String {
    let lp = match logprob {
        Some(l) => format!(
            r#","logprobs":{{"content":[{{"token":{c},"logprob":{l},"top_logprobs":[{{"token":{c},"logprob":{l}}},{{"token":"alt","logprob":-3.0}}]}}]}}"#,
            c = serde_json::to_string(content).unwrap()
        ),
        None => String::new(),
    };
    format!(
        r#"{{"id":"x","choices":[{{"index":0,"delta":{{"content":{}}},"finish_reason":null{lp}}}]}}"#,
        serde_json::to_string(content).unwrap()
    )
}

// ---------------------------------------------------------------------------
// OpenAI wire format
// ---------------------------------------------------------------------------

#[tokio::test]
async fn openai_stream_parses_tricky_sse_framing() {
    // "café" has a two-byte "é"; split the bytes so the first network chunk
    // ends halfway through the character. CRLF line endings, a comment line
    // and "data:" without a space are all legal SSE.
    let first = format!("data: {}\r\n\r\n", openai_chunk("Hello", Some(-0.1)));
    let second = format!(": keep-alive comment\r\n\r\ndata:{}\r\n\r\n", openai_chunk(" café", Some(-0.7)));
    let second = second.into_bytes();
    let cut = second.iter().position(|&b| b == 0xC3).expect("é lead byte") + 1;
    let (a, b) = second.split_at(cut);
    let done = b"data: [DONE]\r\n\r\n".to_vec();
    let (base, seen) = serve(vec![sse_ok(vec![first.into_bytes(), a.to_vec(), b.to_vec(), done])]).await;

    let (i, mut rx) = interceptor(Provider::Openai, "gpt-4o-mini", &base);
    let mut i = i.with_api_key("sk-test-key").with_top_logprobs(3);
    i.intercept_stream("say hello").await.expect("stream ok");
    let events = drain(&mut rx);

    assert_eq!(joined_original(&events), "Hello café");
    assert!(events.iter().any(|e| e.original.contains("café")), "é must survive the chunk split");
    let first_conf = events[0].confidence.expect("logprob confidence");
    assert!((first_conf - (-0.1f32).exp()).abs() < 1e-4);
    assert!(events.iter().any(|e| e.alternatives.iter().any(|a| a.token == "alt")));

    let req = seen.lock().unwrap()[0].clone();
    assert!(req.starts_with("POST /v1/chat/completions "), "{req}");
    assert_eq!(header(&req, "authorization"), Some("Bearer sk-test-key"));
    let body = body_json(&req);
    assert_eq!(body["model"], "gpt-4o-mini");
    assert_eq!(body["stream"], true);
    assert_eq!(body["logprobs"], true);
    assert_eq!(body["top_logprobs"], 3);
    assert_eq!(body["messages"][0]["content"], "say hello");
}

#[tokio::test]
async fn openai_stream_with_system_prompt_sends_system_message() {
    let parts = vec![
        format!("data: {}\n\n", openai_chunk("ok", Some(-0.2))).into_bytes(),
        b"data: [DONE]\n\n".to_vec(),
    ];
    let (base, seen) = serve(vec![sse_ok(parts)]).await;
    let (i, mut rx) = interceptor(Provider::Openai, "gpt-4o", &base);
    let mut i = i.with_api_key("sk-x").with_system_prompt("Be brief.");
    i.intercept_stream("hi").await.expect("stream ok");
    assert_eq!(joined_original(&drain(&mut rx)), "ok");
    let body = body_json(&seen.lock().unwrap()[0]);
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["messages"][0]["content"], "Be brief.");
    assert_eq!(body["messages"][1]["role"], "user");
}

#[tokio::test]
async fn ollama_needs_no_key_and_falls_back_to_timing_confidence() {
    let parts = vec![
        format!("data: {}\n\n", openai_chunk("one", None)).into_bytes(),
        format!("data: {}\n\n", openai_chunk(" two", None)).into_bytes(),
        format!("data: {}\n\n", openai_chunk(" three", None)).into_bytes(),
        b"data: [DONE]\n\n".to_vec(),
    ];
    let (base, seen) = serve(vec![sse_ok(parts)]).await;
    let (mut i, mut rx) = interceptor(Provider::Ollama, "llama3.2", &base);
    i.intercept_stream("count").await.expect("stream ok");
    let events = drain(&mut rx);
    assert_eq!(joined_original(&events), "one two three");
    // First token has no previous gap to time; later ones get the estimate.
    assert!(events.last().unwrap().confidence.is_some());

    let req = seen.lock().unwrap()[0].clone();
    assert_eq!(header(&req, "authorization"), None, "Ollama gets no auth header");
    assert_eq!(body_json(&req)["model"], "llama3.2");
}

#[tokio::test]
async fn gemini_request_leaves_out_logprob_fields() {
    let parts = vec![
        format!("data: {}\n\n", openai_chunk("hi", None)).into_bytes(),
        b"data: [DONE]\n\n".to_vec(),
    ];
    let (base, seen) = serve(vec![sse_ok(parts)]).await;
    let (i, mut rx) = interceptor(Provider::Gemini, "gemini-2.5-flash", &base);
    let mut i = i.with_api_key("g-key");
    i.intercept_stream("hello").await.expect("stream ok");
    assert_eq!(joined_original(&drain(&mut rx)), "hi");
    let body = body_json(&seen.lock().unwrap()[0]);
    assert!(body.get("logprobs").is_none(), "{body}");
    assert!(body.get("top_logprobs").is_none(), "{body}");
}

#[tokio::test]
async fn openrouter_sends_attribution_headers() {
    let parts = vec![
        b": OPENROUTER PROCESSING\n\n".to_vec(),
        format!("data: {}\n\n", openai_chunk("routed", Some(-0.3))).into_bytes(),
        b"data: [DONE]\n\n".to_vec(),
    ];
    let (base, seen) = serve(vec![sse_ok(parts)]).await;
    let (i, mut rx) = interceptor(Provider::Openrouter, "openai/gpt-4o-mini", &base);
    let mut i = i.with_api_key("or-key");
    i.intercept_stream("hello").await.expect("stream ok");
    assert_eq!(joined_original(&drain(&mut rx)), "routed");
    let req = seen.lock().unwrap()[0].clone();
    assert_eq!(header(&req, "x-title"), Some("every-other-token"));
    assert_eq!(header(&req, "authorization"), Some("Bearer or-key"));
}

#[tokio::test]
async fn openai_mid_stream_error_is_reported() {
    let parts = vec![
        format!("data: {}\n\n", openai_chunk("partial", Some(-0.1))).into_bytes(),
        br#"data: {"error":{"message":"upstream overloaded","code":502}}"#.to_vec(),
        b"\n\n".to_vec(),
    ];
    let (base, _) = serve(vec![sse_ok(parts)]).await;
    let (i, _rx) = interceptor(Provider::Openai, "gpt-4o", &base);
    let mut i = i.with_api_key("sk-x");
    let err = i.intercept_stream("hi").await.expect_err("must fail");
    assert!(err.to_string().contains("upstream overloaded"), "{err}");
}

#[tokio::test]
async fn openai_http_error_includes_status_and_body() {
    let (base, _) = serve(vec![status(
        401,
        "Unauthorized",
        "",
        r#"{"error":{"message":"bad key"}}"#,
    )])
    .await;
    let (i, _rx) = interceptor(Provider::Openai, "gpt-4o", &base);
    let mut i = i.with_api_key("sk-wrong");
    let err = i.intercept_stream("hi").await.expect_err("must fail").to_string();
    assert!(err.contains("401") && err.contains("bad key"), "{err}");
}

// ---------------------------------------------------------------------------
// Anthropic wire format
// ---------------------------------------------------------------------------

fn anthropic_event(name: &str, data: &str) -> Vec<u8> {
    format!("event: {name}\ndata: {data}\n\n").into_bytes()
}

#[tokio::test]
async fn anthropic_stream_reads_text_deltas() {
    let parts = vec![
        anthropic_event("message_start", r#"{"type":"message_start","message":{"id":"m"}}"#),
        anthropic_event("ping", r#"{"type":"ping"}"#),
        anthropic_event(
            "content_block_delta",
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Bonjour"}}"#,
        ),
        anthropic_event(
            "content_block_delta",
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" à tous"}}"#,
        ),
        anthropic_event("message_stop", r#"{"type":"message_stop"}"#),
    ];
    let (base, seen) = serve(vec![sse_ok(parts)]).await;
    let (i, mut rx) = interceptor(Provider::Anthropic, "claude-sonnet-4-6", &base);
    let mut i = i.with_api_key("sk-ant-test").with_system_prompt("Speak French.");
    i.intercept_stream("greet").await.expect("stream ok");
    assert_eq!(joined_original(&drain(&mut rx)), "Bonjour à tous");

    let req = seen.lock().unwrap()[0].clone();
    assert!(req.starts_with("POST /v1/messages "), "{req}");
    assert_eq!(header(&req, "x-api-key"), Some("sk-ant-test"));
    assert!(header(&req, "anthropic-version").is_some());
    let body = body_json(&req);
    assert_eq!(body["system"], "Speak French.");
    assert_eq!(body["stream"], true);
}

#[tokio::test]
async fn anthropic_error_event_is_reported() {
    let parts = vec![anthropic_event(
        "error",
        r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
    )];
    let (base, _) = serve(vec![sse_ok(parts)]).await;
    let (i, _rx) = interceptor(Provider::Anthropic, "claude-sonnet-4-6", &base);
    let mut i = i.with_api_key("sk-ant-test");
    let err = i.intercept_stream("hi").await.expect_err("must fail");
    assert!(err.to_string().contains("Overloaded"), "{err}");
}

// ---------------------------------------------------------------------------
// Endpoint resolution
// ---------------------------------------------------------------------------

#[test]
fn endpoint_urls_follow_provider_and_base_url() {
    assert_eq!(
        Provider::Openai.endpoint_url(None),
        "https://api.openai.com/v1/chat/completions"
    );
    assert_eq!(
        Provider::Anthropic.endpoint_url(None),
        "https://api.anthropic.com/v1/messages"
    );
    assert_eq!(
        Provider::Ollama.endpoint_url(None),
        "http://localhost:11434/v1/chat/completions"
    );
    assert_eq!(
        Provider::Openai.endpoint_url(Some("http://localhost:8080/v1/")),
        "http://localhost:8080/v1/chat/completions"
    );
}

#[test]
fn base_url_makes_api_key_optional() {
    // With a custom base URL the key is optional, so construction succeeds
    // whether or not OPENAI_API_KEY happens to be set in this environment.
    let i = TokenInterceptor::new_with_base_url(
        Provider::Openai,
        Transform::Reverse,
        "local-model".into(),
        false,
        false,
        false,
        Some("http://localhost:8080/v1".into()),
    )
    .expect("no key needed with a base URL");
    assert_eq!(i.endpoint_url(), "http://localhost:8080/v1/chat/completions");
}

#[test]
fn every_provider_round_trips_through_from_str() {
    for p in Provider::ALL {
        let parsed: Provider = p.to_string().parse().expect("parse");
        assert_eq!(parsed, p);
        assert!(!p.default_model().is_empty());
    }
    assert!("bogus".parse::<Provider>().is_err());
}

// ---------------------------------------------------------------------------
// Retry policy (backon)
// ---------------------------------------------------------------------------

fn one_token_stream() -> Canned {
    sse_ok(vec![
        format!("data: {}\n\n", openai_chunk("recovered", Some(-0.2))).into_bytes(),
        b"data: [DONE]\n\n".to_vec(),
    ])
}

#[tokio::test]
async fn retries_a_503_then_succeeds() {
    // No Retry-After header, so this waits out the normal back-off.
    let (base, seen) = serve(vec![
        status(503, "Service Unavailable", "", r#"{"error":"busy"}"#),
        one_token_stream(),
    ])
    .await;
    let (i, mut rx) = interceptor(Provider::Openai, "gpt-4o", &base);
    let mut i = i.with_api_key("sk-x");
    i.intercept_stream("hi").await.expect("second attempt succeeds");
    assert_eq!(joined_original(&drain(&mut rx)), "recovered");
    assert_eq!(seen.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn honours_retry_after_on_429() {
    let (base, seen) = serve(vec![
        status(429, "Too Many Requests", "Retry-After: 0\r\n", r#"{"error":"slow down"}"#),
        status(429, "Too Many Requests", "retry-after-ms: 10\r\n", r#"{"error":"slow down"}"#),
        one_token_stream(),
    ])
    .await;
    let (i, mut rx) = interceptor(Provider::Openai, "gpt-4o", &base);
    let mut i = i.with_api_key("sk-x");
    let started = std::time::Instant::now();
    i.intercept_stream("hi").await.expect("third attempt succeeds");
    // Retry-After 0 s and 10 ms replace the 800 ms+ exponential delays.
    assert!(started.elapsed() < Duration::from_millis(700), "{:?}", started.elapsed());
    assert_eq!(joined_original(&drain(&mut rx)), "recovered");
    assert_eq!(seen.lock().unwrap().len(), 3);
}

#[tokio::test]
async fn gives_up_after_max_retries_and_reports_last_status() {
    let (base, seen) = serve(vec![
        status(502, "Bad Gateway", "Retry-After: 0\r\n", r#"{"error":"down"}"#),
        status(502, "Bad Gateway", "Retry-After: 0\r\n", r#"{"error":"still down"}"#),
        one_token_stream(), // never reached
    ])
    .await;
    let (i, _rx) = interceptor(Provider::Openai, "gpt-4o", &base);
    let mut i = i.with_api_key("sk-x").with_max_retries(2);
    let err = i.intercept_stream("hi").await.expect_err("must fail").to_string();
    assert!(err.contains("502") && err.contains("still down"), "{err}");
    assert_eq!(seen.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn does_not_retry_a_400() {
    let (base, seen) = serve(vec![
        status(400, "Bad Request", "", r#"{"error":{"message":"bad model"}}"#),
        one_token_stream(),
    ])
    .await;
    let (i, _rx) = interceptor(Provider::Openai, "gpt-4o", &base);
    let mut i = i.with_api_key("sk-x");
    let err = i.intercept_stream("hi").await.expect_err("must fail").to_string();
    assert!(err.contains("bad model"), "{err}");
    assert_eq!(seen.lock().unwrap().len(), 1);
}
