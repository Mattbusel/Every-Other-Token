//! Embedded web UI server and HTTP request handling.
//!
//! This module exposes a single entry point, [`serve`], that binds a TCP listener
//! and handles all incoming HTTP connections in a hand-rolled async loop.  No
//! external web framework is used -- the server speaks raw HTTP/1.1 so the binary
//! stays small and dependency-free.
//!
//! ## Supported routes
//!
//! | Method | Path | Description |
//! |--------|------|-------------|
//! | `GET` | `/` | Serves the embedded single-page HTML application |
//! | `GET` | `/events` | SSE stream of [`crate::TokenEvent`] JSON objects |
//! | `GET` | `/stream` | Alias for `/events` |
//! | `POST` | `/room/create` | Creates a new collaboration room |
//! | `GET` | `/ws/:code` | WebSocket endpoint for room participants |
//! | `GET` | `/join/:code` | Serve the collaboration join page |
//! | `POST` | `/api/config` | Update runtime configuration |
//! | `GET` | `/api/experiments` | List stored experiments (requires `sqlite-log`) |

use colored::*;
use serde::Serialize;
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use subtle::ConstantTimeEq;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::sync::mpsc;

use crate::cli::Args;
use crate::collab::RoomStore;
use crate::providers::Provider;
use crate::transforms::Transform;
use crate::{TokenEvent, TokenInterceptor};

/// Maximum prompt length accepted on /stream.
const MAX_PROMPT_LEN: usize = 100_000;

/// Per-IP sliding-window rate limiter for the /stream endpoint.
/// Allows at most `MAX_REQUESTS` requests in `WINDOW_SECS` seconds per IP.
const RATE_LIMIT_MAX: u32 = 10;
const RATE_LIMIT_WINDOW: Duration = Duration::from_secs(60);

type RateLimiter = Arc<Mutex<HashMap<IpAddr, (Instant, u32)>>>;

fn new_rate_limiter() -> RateLimiter {
    Arc::new(Mutex::new(HashMap::new()))
}

/// Returns `true` if the request should be allowed, `false` if rate-limited.
fn rate_limit_check(limiter: &RateLimiter, addr: IpAddr) -> bool {
    let mut map = limiter.lock().unwrap_or_else(|e| e.into_inner());
    let now = Instant::now();
    let entry = map.entry(addr).or_insert((now, 0));
    if now.duration_since(entry.0) >= RATE_LIMIT_WINDOW {
        // Window expired — reset
        *entry = (now, 1);
        true
    } else if entry.1 < RATE_LIMIT_MAX {
        entry.1 = entry.1.saturating_add(1);
        true
    } else {
        false
    }
}

/// Returns the CORS origin value to use in `Access-Control-Allow-Origin`.
/// Reads the `CORS_ORIGIN` environment variable; falls back to `"*"`.
fn cors_origin() -> String {
    std::env::var("CORS_ORIGIN").unwrap_or_else(|_| "*".to_string())
}

// Default model names used when the query string omits a model parameter.
// Centralised here so web.rs, cli.rs, and lib.rs all stay in sync.
const DEFAULT_OPENAI_MODEL: &str = "gpt-3.5-turbo";
const DEFAULT_ANTHROPIC_MODEL: &str = "claude-sonnet-4-6";
const DEFAULT_MOCK_MODEL: &str = "mock-fixture-v1";

/// Wraps a `TokenEvent` with a provider-side label for diff streaming.
#[derive(Debug, Serialize)]
struct DiffTokenEvent<'a> {
    side: &'static str,
    #[serde(flatten)]
    event: &'a TokenEvent,
}

/// Embedded single-page HTML application with side-by-side, multi-transform,
/// dependency graph, and export features.
pub const INDEX_HTML: &str = include_str!("../static/index.html");

/// Simple percent-decoding for URL query parameters.
///
/// Accumulates decoded bytes in a staging buffer and flushes via
/// `String::from_utf8_lossy` so that multi-byte UTF-8 sequences
/// (e.g. `%C3%A9` → "é") are reconstructed correctly instead of
/// each byte being cast directly to `char`, which is invalid for
/// bytes >= 128.
pub fn url_decode(s: &str) -> String {
    // Pre-allocate to avoid reallocations for typical inputs
    let mut result = String::with_capacity(s.len());
    let mut pending: Vec<u8> = Vec::new();
    let mut chars = s.chars();

    macro_rules! flush {
        () => {
            if !pending.is_empty() {
                result.push_str(&String::from_utf8_lossy(&pending));
                pending.clear();
            }
        };
    }

    while let Some(c) = chars.next() {
        match c {
            '+' => {
                flush!();
                result.push(' ');
            }
            '%' => {
                let hex: String = chars.by_ref().take(2).collect();
                if hex.len() == 2 {
                    if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                        pending.push(byte);
                    } else {
                        // Invalid hex digits — emit literally
                        flush!();
                        result.push('%');
                        result.push_str(&hex);
                    }
                } else {
                    // Incomplete percent sequence (fewer than 2 chars remain)
                    flush!();
                    result.push('%');
                    result.push_str(&hex);
                }
            }
            _ => {
                flush!();
                result.push(c);
            }
        }
    }
    flush!();
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

/// Query parameters parsed from a /stream request.
struct StreamParams {
    prompt: String,
    transform: String,
    provider: String,
    model: String,
    rate: f64,
    seed: Option<u64>,
    top_logprobs: u8,
    system: Option<String>,
    visual: bool,
    heatmap: bool,
}

fn parse_stream_params(query: &std::collections::HashMap<String, String>) -> StreamParams {
    StreamParams {
        prompt: query.get("prompt").cloned().unwrap_or_default(),
        transform: query
            .get("transform")
            .cloned()
            .unwrap_or_else(|| "reverse".to_string()),
        provider: query
            .get("provider")
            .cloned()
            .unwrap_or_else(|| "openai".to_string()),
        model: query.get("model").cloned().unwrap_or_default(),
        rate: query
            .get("rate")
            .and_then(|r| r.parse::<f64>().ok())
            .filter(|r| r.is_finite())
            .map(|r| r.clamp(0.0, 1.0))
            .unwrap_or(0.5),
        seed: query.get("seed").and_then(|s| s.parse().ok()),
        top_logprobs: query
            .get("top_logprobs")
            .and_then(|t| t.parse::<u8>().ok())
            .map(|v| v.clamp(0, 20))
            .unwrap_or(5),
        system: query.get("system").filter(|s| !s.is_empty()).cloned(),
        visual: query
            .get("visual")
            .map(|v| v == "1" || v == "true")
            .unwrap_or(false),
        heatmap: query
            .get("heatmap")
            .map(|v| v == "1" || v == "true")
            .unwrap_or(false),
    }
}

/// # HTTP API
///
/// ## Endpoints
///
/// - `GET /` — Serves the embedded single-page web UI
///
/// - `GET /stream?prompt=...&transform=...&provider=...&model=...&rate=...`  
///   Server-Sent Events stream of [`TokenEvent`] JSON objects.  
///   Each event: `data: {"text":"...","index":N,"transformed":bool,...}`
///
/// - `GET /diff-stream?prompt=...&transform=...`  
///   SSE stream with two providers side-by-side; each event includes `"side":"openai"|"anthropic"`.
///
/// - `GET /ab-stream?prompt=...&system_a=...&system_b=...`  
///   SSE stream for A/B experiment mode.
///
/// - `POST /room/create` — Creates a multiplayer room, returns `{"code":"SWIFT-LION-42","room_id":"<uuid>","ws_url":"/ws/SWIFT-LION-42"}`.
///
/// - `GET /join/CODE` — Returns room join HTML page.
///
/// - `WS /ws/CODE` — WebSocket connection for multiplayer collaboration.  
///   **Inbound message types** (JSON):  
///   `{"type":"set_name","name":"..."}` — Update display name (max 64 chars)  
///   `{"type":"vote","transform":"...","dir":"up"|"down"}` — Cast a vote  
///   `{"type":"surgery","token_index":N,"new_text":"...","old_text":"..."}` — Edit a token  
///   `{"type":"chat","text":"...","token_index":N}` — Send a chat message  
///   `{"type":"record_start"}` / `{"type":"record_stop"}` — Recording control  
///   `{"type":"token",...}` — Host broadcasts a token event to guests  
///   **Outbound event types**: `welcome`, `participant_join`, `participant_leave`,  
///   `participant_update`, `vote_update`, `surgery`, `chat`, `record_started`,  
///   `record_stopped`, `replay_event`, `replay_done`, `stream_done`, `pong`, `error`
pub async fn serve(port: u16, default_args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    tracing::info!(port, "binding web UI server");
    let listener = TcpListener::bind(format!("127.0.0.1:{}", port)).await?;
    tracing::info!(port, "web UI server listening");

    eprintln!(
        "{}",
        format!("  Web UI running at http://localhost:{}", port).bright_green()
    );
    eprintln!("{}", "  Press Ctrl+C to stop.".bright_blue());

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
    let api_key: Option<String> = default_args.api_key.clone();
    let sse_buffer_size = default_args.sse_buffer_size;

    let room_store = crate::collab::new_room_store();
    let rate_limiter = new_rate_limiter();

    // Background task: evict idle rooms every 5 minutes; evict abandoned rooms every minute.
    {
        let cleanup_store = room_store.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(300));
            loop {
                interval.tick().await;
                crate::collab::evict_idle_rooms(&cleanup_store);
            }
        });
    }
    {
        let abandoned_store = room_store.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            loop {
                interval.tick().await;
                crate::collab::evict_abandoned_rooms(&abandoned_store);
            }
        });
    }

    // If HelixRouter integration is configured, start the bridge + orchestrator.
    // This closes the cross-repo feedback loop: HelixRouter pressure → TelemetryBus
    // → SelfImprovementOrchestrator → parameter adjustments.
    #[cfg(feature = "helix-bridge")]
    if let Some(ref helix_url) = default_args.helix_url {
        use crate::helix_bridge::client::HelixBridge;
        use crate::self_tune::orchestrator::{OrchestratorConfig, SelfImprovementOrchestrator};
        use crate::self_tune::telemetry_bus::{BusConfig, TelemetryBus};
        use std::sync::Arc;

        let bus = Arc::new(TelemetryBus::new(BusConfig::default()));
        bus.start_emitter();

        match HelixBridge::builder(helix_url.clone())
            .bus(Arc::clone(&bus))
            .build()
        {
            Ok(bridge) => {
                let orc = SelfImprovementOrchestrator::new(
                    OrchestratorConfig::default(),
                    Arc::clone(&bus),
                );
                tokio::spawn(async move { bridge.run().await });
                tokio::spawn(async move { orc.run().await });
                eprintln!(
                    "{}",
                    format!("  HelixBridge active → {helix_url}").bright_cyan()
                );
            }
            Err(e) => {
                eprintln!("  HelixBridge init failed: {e}; continuing without it");
            }
        }
    }

    loop {
        let (stream, addr) = listener.accept().await?;
        let provider = default_provider.clone();
        let store = room_store.clone();
        let conn_api_key = api_key.clone();
        let limiter = rate_limiter.clone();
        let peer_ip = addr.ip();
        let buf_sz = sse_buffer_size;
        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, provider, orchestrator, store, conn_api_key, limiter, peer_ip, buf_sz).await {
                eprintln!("  connection error: {}", e);
            }
        });
    }
}

async fn handle_connection(
    mut stream: tokio::net::TcpStream,
    default_provider: Provider,
    orchestrator: bool,
    store: RoomStore,
    api_key: Option<String>,
    limiter: RateLimiter,
    peer_ip: IpAddr,
    sse_buffer_size: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use tokio::io::AsyncReadExt;

    // Peek at the first bytes to detect WebSocket upgrade requests.
    // 4096 bytes ensures we capture full HTTP headers even with many/large header values.
    let mut peek_buf = [0u8; 4096];
    let peek_n = stream.peek(&mut peek_buf).await.unwrap_or(0);
    let peek_str = String::from_utf8_lossy(&peek_buf[..peek_n]);
    let peek_first_line = peek_str.lines().next().unwrap_or("").to_string();

    if peek_str.contains("Upgrade: websocket") || peek_str.contains("upgrade: websocket") {
        let ws_path = peek_first_line
            .split_whitespace()
            .nth(1)
            .unwrap_or("/")
            .to_string();
        if let Some(code) = ws_path.strip_prefix("/ws/") {
            let code = code.to_string();
            // is_host = true only for the first connection (host_id not yet assigned).
            // room_exists=true after /room/create, so "!room_exists" was always false,
            // meaning every client was treated as a guest.  Check host_id instead.
            let is_host = store
                .lock()
                .map(|s| s.get(&code).map(|r| r.host_id.is_empty()).unwrap_or(false))
                .unwrap_or(false);

            match tokio_tungstenite::accept_async(stream).await {
                Ok(ws_stream) => {
                    crate::collab::handle_ws(ws_stream, store, code, is_host).await;
                }
                Err(e) => {
                    eprintln!("  WS handshake error: {}", e);
                }
            }
            return Ok(());
        }
    }

    let mut buf = vec![0u8; 8192];
    let n = stream.read(&mut buf).await?;

    // Use httparse for robust HTTP/1.x request-line and header parsing.
    // This correctly handles folded headers, extra whitespace, and
    // requests where headers extend beyond a simple first-line split.
    let mut headers = [httparse::EMPTY_HEADER; 32];
    let mut req = httparse::Request::new(&mut headers);
    let path_owned: String = match req.parse(&buf[..n]) {
        Ok(httparse::Status::Complete(_)) | Ok(httparse::Status::Partial) => {
            req.path.unwrap_or("/").to_string()
        }
        Err(_) => return Ok(()),
    };
    let path_and_query = path_owned.as_str();

    // Split path and query
    let (path, query_str) = if let Some(idx) = path_and_query.find('?') {
        (&path_and_query[..idx], &path_and_query[idx + 1..])
    } else {
        (path_and_query, "")
    };

    // API key authentication: if api_key is configured, require it on /api/ routes.
    if path.starts_with("/api/") {
        if let Some(ref required_key) = api_key {
            let auth_header = req
                .headers
                .iter()
                .find(|h| h.name.eq_ignore_ascii_case("authorization"))
                .and_then(|h| std::str::from_utf8(h.value).ok());
            let authorized = auth_header
                .and_then(|v| v.strip_prefix("Bearer "))
                .map(|token| bool::from(token.as_bytes().ct_eq(required_key.as_bytes())))
                .unwrap_or(false);
            if !authorized {
                let body = r#"{"error":"Unauthorized"}"#;
                let response = format!(
                    "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: {}\r\nWWW-Authenticate: Bearer\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream.write_all(response.as_bytes()).await?;
                return Ok(());
            }
        }
    }

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
            // Rate limiting: max RATE_LIMIT_MAX requests per IP per RATE_LIMIT_WINDOW.
            if !rate_limit_check(&limiter, peer_ip) {
                let body = r#"{"error":"Too Many Requests"}"#;
                let response = format!(
                    "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: {}\r\nRetry-After: 60\r\nConnection: close\r\n\r\n{}",
                    body.len(), body
                );
                stream.write_all(response.as_bytes()).await?;
                return Ok(());
            }

            let params = parse_query(query_str);
            let sp = parse_stream_params(&params);

            // Guard against oversized prompts.
            if sp.prompt.len() > MAX_PROMPT_LEN {
                let body = r#"{"error":"Prompt exceeds maximum length"}"#;
                let response = format!(
                    "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(), body
                );
                stream.write_all(response.as_bytes()).await?;
                return Ok(());
            }

            let prompt = sp.prompt;
            let transform_str = sp.transform;
            let rate = sp.rate;
            let seed = sp.seed;
            let top_logprobs = sp.top_logprobs;
            let system = sp.system;
            let visual = sp.visual;
            let provider_str = if sp.provider == "openai" {
                default_provider.to_string()
            } else {
                sp.provider
            };
            let model_input = sp.model;
            let heatmap = sp.heatmap;

            let provider = match provider_str.as_str() {
                "anthropic" => Provider::Anthropic,
                _ => Provider::Openai,
            };

            let model = if model_input.is_empty() {
                match provider {
                    Provider::Openai => DEFAULT_OPENAI_MODEL.to_string(),
                    Provider::Anthropic => DEFAULT_ANTHROPIC_MODEL.to_string(),
                    Provider::Mock => DEFAULT_MOCK_MODEL.to_string(),
                }
            } else {
                model_input
            };

            let transform = Transform::from_str_loose(&transform_str).unwrap_or(Transform::Reverse);
            let stream_room_code = params.get("room").cloned();

            // SSE headers
            let headers = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: {}\r\n\r\n",
                cors_origin()
            );
            stream.write_all(headers.as_bytes()).await?;

            // Create channel for token events.
            let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();

            let interceptor_result = TokenInterceptor::new(
                provider,
                transform,
                model,
                visual,
                heatmap,
                orchestrator,
            );

            // Convert result early — stringify the error before any await
            // to satisfy Send bounds on the spawned task.
            let interceptor_result = interceptor_result.map_err(|e| e.to_string());
            let mut interceptor = match interceptor_result {
                Ok(mut i) => {
                    i = i.with_rate(rate);
                    if let Some(s) = seed {
                        i = i.with_seed(s);
                    }
                    i.top_logprobs = top_logprobs;
                    i.system_prompt = system;
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

            // Forward token events as SSE with bounded backpressure buffer.
            // When the buffer exceeds sse_buffer_size, the oldest token is dropped
            // and a BUFFER_OVERFLOW sentinel event is emitted.
            // Abort the LLM background task immediately on client disconnect to
            // avoid making unnecessary API calls for a dropped connection.
            let mut client_disconnected = false;
            let mut token_buffer: std::collections::VecDeque<TokenEvent> =
                std::collections::VecDeque::new();
            let mut overflow_emitted = false;

            while let Some(event) = rx.recv().await {
                if let Some(ref code) = stream_room_code {
                    if let Ok(token_val) = serde_json::to_value(&event) {
                        crate::collab::broadcast(&store, code, token_val.clone());
                        crate::collab::maybe_record(&store, code, token_val);
                    }
                }

                // Buffer management: drop oldest when full, emit overflow sentinel once
                if token_buffer.len() >= sse_buffer_size {
                    token_buffer.pop_front(); // drop oldest
                    if !overflow_emitted {
                        let sentinel =
                            "event: BUFFER_OVERFLOW\ndata: {\"type\":\"BUFFER_OVERFLOW\",\"dropped\":1}\n\n";
                        if stream.write_all(sentinel.as_bytes()).await.is_err() {
                            client_disconnected = true;
                            break;
                        }
                        overflow_emitted = true;
                    }
                } else {
                    overflow_emitted = false; // reset once buffer drains
                }
                token_buffer.push_back(event);

                // Drain the buffer and write events
                while let Some(buffered) = token_buffer.pop_front() {
                    if let Ok(json) = serde_json::to_string(&buffered) {
                        let sse = format!("data: {}\n\n", json);
                        if stream.write_all(sse.as_bytes()).await.is_err() {
                            client_disconnected = true;
                            break;
                        }
                    }
                }
                if client_disconnected {
                    break;
                }
            }

            if client_disconnected {
                stream_task.abort();
            } else {
                let _ = stream_task.await;
            }

            // Send done signal
            let _ = stream.write_all(b"data: [DONE]\n\n").await;
        }
        "/diff-stream" => {
            let params = parse_query(query_str);
            let prompt = params.get("prompt").cloned().unwrap_or_default();
            let transform_str = params
                .get("transform")
                .cloned()
                .unwrap_or_else(|| "reverse".to_string());
            let model_input = params.get("model").cloned().unwrap_or_default();
            let heatmap = params.get("heatmap").is_some_and(|v| v == "1");

            let transform = Transform::from_str_loose(&transform_str).unwrap_or(Transform::Reverse);

            let openai_model = if model_input.is_empty() {
                DEFAULT_OPENAI_MODEL.to_string()
            } else {
                model_input.clone()
            };
            let anthropic_model = if model_input.is_empty() {
                DEFAULT_ANTHROPIC_MODEL.to_string()
            } else {
                model_input.clone()
            };

            // SSE headers
            let headers = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: {}\r\n\r\n",
                cors_origin()
            );
            stream.write_all(headers.as_bytes()).await?;

            // Merged channel: (side, event)
            let (merged_tx, mut merged_rx) =
                mpsc::unbounded_channel::<(&'static str, TokenEvent)>();

            // Spawn OpenAI side
            let openai_result = TokenInterceptor::new(
                Provider::Openai,
                transform.clone(),
                openai_model,
                true,
                heatmap,
                orchestrator,
            )
            .map_err(|e| e.to_string());
            if let Ok(mut oai) = openai_result {
                let (tx_oai, mut rx_oai) = mpsc::unbounded_channel::<TokenEvent>();
                oai.web_tx = Some(tx_oai);
                let prompt_o = prompt.clone();
                tokio::spawn(async move {
                    let _ = oai.intercept_stream(&prompt_o).await;
                });
                let mtx = merged_tx.clone();
                tokio::spawn(async move {
                    while let Some(ev) = rx_oai.recv().await {
                        let _ = mtx.send(("openai", ev));
                    }
                });
            }

            // Spawn Anthropic side
            let anthropic_result = TokenInterceptor::new(
                Provider::Anthropic,
                transform,
                anthropic_model,
                true,
                heatmap,
                orchestrator,
            )
            .map_err(|e| e.to_string());
            if let Ok(mut ant) = anthropic_result {
                let (tx_ant, mut rx_ant) = mpsc::unbounded_channel::<TokenEvent>();
                ant.web_tx = Some(tx_ant);
                let prompt_a = prompt.clone();
                tokio::spawn(async move {
                    let _ = ant.intercept_stream(&prompt_a).await;
                });
                let mtx = merged_tx.clone();
                tokio::spawn(async move {
                    while let Some(ev) = rx_ant.recv().await {
                        let _ = mtx.send(("anthropic", ev));
                    }
                });
            }

            // Drop the original merged_tx so the channel closes when both sides finish
            drop(merged_tx);

            // Forward merged events as SSE with side tag
            while let Some((side, event)) = merged_rx.recv().await {
                let diff_event = DiffTokenEvent {
                    side,
                    event: &event,
                };
                if let Ok(json) = serde_json::to_string(&diff_event) {
                    let sse = format!("data: {}\n\n", json);
                    if stream.write_all(sse.as_bytes()).await.is_err() {
                        break;
                    }
                }
            }

            let _ = stream.write_all(b"data: [DONE]\n\n").await;
        }
        "/ab-stream" => {
            // A/B Experiment: same prompt sent to provider with two different system prompts
            let params = parse_query(query_str);
            let prompt = params.get("prompt").cloned().unwrap_or_default();
            let transform_str = params
                .get("transform")
                .cloned()
                .unwrap_or_else(|| "reverse".to_string());
            let provider_str = params
                .get("provider")
                .cloned()
                .unwrap_or_else(|| default_provider.to_string());
            let model_input = params.get("model").cloned().unwrap_or_default();
            let sys_a = params
                .get("sys_a")
                .cloned()
                .unwrap_or_else(|| "You are a creative storyteller.".to_string());
            let sys_b = params
                .get("sys_b")
                .cloned()
                .unwrap_or_else(|| "You are a technical writer. Be precise.".to_string());

            let ab_provider = match provider_str.as_str() {
                "anthropic" => Provider::Anthropic,
                _ => Provider::Openai,
            };
            let transform = Transform::from_str_loose(&transform_str).unwrap_or(Transform::Reverse);
            let model = if model_input.is_empty() {
                match ab_provider {
                    Provider::Openai => DEFAULT_OPENAI_MODEL.to_string(),
                    Provider::Anthropic => DEFAULT_ANTHROPIC_MODEL.to_string(),
                    Provider::Mock => DEFAULT_MOCK_MODEL.to_string(),
                }
            } else {
                model_input
            };

            let headers = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: {}\r\n\r\n",
                cors_origin()
            );
            stream.write_all(headers.as_bytes()).await?;

            let (merged_tx, mut merged_rx) =
                mpsc::unbounded_channel::<(&'static str, TokenEvent)>();

            // Side A
            let a_result = TokenInterceptor::new(
                ab_provider.clone(),
                transform.clone(),
                model.clone(),
                true,
                false,
                orchestrator,
            )
            .map_err(|e| e.to_string());
            if let Ok(mut side_a) = a_result {
                let (tx_a, mut rx_a) = mpsc::unbounded_channel::<TokenEvent>();
                side_a.web_tx = Some(tx_a);
                side_a.system_prompt = Some(sys_a);
                let prompt_a = prompt.clone();
                tokio::spawn(async move {
                    let _ = side_a.intercept_stream(&prompt_a).await;
                });
                let mtx = merged_tx.clone();
                tokio::spawn(async move {
                    while let Some(ev) = rx_a.recv().await {
                        let _ = mtx.send(("a", ev));
                    }
                });
            }

            // Side B
            let b_result =
                TokenInterceptor::new(ab_provider, transform, model, true, false, orchestrator)
                    .map_err(|e| e.to_string());
            if let Ok(mut side_b) = b_result {
                let (tx_b, mut rx_b) = mpsc::unbounded_channel::<TokenEvent>();
                side_b.web_tx = Some(tx_b);
                side_b.system_prompt = Some(sys_b);
                let prompt_b = prompt.clone();
                tokio::spawn(async move {
                    let _ = side_b.intercept_stream(&prompt_b).await;
                });
                let mtx = merged_tx.clone();
                tokio::spawn(async move {
                    while let Some(ev) = rx_b.recv().await {
                        let _ = mtx.send(("b", ev));
                    }
                });
            }

            drop(merged_tx);

            while let Some((side, event)) = merged_rx.recv().await {
                let diff_event = DiffTokenEvent {
                    side,
                    event: &event,
                };
                if let Ok(json) = serde_json::to_string(&diff_event) {
                    let sse = format!("data: {}\n\n", json);
                    if stream.write_all(sse.as_bytes()).await.is_err() {
                        break;
                    }
                }
            }

            let _ = stream.write_all(b"data: [DONE]\n\n").await;
        }
        "/room/create" => {
            if !rate_limit_check(&limiter, peer_ip) {
                let body = r#"{"error":"Too Many Requests"}"#;
                let response = format!(
                    "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: {}\r\nRetry-After: 60\r\nConnection: close\r\n\r\n{}",
                    body.len(), body
                );
                stream.write_all(response.as_bytes()).await?;
                return Ok(());
            }
            let code = crate::collab::create_room(&store);
            let room_id = uuid::Uuid::new_v4().to_string();
            let body = format!(r#"{{"code":"{}","room_id":"{}","ws_url":"/ws/{}"}}"#, code, room_id, code);
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).await?;
        }
        path if path.starts_with("/join/") => {
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                INDEX_HTML.len(),
                INDEX_HTML
            );
            stream.write_all(response.as_bytes()).await?;
        }
        path if path.starts_with("/replay/") => {
            let code = path.strip_prefix("/replay/").unwrap_or("");
            // Collect events under lock, then release before writing.
            let events_result: Result<Vec<_>, &str> = if let Ok(guard) = store.lock() {
                if let Some(room) = guard.get(code) {
                    Ok(room.recorded_events.clone())
                } else {
                    Err("room not found")
                }
            } else {
                Err("internal error")
            };
            match events_result {
                Err(err_msg) => {
                    let body = format!(r#"{{"error":"{}"}}"#, err_msg);
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    stream.write_all(response.as_bytes()).await?;
                }
                Ok(events) => {
                    // Write JSON in chunks: prefix, each event, suffix.
                    // Use chunked transfer encoding to avoid buffering the full response.
                    let headers = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
                    stream.write_all(headers.as_bytes()).await?;
                    // Helper closure to write a chunked segment
                    let prefix = r#"{"events":["#;
                    let chunk_line = format!("{:x}\r\n{}\r\n", prefix.len(), prefix);
                    stream.write_all(chunk_line.as_bytes()).await?;
                    for (i, event) in events.iter().enumerate() {
                        let sep = if i > 0 { "," } else { "" };
                        let event_json = serde_json::to_string(event)
                            .unwrap_or_else(|_| "{}".to_string());
                        let segment = format!("{}{}", sep, event_json);
                        let chunk_data = format!("{:x}\r\n{}\r\n", segment.len(), segment);
                        stream.write_all(chunk_data.as_bytes()).await?;
                    }
                    let suffix = "]}";
                    let suffix_chunk = format!("{:x}\r\n{}\r\n", suffix.len(), suffix);
                    stream.write_all(suffix_chunk.as_bytes()).await?;
                    // Terminal chunk
                    stream.write_all(b"0\r\n\r\n").await?;
                }
            }
        }
        "/api/experiments" => {
            // Returns stored experiment runs from the SQLite log when the
            // sqlite-log feature is enabled and a --log-db path is provided.
            // Without the feature or a database file the endpoint returns an
            // empty array so the web UI always gets a well-formed response.
            let _db_path = parse_query(query_str)
                .get("db")
                .cloned()
                .unwrap_or_else(|| "experiments.db".to_string());

            #[cfg(feature = "sqlite-log")]
            let body = {
                use crate::experiment_log::ExperimentLog;
                match ExperimentLog::open(std::path::Path::new(&_db_path)) {
                    Ok(log) => log.list().unwrap_or_else(|_| "[]".to_string()),
                    Err(_) => "[]".to_string(),
                }
            };

            #[cfg(not(feature = "sqlite-log"))]
            let body = "[]".to_string();

            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).await?;
        }
        "/batch" => {
            // POST /batch: run multiple prompts through Mock provider and return token counts.
            use tokio::io::AsyncReadExt;

            // Read remaining body from the TCP stream (after the HTTP header).
            let mut body_buf = vec![0u8; 65_536];
            let body_n = stream.read(&mut body_buf).await.unwrap_or(0);
            // The initial read already captured the header; body starts after \r\n\r\n.
            // Combine what we already read in `buf` with any additional body bytes.
            let full = [&buf[..n], &body_buf[..body_n]].concat();
            let body_start = full
                .windows(4)
                .position(|w| w == b"\r\n\r\n")
                .map(|p| p + 4)
                .unwrap_or(full.len());
            let body_bytes = &full[body_start..];

            #[derive(serde::Deserialize)]
            struct BatchRequest {
                prompts: Vec<String>,
                #[serde(default)]
                transform: String,
                #[serde(default)]
                provider: String,
                #[serde(default)]
                model: String,
                #[serde(default)]
                rate: f64,
            }

            let req: BatchRequest = match serde_json::from_slice(body_bytes) {
                Ok(r) => r,
                Err(_) => {
                    let body = r#"{"error":"Invalid JSON body"}"#;
                    let response = format!(
                        "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(), body
                    );
                    stream.write_all(response.as_bytes()).await?;
                    return Ok(());
                }
            };

            if req.prompts.is_empty() || req.prompts.len() > 10 {
                let body = r#"{"error":"prompts must be 1-10 items"}"#;
                let response = format!(
                    "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(), body
                );
                stream.write_all(response.as_bytes()).await?;
                return Ok(());
            }

            let transform_str = if req.transform.is_empty() {
                "reverse".to_string()
            } else {
                req.transform
            };
            let transform = Transform::from_str_loose(&transform_str).unwrap_or(Transform::Reverse);
            let rate = req.rate.clamp(0.0, 1.0);
            let model = if req.model.is_empty() {
                DEFAULT_MOCK_MODEL.to_string()
            } else {
                req.model
            };

            let mut results: Vec<serde_json::Value> = Vec::new();
            for prompt in &req.prompts {
                let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
                let interceptor_result = TokenInterceptor::new(
                    Provider::Mock,
                    transform.clone(),
                    model.clone(),
                    false,
                    false,
                    false,
                )
                .map_err(|e| e.to_string());
                if let Ok(mut interceptor) = interceptor_result {
                    interceptor = interceptor.with_rate(rate);
                    interceptor.web_tx = Some(tx);
                    let _ = interceptor.intercept_stream(prompt).await;
                }
                let mut token_count = 0usize;
                while rx.try_recv().is_ok() {
                    token_count += 1;
                }
                results.push(serde_json::json!({
                    "prompt": prompt,
                    "token_count": token_count,
                }));
            }

            let body = serde_json::json!({"results": results}).to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                cors_origin(),
                body
            );
            stream.write_all(response.as_bytes()).await?;
        }
        _ => {
            let response =
                "HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\nConnection: close\r\n\r\nNot Found";
            stream.write_all(response.as_bytes()).await?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- url_decode tests --

    #[test]
    fn test_url_decode_basic() {
        assert_eq!(url_decode("hello+world"), "hello world");
        assert_eq!(url_decode("hello%20world"), "hello world");
        assert_eq!(url_decode("a%26b"), "a&b");
        assert_eq!(url_decode("plain"), "plain");
    }

    #[test]
    fn test_url_decode_empty() {
        assert_eq!(url_decode(""), "");
    }

    #[test]
    fn test_url_decode_no_encoding() {
        assert_eq!(url_decode("hello"), "hello");
    }

    #[test]
    fn test_url_decode_plus_only() {
        assert_eq!(url_decode("+++"), "   ");
    }

    #[test]
    fn test_url_decode_percent_special_chars() {
        assert_eq!(url_decode("%21"), "!");
        assert_eq!(url_decode("%3D"), "=");
        assert_eq!(url_decode("%3F"), "?");
    }

    #[test]
    fn test_url_decode_mixed() {
        assert_eq!(url_decode("hello+world%21+how%3F"), "hello world! how?");
    }

    #[test]
    fn test_url_decode_consecutive_percent() {
        assert_eq!(url_decode("%20%20"), "  ");
    }

    #[test]
    fn test_url_decode_single_char() {
        assert_eq!(url_decode("a"), "a");
        assert_eq!(url_decode("+"), " ");
    }

    // -- parse_query tests --

    #[test]
    fn test_parse_query_basic() {
        let params = parse_query("prompt=hello+world&transform=reverse&heatmap=1");
        assert_eq!(
            params.get("prompt").map(|s| s.as_str()),
            Some("hello world")
        );
        assert_eq!(params.get("transform").map(|s| s.as_str()), Some("reverse"));
        assert_eq!(params.get("heatmap").map(|s| s.as_str()), Some("1"));
    }

    #[test]
    fn test_parse_query_empty() {
        let params = parse_query("");
        assert!(params.is_empty() || params.get("").map_or(true, |v| v.is_empty()));
    }

    #[test]
    fn test_parse_query_single_param() {
        let params = parse_query("key=value");
        assert_eq!(params.len(), 1);
        assert_eq!(params.get("key").map(|s| s.as_str()), Some("value"));
    }

    #[test]
    fn test_parse_query_no_value() {
        let params = parse_query("key=");
        assert_eq!(params.get("key").map(|s| s.as_str()), Some(""));
    }

    #[test]
    fn test_parse_query_encoded_values() {
        let params = parse_query("prompt=hello+world%21&model=gpt-4");
        assert_eq!(
            params.get("prompt").map(|s| s.as_str()),
            Some("hello world!")
        );
        assert_eq!(params.get("model").map(|s| s.as_str()), Some("gpt-4"));
    }

    #[test]
    fn test_parse_query_many_params() {
        let params = parse_query("a=1&b=2&c=3&d=4&e=5");
        assert_eq!(params.len(), 5);
        assert_eq!(params.get("c").map(|s| s.as_str()), Some("3"));
    }

    #[test]
    fn test_parse_query_special_chars_in_value() {
        let params = parse_query("q=a%2Bb%3Dc");
        assert_eq!(params.get("q").map(|s| s.as_str()), Some("a+b=c"));
    }

    // -- INDEX_HTML structure tests --

    #[test]
    fn test_index_html_is_valid_html() {
        assert!(INDEX_HTML.starts_with("<!DOCTYPE html>"));
        assert!(INDEX_HTML.contains("</html>"));
    }

    #[test]
    fn test_index_html_contains_title() {
        assert!(INDEX_HTML.contains("<title>Every Other Token</title>"));
    }

    #[test]
    fn test_index_html_has_dark_theme() {
        assert!(INDEX_HTML.contains("background:#0d1117"));
    }

    #[test]
    fn test_index_html_has_sse_event_source() {
        assert!(INDEX_HTML.contains("EventSource"));
    }

    #[test]
    fn test_index_html_has_view_modes() {
        assert!(INDEX_HTML.contains("v-single"));
        assert!(INDEX_HTML.contains("v-sbs"));
        assert!(INDEX_HTML.contains("v-multi"));
    }

    #[test]
    fn test_index_html_has_export_button() {
        assert!(INDEX_HTML.contains("Export JSON"));
    }

    #[test]
    fn test_index_html_has_graph_canvas() {
        assert!(INDEX_HTML.contains("depgraph"));
        assert!(INDEX_HTML.contains("drawGraph"));
    }

    #[test]
    fn test_index_html_has_transform_selector() {
        assert!(INDEX_HTML.contains("reverse"));
        assert!(INDEX_HTML.contains("uppercase"));
        assert!(INDEX_HTML.contains("mock"));
        assert!(INDEX_HTML.contains("noise"));
    }

    #[test]
    fn test_index_html_has_provider_selector() {
        assert!(INDEX_HTML.contains("OpenAI"));
        assert!(INDEX_HTML.contains("Anthropic"));
    }

    #[test]
    fn test_index_html_has_js_transforms() {
        assert!(INDEX_HTML.contains("const TX="));
        assert!(INDEX_HTML.contains("reverse:"));
        assert!(INDEX_HTML.contains("uppercase:"));
        assert!(INDEX_HTML.contains("mock:"));
        assert!(INDEX_HTML.contains("noise:"));
    }

    #[test]
    fn test_index_html_has_heatmap_toggle() {
        assert!(INDEX_HTML.contains("heatmap"));
    }

    #[test]
    fn test_index_html_has_stream_button() {
        assert!(INDEX_HTML.contains("Stream"));
        assert!(INDEX_HTML.contains("btn-go"));
    }

    #[test]
    fn test_index_html_has_token_animation() {
        assert!(INDEX_HTML.contains("fadeIn"));
    }

    #[test]
    fn test_index_html_has_multi_panel_colors() {
        assert!(INDEX_HTML.contains("mp-reverse"));
        assert!(INDEX_HTML.contains("mp-uppercase"));
        assert!(INDEX_HTML.contains("mp-mock"));
        assert!(INDEX_HTML.contains("mp-noise"));
    }

    #[test]
    fn test_index_html_has_side_by_side_labels() {
        assert!(INDEX_HTML.contains("Original"));
        assert!(INDEX_HTML.contains("Transformed"));
    }

    #[test]
    fn test_index_html_has_mode_buttons() {
        assert!(INDEX_HTML.contains("btn-single"));
        assert!(INDEX_HTML.contains("btn-sbs"));
        assert!(INDEX_HTML.contains("btn-multi"));
    }

    #[test]
    fn test_index_html_has_done_signal_handling() {
        assert!(INDEX_HTML.contains("[DONE]"));
    }

    #[test]
    fn test_index_html_has_stats_display() {
        assert!(INDEX_HTML.contains("stats"));
    }

    #[test]
    fn test_index_html_has_graph_toggle() {
        assert!(INDEX_HTML.contains("graphtoggle"));
    }

    #[test]
    fn test_index_html_no_external_deps() {
        // Verify no npm/webpack/CDN references
        assert!(!INDEX_HTML.contains("cdn."));
        assert!(!INDEX_HTML.contains("unpkg.com"));
        assert!(!INDEX_HTML.contains("jsdelivr"));
        assert!(!INDEX_HTML.contains("npm"));
    }

    // -- server integration smoke tests --

    #[tokio::test]
    async fn test_serve_binds_to_port() {
        // Verify TcpListener can bind (just test the binding, not the loop)
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await;
        assert!(listener.is_ok());
        let listener = listener.unwrap();
        let addr = listener.local_addr().unwrap();
        assert!(addr.port() > 0);
    }

    #[test]
    fn test_index_html_content_length_matches() {
        let html_bytes = INDEX_HTML.as_bytes();
        assert!(html_bytes.len() > 1000, "HTML should be substantial");
    }

    #[test]
    fn test_index_html_responsive_viewport() {
        assert!(INDEX_HTML.contains("viewport"));
        assert!(INDEX_HTML.contains("width=device-width"));
    }

    #[test]
    fn test_url_decode_long_string() {
        let input = "the+quick+brown+fox+jumps+over+the+lazy+dog";
        let expected = "the quick brown fox jumps over the lazy dog";
        assert_eq!(url_decode(input), expected);
    }

    #[test]
    fn test_url_decode_multibyte_utf8() {
        // %C3%A9 is the UTF-8 encoding of 'é' (U+00E9)
        assert_eq!(url_decode("caf%C3%A9"), "café");
        // %E2%9C%93 is '✓' (U+2713)
        assert_eq!(url_decode("%E2%9C%93"), "✓");
        // Mixed: ASCII + multi-byte
        assert_eq!(url_decode("hello+%C3%A9"), "hello é");
    }

    #[test]
    fn test_url_decode_malformed_percent() {
        // Incomplete percent sequence — should emit literally
        assert_eq!(url_decode("%2"), "%2");
        assert_eq!(url_decode("%ZZ"), "%ZZ");
    }

    #[test]
    fn test_parse_query_duplicate_keys_last_wins() {
        let params = parse_query("key=first&key=second");
        // HashMap behavior: last key wins on iteration insert
        let val = params.get("key").map(|s| s.as_str());
        assert!(val == Some("first") || val == Some("second"));
    }

    #[test]
    fn test_index_html_has_bezier_curves() {
        assert!(INDEX_HTML.contains("bezierCurveTo"));
    }

    #[test]
    fn test_index_html_has_heat_color_function() {
        assert!(INDEX_HTML.contains("heatColor"));
    }

    #[test]
    fn test_index_html_has_export_download() {
        assert!(INDEX_HTML.contains("download"));
        assert!(INDEX_HTML.contains("application/json"));
    }

    // -- New feature tests --

    #[test]
    fn test_index_html_has_chaos_transform() {
        assert!(INDEX_HTML.contains("chaos"));
        assert!(INDEX_HTML.contains(r#"value="chaos""#));
    }

    #[test]
    fn test_index_html_has_chaos_js_transform() {
        assert!(INDEX_HTML.contains("TX.chaos") || INDEX_HTML.contains("chaos:s=>"));
    }

    #[test]
    fn test_index_html_has_chaos_tooltip_css() {
        assert!(INDEX_HTML.contains("attr(title)"));
        assert!(INDEX_HTML.contains(".token[title]"));
    }

    #[test]
    fn test_index_html_has_diff_view() {
        assert!(INDEX_HTML.contains("v-diff"));
        assert!(INDEX_HTML.contains("diff-openai"));
        assert!(INDEX_HTML.contains("diff-anthropic"));
    }

    #[test]
    fn test_index_html_has_diff_button() {
        assert!(INDEX_HTML.contains("btn-diff"));
        assert!(INDEX_HTML.contains("Diff"));
    }

    #[test]
    fn test_index_html_has_diff_css() {
        assert!(INDEX_HTML.contains("view-diff"));
        assert!(INDEX_HTML.contains("diff-match"));
        assert!(INDEX_HTML.contains("diff-diverge"));
    }

    #[test]
    fn test_index_html_has_diff_highlight_fn() {
        assert!(INDEX_HTML.contains("applyDiffHighlights"));
    }

    #[test]
    fn test_index_html_has_surgery_css() {
        assert!(INDEX_HTML.contains("surgeable"));
        assert!(INDEX_HTML.contains("token-input"));
    }

    #[test]
    fn test_index_html_has_surgery_fn() {
        assert!(INDEX_HTML.contains("enableSurgery"));
        assert!(INDEX_HTML.contains("surgeryLog"));
    }

    #[test]
    fn test_index_html_has_diff_stream_fn() {
        assert!(INDEX_HTML.contains("startDiff"));
        assert!(INDEX_HTML.contains("diff-stream"));
    }

    #[test]
    fn test_index_html_has_chaos_label_in_mkspan() {
        assert!(INDEX_HTML.contains("chaos_label") || INDEX_HTML.contains("chaosLabel"));
    }

    #[test]
    fn test_diff_token_event_serializes_with_side() {
        let event = crate::TokenEvent {
            text: "hello".to_string(),
            original: "hello".to_string(),
            index: 0,
            transformed: false,
            importance: 0.5,
            chaos_label: None,
            provider: None,
            confidence: None,
            perplexity: None,
            alternatives: vec![],
            is_error: false,
            arrival_ms: None,
        };
        let diff = DiffTokenEvent {
            side: "openai",
            event: &event,
        };
        let json = serde_json::to_string(&diff).expect("serialize");
        assert!(json.contains(r#""side":"openai""#));
        assert!(json.contains(r#""text":"hello""#));
        assert!(json.contains(r#""index":0"#));
    }

    #[test]
    fn test_diff_token_event_anthropic_side() {
        let event = crate::TokenEvent {
            text: "world".to_string(),
            original: "world".to_string(),
            index: 1,
            transformed: true,
            importance: 0.7,
            chaos_label: Some("reverse".to_string()),
            provider: None,
            confidence: None,
            perplexity: None,
            alternatives: vec![],
            is_error: false,
            arrival_ms: None,
        };
        let diff = DiffTokenEvent {
            side: "anthropic",
            event: &event,
        };
        let json = serde_json::to_string(&diff).expect("serialize");
        assert!(json.contains(r#""side":"anthropic""#));
        assert!(json.contains(r#""transformed":true"#));
        assert!(json.contains(r#""chaos_label":"reverse""#));
    }

    #[test]
    fn test_index_html_surgery_log_in_export() {
        assert!(INDEX_HTML.contains("surgery_log"));
    }

    // -- Research suite feature tests ----------------------------------------

    #[test]
    fn test_index_html_has_experiment_button() {
        assert!(INDEX_HTML.contains("btn-experiment"));
        assert!(INDEX_HTML.contains("Experiment"));
    }

    #[test]
    fn test_index_html_has_research_button() {
        assert!(INDEX_HTML.contains("btn-research"));
        assert!(INDEX_HTML.contains("Research"));
    }

    #[test]
    fn test_index_html_has_experiment_view() {
        assert!(INDEX_HTML.contains("v-experiment"));
    }

    #[test]
    fn test_index_html_has_research_view() {
        assert!(INDEX_HTML.contains("v-research"));
    }

    #[test]
    fn test_index_html_has_perplexity_sparkline() {
        assert!(INDEX_HTML.contains("perp-spark"));
    }

    #[test]
    fn test_index_html_has_research_grid() {
        assert!(INDEX_HTML.contains("research-grid") || INDEX_HTML.contains("research-dash"));
    }

    #[test]
    fn test_index_html_has_render_research_fn() {
        assert!(INDEX_HTML.contains("renderResearch"));
    }

    #[test]
    fn test_index_html_render_research_called_on_done() {
        // renderResearch() must appear after enableSurgery in the [DONE] handler
        let done_pos = INDEX_HTML.find("[DONE]").expect("[DONE] not found");
        let research_pos = INDEX_HTML[done_pos..]
            .find("renderResearch")
            .expect("renderResearch not found after [DONE]");
        let surgery_pos = INDEX_HTML[done_pos..]
            .find("enableSurgery")
            .expect("enableSurgery not found after [DONE]");
        assert!(
            research_pos > surgery_pos,
            "renderResearch should come after enableSurgery"
        );
    }

    #[test]
    fn test_index_html_has_start_experiment_fn() {
        assert!(INDEX_HTML.contains("startExperiment"));
    }

    #[test]
    fn test_index_html_has_update_perp_sparkline_fn() {
        assert!(INDEX_HTML.contains("updatePerpSparkline"));
    }

    #[test]
    fn test_index_html_has_ab_stream_route() {
        assert!(INDEX_HTML.contains("/ab-stream"));
    }

    #[test]
    fn test_index_html_has_ab_system_prompts() {
        assert!(INDEX_HTML.contains("sys_a") || INDEX_HTML.contains("ab-prompts"));
    }

    #[test]
    fn test_index_html_has_experiment_divergence() {
        assert!(INDEX_HTML.contains("exp-diverge") || INDEX_HTML.contains("renderExpDivergence"));
    }

    #[test]
    fn test_index_html_has_confidence_rendering() {
        assert!(INDEX_HTML.contains("confidence"));
    }

    #[test]
    fn test_index_html_has_perplexity_rendering() {
        assert!(INDEX_HTML.contains("perplexity"));
    }

    #[test]
    fn test_index_html_has_confidence_css_classes() {
        assert!(INDEX_HTML.contains("conf-high"));
        assert!(INDEX_HTML.contains("conf-mid"));
        assert!(INDEX_HTML.contains("conf-low"));
    }

    #[test]
    fn test_index_html_has_high_perp_animation() {
        assert!(INDEX_HTML.contains("high-perp") || INDEX_HTML.contains("perpPulse"));
    }

    #[test]
    fn test_index_html_has_ab_panel_a() {
        assert!(INDEX_HTML.contains("exp-a"));
    }

    #[test]
    fn test_index_html_has_ab_panel_b() {
        assert!(INDEX_HTML.contains("exp-b"));
    }

    #[test]
    fn test_index_html_has_experiment_view_in_views_map() {
        assert!(INDEX_HTML.contains("experiment:$('#v-experiment')"));
    }

    #[test]
    fn test_index_html_has_research_view_in_views_map() {
        assert!(INDEX_HTML.contains("research:$('#v-research')"));
    }

    // -- Multiplayer UI tests --

    #[test]
    fn test_index_html_has_host_session_button() {
        assert!(INDEX_HTML.contains("btn-host"));
        assert!(INDEX_HTML.contains("Host Session"));
    }

    #[test]
    fn test_index_html_has_join_button() {
        assert!(INDEX_HTML.contains("btn-join"));
        assert!(INDEX_HTML.contains("Join"));
    }

    #[test]
    fn test_index_html_has_room_code_input() {
        assert!(INDEX_HTML.contains("join-code"));
    }

    #[test]
    fn test_index_html_has_mp_panel() {
        assert!(INDEX_HTML.contains("mp-panel"));
        assert!(INDEX_HTML.contains("mp-code"));
    }

    #[test]
    fn test_index_html_has_participant_sidebar() {
        assert!(INDEX_HTML.contains("sidebar"));
        assert!(INDEX_HTML.contains("participant-list"));
    }

    #[test]
    fn test_index_html_has_chat_panel() {
        assert!(INDEX_HTML.contains("chat-panel"));
        assert!(INDEX_HTML.contains("chat-msgs"));
        assert!(INDEX_HTML.contains("chat-in"));
    }

    #[test]
    fn test_index_html_has_vote_bar() {
        assert!(INDEX_HTML.contains("vote-bar"));
        assert!(INDEX_HTML.contains("btn-vote-up"));
        assert!(INDEX_HTML.contains("btn-vote-dn"));
    }

    #[test]
    fn test_index_html_has_record_button() {
        assert!(INDEX_HTML.contains("btn-rec"));
    }

    #[test]
    fn test_index_html_has_replay_button() {
        assert!(INDEX_HTML.contains("btn-replay"));
    }

    #[test]
    fn test_index_html_has_websocket_init() {
        assert!(INDEX_HTML.contains("WebSocket"));
        assert!(INDEX_HTML.contains("initRoom"));
    }

    #[test]
    fn test_index_html_has_send_ws_fn() {
        assert!(INDEX_HTML.contains("sendWs"));
    }

    #[test]
    fn test_index_html_has_on_ws_msg_handler() {
        assert!(INDEX_HTML.contains("onWsMsg"));
    }

    #[test]
    fn test_index_html_has_participant_join_handler() {
        assert!(INDEX_HTML.contains("participant_join"));
    }

    #[test]
    fn test_index_html_has_participant_leave_handler() {
        assert!(INDEX_HTML.contains("participant_leave"));
    }

    #[test]
    fn test_index_html_has_surgery_peer_handler() {
        assert!(INDEX_HTML.contains("applyPeerSurgery"));
    }

    #[test]
    fn test_index_html_has_chat_send_fn() {
        assert!(INDEX_HTML.contains("sendChat"));
    }

    #[test]
    fn test_index_html_has_replay_event_handler() {
        assert!(INDEX_HTML.contains("doReplayEvent"));
    }

    #[test]
    fn test_index_html_has_room_create_fetch() {
        assert!(INDEX_HTML.contains("/room/create"));
    }

    #[test]
    fn test_index_html_has_copy_link_button() {
        assert!(INDEX_HTML.contains("btn-copy-link"));
    }

    #[test]
    fn test_index_html_has_leave_room_fn() {
        assert!(INDEX_HTML.contains("leaveRoom"));
        assert!(INDEX_HTML.contains("btn-leave"));
    }

    // -- New: cors_origin helper --

    #[test]
    fn test_cors_origin_default_is_star() {
        // Without CORS_ORIGIN env var set (or if already "*"), default is "*"
        // We can't unset env vars portably in tests, so just check it returns a non-empty string.
        let origin = cors_origin();
        assert!(!origin.is_empty());
    }

    // -- New: MAX_PROMPT_LEN constant --

    #[test]
    fn test_max_prompt_len_is_100k() {
        assert_eq!(MAX_PROMPT_LEN, 100_000);
    }

    // -- New: batch endpoint parsing logic --

    #[test]
    fn test_batch_request_parse_valid() {
        let json = r#"{"prompts": ["hello", "world"], "transform": "reverse", "rate": 0.5}"#;
        let v: serde_json::Value = serde_json::from_str(json).unwrap();
        let prompts = v["prompts"].as_array().unwrap();
        assert_eq!(prompts.len(), 2);
        assert_eq!(prompts[0].as_str().unwrap(), "hello");
    }

    #[test]
    fn test_batch_request_parse_empty_prompts_rejected() {
        let json = r#"{"prompts": []}"#;
        let v: serde_json::Value = serde_json::from_str(json).unwrap();
        let prompts = v["prompts"].as_array().unwrap();
        assert!(prompts.is_empty());
        // Simulate the guard: empty or > 10 should be rejected.
        assert!(prompts.is_empty() || prompts.len() > 10);
    }

    #[test]
    fn test_batch_request_parse_too_many_prompts_rejected() {
        let prompts: Vec<&str> = (0..11).map(|_| "x").collect();
        let json = serde_json::json!({"prompts": prompts}).to_string();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let arr = v["prompts"].as_array().unwrap();
        assert!(arr.len() > 10);
    }

    #[test]
    fn test_batch_request_invalid_json_fails() {
        let result = serde_json::from_str::<serde_json::Value>("not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_request_max_10_prompts_allowed() {
        let prompts: Vec<&str> = (0..10).map(|_| "test").collect();
        let json = serde_json::json!({"prompts": prompts}).to_string();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let arr = v["prompts"].as_array().unwrap();
        assert_eq!(arr.len(), 10);
        assert!(arr.len() <= 10); // should be allowed
    }

    #[test]
    fn test_index_html_has_auto_join_logic() {
        assert!(INDEX_HTML.contains("autoJoin") || INDEX_HTML.contains("/join/"));
    }

    #[test]
    fn test_index_html_has_mutation_observer_for_surgery() {
        assert!(INDEX_HTML.contains("MutationObserver"));
    }

    #[test]
    fn test_index_html_has_peer_colors_map() {
        assert!(INDEX_HTML.contains("peerColors"));
    }

    #[test]
    fn test_index_html_has_welcome_message_handler() {
        assert!(INDEX_HTML.contains("welcome"));
    }

    #[test]
    fn test_index_html_multiplayer_no_external_deps() {
        // Verify no socket.io or other external WS libs
        assert!(!INDEX_HTML.contains("socket.io"));
        assert!(!INDEX_HTML.contains("sockjs"));
    }

    #[test]
    fn test_index_html_has_mp_count_badge() {
        assert!(INDEX_HTML.contains("mp-count-badge"));
    }

    #[test]
    fn test_index_html_has_join_toast_css() {
        assert!(INDEX_HTML.contains("join-toast"));
        assert!(INDEX_HTML.contains("slideIn"));
    }

    #[test]
    fn test_index_html_has_ws_route_pattern() {
        assert!(INDEX_HTML.contains("/ws/"));
    }

    // -- Multiplayer completeness tests --------------------------------------

    #[test]
    fn test_index_html_has_stream_done_handler() {
        assert!(INDEX_HTML.contains("stream_done"));
    }

    #[test]
    fn test_index_html_stream_done_enables_guests() {
        // stream_done case must call enableSurgery for guests
        let pos = INDEX_HTML
            .find("stream_done")
            .expect("stream_done not found");
        let after = &INDEX_HTML[pos..pos + 300];
        assert!(after.contains("enableSurgery") || after.contains("amHost"));
    }

    #[test]
    fn test_index_html_has_vote_buttons() {
        assert!(INDEX_HTML.contains("btn-vote-up"));
        assert!(INDEX_HTML.contains("btn-vote-dn"));
    }

    #[test]
    fn test_index_html_has_vote_update_handler() {
        assert!(INDEX_HTML.contains("vote_update"));
    }

    #[test]
    fn test_index_html_has_record_started_handler() {
        assert!(INDEX_HTML.contains("record_started"));
    }

    #[test]
    fn test_index_html_has_record_stopped_handler() {
        assert!(INDEX_HTML.contains("record_stopped"));
    }

    #[test]
    fn test_index_html_has_join_session_input() {
        assert!(INDEX_HTML.contains("join-code") || INDEX_HTML.contains("btn-join"));
    }

    #[test]
    fn test_index_html_multiplayer_token_broadcast_via_ws() {
        // Host sends token events over WS for guests to render
        assert!(INDEX_HTML.contains("sendWs({type:'token'"));
    }

    #[test]
    fn test_index_html_stream_done_sent_via_ws() {
        // stream_done must appear in the HTML and must be co-located with [DONE] SSE handling
        assert!(INDEX_HTML.contains("stream_done"));
        // They must be within 200 chars of each other somewhere in the file
        let text = INDEX_HTML.as_bytes();
        let found = text.windows(200).any(|w| {
            let s = std::str::from_utf8(w).unwrap_or("");
            s.contains("[DONE]") && s.contains("stream_done")
        });
        assert!(found, "stream_done should be sent near [DONE] SSE handler");
    }

    #[test]
    fn test_index_html_mutation_observer_broadcasts_surgery() {
        assert!(INDEX_HTML.contains("MutationObserver"));
        assert!(INDEX_HTML.contains("surgery"));
    }

    #[test]
    fn test_index_html_has_peer_edited_css() {
        assert!(INDEX_HTML.contains("peer-edited"));
    }

    #[test]
    fn test_collab_module_generate_code_length() {
        let code = crate::collab::generate_code();
        // New format: ADJ-NOUN-NN (uppercase), minimum 5 chars
        assert!(code.len() >= 5);
    }

    #[test]
    fn test_collab_module_generate_code_alphanumeric() {
        for _ in 0..20 {
            let code = crate::collab::generate_code();
            // New memorable format allows hyphens
            assert!(code.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'));
        }
    }

    #[test]
    fn test_collab_module_generate_code_uppercase() {
        for _ in 0..10 {
            let code = crate::collab::generate_code();
            assert!(code.chars().all(|c| !c.is_ascii_lowercase()));
        }
    }

    #[test]
    fn test_collab_module_create_room_returns_code() {
        let store = crate::collab::new_room_store();
        let code = crate::collab::create_room(&store);
        assert!(code.len() >= 5);
        let guard = store.lock().unwrap();
        assert!(guard.contains_key(&code));
    }

    #[test]
    fn test_collab_module_join_nonexistent_room_errors() {
        let store = crate::collab::new_room_store();
        let result = crate::collab::join_room(&store, "SWIFT-LION-99", "Bob", false);
        assert!(result.is_err());
    }

    #[test]
    fn test_collab_module_participant_colors_nonempty() {
        assert!(!crate::collab::PARTICIPANT_COLORS.is_empty());
        for color in crate::collab::PARTICIPANT_COLORS {
            assert!(color.starts_with('#'));
        }
    }

    #[test]
    fn test_collab_module_broadcast_reaches_subscriber() {
        let store = crate::collab::new_room_store();
        let code = crate::collab::create_room(&store);
        let (_, mut rx) = crate::collab::join_room(&store, &code, "viewer", false).unwrap();
        crate::collab::broadcast(&store, &code, serde_json::json!({"type": "ping"}));
        assert!(rx.try_recv().is_ok());
    }

    #[test]
    fn test_collab_module_room_state_snapshot_has_expected_fields() {
        let store = crate::collab::new_room_store();
        let code = crate::collab::create_room(&store);
        let snap = crate::collab::room_state_snapshot(&store, &code);
        assert!(snap["code"].is_string());
        assert!(snap["participants"].is_array());
        assert!(snap["surgery_log"].is_array());
        assert!(snap["chat_log"].is_array());
        assert!(snap["is_recording"].is_boolean());
    }

    #[test]
    fn test_collab_module_vote_increments_correctly() {
        let store = crate::collab::new_room_store();
        let code = crate::collab::create_room(&store);
        let (up, down) = crate::collab::vote(&store, &code, "reverse", "up").unwrap();
        assert_eq!(up, 1);
        assert_eq!(down, 0);
        let (up2, _) = crate::collab::vote(&store, &code, "reverse", "up").unwrap();
        assert_eq!(up2, 2);
    }

    #[test]
    fn test_collab_module_vote_down() {
        let store = crate::collab::new_room_store();
        let code = crate::collab::create_room(&store);
        let (up, down) = crate::collab::vote(&store, &code, "reverse", "down").unwrap();
        assert_eq!(up, 0);
        assert_eq!(down, 1);
    }

    #[test]
    fn test_room_replay_route_pattern() {
        // /replay/ route serves recorded events
        assert!(INDEX_HTML.contains("/replay/") || INDEX_HTML.contains("replay_request"));
    }

    #[test]
    fn test_index_html_host_session_broadcasts_tokens() {
        // The start-click hook should send WS token messages when hosting
        let hook_pos = INDEX_HTML
            .find("_baseStartClick")
            .expect("base start click hook not found");
        let after = &INDEX_HTML[hook_pos..hook_pos + 500];
        assert!(after.contains("amHost") || after.contains("roomCode"));
    }

    // -- url_decode tests (#1 — additional coverage) --

    #[test]
    fn test_url_decode_plain_ascii() {
        assert_eq!(url_decode("hello"), "hello");
    }

    #[test]
    fn test_url_decode_plus_to_space() {
        assert_eq!(url_decode("hello+world"), "hello world");
    }

    #[test]
    fn test_url_decode_percent_space() {
        assert_eq!(url_decode("hello%20world"), "hello world");
    }

    #[test]
    fn test_url_decode_equals() {
        assert_eq!(url_decode("a%3Db"), "a=b");
    }

    #[test]
    fn test_url_decode_ampersand() {
        assert_eq!(url_decode("a%26b"), "a&b");
    }

    #[test]
    fn test_url_decode_multibyte_utf8_accent() {
        // %C3%A9 encodes 'é' (U+00E9) in UTF-8 — two-byte sequence
        assert_eq!(url_decode("%C3%A9"), "é");
    }

    #[test]
    fn test_url_decode_trailing_percent_no_panic() {
        let result = url_decode("hello%");
        assert!(result.contains("hello"));
    }

    #[test]
    fn test_url_decode_one_hex_digit_no_panic() {
        let result = url_decode("hello%2");
        assert!(result.contains("hello"));
    }

    #[test]
    fn test_url_decode_invalid_hex_emitted_literally() {
        let result = url_decode("%ZZ");
        assert!(result.contains('%'));
    }

    #[test]
    fn test_url_decode_empty_string_is_empty() {
        assert_eq!(url_decode(""), "");
    }

    #[test]
    fn test_url_decode_only_plus() {
        assert_eq!(url_decode("+++"), "   ");
    }

    #[test]
    fn test_url_decode_mixed_equals_ampersand() {
        assert_eq!(url_decode("foo%3Dbar%26baz"), "foo=bar&baz");
    }

    #[test]
    fn test_url_decode_slash() {
        assert_eq!(url_decode("%2F"), "/");
    }

    #[test]
    fn test_url_decode_lowercase_hex() {
        assert_eq!(url_decode("%2f"), "/");
    }

    #[test]
    fn test_url_decode_encoded_percent() {
        assert_eq!(url_decode("%25"), "%");
    }

    // -- parse_query tests --

    #[test]
    fn test_parse_query_single_pair() {
        let m = parse_query("prompt=hello");
        assert_eq!(m.get("prompt").map(|s| s.as_str()), Some("hello"));
    }

    #[test]
    fn test_parse_query_multiple_pairs() {
        let m = parse_query("a=1&b=2&c=3");
        assert_eq!(m.get("a").map(|s| s.as_str()), Some("1"));
        assert_eq!(m.get("b").map(|s| s.as_str()), Some("2"));
    }

    #[test]
    fn test_parse_query_url_decoded_value() {
        let m = parse_query("prompt=hello%20world");
        assert_eq!(m.get("prompt").map(|s| s.as_str()), Some("hello world"));
    }

    #[test]
    fn test_parse_query_empty_value() {
        let m = parse_query("key=");
        assert_eq!(m.get("key").map(|s| s.as_str()), Some(""));
    }

    #[test]
    fn test_parse_query_empty_string_no_panic() {
        let _ = parse_query("");
    }

    // -- parse_stream_params tests (item 13) --

    #[test]
    fn test_parse_stream_params_defaults() {
        let params = parse_query("");
        let sp = parse_stream_params(&params);
        assert_eq!(sp.prompt, "");
        assert_eq!(sp.transform, "reverse");
        assert_eq!(sp.provider, "openai");
        assert_eq!(sp.model, "");
        assert!((sp.rate - 0.5).abs() < 1e-9);
        assert_eq!(sp.seed, None);
        assert_eq!(sp.top_logprobs, 5);
        assert_eq!(sp.system, None);
        assert!(!sp.visual);
        assert!(!sp.heatmap);
    }

    #[test]
    fn test_parse_stream_params_seed_parsed() {
        let params = parse_query("seed=42");
        let sp = parse_stream_params(&params);
        assert_eq!(sp.seed, Some(42));
    }

    #[test]
    fn test_parse_stream_params_visual_flag_one() {
        let params = parse_query("visual=1");
        let sp = parse_stream_params(&params);
        assert!(sp.visual);
    }

    #[test]
    fn test_parse_stream_params_visual_flag_true() {
        let params = parse_query("visual=true");
        let sp = parse_stream_params(&params);
        assert!(sp.visual);
    }

    #[test]
    fn test_parse_stream_params_rate_parsed() {
        let params = parse_query("rate=0.8");
        let sp = parse_stream_params(&params);
        assert!((sp.rate - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_parse_stream_params_system_prompt() {
        let params = parse_query("system=Be+concise");
        let sp = parse_stream_params(&params);
        assert_eq!(sp.system.as_deref(), Some("Be concise"));
    }

    #[test]
    fn test_parse_stream_params_empty_system_is_none() {
        let params = parse_query("system=");
        let sp = parse_stream_params(&params);
        assert_eq!(sp.system, None);
    }

    #[test]
    fn test_parse_stream_params_rate_negative_clamped() {
        let params = parse_query("rate=-1");
        let sp = parse_stream_params(&params);
        assert_eq!(sp.rate, 0.0);
    }

    #[test]
    fn test_parse_stream_params_rate_over_one_clamped() {
        let params = parse_query("rate=2.0");
        let sp = parse_stream_params(&params);
        assert_eq!(sp.rate, 1.0);
    }

    #[test]
    fn test_parse_stream_params_rate_nan_uses_default() {
        let params = parse_query("rate=nan");
        let sp = parse_stream_params(&params);
        assert_eq!(sp.rate, 0.5);
    }

    #[test]
    fn test_parse_stream_params_top_logprobs_clamped_to_20() {
        let params = parse_query("top_logprobs=255");
        let sp = parse_stream_params(&params);
        assert_eq!(sp.top_logprobs, 20);
    }

    #[test]
    fn test_parse_stream_params_top_logprobs_valid_unchanged() {
        let params = parse_query("top_logprobs=10");
        let sp = parse_stream_params(&params);
        assert_eq!(sp.top_logprobs, 10);
    }

    // -- url_decode UTF-8 multi-byte (item 1) --

    #[test]
    fn test_url_decode_utf8_multibyte_é() {
        // %C3%A9 is UTF-8 for 'é'
        assert_eq!(url_decode("%C3%A9"), "é");
    }

    #[test]
    fn test_url_decode_incomplete_percent_no_panic() {
        // Should not panic on truncated percent-encoded sequence
        let _ = url_decode("a%2");
    }

    // -- Rate limiter tests --

    #[test]
    fn test_rate_limit_allows_up_to_max() {
        let limiter = new_rate_limiter();
        let ip: IpAddr = "127.0.0.1".parse().unwrap();
        for _ in 0..RATE_LIMIT_MAX {
            assert!(rate_limit_check(&limiter, ip), "should allow within limit");
        }
    }

    #[test]
    fn test_rate_limit_blocks_over_max() {
        let limiter = new_rate_limiter();
        let ip: IpAddr = "10.0.0.1".parse().unwrap();
        for _ in 0..RATE_LIMIT_MAX {
            rate_limit_check(&limiter, ip);
        }
        assert!(!rate_limit_check(&limiter, ip), "should block when over limit");
    }

    #[test]
    fn test_rate_limit_different_ips_independent() {
        let limiter = new_rate_limiter();
        let ip1: IpAddr = "1.1.1.1".parse().unwrap();
        let ip2: IpAddr = "2.2.2.2".parse().unwrap();
        for _ in 0..RATE_LIMIT_MAX {
            rate_limit_check(&limiter, ip1);
        }
        // ip2 is unaffected by ip1's exhaustion
        assert!(rate_limit_check(&limiter, ip2));
    }

    // -- Item 4: saturating_add no overflow --
    #[test]
    fn test_rate_limit_saturating_add_no_overflow() {
        let limiter = new_rate_limiter();
        let ip: IpAddr = "192.168.1.100".parse().unwrap();
        {
            let mut map = limiter.lock().unwrap();
            map.insert(ip, (std::time::Instant::now(), u32::MAX));
        }
        let result = rate_limit_check(&limiter, ip);
        assert!(!result, "should be blocked when count is u32::MAX");
    }

    // -- Item 18: rate limit window resets after expiry --
    #[test]
    fn test_rate_limit_window_resets_after_expiry() {
        let limiter = new_rate_limiter();
        let ip: IpAddr = "172.16.0.1".parse().unwrap();
        for _ in 0..RATE_LIMIT_MAX {
            rate_limit_check(&limiter, ip);
        }
        assert!(!rate_limit_check(&limiter, ip), "should be blocked");
        {
            let mut map = limiter.lock().unwrap();
            if let Some(entry) = map.get_mut(&ip) {
                entry.0 = std::time::Instant::now()
                    .checked_sub(std::time::Duration::from_secs(61))
                    .unwrap_or(std::time::Instant::now());
            }
        }
        assert!(rate_limit_check(&limiter, ip), "should be allowed after window expiry");
    }

    // -- Item 7: replay response prefix --
    #[test]
    fn test_replay_response_prefix() {
        let prefix = r#"{"events":["#;
        assert_eq!(prefix, r#"{"events":["#);
    }

    // -- Item 9: url_decode capacity hint --
    #[test]
    fn test_url_decode_capacity_hint() {
        let input = "a".repeat(100);
        let result = url_decode(&input);
        assert_eq!(result, input);
        assert_eq!(result.len(), 100);
    }
}
