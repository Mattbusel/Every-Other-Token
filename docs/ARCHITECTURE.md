# Architecture

## Overview
Every-Other-Token (EOT) is a Rust CLI + embedded web UI for real-time LLM token stream mutation and interpretability research. It intercepts streaming API responses, applies configurable transforms to tokens at Bresenham-spread positions, and fans out enriched token events to terminal output, a web UI, or JSON streams.

## Why raw Tokio HTTP instead of Axum/Actix?
The server speaks raw HTTP/1.1 via a hand-rolled async loop over a `TcpListener`. This keeps the binary small (no web framework dependency), eliminates framework version churn, and gives full control over SSE framing and WebSocket upgrade detection. The trade-off is more boilerplate in `web.rs`.

## Why a single embedded HTML file?
The web UI is a single `include_str!("../static/index.html")` with no build step. This means:
- Zero npm/webpack dependencies
- Single binary deployment (no static file serving)
- Instant iteration (edit HTML, restart server)
The trade-off is that the file grows large and lacks module boundaries.

## Module map
| Module | Responsibility |
|--------|---------------|
| `main.rs` | CLI entry point, wires Args → TokenInterceptor |
| `lib.rs` | `TokenInterceptor`, `TokenEvent`, circuit breaker, streaming engine (SSE parsed by `eventsource-stream`, retries by `backon`) |
| `web.rs` | Raw HTTP server, SSE/WS routing, rate limiting |
| `cli.rs` | `clap`-derived `Args` struct |
| `config.rs` | TOML file config, merge precedence |
| `providers.rs` | Provider list, endpoints, keys and wire types (OpenAI-compatible: OpenAI, Ollama, OpenRouter, Gemini; plus Anthropic and Mock) |
| `tui.rs` | Full-screen terminal view (`--tui`), built on `ratatui` |
| `transforms.rs` | Token mutation strategies (Reverse, Uppercase, Chaos, ...) |
| `collab.rs` | Multiplayer room state, WebSocket handling |
| `research.rs` | Headless N-run batch mode, statistics (`statrs`), CSV and JSONL exports (`csv`) |
| `render.rs` | ANSI colour rendering, confidence bands |

## Configuration precedence
`hard-coded defaults` → `~/.eot.toml` → `./.eot.toml` → `CLI flags` → `query-string params`

## Streaming pipeline
```
Provider API (SSE)
    |
TokenInterceptor.intercept_stream()
    | per token
Bresenham spread check  ->  Transform::apply_with_label()
    |
TokenEvent { text, original, confidence, perplexity, alternatives, ... }
    | fan-out via mpsc::UnboundedSender
+------------+--------------+---------------+------------+
| Terminal   |  Web SSE     |  JSON stream  |  --tui     |
| ANSI out   |  /stream     |  stdout       |  ratatui   |
+------------+--------------+---------------+------------+
```

## Retries and circuit breaker
Requests that get 429, 500, 502, 503 or 529 back, or fail on the network, are retried by `backon` with exponential back-off (800 ms doubling, capped at 30 s) and jitter. A `Retry-After` or `retry-after-ms` header replaces the computed delay.

After 5 consecutive API failures, the circuit opens for 30 seconds, rejecting all calls immediately. A single success resets the counter. Rate limits (429) do not count as failures.

## Feature flags
See `docs/features.md` for the full matrix.
