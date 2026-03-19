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
| `lib.rs` | `TokenInterceptor`, `TokenEvent`, circuit breaker, streaming engine |
| `web.rs` | Raw HTTP server, SSE/WS routing, rate limiting |
| `cli.rs` | `clap`-derived `Args` struct |
| `config.rs` | TOML file config, merge precedence |
| `providers.rs` | OpenAI / Anthropic / Mock HTTP backends |
| `transforms.rs` | Token mutation strategies (Reverse, Uppercase, Chaos, ...) |
| `collab.rs` | Multiplayer room state, WebSocket handling |
| `research.rs` | Headless N-run batch mode, statistics |
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
+------------+--------------+---------------+
| Terminal   |  Web SSE     |  JSON stream  |
| ANSI out   |  /stream     |  stdout       |
+------------+--------------+---------------+
```

## Circuit breaker
After 5 consecutive API failures, the circuit opens for 30 seconds, rejecting all calls immediately. A single success resets the counter.

## Feature flags
See `docs/features.md` for the full matrix.
