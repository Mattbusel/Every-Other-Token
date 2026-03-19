# every-other-token

[![CI](https://github.com/Mattbusel/Every-Other-Token/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/Every-Other-Token/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/Mattbusel/Every-Other-Token/branch/main/graph/badge.svg)](https://codecov.io/gh/Mattbusel/Every-Other-Token)
[![crates.io](https://img.shields.io/crates/v/every-other-token.svg)](https://crates.io/crates/every-other-token)
[![docs.rs](https://docs.rs/every-other-token/badge.svg)](https://docs.rs/every-other-token)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![GitHub Stars](https://img.shields.io/github/stars/Mattbusel/Every-Other-Token?style=social)](https://github.com/Mattbusel/Every-Other-Token)
[![API Reference](https://img.shields.io/badge/docs-api%20reference-blue)](docs/api.md)

A real-time LLM token stream interceptor for interpretability research. Sits between your application and the model, mutates tokens mid-generation, captures per-token confidence and perplexity signals, and renders results in a zero-dependency terminal or web UI.

---

## What it does

Aggregate benchmarks measure final outputs. `every-other-token` measures what happens **during** generation — token by token, position by position — with confidence scores, perplexity signals, and cross-provider structural comparison running simultaneously.

It directly enables four research directions:

1. **Semantic fragility** — At what perturbation rate does coherent reasoning collapse?
2. **Cross-provider divergence** — Do OpenAI and Anthropic produce structurally different token sequences for identical prompts?
3. **System prompt sensitivity** — How much does framing shift per-token confidence distributions?
4. **Chaos resilience** — Do models self-correct when every other token is randomly mutated?

---

## How it works

The tool intercepts the SSE (server-sent events) stream produced by the provider API. Each chunk is parsed into individual tokens. For every token a decision is made — based on a Bresenham-spread rate schedule — whether to apply the active transform. The enriched event is then routed to the terminal renderer, web UI, WebSocket collaboration room, JSON-stream output, or replay recorder simultaneously.

### Pipeline overview

```
CLI (clap) ──► main.rs
                  │
                  ▼
           TokenInterceptor (lib.rs)
           │              │
           ▼              ▼
    OpenAiPlugin    AnthropicPlugin
    (SSE + logprobs) (SSE)
           │
           ▼  per token
  process_content_logprob()
  ├── confidence  = exp(logprob)
  ├── perplexity  = exp(-logprob)
  ├── alternatives = top_logprobs[0..N]
  └── transform   = Transform::apply(token)
           │
           ├──► web_tx  ──► SSE ──► browser (web.rs)
           ├──► collab  ──► WebSocket ──► room participants (collab.rs)
           ├──► stdout  ──► terminal renderer (render.rs)
           ├──► Recorder ──► JSON replay file (replay.rs)
           └──► HeatmapExporter ──► CSV (heatmap.rs)

Research mode (research.rs)
  run_research() × N ──► ResearchOutput (JSON)
  run_research_suite() ──► batch over prompt file

Self-tune (feature = "self-tune")
  TelemetryBus ──► AnomalyDetector ──► TuningController ──► SnapshotStore

Self-modify (feature = "self-modify")
  TaskGen ──► ValidationGate ──► Deploy ──► Memory
```

### Key modules

| Module | Responsibility |
|--------|----------------|
| `lib.rs` | `TokenInterceptor`, `TokenEvent`, stream parsing, retry logic |
| `transforms.rs` | All transform strategies (`Reverse`, `Noise`, `Chaos`, `Chain`, …) |
| `providers.rs` | `ProviderPlugin` trait, OpenAI and Anthropic SSE wire types, MCP types |
| `web.rs` | Embedded HTTP/1.1 server, SSE fan-out, WebSocket upgrade |
| `collab.rs` | Room store, participant management, surgery edits, chat, recording |
| `research.rs` | Headless research loop, aggregate statistics, A/B mode, heatmap export |
| `store.rs` | SQLite-backed experiment persistence, cross-session dedup cache |
| `heatmap.rs` | Per-position confidence matrix → CSV |
| `replay.rs` | JSON recording and deterministic replay |
| `render.rs` | Terminal ANSI colouring, confidence indicators, visual-mode formatting |
| `config.rs` | `~/.eot.toml` / `./.eot.toml` config file with merge semantics |
| `cli.rs` | Clap argument definitions and helper functions |
| `error.rs` | `EotError` enum — one variant per failure domain |

---

## Feature flags

| Flag | Default | Description |
|------|---------|-------------|
| *(none)* | ✓ | Terminal and web UI streaming, all transforms, research mode, collab rooms |
| `sqlite-log` | | Persist experiment runs to a local SQLite database via `store::ExperimentStore` |
| `self-tune` | | Background PID-based parameter tuning loop + telemetry bus |
| `self-modify` | | Agent loop for automated pipeline improvement (requires `self-tune`) |
| `intelligence` | | Reserved namespace for future interpretability features |
| `evolution` | | Reserved namespace for evolutionary optimisation |
| `helix-bridge` | | HTTP bridge that polls HelixRouter `/api/stats` and pushes config patches |
| `redis-backing` | | Write-through Redis persistence for agent memory and snapshots |
| `wasm` | | WASM target bindings via `wasm-bindgen` |

---

## Quickstart

### Prerequisites

- Rust 1.75 or later
- An OpenAI API key (`OPENAI_API_KEY`) and/or Anthropic API key (`ANTHROPIC_API_KEY`)

```bash
git clone https://github.com/Mattbusel/Every-Other-Token
cd Every-Other-Token
cargo build --release

export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Basic usage

```bash
# Terminal output with per-token confidence color bands
./target/release/every-other-token "What is consciousness?" --visual

# Web UI on http://localhost:8888 — opens browser automatically
./target/release/every-other-token "What is consciousness?" --web

# Headless research: 20 runs, JSON aggregate stats
./target/release/every-other-token "Explain recursion" \
    --research --runs 20 --output results.json

# Side-by-side OpenAI vs Anthropic diff in the terminal
./target/release/every-other-token "Describe entropy" --diff-terminal

# A/B system-prompt experiment with significance testing
./target/release/every-other-token "Tell me a story" \
    --research --runs 20 \
    --system-a "Be poetic." --system-b "Be literal." \
    --significance
```

### Shell completions

```bash
./target/release/every-other-token --completions bash >> ~/.bash_completion
./target/release/every-other-token --completions zsh  >  ~/.zfunc/_every-other-token
./target/release/every-other-token --completions fish > ~/.config/fish/completions/every-other-token.fish
```

### Dry run (no API key required)

```bash
./target/release/every-other-token "hello" --dry-run --transform chaos
```

---

## Transforms

| Name | Behavior | Deterministic? |
|------|----------|----------------|
| `reverse` | Reverses token characters: `"hello"` → `"olleh"` | Yes |
| `uppercase` | Converts to uppercase: `"hello"` → `"HELLO"` | Yes |
| `mock` | Alternating lower/upper per char: `"hello"` → `"hElLo"` | Yes |
| `noise` | Appends a random symbol from `* + ~ @ # $ %` | No (seeded with `--seed`) |
| `chaos` | Randomly selects one of the above per token | No (seeded with `--seed`) |
| `scramble` | Fisher-Yates shuffles token characters | No (seeded with `--seed`) |
| `delete` | Replaces the token with the empty string | Yes |
| `synonym` | Substitutes from a 200-entry static synonym table | Yes |
| `delay:N` | Passes through after N ms pause | Yes |
| `A,B,...` | Chain: applies A, then B, then … in sequence | Depends on chain |

**Rate control** — `--rate 0.5` (default): every other token is transformed.
Uses a Bresenham spread for deterministic, uniform distribution at any rate.
Combine with `--seed` for fully reproducible runs.

**Stochastic rate** — `--rate-range 0.3-0.7` picks a random rate in [min, max] per run.

**Confidence gating** — `--min-confidence 0.8` only transforms tokens whose API confidence is below 0.8. High-confidence tokens pass through unchanged.

---

## Web UI modes

Launch with `--web` to open the single-page application:

| Mode | Description |
|------|-------------|
| **Single** | Live token stream with per-token confidence bars and perplexity pulse |
| **Split** | Original vs transformed side by side |
| **Quad** | All four transforms applied simultaneously in a 2×2 grid |
| **Diff** | OpenAI and Anthropic streaming the same prompt in parallel, diverging positions highlighted |
| **Experiment** | A/B mode: two system prompts, same user prompt, live divergence map |
| **Research** | Aggregate stats dashboard: perplexity histogram, confidence distribution, vocab diversity |

---

## Configuration file

Create `~/.eot.toml` (global) or `.eot.toml` in the working directory (local wins over global):

```toml
provider     = "anthropic"
model        = "claude-sonnet-4-6"
transform    = "reverse"
rate         = 0.5
port         = 8888
top_logprobs = 5
system_a     = "You are a concise assistant."
```

CLI flags override config file values. The `rate` field is clamped to `[0.0, 1.0]` with a warning if out of range.

---

## CLI reference

```
USAGE:
    every-other-token [OPTIONS] <PROMPT> [TRANSFORM] [MODEL]

ARGS:
    <PROMPT>      Input prompt (use "-" to read from stdin)
    [TRANSFORM]   Transform type [default: reverse]
    [MODEL]       Model name [default: gpt-3.5-turbo]

OPTIONS:
    --provider <PROVIDER>           openai | anthropic | mock [default: openai]
    --visual, -v                    Enable ANSI confidence-colored output
    --heatmap                       Enable token importance heatmap
    --orchestrator                  Route through MCP pipeline at localhost:3000
    --web                           Launch web UI instead of terminal
    --port <PORT>                   Web UI port [default: 8888]
    --research                      Headless research mode
    --runs <RUNS>                   Number of research runs [default: 10]
    --output <FILE>                 Research output JSON path [default: research_output.json]
    --system-a <PROMPT>             System prompt A (A/B mode)
    --system-b <PROMPT>             System prompt B (A/B mode)
    --top-logprobs <N>              Top alternative tokens per position (0–20) [default: 5]
    --db <FILE>                     SQLite database for experiment persistence
    --significance                  Compute Welch's t-test across A/B confidence distributions
    --heatmap-export <FILE>         Export per-position confidence heatmap to CSV
    --heatmap-min-confidence <F>    Minimum mean confidence for heatmap rows [default: 0.0]
    --heatmap-sort-by <FIELD>       Sort heatmap by "position" or "confidence" [default: position]
    --record <FILE>                 Record token events to JSON replay file
    --replay <FILE>                 Replay token events from file (no API call)
    --rate <F>                      Fraction of tokens to transform (0.0–1.0) [default: 0.5]
    --rate-range <MIN-MAX>          Stochastic rate from interval (e.g. "0.3-0.7")
    --seed <N>                      Fixed RNG seed for reproducible Noise/Chaos runs
    --baseline                      Compare against stored "none" transform runs in SQLite
    --prompt-file <FILE>            Batch research: one prompt per line
    --diff-terminal                 Parallel OpenAI + Anthropic streams side by side
    --json-stream                   One JSON line per token to stdout
    --dry-run                       Validate transform without calling any API
    --template <TPL>                Prompt template with {input} placeholder
    --min-confidence <F>            Only transform tokens with confidence below this value
    --format <FMT>                  Research output format: "json" or "jsonl" [default: json]
    --collapse-window <N>           Confidence collapse detection window [default: 5]
    --orchestrator-url <URL>        MCP orchestrator base URL [default: http://localhost:3000]
    --max-retries <N>               API retry attempts on 429/5xx [default: 3]
    --completions <SHELL>           Generate shell completions (bash/zsh/fish/…)
    --log-db <FILE>                 SQLite experiment log (requires sqlite-log feature)
```

---

## API reference (library)

Add to `Cargo.toml`:

```toml
[dependencies]
every-other-token = "4"
```

### Core types

```rust
use every_other_token::{TokenInterceptor, TokenEvent};
use every_other_token::providers::Provider;
use every_other_token::transforms::Transform;

let mut interceptor = TokenInterceptor::new(
    Provider::Openai,
    Transform::Reverse,
    "gpt-4".to_string(),
    false,  // visual_mode
    false,  // heatmap_mode
    false,  // orchestrator
)?
.with_rate(0.5)
.with_seed(42);

interceptor.intercept_stream("What is entropy?").await?;
```

### Web UI / channel mode

```rust
use every_other_token::{TokenInterceptor, TokenEvent};
use tokio::sync::mpsc;

let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
interceptor.web_tx = Some(tx);
interceptor.intercept_stream("Explain recursion").await?;

while let Some(event) = rx.recv().await {
    println!("{}: {:?}", event.index, event.text);
}
```

### Experiment store

```rust
use every_other_token::store::{ExperimentStore, RunRecord};

let store = ExperimentStore::open("experiments.db")?;
let id = store.insert_experiment("2026-01-01", "my prompt", "openai", "reverse", "gpt-4")?;
store.insert_run(id, &RunRecord {
    run_index: 0,
    token_count: 100,
    transformed_count: 50,
    avg_confidence: Some(0.82),
    avg_perplexity: Some(1.2),
    vocab_diversity: 0.73,
})?;
```

---

## Performance

- Sub-millisecond per-token processing overhead (Bresenham spread, no heap allocation per token)
- Zero-copy async streaming via Tokio with back-pressure on the broadcast channel
- ~4 MB release binary with LTO + `strip = true`
- Parallel provider streams via `tokio::select!` / `tokio::join!`
- Exponential back-off retry on 429 / 5xx responses (up to `--max-retries` attempts)

---

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository and create a feature branch from `main`.
2. Run `cargo fmt` and `cargo clippy -- -D warnings` before committing.
3. Add tests for any new public API surface. The CI gate requires all tests to pass on stable and the MSRV (1.75).
4. Open a pull request against `main` with a clear description of the change and why it is needed.
5. For significant changes, open an issue first to discuss the design.

### Development commands

```bash
# Build
cargo build

# Tests (all feature combinations)
cargo test
cargo test --features sqlite-log
cargo test --features self-tune
cargo test --features self-modify
cargo test --features helix-bridge

# Lint
cargo clippy -- -D warnings
cargo clippy --all-features -- -D warnings

# Format check
cargo fmt --check

# Docs (with warnings-as-errors)
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --open

# Security audit
cargo audit

# Dependency policy check
cargo deny check

# Release build
cargo build --release
```

### Project layout

```
src/
├── lib.rs              # TokenInterceptor, TokenEvent, stream loop
├── main.rs             # CLI entry point
├── cli.rs              # Clap argument struct + helpers
├── config.rs           # .eot.toml config file support
├── error.rs            # EotError enum
├── providers.rs        # ProviderPlugin trait + wire types
├── transforms.rs       # Transform enum + all strategies
├── render.rs           # Terminal ANSI rendering helpers
├── web.rs              # Embedded HTTP/WebSocket server
├── collab.rs           # Multiplayer room management
├── research.rs         # Headless research loop + stats
├── store.rs            # SQLite experiment persistence
├── heatmap.rs          # Per-position confidence CSV export
├── replay.rs           # Token event recording + replay
├── self_tune/          # (feature: self-tune) PID tuning loop
├── self_modify/        # (feature: self-modify) Agent improvement loop
├── helix_bridge/       # (feature: helix-bridge) HelixRouter HTTP bridge
├── semantic_dedup.rs   # (feature: self-modify) In-session prompt dedup
└── experiment_log.rs   # (feature: sqlite-log) SQLite experiment logger
tests/
├── collab_tests.rs
├── providers_tests.rs
├── transforms_tests.rs
├── store_heatmap_replay_tests.rs
├── self_tune_integration.rs
└── web_integration.rs
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Ecosystem

- [tokio-prompt-orchestrator](https://github.com/Mattbusel/tokio-prompt-orchestrator) — the orchestration layer that uses Every-Other-Token telemetry to drive HelixRouter adaptation
- [LLM-Hallucination-Detection-Script](https://github.com/Mattbusel/LLM-Hallucination-Detection-Script) — companion tool for output-level reliability analysis
- [Token-Visualizer](https://github.com/Mattbusel/Token-Visualizer) — interactive tokenizer and prompt-engineering visualization tool

---

## Troubleshooting

**"Missing API key"** — Set the `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` environment variable before running. Use `export OPENAI_API_KEY=sk-...` (Linux/macOS) or `set OPENAI_API_KEY=sk-...` (Windows).

**"Model not found"** — Check the model name spelling; valid examples are `gpt-4o`, `gpt-3.5-turbo`, `claude-sonnet-4-6`. Use `--dry-run` to test transform logic without making any API call.

**"Rate limited (429)"** — The built-in circuit breaker will retry automatically after a backoff. If the circuit opens (5 consecutive failures), it stays open for 30 seconds then resets. Wait 30 s and retry, or reduce request frequency.

**"Stream times out"** — Try a shorter prompt or a faster model. Increase the timeout with `--timeout 300` (seconds) if the model is legitimately slow. Default timeout is 120 s.

**"Web UI blank"** — Open the browser developer console (F12) and check for errors. Common causes: `--provider` does not match the API key that is set, the server is not running on the expected port, or a browser extension is blocking the SSE connection.

**"Port already in use"** — Use `--port 9000` (or any free port) to pick a different port. Find the occupying process with `lsof -i :<port>` (Linux/macOS) or `netstat -ano | findstr :<port>` (Windows).
