# every-other-token

[![CI](https://github.com/Mattbusel/Every-Other-Token/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/Every-Other-Token/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/Mattbusel/Every-Other-Token/branch/main/graph/badge.svg)](https://codecov.io/gh/Mattbusel/Every-Other-Token)
[![crates.io](https://img.shields.io/crates/v/every-other-token.svg)](https://crates.io/crates/every-other-token)
[![docs.rs](https://docs.rs/every-other-token/badge.svg)](https://docs.rs/every-other-token)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![GitHub Stars](https://img.shields.io/github/stars/Mattbusel/Every-Other-Token?style=social)](https://github.com/Mattbusel/Every-Other-Token)

A real-time LLM token stream interceptor for interpretability research. Sits between your application and the model, mutates tokens mid-generation, captures per-token confidence and perplexity signals, and renders results in a zero-dependency terminal or web UI.

---

## Why this exists

Aggregate benchmarks measure final outputs. This tool measures what happens **during** generation -- token by token, position by position -- with confidence scores, perplexity signals, and cross-provider structural comparison running simultaneously.

It directly enables four research directions:

1. **Semantic fragility** -- At what perturbation rate does coherent reasoning collapse?
2. **Cross-provider divergence** -- Do OpenAI and Anthropic produce structurally different token sequences for identical prompts?
3. **System prompt sensitivity** -- How much does framing shift per-token confidence distributions?
4. **Chaos resilience** -- Do models self-correct when every other token is randomly mutated?

---

## Feature table

| Feature flag | Description |
|---|---|
| *(default)* | Terminal and web UI streaming, all transforms, research mode |
| `sqlite-log` | Persist experiment runs to a local SQLite database |
| `self-tune` | Background PID-based parameter tuning loop |
| `self-modify` | Agent loop for automated pipeline improvement |
| `helix-bridge` | HTTP bridge that polls HelixRouter `/api/stats` |
| `redis-backing` | Redis-backed persistence for agent memory and snapshots |

---

## Quickstart

```bash
git clone https://github.com/Mattbusel/Every-Other-Token
cd Every-Other-Token
cargo build --release

export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Terminal output with per-token confidence colors
./target/release/every-other-token "What is consciousness?" --visual

# Web UI on http://localhost:8888
./target/release/every-other-token "What is consciousness?" --web

# Headless research: 20 runs, JSON aggregate stats
./target/release/every-other-token "Explain recursion" \
    --research --runs 20 --output results.json

# Side-by-side OpenAI vs Anthropic diff in the terminal
./target/release/every-other-token "Describe entropy" --diff-terminal

# A/B system-prompt experiment
./target/release/every-other-token "Tell me a story" \
    --research --runs 10 \
    --system-a "Be poetic." --system-b "Be literal."
```

### Shell completions

```bash
./target/release/every-other-token --completions bash >> ~/.bash_completion
./target/release/every-other-token --completions zsh  >  ~/.zfunc/_every-other-token
```

---

## Transforms

| Name | CLI flag | Behavior |
|---|---|---|
| `reverse` | `reverse` | Reverses the characters of intercepted tokens |
| `uppercase` | `uppercase` | Converts intercepted tokens to uppercase |
| `mock` | `mock` | Alternating lower/upper case per character |
| `noise` | `noise` | Appends a random symbol (`* + ~ @ # $ %`) |
| `chaos` | `chaos` | Randomly selects one of the above per token |
| `scramble` | `scramble` | Fisher-Yates shuffles the token's characters |
| `delete` | `delete` | Replaces intercepted tokens with the empty string |
| `synonym` | `synonym` | Substitutes from a 200-entry static synonym table |
| `delay:N` | `delay:N` | Passes the token through after N ms pause |
| *chain* | `A,B` | Applies A then B in sequence (e.g. `reverse,uppercase`) |

The `--rate` flag controls the fraction of tokens intercepted (default 0.5, i.e. every other token). A Bresenham spread ensures uniform distribution. Combine with `--seed` for reproducible results.

---

## Web UI modes

| Mode | Description |
|---|---|
| **Single** | Live token stream with per-token confidence bars and perplexity pulse |
| **Split** | Original vs transformed side by side |
| **Quad** | All four transforms applied simultaneously in a 2x2 grid |
| **Diff** | OpenAI and Anthropic streaming the same prompt in parallel, diverging positions highlighted |
| **Experiment** | A/B mode: two system prompts, same user prompt, live divergence map |
| **Research** | Aggregate stats dashboard: perplexity histogram, confidence distribution, vocab diversity |

---

## Architecture

```
CLI (clap) --> main.rs
                |
                v
         TokenInterceptor
         |              |
         v              v
   OpenAiPlugin   AnthropicPlugin
   (SSE + logprobs)   (SSE)
         |
         v  (per token)
  process_content_with_logprob()
  |-- confidence  = exp(logprob)
  |-- perplexity  = exp(-logprob)
  |-- alternatives = top_logprobs[0..N]
  |-- transform   = Transform::apply(token)
         |
         +--> web_tx  --> SSE --> browser (web.rs)
         +--> collab  --> WebSocket --> room participants (collab.rs)
         +--> stdout  --> terminal renderer (render.rs)
         +--> Recorder --> replay file (replay.rs)
         +--> HeatmapExporter --> CSV (heatmap.rs)

Research mode (research.rs)
  run_research() x N --> ResearchOutput (JSON)
  run_research_suite() --> batch over prompt file

Self-tune (feature = "self-tune")
  TelemetryBus --> AnomalyDetector --> TuningController --> SnapshotStore

Self-modify (feature = "self-modify")
  TaskGen --> ValidationGate --> Deploy --> Memory
```

---

## Configuration file

Create `~/.eot.toml` or `./.eot.toml` (local wins):

```toml
provider     = "anthropic"
model        = "claude-sonnet-4-6"
transform    = "reverse"
rate         = 0.5
port         = 8888
top_logprobs = 5
```

---

## Performance

- Sub-millisecond per-token processing overhead
- Zero-copy async streaming via Tokio
- 4 MB release binary with LTO
- Parallel provider streams via `tokio::select!`

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

# Tests
cargo test
cargo test --features sqlite-log

# Lint
cargo clippy -- -D warnings
cargo clippy --features sqlite-log -- -D warnings

# Format check
cargo fmt --check

# Docs
cargo doc --no-deps --open

# Dependency audit
cargo audit
```

---

## License

MIT -- see [LICENSE](LICENSE).

---

## Ecosystem

- [tokio-prompt-orchestrator](https://github.com/Mattbusel/tokio-prompt-orchestrator) -- the orchestration layer that uses Every-Other-Token telemetry to drive HelixRouter adaptation
- [LLM-Hallucination-Detection-Script](https://github.com/Mattbusel/LLM-Hallucination-Detection-Script) -- companion tool for output-level reliability analysis
- [Token-Visualizer](https://github.com/Mattbusel/Token-Visualizer) -- interactive tokenizer and prompt-engineering visualization tool

---

See [CHANGELOG.md](CHANGELOG.md) for release history and migration notes.
