# Every Other Token

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-393%20passing-brightgreen.svg)](https://github.com/Mattbusel/Every-Other-Token)
[![GitHub Stars](https://img.shields.io/github/stars/Mattbusel/Every-Other-Token?style=social)](https://github.com/Mattbusel/Every-Other-Token)

**Token-level perturbation, probabilistic analysis, and real-time collaboration for live LLM streams.**

LLMs produce tokens. Every Other Token intercepts them mid-stream, applies transforms, measures confidence and perplexity at each position, and surfaces the results — in the terminal, a zero-dependency web UI, or to multiple collaborators simultaneously. The core question: *how robust, confident, and reproducible is language model reasoning at the token level?*

Built in Rust. Dual-provider. 393 tests. Ships as a single binary.

---

## Why This Matters

Token-level analysis is an underexplored axis of LLM evaluation. Aggregate benchmarks measure outputs. This tool measures what happens *during* generation — token by token, position by position — with confidence scores, perplexity signals, and cross-provider structural comparison running simultaneously.

**Four open research questions this directly enables:**

1. **Semantic fragility** — At what perturbation rate does coherent reasoning collapse?
2. **Cross-provider divergence** — Do OpenAI and Anthropic produce structurally different token sequences for identical prompts?
3. **System prompt sensitivity** — How much does framing shift per-token confidence distributions?
4. **Chaos resilience** — Do models self-correct when every other token is randomly mutated?

---

## Recent additions

**Self-Improving Loop** — `SelfImprovementOrchestrator` closes the feedback cycle: telemetry signals from token streams feed a PID controller that adjusts pipeline parameters in real time. Configuration snapshots are Redis-backed so best-performing parameter sets survive restarts.

**Redis-backed persistence** — `RedisSnapshotStore` and `RedisMemory` replace the in-memory stores. Experiment results, anomaly history, and agent modification records now survive process restarts and are accessible across instances.

**HelixRouter bridge** — `helix_bridge` connects Every-Other-Token's token stream output to HelixRouter's routing engine. Token throughput and perplexity signals are forwarded as pressure metrics; HelixRouter can shed load or shift strategy based on live stream velocity.

**SemanticDedup** — Embedding-based deduplication layer collapses semantically redundant prompts before they hit the provider API, complementing the exact-match dedup already in tokio-prompt-orchestrator.

**Staged deployment pipeline** — Proposed configuration changes from the self-tune loop run through a blue-green canary pipeline (5% → 25% → 100%) before full promotion. Automatic rollback on quality regression.

---

## Quick Start

```bash
git clone https://github.com/Mattbusel/Every-Other-Token
cd Every-Other-Token
cargo build --release

export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."   # required for Diff mode

# Terminal
cargo run -- "What is consciousness?" --visual

# Web UI — opens http://localhost:8888
cargo run -- "What is consciousness?" --web

# Headless research — 20 runs, aggregate stats to JSON
cargo run -- "Explain recursion" --research --runs 20 --output results.json

# Route through tokio-prompt-orchestrator MCP pipeline
cargo run -- "prompt" --orchestrator
```

---

## Web UI

`--web` launches a local server with an embedded single-page application. No npm. No webpack. No build step.

### View Modes

| Mode | Description |
|------|-------------|
| **Single** | Live token stream with per-token confidence bars and perplexity pulse |
| **Split** | Original vs transformed side by side |
| **Quad** | All four transforms applied simultaneously in a 2×2 grid |
| **Diff** | OpenAI and Anthropic streaming the same prompt in parallel — diverging positions highlighted red |
| **Experiment** | A/B mode: two system prompts, same user prompt, live divergence map |
| **Research** | Aggregate stats dashboard — perplexity, confidence histogram, vocab diversity, cost |

---

## Research Features

### Token Probability Scores

Every token from OpenAI carries a **confidence score** (exp(logprob), range 0–1) and **top-5 alternative completions** at that position. Color-coded per token: green ≥70%, yellow 40–70%, red <40%. All alternatives included in Export JSON.

### Perplexity Meter

Per-token **perplexity** (exp(-logprob)) streams live. High-perplexity tokens pulse in the UI. A 60-token rolling sparkline shows the model's uncertainty trajectory.

### A/B Experiment Mode

Fire one user prompt to the same provider under **two different system prompts**. Tokens stream into parallel columns. Live divergence map: green if identical, red if divergent. Final similarity score at stream end.

### Headless Research Mode

```bash
cargo run -- "Explain the halting problem" --research --runs 50 --output halting.json
cargo run -- "Tell me a story" --research --runs 10 --system-a "Be poetic." --system-b "Be literal." --output ab.json
```

### Real-Time Multiplayer Collaboration

Multiple researchers observe the same token stream simultaneously. Room codes, shared token surgery, session recording, in-room chat, token voting.

---

## Transforms

| Transform | Behavior |
|-----------|----------|
| `reverse` | Reverses token characters at odd positions |
| `uppercase` | Uppercases token at odd positions |
| `mock` | Alternating case (`hElLo`) at odd positions |
| `noise` | Appends a random symbol (`world$`) at odd positions |
| `chaos` | Randomly selects one of the above per token |

---

## Flags

| Flag | Description |
|------|-------------|
| `--provider` | `openai` (default) or `anthropic` |
| `--visual` / `-v` | Color-code even vs odd tokens |
| `--heatmap` | Token importance heatmap |
| `--web` | Launch web UI |
| `--port` | Web UI port (default: 8888) |
| `--orchestrator` | Route through tokio-prompt-orchestrator MCP pipeline |
| `--research` | Headless research mode |
| `--runs N` | Number of research runs (default: 10) |
| `--output path` | Research output file |
| `--system-a / --system-b` | System prompts for A/B experiment mode |

---

## Architecture

```
CLI args / Web request / WebSocket
          │
          ▼
   TokenInterceptor
    ├── Provider::Openai  ──→ OpenAI SSE stream (logprobs=true)
    └── Provider::Anthropic ─→ Anthropic SSE stream
          │
          ▼ (per token)
    process_content_with_logprob()
    │   ├── confidence = exp(logprob)
    │   ├── perplexity = exp(-logprob)
    │   └── alternatives = top_logprobs[0..5]
          │
          ├── web_tx channel ──→ SSE → browser
          ├── collab broadcast → WebSocket → all room participants
          ├── helix_bridge ───→ HelixRouter pressure metrics
          └── stdout (terminal mode)

SelfImprovementOrchestrator (background)
    TelemetryBus → AnomalyDetector → TuningController → RedisSnapshotStore
```

**Modules:** `cli` · `providers` · `transforms` · `lib` · `web` · `research` · `collab` · `helix_bridge` · `semantic_dedup` · `self_tune` · `self_modify`

---

## Performance

- ~10,000 tokens/sec on commodity hardware
- Sub-millisecond per-token transform + logprob overhead
- 4 MB release binary (LTO + single codegen unit)
- Zero-copy async streaming via Tokio
- 393 tests — unit, integration, property-based

---

## Investment Thesis

Every Other Token is an interpretability instrument at the only layer that matters: token generation, as it happens. The same infrastructure — real-time perturbation, confidence scoring, cross-provider comparison, multiplayer annotation — is what research labs will need to evaluate increasingly capable models systematically.

The recent additions extend this from a single-session tool to a persistent research platform: Redis-backed experiment storage, self-tuning pipeline parameters, cross-repo integration with HelixRouter for load-aware routing, and a staged deployment system for configuration changes. It's now composable with the rest of the stack.

---

## License

MIT

---

*Every Other Token is an open research instrument. If you're working on LLM evaluation, interpretability, inference infrastructure, or collaborative annotation tooling, open an issue.*
