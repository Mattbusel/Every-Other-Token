# Every Other Token

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-483%20passing-brightgreen.svg)](https://github.com/Mattbusel/Every-Other-Token)
[![GitHub Stars](https://img.shields.io/github/stars/Mattbusel/Every-Other-Token?style=social)](https://github.com/Mattbusel/Every-Other-Token)

**Token-level perturbation, probabilistic analysis, and real-time collaboration for live LLM streams.**

LLMs produce tokens. Every Other Token intercepts them mid-stream, applies transforms, measures confidence and perplexity at each position, and surfaces the results — in the terminal, a zero-dependency web UI, or to multiple collaborators simultaneously. The core question: *how robust, confident, and reproducible is language model reasoning at the token level?*

Built in Rust. Dual-provider. 483 tests. Ships as a single binary.

---

## Why This Matters

Token-level analysis is an underexplored axis of LLM evaluation. Aggregate benchmarks measure outputs. This tool measures what happens *during* generation — token by token, position by position — with confidence scores, perplexity signals, and cross-provider structural comparison running simultaneously.

**Four open research questions this directly enables:**

1. **Semantic fragility** — At what perturbation rate does coherent reasoning collapse?
2. **Cross-provider divergence** — Do OpenAI and Anthropic produce structurally different token sequences for identical prompts?
3. **System prompt sensitivity** — How much does framing shift per-token confidence distributions?
4. **Chaos resilience** — Do models self-correct when every other token is randomly mutated?

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
```

---

## Web UI

`--web` launches a local server with an embedded single-page application. No npm. No webpack. No build step.

```bash
cargo run -- "prompt" --web
cargo run -- "prompt" --web --port 9000
```

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

Every token from OpenAI carries a **confidence score** (exp(logprob), range 0–1) and **top-5 alternative completions** at that position. The web UI renders a colored underline per token:

- **Green** — confidence ≥ 70%
- **Yellow** — confidence 40–70%
- **Red** — confidence < 40%

Hover to see the confidence percentage. All alternatives are included in the Export JSON.

### Perplexity Meter

Per-token **perplexity** (exp(-logprob)) streams live. High-perplexity tokens pulse in the UI. A 60-token rolling sparkline shows the model's uncertainty trajectory across the response.

### A/B Experiment Mode

Fire one user prompt to the same provider simultaneously under **two different system prompts**. Tokens stream into parallel columns. A live divergence map compares each position: green if identical, red if divergent. Final similarity score reported at stream end.

```bash
# Via web UI: enter System Prompt A and B, click Experiment, click Stream
```

### Research Dashboard

After any stream completes, the **Research** view shows:
- Vocabulary diversity (unique/total token ratio)
- Average token length
- Average perplexity and confidence
- Token count and transform coverage
- Top 10 highest-perplexity tokens ranked
- Confidence distribution histogram
- Running cost estimate
- Copy-paste citation block

### Headless Research Mode

Run N inference passes without the UI. Aggregate stats written to structured JSON.

```bash
cargo run -- "Explain the halting problem" --research --runs 50 --output halting.json
cargo run -- "Tell me a story" --research --runs 10 --system-a "Be poetic." --system-b "Be literal." --output ab.json
```

Output schema:
```json
{
  "schema_version": 1,
  "prompt": "...",
  "provider": "openai",
  "transform": "reverse",
  "runs": [
    { "run_index": 0, "token_count": 312, "avg_confidence": 0.847, "avg_perplexity": 1.18, "vocab_diversity": 0.71 }
  ],
  "aggregate": { "mean_token_count": 298, "mean_confidence": 0.851, "mean_perplexity": 1.21, "mean_vocab_diversity": 0.69 }
}
```

### Real-Time Multiplayer Collaboration

Multiple researchers can join a shared room and observe the same token stream simultaneously.

- **Room codes** — 6-character alphanumeric, generated on creation
- **Shared token surgery** — edits broadcast instantly to all participants
- **Session recording** — capture the full event sequence for replay
- **In-room chat** — timestamped messages tied to stream position
- **Token voting** — participants can flag or endorse individual tokens

```bash
# Create a room via the web UI — share the room code with collaborators
cargo run -- "prompt" --web
```

---

## Transforms

| Transform | Behavior | Applied to |
|-----------|----------|------------|
| `reverse` | Reverses token characters | Odd positions |
| `uppercase` | Uppercases token | Odd positions |
| `mock` | Alternating case (`hElLo`) | Odd positions |
| `noise` | Appends a random symbol (`world$`) | Odd positions |
| `chaos` | Randomly selects one of the above per token; hover tooltip shows which | Odd positions |

---

## Flags

| Flag | Description |
|------|-------------|
| `--provider` | `openai` (default) or `anthropic` |
| `--visual` / `-v` | Color-code even (normal) vs odd (cyan+bold) tokens |
| `--heatmap` | Token importance heatmap (red = critical, blue = low) |
| `--web` | Launch web UI |
| `--port` | Web UI port (default: 8888) |
| `--orchestrator` | Route through tokio-prompt-orchestrator MCP pipeline |
| `--research` | Headless research mode |
| `--runs N` | Number of research runs (default: 10) |
| `--output path` | Research output file (default: research_output.json) |
| `--system-a` | System prompt A for A/B experiment mode |
| `--system-b` | System prompt B for A/B experiment mode |

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
          └── stdout (terminal mode)
```

**Diff mode** — two `TokenInterceptor` instances (OpenAI + Anthropic) feed a single merged `mpsc` channel. Events tagged by source side, forwarded as one SSE stream.

**A/B mode** — two `TokenInterceptor` instances (same provider, different system prompts) feed a merged channel. Side tagged `"a"` or `"b"`.

**Collab** — `RoomStore` (Arc<Mutex<HashMap>>) shared across all WebSocket connections. Broadcast channel per room via `tokio::sync::broadcast`. Events: token, surgery, chat, vote, stream_done, recording.

**Modules:** `cli` · `providers` · `transforms` · `lib` (interceptor core) · `web` (embedded SPA + TCP server) · `research` (headless runner) · `collab` (multiplayer rooms)

---

## Performance

- ~10,000 tokens/sec on commodity hardware
- Sub-millisecond per-token transform + logprob overhead
- 4 MB release binary (LTO + single codegen unit)
- Zero-copy async streaming via Tokio
- 483 tests — unit, integration, property-based

---

## Export

The **Export JSON** button produces a structured payload:

```json
{
  "prompt": "What is consciousness?",
  "provider": "openai",
  "transform": "chaos",
  "timestamp": "2026-02-21T...",
  "token_count": 312,
  "tokens": [
    {
      "text": "WHAT", "original": "What", "index": 1, "transformed": true,
      "importance": 0.74, "chaos_label": "uppercase",
      "confidence": 0.912, "perplexity": 1.097,
      "alternatives": [
        { "token": "what", "probability": 0.912 },
        { "token": "Why", "probability": 0.043 }
      ]
    }
  ],
  "surgery_log": [
    { "index": 4, "original": "is", "replacement": "was", "timestamp": "..." }
  ]
}
```

---

## Research Applications

- **Perturbation benchmarking** — measure coherence decay rate as transform intensity increases
- **Provider fingerprinting** — identify systematic token-sequence divergence between OpenAI and Anthropic
- **System prompt sensitivity analysis** — A/B experiment across prompt framings at the token level
- **Dataset annotation** — Token Surgery to correct or label streamed outputs in real time
- **Collaborative interpretability** — shared rooms for multi-researcher annotation sessions
- **Chaos resilience scoring** — Chaos mode across a prompt suite, score model self-correction rate
- **Pipeline observability** — attach to tokio-prompt-orchestrator for full-stack token-level tracing

---

## License

MIT

---

*Every Other Token is an open research instrument. If you're working on LLM evaluation, interpretability, inference infrastructure, or collaborative annotation tooling and want to discuss or collaborate, open an issue.*
