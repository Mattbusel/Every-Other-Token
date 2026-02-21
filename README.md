# Every Other Token

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Mattbusel/Every-Other-Token?style=social)](https://github.com/Mattbusel/Every-Other-Token)

**Token-level perturbation and comparative analysis for live LLM streams.**

LLMs produce tokens. Every Other Token intercepts them mid-stream, applies transforms, and surfaces the results — in the terminal or in a zero-dependency web UI. The core question: how robust is language model reasoning at the token level?

Built in Rust. Dual-provider. 208 tests. Ships as a single binary.

---

## Why This Matters

Token-level perturbation is an underexplored axis of LLM evaluation. Aggregate benchmarks measure outputs. This tool measures what happens *during* generation — token by token, position by position — across providers, transforms, and prompts simultaneously.

Three open research questions this directly enables:

1. **Semantic fragility** — At what perturbation rate does coherent reasoning collapse?
2. **Cross-provider divergence** — Do OpenAI and Anthropic produce structurally different token sequences for identical prompts?
3. **Chaos resilience** — Do models self-correct when every other token is randomly mutated?

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
```

---

## Web UI

`--web` launches a local server with an embedded single-page application. No npm. No webpack. No build step. One binary, one command.

```bash
cargo run -- "prompt" --web
cargo run -- "prompt" --web --port 9000
```

### View Modes

| Mode | Description |
|------|-------------|
| **Single** | Live token stream with per-token animation |
| **Split** | Original vs transformed side by side |
| **Quad** | All four transforms applied simultaneously in a 2×2 grid |
| **Diff** | OpenAI and Anthropic streaming the same prompt in parallel — diverging positions highlighted red, matching positions highlighted green |

### Diff Mode

Fires one prompt to both providers simultaneously. Tokens stream into two columns. When both providers have emitted position *N*, they are compared: green if identical, red if divergent. A live stats bar tracks divergence rate across the full stream.

Requires both `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`.

### Token Surgery

After a stream completes, every rendered token becomes an inline editor. Click any token, type a replacement, press Enter to commit (or Escape to revert). Edits are logged and included in the Export JSON payload — useful for annotating outputs or building fine-tuning datasets.

---

## Transforms

| Transform   | Behavior                               | Applied to    |
|-------------|----------------------------------------|---------------|
| `reverse`   | Reverses token characters              | Odd positions |
| `uppercase` | Uppercases token                       | Odd positions |
| `mock`      | Alternating case (`hElLo`)             | Odd positions |
| `noise`     | Appends a random symbol (`world$`)     | Odd positions |
| `chaos`     | Randomly selects one of the above per token, with hover tooltip showing which was applied | Odd positions |

---

## Flags

| Flag              | Description                                           |
|-------------------|-------------------------------------------------------|
| `--provider`      | `openai` (default) or `anthropic`                     |
| `--visual` / `-v` | Color-code even (normal) vs odd (cyan+bold) tokens    |
| `--heatmap`       | Token importance heatmap (red = critical, blue = low) |
| `--web`           | Launch web UI                                         |
| `--port`          | Web UI port (default: 8888)                           |
| `--orchestrator`  | Route through tokio-prompt-orchestrator MCP pipeline  |

---

## Orchestrator Integration

Plugs directly into [tokio-prompt-orchestrator](https://github.com/Mattbusel/tokio-prompt-orchestrator): a production-grade async pipeline with deduplication, circuit breaking, rate limiting, and priority queuing. Use `--orchestrator` to pre-process prompts through a local llama.cpp worker before cloud inference — every output token still intercepted and transformed.

```bash
cargo run -- "Analyze this" reverse --orchestrator --web
```

---

## Architecture

```
CLI args / Web request
        │
        ▼
 TokenInterceptor
  ├── Provider::Openai  ──→ OpenAI SSE stream
  └── Provider::Anthropic ─→ Anthropic SSE stream
        │
        ▼ (per token)
  Transform::apply_with_label()
        │
        ├── web_tx channel ──→ SSE → browser
        └── stdout (terminal mode)
```

**Diff mode** creates two `TokenInterceptor` instances, one per provider, feeding a single merged `mpsc` channel. Events are tagged with their source side and forwarded as a single SSE stream.

**Modules:** `cli` · `providers` · `transforms` · `lib` (interceptor core) · `web` (embedded SPA + TCP server)

---

## Performance

- ~10,000 tokens/sec on commodity hardware
- Sub-millisecond per-token transform overhead
- 4 MB release binary (LTO + single codegen unit)
- Zero-copy async streaming via Tokio
- 208 tests — unit, integration, property-based

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
    { "text": "WHAT", "original": "What", "index": 1, "transformed": true,
      "importance": 0.74, "chaos_label": "uppercase" }
  ],
  "surgery_log": [
    { "index": 4, "original": "is", "replacement": "was", "timestamp": "..." }
  ]
}
```

Every token carries its original text, transformed text, position index, importance score, and — for Chaos mode — the name of the sub-transform applied. Surgery edits are tracked separately for downstream analysis.

---

## Research Applications

- **Perturbation benchmarking** — measure coherence decay rate as transform intensity increases
- **Provider fingerprinting** — identify systematic token-sequence divergence between OpenAI and Anthropic
- **Dataset annotation** — use Token Surgery to correct or label streamed outputs in real time
- **Pipeline observability** — attach to tokio-prompt-orchestrator for full-stack token-level tracing across local and cloud models
- **Chaos resilience scoring** — run Chaos mode across a prompt suite and score model self-correction rate

---

## License

MIT

---

*Every Other Token is an open research instrument. If you're working on LLM evaluation, interpretability, or inference infrastructure and want to collaborate or discuss, open an issue.*
