# Every Other Token

[![Star History Chart](https://api.star-history.com/svg?repos=Mattbusel/Every-Other-Token&type=Date)](https://star-history.com/#Mattbusel/Every-Other-Token)


[![CI](https://github.com/Mattbusel/Every-Other-Token/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/Every-Other-Token/actions/workflows/ci.yml) [![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/) [![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Tests](https://img.shields.io/badge/tests-1165%20passing-brightgreen.svg)](https://github.com/Mattbusel/Every-Other-Token) [![GitHub Stars](https://img.shields.io/github/stars/Mattbusel/Every-Other-Token?style=social)](https://github.com/Mattbusel/Every-Other-Token)

> What if you intercept every other token from a live LLM stream and measure what breaks?

That question led to this: a Rust tool that sits between you and the model, mutates tokens mid-generation, scores the model's confidence at each position, and visualizes the result in real time. It turns out LLMs are surprisingly fragile — and surprisingly resilient — in ways that aggregate benchmarks completely miss.

**What it does:** intercept → perturb → score → visualize → compare across providers, simultaneously, in a zero-dependency web UI or terminal. Multiple researchers can watch the same stream live.

Built in Rust. Dual-provider (OpenAI + Anthropic). 1,165 tests. Ships as a single binary.

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
export ANTHROPIC_API_KEY="sk-ant-..."

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

| Mode | Description |
|------|-------------|
| **Single** | Live token stream with per-token confidence bars and perplexity pulse |
| **Split** | Original vs transformed side by side |
| **Quad** | All four transforms applied simultaneously in a 2x2 grid |
| **Diff** | OpenAI and Anthropic streaming the same prompt in parallel — diverging positions highlighted red |
| **Experiment** | A/B mode: two system prompts, same user prompt, live divergence map |
| **Research** | Aggregate stats dashboard — perplexity, confidence histogram, vocab diversity, cost |

---

## Research Features

Every token from OpenAI carries a **confidence score** (exp(logprob), 0–1) and **top-5 alternatives** at that position. Color-coded per token: green ≥70%, yellow 40–70%, red <40%.

Per-token **perplexity** streams live. High-perplexity tokens pulse in the UI. A 60-token rolling sparkline shows the model's uncertainty trajectory.

**Headless mode:**
```bash
cargo run -- "Explain the halting problem" --research --runs 50 --output halting.json
cargo run -- "Tell me a story" --research --runs 10 --system-a "Be poetic." --system-b "Be literal."
```

---

## Transforms

| Transform | Behavior |
|-----------|----------|
| `reverse` | Reverses token characters at odd positions |
| `uppercase` | Uppercases token at odd positions |
| `mock` | Alternating case at odd positions |
| `noise` | Appends a random symbol at odd positions |
| `chaos` | Randomly selects one of the above per token |

---

## Architecture

```
TokenInterceptor
 ├── Provider::Openai   → OpenAI SSE stream (logprobs=true)
 └── Provider::Anthropic → Anthropic SSE stream
       │
       ▼ (per token)
 process_content_with_logprob()
  ├── confidence = exp(logprob)
  ├── perplexity = exp(-logprob)
  └── alternatives = top_logprobs[0..5]
       │
       ├── web_tx  ──→ SSE → browser
       ├── collab  ──→ WebSocket → all room participants
       └── stdout  ──→ terminal mode

SelfImprovementOrchestrator (background)
  TelemetryBus → AnomalyDetector → TuningController → RedisSnapshotStore
```

## Performance

- ~10,000 tokens/sec on commodity hardware
- Sub-millisecond per-token overhead
- 4 MB release binary (LTO)
- Zero-copy async streaming via Tokio
- 1,165 tests

---

## License

MIT

---

*Open an issue if you're working on LLM evaluation, interpretability, or inference infrastructure.*

---

## Ecosystem

- [tokio-prompt-orchestrator](https://github.com/Mattbusel/tokio-prompt-orchestrator) -- the orchestration layer that uses Every-Other-Token telemetry to drive HelixRouter adaptation
- [LLM-Hallucination-Detection-Script](https://github.com/Mattbusel/LLM-Hallucination-Detection-Script) -- companion tool for output-level reliability analysis
- [llm-cpp](https://github.com/Mattbusel/llm-cpp) -- C++ streaming primitive that inspired the Rust token pipeline here