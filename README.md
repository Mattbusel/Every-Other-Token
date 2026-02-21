# Every Other Token

**A real-time token stream mutator for LLM interpretability research.**

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green.svg)](https://openai.com/api/)
[![Anthropic](https://img.shields.io/badge/Anthropic-API-blueviolet.svg)](https://docs.anthropic.com/)
[![GitHub Stars](https://img.shields.io/github/stars/Mattbusel/Every-Other-Token?style=social)](https://github.com/Mattbusel/Every-Other-Token)

> Intercept, mutate, and visualize LLM token streams in real time.
> OpenAI and Anthropic. CLI and browser. One binary. Zero overhead.

```
Prompt  -->  LLM (streaming)  -->  Token Interceptor  -->  Mutated Output
                                        |                    |
                                   every other token     terminal + web UI
                                   gets transformed      simultaneously
```

---

## Why This Exists

LLM outputs look like solid text, but they're generated one token at a time. Every Other Token makes that process visible by intercepting the stream and applying controlled perturbations at the token level. This lets researchers study:

- **Semantic fragility** — how much token-level noise before coherence breaks down?
- **Attention pattern visualization** — which tokens carry semantic weight?
- **Error propagation** — how do corrupted tokens affect downstream reasoning?
- **Cross-provider comparison** — same mutations, different models, different behaviors

Built in Rust for **~10,000 tokens/sec throughput**, **60% lower latency** than equivalent Python, and **90% less memory**. 20+ GitHub stars.

---

## Install

```bash
git clone https://github.com/Mattbusel/Every-Other-Token
cd Every-Other-Token
cargo build --release
```

Set your API key:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## CLI Usage

```bash
every-other-token <PROMPT> [TRANSFORM] [MODEL] [FLAGS]
```

### Providers

```bash
# OpenAI (default)
cargo run -- "Explain quantum entanglement"

# Anthropic / Claude
cargo run -- "Explain quantum entanglement" --provider anthropic
```

When using `--provider anthropic` with the default model, it auto-selects `claude-sonnet-4-20250514`.

### Transformations

| Transform   | Effect                   | Example (`hello world`)  |
|-------------|--------------------------|--------------------------|
| `reverse`   | Reverses odd tokens      | `hello dlrow`            |
| `uppercase` | Uppercases odd tokens    | `hello WORLD`            |
| `mock`      | Alternating case         | `hello wOrLd`            |
| `noise`     | Injects noise characters | `hello world$`           |

### Examples

```bash
# Basic (reverse transform, OpenAI, gpt-3.5-turbo)
cargo run -- "Tell me about black holes"

# Specific transform and model
cargo run -- "Write a haiku" uppercase gpt-4

# Anthropic with visual mode
cargo run -- "What is consciousness?" mock --provider anthropic --visual

# Full options
cargo run -- "Explain transformers" reverse claude-sonnet-4-20250514 \
  --provider anthropic --visual --heatmap

# Web UI in browser
cargo run -- "prompt" --web
```

### All Flags

| Flag              | Description                                                |
|-------------------|------------------------------------------------------------|
| `--provider`      | `openai` (default) or `anthropic`                          |
| `--visual` / `-v` | Color-code even (normal) vs odd (cyan+bold) tokens         |
| `--heatmap`       | Token importance heatmap (red=critical, blue=low)          |
| `--orchestrator`  | Route through tokio-prompt-orchestrator MCP pipeline       |
| `--web`           | Launch interactive web UI at `http://localhost:8888`        |
| `--port`          | Custom port for web UI (default: `8888`)                   |

---

## Web UI

Pass `--web` to launch a dark-themed interactive token stream viewer in your browser. No build step, no npm, no dependencies — a single HTML page embedded in the binary and served over raw TCP.

```bash
cargo run -- "anything" --web
```

This opens `http://localhost:8888` with:

- **Interactive controls** — change prompt, transform, provider, and model directly in the browser
- **Live token animation** — tokens fade in as they stream from the LLM
- **Color coding** — even tokens in white, odd (transformed) tokens in cyan
- **Heatmap toggle** — token importance mapped to red/orange/yellow/blue backgrounds
- **Live stats** — token count and transformed count update in real time
- **Provider switching** — select OpenAI or Anthropic from a dropdown, auto-selects default model

The web UI streams tokens via Server-Sent Events (SSE) from the `/stream` endpoint. Each request creates a fresh interceptor, so you can run multiple prompts without restarting.

```bash
# Custom port
cargo run -- "prompt" --web --port 9090
```

---

## Visual Mode

Color-codes tokens by position in the even/odd cycle:
- **Even tokens**: rendered normally
- **Odd tokens**: bright cyan + bold (post-transformation)

```bash
cargo run -- "Describe the ocean" reverse --visual
```

## Heatmap Mode

Simulated token importance scoring based on length, position, content type, and syntactic weight:

| Color      | Importance         |
|------------|--------------------|
| Bright Red | Critical (0.8-1.0) |
| Red        | High (0.6-0.8)     |
| Yellow     | Medium (0.4-0.6)   |
| Blue       | Low (0.2-0.4)      |
| Normal     | Minimal (0.0-0.2)  |

```bash
cargo run -- "How do neural networks learn?" --heatmap
```

---

## Use as an Orchestrator Transform Stage

Every Other Token integrates with [tokio-prompt-orchestrator](https://github.com/Mattbusel/Tokio-Prompt) as a real-time transform stage in a multi-model inference pipeline.

When `--orchestrator` is enabled, the prompt is first routed through the orchestrator's MCP `infer` tool at `localhost:3000` (using the `llama_cpp` worker) before being streamed through your chosen provider. This lets you:

- **Chain local + cloud inference**: pre-process with a local LLM, then stream through OpenAI/Anthropic
- **Apply pipeline stages**: deduplication, rate limiting, circuit breaking, and caching happen upstream
- **Research multi-model token interactions**: observe how orchestrator-enriched prompts change downstream token distributions

```bash
# Start the orchestrator first
cd ../Tokio-Prompt && cargo run --bin mcp

# Then run Every Other Token with pipeline routing
cargo run -- "Compare these approaches" uppercase --orchestrator --visual
```

The orchestrator gracefully degrades: if `localhost:3000` is unreachable, the raw prompt is used directly.

```
User Prompt
    |
    v
[Orchestrator Pipeline]  <-- dedup, circuit breaker, RAG
    |
    v
[LLM Provider]  <-- OpenAI or Anthropic streaming
    |
    v
[Token Interceptor]  <-- every-other-token transform
    |
    v
Terminal + Web UI
```

---

## Performance

Built in Rust with zero-copy streaming and async I/O:

| Metric           | Rust              | Python           |
|------------------|-------------------|------------------|
| Token throughput  | ~10,000 tok/sec   | ~4,000 tok/sec   |
| Stream latency    | <1ms overhead     | ~3ms overhead    |
| Memory (1k tokens)| ~2 MB             | ~20 MB           |
| Binary size       | 4 MB (LTO)        | N/A (runtime)    |

Compiled with `lto = true`, `codegen-units = 1`, `panic = "abort"`.

---

## Research Applications

**Semantic fragility testing** — how much noise before coherence breaks:
```bash
cargo run -- "What is the capital of France?" noise --visual
```

**Arithmetic robustness** — do reversed tokens degrade math reasoning:
```bash
cargo run -- "Solve: 87 * 45 =" reverse
```

**Cross-provider comparison** — same prompt, same transform, different provider:
```bash
cargo run -- "Explain attention mechanisms" mock --provider openai --heatmap
cargo run -- "Explain attention mechanisms" mock --provider anthropic --heatmap
```

**Orchestrator-augmented analysis** — local LLM pre-processing before cloud inference:
```bash
cargo run -- "Summarize this research" uppercase --orchestrator --provider anthropic
```

**Browser-based demos** — interactive token streaming for presentations:
```bash
cargo run -- "anything" --web
```

---

## Contributing

Pull requests welcome.

1. Fork the repo
2. Create a branch
3. Make your changes
4. Run `cargo test` (30+ tests must pass)
5. Submit a PR

---

## License

MIT
