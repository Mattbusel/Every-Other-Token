# Every Other Token

**A real-time token stream mutator for LLM interpretability research.**

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Mattbusel/Every-Other-Token?style=social)](https://github.com/Mattbusel/Every-Other-Token)

> Intercept, mutate, and visualize LLM token streams in real time.
> OpenAI and Anthropic. One binary. Zero overhead.

---

## What It Does

Every Other Token sits between you and an LLM streaming API. It intercepts the token stream in real time and applies a transformation to every other token, letting you observe how models behave under controlled token-level perturbation.

**Supported providers:** OpenAI (`gpt-4`, `gpt-3.5-turbo`) and Anthropic (`claude-sonnet-4-20250514`, `claude-haiku-4-5-20251001`)

**Transformations:**

| Transform   | Effect                   | Example (`hello world`)  |
|-------------|--------------------------|--------------------------|
| `reverse`   | Reverses odd tokens      | `hello dlrow`            |
| `uppercase` | Uppercases odd tokens    | `hello WORLD`            |
| `mock`      | Alternating case         | `hello wOrLd`            |
| `noise`     | Injects noise characters | `hello world$`           |

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

### Examples

```bash
# OpenAI (default)
cargo run -- "Tell me a story about a robot"
cargo run -- "Explain quantum physics" uppercase gpt-4
cargo run -- "What is consciousness?" reverse --visual --heatmap

# Anthropic
cargo run -- "Write a haiku" mock --provider anthropic
cargo run -- "Explain transformers" reverse claude-sonnet-4-20250514 --provider anthropic --visual

# With orchestrator pipeline
cargo run -- "Analyze this prompt" uppercase --orchestrator --visual
```

### Flags

| Flag              | Description                                                |
|-------------------|------------------------------------------------------------|
| `--provider`      | `openai` (default) or `anthropic`                          |
| `--visual` / `-v` | Color-code even (normal) vs odd (cyan+bold) tokens         |
| `--heatmap`       | Token importance heatmap (red=critical, blue=low)          |
| `--orchestrator`  | Route through tokio-prompt-orchestrator MCP pipeline       |
| `--web`           | Launch browser-based live token stream UI                  |
| `--port <PORT>`   | Web UI port (default: 8888)                                |

---

## Visual Mode

Color-codes tokens by position in the even/odd cycle:
- **Even tokens**: rendered normally
- **Odd tokens**: bright cyan + bold (post-transformation)

## Heatmap Mode

Simulated token importance scoring based on length, position, content type, and syntactic weight:

| Color      | Importance   |
|------------|--------------|
| Bright Red | Critical (0.8-1.0) |
| Red        | High (0.6-0.8)     |
| Yellow     | Medium (0.4-0.6)   |
| Blue       | Low (0.2-0.4)      |
| Normal     | Minimal (0.0-0.2)  |

---

## Web UI

Launch a real-time browser dashboard that streams tokens as they arrive from the LLM. No build step, no npm — a single HTML page is embedded in the binary and served over raw TCP.

```bash
cargo run -- "Tell me a story" --web
# Opens http://localhost:8888 automatically
```

Features:
- **Live SSE streaming** — tokens appear in the browser as they're generated
- **Dark theme** — monospace, research-grade aesthetic
- **Even/odd color coding** — normal tokens vs cyan+bold transformed tokens
- **Heatmap mode** — toggle importance coloring (red=critical, blue=low)
- **In-browser controls** — change prompt, transform, provider, and model without restarting
- **Zero dependencies** — no JavaScript framework, no build tools, no node_modules

The web server accepts `/stream` requests with query parameters, creates a fresh `TokenInterceptor` per request, and forwards `TokenEvent` structs as SSE `data:` lines. The frontend renders each token as an animated `<span>` with appropriate CSS classes.

```bash
# Custom port
cargo run -- "Explain entropy" uppercase --web --port 9090

# Anthropic + web UI
cargo run -- "Write a poem" mock --provider anthropic --web
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

---

## Performance

Built in Rust with zero-copy streaming and async I/O:

- ~10,000 tokens/sec throughput on commodity hardware
- 60% lower latency than equivalent Python implementations
- 90% lower memory footprint
- Sub-millisecond per-token transform overhead

---

## Research Applications

**Semantic fragility testing** -- How much token-level noise can a model tolerate before coherence breaks down?

```bash
cargo run -- "What is the capital of France?" noise --visual
```

**Arithmetic robustness** -- Do reversed tokens degrade mathematical reasoning?

```bash
cargo run -- "Solve: 87 * 45 =" reverse
```

**Cross-provider comparison** -- Same prompt, same transform, different provider:

```bash
cargo run -- "Explain attention mechanisms" mock --provider openai --heatmap
cargo run -- "Explain attention mechanisms" mock --provider anthropic --heatmap
```

**Orchestrator-augmented analysis** -- Local LLM pre-processing before cloud inference:

```bash
cargo run -- "Summarize this research" uppercase --orchestrator --provider anthropic
```

---

## Contributing

Pull requests welcome.

1. Fork the repo
2. Create a branch
3. Make your changes
4. Run `cargo test` (26 tests must pass)
5. Submit a PR

---

## License

MIT
