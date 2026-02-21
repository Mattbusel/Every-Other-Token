# Every Other Token

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Mattbusel/Every-Other-Token?style=social)](https://github.com/Mattbusel/Every-Other-Token)

LLMs produce tokens. We intercept them.

Every Other Token transforms every other token in a live LLM stream — reversing, uppercasing, injecting noise — and visualizes the result in real time. The question it answers: how fragile is coherence at the token level? One transformed token in two, and you can watch a model's reasoning degrade, adapt, or hold.

20 stars. Built in Rust. ~10,000 tokens/sec with sub-millisecond per-token overhead. Streams from OpenAI and Anthropic. Ships a web UI with zero build step — `--web` opens a dark-themed live viewer in your browser with animated token streaming and heatmap visualization.

Plugs directly into [tokio-prompt-orchestrator](https://github.com/Mattbusel/tokio-prompt-orchestrator) for multi-model pipeline research: local llama.cpp pre-processing, cloud inference, deduplication, circuit breaking — all upstream of the token transform.

## Quick Start

```bash
git clone https://github.com/Mattbusel/Every-Other-Token
cd Every-Other-Token
cargo build --release
```

```bash
export OPENAI_API_KEY="sk-..."        # or
export ANTHROPIC_API_KEY="sk-ant-..." # or both
```

```bash
# Terminal mode
cargo run -- "Tell me a story about a robot"
cargo run -- "Explain quantum physics" uppercase gpt-4 --visual
cargo run -- "Write a haiku" mock --provider anthropic --heatmap

# Web UI — opens http://localhost:8888
cargo run -- "What is consciousness?" --web

# With orchestrator pipeline
cargo run -- "Analyze this" uppercase --orchestrator --visual
```

## Web UI

`--web` launches a local HTTP server with an embedded single-page app. No npm, no webpack, no build step. Dark theme. Tokens animate in as they stream. Even tokens render normal, odd tokens highlighted cyan. Toggle heatmap mode for importance-weighted color coding.

```bash
cargo run -- "prompt" --web                    # default port 8888
cargo run -- "prompt" --web --port 9000        # custom port
```

The web UI lets you change provider, transform, model, and heatmap mode live from the browser controls.

## Transforms

| Transform   | Effect                    | Example              |
|-------------|---------------------------|----------------------|
| `reverse`   | Reverses odd tokens       | `hello dlrow`        |
| `uppercase` | Uppercases odd tokens     | `hello WORLD`        |
| `mock`      | Alternating case          | `hello wOrLd`        |
| `noise`     | Injects noise characters  | `hello world$`       |

## Flags

| Flag              | Description                                          |
|-------------------|------------------------------------------------------|
| `--provider`      | `openai` (default) or `anthropic`                    |
| `--visual` / `-v` | Color-code even (normal) vs odd (cyan+bold) tokens   |
| `--heatmap`       | Token importance heatmap (red=critical, blue=low)    |
| `--web`           | Launch web UI in browser with live streaming          |
| `--port`          | Web UI port (default: 8888)                          |
| `--orchestrator`  | Route through tokio-prompt-orchestrator MCP pipeline  |

## Research Applications

**Semantic fragility** — How much token-level noise before coherence breaks?

**Arithmetic robustness** — Do reversed tokens degrade mathematical reasoning?

**Cross-provider comparison** — Same prompt, same transform, OpenAI vs Anthropic. Side by side.

**Multi-model pipelines** — Local LLM pre-processing through the orchestrator before cloud inference with token-level observation.

## Performance

- ~10,000 tokens/sec on commodity hardware
- Sub-millisecond per-token transform overhead
- 4MB binary with LTO
- Zero-copy async streaming

## License

MIT
