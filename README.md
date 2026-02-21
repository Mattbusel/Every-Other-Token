# Every Other Token

A real-time token stream mutator for LLM interpretability research.

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/Mattbusel/Every-Other-Token?style=social)](https://github.com/Mattbusel/Every-Other-Token)

Intercepts streaming LLM responses token-by-token and applies real-time mutations. Built in Rust for zero-overhead stream processing. Supports **OpenAI** and **Anthropic** providers, with optional routing through [tokio-prompt-orchestrator](https://github.com/Mattbusel/Tokio-Prompt) for pipeline integration.

```
Prompt  -->  LLM (streaming)  -->  Token Interceptor  -->  Mutated Output
                                        |
                                   every other token
                                   gets transformed
```

---

## Install

```bash
git clone https://github.com/Mattbusel/Every-Other-Token
cd Every-Other-Token
cargo build --release
```

Requires Rust 1.70+ and an API key for your chosen provider.

---

## CLI Usage

```bash
every-other-token <PROMPT> [TRANSFORM] [OPTIONS]
```

### Providers

```bash
# OpenAI (default)
export OPENAI_API_KEY=sk-...
cargo run -- "Explain quantum entanglement" --provider openai

# Anthropic / Claude
export ANTHROPIC_API_KEY=sk-ant-...
cargo run -- "Explain quantum entanglement" --provider anthropic
```

### Transformations

| Transform   | Effect                         | Example (`hello world`)  |
|-------------|--------------------------------|--------------------------|
| `reverse`   | Reverses every other token     | `hello dlrow`            |
| `uppercase` | Uppercases every other token   | `hello WORLD`            |
| `mock`      | aLtErNaTiNg case               | `hello wOrLd`            |
| `noise`     | Appends random noise character | `hello world$`           |

### Examples

```bash
# Basic usage (reverse transform, OpenAI, gpt-3.5-turbo)
cargo run -- "Tell me about black holes"

# Specific transform and model
cargo run -- "Write a haiku" uppercase -m gpt-4

# Anthropic with visual mode
cargo run -- "What is consciousness?" mock --provider anthropic --visual

# Full options
cargo run -- "Explain transformers" reverse -m claude-sonnet-4-20250514 \
  --provider anthropic --visual --heatmap
```

### All Flags

| Flag                 | Description                                          |
|----------------------|------------------------------------------------------|
| `--provider`         | `openai` (default) or `anthropic`                    |
| `-m, --model`        | Model name (auto-selects per provider if omitted)    |
| `-v, --visual`       | Color-code even vs odd tokens                        |
| `--heatmap`          | Token importance heatmap (color = salience)          |
| `--orchestrator`     | Route through tokio-prompt-orchestrator pipeline      |
| `--orchestrator-url` | Orchestrator endpoint (default: `http://localhost:3000`) |

---

## Visual Mode

Color-codes token parity in the stream:

- **Even tokens**: normal text (pass-through)
- **Odd tokens**: bright cyan + bold (transformed)

```bash
cargo run -- "Describe the ocean" reverse --visual
```

## Heatmap Mode

Assigns each token an importance score based on length, position, content type, and syntactic role. Maps score to color:

| Color     | Importance  |
|-----------|-------------|
| Red       | 0.8 - 1.0   |
| Orange    | 0.6 - 0.8   |
| Yellow    | 0.4 - 0.6   |
| Blue      | 0.2 - 0.4   |
| White     | 0.0 - 0.2   |

```bash
cargo run -- "How do neural networks learn?" --heatmap
```

---

## Use as an Orchestrator Transform Stage

Every Other Token integrates with [tokio-prompt-orchestrator](https://github.com/Mattbusel/Tokio-Prompt) as a transform stage in multi-agent LLM pipelines.

### How it works

1. Start the orchestrator dashboard:
   ```bash
   cd /path/to/tokio-prompt-orchestrator
   cargo run --features dashboard --bin dashboard -- --worker echo --port 3000
   ```

2. Run Every Other Token with `--orchestrator`:
   ```bash
   cargo run -- "Analyze this data" reverse --orchestrator
   ```

3. The prompt flows through the orchestrator's 5-stage pipeline (RAG, Assemble, Inference, Post, Stream) before reaching the LLM provider. The orchestrator enriches, deduplicates, and rate-limits the request. The response then gets token-level mutations applied.

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
Terminal Output
```

### Custom orchestrator URL

```bash
cargo run -- "prompt" reverse --orchestrator --orchestrator-url http://192.168.1.10:3000
```

If the orchestrator is unreachable, the tool falls back to direct provider access with a warning.

---

## Performance

Rust implementation benchmarks against equivalent Python:

| Metric           | Rust              | Python           |
|------------------|-------------------|------------------|
| Token throughput  | ~10,000 tok/sec   | ~4,000 tok/sec   |
| Stream latency    | <1ms overhead     | ~3ms overhead    |
| Memory (1k tokens)| ~2 MB             | ~20 MB           |
| Binary size       | 4 MB (LTO)        | N/A (runtime)    |

Built with `lto = true`, `codegen-units = 1`, `panic = "abort"` for maximum throughput.

---

## Research Applications

**Semantic fragility testing** -- measure how token-level mutations degrade model coherence:
```bash
cargo run -- "What is the capital of France?" noise --visual
```

**Attention pattern visualization** -- see which tokens carry semantic weight:
```bash
cargo run -- "Explain how transformers work" --heatmap
```

**Error propagation analysis** -- observe how corrupted tokens affect downstream reasoning:
```bash
cargo run -- "Solve: 87 * 45 =" reverse
```

**Cross-provider comparison** -- identical mutations, different models:
```bash
cargo run -- "Define entropy" uppercase --provider openai -m gpt-4
cargo run -- "Define entropy" uppercase --provider anthropic -m claude-sonnet-4-20250514
```

---

## Contributing

Pull requests welcome.

1. Fork the repo
2. Create a feature branch
3. Add tests for new functionality
4. Submit a PR

---

## License

MIT
