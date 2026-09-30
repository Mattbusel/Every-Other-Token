# Using every-other-token

The command-line and web UI guide: every transform, flag group, mode and config option. For a one-screen overview see the [README](../README.md); for the library API see [REFERENCE.md](REFERENCE.md).

## Unique capabilities vs. standard LLM clients

| Capability | Standard clients | every-other-token |
|------------|-----------------|-------------------|
| Per-token confidence scores | No | Yes: `exp(logprob)` at each position |
| Per-token perplexity | No | Yes: `exp(-logprob)` at each position |
| Live stream mutation | No | Yes: 9 transform types with rate and seed control |
| Cross-provider structural diff | No | Yes: Jensen-Shannon divergence, Pearson correlation |
| A/B system-prompt significance testing | No | Yes: Welch's t-test across confidence distributions |
| Token attribution export | No | Yes: JSONL, CSV, self-contained HTML heatmap |
| Causal attribution map | No | Yes: leave-one-out input-token influence scores |
| Prompt mutation lab | No | Yes: systematic variant ranking by perplexity/length/etc. |
| Semantic drift detection | No | Yes: confidence decay from start to end of sequence |
| Replay determinism | No | Yes: record and replay any run from JSON |
| Collaborative rooms | No | Yes: WebSocket multi-participant token surgery |
| TF-IDF semantic heatmaps | No | Yes: no embedding service required |

## Use cases

<details>
<summary>Interpretability research, red-teaming, prompt engineering</summary>

### Interpretability research

- Map which input tokens causally drive each output token using
  `AttributionMap` (leave-one-out proxy).
- Visualize confidence and perplexity trajectories across 20+ runs with
  `--research --runs 20`.
- Export confidence heatmaps to CSV for cross-model comparisons in pandas or R.
- Detect semantic drift: identify responses where model certainty collapses
  mid-generation.
- Run systematic ablations: mask one input concept at a time and measure the
  effect on every output position.

### Red-teaming

- Use `MutationLab` with `MutationTarget::SystemPrompt` to rank which system
  prompt variants produce the longest outputs (potential jailbreak signal).
- Vary candidate adversarial phrases with `MutationTarget::Word` and measure
  `OutputLength` to detect instruction-override candidates.
- Combine `--min-confidence 0.5` with the `delete` transform to find positions
  where the model is easiest to steer by omission.
- Record sessions with `--record` and replay them after a model update to
  detect behavioral regressions.

### Prompt engineering

- Rank synonym variants with `MutationLab` and `MutationMetric::Perplexity`
  to find the wording that the model handles most confidently.
- A/B test system prompts across 30 runs with `--significance` to find
  statistically significant confidence shifts.
- Use the `--visual` heatmap to spot high-perplexity tokens in your template
  that signal ambiguity to the model.
- Export a self-contained HTML heatmap with `AttributionExporter::to_html_heatmap`
  to share results without requiring a Python environment.

</details>

## Build and run from source

### Prerequisites

- Rust 1.81 or later
- For real models: an OpenAI API key (`OPENAI_API_KEY`) and/or an Anthropic API key (`ANTHROPIC_API_KEY`). The mock provider needs neither.

```bash
git clone https://gitlab.com/mattbusel/Every-Other-Token
cd Every-Other-Token
cargo build --release
```

### Try it without an API key

The `mock` provider replays a canned token stream (with logprobs) through the real interception pipeline, so you can see what the tool does before spending any tokens:

```bash
# Every other token reversed, transformed tokens highlighted
./target/release/every-other-token "What is consciousness?" --provider mock --visual

# Per-token table: original vs. intercepted token, confidence, perplexity
cargo run --example mock_stream
cargo run --example mock_stream -- "Your prompt here" uppercase
```

The reply text is fixed (a pangram plus your prompt echoed back); only the provider is fake.

### Use a real model

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Run immediately

```bash
# Terminal output with per-token confidence color bands
./target/release/every-other-token "What is consciousness?" --visual

# Web UI, opens http://localhost:8888 automatically
./target/release/every-other-token "What is consciousness?" --web

# No API key needed, dry run with chaos transform
./target/release/every-other-token "hello world" chaos --dry-run

# Side-by-side OpenAI vs Anthropic diff
./target/release/every-other-token "Describe entropy" --diff-terminal

# Headless research: 20 runs, JSON aggregate stats
./target/release/every-other-token "Explain recursion" \
    --research --runs 20 --output results.json
```

### Shell completions

```bash
./target/release/every-other-token --completions bash >> ~/.bash_completion
./target/release/every-other-token --completions zsh  >  ~/.zfunc/_every-other-token
./target/release/every-other-token --completions fish > ~/.config/fish/completions/every-other-token.fish
```

## Transform types

Each token in the stream can be independently mutated before it reaches the output.

| Name | Effect | Deterministic |
|------|--------|---------------|
| `reverse` | Reverses characters: `"hello"` -> `"olleh"` | Yes |
| `uppercase` | To uppercase: `"hello"` -> `"HELLO"` | Yes |
| `mock` | Alternating case per char: `"hello"` -> `"hElLo"` | Yes |
| `noise` | Appends a random symbol from `* + ~ @ # $ %` | No (use `--seed`) |
| `chaos` | Randomly selects one of the above per token | No (use `--seed`) |
| `scramble` | Fisher-Yates shuffles token characters | No (use `--seed`) |
| `delete` | Replaces the token with the empty string | Yes |
| `synonym` | Substitutes from a 200-entry static synonym table | Yes |
| `delay:N` | Passes through after an N-millisecond pause | Yes |
| `A,B,...` | Chain: applies A, then B, then ... in sequence | Depends on chain |

### Rate control

`--rate 0.5` (default) transforms every other token. Uses a Bresenham spread for uniform distribution at any rate. Combine with `--seed N` for fully reproducible runs.

`--rate-range 0.3-0.7` picks a random rate in [min, max] per run.

`--min-confidence 0.8` only transforms tokens whose API confidence is below the threshold. High-confidence tokens pass through unchanged.

## A/B system prompt testing

Test how two different system prompts affect per-token confidence distributions, with automatic statistical significance testing.

```bash
./target/release/every-other-token "Explain machine learning" \
    --research --runs 30 \
    --system-a "You are a concise technical expert." \
    --system-b "You are a friendly tutor explaining to a beginner." \
    --significance \
    --output ab_results.json
```

The output JSON includes:
- Per-run confidence histograms for system A and system B
- Welch's t-test statistic and p-value
- Mean confidence delta between the two system prompts
- Positions where the distributions diverged most

### A/B via the web UI

Launch with `--web` and select the **Experiment** view to see both system prompts streaming side-by-side with a live divergence map.

## Web UI guide

Launch with `--web` to open the single-page application at `http://localhost:8888`.

| View | Description |
|------|-------------|
| **Single** | Live token stream with per-token confidence bars and perplexity pulse |
| **Split** | Original vs transformed output side by side |
| **Quad** | Four transforms applied simultaneously in a 2x2 grid |
| **Diff** | OpenAI and Anthropic streaming the same prompt; diverging positions highlighted |
| **Experiment** | A/B mode: two system prompts, live divergence map |
| **Research** | Aggregate stats dashboard: perplexity histogram, confidence distribution, vocabulary diversity |

Change the port with `--port 9000` (also settable in `~/.eot.toml`); `--no-open` skips opening a browser tab. With no API key, run `every-other-token --web --provider mock` or pick **Mock (no API key)** in the provider menu.

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

All CLI flags override config file values.

## Building from source

```bash
git clone https://gitlab.com/mattbusel/Every-Other-Token
cd Every-Other-Token
cargo build --release
cargo test --lib
```

Enable optional features:

```bash
cargo build --release --features sqlite-log,self-tune
```
