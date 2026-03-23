# every-other-token

[![CI](https://github.com/Mattbusel/Every-Other-Token/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/Every-Other-Token/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/Mattbusel/Every-Other-Token/branch/main/graph/badge.svg)](https://codecov.io/gh/Mattbusel/Every-Other-Token)
[![crates.io](https://img.shields.io/crates/v/every-other-token.svg)](https://crates.io/crates/every-other-token)
[![docs.rs](https://docs.rs/every-other-token/badge.svg)](https://docs.rs/every-other-token)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.81+-orange.svg)](https://www.rust-lang.org/)

**every-other-token** is a real-time LLM token stream interceptor for interpretability research. It sits between your application and the model, intercepts each token as it arrives over SSE, applies configurable mutation transforms, captures per-token confidence and perplexity from the logprob API, and routes enriched events simultaneously to a color-coded terminal renderer, a zero-dependency web UI, WebSocket collaboration rooms, a JSON replay recorder, and the new token attribution exporter — all without buffering the full response.

---

## Unique capabilities vs. standard LLM clients

| Capability | Standard clients | every-other-token |
|------------|-----------------|-------------------|
| Per-token confidence scores | No | Yes — `exp(logprob)` at each position |
| Per-token perplexity | No | Yes — `exp(-logprob)` at each position |
| Live stream mutation | No | Yes — 9 transform types with rate and seed control |
| Cross-provider structural diff | No | Yes — Jensen-Shannon divergence, Pearson correlation |
| A/B system-prompt significance testing | No | Yes — Welch's t-test across confidence distributions |
| Token attribution export | No | Yes — JSONL, CSV, self-contained HTML heatmap |
| Semantic drift detection | No | Yes — confidence decay from start to end of sequence |
| Replay determinism | No | Yes — record and replay any run from JSON |
| Collaborative rooms | No | Yes — WebSocket multi-participant token surgery |
| TF-IDF semantic heatmaps | No | Yes — no embedding service required |

---

## 5-minute quickstart

### Prerequisites

- Rust 1.81 or later
- An OpenAI API key (`OPENAI_API_KEY`) and/or an Anthropic API key (`ANTHROPIC_API_KEY`)

```bash
git clone https://github.com/Mattbusel/Every-Other-Token
cd Every-Other-Token
cargo build --release

export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Run immediately

```bash
# Terminal output with per-token confidence color bands
./target/release/every-other-token "What is consciousness?" --visual

# Web UI — opens http://localhost:8888 automatically
./target/release/every-other-token "What is consciousness?" --web

# No API key needed — dry run with chaos transform
./target/release/every-other-token "hello world" --dry-run --transform chaos

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

---

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

---

## Token attribution export

`attribution::AttributionExporter` exports per-token confidence, perplexity, and attribution scores to three formats compatible with LIME, Captum, and custom dashboards.

### Data types

| Type | Description |
|------|-------------|
| `TokenAttribution` | One token: position, confidence, perplexity, attribution score, up to 5 top alternatives |
| `SequenceAttribution` | Full sequence: all tokens + aggregate confidence, aggregate perplexity, semantic drift |
| `AttributionExporter` | Serializes to JSONL, CSV, or self-contained HTML heatmap |

### Export to JSONL (LIME / Captum compatible)

```rust
use every_other_token::attribution::{AttributionExporter, SequenceAttribution, TokenAttribution};

let seq = SequenceAttribution {
    prompt: "What is recursion?".into(),
    model: "gpt-4o".into(),
    timestamp: "2026-03-22T00:00:00Z".into(),
    tokens: vec![
        TokenAttribution {
            token: "Recursion".into(),
            position: 0,
            confidence: 0.92,
            perplexity: 1.09,
            attribution_score: 0.45,
            top_alternatives: vec![("Iteration".into(), 0.05)],
        },
    ],
    aggregate_confidence: 0.92,
    aggregate_perplexity: 1.09,
    semantic_drift: 0.0,
};

let exporter = AttributionExporter::new();

// JSONL -- one JSON object per line, pipe directly to Python
let jsonl = exporter.to_jsonl(&[seq.clone()]);

// CSV -- flat table, one row per token
let csv = exporter.to_csv(&seq);

// Self-contained HTML heatmap -- open in any browser, no external deps
let html = exporter.to_html_heatmap(&seq);
std::fs::write("heatmap.html", html).unwrap();
```

### Semantic drift detection

Semantic drift measures the confidence decay from the first half of the generated sequence to the second half. A positive value means the model became less certain toward the end of the response.

```rust
let drift = exporter.detect_drift(&seq);
if drift > 0.15 {
    println!("Warning: high semantic drift ({:.3}) -- model confidence collapsed mid-response", drift);
}
```

### Finding the most and least confident tokens

```rust
let (most_confident, least_confident) = exporter.confidence_extremes(&seq);
println!("Most confident positions: {:?}", most_confident);
println!("Least confident positions: {:?}", least_confident);
```

### Python interoperability

The JSONL format maps directly to a pandas DataFrame:

```python
import json, pandas as pd

rows = []
with open("attribution.jsonl") as f:
    for line in f:
        seq = json.loads(line)
        for tok in seq["tokens"]:
            rows.append({
                "prompt":      seq["prompt"],
                "model":       seq["model"],
                "position":    tok["position"],
                "token":       tok["token"],
                "confidence":  tok["confidence"],
                "perplexity":  tok["perplexity"],
                "attribution": tok["attribution_score"],
            })
df = pd.DataFrame(rows)
```

---

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

---

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

Change the port with `--port 9000`. The port can also be set in `~/.eot.toml`.

---

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

---

## CLI reference

```
USAGE:
    every-other-token [OPTIONS] <PROMPT> [TRANSFORM] [MODEL]

ARGS:
    <PROMPT>      Input prompt (use "-" to read from stdin)
    [TRANSFORM]   Transform type [default: reverse]
    [MODEL]       Model name [default: gpt-3.5-turbo]

OPTIONS:
    --provider <PROVIDER>           openai | anthropic | mock [default: openai]
    --visual, -v                    Enable ANSI confidence-colored output
    --heatmap                       Enable token importance heatmap
    --web                           Launch web UI instead of terminal
    --port <PORT>                   Web UI port [default: 8888]
    --research                      Headless research mode
    --runs <RUNS>                   Number of research runs [default: 10]
    --output <FILE>                 Research output JSON path [default: research_output.json]
    --system-a <PROMPT>             System prompt A (A/B mode)
    --system-b <PROMPT>             System prompt B (A/B mode)
    --top-logprobs <N>              Top alternative tokens per position (0-20) [default: 5]
    --significance                  Compute Welch's t-test across A/B confidence distributions
    --heatmap-export <FILE>         Export per-position confidence heatmap to CSV
    --record <FILE>                 Record token events to JSON replay file
    --replay <FILE>                 Replay token events from file (no API call)
    --rate <F>                      Fraction of tokens to transform (0.0-1.0) [default: 0.5]
    --rate-range <MIN-MAX>          Stochastic rate from interval (e.g. "0.3-0.7")
    --seed <N>                      Fixed RNG seed for reproducible Noise/Chaos runs
    --dry-run                       Validate transform without calling any API
    --min-confidence <F>            Only transform tokens below this confidence value
    --diff-terminal                 Parallel OpenAI + Anthropic streams side by side
    --json-stream                   One JSON line per token to stdout
    --format <FMT>                  Research output format: "json" or "jsonl" [default: json]
    --completions <SHELL>           Generate shell completions (bash/zsh/fish/...)
    --log-db <FILE>                 SQLite experiment log (requires sqlite-log feature)
```

---

## Divergence Detection Engine

`divergence.rs` runs the same prompt through N model configurations simultaneously and computes a per-position **Jensen-Shannon divergence** score showing exactly where and how much models disagree.

### Concepts

| Term | Description |
|------|-------------|
| `ModelConfig` | One model configuration: provider, model name, temperature, top_p |
| `TokenStream` | Sequence of tokens + per-position probability distributions from one model |
| `DivergencePoint` | A position where JS score exceeds the threshold; records each model's token |
| `DivergenceResult` | Complete run: all streams, all divergence points, aggregate statistics |
| `DivergenceDetector` | Orchestrates analysis; threshold configurable |

### Jensen-Shannon divergence

At each token position `t`, the detector collects a probability distribution `P_i(t)` from each model (from the logprobs API). It then computes:

```
JSD(P₁, …, Pₙ) = (1/n) Σᵢ KL(Pᵢ || M)    where M = (1/n) Σᵢ Pᵢ
```

normalised by `log(n)` to give a score in `[0, 1]`.

| Score | Meaning |
|-------|---------|
| 0.0 | All models produce the same token with the same probability |
| ~0.25 | Mild preference difference |
| ~0.5 | Substantial disagreement |
| 1.0 | Models produce completely different tokens |

### Usage

```rust
use every_other_token::divergence::{DivergenceDetector, ModelConfig};
use std::collections::HashMap;

let configs = vec![
    ModelConfig::new("openai", "gpt-4o", 0.0, 1.0),
    ModelConfig::new("openai", "gpt-4o", 1.0, 1.0),
    ModelConfig::new("anthropic", "claude-sonnet-4-6", 0.7, 1.0),
];

let detector = DivergenceDetector::new(configs).with_threshold(0.1);

// After collecting token streams from provider APIs:
let result = detector.analyse("Why is the sky blue?", streams);

println!("{}", result.render_report());
println!("Mean JS: {:.4}", result.mean_divergence());
println!("High-divergence positions: {:?}", result.high_divergence_positions(0.5));
```

### Coloured diff report

`DivergenceResult::render_report()` produces a multi-line ANSI report:

- **Green background**: models agree (JS < 0.25).
- **Yellow background**: moderate divergence (JS ∈ (0.25, 0.5]).
- **Red background**: high divergence (JS > 0.5).

Each model's full token stream is printed with its disagreement positions highlighted.

---

## Token Intervention Mode

`intervention.rs` enables **causal interventions**: pausing generation at any token, injecting a correction, continuing generation from the injected token, and measuring the causal effect downstream.

### Concepts

| Term | Description |
|------|-------------|
| `Intervention` | A single `(position, original_token, injected_token)` triple |
| `InterventionHistory` | Full record: original stream, all interventions, final counterfactual stream |
| `TokenEditor` | ANSI terminal widget for cursor-based token editing |
| `causal_effect()` | Normalised token-level edit distance between original and final streams |
| `causal_influence_map()` | Per-position influence score: how much each position affects downstream tokens |

### Causal effect measurement

```rust
use every_other_token::intervention::{InterventionHistory, Intervention};

// Start with the original model output.
let original = vec!["The".into(), " cat".into(), " sat".into(), " on".into(), " the".into(), " mat".into()];
let mut history = InterventionHistory::new(original);

// Inject " dog" at position 1.
let intervention = Intervention::new(1, " cat", " dog");
// (continue generation from this point, get the counterfactual)
let counterfactual = vec!["The".into(), " dog".into(), " ran".into(), " through".into(), " the".into(), " park".into()];
history.apply(intervention, counterfactual);

println!("Causal effect: {:.3}", history.causal_effect()); // fraction of tokens that changed
```

### TUI token editor

`TokenEditor` provides cursor navigation and token-level editing in any VT100 terminal:

```rust
use every_other_token::intervention::TokenEditor;

let tokens = vec!["The".into(), " sky".into(), " is".into(), " blue".into()];
let mut editor = TokenEditor::new(tokens);

editor.cursor_right();      // move to " sky"
editor.enter_edit_mode();   // start editing
editor.edit_buffer = " ocean".into();
let edit = editor.commit_edit(); // Some((1, " sky", " ocean"))

println!("{}", editor.render()); // ANSI-highlighted stream
```

### Causal influence map

```rust
use every_other_token::intervention::causal_influence_map;

let original     = vec!["a".into(), "b".into(), "c".into(), "d".into()];
let counterfact  = vec!["X".into(), "Y".into(), "c".into(), "d".into()];

let map = causal_influence_map(&original, &counterfact);
// map[0] = fraction of positions >0 that changed = 0.5  (b changed, c/d didn't)
// map[1] = fraction of positions >1 that changed = 0.0  (c and d unchanged)
```

---

## Key modules

| Module | Responsibility |
|--------|----------------|
| `lib.rs` | `TokenInterceptor`, `TokenEvent`, stream parsing, retry logic |
| `attribution.rs` | Per-token JSONL, CSV, and HTML heatmap export; drift detection |
| `divergence.rs` | Jensen-Shannon divergence between model configurations; coloured diff report |
| `intervention.rs` | Token injection, `InterventionHistory`, `TokenEditor`, causal influence map |
| `transforms.rs` | All transform strategies (`Reverse`, `Noise`, `Chaos`, `Chain`, ...) |
| `providers.rs` | `ProviderPlugin` trait, OpenAI and Anthropic SSE wire types |
| `web.rs` | Embedded HTTP/1.1 server, SSE fan-out, WebSocket upgrade |
| `collab.rs` | Room store, participant management, token surgery, chat, recording |
| `research.rs` | Headless research loop, aggregate statistics, A/B mode |
| `comparison.rs` | Cross-model JS divergence, Pearson correlation, structural diff |
| `semantic_heatmap.rs` | TF-IDF cosine similarity windows -> SVG / CSV heatmap |
| `token_dictionary.rs` | Per-token sentiment dictionary with EMA learning |
| `heatmap.rs` | Per-position confidence matrix -> CSV |
| `replay.rs` | JSON recording and deterministic replay |
| `render.rs` | Terminal ANSI coloring, confidence indicators |
| `store.rs` | SQLite-backed experiment persistence |
| `config.rs` | `~/.eot.toml` / `./.eot.toml` config with merge semantics |
| `bayesian.rs` | Bayesian confidence interval estimation across runs |
| `checkpoint.rs` | Snapshot/restore for long research sessions |

---

## Feature flags

| Flag | Default | Description |
|------|---------|-------------|
| *(none)* | On | Terminal and web UI streaming, all transforms, research mode, collab rooms |
| `sqlite-log` | Off | Persist experiment runs to local SQLite via `store::ExperimentStore` |
| `self-tune` | Off | Background PID-based parameter tuning loop and telemetry bus |
| `self-modify` | Off | Agent loop for automated pipeline improvement (requires `self-tune`) |
| `intelligence` | Off | Reserved namespace for future interpretability features |
| `evolution` | Off | Reserved namespace for evolutionary optimisation |
| `helix-bridge` | Off | HTTP bridge polling HelixRouter `/api/stats` |
| `redis-backing` | Off | Write-through Redis persistence for agent memory and snapshots |
| `wasm` | Off | WASM target bindings via `wasm-bindgen` |

---

## Building from source

```bash
git clone https://github.com/Mattbusel/Every-Other-Token
cd Every-Other-Token
cargo build --release
cargo test --lib
```

Enable optional features:

```bash
cargo build --release --features sqlite-log,self-tune
```

---

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b my-feature`.
3. Ensure `cargo fmt`, `cargo clippy -- -D warnings`, and `cargo test --lib` all pass.
4. Open a pull request against `main`.

CI enforces formatting, Clippy, the full test suite, and a release build before merging.

---

## License

MIT -- see [LICENSE](LICENSE) for details.
