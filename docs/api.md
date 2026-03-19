# API Reference

> Auto-generated summary of every public primitive in the crate.
> Full rustdoc is published at **<https://docs.rs/every-other-token>**.

---

## Core types

### `TokenInterceptor`
*`src/lib.rs`*

The main streaming engine. Constructs an HTTP connection to the configured
provider, walks the SSE stream, applies the active `Transform` at every
Bresenham-spread position, attaches per-token confidence/perplexity from API
logprobs, and fans out enriched `TokenEvent`s to the terminal, web UI, or
JSON-stream output.

**Construction (builder pattern)**

```rust
let interceptor = TokenInterceptor::new(provider, transform, model, visual, heatmap, orchestrator)?
    .with_rate(0.4)
    .with_seed(42)
    .with_top_logprobs(5)
    .with_system_prompt("Be concise.")
    .with_web_tx(tx)
    .with_max_retries(3)
    .with_orchestrator_url("http://localhost:3000")
    .with_json_stream(false)
    .with_min_confidence(0.8);

interceptor.intercept_stream("What is consciousness?").await?;
```

**Key public fields**

| Field | Type | Description |
|-------|------|-------------|
| `provider` | `Provider` | Which API backend to use |
| `transform` | `Transform` | Active token mutation strategy |
| `model` | `String` | Model name forwarded to the provider API |
| `rate` | `f64` | Fraction of tokens transformed (0.0–1.0) |
| `top_logprobs` | `u8` | Number of alternative tokens per position (OpenAI only) |
| `visual_mode` | `bool` | Enable ANSI colour output |
| `heatmap_mode` | `bool` | Enable importance heatmap colouring |
| `web_tx` | `Option<UnboundedSender<TokenEvent>>` | Fan-out channel for the web UI |
| `system_prompt` | `Option<String>` | Prepended system message |
| `max_retries` | `u32` | Retry budget for 429/5xx errors |
| `min_confidence` | `Option<f64>` | Gate transforms on per-token confidence |

---

### `TokenEvent`
*`src/lib.rs`*

A single processed token emitted by the pipeline. Serialised as a JSON object
and sent over SSE to the browser, printed in `--json-stream` mode, or stored in
replay recordings.

```json
{
  "text": "esreveR",
  "original": "Reverse",
  "index": 4,
  "transformed": true,
  "importance": 0.72,
  "confidence": 0.91,
  "perplexity": 1.09,
  "alternatives": [
    { "token": " Reverse", "probability": 0.88 },
    { "token": " Invert",  "probability": 0.05 }
  ]
}
```

| Field | Type | Notes |
|-------|------|-------|
| `text` | `String` | Possibly-transformed token shown to the user |
| `original` | `String` | Token before transformation |
| `index` | `usize` | Zero-based position in the response |
| `transformed` | `bool` | Whether the transform was applied |
| `importance` | `f64` | 0–1 scalar; API confidence or heuristic fallback |
| `confidence` | `Option<f32>` | Linear probability from `exp(logprob)`; `None` for Anthropic |
| `perplexity` | `Option<f32>` | `exp(-logprob)`; `None` when logprobs unavailable |
| `alternatives` | `Vec<TokenAlternative>` | Top-K alternatives (OpenAI `top_logprobs`) |
| `chaos_label` | `Option<String>` | Sub-transform chosen by `Chaos`; `None` otherwise |
| `provider` | `Option<String>` | `"openai"` or `"anthropic"` in diff mode |
| `is_error` | `bool` | `true` for synthetic error-notification events |

---

### `TokenAlternative`
*`src/lib.rs`*

One entry in the `alternatives` array of a `TokenEvent`.

| Field | Type | Notes |
|-------|------|-------|
| `token` | `String` | Alternative token string (may include leading space) |
| `probability` | `f32` | Linear probability in `[0.0, 1.0]`, computed as `exp(logprob)` |

---

### `Transform`
*`src/transforms.rs`*

Enum of all available token mutation strategies. Pass via `--transform` or
construct programmatically.

| Variant | Effect |
|---------|--------|
| `Reverse` | Reverse Unicode characters: `"hello"` → `"olleh"` |
| `Uppercase` | Uppercase all characters: `"hello"` → `"HELLO"` |
| `Mock` | Alternating case (SpongeBob): `"hello"` → `"hElLo"` |
| `Noise` | Append a random symbol (`* + ~ @ # $ %`) |
| `Chaos` | Randomly pick one of Reverse / Uppercase / Mock / Noise per token |
| `Scramble` | Fisher-Yates shuffle of characters |
| `Delete` | Drop the token entirely (returns empty string) |
| `Synonym` | Replace with a synonym from the 200-entry built-in map |
| `Delay(ms)` | Return unchanged after sleeping `ms` milliseconds |
| `Chain(vec)` | Apply a sequence of transforms in order |

**Parsing**

```rust
let t = Transform::from_str_loose("reverse").unwrap();
let t = Transform::from_str_loose("reverse,uppercase").unwrap(); // Chain
let t = Transform::from_str_loose("delay:50").unwrap();          // Delay(50)
```

**Applying**

```rust
let (output, label) = transform.apply_with_label("hello");
// output = "olleh", label = "reverse"
```

---

### `Provider`
*`src/providers.rs`*

Selects the LLM API backend.

| Variant | Description |
|---------|-------------|
| `Openai` | OpenAI Chat Completions API (GPT-3.5-Turbo, GPT-4, GPT-4o, …) |
| `Anthropic` | Anthropic Messages API (Claude family) |
| `Mock` | In-process fixture provider for tests and dry-run mode |

---

### `EotConfig`
*`src/config.rs`*

Optional file-based configuration loaded from `~/.eot.toml` (lower priority)
and `./.eot.toml` (higher priority, local wins).  All fields are `Option<T>`;
absent fields fall back to the CLI defaults.

```toml
provider          = "anthropic"
model             = "claude-sonnet-4-6"
transform         = "reverse"
rate              = 0.4
port              = 9000
top_logprobs      = 5
anthropic_max_tokens = 8192
system_a          = "Be concise."
api_key           = "sk-ant-..."   # optional override for the env var
```

---

## HTTP API

The web server started by `--web` listens on `localhost:<port>` (default 8888).

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Embedded single-page UI |
| `GET` | `/stream?prompt=...&transform=...&provider=...&model=...&rate=...` | SSE token stream |
| `GET` | `/diff-stream?prompt=...&transform=...` | Two-provider SSE stream |
| `GET` | `/ab-stream?prompt=...&sys_a=...&sys_b=...` | A/B system-prompt SSE stream |
| `POST` | `/room/create` | Create a multiplayer collaboration room |
| `GET` | `/join/:code` | Serve the join page for a room |
| `WS` | `/ws/:code` | WebSocket for real-time collaboration |
| `GET` | `/replay/:code` | JSON replay of a recorded session |
| `GET` | `/api/experiments?db=...` | List stored experiment rows (sqlite-log feature) |

### `/stream` query parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prompt` | *(required)* | The input text |
| `transform` | `reverse` | Transform strategy name |
| `provider` | `openai` | `openai` or `anthropic` |
| `model` | provider default | Model name |
| `rate` | `0.5` | Transform fraction (0.0–1.0) |
| `seed` | *(random)* | RNG seed for reproducibility |
| `top_logprobs` | `5` | Alternative tokens per position |
| `system` | *(none)* | System prompt |
| `visual` | `0` | `1` to enable ANSI colouring |
| `heatmap` | `0` | `1` to enable heatmap colouring |
| `room` | *(none)* | Collaboration room code |

### WebSocket inbound message types

```jsonc
{ "type": "set_name",  "name": "Alice" }
{ "type": "vote",      "transform": "reverse", "dir": "up" }
{ "type": "surgery",   "token_index": 4, "new_text": "hello", "old_text": "world" }
{ "type": "chat",      "text": "interesting!", "token_index": 4 }
{ "type": "record_start" }
{ "type": "record_stop" }
```

---

## CLI flags

See `every-other-token --help` for the full list. Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--transform` | `reverse` | Token mutation strategy |
| `--model` | `gpt-3.5-turbo` | Model name |
| `--provider` | `openai` | API provider |
| `--rate` | `0.5` | Transform fraction |
| `--rate-range` | *(none)* | e.g. `"0.2-0.8"` — random rate per run |
| `--seed` | *(entropy)* | Fixed RNG seed |
| `--top-logprobs` | `5` | Alternative tokens per position |
| `--web` | `false` | Launch the web UI instead of terminal output |
| `--port` | `8888` | Web UI TCP port |
| `--visual` / `-v` | `false` | ANSI colour output |
| `--heatmap` | `false` | Token importance heatmap |
| `--research` | `false` | Headless N-run research mode |
| `--runs` | `10` | Number of research iterations |
| `--output` | `research_output.json` | Research output path |
| `--json-stream` | `false` | One JSON line per token |
| `--system-a` | *(none)* | System prompt A (A/B mode) |
| `--system-b` | *(none)* | System prompt B (A/B mode) |
| `--diff-terminal` | `false` | Side-by-side terminal diff (OpenAI + Anthropic) |
| `--dry-run` | `false` | Show transform effects without calling any API |
| `--record` | *(none)* | Path to save a JSON token replay |
| `--replay` | *(none)* | Path to replay a saved session |
| `--max-retries` | `3` | Retry budget for 429/5xx errors |
| `--min-confidence` | *(none)* | Only transform tokens below this confidence |

---

## Feature flags

| Flag | Enables |
|------|---------|
| `sqlite-log` | Persist research runs to SQLite; exposes `/api/experiments` |
| `self-tune` | Telemetry bus + self-improvement controller |
| `self-modify` | Snapshot-based parameter mutation (requires `self-tune`) |
| `intelligence` | Reserved namespace for interpretability features |
| `evolution` | Reserved namespace for evolutionary optimisation |
| `self-improving` | All of the above combined |
| `helix-bridge` | HTTP bridge that polls a HelixRouter `/api/stats` endpoint |
| `redis-backing` | Write-through Redis persistence for snapshots |
| `wasm` | WASM target bindings via `wasm-bindgen` |

---

## Circuit breaker

The `execute_with_retry` helper integrates a process-wide circuit breaker.
After **5 consecutive failures** (configurable at compile time via
`CB_TRIP_THRESHOLD`) it opens for **30 seconds** (`CB_RECOVERY_MS`), rejecting
all API calls immediately with a clear error message.  A single successful
response resets the failure counter.

---

*Generated from source — for the authoritative rustdoc see <https://docs.rs/every-other-token>.*
