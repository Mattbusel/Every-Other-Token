# Architecture

This document describes the internal design of `every-other-token` in enough detail to onboard a new contributor or reproduce the system from scratch.

---

## Token Interception Mechanism

The core component is `TokenInterceptor` in `src/lib.rs`. It manages a single stateful connection to one provider (OpenAI or Anthropic) and emits `TokenEvent` structs as tokens arrive.

### SSE Streaming

Both providers expose a server-sent events (SSE) endpoint that emits newline-delimited JSON chunks as the model generates tokens:

- **OpenAI** uses `POST /v1/chat/completions` with `"stream": true`. Each chunk is a `data: <json>` line containing `choices[0].delta.content` (the token text) and, when `logprobs: true` is set, `choices[0].logprobs.content[0].logprob` (log-probability) and `top_logprobs` (up to 20 alternative tokens with their log-probabilities).

- **Anthropic** uses `POST /v1/messages` with `"stream": true`. Each chunk of type `content_block_delta` contains `delta.text`. Anthropic does not expose log-probabilities, so confidence and perplexity are unavailable when using this provider.

The interception loop uses `reqwest` with the `stream` feature. The response body is consumed as an async byte stream via `tokio-stream`. Each `data:` line is extracted, the `[DONE]` sentinel is detected, and the JSON payload is parsed with `serde_json`.

### Request Structure

The `ProviderPlugin` trait (`src/providers.rs`) abstracts the two providers:

```
pub trait ProviderPlugin: Send + Sync {
    fn name(&self) -> &str;
    fn default_model(&self) -> &str;
    fn api_url(&self) -> &str;
    fn build_request(&self, prompt: &str, system: Option<&str>, model: &str) -> serde_json::Value;
}
```

`OpenAiPlugin` and `AnthropicPlugin` are zero-sized structs that implement this trait. The interceptor holds a boxed plugin selected at construction time. This allows the main loop to remain provider-agnostic.

### Retry Logic

The HTTP request is wrapped in `with_retry(max_retries, ...)`. On HTTP 429 or 5xx responses, the loop sleeps for `2^attempt` seconds (capped at 32 s) before retrying. The retry count defaults to 3 and is configurable via `--max-retries`.

---

## Confidence Scoring Algorithm

OpenAI returns the log-probability of each generated token. The log-probability `lp` is in the range `(-inf, 0]`, where 0 means the model assigned probability 1.0 to that token.

Two derived quantities are computed per token:

```
confidence  = exp(lp)          // in [0.0, 1.0]
perplexity  = exp(-lp)         // in [1.0, +inf)
```

These are stored in `TokenEvent.confidence` and `TokenEvent.perplexity`. The `render` module classifies confidence into three bands:

| Band | Threshold | Terminal colour |
|------|-----------|-----------------|
| High | >= 0.7    | Green           |
| Mid  | 0.4-0.7   | Yellow          |
| Low  | < 0.4     | Red             |

The `top_logprobs` field (up to 20 alternatives) is also captured per token and stored in `TokenEvent.alternatives`. In visual mode, the top 3 are printed inline.

The importance score used for heatmap colouring is computed by `calculate_token_importance` in `src/transforms.rs`. It is a heuristic combining token length, position in the sequence, whether any character is uppercase, keyword matching against a list of high-information words, and random jitter. It is not derived from logprobs.

---

## The Five Transform Strategies

All transforms are applied only at positions selected by the perturbation rate (default 0.5, i.e. every other token). The Bresenham spread algorithm ensures the positions are uniformly distributed rather than clustered.

### Reverse

Reverses the Unicode codepoints of the token. `"hello"` becomes `"olleh"`. For multi-byte characters this operates at codepoint granularity, not byte granularity.

Research use: measures whether the model can recover syntactic coherence when word fragments are inverted.

### Uppercase

Converts every codepoint to uppercase using Rust's Unicode-aware `str::to_uppercase`. `"hello"` becomes `"HELLO"`.

Research use: isolates the effect of capitalisation on model confidence. GPT-family models are trained on mixed-case text so capitalisation shift is a mild perturbation.

### Mock

Alternates lowercase and uppercase per character index. Character at even index is lowercased; character at odd index is uppercased. `"hello"` becomes `"hElLo"`.

Research use: a subtler perturbation than full uppercase. The token still contains the same characters but the casing pattern is unusual.

### Noise

Appends one character sampled uniformly from `['*', '+', '~', '@', '#', '$', '%']`. `"hello"` becomes `"hello*"` (or similar).

Research use: tests sensitivity to extraneous trailing characters. The appended symbol is likely out-of-distribution for the model's tokeniser at that position.

### Chaos

Selects one of the four strategies above uniformly at random for each targeted token. The selected sub-strategy name is recorded in `TokenEvent.chaos_label`.

Research use: produces the highest entropy perturbation pattern. Useful for baseline resilience studies. In research mode, `per_transform_perplexity` breaks down the mean perplexity per sub-strategy.

---

## WebSocket Collaboration Protocol

The collaboration system is implemented in `src/collab.rs`. It provides multi-participant rooms where a live token stream can be observed, edited, and discussed in real time.

### Room Lifecycle

1. Client calls `POST /room/create`. The server generates a 6-character alphanumeric room code, creates a `Room` struct in the shared `RoomStore` (`Arc<Mutex<HashMap<String, Room>>>`), and returns the code as JSON.
2. The host connects to `GET /ws/:code` via WebSocket. The handler upgrades the HTTP connection using `tokio-tungstenite`. The participant is assigned the first colour from `PARTICIPANT_COLORS` and receives the `host` role.
3. Guests open `/join/:code` in a browser (served as HTML) and connect to `GET /ws/:code`. Each receives a unique UUID, the next available colour, and the `guest` role.
4. The host starts a token stream (via the web UI). Each `TokenEvent` is broadcast to all room participants via a `tokio::sync::broadcast` channel. The channel capacity is 1,024 events; lagging readers are dropped and rejoin cleanly.
5. Any participant can send a `surgery` message to edit a token already in the stream. The edited event is broadcast to all participants.
6. Participants send `chat` messages (plain text, stored in the room log up to 10,000 entries) and `vote` messages to accumulate transform preference votes.

### Message Format

All WebSocket messages are JSON objects. The `type` field discriminates the message kind:

| Type | Direction | Description |
|------|-----------|-------------|
| `token` | Server -> Client | A `TokenEvent` wrapped with provider label |
| `chat` | Client -> Server, Server -> All | A chat message |
| `surgery` | Client -> Server, Server -> All | A token edit with index and new text |
| `vote` | Client -> Server, Server -> All | A transform preference vote |
| `participant_joined` | Server -> All | New participant joined |
| `participant_left` | Server -> All | Participant disconnected |

### Idle TTL Eviction

A background task periodically scans the `RoomStore` and removes rooms that have not been mutated in `ROOM_IDLE_TTL_MS` (1 hour). This prevents unbounded memory growth in long-running server processes.

---

## Research Mode Design

Research mode (`src/research.rs`) runs `N` independent token streams against the live provider API and aggregates statistics across runs. It is designed for headless batch execution.

### Run Structure

Each run creates a fresh `TokenInterceptor`, streams the prompt to completion, and drains the `mpsc::unbounded_channel` that receives `TokenEvent`s. The following per-run metrics are computed:

- `token_count`: total tokens in the response.
- `transformed_count`: tokens where the transform was applied.
- `avg_confidence`: mean of `exp(logprob)` over all tokens that have a logprob.
- `avg_perplexity`: mean of `exp(-logprob)` over all tokens that have a logprob.
- `vocab_diversity`: ratio of unique original tokens to total tokens (type-token ratio).
- `collapse_positions`: starting positions of runs of `collapse_window` or more consecutive tokens with confidence below 0.4.
- `per_transform_perplexity`: in Chaos mode, the mean perplexity per sub-strategy label.
- `elapsed_ms`: wall-clock duration of the run.
- `tokens_per_second`: derived from `token_count / elapsed_ms`.

### Aggregate Statistics

After all runs, `build_aggregate` computes cross-run means, sample standard deviations (Bessel's correction), and 95% confidence intervals using the Z-score approximation `mean +/- 1.96 * sd / sqrt(n)`. The Z-score approximation is valid when `n >= 30` (central limit theorem); a warning is printed for smaller samples.

### A/B Comparison

When `--system-a` and `--system-b` are set, even-indexed runs use system prompt A and odd-indexed runs use system prompt B. After all runs, a Welch two-sample t-test is run on the even-run vs odd-run mean confidence values. The p-value is computed using the large-sample normal approximation.

Auto-baseline comparison (even vs odd) runs unconditionally whenever there are at least two runs, even without `--significance`.

### Suite Mode

`run_research_suite` reads a prompt file (one prompt per line, `#` for comments) and calls `run_research_for_prompt` for each. Output files are named `output_0.json`, `output_1.json`, etc. This enables batch studies across many prompts without manual invocation.

### SQLite Store

When `--db path.sqlite` is provided (requires `sqlite-log` feature), each experiment and each run is persisted to a SQLite database. The `ExperimentStore` supports cross-session baseline comparison: a prior run with `--transform none` can be loaded and compared against a current perturbed run.
