# Changelog

All notable changes to `every-other-token` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [5.0.0] - 2026-10-04

4.4.0 was tagged in git but never published to crates.io; its changes ship in this release too.

### Added
- **A model running inside the program: `--provider local`.** Build with `--features local` and the tool downloads a small open model once (SmolLM2-135M-Instruct, 269 MB, into your cache directory) and runs it on the CPU with [candle](https://github.com/huggingface/candle). Every token comes with its exact log-probability and the model's real top-5 alternatives, with no API key and no server. `--model` takes another Llama-architecture model from the Hugging Face Hub; `--seed N` samples reproducibly instead of greedy decoding; `--local-max-tokens` caps the reply.
- **What did the answer depend on? `--attribute`.** With `--provider local`, after the reply the tool removes each prompt word in turn, re-scores the same reply with the model, and shows how much the reply's probability fell (occlusion attribution). For "Reply with only the city name. Capital of France?" it ranks France (+4.69 nats), city (+3.50) and Capital (+3.23) far above "the" (+0.36). Library: `local_model::LocalModel::{generate, score, occlusion}`.
- Tests against the real model (`tests/local_model_tests.rs`, ignored by default because of the download), including one that checks re-scoring the model's own answer reproduces exactly the log-probabilities it reported while generating.

### Changed (breaking)
- **No invented confidence.** Providers that return no token probabilities (Anthropic, Gemini, local servers without logprobs) used to get a "confidence" made up from the time between network chunks, shown and exported exactly like a real probability. Those tokens now have no confidence, and the terminal says why and how to get real numbers (`--provider openai` or `--provider local`).
- `Provider` has a `Local` variant and is `#[non_exhaustive]`.
- `attention::AttentionTracer::approximate_from_logprobs` is deprecated: its weights depend only on position, so it never measured attribution. The module docs now say it does arithmetic on log-probabilities you supply and point to `LocalModel::occlusion` for the real thing.

### Fixed
- Running the test suite opened a browser tab: the web server test did not pass `--no-open`.

## [4.4.0] - 2026-10-02

This release swaps several hand-written parts for well-known open-source crates and adds what they made easy.

### Added
- **More providers.** `--provider ollama` streams from a local Ollama with no API key. `--provider openrouter` (`OPENROUTER_API_KEY`) and `--provider gemini` (`GEMINI_API_KEY`) reach hosted models. All three are also in the web UI's provider menu.
- **Any OpenAI-compatible server.** `--base-url URL` (or `base_url` in `.eot.toml`) points a provider at another server, such as llama.cpp, vLLM or LM Studio. With it the API key is optional. `--validate-config` now prints the endpoint it will call.
- **Full-screen terminal view.** `--tui` shows the reply colored by confidence, rewritten tokens underlined, running stats, a confidence sparkline and the latest token's top alternatives. Built on [ratatui](https://ratatui.rs).
- **JSON-lines export.** `--export-logprobs tokens.jsonl` writes one JSON object per token, with every field.
- For library users: `TokenInterceptor::new_with_base_url`, `with_base_url`, `with_api_key`, `endpoint_url`, and `Provider::{default_base_url, endpoint_url, api_key_env, default_model}`.

### Fixed
- **Streams no longer lose text.** The stream reader decoded each network chunk on its own, so an accented letter or emoji split across two chunks made the whole chunk get skipped. SSE is now parsed by [eventsource-stream](https://crates.io/crates/eventsource-stream), which also accepts `data:` without a space and CRLF line endings.
- **Errors inside a stream are reported.** An error event in the middle of a reply (for example Anthropic's "overloaded") used to be ignored, ending the run with a partial answer and no message. It now fails the run with the server's message. HTTP errors also show the status code.
- **Research-mode statistics are correct for small samples.** The A/B t-test used a normal approximation and the 95% confidence intervals used 1.96, both only right for large samples (the default is 10 runs). For five runs a side the old p-value could read 0.058 where the exact value is 0.108. Both now use Student's t distribution from [statrs](https://crates.io/crates/statrs).
- **Retries wait the right amount.** Retries now use [backon](https://crates.io/crates/backon): exponential back-off with jitter, and a `Retry-After` header from the server sets the wait. Anthropic's 529 "overloaded" status is retried too.
- **CSV exports quote properly.** The logprob, timeseries and attribution CSV files are written with the [csv](https://crates.io/crates/csv) crate.

### Changed
- `--export-logprobs` CSV: a missing logprob is now an empty cell instead of `-inf`, and four columns are added at the end (`original`, `transformed`, `confidence`, `perplexity`). The first five columns are unchanged.
- Providers without logprobs other than Anthropic (Gemini, most local servers) get the same token-timing confidence estimate Anthropic already used. (Removed again in 5.0.0: that estimate does not reflect the model.)
- `--max-retries 0` used to skip the request entirely; it now makes one attempt.
- Minimum Rust version is now 1.89 (the previous 1.81 was already out of date: the locked dependencies needed 1.85). The prebuilt downloads are not affected.
- HTTPS is still rustls only; none of the new crates bring in OpenSSL.

## [4.3.2] - 2026-09-30

- HTTPS through rustls instead of the system OpenSSL. The 4.3.1 Linux download needed OpenSSL 1.1, which Ubuntu 22.04 and newer do not ship; this one runs on any glibc 2.31+ distribution.

## [Unreleased]

## [4.3.1] - 2026-09-28

### Fixed
- `--version` printed 4.0.0; it now reports the real crate version (research citations too).
- `--help` explains the tool in one plain sentence, lists every transform and the mock provider, and ends with copy-paste examples.
- Starting with no arguments (double-clicking the Windows .exe) says how to try it without an API key.

### Changed
- Releases also attach versionless files (`every-other-token-windows-x86_64.exe` and per-platform archives) so `releases/latest/download/...` links stay valid.
- README cut to a one-screen overview with a "How it works" diagram and real examples; the full reference moved to `docs/`.

## [4.3.0] - 2026-09-25

### Fixed
- Web UI and `--json-stream` keep the spaces between words with real providers (#3).
- Web UI no longer drops tokens that arrive before the end-of-stream event, so fast streams render.
- Choosing the mock provider in the web UI uses the mock instead of silently falling back to OpenAI.
- `--json-stream` output is pure JSON: the human header and footer are no longer mixed in.
- The mock provider's reply reads as a sentence and echoes up to 40 characters of the prompt.

### Added
- `--no-open` flag so `--web` does not open a browser tab.
- Web UI "Mock (no API key)" provider option, confidence underlines per token, restyled terminal header and footer.
- Animated README hero and project site at https://mattbusel.github.io/Every-Other-Token/.

## [4.2.0] - 2026-09-25

First release since 4.1.2. It carries the library modules added since then
(divergence detection, logit lens, activation patching, circuit discovery,
Thompson-sampling bandit, checkpointing, SSE backpressure, batch research mode
and the many analysis modules added in rounds 3 to 32; see `git log v4.1.2..v4.2.0`),
plus the fixes below.

### Added

- `examples/mock_stream.rs`: `cargo run --example mock_stream` runs the real
  interceptor against the mock provider (no API key) and prints a per-token
  table of original and intercepted text, confidence and perplexity.
- `.github/workflows/release.yml`: pushing a `vX.Y.Z` tag builds binaries for
  Linux x86_64, macOS arm64 and x86_64, and Windows x86_64 and attaches them,
  with SHA-256 checksums, to a GitHub Release. crates.io publishing is manual.

### Fixed

- Mock provider no longer panics on prompts whose 20th byte falls inside a
  multi-byte UTF-8 character.
- Mock provider in terminal mode counted every token twice, so the CLI
  transformed every token instead of every other one.
- Terminal output now keeps the spaces between words (whitespace tokens were
  dropped, so words ran together).
- `StanceClassifier` matched keywords as substrings, so "disagree" counted as
  support; it now matches words.
- `NgramModel::generate` is reproducible for a given seed and no longer stops
  early at a context with no observed continuation.
- `SimilarityEngine` uses smoothed IDF, so terms present in every document
  (including any term of a one-document corpus) no longer get zero weight.
- Test suite compiles and passes again (stale `cli::Args` literals, doctests
  referencing APIs that did not exist, wrong expectations in two tests).

## Earlier unreleased notes (included in 4.2.0)

### Added

- Module-level `//!` doc comments on all previously undocumented public modules
  (`providers`, `transforms`, `store`, `web`, `research`, `heatmap`, `cli`).
- Field-level `///` doc comments on `ResearchRun`, `ResearchOutput`, and
  `ResearchAggregate` structs.
- `documentation` key in `Cargo.toml` pointing to `docs.rs`.
- `crates.io` and `docs.rs` badges in `README.md`.
- `///` doc comments on `TokenInterceptor` struct, `TokenAlternative`,
  `TokenEvent`, `ResearchSession`, `HeatmapExporter::new`,
  `TokenInterceptor::print_header`, and `TokenInterceptor::print_footer`.
- Field-level `///` doc comments on all public fields of `TokenAlternative`,
  `TokenEvent`, and `ResearchSession`.
- Unit tests for `TokenInterceptor::with_rate` covering clamping at 0.0 and 1.0
  and Bresenham-spread behaviour at the boundary rates.
- Unit tests for `TokenInterceptor::with_seed` verifying deterministic output
  from the Noise transform when the same seed is supplied twice.
- Async unit tests for `run_research_headless` using the Mock provider (no API
  key required): empty-prompt error path, multi-run accumulation, vocab
  diversity bounds, and citation string format.
- `EotConfig` struct and all its fields now carry `///` doc comments explaining
  each configuration option and its valid range.
- `make_test_interceptor` helper added to the `research_tests` module so
  `with_rate` and `with_seed` tests can construct interceptors without importing
  the private `tests` module helper.

### Fixed

- `eprintln!` calls in `config.rs` replaced with `tracing::warn!` structured log
  events, matching the rest of the codebase's structured logging style.
- `research_tests` module tests calling `make_test_interceptor` now resolve
  correctly; the function is defined locally in that module.

### Changed
- `README.md` comprehensively rewritten: what it does, architectural pipeline
  diagram, feature flag table, detailed quickstart with all common invocations,
  full CLI reference table, library API examples, performance notes, contributing
  guide with all dev commands, and annotated project layout tree.
- `.github/workflows/ci.yml`: added `self-tune`, `self-modify`, `helix-bridge`
  Clippy/test jobs; `release-build` job verifying the release binary is produced
  on every push; `--all-features` Clippy pass; multi-feature doc build; named all
  jobs; moved `RUSTDOCFLAGS` to workflow-level env.
- `Cargo.toml`: added `documentation` field; extended `categories` to include
  `development-tools`; updated `keywords` to include `interpretability`; added
  `opt-level = 3` and `strip = true` to `[profile.release]`.

### Added (production hardening, 2026-03-18)
- External integration test suites `tests/transforms_tests.rs` and
  `tests/store_heatmap_replay_tests.rs` covering `Transform`, `ExperimentStore`,
  `HeatmapExporter`, `Recorder`, and `Replayer` from outside the crate boundary.
- `tracing::info_span!` on `TokenInterceptor::intercept_stream` entry point with
  provider, model, transform, and rate as structured span fields.
- `tracing::info!` / `tracing::warn!` in `research::run_research`,
  `run_research_suite`, and `run_diff_terminal`; `tracing::info!` in `web::serve`.
- `release-build` CI job: verifies `cargo build --release` succeeds on every push.
- `[profile.release]` now includes `opt-level = 3` and `strip = true`.

### Fixed (2026-03-18)
- Removed duplicate `tracing::warn!` call in `execute_with_retry` that emitted the
  same warning twice per retryable HTTP status (copy-paste regression).

---

## [4.0.0] – 2025-07-12

### Added
- **Full module suite**: `self_tune`, `self_modify`, `semantic_dedup`, `helix_bridge`, `experiment_log`
  behind feature flags so the default build stays lean.
- **ProviderPlugin trait**: zero-sized plugin structs (`OpenAiPlugin`, `AnthropicPlugin`) centralise
  endpoint URLs and request construction.
- **Chain transform**: comma-separated transform pipeline (e.g. `reverse,uppercase`).
- **Scramble / Delete / Synonym transforms** with full test coverage.
- **Delay transform**: injects configurable per-token latency for pacing experiments.
- **Chaos transform**: randomly selects a sub-transform per token; label recorded on the event.
- **Rate control**: `--rate` flag with Bresenham-spread selection for deterministic uniform distribution.
- **Seeded RNG**: `--seed` for fully reproducible Noise / Chaos / Scramble runs.
- **`--rate-range`**: stochastic rate selection from a `MIN-MAX` interval.
- **`--dry-run`**: validate transform and show sample mutations without calling any API.
- **`--template`**: `{input}` prompt substitution with injection-safe split/join logic.
- **`--min-confidence`**: skip transforms on high-confidence tokens.
- **`--diff-terminal`**: parallel OpenAI + Anthropic streams with live diff output.
- **`--json-stream`**: one JSON line per token for pipeline integration.
- **`--max-retries`**: configurable exponential back-off on 429 / 5xx responses.
- **`--baseline`**: compare current run against stored "none" transform runs in SQLite.
- **`--significance`**: Welch's t-test across A / B confidence distributions.
- **`--heatmap-export`** / `--heatmap-sort-by` / `--heatmap-min-confidence`: per-position confidence CSV export.
- **`--record` / `--replay`**: deterministic token-event capture and replay.
- **`--prompt-file`**: batch research across multiple prompts.
- **`--format jsonl`**: newline-delimited JSON output for streaming pipelines.
- **`--collapse-window`**: configurable confidence-collapse detection window.
- **Config file** (`~/.eot.toml` + `./.eot.toml`) with merge semantics and rate clamping.
- **Shell completions** via `--completions <SHELL>`.
- **Web UI** (`--web`): embedded SPA with Single / Split / Quad / Diff / Experiment / Research modes.
- **Collab rooms**: WebSocket-based multiplayer with surgery edits, chat, voting, and session recording.
- **SQLite experiment store** with atomic `insert_experiment_with_run` transaction.
- **Cross-session dedup cache** in SQLite with TTL eviction.
- **HeatmapExporter**: multi-run confidence matrix to CSV.
- **Recorder / Replayer**: JSON replay file serialisation.
- **`cargo doc`** step in CI with `-D warnings` to keep docs buildable.
- **134 + 1 000 + tests** across all modules.

### Changed
- Module layout restructured: `CMakeLists.txt` equivalents removed from non-standard locations.
- `TradeSignal` / `TokenInterceptor` field naming aligned with stable naming conventions.

### Fixed
- `unwrap()` on safe-but-unchecked paths replaced with `?`-propagation or guarded `if let`.
- `run_start.unwrap()` in collapse detector replaced with `if let Some`.
- `transforms::from_str_loose` single-element path now uses `ok_or_else` instead of `unwrap`.

---

## [3.0.0] – 2025-06-01

### Added
- Anthropic provider support with SSE streaming.
- Per-token logprob / confidence / perplexity tracking.
- Top-K alternative tokens in visual mode.
- Research mode aggregate statistics (mean, std dev, 95 % CI).
- A/B experiment support with system prompt alternation.

---

## [2.0.0] – 2025-04-15

### Added
- OpenAI SSE streaming with per-token logprobs.
- Visual mode with ANSI confidence colour bands.
- Heatmap mode (importance scoring + terminal colouring).
- Headless research mode writing JSON output.

---

## [1.0.0] – 2025-03-01

### Added
- Initial release: token interception with Reverse / Uppercase / Mock / Noise transforms.
- OpenAI Chat Completions streaming via `reqwest`.
- `--visual` flag for coloured terminal output.
