# Changelog

All notable changes to `every-other-token` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

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
- `.github/workflows/release.yml`: release workflow that triggers on `v*.*.*`
  tags, builds release binaries for Linux (musl), macOS, and Windows, publishes
  a GitHub Release with attached binaries, and publishes to crates.io.
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
- `README.md` rewritten: clearer description, feature table, complete quickstart,
  architecture ASCII diagram, contributing guide with development commands.
- `.github/workflows/ci.yml`: added `self-tune`, `self-modify`, and `helix-bridge`
  Clippy/test jobs; named all jobs; moved `RUSTDOCFLAGS` to workflow-level env.
- `Cargo.toml`: added `documentation` field; extended `categories` to include
  `development-tools`; updated `keywords` to include `interpretability`.

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
