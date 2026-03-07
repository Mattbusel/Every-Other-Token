# Contributing to Every-Other-Token

This project is primarily a research tool for token-level LLM analysis. Contributions that extend research capability, add providers, improve the web UI, or expand the test suite are welcome.

## What we want

- **New providers** — Mistral, Gemini, local Ollama with logprob support
- **New transforms** — semantically interesting token mutation strategies
- **Research features** — new metrics, export formats, comparison modes
- **Web UI improvements** — new visualization modes, better diff views
- **Bug fixes** — especially anything that breaks the token confidence pipeline

## What we don't want

- Changes that hide logprob data or reduce analysis fidelity
- PRs that replace `Result` returns with `unwrap()`
- New dependencies without clear justification

## How to contribute

1. Fork and clone
2. `cargo test --all-features` — must pass
3. `cargo clippy -- -D warnings` — zero warnings
4. Open a PR with a clear description of the research motivation

## Research ideas

If you're using this for research or have ideas for new experiments, open a [Discussion](https://github.com/Mattbusel/Every-Other-Token/discussions). Good starting points:

- What token positions are most sensitive to perturbation?
- Do different model sizes show different fragility patterns?
- Can perplexity trajectories predict response quality?

## Questions

Open a [Discussion](https://github.com/Mattbusel/Every-Other-Token/discussions).
