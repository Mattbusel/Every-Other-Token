# Contributing to Every-Other-Token

This project is primarily a research tool for token-level LLM analysis. Contributions that extend research capability, add providers, improve the web UI, or expand the test suite are welcome.

## What we want

- **New providers**: Mistral, Gemini, local Ollama with logprob support
- **New transforms**: semantically interesting token mutation strategies
- **Research features**: new metrics, export formats, comparison modes
- **Web UI improvements**: new visualization modes, better diff views
- **Bug fixes**: especially anything that breaks the token confidence pipeline

## What we don't want

- Changes that hide logprob data or reduce analysis fidelity
- PRs that replace `Result` returns with `unwrap()`
- New dependencies without clear justification

## How to contribute

1. Fork and clone.
2. Try it without an API key: `cargo run --example mock_stream`, or
   `cargo run -- "hello" --provider mock --visual`.
3. `cargo test` must pass (this is what CI runs). If you touch a feature-gated
   module, also run `cargo check --no-default-features --features self-improving`
   (or the feature you changed).
4. `cargo clippy` should not report new warnings in the code you changed. The
   codebase has existing warnings, so `-D warnings` is not enforced yet;
   cleaning some up is a welcome first PR.
5. New behaviour needs a test; bug fixes need a regression test.
6. Open a PR with a clear description of what changed and why.

Issues labelled [good first issue](https://github.com/Mattbusel/Every-Other-Token/labels/good%20first%20issue) are small, self-contained starting points.

## Releases

Maintainers bump `version` in `Cargo.toml`, add a `CHANGELOG.md` entry, merge,
then push a tag such as `v4.2.0`. `.github/workflows/release.yml` builds the
Linux, macOS and Windows binaries and attaches them to the GitHub Release.

## Research ideas

If you're using this for research or have ideas for new experiments, open a [Discussion](https://github.com/Mattbusel/Every-Other-Token/discussions). Good starting points:

- What token positions are most sensitive to perturbation?
- Do different model sizes show different fragility patterns?
- Can perplexity trajectories predict response quality?

## Questions

Open a [Discussion](https://github.com/Mattbusel/Every-Other-Token/discussions).
