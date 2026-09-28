<p align="center">
  <a href="https://mattbusel.github.io/Every-Other-Token/"><img src="https://raw.githubusercontent.com/Mattbusel/Every-Other-Token/main/assets/hero.svg" width="100%" alt="every-other-token streaming a reply token by token: every other token is reversed and highlighted, with a confidence bar under each token"></a>
</p>

<h1 align="center">every-other-token</h1>

<p align="center"><b>See how sure an AI model is about every single word it writes, and change its words while it is still writing.</b></p>

<p align="center">
  <a href="https://github.com/Mattbusel/Every-Other-Token/releases/latest/download/every-other-token-windows-x86_64.exe"><b>Download for Windows (.exe)</b></a> &nbsp;&middot;&nbsp;
  <a href="#install">macOS and Linux</a> &nbsp;&middot;&nbsp;
  <a href="https://mattbusel.github.io/Every-Other-Token/">Project site</a> &nbsp;&middot;&nbsp;
  <a href="#documentation">Docs</a>
</p>

<p align="center">
  <a href="https://crates.io/crates/every-other-token"><img src="https://img.shields.io/crates/v/every-other-token.svg?color=ff6a2b&labelColor=0c0d0b" alt="crates.io version"></a>
  <a href="https://github.com/Mattbusel/Every-Other-Token/actions/workflows/ci.yml"><img src="https://github.com/Mattbusel/Every-Other-Token/actions/workflows/ci.yml/badge.svg" alt="CI status"></a>
  <a href="https://docs.rs/every-other-token"><img src="https://img.shields.io/docsrs/every-other-token?labelColor=0c0d0b" alt="docs.rs"></a>
</p>

`every-other-token` is a free LLM token stream viewer and interceptor for the command line and the browser. It sits on the live OpenAI or Anthropic stream, shows the confidence and perplexity of each token from the logprobs, and can rewrite every other token (or any fraction) as it arrives. A built-in mock provider lets you try all of it with no API key.

**Who it's for:** anyone curious how language models pick their words, plus people doing LLM interpretability research, red-teaming and prompt engineering.

## How it works

<img src="https://raw.githubusercontent.com/Mattbusel/Every-Other-Token/main/docs/img/how-it-works.svg" width="100%" alt="Diagram of the four stages on a real run: 1 Intercept reads the live stream, 2 Score gives each token confidence exp(logprob) and perplexity exp(-logprob), 3 Mutate reverses the odd-numbered tokens, 4 Output shows 'The kciuq brown xof jumps revo the yzal dog' in the terminal, web UI or JSON">

1. **Intercept.** It opens the model's streaming connection (SSE) itself, so it sees each chunk the moment it arrives.
2. **Score.** Each token gets `confidence = exp(logprob)` and `perplexity = exp(-logprob)`, plus the top alternatives the model considered.
3. **Mutate.** The chosen tokens (every other one by default, or only the ones the model was unsure about) go through a transform: reverse, uppercase, noise, delete, synonym and more.
4. **Output.** You watch it in the terminal or the web UI, or save it as JSON lines, CSV or an HTML heatmap.

## Install

| Platform | How |
|---|---|
| **Windows** | [**Download every-other-token-windows-x86_64.exe**](https://github.com/Mattbusel/Every-Other-Token/releases/latest/download/every-other-token-windows-x86_64.exe), then double-click it. The web UI opens in your browser. (Unsigned, so SmartScreen may ask: *More info*, then *Run anyway*.) |
| **macOS** (Apple Silicon) | [every-other-token-aarch64-apple-darwin.tar.gz](https://github.com/Mattbusel/Every-Other-Token/releases/latest/download/every-other-token-aarch64-apple-darwin.tar.gz) |
| **macOS** (Intel) | [every-other-token-x86_64-apple-darwin.tar.gz](https://github.com/Mattbusel/Every-Other-Token/releases/latest/download/every-other-token-x86_64-apple-darwin.tar.gz) |
| **Linux** (x86_64) | [every-other-token-x86_64-unknown-linux-gnu.tar.gz](https://github.com/Mattbusel/Every-Other-Token/releases/latest/download/every-other-token-x86_64-unknown-linux-gnu.tar.gz) |
| **Scoop** (Windows) | `scoop bucket add mattbusel https://github.com/Mattbusel/scoop-bucket` then `scoop install every-other-token` |
| **Homebrew** (macOS, Linux) | `brew install mattbusel/tap/every-other-token` |
| **Cargo** (any OS with Rust) | `cargo install every-other-token` |

No Rust toolchain is needed for the downloads. Every release also lists versioned archives with SHA-256 checksums on the [releases page](https://github.com/Mattbusel/Every-Other-Token/releases/latest).

## Examples

All of these are real output from the mock provider (a fixed reply with fixed logprobs, run through the real pipeline), so you can reproduce them exactly without an API key.

**1. Reverse every other token** (the default)

```console
$ every-other-token "Why is the sky blue?" --provider mock
The kciuq brown xof jumps revo the yzal dog. This si a kcom response rof prompt: Why si the yks blue?
24 tokens streamed, 12 transformed
```

**2. Rewrite only the words the model was unsure about** (confidence at or below 0.6)

```console
$ every-other-token "Why is the sky blue?" uppercase --provider mock --rate 1 --min-confidence 0.6
The quick BROWN fox JUMPS over the LAZY dog. THIS is a mock response for PROMPT: WHY IS THE SKY BLUE?
24 tokens streamed, 11 transformed
```

"brown" (0.46), "jumps" (0.57), "lazy" (0.41), "This" (0.54) and "prompt" (0.59) were below the line. The echoed prompt at the end carries no logprobs in the mock, so it falls back to the plain rate.

**3. Get the numbers as JSON, one line per token**

```console
$ every-other-token "Why is the sky blue?" uppercase --provider mock --json-stream
{"text":"The","original":"The","index":0,"transformed":false,"importance":0.8869204521179199,"confidence":0.88692045,"perplexity":1.1274968,"is_error":false,"arrival_ms":0}
{"text":" QUICK","original":" quick","index":1,"transformed":true,"importance":0.6376281380653381,"confidence":0.63762814,"perplexity":1.5683122,"is_error":false,"arrival_ms":0}
...
```

**4. Watch it in the browser** with `every-other-token --web --provider mock` (or just double-click the .exe). Split view: the original stream on the left, the rewritten one on the right, each token underlined by its confidence.

<img src="https://raw.githubusercontent.com/Mattbusel/Every-Other-Token/main/assets/web-ui.png" width="100%" alt="every-other-token web UI in split view: original token stream on the left, transformed stream on the right, each token underlined by its confidence">

## Use it in 3 steps

1. **Get it.** Download the .exe above (or `brew`, `scoop`, `cargo install`).
2. **Try it offline.** Run `every-other-token "Why is the sky blue?" --provider mock`, or double-click the .exe and pick **Mock (no API key)** in the provider menu.
3. **Point it at a real model.** Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, then run `every-other-token "Why is the sky blue?" --visual` (terminal, colored by confidence) or `every-other-token --web` (browser). Real confidence numbers come from OpenAI logprobs; Anthropic's stream has none, so there the tool uses its own importance score.

Run `every-other-token --help` for every flag, with examples at the bottom.

## Documentation

| Doc | What's in it |
|---|---|
| [Usage guide](docs/USAGE.md) | All transforms, `--rate` and `--min-confidence`, A/B system prompt tests, provider diff, research mode, web UI views, config file, shell completions, build from source |
| [Library reference](docs/REFERENCE.md) | The Rust modules behind the CLI (attribution export, drift detection, mutation lab, causal maps and more) with examples |
| [Architecture](docs/ARCHITECTURE.md) | How the stream, transforms and outputs fit together |
| [HTTP API](docs/api.md) and [WebSocket rooms](docs/websocket.md) | The web server endpoints and collaborative token editing |
| [Feature flags](docs/features.md) | Optional Cargo features |
| [docs.rs](https://docs.rs/every-other-token) | Full API docs for using it as a library |
| [Changelog](CHANGELOG.md) and [Contributing](CONTRIBUTING.md) | Release history and how to send a change |

## License

MIT, see [LICENSE](LICENSE).

## Hire the author

**Need this kind of engineering on your product?** I take on a small number of client builds: LLM features, iOS apps and performance work, fixed price. [Services and pricing](https://mattbusel.github.io/) · [Email](mailto:mattbusel@gmail.com) · [LinkedIn](https://www.linkedin.com/in/matthewbusel/)
