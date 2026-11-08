# every-other-token

English | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

[![crates.io](https://img.shields.io/crates/v/every-other-token.svg?color=ff6a2b&labelColor=0c0d0b)](https://crates.io/crates/every-other-token) [![docs.rs](https://img.shields.io/docsrs/every-other-token?labelColor=0c0d0b)](https://docs.rs/every-other-token) [![CI](https://github.com/Mattbusel/Every-Other-Token/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/Every-Other-Token/actions/workflows/ci.yml)

See how sure an LLM is about each word.

One binary, 1,165 tests.

- Shows each token's confidence live
- Rewrites every other token as it writes
- Runs OpenAI, Anthropic, Gemini, Ollama
- Tries offline with no API key

## Install

```sh
cargo install --locked every-other-token
```

## Use

```console
$ every-other-token "Why is the sky blue?" --provider mock
The kciuq brown xof jumps revo the yzal dog. This si a kcom response rof prompt: Why si the yks blue?
24 tokens streamed, 12 transformed
```

[Try it in the browser](https://every-other-token.vercel.app/play/) · [Windows .exe](https://gitlab.com/mattbusel/Every-Other-Token/-/releases/permalink/latest/downloads/every-other-token-windows-x86_64.exe) · [crates.io](https://crates.io/crates/every-other-token) · [docs.rs](https://docs.rs/every-other-token)

More: [docs/DETAILS.md](docs/DETAILS.md)

License: MIT
