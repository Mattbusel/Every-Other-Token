# Python: per-token confidence in notebooks and LLM apps

The `every-other-token` Python package shows how sure a model was about each
token of an answer you already have. Ask the API for logprobs, pass the
response to `render()`, and you get an HTML heatmap: green where the model was
confident, amber where it hesitated, red where it was close to guessing.
Hovering a token shows its probability, logprob, perplexity and the model's
top alternatives at that spot.

It is pure Python with no dependencies, and it does the same arithmetic as the
Rust CLI: `confidence = exp(logprob)` clamped to 0..1, per-token
`perplexity = exp(-logprob)`, green at 70% and up, amber from 40%.

```
pip install every-other-token
```

The Rust CLI streams live from a provider. This package works on a response
you already have, inside your own code.

## In a notebook

```python
from openai import OpenAI
from every_other_token import render

client = OpenAI()
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Who won the 1987 Tour de France?"}],
    logprobs=True,
    top_logprobs=5,   # optional: the alternatives shown on hover
)
render(resp)
```

The last line displays the heatmap. If the model was unsure of the name, that
token shows up red, and hovering it shows which other names it nearly picked.

`render()` returns a `Heatmap`, which is a `str` of HTML with extras:

```python
h = render(resp)
h.summary.mean_confidence   # 0.0..1.0
h.summary.perplexity        # exp(-mean logprob) over the answer
h.summary.low               # how many tokens were below the amber line
h.summary.least_confident   # the 5 least confident tokens, worst first
h.tokens                    # every Token(text, logprob, alternatives)
open("answer.html", "w", encoding="utf-8").write(h.page("My answer"))
```

`render()` options:

| Option | Default | Meaning |
|---|---|---|
| `choice` | `0` | which choice to render when you asked for `n > 1` |
| `high` | `0.7` | confidence at or above this is green |
| `mid` | `0.4` | confidence at or above this (and below `high`) is amber |
| `max_alternatives` | `5` | alternatives listed in each tooltip |
| `show_summary` | `True` | the stats line and legend above the text |

## What it accepts

| Source | How to get logprobs |
|---|---|
| OpenAI Chat Completions (SDK object or dict) | `logprobs=True`, `top_logprobs=N` |
| A list of streamed chat chunks | `stream=True, logprobs=True`; collect the chunks into a list |
| OpenAI legacy Completions | `logprobs=N` |
| vLLM, llama.cpp, LM Studio, other OpenAI-compatible servers | same as OpenAI (same response shape) |
| vLLM offline `LLM.generate` (one `RequestOutput`, or the one-item list it returns for a single prompt; for several prompts pass `outputs[i]`) | `SamplingParams(logprobs=N)` |
| JSON text or a dict of any of the above | |

When a response has no logprobs, `render()` raises `ResponseShapeError` and the
message says which parameter to add.

Some characters (emoji, many CJK characters) are split across several tokens
that each carry part of the UTF-8 bytes; OpenAI sends them as `bytes:\xe2...`.
Those pieces are joined back into the character, and the joined token's
probability is the product of the pieces. This works for streamed chunks too,
where the pieces usually arrive in different chunks.

A response made with logprobs that has no tokens (an empty answer, or
`max_tokens=0`) gives an empty heatmap rather than an error. When the model
refused, the chat logprobs carry the refusal's tokens instead of `content`,
and those are what the heatmap shows.

### Streaming

```python
chunks = list(client.chat.completions.create(..., stream=True, logprobs=True))
render(chunks)
```

### vLLM offline

```python
from vllm import LLM, SamplingParams
from every_other_token import render

outputs = LLM("Qwen/Qwen2.5-0.5B-Instruct").generate(
    ["The capital of Australia is"], SamplingParams(max_tokens=16, logprobs=5)
)
render(outputs[0])
```

## Without the heatmap

```python
from every_other_token import parse, render_text, summarize

tokens = parse(resp)
for t in tokens:
    if t.confidence is not None and t.confidence < 0.4:
        print(repr(t.text), f"{t.confidence:.0%}", [a.text for a in t.alternatives])

print(render_text(tokens))       # ANSI coloured, for a terminal
print(summarize(tokens).perplexity)
```

## Command line: `eot-py`

Save a response as JSON (for example `resp.model_dump_json()`), then:

```
eot-py response.json -o heatmap.html     # a standalone HTML page
eot-py response.json --format text       # coloured text in the terminal
eot-py response.json --format json       # every token with confidence, band, alternatives
cat response.json | eot-py -             # read from stdin
```

Options: `--choice N`, `--high 0.7`, `--mid 0.4`, `--no-color`, `--title`.
Exit code 1 means the file has no logprobs; 2 means it could not be read or the
options were invalid.

## Releasing

The package lives in `python/` and is versioned on its own. Pushing a tag
`py-vX.Y.Z` (matching `version` in `python/pyproject.toml`) runs
`.github/workflows/python.yml`, which tests on Linux, macOS and Windows,
builds the wheel and sdist, and publishes to PyPI with trusted publishing.
Before the first release, add a pending trusted publisher on PyPI for project
`every-other-token`, repository `Mattbusel/Every-Other-Token`, workflow
`python.yml`, environment `pypi`.

Running the tests locally:

```
cd python
python -m venv .venv
.venv/bin/pip install -e ".[test]" openai   # openai is optional
.venv/bin/python -m pytest
```
