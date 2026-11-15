# every-other-token (Python)

See how sure an LLM was about every token of its answer. Ask for logprobs,
pass the response to `render()`, and get an HTML heatmap: green where the
model was confident, amber where it hesitated, red where it was guessing.
Hover a token for its probability, logprob, perplexity and the model's top
alternatives.

```python
from openai import OpenAI
from every_other_token import render

resp = OpenAI().chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Name three moons of Jupiter."}],
    logprobs=True,
    top_logprobs=5,
)
render(resp)  # in Jupyter this displays the heatmap
```

Works with OpenAI Chat Completions and legacy Completions (SDK objects or
dicts), lists of streamed chunks, vLLM / llama.cpp / LM Studio
(OpenAI-compatible servers) and vLLM's offline `RequestOutput`. Pure Python,
no dependencies.

From the shell:

```
eot-py response.json -o heatmap.html
eot-py response.json --format text
```

Full guide: https://github.com/Mattbusel/Every-Other-Token/blob/main/docs/python.md

This is the Python companion to the
[every-other-token](https://crates.io/crates/every-other-token) Rust CLI,
which streams live from OpenAI, Anthropic, Gemini, OpenRouter and local
models. MIT licensed.
