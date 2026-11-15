"""See how sure an LLM was about every token of its answer.

>>> from every_other_token import render
>>> resp = client.chat.completions.create(model="gpt-4o-mini", messages=msgs,
...                                       logprobs=True, top_logprobs=5)  # doctest: +SKIP
>>> render(resp)  # in Jupyter: a confidence heatmap  # doctest: +SKIP

The Python companion to the ``every-other-token`` Rust CLI. Pure Python, no
dependencies. Accepts OpenAI Chat Completions and legacy Completions (SDK
objects or dicts), lists of streamed chunks, any OpenAI-compatible server
(vLLM, llama.cpp, LM Studio) and vLLM's offline ``RequestOutput``.
"""

from ._parse import Alternative, ResponseShapeError, Token, parse
from ._render import Heatmap, Summary, band, render, render_text, render_tokens, summarize

__version__ = "0.1.0"

__all__ = [
    "Alternative",
    "Heatmap",
    "ResponseShapeError",
    "Summary",
    "Token",
    "band",
    "parse",
    "render",
    "render_text",
    "render_tokens",
    "summarize",
    "__version__",
]
