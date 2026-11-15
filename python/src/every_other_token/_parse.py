"""Turn LLM responses that carry log-probabilities into a flat list of tokens.

Supported shapes (objects from the ``openai`` SDK, plain dicts, or JSON):

* OpenAI **Chat Completions** with ``logprobs=True``:
  ``choices[i].logprobs.content[j] = {token, logprob, bytes, top_logprobs}``.
  vLLM, llama.cpp, LM Studio and other OpenAI-compatible servers return the
  same shape.
* A list of **streamed chat chunks** (``stream=True``), each carrying its own
  ``choices[i].logprobs.content`` slice.
* OpenAI legacy **Completions** with ``logprobs=N``:
  ``choices[i].logprobs = {tokens, token_logprobs, top_logprobs}``.
* vLLM **offline** ``RequestOutput`` (``LLM.generate`` with
  ``SamplingParams(logprobs=N)``): ``outputs[i].token_ids`` plus
  ``outputs[i].logprobs[j] = {token_id: Logprob(logprob, rank, decoded_token)}``.

The arithmetic matches the Rust crate: ``confidence = clamp(exp(logprob), 0, 1)``
and per-token ``perplexity = exp(-logprob)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = ["Alternative", "Token", "parse", "ResponseShapeError"]


class ResponseShapeError(ValueError):
    """The response has no per-token log-probabilities in a shape we know."""


@dataclass(frozen=True)
class Alternative:
    """One of the model's top candidates at a position."""

    text: str
    logprob: float

    @property
    def probability(self) -> float:
        return _confidence(self.logprob)


@dataclass(frozen=True)
class Token:
    """A generated token and how sure the model was about it.

    ``logprob`` is ``None`` when the provider sent no number for this token
    (for example the first prompt token of an ``echo=True`` completion).
    """

    text: str
    logprob: Optional[float]
    alternatives: Tuple[Alternative, ...] = field(default_factory=tuple)

    @property
    def confidence(self) -> Optional[float]:
        """``exp(logprob)`` clamped to ``[0, 1]``, or ``None``."""
        return None if self.logprob is None else _confidence(self.logprob)

    @property
    def perplexity(self) -> Optional[float]:
        """Per-token perplexity ``exp(-logprob)``, or ``None``."""
        if self.logprob is None:
            return None
        try:
            return math.exp(-self.logprob)
        except OverflowError:
            return math.inf


def _confidence(logprob: float) -> float:
    if math.isnan(logprob):
        return 0.0
    if logprob >= 0:
        return 1.0
    return min(1.0, max(0.0, math.exp(logprob)))


# ---------------------------------------------------------------------------
# Generic accessors: work on dicts, pydantic models and plain objects alike.
# ---------------------------------------------------------------------------

_MISSING = object()


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    value = getattr(obj, key, _MISSING)
    return default if value is _MISSING else value


def _to_plain(resp: Any) -> Any:
    """Convert SDK objects to dicts when they offer a lossless dump."""
    if isinstance(resp, (Mapping, list, tuple)):
        return resp
    for method in ("model_dump", "to_dict", "dict"):
        fn = getattr(resp, method, None)
        if callable(fn):
            try:
                out = fn()
            except TypeError:
                continue
            if isinstance(out, Mapping):
                return out
    return resp


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Shape handlers
# ---------------------------------------------------------------------------


# (text, logprob, raw UTF-8 bytes or None, alternatives) before byte merging.
_Entry = Tuple[str, Optional[float], Optional[bytes], Tuple[Alternative, ...]]


def _merge_byte_fragments(entries: List[_Entry]) -> List[Token]:
    """Join tokens that each hold part of one UTF-8 character.

    OpenAI splits some characters (emoji, CJK) across tokens and sends each
    fragment as ``"bytes:\\xe2\\x80"`` with the raw ``bytes``. We join the
    fragments until they decode; the joined token's logprob is the sum (the
    probability of the whole run), and its alternatives are the first
    fragment's.
    """
    out: List[Token] = []
    pending: List[_Entry] = []

    def flush_raw() -> None:
        for text, lp, _b, alts in pending:
            out.append(Token(text, lp, alts))
        pending.clear()

    for entry in entries:
        text, lp, raw, alts = entry
        if raw is None and not pending:
            out.append(Token(text, lp, alts))
            continue
        if raw is None:
            flush_raw()
            out.append(Token(text, lp, alts))
            continue
        pending.append(entry)
        buf = b"".join(p[2] or b"" for p in pending)
        try:
            decoded = buf.decode("utf-8")
        except UnicodeDecodeError:
            if len(pending) >= 4:
                flush_raw()
            continue
        if len(pending) == 1:
            out.append(Token(text if not text.startswith("bytes:") else decoded, lp, alts))
        else:
            lps = [p[1] for p in pending]
            total = None if any(v is None for v in lps) else sum(lps)  # type: ignore[arg-type]
            out.append(Token(decoded, total, pending[0][3]))
        pending.clear()
    flush_raw()
    return out


def _chat_entries(content: Iterable[Any]) -> List[_Entry]:
    entries = []
    for item in content or []:
        text = _get(item, "token", "") or ""
        lp = _num(_get(item, "logprob"))
        raw_bytes = _get(item, "bytes")
        raw = bytes(raw_bytes) if isinstance(raw_bytes, (list, tuple, bytes, bytearray)) else None
        alts = tuple(
            Alternative(_get(a, "token", "") or "", float(_get(a, "logprob")))
            for a in (_get(item, "top_logprobs") or [])
            if _num(_get(a, "logprob")) is not None
        )
        entries.append((text, lp, raw, alts))
    return entries


def _from_completion_logprobs(logprobs: Any) -> Optional[List[Token]]:
    tokens = _get(logprobs, "tokens")
    if tokens is None:
        return None
    lps = _get(logprobs, "token_logprobs") or []
    tops = _get(logprobs, "top_logprobs") or []
    out = []
    for i, text in enumerate(tokens):
        lp = _num(lps[i]) if i < len(lps) else None
        top = tops[i] if i < len(tops) else None
        alts: Tuple[Alternative, ...] = ()
        if isinstance(top, Mapping):
            alts = tuple(
                Alternative(str(k), float(v))
                for k, v in sorted(top.items(), key=lambda kv: _desc(_num(kv[1])))
                if _num(v) is not None
            )
        out.append(Token(str(text), lp, alts))
    return out


def _desc(value: Optional[float]) -> float:
    """Sort key: highest value first, missing values last. 0.0 is a real value."""
    return math.inf if value is None else -value


def _has(obj: Any, key: str) -> bool:
    if isinstance(obj, Mapping):
        return key in obj
    return hasattr(obj, key)


def _logprobs_entries(logprobs: Any) -> Optional[List[_Entry]]:
    """Raw entries of one ``choices[i].logprobs`` object, before byte merging.

    Returns ``[]`` for a chat logprobs object with nothing in it (an empty
    answer, ``max_tokens=0``, or ``content: null``), and ``None`` only when the
    object is not a logprobs shape we know.
    """
    content = _get(logprobs, "content")
    if isinstance(content, (list, tuple)):
        return _chat_entries(content)
    tokens = _get(logprobs, "tokens")
    if isinstance(tokens, (list, tuple)):
        return [(t.text, t.logprob, None, t.alternatives) for t in _from_completion_logprobs(logprobs) or []]
    # A refusal carries its own token logprobs; those are the generated tokens.
    refusal = _get(logprobs, "refusal")
    if isinstance(refusal, (list, tuple)):
        return _chat_entries(refusal)
    if content is None and (_has(logprobs, "content") or _has(logprobs, "refusal")):
        return []
    return None


def _from_vllm_output(output: Any) -> Optional[List[Token]]:
    logprobs = _get(output, "logprobs")
    token_ids = _get(output, "token_ids")
    if logprobs is None or token_ids is None:
        return None
    out = []
    for i, tid in enumerate(token_ids):
        pos = logprobs[i] if i < len(logprobs) else None
        if not pos:
            out.append(Token(str(tid), None))
            continue
        # Keys may be ints (Python objects) or strings (after a JSON round trip).
        chosen = pos.get(tid, pos.get(str(tid))) if isinstance(pos, Mapping) else None
        by_rank = sorted(pos.items(), key=lambda kv: _rank(kv[1]))
        alts = tuple(
            Alternative(_decoded(lp, key), float(_get(lp, "logprob")))
            for key, lp in by_rank
            if _num(_get(lp, "logprob")) is not None
        )
        if chosen is None:
            out.append(Token(str(tid), None, alts))
        else:
            out.append(Token(_decoded(chosen, tid), _num(_get(chosen, "logprob")), alts))
    return out


def _rank(lp: Any) -> Tuple[int, float]:
    """Sort key: vLLM's rank when present, then most probable first."""
    rank = _num(_get(lp, "rank"))
    if rank is not None:
        return (0, rank)
    lpv = _num(_get(lp, "logprob"))
    return (1, math.inf if lpv is None else -lpv)


def _decoded(lp: Any, key: Any) -> str:
    text = _get(lp, "decoded_token")
    return str(key) if text is None else str(text)


def _choice_entries(choice: Any) -> Optional[List[_Entry]]:
    logprobs = _get(choice, "logprobs")
    if logprobs is None:
        return None
    return _logprobs_entries(logprobs)


def _pick(seq: Sequence[Any], index: int, what: str) -> Any:
    for item in seq:
        if _get(item, "index", None) == index:
            return item
    if 0 <= index < len(seq):
        return seq[index]
    raise ResponseShapeError(f"response has no {what} with index {index} (it has {len(seq)})")


def parse(resp: Any, choice: int = 0) -> List[Token]:
    """Extract the tokens of one choice, with their log-probabilities.

    ``resp`` may be an ``openai`` SDK object, a dict, a JSON string, a list of
    streamed chunks, or a vLLM ``RequestOutput`` (object or dict). Raises
    :class:`ResponseShapeError` when no logprobs are present, with a hint on
    how to ask for them.
    """
    if isinstance(resp, (str, bytes)):
        import json

        resp = json.loads(resp)
    resp = _to_plain(resp)

    # A list of streamed chunks (or the list of RequestOutputs LLM.generate returns).
    if isinstance(resp, (list, tuple)):
        n_outputs = sum(1 for item in resp if _get(item, "outputs") is not None)
        if n_outputs == 1 and len(resp) == 1:
            return parse(resp[0], choice)
        if n_outputs:
            raise ResponseShapeError(
                f"got a list of {len(resp)} vLLM RequestOutputs (one per prompt); "
                "pass one of them, for example render(outputs[0])"
            )
        # Collect every chunk's raw entries first and merge once: a character
        # split into byte fragments usually arrives across several chunks.
        entries: List[_Entry] = []
        saw_logprobs = False
        for chunk in resp:
            chunk = _to_plain(chunk)
            for c in _get(chunk, "choices") or []:
                if (_get(c, "index", 0) or 0) != choice:
                    continue
                part = _choice_entries(c)
                if part is not None:
                    saw_logprobs = True
                    entries.extend(part)
        if not saw_logprobs:
            raise ResponseShapeError(_NO_LOGPROBS)
        return _merge_byte_fragments(entries)

    choices = _get(resp, "choices")
    if choices is not None:
        picked = _pick(choices, choice, "choice")
        picked_entries = _choice_entries(picked)
        if picked_entries is None:
            raise ResponseShapeError(_NO_LOGPROBS)
        return _merge_byte_fragments(picked_entries)

    outputs = _get(resp, "outputs")
    if outputs is not None:
        out = _from_vllm_output(_pick(outputs, choice, "output"))
        if out is None:
            raise ResponseShapeError(
                "vLLM output has no logprobs: pass SamplingParams(logprobs=5) to LLM.generate"
            )
        return out

    # A bare logprobs object (choices[i].logprobs) is accepted too.
    # Only list-valued content/tokens/refusal count, so a chat message dict
    # ({"role": ..., "content": "text"}) is not mistaken for one.
    if any(isinstance(_get(resp, k), (list, tuple)) for k in ("content", "tokens", "refusal")):
        bare = _logprobs_entries(resp)
        if bare is not None:
            return _merge_byte_fragments(bare)
    raise ResponseShapeError(
        "not a Chat Completion, Completion, list of stream chunks or vLLM RequestOutput"
    )


_NO_LOGPROBS = (
    "response has no logprobs: call the API with logprobs=True (and top_logprobs=5 "
    "for alternatives) for chat completions, or logprobs=5 for legacy completions"
)
