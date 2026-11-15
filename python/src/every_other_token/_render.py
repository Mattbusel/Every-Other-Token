"""Render parsed tokens as an HTML confidence heatmap or ANSI terminal text."""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from ._parse import Token, parse

__all__ = ["Heatmap", "Summary", "band", "render", "render_text", "summarize"]

# Same thresholds and colours as the Rust CLI (render::ConfidenceThresholds)
# and the web UI (.conf-high / .conf-mid / .conf-low).
DEFAULT_HIGH = 0.7
DEFAULT_MID = 0.4
_RGB = {"high": (63, 185, 80), "mid": (227, 179, 65), "low": (248, 81, 73)}


def band(confidence: Optional[float], high: float = DEFAULT_HIGH, mid: float = DEFAULT_MID) -> str:
    """Classify a confidence as ``"high"``, ``"mid"``, ``"low"`` or ``"none"``."""
    if confidence is None:
        return "none"
    if confidence >= high:
        return "high"
    if confidence >= mid:
        return "mid"
    return "low"


@dataclass(frozen=True)
class Summary:
    """Whole-answer numbers.

    ``perplexity`` is the sequence perplexity ``exp(-mean(logprob))`` over the
    tokens that have a logprob; ``mean_confidence`` is the mean of
    ``exp(logprob)``. ``low``/``mid``/``high`` count tokens per band.
    """

    tokens: int
    scored: int
    mean_confidence: Optional[float]
    perplexity: Optional[float]
    high: int
    mid: int
    low: int
    least_confident: List[Token]


def summarize(tokens: Sequence[Token], high: float = DEFAULT_HIGH, mid: float = DEFAULT_MID, worst: int = 5) -> Summary:
    scored = [t for t in tokens if t.logprob is not None]
    counts = {"high": 0, "mid": 0, "low": 0}
    for t in scored:
        counts[band(t.confidence, high, mid)] += 1
    if scored:
        mean_lp = sum(t.logprob for t in scored) / len(scored)  # type: ignore[misc]
        mean_conf = sum(t.confidence for t in scored) / len(scored)  # type: ignore[misc]
        try:
            ppl: Optional[float] = math.exp(-mean_lp)
        except OverflowError:
            ppl = math.inf
    else:
        mean_conf = None
        ppl = None
    least = sorted(scored, key=lambda t: t.logprob)[:worst]  # type: ignore[arg-type,return-value]
    return Summary(len(tokens), len(scored), mean_conf, ppl, counts["high"], counts["mid"], counts["low"], least)


class Heatmap(str):
    """The rendered HTML (a ``str``), shown as a heatmap by Jupyter.

    Also carries ``tokens`` and ``summary`` for programmatic use. ``str(h)``
    is the HTML fragment; :meth:`page` wraps it in a full HTML document.
    """

    tokens: List[Token]
    summary: Summary

    def __new__(cls, markup: str, tokens: List[Token], summary: Summary) -> "Heatmap":
        obj = super().__new__(cls, markup)
        obj.tokens = tokens
        obj.summary = summary
        return obj

    def _repr_html_(self) -> str:
        return str(self)

    def __repr__(self) -> str:
        s = self.summary
        if not s.scored:
            return f"<Heatmap {s.tokens} tokens, no logprobs>"
        return (
            f"<Heatmap {s.tokens} tokens, mean confidence {s.mean_confidence * 100:.1f}%, "  # type: ignore[operator]
            f"perplexity {s.perplexity:.2f}>"
        )

    def page(self, title: str = "Token confidence") -> str:
        """A standalone HTML document, ready to write to a ``.html`` file."""
        return (
            "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{html.escape(title)}</title></head>"
            "<body style=\"margin:0;padding:24px 16px;font-family:system-ui,sans-serif;"
            "background:#fff;color:#1f2328\">"
            f"{self}</body></html>\n"
        )


def _fmt_pct(c: Optional[float]) -> str:
    return "n/a" if c is None else f"{c * 100:.1f}%"


def _tooltip(t: Token, max_alts: int) -> str:
    if t.logprob is None:
        return "no logprob for this token"
    lines = [
        f"confidence {_fmt_pct(t.confidence)}",
        f"logprob {t.logprob:.4f}",
        f"perplexity {t.perplexity:.2f}",
    ]
    if t.alternatives:
        lines.append("top alternatives:")
        for a in t.alternatives[:max_alts]:
            lines.append(f"  {a.text!r} {_fmt_pct(a.probability)}")
    return "\n".join(lines)


def _style(t: Token, high: float, mid: float) -> str:
    b = band(t.confidence, high, mid)
    if b == "none":
        return "border-bottom:1px dashed #8b949e"
    r, g, bl = _RGB[b]
    c = t.confidence or 0.0
    # Stronger tint the less sure the model was, so trouble spots stand out.
    alpha = 0.12 + 0.5 * (1.0 - c)
    return f"background:rgba({r},{g},{bl},{alpha:.2f});border-radius:2px"


def render_tokens(
    tokens: Sequence[Token],
    *,
    high: float = DEFAULT_HIGH,
    mid: float = DEFAULT_MID,
    max_alternatives: int = 5,
    show_summary: bool = True,
) -> Heatmap:
    """Render already-parsed tokens. See :func:`render`."""
    tokens = list(tokens)
    s = summarize(tokens, high, mid)
    spans = []
    for t in tokens:
        title = html.escape(_tooltip(t, max_alternatives), quote=True)
        conf = "" if t.confidence is None else f" data-confidence=\"{t.confidence:.4f}\""
        spans.append(
            f"<span class=\"eot-tok eot-{band(t.confidence, high, mid)}\"{conf} "
            f"title=\"{title}\" style=\"{_style(t, high, mid)}\">{html.escape(t.text)}</span>"
        )
    parts = [
        "<div class=\"eot-heatmap\" style=\"font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;"
        "font-size:14px;line-height:1.7\">"
    ]
    if show_summary:
        parts.append(_summary_html(s, high, mid))
    parts.append("<div class=\"eot-text\" style=\"white-space:pre-wrap;word-wrap:break-word\">")
    parts.append("".join(spans))
    parts.append("</div></div>")
    return Heatmap("".join(parts), tokens, s)


def _swatch(name: str, label: str) -> str:
    r, g, b = _RGB[name]
    return (
        f"<span style=\"background:rgba({r},{g},{b},0.45);border-radius:2px;padding:0 4px;"
        f"margin-right:6px\">{html.escape(label)}</span>"
    )


def _summary_html(s: Summary, high: float, mid: float) -> str:
    if s.scored:
        stats = (
            f"{s.tokens} tokens &middot; mean confidence {_fmt_pct(s.mean_confidence)} &middot; "
            f"perplexity {s.perplexity:.2f} &middot; {s.low} low / {s.mid} mid / {s.high} high"
        )
    else:
        stats = f"{s.tokens} tokens &middot; no logprobs"
    legend = (
        _swatch("high", f">= {high:.0%}")
        + _swatch("mid", f"{mid:.0%}-{high:.0%}")
        + _swatch("low", f"< {mid:.0%}")
    )
    return (
        "<div class=\"eot-summary\" style=\"font-family:system-ui,sans-serif;font-size:12px;"
        f"opacity:.8;margin-bottom:6px\">{stats}<br>{legend}"
        "<span>hover a token for its alternatives</span></div>"
    )


def render(
    resp: Any,
    *,
    choice: int = 0,
    high: float = DEFAULT_HIGH,
    mid: float = DEFAULT_MID,
    max_alternatives: int = 5,
    show_summary: bool = True,
) -> Heatmap:
    """Render a response that has logprobs as an HTML confidence heatmap.

    ``resp`` is anything :func:`every_other_token.parse` accepts: an OpenAI
    ``ChatCompletion`` or ``Completion`` (SDK object or dict) created with
    logprobs, a list of streamed chunks, a JSON string, or a vLLM
    ``RequestOutput``. In Jupyter the result displays as the heatmap; it is
    also a plain ``str`` of HTML.

    Each token is tinted green (confidence >= ``high``), amber (>= ``mid``)
    or red (below), more strongly the less confident the model was. Hovering
    shows confidence, logprob, perplexity and the top alternatives.
    """
    return render_tokens(
        parse(resp, choice=choice),
        high=high,
        mid=mid,
        max_alternatives=max_alternatives,
        show_summary=show_summary,
    )


_ANSI = {"high": "\x1b[32m", "mid": "\x1b[33m", "low": "\x1b[31m", "none": "\x1b[2m"}


def render_text(tokens: Sequence[Token], *, color: bool = True, high: float = DEFAULT_HIGH, mid: float = DEFAULT_MID) -> str:
    """The answer with each token coloured by band (ANSI), like the Rust CLI."""
    if not color:
        return "".join(t.text for t in tokens)
    return "".join(f"{_ANSI[band(t.confidence, high, mid)]}{t.text}\x1b[0m" for t in tokens)
