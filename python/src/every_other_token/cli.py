"""``eot-py``: render a saved LLM response with logprobs.

Examples::

    eot-py response.json -o heatmap.html      # standalone HTML page
    eot-py response.json --format text        # coloured terminal output
    cat response.json | eot-py - --format json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import __version__
from ._parse import ResponseShapeError, parse
from ._render import DEFAULT_HIGH, DEFAULT_MID, band, render_text, render_tokens, summarize


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eot-py",
        description=(
            "Show how confident the model was about each token of a saved response "
            "(OpenAI Chat Completion or Completion with logprobs, a JSON list of stream "
            "chunks, or a vLLM RequestOutput dumped to JSON)."
        ),
    )
    p.add_argument("input", help="path to the JSON response, or - for stdin")
    p.add_argument("-o", "--output", help="write here instead of stdout")
    p.add_argument(
        "-f",
        "--format",
        choices=["html", "text", "json"],
        default=None,
        help="html (default with -o, a full page), text (default on a terminal) or json",
    )
    p.add_argument("--choice", type=int, default=0, help="which choice to render (default 0)")
    p.add_argument("--high", type=float, default=DEFAULT_HIGH, help="confidence for green (default 0.7)")
    p.add_argument("--mid", type=float, default=DEFAULT_MID, help="confidence for amber (default 0.4)")
    p.add_argument("--no-color", action="store_true", help="plain text output")
    p.add_argument("--title", default="Token confidence", help="HTML page title")
    p.add_argument("--version", action="version", version=f"eot-py {__version__}")
    return p


def _read(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if not 0.0 <= args.mid <= args.high <= 1.0:
        print("eot-py: need 0 <= --mid <= --high <= 1", file=sys.stderr)
        return 2
    try:
        data = json.loads(_read(args.input))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"eot-py: cannot read {args.input}: {exc}", file=sys.stderr)
        return 2
    try:
        tokens = parse(data, choice=args.choice)
    except ResponseShapeError as exc:
        print(f"eot-py: {exc}", file=sys.stderr)
        return 1

    fmt = args.format or ("html" if args.output or not sys.stdout.isatty() else "text")
    if fmt == "html":
        out = render_tokens(tokens, high=args.high, mid=args.mid).page(args.title)
    elif fmt == "json":
        s = summarize(tokens, args.high, args.mid)
        out = json.dumps(
            {
                "summary": {
                    "tokens": s.tokens,
                    "scored": s.scored,
                    "mean_confidence": s.mean_confidence,
                    "perplexity": s.perplexity,
                    "high": s.high,
                    "mid": s.mid,
                    "low": s.low,
                },
                "tokens": [
                    {
                        "text": t.text,
                        "logprob": t.logprob,
                        "confidence": t.confidence,
                        "perplexity": t.perplexity,
                        "band": band(t.confidence, args.high, args.mid),
                        "alternatives": [{"text": a.text, "logprob": a.logprob} for a in t.alternatives],
                    }
                    for t in tokens
                ],
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n"
    else:
        color = not args.no_color and not args.output and sys.stdout.isatty()
        s = summarize(tokens, args.high, args.mid)
        text = render_text(tokens, color=color, high=args.high, mid=args.mid)
        if s.scored:
            footer = (
                f"\n\n{s.tokens} tokens, mean confidence {s.mean_confidence * 100:.1f}%, "  # type: ignore[operator]
                f"perplexity {s.perplexity:.2f}, {s.low} low / {s.mid} mid / {s.high} high\n"
            )
        else:
            footer = f"\n\n{s.tokens} tokens, no logprobs\n"
        out = text + footer

    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(out)
        print(f"wrote {args.output} ({len(tokens)} tokens)", file=sys.stderr)
    else:
        stream = sys.stdout
        try:
            stream.write(out)
        except UnicodeEncodeError:
            stream.buffer.write(out.encode("utf-8"))  # type: ignore[attr-defined]
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
