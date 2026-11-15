import json
import math
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import every_other_token as eot
from every_other_token import ResponseShapeError, band, parse, render
from every_other_token.cli import main

FIX = Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def test_chat_completion_dict_tokens_and_math():
    toks = parse(load("chat_completion.json"))
    assert "".join(t.text for t in toks) == "The capital is Paris ❤"
    first = toks[0]
    assert first.logprob == pytest.approx(-0.0012)
    # Same arithmetic as the Rust crate: exp(lp) clamped, perplexity exp(-lp).
    assert first.confidence == pytest.approx(math.exp(-0.0012))
    assert first.perplexity == pytest.approx(math.exp(0.0012))
    paris = toks[3]
    assert paris.text == " Paris"
    assert [a.text for a in paris.alternatives] == [" Lyon", " Paris", " <b>Nice</b>"]
    assert paris.alternatives[0].probability == pytest.approx(math.exp(-0.5))


def test_byte_fragments_are_joined_into_one_character():
    toks = parse(load("chat_completion.json"))
    heart = toks[-1]
    assert heart.text == "❤"
    # Probability of the whole character = product of the fragments.
    assert heart.logprob == pytest.approx(-0.4)
    assert len(toks) == 6


def test_choice_index_selects_other_choice():
    toks = parse(load("chat_completion.json"), choice=1)
    assert [t.text for t in toks] == ["Paris"]
    with pytest.raises(ResponseShapeError, match="index 5"):
        parse(load("chat_completion.json"), choice=5)


def test_legacy_completion_with_null_first_logprob():
    toks = parse(load("completion.json"))
    assert [t.text for t in toks] == ["Say", "Hello", " world", "!"]
    assert toks[0].logprob is None and toks[0].confidence is None and toks[0].perplexity is None
    assert toks[2].logprob == pytest.approx(-1.2)
    # Alternatives come sorted most likely first.
    assert [a.text for a in toks[2].alternatives] == [" there", " world"]


def test_stream_chunks_are_concatenated():
    toks = parse(load("chat_stream_chunks.json"))
    assert [t.text for t in toks] == ["Yes", ",", " maybe"]


def _chunk(*content):
    entries = [{"token": t, "logprob": lp, "bytes": b, "top_logprobs": alts} for t, lp, b, alts in content]
    return {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "logprobs": {"content": entries}}]}


def test_stream_byte_fragments_split_across_chunks_are_joined():
    # OpenAI streams about one token per chunk, so a character's fragments
    # land in different chunks and must be merged across them.
    alts = [{"token": "bytes:\\xe2\\x9d", "logprob": -0.1}, {"token": "!", "logprob": -2.5}]
    chunks = [
        {"choices": [{"index": 0, "delta": {"role": "assistant"}, "logprobs": None}]},
        _chunk(("Hi", -0.05, [72, 105], [])),
        _chunk(("bytes:\\xe2\\x9d", -0.1, [0xE2, 0x9D], alts)),
        _chunk(("bytes:\\xa4", -0.3, [0xA4], [])),
        _chunk((" ok", -0.2, [32, 111, 107], [])),
        {"choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}]},
    ]
    toks = parse(chunks)
    assert [t.text for t in toks] == ["Hi", "❤", " ok"]
    assert toks[1].logprob == pytest.approx(-0.4)
    assert [a.text for a in toks[1].alternatives] == ["bytes:\\xe2\\x9d", "!"]
    # The same fragments in one non-streamed response give the same tokens.
    whole = {"choices": [{"index": 0, "logprobs": {"content": [e for c in chunks[1:5] for e in c["choices"][0]["logprobs"]["content"]]}}]}
    assert [(t.text, t.logprob) for t in parse(whole)] == [(t.text, t.logprob) for t in toks]


def test_legacy_completion_zero_logprob_alternative_sorts_first():
    resp = {"choices": [{"index": 0, "logprobs": {
        "tokens": ["A"], "token_logprobs": [0.0], "top_logprobs": [{"B": -1.0, "A": 0.0, "C": -2.0}],
    }}]}
    toks = parse(resp)
    assert [a.text for a in toks[0].alternatives] == ["A", "B", "C"]
    assert toks[0].confidence == 1.0


def test_empty_chat_logprobs_give_an_empty_heatmap():
    for logprobs in ({"content": []}, {"content": None, "refusal": None}, {"content": [], "refusal": None}):
        resp = {"choices": [{"index": 0, "logprobs": logprobs}]}
        assert parse(resp) == []
        hm = render(resp)
        assert hm.summary.tokens == 0
        assert "<" in hm._repr_html_()
    # Empty stream slices are fine too.
    assert parse([_chunk()]) == []


def test_refusal_logprobs_are_the_generated_tokens():
    resp = {"choices": [{"index": 0, "logprobs": {
        "content": None,
        "refusal": [{"token": "I", "logprob": -0.01, "bytes": [73], "top_logprobs": []},
                    {"token": " can't", "logprob": -0.2, "bytes": [32, 99, 97, 110, 39, 116], "top_logprobs": []}],
    }}]}
    assert [t.text for t in parse(resp)] == ["I", " can't"]


def test_list_of_several_vllm_request_outputs_says_pick_one():
    out = {"index": 0, "token_ids": [1], "logprobs": [{"1": {"logprob": -0.1, "rank": 1, "decoded_token": "x"}}]}
    with pytest.raises(ResponseShapeError, match=r"2 vLLM RequestOutputs.*outputs\[0\]"):
        parse([{"outputs": [out]}, {"outputs": [out]}])
    assert [t.text for t in parse([{"outputs": [out]}])] == ["x"]


def test_chat_message_dict_is_not_a_bare_logprobs_object():
    with pytest.raises(ResponseShapeError, match="not a Chat Completion"):
        parse({"role": "assistant", "content": "hello"})


def test_vllm_request_output_json_round_trip():
    toks = parse(load("vllm_request_output.json"))
    assert [t.text for t in toks] == [" blue", " today"]
    assert toks[1].logprob == pytest.approx(-1.2)
    # vLLM ranks: rank 1 first even though the chosen token was rank 2.
    assert [a.text for a in toks[1].alternatives] == [" and", " today"]


def test_vllm_request_output_objects_with_int_keys():
    Logprob = lambda lp, rank, tok: SimpleNamespace(logprob=lp, rank=rank, decoded_token=tok)  # noqa: E731
    out = SimpleNamespace(
        index=0,
        token_ids=[11, 22],
        logprobs=[
            {11: Logprob(-0.05, 1, " Hi"), 99: Logprob(-3.1, 2, " Yo")},
            {22: Logprob(-2.0, 3, "!"), 33: Logprob(-0.3, 1, "."), 44: Logprob(-1.5, 2, "?")},
        ],
    )
    toks = parse(SimpleNamespace(outputs=[out]))
    assert [t.text for t in toks] == [" Hi", "!"]
    assert toks[1].logprob == -2.0
    assert [a.text for a in toks[1].alternatives] == [".", "?", "!"]
    # A list holding one RequestOutput (what LLM.generate returns) works too.
    assert [t.text for t in parse([SimpleNamespace(outputs=[out])])] == [" Hi", "!"]


def test_vllm_without_logprobs_explains():
    out = SimpleNamespace(token_ids=[1], logprobs=None)
    with pytest.raises(ResponseShapeError, match="SamplingParams"):
        parse(SimpleNamespace(outputs=[out]))


def test_json_string_and_bare_logprobs_object():
    raw = (FIX / "chat_completion.json").read_text(encoding="utf-8")
    assert len(parse(raw)) == 6
    bare = load("chat_completion.json")["choices"][0]["logprobs"]
    assert len(parse(bare)) == 6


def test_missing_logprobs_gives_actionable_error():
    resp = load("chat_completion.json")
    resp["choices"][0]["logprobs"] = None
    with pytest.raises(ResponseShapeError, match="logprobs=True"):
        parse(resp)
    with pytest.raises(ResponseShapeError):
        parse({"hello": "world"})


def test_openai_sdk_objects():
    openai_types = pytest.importorskip("openai.types")
    chat = openai_types.chat.ChatCompletion.model_validate(load("chat_completion.json"))
    assert [t.text for t in parse(chat)] == [t.text for t in parse(load("chat_completion.json"))]
    # The SDK model rejects the null first entry of an echo=True completion,
    # so validate the generated part only.
    raw = load("completion.json")
    lp = raw["choices"][0]["logprobs"]
    for key in ("tokens", "token_logprobs", "top_logprobs", "text_offset"):
        lp[key] = lp[key][1:]
    comp = openai_types.Completion.model_validate(raw)
    toks = parse(comp)
    assert [t.text for t in toks] == ["Hello", " world", "!"]
    assert [a.text for a in toks[1].alternatives] == [" there", " world"]
    chunks = [
        openai_types.chat.ChatCompletionChunk.model_validate(c) for c in load("chat_stream_chunks.json")
    ]
    assert [t.text for t in parse(chunks)] == ["Yes", ",", " maybe"]
    assert "eot-heatmap" in render(chat)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def test_band_thresholds_match_rust_defaults():
    assert band(0.9) == "high" and band(0.7) == "high"
    assert band(0.5) == "mid" and band(0.4) == "mid"
    assert band(0.1) == "low"
    assert band(None) == "none"
    assert band(0.5, high=0.9, mid=0.6) == "low"


def test_render_html_heatmap():
    h = render(load("chat_completion.json"))
    assert isinstance(h, str)
    assert h._repr_html_() == str(h)
    assert repr(h) == "<Heatmap 6 tokens, mean confidence 69.8%, perplexity 1.61>"
    spans = re.findall(r'<span class="eot-tok eot-(\w+)"', h)
    # The(0.999) capital(0.95) is(0.55) Paris(0.20) " "(0.82) heart(0.67)
    assert spans == ["high", "high", "mid", "low", "high", "mid"]
    # Token text and alternatives are escaped, never injected.
    assert "<b>Nice</b>" not in h
    assert "&lt;b&gt;Nice&lt;/b&gt;" in h
    assert "confidence 20.2%" in h
    assert h.summary.low == 1 and h.summary.mid == 2 and h.summary.high == 3
    assert h.summary.least_confident[0].text == " Paris"


def test_render_summary_numbers():
    h = render(load("completion.json"))
    s = h.summary
    assert s.tokens == 4 and s.scored == 3
    lps = [-0.25, -1.2, -2.5]
    assert s.perplexity == pytest.approx(math.exp(-sum(lps) / 3))
    assert s.mean_confidence == pytest.approx(sum(math.exp(x) for x in lps) / 3)
    assert "no logprob for this token" in h
    assert "border-bottom:1px dashed" in h


def test_page_is_a_full_document():
    page = render(load("completion.json")).page("My <run>")
    assert page.startswith("<!doctype html>")
    assert "<title>My &lt;run&gt;</title>" in page
    assert page.rstrip().endswith("</html>")


def test_render_text_ansi():
    toks = parse(load("chat_stream_chunks.json"))
    assert eot.render_text(toks, color=False) == "Yes, maybe"
    colored = eot.render_text(toks)
    assert colored.startswith("\x1b[32mYes\x1b[0m")
    assert "\x1b[31m maybe" in colored


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_writes_html(tmp_path, capsys):
    out = tmp_path / "h.html"
    assert main([str(FIX / "chat_completion.json"), "-o", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>") and "eot-heatmap" in text


def test_cli_json_format(capsys):
    assert main([str(FIX / "completion.json"), "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["summary"]["scored"] == 3
    assert data["tokens"][3]["band"] == "low"
    assert data["tokens"][2]["alternatives"][0]["text"] == " there"


def test_cli_text_format(capsys):
    assert main([str(FIX / "chat_stream_chunks.json"), "--format", "text", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Yes, maybe")
    assert "3 tokens" in out


def test_cli_errors(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text('{"choices": [{"index": 0, "logprobs": null}]}', encoding="utf-8")
    assert main([str(bad)]) == 1
    assert "logprobs=True" in capsys.readouterr().err
    assert main([str(tmp_path / "missing.json")]) == 2
    assert main([str(FIX / "completion.json"), "--mid", "0.9", "--high", "0.5"]) == 2


def test_cli_entry_point_reads_stdin():
    # Runs the installed console script (or the module) as a real process.
    raw = (FIX / "completion.json").read_bytes()
    proc = subprocess.run(
        [sys.executable, "-m", "every_other_token.cli", "-", "--format", "json"],
        input=raw,
        capture_output=True,
        check=True,
    )
    assert json.loads(proc.stdout)["summary"]["tokens"] == 4
