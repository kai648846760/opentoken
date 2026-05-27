"""SSE-payload parser unit tests.

The grok and chatgpt non-stream paths receive a buffered SSE body and extract
text via _parse_*_sse_text(). These pure functions are central to producing
correct chat responses, but the original review noted they had no targeted
tests. Cover the realistic shapes here.
"""
from __future__ import annotations

from opentoken.providers.grok import _iter_grok_sse_text, _parse_grok_sse_text
from opentoken.providers.chatgpt import _advance_streamed_text_state, _parse_chatgpt_sse_text


# ── grok ─────────────────────────────────────────────────────────────────────


def test_parse_grok_sse_text_collects_openai_style_deltas() -> None:
    payload = (
        'data: {"choices":[{"delta":{"content":"hello"}}]}\n'
        'data: {"choices":[{"delta":{"content":" world"}}]}\n'
        "data: [DONE]\n"
    )
    assert _parse_grok_sse_text(payload) == "hello world"


def test_parse_grok_sse_text_does_not_double_emit_when_text_field_also_present() -> None:
    """Round-2 regression: if both choices[].delta.content AND a top-level
    text/content/delta field carry the same chunk, _parse_grok_sse_text must
    emit ONLY once (the choice-derived chunk wins)."""
    payload = (
        'data: {"choices":[{"delta":{"content":"AB"}}],"text":"AB"}\n'
        "data: [DONE]\n"
    )
    assert _parse_grok_sse_text(payload) == "AB"


def test_parse_grok_sse_text_falls_back_to_text_field_only_when_no_choice() -> None:
    payload = (
        'data: {"text":"hi"}\n'
        "data: [DONE]\n"
    )
    assert _parse_grok_sse_text(payload) == "hi"


def test_parse_grok_sse_text_ignores_done_and_malformed_lines() -> None:
    payload = (
        "\n"
        "data:\n"
        "data: [DONE]\n"
        "data: not-valid-json\n"
        'data: {"choices":[{"delta":{"content":"x"}}]}\n'
    )
    assert _parse_grok_sse_text(payload) == "x"


def test_iter_grok_sse_text_yields_pieces_in_order_and_skips_done() -> None:
    lines = iter([
        'data: {"choices":[{"delta":{"content":"a"}}]}',
        'data: {"choices":[{"delta":{"content":"b"}}]}',
        "data: [DONE]",
    ])
    assert list(_iter_grok_sse_text(lines)) == ["a", "b"]


def test_iter_grok_sse_text_does_not_double_emit() -> None:
    """Streaming counterpart of the round-2 fix — already has its own `continue`,
    locked here too."""
    lines = iter([
        'data: {"choices":[{"delta":{"content":"AB"}}],"text":"AB"}',
        "data: [DONE]",
    ])
    assert list(_iter_grok_sse_text(lines)) == ["AB"]


# ── chatgpt ──────────────────────────────────────────────────────────────────


def test_parse_chatgpt_sse_text_extracts_content() -> None:
    # ChatGPT backend-api SSE chunks expose deltas inside `message.content.parts`
    # nested under data.message; the parser should reassemble the final text.
    payload = (
        'data: {"message":{"id":"m1","author":{"role":"assistant"},'
        '"content":{"content_type":"text","parts":["hello"]}}}\n'
        'data: {"message":{"id":"m1","author":{"role":"assistant"},'
        '"content":{"content_type":"text","parts":["hello world"]}}}\n'
        "data: [DONE]\n"
    )
    result = _parse_chatgpt_sse_text(payload)
    # The parser returns the final accumulated assistant text — at minimum it
    # must contain "hello world" (the last frame). We don't pin the exact
    # whitespace because the parser dedupes overlapping deltas.
    assert "hello world" in result


def test_advance_streamed_text_state_normal_growth() -> None:
    """Append-only snapshots produce monotonic deltas."""
    suffix, state = _advance_streamed_text_state("", "hello")
    assert (suffix, state) == ("hello", "hello")
    suffix, state = _advance_streamed_text_state(state, "hello world")
    assert (suffix, state) == (" world", "hello world")


def test_advance_streamed_text_state_handles_stale_short_snapshot() -> None:
    """A re-delivered earlier snapshot is a no-op (no re-emission, state held)."""
    suffix, state = _advance_streamed_text_state("hello world", "hello")
    assert (suffix, state) == ("", "hello world")


def test_advance_streamed_text_state_divergent_does_not_cascade() -> None:
    """When a snapshot diverges (regenerate/rewrite), the new baseline must be
    `candidate` alone — not `current + candidate`. With the buggy concatenation
    every subsequent extension also fails the prefix check, causing the entire
    text to be re-emitted on every frame (cascading duplication).
    """
    suffix, state = _advance_streamed_text_state("foo", "bar")
    # Emit the divergent snapshot once, but the baseline must reset to it.
    assert suffix == "bar"
    assert state == "bar"

    # A normal extension of `bar` must now produce only its incremental tail —
    # NOT the whole text again.
    suffix, state = _advance_streamed_text_state(state, "bar baz")
    assert (suffix, state) == (" baz", "bar baz")


def test_parse_chatgpt_sse_text_handles_done_and_blanks() -> None:
    payload = (
        "\n"
        "data: [DONE]\n"
        "\n"
    )
    # No content frames -> empty result (caller treats as no-content error).
    assert _parse_chatgpt_sse_text(payload) == ""
