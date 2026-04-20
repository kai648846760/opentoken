from __future__ import annotations

from openclaw_algae.verification.stream_probe import analyze_stream_lines


def _clock(step: float = 0.1):
    current = 0.0

    def now() -> float:
        nonlocal current
        value = current
        current += step
        return value

    return now


def test_analyze_responses_lines_reports_generic_error_event() -> None:
    result = analyze_stream_lines(
        [
            "event: response.created",
            'data: {"type":"response.created"}',
            "event: response.in_progress",
            'data: {"type":"response.in_progress"}',
            "event: error",
            'data: {"type":"error","error":{"message":"rate limited","type":"rate_limit_error"}}',
        ],
        mode="responses",
        now_fn=_clock(),
    )

    assert result["visible_chunks"] == 0
    assert result["done"] is False
    assert "rate_limit_error" in str(result["error"])
    assert "rate limited" in str(result["error"])



def test_analyze_responses_lines_counts_reasoning_delta_as_visible() -> None:
    result = analyze_stream_lines(
        [
            "event: response.created",
            'data: {"type":"response.created"}',
            "event: response.output_item.added",
            'data: {"type":"response.output_item.added"}',
            "event: response.reasoning_text.delta",
            'data: {"type":"response.reasoning_text.delta","delta":"先想一想"}',
            "event: response.completed",
            'data: {"type":"response.completed","response":{"status":"completed"}}',
        ],
        mode="responses",
        now_fn=_clock(),
    )

    assert result["visible_chunks"] == 1
    assert result["reasoning_chunks"] == 1
    assert result["message_chunks"] == 0
    assert result["first_visible_s"] is not None
    assert result["done"] is True
    assert result["preview"] == "先想一想"



def test_analyze_chat_lines_drains_stream_after_proving_visibility() -> None:
    seen_done = {"value": False}

    def lines():
        yield 'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}'
        for index in range(6):
            yield f'data: {{"choices":[{{"delta":{{"content":"片段{index}"}},"finish_reason":null}}]}}'
        seen_done["value"] = True
        yield "data: [DONE]"

    result = analyze_stream_lines(
        lines(),
        mode="chat",
        now_fn=_clock(),
        max_visible_chunks=3,
        observation_window_s=0.0,
    )

    assert result["visible_chunks"] == 3
    assert result["done"] is True
    assert seen_done["value"] is True
