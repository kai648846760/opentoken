from openclaw_algae.api.streaming import ProtocolMarkupProjector, chunk_visible_text, strip_tool_protocol_markup


def test_protocol_markup_projector_preserves_fragmented_think_tags_and_hides_tool_markup() -> None:
    projector = ProtocolMarkupProjector()

    pieces = [
        "<thi",
        "nk>先想",
        "一想</think><tool_",
        'call id="call_weather_1" name="get_weather">',
        '{"location":"Tokyo"}',
        "</tool_call><final_",
        "answer>最终答案</final_answer>",
    ]

    deltas = [projector.push(piece) for piece in pieces]

    assert deltas == [
        "",
        "<think>先想",
        "一想</think>",
        "",
        "",
        "",
        "最终答案",
    ]
    assert projector.visible_text == "<think>先想一想</think>最终答案"


def test_strip_tool_protocol_markup_keeps_think_but_removes_tool_and_final_tags() -> None:
    assert (
        strip_tool_protocol_markup(
            '<think>先想一想</think><tool_calls>[{"name":"get_weather","arguments":{"location":"Tokyo"}}]</tool_calls><final_answer>最终答案</final_answer>'
        )
        == "<think>先想一想</think>最终答案"
    )


def test_strip_tool_protocol_markup_can_hide_think_for_chat_completions() -> None:
    assert (
        strip_tool_protocol_markup(
            '<think>先想一想</think><tool_calls>[{"name":"get_weather","arguments":{"location":"Tokyo"}}]</tool_calls><final_answer>最终答案</final_answer>',
            include_think=False,
        )
        == "最终答案"
    )


def test_protocol_markup_projector_can_hide_fragmented_think_for_chat_completions() -> None:
    projector = ProtocolMarkupProjector(include_think=False)

    pieces = [
        "<thi",
        "nk>先想",
        "一想</think><final_",
        "answer>最终答案</final_answer>",
    ]

    deltas = [projector.push(piece) for piece in pieces]

    assert deltas == ["", "", "", "最终答案"]
    assert projector.visible_text == "最终答案"


def test_chunk_visible_text_preserves_complete_think_tag_boundaries() -> None:
    assert chunk_visible_text("<think>先想一想</think>最终答案") == [
        "<think>",
        "先想一想",
        "</think>",
        "最终答案",
    ]
