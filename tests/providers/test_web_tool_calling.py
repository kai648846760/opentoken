from __future__ import annotations

import pytest

from opentoken.gateway.normalized import NormalizedChatRequest
from opentoken.providers.web_tool_calling import (
    build_web_tool_prompt,
    parse_web_tool_response,
)


def test_parse_web_tool_response_extracts_fenced_tool_json() -> None:
    content, tool_calls, finish_reason = parse_web_tool_response(
        '```tool_json\n{"tool":"write_file","parameters":{"path":"/tmp/a.txt","content":"hello"}}\n```'
    )

    assert content is None
    assert finish_reason == "tool_calls"
    assert tool_calls == [
        {
            "id": "call_write_file_1",
            "type": "function",
            "function": {
                "name": "write_file",
                "arguments": '{"path":"/tmp/a.txt","content":"hello"}',
            },
        }
    ]


def test_parse_web_tool_response_extracts_xml_openai_format() -> None:
    content, tool_calls, finish_reason = parse_web_tool_response(
        '<tool_call>{"name":"read_file","arguments":{"path":"/tmp/a.txt"}}</tool_call>'
    )

    assert content is None
    assert finish_reason == "tool_calls"
    assert tool_calls[0]["function"]["name"] == "read_file"
    assert tool_calls[0]["function"]["arguments"] == '{"path":"/tmp/a.txt"}'


def test_parse_web_tool_response_extracts_qwen_style_xml_tool_call() -> None:
    content, tool_calls, finish_reason = parse_web_tool_response(
        '<tool_call id="call_weather_1" name="get_weather">{"location":"Tokyo"}</tool_call>'
    )

    assert content is None
    assert finish_reason == "tool_calls"
    assert tool_calls == [
        {
            "id": "call_weather_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location":"Tokyo"}',
            },
        }
    ]


def test_parse_web_tool_response_extracts_multiple_xml_tool_calls() -> None:
    content, tool_calls, finish_reason = parse_web_tool_response(
        '<tool_call id="call_1" name="read_file">{"path":"/tmp/a.txt"}</tool_call>'
        '<tool_call id="call_2" name="message">{"text":"done"}</tool_call>'
    )

    assert content is None
    assert finish_reason == "tool_calls"
    assert tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": '{"path":"/tmp/a.txt"}',
            },
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {
                "name": "message",
                "arguments": '{"text":"done"}',
            },
        },
    ]


def test_parse_web_tool_response_extracts_parallel_tagged_tool_calls() -> None:
    content, tool_calls, finish_reason = parse_web_tool_response(
        '<think>Need tools.</think><tool_calls>[{"name":"write_file","arguments":{"path":"/tmp/a.txt","content":"hello"}},{"name":"message","arguments":{"text":"done"}}]</tool_calls>'
    )

    assert content == "<think>Need tools.</think>"
    assert finish_reason == "tool_calls"
    assert [call["function"]["name"] for call in tool_calls] == ["write_file", "message"]
    assert tool_calls[0]["function"]["arguments"] == '{"path":"/tmp/a.txt","content":"hello"}'
    assert tool_calls[1]["function"]["arguments"] == '{"text":"done"}'


def test_parse_web_tool_response_salvages_single_tool_call_object_without_array() -> None:
    """A single tool-call OBJECT (not wrapped in an array) is a common model
    mistake — salvage it instead of hard-failing into a repair round-trip."""
    content, tool_calls, finish_reason = parse_web_tool_response(
        '<tool_calls>{"name":"web_search","arguments":{"query":"hi"}}</tool_calls>'
    )
    assert finish_reason == "tool_calls"
    assert [call["function"]["name"] for call in tool_calls] == ["web_search"]
    assert tool_calls[0]["function"]["arguments"] == '{"query":"hi"}'


def test_parse_web_tool_response_xml_body_non_object_treated_as_raw_arguments() -> None:
    """An XML tool_call whose body decodes to a non-object JSON value (a bare
    string) must be treated as raw arguments, not rejected outright."""
    content, tool_calls, finish_reason = parse_web_tool_response(
        '<tool_call id="c1" name="echo">"just a string"</tool_call>'
    )
    assert finish_reason == "tool_calls"
    assert tool_calls[0]["function"]["name"] == "echo"
    # The raw string body is preserved verbatim as the arguments payload.
    assert tool_calls[0]["function"]["arguments"] == '"just a string"'


def test_parse_web_tool_response_coerces_plain_reasoning_prefix_into_think_block() -> None:
    content, tool_calls, finish_reason = parse_web_tool_response(
        '用户需要查询东京的天气，需要调用get_weather工具，传入location参数为Tokyo\n'
        '<tool_calls>[{"name":"get_weather","arguments":{"location":"Tokyo"}}]</tool_calls>',
        available_tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                },
            }
        ],
        tool_choice="required",
    )

    assert content == "<think>用户需要查询东京的天气，需要调用get_weather工具，传入location参数为Tokyo</think>"
    assert finish_reason == "tool_calls"
    assert tool_calls == [
        {
            "id": "call_get_weather_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location":"Tokyo"}',
            },
        }
    ]


def test_parse_web_tool_response_coerces_malformed_orphan_protocol_tags_into_think_block() -> None:
    content, tool_calls, finish_reason = parse_web_tool_response(
        '<tool_call>用户要求我先思考，然后必须调用 get_weather 查询 Tokyo 的天气。</think>'
        '<tool_calls>[{"name":"get_weather","arguments":{"location":"Tokyo"}}]</tool_calls>',
        available_tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                },
            }
        ],
        tool_choice="required",
    )

    assert content == "<think>用户要求我先思考，然后必须调用 get_weather 查询 Tokyo 的天气。</think>"
    assert finish_reason == "tool_calls"
    assert tool_calls == [
        {
            "id": "call_get_weather_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location":"Tokyo"}',
            },
        }
    ]


def test_parse_web_tool_response_extracts_json_array_from_fenced_tool_calls_block() -> None:
    content, tool_calls, finish_reason = parse_web_tool_response(
        '<tool_calls>```json\n[{"name":"web_search","arguments":{"query":"OpenAI"}}]\n```</tool_calls>',
        available_tools=[{"type": "web_search_preview"}],
        tool_choice="required",
    )

    assert content is None
    assert finish_reason == "tool_calls"
    assert tool_calls[0]["function"]["name"] == "web_search"
    assert tool_calls[0]["function"]["arguments"] == '{"query":"OpenAI"}'


def test_parse_web_tool_response_extracts_json_array_from_noisy_tool_calls_block() -> None:
    content, tool_calls, finish_reason = parse_web_tool_response(
        '<tool_calls>Use the tool now:\n[{"name":"web_search","arguments":{"query":"OpenAI"}}]\n</tool_calls>',
        available_tools=[{"type": "web_search_preview"}],
        tool_choice="required",
    )

    assert content is None
    assert finish_reason == "tool_calls"
    assert tool_calls[0]["function"]["name"] == "web_search"
    assert tool_calls[0]["function"]["arguments"] == '{"query":"OpenAI"}'


def test_parse_web_tool_response_extracts_tagged_final_answer() -> None:
    content, tool_calls, finish_reason = parse_web_tool_response(
        "<think>Done.</think><final_answer>All finished.</final_answer>"
    )

    assert content == "<think>Done.</think>All finished."
    assert finish_reason == "stop"
    assert tool_calls == []


def test_parse_web_tool_response_rejects_malformed_tagged_output_without_leaking_raw_thinking() -> None:
    with pytest.raises(RuntimeError, match="strict tagged tool protocol"):
        parse_web_tool_response("<think>Need tools.</think>Tool read_file does not exists.")


def test_parse_web_tool_response_rejects_tool_call_when_tool_choice_is_none() -> None:
    with pytest.raises(RuntimeError, match="tool_choice='none'"):
        parse_web_tool_response(
            '<tool_call>{"name":"read_file","arguments":{"path":"/tmp/a.txt"}}</tool_call>',
            available_tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            tool_choice="none",
        )


def test_parse_web_tool_response_accepts_versioned_web_search_tool_types() -> None:
    content, tool_calls, finish_reason = parse_web_tool_response(
        '<tool_calls>[{"name":"web_search","arguments":{"query":"上海天气"}}]</tool_calls>',
        available_tools=[{"type": "web_search_preview_2026_01_01"}],
        tool_choice="required",
    )

    assert content is None
    assert finish_reason == "tool_calls"
    assert tool_calls[0]["function"]["name"] == "web_search"


def test_parse_web_tool_response_accepts_versioned_web_fetch_tool_types() -> None:
    content, tool_calls, finish_reason = parse_web_tool_response(
        '<tool_calls>[{"name":"web_fetch","arguments":{"url":"https://example.com"}}]</tool_calls>',
        available_tools=[{"type": "web_fetch_preview_2026_01_01"}],
        tool_choice="required",
    )

    assert content is None
    assert finish_reason == "tool_calls"
    assert tool_calls[0]["function"]["name"] == "web_fetch"


def test_parse_web_tool_response_accepts_builtin_web_search_preview_tools() -> None:
    content, tool_calls, finish_reason = parse_web_tool_response(
        '<tool_calls>[{"name":"web_search","arguments":{"query":"上海天气"}}]</tool_calls>',
        available_tools=[{"type": "web_search_preview"}],
        tool_choice="required",
    )

    assert content is None
    assert finish_reason == "tool_calls"
    assert tool_calls == [
        {
            "id": "call_web_search_1",
            "type": "function",
            "function": {
                "name": "web_search",
                "arguments": '{"query":"上海天气"}',
            },
        }
    ]


def test_parse_web_tool_response_aliases_write_file_to_write() -> None:
    content, tool_calls, finish_reason = parse_web_tool_response(
        '<tool_calls>[{"name":"write_file","arguments":{"path":"/tmp/a.txt","content":"hello"}}]</tool_calls>',
        available_tools=[
            {
                "type": "function",
                "function": {
                    "name": "write",
                    "parameters": {"type": "object"},
                },
            }
        ],
        tool_choice="required",
    )

    assert content is None
    assert finish_reason == "tool_calls"
    assert tool_calls[0]["function"]["name"] == "write"


def test_parse_web_tool_response_rejects_final_answer_when_tool_choice_is_required() -> None:
    with pytest.raises(RuntimeError, match="requires a tool call"):
        parse_web_tool_response(
            "<final_answer>All finished.</final_answer>",
            available_tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            tool_choice="required",
        )


def test_build_web_tool_prompt_includes_tool_defs_and_history() -> None:
    request = NormalizedChatRequest(
        model="algae/chatgpt/gpt-4",
        messages=[
            {"role": "system", "content": "be precise"},
            {"role": "user", "content": "把 /tmp/demo.txt 写入 hello"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_write_1",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": '{"path":"/tmp/demo.txt","content":"hello"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_write_1",
                "content": '{"ok":true}',
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            }
        ],
        tool_choice="required",
    )

    prompt = build_web_tool_prompt(request, provider="chatgpt")

    assert "You must respond using only the following XML-like tags" in prompt
    assert "<tool_calls>" in prompt
    assert "<final_answer>" in prompt
    assert "Tool choice for this response is required." in prompt
    assert "write_file" in prompt
    assert "Use the strict tagged tool protocol when responding." in prompt
    assert 'Assistant tool calls: call_write_1' in prompt
    assert "<tool_result>" in prompt
    assert 'Tool result for call_id=call_write_1:' in prompt


def test_build_web_tool_prompt_supports_responses_style_function_tools() -> None:
    request = NormalizedChatRequest(
        model="algae/chatgpt/gpt-4",
        messages=[{"role": "user", "content": "上海天气"}],
        tools=[
            {
                "type": "function",
                "name": "web_search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ],
        tool_choice="required",
    )

    prompt = build_web_tool_prompt(request, provider="chatgpt")

    assert "- web_search(query: string (required)): Search the web" in prompt


def test_balanced_bracket_substring_ignores_trailing_prose() -> None:
    """Trailing prose with stray brackets must not bleed into the extracted
    JSON fragment (the old find/rfind grabbed up to the last ']')."""
    from opentoken.providers.web_tool_calling import _extract_json_container_fragment

    extracted = _extract_json_container_fragment('[{"name":"x","arguments":{}}] note: see [docs]')
    assert extracted == '[{"name":"x","arguments":{}}]'


def test_balanced_bracket_substring_respects_string_brackets() -> None:
    """A bracket inside a JSON string value must not affect depth counting."""
    from opentoken.providers.web_tool_calling import _extract_json_container_fragment

    extracted = _extract_json_container_fragment('prefix [{"q":"a [bracketed] phrase"}] suffix')
    assert extracted == '[{"q":"a [bracketed] phrase"}]'
    import json as _json
    assert _json.loads(extracted) == [{"q": "a [bracketed] phrase"}]
