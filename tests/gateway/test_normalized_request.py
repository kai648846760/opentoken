from opentoken.gateway.normalized import (
    normalize_chat_completions_request,
    normalize_responses_request,
)


def test_normalize_chat_completions_request_preserves_core_fields() -> None:
    payload = {
        "model": "algae/deepseek/deepseek-chat",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.2,
        "stream": False,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "tool_choice": "auto",
    }

    normalized = normalize_chat_completions_request(payload)

    assert normalized.model == "algae/deepseek/deepseek-chat"
    assert normalized.temperature == 0.2
    assert normalized.stream is False
    assert normalized.messages[0]["role"] == "user"
    assert normalized.tools == payload["tools"]
    assert normalized.tool_choice == "auto"


def test_normalize_responses_request_maps_input_to_user_message() -> None:
    payload = {
        "model": "algae/deepseek/deepseek-chat",
        "input": "hello",
    }

    normalized = normalize_responses_request(payload)

    assert normalized.model == "algae/deepseek/deepseek-chat"
    assert normalized.messages == [{"role": "user", "content": "hello"}]


def test_normalize_responses_request_wraps_content_item_list_as_user_message() -> None:
    payload = {
        "model": "algae/deepseek/deepseek-chat",
        "input": [
            {"type": "input_text", "text": "hello"},
            {"type": "input_text", "text": "world"},
        ],
    }

    normalized = normalize_responses_request(payload)

    assert normalized.messages == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "hello"},
                {"type": "input_text", "text": "world"},
            ],
        }
    ]


def test_normalize_responses_request_prepends_instructions_as_system_message() -> None:
    payload = {
        "model": "algae/deepseek/deepseek-chat",
        "instructions": "be terse",
        "input": "hello",
    }

    normalized = normalize_responses_request(payload)

    assert normalized.messages == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hello"},
    ]


def test_normalize_responses_request_maps_function_call_items_to_chat_messages() -> None:
    payload = {
        "model": "algae/deepseek/deepseek-chat",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "check weather"}],
            },
            {
                "type": "function_call",
                "id": "fc_item_1",
                "call_id": "call_weather_1",
                "name": "get_weather",
                "arguments": '{"location":"Tokyo"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_weather_1",
                "output": '{"temp":22}',
            },
        ],
    }

    normalized = normalize_responses_request(payload)

    assert normalized.messages == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "check weather"}],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_weather_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location":"Tokyo"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_weather_1",
            "content": '{"temp":22}',
        },
    ]

from opentoken.storage.file_store import create_file
import opentoken.gateway.normalized as normalized_module


def test_normalize_chat_completions_request_resolves_uploaded_file_ids(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(normalized_module, "resolve_state_dir", lambda: tmp_path)
    stored = create_file(
        tmp_path,
        filename="note.txt",
        content=b"hello from file",
        purpose="assistants",
        mime_type="text/plain",
    )

    normalized = normalize_chat_completions_request(
        {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "read this"},
                        {"type": "input_file", "file_id": stored["id"]},
                    ],
                }
            ],
        }
    )

    content = normalized.messages[0]["content"]
    assert isinstance(content, list)
    file_item = content[1]
    assert file_item["type"] == "input_file"
    assert file_item["filename"] == "note.txt"
    assert file_item["file_data"].startswith("data:text/plain;base64,")



def test_normalize_responses_request_resolves_uploaded_image_ids(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(normalized_module, "resolve_state_dir", lambda: tmp_path)
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc``\x00\x00\x00\x04\x00\x01"
        b"\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    stored = create_file(
        tmp_path,
        filename="pixel.png",
        content=png_bytes,
        purpose="vision",
        mime_type="image/png",
    )

    normalized = normalize_responses_request(
        {
            "model": "deepseek-chat",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_image", "file_id": stored["id"]},
                    ],
                }
            ],
        }
    )

    content = normalized.messages[0]["content"]
    assert isinstance(content, list)
    image_item = content[0]
    assert image_item["type"] == "input_image"
    assert image_item["image_url"]["url"].startswith("data:image/png;base64,")


def test_normalize_responses_request_remaps_developer_role_to_system() -> None:
    """Responses API uses `role: developer` for what Chat Completions calls
    `system`. The normalizer collapses that to system so downstream prompt
    builders see one canonical role."""
    payload = {
        "model": "algae/deepseek/deepseek-chat",
        "input": [{"type": "message", "role": "developer", "content": "rules"}],
    }
    normalized = normalize_responses_request(payload)
    assert normalized.messages == [{"role": "system", "content": "rules"}]


def test_normalize_responses_request_rejects_function_call_without_call_id() -> None:
    import pytest

    payload = {
        "model": "algae/deepseek/deepseek-chat",
        "input": [{"type": "function_call", "name": "x", "arguments": "{}"}],
    }
    with pytest.raises(RuntimeError, match="call_id is required"):
        normalize_responses_request(payload)


def test_normalize_responses_request_rejects_function_call_without_name() -> None:
    import pytest

    payload = {
        "model": "algae/deepseek/deepseek-chat",
        "input": [
            {"type": "function_call", "call_id": "call_1", "arguments": "{}"},
        ],
    }
    with pytest.raises(RuntimeError, match="name is required"):
        normalize_responses_request(payload)


def test_normalize_responses_request_rejects_function_call_output_without_call_id() -> None:
    import pytest

    payload = {
        "model": "algae/deepseek/deepseek-chat",
        "input": [{"type": "function_call_output", "output": "{}"}],
    }
    with pytest.raises(RuntimeError, match="call_id is required"):
        normalize_responses_request(payload)
