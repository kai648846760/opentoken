from fastapi.testclient import TestClient
import httpx
import json

from opentoken.providers.base import ChatResponse
from opentoken.providers.base import ProviderRateLimitError

from opentoken.api.app import create_app
import opentoken.api.routes.chat as chat_route_module


class FakeRouter:
    def chat(self, request):
        assert request.model == "algae/deepseek/deepseek-chat"
        assert request.messages == [{"role": "user", "content": "hello"}]
        return ChatResponse(model=request.model, content="provider answer")


def test_chat_completions_returns_openai_style_response(monkeypatch) -> None:
    monkeypatch.setattr(chat_route_module, "get_default_router", lambda: FakeRouter())
    client = TestClient(create_app())

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "algae/deepseek/deepseek-chat",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {
        "id",
        "object",
        "created",
        "model",
        "choices",
        "usage",
        "system_fingerprint",
    }
    assert payload["object"] == "chat.completion"
    assert isinstance(payload["id"], str)
    assert payload["id"].startswith("chatcmpl-")
    assert isinstance(payload["created"], int)
    assert payload["model"] == "algae/deepseek/deepseek-chat"
    assert isinstance(payload["system_fingerprint"], str) and payload["system_fingerprint"]
    assert len(payload["choices"]) == 1
    assert set(payload["choices"][0].keys()) == {"index", "message", "finish_reason"}
    assert set(payload["choices"][0]["message"].keys()) == {"role", "content"}
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert payload["choices"][0]["message"]["content"] == "provider answer"
    # usage is now estimated, not hardcoded zero — assert structure + non-negative ints.
    assert set(payload["usage"].keys()) == {"prompt_tokens", "completion_tokens", "total_tokens"}
    assert all(isinstance(v, int) and v >= 0 for v in payload["usage"].values())
    assert payload["usage"]["prompt_tokens"] > 0
    assert payload["usage"]["completion_tokens"] > 0
    assert (
        payload["usage"]["total_tokens"]
        == payload["usage"]["prompt_tokens"] + payload["usage"]["completion_tokens"]
    )


class HttpErrorRouter:
    def chat(self, request):
        req = httpx.Request("POST", "https://example.com/upstream")
        resp = httpx.Response(403, request=req, text="forbidden")
        raise httpx.HTTPStatusError("upstream rejected request", request=req, response=resp)


class RateLimitRouter:
    def chat(self, request):
        raise ProviderRateLimitError("Doubao rate limit exceeded")


class ToolCallRouter:
    def chat(self, request):
        return ChatResponse(
            model=request.model,
            content=None,
            tool_calls=[
                {
                    "id": "call_weather_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location":"Tokyo"}',
                    },
                }
            ],
            finish_reason="tool_calls",
        )


class StreamingMarkupRouter:
    def stream_chat(self, request):
        assert request.model == "algae/glm-cn/glm-5"
        yield "<final"
        yield "_answer>"
        yield "glm stream ok"
        yield "</final_answer>"


class ThinkingRouter:
    def chat(self, request):
        return ChatResponse(
            model=request.model,
            content="<think>先想一想</think><final_answer>最终答案</final_answer>",
        )

    def stream_chat(self, request):
        yield "<thi"
        yield "nk>先想一想</think><final_answer>最"
        yield "终答案</final_answer>"


class ThinkingToolCallRouter:
    def chat(self, request):
        return ChatResponse(
            model=request.model,
            content="<think>先想一想</think>",
            tool_calls=[
                {
                    "id": "call_weather_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location":"Tokyo"}',
                    },
                }
            ],
            finish_reason="tool_calls",
        )


def test_chat_completions_maps_upstream_http_errors_to_bad_gateway(monkeypatch) -> None:
    monkeypatch.setattr(chat_route_module, "get_default_router", lambda: HttpErrorRouter())
    client = TestClient(create_app())

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "algae/deepseek/deepseek-chat",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 502
    assert response.json() == {
        "error": {
            "message": "upstream rejected request",
            "type": "api_error",
        }
    }


def test_chat_completions_maps_rate_limit_errors(monkeypatch) -> None:
    monkeypatch.setattr(chat_route_module, "get_default_router", lambda: RateLimitRouter())
    client = TestClient(create_app())

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "algae/doubao/doubao-seed-2.0",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 429
    assert response.json() == {
        "error": {
            "message": "Doubao rate limit exceeded",
            "type": "rate_limit_error",
        }
    }


def test_chat_completions_supports_streaming_sse(monkeypatch) -> None:
    monkeypatch.setattr(chat_route_module, "get_default_router", lambda: FakeRouter())
    client = TestClient(create_app())

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "algae/deepseek/deepseek-chat",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache, no-transform"
    assert response.headers["x-accel-buffering"] == "no"
    events = _parse_chat_sse(body)
    assert events[-1] == "[DONE]"
    chunks = events[:-1]
    assert len(chunks) >= 3
    first = chunks[0]
    final = chunks[-1]
    assert first["object"] == "chat.completion.chunk"
    assert final["object"] == "chat.completion.chunk"
    assert first["id"] == final["id"]
    assert first["created"] == final["created"]
    assert first["model"] == "algae/deepseek/deepseek-chat"
    assert first["choices"] == [
        {
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None,
        }
    ]
    streamed_text = "".join(
        str(chunk["choices"][0]["delta"].get("content", ""))
        for chunk in chunks[1:-1]
    )
    assert streamed_text == "provider answer"
    assert final["choices"] == [
        {
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }
    ]


class StreamRouter:
    def stream_chat(self, request):
        assert request.model == "algae/qwen-intl/qwen3.6-plus"
        assert request.messages == [{"role": "user", "content": "hello"}]
        yield "你好"
        yield "！"

    def chat(self, request):
        raise AssertionError("stream=true should use stream_chat when available")


def test_chat_completions_prefers_incremental_stream_when_router_supports_it(monkeypatch) -> None:
    monkeypatch.setattr(chat_route_module, "get_default_router", lambda: StreamRouter())
    client = TestClient(create_app())

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "algae/qwen-intl/qwen3.6-plus",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache, no-transform"
    assert response.headers["x-accel-buffering"] == "no"
    events = _parse_chat_sse(body)
    chunks = events[:-1]
    streamed_text = "".join(
        str(chunk["choices"][0]["delta"].get("content", ""))
        for chunk in chunks[1:-1]
    )
    assert streamed_text == "你好！"
    assert events[-1] == "[DONE]"


class CoarseStreamRouter:
    def stream_chat(self, request):
        assert request.model == "algae/qwen-intl/qwen3.6-plus"
        yield "这是一段非常长的中文内容，没有空格，也不应该在 chat/completions 里被一次性整块吐出给客户端。"

    def chat(self, request):
        raise AssertionError("stream=true should use stream_chat when available")


def test_chat_completions_rechunks_coarse_stream_pieces_into_smaller_deltas(monkeypatch) -> None:
    monkeypatch.setattr(chat_route_module, "get_default_router", lambda: CoarseStreamRouter())
    client = TestClient(create_app())

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "algae/qwen-intl/qwen3.6-plus",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    events = _parse_chat_sse(body)
    content_chunks = [
        str(chunk["choices"][0]["delta"].get("content", ""))
        for chunk in events[:-1]
        if isinstance(chunk, dict) and "content" in chunk["choices"][0]["delta"]
    ]
    assert "".join(content_chunks) == (
        "这是一段非常长的中文内容，没有空格，也不应该在 chat/completions 里被一次性整块吐出给客户端。"
    )
    assert len(content_chunks) >= 2


def test_chat_completions_returns_tool_calls(monkeypatch) -> None:
    monkeypatch.setattr(chat_route_module, "get_default_router", lambda: ToolCallRouter())
    client = TestClient(create_app())

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "algae/qwen-intl/qwen3.6-plus",
            "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
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
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"] == [
        {
            "index": 0,
            "message": {
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
            "finish_reason": "tool_calls",
        }
    ]


def test_chat_completions_streams_tool_calls(monkeypatch) -> None:
    monkeypatch.setattr(chat_route_module, "get_default_router", lambda: ToolCallRouter())
    client = TestClient(create_app())

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "algae/qwen-intl/qwen3.6-plus",
            "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
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
            "stream": True,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    events = _parse_chat_sse(body)
    assert events[0]["choices"] == [
        {
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None,
        }
    ]
    assert events[1]["choices"] == [
        {
            "index": 0,
            "delta": {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_weather_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "",
                        },
                    }
                ],
            },
            "finish_reason": None,
        }
    ]
    assert events[2]["choices"] == [
        {
            "index": 0,
            "delta": {
                "tool_calls": [
                    {
                        "index": 0,
                        "function": {
                            "arguments": '{"location":"Tokyo"}',
                        },
                    }
                ],
            },
            "finish_reason": None,
        }
    ]
    assert _collect_streamed_tool_calls(events) == [
        {
            "id": "call_weather_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location":"Tokyo"}',
            },
        }
    ]
    assert events[3]["choices"] == [
        {
            "index": 0,
            "delta": {},
            "finish_reason": "tool_calls",
        }
    ]
    assert events[4] == "[DONE]"


def test_chat_completions_hides_think_content_before_tool_calls(monkeypatch) -> None:
    monkeypatch.setattr(chat_route_module, "get_default_router", lambda: ThinkingToolCallRouter())
    client = TestClient(create_app())

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "algae/qwen-intl/qwen3.6-plus",
            "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
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
            "stream": True,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    events = _parse_chat_sse(body)
    assert events[0]["choices"] == [
        {
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None,
        }
    ]
    content_deltas = [
        str(event["choices"][0]["delta"].get("content", ""))
        for event in events[1:]
        if isinstance(event, dict) and "content" in event["choices"][0]["delta"]
    ]
    assert "".join(content_deltas) == ""
    tool_call_events = [
        event
        for event in events
        if isinstance(event, dict) and "tool_calls" in event["choices"][0]["delta"]
    ]
    assert len(tool_call_events) == 2
    assert tool_call_events[0]["choices"] == [
        {
            "index": 0,
            "delta": {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_weather_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "",
                        },
                    }
                ],
            },
            "finish_reason": None,
        }
    ]
    assert tool_call_events[1]["choices"] == [
        {
            "index": 0,
            "delta": {
                "tool_calls": [
                    {
                        "index": 0,
                        "function": {
                            "arguments": '{"location":"Tokyo"}',
                        },
                    }
                ],
            },
            "finish_reason": None,
        }
    ]
    assert _collect_streamed_tool_calls(events) == [
        {
            "id": "call_weather_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location":"Tokyo"}',
            },
        }
    ]
    assert events[-2]["choices"] == [
        {
            "index": 0,
            "delta": {},
            "finish_reason": "tool_calls",
        }
    ]
    assert events[-1] == "[DONE]"


def test_chat_completions_stream_strips_protocol_tags_from_content(monkeypatch) -> None:
    monkeypatch.setattr(chat_route_module, "get_default_router", lambda: StreamingMarkupRouter())
    client = TestClient(create_app())

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "algae/glm-cn/glm-5",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    events = _parse_chat_sse(body)
    streamed_text = "".join(
        str(chunk["choices"][0]["delta"].get("content", ""))
        for chunk in events[:-1]
        if isinstance(chunk, dict)
    )
    assert streamed_text == "glm stream ok"
    assert "<final_answer>" not in streamed_text
    assert "</final_answer>" not in streamed_text


def test_chat_completions_hides_think_tags_in_non_stream_content(monkeypatch) -> None:
    monkeypatch.setattr(chat_route_module, "get_default_router", lambda: ThinkingRouter())
    client = TestClient(create_app())

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "algae/glm-cn/glm-5",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "最终答案"


def test_chat_completions_hides_think_only_content_when_returning_tool_calls(monkeypatch) -> None:
    monkeypatch.setattr(chat_route_module, "get_default_router", lambda: ThinkingToolCallRouter())
    client = TestClient(create_app())

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "algae/qwen-intl/qwen3.6-plus",
            "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
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
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] is None


def test_chat_completions_stream_hides_think_tags(monkeypatch) -> None:
    monkeypatch.setattr(chat_route_module, "get_default_router", lambda: ThinkingRouter())
    client = TestClient(create_app())

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "algae/glm-cn/glm-5",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    events = _parse_chat_sse(body)
    streamed_text = "".join(
        str(chunk["choices"][0]["delta"].get("content", ""))
        for chunk in events[:-1]
        if isinstance(chunk, dict)
    )
    assert streamed_text == "最终答案"


def test_chat_completions_stream_preserves_think_tags_for_reasoning_models(monkeypatch) -> None:
    monkeypatch.setattr(chat_route_module, "get_default_router", lambda: ThinkingRouter())
    client = TestClient(create_app())

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "algae/deepseek/deepseek-reasoner",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    events = _parse_chat_sse(body)
    streamed_text = "".join(
        str(chunk["choices"][0]["delta"].get("content", ""))
        for chunk in events[:-1]
        if isinstance(chunk, dict)
    )
    assert streamed_text == "<think>先想一想</think>最终答案"


def test_chat_completions_returns_bad_request_for_missing_model(monkeypatch) -> None:
    monkeypatch.setattr(chat_route_module, "get_default_router", lambda: FakeRouter())
    client = TestClient(create_app())

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "Model is required.",
            "type": "invalid_request_error",
        }
    }


def _parse_chat_sse(body: str) -> list[object]:
    events: list[object] = []
    for block in body.strip().split("\n\n"):
        data_lines = [line.removeprefix("data: ").strip() for line in block.splitlines() if line.startswith("data: ")]
        if not data_lines:
            continue
        data = "\n".join(data_lines)
        if data == "[DONE]":
            events.append(data)
            continue
        events.append(json.loads(data))
    return events


def _collect_streamed_tool_calls(events: list[object]) -> list[dict[str, object]]:
    collected: dict[int, dict[str, object]] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        choices = event.get("choices", [])
        if not isinstance(choices, list) or not choices:
            continue
        delta = choices[0].get("delta", {})
        if not isinstance(delta, dict):
            continue
        streamed_tool_calls = delta.get("tool_calls", [])
        if not isinstance(streamed_tool_calls, list):
            continue
        for item in streamed_tool_calls:
            if not isinstance(item, dict):
                continue
            index = int(item.get("index", 0))
            current = collected.setdefault(
                index,
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            item_id = item.get("id")
            if isinstance(item_id, str) and item_id:
                current["id"] = item_id
            item_type = item.get("type")
            if isinstance(item_type, str) and item_type:
                current["type"] = item_type
            function = item.get("function", {})
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if isinstance(name, str) and name:
                current["function"]["name"] = name
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                current["function"]["arguments"] += arguments
    return [collected[index] for index in sorted(collected)]


class MissingCredsRouter:
    def chat(self, request):
        raise RuntimeError("Missing deepseek credentials. Run `opentoken login deepseek` first.")


class UpstreamFailedRouter:
    def chat(self, request):
        raise RuntimeError("All browser workers failed for doubao: page crashed")


def test_chat_completions_maps_missing_credentials_to_401(monkeypatch) -> None:
    monkeypatch.setattr(chat_route_module, "get_default_router", lambda: MissingCredsRouter())
    client = TestClient(create_app())

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "algae/deepseek/deepseek-chat",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    # A provider the user hasn't logged in to is an auth problem, not a malformed
    # request — must not be 400 invalid_request_error.
    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"


def test_chat_completions_maps_upstream_provider_failure_to_502(monkeypatch) -> None:
    monkeypatch.setattr(chat_route_module, "get_default_router", lambda: UpstreamFailedRouter())
    client = TestClient(create_app())

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "algae/doubao/doubao-seed-2.0",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    # Worker / upstream failure is a gateway-side error, not a client request
    # error — 502, not 400.
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "api_error"
