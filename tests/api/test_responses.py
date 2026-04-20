from fastapi.testclient import TestClient
import httpx
import json
import time

from openclaw_algae.providers.base import ChatResponse
from openclaw_algae.providers.base import ProviderRateLimitError
from openclaw_algae.api.app import create_app
import openclaw_algae.api.routes.responses as responses_route_module
import openclaw_algae.storage.response_store as response_store_module


class FakeRouter:
    def chat(self, request):
        assert request.model == "algae/deepseek/deepseek-chat"
        assert request.messages == [{"role": "user", "content": "hello"}]
        return ChatResponse(model=request.model, content="provider answer")


def test_responses_returns_openai_style_response(monkeypatch) -> None:
    monkeypatch.setattr(responses_route_module, "get_default_router", lambda: FakeRouter())
    client = TestClient(create_app())

    response = client.post(
        "/v1/responses",
        json={
            "model": "algae/deepseek/deepseek-chat",
            "input": "hello",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {"id", "object", "created_at", "status", "model", "output", "usage"}
    assert payload["object"] == "response"
    assert isinstance(payload["id"], str)
    assert payload["id"].startswith("resp-")
    assert payload["model"] == "algae/deepseek/deepseek-chat"
    assert isinstance(payload["created_at"], int)
    assert payload["status"] == "completed"
    assert len(payload["output"]) == 1
    assert set(payload["output"][0].keys()) == {"type", "id", "role", "status", "content"}
    assert payload["output"][0]["type"] == "message"
    assert isinstance(payload["output"][0]["id"], str)
    assert payload["output"][0]["status"] == "completed"
    assert payload["output"][0]["role"] == "assistant"
    assert payload["output"][0]["content"] == [{"type": "output_text", "text": "provider answer"}]
    assert payload["output"][0]["content"][0]["text"] == "provider answer"
    assert payload["usage"] == {
        "input_tokens": 0,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 0,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 0,
    }


def test_responses_route_reuses_provider_router(monkeypatch) -> None:
    monkeypatch.setattr(responses_route_module, "get_default_router", lambda: FakeRouter())
    client = TestClient(create_app())

    response = client.post(
        "/v1/responses",
        json={
            "model": "algae/deepseek/deepseek-chat",
            "input": "hello",
        },
    )

    assert response.status_code == 200
    assert response.json()["output"][0]["content"][0]["text"] == "provider answer"


class ContentItemRouter:
    def chat(self, request):
        assert request.model == "algae/deepseek/deepseek-chat"
        assert request.messages == [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "hello"},
                    {"type": "input_text", "text": "world"},
                ],
            }
        ]
        return ChatResponse(model=request.model, content="provider answer")


class StreamingMarkupRouter:
    def stream_chat(self, request):
        assert request.model == "algae/glm-cn/glm-5"
        yield "<final"
        yield "_answer>"
        yield "glm responses stream ok"
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


class DelayedPlainMessageRouter:
    def stream_chat(self, request):
        yield "最"
        time.sleep(0.08)
        yield "终"
        time.sleep(0.08)
        yield "答"
        time.sleep(0.08)
        yield "案"


def test_responses_wraps_content_items_as_user_message(monkeypatch) -> None:
    monkeypatch.setattr(responses_route_module, "get_default_router", lambda: ContentItemRouter())
    client = TestClient(create_app())

    response = client.post(
        "/v1/responses",
        json={
            "model": "algae/deepseek/deepseek-chat",
            "input": [
                {"type": "input_text", "text": "hello"},
                {"type": "input_text", "text": "world"},
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["output"][0]["content"][0]["text"] == "provider answer"


class InstructionsRouter:
    def chat(self, request):
        assert request.model == "algae/deepseek/deepseek-chat"
        assert request.messages == [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hello"},
        ]
        return ChatResponse(model=request.model, content="provider answer")


def test_responses_maps_instructions_to_system_message(monkeypatch) -> None:
    monkeypatch.setattr(responses_route_module, "get_default_router", lambda: InstructionsRouter())
    client = TestClient(create_app())

    response = client.post(
        "/v1/responses",
        json={
            "model": "algae/deepseek/deepseek-chat",
            "instructions": "be terse",
            "input": "hello",
        },
    )

    assert response.status_code == 200
    assert response.json()["output"][0]["content"][0]["text"] == "provider answer"


class HttpErrorRouter:
    def chat(self, request):
        req = httpx.Request("POST", "https://example.com/upstream")
        resp = httpx.Response(429, request=req, text="rate limited")
        raise httpx.HTTPStatusError("upstream rate limited", request=req, response=resp)


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


def test_responses_maps_upstream_http_errors_to_bad_gateway(monkeypatch) -> None:
    monkeypatch.setattr(responses_route_module, "get_default_router", lambda: HttpErrorRouter())
    client = TestClient(create_app())

    response = client.post(
        "/v1/responses",
        json={
            "model": "algae/deepseek/deepseek-chat",
            "input": "hello",
        },
    )

    assert response.status_code == 502
    assert response.json() == {
        "error": {
            "message": "upstream rate limited",
            "type": "api_error",
        }
    }


def test_responses_maps_rate_limit_errors(monkeypatch) -> None:
    monkeypatch.setattr(responses_route_module, "get_default_router", lambda: RateLimitRouter())
    client = TestClient(create_app())

    response = client.post(
        "/v1/responses",
        json={
            "model": "algae/doubao/doubao-seed-2.0",
            "input": "hello",
        },
    )

    assert response.status_code == 429
    assert response.json() == {
        "error": {
            "message": "Doubao rate limit exceeded",
            "type": "rate_limit_error",
        }
    }


def test_responses_supports_streaming_sse(monkeypatch) -> None:
    monkeypatch.setattr(responses_route_module, "get_default_router", lambda: FakeRouter())
    client = TestClient(create_app())

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "algae/deepseek/deepseek-chat",
            "input": "hello",
            "stream": True,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache, no-transform"
    assert response.headers["x-accel-buffering"] == "no"
    events = _parse_named_sse(body)
    event_names = [name for name, _ in events]
    assert event_names[:4] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
    ]
    assert event_names[-4:] == [
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert set(event_names[4:-4]) == {"response.output_text.delta"}
    assert len(event_names[4:-4]) >= 1

    created_name, created_payload = events[0]
    in_progress_name, in_progress_payload = events[1]
    output_added_name, output_added_payload = events[2]
    part_added_name, part_added_payload = events[3]
    delta_events = events[4:-4]
    text_done_name, text_done_payload = events[-4]
    part_done_name, part_done_payload = events[-3]
    item_done_name, item_done_payload = events[-2]
    completed_name, completed_payload = events[-1]

    assert created_name == "response.created"
    assert created_payload["type"] == "response.created"
    assert created_payload["response"]["status"] == "in_progress"
    assert in_progress_name == "response.in_progress"
    assert in_progress_payload["type"] == "response.in_progress"
    assert in_progress_payload["response"] == created_payload["response"]

    assert output_added_name == "response.output_item.added"
    assert output_added_payload["type"] == "response.output_item.added"
    assert output_added_payload["output_index"] == 0
    assert output_added_payload["item"]["status"] == "in_progress"
    assert output_added_payload["item"]["content"] == [{"type": "output_text", "text": ""}]

    assert part_added_name == "response.content_part.added"
    assert part_added_payload == {
        "type": "response.content_part.added",
        "item_id": output_added_payload["item"]["id"],
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "output_text", "text": ""},
    }

    streamed_delta = "".join(payload["delta"] for _, payload in delta_events)
    assert streamed_delta == "provider answer"
    for delta_name, delta_payload in delta_events:
        assert delta_name == "response.output_text.delta"
        assert delta_payload["type"] == "response.output_text.delta"
        assert delta_payload["item_id"] == output_added_payload["item"]["id"]
        assert delta_payload["output_index"] == 0
        assert delta_payload["content_index"] == 0

    assert text_done_name == "response.output_text.done"
    assert text_done_payload == {
        "type": "response.output_text.done",
        "item_id": output_added_payload["item"]["id"],
        "output_index": 0,
        "content_index": 0,
        "text": "provider answer",
    }

    assert part_done_name == "response.content_part.done"
    assert part_done_payload == {
        "type": "response.content_part.done",
        "item_id": output_added_payload["item"]["id"],
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "output_text", "text": "provider answer"},
    }

    assert item_done_name == "response.output_item.done"
    assert item_done_payload["type"] == "response.output_item.done"
    assert item_done_payload["output_index"] == 0
    assert item_done_payload["item"]["id"] == output_added_payload["item"]["id"]
    assert item_done_payload["item"]["status"] == "completed"
    assert item_done_payload["item"]["content"] == [{"type": "output_text", "text": "provider answer"}]

    assert completed_name == "response.completed"
    assert completed_payload["type"] == "response.completed"
    assert completed_payload["response"]["id"] == created_payload["response"]["id"]
    assert completed_payload["response"]["status"] == "completed"
    assert completed_payload["response"]["output"] == [item_done_payload["item"]]


class StreamRouter:
    def stream_chat(self, request):
        assert request.model == "algae/qwen-intl/qwen3.6-plus"
        assert request.messages == [{"role": "user", "content": "hello"}]
        yield "你好"
        yield "！"

    def chat(self, request):
        raise AssertionError("stream=true should use stream_chat when available")


def test_responses_prefers_incremental_stream_when_router_supports_it(monkeypatch) -> None:
    monkeypatch.setattr(responses_route_module, "get_default_router", lambda: StreamRouter())
    client = TestClient(create_app())

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "algae/qwen-intl/qwen3.6-plus",
            "input": "hello",
            "stream": True,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    events = _parse_named_sse(body)
    delta_events = [payload for name, payload in events if name == "response.output_text.delta"]
    assert "".join(item["delta"] for item in delta_events) == "你好！"
    assert events[-1][0] == "response.completed"


def test_responses_flushes_plain_message_deltas_incrementally(monkeypatch) -> None:
    monkeypatch.setattr(responses_route_module, "get_default_router", lambda: DelayedPlainMessageRouter())
    client = TestClient(create_app())

    output_text_events: list[tuple[float, str]] = []
    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "algae/qwen-intl/qwen3.6-plus",
            "input": "hello",
            "stream": True,
        },
    ) as response:
        started = time.perf_counter()
        event_name = ""
        for line in response.iter_lines():
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ").strip()
                continue
            if not line.startswith("data: "):
                continue
            payload = json.loads(line.removeprefix("data: "))
            if event_name == "response.output_text.delta":
                output_text_events.append((time.perf_counter() - started, payload["delta"]))

    assert response.status_code == 200
    assert "".join(chunk for _, chunk in output_text_events) == "最终答案"
    assert len(output_text_events) >= 2


def test_responses_preserve_think_tags_in_non_stream_output(monkeypatch) -> None:
    monkeypatch.setattr(responses_route_module, "get_default_router", lambda: ThinkingRouter())
    client = TestClient(create_app())

    response = client.post(
        "/v1/responses",
        json={
            "model": "algae/glm-cn/glm-5",
            "input": "hello",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["output"][0] == {
        "type": "reasoning",
        "id": payload["output"][0]["id"],
        "status": "completed",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": "先想一想"}],
    }
    assert payload["output"][1] == {
        "type": "message",
        "id": payload["output"][1]["id"],
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "最终答案"}],
    }


def test_responses_stream_preserve_think_tags(monkeypatch) -> None:
    monkeypatch.setattr(responses_route_module, "get_default_router", lambda: ThinkingRouter())
    client = TestClient(create_app())

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "algae/glm-cn/glm-5",
            "input": "hello",
            "stream": True,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    events = _parse_named_sse(body)
    reasoning_delta_events = [payload for name, payload in events if name == "response.reasoning_text.delta"]
    message_delta_events = [payload for name, payload in events if name == "response.output_text.delta"]
    assert "".join(item["delta"] for item in reasoning_delta_events) == "先想一想"
    assert "".join(item["delta"] for item in message_delta_events) == "最终答案"


def test_responses_returns_function_call_output(monkeypatch) -> None:
    monkeypatch.setattr(responses_route_module, "get_default_router", lambda: ToolCallRouter())
    client = TestClient(create_app())

    response = client.post(
        "/v1/responses",
        json={
            "model": "algae/qwen-intl/qwen3.6-plus",
            "input": "What's the weather in Tokyo?",
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
    assert payload["status"] == "incomplete"
    assert len(payload["output"]) == 1
    assert payload["output"][0] == {
        "type": "function_call",
        "id": payload["output"][0]["id"],
        "call_id": "call_weather_1",
        "name": "get_weather",
        "arguments": '{"location":"Tokyo"}',
    }


def test_responses_returns_think_message_before_function_call_output(monkeypatch) -> None:
    monkeypatch.setattr(responses_route_module, "get_default_router", lambda: ThinkingToolCallRouter())
    client = TestClient(create_app())

    response = client.post(
        "/v1/responses",
        json={
            "model": "algae/qwen-intl/qwen3.6-plus",
            "input": "What's the weather in Tokyo?",
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
    assert payload["status"] == "incomplete"
    assert payload["output"][0] == {
        "type": "reasoning",
        "id": payload["output"][0]["id"],
        "status": "completed",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": "先想一想"}],
    }
    assert payload["output"][1] == {
        "type": "function_call",
        "id": payload["output"][1]["id"],
        "call_id": "call_weather_1",
        "name": "get_weather",
        "arguments": '{"location":"Tokyo"}',
    }


def test_responses_streams_function_call_items(monkeypatch) -> None:
    monkeypatch.setattr(responses_route_module, "get_default_router", lambda: ToolCallRouter())
    client = TestClient(create_app())

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "algae/qwen-intl/qwen3.6-plus",
            "input": "What's the weather in Tokyo?",
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
    events = _parse_named_sse(body)
    assert [name for name, _ in events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]

    _, added_payload = events[2]
    delta_payloads = [payload for name, payload in events if name == "response.function_call_arguments.delta"]
    done_event = next(payload for name, payload in events if name == "response.output_item.done")
    completed_payload = events[-1][1]

    assert added_payload == {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "type": "function_call",
            "id": added_payload["item"]["id"],
            "call_id": "call_weather_1",
            "name": "get_weather",
            "arguments": "",
            "status": "in_progress",
        },
    }
    assert "".join(payload["delta"] for payload in delta_payloads) == '{"location":"Tokyo"}'
    assert done_event == {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "type": "function_call",
            "id": added_payload["item"]["id"],
            "call_id": "call_weather_1",
            "name": "get_weather",
            "arguments": '{"location":"Tokyo"}',
            "status": "completed",
        },
    }
    assert completed_payload["response"]["status"] == "incomplete"
    assert completed_payload["response"]["output"] == [
        {
            "type": "function_call",
            "id": added_payload["item"]["id"],
            "call_id": "call_weather_1",
            "name": "get_weather",
            "arguments": '{"location":"Tokyo"}',
        }
    ]


def test_responses_streams_think_message_before_function_call_items(monkeypatch) -> None:
    monkeypatch.setattr(responses_route_module, "get_default_router", lambda: ThinkingToolCallRouter())
    client = TestClient(create_app())

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "algae/qwen-intl/qwen3.6-plus",
            "input": "What's the weather in Tokyo?",
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
    events = _parse_named_sse(body)
    event_names = [name for name, _ in events]
    assert event_names[:3] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
    ]
    reasoning_done_index = event_names.index("response.output_item.done")
    assert event_names[reasoning_done_index - 2 : reasoning_done_index + 1] == [
        "response.reasoning_text.delta",
        "response.reasoning_text.done",
        "response.output_item.done",
    ]
    assert event_names[reasoning_done_index + 1 :] == [
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]

    reasoning_added = events[2][1]["item"]
    reasoning_delta_payloads = [payload for name, payload in events if name == "response.reasoning_text.delta"]
    reasoning_done = events[reasoning_done_index][1]["item"]
    tool_added = events[-5][1]["item"]
    tool_argument_deltas = [payload for name, payload in events if name == "response.function_call_arguments.delta"]
    tool_arguments_done = events[-3][1]
    tool_done = events[-2][1]["item"]
    completed = events[-1][1]["response"]

    assert "".join(payload["delta"] for payload in reasoning_delta_payloads) == "先想一想"
    assert reasoning_done == {
        "type": "reasoning",
        "id": reasoning_added["id"],
        "status": "completed",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": "先想一想"}],
    }
    assert tool_added == {
        "type": "function_call",
        "id": tool_added["id"],
        "call_id": "call_weather_1",
        "name": "get_weather",
        "arguments": "",
        "status": "in_progress",
    }
    assert "".join(payload["delta"] for payload in tool_argument_deltas) == '{"location":"Tokyo"}'
    assert tool_arguments_done == {
        "type": "response.function_call_arguments.done",
        "item_id": tool_added["id"],
        "output_index": 1,
        "arguments": '{"location":"Tokyo"}',
        "name": "get_weather",
    }
    assert tool_done == {
        "type": "function_call",
        "id": tool_added["id"],
        "call_id": "call_weather_1",
        "name": "get_weather",
        "arguments": '{"location":"Tokyo"}',
        "status": "completed",
    }
    assert completed["status"] == "incomplete"
    assert completed["output"] == [
        {
            "type": "reasoning",
            "id": reasoning_added["id"],
            "status": "completed",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "先想一想"}],
        },
        {
            "type": "function_call",
            "id": tool_added["id"],
            "call_id": "call_weather_1",
            "name": "get_weather",
            "arguments": '{"location":"Tokyo"}',
        },
    ]


def test_responses_stream_strips_protocol_tags_from_output_text(monkeypatch) -> None:
    monkeypatch.setattr(responses_route_module, "get_default_router", lambda: StreamingMarkupRouter())
    client = TestClient(create_app())

    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "algae/glm-cn/glm-5",
            "input": "hello",
            "stream": True,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    events = _parse_named_sse(body)
    delta_events = [payload for name, payload in events if name == "response.output_text.delta"]
    streamed_text = "".join(item["delta"] for item in delta_events)
    assert streamed_text == "glm responses stream ok"
    assert "<final_answer>" not in streamed_text
    assert "</final_answer>" not in streamed_text


class PreviousResponseRouter:
    def __init__(self) -> None:
        self.seen_requests: list[object] = []

    def chat(self, request):
        self.seen_requests.append(request)
        return ChatResponse(model=request.model, content=f"turn-{len(self.seen_requests)}")


def test_responses_previous_response_id_reuses_stored_messages(monkeypatch, tmp_path) -> None:
    router = PreviousResponseRouter()
    monkeypatch.setattr(responses_route_module, "get_default_router", lambda: router)
    monkeypatch.setattr(responses_route_module, "resolve_state_dir", lambda: tmp_path)
    monkeypatch.setattr(response_store_module, "_resolve_response_store_path", lambda state_dir: tmp_path / "responses.json")
    client = TestClient(create_app())

    first = client.post(
        "/v1/responses",
        json={
            "model": "algae/deepseek/deepseek-chat",
            "input": "hello",
        },
    )
    assert first.status_code == 200
    first_id = first.json()["id"]

    second = client.post(
        "/v1/responses",
        json={
            "model": "algae/deepseek/deepseek-chat",
            "previous_response_id": first_id,
            "input": "again",
        },
    )

    assert second.status_code == 200
    assert router.seen_requests[0].messages == [{"role": "user", "content": "hello"}]
    assert router.seen_requests[1].messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "turn-1"},
        {"role": "user", "content": "again"},
    ]


def test_responses_previous_response_id_rejects_unknown_id(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(responses_route_module, "get_default_router", lambda: FakeRouter())
    monkeypatch.setattr(responses_route_module, "resolve_state_dir", lambda: tmp_path)
    client = TestClient(create_app())

    response = client.post(
        "/v1/responses",
        json={
            "model": "algae/deepseek/deepseek-chat",
            "previous_response_id": "resp_missing",
            "input": "hello",
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "Unknown previous_response_id: resp_missing",
            "type": "invalid_request_error",
        }
    }


def test_responses_returns_bad_request_for_missing_model(monkeypatch) -> None:
    monkeypatch.setattr(responses_route_module, "get_default_router", lambda: FakeRouter())
    client = TestClient(create_app())

    response = client.post(
        "/v1/responses",
        json={
            "input": "hello",
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "Model is required.",
            "type": "invalid_request_error",
        }
    }


def _parse_named_sse(body: str) -> list[tuple[str, dict[str, object]]]:
    events: list[tuple[str, dict[str, object]]] = []
    for block in body.strip().split("\n\n"):
        event_name = ""
        data = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ").strip()
            elif line.startswith("data: "):
                data = line.removeprefix("data: ").strip()
        if event_name and data:
            events.append((event_name, json.loads(data)))
    return events
