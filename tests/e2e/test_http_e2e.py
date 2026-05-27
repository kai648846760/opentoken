import json
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

import opentoken.api.routes.chat as chat_route_module
import opentoken.api.routes.responses as responses_route_module
from opentoken.api.app import create_app
from opentoken.gateway.normalized import NormalizedChatRequest
from opentoken.gateway.router import ProviderRouter
from opentoken.models.provider_credentials import ProviderCredentialRecord
from opentoken.providers.base import ChatResponse, ProviderAdapter
from opentoken.storage.provider_store import save_provider_credentials


class StaticAdapter(ProviderAdapter):
    def __init__(self, content: str) -> None:
        self._content = content

    def chat(
        self,
        request: NormalizedChatRequest,
        credentials: ProviderCredentialRecord | None = None,
    ) -> ChatResponse:
        assert credentials is not None
        return ChatResponse(model=request.model, content=self._content)


class UpstreamErrorAdapter(ProviderAdapter):
    def chat(
        self,
        request: NormalizedChatRequest,
        credentials: ProviderCredentialRecord | None = None,
    ) -> ChatResponse:
        req = httpx.Request("POST", "https://example.com/upstream")
        resp = httpx.Response(429, request=req, text="rate limited")
        raise httpx.HTTPStatusError("upstream rate limited", request=req, response=resp)


class DelayedThinkingStreamRouter:
    def stream_chat(self, request):
        assert request.messages
        yield "<thi"
        time.sleep(0.08)
        yield "nk>先想"
        time.sleep(0.08)
        yield "一想</think>"
        time.sleep(0.08)
        yield "<final_answer>最终"
        time.sleep(0.08)
        yield "答案</final_answer>"


class DelayedCoarseChatStreamRouter:
    def stream_chat(self, request):
        assert request.messages
        time.sleep(0.08)
        yield "这是一段非常长的中文内容，没有空格，也不应该在 chat/completions 里被一次性整块吐出给客户端。"


class ThinkingToolCallHttpRouter:
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


@pytest.fixture
def http_server(monkeypatch, tmp_path: Path):
    servers: list[tuple[uvicorn.Server, threading.Thread]] = []

    def start(
        *,
        chat_router: ProviderRouter | None = None,
        responses_router: ProviderRouter | None = None,
        api_key: str = "test-key",
    ) -> tuple[str, dict[str, str]]:
        state_dir = tmp_path / ".opentoken"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "config.json").write_text(
            json.dumps(
                {
                    "api_key": api_key,
                    "host": "127.0.0.1",
                    "port": 32117,
                }
            ),
            encoding="utf-8",
        )

        if chat_router is not None:
            monkeypatch.setattr(chat_route_module, "get_default_router", lambda: chat_router)
        if responses_router is not None:
            monkeypatch.setattr(responses_route_module, "get_default_router", lambda: responses_router)

        port = _find_free_port()
        server = uvicorn.Server(
            uvicorn.Config(
                app=create_app(),
                host="127.0.0.1",
                port=port,
                log_level="warning",
            )
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        _wait_for_server(f"http://127.0.0.1:{port}/health")
        servers.append((server, thread))
        return f"http://127.0.0.1:{port}", {"Authorization": f"Bearer {api_key}"}

    yield start

    for server, thread in reversed(servers):
        server.should_exit = True
        thread.join(timeout=5)


def test_models_endpoint_works_over_real_http(http_server) -> None:
    base_url, headers = http_server()

    with httpx.Client(base_url=base_url, timeout=10.0, trust_env=False) as client:
        unauthorized = client.get("/v1/models")
        authorized = client.get("/v1/models", headers=headers)

    assert unauthorized.status_code == 401
    assert unauthorized.json() == {
        "error": {
            "message": "Invalid or missing API key.",
            "type": "authentication_error",
        }
    }
    assert authorized.status_code == 200
    payload = authorized.json()
    assert payload["object"] == "list"
    assert isinstance(payload["data"], list)
    # The catalog is now derived from live discovery, so the specific provider
    # entries depend on which providers are logged in at test time. The contract
    # we still expect over HTTP: every entry is OpenAI-shaped with owned_by ==
    # "opentoken".
    for item in payload["data"]:
        assert set(item.keys()) == {"id", "object", "owned_by"}
        assert item["object"] == "model"
        assert item["owned_by"] == "opentoken"


def test_post_endpoints_require_auth_over_real_http(http_server) -> None:
    base_url, _headers = http_server()

    with httpx.Client(base_url=base_url, timeout=10.0, trust_env=False) as client:
        chat_response = client.post(
            "/v1/chat/completions",
            json={
                "model": "algae/deepseek/deepseek-chat",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        responses_response = client.post(
            "/v1/responses",
            json={
                "model": "algae/deepseek/deepseek-chat",
                "input": "hello",
            },
        )

    expected = {
        "error": {
            "message": "Invalid or missing API key.",
            "type": "authentication_error",
        }
    }
    assert chat_response.status_code == 401
    assert chat_response.json() == expected
    assert responses_response.status_code == 401
    assert responses_response.json() == expected


def test_chat_completions_round_trip_over_real_http(http_server, tmp_path: Path) -> None:
    router = _provider_router_with_adapter(
        tmp_path,
        provider="deepseek",
        adapter=StaticAdapter("e2e provider answer"),
    )
    base_url, headers = http_server(chat_router=router)

    with httpx.Client(base_url=base_url, timeout=10.0, trust_env=False) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "algae/deepseek/deepseek-chat",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "algae/deepseek/deepseek-chat"
    assert payload["choices"][0]["message"] == {
        "role": "assistant",
        "content": "e2e provider answer",
    }


def test_chat_completions_stream_round_trip_over_real_http(http_server, tmp_path: Path) -> None:
    router = _provider_router_with_adapter(
        tmp_path,
        provider="deepseek",
        adapter=StaticAdapter("streamed e2e answer"),
    )
    base_url, headers = http_server(chat_router=router)

    with httpx.Client(base_url=base_url, timeout=10.0, trust_env=False) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "algae/deepseek/deepseek-chat",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        ) as response:
            lines = [line for line in response.iter_lines() if line]

    assert response.status_code == 200
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
    assert len(chunks) >= 3
    assert chunks[0]["object"] == "chat.completion.chunk"
    assert chunks[-1]["object"] == "chat.completion.chunk"
    assert chunks[0]["id"] == chunks[-1]["id"]
    assert chunks[0]["model"] == chunks[-1]["model"] == "algae/deepseek/deepseek-chat"
    assert chunks[0]["choices"][0]["delta"] == {
        "role": "assistant",
    }
    assert "".join(
        chunk["choices"][0]["delta"].get("content", "")
        for chunk in chunks[1:-1]
    ) == "streamed e2e answer"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_chat_completions_returns_error_envelope_over_real_http(http_server, tmp_path: Path) -> None:
    router = _provider_router_with_adapter(
        tmp_path,
        provider="deepseek",
        adapter=StaticAdapter("unused"),
    )
    base_url, headers = http_server(chat_router=router)

    with httpx.Client(base_url=base_url, timeout=10.0, trust_env=False) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "algae/nonexist/test-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "Unsupported model: algae/nonexist/test-model",
            "type": "invalid_request_error",
        }
    }


def test_chat_completions_maps_upstream_http_error_over_real_http(http_server, tmp_path: Path) -> None:
    router = _provider_router_with_adapter(
        tmp_path,
        provider="deepseek",
        adapter=UpstreamErrorAdapter(),
    )
    base_url, headers = http_server(chat_router=router)

    with httpx.Client(base_url=base_url, timeout=10.0, trust_env=False) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "algae/deepseek/deepseek-chat",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 502
    assert response.json() == {
        "error": {
            "message": "upstream rate limited",
            "type": "api_error",
        }
    }


def test_chat_completions_validates_missing_model_over_real_http(http_server, tmp_path: Path) -> None:
    router = _provider_router_with_adapter(
        tmp_path,
        provider="deepseek",
        adapter=StaticAdapter("unused"),
    )
    base_url, headers = http_server(chat_router=router)

    with httpx.Client(base_url=base_url, timeout=10.0, trust_env=False) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=headers,
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "Model is required.",
            "type": "invalid_request_error",
        }
    }


def test_responses_round_trip_over_real_http(http_server, tmp_path: Path) -> None:
    router = _provider_router_with_adapter(
        tmp_path,
        provider="deepseek",
        adapter=StaticAdapter("responses e2e answer"),
    )
    base_url, headers = http_server(responses_router=router)

    with httpx.Client(base_url=base_url, timeout=10.0, trust_env=False) as client:
        response = client.post(
            "/v1/responses",
            headers=headers,
            json={
                "model": "algae/deepseek/deepseek-chat",
                "input": "hello",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "response"
    assert payload["status"] == "completed"
    assert payload["output"][0]["content"] == [{"type": "output_text", "text": "responses e2e answer"}]


def test_responses_stream_round_trip_over_real_http(http_server, tmp_path: Path) -> None:
    router = _provider_router_with_adapter(
        tmp_path,
        provider="deepseek",
        adapter=StaticAdapter("responses stream answer"),
    )
    base_url, headers = http_server(responses_router=router)

    with httpx.Client(base_url=base_url, timeout=10.0, trust_env=False) as client:
        with client.stream(
            "POST",
            "/v1/responses",
            headers=headers,
            json={
                "model": "algae/deepseek/deepseek-chat",
                "input": "hello",
                "stream": True,
            },
        ) as response:
            lines = [line for line in response.iter_lines() if line]

    assert response.status_code == 200
    event_names = [line.removeprefix("event: ") for line in lines if line.startswith("event: ")]
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
    payloads = [json.loads(line.removeprefix("data: ")) for line in lines if line.startswith("data: ")]
    created_payload = payloads[0]["response"]
    output_added_payload = payloads[2]["item"]
    delta_payloads = payloads[4:-4]
    item_done_payload = payloads[-2]["item"]
    completed_payload = payloads[-1]["response"]

    assert created_payload["status"] == "in_progress"
    assert output_added_payload["status"] == "in_progress"
    assert "".join(payload["delta"] for payload in delta_payloads) == "responses stream answer"
    assert item_done_payload["status"] == "completed"
    assert item_done_payload["content"] == [{"type": "output_text", "text": "responses stream answer"}]
    assert completed_payload["status"] == "completed"
    assert completed_payload["output"] == [item_done_payload]


def test_chat_completions_stream_flushes_incrementally_and_hides_think_tags_over_real_http(http_server) -> None:
    base_url, headers = http_server(chat_router=DelayedThinkingStreamRouter())

    content_events: list[tuple[float, str]] = []
    with httpx.Client(base_url=base_url, timeout=10.0, trust_env=False) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "algae/deepseek/deepseek-chat",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        ) as response:
            started = time.perf_counter()
            for line in response.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                payload = json.loads(line.removeprefix("data: "))
                delta = payload["choices"][0]["delta"]
                content = delta.get("content")
                if isinstance(content, str) and content:
                    content_events.append((time.perf_counter() - started, content))

    assert response.status_code == 200
    assert "".join(chunk for _, chunk in content_events) == "最终答案"
    assert len(content_events) >= 2
    assert content_events[-1][0] - content_events[0][0] >= 0.05


def test_chat_completions_stream_rechunks_coarse_provider_pieces_over_real_http(http_server) -> None:
    base_url, headers = http_server(chat_router=DelayedCoarseChatStreamRouter())

    content_events: list[str] = []
    with httpx.Client(base_url=base_url, timeout=10.0, trust_env=False) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "algae/deepseek/deepseek-chat",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        ) as response:
            for line in response.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                payload = json.loads(line.removeprefix("data: "))
                delta = payload["choices"][0]["delta"]
                content = delta.get("content")
                if isinstance(content, str) and content:
                    content_events.append(content)

    assert response.status_code == 200
    assert "".join(content_events) == (
        "这是一段非常长的中文内容，没有空格，也不应该在 chat/completions 里被一次性整块吐出给客户端。"
    )
    assert len(content_events) >= 2


def test_chat_completions_streams_openai_style_tool_call_deltas_over_real_http(http_server) -> None:
    base_url, headers = http_server(chat_router=ThinkingToolCallHttpRouter())

    with httpx.Client(base_url=base_url, timeout=10.0, trust_env=False) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "algae/deepseek/deepseek-chat",
                "messages": [{"role": "user", "content": "hello"}],
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
            lines = [line for line in response.iter_lines() if line]

    assert response.status_code == 200
    assert lines[-1] == "data: [DONE]"
    events = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
    content_deltas = [
        event["choices"][0]["delta"].get("content", "")
        for event in events
        if "content" in event["choices"][0]["delta"]
    ]
    assert "".join(content_deltas) == ""
    tool_call_events = [
        event
        for event in events
        if "tool_calls" in event["choices"][0]["delta"]
    ]
    assert tool_call_events[0]["choices"][0]["delta"]["tool_calls"] == [
        {
            "index": 0,
            "id": "call_weather_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": ""},
        }
    ]
    assert tool_call_events[1]["choices"][0]["delta"]["tool_calls"] == [
        {
            "index": 0,
            "function": {"arguments": '{"location":"Tokyo"}'},
        }
    ]
    assert _collect_chat_stream_tool_calls(events) == [
        {
            "id": "call_weather_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location":"Tokyo"}',
            },
        }
    ]
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_responses_stream_flushes_reasoning_incrementally_over_real_http(http_server) -> None:
    base_url, headers = http_server(responses_router=DelayedThinkingStreamRouter())

    reasoning_events: list[tuple[float, str]] = []
    output_text_events: list[tuple[float, str]] = []
    with httpx.Client(base_url=base_url, timeout=10.0, trust_env=False) as client:
        with client.stream(
            "POST",
            "/v1/responses",
            headers=headers,
            json={
                "model": "algae/deepseek/deepseek-chat",
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
                now = time.perf_counter() - started
                if event_name == "response.reasoning_text.delta":
                    reasoning_events.append((now, payload["delta"]))
                elif event_name == "response.output_text.delta":
                    output_text_events.append((now, payload["delta"]))

    assert response.status_code == 200
    assert "".join(chunk for _, chunk in reasoning_events) == "先想一想"
    assert "".join(chunk for _, chunk in output_text_events) == "最终答案"
    assert reasoning_events
    assert output_text_events
    assert output_text_events[-1][0] - reasoning_events[0][0] >= 0.07


def test_responses_streams_reasoning_and_function_call_arguments_over_real_http(http_server) -> None:
    base_url, headers = http_server(responses_router=ThinkingToolCallHttpRouter())

    with httpx.Client(base_url=base_url, timeout=10.0, trust_env=False) as client:
        with client.stream(
            "POST",
            "/v1/responses",
            headers=headers,
            json={
                "model": "algae/deepseek/deepseek-chat",
                "input": "hello",
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
    assert event_names == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.reasoning_text.delta",
        "response.reasoning_text.done",
        "response.output_item.done",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    reasoning_added = events[2][1]["item"]
    function_added = events[6][1]["item"]
    assert reasoning_added == {
        "type": "reasoning",
        "id": reasoning_added["id"],
        "status": "in_progress",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": ""}],
    }
    assert events[3][1]["delta"] == "先想一想"
    assert function_added == {
        "type": "function_call",
        "id": function_added["id"],
        "call_id": "call_weather_1",
        "name": "get_weather",
        "arguments": "",
        "status": "in_progress",
    }
    assert events[7][1]["delta"] == '{"location":"Tokyo"}'
    assert events[8][1] == {
        "type": "response.function_call_arguments.done",
        "item_id": function_added["id"],
        "output_index": 1,
        "arguments": '{"location":"Tokyo"}',
        "name": "get_weather",
    }


def test_responses_maps_upstream_http_error_over_real_http(http_server, tmp_path: Path) -> None:
    router = _provider_router_with_adapter(
        tmp_path,
        provider="deepseek",
        adapter=UpstreamErrorAdapter(),
    )
    base_url, headers = http_server(responses_router=router)

    with httpx.Client(base_url=base_url, timeout=10.0, trust_env=False) as client:
        response = client.post(
            "/v1/responses",
            headers=headers,
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


def test_responses_validates_missing_model_over_real_http(http_server, tmp_path: Path) -> None:
    router = _provider_router_with_adapter(
        tmp_path,
        provider="deepseek",
        adapter=StaticAdapter("unused"),
    )
    base_url, headers = http_server(responses_router=router)

    with httpx.Client(base_url=base_url, timeout=10.0, trust_env=False) as client:
        response = client.post(
            "/v1/responses",
            headers=headers,
            json={"input": "hello"},
        )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "Model is required.",
            "type": "invalid_request_error",
        }
    }


def _provider_router_with_adapter(
    tmp_path: Path,
    *,
    provider: str,
    adapter: ProviderAdapter,
) -> ProviderRouter:
    providers_dir = tmp_path / "providers"
    save_provider_credentials(
        providers_dir,
        ProviderCredentialRecord(
            provider=provider,
            kind="web_session",
            cookie="session=value",
            headers={"authorization": "Bearer token"},
            user_agent="ua",
            status="valid",
        ),
    )
    return ProviderRouter(adapters={provider: adapter}, providers_dir=providers_dir)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _wait_for_server(url: str) -> None:
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=1.0, trust_env=False)
        except httpx.HTTPError:
            time.sleep(0.05)
            continue
        if response.status_code == 200:
            return
        time.sleep(0.05)
    raise RuntimeError(f"Timed out waiting for server: {url}")


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


def _collect_chat_stream_tool_calls(events: list[dict[str, object]]) -> list[dict[str, object]]:
    collected: dict[int, dict[str, object]] = {}
    for event in events:
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

from opentoken.providers.prompts import stringify_message_content


class AttachmentEchoAdapter(ProviderAdapter):
    def chat(
        self,
        request: NormalizedChatRequest,
        credentials: ProviderCredentialRecord | None = None,
    ) -> ChatResponse:
        assert credentials is not None
        rendered = "\n\n".join(
            stringify_message_content(message.get("content", ""))
            for message in request.messages
        )
        return ChatResponse(model=request.model, content=rendered)



def test_files_and_file_id_round_trip_over_real_http(http_server, tmp_path: Path) -> None:
    router = _provider_router_with_adapter(
        tmp_path,
        provider="deepseek",
        adapter=AttachmentEchoAdapter(),
    )
    base_url, headers = http_server(responses_router=router)

    with httpx.Client(base_url=base_url, timeout=10.0, trust_env=False) as client:
        upload_response = client.post(
            "/v1/files",
            headers=headers,
            files={"file": ("note.txt", b"hello upload chain", "text/plain")},
            data={"purpose": "assistants"},
        )
        file_id = upload_response.json()["id"]
        response = client.post(
            "/v1/responses",
            headers=headers,
            json={
                # Use the prefixed form so resolution doesn't depend on the
                # live-discovery catalog (which is empty in tests).
                "model": "algae/deepseek/deepseek-chat",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "summarize the attachment"},
                            {"type": "input_file", "file_id": file_id},
                        ],
                    }
                ],
            },
        )

    assert upload_response.status_code == 200
    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "algae/deepseek/deepseek-chat"
    output_text = payload["output"][0]["content"][0]["text"]
    assert "summarize the attachment" in output_text
    assert "hello upload chain" in output_text



def test_uploads_and_embeddings_over_real_http(http_server, tmp_path: Path) -> None:
    base_url, headers = http_server()

    with httpx.Client(base_url=base_url, timeout=10.0, trust_env=False) as client:
        create_upload = client.post(
            "/v1/uploads",
            headers=headers,
            json={
                "filename": "joined.txt",
                "bytes": 11,
                "mime_type": "text/plain",
                "purpose": "assistants",
            },
        )
        upload_id = create_upload.json()["id"]
        add_part = client.post(
            f"/v1/uploads/{upload_id}/parts",
            headers=headers,
            files={"data": ("part-1", b"hello world", "application/octet-stream")},
        )
        complete = client.post(
            f"/v1/uploads/{upload_id}/complete",
            headers=headers,
            json={},
        )
        embeddings = client.post(
            "/v1/embeddings",
            headers=headers,
            json={"model": "text-embedding-3-small", "input": ["alpha", "beta"]},
        )

    assert create_upload.status_code == 200
    assert add_part.status_code == 200
    assert complete.status_code == 200
    assert complete.json()["filename"] == "joined.txt"
    assert complete.json()["bytes"] == 11

    # /v1/embeddings now returns 501 — opentoken doesn't host a real embedding
    # backend. See tests/api/test_embeddings.py for the canonical contract test.
    assert embeddings.status_code == 501
    assert embeddings.json()["error"]["type"] == "not_implemented"
