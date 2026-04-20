import base64
import json
from hashlib import sha256

import httpx
import pytest

from openclaw_algae.gateway.normalized import NormalizedChatRequest
from openclaw_algae.models.provider_credentials import ProviderCredentialRecord
from openclaw_algae.providers.base import ProviderRateLimitError
from openclaw_algae.providers import deepseek as deepseek_provider
from openclaw_algae.providers.deepseek import DeepSeekWebAdapter, DeepSeekWebClient


def test_deepseek_web_client_builds_reference_headers() -> None:
    credentials = ProviderCredentialRecord(
        provider="deepseek",
        kind="web_session",
        cookie="session=value",
        headers={"authorization": "Bearer token"},
        user_agent="ua",
        status="valid",
    )

    headers = DeepSeekWebClient(credentials).build_headers()

    assert headers["Cookie"] == "session=value"
    assert headers["User-Agent"] == "ua"
    assert headers["Authorization"] == "Bearer token"
    assert headers["Referer"] == "https://chat.deepseek.com/"
    assert headers["Origin"] == "https://chat.deepseek.com"


def test_deepseek_adapter_translates_request_for_client() -> None:
    credentials = ProviderCredentialRecord(
        provider="deepseek",
        kind="web_session",
        cookie="session=value",
        headers={"authorization": "Bearer token"},
        user_agent="ua",
        status="valid",
    )
    calls: dict[str, object] = {}

    class FakeClient:
        def chat_completion(self, *, message: str, model: str) -> str:
            calls["message"] = message
            calls["model"] = model
            return "deepseek answer"

    adapter = DeepSeekWebAdapter(client_factory=lambda _: FakeClient())

    response = adapter.chat(
        NormalizedChatRequest(
            model="algae/deepseek/deepseek-chat",
            messages=[
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "next step"},
            ],
        ),
        credentials,
    )

    assert calls["message"] == "System: be concise\n\nUser: hello\n\nAssistant: hi\n\nUser: next step"
    assert calls["model"] == "deepseek-chat"
    assert response.content == "deepseek answer"


def test_deepseek_adapter_keeps_reasoning_markup_for_non_tool_requests() -> None:
    credentials = ProviderCredentialRecord(
        provider="deepseek",
        kind="web_session",
        cookie="session=value",
        headers={"authorization": "Bearer token"},
        user_agent="ua",
        status="valid",
    )

    class FakeClient:
        def chat_completion(self, *, message: str, model: str) -> str:
            assert model == "deepseek-reasoner"
            assert "User: 13*17=?" in message
            return "<think>先想一想</think>221"

    adapter = DeepSeekWebAdapter(client_factory=lambda _: FakeClient())

    response = adapter.chat(
        NormalizedChatRequest(
            model="algae/deepseek/deepseek-reasoner",
            messages=[{"role": "user", "content": "13*17=?"}],
        ),
        credentials,
    )

    assert response.content == "<think>先想一想</think>221"
    assert response.tool_calls == []
    assert response.finish_reason == "stop"


def test_deepseek_web_client_validates_credentials_via_users_current() -> None:
    credentials = ProviderCredentialRecord(
        provider="deepseek",
        kind="web_session",
        cookie="session=value",
        headers={"authorization": "Bearer old-token"},
        user_agent="ua",
        status="valid",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v0/users/current"
        assert request.headers["Authorization"] == "Bearer old-token"
        return httpx.Response(
            200,
            json={"data": {"biz_data": {"token": "fresh-token"}}},
        )

    client = DeepSeekWebClient(
        credentials,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert client.validate_credentials() == "fresh-token"


def test_deepseek_web_client_retries_after_refresh_when_session_creation_is_unauthorized() -> None:
    credentials = ProviderCredentialRecord(
        provider="deepseek",
        kind="web_session",
        cookie="session=value",
        headers={"authorization": "Bearer stale-token"},
        user_agent="ua",
        status="valid",
    )
    seen_auth_headers: list[str] = []
    attempts = {"create": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v0/users/current":
            return httpx.Response(
                200,
                json={"data": {"biz_data": {"token": "fresh-token"}}},
            )
        if request.url.path == "/api/v0/chat_session/create":
            seen_auth_headers.append(request.headers.get("Authorization", ""))
            attempts["create"] += 1
            if attempts["create"] == 1:
                return httpx.Response(401, text="expired", request=request)
            return httpx.Response(
                200,
                json={"data": {"biz_data": {"id": "session-123"}}},
            )
        raise AssertionError(f"Unexpected path {request.url.path}")

    client = DeepSeekWebClient(
        credentials,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    session_id = client.create_chat_session()

    assert session_id == "session-123"
    assert seen_auth_headers == ["Bearer stale-token", "Bearer fresh-token"]


def test_deepseek_web_client_solves_sha256_pow() -> None:
    credentials = ProviderCredentialRecord(
        provider="deepseek",
        kind="web_session",
        cookie="session=value",
        headers={},
        user_agent="ua",
        status="valid",
    )
    client = DeepSeekWebClient(credentials)
    challenge = {
        "algorithm": "sha256",
        "challenge": "abc",
        "difficulty": 8,
        "salt": "salt",
        "signature": "sig",
    }

    answer = client.solve_pow(challenge)
    digest = sha256(f"saltabc{answer}".encode("utf-8")).hexdigest()

    assert isinstance(answer, int)
    assert digest.startswith("00")


def test_deepseek_web_client_solves_deepseek_hash_v1_pow(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="deepseek",
        kind="web_session",
        cookie="session=value",
        headers={},
        user_agent="ua",
        status="valid",
    )
    client = DeepSeekWebClient(credentials)
    seen: dict[str, object] = {}

    def fake_solver(*, challenge: str, prefix: str, difficulty: float) -> float | None:
        seen["challenge"] = challenge
        seen["prefix"] = prefix
        seen["difficulty"] = difficulty
        return 42.0

    monkeypatch.setattr(
        deepseek_provider,
        "_solve_deepseek_hash_v1_wasm",
        fake_solver,
        raising=False,
    )

    answer = client.solve_pow(
        {
            "algorithm": "DeepSeekHashV1",
            "challenge": "abc123",
            "difficulty": 144000,
            "salt": "salt",
            "signature": "sig",
            "expire_at": 1740000000,
        }
    )

    assert answer == 42
    assert seen == {
        "challenge": "abc123",
        "prefix": "salt_1740000000_",
        "difficulty": 144000.0,
    }


def test_deepseek_web_client_raises_when_deepseek_hash_v1_pow_has_no_solution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials = ProviderCredentialRecord(
        provider="deepseek",
        kind="web_session",
        cookie="session=value",
        headers={},
        user_agent="ua",
        status="valid",
    )
    client = DeepSeekWebClient(credentials)

    monkeypatch.setattr(
        deepseek_provider,
        "_solve_deepseek_hash_v1_wasm",
        lambda **_: None,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="DeepSeekHashV1 failed to find solution"):
        client.solve_pow(
            {
                "algorithm": "DeepSeekHashV1",
                "challenge": "abc123",
                "difficulty": 144000,
                "salt": "salt",
                "signature": "sig",
                "expire_at": 1740000000,
            }
        )


def test_deepseek_web_client_chat_completion_sends_pow_header_and_parses_sse() -> None:
    credentials = ProviderCredentialRecord(
        provider="deepseek",
        kind="web_session",
        cookie="session=value",
        headers={"authorization": "Bearer token"},
        user_agent="ua",
        status="valid",
    )
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v0/chat_session/create":
            return httpx.Response(
                200,
                json={
                    "data": {
                        "biz_data": {
                            "id": "session-123",
                        }
                    }
                },
            )
        if request.url.path == "/api/v0/chat/create_pow_challenge":
            return httpx.Response(
                200,
                json={
                    "data": {
                        "biz_data": {
                            "challenge": {
                                "algorithm": "sha256",
                                "challenge": "abc",
                                "difficulty": 1,
                                "salt": "salt",
                                "signature": "sig",
                            }
                        }
                    }
                },
            )

        assert request.url.path == "/api/v0/chat/completion"
        seen["headers"] = dict(request.headers)
        seen["payload"] = json.loads(request.content.decode("utf-8"))
        decoded = json.loads(
            base64.b64decode(request.headers["x-ds-pow-response"]).decode("utf-8")
        )
        seen["pow"] = decoded
        return httpx.Response(
            200,
            text='data: {"v":"hello"}\n\ndata: {"v":" world"}\n\ndata: [DONE]\n',
            headers={"content-type": "text/event-stream"},
        )

    client = DeepSeekWebClient(
        credentials,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    content = client.chat_completion(
        message="hello",
        model="deepseek-chat",
    )

    assert seen["payload"]["parent_message_id"] is None
    assert seen["payload"]["prompt"] == "hello"
    assert seen["payload"]["thinking_enabled"] is False
    assert seen["pow"]["target_path"] == "/api/v0/chat/completion"
    assert isinstance(seen["pow"]["answer"], int)


def test_deepseek_web_client_uses_trust_env_false_by_default() -> None:
    credentials = ProviderCredentialRecord(
        provider="deepseek",
        kind="web_session",
        cookie="session=value",
        headers={},
        user_agent="ua",
        status="valid",
    )

    client = DeepSeekWebClient(credentials)

    assert client._client.trust_env is False


def test_deepseek_web_client_chat_completion_normalizes_hash_v1_pow_answer_to_int(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials = ProviderCredentialRecord(
        provider="deepseek",
        kind="web_session",
        cookie="session=value",
        headers={"authorization": "Bearer token"},
        user_agent="ua",
        status="valid",
    )
    seen: dict[str, object] = {}

    def fake_solver(*, challenge: str, prefix: str, difficulty: float) -> float | None:
        return 74670.0

    monkeypatch.setattr(
        deepseek_provider,
        "_solve_deepseek_hash_v1_wasm",
        fake_solver,
        raising=False,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v0/chat_session/create":
            return httpx.Response(
                200,
                json={
                    "data": {
                        "biz_data": {
                            "id": "session-123",
                        }
                    }
                },
            )
        if request.url.path == "/api/v0/chat/create_pow_challenge":
            return httpx.Response(
                200,
                json={
                    "data": {
                        "biz_data": {
                            "challenge": {
                                "algorithm": "DeepSeekHashV1",
                                "challenge": "abc",
                                "difficulty": 144000,
                                "salt": "salt",
                                "signature": "sig",
                                "expire_at": 1740000000,
                            }
                        }
                    }
                },
            )

        seen["pow"] = json.loads(base64.b64decode(request.headers["x-ds-pow-response"]).decode("utf-8"))
        return httpx.Response(
            200,
            text='data: {"v":"algae-ok"}\n\ndata: [DONE]\n',
            headers={"content-type": "text/event-stream"},
        )

    client = DeepSeekWebClient(
        credentials,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    content = client.chat_completion(
        message="hello",
        model="deepseek-chat",
    )

    assert seen["pow"]["answer"] == 74670
    assert isinstance(seen["pow"]["answer"], int)
    assert content == "algae-ok"


def test_deepseek_web_client_chat_completion_surfaces_json_error_body() -> None:
    credentials = ProviderCredentialRecord(
        provider="deepseek",
        kind="web_session",
        cookie="session=value",
        headers={"authorization": "Bearer token"},
        user_agent="ua",
        status="valid",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v0/chat_session/create":
            return httpx.Response(
                200,
                json={
                    "data": {
                        "biz_data": {
                            "id": "session-123",
                        }
                    }
                },
            )
        if request.url.path == "/api/v0/chat/create_pow_challenge":
            return httpx.Response(
                200,
                json={
                    "data": {
                        "biz_data": {
                            "challenge": {
                                "algorithm": "sha256",
                                "challenge": "abc",
                                "difficulty": 1,
                                "salt": "salt",
                                "signature": "sig",
                            }
                        }
                    }
                },
            )
        return httpx.Response(
            200,
            json={"code": 40301, "msg": "INVALID_POW_RESPONSE", "data": None},
            headers={"content-type": "application/json"},
        )

    client = DeepSeekWebClient(
        credentials,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(RuntimeError, match="INVALID_POW_RESPONSE"):
        client.chat_completion(
            message="hello",
            model="deepseek-chat",
        )


def test_deepseek_web_client_chat_completion_surfaces_sse_rate_limit_hint() -> None:
    credentials = ProviderCredentialRecord(
        provider="deepseek",
        kind="web_session",
        cookie="session=value",
        headers={"authorization": "Bearer token"},
        user_agent="ua",
        status="valid",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v0/chat_session/create":
            return httpx.Response(200, json={"data": {"biz_data": {"id": "session-123"}}})
        if request.url.path == "/api/v0/chat/create_pow_challenge":
            return httpx.Response(
                200,
                json={
                    "data": {
                        "biz_data": {
                            "challenge": {
                                "algorithm": "sha256",
                                "challenge": "abc",
                                "difficulty": 1,
                                "salt": "salt",
                                "signature": "sig",
                            }
                        }
                    }
                },
            )
        return httpx.Response(
            200,
            text="\n".join(
                [
                    "event: ready",
                    'data: {"request_message_id":1,"response_message_id":2}',
                    "",
                    "event: hint",
                    'data: {"type":"error","content":"消息发送过于频繁，请稍后重试","clear_response":true,"finish_reason":"rate_limit_reached"}',
                    "",
                    "event: close",
                    'data: {"click_behavior":"retry","auto_resume":false}',
                    "",
                ]
            ),
            headers={"content-type": "text/event-stream"},
        )

    client = DeepSeekWebClient(
        credentials,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ProviderRateLimitError, match="消息发送过于频繁"):
        client.chat_completion(
            message="hello",
            model="deepseek-chat",
        )


def test_parse_deepseek_sse_text_handles_fragment_snapshot_then_append() -> None:
    payload = "\n".join(
        [
            'event: ready',
            'data: {"request_message_id":1,"response_message_id":2}',
            "",
            'data: {"v":{"response":{"fragments":[{"content":"algae"}]}}}',
            "",
            'data: {"p":"response/fragments/-1/content","o":"APPEND","v":"-"}',
            "",
            'data: {"v":"ok"}',
            "",
            'data: [DONE]',
        ]
    )

    assert deepseek_provider._parse_deepseek_sse_text(payload) == "algae-ok"


def test_parse_deepseek_sse_text_preserves_think_fragments_with_tags() -> None:
    payload = "\n".join(
        [
            'data: {"v":{"response":{"fragments":[{"type":"THINK","content":"We"}]}}}',
            "",
            'data: {"v":" should"}',
            "",
            'data: {"v":" think"}',
            "",
            'data: {"p":"response/fragments","o":"APPEND","v":[{"type":"RESPONSE","content":"reason"}]}',
            "",
            'data: {"p":"response/fragments/-1/content","v":"er"}',
            "",
            'data: {"v":"-"}',
            "",
            'data: {"v":"ok"}',
            "",
            'data: [DONE]',
        ]
    )

    assert deepseek_provider._parse_deepseek_sse_text(payload) == "<think>We should think</think>reasoner-ok"


def test_deepseek_web_client_streams_incremental_text_with_think_tags() -> None:
    credentials = ProviderCredentialRecord(
        provider="deepseek",
        kind="web_session",
        cookie="session=value",
        headers={"authorization": "Bearer token"},
        user_agent="ua",
        status="valid",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat_session/create"):
            return httpx.Response(
                200,
                json={"data": {"biz_data": {"id": "session-1"}}},
            )
        if request.url.path.endswith("/create_pow_challenge"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "biz_data": {
                            "challenge": {
                                "algorithm": "sha256",
                                "challenge": "abc",
                                "difficulty": 1,
                                "salt": "salt",
                                "signature": "sig",
                            }
                        }
                    }
                },
            )
        if request.url.path.endswith("/chat/completion"):
            return httpx.Response(
                200,
                text="\n".join(
                    [
                        'data: {"v":{"response":{"fragments":[{"type":"THINK","content":"We"}]}}}',
                        "",
                        'data: {"v":" think"}',
                        "",
                        'data: {"p":"response/fragments","o":"APPEND","v":[{"type":"RESPONSE","content":"ok"}]}',
                        "",
                        'data: {"v":"!"}',
                        "",
                        "data: [DONE]",
                    ]
                ),
                headers={"content-type": "text/event-stream"},
            )
        raise AssertionError(f"Unexpected path: {request.url.path}")

    client = DeepSeekWebClient(
        credentials,
        client=httpx.Client(transport=httpx.MockTransport(handler), trust_env=False),
    )
    client.solve_pow = lambda challenge: 1  # type: ignore[method-assign]

    assert list(client.iter_chat_completion_text(message="hello", model="deepseek-reasoner")) == [
        "<think>We",
        " think",
        "</think>ok",
        "!",
    ]


def test_deepseek_web_client_stream_raises_provider_rate_limit_on_hint_error() -> None:
    credentials = ProviderCredentialRecord(
        provider="deepseek",
        kind="web_session",
        cookie="session=value",
        headers={"authorization": "Bearer token"},
        user_agent="ua",
        status="valid",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat_session/create"):
            return httpx.Response(200, json={"data": {"biz_data": {"id": "session-1"}}})
        if request.url.path.endswith("/create_pow_challenge"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "biz_data": {
                            "challenge": {
                                "algorithm": "sha256",
                                "challenge": "abc",
                                "difficulty": 1,
                                "salt": "salt",
                                "signature": "sig",
                            }
                        }
                    }
                },
            )
        if request.url.path.endswith("/chat/completion"):
            return httpx.Response(
                200,
                text="\n".join(
                    [
                        "event: ready",
                        'data: {"request_message_id":1,"response_message_id":2}',
                        "",
                        "event: hint",
                        'data: {"type":"error","content":"消息发送过于频繁，请稍后重试","clear_response":true,"finish_reason":"rate_limit_reached"}',
                        "",
                    ]
                ),
                headers={"content-type": "text/event-stream"},
            )
        raise AssertionError(f"Unexpected path: {request.url.path}")

    client = DeepSeekWebClient(
        credentials,
        client=httpx.Client(transport=httpx.MockTransport(handler), trust_env=False),
    )
    client.solve_pow = lambda challenge: 1  # type: ignore[method-assign]

    with pytest.raises(ProviderRateLimitError, match="消息发送过于频繁"):
        list(client.iter_chat_completion_text(message="hello", model="deepseek-chat"))


def test_deepseek_web_client_stream_maps_generating_message_to_rate_limit_error() -> None:
    credentials = ProviderCredentialRecord(
        provider="deepseek",
        kind="web_session",
        cookie="session=value",
        headers={"authorization": "Bearer token"},
        user_agent="ua",
        status="valid",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat_session/create"):
            return httpx.Response(200, json={"data": {"biz_data": {"id": "session-1"}}})
        if request.url.path.endswith("/create_pow_challenge"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "biz_data": {
                            "challenge": {
                                "algorithm": "sha256",
                                "challenge": "abc",
                                "difficulty": 1,
                                "salt": "salt",
                                "signature": "sig",
                            }
                        }
                    }
                },
            )
        if request.url.path.endswith("/chat/completion"):
            return httpx.Response(
                200,
                text="\n".join(
                    [
                        "event: error",
                        'data: {"type":"error","content":"有消息正在生成，请稍后再试"}',
                        "",
                    ]
                ),
                headers={"content-type": "text/event-stream"},
            )
        raise AssertionError(f"Unexpected path: {request.url.path}")

    client = DeepSeekWebClient(
        credentials,
        client=httpx.Client(transport=httpx.MockTransport(handler), trust_env=False),
    )
    client.solve_pow = lambda challenge: 1  # type: ignore[method-assign]

    with pytest.raises(ProviderRateLimitError, match="有消息正在生成"):
        list(client.iter_chat_completion_text(message="hello", model="deepseek-chat"))


def test_deepseek_web_client_json_error_maps_generating_message_to_rate_limit_error() -> None:
    credentials = ProviderCredentialRecord(
        provider="deepseek",
        kind="web_session",
        cookie="session=value",
        headers={"authorization": "Bearer token"},
        user_agent="ua",
        status="valid",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat_session/create"):
            return httpx.Response(200, json={"data": {"biz_data": {"id": "session-1"}}})
        if request.url.path.endswith("/create_pow_challenge"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "biz_data": {
                            "challenge": {
                                "algorithm": "sha256",
                                "challenge": "abc",
                                "difficulty": 1,
                                "salt": "salt",
                                "signature": "sig",
                            }
                        }
                    }
                },
            )
        if request.url.path.endswith("/chat/completion"):
            return httpx.Response(
                200,
                json={"code": 1234, "message": "有消息正在生成，请稍后再试"},
                headers={"content-type": "application/json"},
            )
        raise AssertionError(f"Unexpected path: {request.url.path}")

    client = DeepSeekWebClient(
        credentials,
        client=httpx.Client(transport=httpx.MockTransport(handler), trust_env=False),
    )
    client.solve_pow = lambda challenge: 1  # type: ignore[method-assign]

    with pytest.raises(ProviderRateLimitError, match="有消息正在生成"):
        client.chat_completion(message="hello", model="deepseek-chat")


def test_deepseek_web_client_creates_fresh_session_for_each_stream_request() -> None:
    credentials = ProviderCredentialRecord(
        provider="deepseek",
        kind="web_session",
        cookie="session=value",
        headers={"authorization": "Bearer token"},
        user_agent="ua",
        status="valid",
    )
    seen_session_ids: list[str] = []
    created_sessions = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat_session/create"):
            created_sessions["count"] += 1
            return httpx.Response(
                200,
                json={"data": {"biz_data": {"id": f"session-{created_sessions['count']}"}}},
            )
        if request.url.path.endswith("/create_pow_challenge"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "biz_data": {
                            "challenge": {
                                "algorithm": "sha256",
                                "challenge": "abc",
                                "difficulty": 1,
                                "salt": "salt",
                                "signature": "sig",
                            }
                        }
                    }
                },
            )
        if request.url.path.endswith("/chat/completion"):
            payload = json.loads(request.content.decode("utf-8"))
            seen_session_ids.append(str(payload["chat_session_id"]))
            return httpx.Response(
                200,
                text='data: {"v":"hello"}\n\ndata: [DONE]\n',
                headers={"content-type": "text/event-stream"},
            )
        raise AssertionError(f"Unexpected path: {request.url.path}")

    client = DeepSeekWebClient(
        credentials,
        client=httpx.Client(transport=httpx.MockTransport(handler), trust_env=False),
    )
    client.solve_pow = lambda challenge: 1  # type: ignore[method-assign]

    assert list(client.iter_chat_completion_text(message="hello", model="deepseek-chat")) == ["hello"]
    assert list(client.iter_chat_completion_text(message="hello again", model="deepseek-chat")) == ["hello"]
    assert seen_session_ids == ["session-1", "session-2"]


def test_deepseek_adapter_includes_attachment_descriptions_and_text_content() -> None:
    credentials = ProviderCredentialRecord(
        provider="deepseek",
        kind="web_session",
        cookie="session=value",
        headers={"authorization": "Bearer token"},
        user_agent="ua",
        status="valid",
    )
    calls: dict[str, object] = {}

    class FakeClient:
        def chat_completion(self, *, message: str, model: str) -> str:
            calls["message"] = message
            calls["model"] = model
            return "deepseek answer"

    adapter = DeepSeekWebAdapter(client_factory=lambda _: FakeClient())

    response = adapter.chat(
        NormalizedChatRequest(
            model="deepseek-chat",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Please quote the attachment."},
                        {
                            "type": "input_file",
                            "filename": "note.txt",
                            "file_data": "data:text/plain;base64,bGl2ZSBhdHRhY2htZW50IGJvZHk=",
                        },
                    ],
                }
            ],
        ),
        credentials,
    )

    assert response.content == "deepseek answer"
    assert "Please quote the attachment." in calls["message"]
    assert "[Attached file: note.txt | data URI (text/plain)]" in calls["message"]
    assert "live attachment body" in calls["message"]


def test_deepseek_adapter_streams_using_client_when_available() -> None:
    credentials = ProviderCredentialRecord(
        provider="deepseek",
        kind="web_session",
        cookie="session=value",
        headers={"authorization": "Bearer token"},
        user_agent="ua",
        status="valid",
    )

    class FakeClient:
        def iter_chat_completion_text(self, *, message: str, model: str):
            assert model == "deepseek-chat"
            assert "User: hello" in message
            yield "he"
            yield "llo"

    adapter = DeepSeekWebAdapter(client_factory=lambda _: FakeClient())

    chunks = list(
        adapter.stream_chat(
            NormalizedChatRequest(
                model="algae/deepseek/deepseek-chat",
                messages=[{"role": "user", "content": "hello"}],
            ),
            credentials,
        )
        or []
    )

    assert chunks == ["he", "llo"]
