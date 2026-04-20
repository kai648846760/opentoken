import json

import httpx
import pytest

from openclaw_algae.gateway.normalized import NormalizedChatRequest
from openclaw_algae.models.provider_credentials import ProviderCredentialRecord
from openclaw_algae.providers.base import ChatResponse, ProviderRateLimitError
from openclaw_algae.providers.claude import ClaudeWebAdapter, ClaudeWebClient
from openclaw_algae.providers.doubao import (
    DoubaoWebAdapter,
    DoubaoWebClient,
    _parse_doubao_response_text,
)
from openclaw_algae.providers.gemini import GeminiApiClient, GeminiWebAdapter
from openclaw_algae.providers.glm import (
    GLMApiClient,
    GLMIntlApiClient,
    GLMWebAdapter,
    _compute_glm_intl_request_signature,
)
from openclaw_algae.providers.grok import GrokApiClient, GrokWebAdapter
from openclaw_algae.providers.kimi import KimiWebClient
from openclaw_algae.providers.manus import ManusApiAdapter, ManusApiClient
from openclaw_algae.providers.mimo import MimoWebAdapter, MimoWebClient
from openclaw_algae.providers.qwen import (
    QwenCnApiClient,
    QwenApiClient,
    QwenWebAdapter,
    _iter_qwen_cn_sse_text_chunks,
)




def test_glm_api_client_uses_trust_env_false_by_default() -> None:
    credentials = ProviderCredentialRecord(
        provider="glm-cn",
        kind="browser_session",
        cookie="chatglm_token=value",
        headers={},
        user_agent="ua",
        status="valid",
    )

    client = GLMApiClient(credentials)

    assert client._client.trust_env is False


def test_kimi_web_client_uses_trust_env_false_by_default() -> None:
    credentials = ProviderCredentialRecord(
        provider="kimi",
        kind="browser_session",
        cookie="kimi-auth=value",
        headers={},
        user_agent="ua",
        status="valid",
    )

    client = KimiWebClient(credentials)

    assert client._client.trust_env is False


def test_chatgpt_api_client_uses_trust_env_false_by_default(tmp_path) -> None:
    from openclaw_algae.providers.chatgpt import ChatGPTApiClient

    credentials = ProviderCredentialRecord(
        provider="chatgpt",
        kind="browser_session",
        cookie="session=1",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    client = ChatGPTApiClient(credentials, state_dir=tmp_path)

    assert client._client.trust_env is False


def test_claude_web_client_uses_trust_env_false_by_default() -> None:
    credentials = ProviderCredentialRecord(
        provider="claude",
        kind="browser_session",
        cookie="sessionKey=test",
        headers={},
        user_agent="ua",
        metadata={"organization_id": "org-1"},
        status="valid",
    )

    client = ClaudeWebClient(credentials)

    assert client._client.trust_env is False


def test_doubao_web_client_uses_trust_env_false_by_default() -> None:
    credentials = ProviderCredentialRecord(
        provider="doubao",
        kind="browser_session",
        cookie="sessionid=test",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    client = DoubaoWebClient(credentials)

    assert client._client.trust_env is False


def test_gemini_api_client_uses_trust_env_false_by_default() -> None:
    credentials = ProviderCredentialRecord(
        provider="gemini",
        kind="browser_session",
        cookie="__Secure-1PSIDTS=test",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    client = GeminiApiClient(credentials)

    assert client._client.trust_env is False


def test_grok_api_client_uses_trust_env_false_by_default() -> None:
    credentials = ProviderCredentialRecord(
        provider="grok",
        kind="browser_session",
        cookie="session=test",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    client = GrokApiClient(credentials)

    assert client._client.trust_env is False


def test_mimo_web_client_uses_trust_env_false_by_default() -> None:
    credentials = ProviderCredentialRecord(
        provider="mimo",
        kind="browser_session",
        cookie='serviceToken="svc-token"',
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    client = MimoWebClient(credentials)

    assert client._client.trust_env is False


def test_manus_api_client_uses_trust_env_false_by_default() -> None:
    credentials = ProviderCredentialRecord(
        provider="manus",
        kind="api_key",
        cookie=None,
        headers={"api_key": "manus-key"},
        user_agent=None,
        metadata={},
        status="valid",
    )

    client = ManusApiClient(credentials)

    assert client._client.trust_env is False


def test_kimi_web_client_maps_resource_exhausted_to_rate_limit_error() -> None:
    credentials = ProviderCredentialRecord(
        provider="kimi",
        kind="browser_session",
        cookie="kimi-auth=value",
        headers={},
        user_agent="ua",
        status="valid",
    )
    payload = {
        "error": {
            "code": "resource_exhausted",
            "details": [
                {
                    "debug": {
                        "localizedMessage": {
                            "message": "The current model has reached its conversation limit."
                        }
                    }
                }
            ],
        }
    }
    encoded = json.dumps(payload).encode("utf-8")
    frame = b"\x00" + len(encoded).to_bytes(4, "big") + encoded

    client = KimiWebClient(credentials)

    with pytest.raises(ProviderRateLimitError, match="conversation limit"):
        client._parse_response(frame)


def test_kimi_web_client_retries_once_when_first_response_is_empty() -> None:
    credentials = ProviderCredentialRecord(
        provider="kimi",
        kind="browser_session",
        cookie="kimi-auth=value",
        headers={},
        user_agent="ua",
        status="valid",
    )
    calls = {"count": 0}

    def frame(payload: dict[str, object]) -> bytes:
        encoded = json.dumps(payload).encode("utf-8")
        return b"\x00" + len(encoded).to_bytes(4, "big") + encoded

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(200, content=frame({"op": "set", "block": {}}))
        return httpx.Response(
            200,
            content=frame({"op": "set", "block": {"text": {"content": "kimi ok"}}}),
        )

    client = KimiWebClient(credentials)
    client._client = httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)

    assert client.chat_completion(message="hello", model="moonshot-v1-8k") == "kimi ok"
    assert calls["count"] == 2
    credentials = ProviderCredentialRecord(
        provider="claude",
        kind="browser_session",
        cookie="sessionKey=sk-ant-sid01-test; anthropic-device-id=device-1",
        headers={},
        user_agent="ua",
        metadata={"session_key": "sk-ant-sid01-test"},
        status="valid",
    )
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/organizations":
            seen["org_headers"] = dict(request.headers)
            return httpx.Response(200, json=[{"uuid": "org-1"}])
        if request.url.path == "/api/organizations/org-1/chat_conversations":
            seen["conversation_payload"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(200, json={"uuid": "conv-1", "name": "Conversation"})
        if request.url.path == "/api/organizations/org-1/chat_conversations/conv-1/completion":
            seen["completion_payload"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                text=(
                    'data: {"type":"content_block_delta","delta":{"text":"hello"}}\n\n'
                    'data: {"type":"content_block_delta","delta":{"text":" world"}}\n\n'
                    'data: [DONE]\n\n'
                ),
                headers={"content-type": "text/event-stream"},
            )
        raise AssertionError(f"Unexpected path: {request.url.path}")

    client = ClaudeWebClient(
        credentials,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert client.discover_organization_id() == "org-1"
    assert client.chat_completion(message="hello", model="claude-sonnet-4-6") == "hello world"
    assert seen["org_headers"]["anthropic-device-id"] == "device-1"
    assert seen["completion_payload"]["model"] == "claude-sonnet-4-6"


def test_chatgpt_adapter_reuses_client_instance_between_calls() -> None:
    from openclaw_algae.providers.chatgpt import ChatGPTWebAdapter

    credentials = ProviderCredentialRecord(
        provider="chatgpt",
        kind="browser_session",
        cookie="session=1",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )
    created = {"count": 0}

    class FakeClient:
        def __init__(self) -> None:
            created["count"] += 1

        def chat_completion(self, *, message: str, model: str) -> str:
            return "chatgpt answer"

    adapter = ChatGPTWebAdapter(client_factory=lambda _: FakeClient())

    request = NormalizedChatRequest(
        model="algae/chatgpt/gpt-4",
        messages=[{"role": "user", "content": "hello"}],
    )

    first = adapter.chat(request, credentials)
    second = adapter.chat(request, credentials)

    assert first.content == "chatgpt answer"
    assert second.content == "chatgpt answer"
    assert created["count"] == 1


def test_chatgpt_adapter_streams_using_client_when_available() -> None:
    from openclaw_algae.providers.chatgpt import ChatGPTWebAdapter

    credentials = ProviderCredentialRecord(
        provider="chatgpt",
        kind="browser_session",
        cookie="session=1",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    class FakeClient:
        def iter_chat_completion_text(self, *, message: str, model: str):
            assert model == "gpt-4"
            assert message == "User: hello"
            yield "he"
            yield "llo"

    adapter = ChatGPTWebAdapter(client_factory=lambda _: FakeClient())

    assert list(
        adapter.stream_chat(
            NormalizedChatRequest(
                model="algae/chatgpt/gpt-4",
                messages=[{"role": "user", "content": "hello"}],
            ),
            credentials,
        )
        or []
    ) == ["he", "llo"]


def test_claude_adapter_uses_prompt_history_and_model() -> None:
    credentials = ProviderCredentialRecord(
        provider="claude",
        kind="browser_session",
        cookie="sessionKey=sk-ant-sid01-test",
        headers={},
        user_agent="ua",
        metadata={"session_key": "sk-ant-sid01-test"},
        status="valid",
    )
    calls: dict[str, object] = {}

    class FakeClient:
        def chat_completion(self, *, message: str, model: str, conversation_id=None) -> str:
            calls["message"] = message
            calls["model"] = model
            calls["conversation_id"] = conversation_id
            return "claude answer"

    adapter = ClaudeWebAdapter(client_factory=lambda _: FakeClient())

    response = adapter.chat(
        NormalizedChatRequest(
            model="algae/claude/claude-sonnet-4-6",
            messages=[
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "hello"},
            ],
        ),
        credentials,
    )

    assert response.content == "claude answer"
    assert calls["model"] == "claude-sonnet-4-6"
    assert calls["message"] == "System: be concise\n\nUser: hello"


def test_claude_web_client_streams_incremental_text() -> None:
    credentials = ProviderCredentialRecord(
        provider="claude",
        kind="browser_session",
        cookie="sessionKey=sk-ant-sid01-test",
        headers={},
        user_agent="ua",
        metadata={"session_key": "sk-ant-sid01-test", "organization_id": "org-1"},
        status="valid",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/organizations/org-1/chat_conversations":
            return httpx.Response(200, json={"uuid": "conv-1", "name": "Conversation"})
        if request.url.path == "/api/organizations/org-1/chat_conversations/conv-1/completion":
            return httpx.Response(
                200,
                text=(
                    'data: {"type":"content_block_delta","delta":{"text":"hello"}}\n\n'
                    'data: {"type":"content_block_delta","delta":{"text":" world"}}\n\n'
                    'data: [DONE]\n\n'
                ),
                headers={"content-type": "text/event-stream"},
            )
        raise AssertionError(f"Unexpected path: {request.url.path}")

    client = ClaudeWebClient(
        credentials,
        client=httpx.Client(transport=httpx.MockTransport(handler), trust_env=False),
    )

    assert list(client.iter_chat_completion_text(message="hello", model="claude-sonnet-4-6")) == [
        "hello",
        " world",
    ]


def test_claude_adapter_streams_using_client_when_available() -> None:
    credentials = ProviderCredentialRecord(
        provider="claude",
        kind="browser_session",
        cookie="sessionKey=1",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    class FakeClient:
        def iter_chat_completion_text(self, *, message: str, model: str):
            assert model == "claude-sonnet-4-6"
            assert message == "User: hello"
            yield "he"
            yield "llo"

    adapter = ClaudeWebAdapter(client_factory=lambda _: FakeClient())

    assert list(
        adapter.stream_chat(
            NormalizedChatRequest(
                model="algae/claude/claude-sonnet-4-6",
                messages=[{"role": "user", "content": "hello"}],
            ),
            credentials,
        )
        or []
    ) == ["he", "llo"]


def test_doubao_web_client_builds_samantha_request_and_parses_text() -> None:
    credentials = ProviderCredentialRecord(
        provider="doubao",
        kind="browser_session",
        cookie="sessionid=session-1; ttwid=ttwid-1; msToken=ms-token-1; s_v_web_id=verify_fp_1",
        headers={},
        user_agent="ua",
        metadata={"sessionid": "session-1", "ttwid": "ttwid-1"},
        status="valid",
    )
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/samantha/chat/completion"
        seen["query"] = dict(request.url.params)
        seen["headers"] = dict(request.headers)
        seen["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            text=(
                '{"event_type":2001,"event_data":"{\\"message\\":{\\"content\\":\\"{\\\\\\"text\\\\\\":\\\\\\"hello\\\\\\"}\\",\\"content_type\\":2001}}"}\n'
                'event: CHUNK_DELTA\n'
                'data: {"text":" world"}\n\n'
                'event: SSE_REPLY_END\n'
                'data: {"end_type":1}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    client = DoubaoWebClient(
        credentials,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    content = client.chat_completion(message="hello", model="doubao-seed-2.0")

    assert content == "hello world"
    assert seen["headers"]["agw-js-conv"] == "str"
    assert seen["query"]["aid"] == "497858"
    assert seen["query"]["fp"] == "verify_fp_1"
    assert seen["query"]["msToken"] == "ms-token-1"
    assert seen["payload"]["conversation_id"] == "0"


def test_doubao_adapter_uses_prompt_history() -> None:
    credentials = ProviderCredentialRecord(
        provider="doubao",
        kind="browser_session",
        cookie="sessionid=session-1",
        headers={},
        user_agent="ua",
        metadata={"sessionid": "session-1"},
        status="valid",
    )
    calls: dict[str, object] = {}

    class FakeClient:
        def chat_completion(self, *, message: str, model: str) -> str:
            calls["message"] = message
            calls["model"] = model
            return "doubao answer"

    adapter = DoubaoWebAdapter(client_factory=lambda _: FakeClient())

    response = adapter.chat(
        NormalizedChatRequest(
            model="algae/doubao/doubao-seed-2.0",
            messages=[
                {"role": "system", "content": "be careful"},
                {"role": "user", "content": "hello"},
            ],
        ),
        credentials,
    )

    assert response.content == "doubao answer"
    assert "<|im_start|>system\nbe careful\n" in calls["message"]
    assert calls["model"] == "doubao-seed-2.0"


def test_doubao_adapter_streams_using_client_when_available() -> None:
    credentials = ProviderCredentialRecord(
        provider="doubao",
        kind="browser_session",
        cookie="sessionid=session-1",
        headers={},
        user_agent="ua",
        metadata={"sessionid": "session-1"},
        status="valid",
    )

    class FakeClient:
        def iter_chat_completion_text(self, *, message: str, model: str):
            assert model == "doubao-seed-2.0"
            assert "<|im_start|>user\nhello\n" in message
            yield "你"
            yield "好"

    adapter = DoubaoWebAdapter(client_factory=lambda _: FakeClient())

    assert list(
        adapter.stream_chat(
            NormalizedChatRequest(
                model="algae/doubao/doubao-seed-2.0",
                messages=[{"role": "user", "content": "hello"}],
            ),
            credentials,
        )
        or []
    ) == ["你", "好"]


def test_parse_doubao_response_text_ignores_duplicate_tts_chunks() -> None:
    payload = """id: 5
event: STREAM_MSG_NOTIFY
data: {"content":{"content_block":[{"block_type":10000,"content":{"text_block":{"text":"al"}},"is_finish":false}],"tts_content":"al"}}

id: 8
event: STREAM_CHUNK
data: {"patch_op":[{"patch_object":1,"patch_type":1,"patch_value":{"content_block":[{"block_type":10000,"content":{"text_block":{"text":"gae"}},"is_finish":false}]}}]}

id: 9
event: STREAM_CHUNK
data: {"patch_op":[{"patch_object":111,"patch_type":1,"patch_value":{"tts_content":"gae"}}]}

id: 12
event: CHUNK_DELTA
data: {"text":"-d"}

id: 13
event: STREAM_CHUNK
data: {"patch_op":[{"patch_object":111,"patch_type":1,"patch_value":{"tts_content":"-d"}}]}

id: 16
event: CHUNK_DELTA
data: {"text":"ou"}

id: 17
event: STREAM_CHUNK
data: {"patch_op":[{"patch_object":111,"patch_type":1,"patch_value":{"tts_content":"ou"}}]}

id: 20
event: CHUNK_DELTA
data: {"text":"bao"}

id: 21
event: STREAM_CHUNK
data: {"patch_op":[{"patch_object":111,"patch_type":1,"patch_value":{"tts_content":"bao"}}]}

id: 24
event: CHUNK_DELTA
data: {"text":"-check"}

id: 25
event: STREAM_CHUNK
data: {"patch_op":[{"patch_object":111,"patch_type":1,"patch_value":{"tts_content":"-check"}}]}
"""

    assert _parse_doubao_response_text(payload) == "algae-doubao-check"


def test_mimo_web_client_builds_headers_and_extracts_text() -> None:
    credentials = ProviderCredentialRecord(
        provider="mimo",
        kind="browser_session",
        cookie='serviceToken="svc-token"; xiaomichatbot_ph="bot-ph"',
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        seen["payload"] = json.loads(request.content.decode("utf-8"))
        assert request.url.path == "/open-apis/bot/chat"
        assert request.url.params["xiaomichatbot_ph"] == "bot-ph"
        return httpx.Response(
            200,
            text='{"type":"text","content":"hello"}\n{"type":"text","content":" world"}\n{"conversationId":"abc123"}',
            headers={"content-type": "text/event-stream"},
        )

    client = MimoWebClient(
        credentials,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert client.chat_completion(message="hello", model="mimo-v2-pro") == "hello world"
    assert seen["headers"]["authorization"] == "Bearer svc-token"
    assert seen["payload"]["query"] == "hello"
    assert seen["payload"]["modelConfig"]["model"] == "mimo-v2-pro"


def test_mimo_adapter_uses_prompt_history() -> None:
    credentials = ProviderCredentialRecord(
        provider="mimo",
        kind="browser_session",
        cookie="serviceToken=svc-token",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )
    calls: dict[str, object] = {}

    class FakeClient:
        def chat_completion(self, *, message: str, model: str) -> str:
            calls["message"] = message
            calls["model"] = model
            return "mimo answer"

    adapter = MimoWebAdapter(client_factory=lambda _: FakeClient())

    response = adapter.chat(
        NormalizedChatRequest(
            model="algae/mimo/mimo-v2-pro",
            messages=[
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "hello"},
            ],
        ),
        credentials,
    )

    assert response.content == "mimo answer"
    assert calls["model"] == "mimo-v2-pro"
    assert calls["message"] == "System: be concise\n\nUser: hello"


def test_mimo_web_client_streams_incremental_text() -> None:
    credentials = ProviderCredentialRecord(
        provider="mimo",
        kind="browser_session",
        cookie='serviceToken="svc-token"; xiaomichatbot_ph="bot-ph"',
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text='{"type":"text","content":"hello"}\n{"type":"text","content":" world"}\n',
            headers={"content-type": "text/event-stream"},
        )

    client = MimoWebClient(
        credentials,
        client=httpx.Client(transport=httpx.MockTransport(handler), trust_env=False),
    )

    assert list(client.iter_chat_completion_text(message="hello", model="mimo-v2-pro")) == [
        "hello",
        " world",
    ]


def test_mimo_adapter_streams_using_client_when_available() -> None:
    credentials = ProviderCredentialRecord(
        provider="mimo",
        kind="browser_session",
        cookie='serviceToken="svc-token"',
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    class FakeClient:
        def iter_chat_completion_text(self, *, message: str, model: str):
            assert model == "mimo-v2-pro"
            assert message == "User: hello"
            yield "你"
            yield "好"

    adapter = MimoWebAdapter(client_factory=lambda _: FakeClient())

    assert list(
        adapter.stream_chat(
            NormalizedChatRequest(
                model="algae/mimo/mimo-v2-pro",
                messages=[{"role": "user", "content": "hello"}],
            ),
            credentials,
        )
        or []
    ) == ["你", "好"]


def test_manus_api_client_polls_until_completed(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="manus",
        kind="api_key",
        cookie=None,
        headers={"api_key": "manus-key"},
        user_agent=None,
        metadata={},
        status="valid",
    )
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        if request.method == "POST" and request.url.path == "/v1/tasks":
            assert request.headers["API_KEY"] == "manus-key"
            payload = json.loads(request.content.decode("utf-8"))
            assert payload["prompt"] == "hello"
            return httpx.Response(200, json={"task_id": "task-1"})
        if request.method == "GET" and request.url.path == "/v1/tasks/task-1":
            if calls.count("GET /v1/tasks/task-1") == 1:
                return httpx.Response(200, json={"id": "task-1", "status": "running"})
            return httpx.Response(
                200,
                json={
                    "id": "task-1",
                    "status": "completed",
                    "output": [
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "hello"},
                                {"type": "output_text", "text": " world"},
                            ],
                        }
                    ],
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url.path}")

    monkeypatch.setattr("time.sleep", lambda _: None)
    client = ManusApiClient(
        credentials,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        poll_interval_seconds=0,
        max_poll_seconds=1,
    )

    assert client.chat_completion(message="hello", model="manus-1.6") == "hello\n\n world"


def test_manus_api_client_streams_incremental_text(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="manus",
        kind="api_key",
        cookie=None,
        headers={"api_key": "manus-key"},
        user_agent=None,
        metadata={},
        status="valid",
    )
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        if request.method == "POST" and request.url.path == "/v1/tasks":
            return httpx.Response(200, json={"task_id": "task-1"})
        if request.method == "GET" and request.url.path == "/v1/tasks/task-1":
            if calls.count("GET /v1/tasks/task-1") == 1:
                return httpx.Response(
                    200,
                    json={
                        "id": "task-1",
                        "status": "running",
                        "output": [{"role": "assistant", "content": [{"type": "output_text", "text": "hello"}]}],
                    },
                )
            return httpx.Response(
                200,
                json={
                    "id": "task-1",
                    "status": "completed",
                    "output": [
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "hello"},
                                {"type": "output_text", "text": " world"},
                            ],
                        }
                    ],
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url.path}")

    monkeypatch.setattr("time.sleep", lambda _: None)
    client = ManusApiClient(
        credentials,
        client=httpx.Client(transport=httpx.MockTransport(handler), trust_env=False),
        poll_interval_seconds=0,
        max_poll_seconds=1,
    )

    assert list(client.iter_chat_completion_text(message="hello", model="manus-1.6")) == ["hello", "\n\n world"]


def test_manus_adapter_uses_manual_api_key() -> None:
    credentials = ProviderCredentialRecord(
        provider="manus",
        kind="api_key",
        cookie=None,
        headers={"api_key": "manus-key"},
        user_agent=None,
        metadata={},
        status="valid",
    )
    calls: dict[str, object] = {}

    class FakeClient:
        def chat_completion(self, *, message: str, model: str) -> str:
            calls["message"] = message
            calls["model"] = model
            return "manus answer"

    adapter = ManusApiAdapter(client_factory=lambda _: FakeClient())

    response = adapter.chat(
        NormalizedChatRequest(
            model="algae/manus/manus-1.6",
            messages=[{"role": "user", "content": "hello"}],
        ),
        credentials,
    )

    assert response.content == "manus answer"
    assert calls["message"] == "User: hello"
    assert calls["model"] == "manus-1.6"


def test_manus_adapter_streams_using_client_when_available() -> None:
    credentials = ProviderCredentialRecord(
        provider="manus",
        kind="api_key",
        cookie=None,
        headers={"api_key": "manus-key"},
        user_agent=None,
        metadata={},
        status="valid",
    )

    class FakeClient:
        def iter_chat_completion_text(self, *, message: str, model: str):
            assert model == "manus-1.6"
            assert message == "User: hello"
            yield "he"
            yield "llo"

    adapter = ManusApiAdapter(client_factory=lambda _: FakeClient())

    assert list(
        adapter.stream_chat(
            NormalizedChatRequest(
                model="algae/manus/manus-1.6",
                messages=[{"role": "user", "content": "hello"}],
            ),
            credentials,
        )
        or []
    ) == ["he", "llo"]


def test_gemini_adapter_streams_using_client_when_available() -> None:
    credentials = ProviderCredentialRecord(
        provider="gemini",
        kind="browser_session",
        cookie="__Secure-1PSIDTS=test",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    class FakeClient:
        def iter_chat_completion_text(self, *, message: str, model: str):
            assert model == "gemini-pro"
            assert message == "User: hello"
            yield "he"
            yield "llo"

    adapter = GeminiWebAdapter(client_factory=lambda _: FakeClient())

    assert list(
        adapter.stream_chat(
            NormalizedChatRequest(
                model="algae/gemini/gemini-pro",
                messages=[{"role": "user", "content": "hello"}],
            ),
            credentials,
        )
        or []
    ) == ["he", "llo"]


def test_grok_adapter_streams_using_client_when_available() -> None:
    credentials = ProviderCredentialRecord(
        provider="grok",
        kind="browser_session",
        cookie="session=test",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    class FakeClient:
        def iter_chat_completion_text(self, *, message: str, model: str):
            assert model == "grok-2"
            assert message == "User: hello"
            yield "he"
            yield "llo"

    adapter = GrokWebAdapter(client_factory=lambda _: FakeClient())

    assert list(
        adapter.stream_chat(
            NormalizedChatRequest(
                model="algae/grok/grok-2",
                messages=[{"role": "user", "content": "hello"}],
            ),
            credentials,
        )
        or []
    ) == ["he", "llo"]


def test_qwen_adapter_builds_tool_prompt_and_tool_result_history() -> None:
    credentials = ProviderCredentialRecord(
        provider="qwen-intl",
        kind="browser_session",
        cookie="qwen_session=session-1",
        headers={},
        user_agent="ua",
        metadata={"session_token": "session-1"},
        status="valid",
    )
    calls: dict[str, object] = {}

    class FakeClient:
        def chat_completion_text(self, *, message: str, model: str) -> str:
            calls["message"] = message
            calls["model"] = model
            return '<tool_calls>[{"name":"get_weather","arguments":{"location":"Tokyo"}}]</tool_calls>'

    adapter = QwenWebAdapter(client_factory=lambda _: FakeClient())

    response = adapter.chat(
        NormalizedChatRequest(
            model="algae/qwen-intl/qwen3.6-plus",
            messages=[
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "What is the weather in Tokyo?"},
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
                    "content": '{"temp":22,"unit":"C"}',
                },
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    },
                }
            ],
            tool_choice="required",
        ),
        credentials,
    )

    assert response.content is None
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0]["function"]["name"] == "get_weather"
    assert calls["model"] == "qwen3.6-plus"
    assert "You must respond using only the following XML-like tags" in calls["message"]
    assert "Tool choice for this response is required." in calls["message"]
    assert "- get_weather(location: string (required)): Get weather" in calls["message"]
    assert "Assistant tool calls: call_weather_1" in calls["message"]
    assert "<tool_result>" in calls["message"]


def test_chatgpt_adapter_maps_tool_json_response_to_openai_tool_calls() -> None:
    from openclaw_algae.providers.chatgpt import ChatGPTWebAdapter

    credentials = ProviderCredentialRecord(
        provider="chatgpt",
        kind="browser_session",
        cookie="session=1",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    class FakeClient:
        def chat_completion(self, *, message: str, model: str) -> str:
            return '```tool_json\n{"tool":"write_file","parameters":{"path":"/tmp/demo.txt","content":"hello"}}\n```'

    adapter = ChatGPTWebAdapter(client_factory=lambda _: FakeClient())

    response = adapter.chat(
        NormalizedChatRequest(
            model="algae/chatgpt/gpt-4",
            messages=[{"role": "user", "content": "write a file"}],
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
        ),
        credentials,
    )

    assert response.finish_reason == "tool_calls"
    assert response.content is None
    assert response.tool_calls == [
        {
            "id": "call_write_file_1",
            "type": "function",
            "function": {
                "name": "write_file",
                "arguments": '{"path":"/tmp/demo.txt","content":"hello"}',
            },
        }
    ]


def test_qwen_adapter_maps_xml_tool_call_response_to_openai_tool_calls() -> None:
    credentials = ProviderCredentialRecord(
        provider="qwen-intl",
        kind="browser_session",
        cookie="qwen_session=session-1",
        headers={},
        user_agent="ua",
        metadata={"session_token": "session-1"},
        status="valid",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v2/chats/new":
            return httpx.Response(200, json={"data": {"id": "chat-1"}})
        if request.url.path == "/api/v2/chat/completions":
            return httpx.Response(
                200,
                text=(
                    'data: {"choices":[{"delta":{"content":"<think>Need a tool.</think><tool_call id=\\"call_weather_1\\" name=\\"get_weather\\">{\\"location\\":\\"Tokyo\\"}</tool_call>"}}]}\n\n'
                    'data: [DONE]\n\n'
                ),
                headers={"content-type": "text/event-stream"},
            )
        raise AssertionError(f"Unexpected path: {request.url.path}")

    client = httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)
    adapter = QwenWebAdapter(client_factory=lambda cred: QwenApiClient(cred, client=client))

    response = adapter.chat(
        NormalizedChatRequest(
            model="algae/qwen-intl/qwen3.6-plus",
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            tool_choice="auto",
        ),
        credentials,
    )

    assert response.content == "<think>Need a tool.</think>"
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls == [
        {
            "id": "call_weather_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location":"Tokyo"}',
            },
        }
    ]


def test_qwen_stream_parser_preserves_think_phase_text_with_tags() -> None:
    from openclaw_algae.providers.qwen import _parse_qwen_sse_text

    payload = (
        'data: {"choices":[{"delta":{"content":"这是思考","phase":"think","status":"typing"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"你好","phase":"answer","status":"typing"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"！","phase":"answer","status":"typing"}}]}\n\n'
    )

    assert _parse_qwen_sse_text(payload) == "<think>这是思考</think>你好！"


def test_qwen_incremental_stream_parser_preserves_think_phase_text_with_tags() -> None:
    from openclaw_algae.providers.qwen import _iter_qwen_sse_text_chunks

    lines = iter(
        [
            'data: {"choices":[{"delta":{"content":"这是思考","phase":"think","status":"typing"}}]}',
            'data: {"choices":[{"delta":{"content":"你好","phase":"answer","status":"typing"}}]}',
            'data: {"choices":[{"delta":{"content":"！","phase":"answer","status":"typing"}}]}',
            "data: [DONE]",
        ]
    )

    assert "".join(_iter_qwen_sse_text_chunks(lines)) == "<think>这是思考</think>你好！"


def test_qwen_cn_incremental_stream_parser_ignores_semantically_duplicate_numbered_snapshot() -> None:
    summary_text = (
        "无需等待完整结果，边生成边展示，大幅降低用户等待焦虑。"
        " 交互更流畅自然，尤其适合对话、翻译等实时场景。"
        " 提升系统响应体验，避免长时间空白导致用户流失。"
    )
    numbered_snapshot = (
        "1. 无需等待完整结果，边生成边展示，大幅降低用户等待焦虑。\n"
        "2. 交互更流畅自然，尤其适合对话、翻译等实时场景。\n"
        "3. 提升系统响应体验，避免长时间空白导致用户流失。"
    )

    lines = iter(
        [
            f'data: {json.dumps({"data": {"messages": [{"content": summary_text}]}} , ensure_ascii=False)}',
            f'data: {json.dumps({"data": {"messages": [{"content": numbered_snapshot}]}} , ensure_ascii=False)}',
            "data: [DONE]",
        ]
    )

    assert list(_iter_qwen_cn_sse_text_chunks(lines)) == [summary_text]


def test_qwen_cn_incremental_stream_parser_uses_latest_message_snapshot_only() -> None:
    lines = iter(
        [
            'data: {"data":{"messages":[{"content":"旧快照"},{"content":"最新快照-1"}]}}',
            'data: {"data":{"messages":[{"content":"旧快照"},{"content":"最新快照-12"}]}}',
            "data: [DONE]",
        ]
    )

    assert list(_iter_qwen_cn_sse_text_chunks(lines)) == ["最新快照-1", "2"]


def test_qwen_cn_api_client_uses_fresh_session_id_per_request() -> None:
    credentials = ProviderCredentialRecord(
        provider="qwen-cn",
        kind="browser_session",
        cookie="XSRF-TOKEN=token; b-user-id=user-1",
        headers={},
        user_agent="ua",
        status="valid",
    )

    seen_session_ids: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        seen_session_ids.append(str(payload["session_id"]))
        return httpx.Response(
            200,
            text='data: {"data":{"messages":[{"content":"QWEN_CN_OK"}]}}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    client = QwenCnApiClient(
        credentials,
        client=httpx.Client(transport=httpx.MockTransport(handler), trust_env=False),
    )

    assert client.chat_completion_text(message="hello-1", model="Qwen3-Max") == "QWEN_CN_OK"
    assert client.chat_completion_text(message="hello-2", model="Qwen3-Max") == "QWEN_CN_OK"
    assert len(seen_session_ids) == 2
    assert seen_session_ids[0] != seen_session_ids[1]


def test_qwen_cn_api_client_chat_completion_falls_back_from_outline_workflow() -> None:
    credentials = ProviderCredentialRecord(
        provider="qwen-cn",
        kind="browser_session",
        cookie="XSRF-TOKEN=token; b-user-id=user-1",
        headers={},
        user_agent="ua",
        status="valid",
    )

    seen_models: list[str] = []
    outline_stream = (
        'data: {"data":{"extra_info":{"sub_scene":"creator/outline","document_scene":"document_longtext"},'
        '"messages":[{"mime_type":"signal/post","meta_data":{"scene":"chat_writer"}}]}}\n\n'
        'data: {"data":{"messages":[{"mime_type":"multi_load/iframe","content":"为确保完全符合你的创作要求，以下为写作大纲：\\n\\n[(outline_1)]","meta_data":{"multi_load":[{"type":"outline","content":{"text":"### 大纲"}}]}}]}}\n\n'
        "data: [DONE]\n\n"
    )
    normal_stream = (
        'data: {"data":{"messages":[{"content":"正"}]}}\n\n'
        'data: {"data":{"messages":[{"content":"正文"}]}}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        seen_models.append(str(payload["model"]))
        model = str(payload["model"])
        if model == "Qwen3-Max-Thinking":
            return httpx.Response(
                200,
                text=outline_stream,
                headers={"content-type": "text/event-stream"},
            )
        if model == "Qwen3.5-Flash":
            return httpx.Response(
                200,
                text=normal_stream,
                headers={"content-type": "text/event-stream"},
            )
        raise AssertionError(f"Unexpected model {model}")

    client = QwenCnApiClient(
        credentials,
        client=httpx.Client(transport=httpx.MockTransport(handler), trust_env=False),
    )

    response = client.chat_completion(
        message="来一个3000字自我介绍",
        model="Qwen3-Max-Thinking",
    )

    assert response.content == "正文"
    assert seen_models == ["Qwen3-Max-Thinking", "Qwen3.5-Flash"]


def test_qwen_cn_api_client_stream_falls_back_from_outline_workflow_before_placeholder() -> None:
    credentials = ProviderCredentialRecord(
        provider="qwen-cn",
        kind="browser_session",
        cookie="XSRF-TOKEN=token; b-user-id=user-1",
        headers={},
        user_agent="ua",
        status="valid",
    )

    seen_models: list[str] = []
    outline_stream = (
        'data: {"data":{"extra_info":{"sub_scene":"creator/outline","document_scene":"document_longtext"},'
        '"messages":[{"mime_type":"signal/post","meta_data":{"scene":"chat_writer"}}]}}\n\n'
        'data: {"data":{"messages":[{"mime_type":"multi_load/iframe","content":"为确保完全符合你的创作要求，以下为写作大纲：\\n\\n[(outline_1)]","meta_data":{"multi_load":[{"type":"outline","content":{"text":"### 大纲"}}]}}]}}\n\n'
        "data: [DONE]\n\n"
    )
    normal_stream = (
        'data: {"data":{"messages":[{"content":"正"}]}}\n\n'
        'data: {"data":{"messages":[{"content":"正文"}]}}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        seen_models.append(str(payload["model"]))
        model = str(payload["model"])
        if model == "Qwen3.5-千问":
            return httpx.Response(
                200,
                text=outline_stream,
                headers={"content-type": "text/event-stream"},
            )
        if model == "Qwen3.5-Flash":
            return httpx.Response(
                200,
                text=normal_stream,
                headers={"content-type": "text/event-stream"},
            )
        raise AssertionError(f"Unexpected model {model}")

    client = QwenCnApiClient(
        credentials,
        client=httpx.Client(transport=httpx.MockTransport(handler), trust_env=False),
    )

    assert list(
        client.iter_chat_completion_text(
            message="来一个3000字自我介绍",
            model="Qwen3.5-千问",
        )
    ) == ["正", "文"]
    assert seen_models == ["Qwen3.5-千问", "Qwen3.5-Flash"]


def test_qwen_adapter_normalizes_compat_model_aliases() -> None:
    credentials = ProviderCredentialRecord(
        provider="qwen-intl",
        kind="browser_session",
        cookie="qwen_session=session-1",
        headers={},
        user_agent="ua",
        metadata={"session_token": "session-1"},
        status="valid",
    )
    calls: dict[str, object] = {}

    class FakeClient:
        def chat_completion(self, *, message: str, model: str) -> object:
            calls["message"] = message
            calls["model"] = model
            return ChatResponse(model=model, content="compat answer")

    adapter = QwenWebAdapter(client_factory=lambda _: FakeClient())

    response = adapter.chat(
        NormalizedChatRequest(
            model="algae/qwen/qwen-3.6-235b-a22b",
            messages=[{"role": "user", "content": "hello"}],
        ),
        credentials,
    )

    assert response.content == "compat answer"
    assert calls["model"] == "qwen3.6-plus"


def test_qwen_adapter_prefers_browser_stream_client_when_available() -> None:
    credentials = ProviderCredentialRecord(
        provider="qwen-intl",
        kind="browser_session",
        cookie="qwen_session=session-1",
        headers={},
        user_agent="ua",
        metadata={"session_token": "session-1"},
        status="valid",
    )
    seen: dict[str, object] = {}

    class UnexpectedHttpClient:
        def iter_chat_completion_text(self, *, message: str, model: str):
            raise AssertionError("HTTP qwen stream client should not be used when browser stream client is configured")

    class BrowserStreamClient:
        def stream_chat_completion(self, *, message: str, model: str):
            seen["message"] = message
            seen["model"] = model
            yield "你"
            yield "好"

    adapter = QwenWebAdapter(
        client_factory=lambda _: UnexpectedHttpClient(),
        stream_client_factory=lambda _: BrowserStreamClient(),
    )

    chunks = list(
        adapter.stream_chat(
            NormalizedChatRequest(
                model="algae/qwen-intl/qwen3.6-plus",
                messages=[
                    {"role": "system", "content": "be concise"},
                    {"role": "user", "content": "hello"},
                ],
            ),
            credentials,
        )
        or []
    )

    assert chunks == ["你", "好"]
    assert seen["model"] == "qwen3.6-plus"
    assert seen["message"] == "System: be concise\n\nUser: hello"


def test_qwen_api_client_stream_retries_read_timeout_before_first_visible_chunk() -> None:
    credentials = ProviderCredentialRecord(
        provider="qwen-intl",
        kind="browser_session",
        cookie="qwen_session=session-1",
        headers={},
        user_agent="ua",
        metadata={"session_token": "session-1"},
        status="valid",
    )

    class TimeoutStreamResponse:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self):
            raise httpx.ReadTimeout("timed out")

    class SuccessStreamResponse:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self):
            return iter(
                [
                    'data: {"choices":[{"delta":{"content":"你","phase":"answer","status":"typing"}}]}',
                    'data: {"choices":[{"delta":{"content":"好","phase":"answer","status":"typing"}}]}',
                    "data: [DONE]",
                ]
            )

    class FakeClient:
        def __init__(self) -> None:
            self.stream_calls = 0

        def post(self, url: str, headers=None, json=None, params=None):
            assert url.endswith("/api/v2/chats/new")
            return httpx.Response(
                200,
                json={"data": {"id": "chat-1"}},
                request=httpx.Request("POST", url),
            )

        def stream(self, method: str, url: str, headers=None, params=None, json=None):
            self.stream_calls += 1
            if self.stream_calls == 1:
                return TimeoutStreamResponse()
            return SuccessStreamResponse()

    fake_client = FakeClient()
    client = QwenApiClient(credentials, client=fake_client)

    assert list(client.iter_chat_completion_text(message="hello", model="qwen3.6-plus")) == ["你", "好"]
    assert fake_client.stream_calls == 2


def test_claude_adapter_maps_tool_json_response_to_openai_tool_calls() -> None:
    credentials = ProviderCredentialRecord(
        provider="claude",
        kind="browser_session",
        cookie="sessionKey=abc",
        headers={},
        user_agent="ua",
        metadata={"session_key": "abc"},
        status="valid",
    )

    class FakeClient:
        def chat_completion(self, *, message: str, model: str, conversation_id=None) -> str:
            return '```tool_json\n{"tool":"web_search","parameters":{"query":"上海天气"}}\n```'

    adapter = ClaudeWebAdapter(client_factory=lambda _: FakeClient())

    response = adapter.chat(
        NormalizedChatRequest(
            model="algae/claude/claude-sonnet-4-6",
            messages=[{"role": "user", "content": "搜索上海天气"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search the web",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                }
            ],
            tool_choice="required",
        ),
        credentials,
    )

    assert response.content is None
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0]["function"]["name"] == "web_search"


def test_chatgpt_web_client_persists_conversation_id_in_session_store(tmp_path) -> None:
    from openclaw_algae.providers.chatgpt import ChatGPTApiClient
    from openclaw_algae.storage.provider_sessions import load_provider_session

    credentials = ProviderCredentialRecord(
        provider="chatgpt",
        kind="browser_session",
        cookie="session=1",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["conversation_id"] is None
        return httpx.Response(
            200,
            text='data: {"message":{"content":{"parts":["hello"]}}}\n\n',
            headers={"x-conversation-id": "conv-42"},
        )

    client = ChatGPTApiClient(
        credentials,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        state_dir=tmp_path,
    )

    assert client.chat_completion(message="hello", model="gpt-4") == "hello"
    assert load_provider_session(tmp_path, provider="chatgpt", credentials=credentials) == {"conversation_id": "conv-42"}


def test_chatgpt_web_client_streams_incremental_text_and_persists_conversation_id(tmp_path) -> None:
    from openclaw_algae.providers.chatgpt import ChatGPTApiClient
    from openclaw_algae.storage.provider_sessions import load_provider_session

    credentials = ProviderCredentialRecord(
        provider="chatgpt",
        kind="browser_session",
        cookie="session=1",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["conversation_id"] is None
        return httpx.Response(
            200,
            text=(
                'data: {"message":{"content":{"parts":["he"]}}}\n\n'
                'data: {"message":{"content":{"parts":["hello"]}}}\n\n'
                'data: {"text":" world"}\n\n'
                'data: [DONE]\n\n'
            ),
            headers={
                "content-type": "text/event-stream",
                "x-conversation-id": "conv-43",
            },
        )

    client = ChatGPTApiClient(
        credentials,
        client=httpx.Client(transport=httpx.MockTransport(handler), trust_env=False),
        state_dir=tmp_path,
    )

    assert list(client.iter_chat_completion_text(message="hello", model="gpt-4")) == ["he", "llo", " world"]
    assert load_provider_session(tmp_path, provider="chatgpt", credentials=credentials) == {"conversation_id": "conv-43"}


def test_kimi_web_client_streams_incremental_text() -> None:
    credentials = ProviderCredentialRecord(
        provider="kimi",
        kind="browser_session",
        cookie="kimi-auth=value",
        headers={},
        user_agent="ua",
        status="valid",
    )

    def frame(payload: dict[str, object]) -> bytes:
        encoded = json.dumps(payload).encode("utf-8")
        return b"\x00" + len(encoded).to_bytes(4, "big") + encoded

    stream_bytes = (
        frame({"op": "set", "block": {"text": {"content": "你"}}})
        + frame({"op": "set", "block": {"text": {"content": "你好"}}})
        + frame({"op": "append", "block": {"text": {"content": "！"}}})
    )

    def chunked_bytes() -> bytes:
        return stream_bytes

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=chunked_bytes(),
            headers={"content-type": "application/connect+json"},
        )

    client = KimiWebClient(
        credentials,
    )
    client._client = httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)

    assert list(client.iter_chat_completion_text(message="hello", model="moonshot-v1-32k")) == ["你", "好", "！"]


def test_kimi_adapter_streams_using_client_when_available() -> None:
    from openclaw_algae.providers.kimi import KimiWebAdapter

    credentials = ProviderCredentialRecord(
        provider="kimi",
        kind="browser_session",
        cookie="kimi-auth=value",
        headers={},
        user_agent="ua",
        status="valid",
    )

    class FakeClient:
        def iter_chat_completion_text(self, *, message: str, model: str):
            assert model == "moonshot-v1-32k"
            assert message == "User: hello"
            yield "你"
            yield "好"

    adapter = KimiWebAdapter(client_factory=lambda _: FakeClient())

    assert list(
        adapter.stream_chat(
            NormalizedChatRequest(
                model="algae/kimi/moonshot-v1-32k",
                messages=[{"role": "user", "content": "hello"}],
            ),
            credentials,
        )
        or []
    ) == ["你", "好"]


def test_qwen_web_client_starts_fresh_chat_even_when_persisted_chat_id_exists(tmp_path) -> None:
    from openclaw_algae.providers.qwen import QwenApiClient
    from openclaw_algae.storage.provider_sessions import save_provider_session

    credentials = ProviderCredentialRecord(
        provider="qwen-intl",
        kind="browser_session",
        cookie="token=1",
        headers={},
        user_agent="ua",
        metadata={"session_token": "token-1"},
        status="valid",
    )
    save_provider_session(tmp_path, provider="qwen-intl", credentials=credentials, state={"chat_id": "chat-persisted"})
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/v2/chats/new"):
            return httpx.Response(200, json={"data": {"id": "chat-fresh"}})
        seen["query"] = dict(request.url.params)
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"hello qwen"}}]}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    client = QwenApiClient(
        credentials,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        state_dir=tmp_path,
    )

    response = client.chat_completion(message="hi", model="qwen3.5-plus")

    assert response.content == "hello qwen"
    assert seen["query"]["chat_id"] == "chat-fresh"


def test_qwen_web_client_retries_with_fresh_chat_id_when_chat_is_in_progress(tmp_path) -> None:
    credentials = ProviderCredentialRecord(
        provider="qwen-intl",
        kind="browser_session",
        cookie="token=1",
        headers={},
        user_agent="ua",
        metadata={"session_token": "token-1"},
        status="valid",
    )
    calls: list[tuple[str, str]] = []
    issued_chat_ids = iter(["chat-stale", "chat-fresh"])

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        if request.url.path.endswith("/api/v2/chat/completions"):
            chat_id = dict(request.url.params).get("chat_id", "")
            if chat_id == "chat-stale":
                return httpx.Response(
                    200,
                    text=(
                        '{"success":false,"request_id":"req-1","data":'
                        '{"code":"Bad_Request","details":"The chat is in progress!"}}'
                    ),
                    headers={"content-type": "application/json"},
                )
            return httpx.Response(
                200,
                text='data: {"choices":[{"delta":{"content":"hello qwen retry"}}]}\n\n',
                headers={"content-type": "text/event-stream"},
            )
        if request.url.path.endswith("/api/v2/chats/new"):
            return httpx.Response(
                200,
                json={"data": {"id": next(issued_chat_ids)}},
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")
    client = QwenApiClient(
        credentials,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        state_dir=tmp_path,
    )

    response = client.chat_completion(message="hi", model="qwen3.6-plus")

    assert response.content == "hello qwen retry"
    assert any(url.endswith("/api/v2/chats/new") for _, url in calls)
    assert any("chat_id=chat-fresh" in url for _, url in calls)


def test_qwen_web_client_disables_thinking_for_standard_models(tmp_path) -> None:
    credentials = ProviderCredentialRecord(
        provider="qwen-intl",
        kind="browser_session",
        cookie="token=1",
        headers={},
        user_agent="ua",
        metadata={"session_token": "token-1"},
        status="valid",
    )
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/v2/chats/new"):
            return httpx.Response(200, json={"data": {"id": "chat-fresh"}})
        if request.url.path.endswith("/api/v2/chat/completions"):
            seen["payload"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                text='data: {"choices":[{"delta":{"phase":"answer","content":"hello qwen"}}]}\n\n',
                headers={"content-type": "text/event-stream"},
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    client = QwenApiClient(
        credentials,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        state_dir=tmp_path,
    )

    response = client.chat_completion(message="hi", model="qwen3.5-plus")

    assert response.content == "hello qwen"
    assert seen["payload"]["messages"][0]["feature_config"]["thinking_enabled"] is False
    assert seen["payload"]["messages"][0]["feature_config"]["output_schema"] == "phase"


def test_glm_api_client_uses_normal_chat_mode_for_non_think_models() -> None:
    credentials = ProviderCredentialRecord(
        provider="glm-cn",
        kind="browser_session",
        cookie="chatglm_token=value",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chatglm/backend-api/assistant/stream"):
            seen["payload"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                text='data: {"conversation_id":"conv-1","parts":[{"content":[{"type":"text","text":"glm ok"}]}]}\n',
                headers={"content-type": "text/event-stream"},
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    client = GLMApiClient(
        credentials,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = client.chat_completion(message="hi", model="glm-4-plus")

    assert response == "glm ok"
    assert seen["payload"]["meta_data"]["chat_mode"] == "normal"


def test_glm_api_client_keeps_zero_chat_mode_for_think_models() -> None:
    credentials = ProviderCredentialRecord(
        provider="glm-cn",
        kind="browser_session",
        cookie="chatglm_token=value",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chatglm/backend-api/assistant/stream"):
            seen["payload"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                text='data: {"conversation_id":"conv-1","parts":[{"content":[{"type":"text","text":"glm think ok"}]}]}\n',
                headers={"content-type": "text/event-stream"},
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    client = GLMApiClient(
        credentials,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = client.chat_completion(message="hi", model="glm-4-think")

    assert response == "glm think ok"
    assert seen["payload"]["meta_data"]["chat_mode"] == "zero"


def test_glm_intl_request_signature_matches_browser_capture() -> None:
    signature = _compute_glm_intl_request_signature(
        sorted_payload=(
            "requestId,fcb0273e-78c3-430d-a708-c9b46f9f6634,"
            "timestamp,1776523109947,user_id,a2cd2187-19d5-4fbc-978c-4a9d3e9485d7"
        ),
        signature_prompt="请只回复：REQ_CAPTURE",
        timestamp_ms="1776523109947",
    )

    assert signature == "1356c56609d3ae248538cf251bddf10b82abbd1a9e138ef8256ab45d46d95f8a"


def test_glm_intl_api_client_streams_answer_text_from_v2_sse() -> None:
    credentials = ProviderCredentialRecord(
        provider="glm-intl",
        kind="browser_session",
        cookie="token=cookie-token",
        headers={},
        user_agent="Mozilla/5.0",
        metadata={},
        status="valid",
    )
    seen: dict[str, object] = {"requests": []}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["requests"].append((request.method, str(request.url)))
        if request.url.path == "/api/v1/auths/":
            return httpx.Response(
                200,
                json={
                    "id": "user-123",
                    "name": "loki",
                    "token": "bearer-123",
                },
            )
        if request.url.path == "/api/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "GLM-5.1"},
                        {"id": "GLM-5-Turbo"},
                    ]
                },
            )
        if request.url.path == "/api/v1/chats/new":
            seen["new_chat_payload"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(200, json={"id": "chat-123"})
        if request.url.path == "/api/v2/chat/completions":
            seen["completion_headers"] = dict(request.headers)
            seen["completion_payload"] = json.loads(request.content.decode("utf-8"))
            seen["completion_query"] = dict(request.url.params)
            return httpx.Response(
                200,
                text=(
                    'data: {"type":"chat:completion","data":{"delta_content":"思","phase":"thinking"}}\n\n'
                    'data: {"type":"chat:completion","data":{"delta_content":"考","phase":"thinking"}}\n\n'
                    'data: {"type":"chat:completion","data":{"delta_content":"答","phase":"answer"}}\n\n'
                    'data: {"type":"chat:completion","data":{"delta_content":"案","phase":"answer"}}\n\n'
                    'data: {"type":"chat:completion","data":{"phase":"done","done":true}}\n\n'
                    "data: [DONE]\n\n"
                ),
                headers={"content-type": "text/event-stream"},
            )
        if request.url.path == "/":
            return httpx.Response(
                200,
                text=(
                    '<script src="https://z-cdn.chatglm.cn/z-ai/frontend/prod-fe-1.1.12/'
                    '_app/immutable/entry/start.js"></script>'
                ),
                headers={"content-type": "text/html"},
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    client = GLMIntlApiClient(
        credentials,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert list(
        client.iter_marked_chat_completion_text(
            message="hello glm-intl",
            model="glm-4-think",
        )
    ) == ["<think>", "思", "考", "</think>", "答", "案"]

    text = client.chat_completion(message="hello glm-intl", model="glm-4-plus")

    assert text == "答案"
    assert seen["new_chat_payload"]["chat"]["models"] == ["GLM-5.1"]
    assert seen["completion_payload"]["model"] == "GLM-5-Turbo"
    assert seen["completion_query"]["signature_timestamp"] == seen["completion_query"]["timestamp"]
    assert seen["completion_headers"]["x-fe-version"] == "prod-fe-1.1.12"
    assert seen["completion_headers"]["x-signature"]


def test_glm_adapter_streams_using_client_when_available() -> None:
    credentials = ProviderCredentialRecord(
        provider="glm-cn",
        kind="browser_session",
        cookie="chatglm_token=value",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    class FakeClient:
        def iter_chat_completion_text(self, *, message: str, model: str):
            assert model == "glm-4-plus"
            assert message == "User: hello"
            yield "你"
            yield "好"

    adapter = GLMWebAdapter(client_factory=lambda _: FakeClient())

    assert list(
        adapter.stream_chat(
            NormalizedChatRequest(
                model="algae/glm-cn/glm-4-plus",
                messages=[{"role": "user", "content": "hello"}],
            ),
            credentials,
        )
        or []
    ) == ["你", "好"]


def test_qwen_web_client_retries_with_fresh_chat_id_when_sse_has_no_visible_text(tmp_path) -> None:
    credentials = ProviderCredentialRecord(
        provider="qwen-intl",
        kind="browser_session",
        cookie="token=1",
        headers={},
        user_agent="ua",
        metadata={"session_token": "token-1"},
        status="valid",
    )
    calls: list[tuple[str, str]] = []
    issued_chat_ids = iter(["chat-stale", "chat-fresh"])

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        if request.url.path.endswith("/api/v2/chat/completions"):
            chat_id = dict(request.url.params).get("chat_id", "")
            if chat_id == "chat-stale":
                return httpx.Response(
                    200,
                    text='data: {"response.created":{"chat_id":"chat-stale"}}\n\n',
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(
                200,
                text='data: {"choices":[{"delta":{"content":"hello qwen retry"}}]}\n\n',
                headers={"content-type": "text/event-stream"},
            )
        if request.url.path.endswith("/api/v2/chats/new"):
            return httpx.Response(200, json={"data": {"id": next(issued_chat_ids)}})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")
    client = QwenApiClient(
        credentials,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        state_dir=tmp_path,
    )

    response = client.chat_completion(message="hi", model="qwen3.6-plus")

    assert response.content == "hello qwen retry"
    assert any(url.endswith("/api/v2/chats/new") for _, url in calls)
    assert any("chat_id=chat-fresh" in url for _, url in calls)
