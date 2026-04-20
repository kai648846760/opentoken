import threading

from openclaw_algae.gateway.normalized import NormalizedChatRequest
from openclaw_algae.models.provider_credentials import ProviderCredentialRecord
from openclaw_algae.providers.browser import (
    BrowserChatAdapter,
    BrowserProviderClient,
    _run_browser_stream,
)


class FakeBrowserClient(BrowserProviderClient):
    def __init__(self) -> None:
        self.seen: dict[str, object] = {}

    def chat_completion(self, *, message: str, model: str) -> str:
        self.seen["message"] = message
        self.seen["model"] = model
        return "browser answer"


def test_browser_chat_adapter_uses_prompt_history() -> None:
    credentials = ProviderCredentialRecord(
        provider="qwen-intl",
        kind="browser_session",
        cookie="session=value",
        headers={},
        user_agent="ua",
        metadata={"session_token": "token"},
        status="valid",
    )
    client = FakeBrowserClient()
    adapter = BrowserChatAdapter(
        provider_name="Qwen International",
        login_hint="openclaw-algae login qwen international",
        client_factory=lambda _: client,
    )

    response = adapter.chat(
        NormalizedChatRequest(
            model="algae/qwen-intl/qwen3.5-plus",
            messages=[
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "hello"},
            ],
        ),
        credentials,
    )

    assert response.content == "browser answer"
    assert client.seen["model"] == "qwen3.5-plus"
    assert client.seen["message"] == "System: be concise\n\nUser: hello"


def test_browser_chat_adapter_requires_credentials() -> None:
    adapter = BrowserChatAdapter(
        provider_name="Gemini",
        login_hint="openclaw-algae login gemini",
        client_factory=lambda _: FakeBrowserClient(),
    )

    try:
        adapter.chat(
            NormalizedChatRequest(
                model="algae/gemini/gemini-pro",
                messages=[{"role": "user", "content": "hello"}],
            ),
            None,
        )
    except RuntimeError as exc:
        assert "login gemini" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


def test_browser_chat_adapter_maps_tagged_tool_response_to_openai_tool_calls() -> None:
    credentials = ProviderCredentialRecord(
        provider="gemini",
        kind="browser_session",
        cookie="session=value",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    class ToolBrowserClient(FakeBrowserClient):
        def chat_completion(self, *, message: str, model: str) -> str:
            self.seen["message"] = message
            self.seen["model"] = model
            return '<tool_calls>[{"name":"read_file","arguments":{"path":"/tmp/demo.txt"}}]</tool_calls>'

    client = ToolBrowserClient()
    adapter = BrowserChatAdapter(
        provider_name="Gemini",
        login_hint="openclaw-algae login gemini",
        client_factory=lambda _: client,
    )

    response = adapter.chat(
        NormalizedChatRequest(
            model="algae/gemini/gemini-pro",
            messages=[{"role": "user", "content": "read file"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
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
    assert response.tool_calls[0]["function"]["name"] == "read_file"
    assert "<tool_calls>" in client.seen["message"]
    assert "strict tagged tool protocol" in client.seen["message"]


def test_browser_chat_adapter_repairs_malformed_tagged_output_before_returning_tool_call() -> None:
    credentials = ProviderCredentialRecord(
        provider="gemini",
        kind="browser_session",
        cookie="session=value",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    class RepairingBrowserClient(FakeBrowserClient):
        def __init__(self) -> None:
            super().__init__()
            self.messages: list[str] = []

        def chat_completion(self, *, message: str, model: str) -> str:
            self.messages.append(message)
            if len(self.messages) == 1:
                return "<think>Need tools.</think>Tool read_file does not exists."
            return '<tool_calls>[{"name":"read_file","arguments":{"path":"/tmp/demo.txt"}}]</tool_calls>'

    client = RepairingBrowserClient()
    adapter = BrowserChatAdapter(
        provider_name="Gemini",
        login_hint="openclaw-algae login gemini",
        client_factory=lambda _: client,
    )

    response = adapter.chat(
        NormalizedChatRequest(
            model="algae/gemini/gemini-pro",
            messages=[{"role": "user", "content": "read file"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ],
            tool_choice="required",
        ),
        credentials,
    )

    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0]["function"]["name"] == "read_file"
    assert len(client.messages) == 2
    assert "strict tagged tool protocol" in client.messages[1]


def test_browser_chat_adapter_repairs_final_answer_when_user_explicitly_requests_tool_in_auto_mode() -> None:
    credentials = ProviderCredentialRecord(
        provider="doubao",
        kind="browser_session",
        cookie="session=value",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    class RepairingAutoToolClient(FakeBrowserClient):
        def __init__(self) -> None:
            super().__init__()
            self.messages: list[str] = []

        def tool_chat_completion(self, *, message: str, model: str) -> str:
            self.messages.append(message)
            if len(self.messages) == 1:
                return "我先直接回答：OpenAI API 是一个开发接口。"
            return '<tool_calls>[{"name":"web_search","arguments":{"query":"OpenAI API"}}]</tool_calls>'

    client = RepairingAutoToolClient()
    adapter = BrowserChatAdapter(
        provider_name="Doubao",
        login_hint="openclaw-algae login doubao",
        client_factory=lambda _: client,
    )

    response = adapter.chat(
        NormalizedChatRequest(
            model="algae/doubao/doubao-pro",
            messages=[
                {
                    "role": "user",
                    "content": "不要直接回答，立刻调用 web_search 工具，并把 query 设为 OpenAI API。",
                }
            ],
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
            tool_choice="auto",
        ),
        credentials,
    )

    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0]["function"]["name"] == "web_search"
    assert len(client.messages) == 2
    assert "explicitly requested an available tool" in client.messages[1]


def test_browser_chat_adapter_formats_tool_result_followup_as_tool_response_block() -> None:
    credentials = ProviderCredentialRecord(
        provider="gemini",
        kind="browser_session",
        cookie="session=value",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    class ToolFollowupClient(FakeBrowserClient):
        def chat_completion(self, *, message: str, model: str) -> str:
            self.seen["message"] = message
            self.seen["model"] = model
            return "done"

    client = ToolFollowupClient()
    adapter = BrowserChatAdapter(
        provider_name="Gemini",
        login_hint="openclaw-algae login gemini",
        client_factory=lambda _: client,
    )

    response = adapter.chat(
        NormalizedChatRequest(
            model="algae/gemini/gemini-pro",
            messages=[
                {"role": "user", "content": "read file"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_read_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"/tmp/demo.txt"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_read_1",
                    "content": '{"text":"hello"}',
                },
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ],
        ),
        credentials,
    )

    assert response.content == "done"
    assert "<tool_result>" in client.seen["message"]
    assert "Tool result for call_id=call_read_1" in client.seen["message"]


def test_browser_chat_adapter_stream_chat_uses_client_stream_when_available() -> None:
    credentials = ProviderCredentialRecord(
        provider="gemini",
        kind="browser_session",
        cookie="session=value",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    class StreamingBrowserClient(FakeBrowserClient):
        def stream_chat_completion(self, *, message: str, model: str):
            self.seen["message"] = message
            self.seen["model"] = model
            yield "hello"
            yield " world"

    client = StreamingBrowserClient()
    adapter = BrowserChatAdapter(
        provider_name="Gemini",
        login_hint="openclaw-algae login gemini",
        client_factory=lambda _: client,
    )

    chunks = list(
        adapter.stream_chat(
            NormalizedChatRequest(
                model="algae/gemini/gemini-pro",
                messages=[
                    {"role": "system", "content": "be concise"},
                    {"role": "user", "content": "hello"},
                ],
            ),
            credentials,
        )
        or []
    )

    assert chunks == ["hello", " world"]
    assert client.seen["model"] == "gemini-pro"
    assert client.seen["message"] == "System: be concise\n\nUser: hello"


def test_browser_chat_adapter_stream_chat_falls_back_to_non_stream_chat_when_stream_empty() -> None:
    credentials = ProviderCredentialRecord(
        provider="doubao",
        kind="browser_session",
        cookie="session=value",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    class EmptyStreamingClient(FakeBrowserClient):
        def stream_chat_completion(self, *, message: str, model: str):
            if False:
                yield ""

        def chat_completion(self, *, message: str, model: str) -> str:
            self.seen["fallback_message"] = message
            self.seen["fallback_model"] = model
            return "hello world"

    client = EmptyStreamingClient()
    adapter = BrowserChatAdapter(
        provider_name="Doubao",
        login_hint="openclaw-algae login doubao",
        client_factory=lambda _: client,
    )

    chunks = list(
        adapter.stream_chat(
            NormalizedChatRequest(
                model="algae/doubao/doubao-pro",
                messages=[{"role": "user", "content": "hello"}],
            ),
            credentials,
        )
        or []
    )

    assert "".join(chunks) == "hello world"
    assert client.seen["fallback_model"] == "doubao-pro"


def test_browser_chat_adapter_stream_chat_can_disable_non_stream_fallback_on_failure() -> None:
    credentials = ProviderCredentialRecord(
        provider="doubao",
        kind="browser_session",
        cookie="session=value",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    class FailingStreamingClient(FakeBrowserClient):
        def stream_chat_completion(self, *, message: str, model: str):
            raise RuntimeError("real stream failed")
            yield ""

        def chat_completion(self, *, message: str, model: str) -> str:
            self.seen["fallback_called"] = True
            return "should-not-be-used"

    client = FailingStreamingClient()
    adapter = BrowserChatAdapter(
        provider_name="Doubao",
        login_hint="openclaw-algae login doubao",
        client_factory=lambda _: client,
        fallback_to_non_stream_chat_on_stream_failure=False,
    )

    try:
        list(
            adapter.stream_chat(
                NormalizedChatRequest(
                    model="algae/doubao/doubao-pro",
                    messages=[{"role": "user", "content": "hello"}],
                ),
                credentials,
            )
            or []
        )
    except RuntimeError as exc:
        assert "real stream failed" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")

    assert "fallback_called" not in client.seen


def test_run_browser_stream_waits_for_consumer_before_requesting_next_piece() -> None:
    class PullAwareIterator:
        def __init__(self) -> None:
            self._sent_first = False
            self.second_requested = threading.Event()
            self.closed = threading.Event()

        def __iter__(self):
            return self

        def __next__(self) -> str:
            if not self._sent_first:
                self._sent_first = True
                return "hello"
            self.second_requested.set()
            self.closed.wait(timeout=1.0)
            raise StopIteration

        def close(self) -> None:
            self.closed.set()

    iterator = PullAwareIterator()
    stream = _run_browser_stream(
        provider_name="Browser Pull Test",
        invoke=lambda: iterator,
        timeout_seconds=1.0,
    )

    assert next(stream) == "hello"
    assert not iterator.second_requested.wait(0.2)

    stream.close()


def test_run_browser_stream_closes_underlying_iterator_when_consumer_stops_early() -> None:
    class CloseAwareIterator:
        def __init__(self) -> None:
            self._sent_first = False
            self.closed = threading.Event()

        def __iter__(self):
            return self

        def __next__(self) -> str:
            if not self._sent_first:
                self._sent_first = True
                return "hello"
            self.closed.wait(timeout=1.0)
            raise StopIteration

        def close(self) -> None:
            self.closed.set()

    iterator = CloseAwareIterator()
    stream = _run_browser_stream(
        provider_name="Browser Close Test",
        invoke=lambda: iterator,
        timeout_seconds=1.0,
    )

    assert next(stream) == "hello"
    stream.close()

    assert iterator.closed.wait(0.5)


def test_run_browser_stream_timeout_does_not_poison_later_streams() -> None:
    unblock = threading.Event()

    class StuckIterator:
        def __iter__(self):
            return self

        def __next__(self) -> str:
            unblock.wait(timeout=5.0)
            raise StopIteration

        def close(self) -> None:
            return None

    timed_out_stream = _run_browser_stream(
        provider_name="Browser Poison Test",
        invoke=lambda: StuckIterator(),
        timeout_seconds=0.1,
    )

    try:
        next(timed_out_stream)
    except RuntimeError as exc:
        assert "timed out" in str(exc)
    else:
        raise AssertionError("Expected stream timeout")

    class GoodIterator:
        def __iter__(self):
            return self

        def __next__(self) -> str:
            return "ok"

        def close(self) -> None:
            return None

    recovered_stream = _run_browser_stream(
        provider_name="Browser Poison Test",
        invoke=lambda: GoodIterator(),
        timeout_seconds=0.5,
    )
    try:
        assert next(recovered_stream) == "ok"
    finally:
        unblock.set()
        recovered_stream.close()


def test_browser_chat_adapter_prefers_tool_chat_completion_when_available() -> None:
    credentials = ProviderCredentialRecord(
        provider="doubao",
        kind="browser_session",
        cookie="session=value",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    class ToolCapableClient(FakeBrowserClient):
        def tool_chat_completion(self, *, message: str, model: str) -> str:
            self.seen["tool_message"] = message
            self.seen["tool_model"] = model
            return '<tool_call id="call_read_1" name="read_file">{"path":"/tmp/demo.txt"}</tool_call>'

        def chat_completion(self, *, message: str, model: str) -> str:
            raise AssertionError("tool requests should prefer tool_chat_completion")

    client = ToolCapableClient()
    adapter = BrowserChatAdapter(
        provider_name="Doubao",
        login_hint="openclaw-algae login doubao",
        client_factory=lambda _: client,
    )

    response = adapter.chat(
        NormalizedChatRequest(
            model="algae/doubao/doubao-pro",
            messages=[{"role": "user", "content": "read file"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ],
            tool_choice="required",
        ),
        credentials,
    )

    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0]["function"]["name"] == "read_file"
    assert "<tool_calls>" in client.seen["tool_message"]
    assert "read_file" in client.seen["tool_message"]


def test_browser_chat_adapter_runs_tool_chat_completion_on_browser_worker(
    monkeypatch,
) -> None:
    credentials = ProviderCredentialRecord(
        provider="doubao",
        kind="browser_session",
        cookie="session=value",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    calls: list[str] = []

    class ToolCapableClient(FakeBrowserClient):
        def tool_chat_completion(self, *, message: str, model: str) -> str:
            self.seen["tool_message"] = message
            self.seen["tool_model"] = model
            return '<tool_call id="call_read_1" name="read_file">{"path":"/tmp/demo.txt"}</tool_call>'

    def fake_run_browser_completion(*, provider_name: str, invoke, timeout_seconds: float = 300.0):
        calls.append(provider_name)
        return str(invoke())

    monkeypatch.setattr(
        "openclaw_algae.providers.browser._run_browser_completion",
        fake_run_browser_completion,
    )

    client = ToolCapableClient()
    adapter = BrowserChatAdapter(
        provider_name="Doubao",
        login_hint="openclaw-algae login doubao",
        client_factory=lambda _: client,
    )

    response = adapter.chat(
        NormalizedChatRequest(
            model="algae/doubao/doubao-pro",
            messages=[{"role": "user", "content": "read file"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ],
            tool_choice="required",
        ),
        credentials,
    )

    assert response.finish_reason == "tool_calls"
    assert calls == ["Doubao"]
