import base64
import json
import threading
import time

import httpx
import openclaw_algae.providers.camoufox_clients as camoufox_module
from openclaw_algae.models.provider_credentials import ProviderCredentialRecord
from openclaw_algae.providers.base import ProviderRateLimitError
from openclaw_algae.providers.camoufox_clients import (
    CamoufoxProviderClient,
    _DOUBAO_URL,
    _GLMIntlDomStreamProjector,
    _advance_text_stream_state,
    _dom_send_and_wait_glm_intl,
    _dom_send_and_wait_doubao,
    _dom_send_and_wait_glm_cn,
    _extract_glm_error_detail,
    _fetch_doubao_browser_completion,
    _fetch_qwen_cn_browser_completion,
    _parse_qwen_cn_response_text,
    _extract_doubao_conversation_id,
    _is_doubao_chat_completion_response,
    _extract_doubao_stream_error,
    _is_reformatted_snapshot_duplicate,
    _dom_send_and_wait_qwen_cn,
    _extract_qwen_cn_candidate_text,
    _get_or_create_browser_session,
    _resolve_qwen_cn_request_auth,
    _send_qwen_intl_dom_message,
    _stream_qwen_intl_dom_completion,
    _stream_qwen_intl_browser_completion,
    _stream_doubao_browser_completion,
    _stream_doubao_dom_completion,
    _stream_glm_cn_browser_completion,
    _stream_glm_intl_dom_completion,
    _ProviderBrowserSession,
    _PROVIDER_GLOBAL_SESSIONS,
)


def _make_test_jwt(payload: dict[str, object]) -> str:
    def _b64(data: dict[str, object]) -> str:
        raw = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{_b64({'alg': 'HS256', 'typ': 'JWT'})}.{_b64(payload)}.sig"


def test_reformatted_snapshot_duplicate_ignores_markdown_emphasis() -> None:
    assert _is_reformatted_snapshot_duplicate(
        "流式输出就是内容一边生成一边输出，不用等全部完成才一次性显示，用户能实时看到结果。",
        "流式输出就是**内容一边生成一边输出**，不用等全部完成才一次性显示，用户能实时看到结果。",
    )


def test_advance_text_stream_state_emits_only_new_suffix_for_reformatted_snapshot() -> None:
    current = "这是今天开源圈的 5 条趣闻，速览如下： Claude Code 源码泄露引发 “光速复刻” An"
    candidate = (
        "这是今天开源圈的 5 条趣闻，速览如下： Claude Code 源码泄露引发 “光速复刻”Anthropic "
        "不慎泄露 51.2 万行 Claude Code 源码微博。25 岁留学生在女友提醒下，仅用数小时便重写架构，推出 ClawCode"
    )

    suffix, updated = _advance_text_stream_state(current, candidate)

    assert suffix == "thropic 不慎泄露 51.2 万行 Claude Code 源码微博。25 岁留学生在女友提醒下，仅用数小时便重写架构，推出 ClawCode"
    assert updated == candidate


def test_advance_text_stream_state_swallows_markdown_reformat_snapshot_without_duplication() -> None:
    current = "速览如下：Claude Code 源码泄露。"
    candidate = "速览如下：\n\n1.  **Claude Code** 源码泄露。"

    suffix, updated = _advance_text_stream_state(current, candidate)

    assert suffix == ""
    assert updated == candidate


def test_qwen_cn_request_auth_prefers_live_cookie_values_over_stale_saved_metadata() -> None:
    credentials = ProviderCredentialRecord(
        provider="qwen-cn",
        kind="browser_session",
        cookie="XSRF-TOKEN=old-token; b-user-id=old-user",
        headers={},
        user_agent="ua",
        metadata={
            "xsrf_token": "stale-token",
            "ut": "stale-user",
            "device_id": "stale-device",
        },
        status="valid",
    )

    auth = _resolve_qwen_cn_request_auth(
        credentials,
        {
            "XSRF-TOKEN": "live-token",
            "b-user-id": "live-user",
        },
    )

    assert auth["xsrf_token"] == "live-token"
    assert auth["ut"] == "live-user"
    assert auth["device_id"] == "stale-device"


def test_qwen_cn_request_auth_falls_back_to_saved_metadata_when_cookie_values_missing() -> None:
    credentials = ProviderCredentialRecord(
        provider="qwen-cn",
        kind="browser_session",
        cookie="tongyi_sso_ticket=ticket",
        headers={},
        user_agent="ua",
        metadata={
            "xsrf_token": "saved-token",
            "ut": "saved-user",
        },
        status="valid",
    )

    auth = _resolve_qwen_cn_request_auth(credentials, {})

    assert auth["xsrf_token"] == "saved-token"
    assert auth["ut"] == "saved-user"
    assert auth["device_id"] == "saved-user"


def test_extract_qwen_cn_candidate_text_prefers_latest_markdown_reply() -> None:
    text = _extract_qwen_cn_candidate_text(
        "Reply with exactly: qwen-dom-scan",
        markdown_texts=["older", "qwen-dom-scan"],
        message_texts=["Reply with exactly: qwen-dom-scanqwen-dom-scan8篇来源"],
    )

    assert text == "qwen-dom-scan"


def test_extract_qwen_cn_candidate_text_strips_prompt_echo_from_message_fallback() -> None:
    text = _extract_qwen_cn_candidate_text(
        "Reply with exactly: qwen-fallback",
        markdown_texts=[],
        message_texts=["Reply with exactly: qwen-fallback这是回答正文"],
    )

    assert text == "这是回答正文"


def test_extract_qwen_cn_candidate_text_ignores_pure_prompt_echo() -> None:
    prompt = "Reply with exactly: qwen-prompt-echo"

    text = _extract_qwen_cn_candidate_text(
        prompt,
        markdown_texts=[],
        message_texts=[prompt],
    )

    assert text == ""


def test_stream_qwen_intl_browser_completion_yields_incremental_chunks() -> None:
    class FakePage:
        def __init__(self) -> None:
            self.poll_count = 0
            self.deleted_stream_ids: list[str] = []

        def evaluate(self, script: str, arg: dict[str, object]):
            if "window[streamId] = state;" in script:
                return {"ok": True}
            if "const state = window[streamId];" in script:
                self.poll_count += 1
                if self.poll_count == 1:
                    return {
                        "lines": [
                            'data: {"response.created":{"chat_id":"chat-1"}}',
                            "",
                            'data: {"choices":[{"delta":{"content":"你","phase":"answer","status":"typing"}}]}',
                            "",
                        ],
                        "done": False,
                        "error": "",
                    }
                if self.poll_count == 2:
                    return {
                        "lines": [
                            'data: {"choices":[{"delta":{"content":"好","phase":"answer","status":"typing"}}]}',
                            "",
                        ],
                        "done": True,
                        "error": "",
                    }
                return {"lines": [], "done": True, "error": ""}
            if "delete window[streamId]" in script:
                self.deleted_stream_ids.append(str(arg["streamId"]))
                return True
            raise AssertionError(f"Unexpected script: {script[:80]}")

    pieces = list(
        _stream_qwen_intl_browser_completion(
            FakePage(),
            message="hello",
            model="qwen3.6-plus",
            poll_interval_seconds=0.0,
        )
    )

    assert pieces == ["你", "好"]


def test_stream_qwen_intl_dom_completion_yields_incremental_answer_deltas(monkeypatch) -> None:
    prepared: dict[str, object] = {}
    snapshots = iter(
        (
            {
                "answer_text": "",
                "assistant_text": "",
                "thinking_text": "",
                "is_streaming": True,
            },
            {
                "answer_text": "你",
                "assistant_text": "Thinking completed 你",
                "thinking_text": "Thinking completed",
                "is_streaming": True,
            },
            {
                "answer_text": "你好",
                "assistant_text": "Thinking completed 你好",
                "thinking_text": "Thinking completed",
                "is_streaming": False,
            },
        )
    )

    monkeypatch.setattr(
        camoufox_module,
        "_prepare_qwen_intl_dom_page",
        lambda page, *, model: prepared.setdefault("model", model),
    )
    monkeypatch.setattr(
        camoufox_module,
        "_send_qwen_intl_dom_message",
        lambda page, *, message: prepared.setdefault("message", message),
    )
    monkeypatch.setattr(
        camoufox_module,
        "_capture_qwen_intl_dom_stream_state",
        lambda page: next(snapshots),
    )

    pieces = list(
        _stream_qwen_intl_dom_completion(
            object(),
            message="hello",
            model="qwen3.6-plus",
            poll_interval_seconds=0.0,
            timeout_seconds=1.0,
        )
    )

    assert pieces == ["你", "好"]
    assert prepared == {"model": "qwen3.6-plus", "message": "hello"}


def test_glm_intl_dom_stream_projector_wraps_thinking_and_answer_incrementally() -> None:
    projector = _GLMIntlDomStreamProjector()

    assert projector.push_snapshot(thinking_text="分", answer_text="") == ["<think>", "分"]
    assert projector.push_snapshot(thinking_text="分析", answer_text="") == ["析"]
    assert projector.push_snapshot(thinking_text="分析", answer_text="你") == ["</think>", "你"]
    assert projector.push_snapshot(thinking_text="分析", answer_text="你好") == ["好"]
    assert projector.finish() == ""


def test_stream_glm_intl_dom_completion_waits_for_input_and_yields_incremental_pieces(monkeypatch) -> None:
    class FakePage:
        def evaluate(self, script: str):
            return 0

    page = FakePage()
    prepared: dict[str, bool] = {}
    waits: list[tuple[str, str, int]] = []
    sent: dict[str, object] = {}
    snapshots = iter(
        (
            {
                "assistant_count": 0,
                "answer_text": "",
                "thinking_text": "",
                "regenerate_visible": False,
            },
            {
                "assistant_count": 1,
                "answer_text": "",
                "thinking_text": "分",
                "regenerate_visible": False,
            },
            {
                "assistant_count": 1,
                "answer_text": "",
                "thinking_text": "分析",
                "regenerate_visible": True,
            },
            {
                "assistant_count": 1,
                "answer_text": "你",
                "thinking_text": "分析",
                "regenerate_visible": False,
            },
            {
                "assistant_count": 1,
                "answer_text": "你好",
                "thinking_text": "分析",
                "regenerate_visible": True,
            },
            {
                "assistant_count": 1,
                "answer_text": "你好",
                "thinking_text": "分析",
                "regenerate_visible": True,
            },
        )
    )

    monkeypatch.setattr(
        camoufox_module,
        "_prepare_glm_intl_dom_page",
        lambda page: prepared.setdefault("called", True),
    )
    monkeypatch.setattr(
        camoufox_module,
        "_wait_for_glm_intl_input_ready",
        lambda page, *, timeout_ms=120000: waits.append(("#chat-input", "visible", timeout_ms)),
    )
    monkeypatch.setattr(
        camoufox_module,
        "_send_glm_intl_dom_message",
        lambda page, *, message: sent.setdefault("message", message),
    )
    monkeypatch.setattr(
        camoufox_module,
        "_capture_glm_intl_dom_stream_state",
        lambda page: next(snapshots),
    )

    pieces = list(
        _stream_glm_intl_dom_completion(
            page,
            message="hello",
            poll_interval_seconds=0.0,
            timeout_seconds=1.0,
        )
    )

    assert prepared == {"called": True}
    assert waits == [("#chat-input", "visible", 120000)]
    assert sent == {"message": "hello"}
    assert pieces == ["<think>", "分", "析", "</think>", "你", "好"]


def test_dom_send_and_wait_glm_intl_strips_think_markup(monkeypatch) -> None:
    monkeypatch.setattr(
        camoufox_module,
        "_stream_glm_intl_dom_completion",
        lambda page, *, message: iter(["<think>", "分析", "</think>", "最终答案"]),
    )

    text = _dom_send_and_wait_glm_intl(object(), "hello")

    assert text == "最终答案"


def test_chat_glm_intl_prefers_api_completion_before_dom_fallback(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="glm-intl",
        kind="browser_session",
        cookie="token=1",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    call_order: list[str] = []

    monkeypatch.setattr(
        camoufox_module,
        "_chat_glm_intl_api_completion",
        lambda credentials, *, message, model: (
            call_order.append("api") or "glm-intl-api-ok"
        ),
    )
    monkeypatch.setattr(
        CamoufoxProviderClient,
        "_with_page",
        lambda self, *, start_url, cookie_domains, action: (_ for _ in ()).throw(
            AssertionError("DOM fallback should not run when API completion succeeds")
        ),
    )

    client = CamoufoxProviderClient("glm-intl", credentials)

    assert client._chat_glm_intl(message="hello", model="glm-4-plus") == "glm-intl-api-ok"
    assert call_order == ["api"]


def test_stream_glm_intl_prefers_api_stream_before_dom_fallback(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="glm-intl",
        kind="browser_session",
        cookie="token=1",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    call_order: list[str] = []

    monkeypatch.setattr(
        camoufox_module,
        "_stream_glm_intl_api_completion",
        lambda credentials, *, message, model: (
            call_order.append("api") or iter(("<think>", "分析", "</think>", "答案"))
        ),
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._get_or_create_browser_session",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("Browser session should not be created when API stream succeeds")
        ),
    )
    monkeypatch.setattr(
        camoufox_module,
        "_stream_glm_intl_dom_completion",
        lambda page, *, message: (
            call_order.append("dom")
            or (_ for _ in ()).throw(
                AssertionError("DOM fallback should not run when API stream succeeds")
            )
        ),
    )

    client = CamoufoxProviderClient("glm-intl", credentials)

    assert list(client._stream_glm_intl(message="hello", model="glm-4-plus")) == [
        "<think>",
        "分析",
        "</think>",
        "答案",
    ]
    assert call_order == ["api"]


def test_stream_glm_intl_api_completion_uses_fast_fail_http_timeout(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="glm-intl",
        kind="browser_session",
        cookie="token=1",
        headers={},
        user_agent="ua",
        status="valid",
    )
    captured: dict[str, object] = {}

    class FakeGLMIntlApiClient:
        def __init__(self, credentials, *, client=None) -> None:
            captured["client"] = client

        def iter_marked_chat_completion_text(self, *, message: str, model: str):
            yield "<think>"
            yield "分析"
            yield "</think>"
            yield "答案"

    monkeypatch.setattr(camoufox_module, "GLMIntlApiClient", FakeGLMIntlApiClient)

    pieces = list(
        camoufox_module._stream_glm_intl_api_completion(
            credentials,
            message="hello",
            model="glm-4-plus",
        )
    )

    assert pieces == ["<think>", "分析", "</think>", "答案"]
    timeout = captured["client"].timeout
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == 6.0
    assert timeout.read == 6.0


def test_stream_glm_intl_falls_back_to_dom_when_api_stream_startup_times_out(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="glm-intl",
        kind="browser_session",
        cookie="token=1",
        headers={},
        user_agent="ua",
        status="valid",
    )

    class FakePage:
        url = "https://chat.z.ai/"

    class FakeContext:
        def __init__(self) -> None:
            self.pages = [FakePage()]

        def new_page(self):
            return self.pages[0]

    class FakeSession:
        def __init__(self) -> None:
            self.context = FakeContext()
            self.page = self.context.pages[0]
            self.headless = True
            self.owner_thread = threading.current_thread()
            self.metadata = {}

    call_order: list[str] = []

    def slow_api_stream():
        time.sleep(0.05)
        yield "<think>"

    monkeypatch.setattr(
        camoufox_module,
        "_GLM_INTL_API_STREAM_STARTUP_TIMEOUT_SECONDS",
        0.01,
        raising=False,
    )
    monkeypatch.setattr(
        camoufox_module,
        "_stream_glm_intl_api_completion",
        lambda credentials, *, message, model: (
            call_order.append("api") or slow_api_stream()
        ),
    )
    monkeypatch.setattr(
        camoufox_module,
        "_get_or_create_browser_session",
        lambda **kwargs: FakeSession(),
    )
    monkeypatch.setattr(
        camoufox_module,
        "_page_is_closed",
        lambda page: False,
    )
    monkeypatch.setattr(
        CamoufoxProviderClient,
        "_inject_cookie_string",
        lambda self, context, domains: None,
    )
    monkeypatch.setattr(
        camoufox_module,
        "_stream_glm_intl_dom_completion",
        lambda page, *, message: (
            call_order.append("dom") or iter(("<think>", "分析", "</think>", "答案"))
        ),
    )

    client = CamoufoxProviderClient("glm-intl", credentials)

    assert list(client._stream_glm_intl(message="hello", model="glm-4-plus")) == [
        "<think>",
        "分析",
        "</think>",
        "答案",
    ]
    assert call_order == ["api", "dom"]


def test_stream_glm_intl_uses_short_bootstrap_navigation_timeout_for_dom_fallback(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="glm-intl",
        kind="browser_session",
        cookie="token=1",
        headers={},
        user_agent="ua",
        status="valid",
    )
    call_order: list[str] = []

    class FakePage:
        def __init__(self) -> None:
            self.url = "about:blank"
            self.goto_calls: list[tuple[str, str, int]] = []

        def goto(self, url: str, *, wait_until: str, timeout: int) -> None:
            self.goto_calls.append((url, wait_until, timeout))
            self.url = url

    class FakeContext:
        def __init__(self, page: FakePage) -> None:
            self.pages = [page]

        def new_page(self):
            return self.pages[0]

    class FakeSession:
        def __init__(self) -> None:
            self.page = FakePage()
            self.context = FakeContext(self.page)
            self.headless = True
            self.owner_thread = threading.current_thread()
            self.metadata = {}

    session = FakeSession()

    monkeypatch.setattr(
        camoufox_module,
        "_stream_glm_intl_api_completion",
        lambda credentials, *, message, model: (
            call_order.append("api")
            or (_ for _ in ()).throw(RuntimeError("api stream failed"))
        ),
    )
    monkeypatch.setattr(
        camoufox_module,
        "_get_or_create_browser_session",
        lambda **kwargs: session,
    )
    monkeypatch.setattr(
        camoufox_module,
        "_page_is_closed",
        lambda page: False,
    )
    monkeypatch.setattr(
        CamoufoxProviderClient,
        "_inject_cookie_string",
        lambda self, context, domains: None,
    )
    monkeypatch.setattr(
        camoufox_module,
        "_stream_glm_intl_dom_completion",
        lambda page, *, message: (
            call_order.append("dom") or iter(("<think>", "分析", "</think>", "答案"))
        ),
    )

    client = CamoufoxProviderClient("glm-intl", credentials)

    assert list(client._stream_glm_intl(message="hello", model="glm-4-plus")) == [
        "<think>",
        "分析",
        "</think>",
        "答案",
    ]
    assert call_order == ["api", "dom"]
    assert session.page.goto_calls == [("https://chat.z.ai/", "domcontentloaded", 10000)]


def test_stream_qwen_intl_prefers_api_stream_by_default(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="qwen-intl",
        kind="browser_session",
        cookie="session=1",
        headers={},
        user_agent="ua",
        status="valid",
    )

    call_order: list[str] = []

    monkeypatch.setattr(
        camoufox_module,
        "_stream_qwen_intl_api_completion",
        lambda credentials, *, message, model: (
            call_order.append("api") or iter(("真", "流"))
        ),
    )
    monkeypatch.setattr(
        camoufox_module,
        "_get_or_create_browser_session",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("Browser session should not be created when API stream succeeds")
        ),
    )
    monkeypatch.setattr(
        camoufox_module,
        "_stream_qwen_intl_browser_completion",
        lambda page, *, message, model: (
            call_order.append("browser") or iter(("真", "流"))
        ),
    )
    monkeypatch.setattr(
        camoufox_module,
        "_stream_qwen_intl_dom_completion",
        lambda page, *, message, model: (
            call_order.append("dom")
            or (_ for _ in ()).throw(
                AssertionError("DOM fallback should not run when browser fetch stream succeeds")
            )
        ),
    )

    client = CamoufoxProviderClient("qwen-intl", credentials)

    assert list(client._stream_qwen_intl(message="hello", model="qwen3.6-plus")) == ["真", "流"]
    assert call_order == ["api"]


def test_stream_qwen_intl_api_completion_uses_fast_fail_http_timeout(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="qwen-intl",
        kind="browser_session",
        cookie="session=1",
        headers={},
        user_agent="ua",
        status="valid",
    )
    captured: dict[str, object] = {}

    class FakeQwenApiClient:
        def __init__(self, credentials, *, base_url: str, client=None) -> None:
            captured["base_url"] = base_url
            captured["client"] = client

        def iter_chat_completion_text(self, *, message: str, model: str):
            yield "ok"

    monkeypatch.setattr(camoufox_module, "QwenApiClient", FakeQwenApiClient)

    pieces = list(
        camoufox_module._stream_qwen_intl_api_completion(
            credentials,
            message="hello",
            model="qwen3.6-plus",
        )
    )

    assert pieces == ["ok"]
    assert captured["base_url"] == "https://chat.qwen.ai"
    timeout = captured["client"].timeout
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == 6.0
    assert timeout.read == 12.0


def test_stream_qwen_intl_falls_back_to_browser_stream_when_api_stream_fails(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="qwen-intl",
        kind="browser_session",
        cookie="session=1",
        headers={},
        user_agent="ua",
        status="valid",
    )

    class FakePage:
        url = "https://chat.qwen.ai/"

    class FakeContext:
        def __init__(self) -> None:
            self.pages = [FakePage()]

        def new_page(self):
            return self.pages[0]

    class FakeSession:
        def __init__(self) -> None:
            self.context = FakeContext()
            self.page = self.context.pages[0]
            self.headless = True
            self.owner_thread = threading.current_thread()
            self.metadata = {}

    call_order: list[str] = []

    monkeypatch.setattr(
        camoufox_module,
        "_get_or_create_browser_session",
        lambda **kwargs: FakeSession(),
    )
    monkeypatch.setattr(
        camoufox_module,
        "_page_is_closed",
        lambda page: False,
    )
    monkeypatch.setattr(
        CamoufoxProviderClient,
        "_inject_cookie_string",
        lambda self, context, domains: None,
    )
    monkeypatch.setattr(
        camoufox_module,
        "_stream_qwen_intl_api_completion",
        lambda credentials, *, message, model: (
            call_order.append("api")
            or (_ for _ in ()).throw(RuntimeError("api stream failed"))
        ),
    )
    monkeypatch.setattr(
        camoufox_module,
        "_stream_qwen_intl_browser_completion",
        lambda page, *, message, model: (
            call_order.append("browser") or iter(("真", "流"))
        ),
    )
    monkeypatch.setattr(
        camoufox_module,
        "_stream_qwen_intl_dom_completion",
        lambda page, *, message, model: (
            call_order.append("dom")
            or (_ for _ in ()).throw(
                AssertionError("DOM fallback should not run when browser stream fallback succeeds")
            )
        ),
    )

    client = CamoufoxProviderClient("qwen-intl", credentials)

    assert list(client._stream_qwen_intl(message="hello", model="qwen3.6-plus")) == ["真", "流"]
    assert call_order == ["api", "browser"]


def test_stream_qwen_intl_falls_back_to_browser_stream_when_api_stream_startup_times_out(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="qwen-intl",
        kind="browser_session",
        cookie="session=1",
        headers={},
        user_agent="ua",
        status="valid",
    )

    class FakePage:
        url = "https://chat.qwen.ai/"

    class FakeContext:
        def __init__(self) -> None:
            self.pages = [FakePage()]

        def new_page(self):
            return self.pages[0]

    class FakeSession:
        def __init__(self) -> None:
            self.context = FakeContext()
            self.page = self.context.pages[0]
            self.headless = True
            self.owner_thread = threading.current_thread()
            self.metadata = {}

    call_order: list[str] = []

    def slow_api_stream():
        time.sleep(0.05)
        yield "慢"

    monkeypatch.setattr(
        camoufox_module,
        "_QWEN_INTL_API_STREAM_STARTUP_TIMEOUT_SECONDS",
        0.01,
        raising=False,
    )
    monkeypatch.setattr(
        camoufox_module,
        "_stream_qwen_intl_api_completion",
        lambda credentials, *, message, model: (
            call_order.append("api") or slow_api_stream()
        ),
    )
    monkeypatch.setattr(
        camoufox_module,
        "_get_or_create_browser_session",
        lambda **kwargs: FakeSession(),
    )
    monkeypatch.setattr(
        camoufox_module,
        "_page_is_closed",
        lambda page: False,
    )
    monkeypatch.setattr(
        CamoufoxProviderClient,
        "_inject_cookie_string",
        lambda self, context, domains: None,
    )
    monkeypatch.setattr(
        camoufox_module,
        "_stream_qwen_intl_browser_completion",
        lambda page, *, message, model: (
            call_order.append("browser") or iter(("真", "流"))
        ),
    )
    monkeypatch.setattr(
        camoufox_module,
        "_stream_qwen_intl_dom_completion",
        lambda page, *, message, model: (
            call_order.append("dom")
            or (_ for _ in ()).throw(
                AssertionError("DOM fallback should not run when browser stream fallback succeeds")
            )
        ),
    )

    client = CamoufoxProviderClient("qwen-intl", credentials)

    assert list(client._stream_qwen_intl(message="hello", model="qwen3.6-plus")) == ["真", "流"]
    assert call_order == ["api", "api", "browser"]


def test_stream_qwen_intl_retries_api_startup_timeout_before_browser_fallback(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="qwen-intl",
        kind="browser_session",
        cookie="session=1",
        headers={},
        user_agent="ua",
        status="valid",
    )

    call_order: list[str] = []
    api_attempts = {"count": 0}

    def api_stream():
        api_attempts["count"] += 1
        call_order.append(f"api-{api_attempts['count']}")
        if api_attempts["count"] == 1:
            raise RuntimeError("Qwen International API stream startup timed out after 4s")
        yield "真"
        yield "流"

    monkeypatch.setattr(
        camoufox_module,
        "_stream_qwen_intl_api_completion",
        lambda credentials, *, message, model: api_stream(),
    )
    monkeypatch.setattr(
        camoufox_module,
        "_get_or_create_browser_session",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("Browser fallback should not run when the second API attempt succeeds")
        ),
    )

    client = CamoufoxProviderClient("qwen-intl", credentials)

    assert list(client._stream_qwen_intl(message="hello", model="qwen3.6-plus")) == ["真", "流"]
    assert call_order == ["api-1", "api-2"]


def test_stream_qwen_intl_uses_commit_navigation_when_bootstrapping_browser_stream(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="qwen-intl",
        kind="browser_session",
        cookie="session=1",
        headers={},
        user_agent="ua",
        status="valid",
    )
    call_order: list[str] = []

    class FakePage:
        def __init__(self) -> None:
            self.url = "about:blank"
            self.goto_calls: list[tuple[str, str, int]] = []

        def goto(self, url: str, *, wait_until: str, timeout: int) -> None:
            self.goto_calls.append((url, wait_until, timeout))
            self.url = url

    class FakeContext:
        def __init__(self, page: FakePage) -> None:
            self.pages = [page]

        def new_page(self):
            return self.pages[0]

    class FakeSession:
        def __init__(self) -> None:
            self.page = FakePage()
            self.context = FakeContext(self.page)
            self.headless = True
            self.owner_thread = threading.current_thread()
            self.metadata = {}

    session = FakeSession()

    monkeypatch.setattr(
        camoufox_module,
        "_stream_qwen_intl_api_completion",
        lambda credentials, *, message, model: (
            call_order.append("api")
            or (_ for _ in ()).throw(RuntimeError("api stream failed"))
        ),
    )
    monkeypatch.setattr(
        camoufox_module,
        "_get_or_create_browser_session",
        lambda **kwargs: session,
    )
    monkeypatch.setattr(
        camoufox_module,
        "_page_is_closed",
        lambda page: False,
    )
    monkeypatch.setattr(
        CamoufoxProviderClient,
        "_inject_cookie_string",
        lambda self, context, domains: None,
    )
    monkeypatch.setattr(
        camoufox_module,
        "_stream_qwen_intl_browser_completion",
        lambda page, *, message, model: (
            call_order.append("browser") or iter(("真", "流"))
        ),
    )
    monkeypatch.setattr(
        camoufox_module,
        "_stream_qwen_intl_dom_completion",
        lambda page, *, message, model: (
            call_order.append("dom")
            or (_ for _ in ()).throw(
                AssertionError("DOM fallback should not run when browser stream succeeds")
            )
        ),
    )

    client = CamoufoxProviderClient("qwen-intl", credentials)

    assert list(client._stream_qwen_intl(message="hello", model="qwen3.6-plus")) == ["真", "流"]
    assert call_order == ["api", "browser"]
    assert session.page.goto_calls == [("https://chat.qwen.ai/", "commit", 10000)]


def test_stream_qwen_intl_falls_back_to_dom_when_api_and_browser_stream_fail(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="qwen-intl",
        kind="browser_session",
        cookie="session=1",
        headers={},
        user_agent="ua",
        status="valid",
    )

    class FakePage:
        url = "https://chat.qwen.ai/"

    class FakeContext:
        def __init__(self) -> None:
            self.pages = [FakePage()]

        def new_page(self):
            return self.pages[0]

    class FakeSession:
        def __init__(self) -> None:
            self.context = FakeContext()
            self.page = self.context.pages[0]
            self.headless = True
            self.owner_thread = threading.current_thread()
            self.metadata = {}

    call_order: list[str] = []

    monkeypatch.setattr(
        camoufox_module,
        "_get_or_create_browser_session",
        lambda **kwargs: FakeSession(),
    )
    monkeypatch.setattr(
        camoufox_module,
        "_page_is_closed",
        lambda page: False,
    )
    monkeypatch.setattr(
        CamoufoxProviderClient,
        "_inject_cookie_string",
        lambda self, context, domains: None,
    )
    monkeypatch.setattr(
        camoufox_module,
        "_stream_qwen_intl_api_completion",
        lambda credentials, *, message, model: (
            call_order.append("api")
            or (_ for _ in ()).throw(RuntimeError("api stream failed"))
        ),
    )
    monkeypatch.setattr(
        camoufox_module,
        "_stream_qwen_intl_browser_completion",
        lambda page, *, message, model: (
            call_order.append("browser")
            or (_ for _ in ()).throw(RuntimeError("browser fetch failed"))
        ),
    )
    monkeypatch.setattr(
        camoufox_module,
        "_stream_qwen_intl_dom_completion",
        lambda page, *, message, model: (
            call_order.append("dom") or iter(("退", "路"))
        ),
    )

    client = CamoufoxProviderClient("qwen-intl", credentials)

    assert list(client._stream_qwen_intl(message="hello", model="qwen3.6-plus")) == ["退", "路"]
    assert call_order == ["api", "browser", "dom"]


def test_stream_qwen_intl_does_not_fallback_after_browser_stream_emits_then_errors(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="qwen-intl",
        kind="browser_session",
        cookie="session=1",
        headers={},
        user_agent="ua",
        status="valid",
    )

    class FakePage:
        url = "https://chat.qwen.ai/"

    class FakeContext:
        def __init__(self) -> None:
            self.pages = [FakePage()]

        def new_page(self):
            return self.pages[0]

    class FakeSession:
        def __init__(self) -> None:
            self.context = FakeContext()
            self.page = self.context.pages[0]
            self.headless = True
            self.owner_thread = threading.current_thread()
            self.metadata = {}

    call_order: list[str] = []

    def broken_browser_stream():
        yield "真"
        raise RuntimeError("mid-stream failure")

    monkeypatch.setattr(
        camoufox_module,
        "_stream_qwen_intl_api_completion",
        lambda credentials, *, message, model: (
            call_order.append("api")
            or (_ for _ in ()).throw(RuntimeError("api stream failed"))
        ),
    )
    monkeypatch.setattr(
        camoufox_module,
        "_get_or_create_browser_session",
        lambda **kwargs: FakeSession(),
    )
    monkeypatch.setattr(
        camoufox_module,
        "_page_is_closed",
        lambda page: False,
    )
    monkeypatch.setattr(
        CamoufoxProviderClient,
        "_inject_cookie_string",
        lambda self, context, domains: None,
    )
    monkeypatch.setattr(
        camoufox_module,
        "_stream_qwen_intl_browser_completion",
        lambda page, *, message, model: (
            call_order.append("browser") or broken_browser_stream()
        ),
    )
    monkeypatch.setattr(
        camoufox_module,
        "_stream_qwen_intl_dom_completion",
        lambda page, *, message, model: (
            call_order.append("dom")
            or (_ for _ in ()).throw(
                AssertionError("DOM fallback should not run after browser stream already emitted data")
            )
        ),
    )

    client = CamoufoxProviderClient("qwen-intl", credentials)

    try:
        list(client._stream_qwen_intl(message="hello", model="qwen3.6-plus"))
    except RuntimeError as exc:
        assert "mid-stream failure" in str(exc)
    else:
        raise AssertionError("Expected mid-stream browser failure to be raised")
    assert call_order == ["api", "browser"]


def test_stream_qwen_intl_does_not_fallback_after_api_stream_emits_then_errors(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="qwen-intl",
        kind="browser_session",
        cookie="session=1",
        headers={},
        user_agent="ua",
        status="valid",
    )

    call_order: list[str] = []

    def broken_api_stream():
        yield "真"
        raise RuntimeError("api mid-stream failure")

    monkeypatch.setattr(
        camoufox_module,
        "_stream_qwen_intl_api_completion",
        lambda credentials, *, message, model: (
            call_order.append("api") or broken_api_stream()
        ),
    )
    monkeypatch.setattr(
        camoufox_module,
        "_get_or_create_browser_session",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("Browser session should not be created after API stream already emitted data")
        ),
    )

    client = CamoufoxProviderClient("qwen-intl", credentials)

    try:
        list(client._stream_qwen_intl(message="hello", model="qwen3.6-plus"))
    except RuntimeError as exc:
        assert "api mid-stream failure" in str(exc)
    else:
        raise AssertionError("Expected mid-stream API failure to be raised")
    assert call_order == ["api"]


def test_send_qwen_intl_dom_message_prefers_js_textarea_setter(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class FakeLocator:
        def __init__(self, selector: str) -> None:
            self.selector = selector
            self.first = self

        def wait_for(self, timeout=None) -> None:
            calls.append(("wait_for", self.selector))

        def click(self, timeout=None) -> None:
            calls.append(("click", self.selector))

        def fill(self, value: str) -> None:
            calls.append(("fill", value))

    class FakeKeyboard:
        def press(self, key: str) -> None:
            calls.append(("press", key))

        def type(self, value: str, delay=None) -> None:
            calls.append(("type", value))

    class FakePage:
        def __init__(self) -> None:
            self.keyboard = FakeKeyboard()

        def locator(self, selector: str) -> FakeLocator:
            return FakeLocator(selector)

        def wait_for_timeout(self, timeout: int) -> None:
            calls.append(("wait_timeout", timeout))

    monkeypatch.setattr(
        camoufox_module,
        "_set_visible_textarea_value",
        lambda page, selectors, value: True,
    )

    _send_qwen_intl_dom_message(FakePage(), message="hello")

    assert ("type", "hello") not in calls
    assert ("fill", "hello") not in calls
    assert ("click", "button.send-button") in calls


def test_parse_qwen_cn_response_text_prefers_latest_full_message() -> None:
    payload = """data: {"data":{"messages":[{"content":"QWEN"}]}}
data: {"data":{"messages":[{"content":"QWEN_CN"}]}}
data: {"data":{"messages":[{"content":"QWEN_CN_API_OK"}]}}
"""

    assert _parse_qwen_cn_response_text(payload) == "QWEN_CN_API_OK"


def test_parse_qwen_cn_response_text_falls_back_to_delta_chunks() -> None:
    payload = """data: {"choices":[{"delta":{"content":"QWEN_"}}]}
data: {"choices":[{"delta":{"content":"CN_"}}]}
data: {"choices":[{"delta":{"content":"STREAM_OK"}}]}
"""

    assert _parse_qwen_cn_response_text(payload) == "QWEN_CN_STREAM_OK"


def test_fetch_qwen_cn_browser_completion_builds_request_and_parses_sse() -> None:
    captured: dict[str, object] = {}

    class FakePage:
        def evaluate(self, script: str, arg: dict[str, object]) -> dict[str, object]:
            captured["script"] = script
            captured["arg"] = arg
            return {
                "ok": True,
                "status": 200,
                "text": """data: {"data":{"messages":[{"content":"QWEN_CN_BROWSER_OK"}]}}""",
            }

    text = _fetch_qwen_cn_browser_completion(
        FakePage(),
        message="Reply with exactly: QWEN_CN_BROWSER_OK",
        model="Qwen3.5-Plus",
        auth={
            "xsrf_token": "xsrf-live",
            "ut": "user-live",
            "device_id": "device-live",
        },
    )

    assert text == "QWEN_CN_BROWSER_OK"
    request = captured["arg"]
    assert request["baseUrl"] == "https://chat2.qianwen.com"
    assert request["model"] == "Qwen3.5-Plus"
    assert request["ut"] == "user-live"
    assert request["xsrfToken"] == "xsrf-live"
    assert request["deviceId"] == "device-live"
    assert request["message"] == "Reply with exactly: QWEN_CN_BROWSER_OK"


def test_chat_qwen_cn_prefers_browser_fetch_over_dom_fallback(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="qwen-cn",
        kind="browser_session",
        cookie="tongyi_sso_ticket=ticket",
        headers={},
        user_agent="ua",
        metadata={
            "xsrf_token": "saved-token",
            "ut": "saved-user",
        },
        status="valid",
    )
    client = CamoufoxProviderClient("qwen-cn", credentials)
    calls: dict[str, object] = {}

    class FakeContext:
        def cookies(self, urls: list[str]) -> list[dict[str, str]]:
            assert "https://www.qianwen.com" in urls
            return [
                {"name": "XSRF-TOKEN", "value": "live-token"},
                {"name": "b-user-id", "value": "live-user"},
            ]

    class FakePage:
        url = "https://www.qianwen.com/"

        def goto(self, url: str, wait_until: str, timeout: int) -> None:
            self.url = url

        def wait_for_timeout(self, timeout_ms: int) -> None:
            return None

    def fake_with_page(self, *, start_url: str, cookie_domains: tuple[str, ...], action):
        assert start_url == "https://www.qianwen.com/"
        assert cookie_domains == (".qianwen.com",)
        return action(FakeContext(), FakePage())

    def fake_fetch(page, *, message: str, model: str, auth: dict[str, str]) -> str:
        calls["message"] = message
        calls["model"] = model
        calls["auth"] = auth
        return "qwen api answer"

    def unexpected_dom(*args, **kwargs):
        raise AssertionError("DOM fallback should not be used when browser fetch succeeds")

    monkeypatch.setattr(CamoufoxProviderClient, "_with_page", fake_with_page)
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._fetch_qwen_cn_browser_completion",
        fake_fetch,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._dom_send_and_wait_qwen_cn",
        unexpected_dom,
    )

    text = client._chat_qwen_cn(
        message="Reply with exactly: QWEN_CN_BROWSER_OK",
        model="Qwen3.5-Plus",
    )

    assert text == "qwen api answer"
    assert calls["message"] == "Reply with exactly: QWEN_CN_BROWSER_OK"
    assert calls["model"] == "Qwen3.5-Plus"
    assert calls["auth"] == {
        "xsrf_token": "live-token",
        "ut": "live-user",
        "device_id": "live-user",
    }


def test_extract_doubao_stream_error_returns_stream_error_message() -> None:
    payload = """id: 0
event: STREAM_ERROR
data: {"error_code":710012000,"error_msg":"user invalid"}

event: gateway-error
data: {"code":"Internal","message":"ErrorX:code=710012000 stable=true message=[user invalid] [biz error]"}
"""

    assert _extract_doubao_stream_error(payload) == "STREAM_ERROR 710012000: user invalid"


def test_extract_doubao_stream_error_returns_samantha_error_message() -> None:
    payload = """data: {"event_data":"{\\"code\\":710022004,\\"message\\":\\"rate limited\\",\\"error_detail\\":{\\"message\\":\\"系统错误\\"}}","event_id":"0","event_type":2005}"""

    assert _extract_doubao_stream_error(payload) == "SAMANTHA_ERROR 710022004: rate limited: 系统错误"


def test_extract_glm_error_detail_returns_message_from_json_envelope() -> None:
    payload = '{"status":500,"message":"您已多次体验过对话, 请登录后继续使用","result":null}'

    assert _extract_glm_error_detail(payload) == "您已多次体验过对话, 请登录后继续使用"


def test_extract_doubao_conversation_id_returns_ack_conversation_id() -> None:
    payload = """id: 0
event: SSE_ACK
data: {"ack_client_meta":{"conversation_id":"38418988375786754","local_conversation_id":"local_123"}}
"""

    assert _extract_doubao_conversation_id(payload) == "38418988375786754"


def test_is_doubao_chat_completion_response_requires_matching_prompt() -> None:
    class FakeRequest:
        def __init__(self, method: str, post_data: str) -> None:
            self.method = method
            self.post_data = post_data

    class FakeResponse:
        def __init__(self, url: str, method: str, post_data: str) -> None:
            self.url = url
            self.request = FakeRequest(method, post_data)

    response = FakeResponse(
        "https://www.doubao.com/samantha/chat/completion",
        "POST",
        '{"messages":[{"content":"{\\"text\\":\\"Reply with exactly: algae-doubao-new\\"}"}]}',
    )
    stale = FakeResponse(
        "https://www.doubao.com/samantha/chat/completion",
        "POST",
        '{"messages":[{"content":"{\\"text\\":\\"Reply with exactly: algae-doubao-old\\"}"}]}',
    )

    assert _is_doubao_chat_completion_response(
        response, "Reply with exactly: algae-doubao-new"
    ) is True
    assert _is_doubao_chat_completion_response(
        stale, "Reply with exactly: algae-doubao-new"
    ) is False


def test_is_doubao_chat_completion_response_matches_multiline_prompt_with_json_escapes() -> None:
    class FakeRequest:
        def __init__(self, method: str, post_data: str) -> None:
            self.method = method
            self.post_data = post_data

    class FakeResponse:
        def __init__(self, post_data: str) -> None:
            self.url = "https://www.doubao.com/chat/completion?aid=497858"
            self.request = FakeRequest("POST", post_data)

    message = "System: line 1\n\nUser: line 2"
    post_data = (
        '{"messages":[{"content_block":[{"content":{"text_block":{"text":"System: line 1\\\\n\\\\nUser: line 2"}}}]}]}'
    )

    assert _is_doubao_chat_completion_response(FakeResponse(post_data), message) is True


def test_fetch_doubao_browser_completion_matches_reference_request_shape() -> None:
    captured: dict[str, object] = {}

    class FakePage:
        def evaluate(self, script: str, arg: dict[str, object]) -> dict[str, object]:
            captured["arg"] = arg
            return {
                "ok": True,
                "status": 200,
                "text": (
                    '{"event_type":2001,"event_data":"{\\"message\\":{\\"content\\":\\"{\\\\\\"text\\\\\\":\\\\\\"DOUBAO_OK\\\\\\"}\\",\\"content_type\\":2001}}"}\n'
                ),
            }

    credentials = ProviderCredentialRecord(
        provider="doubao",
        kind="browser_session",
        cookie="sessionid=session-1",
        headers={},
        user_agent="ua",
        metadata={"sessionid": "session-1"},
        status="valid",
    )
    client = CamoufoxProviderClient("doubao", credentials)
    session = _ProviderBrowserSession(
        manager=object(),
        context=object(),
        page=object(),
        headless=True,
        owner_thread=threading.current_thread(),
        metadata={
            "doubao_conversation_id": "0",
            "doubao_request_params": {"aid": "497858"},
        },
    )

    text = _fetch_doubao_browser_completion(
        FakePage(),
        session=session,
        client=client,
        message="Reply exactly: DOUBAO_OK",
        model="doubao-seed-2.0",
    )

    assert text == "DOUBAO_OK"
    request_body = captured["arg"]["body"]
    assert request_body["conversation_id"] == "0"
    assert request_body["completion_option"]["need_create_conversation"] is True
    assert "model" not in request_body


def test_chat_glm_cn_ignores_persisted_conversation_and_falls_back_to_dom_on_timeout(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="glm-cn",
        kind="browser_session",
        cookie="chatglm_token=header.payload.sig; chatglm_refresh_token=refresh",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )
    client = CamoufoxProviderClient("glm-cn", credentials)

    class FakeContext:
        pass

    class FakePage:
        url = "https://chatglm.cn/main/all"

        def goto(self, url: str, wait_until: str, timeout: int) -> None:
            self.url = url

        def wait_for_timeout(self, timeout_ms: int) -> None:
            return None

        def evaluate(self, script: str, arg=None):
            if "assistant/stream" in script:
                captured["body"] = arg["body"]
                captured["device_id"] = arg["deviceId"]
                return {
                    "ok": False,
                    "status": 408,
                    "error": "ChatGLM API request timed out after 120000ms",
                    "rawText": "",
                }
            raise AssertionError(f"Unexpected evaluate call: {script[:80]}")

    captured: dict[str, object] = {}
    saved_states: list[dict[str, str]] = []

    session = _ProviderBrowserSession(
        manager=object(),
        context=FakeContext(),
        page=FakePage(),
        headless=True,
        owner_thread=threading.current_thread(),
        metadata={},
    )

    monkeypatch.setitem(_PROVIDER_GLOBAL_SESSIONS, "glm-cn", session)
    monkeypatch.setattr(client, "_cookie_map", lambda context, urls: {
        "chatglm_token": "header.payload.sig",
        "chatglm_refresh_token": "refresh",
    })
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients.load_provider_session",
        lambda *args, **kwargs: {"device_id": "device-from-store", "conversation_id": "conv-from-store"},
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients.save_provider_session",
        lambda *args, **kwargs: saved_states.append(kwargs["state"]),
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._refresh_glm_access_token",
        lambda *args, **kwargs: "fresh-token",
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._dom_send_and_wait_glm_cn",
        lambda page, message: "glm dom fallback ok",
    )
    monkeypatch.setattr(CamoufoxProviderClient, "_with_page", lambda self, *, start_url, cookie_domains, action: action(FakeContext(), session.page))

    result = client._chat_glm_cn(message="Reply exactly: OK", model="glm-4-plus")

    assert result == "glm dom fallback ok"
    assert session.metadata["glm_cn_device_id"] == "device-from-store"
    assert captured["device_id"] == "device-from-store"
    assert captured["body"]["conversation_id"] == ""
    assert any(state.get("device_id") == "device-from-store" for state in saved_states)


def test_stream_glm_cn_prefers_browser_stream_and_ignores_persisted_conversation_id(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="glm-cn",
        kind="browser_session",
        cookie="chatglm_token=token; chatglm_refresh_token=refresh",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    seen: dict[str, object] = {}

    client = CamoufoxProviderClient("glm-cn", credentials)

    class FakePage:
        url = "https://chatglm.cn/main/all"

        def goto(self, url: str, wait_until: str, timeout: int) -> None:
            self.url = url

        def wait_for_timeout(self, timeout_ms: int) -> None:
            return None

    class FakeContext:
        pages = []

    session = _ProviderBrowserSession(
        manager=object(),
        context=FakeContext(),
        page=FakePage(),
        headless=True,
        owner_thread=threading.current_thread(),
        metadata={},
    )

    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._get_or_create_browser_session",
        lambda **kwargs: session,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._page_is_closed",
        lambda page: False,
    )
    monkeypatch.setattr(
        CamoufoxProviderClient,
        "_inject_cookie_string",
        lambda self, context, cookie_domains: None,
    )
    monkeypatch.setattr(
        CamoufoxProviderClient,
        "_cookie_map",
        lambda self, context, urls: {
            "chatglm_token": "token",
            "chatglm_refresh_token": "refresh",
            "chatglm_user_id": "user-1",
        },
    )
    monkeypatch.setattr(
        CamoufoxProviderClient,
        "_sync_live_glm_cn_credentials",
        lambda self, context, cookie_map: None,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._is_jwt_expired",
        lambda token, buffer_seconds=60: False,
    )
    monkeypatch.setattr(
        client,
        "_load_glm_cn_session_state",
        lambda: {"device_id": "device-from-store", "conversation_id": "conv-from-store"},
    )
    persisted: list[dict[str, str]] = []
    monkeypatch.setattr(
        client,
        "_persist_glm_cn_session_state",
        lambda *, device_id=None, conversation_id=None: persisted.append(
            {
                "device_id": str(device_id or ""),
                "conversation_id": str(conversation_id or ""),
            }
        ),
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._stream_glm_cn_browser_completion",
        lambda page, *, session, client, message, model, access_token, device_id: (
            seen.update(
                {
                    "message": message,
                    "model": model,
                    "device_id": device_id,
                    "conversation_id": str(session.metadata.get("glm_cn_conversation_id") or ""),
                    "access_token": access_token,
                }
            )
            or iter(["glm", " ok"])
        ),
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._dom_send_and_wait_glm_cn",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("browser stream should avoid DOM fallback in this test")
        ),
    )

    assert list(client._stream_glm_cn(message="hello", model="glm-5")) == ["glm", " ok"]
    assert seen["device_id"] == "device-from-store"
    assert seen["conversation_id"] == ""
    assert seen["access_token"] == "token"
    assert persisted[-1] == {"device_id": "device-from-store", "conversation_id": ""}


def test_stream_glm_cn_browser_completion_yields_incremental_chunks_and_persists_device_id() -> None:
    captured: dict[str, object] = {}

    class FakePage:
        def __init__(self) -> None:
            self.polls = [
                {
                    "lines": [
                        'data: {"conversation_id":"glm-conv-1","text":"glm"}',
                    ],
                    "done": False,
                    "error": "",
                },
                {
                    "lines": [
                        'data: {"conversation_id":"glm-conv-1","text":"glm ok"}',
                    ],
                    "done": True,
                    "error": "",
                },
            ]

        def evaluate(self, script: str, arg: dict[str, object]) -> dict[str, object]:
            if "window[streamId] = state" in script:
                captured["start_arg"] = arg
                return {"ok": True}
            if "state.lines.splice" in script:
                return self.polls.pop(0)
            if "delete window[streamId]" in script:
                captured["cleaned"] = True
                return {"ok": True}
            raise AssertionError(f"Unexpected evaluate script: {script[:120]}")

    class FakeClient:
        def _persist_glm_cn_session_state(
            self,
            *,
            device_id: str | None = None,
            conversation_id: str | None = None,
        ) -> None:
            captured["persisted_state"] = {
                "device_id": str(device_id or ""),
                "conversation_id": str(conversation_id or ""),
            }

    session = _ProviderBrowserSession(
        manager=object(),
        context=object(),
        page=object(),
        headless=True,
        owner_thread=threading.current_thread(),
        metadata={"glm_cn_device_id": "glm-device-1", "glm_cn_conversation_id": ""},
    )

    pieces = list(
        _stream_glm_cn_browser_completion(
            FakePage(),
            session=session,
            client=FakeClient(),
            message="Reply exactly: GLM_STREAM_OK",
            model="glm-5",
            access_token="live-token",
            device_id="glm-device-1",
            poll_interval_seconds=0.0,
        )
    )

    assert pieces == ["glm", " ok"]
    assert session.metadata["glm_cn_conversation_id"] == "glm-conv-1"
    assert captured["persisted_state"] == {
        "device_id": "glm-device-1",
        "conversation_id": "",
    }
    assert captured["start_arg"]["accessToken"] == "live-token"
    assert captured["start_arg"]["body"]["conversation_id"] == ""
    assert captured["cleaned"] is True


def test_chat_glm_cn_uses_normal_chat_mode_for_non_think_models(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="glm-cn",
        kind="browser_session",
        cookie="chatglm_token=header.payload.sig; chatglm_refresh_token=refresh",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )
    client = CamoufoxProviderClient("glm-cn", credentials)

    class FakeContext:
        pass

    class FakePage:
        url = "https://chatglm.cn/main/all"

        def goto(self, url: str, wait_until: str, timeout: int) -> None:
            self.url = url

        def wait_for_timeout(self, timeout_ms: int) -> None:
            return None

        def evaluate(self, script: str, arg=None):
            if "assistant/stream" in script:
                captured["body"] = arg["body"]
                return {
                    "ok": True,
                    "status": 200,
                    "rawText": 'data: {"conversation_id":"conv-1","parts":[{"content":[{"type":"text","text":"glm ok"}]}]}\n',
                }
            raise AssertionError(f"Unexpected evaluate call: {script[:80]}")

    captured: dict[str, object] = {}
    session = _ProviderBrowserSession(
        manager=object(),
        context=FakeContext(),
        page=FakePage(),
        headless=True,
        owner_thread=threading.current_thread(),
        metadata={},
    )

    monkeypatch.setitem(_PROVIDER_GLOBAL_SESSIONS, "glm-cn", session)
    monkeypatch.setattr(client, "_cookie_map", lambda context, urls: {
        "chatglm_token": "header.payload.sig",
        "chatglm_refresh_token": "refresh",
    })
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients.load_provider_session",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._refresh_glm_access_token",
        lambda *args, **kwargs: "fresh-token",
    )
    monkeypatch.setattr(
        CamoufoxProviderClient,
        "_with_page",
        lambda self, *, start_url, cookie_domains, action: action(FakeContext(), session.page),
    )

    result = client._chat_glm_cn(message="Reply exactly: OK", model="glm-4-plus")

    assert result == "glm ok"
    assert captured["body"]["meta_data"]["chat_mode"] == "normal"


def test_fetch_doubao_browser_completion_ignores_stale_conversation_id() -> None:
    captured: dict[str, object] = {}

    class FakePage:
        def evaluate(self, script: str, arg: dict[str, object]) -> dict[str, object]:
            captured["arg"] = arg
            return {
                "ok": True,
                "status": 200,
                "text": """event: CHUNK_DELTA
data: {"text":"doubao fresh conversation"}""",
            }

    class FakeSession:
        def __init__(self) -> None:
            self.metadata = {
                "doubao_request_params": {"aid": "497858"},
                "doubao_conversation_id": "stale-conversation-id",
            }

    class FakeClient:
        def _persist_doubao_conversation_id(self, conversation_id: str) -> None:
            captured["persisted_conversation_id"] = conversation_id

    session = FakeSession()

    text = _fetch_doubao_browser_completion(
        FakePage(),
        session=session,
        client=FakeClient(),
        message="Reply with exactly: DOUBAO_FRESH",
        model="doubao-seed-2.0",
    )

    assert text == "doubao fresh conversation"
    request = captured["arg"]["body"]
    assert request["conversation_id"] == "stale-conversation-id"
    assert request["completion_option"]["need_create_conversation"] is False


def test_stream_doubao_browser_completion_yields_incremental_chunks_and_persists_conversation_id() -> None:
    captured: dict[str, object] = {}

    class FakePage:
        def __init__(self) -> None:
            self.polls = [
                {
                    "lines": [
                        'event: SSE_ACK',
                        'data: {"ack_client_meta":{"conversation_id":"doubao-conv-2"}}',
                        "",
                        'event: CHUNK_DELTA',
                        'data: {"text":"dou"}',
                        "",
                    ],
                    "done": False,
                    "error": "",
                },
                {
                    "lines": [
                        'event: CHUNK_DELTA',
                        'data: {"text":"bao"}',
                        "",
                    ],
                    "done": True,
                    "error": "",
                },
            ]

        def evaluate(self, script: str, arg: dict[str, object]) -> dict[str, object]:
            if "window[streamId] = state" in script:
                captured["start_arg"] = arg
                return {"ok": True}
            if "state.lines.splice" in script:
                return self.polls.pop(0)
            raise AssertionError(f"Unexpected evaluate script: {script[:120]}")

    class FakeClient:
        def _persist_doubao_conversation_id(self, conversation_id: str) -> None:
            captured["persisted_conversation_id"] = conversation_id

    session = _ProviderBrowserSession(
        manager=object(),
        context=object(),
        page=object(),
        headless=True,
        owner_thread=threading.current_thread(),
        metadata={
            "doubao_conversation_id": "0",
            "doubao_request_params": {"aid": "497858"},
        },
    )

    pieces = list(
        _stream_doubao_browser_completion(
            FakePage(),
            session=session,
            client=FakeClient(),
            message="Reply exactly: DOUBAO_STREAM_OK",
            model="doubao-seed-2.0",
        )
    )

    assert pieces == ["dou", "bao"]
    assert session.metadata["doubao_conversation_id"] == "doubao-conv-2"
    assert captured["persisted_conversation_id"] == "doubao-conv-2"
    request_body = captured["start_arg"]["body"]
    assert request_body["conversation_id"] == "0"
    assert request_body["completion_option"]["need_create_conversation"] is True


def test_stream_doubao_browser_completion_coalesces_terminal_full_text_chunk() -> None:
    class FakePage:
        def __init__(self) -> None:
            self.polls = [
                {
                    "lines": [
                        'event: CHUNK_DELTA',
                        'data: {"text":"dou"}',
                        "",
                    ],
                    "done": False,
                    "error": "",
                },
                {
                    "lines": [
                        (
                            'event: STREAM_MSG_NOTIFY\n'
                            'data: {"content":{"content_block":[{"content":{"text_block":{"text":"doubao"}}}]}}'
                        ),
                        "",
                    ],
                    "done": True,
                    "error": "",
                },
            ]

        def evaluate(self, script: str, arg: dict[str, object]) -> dict[str, object]:
            if "window[streamId] = state" in script:
                return {"ok": True}
            if "state.lines.splice" in script:
                return self.polls.pop(0)
            raise AssertionError(f"Unexpected evaluate script: {script[:120]}")

    session = _ProviderBrowserSession(
        manager=object(),
        context=object(),
        page=object(),
        headless=True,
        owner_thread=threading.current_thread(),
        metadata={
            "doubao_conversation_id": "0",
            "doubao_request_params": {"aid": "497858"},
        },
    )

    pieces = list(
        _stream_doubao_browser_completion(
            FakePage(),
            session=session,
            client=object(),
            message="Reply exactly: DOUBAO_STREAM_OK",
            model="doubao-seed-2.0",
        )
    )

    assert pieces == ["dou", "bao"]


def test_stream_doubao_browser_completion_ignores_non_prefix_terminal_snapshot_after_deltas() -> None:
    class FakePage:
        def __init__(self) -> None:
            self.polls = [
                {
                    "lines": [
                        'event: CHUNK_DELTA',
                        'data: {"text":"1. first2. second"}',
                        "",
                    ],
                    "done": False,
                    "error": "",
                },
                {
                    "lines": [
                        'event: STREAM_MSG_NOTIFY',
                        'data: {"content":{"content_block":[{"content":{"text_block":{"text":"1. first\\n2. second"}}}]}}',
                        "",
                    ],
                    "done": True,
                    "error": "",
                },
            ]

        def evaluate(self, script: str, arg: dict[str, object]) -> dict[str, object]:
            if "window[streamId] = state" in script:
                return {"ok": True}
            if "state.lines.splice" in script:
                return self.polls.pop(0)
            raise AssertionError(f"Unexpected evaluate script: {script[:120]}")

    session = _ProviderBrowserSession(
        manager=object(),
        context=object(),
        page=object(),
        headless=True,
        owner_thread=threading.current_thread(),
        metadata={
            "doubao_conversation_id": "0",
            "doubao_request_params": {"aid": "497858"},
        },
    )

    pieces = list(
        _stream_doubao_browser_completion(
            FakePage(),
            session=session,
            client=object(),
            message="Reply exactly: DOUBAO_STREAM_OK",
            model="doubao-seed-2.0",
        )
    )

    assert pieces == ["1. first2. second"]


def test_stream_doubao_browser_completion_ignores_non_prefix_samantha_snapshot_after_deltas() -> None:
    compact_snapshot = json.dumps(
        {
            "event_type": 2001,
            "event_data": json.dumps(
                {
                    "is_finish": False,
                    "message": {
                        "content": json.dumps({"text": "1. first\n2. second"}),
                    },
                }
            ),
        },
        ensure_ascii=False,
    )

    class FakePage:
        def __init__(self) -> None:
            self.polls = [
                {
                    "lines": [
                        'event: CHUNK_DELTA',
                        'data: {"text":"1. first2. second"}',
                        "",
                    ],
                    "done": False,
                    "error": "",
                },
                {
                    "lines": [
                        f"data: {compact_snapshot}",
                    ],
                    "done": True,
                    "error": "",
                },
            ]

        def evaluate(self, script: str, arg: dict[str, object]) -> dict[str, object]:
            if "window[streamId] = state" in script:
                return {"ok": True}
            if "state.lines.splice" in script:
                return self.polls.pop(0)
            raise AssertionError(f"Unexpected evaluate script: {script[:120]}")

    session = _ProviderBrowserSession(
        manager=object(),
        context=object(),
        page=object(),
        headless=True,
        owner_thread=threading.current_thread(),
        metadata={
            "doubao_conversation_id": "0",
            "doubao_request_params": {"aid": "497858"},
        },
    )

    pieces = list(
        _stream_doubao_browser_completion(
            FakePage(),
            session=session,
            client=object(),
            message="Reply exactly: DOUBAO_STREAM_OK",
            model="doubao-seed-2.0",
        )
    )

    assert pieces == ["1. first2. second"]


def test_stream_doubao_browser_completion_ignores_semantically_duplicate_numbered_snapshot() -> None:
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
    compact_snapshot = json.dumps(
        {
            "event_type": 2001,
            "event_data": json.dumps(
                {
                    "is_finish": False,
                    "message": {
                        "content": json.dumps({"text": numbered_snapshot}),
                    },
                }
            ),
        },
        ensure_ascii=False,
    )

    class FakePage:
        def __init__(self) -> None:
            self.polls = [
                {
                    "lines": [
                        'event: CHUNK_DELTA',
                        f'data: {json.dumps({"text": summary_text}, ensure_ascii=False)}',
                        "",
                    ],
                    "done": False,
                    "error": "",
                },
                {
                    "lines": [
                        f"data: {compact_snapshot}",
                    ],
                    "done": True,
                    "error": "",
                },
            ]

        def evaluate(self, script: str, arg: dict[str, object]) -> dict[str, object]:
            if "window[streamId] = state" in script:
                return {"ok": True}
            if "state.lines.splice" in script:
                return self.polls.pop(0)
            raise AssertionError(f"Unexpected evaluate script: {script[:120]}")

    session = _ProviderBrowserSession(
        manager=object(),
        context=object(),
        page=object(),
        headless=True,
        owner_thread=threading.current_thread(),
        metadata={
            "doubao_conversation_id": "0",
            "doubao_request_params": {"aid": "497858"},
        },
    )

    pieces = list(
        _stream_doubao_browser_completion(
            FakePage(),
            session=session,
            client=object(),
            message="Reply exactly: DOUBAO_STREAM_OK",
            model="doubao-seed-2.0",
        )
    )

    assert pieces == [summary_text]


def test_stream_doubao_browser_completion_ignores_duplicate_numbered_chunk_delta_snapshot() -> None:
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

    class FakePage:
        def __init__(self) -> None:
            self.polls = [
                {
                    "lines": [
                        'event: CHUNK_DELTA',
                        f'data: {json.dumps({"text": summary_text}, ensure_ascii=False)}',
                        "",
                    ],
                    "done": False,
                    "error": "",
                },
                {
                    "lines": [
                        'event: CHUNK_DELTA',
                        f'data: {json.dumps({"text": numbered_snapshot}, ensure_ascii=False)}',
                        "",
                    ],
                    "done": True,
                    "error": "",
                },
            ]

        def evaluate(self, script: str, arg: dict[str, object]) -> dict[str, object]:
            if "window[streamId] = state" in script:
                return {"ok": True}
            if "state.lines.splice" in script:
                return self.polls.pop(0)
            raise AssertionError(f"Unexpected evaluate script: {script[:120]}")

    session = _ProviderBrowserSession(
        manager=object(),
        context=object(),
        page=object(),
        headless=True,
        owner_thread=threading.current_thread(),
        metadata={
            "doubao_conversation_id": "0",
            "doubao_request_params": {"aid": "497858"},
        },
    )

    pieces = list(
        _stream_doubao_browser_completion(
            FakePage(),
            session=session,
            client=object(),
            message="Reply exactly: DOUBAO_STREAM_OK",
            model="doubao-seed-2.0",
        )
    )

    assert pieces == [summary_text]


def test_stream_doubao_browser_completion_raises_rate_limit_before_startup_timeout() -> None:
    rate_limit_line = json.dumps(
        {
            "event_data": json.dumps(
                {
                    "code": 710022004,
                    "message": "rate limited",
                    "error_detail": {"message": "系统错误"},
                },
                ensure_ascii=False,
            ),
            "event_id": "0",
            "event_type": 2005,
        },
        ensure_ascii=False,
    )

    class FakePage:
        def __init__(self) -> None:
            self.polls = [
                {
                    "lines": [f"data: {rate_limit_line}"],
                    "done": False,
                    "error": "",
                }
            ]

        def evaluate(self, script: str, arg: dict[str, object]) -> dict[str, object]:
            if "window[streamId] = state" in script:
                return {"ok": True}
            if "state.lines.splice" in script:
                return self.polls.pop(0)
            raise AssertionError(f"Unexpected evaluate script: {script[:120]}")

    session = _ProviderBrowserSession(
        manager=object(),
        context=object(),
        page=object(),
        headless=True,
        owner_thread=threading.current_thread(),
        metadata={
            "doubao_conversation_id": "0",
            "doubao_request_params": {"aid": "497858"},
        },
    )

    try:
        list(
            _stream_doubao_browser_completion(
                FakePage(),
                session=session,
                client=object(),
                message="Reply exactly: DOUBAO_STREAM_OK",
                model="doubao-seed-2.0",
                poll_interval_seconds=0.0,
                startup_timeout_seconds=10.0,
            )
        )
    except ProviderRateLimitError as exc:
        assert "Doubao rate limit exceeded" in str(exc)
    else:
        raise AssertionError("Expected ProviderRateLimitError")


def test_stream_doubao_dom_completion_yields_incremental_text_and_persists_conversation_id() -> None:
    prompt = "Reply with exactly: doubao-stream-dom-ok"
    captured: dict[str, object] = {}

    class FakeKeyboard:
        def __init__(self) -> None:
            self.actions: list[tuple[str, str]] = []

        def press(self, key: str) -> None:
            self.actions.append(("press", key))

        def type(self, text: str, delay: int = 0) -> None:
            self.actions.append(("type", text))

    class FakeLocator:
        def __init__(self) -> None:
            self.clicked = False

        @property
        def first(self) -> "FakeLocator":
            return self

        def wait_for(self, timeout: int | None = None) -> None:
            return None

        def click(self, timeout: int | None = None, force: bool | None = None) -> None:
            self.clicked = True

    class FakeRequest:
        method = "POST"
        post_data = (
            '{"messages":[{"content":"{\\"text\\":\\"Reply with exactly: doubao-stream-dom-ok\\"}"}]}'
        )

    class FakeResponse:
        url = "https://www.doubao.com/chat/completion?aid=497858"
        request = FakeRequest()
        headers = {"content-type": "text/event-stream"}
        status = 200

        def text(self) -> str:
            return (
                'event: SSE_ACK\n'
                'data: {"ack_client_meta":{"conversation_id":"doubao-conv-dom"}}\n\n'
                'event: CHUNK_DELTA\n'
                'data: {"text":"doubao-stream-dom-ok"}\n'
            )

    class FakePage:
        def __init__(self) -> None:
            self.keyboard = FakeKeyboard()
            self.locator_instance = FakeLocator()
            self.listeners: dict[str, object] = {}
            self.snapshots = [
                {"markdownTexts": [], "messageTexts": [prompt], "composerBusy": True},
                {"markdownTexts": ["dou"], "messageTexts": [f"{prompt} dou"], "composerBusy": True},
                {"markdownTexts": ["doubao"], "messageTexts": [f"{prompt} doubao"], "composerBusy": False},
                {"markdownTexts": ["doubao"], "messageTexts": [f"{prompt} doubao"], "composerBusy": False},
            ]

        def locator(self, selector: str) -> FakeLocator:
            assert "textarea" in selector or "button" in selector
            return self.locator_instance

        def wait_for_timeout(self, timeout_ms: int) -> None:
            return None

        def on(self, event_name: str, callback) -> None:
            self.listeners[event_name] = callback

        def remove_listener(self, event_name: str, callback) -> None:
            self.listeners.pop(event_name, None)

        def evaluate(self, script: str, arg: object | None = None):
            if "markdownCount" in script:
                return {"markdownCount": 0, "messageCount": 0}
            if "markdownTexts" in script:
                return self.snapshots.pop(0)
            if "g-send-msg-btn-bg" in script:
                callback = self.listeners.get("response")
                if callable(callback):
                    callback(FakeResponse())
                return True
            return None

    class FakeClient:
        def _persist_doubao_conversation_id(self, conversation_id: str) -> None:
            captured["persisted_conversation_id"] = conversation_id

    session = type("Session", (), {"metadata": {}})()
    page = FakePage()

    pieces = list(
        _stream_doubao_dom_completion(
            page,
            session=session,
            client=FakeClient(),
            message=prompt,
            model="doubao-seed-2.0",
            poll_interval_seconds=0.0,
        )
    )

    assert pieces == ["dou", "bao", "-stream-dom-ok"]
    assert session.metadata["doubao_conversation_id"] == "doubao-conv-dom"
    assert captured["persisted_conversation_id"] == "doubao-conv-dom"
    assert page.locator_instance.clicked is True
    assert page.keyboard.actions == [
        ("press", "Meta+A"),
        ("press", "Backspace"),
        ("type", prompt),
    ]


def test_stream_doubao_dom_completion_raises_rate_limit_before_startup_timeout() -> None:
    prompt = "Reply with exactly: doubao-dom-rate-limit"
    rate_limit_payload = (
        'data: {"event_data":"{\\"code\\":710022004,\\"message\\":\\"rate limited\\",'
        '\\"error_detail\\":{\\"message\\":\\"系统错误\\"}}","event_id":"0","event_type":2005}\n'
        'data: {"event_data":"{}","event_id":"1","event_type":2003}\n'
    )

    class FakeKeyboard:
        def press(self, key: str) -> None:
            return None

        def type(self, text: str, delay: int = 0) -> None:
            return None

    class FakeLocator:
        @property
        def first(self) -> "FakeLocator":
            return self

        def wait_for(self, timeout: int | None = None) -> None:
            return None

        def click(self, timeout: int | None = None, force: bool | None = None) -> None:
            return None

        def fill(self, value: str) -> None:
            return None

    class FakeRequest:
        method = "POST"
        post_data = (
            '{"messages":[{"content":"{\\"text\\":\\"Reply with exactly: doubao-dom-rate-limit\\"}"}]}'
        )

    class FakeResponse:
        url = "https://www.doubao.com/chat/completion?aid=497858"
        request = FakeRequest()
        headers = {"content-type": "text/event-stream"}
        status = 200

        def text(self) -> str:
            return rate_limit_payload

    class FakePage:
        def __init__(self) -> None:
            self.keyboard = FakeKeyboard()
            self.locator_instance = FakeLocator()
            self.listeners: dict[str, object] = {}
            self.snapshots = [
                {"markdownTexts": [], "messageTexts": [prompt], "composerBusy": False},
            ]

        def locator(self, selector: str) -> FakeLocator:
            return self.locator_instance

        def wait_for_timeout(self, timeout_ms: int) -> None:
            return None

        def on(self, event_name: str, callback) -> None:
            self.listeners[event_name] = callback

        def remove_listener(self, event_name: str, callback) -> None:
            self.listeners.pop(event_name, None)

        def evaluate(self, script: str, arg: object | None = None):
            if "markdownCount" in script:
                return {"markdownCount": 0, "messageCount": 0}
            if "markdownTexts" in script:
                return self.snapshots.pop(0)
            if "g-send-msg-btn-bg" in script:
                callback = self.listeners.get("response")
                if callable(callback):
                    callback(FakeResponse())
                return True
            return None

    session = type("Session", (), {"metadata": {}})()

    try:
        list(
            _stream_doubao_dom_completion(
                FakePage(),
                session=session,
                client=type("Client", (), {"_persist_doubao_conversation_id": lambda self, conversation_id: None})(),
                message=prompt,
                model="doubao-seed-2.0",
                poll_interval_seconds=0.0,
                startup_timeout_seconds=10.0,
            )
        )
    except ProviderRateLimitError as exc:
        assert "Doubao rate limit exceeded" in str(exc)
    else:
        raise AssertionError("Expected ProviderRateLimitError")


def test_stream_doubao_dom_completion_ignores_semantically_duplicate_final_snapshot() -> None:
    prompt = "Reply with exactly: doubao-dom-summary"
    streamed_text = (
        "无需等待完整结果，边生成边展示，大幅降低用户等待焦虑。"
        " 交互更流畅自然，尤其适合对话、翻译等实时场景。"
        " 提升系统响应体验，避免长时间空白导致用户流失。"
    )
    final_snapshot = (
        "1. 无需等待完整结果，边生成边展示，大幅降低用户等待焦虑。\n"
        "2. 交互更流畅自然，尤其适合对话、翻译等实时场景。\n"
        "3. 提升系统响应体验，避免长时间空白导致用户流失。"
    )

    class FakeKeyboard:
        def press(self, key: str) -> None:
            return None

        def type(self, text: str, delay: int = 0) -> None:
            return None

    class FakeLocator:
        @property
        def first(self) -> "FakeLocator":
            return self

        def wait_for(self, timeout: int | None = None) -> None:
            return None

        def click(self, timeout: int | None = None, force: bool | None = None) -> None:
            return None

        def fill(self, value: str) -> None:
            return None

    class FakeRequest:
        method = "POST"
        post_data = (
            '{"messages":[{"content":"{\\"text\\":\\"Reply with exactly: doubao-dom-summary\\"}"}]}'
        )

    class FakeResponse:
        url = "https://www.doubao.com/chat/completion?aid=497858"
        request = FakeRequest()
        headers = {"content-type": "text/event-stream"}
        status = 200

        def text(self) -> str:
            return (
                'event: SSE_ACK\n'
                'data: {"ack_client_meta":{"conversation_id":"doubao-conv-dom"}}\n\n'
                'event: CHUNK_DELTA\n'
                f'data: {json.dumps({"text": final_snapshot}, ensure_ascii=False)}\n'
            )

    class FakePage:
        def __init__(self) -> None:
            self.keyboard = FakeKeyboard()
            self.locator_instance = FakeLocator()
            self.listeners: dict[str, object] = {}
            self.snapshots = [
                {"markdownTexts": [], "messageTexts": [prompt], "composerBusy": True},
                {"markdownTexts": [streamed_text], "messageTexts": [f"{prompt} {streamed_text}"], "composerBusy": False},
                {"markdownTexts": [streamed_text], "messageTexts": [f"{prompt} {streamed_text}"], "composerBusy": False},
            ]

        def locator(self, selector: str) -> FakeLocator:
            return self.locator_instance

        def wait_for_timeout(self, timeout_ms: int) -> None:
            return None

        def on(self, event_name: str, callback) -> None:
            self.listeners[event_name] = callback

        def remove_listener(self, event_name: str, callback) -> None:
            self.listeners.pop(event_name, None)

        def evaluate(self, script: str, arg: object | None = None):
            if "markdownCount" in script:
                return {"markdownCount": 0, "messageCount": 0}
            if "markdownTexts" in script:
                return self.snapshots.pop(0)
            if "g-send-msg-btn-bg" in script:
                callback = self.listeners.get("response")
                if callable(callback):
                    callback(FakeResponse())
                return True
            return None

    class FakeClient:
        def _persist_doubao_conversation_id(self, conversation_id: str) -> None:
            return None

    session = type("Session", (), {"metadata": {}})()
    page = FakePage()

    pieces = list(
        _stream_doubao_dom_completion(
            page,
            session=session,
            client=FakeClient(),
            message=prompt,
            model="doubao-seed-2.0",
            poll_interval_seconds=0.0,
            timeout_seconds=1.0,
        )
    )

    assert pieces == [streamed_text]


def test_stream_doubao_dom_completion_resets_after_large_replacement_snapshot() -> None:
    prompt = "Reply with exactly: doubao-dom-restart"

    class FakeKeyboard:
        def press(self, key: str) -> None:
            return None

        def type(self, text: str, delay: int = 0) -> None:
            return None

    class FakeLocator:
        @property
        def first(self) -> "FakeLocator":
            return self

        def wait_for(self, timeout: int | None = None) -> None:
            return None

        def click(self, timeout: int | None = None, force: bool | None = None) -> None:
            return None

        def fill(self, value: str) -> None:
            return None

    class FakeRequest:
        method = "POST"
        post_data = (
            '{"messages":[{"content":"{\\"text\\":\\"Reply with exactly: doubao-dom-restart\\"}"}]}'
        )

    class FakeResponse:
        url = "https://www.doubao.com/chat/completion?aid=497858"
        request = FakeRequest()
        headers = {"content-type": "text/event-stream"}
        status = 200

        def text(self) -> str:
            return 'event: SSE_ACK\ndata: {"ack_client_meta":{"conversation_id":"doubao-conv-dom"}}\n\n'

    class FakePage:
        def __init__(self) -> None:
            self.keyboard = FakeKeyboard()
            self.locator_instance = FakeLocator()
            self.listeners: dict[str, object] = {}
            self.snapshots = [
                {"markdownTexts": [], "messageTexts": [prompt], "composerBusy": True},
                {"markdownTexts": ["第一版结尾。"], "messageTexts": [f"{prompt} 第一版结尾。"], "composerBusy": True},
                {
                    "markdownTexts": ["重新开始：完整版第一段。"],
                    "messageTexts": [f"{prompt} 重新开始：完整版第一段。"],
                    "composerBusy": True,
                },
                {
                    "markdownTexts": ["重新开始：完整版第一段。第二段。"],
                    "messageTexts": [f"{prompt} 重新开始：完整版第一段。第二段。"],
                    "composerBusy": False,
                },
                {
                    "markdownTexts": ["重新开始：完整版第一段。第二段。"],
                    "messageTexts": [f"{prompt} 重新开始：完整版第一段。第二段。"],
                    "composerBusy": False,
                },
            ]

        def locator(self, selector: str) -> FakeLocator:
            return self.locator_instance

        def wait_for_timeout(self, timeout_ms: int) -> None:
            return None

        def on(self, event_name: str, callback) -> None:
            self.listeners[event_name] = callback

        def remove_listener(self, event_name: str, callback) -> None:
            self.listeners.pop(event_name, None)

        def evaluate(self, script: str, arg: object | None = None):
            if "markdownCount" in script:
                return {"markdownCount": 0, "messageCount": 0}
            if "markdownTexts" in script:
                return self.snapshots.pop(0)
            if "g-send-msg-btn-bg" in script:
                callback = self.listeners.get("response")
                if callable(callback):
                    callback(FakeResponse())
                return True
            return None

    session = type("Session", (), {"metadata": {}})()
    page = FakePage()

    pieces = list(
        _stream_doubao_dom_completion(
            page,
            session=session,
            client=type("Client", (), {"_persist_doubao_conversation_id": lambda self, conversation_id: None})(),
            message=prompt,
            model="doubao-thinking",
            poll_interval_seconds=0.0,
            timeout_seconds=1.0,
        )
    )

    assert pieces == ["第一版结尾。", "重新开始：完整版第一段。", "第二段。"]


def test_stream_doubao_prefers_browser_stream_over_dom_fallback(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="doubao",
        kind="browser_session",
        cookie="sessionid=session-1; ttwid=ttwid-1",
        headers={},
        user_agent="ua",
        metadata={"sessionid": "session-1", "ttwid": "ttwid-1"},
        status="valid",
    )
    client = CamoufoxProviderClient("doubao", credentials)

    class FakePage:
        url = _DOUBAO_URL

        def goto(self, url: str, wait_until: str, timeout: int) -> None:
            self.url = url

        def wait_for_timeout(self, timeout_ms: int) -> None:
            return None

    class FakeContext:
        pages = []

    session = _ProviderBrowserSession(
        manager=object(),
        context=FakeContext(),
        page=FakePage(),
        headless=True,
        owner_thread=threading.current_thread(),
        metadata={"doubao_request_params": {"aid": "497858"}},
    )

    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._get_or_create_browser_session",
        lambda **kwargs: session,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._page_is_closed",
        lambda page: False,
    )
    monkeypatch.setattr(
        CamoufoxProviderClient,
        "_inject_cookie_string",
        lambda self, context, cookie_domains: None,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._select_doubao_model",
        lambda page, model: None,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._stream_doubao_browser_completion",
        lambda page, *, session, client, message, model: iter(["real", " stream"]),
    )

    def unexpected_dom(*args, **kwargs):
        raise AssertionError("DOM fallback should not run when browser stream succeeds")

    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._stream_doubao_dom_completion",
        unexpected_dom,
    )

    pieces = list(client._stream_doubao(message="hello", model="doubao-seed-2.0"))

    assert pieces == ["real", " stream"]


def test_stream_doubao_defaults_to_browser_first_even_for_pro_and_thinking_models(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="doubao",
        kind="browser_session",
        cookie="sessionid=session-1; ttwid=ttwid-1",
        headers={},
        user_agent="ua",
        metadata={"sessionid": "session-1", "ttwid": "ttwid-1"},
        status="valid",
    )
    client = CamoufoxProviderClient("doubao", credentials)

    class FakePage:
        url = _DOUBAO_URL

        def goto(self, url: str, wait_until: str, timeout: int) -> None:
            self.url = url

        def wait_for_timeout(self, timeout_ms: int) -> None:
            return None

    class FakeContext:
        pages = []

    for model in ("doubao-thinking", "doubao-pro"):
        call_order: list[str] = []
        session = _ProviderBrowserSession(
            manager=object(),
            context=FakeContext(),
            page=FakePage(),
            headless=True,
            owner_thread=threading.current_thread(),
            metadata={"doubao_request_params": {"aid": "497858"}},
        )

        monkeypatch.setattr(
            "openclaw_algae.providers.camoufox_clients._get_or_create_browser_session",
            lambda **kwargs: session,
        )
        monkeypatch.setattr(
            "openclaw_algae.providers.camoufox_clients._page_is_closed",
            lambda page: False,
        )
        monkeypatch.setattr(
            CamoufoxProviderClient,
            "_inject_cookie_string",
            lambda self, context, cookie_domains: None,
        )
        monkeypatch.setattr(
            "openclaw_algae.providers.camoufox_clients._select_doubao_model",
            lambda page, selected_model: call_order.append(f"select:{selected_model}"),
        )
        monkeypatch.setattr(
            "openclaw_algae.providers.camoufox_clients._stream_doubao_browser_completion",
            lambda page, *, session, client, message, model: (
                call_order.append("browser") or iter(["ok"])
            ),
        )
        monkeypatch.setattr(
            "openclaw_algae.providers.camoufox_clients._stream_doubao_dom_completion",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("DOM should not run before browser by default")
            ),
        )

        assert list(client._stream_doubao(message="hello", model=model)) == ["ok"]
        assert call_order == [f"select:{model}", "browser"]


def test_stream_doubao_falls_back_to_dom_when_browser_stream_completes_silently(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="doubao",
        kind="browser_session",
        cookie="sessionid=session-1; ttwid=ttwid-1",
        headers={},
        user_agent="ua",
        metadata={"sessionid": "session-1", "ttwid": "ttwid-1"},
        status="valid",
    )
    client = CamoufoxProviderClient("doubao", credentials)
    call_order: list[str] = []

    class FakePage:
        url = _DOUBAO_URL

        def goto(self, url: str, wait_until: str, timeout: int) -> None:
            self.url = url

        def wait_for_timeout(self, timeout_ms: int) -> None:
            return None

    class FakeContext:
        pages = []

    session = _ProviderBrowserSession(
        manager=object(),
        context=FakeContext(),
        page=FakePage(),
        headless=True,
        owner_thread=threading.current_thread(),
        metadata={"doubao_request_params": {"aid": "497858"}},
    )

    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._get_or_create_browser_session",
        lambda **kwargs: session,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._page_is_closed",
        lambda page: False,
    )
    monkeypatch.setattr(
        CamoufoxProviderClient,
        "_inject_cookie_string",
        lambda self, context, cookie_domains: None,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._select_doubao_model",
        lambda page, model: call_order.append(f"select:{model}"),
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._stream_doubao_browser_completion",
        lambda page, *, session, client, message, model: (
            call_order.append("browser") or iter(())
        ),
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._stream_doubao_dom_completion",
        lambda page, *, session, client, message, model: (
            call_order.append("dom") or iter(["补", "救"])
        ),
    )

    pieces = list(client._stream_doubao(message="hello", model="doubao-seed-2.0"))

    assert pieces == ["补", "救"]
    assert call_order == ["select:doubao-seed-2.0", "browser", "dom"]


def test_stream_doubao_falls_back_to_dom_when_browser_stream_startup_times_out(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="doubao",
        kind="browser_session",
        cookie="sessionid=session-1; ttwid=ttwid-1",
        headers={},
        user_agent="ua",
        metadata={"sessionid": "session-1", "ttwid": "ttwid-1"},
        status="valid",
    )
    client = CamoufoxProviderClient("doubao", credentials)
    call_order: list[str] = []

    class FakePage:
        url = _DOUBAO_URL

        def goto(self, url: str, wait_until: str, timeout: int) -> None:
            self.url = url

        def wait_for_timeout(self, timeout_ms: int) -> None:
            return None

    class FakeContext:
        pages = []

    session = _ProviderBrowserSession(
        manager=object(),
        context=FakeContext(),
        page=FakePage(),
        headless=True,
        owner_thread=threading.current_thread(),
        metadata={"doubao_request_params": {"aid": "497858"}},
    )

    def slow_browser_stream(*, startup_timeout_seconds=None):
        if startup_timeout_seconds is not None and startup_timeout_seconds < 0.05:
            raise RuntimeError(
                f"Doubao browser stream startup timed out after {startup_timeout_seconds:g}s"
            )
        time.sleep(0.05)
        yield "慢"

    monkeypatch.setattr(
        camoufox_module,
        "_DOUBAO_BROWSER_STREAM_STARTUP_TIMEOUT_SECONDS",
        0.01,
        raising=False,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._get_or_create_browser_session",
        lambda **kwargs: session,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._page_is_closed",
        lambda page: False,
    )
    monkeypatch.setattr(
        CamoufoxProviderClient,
        "_inject_cookie_string",
        lambda self, context, cookie_domains: None,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._select_doubao_model",
        lambda page, model: call_order.append(f"select:{model}"),
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._stream_doubao_browser_completion",
        lambda page, *, session, client, message, model, startup_timeout_seconds=None: (
            call_order.append("browser")
            or slow_browser_stream(startup_timeout_seconds=startup_timeout_seconds)
        ),
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._stream_doubao_dom_completion",
        lambda page, *, session, client, message, model: (
            call_order.append("dom") or iter(["补", "救"])
        ),
    )

    pieces = list(client._stream_doubao(message="hello", model="doubao-seed-2.0"))

    assert pieces == ["补", "救"]
    assert call_order == ["select:doubao-seed-2.0", "browser", "dom"]


def test_stream_doubao_does_not_fallback_to_dom_when_browser_stream_rate_limited(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="doubao",
        kind="browser_session",
        cookie="sessionid=session-1; ttwid=ttwid-1",
        headers={},
        user_agent="ua",
        metadata={"sessionid": "session-1", "ttwid": "ttwid-1"},
        status="valid",
    )
    client = CamoufoxProviderClient("doubao", credentials)
    call_order: list[str] = []

    class FakePage:
        url = _DOUBAO_URL

        def goto(self, url: str, wait_until: str, timeout: int) -> None:
            self.url = url

        def wait_for_timeout(self, timeout_ms: int) -> None:
            return None

    class FakeContext:
        pages = []

    session = _ProviderBrowserSession(
        manager=object(),
        context=FakeContext(),
        page=FakePage(),
        headless=True,
        owner_thread=threading.current_thread(),
        metadata={"doubao_request_params": {"aid": "497858"}},
    )

    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._get_or_create_browser_session",
        lambda **kwargs: session,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._page_is_closed",
        lambda page: False,
    )
    monkeypatch.setattr(
        CamoufoxProviderClient,
        "_inject_cookie_string",
        lambda self, context, cookie_domains: None,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._select_doubao_model",
        lambda page, model: call_order.append(f"select:{model}"),
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._stream_doubao_browser_completion",
        lambda page, *, session, client, message, model, startup_timeout_seconds=None: (
            call_order.append("browser")
            or (_ for _ in ()).throw(ProviderRateLimitError("rate limited"))
        ),
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._stream_doubao_dom_completion",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("DOM fallback should not run after browser rate limit")
        ),
    )

    try:
        list(client._stream_doubao(message="hello", model="doubao-seed-2.0"))
    except ProviderRateLimitError as exc:
        assert "rate limited" in str(exc)
    else:
        raise AssertionError("Expected ProviderRateLimitError")

    assert call_order == ["select:doubao-seed-2.0", "browser"]


def test_stream_doubao_ignores_persisted_conversation_id(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="doubao",
        kind="browser_session",
        cookie="sessionid=session-1; ttwid=ttwid-1",
        headers={},
        user_agent="ua",
        metadata={"sessionid": "session-1", "ttwid": "ttwid-1"},
        status="valid",
    )
    client = CamoufoxProviderClient("doubao", credentials)
    seen: dict[str, object] = {}

    class FakePage:
        url = _DOUBAO_URL

        def goto(self, url: str, wait_until: str, timeout: int) -> None:
            self.url = url

        def wait_for_timeout(self, timeout_ms: int) -> None:
            return None

    class FakeContext:
        pages = []

    session = _ProviderBrowserSession(
        manager=object(),
        context=FakeContext(),
        page=FakePage(),
        headless=True,
        owner_thread=threading.current_thread(),
        metadata={},
    )

    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._get_or_create_browser_session",
        lambda **kwargs: session,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._page_is_closed",
        lambda page: False,
    )
    monkeypatch.setattr(
        CamoufoxProviderClient,
        "_inject_cookie_string",
        lambda self, context, cookie_domains: None,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients.load_provider_session",
        lambda *args, **kwargs: {"conversation_id": "conv-from-store"},
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients.resolve_doubao_query_params",
        lambda credentials: {"aid": "497858"},
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._select_doubao_model",
        lambda page, model: None,
    )

    def fake_browser_stream(page, *, session, client, message, model):
        seen["conversation_id"] = session.metadata.get("doubao_conversation_id")
        yield "fresh"

    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._stream_doubao_browser_completion",
        fake_browser_stream,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._stream_doubao_dom_completion",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("DOM fallback should not run when browser stream succeeds")
        ),
    )

    assert list(client._stream_doubao(message="hello", model="doubao-seed-2.0")) == ["fresh"]
    assert seen["conversation_id"] == "0"


def test_stream_doubao_prefers_dom_stream_for_pro_models(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="doubao",
        kind="browser_session",
        cookie="sessionid=session-1; ttwid=ttwid-1",
        headers={},
        user_agent="ua",
        metadata={"sessionid": "session-1", "ttwid": "ttwid-1"},
        status="valid",
    )
    client = CamoufoxProviderClient("doubao", credentials)
    call_order: list[str] = []

    class FakePage:
        url = _DOUBAO_URL

        def goto(self, url: str, wait_until: str, timeout: int) -> None:
            self.url = url

        def wait_for_timeout(self, timeout_ms: int) -> None:
            return None

    class FakeContext:
        pages = []

    session = _ProviderBrowserSession(
        manager=object(),
        context=FakeContext(),
        page=FakePage(),
        headless=True,
        owner_thread=threading.current_thread(),
        metadata={"doubao_request_params": {"aid": "497858"}},
    )

    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._get_or_create_browser_session",
        lambda **kwargs: session,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._page_is_closed",
        lambda page: False,
    )
    monkeypatch.setattr(
        CamoufoxProviderClient,
        "_inject_cookie_string",
        lambda self, context, cookie_domains: None,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._select_doubao_model",
        lambda page, model: call_order.append(f"select:{model}"),
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients.resolve_doubao_query_params",
        lambda credentials: {"aid": "497858"},
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._prefer_dom_first_doubao_stream",
        lambda model: True,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._stream_doubao_dom_completion",
        lambda page, *, session, client, message, model: (
            call_order.append("dom") or iter(["实", "时"])
        ),
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._stream_doubao_browser_completion",
        lambda page, *, session, client, message, model: (
            call_order.append("browser")
            or (_ for _ in ()).throw(
                AssertionError("doubao-pro should prefer DOM stream first")
            )
        ),
    )

    pieces = list(client._stream_doubao(message="hello", model="doubao-pro"))

    assert pieces == ["实", "时"]
    assert call_order == ["select:doubao-pro", "dom"]


def test_stream_doubao_falls_back_to_browser_when_dom_stream_completes_silently(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="doubao",
        kind="browser_session",
        cookie="sessionid=session-1; ttwid=ttwid-1",
        headers={},
        user_agent="ua",
        metadata={"sessionid": "session-1", "ttwid": "ttwid-1"},
        status="valid",
    )
    client = CamoufoxProviderClient("doubao", credentials)
    call_order: list[str] = []

    class FakePage:
        url = _DOUBAO_URL

        def goto(self, url: str, wait_until: str, timeout: int) -> None:
            self.url = url

        def wait_for_timeout(self, timeout_ms: int) -> None:
            return None

    class FakeContext:
        pages = []

    session = _ProviderBrowserSession(
        manager=object(),
        context=FakeContext(),
        page=FakePage(),
        headless=True,
        owner_thread=threading.current_thread(),
        metadata={"doubao_request_params": {"aid": "497858"}},
    )

    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._get_or_create_browser_session",
        lambda **kwargs: session,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._page_is_closed",
        lambda page: False,
    )
    monkeypatch.setattr(
        CamoufoxProviderClient,
        "_inject_cookie_string",
        lambda self, context, cookie_domains: None,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._prefer_dom_first_doubao_stream",
        lambda model: True,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._select_doubao_model",
        lambda page, model: call_order.append(f"select:{model}"),
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._stream_doubao_dom_completion",
        lambda page, *, session, client, message, model: (
            call_order.append("dom") or iter(())
        ),
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._stream_doubao_browser_completion",
        lambda page, *, session, client, message, model: (
            call_order.append("browser") or iter(["补", "救"])
        ),
    )

    pieces = list(client._stream_doubao(message="hello", model="doubao-pro"))

    assert pieces == ["补", "救"]
    assert call_order == ["select:doubao-pro", "dom", "browser"]


def test_stream_doubao_falls_back_to_browser_when_dom_stream_startup_times_out(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="doubao",
        kind="browser_session",
        cookie="sessionid=session-1; ttwid=ttwid-1",
        headers={},
        user_agent="ua",
        metadata={"sessionid": "session-1", "ttwid": "ttwid-1"},
        status="valid",
    )
    client = CamoufoxProviderClient("doubao", credentials)
    call_order: list[str] = []

    class FakePage:
        url = _DOUBAO_URL

        def goto(self, url: str, wait_until: str, timeout: int) -> None:
            self.url = url

        def wait_for_timeout(self, timeout_ms: int) -> None:
            return None

    class FakeContext:
        pages = []

    session = _ProviderBrowserSession(
        manager=object(),
        context=FakeContext(),
        page=FakePage(),
        headless=True,
        owner_thread=threading.current_thread(),
        metadata={"doubao_request_params": {"aid": "497858"}},
    )

    def slow_dom_stream(*, startup_timeout_seconds=None):
        if startup_timeout_seconds is not None and startup_timeout_seconds < 0.05:
            raise RuntimeError(
                f"Doubao DOM stream startup timed out after {startup_timeout_seconds:g}s"
            )
        time.sleep(0.05)
        yield "慢"

    monkeypatch.setattr(
        camoufox_module,
        "_DOUBAO_DOM_STREAM_STARTUP_TIMEOUT_SECONDS",
        0.01,
        raising=False,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._get_or_create_browser_session",
        lambda **kwargs: session,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._page_is_closed",
        lambda page: False,
    )
    monkeypatch.setattr(
        CamoufoxProviderClient,
        "_inject_cookie_string",
        lambda self, context, cookie_domains: None,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._prefer_dom_first_doubao_stream",
        lambda model: True,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._select_doubao_model",
        lambda page, model: call_order.append(f"select:{model}"),
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._stream_doubao_dom_completion",
        lambda page, *, session, client, message, model, startup_timeout_seconds=None: (
            call_order.append("dom") or slow_dom_stream(startup_timeout_seconds=startup_timeout_seconds)
        ),
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._stream_doubao_browser_completion",
        lambda page, *, session, client, message, model: (
            call_order.append("browser") or iter(["补", "救"])
        ),
    )

    pieces = list(client._stream_doubao(message="hello", model="doubao-pro"))

    assert pieces == ["补", "救"]
    assert call_order == ["select:doubao-pro", "dom", "browser"]


def test_stream_doubao_does_not_fallback_to_browser_when_dom_stream_rate_limited(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="doubao",
        kind="browser_session",
        cookie="sessionid=session-1; ttwid=ttwid-1",
        headers={},
        user_agent="ua",
        metadata={"sessionid": "session-1", "ttwid": "ttwid-1"},
        status="valid",
    )
    client = CamoufoxProviderClient("doubao", credentials)
    call_order: list[str] = []

    class FakePage:
        url = _DOUBAO_URL

        def goto(self, url: str, wait_until: str, timeout: int) -> None:
            self.url = url

        def wait_for_timeout(self, timeout_ms: int) -> None:
            return None

    class FakeContext:
        pages = []

    session = _ProviderBrowserSession(
        manager=object(),
        context=FakeContext(),
        page=FakePage(),
        headless=True,
        owner_thread=threading.current_thread(),
        metadata={"doubao_request_params": {"aid": "497858"}},
    )

    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._get_or_create_browser_session",
        lambda **kwargs: session,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._page_is_closed",
        lambda page: False,
    )
    monkeypatch.setattr(
        CamoufoxProviderClient,
        "_inject_cookie_string",
        lambda self, context, cookie_domains: None,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._prefer_dom_first_doubao_stream",
        lambda model: True,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._select_doubao_model",
        lambda page, model: call_order.append(f"select:{model}"),
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._stream_doubao_dom_completion",
        lambda page, *, session, client, message, model, startup_timeout_seconds=None: (
            call_order.append("dom")
            or (_ for _ in ()).throw(ProviderRateLimitError("rate limited"))
        ),
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._stream_doubao_browser_completion",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Browser fallback should not run after DOM rate limit")
        ),
    )

    try:
        list(client._stream_doubao(message="hello", model="doubao-pro"))
    except ProviderRateLimitError as exc:
        assert "rate limited" in str(exc)
    else:
        raise AssertionError("Expected ProviderRateLimitError")

    assert call_order == ["select:doubao-pro", "dom"]


def test_stream_doubao_prefers_dom_stream_for_thinking_models(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="doubao",
        kind="browser_session",
        cookie="sessionid=session-1; ttwid=ttwid-1",
        headers={},
        user_agent="ua",
        metadata={"sessionid": "session-1", "ttwid": "ttwid-1"},
        status="valid",
    )
    client = CamoufoxProviderClient("doubao", credentials)
    call_order: list[str] = []

    class FakePage:
        url = _DOUBAO_URL

        def goto(self, url: str, wait_until: str, timeout: int) -> None:
            self.url = url

        def wait_for_timeout(self, timeout_ms: int) -> None:
            return None

    class FakeContext:
        pages = []

    session = _ProviderBrowserSession(
        manager=object(),
        context=FakeContext(),
        page=FakePage(),
        headless=True,
        owner_thread=threading.current_thread(),
        metadata={"doubao_request_params": {"aid": "497858"}},
    )

    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._get_or_create_browser_session",
        lambda **kwargs: session,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._page_is_closed",
        lambda page: False,
    )
    monkeypatch.setattr(
        CamoufoxProviderClient,
        "_inject_cookie_string",
        lambda self, context, cookie_domains: None,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._select_doubao_model",
        lambda page, model: call_order.append(f"select:{model}"),
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients.resolve_doubao_query_params",
        lambda credentials: {"aid": "497858"},
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._prefer_dom_first_doubao_stream",
        lambda model: True,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._stream_doubao_dom_completion",
        lambda page, *, session, client, message, model: (
            call_order.append("dom") or iter(["思", "考"])
        ),
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._stream_doubao_browser_completion",
        lambda page, *, session, client, message, model: (
            call_order.append("browser")
            or (_ for _ in ()).throw(
                AssertionError("doubao-thinking should prefer DOM stream first")
            )
        ),
    )

    pieces = list(client._stream_doubao(message="hello", model="doubao-thinking"))

    assert pieces == ["思", "考"]
    assert call_order == ["select:doubao-thinking", "dom"]


def test_dom_send_and_wait_qwen_cn_types_prompt_and_returns_stable_markdown_reply() -> None:
    prompt = "Reply with exactly: qwen-dom-scan"

    class FakeKeyboard:
        def __init__(self) -> None:
            self.actions: list[tuple[str, str]] = []

        def press(self, key: str) -> None:
            self.actions.append(("press", key))

        def type(self, text: str, delay: int = 0) -> None:
            self.actions.append(("type", text))

    class FakeLocator:
        def __init__(self) -> None:
            self.clicked = False

        @property
        def first(self) -> "FakeLocator":
            return self

        def wait_for(self, timeout: int | None = None) -> None:
            return None

        def click(self, timeout: int | None = None) -> None:
            self.clicked = True

    class FakePage:
        def __init__(self) -> None:
            self.keyboard = FakeKeyboard()
            self.locator_instance = FakeLocator()
            self.snapshots = [
                {
                    "markdownTexts": [],
                    "messageTexts": [prompt],
                    "composerBusy": True,
                },
                {
                    "markdownTexts": ["qwen-dom-scan"],
                    "messageTexts": [f"{prompt} qwen-dom-scan 6篇来源"],
                    "composerBusy": False,
                },
                {
                    "markdownTexts": ["qwen-dom-scan"],
                    "messageTexts": [f"{prompt} qwen-dom-scan 6篇来源"],
                    "composerBusy": False,
                },
                {
                    "markdownTexts": ["qwen-dom-scan"],
                    "messageTexts": [f"{prompt} qwen-dom-scan 6篇来源"],
                    "composerBusy": False,
                },
            ]

        def locator(self, selector: str) -> FakeLocator:
            assert selector == '[contenteditable="true"][role="textbox"]'
            return self.locator_instance

        def wait_for_timeout(self, timeout_ms: int) -> None:
            return None

        def evaluate(self, script: str, arg: object | None = None) -> dict[str, object]:
            if "markdownTexts" in script:
                return self.snapshots.pop(0)
            return {}

    page = FakePage()

    text = _dom_send_and_wait_qwen_cn(page, prompt)

    assert text == "qwen-dom-scan"
    assert page.locator_instance.clicked is True
    assert page.keyboard.actions == [
        ("press", "Meta+A"),
        ("press", "Backspace"),
        ("type", prompt),
        ("press", "Enter"),
    ]


def test_dom_send_and_wait_doubao_uses_live_textarea_selector_and_captures_response() -> None:
    prompt = "Reply with exactly: doubao-dom-ok"
    persisted: dict[str, str] = {}

    class FakeKeyboard:
        def __init__(self) -> None:
            self.actions: list[tuple[str, str]] = []

        def press(self, key: str) -> None:
            self.actions.append(("press", key))

        def type(self, text: str, delay: int = 0) -> None:
            self.actions.append(("type", text))

    class FakeLocator:
        def __init__(self) -> None:
            self.clicked = False

        @property
        def first(self) -> "FakeLocator":
            return self

        def wait_for(self, timeout: int | None = None) -> None:
            return None

        def click(self, timeout: int | None = None) -> None:
            self.clicked = True

    class FakeRequest:
        def __init__(self) -> None:
            self.method = "POST"
            self.post_data = (
                '{"messages":[{"content":"{\\"text\\":\\"Reply with exactly: doubao-dom-ok\\"}"}]}'
            )

    class FakeResponse:
        def __init__(self) -> None:
            self.url = "https://www.doubao.com/chat/completion?aid=497858"
            self.request = FakeRequest()
            self.headers = {"content-type": "text/event-stream"}
            self.status = 200

        def text(self) -> str:
            return (
                'event: SSE_ACK\n'
                'data: {"ack_client_meta":{"conversation_id":"doubao-conv-1"}}\n\n'
                'event: CHUNK_DELTA\n'
                'data: {"text":"doubao-dom-ok"}\n'
            )

    class FakePage:
        def __init__(self) -> None:
            self.keyboard = FakeKeyboard()
            self.locator_instance = FakeLocator()
            self.listeners: dict[str, object] = {}

        def locator(self, selector: str) -> FakeLocator:
            assert "textarea" in selector
            return self.locator_instance

        def wait_for_timeout(self, timeout_ms: int) -> None:
            return None

        def on(self, event_name: str, callback) -> None:
            self.listeners[event_name] = callback

        def remove_listener(self, event_name: str, callback) -> None:
            self.listeners.pop(event_name, None)

        def evaluate(self, script: str, arg: object | None = None):
            if "g-send-msg-btn-bg" in script:
                callback = self.listeners.get("response")
                if callable(callback):
                    callback(FakeResponse())
                return True
            return None

    class FakeSession:
        def __init__(self) -> None:
            self.metadata: dict[str, object] = {}

    class FakeClient:
        def _persist_doubao_conversation_id(self, conversation_id: str) -> None:
            persisted["conversation_id"] = conversation_id

    page = FakePage()
    session = FakeSession()
    client = FakeClient()

    text = _dom_send_and_wait_doubao(
        page,
        session=session,
        client=client,
        message=prompt,
        model="doubao-seed-2.0",
    )

    assert text == "doubao-dom-ok"
    assert session.metadata["doubao_conversation_id"] == "doubao-conv-1"
    assert persisted["conversation_id"] == "doubao-conv-1"
    assert page.locator_instance.clicked is True
    assert page.keyboard.actions == [
        ("press", "Meta+A"),
        ("press", "Backspace"),
        ("type", prompt),
    ]


def test_dom_send_and_wait_doubao_recovers_when_composer_click_is_intercepted() -> None:
    prompt = "Reply with exactly: doubao-dom-ok"

    class FakeKeyboard:
        def __init__(self) -> None:
            self.actions: list[tuple[str, str]] = []

        def press(self, key: str) -> None:
            self.actions.append(("press", key))

        def type(self, text: str, delay: int = 0) -> None:
            self.actions.append(("type", text))

    class FakeLocator:
        def __init__(self) -> None:
            self.clicked = 0
            self.filled: list[str] = []

        @property
        def first(self) -> "FakeLocator":
            return self

        def wait_for(self, timeout: int | None = None) -> None:
            return None

        def click(self, timeout: int | None = None, force: bool | None = None) -> None:
            self.clicked += 1
            raise RuntimeError("click intercepted")

        def fill(self, value: str) -> None:
            self.filled.append(value)

    class FakeRequest:
        method = "POST"
        post_data = '{"messages":[{"content":"{\\"text\\":\\"Reply with exactly: doubao-dom-ok\\"}"}]}'

    class FakeResponse:
        url = "https://www.doubao.com/chat/completion?aid=497858"
        request = FakeRequest()
        headers = {"content-type": "text/event-stream"}
        status = 200

        def text(self) -> str:
            return (
                'event: CHUNK_DELTA\n'
                'data: {"text":"doubao-dom-ok"}\n'
            )

    class FakePage:
        def __init__(self) -> None:
            self.keyboard = FakeKeyboard()
            self.locator_instance = FakeLocator()
            self.listeners: dict[str, object] = {}

        def locator(self, selector: str) -> FakeLocator:
            assert "textarea" in selector
            return self.locator_instance

        def wait_for_timeout(self, timeout_ms: int) -> None:
            return None

        def on(self, event_name: str, callback) -> None:
            self.listeners[event_name] = callback

        def remove_listener(self, event_name: str, callback) -> None:
            self.listeners.pop(event_name, None)

        def evaluate(self, script: str, arg: object | None = None):
            callback = self.listeners.get("response")
            if callable(callback):
                callback(FakeResponse())
            return True

    page = FakePage()
    session = type("Session", (), {"metadata": {}})()
    client = type("Client", (), {"_persist_doubao_conversation_id": lambda self, conversation_id: None})()

    text = _dom_send_and_wait_doubao(
        page,
        session=session,
        client=client,
        message=prompt,
        model="doubao-seed-2.0",
    )

    assert text == "doubao-dom-ok"
    assert page.locator_instance.clicked >= 1
    assert page.locator_instance.filled == [prompt]


def test_dom_send_and_wait_doubao_uses_fill_for_multiline_prompt() -> None:
    prompt = "System: line 1\n\nUser: line 2"

    class FakeKeyboard:
        def __init__(self) -> None:
            self.actions: list[tuple[str, str]] = []

        def press(self, key: str) -> None:
            self.actions.append(("press", key))

        def type(self, text: str, delay: int = 0) -> None:
            self.actions.append(("type", text))

    class FakeLocator:
        def __init__(self) -> None:
            self.filled: list[str] = []

        @property
        def first(self) -> "FakeLocator":
            return self

        def wait_for(self, timeout: int | None = None) -> None:
            return None

        def click(self, timeout: int | None = None, force: bool | None = None) -> None:
            return None

        def fill(self, value: str) -> None:
            self.filled.append(value)

    class FakeRequest:
        method = "POST"
        post_data = '{"messages":[{"content_block":[{"content":{"text_block":{"text":"System: line 1\\\\n\\\\nUser: line 2"}}}]}]}'

    class FakeResponse:
        url = "https://www.doubao.com/chat/completion?aid=497858"
        request = FakeRequest()
        headers = {"content-type": "text/event-stream"}
        status = 200

        def text(self) -> str:
            return 'event: CHUNK_DELTA\ndata: {"text":"doubao-dom-ok"}\n'

    class FakePage:
        def __init__(self) -> None:
            self.keyboard = FakeKeyboard()
            self.locator_instance = FakeLocator()
            self.listeners: dict[str, object] = {}

        def locator(self, selector: str) -> FakeLocator:
            return self.locator_instance

        def wait_for_timeout(self, timeout_ms: int) -> None:
            return None

        def on(self, event_name: str, callback) -> None:
            self.listeners[event_name] = callback

        def remove_listener(self, event_name: str, callback) -> None:
            self.listeners.pop(event_name, None)

        def evaluate(self, script: str, arg: object | None = None):
            callback = self.listeners.get("response")
            if callable(callback):
                callback(FakeResponse())
            return True

    page = FakePage()
    session = type("Session", (), {"metadata": {}})()
    client = type("Client", (), {"_persist_doubao_conversation_id": lambda self, conversation_id: None})()

    text = _dom_send_and_wait_doubao(
        page,
        session=session,
        client=client,
        message=prompt,
        model="doubao-seed-2.0",
    )

    assert text == "doubao-dom-ok"
    assert page.locator_instance.filled == [prompt]
    assert ("type", prompt) not in page.keyboard.actions


def test_dom_send_and_wait_glm_cn_presses_enter_and_parses_stream_response() -> None:
    prompt = "Reply with exactly: glm-dom-ok"
    persisted: list[dict[str, str]] = []

    class FakeKeyboard:
        def __init__(self, page: "FakePage") -> None:
            self.page = page
            self.actions: list[tuple[str, str]] = []

        def press(self, key: str) -> None:
            self.actions.append(("press", key))
            if key == "Enter":
                callback = self.page.listeners.get("response")
                if callable(callback):
                    callback(FakeResponse())

        def type(self, text: str, delay: int = 0) -> None:
            self.actions.append(("type", text))

    class FakeLocator:
        @property
        def first(self) -> "FakeLocator":
            return self

        def wait_for(self, timeout: int | None = None) -> None:
            return None

        def click(self, timeout: int | None = None) -> None:
            return None

    class FakeResponse:
        def __init__(self) -> None:
            self.url = "https://chatglm.cn/chatglm/backend-api/assistant/stream"
            self.status = 200

        def text(self) -> str:
            return (
                'data: {"conversation_id":"glm-conv-1","parts":[{"content":[{"type":"text","text":"glm-dom-ok"}]}]}\n'
            )

    class FakePage:
        def __init__(self) -> None:
            self.listeners: dict[str, object] = {}
            self.keyboard = FakeKeyboard(self)

        def goto(self, url: str, **kwargs) -> None:
            return None

        def wait_for_timeout(self, timeout_ms: int) -> None:
            return None

        def locator(self, selector: str) -> FakeLocator:
            assert selector == "textarea"
            return FakeLocator()

        def on(self, event_name: str, callback) -> None:
            self.listeners[event_name] = callback

        def remove_listener(self, event_name: str, callback) -> None:
            self.listeners.pop(event_name, None)

        def evaluate(self, script: str, arg: object | None = None):
            if "!!document.querySelector('textarea')" in script:
                return True
            return None

    class FakeSession:
        def __init__(self) -> None:
            self.metadata: dict[str, object] = {}

    class FakeClient:
        def _persist_glm_cn_session_state(
            self,
            *,
            device_id: str | None = None,
            conversation_id: str | None = None,
        ) -> None:
            persisted.append(
                {
                    "device_id": str(device_id or ""),
                    "conversation_id": str(conversation_id or ""),
                }
            )

    page = FakePage()
    session = FakeSession()
    session.metadata["glm_cn_device_id"] = "glm-device-1"
    client = FakeClient()

    text = _dom_send_and_wait_glm_cn(
        page,
        message=prompt,
        session=session,
        client=client,
    )

    assert text == "glm-dom-ok"
    assert session.metadata["glm_cn_conversation_id"] == "glm-conv-1"
    assert persisted[-1] == {
        "device_id": "glm-device-1",
        "conversation_id": "",
    }
    assert page.keyboard.actions == [
        ("press", "Meta+A"),
        ("press", "Backspace"),
        ("type", prompt),
        ("press", "Enter"),
    ]


def test_dom_send_and_wait_glm_cn_retries_too_frequent_once() -> None:
    prompt = "Reply with exactly: glm-dom-ok"

    class FakeKeyboard:
        def __init__(self, page: "FakePage") -> None:
            self.page = page
            self.actions: list[tuple[str, str]] = []

        def press(self, key: str) -> None:
            self.actions.append(("press", key))
            if key == "Enter":
                callback = self.page.listeners.get("response")
                if callable(callback):
                    callback(self.page.responses.pop(0))

        def type(self, text: str, delay: int = 0) -> None:
            self.actions.append(("type", text))

    class FakeLocator:
        @property
        def first(self) -> "FakeLocator":
            return self

        def wait_for(self, timeout: int | None = None) -> None:
            return None

        def click(self, timeout: int | None = None) -> None:
            return None

    class FakeErrorResponse:
        url = "https://chatglm.cn/chatglm/backend-api/assistant/stream"
        status = 200

        def text(self) -> str:
            return '{"message":"请求过于频繁，请稍后再试"}'

    class FakeOkResponse:
        url = "https://chatglm.cn/chatglm/backend-api/assistant/stream"
        status = 200

        def text(self) -> str:
            return (
                'data: {"conversation_id":"glm-conv-2","parts":[{"content":[{"type":"text","text":"glm-dom-ok"}]}]}\n'
            )

    class FakePage:
        def __init__(self) -> None:
            self.listeners: dict[str, object] = {}
            self.responses = [FakeErrorResponse(), FakeOkResponse()]
            self.keyboard = FakeKeyboard(self)
            self.waits: list[int] = []

        def goto(self, url: str, **kwargs) -> None:
            return None

        def wait_for_timeout(self, timeout_ms: int) -> None:
            self.waits.append(timeout_ms)

        def locator(self, selector: str) -> FakeLocator:
            assert selector == "textarea"
            return FakeLocator()

        def on(self, event_name: str, callback) -> None:
            self.listeners[event_name] = callback

        def remove_listener(self, event_name: str, callback) -> None:
            self.listeners.pop(event_name, None)

        def evaluate(self, script: str, arg: object | None = None):
            if "!!document.querySelector('textarea')" in script:
                return True
            return None

    page = FakePage()
    session = type("Session", (), {"metadata": {"glm_cn_device_id": "glm-device-1"}})()
    client = type(
        "Client",
        (),
        {
            "_persist_glm_cn_session_state": lambda self, *, device_id=None, conversation_id=None: None
        },
    )()

    text = _dom_send_and_wait_glm_cn(
        page,
        message=prompt,
        session=session,
        client=client,
    )

    assert text == "glm-dom-ok"
    assert session.metadata["glm_cn_conversation_id"] == "glm-conv-2"
    assert page.keyboard.actions.count(("press", "Enter")) == 2
    assert any(wait >= 1500 for wait in page.waits)


def test_dom_send_and_wait_glm_cn_uses_fill_for_multiline_prompt() -> None:
    prompt = "System: line 1\n\nUser: line 2"

    class FakeKeyboard:
        def __init__(self, page: "FakePage") -> None:
            self.page = page
            self.actions: list[tuple[str, str]] = []

        def press(self, key: str) -> None:
            self.actions.append(("press", key))
            if key == "Enter":
                callback = self.page.listeners.get("response")
                if callable(callback):
                    callback(FakeResponse())

        def type(self, text: str, delay: int = 0) -> None:
            self.actions.append(("type", text))

    class FakeLocator:
        def __init__(self) -> None:
            self.filled: list[str] = []

        @property
        def first(self) -> "FakeLocator":
            return self

        def wait_for(self, timeout: int | None = None) -> None:
            return None

        def click(self, timeout: int | None = None) -> None:
            return None

        def fill(self, value: str) -> None:
            self.filled.append(value)

    class FakeResponse:
        def __init__(self) -> None:
            self.url = "https://chatglm.cn/chatglm/backend-api/assistant/stream"
            self.status = 200

        def text(self) -> str:
            return (
                'data: {"conversation_id":"glm-conv-3","parts":[{"content":[{"type":"text","text":"glm-dom-ok"}]}]}\n'
            )

    class FakePage:
        def __init__(self) -> None:
            self.listeners: dict[str, object] = {}
            self.locator_instance = FakeLocator()
            self.keyboard = FakeKeyboard(self)

        def goto(self, url: str, **kwargs) -> None:
            return None

        def wait_for_timeout(self, timeout_ms: int) -> None:
            return None

        def locator(self, selector: str) -> FakeLocator:
            assert selector == "textarea"
            return self.locator_instance

        def on(self, event_name: str, callback) -> None:
            self.listeners[event_name] = callback

        def remove_listener(self, event_name: str, callback) -> None:
            self.listeners.pop(event_name, None)

        def evaluate(self, script: str, arg: object | None = None):
            if "!!document.querySelector('textarea')" in script:
                return True
            return None

    page = FakePage()
    session = type("Session", (), {"metadata": {"glm_cn_device_id": "glm-device-1"}})()
    client = type(
        "Client",
        (),
        {
            "_persist_glm_cn_session_state": lambda self, *, device_id=None, conversation_id=None: None
        },
    )()

    text = _dom_send_and_wait_glm_cn(
        page,
        message=prompt,
        session=session,
        client=client,
    )

    assert text == "glm-dom-ok"
    assert page.locator_instance.filled == [prompt]
    assert ("type", prompt) not in page.keyboard.actions
    assert page.keyboard.actions.count(("press", "Enter")) == 1


def test_with_page_serializes_same_provider_browser_profile(monkeypatch, tmp_path) -> None:
    credentials = ProviderCredentialRecord(
        provider="doubao",
        kind="browser_session",
        cookie="session=1",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    class FakePage:
        def __init__(self) -> None:
            self.url = ""

        def goto(self, url: str, **kwargs) -> None:
            self.url = url

    class FakeContext:
        def __init__(self) -> None:
            self.pages: list[FakePage] = []

        def add_cookies(self, cookies) -> None:
            return None

        def new_page(self) -> FakePage:
            page = FakePage()
            self.pages.append(page)
            return page

        def close(self) -> None:
            return None

    class FakePlaywright:
        def __init__(self) -> None:
            self.chromium = self

        def __enter__(self) -> "FakePlaywright":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def launch_persistent_context(self, user_data_dir: str, **kwargs) -> FakeContext:
            return FakeContext()

    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients.require_sync_playwright",
        lambda: lambda: FakePlaywright(),
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients.prepare_browser_state_dir",
        lambda state_dir, provider: tmp_path / provider,
    )

    first_entered = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    results: list[str] = []

    def make_action(name: str):
        def action(_context, _page):
            if name == "first":
                first_entered.set()
                release_first.wait(timeout=2)
            else:
                second_entered.set()
            results.append(name)
            return name

        return action

    client_one = CamoufoxProviderClient("doubao", credentials)
    client_one._state_dir = tmp_path
    client_two = CamoufoxProviderClient("doubao", credentials)
    client_two._state_dir = tmp_path

    first_thread = threading.Thread(
        target=lambda: client_one._with_page(
            start_url="https://www.doubao.com/chat/",
            cookie_domains=(".doubao.com",),
            action=make_action("first"),
        )
    )
    second_thread = threading.Thread(
        target=lambda: client_two._with_page(
            start_url="https://www.doubao.com/chat/",
            cookie_domains=(".doubao.com",),
            action=make_action("second"),
        )
    )

    first_thread.start()
    assert first_entered.wait(timeout=1)
    second_thread.start()
    time.sleep(0.2)

    assert second_entered.is_set() is False

    release_first.set()
    first_thread.join(timeout=1)
    second_thread.join(timeout=1)

    assert second_entered.is_set() is True
    assert results == ["first", "second"]


def test_chat_doubao_prefers_fetch_completion_path_by_default(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="doubao",
        kind="browser_session",
        cookie="session=1",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    class FakePage:
        def __init__(self) -> None:
            self.url = _DOUBAO_URL

        def goto(self, url: str, **kwargs) -> None:
            self.url = url

        def wait_for_timeout(self, timeout_ms: int) -> None:
            return None

        def is_closed(self) -> bool:
            return False

    class FakeContext:
        def __init__(self) -> None:
            self.pages = [FakePage()]

        def add_cookies(self, cookies) -> None:
            pass

        def cookies(self, urls=None):
            return []

    class FakeSession:
        def __init__(self) -> None:
            self.context = FakeContext()
            self.page = self.context.pages[0]
            self.headless = True
            self.owner_thread = None
            self.metadata = {}

    fake_session = FakeSession()

    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._get_or_create_browser_session",
        lambda **kwargs: fake_session,
    )

    def fetch_ok(page, *, session, message, model):
        return "doubao-fetch-ok"

    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._fetch_doubao_browser_completion",
        fetch_ok,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._dom_send_and_wait_doubao",
        lambda page, message, model: (_ for _ in ()).throw(
            AssertionError("DOM fallback should not run when fetch path succeeds")
        ),
    )

    client = CamoufoxProviderClient("doubao", credentials)

    assert (
        client._chat_doubao(message="Reply with exactly: DOUBAO_OK", model="doubao-seed-2.0")
        == "doubao-fetch-ok"
    )


def test_chat_doubao_falls_back_to_dom_when_fetch_path_fails(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="doubao",
        kind="browser_session",
        cookie="session=1",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    class FakePage:
        def __init__(self) -> None:
            self.url = _DOUBAO_URL
            self.goto_calls: list[str] = []

        def goto(self, url: str, **kwargs) -> None:
            self.url = url
            self.goto_calls.append(url)

        def wait_for_timeout(self, timeout_ms: int) -> None:
            return None

        def is_closed(self) -> bool:
            return False

    class FakeContext:
        def __init__(self) -> None:
            self.pages = [FakePage()]

        def add_cookies(self, cookies) -> None:
            pass

        def cookies(self, urls=None):
            return []

    class FakeSession:
        def __init__(self) -> None:
            self.context = FakeContext()
            self.page = self.context.pages[0]
            self.headless = True
            self.owner_thread = None
            self.metadata = {}

    fake_session = FakeSession()

    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._get_or_create_browser_session",
        lambda **kwargs: fake_session,
    )

    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._fetch_doubao_browser_completion",
        lambda page, *, session, message, model: (_ for _ in ()).throw(
            RuntimeError("Doubao fetch timed out")
        ),
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._dom_send_and_wait_doubao",
        lambda page, message, model: "doubao-dom-ok",
    )

    client = CamoufoxProviderClient("doubao", credentials)

    assert (
        client._chat_doubao(message="Reply with exactly: DOUBAO_DOM_OK", model="doubao-seed-2.0")
        == "doubao-dom-ok"
    )


def test_chat_doubao_falls_back_to_dom_when_fetch_path_hits_rate_limit(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="doubao",
        kind="browser_session",
        cookie="session=1",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    class FakePage:
        def __init__(self) -> None:
            self.url = _DOUBAO_URL

        def goto(self, url: str, **kwargs) -> None:
            self.url = url

        def wait_for_timeout(self, timeout_ms: int) -> None:
            return None

        def is_closed(self) -> bool:
            return False

    class FakeContext:
        def __init__(self) -> None:
            self.pages = [FakePage()]

        def add_cookies(self, cookies) -> None:
            pass

        def cookies(self, urls=None):
            return []

    class FakeSession:
        def __init__(self) -> None:
            self.context = FakeContext()
            self.page = self.context.pages[0]
            self.headless = True
            self.owner_thread = None
            self.metadata = {}

    fake_session = FakeSession()

    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._get_or_create_browser_session",
        lambda **kwargs: fake_session,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._fetch_doubao_browser_completion",
        lambda page, *, session, client, message, model: (_ for _ in ()).throw(
            ProviderRateLimitError("rate limited")
        ),
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._dom_send_and_wait_doubao",
        lambda page, session, client, message, model: "doubao-dom-rate-limit-ok",
    )

    client = CamoufoxProviderClient("doubao", credentials)

    assert (
        client._chat_doubao(
            message="Reply with exactly: DOUBAO_DOM_RATE_LIMIT_OK",
            model="doubao-seed-2.0",
        )
        == "doubao-dom-rate-limit-ok"
    )


def test_chat_glm_cn_falls_back_to_dom_when_fetch_path_requires_login(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="glm-cn",
        kind="browser_session",
        cookie=(
            "chatglm_token=token;"
            " chatglm_refresh_token=refresh;"
            " chatglm_user_id=user-1;"
            " chatglm_device_id=device-1"
        ),
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    class FakeContext:
        def cookies(self, urls=None):
            return [
                {"name": "chatglm_token", "value": "token"},
                {"name": "chatglm_refresh_token", "value": "refresh"},
                {"name": "chatglm_user_id", "value": "user-1"},
                {"name": "chatglm_device_id", "value": "device-1"},
            ]

    class FakePage:
        def __init__(self) -> None:
            self.url = "https://chatglm.cn/main/all"

        def goto(self, url: str, **kwargs) -> None:
            self.url = url

        def wait_for_timeout(self, timeout_ms: int) -> None:
            return None

        def evaluate(self, script: str, arg):
            return {
                "ok": True,
                "status": 200,
                "error": "",
                "rawText": '{"message":"请登录后继续使用"}',
            }

    client = CamoufoxProviderClient("glm-cn", credentials)

    monkeypatch.setattr(
        client,
        "_with_page",
        lambda **kwargs: kwargs["action"](FakeContext(), FakePage()),
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._is_jwt_expired",
        lambda token, buffer_seconds=60: False,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._dom_send_and_wait_glm_cn",
        lambda page, message: "glm-dom-ok",
    )

    assert (
        client._chat_glm_cn(message="Reply with exactly: GLM_DOM_OK", model="glm-4-plus")
        == "glm-dom-ok"
    )


def test_chat_glm_cn_prefers_live_non_guest_browser_cookies_over_saved_guest_credentials(
    monkeypatch,
) -> None:
    guest_token = _make_test_jwt(
        {
            "uid": "guest-user",
            "device_id": "device-live",
            "is_guest": True,
            "type": "access",
            "exp": 4_102_444_800,
        }
    )
    guest_refresh = _make_test_jwt(
        {
            "uid": "guest-user",
            "device_id": "device-live",
            "is_guest": True,
            "type": "refresh",
            "exp": 4_102_444_800,
        }
    )
    live_token = _make_test_jwt(
        {
            "uid": "real-user",
            "device_id": "device-live",
            "type": "access",
            "exp": 4_102_444_800,
        }
    )
    live_refresh = _make_test_jwt(
        {
            "uid": "real-user",
            "device_id": "device-live",
            "type": "refresh",
            "exp": 4_102_444_800,
        }
    )
    credentials = ProviderCredentialRecord(
        provider="glm-cn",
        kind="browser_session",
        cookie=(
            f"chatglm_token={guest_token}; "
            f"chatglm_refresh_token={guest_refresh}; "
            "chatglm_user_id=guest-user; "
            "chatglm_device_id=device-live"
        ),
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    class FakeContext:
        def __init__(self) -> None:
            self._cookies = {
                "chatglm_token": live_token,
                "chatglm_refresh_token": live_refresh,
                "chatglm_user_id": "real-user",
                "chatglm_device_id": "device-live",
            }
            self.pages = [FakePage(self)]

        def add_cookies(self, cookies) -> None:
            for cookie in cookies:
                self._cookies[str(cookie["name"])] = str(cookie["value"])

        def cookies(self, urls=None):
            return [{"name": key, "value": value} for key, value in self._cookies.items()]

    class FakePage:
        def __init__(self, context: FakeContext) -> None:
            self.context = context
            self.url = "https://chatglm.cn/main/all"
            self.seen_access_token = None

        def goto(self, url: str, **kwargs) -> None:
            self.url = url

        def wait_for_timeout(self, timeout_ms: int) -> None:
            return None

        def is_closed(self) -> bool:
            return False

        def evaluate(self, script: str, arg):
            self.seen_access_token = arg.get("accessToken")
            if self.seen_access_token != live_token:
                return {
                    "ok": True,
                    "status": 200,
                    "error": "",
                    "rawText": '{"status":500,"message":"您已多次体验过对话, 请登录后继续使用","result":null}',
                }
            return {
                "ok": True,
                "status": 200,
                "error": "",
                "rawText": (
                    'data: {"conversation_id":"conv-live","parts":[{"content":[{"type":"text",'
                    '"text":"GLM_BROWSER_OK"}]}]}\n'
                ),
            }

    class FakeSession:
        def __init__(self) -> None:
            self.context = FakeContext()
            self.page = self.context.pages[0]
            self.headless = True
            self.owner_thread = threading.current_thread()
            self.metadata = {}

    fake_session = FakeSession()
    client = CamoufoxProviderClient("glm-cn", credentials)

    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._get_or_create_browser_session",
        lambda **kwargs: fake_session,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._is_jwt_expired",
        lambda token, buffer_seconds=60: False,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._dom_send_and_wait_glm_cn",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("live browser cookies should avoid DOM fallback")
        ),
    )

    text = client._chat_glm_cn(
        message="Reply with exactly: GLM_BROWSER_OK",
        model="glm-4-plus",
    )

    assert text == "GLM_BROWSER_OK"
    assert fake_session.page.seen_access_token == live_token
    assert f"chatglm_token={live_token}" in (client._credentials.cookie or "")


def test_tool_chat_doubao_prefers_browser_path_over_raw_http_client(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="doubao",
        kind="browser_session",
        cookie="session=1",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )
    client = CamoufoxProviderClient("doubao", credentials)

    monkeypatch.setattr(
        client,
        "_chat_doubao",
        lambda *, message, model: "doubao-browser-tool-ok",
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients.DoubaoWebClient.chat_completion",
        lambda self, *, message, model: (_ for _ in ()).throw(
            AssertionError("tool calling should prefer the browser-backed path")
        ),
    )

    assert (
        client._tool_chat_doubao(
            message="Reply with exactly: DOUBAO_BROWSER_TOOL_OK",
            model="doubao-pro",
        )
        == "doubao-browser-tool-ok"
    )


def test_tool_chat_doubao_enables_temporary_tool_conversation_isolation(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="doubao",
        kind="browser_session",
        cookie="session=1",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )
    client = CamoufoxProviderClient("doubao", credentials)
    seen: dict[str, object] = {}

    def fake_chat(*, message, model):
        seen["during_call"] = getattr(client, "_tool_conversation_isolation", False)
        return "ok"

    monkeypatch.setattr(client, "_chat_doubao", fake_chat)

    assert client._tool_chat_doubao(message="hi", model="doubao-pro") == "ok"
    assert seen["during_call"] is True
    assert getattr(client, "_tool_conversation_isolation", False) is False


def test_tool_chat_glm_cn_prefers_browser_path_over_raw_http_client(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="glm-cn",
        kind="browser_session",
        cookie="chatglm_token=token; chatglm_refresh_token=refresh",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )
    client = CamoufoxProviderClient("glm-cn", credentials)

    monkeypatch.setattr(
        client,
        "_chat_glm_cn",
        lambda *, message, model: "glm-browser-tool-ok",
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients.GLMApiClient.chat_completion",
        lambda self, *, message, model: (_ for _ in ()).throw(
            AssertionError("tool calling should prefer the browser-backed path")
        ),
    )

    assert (
        client._tool_chat_glm_cn(
            message="Reply with exactly: GLM_BROWSER_TOOL_OK",
            model="glm-4-plus",
        )
        == "glm-browser-tool-ok"
    )


def test_tool_chat_glm_cn_enables_temporary_tool_conversation_isolation(monkeypatch) -> None:
    credentials = ProviderCredentialRecord(
        provider="glm-cn",
        kind="browser_session",
        cookie="chatglm_token=token; chatglm_refresh_token=refresh",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )
    client = CamoufoxProviderClient("glm-cn", credentials)
    seen: dict[str, object] = {}

    def fake_chat(*, message, model):
        seen["during_call"] = getattr(client, "_tool_conversation_isolation", False)
        return "ok"

    monkeypatch.setattr(client, "_chat_glm_cn", fake_chat)

    assert client._tool_chat_glm_cn(message="hi", model="glm-5") == "ok"
    assert seen["during_call"] is True
    assert getattr(client, "_tool_conversation_isolation", False) is False


def test_browser_session_is_reused_across_threads(monkeypatch, tmp_path) -> None:
    """Sessions are global per provider: the same session is reused across threads.
    The _BrowserWorkerThread in browser.py ensures all calls are serialised to
    the owner thread, so the underlying playwright context is always accessed from
    its creation thread even when callers come from different threads."""
    launches: list[str] = []

    class FakePage:
        def is_closed(self) -> bool:
            return False

    class FakeContext:
        def __init__(self, label: str) -> None:
            self.label = label
            self.pages = [FakePage()]

        def close(self) -> None:
            return None

    class FakePlaywright:
        def __init__(self) -> None:
            self.chromium = self

        def __enter__(self) -> "FakePlaywright":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def launch_persistent_context(self, user_data_dir: str, **kwargs) -> FakeContext:
            label = f"context-{len(launches) + 1}"
            launches.append(label)
            return FakeContext(label)

    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients.require_sync_playwright",
        lambda: lambda: FakePlaywright(),
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients.prepare_browser_state_dir",
        lambda state_dir, provider: tmp_path / provider,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._close_browser_session",
        lambda provider: None,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._PROVIDER_GLOBAL_SESSIONS",
        {},
    )

    results: list[str] = []

    def worker() -> None:
        session = _get_or_create_browser_session(provider="qwen-cn", state_dir=tmp_path, headless=True)
        results.append(session.context.label)

    first = threading.Thread(target=worker)
    second = threading.Thread(target=worker)
    first.start()
    first.join(timeout=1)
    second.start()
    second.join(timeout=1)

    # Calls from a different thread must recreate the session so playwright
    # objects stay bound to their owner thread.
    assert launches == ["context-1", "context-2"]
    assert results == ["context-1", "context-2"]


def test_get_or_create_browser_session_recovers_with_fresh_manager_when_profile_is_locked(
    monkeypatch, tmp_path
) -> None:
    launches: list[tuple[str, str]] = []
    exits: list[str] = []

    class FakePage:
        def is_closed(self) -> bool:
            return False

    class FakeContext:
        def __init__(self, label: str) -> None:
            self.label = label
            self.pages = [FakePage()]

        def close(self) -> None:
            return None

    class FakePlaywright:
        def __init__(self, label: str) -> None:
            self.label = label
            self.chromium = self

        def __enter__(self) -> "FakePlaywright":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            exits.append(self.label)

        def launch_persistent_context(self, user_data_dir: str, **kwargs) -> FakeContext:
            launches.append((self.label, user_data_dir))
            if self.label == "manager-1":
                raise RuntimeError("Only one copy of Firefox can be open at a time")
            return FakeContext(self.label)

    counter = {"value": 0}

    def factory():
        counter["value"] += 1
        return FakePlaywright(f"manager-{counter['value']}")

    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients.require_sync_playwright",
        lambda: factory,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients.prepare_browser_state_dir",
        lambda state_dir, provider: tmp_path / provider,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._PROVIDER_GLOBAL_SESSIONS",
        {},
    )

    session = _get_or_create_browser_session(provider="glm-cn", state_dir=tmp_path, headless=True)

    assert launches[0] == ("manager-1", str(tmp_path / "glm-cn-runtime"))
    assert launches[1][0] == "manager-2"
    assert "glm-cn-runtime-recovery-" in launches[1][1]
    assert session.context.label == "manager-2"
    assert exits == ["manager-1"]


def test_with_page_does_not_close_browser_session_after_success(monkeypatch, tmp_path) -> None:
    """_with_page should NOT close the session on success — sessions are kept warm for reuse."""
    credentials = ProviderCredentialRecord(
        provider="qwen-cn",
        kind="browser_session",
        cookie="session=1",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    class FakePage:
        def __init__(self) -> None:
            self.url = ""

        def goto(self, url: str, **kwargs) -> None:
            self.url = url

    session = type(
        "Session",
        (),
        {
            "context": type("Context", (), {"add_cookies": lambda self, cookies: None})(),
            "page": FakePage(),
            "headless": True,
            "owner_thread": threading.current_thread(),
            "metadata": {},
        },
    )()
    closed: list[str] = []

    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._get_or_create_browser_session",
        lambda **kwargs: session,
    )
    monkeypatch.setattr(
        "openclaw_algae.providers.camoufox_clients._close_browser_session",
        lambda provider: closed.append(provider),
    )

    client = CamoufoxProviderClient("qwen-cn", credentials)
    client._state_dir = tmp_path

    result = client._with_page(
        start_url="https://www.qianwen.com/",
        cookie_domains=(".qianwen.com",),
        action=lambda context, page: "ok",
    )

    assert result == "ok"
    assert closed == []  # session kept alive for reuse
