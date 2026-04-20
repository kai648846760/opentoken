import asyncio
from pathlib import Path
import threading

import pytest

from openclaw_algae.gateway.normalized import NormalizedChatRequest
from openclaw_algae.gateway.router import ProviderRouter, get_default_router
from openclaw_algae.models.provider_credentials import ProviderCredentialRecord
from openclaw_algae.providers.base import ChatResponse, ProviderAdapter
from openclaw_algae.providers.base import ProviderRateLimitError
from openclaw_algae.providers.browser import BrowserChatAdapter
from openclaw_algae.providers.camoufox_clients import CamoufoxProviderClient
from openclaw_algae.providers.doubao import DoubaoWebAdapter
from openclaw_algae.providers.qwen import QwenCnWebAdapter
from openclaw_algae.storage.provider_store import save_provider_credentials


class RecordingAdapter(ProviderAdapter):
    def __init__(self) -> None:
        self.seen_request = None
        self.seen_credentials = None

    def chat(self, request, credentials=None) -> ChatResponse:
        self.seen_request = request
        self.seen_credentials = credentials
        return ChatResponse(model=request.model, content="adapter response")


def test_router_loads_provider_credentials_for_deepseek(tmp_path: Path) -> None:
    providers_dir = tmp_path / "providers"
    save_provider_credentials(
        providers_dir,
        ProviderCredentialRecord(
            provider="deepseek",
            kind="web_session",
            cookie="session=value",
            headers={"authorization": "Bearer token"},
            user_agent="ua",
            status="valid",
        ),
    )
    adapter = RecordingAdapter()
    router = ProviderRouter(adapters={"deepseek": adapter}, providers_dir=providers_dir)

    response = router.chat(
        NormalizedChatRequest(
            model="algae/deepseek/deepseek-chat",
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    assert response.content == "adapter response"
    assert adapter.seen_request is not None
    assert adapter.seen_credentials is not None
    assert adapter.seen_credentials.provider == "deepseek"
    assert adapter.seen_credentials.headers["authorization"] == "Bearer token"


def test_router_accepts_unprefixed_openclaw_model_refs(tmp_path: Path) -> None:
    providers_dir = tmp_path / "providers"
    save_provider_credentials(
        providers_dir,
        ProviderCredentialRecord(
            provider="deepseek",
            kind="web_session",
            cookie="session=value",
            headers={"authorization": "Bearer token"},
            user_agent="ua",
            status="valid",
        ),
    )
    adapter = RecordingAdapter()
    router = ProviderRouter(adapters={"deepseek": adapter}, providers_dir=providers_dir)

    response = router.chat(
        NormalizedChatRequest(
            model="deepseek/deepseek-chat",
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    assert response.content == "adapter response"
    assert adapter.seen_request is not None
    assert adapter.seen_request.model == "deepseek/deepseek-chat"


def test_router_accepts_native_raw_model_ids(tmp_path: Path) -> None:
    providers_dir = tmp_path / "providers"
    save_provider_credentials(
        providers_dir,
        ProviderCredentialRecord(
            provider="deepseek",
            kind="web_session",
            cookie="session=value",
            headers={"authorization": "Bearer token"},
            user_agent="ua",
            status="valid",
        ),
    )
    adapter = RecordingAdapter()
    router = ProviderRouter(adapters={"deepseek": adapter}, providers_dir=providers_dir)

    response = router.chat(
        NormalizedChatRequest(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    assert response.content == "adapter response"
    assert adapter.seen_request is not None
    assert adapter.seen_request.model == "deepseek-chat"


def test_router_accepts_qwen_alias_provider_model_refs(tmp_path: Path) -> None:
    providers_dir = tmp_path / "providers"
    save_provider_credentials(
        providers_dir,
        ProviderCredentialRecord(
            provider="qwen-intl",
            kind="browser_session",
            cookie="session=value",
            headers={},
            user_agent="ua",
            metadata={"session_token": "token"},
            status="valid",
        ),
    )
    adapter = RecordingAdapter()
    router = ProviderRouter(adapters={"qwen-intl": adapter}, providers_dir=providers_dir)

    response = router.chat(
        NormalizedChatRequest(
            model="algae/qwen/qwen-3.6-235b-a22b",
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    assert response.content == "adapter response"
    assert adapter.seen_request is not None
    assert adapter.seen_credentials is not None
    assert adapter.seen_credentials.provider == "qwen-intl"


def test_router_falls_back_to_stub_when_provider_is_not_logged_in(tmp_path: Path) -> None:
    adapter = RecordingAdapter()
    router = ProviderRouter(adapters={"deepseek": adapter}, providers_dir=tmp_path / "providers")

    try:
        router.chat(
            NormalizedChatRequest(
                model="algae/deepseek/deepseek-chat",
                messages=[{"role": "user", "content": "hello"}],
            )
        )
    except RuntimeError as exc:
        assert "login deepseek" in str(exc)
    else:
        raise AssertionError("Expected missing-credentials error")
    assert adapter.seen_request is None


def test_router_maps_kimi_rate_limit_as_provider_rate_limit(tmp_path: Path) -> None:
    providers_dir = tmp_path / "providers"
    save_provider_credentials(
        providers_dir,
        ProviderCredentialRecord(
            provider="kimi",
            kind="browser_session",
            cookie="kimi-auth=value",
            headers={},
            user_agent="ua",
            status="valid",
        ),
    )

    class RateLimitedKimiAdapter(ProviderAdapter):
        def chat(self, request, credentials=None) -> ChatResponse:
            raise ProviderRateLimitError("Kimi conversation limit")

    router = ProviderRouter(adapters={"kimi": RateLimitedKimiAdapter()}, providers_dir=providers_dir)

    with pytest.raises(ProviderRateLimitError, match="Kimi conversation limit"):
        router.chat(
            NormalizedChatRequest(
                model="algae/kimi/moonshot-v1-32k",
                messages=[{"role": "user", "content": "hello"}],
            )
        )


    providers_dir = tmp_path / "providers"
    save_provider_credentials(
        providers_dir,
        ProviderCredentialRecord(
            provider="claude",
            kind="browser_session",
            cookie="sessionKey=sk-ant-sid01-test",
            headers={},
            user_agent="ua",
            status="valid",
        ),
    )
    adapter = RecordingAdapter()
    router = ProviderRouter(adapters={"claude": adapter}, providers_dir=providers_dir)

    response = router.chat(
        NormalizedChatRequest(
            model="algae/claude/claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    assert response.content == "adapter response"
    assert adapter.seen_credentials is not None
    assert adapter.seen_credentials.provider == "claude"


def test_router_rejects_non_algae_models(tmp_path: Path) -> None:
    router = ProviderRouter(providers_dir=tmp_path / "providers")

    with pytest.raises(RuntimeError, match="Unsupported model"):
        router.chat(
            NormalizedChatRequest(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hello"}],
            )
        )


def test_router_rejects_unknown_algae_provider_models(tmp_path: Path) -> None:
    router = ProviderRouter(adapters={"deepseek": RecordingAdapter()}, providers_dir=tmp_path / "providers")

    with pytest.raises(RuntimeError, match="Unsupported model"):
        router.chat(
            NormalizedChatRequest(
                model="algae/nonexist/test-model",
                messages=[{"role": "user", "content": "hello"}],
            )
        )


def test_default_router_registers_camoufox_clients_for_browser_providers() -> None:
    router = get_default_router()
    browser_providers = {
        "doubao",
        "qwen-intl",
        "qwen-cn",
        "chatgpt",
        "gemini",
        "grok",
        "glm-cn",
        "glm-intl",
    }

    from openclaw_algae.providers.chatgpt import ChatGPTWebAdapter
    from openclaw_algae.providers.gemini import GeminiWebAdapter
    from openclaw_algae.providers.glm import GLMIntlWebAdapter, GLMWebAdapter
    from openclaw_algae.providers.grok import GrokWebAdapter
    from openclaw_algae.providers.qwen import QwenWebAdapter

    # Some providers still use API adapters; others now prefer browser-backed
    # adapters because they are closer to the reference implementation and more
    # reliable with live credentials.
    adapter_type_map = {
        "doubao": BrowserChatAdapter,
        "qwen-intl": QwenWebAdapter,
        "qwen-cn": QwenCnWebAdapter,
        "chatgpt": ChatGPTWebAdapter,
        "gemini": GeminiWebAdapter,
        "grok": GrokWebAdapter,
        "glm-cn": BrowserChatAdapter,
        "glm-intl": BrowserChatAdapter,
    }

    for provider in browser_providers:
        adapter = router._adapters[provider]
        expected_type = adapter_type_map.get(provider)
        if expected_type is not None:
            assert isinstance(adapter, expected_type), f"{provider} adapter type mismatch"
        else:
            # Fallback for providers still using BrowserChatAdapter
            assert isinstance(adapter, BrowserChatAdapter)


def test_browser_chat_adapter_runs_client_in_dedicated_thread() -> None:
    class LoopSensitiveClient:
        def __init__(self) -> None:
            self.called_thread_name: str | None = None

        def chat_completion(self, *, message: str, model: str) -> str:
            self.called_thread_name = threading.current_thread().name
            with pytest.raises(RuntimeError):
                asyncio.get_running_loop()
            return f"{model}:{message}"

    client = LoopSensitiveClient()
    adapter = BrowserChatAdapter(
        provider_name="Doubao",
        login_hint="openclaw-algae login doubao",
        client_factory=lambda credentials: client,
    )
    credentials = ProviderCredentialRecord(
        provider="doubao",
        kind="browser_session",
        cookie="session=value",
        headers={},
        user_agent="ua",
        status="valid",
    )
    request = NormalizedChatRequest(
        model="algae/doubao/doubao-seed-2.0",
        messages=[{"role": "user", "content": "hello"}],
    )

    async def invoke_from_running_loop() -> ChatResponse:
        return adapter.chat(request, credentials)

    response = asyncio.run(invoke_from_running_loop())

    assert response.content.endswith("hello")
    assert client.called_thread_name is not None
    assert client.called_thread_name != threading.current_thread().name


def test_browser_chat_adapter_normalizes_non_runtime_exceptions() -> None:
    class ExplodingClient:
        def chat_completion(self, *, message: str, model: str) -> str:
            raise ValueError("camoufox launch failed")

    adapter = BrowserChatAdapter(
        provider_name="Doubao",
        login_hint="openclaw-algae login doubao",
        client_factory=lambda credentials: ExplodingClient(),
    )
    credentials = ProviderCredentialRecord(
        provider="doubao",
        kind="browser_session",
        cookie="session=value",
        headers={},
        user_agent="ua",
        status="valid",
    )

    with pytest.raises(RuntimeError, match="camoufox launch failed"):
        adapter.chat(
            NormalizedChatRequest(
                model="algae/doubao/doubao-seed-2.0",
                messages=[{"role": "user", "content": "hello"}],
            ),
            credentials,
        )


def test_router_prefers_browser_backed_adapters_for_doubao_and_glm() -> None:
    router = ProviderRouter()

    assert isinstance(router._adapters["doubao"], BrowserChatAdapter)
    assert isinstance(router._adapters["glm-cn"], BrowserChatAdapter)
    assert isinstance(router._adapters["glm-intl"], BrowserChatAdapter)
