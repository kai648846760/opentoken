from __future__ import annotations

import sys
import types

import pytest

from opentoken.gateway.normalized import NormalizedChatRequest
from opentoken.models.provider_credentials import ProviderCredentialRecord
from opentoken.providers.base import ChatResponse


def _credentials() -> ProviderCredentialRecord:
    return ProviderCredentialRecord(
        provider="unified",
        kind="api_key",
        cookie="",
        headers={},
        user_agent="",
        metadata={
            "api_key_openrouter": "sk-or-test",
            "api_key_anthropic": "sk-ant-test",
        },
        status="valid",
    )


def _request(model: str) -> NormalizedChatRequest:
    return NormalizedChatRequest(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
    )


@pytest.fixture
def fake_litellm(monkeypatch):
    """Inject a fake `litellm` module to test the adapter without the real dep."""
    seen_calls: list[dict] = []

    def fake_completion(**kwargs):
        seen_calls.append(kwargs)
        # Mimic the OpenAI-style response object that LiteLLM returns.
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "fake-response",
                        "tool_calls": None,
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    module = types.SimpleNamespace(completion=fake_completion, _seen=seen_calls)
    monkeypatch.setitem(sys.modules, "litellm", module)
    # Reset the cached availability flag from the adapter module so the new fake
    # is picked up even if a previous test marked it unavailable.
    import opentoken.providers.unified_proxy as up
    up._LITELLM_AVAILABLE = None
    return module


def test_unified_proxy_calls_litellm_with_stripped_prefix(fake_litellm):
    from opentoken.providers.unified_proxy import UnifiedProxyAdapter

    adapter = UnifiedProxyAdapter()
    response = adapter.chat(_request("unified/openrouter/anthropic/claude-3.5-sonnet"), _credentials())

    assert isinstance(response, ChatResponse)
    assert response.content == "fake-response"
    assert response.finish_reason == "stop"

    call = fake_litellm._seen[-1]
    # The unified/ prefix is stripped before being passed to litellm.
    assert call["model"] == "openrouter/anthropic/claude-3.5-sonnet"
    assert call["stream"] is False
    assert call["messages"] == [{"role": "user", "content": "hi"}]


def test_unified_proxy_passes_backend_specific_api_key(fake_litellm):
    """The backend prefix in the model id picks the matching api_key_<backend>
    entry from credentials metadata and passes it as litellm's per-call `api_key=`
    kwarg. The previous design mutated os.environ under a global lock, which
    serialised every unified-proxy request process-wide for the duration of the
    upstream completion."""
    from opentoken.providers.unified_proxy import UnifiedProxyAdapter

    UnifiedProxyAdapter().chat(_request("unified/openrouter/anthropic/claude-3.5-sonnet"), _credentials())
    assert fake_litellm._seen[-1]["api_key"] == "sk-or-test"

    UnifiedProxyAdapter().chat(_request("unified/anthropic/claude-3-haiku"), _credentials())
    assert fake_litellm._seen[-1]["api_key"] == "sk-ant-test"


def test_unified_proxy_falls_back_to_generic_api_key(fake_litellm):
    """If a backend doesn't have a dedicated api_key_<backend> entry, fall back
    to a generic api_key (covers single-backend setups that don't need the
    namespaced form)."""
    from opentoken.providers.unified_proxy import UnifiedProxyAdapter

    creds = ProviderCredentialRecord(
        provider="unified",
        kind="api_key",
        cookie="",
        headers={},
        user_agent="",
        metadata={"api_key": "generic-key"},
        status="valid",
    )
    UnifiedProxyAdapter().chat(_request("unified/groq/llama-3"), creds)
    assert fake_litellm._seen[-1]["api_key"] == "generic-key"


def test_unified_proxy_omits_api_key_when_credentials_have_none(fake_litellm):
    """No credentials configured for the requested backend → don't pass api_key,
    let litellm resolve via its own env vars (covers users who set env directly)."""
    from opentoken.providers.unified_proxy import UnifiedProxyAdapter

    creds = ProviderCredentialRecord(
        provider="unified",
        kind="api_key",
        cookie="",
        headers={},
        user_agent="",
        metadata={"api_key_openrouter": "sk-or-test"},  # No 'groq' entry.
        status="valid",
    )
    UnifiedProxyAdapter().chat(_request("unified/groq/llama-3"), creds)
    assert "api_key" not in fake_litellm._seen[-1]


def test_unified_proxy_raises_when_litellm_missing(monkeypatch):
    import opentoken.providers.unified_proxy as up

    monkeypatch.setitem(sys.modules, "litellm", None)
    up._LITELLM_AVAILABLE = None
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name == "litellm":
            raise ImportError("no litellm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(RuntimeError, match="litellm is not installed"):
        up.UnifiedProxyAdapter().chat(_request("unified/openrouter/x"), _credentials())


def test_stream_unified_emits_content_deltas():
    from opentoken.providers.unified_proxy import _stream_unified

    def fake_completion(**kwargs):
        yield {"choices": [{"delta": {"content": "Hel"}}]}
        yield {"choices": [{"delta": {"content": "lo"}}]}

    litellm = types.SimpleNamespace(completion=fake_completion)
    chunks = list(
        _stream_unified(
            litellm=litellm,
            model="openrouter/x",
            messages=[{"role": "user", "content": "hi"}],
            temperature=None,
            tools=None,
            tool_choice=None,
            api_key=None,
        )
    )
    assert "".join(chunks) == "Hello"


def test_stream_unified_raises_on_tool_calls_instead_of_dropping():
    """Streaming can't carry OpenAI structured tool_call deltas (the interface
    is Iterator[str]). Rather than silently emit an empty completion, the
    streamer must fail loudly so the caller retries with stream=false."""
    from opentoken.providers.unified_proxy import _stream_unified

    def fake_completion(**kwargs):
        yield {"choices": [{"delta": {"content": None, "tool_calls": [
            {"index": 0, "function": {"name": "get_weather", "arguments": "{}"}}
        ]}}]}

    litellm = types.SimpleNamespace(completion=fake_completion)
    with pytest.raises(RuntimeError, match="tool_calls during streaming"):
        list(
            _stream_unified(
                litellm=litellm,
                model="openrouter/x",
                messages=[{"role": "user", "content": "hi"}],
                temperature=None,
                tools=[{"type": "function", "function": {"name": "get_weather"}}],
                tool_choice="auto",
                api_key=None,
            )
        )
