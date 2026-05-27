"""Unit tests for the new per-provider model extractors.

These tests don't hit any network. They exercise the regex / JSON-extraction
helpers on small representative payloads so that schema changes in upstream
pages can be caught up front.
"""
from __future__ import annotations

import sys
import types

import pytest

from opentoken.models.discovery import (
    _discover_deepseek_models,
    _discover_nim_models,
    _discover_unified_models,
    _extract_chatgpt_models_from_html,
    _extract_deepseek_models_from_html,
    _extract_gemini_models_from_html,
    _extract_glm_intl_models_from_payload,
    _extract_grok_models_from_html,
    _extract_kimi_models_from_html,
    _extract_mimo_models_from_html,
)
from opentoken.models.provider_credentials import ProviderCredentialRecord


def _credentials(
    provider: str,
    metadata: dict[str, str] | None = None,
) -> ProviderCredentialRecord:
    return ProviderCredentialRecord(
        provider=provider,
        kind="api_key",
        cookie="",
        headers={},
        user_agent="",
        metadata=metadata or {},
        status="valid",
    )


def test_extract_deepseek_models_from_html() -> None:
    html = """
    <script>
    {"model_class":"deepseek-chat","display_name":"DeepSeek Chat","temperature":1.0}
    {"model_class":"deepseek-reasoner","display_name":"DeepSeek Reasoner","temperature":1.0}
    </script>
    """
    assert _extract_deepseek_models_from_html(html) == [
        ("deepseek-chat", "DeepSeek Chat"),
        ("deepseek-reasoner", "DeepSeek Reasoner"),
    ]


def test_extract_kimi_models_from_html() -> None:
    html = """
    <script>
    {"id":"k2","name":"Kimi K2","tier":"free"}
    {"id":"k2-thinking","name":"Kimi K2 思考","tier":"free"}
    {"id":"moonshot-v1-128k","name":"Moonshot v1 128K","tier":"pro"}
    </script>
    """
    assert _extract_kimi_models_from_html(html) == [
        ("k2", "Kimi K2"),
        ("k2-thinking", "Kimi K2 思考"),
        ("moonshot-v1-128k", "Moonshot v1 128K"),
    ]


def test_extract_glm_intl_models_from_payload() -> None:
    html = """
    [{"id":"glm-4.6","name":"GLM-4.6"},{"id":"glm-4-air","name":"GLM-4 Air"}]
    """
    assert _extract_glm_intl_models_from_payload(html) == [
        ("glm-4.6", "GLM-4.6"),
        ("glm-4-air", "GLM-4 Air"),
    ]


def test_extract_gemini_models_from_html() -> None:
    html = """
    ["gemini-2.5-pro","Gemini 2.5 Pro",null,null
    ["gemini-2.5-flash","Gemini 2.5 Flash",null,null
    """
    assert _extract_gemini_models_from_html(html) == [
        ("gemini-2.5-pro", "Gemini 2.5 Pro"),
        ("gemini-2.5-flash", "Gemini 2.5 Flash"),
    ]


def test_extract_grok_models_from_html() -> None:
    html = """
    <script>
    {"modelName":"grok-4","displayName":"Grok 4","beta":false}
    {"modelName":"grok-4-fast","displayName":"Grok 4 Fast","beta":true}
    </script>
    """
    assert _extract_grok_models_from_html(html) == [
        ("grok-4", "Grok 4"),
        ("grok-4-fast", "Grok 4 Fast"),
    ]


def test_extract_mimo_models_from_html() -> None:
    html = """
    <script>
    {"modelKey":"xiaomimo-2.5","displayName":"小米 MiMo 2.5"}
    {"modelKey":"mimo-coder","displayName":"MiMo Coder"}
    </script>
    """
    assert _extract_mimo_models_from_html(html) == [
        ("xiaomimo-2.5", "小米 MiMo 2.5"),
        ("mimo-coder", "MiMo Coder"),
    ]


def test_extract_chatgpt_models_from_html() -> None:
    html = """
    <script>
    {"slug":"gpt-4o","title":"GPT-4o","tier":"plus"}
    {"slug":"gpt-4o-mini","title":"GPT-4o Mini","tier":"free"}
    {"slug":"o1-preview","title":"o1 preview","tier":"plus"}
    </script>
    """
    assert _extract_chatgpt_models_from_html(html) == [
        ("gpt-4o", "GPT-4o"),
        ("gpt-4o-mini", "GPT-4o Mini"),
        ("o1-preview", "o1 preview"),
    ]


def test_discover_nim_models_uses_bearer_auth(monkeypatch) -> None:
    import opentoken.models.discovery as discovery

    captured: dict[str, object] = {}

    def fake_get_json(*, url, credentials, extra_headers=None, timeout_seconds=30.0):
        captured["url"] = url
        captured["headers"] = extra_headers
        return {
            "data": [
                {"id": "deepseek-ai/deepseek-r1"},
                {"id": "meta/llama-3.3-70b-instruct"},
            ]
        }

    monkeypatch.setattr(discovery, "_http_get_json", fake_get_json)

    credentials = _credentials("nim", metadata={"api_key": "nvapi-test"})
    result = _discover_nim_models(credentials, None)  # type: ignore[arg-type]

    assert result == [
        ("deepseek-ai/deepseek-r1", "deepseek-ai/deepseek-r1"),
        ("meta/llama-3.3-70b-instruct", "meta/llama-3.3-70b-instruct"),
    ]
    assert captured["url"] == "https://integrate.api.nvidia.com/v1/models"
    assert captured["headers"] == {"Authorization": "Bearer nvapi-test"}


def test_discover_nim_models_returns_empty_without_token() -> None:
    credentials = _credentials("nim", metadata={})
    assert _discover_nim_models(credentials, None) == []  # type: ignore[arg-type]


def test_discover_nim_models_returns_empty_on_http_failure(monkeypatch) -> None:
    import opentoken.models.discovery as discovery

    monkeypatch.setattr(
        discovery,
        "_http_get_json",
        lambda **kwargs: None,
    )
    assert _discover_nim_models(
        _credentials("nim", metadata={"api_key": "nvapi-test"}),
        None,  # type: ignore[arg-type]
    ) == []


def test_discover_unified_models_lists_backend_filtered_models(monkeypatch) -> None:
    # Fake the litellm.model_cost registry so the test doesn't depend on the
    # real package being installed (or its remote model list).
    fake_litellm = types.SimpleNamespace(
        model_cost={
            "openrouter/anthropic/claude-3.5-sonnet": {"litellm_provider": "openrouter"},
            "openrouter/meta-llama/llama-3.3-70b-instruct": {"litellm_provider": "openrouter"},
            "groq/llama-3.3-70b-versatile": {"litellm_provider": "groq"},
            "anthropic/claude-3.5-sonnet": {"litellm_provider": "anthropic"},
            "gpt-4o": {"litellm_provider": "openai"},  # No prefix -> skipped
        }
    )
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    credentials = _credentials(
        "unified",
        metadata={
            "api_key_openrouter": "sk-or-test",
            "api_key_anthropic": "sk-ant-test",
        },
    )
    result = _discover_unified_models(credentials, None)  # type: ignore[arg-type]
    model_ids = {model_id for model_id, _name in result}

    assert "openrouter/anthropic/claude-3.5-sonnet" in model_ids
    assert "openrouter/meta-llama/llama-3.3-70b-instruct" in model_ids
    assert "anthropic/claude-3.5-sonnet" in model_ids
    # Groq isn't configured in credentials → not enumerated.
    assert "groq/llama-3.3-70b-versatile" not in model_ids
    # Bare model name with no backend prefix is always skipped.
    assert "gpt-4o" not in model_ids


def test_discover_unified_models_returns_empty_without_backends(monkeypatch) -> None:
    # Even if litellm is available, without credentials configured for any
    # backend we don't enumerate anything.
    monkeypatch.setitem(
        sys.modules,
        "litellm",
        types.SimpleNamespace(model_cost={"openrouter/x": {}}),
    )
    assert _discover_unified_models(_credentials("unified"), None) == []  # type: ignore[arg-type]


def test_discover_deepseek_returns_wire_models_when_credentials_authenticate(monkeypatch) -> None:
    import opentoken.models.discovery as discovery

    captured: dict[str, object] = {}

    def fake_get_json(*, url, credentials, extra_headers=None, timeout_seconds=30.0):
        captured["url"] = url
        captured["auth"] = (extra_headers or {}).get("Authorization")
        return {"code": 0, "msg": "", "data": {"biz_data": {"token": "T"}}}

    monkeypatch.setattr(discovery, "_http_get_json", fake_get_json)

    credentials = ProviderCredentialRecord(
        provider="deepseek",
        kind="web_session",
        cookie="x",
        headers={"authorization": "Bearer X"},
        user_agent="",
        metadata={},
        status="valid",
    )

    assert _discover_deepseek_models(credentials, None) == [  # type: ignore[arg-type]
        ("deepseek-chat", "DeepSeek Chat"),
        ("deepseek-reasoner", "DeepSeek Reasoner"),
    ]
    assert captured["url"] == "https://chat.deepseek.com/api/v0/users/current"
    assert captured["auth"] == "Bearer X"


def test_discover_deepseek_returns_empty_without_authorization() -> None:
    creds = ProviderCredentialRecord(
        provider="deepseek",
        kind="web_session",
        cookie="x",
        headers={},  # no authorization header → can't probe
        user_agent="",
        metadata={},
        status="valid",
    )
    assert _discover_deepseek_models(creds, None) == []  # type: ignore[arg-type]


def test_discover_deepseek_returns_empty_when_users_endpoint_errors(monkeypatch) -> None:
    import opentoken.models.discovery as discovery
    monkeypatch.setattr(
        discovery,
        "_http_get_json",
        lambda **kwargs: {"code": 40002, "msg": "Missing Token", "data": None},
    )
    creds = ProviderCredentialRecord(
        provider="deepseek",
        kind="web_session",
        cookie="x",
        headers={"authorization": "Bearer X"},
        user_agent="",
        metadata={},
        status="valid",
    )
    assert _discover_deepseek_models(creds, None) == []  # type: ignore[arg-type]


def test_discover_unified_models_returns_empty_when_litellm_unavailable(monkeypatch) -> None:
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def failing_import(name, *args, **kwargs):
        if name == "litellm":
            raise ImportError("nope")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", failing_import)
    creds = _credentials("unified", metadata={"api_key_openrouter": "sk-or-x"})
    assert _discover_unified_models(creds, None) == []  # type: ignore[arg-type]
