from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderDefinition:
    key: str
    display_name: str
    aliases: tuple[str, ...]
    login_modes: tuple[str, ...]
    manual_auth: tuple[str, ...] = ()


_SUPPORTED_PROVIDERS: tuple[ProviderDefinition, ...] = (
    ProviderDefinition(
        key="deepseek",
        display_name="DeepSeek",
        aliases=("deepseek", "deep seek"),
        login_modes=("manual", "browser"),
        manual_auth=("cookie", "header"),
    ),
    ProviderDefinition(
        key="qwen-intl",
        display_name="Qwen International",
        aliases=("qwen-intl", "qwen international", "qwen intl", "qwen"),
        login_modes=("manual", "browser"),
        manual_auth=("cookie", "header"),
    ),
    ProviderDefinition(
        key="qwen-cn",
        display_name="Qwen China",
        aliases=("qwen-cn", "qwen china", "qwen cn", "qianwen", "tongyi qianwen"),
        login_modes=("manual", "browser"),
        manual_auth=("cookie", "header"),
    ),
    ProviderDefinition(
        key="kimi",
        display_name="Kimi",
        aliases=("kimi",),
        login_modes=("manual", "browser"),
        manual_auth=("cookie", "header"),
    ),
    ProviderDefinition(
        key="claude",
        display_name="Claude",
        aliases=("claude", "anthropic"),
        login_modes=("manual", "browser"),
        manual_auth=("cookie", "header"),
    ),
    ProviderDefinition(
        key="doubao",
        display_name="Doubao",
        aliases=("doubao",),
        login_modes=("manual", "browser"),
        manual_auth=("cookie", "header"),
    ),
    ProviderDefinition(
        key="chatgpt",
        display_name="ChatGPT",
        aliases=("chatgpt", "openai"),
        login_modes=("manual", "browser"),
        manual_auth=("cookie", "header"),
    ),
    ProviderDefinition(
        key="gemini",
        display_name="Gemini",
        aliases=("gemini", "google gemini"),
        login_modes=("manual", "browser"),
        manual_auth=("cookie", "header"),
    ),
    ProviderDefinition(
        key="grok",
        display_name="Grok",
        aliases=("grok", "xai", "x ai"),
        login_modes=("manual", "browser"),
        manual_auth=("cookie", "header"),
    ),
    ProviderDefinition(
        key="glm-cn",
        display_name="GLM China",
        aliases=("glm-cn", "glm cn", "glm china", "zhipu", "chatglm"),
        login_modes=("manual", "browser"),
        manual_auth=("cookie", "header"),
    ),
    ProviderDefinition(
        key="glm-intl",
        display_name="GLM International",
        aliases=("glm-intl", "glm international", "glm intl"),
        login_modes=("manual", "browser"),
        manual_auth=("cookie", "header"),
    ),
    ProviderDefinition(
        key="mimo",
        display_name="Xiaomi Mimo",
        aliases=("mimo", "xiaomi mimo", "xiaomimo"),
        login_modes=("manual", "browser"),
        manual_auth=("cookie", "header"),
    ),
    ProviderDefinition(
        key="manus",
        display_name="Manus",
        aliases=("manus",),
        login_modes=("manual",),
        manual_auth=("api_key",),
    ),
    ProviderDefinition(
        key="nim",
        display_name="NVIDIA NIM",
        aliases=("nim", "nvidia", "nvidia nim", "nvidia-nim"),
        login_modes=("manual",),
        manual_auth=("api_key",),
    ),
)


def list_supported_providers() -> tuple[ProviderDefinition, ...]:
    return _SUPPORTED_PROVIDERS


def supported_provider_keys() -> tuple[str, ...]:
    return tuple(provider.key for provider in _SUPPORTED_PROVIDERS)


def resolve_provider_key(raw: str) -> str | None:
    normalized = _normalize_provider_name(raw)
    for provider in _SUPPORTED_PROVIDERS:
        if normalized == _normalize_provider_name(provider.key):
            return provider.key
        if any(normalized == _normalize_provider_name(alias) for alias in provider.aliases):
            return provider.key
    return None


def get_provider_definition(raw: str) -> ProviderDefinition | None:
    key = resolve_provider_key(raw)
    if key is None:
        return None
    for provider in _SUPPORTED_PROVIDERS:
        if provider.key == key:
            return provider
    return None


def _normalize_provider_name(raw: str) -> str:
    cleaned = raw.strip().lower().replace("/", " ")
    for separator in ("-", "_"):
        cleaned = cleaned.replace(separator, " ")
    return " ".join(cleaned.split())
