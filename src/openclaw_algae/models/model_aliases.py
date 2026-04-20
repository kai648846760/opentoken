from __future__ import annotations


_MODEL_ALIASES: dict[str, dict[str, str]] = {
    "qwen-intl": {
        "qwen-3.6-plus": "qwen3.6-plus",
        "qwen-3.6-235b-a22b": "qwen3.6-plus",
        "qwen3.6-235b-a22b": "qwen3.6-plus",
        "qwen-3.5-plus": "qwen3.5-plus",
        "qwen-3.5-turbo": "qwen3.5-flash",
        "qwen3.5-turbo": "qwen3.5-flash",
        "qwen-max": "qwen-max-latest",
    },
    "qwen-cn": {
        "qwen3.5-plus": "Qwen3.5-千问",
        "Qwen3.5-Plus": "Qwen3.5-千问",
        "qwen3.5-turbo": "Qwen3.5-Flash",
        "Qwen3.5-Turbo": "Qwen3.5-Flash",
        "qwen-max": "Qwen3-Max",
        "qwen-max-thinking": "Qwen3-Max-Thinking",
        "qwen3-coder": "Qwen3-Coder",
    },
    "mimo": {
        "mimo-2.0": "xiaomimo-chat",
        "mimo-2.5-pro": "mimo-v2-pro",
    },
}


def normalize_provider_model(provider: str, model: str) -> str:
    aliases = _MODEL_ALIASES.get(provider, {})
    return aliases.get(model, model)



def list_provider_aliases(provider: str) -> tuple[str, ...]:
    aliases = _MODEL_ALIASES.get(provider, {})
    return tuple(aliases.keys())


__all__ = ["normalize_provider_model", "list_provider_aliases"]
