from openclaw_algae.providers.registry import supported_provider_keys


def test_supported_provider_keys_match_requested_catalog() -> None:
    assert supported_provider_keys() == (
        "deepseek",
        "qwen-intl",
        "qwen-cn",
        "kimi",
        "claude",
        "doubao",
        "chatgpt",
        "gemini",
        "grok",
        "glm-cn",
        "glm-intl",
        "mimo",
        "manus",
    )
