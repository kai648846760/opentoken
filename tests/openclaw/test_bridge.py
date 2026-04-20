from openclaw_algae.openclaw.bridge import build_algae_provider_patch


def test_build_algae_provider_patch_adds_unified_provider() -> None:
    patch = build_algae_provider_patch(
        base_url="http://127.0.0.1:32117/v1",
        api_key="test-algae-key",
    )

    assert "models" in patch
    assert "algae" in patch["models"]["providers"]
    provider = patch["models"]["providers"]["algae"]
    assert provider["api"] == "openai-completions"
    assert provider["apiKey"] == "test-algae-key"
    assert any(model["id"] == "deepseek/deepseek-chat" for model in provider["models"])
    assert any(model["id"] == "claude/claude-sonnet-4-6" for model in provider["models"])
