from opentoken.opentoken.bridge import build_algae_provider_patch


def test_build_algae_provider_patch_writes_envelope_with_dynamic_model_list(monkeypatch) -> None:
    # The bridge now pulls live-discovered models, so seed a deterministic catalog.
    from opentoken.models.catalog import ModelCatalogEntry

    monkeypatch.setattr(
        "opentoken.opentoken.bridge.load_model_catalog",
        lambda: [
            ModelCatalogEntry(id="algae/deepseek/deepseek-chat", provider="opentoken", name="DeepSeek Chat"),
            ModelCatalogEntry(id="algae/claude/claude-sonnet-4-6", provider="opentoken", name="Claude Sonnet 4.6"),
        ],
    )

    patch = build_algae_provider_patch(
        base_url="http://127.0.0.1:32117/v1",
        api_key="test-algae-key",
    )

    assert "models" in patch
    assert "algae" in patch["models"]["providers"]
    provider = patch["models"]["providers"]["algae"]
    assert provider["api"] == "openai-completions"
    assert provider["apiKey"] == "test-algae-key"
    assert provider["baseUrl"] == "http://127.0.0.1:32117/v1"
    assert any(model["id"] == "deepseek/deepseek-chat" for model in provider["models"])
    assert any(model["id"] == "claude/claude-sonnet-4-6" for model in provider["models"])


def test_build_algae_provider_patch_handles_empty_discovery(monkeypatch) -> None:
    # When no providers are logged in / discovery returns nothing, the patch
    # should still produce a valid envelope with an empty models list rather
    # than fall back to a stale hardcoded list.
    monkeypatch.setattr("opentoken.opentoken.bridge.load_model_catalog", lambda: [])
    patch = build_algae_provider_patch(
        base_url="http://127.0.0.1:32117/v1",
        api_key="test-algae-key",
    )
    provider = patch["models"]["providers"]["algae"]
    assert provider["models"] == []
