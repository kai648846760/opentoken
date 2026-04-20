from fastapi.testclient import TestClient

from openclaw_algae.api.app import create_app
from openclaw_algae.models.catalog import default_catalog


def test_models_endpoint_returns_openai_style_list() -> None:
    client = TestClient(create_app())

    response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {"object", "data"}
    assert payload["object"] == "list"
    assert isinstance(payload["data"], list)
    assert len(payload["data"]) >= len(default_catalog())
    for item in payload["data"]:
        assert set(item.keys()) == {"id", "object", "owned_by"}
        assert item["object"] == "model"
        assert item["owned_by"] == "openclaw-algae"


def test_models_endpoint_lists_current_supported_provider_defaults() -> None:
    client = TestClient(create_app())

    response = client.get("/v1/models")

    assert response.status_code == 200
    model_ids = {item["id"] for item in response.json()["data"]}

    expected = {
        "algae/deepseek/deepseek-chat",
        "algae/deepseek/deepseek-reasoner",
        "algae/qwen-intl/qwen3.6-plus",
        "algae/qwen-intl/qwen3.5-plus",
        "algae/qwen-intl/qwen3.5-flash",
        "algae/qwen-intl/qwen3.5-omni-plus",
        "algae/qwen-intl/qwen-max-latest",
        "algae/qwen-cn/Qwen3.5-千问",
        "algae/qwen-cn/Qwen3.5-Flash",
        "algae/qwen-cn/Qwen3-Max",
        "algae/qwen-cn/Qwen3-Max-Thinking",
        "algae/qwen-cn/Qwen3-Coder",
        "algae/kimi/moonshot-v1-8k",
        "algae/kimi/moonshot-v1-32k",
        "algae/kimi/moonshot-v1-128k",
        "algae/claude/claude-sonnet-4-6",
        "algae/claude/claude-opus-4-6",
        "algae/claude/claude-haiku-4-6",
        "algae/doubao/doubao-seed-2.0",
        "algae/doubao/doubao-pro",
        "algae/chatgpt/gpt-4",
        "algae/chatgpt/gpt-4-turbo",
        "algae/gemini/gemini-pro",
        "algae/gemini/gemini-ultra",
        "algae/grok/grok-1",
        "algae/grok/grok-2",
        "algae/glm-cn/glm-4-plus",
        "algae/glm-cn/glm-4-think",
        "algae/glm-intl/glm-4-plus",
        "algae/glm-intl/glm-4-think",
        "algae/mimo/mimo-2.0",
        "algae/mimo/mimo-2.5-pro",
        "algae/manus/manus-1.6",
        "algae/manus/manus-1.6-lite",
    }

    assert expected.issubset(model_ids)


def test_models_endpoint_omits_retired_or_duplicate_ids() -> None:
    client = TestClient(create_app())

    response = client.get("/v1/models")

    assert response.status_code == 200
    model_ids = {item["id"] for item in response.json()["data"]}

    retired = {
        "algae/qwen-intl/qwen3.5-turbo",
        "algae/qwen-cn/qwen3.5-plus",
        "algae/qwen-cn/qwen3.5-turbo",
        "algae/qwen-cn/Qwen3.5-Plus",
        "algae/qwen-cn/Qwen3.5-Turbo",
        "algae/doubao/doubao-lite",
        "algae/glm-cn/glm-4",
        "algae/glm-cn/glm-4-zero",
        "algae/mimo/mimo-v2-pro",
        "algae/mimo/xiaomimo-chat",
    }

    assert retired.isdisjoint(model_ids)


def test_models_endpoint_includes_native_ids_and_embedding_models() -> None:
    client = TestClient(create_app())

    response = client.get("/v1/models")

    assert response.status_code == 200
    model_ids = {item["id"] for item in response.json()["data"]}

    assert {
        "deepseek-chat",
        "deepseek-reasoner",
        "gpt-4",
        "gpt-4-turbo",
        "claude-sonnet-4-6",
        "doubao-seed-2.0",
        "glm-4-plus",
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    }.issubset(model_ids)
