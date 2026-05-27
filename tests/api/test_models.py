from fastapi.testclient import TestClient

from opentoken.api.app import create_app


def test_models_endpoint_returns_openai_style_list() -> None:
    client = TestClient(create_app())

    response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {"object", "data"}
    assert payload["object"] == "list"
    assert isinstance(payload["data"], list)
    for item in payload["data"]:
        assert set(item.keys()) == {"id", "object", "owned_by"}
        assert item["object"] == "model"
        assert item["owned_by"] == "opentoken"


def test_models_endpoint_omits_retired_or_duplicate_ids() -> None:
    client = TestClient(create_app())

    response = client.get("/v1/models")

    assert response.status_code == 200
    model_ids = {item["id"] for item in response.json()["data"]}

    # These ids used to leak from the hardcoded catalog. Now that the catalog is
    # live-discovered, they should not reappear unless a provider's upstream
    # explicitly lists them.
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


def test_models_endpoint_still_advertises_local_embedding_aliases() -> None:
    # /v1/embeddings now returns 501, but /v1/models still lists the canonical
    # OpenAI embedding model ids so clients that probe model availability up
    # front know they exist as aliases. This is intentional decoupling from the
    # /v1/embeddings handler.
    client = TestClient(create_app())

    response = client.get("/v1/models")

    assert response.status_code == 200
    model_ids = {item["id"] for item in response.json()["data"]}

    assert {
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    }.issubset(model_ids)
