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


def test_models_endpoint_does_not_advertise_unimplemented_embedding_models() -> None:
    """/v1/embeddings 永久 501,所以 /v1/models 也不该再把 text-embedding-* 列
    出来 —— 之前的"故意 decouple"会让 SDK auto-discover 拿了再调 → 51X 错误。
    既然端点不可用,model 名也不暴露。"""
    client = TestClient(create_app())

    response = client.get("/v1/models")

    assert response.status_code == 200
    model_ids = {item["id"] for item in response.json()["data"]}

    assert "text-embedding-3-small" not in model_ids
    assert "text-embedding-3-large" not in model_ids
    assert "text-embedding-ada-002" not in model_ids
