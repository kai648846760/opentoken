from __future__ import annotations

from fastapi.testclient import TestClient

from opentoken.api.app import create_app


def test_embeddings_returns_501_until_a_real_backend_is_wired() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/embeddings",
        json={
            "model": "text-embedding-3-small",
            "input": "hello embeddings",
        },
    )

    # Previously this returned deterministic SHA-256 noise dressed up as
    # OpenAI-shaped embeddings, which is worse than nothing — it broke any RAG /
    # vector-store consumer with silent garbage. We now refuse instead.
    assert response.status_code == 501
    body = response.json()
    assert body["error"]["type"] == "not_implemented"
    assert "embedding" in body["error"]["message"].lower()


def test_embeddings_batch_input_also_501() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/embeddings",
        json={
            "model": "text-embedding-3-small",
            "input": ["alpha", "beta"],
        },
    )

    assert response.status_code == 501
