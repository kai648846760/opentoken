from __future__ import annotations

from fastapi.testclient import TestClient

from openclaw_algae.api.app import create_app


def test_embeddings_supports_string_input() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/embeddings",
        json={
            "model": "text-embedding-3-small",
            "input": "hello embeddings",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["model"] == "text-embedding-3-small"
    assert len(payload["data"]) == 1
    assert payload["data"][0]["object"] == "embedding"
    assert payload["data"][0]["index"] == 0
    assert isinstance(payload["data"][0]["embedding"], list)
    assert len(payload["data"][0]["embedding"]) == 256
    assert payload["usage"]["prompt_tokens"] > 0
    assert payload["usage"]["total_tokens"] == payload["usage"]["prompt_tokens"]


def test_embeddings_supports_batch_input() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/embeddings",
        json={
            "model": "text-embedding-3-small",
            "input": ["alpha", "beta"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert [item["index"] for item in payload["data"]] == [0, 1]
    assert len(payload["data"][0]["embedding"]) == 256
    assert len(payload["data"][1]["embedding"]) == 256
