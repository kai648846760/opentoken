from fastapi.testclient import TestClient

from openclaw_algae.api.app import create_app
from openclaw_algae.config.app_config import default_app_config


def test_health_does_not_require_api_key(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200


def test_models_requires_bearer_api_key(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config = default_app_config()
    config["api_key"] = "test-key"
    (tmp_path / ".openclaw-algae").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".openclaw-algae" / "config.json").write_text(
        '{"api_key":"test-key","host":"127.0.0.1","port":32117}',
        encoding="utf-8",
    )
    client = TestClient(create_app())

    missing = client.get("/v1/models")
    wrong = client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
    ok = client.get("/v1/models", headers={"Authorization": "Bearer test-key"})

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert ok.status_code == 200
