from fastapi.testclient import TestClient

from opentoken.api.app import create_app
from opentoken.api.auth import reset_auth_cache
from opentoken.config.app_config import default_app_config


def test_health_does_not_require_api_key(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200


def test_models_requires_bearer_api_key(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config = default_app_config()
    config["api_key"] = "test-key"
    (tmp_path / ".opentoken").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".opentoken" / "config.json").write_text(
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


def test_empty_api_key_fails_closed_without_explicit_keyless(monkeypatch, tmp_path) -> None:
    """`"api_key": ""` 不能再隐式开 keyless 模式 —— rotation 时这会让网关短暂全 open。
    没有 `keyless_local: true` opt-in 就 fail-closed 503。"""
    reset_auth_cache()
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".opentoken").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".opentoken" / "config.json").write_text(
        '{"api_key":"","host":"127.0.0.1","port":32117}', encoding="utf-8"
    )
    client = TestClient(create_app(), raise_server_exceptions=False)
    response = client.get("/v1/models", headers={"Authorization": "Bearer anything"})
    assert response.status_code == 503, response.text
    reset_auth_cache()


def test_explicit_keyless_local_lets_empty_key_pass(monkeypatch, tmp_path) -> None:
    """显式 `keyless_local: true` 让空 api_key 通过 —— 真要 keyless 必须明示意图。"""
    reset_auth_cache()
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".opentoken").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".opentoken" / "config.json").write_text(
        '{"api_key":"","keyless_local":true,"host":"127.0.0.1","port":32117}',
        encoding="utf-8",
    )
    client = TestClient(create_app(), raise_server_exceptions=False)
    response = client.get("/v1/models")
    # 没传 Bearer 也应该被放行（keyless 显式开启）
    assert response.status_code == 200, response.text
    reset_auth_cache()


def test_401_response_carries_x_request_id(monkeypatch, tmp_path) -> None:
    """401 鉴权失败的响应必须带 X-Request-Id —— 之前 require_api_key 在
    assign_request_id 外面,401 不流经 assign_request_id 的 header 注入,
    客户端没法把 401 关联到网关日志。调整 middleware 顺序后必须含 header。"""
    reset_auth_cache()
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".opentoken").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".opentoken" / "config.json").write_text(
        '{"api_key":"real-key","host":"127.0.0.1","port":32117}', encoding="utf-8"
    )
    client = TestClient(create_app())
    response = client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
    assert response.status_code == 401
    assert response.headers.get("x-request-id"), "401 response must carry X-Request-Id"
    reset_auth_cache()


def test_corrupt_config_fails_closed_not_open(monkeypatch, tmp_path) -> None:
    """A corrupt config.json must NOT crash the middleware (500 storm) and must
    NOT fall through to the keyless-open path — it fails closed with 503."""
    reset_auth_cache()
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".opentoken").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".opentoken" / "config.json").write_text("{ truncated", encoding="utf-8")
    client = TestClient(create_app(), raise_server_exceptions=False)

    response = client.get("/v1/models", headers={"Authorization": "Bearer anything"})

    assert response.status_code == 503, response.text
    reset_auth_cache()
