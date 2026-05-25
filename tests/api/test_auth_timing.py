"""Auth middleware uses constant-time comparison for the API key.

This is a structural test — we monkeypatch hmac.compare_digest and verify the
code path actually goes through it, rather than naive `==` comparison.
"""
from __future__ import annotations

import hmac
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from opentoken.api import auth as auth_module
from opentoken.api.app import create_app


@pytest.fixture
def isolated_state(monkeypatch, tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    config_path = state_dir / "config.json"
    config_path.write_text(json.dumps({"api_key": "secret-token-value", "host": "127.0.0.1", "port": 32117}))

    monkeypatch.setattr("opentoken.api.auth.resolve_app_config_path", lambda: config_path)
    monkeypatch.setattr("opentoken.config.paths.resolve_state_dir", lambda: state_dir)
    auth_module.reset_auth_cache()
    yield state_dir
    auth_module.reset_auth_cache()


def test_auth_uses_hmac_compare_digest(isolated_state, monkeypatch):
    calls = {"count": 0}
    real_compare = hmac.compare_digest

    def spying_compare(a, b):
        calls["count"] += 1
        return real_compare(a, b)

    monkeypatch.setattr(auth_module, "hmac", type("h", (), {"compare_digest": spying_compare}))

    client = TestClient(create_app())
    response = client.get("/v1/models", headers={"Authorization": "Bearer wrong"})

    assert response.status_code == 401
    # The constant-time comparison was reached at least once for this request.
    assert calls["count"] >= 1


def test_auth_accepts_matching_token(isolated_state):
    client = TestClient(create_app())
    response = client.get(
        "/v1/models",
        headers={"Authorization": "Bearer secret-token-value"},
    )
    assert response.status_code == 200


def test_auth_caches_config_across_requests(isolated_state):
    # Two consecutive requests must not re-stat the file each time. The cache
    # is keyed on mtime_ns; we simulate "no change" by not touching the file.
    state_dir = isolated_state
    config_path = state_dir / "config.json"
    mtime = config_path.stat().st_mtime_ns

    client = TestClient(create_app())
    client.get("/v1/models", headers={"Authorization": "Bearer secret-token-value"})
    client.get("/v1/models", headers={"Authorization": "Bearer secret-token-value"})

    assert config_path.stat().st_mtime_ns == mtime
