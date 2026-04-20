from typer.testing import CliRunner

from openclaw_algae.cli.app import app
from openclaw_algae.storage.provider_store import load_provider_credentials


def test_login_manual_saves_provider_credentials(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "login",
            "deepseek",
            "--cookie",
            "session=value",
            "--header",
            "authorization=Bearer token",
            "--user-agent",
            "ua",
        ],
    )

    assert result.exit_code == 0
    loaded = load_provider_credentials(tmp_path / ".openclaw-algae" / "providers", "deepseek")
    assert loaded is not None
    assert loaded.cookie == "session=value"
    assert loaded.headers["authorization"] == "Bearer token"


def test_login_without_credentials_rejects_manual_only_provider(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()

    result = runner.invoke(app, ["login", "manus"])

    assert result.exit_code != 0
    assert "manual credentials" in result.stderr.lower()


def test_login_manus_api_key_saves_provider_credentials(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "login",
            "manus",
            "--api-key",
            "manus-key",
        ],
    )

    assert result.exit_code == 0
    assert "Saved API key credentials for manus" in result.stdout
    loaded = load_provider_credentials(tmp_path / ".openclaw-algae" / "providers", "manus")
    assert loaded is not None
    assert loaded.kind == "api_key"
    assert loaded.headers["api_key"] == "manus-key"


def test_login_rejects_api_key_for_provider_without_api_key_support(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "login",
            "deepseek",
            "--api-key",
            "not-supported",
        ],
    )

    assert result.exit_code != 0
    assert "does not support --api-key" in result.stderr


def test_login_rejects_cookie_auth_for_api_key_only_provider(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "login",
            "manus",
            "--cookie",
            "session=value",
        ],
    )

    assert result.exit_code != 0
    assert "requires --api-key" in result.stderr


def test_login_rejects_user_agent_without_real_manual_credentials(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "login",
            "deepseek",
            "--user-agent",
            "ua-only",
        ],
    )

    assert result.exit_code != 0
    assert "Provide --cookie or --header" in result.stderr


def test_login_accepts_multi_token_provider_alias(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "login",
            "qwen",
            "china",
            "--cookie",
            "session=value",
        ],
    )

    assert result.exit_code == 0
    loaded = load_provider_credentials(tmp_path / ".openclaw-algae" / "providers", "qwen-cn")
    assert loaded is not None
    assert loaded.provider == "qwen-cn"
    assert loaded.cookie == "session=value"


def test_login_manual_rejects_malformed_header(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "login",
            "deepseek",
            "--header",
            "authorization",
        ],
    )

    assert result.exit_code != 0
    assert "Invalid header format" in result.stderr
