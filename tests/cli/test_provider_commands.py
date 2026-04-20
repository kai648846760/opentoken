from typer.testing import CliRunner

from openclaw_algae.cli.app import app
from openclaw_algae.models.provider_credentials import ProviderCredentialRecord
from openclaw_algae.storage.provider_store import save_provider_credentials


def test_providers_command_lists_saved_providers(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    state_dir = tmp_path / ".openclaw-algae" / "providers"
    save_provider_credentials(
        state_dir,
        ProviderCredentialRecord(
            provider="deepseek",
            kind="web_session",
            cookie="session=value",
            headers={},
            user_agent="ua",
            status="valid",
        ),
    )

    runner = CliRunner()
    result = runner.invoke(app, ["providers"])

    assert result.exit_code == 0
    assert "deepseek" in result.stdout
    assert "valid" in result.stdout


def test_providers_command_lists_supported_catalog_when_empty(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    runner = CliRunner()
    result = runner.invoke(app, ["providers"])

    assert result.exit_code == 0
    assert "deepseek" in result.stdout
    assert "qwen-intl" in result.stdout
    assert "chatgpt" in result.stdout
    assert "not_logged_in" in result.stdout


def test_login_normalizes_provider_alias_and_saves_canonical_provider(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["login", "qwen china", "--cookie", "session=value"],
    )

    assert result.exit_code == 0
    assert "Saved credentials for qwen-cn" in result.stdout
    assert (tmp_path / ".openclaw-algae" / "providers" / "qwen-cn.json").exists()


def test_login_rejects_unknown_provider(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["login", "mystery-provider", "--cookie", "session=value"],
    )

    assert result.exit_code != 0
    assert "Unsupported provider" in result.stderr


def test_logout_removes_saved_provider(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    providers_dir = tmp_path / ".openclaw-algae" / "providers"
    save_provider_credentials(
        providers_dir,
        ProviderCredentialRecord(
            provider="deepseek",
            kind="web_session",
            cookie="session=value",
            headers={},
            user_agent="ua",
            status="valid",
        ),
    )

    runner = CliRunner()
    result = runner.invoke(app, ["logout", "deepseek"])

    assert result.exit_code == 0
    assert not (providers_dir / "deepseek.json").exists()


def test_logout_accepts_multi_token_provider_alias(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    providers_dir = tmp_path / ".openclaw-algae" / "providers"
    save_provider_credentials(
        providers_dir,
        ProviderCredentialRecord(
            provider="qwen-cn",
            kind="web_session",
            cookie="session=value",
            headers={},
            user_agent="ua",
            status="valid",
        ),
    )

    runner = CliRunner()
    result = runner.invoke(app, ["logout", "qwen", "china"])

    assert result.exit_code == 0
    assert not (providers_dir / "qwen-cn.json").exists()
