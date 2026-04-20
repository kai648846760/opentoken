from typer.testing import CliRunner

import openclaw_algae.cli.app as cli_app_module
from openclaw_algae.cli.app import app


def test_start_invokes_uvicorn_with_configured_bind(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    captured: dict[str, object] = {}

    def fake_run(app_ref: str, *, factory: bool, host: str, port: int) -> None:
        captured["app_ref"] = app_ref
        captured["factory"] = factory
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(cli_app_module.uvicorn, "run", fake_run)
    runner = CliRunner()

    result = runner.invoke(app, ["start"])

    assert result.exit_code == 0
    assert captured["app_ref"] == "openclaw_algae.api.app:create_app"
    assert captured["factory"] is True
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 32117
    assert (tmp_path / ".openclaw-algae" / "config.json").exists()
