from typer.testing import CliRunner

from openclaw_algae.cli.app import app


def test_cli_help_shows_core_commands() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "onboard" in result.stdout
    assert "login" in result.stdout
    assert "start" in result.stdout
    assert "config" in result.stdout
    assert "verify" in result.stdout
