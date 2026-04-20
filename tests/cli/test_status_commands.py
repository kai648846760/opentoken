from typer.testing import CliRunner

from openclaw_algae.cli.app import app


def test_status_command_runs() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "state_dir" in result.stdout

