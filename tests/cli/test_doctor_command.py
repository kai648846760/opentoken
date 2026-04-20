from typer.testing import CliRunner

from openclaw_algae.cli.app import app
from openclaw_algae.browser.common import CamoufoxRuntimeStatus
import openclaw_algae.cli.status_view as status_view_module


def test_doctor_reports_environment_paths(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert f"state_dir={tmp_path / '.openclaw-algae'}" in result.stdout
    assert "state_dir_exists=no" in result.stdout
    assert "app_config_exists=no" in result.stdout
    assert f"openclaw_config={tmp_path / '.openclaw' / 'openclaw.json'}" in result.stdout
    assert "openclaw_config_exists=no" in result.stdout
    assert "providers=0" in result.stdout


def test_doctor_reports_camoufox_runtime_status(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()
    monkeypatch.setattr(
        status_view_module,
        "probe_camoufox_runtime",
        lambda: CamoufoxRuntimeStatus(
            package_installed=True,
            browser_installed=True,
            executable_path="/tmp/camoufox",
            version="0.4.11",
            install_hint="python -m camoufox fetch",
        ),
    )

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "camoufox_package_installed=yes" in result.stdout
    assert "camoufox_browser_installed=yes" in result.stdout
    assert "camoufox_version=0.4.11" in result.stdout
    assert "camoufox_executable=/tmp/camoufox" in result.stdout


def test_doctor_reports_logged_in_provider_count(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()

    onboard = runner.invoke(app, ["onboard"])
    assert onboard.exit_code == 0
    login = runner.invoke(app, ["login", "deepseek", "--cookie", "session=value"])
    assert login.exit_code == 0

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "state_dir_exists=yes" in result.stdout
    assert "app_config_exists=yes" in result.stdout
    assert "providers=1" in result.stdout
    assert "provider_keys=deepseek" in result.stdout
