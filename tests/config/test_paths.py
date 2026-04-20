from pathlib import Path

from openclaw_algae.config.paths import resolve_openclaw_config_path, resolve_state_dir


def test_resolve_state_dir_defaults_to_user_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    assert resolve_state_dir() == tmp_path / ".openclaw-algae"


def test_resolve_openclaw_config_path_defaults_to_openclaw_home(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    assert resolve_openclaw_config_path() == tmp_path / ".openclaw" / "openclaw.json"


def test_resolve_openclaw_config_path_prefers_config_env_override(
    monkeypatch, tmp_path: Path
) -> None:
    override = tmp_path / "custom" / "openclaw.json"
    monkeypatch.setenv("OPENCLAW_CONFIG_PATH", str(override))

    assert resolve_openclaw_config_path() == override


def test_resolve_openclaw_config_path_uses_state_dir_override(
    monkeypatch, tmp_path: Path
) -> None:
    override = tmp_path / "state-dir"
    monkeypatch.setenv("OPENCLAW_STATE_DIR", str(override))

    assert resolve_openclaw_config_path() == override / "openclaw.json"
