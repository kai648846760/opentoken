from pathlib import Path

from openclaw_algae.storage.bootstrap import initialize_state_dir


def test_initialize_state_dir_creates_expected_directories(tmp_path: Path) -> None:
    state_dir = tmp_path / ".openclaw-algae"

    initialize_state_dir(state_dir)

    assert (state_dir / "providers").is_dir()
    assert (state_dir / "browser").is_dir()
    assert (state_dir / "logs").is_dir()
    assert (state_dir / "openclaw").is_dir()

