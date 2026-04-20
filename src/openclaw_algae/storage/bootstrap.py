from pathlib import Path

from openclaw_algae.config.app_config import load_or_create_app_config


def initialize_state_dir(state_dir: Path) -> Path:
    for name in ("providers", "browser", "logs", "openclaw", "files", "uploads"):
        (state_dir / name).mkdir(parents=True, exist_ok=True)
    load_or_create_app_config(state_dir / "config.json")
    return state_dir
