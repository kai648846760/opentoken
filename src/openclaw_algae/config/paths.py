import os
from pathlib import Path


def resolve_state_dir() -> Path:
    return Path.home() / ".openclaw-algae"


def resolve_app_config_path() -> Path:
    return resolve_state_dir() / "config.json"


def resolve_providers_dir() -> Path:
    return resolve_state_dir() / "providers"


def resolve_openclaw_state_dir() -> Path:
    override = os.getenv("OPENCLAW_STATE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".openclaw"


def resolve_openclaw_config_path() -> Path:
    override = os.getenv("OPENCLAW_CONFIG_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return resolve_openclaw_state_dir() / "openclaw.json"
