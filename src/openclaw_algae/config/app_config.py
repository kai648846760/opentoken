import json
import secrets
from pathlib import Path


def default_app_config() -> dict[str, object]:
    return {
        "api_key": secrets.token_hex(16),
        "host": "127.0.0.1",
        "port": 32117,
    }


def load_or_create_app_config(config_path: Path) -> dict[str, object]:
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    payload = default_app_config()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload

