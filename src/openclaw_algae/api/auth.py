import json
from pathlib import Path

from fastapi import Request

from openclaw_algae.api.errors import openai_error_response
from openclaw_algae.config.paths import resolve_app_config_path


def maybe_require_api_key(request: Request):
    if request.url.path == "/health":
        return None

    config = _load_existing_app_config(resolve_app_config_path())
    if config is None:
        return None

    expected_api_key = str(config.get("api_key", "")).strip()
    if not expected_api_key:
        return None

    authorization = request.headers.get("Authorization", "")
    if authorization != f"Bearer {expected_api_key}":
        return openai_error_response(
            status_code=401,
            message="Invalid or missing API key.",
            error_type="authentication_error",
        )
    return None


def _load_existing_app_config(config_path: Path) -> dict[str, object] | None:
    if not config_path.exists():
        return None
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload
