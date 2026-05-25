from __future__ import annotations

import hashlib
import json
from pathlib import Path

from opentoken.storage._atomic import write_json_atomic
from opentoken.models.provider_credentials import ProviderCredentialRecord


def load_provider_session(
    state_dir: Path,
    *,
    provider: str,
    credentials: ProviderCredentialRecord,
) -> dict[str, str]:
    path = _resolve_session_store_path(state_dir)
    store = _load_store(path)
    return dict(store.get(_session_key(provider, credentials), {}))


def save_provider_session(
    state_dir: Path,
    *,
    provider: str,
    credentials: ProviderCredentialRecord,
    state: dict[str, str],
) -> Path:
    path = _resolve_session_store_path(state_dir)
    store = _load_store(path)
    store[_session_key(provider, credentials)] = dict(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, store)
    return path


def credential_fingerprint(credentials: ProviderCredentialRecord) -> str:
    payload = json.dumps(
        {
            "provider": credentials.provider,
            "kind": credentials.kind,
            "cookie": credentials.cookie or "",
            "headers": credentials.headers,
            "user_agent": credentials.user_agent or "",
            "metadata": credentials.metadata,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _session_key(provider: str, credentials: ProviderCredentialRecord) -> str:
    return f"{provider}:{credential_fingerprint(credentials)}"


def _resolve_session_store_path(state_dir: Path) -> Path:
    return state_dir / "provider-sessions.json"


def _load_store(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    normalized: dict[str, dict[str, str]] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, dict):
            normalized[key] = {
                str(item_key): str(item_value)
                for item_key, item_value in value.items()
            }
    return normalized
