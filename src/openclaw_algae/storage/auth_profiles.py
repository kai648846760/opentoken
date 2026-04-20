from __future__ import annotations

import copy
import json
from pathlib import Path

from openclaw_algae.config.paths import resolve_state_dir
from openclaw_algae.models.provider_credentials import ProviderCredentialRecord


_DEFAULT_STORE: dict[str, object] = {
    "version": 1,
    "profiles": {},
    "order": {},
    "lastGood": {},
    "usageStats": {},
}


def resolve_auth_profiles_path(providers_dir: Path) -> Path:
    base_dir = providers_dir.parent if providers_dir.name == "providers" else providers_dir
    return base_dir / "auth-profiles.json"


def resolve_shared_auth_profiles_path(providers_dir: Path) -> Path | None:
    base_dir = providers_dir.parent if providers_dir.name == "providers" else providers_dir
    if base_dir.name == ".openclaw-algae":
        return base_dir.parent / ".openclaw" / "auth-profiles.json"
    try:
        if base_dir.resolve() == resolve_state_dir().resolve():
            return base_dir.parent / ".openclaw" / "auth-profiles.json"
    except OSError:
        return None
    return None


def load_auth_profile_record(providers_dir: Path, provider: str) -> ProviderCredentialRecord | None:
    for path in _candidate_auth_profile_paths(providers_dir):
        store = _load_store(path)
        profiles = store.get("profiles", {})
        if not isinstance(profiles, dict):
            continue

        preferred_id = f"{provider}:default"
        candidates: list[tuple[str, object]] = []
        if preferred_id in profiles:
            candidates.append((preferred_id, profiles[preferred_id]))
        for profile_id, raw in profiles.items():
            if profile_id == preferred_id:
                continue
            if isinstance(raw, dict) and str(raw.get("provider", "")).strip() == provider:
                candidates.append((profile_id, raw))

        for _, raw in candidates:
            record = _decode_profile_record(raw)
            if record is not None:
                return record
    return None


def list_auth_profile_records(providers_dir: Path) -> list[ProviderCredentialRecord]:
    deduped: dict[str, ProviderCredentialRecord] = {}
    for path in _candidate_auth_profile_paths(providers_dir):
        store = _load_store(path)
        profiles = store.get("profiles", {})
        if not isinstance(profiles, dict):
            continue

        for profile_id, raw in profiles.items():
            record = _decode_profile_record(raw)
            if record is None:
                continue
            preferred = deduped.get(record.provider)
            if preferred is None or profile_id == f"{record.provider}:default":
                deduped[record.provider] = record
    return [deduped[key] for key in sorted(deduped)]


def save_auth_profile_record(providers_dir: Path, record: ProviderCredentialRecord) -> Path:
    saved_path: Path | None = None
    for path in _candidate_auth_profile_paths(providers_dir):
        store = _load_store(path)
        profiles = store.setdefault("profiles", {})
        if not isinstance(profiles, dict):
            profiles = {}
            store["profiles"] = profiles
        profiles[f"{record.provider}:default"] = {
            "type": "token",
            "provider": record.provider,
            "token": record.model_dump_json(),
        }
        _save_store(path, store)
        if saved_path is None:
            saved_path = path
    return saved_path or resolve_auth_profiles_path(providers_dir)


def delete_auth_profile_record(providers_dir: Path, provider: str) -> bool:
    to_delete = [
        (
            path,
            [
                profile_id
                for profile_id, raw in profiles.items()
                if profile_id == f"{provider}:default"
                or (isinstance(raw, dict) and str(raw.get("provider", "")).strip() == provider)
            ],
        )
        for path in _candidate_auth_profile_paths(providers_dir)
        for store in [_load_store(path)]
        for profiles in [store.get("profiles", {})]
        if isinstance(profiles, dict)
    ]
    deleted = False
    for path, profile_ids in to_delete:
        if not profile_ids:
            continue
        store = _load_store(path)
        profiles = store.get("profiles", {})
        if not isinstance(profiles, dict):
            continue
        for profile_id in profile_ids:
            profiles.pop(profile_id, None)
        _save_store(path, store)
        deleted = True
    return deleted


def _decode_profile_record(raw: object) -> ProviderCredentialRecord | None:
    if not isinstance(raw, dict):
        return None

    token = raw.get("token")
    if isinstance(token, str) and token.strip():
        try:
            return ProviderCredentialRecord.model_validate_json(token)
        except Exception:
            pass

    try:
        return ProviderCredentialRecord.model_validate(raw)
    except Exception:
        return None


def _load_store(path: Path) -> dict[str, object]:
    if not path.exists():
        return copy.deepcopy(_DEFAULT_STORE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return copy.deepcopy(_DEFAULT_STORE)
    if not isinstance(payload, dict):
        return copy.deepcopy(_DEFAULT_STORE)
    store = copy.deepcopy(_DEFAULT_STORE)
    store.update(payload)
    return store


def _save_store(path: Path, store: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


def _candidate_auth_profile_paths(providers_dir: Path) -> list[Path]:
    paths = [resolve_auth_profiles_path(providers_dir)]
    shared = resolve_shared_auth_profiles_path(providers_dir)
    if shared is not None and shared not in paths:
        paths.append(shared)
    return paths
