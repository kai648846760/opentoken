import json
from pathlib import Path

from pydantic import ValidationError

from openclaw_algae.models.provider_credentials import ProviderCredentialRecord
from openclaw_algae.storage.auth_profiles import (
    delete_auth_profile_record,
    list_auth_profile_records,
    load_auth_profile_record,
    save_auth_profile_record,
)


def _provider_path(state_dir: Path, provider: str) -> Path:
    return state_dir / f"{provider}.json"


def save_provider_credentials(state_dir: Path, record: ProviderCredentialRecord) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    target = _provider_path(state_dir, record.provider)
    target.write_text(json.dumps(record.model_dump(), indent=2), encoding="utf-8")
    save_auth_profile_record(state_dir, record)
    return target


def load_provider_credentials(state_dir: Path, provider: str) -> ProviderCredentialRecord | None:
    auth_record = load_auth_profile_record(state_dir, provider)
    if auth_record is not None:
        return auth_record
    target = _provider_path(state_dir, provider)
    if not target.exists():
        return None
    return _load_record(target)


def list_provider_credentials(state_dir: Path) -> list[ProviderCredentialRecord]:
    records_by_provider = {
        record.provider: record for record in list_auth_profile_records(state_dir)
    }
    if state_dir.exists():
        for path in sorted(state_dir.glob("*.json")):
            record = _load_record(path)
            if record is not None and record.provider not in records_by_provider:
                records_by_provider[record.provider] = record
    return [records_by_provider[key] for key in sorted(records_by_provider)]


def delete_provider_credentials(state_dir: Path, provider: str) -> bool:
    deleted = delete_auth_profile_record(state_dir, provider)
    target = _provider_path(state_dir, provider)
    if target.exists():
        target.unlink()
        deleted = True
    return deleted


def _load_record(path: Path) -> ProviderCredentialRecord | None:
    try:
        return ProviderCredentialRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, json.JSONDecodeError):
        return None
