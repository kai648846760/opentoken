import json
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from opentoken.storage._atomic import file_lock, write_json_atomic
from opentoken.models.provider_credentials import ProviderCredentialRecord
from opentoken.storage.auth_profiles import (
    delete_auth_profile_record,
    list_auth_profile_records,
    load_auth_profile_record,
    save_auth_profile_record,
)


def _provider_path(state_dir: Path, provider: str) -> Path:
    return state_dir / f"{provider}.json"


def save_provider_credentials(
    state_dir: Path,
    record: ProviderCredentialRecord,
    *,
    validator: Callable[[ProviderCredentialRecord], bool] | None = None,
) -> Path | None:
    """Persist a provider credential record.

    If a `validator` is provided, it must return True before any existing record
    is overwritten — if validation fails the old credentials are kept and this
    function returns None. This is the dry-run-before-overwrite contract used
    after browser harvest, so a botched harvest can't replace a previously-good
    cookie with a broken one.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    if validator is not None:
        try:
            ok = bool(validator(record))
        except Exception:
            ok = False
        if not ok:
            return None
    target = _provider_path(state_dir, record.provider)
    # save and delete each update two stores (<provider>.json + the auth-profile
    # store). Hold a per-provider lock across both so a concurrent re-login and
    # logout of the same provider can't interleave and leave them diverged
    # (e.g. json written but profile deleted), which load_provider_credentials
    # would then read inconsistently. The provider-path lock is always the
    # OUTER lock in both functions, so there's no ordering deadlock with the
    # auth-profile store's own internal lock.
    with file_lock(target):
        write_json_atomic(target, record.model_dump(), sensitive=True)
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
    target = _provider_path(state_dir, provider)
    with file_lock(target):
        deleted = delete_auth_profile_record(state_dir, provider)
        if target.exists():
            target.unlink()
            deleted = True
    return deleted


def _load_record(path: Path) -> ProviderCredentialRecord | None:
    try:
        return ProviderCredentialRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, json.JSONDecodeError):
        return None
