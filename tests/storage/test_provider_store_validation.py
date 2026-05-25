"""save_provider_credentials validator dry-run behaviour."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from opentoken.models.provider_credentials import ProviderCredentialRecord
from opentoken.storage.provider_store import (
    load_provider_credentials,
    save_provider_credentials,
)


def _record(provider: str, marker: str) -> ProviderCredentialRecord:
    return ProviderCredentialRecord(
        provider=provider,
        kind="web_session",
        cookie=f"sess={marker}",
        headers={},
        user_agent="test",
        metadata={},
        status="valid",
    )


def test_save_without_validator_writes_file(tmp_path):
    saved = save_provider_credentials(tmp_path, _record("dummy", "first"))
    assert saved is not None
    loaded = load_provider_credentials(tmp_path, "dummy")
    assert loaded is not None
    assert loaded.cookie == "sess=first"


def test_save_validator_pass_writes_file(tmp_path):
    saved = save_provider_credentials(
        tmp_path,
        _record("dummy", "ok"),
        validator=lambda record: True,
    )
    assert saved is not None
    loaded = load_provider_credentials(tmp_path, "dummy")
    assert loaded is not None
    assert loaded.cookie == "sess=ok"


def test_save_validator_fail_keeps_existing_file(tmp_path):
    # Seed an existing valid record.
    save_provider_credentials(tmp_path, _record("dummy", "good"))

    # Try to overwrite with a record the validator rejects.
    result = save_provider_credentials(
        tmp_path,
        _record("dummy", "bad"),
        validator=lambda record: False,
    )
    assert result is None

    # The original record must still be on disk untouched.
    loaded = load_provider_credentials(tmp_path, "dummy")
    assert loaded is not None
    assert loaded.cookie == "sess=good"


def test_save_validator_exception_treated_as_fail(tmp_path):
    save_provider_credentials(tmp_path, _record("dummy", "good"))

    def crashing_validator(record):
        raise RuntimeError("upstream is down")

    result = save_provider_credentials(
        tmp_path,
        _record("dummy", "replaced"),
        validator=crashing_validator,
    )
    assert result is None

    loaded = load_provider_credentials(tmp_path, "dummy")
    assert loaded is not None
    assert loaded.cookie == "sess=good"
