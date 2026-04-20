from __future__ import annotations

import json
from pathlib import Path

from openclaw_algae.models.provider_credentials import ProviderCredentialRecord
from openclaw_algae.storage.provider_store import (
    list_provider_credentials,
    load_provider_credentials,
    save_provider_credentials,
)


def test_save_provider_credentials_writes_reference_style_auth_profiles_store(tmp_path: Path) -> None:
    providers_dir = tmp_path / "providers"
    record = ProviderCredentialRecord(
        provider="qwen-intl",
        kind="browser_session",
        cookie="token=value",
        headers={"authorization": "Bearer x"},
        user_agent="ua",
        metadata={"session_token": "token-value"},
        status="valid",
    )

    save_provider_credentials(providers_dir, record)

    auth_store = json.loads((tmp_path / "auth-profiles.json").read_text(encoding="utf-8"))
    assert auth_store["version"] >= 1
    assert auth_store["profiles"]["qwen-intl:default"]["type"] == "token"
    assert auth_store["profiles"]["qwen-intl:default"]["provider"] == "qwen-intl"

    saved_token = auth_store["profiles"]["qwen-intl:default"]["token"]
    restored = ProviderCredentialRecord.model_validate_json(saved_token)
    assert restored.provider == "qwen-intl"
    assert restored.metadata["session_token"] == "token-value"


def test_load_provider_credentials_reads_from_auth_profiles_store_without_legacy_file(
    tmp_path: Path,
) -> None:
    providers_dir = tmp_path / "providers"
    providers_dir.mkdir(parents=True)
    record = ProviderCredentialRecord(
        provider="claude",
        kind="browser_session",
        cookie="sessionKey=abc",
        headers={},
        user_agent="ua",
        metadata={"session_key": "abc"},
        status="valid",
    )
    (tmp_path / "auth-profiles.json").write_text(
        json.dumps(
            {
                "version": 1,
                "profiles": {
                    "claude:default": {
                        "type": "token",
                        "provider": "claude",
                        "token": record.model_dump_json(),
                    }
                },
                "order": {},
                "lastGood": {},
                "usageStats": {},
            }
        ),
        encoding="utf-8",
    )

    loaded = load_provider_credentials(providers_dir, "claude")

    assert loaded is not None
    assert loaded.provider == "claude"
    assert loaded.metadata["session_key"] == "abc"


def test_save_provider_credentials_mirrors_shared_openclaw_auth_profiles_for_default_layout(
    tmp_path: Path,
) -> None:
    providers_dir = tmp_path / ".openclaw-algae" / "providers"
    record = ProviderCredentialRecord(
        provider="chatgpt",
        kind="browser_session",
        cookie="__Secure-next-auth.session-token=abc",
        headers={},
        user_agent="ua",
        metadata={"session_token": "abc"},
        status="valid",
    )

    save_provider_credentials(providers_dir, record)

    shared_store = json.loads(
        (tmp_path / ".openclaw" / "auth-profiles.json").read_text(encoding="utf-8")
    )
    assert shared_store["profiles"]["chatgpt:default"]["provider"] == "chatgpt"


def test_load_provider_credentials_reads_from_shared_openclaw_auth_profiles_store(
    tmp_path: Path,
) -> None:
    providers_dir = tmp_path / ".openclaw-algae" / "providers"
    providers_dir.mkdir(parents=True)
    record = ProviderCredentialRecord(
        provider="gemini",
        kind="browser_session",
        cookie="SID=value",
        headers={},
        user_agent="ua",
        metadata={"sid": "value"},
        status="valid",
    )
    shared_path = tmp_path / ".openclaw" / "auth-profiles.json"
    shared_path.parent.mkdir(parents=True, exist_ok=True)
    shared_path.write_text(
        json.dumps(
            {
                "version": 1,
                "profiles": {
                    "gemini:default": {
                        "type": "token",
                        "provider": "gemini",
                        "token": record.model_dump_json(),
                    }
                },
                "order": {},
                "lastGood": {},
                "usageStats": {},
            }
        ),
        encoding="utf-8",
    )

    loaded = load_provider_credentials(providers_dir, "gemini")

    assert loaded is not None
    assert loaded.provider == "gemini"
    assert loaded.metadata["sid"] == "value"


def test_list_provider_credentials_merges_auth_profiles_and_legacy_without_duplicates(
    tmp_path: Path,
) -> None:
    providers_dir = tmp_path / "providers"
    providers_dir.mkdir(parents=True)

    legacy = ProviderCredentialRecord(
        provider="deepseek",
        kind="browser_session",
        cookie="session=value",
        headers={"authorization": "Bearer t"},
        user_agent="ua",
        metadata={},
        status="valid",
    )
    (providers_dir / "deepseek.json").write_text(legacy.model_dump_json(indent=2), encoding="utf-8")

    from_store = ProviderCredentialRecord(
        provider="qwen-cn",
        kind="browser_session",
        cookie="xsrf=value",
        headers={},
        user_agent="ua",
        metadata={"xsrf_token": "value"},
        status="valid",
    )
    (tmp_path / "auth-profiles.json").write_text(
        json.dumps(
            {
                "version": 1,
                "profiles": {
                    "qwen-cn:default": {
                        "type": "token",
                        "provider": "qwen-cn",
                        "token": from_store.model_dump_json(),
                    },
                    "deepseek:default": {
                        "type": "token",
                        "provider": "deepseek",
                        "token": legacy.model_dump_json(),
                    },
                },
                "order": {},
                "lastGood": {},
                "usageStats": {},
            }
        ),
        encoding="utf-8",
    )

    records = list_provider_credentials(providers_dir)

    assert [record.provider for record in records] == ["deepseek", "qwen-cn"]
