from pathlib import Path

from openclaw_algae.models.provider_credentials import ProviderCredentialRecord
from openclaw_algae.storage.provider_store import (
    list_provider_credentials,
    load_provider_credentials,
    save_provider_credentials,
)


def test_save_and_load_provider_credentials(tmp_path: Path) -> None:
    record = ProviderCredentialRecord(
        provider="deepseek",
        kind="web_session",
        cookie="session=value",
        headers={"authorization": "Bearer token"},
        user_agent="ua",
        metadata={"xsrf_token": "abc", "ut": "user-1"},
        status="valid",
    )

    save_provider_credentials(tmp_path, record)
    loaded = load_provider_credentials(tmp_path, "deepseek")

    assert loaded is not None
    assert loaded.provider == "deepseek"
    assert loaded.cookie == "session=value"
    assert loaded.metadata["xsrf_token"] == "abc"


def test_load_provider_credentials_returns_none_for_invalid_json(tmp_path: Path) -> None:
    (tmp_path / "deepseek.json").write_text("{bad json", encoding="utf-8")

    loaded = load_provider_credentials(tmp_path, "deepseek")

    assert loaded is None


def test_list_provider_credentials_skips_invalid_files(tmp_path: Path) -> None:
    save_provider_credentials(
        tmp_path,
        ProviderCredentialRecord(
            provider="deepseek",
            kind="web_session",
            cookie="session=value",
            headers={"authorization": "Bearer token"},
            user_agent="ua",
            metadata={},
            status="valid",
        ),
    )
    (tmp_path / "broken.json").write_text("{bad json", encoding="utf-8")

    records = list_provider_credentials(tmp_path)

    assert [record.provider for record in records] == ["deepseek"]
