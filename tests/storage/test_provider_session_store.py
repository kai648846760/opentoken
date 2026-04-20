from __future__ import annotations

from pathlib import Path

from openclaw_algae.models.provider_credentials import ProviderCredentialRecord
from openclaw_algae.storage.provider_sessions import (
    load_provider_session,
    save_provider_session,
)


def test_provider_session_store_roundtrips_conversation_state(tmp_path: Path) -> None:
    credentials = ProviderCredentialRecord(
        provider="chatgpt",
        kind="browser_session",
        cookie="session=abc",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    save_provider_session(
        tmp_path,
        provider="chatgpt",
        credentials=credentials,
        state={"conversation_id": "conv-1", "parent_message_id": "msg-1"},
    )

    loaded = load_provider_session(tmp_path, provider="chatgpt", credentials=credentials)

    assert loaded == {"conversation_id": "conv-1", "parent_message_id": "msg-1"}


def test_provider_session_store_is_isolated_by_credential_fingerprint(tmp_path: Path) -> None:
    credentials_one = ProviderCredentialRecord(
        provider="qwen-intl",
        kind="browser_session",
        cookie="token=one",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )
    credentials_two = ProviderCredentialRecord(
        provider="qwen-intl",
        kind="browser_session",
        cookie="token=two",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )

    save_provider_session(tmp_path, provider="qwen-intl", credentials=credentials_one, state={"chat_id": "chat-1"})

    assert load_provider_session(tmp_path, provider="qwen-intl", credentials=credentials_one) == {"chat_id": "chat-1"}
    assert load_provider_session(tmp_path, provider="qwen-intl", credentials=credentials_two) == {}
