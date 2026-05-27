"""Storage modules must survive corrupt / non-dict / truncated JSON on disk.

A crash mid-write (pre-atomic-write era) or external tampering can leave a
store file unparseable. Every store's _load_store should degrade to the
default empty store rather than raise, so the gateway keeps working (and the
next atomic write heals the file).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from opentoken.models.provider_credentials import ProviderCredentialRecord


def _corrupt(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.mark.parametrize("bad", ['{"truncated":', "not json at all", "[1,2,3]", "null", "12345"])
def test_file_store_survives_corrupt_store(tmp_path: Path, bad: str) -> None:
    from opentoken.storage import file_store

    _corrupt(tmp_path / "files.json", bad)
    # Listing should not raise; it returns an empty list for a corrupt store.
    assert file_store.list_files(tmp_path) == []
    # And a subsequent write heals it.
    created = file_store.create_file(
        tmp_path, filename="a.txt", content=b"hi", purpose="assistants", mime_type="text/plain"
    )
    assert created["id"].startswith("file-")
    assert any(item["id"] == created["id"] for item in file_store.list_files(tmp_path))


@pytest.mark.parametrize("bad", ['{"x":', "garbage", "[]", "true"])
def test_response_store_survives_corrupt_store(tmp_path: Path, bad: str) -> None:
    from opentoken.storage import response_store

    _corrupt(tmp_path / "responses.json", bad)
    assert response_store.load_response_messages(tmp_path, "resp-x") is None
    response_store.save_response_messages(
        tmp_path, response_id="resp-1", model="m", messages=[{"role": "user", "content": "hi"}]
    )
    loaded = response_store.load_response_messages(tmp_path, "resp-1")
    assert loaded == [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize("bad", ['{"profiles":', "nope", "[]", "42"])
def test_auth_profiles_survives_corrupt_store(tmp_path: Path, bad: str) -> None:
    from opentoken.storage import auth_profiles

    providers_dir = tmp_path / "providers"
    providers_dir.mkdir(parents=True, exist_ok=True)
    _corrupt(auth_profiles.resolve_auth_profiles_path(providers_dir), bad)

    assert auth_profiles.load_auth_profile_record(providers_dir, "deepseek") is None

    record = ProviderCredentialRecord(
        provider="deepseek",
        kind="web_session",
        cookie="c",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )
    auth_profiles.save_auth_profile_record(providers_dir, record)
    loaded = auth_profiles.load_auth_profile_record(providers_dir, "deepseek")
    assert loaded is not None
    assert loaded.cookie == "c"


@pytest.mark.parametrize("bad", ['{"uploads":', "??", "[]", "false"])
def test_upload_store_survives_corrupt_store(tmp_path: Path, bad: str) -> None:
    from opentoken.storage import upload_store

    _corrupt(tmp_path / "uploads.json", bad)
    # A corrupt store yields no upload; creating a new one heals it.
    created = upload_store.create_upload(
        tmp_path, filename="j.txt", expected_bytes=2, purpose="assistants", mime_type="text/plain"
    )
    assert created["id"].startswith("upload")


def test_provider_sessions_concurrent_saves_do_not_lose_updates(tmp_path) -> None:
    """save_provider_session locks its read-modify-write so concurrent saves
    for different providers don't clobber each other (lost-update race)."""
    import threading
    from opentoken.storage import provider_sessions

    def creds(provider: str) -> ProviderCredentialRecord:
        return ProviderCredentialRecord(
            provider=provider,
            kind="web_session",
            cookie=f"c-{provider}",
            headers={},
            user_agent="ua",
            metadata={},
            status="valid",
        )

    providers = [f"prov{i}" for i in range(12)]
    barrier = threading.Barrier(len(providers))

    def worker(provider: str) -> None:
        barrier.wait()  # maximise overlap
        provider_sessions.save_provider_session(
            tmp_path,
            provider=provider,
            credentials=creds(provider),
            state={"chat_id": provider},
        )

    threads = [threading.Thread(target=worker, args=(p,)) for p in providers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Every provider's session must have survived — no lost updates.
    for p in providers:
        loaded = provider_sessions.load_provider_session(tmp_path, provider=p, credentials=creds(p))
        assert loaded.get("chat_id") == p
