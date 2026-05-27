"""Owner-only (0600) permission checks for stores holding user data.

Conversation history (response_store) and uploaded content (upload_store)
are user data that must not be world-readable on a shared host. These tests
pin the file mode bits so a future refactor can't silently drop sensitive=True
or the blob chmod.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="POSIX-only file mode bits"
)


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def test_response_store_is_owner_only(tmp_path: Path) -> None:
    from opentoken.storage import response_store

    response_store.save_response_messages(
        tmp_path,
        response_id="resp-1",
        model="m",
        messages=[{"role": "user", "content": "my secret prompt"}],
    )
    store_path = tmp_path / "responses.json"
    assert store_path.exists()
    assert _mode(store_path) & 0o077 == 0, (
        f"responses.json is too permissive: {oct(_mode(store_path))}"
    )


def test_upload_part_blob_is_owner_only(tmp_path: Path) -> None:
    from opentoken.storage import upload_store

    upload = upload_store.create_upload(
        tmp_path,
        filename="a.bin",
        purpose="assistants",
        expected_bytes=4,
        mime_type="application/octet-stream",
    )
    part = upload_store.add_upload_part(
        tmp_path,
        upload["id"],
        content=b"data",
        content_type="application/octet-stream",
    )
    assert part is not None

    blobs = list((tmp_path / "uploads").rglob("*")) if (tmp_path / "uploads").exists() else []
    blob_files = [p for p in blobs if p.is_file()]
    assert blob_files, "expected an upload part blob on disk"
    for blob in blob_files:
        assert _mode(blob) & 0o077 == 0, f"{blob} too permissive: {oct(_mode(blob))}"
