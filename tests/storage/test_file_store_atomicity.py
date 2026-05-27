"""file_store blob/metadata atomicity.

create_file commits the JSON metadata and writes the blob bytes together under
a single exclusive lock. Previously the blob was written *after* the lock was
released, opening two windows: a concurrent delete_file could land between the
metadata commit and the blob write (leaving an orphan .bin), and a concurrent
reader could observe metadata-but-no-bytes. These tests pin the invariant that
once create_file returns, the blob is present and readable, with no leftover
temp file.
"""
from __future__ import annotations

import threading
from pathlib import Path

from opentoken.storage import file_store


def test_create_file_blob_present_and_readable(tmp_path: Path) -> None:
    created = file_store.create_file(
        tmp_path,
        filename="a.txt",
        content=b"hello world",
        purpose="assistants",
        mime_type="text/plain",
    )
    result = file_store.read_file_content(tmp_path, created["id"])
    assert result is not None
    _metadata, content = result
    assert content == b"hello world"

    # No stray temp blob left behind by the atomic rename.
    blobs = list((tmp_path / "files").glob("*.tmp"))
    assert blobs == []


def test_create_then_delete_leaves_no_orphan_blob(tmp_path: Path) -> None:
    created = file_store.create_file(
        tmp_path,
        filename="a.txt",
        content=b"data",
        purpose="assistants",
        mime_type="text/plain",
    )
    file_id = created["id"]
    assert file_store.delete_file(tmp_path, file_id) is True

    # Metadata gone and blob gone — invariant: blob exists iff metadata exists.
    assert file_store.get_file(tmp_path, file_id) is None
    assert not (tmp_path / "files" / f"{file_id}.bin").exists()
    assert list((tmp_path / "files").glob("*.tmp")) == []


def test_concurrent_creates_all_persist_with_blobs(tmp_path: Path) -> None:
    """Many concurrent create_file calls (FastAPI threadpool) must each end up
    with both metadata and a readable blob — the lock serialises the
    read-modify-write of files.json and the blob write together."""
    created_ids: list[str] = []
    lock = threading.Lock()

    def _make(i: int) -> None:
        meta = file_store.create_file(
            tmp_path,
            filename=f"f{i}.txt",
            content=f"body-{i}".encode(),
            purpose="assistants",
            mime_type="text/plain",
        )
        with lock:
            created_ids.append(meta["id"])

    threads = [threading.Thread(target=_make, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10.0)

    assert len(created_ids) == 12
    listed = {item["id"] for item in file_store.list_files(tmp_path)}
    assert listed == set(created_ids)
    for file_id in created_ids:
        result = file_store.read_file_content(tmp_path, file_id)
        assert result is not None, f"{file_id} lost its blob"
