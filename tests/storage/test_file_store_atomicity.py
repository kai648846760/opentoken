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

import stat
import sys
import threading
from pathlib import Path

import pytest

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


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX-only file mode bits")
def test_create_file_blob_is_owner_only(tmp_path: Path) -> None:
    """Uploaded content can be sensitive (images, PDFs, source files). The
    blob must be written 0600 so other local users on a shared host can't
    read it. Mode-check via stat.S_IMODE — the rename target inherits the
    chmod we apply to the tmp file before os.replace."""
    created = file_store.create_file(
        tmp_path,
        filename="secret.txt",
        content=b"private",
        purpose="assistants",
        mime_type="text/plain",
    )
    blob = tmp_path / "files" / f"{created['id']}.bin"
    mode = stat.S_IMODE(blob.stat().st_mode)
    assert mode == 0o600, f"blob is {oct(mode)} — should be 0o600 owner-only"


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
