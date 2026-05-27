from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from time import time
from uuid import uuid4

from opentoken.storage._atomic import file_lock, write_json_atomic


_SAFE_FILE_ID = re.compile(r"^file-[A-Za-z0-9]{16,}$")

_DEFAULT_STORE: dict[str, object] = {
    "version": 1,
    "files": {},
}


def create_file(
    state_dir: Path,
    *,
    filename: str,
    content: bytes,
    purpose: str,
    mime_type: str | None = None,
) -> dict[str, object]:
    file_id = f"file-{uuid4().hex}"
    metadata = {
        "id": file_id,
        "object": "file",
        "bytes": len(content),
        "created_at": int(time()),
        "filename": filename,
        "purpose": purpose,
        "status": "processed",
        "mime_type": mime_type or "application/octet-stream",
    }
    path = _resolve_store_path(state_dir)
    blob_path = _resolve_blob_path(state_dir, file_id)
    with file_lock(path):
        # Write blob and commit metadata together under the lock. Without this
        # a concurrent delete_file between metadata commit and blob write would
        # leave an orphan .bin (delete's unlink runs while blob doesn't yet
        # exist, then create's write_bytes lands), and a concurrent reader
        # would see metadata-but-no-bytes. Atomic tmp+rename for the blob keeps
        # readers from observing a partial write even within the lock window.
        tmp_blob = blob_path.with_name(blob_path.name + ".tmp")
        tmp_blob.write_bytes(content)
        os.replace(tmp_blob, blob_path)
        store = _load_store(path)
        files = store.setdefault("files", {})
        if not isinstance(files, dict):
            files = {}
            store["files"] = files
        files[file_id] = copy.deepcopy(metadata)
        _save_store(path, store)
    return copy.deepcopy(metadata)


def list_files(state_dir: Path) -> list[dict[str, object]]:
    store = _load_store(_resolve_store_path(state_dir))
    files = store.get("files", {})
    if not isinstance(files, dict):
        return []
    items = [
        _public_metadata(value)
        for value in files.values()
        if isinstance(value, dict)
    ]
    return sorted(items, key=lambda item: (int(item.get("created_at", 0)), str(item.get("id", ""))))


def get_file(state_dir: Path, file_id: str) -> dict[str, object] | None:
    store = _load_store(_resolve_store_path(state_dir))
    files = store.get("files", {})
    if not isinstance(files, dict):
        return None
    entry = files.get(file_id)
    if not isinstance(entry, dict):
        return None
    return _public_metadata(entry)


def read_file_content(state_dir: Path, file_id: str) -> tuple[dict[str, object], bytes] | None:
    metadata = get_file(state_dir, file_id)
    if metadata is None:
        return None
    blob_path = _resolve_blob_path(state_dir, file_id)
    if not blob_path.exists():
        return None
    return metadata, blob_path.read_bytes()


def delete_file(state_dir: Path, file_id: str) -> bool:
    if not _SAFE_FILE_ID.match(file_id):
        return False
    path = _resolve_store_path(state_dir)
    with file_lock(path):
        store = _load_store(path)
        files = store.get("files", {})
        if not isinstance(files, dict) or file_id not in files:
            return False
        files.pop(file_id, None)
        _save_store(path, store)
    blob_path = _resolve_blob_path(state_dir, file_id)
    try:
        blob_path.unlink(missing_ok=True)
    except OSError:
        pass
    return True


def _public_metadata(entry: dict[str, object]) -> dict[str, object]:
    return {
        "id": str(entry.get("id", "")),
        "object": "file",
        "bytes": int(entry.get("bytes", 0)),
        "created_at": int(entry.get("created_at", 0)),
        "filename": str(entry.get("filename", "")),
        "purpose": str(entry.get("purpose", "")),
        "status": str(entry.get("status", "processed")),
        "mime_type": str(entry.get("mime_type", "application/octet-stream")),
    }


def _resolve_store_path(state_dir: Path) -> Path:
    return state_dir / "files.json"


def _resolve_blob_path(state_dir: Path, file_id: str) -> Path:
    # Reject anything that doesn't look like an id we minted — this is the only
    # surface that turns a caller-controlled string into a filesystem path, so
    # treat it as untrusted and refuse traversal attempts (``../``, embedded
    # ``/`` separators, NULs, empty strings).
    if not _SAFE_FILE_ID.match(file_id):
        raise ValueError(f"Refusing to resolve unsafe file id: {file_id!r}")
    blob_path = state_dir / "files" / f"{file_id}.bin"
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    return blob_path


def _load_store(path: Path) -> dict[str, object]:
    if not path.exists():
        return copy.deepcopy(_DEFAULT_STORE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return copy.deepcopy(_DEFAULT_STORE)
    if not isinstance(payload, dict):
        return copy.deepcopy(_DEFAULT_STORE)
    store = copy.deepcopy(_DEFAULT_STORE)
    store.update(payload)
    return store


def _save_store(path: Path, store: dict[str, object]) -> None:
    write_json_atomic(path, store)
