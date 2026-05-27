from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from opentoken.storage._atomic import file_lock, write_json_atomic
from time import time
from uuid import uuid4

_DEFAULT_STORE: dict[str, object] = {
    "version": 1,
    "uploads": {},
}


def create_upload(
    state_dir: Path,
    *,
    filename: str,
    expected_bytes: int,
    mime_type: str | None,
    purpose: str,
) -> dict[str, object]:
    upload_id = f"upload-{uuid4().hex}"
    metadata = {
        "id": upload_id,
        "object": "upload",
        "created_at": int(time()),
        "filename": filename,
        "bytes": int(expected_bytes),
        "purpose": purpose,
        "mime_type": mime_type or "application/octet-stream",
        "status": "created",
        "parts": [],
    }
    path = _resolve_store_path(state_dir)
    with file_lock(path):
        store = _load_store(path)
        uploads = store.setdefault("uploads", {})
        if not isinstance(uploads, dict):
            uploads = {}
            store["uploads"] = uploads
        uploads[upload_id] = metadata
        _save_store(path, store)
    return _public_upload(metadata)


def get_upload(state_dir: Path, upload_id: str) -> dict[str, object] | None:
    uploads = _load_store(_resolve_store_path(state_dir)).get("uploads", {})
    if not isinstance(uploads, dict):
        return None
    entry = uploads.get(upload_id)
    if not isinstance(entry, dict):
        return None
    return copy.deepcopy(entry)


def add_upload_part(
    state_dir: Path,
    upload_id: str,
    *,
    content: bytes,
    content_type: str | None = None,
) -> dict[str, object] | None:
    path = _resolve_store_path(state_dir)
    # Concurrent parts on the same upload must not lose each other: without a
    # lock both load the same parts list, each appends its own, last write wins.
    with file_lock(path):
        store = _load_store(path)
        uploads = store.get("uploads", {})
        if not isinstance(uploads, dict):
            return None
        entry = uploads.get(upload_id)
        if not isinstance(entry, dict):
            return None
        if str(entry.get("status", "")) != "created":
            return None
        part_id = f"part-{uuid4().hex}"
        part = {
            "id": part_id,
            "object": "upload.part",
            "created_at": int(time()),
            "upload_id": upload_id,
            "bytes": len(content),
            "content_type": content_type or "application/octet-stream",
        }
        parts = entry.setdefault("parts", [])
        if not isinstance(parts, list):
            parts = []
            entry["parts"] = parts
        parts.append(part)
        _save_store(path, store)
        # 0600 owner-only: upload parts hold user-supplied content (images,
        # docs, source files) that must not be world-readable on a shared host.
        blob_path = _resolve_part_blob_path(state_dir, upload_id, part_id)
        blob_path.write_bytes(content)
        try:
            os.chmod(blob_path, 0o600)
        except OSError:
            pass
    return copy.deepcopy(part)


def complete_upload(
    state_dir: Path,
    upload_id: str,
    *,
    part_ids: list[str] | None = None,
) -> tuple[dict[str, object], bytes] | None:
    path = _resolve_store_path(state_dir)
    with file_lock(path):
        store = _load_store(path)
        uploads = store.get("uploads", {})
        if not isinstance(uploads, dict):
            return None
        entry = uploads.get(upload_id)
        if not isinstance(entry, dict):
            return None
        if str(entry.get("status", "")) != "created":
            return None
        parts = entry.get("parts", [])
        if not isinstance(parts, list):
            return None

        ordered_parts: list[dict[str, object]] = []
        if part_ids:
            part_lookup = {
                str(part.get("id", "")): part
                for part in parts
                if isinstance(part, dict)
            }
            for part_id in part_ids:
                part = part_lookup.get(part_id)
                if not isinstance(part, dict):
                    return None
                ordered_parts.append(part)
        else:
            ordered_parts = [part for part in parts if isinstance(part, dict)]

        part_blob_paths = [
            _resolve_part_blob_path(state_dir, upload_id, str(part.get("id", "")))
            for part in ordered_parts
        ]
        content = b"".join(blob_path.read_bytes() for blob_path in part_blob_paths)
        entry["status"] = "completed"
        entry["completed_at"] = int(time())
        _save_store(path, store)

        # Free the per-part blobs once the upload is materialised — leaving them around
        # accumulates per-upload temp files forever and lets a duplicated upload_id step
        # on stale data.
        for blob_path in part_blob_paths:
            try:
                blob_path.unlink(missing_ok=True)
            except OSError:
                pass

    return _public_upload(entry), content


def cancel_upload(state_dir: Path, upload_id: str) -> dict[str, object] | None:
    path = _resolve_store_path(state_dir)
    with file_lock(path):
        store = _load_store(path)
        uploads = store.get("uploads", {})
        if not isinstance(uploads, dict):
            return None
        entry = uploads.get(upload_id)
        if not isinstance(entry, dict):
            return None
        entry["status"] = "cancelled"
        entry["cancelled_at"] = int(time())
        _save_store(path, store)
    return _public_upload(entry)


def _public_upload(entry: dict[str, object]) -> dict[str, object]:
    parts = entry.get("parts", [])
    part_count = len(parts) if isinstance(parts, list) else 0
    return {
        "id": str(entry.get("id", "")),
        "object": "upload",
        "created_at": int(entry.get("created_at", 0)),
        "filename": str(entry.get("filename", "")),
        "bytes": int(entry.get("bytes", 0)),
        "purpose": str(entry.get("purpose", "")),
        "mime_type": str(entry.get("mime_type", "application/octet-stream")),
        "status": str(entry.get("status", "created")),
        "part_count": part_count,
    }


def _resolve_store_path(state_dir: Path) -> Path:
    return state_dir / "uploads.json"


def _resolve_part_blob_path(state_dir: Path, upload_id: str, part_id: str) -> Path:
    blob_path = state_dir / "uploads" / upload_id / f"{part_id}.bin"
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
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, store)
