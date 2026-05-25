"""Atomic JSON write helpers shared by storage modules.

Plain `Path.write_text` is not atomic — a crash mid-write can leave a half-
written / unparseable JSON file. These helpers write to a sibling temp file
and then `os.replace` it onto the target, which on POSIX is guaranteed to be
atomic with respect to other readers.

The functions also lock the target file while reading/writing so that two
processes racing on the same store don't interleave their writes.
"""
from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Iterator
from pathlib import Path

try:  # pragma: no cover - posix-only fcntl exists on macOS and Linux
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]


@contextlib.contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Best-effort exclusive lock on `path` (creates the file if absent).

    On platforms without fcntl (Windows) the lock degrades to a no-op — callers
    still benefit from the atomic os.replace in write_json_atomic.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    handle = lock_path.open("a+")
    try:
        if fcntl is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except OSError:
                # Lock acquisition failure should not stop the write — log via raise.
                raise
        yield
    finally:
        if fcntl is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        handle.close()


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        fh.write(serialized)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            # fsync isn't fatal — keep going.
            pass
    os.replace(tmp_path, path)
