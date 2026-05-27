"""Recovery-profile cleanup for the Camoufox client.

The browser launcher creates a fresh "<name>-recovery-<ms>" sibling directory
each time Firefox refuses to open because another copy is running. Without
cleanup those dirs accumulate until they fill the disk. The sweeper deletes
recovery dirs older than the age threshold; fresh dirs and unrelated siblings
are untouched.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from opentoken.providers.camoufox_clients import _sweep_stale_recovery_dirs


def test_sweep_removes_old_recovery_dirs_keeps_fresh_ones(tmp_path: Path) -> None:
    state = tmp_path / "deepseek"
    state.mkdir()

    old = tmp_path / "deepseek-recovery-111"
    fresh = tmp_path / "deepseek-recovery-222"
    unrelated = tmp_path / "deepseek-other"
    old.mkdir()
    fresh.mkdir()
    unrelated.mkdir()
    # Age out the "old" sibling.
    two_hours_ago = time.time() - 7200
    os.utime(old, (two_hours_ago, two_hours_ago))

    _sweep_stale_recovery_dirs(state, max_age_seconds=3600)

    assert not old.exists()
    assert fresh.exists()
    assert unrelated.exists()
    assert state.exists()


def test_sweep_tolerates_missing_parent(tmp_path: Path) -> None:
    # A state dir whose parent doesn't exist (or isn't readable) should be a
    # silent no-op, not crash the launch.
    state = tmp_path / "does-not-exist" / "deepseek"
    # Don't create anything — just call the sweeper.
    _sweep_stale_recovery_dirs(state, max_age_seconds=3600)  # no exception
