"""Tests for BrowserWorker lifecycle."""
from pathlib import Path

from openclaw_algae.pool.types import WorkerIdentity, WorkerState
from openclaw_algae.pool.worker import BrowserWorker


def test_worker_initial_state() -> None:
    worker = BrowserWorker(
        identity=WorkerIdentity(name="test_worker", provider_type="doubao", instance_name="default"),
        supported_models=["doubao-seed-2.0"],
        user_data_dir=Path("/tmp/test"),
    )
    assert worker.state == WorkerState.IDLE
    assert worker.busy_count == 0
    assert worker.name == "test_worker"
    assert worker.provider_type == "doubao"


def test_worker_acquire_release() -> None:
    worker = BrowserWorker(
        identity=WorkerIdentity(name="test", provider_type="doubao", instance_name="default"),
        supported_models=["doubao-seed-2.0"],
        user_data_dir=Path("/tmp/test"),
    )
    worker.acquire()
    assert worker.state == WorkerState.BUSY
    assert worker.busy_count == 1

    worker.acquire()
    assert worker.busy_count == 2

    worker.release()
    assert worker.busy_count == 1

    worker.release()
    assert worker.state == WorkerState.IDLE
    assert worker.busy_count == 0


def test_worker_mark_crashed() -> None:
    worker = BrowserWorker(
        identity=WorkerIdentity(name="test", provider_type="doubao", instance_name="default"),
        supported_models=["doubao-seed-2.0"],
        user_data_dir=Path("/tmp/test"),
    )
    worker.acquire()
    worker.mark_crashed()
    assert worker.state == WorkerState.CRASHED
    assert worker.busy_count == 0


def test_worker_supports_model() -> None:
    worker = BrowserWorker(
        identity=WorkerIdentity(name="test", provider_type="doubao", instance_name="default"),
        supported_models=["doubao-seed-2.0", "doubao-pro"],
        user_data_dir=Path("/tmp/test"),
    )
    assert worker.supports("doubao-seed-2.0") is True
    assert worker.supports("doubao-pro") is True
    assert worker.supports("algae/doubao/doubao-seed-2.0") is True
    assert worker.supports("deepseek-chat") is False


def test_worker_get_models() -> None:
    worker = BrowserWorker(
        identity=WorkerIdentity(name="test", provider_type="doubao", instance_name="default"),
        supported_models=["doubao-seed-2.0", "doubao-pro"],
        user_data_dir=Path("/tmp/test"),
    )
    models = worker.get_models()
    assert models == ["doubao-seed-2.0", "doubao-pro"]


def test_worker_shutdown() -> None:
    worker = BrowserWorker(
        identity=WorkerIdentity(name="test", provider_type="doubao", instance_name="default"),
        supported_models=["doubao-seed-2.0"],
        user_data_dir=Path("/tmp/test"),
    )
    worker.shutdown()
    assert worker.state == WorkerState.SHUTDOWN
    assert worker.page is None


def test_worker_repr() -> None:
    worker = BrowserWorker(
        identity=WorkerIdentity(name="my_worker", provider_type="doubao", instance_name="default"),
        supported_models=["doubao-seed-2.0"],
        user_data_dir=Path("/tmp/test"),
    )
    assert "my_worker" in repr(worker)
    assert "idle" in repr(worker).lower()
