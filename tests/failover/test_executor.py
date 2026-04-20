"""Tests for failover executor."""
import pytest

from openclaw_algae.failover.errors import (
    NonRetryableError,
    RetryableError,
)
from openclaw_algae.failover.executor import FailoverExecutor


def test_execute_first_succeeds() -> None:
    executor = FailoverExecutor(max_retries=2)
    calls = []

    def work_fn(candidate):
        calls.append(candidate)
        return f"result-{candidate}"

    result = executor.execute(["a", "b", "c"], work_fn)
    assert result == "result-a"
    assert calls == ["a"]


def test_execute_second_succeeds() -> None:
    executor = FailoverExecutor(max_retries=2)
    calls = []

    def work_fn(candidate):
        calls.append(candidate)
        if candidate == "a":
            raise RetryableError("fail-a")
        return f"result-{candidate}"

    result = executor.execute(["a", "b", "c"], work_fn)
    assert result == "result-b"
    assert calls == ["a", "b"]


def test_execute_all_fail() -> None:
    executor = FailoverExecutor(max_retries=2)

    def work_fn(candidate):
        raise RetryableError(f"fail-{candidate}")

    with pytest.raises(RetryableError, match="All 3 candidates failed"):
        executor.execute(["a", "b", "c"], work_fn)


def test_execute_non_retryable_stops() -> None:
    executor = FailoverExecutor(max_retries=2)
    calls = []

    def work_fn(candidate):
        calls.append(candidate)
        if candidate == "a":
            raise NonRetryableError("captcha")
        return f"result-{candidate}"

    with pytest.raises(NonRetryableError, match="captcha"):
        executor.execute(["a", "b", "c"], work_fn)
    assert calls == ["a"]  # Only tried "a", stopped immediately


def test_execute_no_candidates() -> None:
    executor = FailoverExecutor(max_retries=2)

    with pytest.raises(NonRetryableError, match="No candidates"):
        executor.execute([], lambda _: "ok")


def test_execute_max_retries_bounds_candidates() -> None:
    """With max_retries=1 and 5 candidates, should try at most 2."""
    executor = FailoverExecutor(max_retries=1)
    calls = []

    def work_fn(candidate):
        calls.append(candidate)
        raise RetryableError(f"fail-{candidate}")

    with pytest.raises(RetryableError):
        executor.execute(["a", "b", "c", "d", "e"], work_fn)
    assert len(calls) == 2  # max_retries(1) + 1 = 2


def test_execute_on_retry_callback() -> None:
    retries = []
    executor = FailoverExecutor(
        max_retries=2,
        on_retry=lambda c, e: retries.append((c, str(e))),
    )

    def work_fn(candidate):
        if candidate != "c":
            raise RetryableError(f"fail-{candidate}")
        return f"result-{candidate}"

    result = executor.execute(["a", "b", "c"], work_fn)
    assert result == "result-c"
    assert len(retries) == 2
    assert retries[0] == ("a", "fail-a")
    assert retries[1] == ("b", "fail-b")


def test_execute_unknown_error_classified() -> None:
    """Unknown errors are classified and may trigger failover."""
    executor = FailoverExecutor(max_retries=2)
    calls = []

    def work_fn(candidate):
        calls.append(candidate)
        if candidate == "a":
            raise Exception("connection reset")  # retryable
        return f"result-{candidate}"

    result = executor.execute(["a", "b"], work_fn)
    assert result == "result-b"
    assert calls == ["a", "b"]


def test_execute_unknown_non_retryable_stops() -> None:
    """Unknown non-retryable errors should stop failover."""
    executor = FailoverExecutor(max_retries=2)
    calls = []

    def work_fn(candidate):
        calls.append(candidate)
        if candidate == "a":
            raise Exception("captcha required")  # non-retryable
        return f"result-{candidate}"

    with pytest.raises(NonRetryableError):
        executor.execute(["a", "b"], work_fn)
    assert calls == ["a"]
