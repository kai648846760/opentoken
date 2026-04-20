"""Failover executor: tries candidates in sequence until one succeeds."""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

from openclaw_algae.failover.errors import (
    NonRetryableError,
    RetryableError,
    classify_error,
    normalize_error,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class FailoverExecutor:
    """Execute work across a list of candidates with bounded retries.

    Tries each candidate in order. If a candidate fails with a retryable error
    and there are more candidates available, tries the next one. Stops on
    non-retryable errors or when max_attempts is reached.
    """

    def __init__(
        self,
        *,
        max_retries: int = 2,
        on_retry: Callable[[object, Exception], None] | None = None,
    ) -> None:
        self._max_retries = max_retries
        self._on_retry = on_retry

    def execute(
        self,
        candidates: list[object],
        work_fn: Callable[[object], T],
    ) -> T:
        """Execute work_fn against candidates until success or exhaustion.

        Args:
            candidates: List of worker candidates, ordered by preference.
            work_fn: Function that takes a candidate and returns a result.

        Returns:
            The result from the first successful candidate.

        Raises:
            NonRetryableError: If a candidate fails with a non-retryable error.
            RetryableError: If all candidates are exhausted with retryable errors.
        """
        if not candidates:
            raise NonRetryableError("No candidates available for failover.")

        max_attempts = min(self._max_retries + 1, len(candidates))
        last_error: Exception | None = None

        for i in range(max_attempts):
            candidate = candidates[i]
            try:
                return work_fn(candidate)
            except NonRetryableError as exc:
                # Non-retryable: don't try other candidates
                logger.error(
                    "failover: Non-retryable error on candidate #%d: %s",
                    i + 1,
                    exc,
                )
                raise
            except RetryableError as exc:
                # Retryable: try next candidate
                last_error = exc
                logger.warning(
                    "failover: Candidate #%d failed (retryable): %s",
                    i + 1,
                    exc,
                )
                if self._on_retry is not None:
                    self._on_retry(candidate, exc)
            except Exception as exc:
                # Unknown: classify and decide
                classification = classify_error(exc)
                if classification.retryable:
                    last_error = normalize_error(exc)
                    logger.warning(
                        "failover: Candidate #%d failed (retryable): %s",
                        i + 1,
                        classification.message,
                    )
                    if self._on_retry is not None:
                        self._on_retry(candidate, exc)
                else:
                    logger.error(
                        "failover: Candidate #%d failed (non-retryable): %s",
                        i + 1,
                        classification.message,
                    )
                    raise NonRetryableError(classification.message, original=exc)

        # All candidates exhausted
        raise RetryableError(
            f"All {max_attempts} candidates failed. "
            f"Last error: {last_error}",
            original=last_error,
        )
