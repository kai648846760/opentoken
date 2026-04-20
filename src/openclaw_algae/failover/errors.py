"""Error classification for failover decisions."""
from __future__ import annotations

import re
from dataclasses import dataclass


class RetryableError(Exception):
    """Error that should trigger failover to the next candidate."""

    def __init__(self, message: str, *, original: BaseException | None = None) -> None:
        super().__init__(message)
        self.original = original


class NonRetryableError(Exception):
    """Error that should NOT trigger failover (permanent failure)."""

    def __init__(self, message: str, *, original: BaseException | None = None) -> None:
        super().__init__(message)
        self.original = original


@dataclass(frozen=True)
class ErrorClassification:
    retryable: bool
    code: str
    message: str


# Patterns that indicate retryable errors
_RETRYABLE_PATTERNS = [
    re.compile(r"timeout", re.IGNORECASE),
    re.compile(r"timed?\s*out", re.IGNORECASE),
    re.compile(r"connection\s+(reset|refused|aborted|closed)", re.IGNORECASE),
    re.compile(r"network\s+error", re.IGNORECASE),
    re.compile(r"5\d{2}", re.IGNORECASE),
    re.compile(r"rate\s*limit", re.IGNORECASE),
    re.compile(r"too\s*many\s*requests", re.IGNORECASE),
    re.compile(r"page\s*(is\s*)?closed", re.IGNORECASE),
    re.compile(r"browser\s*(is\s*)?closed", re.IGNORECASE),
    re.compile(r"crashed", re.IGNORECASE),
    re.compile(r"session\s+(expired|invalid)", re.IGNORECASE),
    re.compile(r"401", re.IGNORECASE),
    re.compile(r"403", re.IGNORECASE),
]

# Patterns that indicate non-retryable errors
_NON_RETRYABLE_PATTERNS = [
    re.compile(r"captcha", re.IGNORECASE),
    re.compile(r"blocked", re.IGNORECASE),
    re.compile(r"content\s+policy", re.IGNORECASE),
    re.compile(r"account\s+(suspend|disabled|banned)", re.IGNORECASE),
    re.compile(r"unsupported", re.IGNORECASE),
    re.compile(r"invalid\s+(model|provider|request)", re.IGNORECASE),
    re.compile(r"not\s+found", re.IGNORECASE),
    re.compile(r"missing\s+(credential|token|cookie)", re.IGNORECASE),
]


def classify_error(exc: Exception) -> ErrorClassification:
    """Classify an exception as retryable or non-retryable."""
    message = str(exc)
    error_type = type(exc).__name__
    full_text = f"{error_type}: {message}"

    # Check non-retryable first (more specific patterns)
    for pattern in _NON_RETRYABLE_PATTERNS:
        if pattern.search(full_text):
            return ErrorClassification(
                retryable=False,
                code="non_retryable",
                message=message,
            )

    # Check retryable patterns
    for pattern in _RETRYABLE_PATTERNS:
        if pattern.search(full_text):
            return ErrorClassification(
                retryable=True,
                code="retryable",
                message=message,
            )

    # Unknown errors are non-retryable by default
    return ErrorClassification(
        retryable=False,
        code="unknown",
        message=message,
    )


def normalize_error(exc: Exception) -> RetryableError | NonRetryableError:
    """Wrap a generic exception into a typed error based on classification."""
    classification = classify_error(exc)
    if classification.retryable:
        return RetryableError(classification.message, original=exc)
    return NonRetryableError(classification.message, original=exc)
