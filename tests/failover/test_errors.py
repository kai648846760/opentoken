"""Tests for error classification."""
from opentoken.failover.errors import (
    ErrorClassification,
    NonRetryableError,
    RetryableError,
    classify_error,
    normalize_error,
)
import httpx


def test_classify_timeout() -> None:
    result = classify_error(Exception("Request timed out"))
    assert result.retryable is True
    assert result.code == "retryable"


def test_classify_connection_reset() -> None:
    result = classify_error(ConnectionError("connection reset by peer"))
    assert result.retryable is True


def test_classify_500() -> None:
    result = classify_error(Exception("Server returned 500: Internal Server Error"))
    assert result.retryable is True


def test_classify_rate_limit() -> None:
    result = classify_error(Exception("rate limit exceeded"))
    assert result.retryable is True


def test_classify_401_is_non_retryable() -> None:
    # 401 / 403 are auth failures — retrying against the same credentials cannot
    # recover; they should be terminal so callers know to refresh credentials.
    result = classify_error(Exception("401 Unauthorized"))
    assert result.retryable is False


def test_classify_403_is_non_retryable() -> None:
    result = classify_error(Exception("403 Forbidden"))
    assert result.retryable is False


def test_classify_real_5xx_still_retryable() -> None:
    result = classify_error(Exception("502 Bad Gateway"))
    assert result.retryable is True


def test_classify_no_match_inside_word() -> None:
    # The broader regex 5\d{2} previously matched substrings like "X500Y";
    # with \b boundaries it no longer matches when surrounded by word characters.
    result = classify_error(Exception("modelID500abc not found"))
    # 401/403/captcha take precedence as non-retryable so we use a neutral string;
    # this message has no other retryable pattern, so it should fall through to
    # "unknown" → non-retryable.
    assert result.retryable is False


def test_classify_captcha() -> None:
    result = classify_error(Exception("CAPTCHA verification required"))
    assert result.retryable is False


def test_classify_blocked() -> None:
    result = classify_error(Exception("Account has been blocked"))
    assert result.retryable is False


def test_classify_content_policy() -> None:
    result = classify_error(Exception("Content policy violation"))
    assert result.retryable is False


def test_classify_unsupported() -> None:
    result = classify_error(RuntimeError("Unsupported provider: xyz"))
    assert result.retryable is False


def test_classify_missing_credentials() -> None:
    result = classify_error(RuntimeError("Missing credential for provider"))
    assert result.retryable is False


def test_classify_unknown_error() -> None:
    result = classify_error(ValueError("something weird"))
    assert result.retryable is False  # Unknown defaults to non-retryable


def test_classify_page_closed() -> None:
    result = classify_error(Exception("Page is closed"))
    assert result.retryable is True


def test_classify_browser_crashed() -> None:
    result = classify_error(Exception("Browser context crashed"))
    assert result.retryable is True


def test_normalize_retryable() -> None:
    exc = Exception("connection reset")
    result = normalize_error(exc)
    assert isinstance(result, RetryableError)
    assert result.original is exc


def test_normalize_non_retryable() -> None:
    exc = RuntimeError("captcha required")
    result = normalize_error(exc)
    assert isinstance(result, NonRetryableError)
    assert result.original is exc


def test_retryable_error_standalone() -> None:
    err = RetryableError("test", original=ValueError("orig"))
    assert str(err) == "test"
    assert isinstance(err.original, ValueError)


def test_non_retryable_error_standalone() -> None:
    err = NonRetryableError("blocked", original=RuntimeError("orig"))
    assert str(err) == "blocked"
    assert isinstance(err.original, RuntimeError)
