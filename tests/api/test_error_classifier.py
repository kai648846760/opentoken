"""Direct unit tests for the shared provider-RuntimeError classifier.

This function is the single decision point both /v1/chat/completions and
/v1/responses use to turn a router RuntimeError into an OpenAI-shaped
(status, error_type). Lock every branch here so route-level tests don't have
to enumerate them all.
"""
from __future__ import annotations

import pytest

from opentoken.api.errors import classify_provider_runtime_error


@pytest.mark.parametrize(
    "message,expected_status,expected_type",
    [
        # Request-validation errors the router raises -> 400.
        ("Unsupported model: algae/foo/bar", 400, "invalid_request_error"),
        ("No route configured for provider: foo", 400, "invalid_request_error"),
        ("No adapter registered for foo", 400, "invalid_request_error"),
        # Auth / credential problems -> 401.
        ("Missing deepseek credentials. Run `opentoken login deepseek` first.", 401, "authentication_error"),
        ("Manus API key is required.", 401, "authentication_error"),
        ("Claude session expired or invalid. Run `opentoken login claude` to refresh.", 401, "authentication_error"),
        ("Qwen credentials expired or invalid.", 401, "authentication_error"),
        # Everything else -> 502 (upstream / gateway failure).
        ("All browser workers failed for doubao: crashed", 502, "api_error"),
        ("DeepSeek chat completion returned no text content.", 502, "api_error"),
        ("something totally unexpected", 502, "api_error"),
    ],
)
def test_classify_provider_runtime_error(message: str, expected_status: int, expected_type: str) -> None:
    status, error_type = classify_provider_runtime_error(RuntimeError(message))
    assert (status, error_type) == (expected_status, expected_type)


def test_classify_is_case_insensitive() -> None:
    status, error_type = classify_provider_runtime_error(RuntimeError("UNSUPPORTED MODEL: x"))
    assert (status, error_type) == (400, "invalid_request_error")
