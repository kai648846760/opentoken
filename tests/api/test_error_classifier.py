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


@pytest.mark.parametrize(
    "message",
    [
        # Qwen: stale session can't create a chat (returns 200 with no id).
        "Qwen returned no chat id (status=200). Run `opentoken login qwen` to refresh the session.",
        "Qwen credentials expired or invalid. Run `opentoken login qwen` again.",
        # Kimi gRPC body signals auth failure rather than via HTTP status.
        "Kimi error: {'code': 'unauthenticated', 'details': [{'reason': 'REASON_INVALID_AUTH_TOKEN'}]}",
        # Generic re-login hint in any provider message.
        "Something broke. Run `opentoken login grok` to refresh.",
    ],
)
def test_classify_body_signalled_auth_failures_as_401(message: str) -> None:
    status, error_type = classify_provider_runtime_error(RuntimeError(message))
    assert (status, error_type) == (401, "authentication_error")


@pytest.mark.parametrize(
    "message",
    [
        # Non-auth uses of "expired" must NOT route to 401 — sending users into
        # a pointless re-login loop is worse than reporting an upstream failure.
        "upstream certificate expired",
        "TLS handshake failed: server certificate expired 2 days ago",
        "cache entry expired",
        "request expired before completion",
        "deepseek lease expired in the regional pool",
    ],
)
def test_classify_does_not_treat_bare_expired_as_auth_failure(message: str) -> None:
    status, error_type = classify_provider_runtime_error(RuntimeError(message))
    assert (status, error_type) == (502, "api_error")
