"""Streaming auth-retry mapping for Gemini and Grok.

A persistent 401 on the streaming path used to be raised as
httpx.HTTPStatusError → classified as a generic 502 api_error, even though
the actual problem is a dead session that needs `opentoken login`. These
tests pin that the streaming retry surfaces an auth RuntimeError (which the
gateway classifier maps to 401 authentication_error).
"""
from __future__ import annotations

import httpx
import pytest

from opentoken.api.errors import classify_provider_runtime_error
from opentoken.models.provider_credentials import ProviderCredentialRecord
from opentoken.providers.gemini import GeminiApiClient
from opentoken.providers.grok import GrokApiClient


def _credentials(provider: str) -> ProviderCredentialRecord:
    return ProviderCredentialRecord(
        provider=provider,
        kind="browser_session",
        cookie="__Secure-1PSIDTS=test; SIDCC=x",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )


def test_gemini_stream_persistent_401_raises_auth_error_not_502() -> None:
    """Both the initial and the retry stream return 401 → must be classified
    as authentication_error (re-login), not the api_error a raw raise_for_status
    would yield."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    transport = httpx.MockTransport(handler)
    client = GeminiApiClient(
        _credentials("gemini"),
        client=httpx.Client(transport=transport, trust_env=False),
    )

    with pytest.raises(RuntimeError) as exc_info:
        list(client.iter_chat_completion_text(message="hi", model="gemini-pro"))

    status, error_type = classify_provider_runtime_error(exc_info.value)
    assert (status, error_type) == (401, "authentication_error")


def test_grok_stream_401_with_failed_conversation_creation_raises_auth_error() -> None:
    """When the post-401 _create_conversation() also fails to mint a fresh id,
    the previous code called raise_for_status on the original 401 response
    (→ generic 502 via HTTPStatusError) and could have constructed a
    `/conversations/None/message` URL. Verify it now raises an auth RuntimeError
    and never builds a None-id URL."""
    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        # Both the message POST and the conversation-create POST return 401.
        return httpx.Response(401, json={"error": "unauthenticated"})

    transport = httpx.MockTransport(handler)
    client = GrokApiClient(
        _credentials("grok"),
        client=httpx.Client(transport=transport, trust_env=False),
    )

    with pytest.raises(RuntimeError) as exc_info:
        list(client.iter_chat_completion_text(message="hi", model="grok-2"))

    status, error_type = classify_provider_runtime_error(exc_info.value)
    assert (status, error_type) == (401, "authentication_error")
    # Confirm no `.../conversations/None/message` URL was ever requested.
    assert all("/None/message" not in u for u in seen_urls), seen_urls
