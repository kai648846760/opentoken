"""Claude 401/403 surfaces as a friendly RuntimeError, not a 502.

The route's error classifier maps "session expired" RuntimeErrors to a 401
authentication_error so OpenAI-style clients can route through their re-auth
flow. Without the helper, an expired sessionKey would propagate as
httpx.HTTPStatusError -> 502 api_error, hiding the real fix.
"""
from __future__ import annotations

import httpx
import pytest

from opentoken.models.provider_credentials import ProviderCredentialRecord
from opentoken.providers.claude import ClaudeWebClient


def _client_with_status(status: int) -> ClaudeWebClient:
    transport = httpx.MockTransport(lambda request: httpx.Response(status, text="unauth"))
    credentials = ProviderCredentialRecord(
        provider="claude",
        kind="web_session",
        cookie="sessionKey=stale",
        headers={},
        user_agent="ua",
        metadata={"organization_id": "org-1"},
        status="valid",
    )
    return ClaudeWebClient(
        credentials,
        client=httpx.Client(transport=transport, trust_env=False),
    )


@pytest.mark.parametrize("status", [401, 403])
def test_claude_chat_raises_session_expired_runtime_error(status: int) -> None:
    client = _client_with_status(status)
    with pytest.raises(RuntimeError, match="Claude session expired"):
        client.chat_completion(message="hi", model="claude-sonnet-4-6", conversation_id="conv-1")


@pytest.mark.parametrize("status", [401, 403])
def test_claude_stream_raises_session_expired_runtime_error(status: int) -> None:
    client = _client_with_status(status)
    with pytest.raises(RuntimeError, match="Claude session expired"):
        list(client.iter_chat_completion_text(message="hi", model="claude-sonnet-4-6", conversation_id="conv-1"))


def test_claude_500_still_raises_http_error_not_session_expired() -> None:
    # Non-auth upstream errors must keep falling through to httpx.HTTPStatusError
    # so the route's classifier maps them to 502, not 401.
    client = _client_with_status(500)
    with pytest.raises(httpx.HTTPStatusError):
        client.chat_completion(message="hi", model="claude-sonnet-4-6", conversation_id="conv-1")
