"""credentials_probe: conservative dry-run validation.

Only providers with a verified authenticated probe URL are gated; everything
else is trusted (returns True) so an unverified/stricter probe can never
false-reject a valid harvest and block re-login.
"""
from __future__ import annotations

import httpx

from opentoken.models.provider_credentials import ProviderCredentialRecord
from opentoken.verification.credentials_probe import probe_credentials


def _record(provider: str) -> ProviderCredentialRecord:
    return ProviderCredentialRecord(
        provider=provider,
        kind="web_session",
        cookie="sessionKey=x",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )


def test_unregistered_provider_is_trusted() -> None:
    # nim / kimi / qwen etc. have no probe URL -> trust the harvest.
    for provider in ("nim", "kimi", "qwen-intl", "deepseek", "gemini"):
        assert probe_credentials(_record(provider)) is True


def test_claude_probe_passes_on_200() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=[{"uuid": "org"}]))
    ok = probe_credentials(
        _record("claude"),
        client_factory=lambda: httpx.Client(transport=transport, trust_env=False),
    )
    assert ok is True


def test_claude_probe_fails_on_401() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(401, text="unauth"))
    ok = probe_credentials(
        _record("claude"),
        client_factory=lambda: httpx.Client(transport=transport, trust_env=False),
    )
    assert ok is False


def test_browser_session_without_cookie_fails_without_probing() -> None:
    """An empty-cookie browser_session has no auth state — it must fail closed,
    NOT send an unauthenticated request that some providers answer 200 to
    (which would falsely mark a broken credential as ok)."""
    probed = {"called": False}

    def factory():
        probed["called"] = True
        return httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=[{"uuid": "org"}])),
            trust_env=False,
        )

    record = ProviderCredentialRecord(
        provider="claude",
        kind="browser_session",
        cookie="",
        headers={},
        user_agent="ua",
        metadata={},
        status="valid",
    )
    assert probe_credentials(record, client_factory=factory) is False
    assert probed["called"] is False  # never even attempted the request


def test_claude_probe_fails_on_network_error() -> None:
    def boom() -> httpx.Client:
        raise httpx.ConnectError("no network")

    # A connection error is treated as a failed probe (can't confirm validity).
    assert probe_credentials(_record("claude"), client_factory=boom) is False
