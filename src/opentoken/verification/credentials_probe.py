"""Cheap pre-save credential validation.

Before persisting freshly-harvested credentials (cookies, API keys) we ping
a low-cost provider endpoint to confirm the new blob actually works. If the
probe fails we keep the previously-stored credentials instead of replacing
them with a broken set. This is the credentials dry-run contract used by the
browser harvest flow.

The probe is intentionally minimal: a single GET against an authenticated
"who-am-I" / billing / homepage endpoint. The goal is to catch obvious
failures (expired session, malformed cookie) without blocking on a full
chat completion round-trip.
"""
from __future__ import annotations

import httpx

from opentoken.models.provider_credentials import ProviderCredentialRecord


# Each tuple is (URL to GET, status codes that mean "authenticated"). Anything
# else (4xx auth errors, 5xx) is treated as fail.
#
# Deliberately conservative: a probe is only listed here if (a) the endpoint
# genuinely requires auth (so a 200 actually proves the credential works) AND
# (b) it accepts the SAME credential material the chat path uses. A probe that
# is stricter than the chat endpoint would false-reject a valid harvest and
# block the user from re-logging in — worse than having no probe at all. So
# unverified / homepage-style endpoints are intentionally omitted: those
# providers fall through to "trust the harvest" (probe_credentials returns
# True) rather than risk a false rejection.
#
# Verified entries:
#   - claude /api/organizations: returns 401/403 cookie-less, 200 with a valid
#     sessionKey cookie (the same cookie the completion endpoint uses).
#
# Omitted on purpose (and why):
#   - deepseek/gemini homepages always return 200 regardless of auth -> useless.
#   - kimi /api/user needs a Bearer token the cookie-only harvest may not have
#     -> would false-reject. qwen/glm /users/me endpoints are unverified.
_PROVIDER_PROBE_URLS: dict[str, tuple[str, tuple[int, ...]]] = {
    "claude": ("https://claude.ai/api/organizations", (200,)),
}


def probe_credentials(
    record: ProviderCredentialRecord,
    *,
    client_factory=None,
    timeout_seconds: float = 8.0,
) -> bool:
    """Return True if the record looks usable, False if the probe rejected it.

    Providers without a registered probe URL return True (unknown == accept;
    we'd rather over-accept than block the rename of a working credential
    file just because we don't know how to probe it yet).
    """
    target = _PROVIDER_PROBE_URLS.get(record.provider)
    if target is None:
        # Some providers (e.g. api-key providers like nim/manus/unified) don't
        # have a cheap GET we can hit without spending a quota; trust the user.
        return True
    # A browser_session record without a cookie has no auth state — probing it
    # would send an unauthenticated request to the target, which can return 200
    # (some providers serve stub responses to anonymous GETs), falsely marking
    # an empty credential as "ok" and overwriting a previously-working one.
    if record.kind == "browser_session" and not (record.cookie or "").strip():
        return False
    url, ok_status = target

    if client_factory is None:
        def client_factory():  # pragma: no cover - exercised through tests
            return httpx.Client(timeout=timeout_seconds, trust_env=False)

    headers = {
        "User-Agent": record.user_agent or "Mozilla/5.0",
        "Cookie": record.cookie or "",
        "Accept": "application/json,text/html;q=0.9",
    }
    if record.headers:
        for header_key in ("authorization", "Authorization"):
            value = record.headers.get(header_key)
            if value:
                headers["Authorization"] = str(value)
                break

    try:
        with client_factory() as client:
            response = client.get(url, headers=headers)
    except Exception:
        return False
    return response.status_code in ok_status
