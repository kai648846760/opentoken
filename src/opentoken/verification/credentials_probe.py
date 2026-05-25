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


# Each tuple is (URL to GET, list of cookie-relevant status codes that mean
# "authenticated"). Anything else (4xx auth errors, 5xx) is treated as fail.
_PROVIDER_PROBE_URLS: dict[str, tuple[str, tuple[int, ...]]] = {
    "claude": ("https://claude.ai/api/organizations", (200,)),
    "deepseek": ("https://chat.deepseek.com/", (200,)),
    "kimi": ("https://kimi.com/api/user", (200,)),
    "qwen-intl": ("https://chat.qwen.ai/api/v1/users/me", (200,)),
    "qwen-cn": ("https://chat2.qianwen.com/api/v1/users/me", (200,)),
    "glm-intl": ("https://chat.z.ai/api/user", (200,)),
    "gemini": ("https://gemini.google.com/app", (200,)),
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
