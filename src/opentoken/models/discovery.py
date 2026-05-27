from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
import re
import time
from typing import Callable

import httpx

from opentoken.config.paths import resolve_providers_dir, resolve_state_dir
from opentoken.models.catalog import ModelCatalogEntry, default_catalog
from opentoken.models.provider_credentials import ProviderCredentialRecord
from opentoken.providers.camoufox_clients import CamoufoxProviderClient
from opentoken.storage._atomic import file_lock, write_json_atomic
from opentoken.storage.provider_sessions import credential_fingerprint
from opentoken.storage.provider_store import load_provider_credentials

# Overall wall-clock budget for a single cold-cache discovery pass. Each provider
# discoverer runs concurrently; any that haven't returned by this deadline are
# skipped for this pass (they contribute nothing rather than blocking /v1/models).
_DISCOVERY_PASS_DEADLINE_SECONDS = 45.0
_DISCOVERY_MAX_WORKERS = 8

# Each discoverer should return list[(provider_model_id, display_name)]; an empty
# list is a soft failure (provider unreachable, schema changed, …) — the loader
# will fall back to the cached/static catalog for that provider only.

# Minimal known-good wire models per provider, used ONLY as a last resort: the
# provider is logged in, but live discovery returned nothing (e.g. the page is
# now fully JS-rendered so the HTML scrape finds no model list, or a gRPC-only
# provider has no listing endpoint). Without this, a logged-in provider whose
# discovery breaks vanishes entirely from /v1/models and can't be exercised —
# which is wrong, since the provider itself still works when called with a known
# model id. This is NOT the old "hardcoded catalog": live discovery is always
# tried first and wins; these are only the floor so a working login is never
# invisible. Model ids are the canonical wire ids each adapter accepts.
_FALLBACK_MODELS: dict[str, list[tuple[str, str]]] = {
    "qwen-intl": [("qwen3.6-plus", "Qwen 3.6 Plus"), ("qwen-max-latest", "Qwen Max")],
    "kimi": [("k2", "Kimi K2"), ("k1", "Kimi K1")],
}

_DISCOVERY_TTL_SECONDS = 60 * 60 * 6
_QWEN_INTL_MODEL_PATTERN = re.compile(
    r'"id":"([A-Za-z0-9_.:-]+)"\s*,\s*"name":"([^"]+)"\s*,\s*"object":"model"'
)
_QWEN_CN_MODEL_PATTERN = re.compile(
    r"Qwen[0-9A-Za-z.\u4e00-\u9fff-]*-[0-9A-Za-z.\u4e00-\u9fff-]+"
)
_DOUBAO_ACTION_BAR_MENU_PATTERN = re.compile(
    r'"action_bar_menu_config":\{"menu_item_list":(?P<items>\[[\s\S]*?\]),'
    r'"default_deep_think_auto"',
)
_GLM_CN_MODEL_PATTERN = re.compile(
    r"\b(GLM-\d+(?:\.\d+)?(?:[- ](?:Plus|Think|Zero))?)\b",
    flags=re.IGNORECASE,
)
_DISCOVERERS: dict[str, Callable[[ProviderCredentialRecord, Path], list[tuple[str, str]]]] = {}
_DOUBAO_MENU_NAME_TO_MODEL: dict[str, tuple[str, str]] = {
    "快速": ("doubao-seed-2.0", "Doubao 快速"),
    "思考": ("doubao-thinking", "Doubao 思考"),
    "专家": ("doubao-pro", "Doubao 专家"),
}


def _build_catalog_entries(provider: str, models: list[tuple[str, str]]) -> list[ModelCatalogEntry]:
    return [
        ModelCatalogEntry(
            id=f"algae/{provider}/{model_id}",
            provider="opentoken",
            name=name,
        )
        for model_id, name in models
    ]


def _group_fallback_catalog() -> tuple[list[str], dict[str, list[ModelCatalogEntry]]]:
    order: list[str] = []
    grouped: dict[str, list[ModelCatalogEntry]] = {}
    for entry in default_catalog():
        provider = entry.id.split("/", 2)[1]
        if provider not in grouped:
            grouped[provider] = []
            order.append(provider)
        grouped[provider].append(entry)
    return order, grouped


def _resolve_cache_path(state_dir: Path) -> Path:
    return state_dir / "model-catalog-cache.json"


def _load_cache(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): value
        for key, value in payload.items()
        if isinstance(key, str) and isinstance(value, dict)
    }


def _save_cache(path: Path, payload: dict[str, dict[str, object]]) -> None:
    # Atomic write: two concurrent /v1/models requests in the FastAPI threadpool
    # both call load_model_catalog → both call _save_cache. Without tmp+rename
    # they can interleave and produce torn JSON that _load_cache then drops.
    write_json_atomic(path, payload)


def _persist_discovered_models(
    cache_path: Path,
    *,
    discovered_results: dict[str, list[tuple[str, str]]],
    creds_by_provider: dict[str, ProviderCredentialRecord],
    now: float,
) -> None:
    """Merge freshly discovered models into the on-disk cache under an
    exclusive lock. Two concurrent /v1/models passes both read the cache at the
    top of load_model_catalog and arrive here with disjoint discovered sets;
    the lock + re-read keeps the later writer from silently clobbering the
    earlier one's entries (lost update). Atomic tmp+rename alone only prevents
    torn JSON — it does not serialise read-modify-write.
    """
    with file_lock(cache_path):
        merged = _load_cache(cache_path)
        for provider, models in discovered_results.items():
            _store_cached_models(
                merged,
                provider=provider,
                credentials=creds_by_provider[provider],
                models=models,
                now=now,
            )
        _save_cache(cache_path, merged)


def _cache_key(provider: str, credentials: ProviderCredentialRecord) -> str:
    return f"{provider}:{credential_fingerprint(credentials)}"


def _load_cached_models(
    cache: dict[str, dict[str, object]],
    *,
    provider: str,
    credentials: ProviderCredentialRecord,
    now: float,
) -> list[tuple[str, str]] | None:
    payload = cache.get(_cache_key(provider, credentials))
    if not isinstance(payload, dict):
        return None
    expires_at = float(payload.get("expires_at", 0) or 0)
    if expires_at < now:
        return None
    models = payload.get("models")
    if not isinstance(models, list):
        return None
    normalized: list[tuple[str, str]] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id", "")).strip()
        name = str(item.get("name", "")).strip()
        if model_id and name:
            normalized.append((model_id, name))
    return normalized or None


def _store_cached_models(
    cache: dict[str, dict[str, object]],
    *,
    provider: str,
    credentials: ProviderCredentialRecord,
    models: list[tuple[str, str]],
    now: float,
) -> None:
    cache[_cache_key(provider, credentials)] = {
        "expires_at": now + _DISCOVERY_TTL_SECONDS,
        "models": [{"id": model_id, "name": name} for model_id, name in models],
    }


def _dedupe_models(models: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for model_id, name in models:
        normalized_id = model_id.strip()
        if not normalized_id or normalized_id in seen:
            continue
        seen.add(normalized_id)
        deduped.append((normalized_id, name.strip() or normalized_id))
    return deduped


def _fetch_text_page(*, url: str, credentials: ProviderCredentialRecord) -> str:
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "User-Agent": credentials.user_agent or "Mozilla/5.0",
        "Cookie": credentials.cookie or "",
        "Referer": url,
    }
    with httpx.Client(timeout=60.0, trust_env=False, follow_redirects=True) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        return response.text


def _extract_qwen_intl_models_from_html(html: str) -> list[tuple[str, str]]:
    return _dedupe_models(list(_QWEN_INTL_MODEL_PATTERN.findall(html)))


def _extract_qwen_cn_models_from_dialog_text(dialog_text: str) -> list[tuple[str, str]]:
    return _dedupe_models(
        [(model_id, model_id) for model_id in _QWEN_CN_MODEL_PATTERN.findall(dialog_text)]
    )


def _extract_doubao_models_from_html(html: str) -> list[tuple[str, str]]:
    match = _DOUBAO_ACTION_BAR_MENU_PATTERN.search(html)
    if match is None:
        return []
    try:
        raw_items = json.loads(match.group("items"))
    except json.JSONDecodeError:
        return []
    models: list[tuple[str, str]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        raw_name = str(item.get("name", "")).strip()
        if not raw_name:
            continue
        mapped = None
        for label, model in _DOUBAO_MENU_NAME_TO_MODEL.items():
            if raw_name.startswith(label):
                mapped = model
                break
        if mapped is None:
            continue
        models.append(mapped)
    return _dedupe_models(models)


def _extract_glm_cn_models_from_html(html: str) -> list[tuple[str, str]]:
    models: list[tuple[str, str]] = []
    for raw in _GLM_CN_MODEL_PATTERN.findall(html):
        model_id, name = _normalize_glm_cn_model_token(raw)
        if model_id and name:
            models.append((model_id, name))
    return _dedupe_models(models)


def _normalize_glm_cn_model_token(raw: str) -> tuple[str, str]:
    normalized_id = re.sub(r"\s+", "-", raw.strip()).lower()
    if not normalized_id.startswith("glm-"):
        return "", ""
    segments = normalized_id.split("-")
    label_segments: list[str] = []
    for segment in segments:
        if segment == "glm":
            label_segments.append("GLM")
        elif segment in {"plus", "think", "zero"}:
            label_segments.append(segment.title())
        else:
            label_segments.append(segment.upper() if segment.isalpha() and len(segment) <= 2 else segment)
    return normalized_id, "-".join(label_segments)


def _discover_qwen_intl_models(
    credentials: ProviderCredentialRecord,
    _state_dir: Path,
) -> list[tuple[str, str]]:
    html = _fetch_text_page(url="https://chat.qwen.ai/", credentials=credentials)
    return _extract_qwen_intl_models_from_html(html)


def _discover_qwen_cn_models(
    credentials: ProviderCredentialRecord,
    state_dir: Path,
) -> list[tuple[str, str]]:
    client = CamoufoxProviderClient("qwen-cn", credentials)
    client._state_dir = state_dir

    def action(_context, page):
        page.wait_for_timeout(4000)
        current_label = page.evaluate(
            """
            () => {
              const clean = (s) => (s || "").replace(/\\s+/g, " ").trim();
              const texts = Array.from(document.querySelectorAll("*"))
                .map((el) => clean(el.innerText || el.textContent || ""))
                .filter(Boolean);
              return texts.find((text) => /^Qwen[0-9A-Za-z.\u4e00-\u9fff-]*-[0-9A-Za-z.\u4e00-\u9fff-]+$/.test(text)) || "";
            }
            """
        )
        if not current_label:
            return []
        page.locator(f"text={current_label}").first.click(timeout=10000)
        page.wait_for_timeout(1000)
        dialog_text = page.evaluate(
            """
            () => {
              const clean = (s) => (s || "").replace(/\\s+/g, " ").trim();
              const dialog = document.querySelector('[role="dialog"]');
              if (!dialog) {
                return "";
              }
              return clean(dialog.innerText || dialog.textContent || "");
            }
            """
        )
        return _extract_qwen_cn_models_from_dialog_text(str(dialog_text))

    return list(
        client._with_page(
            start_url="https://www.qianwen.com/",
            cookie_domains=(".qianwen.com",),
            action=action,
        )
    )


def _discover_doubao_models(
    credentials: ProviderCredentialRecord,
    _state_dir: Path,
) -> list[tuple[str, str]]:
    html = _fetch_text_page(url="https://www.doubao.com/chat/", credentials=credentials)
    return _extract_doubao_models_from_html(html)


def _discover_glm_cn_models(
    credentials: ProviderCredentialRecord,
    _state_dir: Path,
) -> list[tuple[str, str]]:
    html = _fetch_text_page(url="https://chatglm.cn/main/all", credentials=credentials)
    return _extract_glm_cn_models_from_html(html)


# ─── extra discoverers ────────────────────────────────────────────────────────
# Below this point: discoverers for providers that previously only had hardcoded
# catalog entries. Each follows the same contract: take credentials + state_dir,
# return a list of (model_id, display_name) tuples. Any failure → return [].


_DEEPSEEK_MODEL_HTML_PATTERN = re.compile(
    r'"model_class":"(deepseek-[a-z0-9.-]+)"\s*,\s*"display_name":"([^"]+)"'
)
_DEEPSEEK_MODEL_JS_PATTERN = re.compile(
    r'\{\s*"model"\s*:\s*"(deepseek-[a-z0-9.-]+)"\s*,\s*"label"\s*:\s*"([^"]+)"'
)
_KIMI_MODEL_PATTERN = re.compile(
    r'"id"\s*:\s*"(k[12](?:-thinking|-search)?|moonshot-v1-[0-9a-z]+)"\s*,\s*"name"\s*:\s*"([^"]+)"'
)
_GLM_INTL_MODEL_PATTERN = re.compile(
    r'"id"\s*:\s*"(glm-[a-z0-9.-]+)"\s*,\s*"name"\s*:\s*"([^"]+)"',
    flags=re.IGNORECASE,
)
_GEMINI_MODEL_PATTERN = re.compile(
    r'\["(gemini[-_][a-z0-9_.\-]+)"\s*,\s*"([^"]+)"',
    flags=re.IGNORECASE,
)
_GROK_MODEL_PATTERN = re.compile(
    r'"modelName"\s*:\s*"(grok-[a-z0-9.-]+)"\s*,\s*"displayName"\s*:\s*"([^"]+)"',
    flags=re.IGNORECASE,
)
_MIMO_MODEL_PATTERN = re.compile(
    r'"modelKey"\s*:\s*"([a-z0-9._-]*mimo[a-z0-9._-]*)"\s*,\s*"displayName"\s*:\s*"([^"]+)"',
    flags=re.IGNORECASE,
)
_CHATGPT_MODEL_PATTERN = re.compile(
    r'"slug"\s*:\s*"(gpt-[a-z0-9.-]+|o[1-9][a-z0-9.-]*)"\s*,\s*"title"\s*:\s*"([^"]+)"',
    flags=re.IGNORECASE,
)


def _bearer_token_from_credentials(credentials: ProviderCredentialRecord) -> str | None:
    if credentials.metadata:
        api_key = str(credentials.metadata.get("api_key", "")).strip()
        if api_key:
            return api_key
    if credentials.headers:
        for key in ("authorization", "Authorization"):
            value = str(credentials.headers.get(key, "")).strip()
            if not value:
                continue
            return value[7:].strip() if value.lower().startswith("bearer ") else value
    return None


def _http_get_json(
    *,
    url: str,
    credentials: ProviderCredentialRecord,
    extra_headers: dict[str, str] | None = None,
    timeout_seconds: float = 30.0,
) -> object | None:
    headers: dict[str, str] = {
        "Accept": "application/json",
        "User-Agent": credentials.user_agent or "Mozilla/5.0",
    }
    if credentials.cookie:
        headers["Cookie"] = credentials.cookie
    if extra_headers:
        headers.update(extra_headers)
    try:
        with httpx.Client(timeout=timeout_seconds, trust_env=False, follow_redirects=False) as client:
            response = client.get(url, headers=headers)
        if response.status_code != 200:
            return None
        return response.json()
    except Exception:
        return None


# ─── NVIDIA NIM (standard OpenAI /v1/models) ──────────────────────────────────


def _discover_nim_models(
    credentials: ProviderCredentialRecord,
    _state_dir: Path,
) -> list[tuple[str, str]]:
    token = _bearer_token_from_credentials(credentials)
    if not token:
        return []
    body = _http_get_json(
        url="https://integrate.api.nvidia.com/v1/models",
        credentials=credentials,
        extra_headers={"Authorization": f"Bearer {token}"},
    )
    if not isinstance(body, dict):
        return []
    items = body.get("data")
    if not isinstance(items, list):
        return []
    models: list[tuple[str, str]] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("id") or "").strip()
        if not model_id:
            continue
        # NIM doesn't expose a separate display name; reuse the id, which already
        # looks like "namespace/model".
        models.append((model_id, model_id))
    return _dedupe_models(models)


# ─── Manus (their own model registry) ─────────────────────────────────────────


def _discover_manus_models(
    credentials: ProviderCredentialRecord,
    _state_dir: Path,
) -> list[tuple[str, str]]:
    token = _bearer_token_from_credentials(credentials)
    if not token:
        return []
    body = _http_get_json(
        url="https://api.manus.im/api/v1/agents",
        credentials=credentials,
        extra_headers={"API_KEY": token},
    )
    if not isinstance(body, dict):
        return []
    items = body.get("agents") if isinstance(body.get("agents"), list) else body.get("data")
    if not isinstance(items, list):
        return []
    models: list[tuple[str, str]] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        agent_id = str(entry.get("id") or entry.get("agentId") or "").strip()
        agent_name = str(entry.get("name") or entry.get("displayName") or agent_id).strip()
        if agent_id and agent_name:
            models.append((agent_id, agent_name))
    return _dedupe_models(models)


# ─── DeepSeek (chat.deepseek.com HTML bundle has model list) ──────────────────


def _extract_deepseek_models_from_html(html: str) -> list[tuple[str, str]]:
    models: list[tuple[str, str]] = []
    for model_id, display in _DEEPSEEK_MODEL_HTML_PATTERN.findall(html):
        models.append((model_id, display))
    if not models:
        for model_id, display in _DEEPSEEK_MODEL_JS_PATTERN.findall(html):
            models.append((model_id, display))
    return _dedupe_models(models)


_DEEPSEEK_WIRE_MODELS: tuple[tuple[str, str], ...] = (
    ("deepseek-chat", "DeepSeek Chat"),
    ("deepseek-reasoner", "DeepSeek Reasoner"),
)


def _discover_deepseek_models(
    credentials: ProviderCredentialRecord,
    _state_dir: Path,
) -> list[tuple[str, str]]:
    # The deepseek web app is a JS SPA — its homepage HTML contains no model
    # list. The protocol itself only supports two model "modes" (chat /
    # reasoner) toggled via thinking_enabled, so the discovery contract here is:
    # if the saved credentials still authenticate against /api/v0/users/current,
    # the two wire models are available; otherwise we contribute nothing.
    auth = ""
    if credentials.headers:
        auth = str(credentials.headers.get("authorization", "")).strip()
    if not auth:
        return []
    body = _http_get_json(
        url="https://chat.deepseek.com/api/v0/users/current",
        credentials=credentials,
        extra_headers={"Authorization": auth},
    )
    if not isinstance(body, dict):
        return []
    if body.get("code") != 0:
        return []
    return list(_DEEPSEEK_WIRE_MODELS)


# ─── Kimi (kimi.com / kimi.moonshot.cn HTML) ──────────────────────────────────


def _extract_kimi_models_from_html(html: str) -> list[tuple[str, str]]:
    return _dedupe_models(list(_KIMI_MODEL_PATTERN.findall(html)))


def _discover_kimi_models(
    credentials: ProviderCredentialRecord,
    _state_dir: Path,
) -> list[tuple[str, str]]:
    html = _fetch_text_page(url="https://kimi.com/", credentials=credentials)
    return _extract_kimi_models_from_html(html)


# ─── GLM International (chat.z.ai or similar) ─────────────────────────────────


def _extract_glm_intl_models_from_payload(html: str) -> list[tuple[str, str]]:
    return _dedupe_models(list(_GLM_INTL_MODEL_PATTERN.findall(html)))


def _discover_glm_intl_models(
    credentials: ProviderCredentialRecord,
    _state_dir: Path,
) -> list[tuple[str, str]]:
    body = _http_get_json(
        url="https://chat.z.ai/api/models",
        credentials=credentials,
    )
    if isinstance(body, dict):
        items = body.get("data") if isinstance(body.get("data"), list) else body.get("models")
        if isinstance(items, list):
            models: list[tuple[str, str]] = []
            for entry in items:
                if not isinstance(entry, dict):
                    continue
                model_id = str(entry.get("id") or entry.get("name") or "").strip()
                display = str(entry.get("name") or model_id).strip()
                if model_id:
                    models.append((model_id, display))
            if models:
                return _dedupe_models(models)
    # Fallback: scrape the home HTML for embedded model metadata
    html = _fetch_text_page(url="https://chat.z.ai/", credentials=credentials)
    return _extract_glm_intl_models_from_payload(html)


# ─── ChatGPT (chat.openai.com/backend-api/models) ─────────────────────────────


def _extract_chatgpt_models_from_html(html: str) -> list[tuple[str, str]]:
    return _dedupe_models(list(_CHATGPT_MODEL_PATTERN.findall(html)))


def _discover_chatgpt_models(
    credentials: ProviderCredentialRecord,
    _state_dir: Path,
) -> list[tuple[str, str]]:
    # chat.openai.com now 308-redirects to chatgpt.com; _http_get_json runs
    # with follow_redirects=False (intentional, to avoid SSRF-bypass via
    # follow), so we hit the canonical domain directly. Cookie-only auth is
    # what /backend-api/models accepts here — passing the harvested
    # access_token as a Bearer actually 401s ("Could not parse your
    # authentication token"), so we let _http_get_json forward just the cookie.
    body = _http_get_json(
        url="https://chatgpt.com/backend-api/models",
        credentials=credentials,
    )
    if isinstance(body, dict):
        items = body.get("models") if isinstance(body.get("models"), list) else body.get("data")
        if isinstance(items, list):
            models: list[tuple[str, str]] = []
            for entry in items:
                if not isinstance(entry, dict):
                    continue
                slug = str(entry.get("slug") or entry.get("id") or "").strip()
                title = str(entry.get("title") or entry.get("name") or slug).strip()
                if slug:
                    models.append((slug, title))
            if models:
                return _dedupe_models(models)
    # Fallback: parse the homepage HTML.
    try:
        html = _fetch_text_page(url="https://chatgpt.com/", credentials=credentials)
    except Exception:
        return []
    return _extract_chatgpt_models_from_html(html)


# ─── Claude (claude.ai uses statsig + per-org model availability) ─────────────


def _discover_claude_models(
    credentials: ProviderCredentialRecord,
    _state_dir: Path,
) -> list[tuple[str, str]]:
    orgs = _http_get_json(
        url="https://claude.ai/api/organizations",
        credentials=credentials,
    )
    if not isinstance(orgs, list) or not orgs:
        return []
    org_id: str | None = None
    capabilities: list[str] = []
    for org in orgs:
        if not isinstance(org, dict):
            continue
        uuid_val = str(org.get("uuid") or "").strip()
        if uuid_val:
            org_id = uuid_val
        org_capabilities = org.get("capabilities")
        if isinstance(org_capabilities, list):
            for capability in org_capabilities:
                if isinstance(capability, str):
                    capabilities.append(capability)
        break
    if not org_id:
        return []
    # Claude exposes its currently-available model slugs through statsig dynamic
    # config. The exact endpoint shape moves around; we ask for the chat
    # conversation features set and scrape any "model" strings out of it.
    statsig = _http_get_json(
        url=f"https://claude.ai/api/organizations/{org_id}/statsig/dynamic_configs/chat_models",
        credentials=credentials,
    )
    models: list[tuple[str, str]] = []
    if isinstance(statsig, dict):
        config_value = statsig.get("value")
        if isinstance(config_value, dict):
            for key, value in config_value.items():
                if not isinstance(key, str):
                    continue
                if key.startswith("claude-"):
                    label = str(value) if isinstance(value, str) else key
                    models.append((key, label or key))
    # As a final fallback: extract claude-* slugs that the org's capabilities
    # advertise (e.g. "claude_max_3_5_sonnet_v2_enabled").
    if not models:
        seen: set[str] = set()
        for capability in capabilities:
            match = re.search(r"(claude[-_][a-z0-9._-]+)", capability, flags=re.IGNORECASE)
            if not match:
                continue
            slug = match.group(1).replace("_", "-")
            if slug in seen:
                continue
            seen.add(slug)
            models.append((slug, slug))
    return _dedupe_models(models)


# ─── Gemini (gemini.google.com HTML has model selector) ───────────────────────


def _extract_gemini_models_from_html(html: str) -> list[tuple[str, str]]:
    return _dedupe_models(list(_GEMINI_MODEL_PATTERN.findall(html)))


def _discover_gemini_models(
    credentials: ProviderCredentialRecord,
    _state_dir: Path,
) -> list[tuple[str, str]]:
    try:
        html = _fetch_text_page(url="https://gemini.google.com/app", credentials=credentials)
    except Exception:
        return []
    return _extract_gemini_models_from_html(html)


# ─── Grok (grok.com app HTML) ─────────────────────────────────────────────────


def _extract_grok_models_from_html(html: str) -> list[tuple[str, str]]:
    return _dedupe_models(list(_GROK_MODEL_PATTERN.findall(html)))


def _discover_grok_models(
    credentials: ProviderCredentialRecord,
    _state_dir: Path,
) -> list[tuple[str, str]]:
    try:
        html = _fetch_text_page(url="https://grok.com/", credentials=credentials)
    except Exception:
        return []
    return _extract_grok_models_from_html(html)


# ─── Xiaomi Mimo (xiaomimo.com home HTML) ─────────────────────────────────────


def _extract_mimo_models_from_html(html: str) -> list[tuple[str, str]]:
    return _dedupe_models(list(_MIMO_MODEL_PATTERN.findall(html)))


def _discover_mimo_models(
    credentials: ProviderCredentialRecord,
    _state_dir: Path,
) -> list[tuple[str, str]]:
    try:
        html = _fetch_text_page(url="https://xiaomimo.com/", credentials=credentials)
    except Exception:
        return []
    return _extract_mimo_models_from_html(html)


# ─── Unified Proxy (LiteLLM helper, if installed) ─────────────────────────────


def _discover_unified_models(
    credentials: ProviderCredentialRecord,
    _state_dir: Path,
) -> list[tuple[str, str]]:
    # Only enumerate backends that the credentials file actually has keys for —
    # listing every LiteLLM-supported model regardless of credentials would
    # flood /v1/models with thousands of unreachable entries.
    backends: list[str] = []
    if credentials.metadata:
        for key in credentials.metadata:
            if not isinstance(key, str) or not key.startswith("api_key_"):
                continue
            backend = key[len("api_key_"):].strip()
            if backend:
                backends.append(backend)
    if not backends:
        return []
    try:
        import litellm  # type: ignore
    except ImportError:
        return []
    models: list[tuple[str, str]] = []
    model_cost = getattr(litellm, "model_cost", None)
    if not isinstance(model_cost, dict):
        return []
    for raw_id, spec in model_cost.items():
        if not isinstance(raw_id, str):
            continue
        if "/" not in raw_id:
            # LiteLLM uses bare names for some providers (e.g. "gpt-4o"). Skip
            # — without a backend prefix we can't disambiguate.
            continue
        provider_prefix = raw_id.split("/", 1)[0].lower()
        if provider_prefix in {backend.lower() for backend in backends}:
            display = str(spec.get("litellm_provider") if isinstance(spec, dict) else raw_id) or raw_id
            models.append((raw_id, display))
    return _dedupe_models(models)


_DISCOVERERS = {
    "deepseek": _discover_deepseek_models,
    "doubao": _discover_doubao_models,
    "glm-cn": _discover_glm_cn_models,
    "glm-intl": _discover_glm_intl_models,
    "qwen-intl": _discover_qwen_intl_models,
    "qwen-cn": _discover_qwen_cn_models,
    "kimi": _discover_kimi_models,
    "claude": _discover_claude_models,
    "chatgpt": _discover_chatgpt_models,
    "gemini": _discover_gemini_models,
    "grok": _discover_grok_models,
    "mimo": _discover_mimo_models,
    "manus": _discover_manus_models,
    "nim": _discover_nim_models,
    "unified": _discover_unified_models,
}


def load_model_catalog(
    *,
    state_dir: Path | None = None,
    providers_dir: Path | None = None,
    use_cache: bool = True,
) -> list[ModelCatalogEntry]:
    resolved_state_dir = state_dir or resolve_state_dir()
    resolved_providers_dir = providers_dir or resolve_providers_dir()
    cache_path = _resolve_cache_path(resolved_state_dir)
    cache = _load_cache(cache_path) if use_cache else {}
    now = time.time()
    provider_order, grouped_entries = _group_fallback_catalog()

    # First pass (no I/O): collect logged-in providers, serving any with a valid
    # cache entry immediately and queueing the rest for live discovery.
    cached_results: dict[str, list[tuple[str, str]]] = {}
    to_discover: list[tuple[str, ProviderCredentialRecord]] = []
    for provider, _discoverer in _DISCOVERERS.items():
        credentials = load_provider_credentials(resolved_providers_dir, provider)
        if credentials is None:
            continue
        cached = (
            _load_cached_models(cache, provider=provider, credentials=credentials, now=now)
            if use_cache
            else None
        )
        if cached is not None:
            cached_results[provider] = cached
        else:
            to_discover.append((provider, credentials))

    # Run the un-cached discoverers concurrently with an overall deadline. A
    # single slow provider (e.g. qwen-cn, which launches a Camoufox browser) or a
    # hung HTTP call no longer blocks /v1/models for minutes — providers that
    # don't finish in time simply contribute nothing this pass, and their next
    # request will try again. This is the fix for the cold-cache /v1/models
    # timeout: total time ≈ slowest discoverer instead of the sum of all.
    discovered_results: dict[str, list[tuple[str, str]]] = {}
    if to_discover:
        def _run(provider: str, credentials: ProviderCredentialRecord) -> tuple[str, list[tuple[str, str]]]:
            try:
                models = _dedupe_models(_DISCOVERERS[provider](credentials, resolved_state_dir))
            except Exception:
                models = []
            return provider, models

        max_workers = min(_DISCOVERY_MAX_WORKERS, len(to_discover))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run, provider, credentials): provider
                for provider, credentials in to_discover
            }
            deadline = time.monotonic() + _DISCOVERY_PASS_DEADLINE_SECONDS
            for future in concurrent.futures.as_completed(futures):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    provider, models = future.result(timeout=max(remaining, 0.01))
                except Exception:
                    continue
                if models:
                    discovered_results[provider] = models

    # Persist freshly discovered models to the cache.
    if use_cache and discovered_results:
        creds_by_provider = {p: c for p, c in to_discover}
        _persist_discovered_models(
            cache_path,
            discovered_results=discovered_results,
            creds_by_provider=creds_by_provider,
            now=now,
        )

    # Floor for logged-in providers whose live discovery yielded nothing this
    # pass (qwen-intl's catalog page is now JS-rendered, Kimi's lives behind a
    # gRPC-Connect endpoint we don't speak). Without this the provider vanishes
    # from /v1/models even though chat against a known wire id still works,
    # which made the smoke script skip them. Not cached — every pass retries
    # live first; the fallback only kicks in when discovery has actually failed.
    fallback_results: dict[str, list[tuple[str, str]]] = {}
    for provider, _credentials in to_discover:
        if provider in discovered_results:
            continue
        floor = _FALLBACK_MODELS.get(provider)
        if floor:
            fallback_results[provider] = list(floor)

    for provider, models in {**cached_results, **discovered_results, **fallback_results}.items():
        if not models:
            continue
        grouped_entries[provider] = _build_catalog_entries(provider, models)
        if provider not in provider_order:
            provider_order.append(provider)

    flattened: list[ModelCatalogEntry] = []
    for provider in provider_order:
        flattened.extend(grouped_entries.get(provider, []))
    return flattened


__all__ = [
    "load_model_catalog",
    "_extract_doubao_models_from_html",
    "_extract_glm_cn_models_from_html",
    "_extract_qwen_intl_models_from_html",
    "_extract_qwen_cn_models_from_dialog_text",
]
