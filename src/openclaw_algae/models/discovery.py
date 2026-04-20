from __future__ import annotations

import json
from pathlib import Path
import re
import time
from typing import Callable

import httpx

from openclaw_algae.config.paths import resolve_providers_dir, resolve_state_dir
from openclaw_algae.models.catalog import ModelCatalogEntry, default_catalog
from openclaw_algae.models.provider_credentials import ProviderCredentialRecord
from openclaw_algae.providers.camoufox_clients import CamoufoxProviderClient
from openclaw_algae.storage.provider_sessions import credential_fingerprint
from openclaw_algae.storage.provider_store import load_provider_credentials

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
            provider="openclaw-algae",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


_DISCOVERERS = {
    "doubao": _discover_doubao_models,
    "glm-cn": _discover_glm_cn_models,
    "qwen-intl": _discover_qwen_intl_models,
    "qwen-cn": _discover_qwen_cn_models,
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

    for provider, discoverer in _DISCOVERERS.items():
        credentials = load_provider_credentials(resolved_providers_dir, provider)
        if credentials is None:
            continue
        discovered = (
            _load_cached_models(
                cache,
                provider=provider,
                credentials=credentials,
                now=now,
            )
            if use_cache
            else None
        )
        if discovered is None:
            try:
                discovered = _dedupe_models(discoverer(credentials, resolved_state_dir))
            except Exception:
                discovered = None
            if discovered and use_cache:
                _store_cached_models(
                    cache,
                    provider=provider,
                    credentials=credentials,
                    models=discovered,
                    now=now,
                )
        if discovered:
            grouped_entries[provider] = _build_catalog_entries(provider, discovered)
            if provider not in provider_order:
                provider_order.append(provider)

    if use_cache:
        _save_cache(cache_path, cache)

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
