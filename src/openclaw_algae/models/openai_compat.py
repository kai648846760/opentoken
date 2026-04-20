from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from openclaw_algae.config.paths import resolve_providers_dir
from openclaw_algae.models.catalog import ModelCatalogEntry
from openclaw_algae.models.discovery import load_model_catalog
from openclaw_algae.models.model_aliases import list_provider_aliases, normalize_provider_model
from openclaw_algae.providers.registry import resolve_provider_key
from openclaw_algae.storage.provider_store import load_provider_credentials

_LOCAL_EMBEDDING_MODELS: tuple[str, ...] = (
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
)


@dataclass(frozen=True)
class ResolvedModelRef:
    provider: str
    provider_model: str
    canonical_model: str


@dataclass(frozen=True)
class _ModelCandidate:
    provider: str
    provider_model: str
    canonical_model: str


def resolve_requested_model(
    model_ref: str,
    *,
    providers_dir: Path | None = None,
    catalog: list[ModelCatalogEntry] | None = None,
) -> ResolvedModelRef | None:
    cleaned = model_ref.strip()
    if not cleaned:
        return None

    slash_resolved = _resolve_prefixed_model(cleaned)
    if slash_resolved is not None:
        return slash_resolved

    entries = catalog or load_model_catalog(providers_dir=providers_dir)
    candidates = _find_raw_model_candidates(cleaned, entries)
    if not candidates:
        return None
    chosen = _choose_candidate(candidates, providers_dir=providers_dir)
    return ResolvedModelRef(
        provider=chosen.provider,
        provider_model=chosen.provider_model,
        canonical_model=chosen.canonical_model,
    )


def build_openai_model_objects(
    entries: list[ModelCatalogEntry],
    *,
    providers_dir: Path | None = None,
) -> list[dict[str, str]]:
    ordered: OrderedDict[str, dict[str, str]] = OrderedDict()
    for entry in entries:
        ordered.setdefault(entry.id, entry.to_openai_dict())

    for raw_model_id in _preferred_raw_model_ids(entries, providers_dir=providers_dir):
        ordered.setdefault(
            raw_model_id,
            {
                "id": raw_model_id,
                "object": "model",
                "owned_by": "openclaw-algae",
            },
        )

    for embedding_model in _LOCAL_EMBEDDING_MODELS:
        ordered.setdefault(
            embedding_model,
            {
                "id": embedding_model,
                "object": "model",
                "owned_by": "openclaw-algae",
            },
        )

    return list(ordered.values())


def _preferred_raw_model_ids(
    entries: list[ModelCatalogEntry],
    *,
    providers_dir: Path | None = None,
) -> list[str]:
    ordered: OrderedDict[str, None] = OrderedDict()
    for entry in entries:
        raw_model_id = _split_entry(entry)[1]
        resolved = resolve_requested_model(
            raw_model_id,
            providers_dir=providers_dir,
            catalog=entries,
        )
        if resolved is None:
            continue
        if resolved.canonical_model != entry.id:
            continue
        ordered.setdefault(raw_model_id, None)
    return list(ordered.keys())


def _resolve_prefixed_model(model_ref: str) -> ResolvedModelRef | None:
    parts = model_ref.split("/")
    if len(parts) >= 3 and parts[0] == "algae":
        provider = resolve_provider_key(parts[1])
        if provider is None:
            return None
        provider_model = normalize_provider_model(provider, parts[-1])
        return ResolvedModelRef(
            provider=provider,
            provider_model=provider_model,
            canonical_model=f"algae/{provider}/{provider_model}",
        )
    if len(parts) == 2:
        provider = resolve_provider_key(parts[0])
        if provider is None:
            return None
        provider_model = normalize_provider_model(provider, parts[1])
        return ResolvedModelRef(
            provider=provider,
            provider_model=provider_model,
            canonical_model=f"algae/{provider}/{provider_model}",
        )
    return None


def _find_raw_model_candidates(
    raw_model_ref: str,
    entries: list[ModelCatalogEntry],
) -> list[_ModelCandidate]:
    ordered: OrderedDict[str, _ModelCandidate] = OrderedDict()
    for entry in entries:
        provider, provider_model = _split_entry(entry)
        if raw_model_ref == provider_model:
            ordered.setdefault(
                entry.id,
                _ModelCandidate(
                    provider=provider,
                    provider_model=provider_model,
                    canonical_model=entry.id,
                ),
            )
            continue
        for alias in list_provider_aliases(provider):
            if raw_model_ref != alias:
                continue
            normalized = normalize_provider_model(provider, alias)
            if normalized != provider_model:
                continue
            ordered.setdefault(
                entry.id,
                _ModelCandidate(
                    provider=provider,
                    provider_model=provider_model,
                    canonical_model=entry.id,
                ),
            )
            break
    return list(ordered.values())


def _choose_candidate(
    candidates: list[_ModelCandidate],
    *,
    providers_dir: Path | None = None,
) -> _ModelCandidate:
    if len(candidates) == 1:
        return candidates[0]

    resolved_providers_dir = providers_dir or resolve_providers_dir()
    logged_in = [
        candidate
        for candidate in candidates
        if load_provider_credentials(resolved_providers_dir, candidate.provider) is not None
    ]
    if len(logged_in) == 1:
        return logged_in[0]
    if logged_in:
        return logged_in[0]
    return candidates[0]


def _split_entry(entry: ModelCatalogEntry) -> tuple[str, str]:
    _, provider, provider_model = entry.id.split("/", 2)
    return provider, provider_model


__all__ = [
    "ResolvedModelRef",
    "build_openai_model_objects",
    "resolve_requested_model",
]
