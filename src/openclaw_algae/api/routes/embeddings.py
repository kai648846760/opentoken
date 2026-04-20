from __future__ import annotations

import hashlib
from math import ceil
from typing import Any

from fastapi import APIRouter

from openclaw_algae.api.errors import openai_error_response

router = APIRouter()
_DEFAULT_DIMENSIONS = 256


@router.post("/v1/embeddings")
def embeddings(payload: dict[str, Any]):
    model = str(payload.get("model", "")).strip()
    if not model:
        return openai_error_response(
            status_code=400,
            message="model is required",
            error_type="invalid_request_error",
        )
    if "input" not in payload:
        return openai_error_response(
            status_code=400,
            message="input is required",
            error_type="invalid_request_error",
        )

    dimensions = _resolve_dimensions(payload.get("dimensions"))
    normalized_inputs = _normalize_inputs(payload["input"])
    if normalized_inputs is None:
        return openai_error_response(
            status_code=400,
            message="Unsupported embeddings input format.",
            error_type="invalid_request_error",
        )

    data: list[dict[str, object]] = []
    prompt_tokens = 0
    for index, item in enumerate(normalized_inputs):
        blob = _normalize_input_bytes(item)
        prompt_tokens += _estimate_tokens(item)
        data.append(
            {
                "object": "embedding",
                "index": index,
                "embedding": _deterministic_embedding(blob, dimensions=dimensions),
            }
        )

    return {
        "object": "list",
        "data": data,
        "model": model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
        },
    }


def _resolve_dimensions(raw: Any) -> int:
    if raw is None:
        return _DEFAULT_DIMENSIONS
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_DIMENSIONS
    return value if value > 0 else _DEFAULT_DIMENSIONS


def _normalize_inputs(raw: Any) -> list[str | list[int]] | None:
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        if all(isinstance(item, str) for item in raw):
            return list(raw)
        if all(isinstance(item, int) for item in raw):
            return [list(raw)]
        if all(isinstance(item, list) and all(isinstance(token, int) for token in item) for item in raw):
            return [list(item) for item in raw]
    return None


def _normalize_input_bytes(item: str | list[int]) -> bytes:
    if isinstance(item, str):
        return item.encode("utf-8")
    return ",".join(str(token) for token in item).encode("utf-8")


def _estimate_tokens(item: str | list[int]) -> int:
    if isinstance(item, str):
        return max(1, ceil(len(item.encode("utf-8")) / 4))
    return max(1, len(item))


def _deterministic_embedding(blob: bytes, *, dimensions: int) -> list[float]:
    values: list[float] = []
    counter = 0
    while len(values) < dimensions:
        digest = hashlib.sha256(blob + counter.to_bytes(4, "big")).digest()
        for offset in range(0, len(digest), 2):
            if len(values) >= dimensions:
                break
            chunk = digest[offset : offset + 2]
            if len(chunk) < 2:
                continue
            raw_value = int.from_bytes(chunk, "big")
            normalized = (raw_value / 65535.0) * 2.0 - 1.0
            values.append(round(normalized, 8))
        counter += 1
    return values
