"""Cross-model fallback chain.

When a provider rate-limits a specific model (HTTP 429 / ProviderRateLimitError),
opentoken can hop to the next model in a user-defined chain instead of returning
the error to the caller. This is especially useful for backends like NIM that
expose multiple equivalent-tier models — when DeepSeek R1 throttles, swing to
Llama 3.3 70B and the caller never sees the bump.

The chain is configured per provider in the credentials JSON:

    {
      "metadata": {
        "model_chain": [
          "deepseek-ai/deepseek-r1",
          "meta/llama-3.3-70b-instruct",
          "qwen/qwen2.5-72b-instruct"
        ]
      }
    }

`run_with_chain` finds the requested model in the chain (or appends it to the
front if absent) and tries each downstream model on rate-limit. The current
model the caller asked for is always tried first, even if the chain has it
later — we never demote a request the client explicitly asked for.
"""
from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Iterator
from typing import TypeVar

from opentoken.gateway.normalized import NormalizedChatRequest
from opentoken.models.provider_credentials import ProviderCredentialRecord
from opentoken.providers.base import ChatResponse, ProviderRateLimitError


logger = logging.getLogger(__name__)

T = TypeVar("T")


def chain_from_credentials(credentials: ProviderCredentialRecord | None) -> list[str]:
    if credentials is None or not credentials.metadata:
        return []
    raw = credentials.metadata.get("model_chain")
    parsed: object = raw
    # ProviderCredentialRecord stores metadata as dict[str, str], so a chain
    # configured by `opentoken login nim --model-chain ...` lives there as a
    # JSON-encoded string. Accept either a literal list (in-memory test path)
    # or the JSON-string form.
    if isinstance(raw, str):
        try:
            parsed = __import__("json").loads(raw)
        except ValueError:
            return []
    if not isinstance(parsed, list):
        return []
    chain: list[str] = []
    for item in parsed:
        if isinstance(item, str):
            value = item.strip()
            if value:
                chain.append(value)
    return chain


def _resolve_attempt_order(requested_model: str, chain: list[str]) -> list[str]:
    """Return the list of models to try, requested first, chain alternatives next."""
    seen: set[str] = set()
    ordered: list[str] = []
    if requested_model:
        ordered.append(requested_model)
        seen.add(requested_model)
    for model in chain:
        if model in seen:
            continue
        ordered.append(model)
        seen.add(model)
    return ordered


def _request_with_model(request: NormalizedChatRequest, model: str) -> NormalizedChatRequest:
    if request.model == model:
        return request
    cloned = copy.deepcopy(request)
    # Pydantic v2: set via __setattr__ goes through validation; that's fine here.
    cloned.model = model  # type: ignore[misc]
    return cloned


def run_with_chain(
    request: NormalizedChatRequest,
    chain: list[str],
    invoke: Callable[[NormalizedChatRequest], T],
) -> T:
    """Invoke `invoke` against successive models in the chain on rate-limit.

    The first model attempted is whatever `request.model` already is; chain
    members are tried in order if the previous attempt raises ProviderRateLimitError.
    Returns whatever the eventually-successful `invoke` returns. If every model
    rate-limits, the last ProviderRateLimitError is re-raised.
    """
    attempts = _resolve_attempt_order(request.model, chain)
    if not attempts:
        return invoke(request)

    last_exc: ProviderRateLimitError | None = None
    for model in attempts:
        try:
            return invoke(_request_with_model(request, model))
        except ProviderRateLimitError as exc:
            logger.info(
                "model_chain_rate_limited model=%s remaining=%d",
                model,
                len(attempts) - attempts.index(model) - 1,
            )
            last_exc = exc
            continue
    assert last_exc is not None
    raise last_exc


def stream_with_chain(
    request: NormalizedChatRequest,
    chain: list[str],
    invoke: Callable[[NormalizedChatRequest], Iterator[str] | None],
) -> Iterator[str] | None:
    """Like run_with_chain but for streaming responses.

    Note: we can only fall back BEFORE any bytes are yielded. Once the stream
    starts emitting tokens we have to surface the rate-limit error mid-stream.
    """
    attempts = _resolve_attempt_order(request.model, chain)
    if not attempts:
        return invoke(request)

    last_exc: ProviderRateLimitError | None = None
    for model in attempts:
        try:
            iterator = invoke(_request_with_model(request, model))
        except ProviderRateLimitError as exc:
            last_exc = exc
            continue
        if iterator is None:
            return None
        return iterator
    assert last_exc is not None
    raise last_exc
