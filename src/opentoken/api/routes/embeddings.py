from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from opentoken.api.errors import openai_error_response

router = APIRouter()


@router.post("/v1/embeddings")
def embeddings(payload: dict[str, Any]):
    # opentoken does not currently host or proxy any real embedding model.
    # The previous implementation returned SHA-256-derived "fake" vectors at a
    # default of 256 dimensions, which silently broke any downstream that used
    # the gateway for RAG / vector search (the values are noise, do not match
    # OpenAI's expected dimensionality, and aren't L2-normalised).
    #
    # Returning honest 501 lets callers detect the missing capability and route
    # the embedding call to a real backend. When a real provider is wired up
    # (e.g. NIM or a hosted embedding service), this handler should swap to
    # that implementation; until then `not_implemented` is the right contract.
    model = str(payload.get("model", "")).strip() if isinstance(payload, dict) else ""
    return openai_error_response(
        status_code=501,
        message=(
            "Embeddings are not implemented in opentoken yet. Configure an "
            "embedding-capable provider, or route /v1/embeddings to a backend "
            "such as OpenAI / NIM / a local sentence-transformer."
        ),
        error_type="not_implemented",
        param=None if not model else "model",
    )
