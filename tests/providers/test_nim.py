from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx
import pytest

from opentoken.failover.model_chain import (
    chain_from_credentials,
    run_with_chain,
    stream_with_chain,
)
from opentoken.gateway.normalized import NormalizedChatRequest
from opentoken.models.provider_credentials import ProviderCredentialRecord
from opentoken.providers.base import ChatResponse, ProviderRateLimitError
from opentoken.providers.nim import NimChatAdapter


def _credentials(api_key: str = "nvapi-test", chain: list[str] | None = None) -> ProviderCredentialRecord:
    metadata: dict[str, str] = {"api_key": api_key}
    if chain is not None:
        metadata["model_chain"] = json.dumps(chain)
    return ProviderCredentialRecord(
        provider="nim",
        kind="api_key",
        cookie="",
        headers={},
        user_agent="",
        metadata=metadata,
        status="valid",
    )


def _request(model: str = "deepseek-ai/deepseek-r1") -> NormalizedChatRequest:
    return NormalizedChatRequest(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
    )


def test_nim_chat_returns_first_choice():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "world"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )
    )
    adapter = NimChatAdapter(
        client_factory=lambda credentials: httpx.Client(transport=transport, trust_env=False)
    )
    response = adapter.chat(_request(), _credentials())
    assert isinstance(response, ChatResponse)
    assert response.content == "world"
    assert response.finish_reason == "stop"


def test_nim_chat_raises_provider_rate_limit_on_429():
    transport = httpx.MockTransport(lambda request: httpx.Response(429, text="rate-limited"))
    adapter = NimChatAdapter(
        client_factory=lambda credentials: httpx.Client(transport=transport, trust_env=False)
    )
    with pytest.raises(ProviderRateLimitError):
        adapter.chat(_request(), _credentials())


def test_nim_requires_api_key():
    adapter = NimChatAdapter()
    bare = ProviderCredentialRecord(
        provider="nim",
        kind="api_key",
        cookie="",
        headers={},
        user_agent="",
        metadata={},
        status="valid",
    )
    with pytest.raises(RuntimeError, match="API key"):
        adapter.chat(_request(), bare)


def test_chain_from_credentials_returns_strings_in_order():
    chain = chain_from_credentials(
        _credentials(chain=["deepseek-ai/deepseek-r1", "qwen/qwen2.5-72b-instruct"])
    )
    assert chain == ["deepseek-ai/deepseek-r1", "qwen/qwen2.5-72b-instruct"]


def test_chain_from_credentials_filters_non_strings_and_blanks():
    creds = ProviderCredentialRecord(
        provider="nim",
        kind="api_key",
        cookie="",
        headers={},
        user_agent="",
        metadata={"model_chain": json.dumps(["a", "", 1, None, "b"])},
        status="valid",
    )
    assert chain_from_credentials(creds) == ["a", "b"]


def test_chain_from_credentials_handles_invalid_json():
    creds = ProviderCredentialRecord(
        provider="nim",
        kind="api_key",
        cookie="",
        headers={},
        user_agent="",
        metadata={"model_chain": "not-json"},
        status="valid",
    )
    assert chain_from_credentials(creds) == []


def test_run_with_chain_falls_back_on_rate_limit():
    attempts: list[str] = []

    def invoke(req: NormalizedChatRequest) -> ChatResponse:
        attempts.append(req.model)
        if req.model in {"first-model", "second-model"}:
            raise ProviderRateLimitError(f"limited:{req.model}")
        return ChatResponse(model=req.model, content="hit")

    result = run_with_chain(
        _request("first-model"),
        ["second-model", "third-model"],
        invoke,
    )
    assert attempts == ["first-model", "second-model", "third-model"]
    assert result.content == "hit"
    assert result.model == "third-model"


def test_run_with_chain_reraises_when_all_models_rate_limit():
    def invoke(req: NormalizedChatRequest) -> ChatResponse:
        raise ProviderRateLimitError(f"limited:{req.model}")

    with pytest.raises(ProviderRateLimitError):
        run_with_chain(_request("a"), ["b", "c"], invoke)


def test_run_with_chain_skips_duplicate_models():
    seen: list[str] = []

    def invoke(req: NormalizedChatRequest) -> ChatResponse:
        seen.append(req.model)
        return ChatResponse(model=req.model, content="ok")

    run_with_chain(_request("a"), ["a", "b"], invoke)
    # Even though chain repeats "a", the requested model should only run once.
    assert seen == ["a"]


def test_stream_with_chain_returns_iterator_for_first_successful_model():
    attempts: list[str] = []

    def invoke(req: NormalizedChatRequest):
        attempts.append(req.model)
        if req.model == "first":
            raise ProviderRateLimitError("nope")
        return iter(["hi"])

    iterator = stream_with_chain(_request("first"), ["second"], invoke)
    assert iterator is not None
    assert list(iterator) == ["hi"]
    assert attempts == ["first", "second"]


def test_stream_with_chain_falls_back_on_lazy_rate_limit_during_first_chunk():
    # Real-world shape: the stream adapter returns a lazy generator whose
    # upstream HTTP (and 429 detection) only runs on the first __next__().
    # stream_with_chain must prime that first chunk inside the fallback loop so
    # the rate-limit triggers a hop to the next model instead of surfacing as a
    # mid-stream error to the caller.
    attempts: list[str] = []

    def lazy_rate_limited():
        raise ProviderRateLimitError("429 on first chunk")
        yield  # pragma: no cover - makes this a generator

    def good():
        yield "pong"

    def invoke(req: NormalizedChatRequest):
        attempts.append(req.model)
        if req.model == "first":
            return lazy_rate_limited()
        return good()

    iterator = stream_with_chain(_request("first"), ["second"], invoke)
    assert iterator is not None
    assert list(iterator) == ["pong"]
    # Both models were invoked: first primed -> 429 -> fell back to second.
    assert attempts == ["first", "second"]


def test_stream_with_chain_reraises_when_all_models_lazy_rate_limit():
    def lazy_rate_limited():
        raise ProviderRateLimitError("429")
        yield  # pragma: no cover

    iterator_factory = lambda req: lazy_rate_limited()  # noqa: E731
    with pytest.raises(ProviderRateLimitError):
        stream_with_chain(_request("a"), ["b"], iterator_factory)
