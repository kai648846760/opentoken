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


def test_nim_stream_wraps_reasoning_deltas_with_think_tags():
    """Reasoning models stream their chain of thought as delta.reasoning_content
    BEFORE the answer arrives in delta.content. Open <think> on the first
    reasoning delta, close it on the first content delta. Balanced span so the
    projector treats it correctly."""

    def handler(request):
        body = "\n".join(
            f"data: {chunk}"
            for chunk in [
                '{"choices":[{"delta":{"reasoning_content":"step1"}}]}',
                '{"choices":[{"delta":{"reasoning_content":" step2"}}]}',
                '{"choices":[{"delta":{"content":"answer"}}]}',
                '{"choices":[{"delta":{"content":" more"}}]}',
                "[DONE]",
            ]
        ) + "\n"
        return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    adapter = NimChatAdapter(
        client_factory=lambda credentials: httpx.Client(transport=transport, trust_env=False)
    )
    pieces = list(adapter.stream_chat(_request(), _credentials()) or ())
    assert pieces == ["<think>", "step1", " step2", "</think>", "answer", " more"]


def test_nim_stream_closes_unfinished_think_span():
    """If the stream ends mid-reasoning (truncation / abort), the <think>
    open emitted earlier must still get its </think>, or the projector will
    treat the rest of the response as hidden."""

    def handler(request):
        body = (
            'data: {"choices":[{"delta":{"reasoning_content":"abrupt"}}]}\n'
            'data: [DONE]\n'
        )
        return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    adapter = NimChatAdapter(
        client_factory=lambda credentials: httpx.Client(transport=transport, trust_env=False)
    )
    pieces = list(adapter.stream_chat(_request(), _credentials()) or ())
    assert pieces == ["<think>", "abrupt", "</think>"]


def test_nim_chat_wraps_reasoning_content_in_think_tags():
    # DeepSeek R1 / NIM reasoning models put their chain of thought in
    # message.reasoning_content. The gateway should preserve it (wrapped in
    # <think>) instead of silently dropping it.
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "answer",
                            "reasoning_content": "step 1\nstep 2",
                        },
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
    assert response.content == "<think>step 1\nstep 2</think>answer"


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


def test_nim_model_chain_falls_back_through_router(tmp_path, monkeypatch) -> None:
    """End-to-end: PoolAwareRouter wires run_with_chain into _call_http_provider
    for NIM. When the first model raises ProviderRateLimitError the router
    transparently retries the next model in the chain.
    """
    import json as _json
    from opentoken.gateway.router import PoolAwareRouter
    from opentoken.gateway.normalized import NormalizedChatRequest
    from opentoken.providers.nim import NimChatAdapter
    from opentoken.storage.provider_store import save_provider_credentials
    from opentoken.models.catalog import ModelCatalogEntry

    providers_dir = tmp_path / "providers"
    save_provider_credentials(
        providers_dir,
        ProviderCredentialRecord(
            provider="nim",
            kind="api_key",
            cookie="",
            headers={},
            user_agent="",
            metadata={
                "api_key": "nvapi-test",
                # chain: first model rate-limits, second succeeds.
                "model_chain": _json.dumps(["first-model", "second-model"]),
            },
            status="valid",
        ),
    )

    calls: list[str] = []

    class FakeNim(NimChatAdapter):
        def chat(self, request, credentials=None):
            # Mirror the real adapter: strip algae/nim/ to get the wire-level
            # model id, then take action on that. The chain replays models in
            # the form they're configured (bare wire ids), which after the
            # router populates request.model becomes algae/nim/<id> on the
            # first attempt and bare <id> on chain fallbacks.
            from opentoken.providers.nim import _model_id_from_request

            wire = _model_id_from_request(request)
            calls.append(wire)
            if wire == "first-model":
                raise ProviderRateLimitError("NIM 429 on first-model")
            return ChatResponse(model=request.model, content="hi", finish_reason="stop")

    # The PoolAwareRouter is created with our fake adapter and a stubbed catalog.
    monkeypatch.setattr(
        "opentoken.models.openai_compat.load_model_catalog",
        lambda providers_dir=None: [
            ModelCatalogEntry(id="algae/nim/first-model", provider="opentoken", name="first"),
            ModelCatalogEntry(id="algae/nim/second-model", provider="opentoken", name="second"),
        ],
    )
    router = PoolAwareRouter(
        providers_dir=providers_dir,
        adapters={"nim": FakeNim()},
    )

    response = router.chat(
        NormalizedChatRequest(
            model="algae/nim/first-model",
            messages=[{"role": "user", "content": "hi"}],
        )
    )

    assert response.content == "hi"
    # The router first tries the requested model (algae/nim/first-model -> wire
    # "first-model"), gets a 429, then falls back through the chain. Because
    # "first-model" appears in the chain too, _resolve_attempt_order produces
    # [algae/nim/first-model, first-model (already-tried via dedupe... actually
    # it stays since dedupe is on the prefixed form), second-model]; the chain
    # helper retries until "second-model" succeeds.
    assert "second-model" in calls
    assert calls[-1] == "second-model"
