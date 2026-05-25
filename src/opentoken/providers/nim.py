"""NVIDIA NIM provider — OpenAI-compatible inference for hosted frontier models.

NIM (NVIDIA Inference Microservices) at `integrate.api.nvidia.com/v1` speaks
the OpenAI Chat Completions protocol and exposes a curated catalog of strong
open models (DeepSeek R1, Llama 3.3 70B, Qwen 2.5 family, Kimi K2, …) with a
generous free tier per account (40 RPM at time of writing). It's the lowest-
friction way to give opentoken users immediate access to English-capable
frontier models without browser-harvest auth.

This adapter implements:
- Bearer-token auth against the NIM endpoint
- Streaming + non-streaming chat completions
- Pass-through of OpenAI-shaped tool calls, reasoning_effort, etc.

Credential layout (`~/.opentoken/providers/nim.json`):

    {
      "kind": "api_key",
      "metadata": {
        "api_key": "nvapi-XXXXXXXXXXXXXXXXXXXX",
        "model_chain": [
          "deepseek-ai/deepseek-r1",
          "meta/llama-3.3-70b-instruct",
          "qwen/qwen2.5-72b-instruct"
        ]
      },
      "status": "valid"
    }

The optional `model_chain` field drives the cross-model fallback (see
opentoken.failover.model_chain). If omitted, only the model named in the
request is attempted.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import httpx

from opentoken.gateway.normalized import NormalizedChatRequest
from opentoken.models.provider_credentials import ProviderCredentialRecord
from opentoken.providers._client_cache import BoundedClientCache
from opentoken.providers.base import (
    ChatResponse,
    ProviderAdapter,
    ProviderRateLimitError,
)


NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


def _messages_from_request(request: NormalizedChatRequest) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for msg in request.messages or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if isinstance(content, list):
            # Collapse multi-modal content to plain text — NIM accepts string
            # content uniformly; image parts get described inline upstream.
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
            content = "\n".join(text_parts)
        out.append({"role": role, "content": content if isinstance(content, str) else str(content)})
    return out


def _model_id_from_request(request: NormalizedChatRequest) -> str:
    # Accept both bare model ids ("deepseek-ai/deepseek-r1") and the algae-
    # prefixed form used internally ("algae/nim/deepseek-ai/deepseek-r1").
    raw = (request.model or "").strip()
    if raw.startswith("algae/nim/"):
        return raw[len("algae/nim/"):]
    return raw


class NimChatAdapter(ProviderAdapter):
    """Adapter for NIM OpenAI-compatible chat completions."""

    def __init__(
        self,
        *,
        base_url: str = NIM_BASE_URL,
        client_factory: Callable[[ProviderCredentialRecord], httpx.Client] | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._client_factory = client_factory or self._default_client_factory
        self._client_cache: BoundedClientCache[httpx.Client] = BoundedClientCache(
            closer=lambda client: client.close(),
        )

    def _default_client_factory(self, credentials: ProviderCredentialRecord) -> httpx.Client:
        return httpx.Client(timeout=self._timeout_seconds, trust_env=False)

    def _api_key(self, credentials: ProviderCredentialRecord) -> str:
        key = ""
        if credentials.metadata:
            key = str(credentials.metadata.get("api_key", "")).strip()
        if not key and credentials.headers:
            for header_name in ("authorization", "Authorization", "api_key", "API_KEY"):
                value = str(credentials.headers.get(header_name, "")).strip()
                if value:
                    key = value
                    break
        if not key:
            raise RuntimeError(
                "Missing NVIDIA NIM API key. Run `opentoken login nim --api-key nvapi-...`."
            )
        if key.lower().startswith("bearer "):
            return key[7:].strip()
        return key

    def _client_key(self, credentials: ProviderCredentialRecord) -> str:
        # Don't include the API key in the cache key — credentials may rotate
        # without restarting; key on provider + a stable fingerprint instead.
        return f"{credentials.provider}:{id(credentials)}"

    def _get_client(self, credentials: ProviderCredentialRecord) -> httpx.Client:
        cached = self._client_cache.get(self._client_key(credentials))
        if cached is not None:
            return cached
        client = self._client_factory(credentials)
        self._client_cache.set(self._client_key(credentials), client)
        return client

    def _headers(self, credentials: ProviderCredentialRecord) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key(credentials)}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _build_payload(self, request: NormalizedChatRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": _model_id_from_request(request),
            "messages": _messages_from_request(request),
            "stream": stream,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        return payload

    def chat(
        self,
        request: NormalizedChatRequest,
        credentials: ProviderCredentialRecord | None = None,
    ) -> ChatResponse:
        if credentials is None:
            raise RuntimeError("Missing NIM credentials. Run `opentoken login nim --api-key ...` first.")
        client = self._get_client(credentials)
        response = client.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers(credentials),
            json=self._build_payload(request, stream=False),
        )
        if response.status_code == 429:
            raise ProviderRateLimitError(
                f"NIM rate-limited (HTTP 429) for model {_model_id_from_request(request)}."
            )
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices") or []
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"NIM returned no choices: {body}")
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message") or {}
        content = str(message.get("content") or "")
        raw_tool_calls = message.get("tool_calls") or []
        tool_calls = raw_tool_calls if isinstance(raw_tool_calls, list) else []
        finish_reason = str(choice.get("finish_reason") or "stop")
        return ChatResponse(
            model=request.model,
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

    def stream_chat(
        self,
        request: NormalizedChatRequest,
        credentials: ProviderCredentialRecord | None = None,
    ) -> Iterator[str] | None:
        if credentials is None:
            raise RuntimeError("Missing NIM credentials. Run `opentoken login nim --api-key ...` first.")
        client = self._get_client(credentials)
        return _stream_nim_chunks(
            client=client,
            url=f"{self._base_url}/chat/completions",
            headers=self._headers(credentials),
            payload=self._build_payload(request, stream=True),
        )


def _stream_nim_chunks(
    *,
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> Iterator[str]:
    """Yield only the content deltas from NIM's OpenAI-compatible SSE stream."""
    with client.stream("POST", url, headers=headers, json=payload) as response:
        if response.status_code == 429:
            raise ProviderRateLimitError("NIM rate-limited during streaming chat.")
        response.raise_for_status()
        for line in response.iter_lines():
            line = (line or "").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") if isinstance(chunk, dict) else None
            if not isinstance(choices, list) or not choices:
                continue
            first = choices[0]
            if not isinstance(first, dict):
                continue
            delta = first.get("delta") if isinstance(first.get("delta"), dict) else {}
            content = delta.get("content")
            if isinstance(content, str) and content:
                yield content
