"""Unified backend proxy — bridges opentoken to 100+ providers via LiteLLM.

This adapter exists so users can swing opentoken at any OpenAI-compatible
backend (OpenRouter, Groq, Together, Bedrock, Mistral, Cohere, Perplexity,
xAI, Fireworks, DeepInfra, Ollama, LM Studio…) without writing a one-off
adapter for each. The litellm package is intentionally a soft dependency:
opentoken stays installable in lean environments where you don't need this
proxy, and importing this module is a no-op until you actually configure
the `unified` provider.

Credential layout (`~/.opentoken/providers/unified.json`):

    {
      "kind": "api_key",
      "metadata": {
        "api_key_openrouter":   "sk-or-v1-...",
        "api_key_groq":         "gsk_...",
        "api_key_together":     "...",
        "api_key_anthropic":    "sk-ant-...",
        "api_key_openai":       "sk-..."
      },
      "status": "valid"
    }

Model id format coming in:

    unified/<backend>/<model-id>

For example `unified/openrouter/anthropic/claude-3.5-sonnet` becomes a
LiteLLM call of `openrouter/anthropic/claude-3.5-sonnet`. The backend
segment names what LiteLLM calls the provider; everything after it is the
upstream model id verbatim.
"""
from __future__ import annotations

from collections.abc import Iterator

from opentoken.gateway.normalized import NormalizedChatRequest
from opentoken.models.provider_credentials import ProviderCredentialRecord
from opentoken.providers.base import ChatResponse, ProviderAdapter, ProviderRateLimitError


_LITELLM_AVAILABLE: bool | None = None


def _import_litellm():
    """Lazy-import litellm so the soft dep stays soft.

    Imports are cached; the function returns the litellm module or raises a
    descriptive RuntimeError that points the user at the install command.
    """
    global _LITELLM_AVAILABLE
    if _LITELLM_AVAILABLE is False:
        raise RuntimeError(
            "litellm is not installed. Install with `uv add 'litellm>=1.50.0'` "
            "or `uv sync --extra unified` to enable the unified proxy provider."
        )
    try:
        import litellm  # type: ignore
    except ImportError as exc:
        _LITELLM_AVAILABLE = False
        raise RuntimeError(
            "litellm is not installed. Install with `uv add 'litellm>=1.50.0'` "
            "or `uv sync --extra unified` to enable the unified proxy provider."
        ) from exc
    _LITELLM_AVAILABLE = True
    return litellm


def _resolve_api_key(model: str, credentials: ProviderCredentialRecord) -> str | None:
    """Pick the credential key matching the model's backend prefix.

    A single chat request targets exactly one LiteLLM backend (identified by the
    first segment of the model id, e.g. `openrouter/...` or `groq/...`). We pull
    only that backend's key from the credentials and pass it as litellm's
    per-call `api_key=` kwarg. This replaces the previous approach of mutating
    `os.environ` under a global lock, which serialised every unified-proxy call
    process-wide for the duration of the upstream completion (catastrophic for
    long-running streams).
    """
    if not credentials.metadata or not model:
        return None
    backend = model.split("/", 1)[0].strip().lower()
    if not backend:
        return None
    # Accept either api_key_<backend> (preferred, multi-backend) or a generic
    # api_key fallback so single-backend setups don't need namespacing.
    for candidate_key in (f"api_key_{backend}", "api_key"):
        raw = credentials.metadata.get(candidate_key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def _model_id_from_request(request: NormalizedChatRequest) -> str:
    """Strip the `unified/` / `algae/unified/` prefixes from the model id."""
    raw = (request.model or "").strip()
    if raw.startswith("algae/unified/"):
        return raw[len("algae/unified/"):]
    if raw.startswith("unified/"):
        return raw[len("unified/"):]
    return raw


def _messages_from_request(request: NormalizedChatRequest) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for msg in request.messages or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
            content = "\n".join(text_parts)
        out.append({"role": role, "content": content if isinstance(content, str) else str(content)})
    return out


class UnifiedProxyAdapter(ProviderAdapter):
    """ProviderAdapter that fans requests out to LiteLLM-supported backends."""

    def chat(
        self,
        request: NormalizedChatRequest,
        credentials: ProviderCredentialRecord | None = None,
    ) -> ChatResponse:
        if credentials is None:
            raise RuntimeError(
                "Missing credentials for the unified proxy. "
                "Run `opentoken login unified --header api_key_openrouter=sk-or-...` (or similar)."
            )
        litellm = _import_litellm()
        model = _model_id_from_request(request)
        if not model:
            raise RuntimeError("unified proxy: missing backend/model id in request.")
        api_key = _resolve_api_key(model, credentials)
        kwargs: dict[str, object] = {
            "model": model,
            "messages": _messages_from_request(request),
            "stream": False,
            "temperature": request.temperature,
            "tools": request.tools,
            "tool_choice": request.tool_choice,
        }
        if api_key:
            kwargs["api_key"] = api_key
        response = litellm.completion(**kwargs)
        return _convert_litellm_completion(request.model, response)

    def stream_chat(
        self,
        request: NormalizedChatRequest,
        credentials: ProviderCredentialRecord | None = None,
    ) -> Iterator[str] | None:
        if credentials is None:
            raise RuntimeError(
                "Missing credentials for the unified proxy. "
                "Run `opentoken login unified` first."
            )
        litellm = _import_litellm()
        model = _model_id_from_request(request)
        if not model:
            raise RuntimeError("unified proxy: missing backend/model id in request.")
        api_key = _resolve_api_key(model, credentials)
        return _stream_unified(
            litellm=litellm,
            model=model,
            messages=_messages_from_request(request),
            temperature=request.temperature,
            tools=request.tools,
            tool_choice=request.tool_choice,
            api_key=api_key,
        )


def _convert_litellm_completion(requested_model: str, response: object) -> ChatResponse:
    choices = getattr(response, "choices", None) or (
        response.get("choices") if isinstance(response, dict) else None  # type: ignore[union-attr]
    )
    if not choices:
        raise RuntimeError("unified proxy: backend returned no choices.")
    choice = choices[0]
    message = getattr(choice, "message", None) or (
        choice.get("message") if isinstance(choice, dict) else None
    )
    if message is None:
        raise RuntimeError("unified proxy: backend returned no message in choice.")
    content_raw = getattr(message, "content", None) if not isinstance(message, dict) else message.get("content")
    tool_calls_raw = (
        getattr(message, "tool_calls", None) if not isinstance(message, dict) else message.get("tool_calls")
    )
    finish_reason = (
        getattr(choice, "finish_reason", None) if not isinstance(choice, dict) else choice.get("finish_reason")
    ) or "stop"
    return ChatResponse(
        model=requested_model,
        content=str(content_raw or ""),
        tool_calls=list(tool_calls_raw) if isinstance(tool_calls_raw, list) else [],
        finish_reason=str(finish_reason),
    )


def _stream_unified(
    *,
    litellm,
    model: str,
    messages: list[dict[str, object]],
    temperature: float | None,
    tools: list[dict[str, object]] | None,
    tool_choice: object,
    api_key: str | None,
) -> Iterator[str]:
    kwargs: dict[str, object] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "tools": tools,
        "tool_choice": tool_choice,
    }
    if api_key:
        kwargs["api_key"] = api_key
    stream = litellm.completion(**kwargs)
    for chunk in stream:
        delta = _extract_delta(chunk)
        if delta:
            yield delta


def _extract_delta(chunk: object) -> str:
    choices = getattr(chunk, "choices", None) or (
        chunk.get("choices") if isinstance(chunk, dict) else None  # type: ignore[union-attr]
    )
    if not choices:
        return ""
    first = choices[0]
    delta = getattr(first, "delta", None) or (
        first.get("delta") if isinstance(first, dict) else None
    )
    if delta is None:
        return ""
    content = (
        getattr(delta, "content", None) if not isinstance(delta, dict) else delta.get("content")
    )
    return str(content) if isinstance(content, str) else ""
