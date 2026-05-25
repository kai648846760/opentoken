"""Qwen API clients for both International and China variants."""
from __future__ import annotations

import json
from pathlib import Path
import re
import time
from collections.abc import Callable, Iterator
from typing import Any
from uuid import uuid4

import httpx

from opentoken.config.paths import resolve_state_dir
from opentoken.gateway.normalized import NormalizedChatRequest
from opentoken.models.model_aliases import normalize_provider_model
from opentoken.models.provider_credentials import ProviderCredentialRecord
from opentoken.providers._client_cache import BoundedClientCache
from opentoken.providers.base import ChatResponse, ProviderAdapter
from opentoken.providers.prompts import build_qwen_prompt
from opentoken.providers.web_tool_calling import (
    complete_web_tool_roundtrip,
    request_uses_web_tools,
)

# ── Qwen International ────────────────────────────────────────────────────────

class QwenApiClient:
    """API client for Qwen International (chat.qwen.ai)."""

    def __init__(
        self,
        credentials: ProviderCredentialRecord,
        *,
        base_url: str = "https://chat.qwen.ai",
        client: httpx.Client | None = None,
        state_dir: Path | None = None,
    ) -> None:
        self._credentials = credentials
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=60.0, trust_env=False)
        self._state_dir = state_dir or resolve_state_dir()
        self._chat_id: str | None = None

    def build_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Cookie": self._credentials.cookie or "",
            "User-Agent": self._credentials.user_agent or "Mozilla/5.0",
            "Referer": f"{self._base_url}/",
            "Origin": self._base_url,
        }

    def _ensure_chat_id(self) -> None:
        """Start a fresh upstream chat for each OpenAI-compatible request.

        If Qwen rejects creation (401 typically means the saved cookie is dead),
        raise — silently rebuilding the httpx.Client here used to mask the failure
        so callers proceeded with an empty chat_id and got non-deterministic responses.
        """
        response = self._client.post(
            f"{self._base_url}/api/v2/chats/new",
            headers=self.build_headers(),
            json={},
        )
        if response.status_code == 401:
            raise RuntimeError(
                "Qwen credentials expired or invalid. Run `opentoken login qwen` again."
            )
        response.raise_for_status()
        data = response.json()
        chat_id = data.get("data", {}).get("id") or data.get("data", {}).get("chat_id")
        if not isinstance(chat_id, str) or not chat_id:
            raise RuntimeError(
                f"Qwen returned no chat id (status={response.status_code}). "
                "Run `opentoken login qwen` to refresh the session."
            )
        self._chat_id = chat_id

    def _build_payload(self, *, chat_id: str, message: str, model: str, fid: str) -> dict[str, object]:
        return {
            "stream": True,
            "version": "2.1",
            "incremental_output": True,
            "chat_id": chat_id,
            "chat_mode": "normal",
            "model": model,
            "parent_id": None,
            "messages": [
                {
                    "fid": fid,
                    "parentId": None,
                    "childrenIds": [],
                    "role": "user",
                    "content": message,
                    "user_action": "chat",
                    "files": [],
                    "timestamp": int(time.time()),
                    "models": [model],
                    "chat_type": "t2t",
                    "feature_config": _qwen_feature_config_for_model(model),
                }
            ],
        }

    def _chat_completion_response_text(self, *, message: str, model: str) -> str:
        self._ensure_chat_id()
        chat_id = self._chat_id or ""
        fid = str(uuid4())
        payload = self._build_payload(chat_id=chat_id, message=message, model=model, fid=fid)
        response = self._client.post(
            f"{self._base_url}/api/v2/chat/completions",
            headers=self.build_headers(),
            params={"chat_id": chat_id},
            json=payload,
        )

        if response.status_code == 401 or _qwen_response_requires_fresh_chat(response):
            self._chat_id = None
            self._ensure_chat_id()
            chat_id = self._chat_id or ""
            payload["chat_id"] = chat_id
            payload["messages"][0]["models"] = [model]
            response = self._client.post(
                f"{self._base_url}/api/v2/chat/completions",
                headers=self.build_headers(),
                params={"chat_id": chat_id},
                json=payload,
            )

        response.raise_for_status()
        return response.text

    def iter_chat_completion_text(self, *, message: str, model: str) -> Iterator[str]:
        yield from self._iter_chat_completion_text(message=message, model=model, allow_retry=True)

    def _iter_chat_completion_text(
        self,
        *,
        message: str,
        model: str,
        allow_retry: bool,
    ) -> Iterator[str]:
        self._ensure_chat_id()
        chat_id = self._chat_id or ""
        payload = self._build_payload(
            chat_id=chat_id,
            message=message,
            model=model,
            fid=str(uuid4()),
        )
        yielded_any = False
        try:
            with self._client.stream(
                "POST",
                f"{self._base_url}/api/v2/chat/completions",
                headers=self.build_headers(),
                params={"chat_id": chat_id},
                json=payload,
            ) as response:
                content_type = str(response.headers.get("content-type", "")).lower()
                if response.status_code == 401:
                    body = response.read().decode("utf-8", errors="replace")
                    if allow_retry:
                        self._chat_id = None
                        yield from self._iter_chat_completion_text(
                            message=message,
                            model=model,
                            allow_retry=False,
                        )
                        return
                    response.raise_for_status()
                    raise RuntimeError(body[:300])
                if "text/event-stream" not in content_type:
                    body = response.read().decode("utf-8", errors="replace")
                    if allow_retry and _qwen_payload_requires_fresh_chat(
                        status_code=response.status_code,
                        content_type=content_type,
                        body=body,
                    ):
                        self._chat_id = None
                        yield from self._iter_chat_completion_text(
                            message=message,
                            model=model,
                            allow_retry=False,
                        )
                        return
                    response.raise_for_status()
                    if body:
                        yield body
                    return
                response.raise_for_status()
                for piece in _iter_qwen_sse_text_chunks(response.iter_lines()):
                    yielded_any = True
                    yield piece
        except httpx.ReadTimeout:
            if allow_retry and not yielded_any:
                self._chat_id = None
                yield from self._iter_chat_completion_text(
                    message=message,
                    model=model,
                    allow_retry=False,
                )
                return
            raise

    def chat_completion_text(self, *, message: str, model: str) -> str:
        response_text = self._chat_completion_response_text(message=message, model=model)
        content = _parse_qwen_sse_text(response_text)
        if not content:
            response_text = self._chat_completion_response_text(message=message, model=model)
            content = _parse_qwen_sse_text(response_text)
        return content

    def chat_completion(self, *, message: str, model: str) -> ChatResponse:
        response_text = self._chat_completion_response_text(message=message, model=model)
        content, tool_calls, finish_reason = _parse_qwen_sse_response(response_text)
        if not content and not tool_calls:
            response_text = self._chat_completion_response_text(message=message, model=model)
            content, tool_calls, finish_reason = _parse_qwen_sse_response(response_text)
        if not content and not tool_calls:
            raise RuntimeError(
                f"Qwen Intl chat completion returned no text content. Response: {response_text[:200]}"
            )
        return ChatResponse(
            model=model,
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )


class QwenWebAdapter(ProviderAdapter):
    def __init__(
        self,
        *,
        base_url: str = "https://chat.qwen.ai",
        client_factory: Callable[[ProviderCredentialRecord], QwenApiClient] | None = None,
        stream_client_factory: Callable[[ProviderCredentialRecord], object] | None = None,
    ) -> None:
        self._base_url = base_url
        self._client_factory = client_factory or (
            lambda credentials: QwenApiClient(credentials, base_url=base_url)
        )
        self._stream_client_factory = stream_client_factory
        self._client_cache: BoundedClientCache[QwenApiClient] = BoundedClientCache()

    def _client_key(self, credentials: ProviderCredentialRecord) -> str:
        return f"{credentials.provider}:{credentials.cookie}:{credentials.user_agent}"

    def _get_client(self, credentials: ProviderCredentialRecord) -> QwenApiClient:
        key = self._client_key(credentials)
        client = self._client_cache.get(key)
        if client is None:
            client = self._client_factory(credentials)
            self._client_cache.set(key, client)
        return client

    def chat(
        self,
        request: NormalizedChatRequest,
        credentials: ProviderCredentialRecord | None = None,
    ) -> ChatResponse:
        if credentials is None:
            raise RuntimeError("Missing Qwen credentials. Run `opentoken login qwen international` first.")
        client = self._get_client(credentials)
        model = normalize_provider_model(
            credentials.provider,
            request.model.rsplit("/", 1)[-1],
        )
        if request_uses_web_tools(request):
            parsed_content, tool_calls, finish_reason = complete_web_tool_roundtrip(
                request,
                provider="qwen-intl",
                invoke=lambda message: _invoke_qwen_tool_prompt(client, message=message, model=model),
            )
            return ChatResponse(
                model=request.model,
                content=parsed_content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            )
        response = client.chat_completion(
            message=build_qwen_prompt(request),
            model=model,
        )
        return _coerce_qwen_client_response(request.model, response)

    def stream_chat(
        self,
        request: NormalizedChatRequest,
        credentials: ProviderCredentialRecord | None = None,
    ) -> Iterator[str] | None:
        if credentials is None or request_uses_web_tools(request):
            return None
        model = normalize_provider_model(
            credentials.provider,
            request.model.rsplit("/", 1)[-1],
        )
        if self._stream_client_factory is not None:
            stream_client = self._stream_client_factory(credentials)
            stream_method = getattr(stream_client, "stream_chat_completion", None)
            if callable(stream_method):
                return stream_method(
                    message=build_qwen_prompt(request),
                    model=model,
                )
        client = self._get_client(credentials)
        return client.iter_chat_completion_text(
            message=build_qwen_prompt(request),
            model=model,
        )


# ── Qwen China (chat2.qianwen.com) ────────────────────────────────────────────

_QWEN_CN_BASE_URL = "https://chat2.qianwen.com"
_QWEN_CN_OUTLINE_FALLBACK_MODELS = {
    "Qwen3.5-千问": "Qwen3.5-Flash",
    "Qwen3-Max-Thinking": "Qwen3.5-Flash",
}


def _generate_qwen_cn_nonce() -> str:
    return uuid4().hex.replace("-", "")


def _build_qwen_cn_query_params(credentials: ProviderCredentialRecord) -> dict[str, str]:
    """Build query parameters with anti-bot values for Qwen CN."""
    ts = str(int(time.time() * 1000))
    nonce = _generate_qwen_cn_nonce()

    # Extract cookies for anti-bot params
    cookie = credentials.cookie or ""
    b_user_id = ""
    xsrf_token = ""
    for item in cookie.split(";"):
        if "=" in item:
            k, v = item.strip().split("=", 1)
            if k == "b-user-id":
                b_user_id = v
            elif k == "XSRF-TOKEN":
                xsrf_token = v

    return {
        "biz_id": "ai_qwen",
        "chat_client": "h5",
        "device": "pc",
        "fr": "pc",
        "pr": "qwen",
        "nonce": nonce,
        "timestamp": ts,
        "ut": b_user_id or _generate_qwen_cn_nonce(),
    }


def _build_qwen_cn_headers(credentials: ProviderCredentialRecord) -> dict[str, str]:
    cookie = credentials.cookie or ""
    xsrf_token = ""
    for item in cookie.split(";"):
        if "=" in item:
            k, v = item.strip().split("=", 1)
            if k == "XSRF-TOKEN":
                xsrf_token = v

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Cookie": cookie,
        "User-Agent": credentials.user_agent or "Mozilla/5.0",
        "Referer": f"{_QWEN_CN_BASE_URL}/",
        "Origin": _QWEN_CN_BASE_URL,
    }
    if xsrf_token:
        headers["x-xsrf-token"] = xsrf_token
    return headers


class QwenCnApiClient:
    """Real API client for Qwen China (chat2.qianwen.com)."""

    def __init__(
        self,
        credentials: ProviderCredentialRecord,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self._credentials = credentials
        self._client = client or httpx.Client(timeout=60.0, trust_env=False)

    def _ensure_session_id(self) -> str:
        """Use a fresh session_id for each OpenAI-compatible request."""
        return uuid4().hex

    def _build_payload(self, *, session_id: str, message: str, model: str) -> dict[str, object]:
        return {
            "model": model,
            "messages": [
                {
                    "content": message,
                    "mime_type": "text/plain",
                    "meta_data": {"ori_query": message},
                }
            ],
            "session_id": session_id,
            "parent_req_id": "0",
            "deep_search": "0",
            "req_id": f"req-{uuid4().hex[:12]}",
            "scene": "chat",
            "sub_scene": "chat",
            "temporary": False,
            "from": "default",
            "scene_param": "first_turn",
            "chat_client": "h5",
            "client_tm": str(int(time.time() * 1000)),
            "protocol_version": "v2",
            "biz_id": "ai_qwen",
        }

    def _chat_completion_response_text(
        self,
        *,
        message: str,
        model: str,
        allow_outline_fallback: bool = True,
    ) -> str:
        session_id = self._ensure_session_id()
        params = _build_qwen_cn_query_params(self._credentials)
        headers = _build_qwen_cn_headers(self._credentials)
        payload = self._build_payload(session_id=session_id, message=message, model=model)

        response = self._client.post(
            f"{_QWEN_CN_BASE_URL}/api/v2/chat",
            headers=headers,
            params=params,
            json=payload,
        )

        # Handle 401 by retrying with same session_id and refreshed anti-bot params
        if response.status_code == 401:
            response = self._client.post(
                f"{_QWEN_CN_BASE_URL}/api/v2/chat",
                headers=_build_qwen_cn_headers(self._credentials),
                params=_build_qwen_cn_query_params(self._credentials),
                json=payload,
            )

        response.raise_for_status()
        response_text = response.text
        if allow_outline_fallback and _qwen_cn_payload_indicates_outline_workflow(response_text):
            fallback_model = _qwen_cn_outline_fallback_model(model)
            if fallback_model is not None and fallback_model != model:
                return self._chat_completion_response_text(
                    message=message,
                    model=fallback_model,
                    allow_outline_fallback=False,
                )
        return response_text

    def iter_chat_completion_text(self, *, message: str, model: str) -> Iterator[str]:
        yield from self._iter_chat_completion_text(
            message=message,
            model=model,
            allow_outline_fallback=True,
        )

    def _iter_chat_completion_text(
        self,
        *,
        message: str,
        model: str,
        allow_outline_fallback: bool,
    ) -> Iterator[str]:
        session_id = self._ensure_session_id()
        payload = self._build_payload(session_id=session_id, message=message, model=model)
        with self._client.stream(
            "POST",
            f"{_QWEN_CN_BASE_URL}/api/v2/chat",
            headers=_build_qwen_cn_headers(self._credentials),
            params=_build_qwen_cn_query_params(self._credentials),
            json=payload,
        ) as response:
            if response.status_code == 401:
                with self._client.stream(
                    "POST",
                    f"{_QWEN_CN_BASE_URL}/api/v2/chat",
                    headers=_build_qwen_cn_headers(self._credentials),
                    params=_build_qwen_cn_query_params(self._credentials),
                    json=payload,
                ) as retry:
                    retry.raise_for_status()
                    yield from self._stream_qwen_cn_text_chunks(
                        lines=retry.iter_lines(),
                        message=message,
                        model=model,
                        allow_outline_fallback=allow_outline_fallback,
                    )
                    return
            response.raise_for_status()
            yield from self._stream_qwen_cn_text_chunks(
                lines=response.iter_lines(),
                message=message,
                model=model,
                allow_outline_fallback=allow_outline_fallback,
            )

    def _stream_qwen_cn_text_chunks(
        self,
        *,
        lines: Iterator[str],
        message: str,
        model: str,
        allow_outline_fallback: bool,
    ) -> Iterator[str]:
        emitted = ""
        for raw_line in lines:
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else str(raw_line)
            line = line.strip()
            if not line:
                continue
            if line.startswith("data: "):
                line = line[6:].strip()
            elif line.startswith("data:"):
                line = line[5:].strip()
            else:
                continue
            if line == "[DONE]" or not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            if allow_outline_fallback and _qwen_cn_data_indicates_outline_workflow(data):
                fallback_model = _qwen_cn_outline_fallback_model(model)
                if fallback_model is not None and fallback_model != model:
                    yield from self._iter_chat_completion_text(
                        message=message,
                        model=fallback_model,
                        allow_outline_fallback=False,
                    )
                    return
            for candidate in _extract_qwen_cn_text_candidates(data):
                suffix, emitted = _advance_streamed_text_state(emitted, candidate)
                if suffix:
                    yield suffix

    def chat_completion_text(self, *, message: str, model: str) -> str:
        return _parse_qwen_cn_sse_text(self._chat_completion_response_text(message=message, model=model))

    def chat_completion(self, *, message: str, model: str) -> ChatResponse:
        response_text = self._chat_completion_response_text(
            message=message,
            model=model,
            allow_outline_fallback=True,
        )
        content, tool_calls, finish_reason = _parse_qwen_cn_sse_response(response_text)
        if not content and not tool_calls:
            raise RuntimeError("Qwen CN chat completion returned no text content.")
        return ChatResponse(
            model=model,
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )


class QwenCnWebAdapter(ProviderAdapter):
    def __init__(
        self,
        *,
        client_factory: Callable[[ProviderCredentialRecord], QwenCnApiClient] | None = None,
    ) -> None:
        self._client_factory = client_factory or (
            lambda credentials: QwenCnApiClient(credentials)
        )
        self._client_cache: BoundedClientCache[QwenCnApiClient] = BoundedClientCache()

    def _client_key(self, credentials: ProviderCredentialRecord) -> str:
        return f"{credentials.provider}:{credentials.cookie}:{credentials.user_agent}"

    def chat(
        self,
        request: NormalizedChatRequest,
        credentials: ProviderCredentialRecord | None = None,
    ) -> ChatResponse:
        if credentials is None:
            raise RuntimeError("Missing Qwen CN credentials. Run `opentoken login qwen china` first.")
        key = self._client_key(credentials)
        client = self._client_cache.get(key)
        if client is None:
            client = self._client_factory(credentials)
            self._client_cache.set(key, client)
        model = normalize_provider_model(
            credentials.provider,
            request.model.rsplit("/", 1)[-1],
        )
        if request_uses_web_tools(request):
            parsed_content, tool_calls, finish_reason = complete_web_tool_roundtrip(
                request,
                provider="qwen-cn",
                invoke=lambda message: _invoke_qwen_tool_prompt(client, message=message, model=model),
            )
            return ChatResponse(
                model=request.model,
                content=parsed_content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            )
        response = client.chat_completion(
            message=build_qwen_prompt(request),
            model=model,
        )
        return _coerce_qwen_client_response(request.model, response)

    def stream_chat(
        self,
        request: NormalizedChatRequest,
        credentials: ProviderCredentialRecord | None = None,
    ) -> Iterator[str] | None:
        if credentials is None or request_uses_web_tools(request):
            return None
        key = self._client_key(credentials)
        client = self._client_cache.get(key)
        if client is None:
            client = self._client_factory(credentials)
            self._client_cache.set(key, client)
        model = normalize_provider_model(
            credentials.provider,
            request.model.rsplit("/", 1)[-1],
        )
        return client.iter_chat_completion_text(
            message=build_qwen_prompt(request),
            model=model,
        )


def _qwen_response_requires_fresh_chat(response: httpx.Response) -> bool:
    return _qwen_payload_requires_fresh_chat(
        status_code=response.status_code,
        content_type=str(response.headers.get("content-type", "")).lower(),
        body=response.text,
    )


def _qwen_payload_requires_fresh_chat(*, status_code: int, content_type: str, body: str) -> bool:
    if status_code >= 400:
        return False
    if "text/event-stream" in content_type:
        return False
    return "chat is in progress" in body.lower()


# ── SSE Parsing ───────────────────────────────────────────────────────────────

def _parse_qwen_sse_text(payload: str) -> str:
    """Parse Qwen International SSE response."""
    chunks: list[tuple[str, str]] = []
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:].strip()
        if data_str == "[DONE]" or not data_str:
            continue
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue

        chunks.extend(_extract_qwen_intl_phased_segments(data))

    return _render_qwen_phased_text(chunks)


def _iter_qwen_sse_text_chunks(lines: Iterator[str]) -> Iterator[str]:
    raw_payload = ""
    emitted = ""
    for raw_line in lines:
        raw_payload += f"{raw_line}\n"
        line = raw_line.strip()
        if not line or not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]" or not data_str:
            continue
        candidate = _parse_qwen_sse_text(raw_payload)
        suffix, emitted = _advance_streamed_text_state(emitted, candidate)
        if suffix:
            yield suffix


def _parse_qwen_cn_sse_text(payload: str) -> str:
    """Parse Qwen China SSE response.

    Response format: data:{"error_msg":"","data":{"messages":[{"content":"text","mime_type":"..."}]}}
    Qwen CN sends incremental FULL text in each event. We collect all and take the longest.
    """
    all_texts: list[str] = []
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("data: "):
            line = line[6:].strip()
        elif line.startswith("data:"):
            line = line[5:].strip()
        else:
            continue
        if line == "[DONE]" or not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue

        # Qwen CN format: data.messages[].content
        inner = data.get("data", {})
        if isinstance(inner, dict):
            messages = inner.get("messages", [])
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict):
                        content = msg.get("content")
                        if isinstance(content, str) and content:
                            all_texts.append(content)

        # Also check OpenAI-style format as fallback
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                delta = choice.get("delta", {})
                if isinstance(delta, dict):
                    content = delta.get("content")
                    if isinstance(content, str):
                        all_texts.append(content)

        # Generic fallback
        content = data.get("content") or data.get("text")
        if isinstance(content, str) and content:
            all_texts.append(content)

    # Take the longest text (Qwen CN sends incremental full text)
    if not all_texts:
        return ""
    return max(all_texts, key=len)


def _qwen_cn_outline_fallback_model(model: str) -> str | None:
    return _QWEN_CN_OUTLINE_FALLBACK_MODELS.get(str(model or "").strip())


def _qwen_cn_payload_indicates_outline_workflow(payload: str) -> bool:
    for raw_line in str(payload or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("data: "):
            line = line[6:].strip()
        elif line.startswith("data:"):
            line = line[5:].strip()
        else:
            continue
        if line == "[DONE]" or not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if _qwen_cn_data_indicates_outline_workflow(data):
            return True
    return False


def _qwen_cn_data_indicates_outline_workflow(data: object) -> bool:
    if not isinstance(data, dict):
        return False
    if _qwen_cn_extra_info_indicates_outline(data.get("extra_info")):
        return True
    inner = data.get("data")
    if not isinstance(inner, dict):
        return False
    if _qwen_cn_extra_info_indicates_outline(inner.get("extra_info")):
        return True
    messages = inner.get("messages")
    if not isinstance(messages, list):
        return False
    for item in messages:
        if not isinstance(item, dict):
            continue
        meta_data = item.get("meta_data")
        if isinstance(meta_data, dict):
            if str(meta_data.get("scene", "")).strip().lower() == "chat_writer":
                return True
            multi_load = meta_data.get("multi_load")
            if isinstance(multi_load, list):
                for card in multi_load:
                    if not isinstance(card, dict):
                        continue
                    if str(card.get("type", "")).strip().lower() == "outline":
                        return True
        content = item.get("content")
        if isinstance(content, str) and "[(outline_" in content:
            return True
    return False


def _qwen_cn_extra_info_indicates_outline(extra_info: object) -> bool:
    if not isinstance(extra_info, dict):
        return False
    sub_scene = str(extra_info.get("sub_scene", "")).strip().lower()
    document_scene = str(extra_info.get("document_scene", "")).strip().lower()
    return sub_scene == "creator/outline" or document_scene == "document_longtext"


def _iter_qwen_cn_sse_text_chunks(lines: Iterator[str]) -> Iterator[str]:
    emitted = ""
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("data: "):
            line = line[6:].strip()
        elif line.startswith("data:"):
            line = line[5:].strip()
        else:
            continue
        if line == "[DONE]" or not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        for candidate in _extract_qwen_cn_text_candidates(data):
            suffix, emitted = _advance_streamed_text_state(emitted, candidate)
            if suffix:
                yield suffix


def _extract_qwen_intl_text_candidates(data: dict[str, object]) -> list[str]:
    candidates: list[str] = []
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            delta = choice.get("delta", {})
            if isinstance(delta, dict):
                content = delta.get("content") if _qwen_phase_visible(delta) else None
                if isinstance(content, str) and content:
                    candidates.append(content)
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content") if _qwen_phase_visible(message) else None
                if isinstance(content, str) and content:
                    candidates.append(content)
    content = (
        data.get("content") if _qwen_phase_visible(data) else None
    ) or (
        data.get("text") if _qwen_phase_visible(data) else None
    ) or (
        data.get("delta") if _qwen_phase_visible(data) else None
    )
    if isinstance(content, str) and content:
        candidates.append(content)
    return candidates


def _extract_qwen_intl_phased_segments(data: dict[str, object]) -> list[tuple[str, str]]:
    segments: list[tuple[str, str]] = []
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            delta = choice.get("delta", {})
            if isinstance(delta, dict):
                content = delta.get("content")
                phase = _qwen_phase_name(delta)
                if isinstance(content, str) and content and phase is not None:
                    segments.append((phase, content))
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                phase = _qwen_phase_name(message)
                if isinstance(content, str) and content and phase is not None:
                    segments.append((phase, content))

    for field_name in ("content", "text", "delta"):
        content = data.get(field_name)
        phase = _qwen_phase_name(data)
        if isinstance(content, str) and content and phase is not None:
            segments.append((phase, content))
            break
    return segments


def _extract_qwen_cn_text_candidates(data: dict[str, object]) -> list[str]:
    candidates: list[str] = []
    inner = data.get("data", {})
    if isinstance(inner, dict):
        messages = inner.get("messages", [])
        if isinstance(messages, list):
            latest_message_text = _extract_latest_qwen_cn_message_text(messages)
            if latest_message_text:
                candidates.append(latest_message_text)
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            delta = choice.get("delta", {})
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str) and content:
                    candidates.append(content)
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content:
                    candidates.append(content)
    content = data.get("content") or data.get("text")
    if isinstance(content, str) and content:
        candidates.append(content)
    return candidates


def _extract_latest_qwen_cn_message_text(messages: list[object]) -> str:
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        for key in ("content", "text"):
            value = msg.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _qwen_phase_visible(payload: dict[str, object]) -> bool:
    phase = str(payload.get("phase", "")).strip().lower()
    if not phase:
        return True
    return phase == "answer"


def _qwen_phase_name(payload: dict[str, object]) -> str | None:
    phase = str(payload.get("phase", "")).strip().lower()
    if not phase or phase == "answer":
        return "answer"
    if phase == "think":
        return "think"
    return None


def _render_qwen_phased_text(segments: list[tuple[str, str]]) -> str:
    if not segments:
        return ""

    parts: list[str] = []
    in_think = False
    for phase, text in segments:
        if not text:
            continue
        if phase == "think":
            if not in_think:
                parts.append("<think>")
                in_think = True
            parts.append(text)
            continue
        if in_think:
            parts.append("</think>")
            in_think = False
        parts.append(text)
    if in_think:
        parts.append("</think>")
    return "".join(parts)


def _advance_streamed_text_state(current: str, candidate: str) -> tuple[str, str]:
    if not candidate:
        return "", current
    if not current:
        return candidate, candidate
    if candidate.startswith(current):
        suffix = candidate[len(current) :]
        return suffix, candidate
    if current.startswith(candidate):
        return "", current
    relation, raw_boundary = _normalized_snapshot_relation(current, candidate)
    if relation == "candidate_extends_current":
        return candidate[raw_boundary:], candidate
    if relation == "equivalent":
        return "", candidate
    if relation == "current_extends_candidate":
        return "", current
    if _is_reformatted_snapshot_duplicate(current, candidate):
        return "", candidate
    return candidate, current + candidate


def _is_reformatted_snapshot_duplicate(current: str, candidate: str) -> bool:
    relation, _raw_boundary = _normalized_snapshot_relation(current, candidate)
    if relation == "equivalent":
        return True
    if relation == "candidate_extends_current":
        return False
    if relation == "current_extends_candidate":
        return True
    current_fingerprint = _normalize_snapshot_fingerprint(current)
    candidate_fingerprint = _normalize_snapshot_fingerprint(candidate)
    if not current_fingerprint or not candidate_fingerprint:
        return False
    shorter, longer = sorted((current_fingerprint, candidate_fingerprint), key=len)
    if len(shorter) < 32:
        return False
    if shorter == longer:
        return True
    return shorter in longer and (len(shorter) / len(longer)) >= 0.92


def _normalize_snapshot_fingerprint(text: str) -> str:
    collapsed, _boundaries = _normalize_snapshot_with_boundaries(text)
    return collapsed


def _normalized_snapshot_relation(current: str, candidate: str) -> tuple[str | None, int]:
    current_fingerprint, _current_boundaries = _normalize_snapshot_with_boundaries(current)
    candidate_fingerprint, candidate_boundaries = _normalize_snapshot_with_boundaries(candidate)
    if not current_fingerprint or not candidate_fingerprint:
        return None, 0
    if candidate_fingerprint == current_fingerprint:
        return "equivalent", len(candidate)
    if candidate_fingerprint.startswith(current_fingerprint):
        return "candidate_extends_current", candidate_boundaries[len(current_fingerprint)]
    if current_fingerprint.startswith(candidate_fingerprint):
        return "current_extends_candidate", len(candidate)
    return None, 0


def _normalize_snapshot_with_boundaries(text: str) -> tuple[str, list[int]]:
    raw = str(text or "")
    normalized_parts: list[str] = []
    boundaries: list[int] = [0]
    index = 0
    while index < len(raw):
        char = raw[index]
        if char.isspace() or char in "*_`~":
            index += 1
            continue
        marker_end = _consume_snapshot_list_marker(raw, index, normalized_parts[-1] if normalized_parts else "")
        if marker_end is not None:
            index = marker_end
            continue
        normalized_parts.append(char)
        boundaries.append(index + 1)
        index += 1
    return "".join(normalized_parts), boundaries


def _consume_snapshot_list_marker(text: str, index: int, previous_normalized_char: str) -> int | None:
    if previous_normalized_char and previous_normalized_char not in "。！？.!?:：;；":
        return None
    end = index
    while end < len(text) and _is_snapshot_digit(text[end]):
        end += 1
    if end == index:
        return None
    if end >= len(text) or text[end] not in ".、．)）":
        return None
    return end + 1


def _is_snapshot_digit(char: str) -> bool:
    return ("0" <= char <= "9") or ("０" <= char <= "９")


def _parse_qwen_sse_response(payload: str) -> tuple[str | None, list[dict[str, object]], str]:
    return _extract_qwen_tool_calls(_parse_qwen_sse_text(payload))


def _parse_qwen_cn_sse_response(payload: str) -> tuple[str | None, list[dict[str, object]], str]:
    return _extract_qwen_tool_calls(_parse_qwen_cn_sse_text(payload))


def _extract_qwen_tool_calls(payload: str) -> tuple[str | None, list[dict[str, object]], str]:
    stripped = _strip_qwen_reasoning(payload)
    if not stripped:
        return None, [], "stop"

    tool_calls: list[dict[str, object]] = []
    content_parts: list[str] = []
    last_end = 0

    for match in _QWEN_TOOL_CALL_PATTERN.finditer(stripped):
        prefix = stripped[last_end:match.start()]
        if prefix.strip():
            content_parts.append(prefix)

        attrs = match.group("attrs")
        name = _extract_qwen_tool_attr(attrs, "name")
        if not name:
            content_parts.append(match.group(0))
            last_end = match.end()
            continue

        call_id = _extract_qwen_tool_attr(attrs, "id") or f"call_{uuid4().hex[:8]}"
        arguments = _normalize_qwen_tool_arguments(match.group("body"))
        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        )
        last_end = match.end()

    suffix = stripped[last_end:]
    if suffix.strip():
        content_parts.append(suffix)

    content = "".join(content_parts).strip() or None
    finish_reason = "tool_calls" if tool_calls else "stop"
    return content, tool_calls, finish_reason


def _strip_qwen_reasoning(payload: str) -> str:
    return re.sub(r"<think\b[^>]*>.*?</think\s*>", "", payload, flags=re.IGNORECASE | re.DOTALL).strip()


def _extract_qwen_tool_attr(attrs: str, name: str) -> str | None:
    match = re.search(rf'{name}\s*=\s*["\']?([^"\'>\s]+)', attrs, flags=re.IGNORECASE)
    if match is None:
        return None
    value = match.group(1).strip()
    return value or None


def _normalize_qwen_tool_arguments(arguments: str) -> str:
    cleaned = arguments.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip() or "{}"
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return cleaned
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def _coerce_qwen_client_response(model_ref: str, response: ChatResponse | str) -> ChatResponse:
    if isinstance(response, ChatResponse):
        return ChatResponse(
            model=model_ref,
            content=response.content,
            tool_calls=response.tool_calls,
            finish_reason=response.finish_reason,
        )
    return ChatResponse(model=model_ref, content=str(response))


def _invoke_qwen_tool_prompt(client: object, *, message: str, model: str) -> str:
    text_method = getattr(client, "chat_completion_text", None)
    if callable(text_method):
        return str(text_method(message=message, model=model))
    response = client.chat_completion(message=message, model=model)
    if isinstance(response, ChatResponse):
        return response.content or ""
    return str(response)


_QWEN_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call\b(?P<attrs>[^>]*)>(?P<body>.*?)</tool_call\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)


def _qwen_feature_config_for_model(model: str) -> dict[str, object]:
    return {
        "thinking_enabled": _qwen_model_uses_reasoning(model),
        "output_schema": "phase",
    }


def _qwen_model_uses_reasoning(model: str) -> bool:
    model_lower = model.strip().lower()
    return any(token in model_lower for token in ("think", "reasoner"))
