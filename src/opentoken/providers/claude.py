from __future__ import annotations

import json
import re
from collections.abc import Callable
from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import httpx

from opentoken.gateway.normalized import NormalizedChatRequest
from opentoken.models.model_aliases import normalize_provider_model
from opentoken.models.provider_credentials import ProviderCredentialRecord
from opentoken.providers._client_cache import BoundedClientCache
from opentoken.providers.base import ChatResponse, ProviderAdapter
from opentoken.providers.prompts import build_role_prompt
from opentoken.providers.web_tool_calling import (
    build_web_tool_prompt,
    complete_web_tool_roundtrip,
    parse_web_tool_response,
    request_uses_web_tools,
)


class ClaudeWebClient:
    def __init__(
        self,
        credentials: ProviderCredentialRecord,
        *,
        base_url: str = "https://claude.ai/api",
        client: httpx.Client | None = None,
    ) -> None:
        self._credentials = credentials
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=60.0, trust_env=False)
        self._organization_id = self._resolve_organization_id()
        self._device_id = self._resolve_device_id()

    def _resolve_organization_id(self) -> str | None:
        organization_id = str(self._credentials.metadata.get("organization_id", "")).strip()
        return organization_id or None

    def _resolve_device_id(self) -> str:
        device_id = str(self._credentials.metadata.get("device_id", "")).strip()
        if device_id:
            return device_id
        cookie = self._resolve_cookie()
        match = re.search(r"anthropic-device-id=([^;]+)", cookie)
        if match:
            return match.group(1)
        return str(uuid4())

    def _resolve_cookie(self) -> str:
        if self._credentials.cookie:
            return self._credentials.cookie
        session_key = str(self._credentials.metadata.get("session_key", "")).strip()
        if session_key:
            return f"sessionKey={session_key}"
        return ""

    def build_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Cookie": self._resolve_cookie(),
            "User-Agent": self._credentials.user_agent or "Mozilla/5.0",
            "Accept": "text/event-stream",
            "Referer": "https://claude.ai/",
            "Origin": "https://claude.ai",
            "anthropic-client-platform": "web_claude_ai",
            "anthropic-device-id": self._device_id,
        }

    def discover_organization_id(self) -> str | None:
        if self._organization_id:
            return self._organization_id
        response = self._client.get(
            f"{self._base_url}/organizations",
            headers=self.build_headers(),
        )
        _raise_for_claude_auth(response.status_code)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list) and payload:
            organization_id = payload[0].get("uuid")
            if isinstance(organization_id, str) and organization_id:
                self._organization_id = organization_id
                return organization_id
        return None

    def _conversation_path(self, conversation_id: str | None = None) -> str:
        organization_id = self.discover_organization_id()
        if organization_id and conversation_id:
            return f"/organizations/{organization_id}/chat_conversations/{conversation_id}"
        if organization_id:
            return f"/organizations/{organization_id}/chat_conversations"
        if conversation_id:
            return f"/chat_conversations/{conversation_id}"
        return "/chat_conversations"

    def create_conversation(self) -> str:
        response = self._client.post(
            f"{self._base_url}{self._conversation_path()}",
            headers=self.build_headers(),
            json={
                "name": "Conversation",
                "uuid": str(uuid4()),
            },
        )
        _raise_for_claude_auth(response.status_code)
        response.raise_for_status()
        payload = response.json()
        conversation_id = payload.get("uuid") or payload.get("id")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise RuntimeError("Claude conversation creation returned no conversation id.")
        return conversation_id

    def chat_completion(
        self,
        *,
        message: str,
        model: str,
        conversation_id: str | None = None,
    ) -> str:
        conversation_id = conversation_id or self.create_conversation()
        response = self._client.post(
            f"{self._base_url}{self._conversation_path(conversation_id)}/completion",
            headers=self.build_headers(),
            json={
                "prompt": message,
                "parent_message_uuid": "00000000-0000-4000-8000-000000000000",
                "model": model,
                "timezone": "Asia/Shanghai",
                "rendering_mode": "messages",
                "attachments": [],
                "files": [],
                "locale": "en-US",
                "personalized_styles": [],
                "sync_sources": [],
                "tools": [],
            },
        )
        _raise_for_claude_auth(response.status_code)
        response.raise_for_status()
        content = _parse_claude_sse_text(response.text)
        if not content:
            raise RuntimeError("Claude chat completion returned no text content.")
        return content

    def iter_chat_completion_text(
        self,
        *,
        message: str,
        model: str,
        conversation_id: str | None = None,
    ) -> Iterator[str]:
        conversation_id = conversation_id or self.create_conversation()
        with self._client.stream(
            "POST",
            f"{self._base_url}{self._conversation_path(conversation_id)}/completion",
            headers=self.build_headers(),
            json={
                "prompt": message,
                "parent_message_uuid": "00000000-0000-4000-8000-000000000000",
                "model": model,
                "timezone": "Asia/Shanghai",
                "rendering_mode": "messages",
                "attachments": [],
                "files": [],
                "locale": "en-US",
                "personalized_styles": [],
                "sync_sources": [],
                "tools": [],
            },
        ) as response:
            _raise_for_claude_auth(response.status_code)
            response.raise_for_status()
            for raw_line in response.iter_lines():
                line = raw_line.strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]" or not data:
                    continue
                try:
                    parsed = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    fragment = _extract_text_fragment(parsed)
                    if fragment:
                        yield fragment


class ClaudeWebAdapter(ProviderAdapter):
    def __init__(
        self,
        *,
        client_factory: Callable[[ProviderCredentialRecord], ClaudeWebClient] | None = None,
    ) -> None:
        self._client_factory = client_factory or (lambda credentials: ClaudeWebClient(credentials))
        self._client_cache: BoundedClientCache[ClaudeWebClient] = BoundedClientCache()

    def _client_key(self, credentials: ProviderCredentialRecord) -> str:
        return (
            f"{credentials.provider}:{credentials.cookie}:{credentials.user_agent}:"
            f"{credentials.metadata.get('session_key', '')}"
        )

    def chat(
        self,
        request: NormalizedChatRequest,
        credentials: ProviderCredentialRecord | None = None,
    ) -> ChatResponse:
        if credentials is None:
            raise RuntimeError("Missing Claude credentials. Run `opentoken login claude` first.")
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
                provider="claude",
                invoke=lambda message: client.chat_completion(
                    message=message,
                    model=model,
                ),
            )
        else:
            content = client.chat_completion(
                message=build_role_prompt(request),
                model=model,
            )
            parsed_content, tool_calls, finish_reason = parse_web_tool_response(
                content,
                available_tools=request.tools,
                tool_choice=request.tool_choice,
            )
        return ChatResponse(
            model=request.model,
            content=parsed_content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

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
        stream_method = getattr(client, "iter_chat_completion_text", None)
        if not callable(stream_method):
            return None
        return stream_method(
            message=build_role_prompt(request),
            model=model,
        )


def _raise_for_claude_auth(status_code: int) -> None:
    """Convert claude.ai auth failures into a friendly RuntimeError that the
    gateway's error classifier maps to 401 authentication_error. Without this,
    a 401/403 would fall through to response.raise_for_status() -> httpx
    HTTPStatusError -> the route's 502 api_error bucket, hiding the real fix
    (re-login) behind an opaque upstream-error message."""
    if status_code in (401, 403):
        raise RuntimeError(
            "Claude session expired or invalid. Run `opentoken login claude` to refresh."
        )


def _extract_text_fragment(payload: dict[str, Any]) -> str:
    if payload.get("type") == "content_block_delta":
        delta = payload.get("delta", {})
        if isinstance(delta, dict) and isinstance(delta.get("text"), str):
            return delta["text"]
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        delta = choices[0].get("delta", {})
        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
            return delta["content"]
    for key in ("text", "content", "delta"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return ""


def _parse_claude_sse_text(payload: str) -> str:
    chunks: list[str] = []
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if data == "[DONE]" or not data:
            continue
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            fragment = _extract_text_fragment(parsed)
            if fragment:
                chunks.append(fragment)
    return "".join(chunks)
