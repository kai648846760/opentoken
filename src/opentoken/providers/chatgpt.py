from __future__ import annotations

import json
from collections.abc import Callable
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from opentoken.config.paths import resolve_state_dir
from opentoken.gateway.normalized import NormalizedChatRequest
from opentoken.models.model_aliases import normalize_provider_model
from opentoken.models.provider_credentials import ProviderCredentialRecord
from opentoken.providers._client_cache import BoundedClientCache
from opentoken.providers.base import ChatResponse, ProviderAdapter, raise_for_provider_auth
from opentoken.providers.prompts import build_role_prompt
from opentoken.storage.provider_sessions import load_provider_session, save_provider_session
from opentoken.providers.web_tool_calling import (
    build_web_tool_prompt,
    complete_web_tool_roundtrip,
    parse_web_tool_response,
    request_uses_web_tools,
)


class ChatGPTApiClient:
    """API client for ChatGPT web interface."""

    def __init__(
        self,
        credentials: ProviderCredentialRecord,
        *,
        base_url: str = "https://chatgpt.com",
        client: httpx.Client | None = None,
        state_dir: Path | None = None,
    ) -> None:
        self._credentials = credentials
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=60.0, trust_env=False)
        self._state_dir = state_dir or resolve_state_dir()
        self._conversation_id: str | None = load_provider_session(
            self._state_dir,
            provider=credentials.provider,
            credentials=credentials,
        ).get("conversation_id")

    def build_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Cookie": self._credentials.cookie or "",
            "User-Agent": self._credentials.user_agent or "Mozilla/5.0",
            "Referer": f"{self._base_url}/",
            "Origin": self._base_url,
        }

    def _build_payload(self, *, message: str, model: str, conversation_id: str | None) -> dict[str, Any]:
        return {
            "action": "next",
            "messages": [
                {
                    "id": str(uuid4()),
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": [message]},
                }
            ],
            "model": model,
            "conversation_id": conversation_id,
            "history_and_training_disabled": False,
        }

    def chat_completion(self, *, message: str, model: str) -> str:
        response = self._client.post(
            f"{self._base_url}/backend-api/conversation",
            headers=self.build_headers(),
            json=self._build_payload(
                message=message,
                model=model,
                conversation_id=self._conversation_id,
            ),
        )

        # Handle 401
        if response.status_code == 401:
            self._conversation_id = None
            response = self._client.post(
                f"{self._base_url}/backend-api/conversation",
                headers=self.build_headers(),
                json=self._build_payload(message=message, model=model, conversation_id=None),
            )

        raise_for_provider_auth(
            response.status_code, provider="ChatGPT", login_command="opentoken login chatgpt"
        )
        response.raise_for_status()
        content = _parse_chatgpt_sse_text(response.text)

        # Extract conversation_id for future requests
        self._extract_conversation_id(response.headers.get("x-conversation-id"))

        if not content:
            raise RuntimeError("ChatGPT chat completion returned no text content.")
        return content

    def iter_chat_completion_text(self, *, message: str, model: str) -> Iterator[str]:
        yield from self._iter_chat_completion_text(
            message=message,
            model=model,
            allow_retry=True,
        )

    def _iter_chat_completion_text(
        self,
        *,
        message: str,
        model: str,
        allow_retry: bool,
    ) -> Iterator[str]:
        payload = self._build_payload(
            message=message,
            model=model,
            conversation_id=self._conversation_id,
        )
        with self._client.stream(
            "POST",
            f"{self._base_url}/backend-api/conversation",
            headers=self.build_headers(),
            json=payload,
        ) as response:
            if response.status_code == 401 and allow_retry:
                self._conversation_id = None
                yield from self._iter_chat_completion_text(
                    message=message,
                    model=model,
                    allow_retry=False,
                )
                return
            raise_for_provider_auth(
                response.status_code, provider="ChatGPT", login_command="opentoken login chatgpt"
            )
            response.raise_for_status()
            self._extract_conversation_id(response.headers.get("x-conversation-id"))

            raw_payload = ""
            emitted = ""
            for raw_line in response.iter_lines():
                raw_payload += f"{raw_line}\n"
                candidate = _parse_chatgpt_sse_text(raw_payload)
                suffix, emitted = _advance_streamed_text_state(emitted, candidate)
                if suffix:
                    yield suffix

    def _extract_conversation_id(self, conv_id: str | None) -> None:
        if isinstance(conv_id, str) and conv_id:
            self._conversation_id = conv_id
            save_provider_session(
                self._state_dir,
                provider=self._credentials.provider,
                credentials=self._credentials,
                state={"conversation_id": conv_id},
            )


class ChatGPTWebAdapter(ProviderAdapter):
    def __init__(
        self,
        *,
        client_factory: Callable[[ProviderCredentialRecord], ChatGPTApiClient] | None = None,
    ) -> None:
        self._client_factory = client_factory or (
            lambda credentials: ChatGPTApiClient(credentials)
        )
        self._client_cache: BoundedClientCache[ChatGPTApiClient] = BoundedClientCache()

    def _client_key(self, credentials: ProviderCredentialRecord) -> str:
        return (
            f"{credentials.provider}:{credentials.cookie}:{credentials.user_agent}:"
            f"{credentials.headers.get('authorization', '')}"
        )

    def chat(
        self,
        request: NormalizedChatRequest,
        credentials: ProviderCredentialRecord | None = None,
    ) -> ChatResponse:
        if credentials is None:
            raise RuntimeError("Missing ChatGPT credentials. Run `opentoken login chatgpt` first.")
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
                provider="chatgpt",
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


def _parse_chatgpt_sse_text(payload: str) -> str:
    """Parse ChatGPT SSE response and extract text content."""
    emitted = ""
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

        # ChatGPT format: message.content.parts
        message = data.get("message", {})
        if isinstance(message, dict):
            content = message.get("content", {})
            if isinstance(content, dict):
                parts = content.get("parts", [])
                if isinstance(parts, list):
                    snapshot = "".join(part for part in parts if isinstance(part, str))
                    _suffix, emitted = _advance_streamed_text_state(emitted, snapshot)

        # Also check delta/content for streaming variants
        text = data.get("text") or data.get("content") or data.get("delta")
        if isinstance(text, str) and text:
            _suffix, emitted = _advance_streamed_text_state(emitted, text)

    return emitted


def _advance_streamed_text_state(current: str, candidate: str) -> tuple[str, str]:
    if not candidate:
        return "", current
    if candidate.startswith(current):
        suffix = candidate[len(current) :]
        return suffix, candidate
    if current.startswith(candidate):
        return "", current
    # Divergent snapshot: `candidate` is neither an extension of nor a prefix of
    # what we've emitted (ChatGPT regenerated / a moderation pass rewrote the
    # message). We can't un-send the bytes already streamed, so emit the new
    # snapshot once — but the new baseline must be `candidate`, NOT
    # `current + candidate`. Concatenating poisons the baseline: the next frame
    # (an extension of `candidate`) would also fail the prefix test and the
    # whole text would be re-emitted on every subsequent frame, cascading into
    # massively duplicated output.
    return candidate, candidate
