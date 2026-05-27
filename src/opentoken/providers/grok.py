from __future__ import annotations

import json
from collections.abc import Callable
from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import httpx

from opentoken.gateway.normalized import NormalizedChatRequest
from opentoken.models.model_aliases import normalize_provider_model
from opentoken.models.provider_credentials import ProviderCredentialRecord
from opentoken.providers._client_cache import BoundedClientCache, close_httpx_backed_client
from opentoken.providers.base import ChatResponse, ProviderAdapter, raise_for_provider_auth
from opentoken.providers.prompts import build_role_prompt
from opentoken.providers.web_tool_calling import (
    build_web_tool_prompt,
    complete_web_tool_roundtrip,
    parse_web_tool_response,
    request_uses_web_tools,
)


class GrokApiClient:
    """API client for Grok web interface."""

    def __init__(
        self,
        credentials: ProviderCredentialRecord,
        *,
        base_url: str = "https://grok.com",
        client: httpx.Client | None = None,
    ) -> None:
        self._credentials = credentials
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=60.0, trust_env=False)
        self._conversation_id: str | None = None

    def build_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Cookie": self._credentials.cookie or "",
            "User-Agent": self._credentials.user_agent or "Mozilla/5.0",
            "Referer": f"{self._base_url}/",
            "Origin": self._base_url,
        }

    def _create_conversation(self) -> str | None:
        """Create a fresh conversation. Never reuse an existing one, since a single
        GrokApiClient is shared across requests in a multi-tenant gateway and reusing
        whatever happens to be the first conversation on the account would cross-pollute
        message history between unrelated callers."""
        response = self._client.post(
            f"{self._base_url}/rest/app-chat/conversations",
            headers=self.build_headers(),
            json={},
        )
        if response.status_code == 200:
            try:
                data = response.json()
                conv_id = data.get("conversationId") or data.get("id")
                if isinstance(conv_id, str) and conv_id:
                    self._conversation_id = conv_id
                    return conv_id
            except Exception:
                pass
        return None

    def chat_completion(self, *, message: str, model: str) -> str:
        # Always start a fresh conversation per request to prevent context leakage.
        self._conversation_id = None
        if not self._create_conversation():
            # Conversation creation failed (overwhelmingly a dead session).
            # Raise an auth-flavored error instead of POSTing to
            # `/conversations/None/message`, which would 404 → generic 502.
            raise RuntimeError(
                "Grok could not create a conversation; the session may be expired. "
                "Run `opentoken login grok` to refresh."
            )

        response = self._client.post(
            f"{self._base_url}/rest/app-chat/conversations/{self._conversation_id}/message",
            headers=self.build_headers(),
            json={
                "message": message,
                "fileAttachments": [],
                "imageAttachments": [],
                "disableSearch": False,
                "enableImageGeneration": True,
                "enableImageRecollection": False,
                "sendFinalMetadata": True,
                "customInstructions": "",
                "deepsearchPreset": "",
            },
        )

        # Handle 401
        if response.status_code == 401:
            self._conversation_id = None
            self._create_conversation()
            if self._conversation_id:
                response = self._client.post(
                    f"{self._base_url}/rest/app-chat/conversations/{self._conversation_id}/message",
                    headers=self.build_headers(),
                    json={
                        "message": message,
                        "fileAttachments": [],
                        "imageAttachments": [],
                        "disableSearch": False,
                        "enableImageGeneration": True,
                        "enableImageRecollection": False,
                        "sendFinalMetadata": True,
                        "customInstructions": "",
                        "deepsearchPreset": "",
                    },
                )

        raise_for_provider_auth(
            response.status_code, provider="Grok", login_command="opentoken login grok"
        )
        response.raise_for_status()
        content = _parse_grok_sse_text(response.text)
        if not content:
            raise RuntimeError("Grok chat completion returned no text content.")
        return content

    def iter_chat_completion_text(self, *, message: str, model: str) -> Iterator[str]:
        # Always start a fresh conversation per request (see chat_completion).
        self._conversation_id = None
        if not self._create_conversation():
            raise RuntimeError(
                "Grok could not create a conversation; the session may be expired. "
                "Run `opentoken login grok` to refresh."
            )
        with self._client.stream(
            "POST",
            f"{self._base_url}/rest/app-chat/conversations/{self._conversation_id}/message",
            headers=self.build_headers(),
            json={
                "message": message,
                "fileAttachments": [],
                "imageAttachments": [],
                "disableSearch": False,
                "enableImageGeneration": True,
                "enableImageRecollection": False,
                "sendFinalMetadata": True,
                "customInstructions": "",
                "deepsearchPreset": "",
            },
        ) as response:
            if response.status_code == 401:
                self._conversation_id = None
                self._create_conversation()
                if not self._conversation_id:
                    # Couldn't re-establish a conversation after the 401: the
                    # session is dead. Map to an auth error (re-login) rather
                    # than raise_for_status's generic HTTPStatusError (→ 502),
                    # and never build a `/conversations/None/message` URL.
                    raise_for_provider_auth(
                        response.status_code, provider="Grok", login_command="opentoken login grok"
                    )
                    response.raise_for_status()
                    return
                with self._client.stream(
                    "POST",
                    f"{self._base_url}/rest/app-chat/conversations/{self._conversation_id}/message",
                    headers=self.build_headers(),
                    json={
                        "message": message,
                        "fileAttachments": [],
                        "imageAttachments": [],
                        "disableSearch": False,
                        "enableImageGeneration": True,
                        "enableImageRecollection": False,
                        "sendFinalMetadata": True,
                        "customInstructions": "",
                        "deepsearchPreset": "",
                    },
                ) as retry:
                    raise_for_provider_auth(
                        retry.status_code, provider="Grok", login_command="opentoken login grok"
                    )
                    retry.raise_for_status()
                    yield from _iter_grok_sse_text(retry.iter_lines())
                    return
            raise_for_provider_auth(
                response.status_code, provider="Grok", login_command="opentoken login grok"
            )
            response.raise_for_status()
            yield from _iter_grok_sse_text(response.iter_lines())


class GrokWebAdapter(ProviderAdapter):
    def __init__(
        self,
        *,
        client_factory: Callable[[ProviderCredentialRecord], GrokApiClient] | None = None,
    ) -> None:
        self._client_factory = client_factory or (
            lambda credentials: GrokApiClient(credentials)
        )
        self._client_cache: BoundedClientCache[GrokApiClient] = BoundedClientCache(closer=close_httpx_backed_client)

    def _client_key(self, credentials: ProviderCredentialRecord) -> str:
        return f"{credentials.provider}:{credentials.cookie}:{credentials.user_agent}"

    def chat(
        self,
        request: NormalizedChatRequest,
        credentials: ProviderCredentialRecord | None = None,
    ) -> ChatResponse:
        if credentials is None:
            raise RuntimeError("Missing Grok credentials. Run `opentoken login grok` first.")
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
                provider="grok",
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


def _parse_grok_sse_text(payload: str) -> str:
    """Parse Grok SSE/NDJSON response and extract text content."""
    chunks: list[str] = []
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Strip SSE data: prefix if present
        if line.startswith("data: "):
            line = line[6:].strip()
        elif line.startswith("data:"):
            line = line[5:].strip()

        if line == "[DONE]" or not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue

        # Standard OpenAI-like format
        appended_from_choice = False
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                delta = choice.get("delta", {})
                if isinstance(delta, dict):
                    content = delta.get("content") or delta.get("text")
                    if isinstance(content, str):
                        chunks.append(content)
                        appended_from_choice = True

        # Grok-specific: text/content fields. Only fall back to these if we did NOT
        # already append from choices.delta — some Grok models echo the same payload
        # in both, which would otherwise produce duplicated text.
        if not appended_from_choice:
            text = data.get("text") or data.get("content") or data.get("delta")
            if isinstance(text, str) and text:
                chunks.append(text)

    return "".join(chunks)


def _iter_grok_sse_text(lines: Iterator[str]) -> Iterator[str]:
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("data: "):
            line = line[6:].strip()
        elif line.startswith("data:"):
            line = line[5:].strip()
        if line == "[DONE]" or not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                delta = choice.get("delta", {})
                if isinstance(delta, dict):
                    content = delta.get("content") or delta.get("text")
                    if isinstance(content, str) and content:
                        yield content
                        continue
        text = data.get("text") or data.get("content") or data.get("delta")
        if isinstance(text, str) and text:
            yield text
