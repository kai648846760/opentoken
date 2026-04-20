from __future__ import annotations

import json
import re
from collections.abc import Callable
from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import httpx

from openclaw_algae.gateway.normalized import NormalizedChatRequest
from openclaw_algae.models.model_aliases import normalize_provider_model
from openclaw_algae.models.provider_credentials import ProviderCredentialRecord
from openclaw_algae.providers.base import ChatResponse, ProviderAdapter
from openclaw_algae.providers.prompts import build_role_prompt
from openclaw_algae.providers.web_tool_calling import (
    build_web_tool_prompt,
    complete_web_tool_roundtrip,
    parse_web_tool_response,
    request_uses_web_tools,
)


_MODEL_MAP = {
    "xiaomimo-chat": "mimo-v2-flash-studio",
}


class MimoWebClient:
    def __init__(
        self,
        credentials: ProviderCredentialRecord,
        *,
        base_url: str = "https://aistudio.xiaomimimo.com",
        client: httpx.Client | None = None,
    ) -> None:
        self._credentials = credentials
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=60.0, trust_env=False)

    def _extract_cookie_value(self, name: str) -> str:
        cookie = self._credentials.cookie or ""
        patterns = [
            rf'{re.escape(name)}="([^"]*)"',
            rf"{re.escape(name)}=([^;]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, cookie)
            if match:
                return match.group(1)
        return ""

    def build_headers(self) -> dict[str, str]:
        service_token = self._extract_cookie_value("serviceToken")
        bot_ph = self._extract_cookie_value("xiaomichatbot_ph")
        headers = {
            "Cookie": self._credentials.cookie or "",
            "User-Agent": self._credentials.user_agent or "Mozilla/5.0",
            "Content-Type": "application/json",
            "Accept": "text/event-stream, */*",
            "Referer": f"{self._base_url}/",
            "Origin": self._base_url,
            "x-timezone": "Asia/Shanghai",
            "bot_ph": bot_ph,
        }
        if service_token:
            headers["Authorization"] = f"Bearer {service_token}"
        return headers

    def chat_completion(self, *, message: str, model: str) -> str:
        bot_ph = self._extract_cookie_value("xiaomichatbot_ph")
        params = {"xiaomichatbot_ph": bot_ph} if bot_ph else None
        response = self._client.post(
            f"{self._base_url}/open-apis/bot/chat",
            headers=self.build_headers(),
            params=params,
            json={
                "msgId": str(uuid4()).replace("-", ""),
                "conversationId": "0",
                "query": message,
                "isEditedQuery": False,
                "modelConfig": {
                    "enableThinking": False,
                    "webSearchStatus": "disabled",
                    "model": _MODEL_MAP.get(model, model),
                    "temperature": 0.8,
                    "topP": 0.95,
                },
                "multiMedias": [],
            },
        )
        response.raise_for_status()
        content = _parse_mimo_response_text(response.text)
        if not content:
            raise RuntimeError("MiMo chat completion returned no text content.")
        return content

    def iter_chat_completion_text(self, *, message: str, model: str) -> Iterator[str]:
        bot_ph = self._extract_cookie_value("xiaomichatbot_ph")
        params = {"xiaomichatbot_ph": bot_ph} if bot_ph else None
        with self._client.stream(
            "POST",
            f"{self._base_url}/open-apis/bot/chat",
            headers=self.build_headers(),
            params=params,
            json={
                "msgId": str(uuid4()).replace("-", ""),
                "conversationId": "0",
                "query": message,
                "isEditedQuery": False,
                "modelConfig": {
                    "enableThinking": False,
                    "webSearchStatus": "disabled",
                    "model": _MODEL_MAP.get(model, model),
                    "temperature": 0.8,
                    "topP": 0.95,
                },
                "multiMedias": [],
            },
        ) as response:
            response.raise_for_status()
            yield from _iter_mimo_response_text(response.iter_lines())


class MimoWebAdapter(ProviderAdapter):
    def __init__(
        self,
        *,
        client_factory: Callable[[ProviderCredentialRecord], MimoWebClient] | None = None,
    ) -> None:
        self._client_factory = client_factory or (lambda credentials: MimoWebClient(credentials))
        self._client_cache: dict[str, MimoWebClient] = {}

    def _client_key(self, credentials: ProviderCredentialRecord) -> str:
        return f"{credentials.provider}:{credentials.cookie}:{credentials.user_agent}"

    def chat(
        self,
        request: NormalizedChatRequest,
        credentials: ProviderCredentialRecord | None = None,
    ) -> ChatResponse:
        if credentials is None:
            raise RuntimeError(
                "Missing MiMo credentials. Run `openclaw-algae login xiaomi mimo` first."
            )
        key = self._client_key(credentials)
        client = self._client_cache.get(key)
        if client is None:
            client = self._client_factory(credentials)
            self._client_cache[key] = client
        model = normalize_provider_model(
            credentials.provider,
            request.model.rsplit("/", 1)[-1],
        )
        if request_uses_web_tools(request):
            parsed_content, tool_calls, finish_reason = complete_web_tool_roundtrip(
                request,
                provider="mimo",
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
            self._client_cache[key] = client
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


def _extract_mimo_fragment(payload: dict[str, Any]) -> str:
    payload_type = payload.get("type")
    if payload_type in {"text", "output_text"} and isinstance(payload.get("content"), str):
        return payload["content"]
    if payload_type in {"text", "output_text"} and isinstance(payload.get("text"), str):
        return payload["text"]
    if isinstance(payload.get("content"), dict):
        content = payload["content"]
        if isinstance(content.get("text"), str):
            return content["text"]
    if isinstance(payload.get("delta"), str):
        return payload["delta"]
    return ""


def _parse_mimo_response_text(payload: str) -> str:
    chunks: list[str] = []
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("data: "):
            line = line[6:].strip()
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            fragment = _extract_mimo_fragment(parsed)
            if fragment:
                chunks.append(fragment)
    return "".join(chunks)


def _iter_mimo_response_text(lines: Iterator[str]) -> Iterator[str]:
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("data: "):
            line = line[6:].strip()
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            fragment = _extract_mimo_fragment(parsed)
            if fragment:
                yield fragment
