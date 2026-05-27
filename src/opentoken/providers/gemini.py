from __future__ import annotations

import json
import secrets
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


class GeminiApiClient:
    """API client for Gemini web interface."""

    def __init__(
        self,
        credentials: ProviderCredentialRecord,
        *,
        base_url: str = "https://gemini.google.com",
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
            "Referer": f"{self._base_url}/app",
            "Origin": self._base_url,
        }

    def chat_completion(self, *, message: str, model: str) -> str:
        response = self._client.post(
            f"{self._base_url}/_/BardChatUi/data/assistant.lamda.BardLambdaService/GenerateContent",
            headers=self.build_headers(),
            params={
                "bl": "boq_assistant-bard-web-server_20240101.00_p0",
                "at": self._extract_at_token(),
                "_reqid": str(secrets.randbelow(1_000_000)),
                "rt": "c",
            },
            content=f"f.req={json.dumps([[message]])}",
        )

        # Handle 401
        if response.status_code == 401:
            response = self._client.post(
                f"{self._base_url}/_/BardChatUi/data/assistant.lamda.BardLambdaService/GenerateContent",
                headers=self.build_headers(),
                params={
                    "bl": "boq_assistant-bard-web-server_20240101.00_p0",
                    "at": self._extract_at_token(),
                    "_reqid": str(secrets.randbelow(1_000_000)),
                    "rt": "c",
                },
                content=f"f.req={json.dumps([[message]])}",
            )

        raise_for_provider_auth(
            response.status_code, provider="Gemini", login_command="opentoken login gemini"
        )
        response.raise_for_status()
        content = _parse_gemini_response(response.text)
        if not content:
            raise RuntimeError("Gemini chat completion returned no text content.")
        return content

    def iter_chat_completion_text(self, *, message: str, model: str) -> Iterator[str]:
        params = {
            "bl": "boq_assistant-bard-web-server_20240101.00_p0",
            "at": self._extract_at_token(),
            "_reqid": str(secrets.randbelow(1_000_000)),
            "rt": "c",
        }
        with self._client.stream(
            "POST",
            f"{self._base_url}/_/BardChatUi/data/assistant.lamda.BardLambdaService/GenerateContent",
            headers=self.build_headers(),
            params=params,
            content=f"f.req={json.dumps([[message]])}",
        ) as response:
            if response.status_code == 401:
                with self._client.stream(
                    "POST",
                    f"{self._base_url}/_/BardChatUi/data/assistant.lamda.BardLambdaService/GenerateContent",
                    headers=self.build_headers(),
                    params=params,
                    content=f"f.req={json.dumps([[message]])}",
                ) as retry:
                    retry.raise_for_status()
                    yield from _iter_gemini_response(retry.iter_lines())
                    return
            raise_for_provider_auth(
                response.status_code, provider="Gemini", login_command="opentoken login gemini"
            )
            response.raise_for_status()
            yield from _iter_gemini_response(response.iter_lines())

    def _extract_at_token(self) -> str:
        """Extract AT token from cookie for Gemini API."""
        cookie = self._credentials.cookie or ""
        for item in cookie.split(";"):
            if "=" in item:
                k, v = item.strip().split("=", 1)
                if k == "__Secure-1PSIDTS":
                    return v
        return ""


class GeminiWebAdapter(ProviderAdapter):
    def __init__(
        self,
        *,
        client_factory: Callable[[ProviderCredentialRecord], GeminiApiClient] | None = None,
    ) -> None:
        self._client_factory = client_factory or (
            lambda credentials: GeminiApiClient(credentials)
        )
        self._client_cache: BoundedClientCache[GeminiApiClient] = BoundedClientCache(closer=close_httpx_backed_client)

    def _client_key(self, credentials: ProviderCredentialRecord) -> str:
        return f"{credentials.provider}:{credentials.cookie}:{credentials.user_agent}"

    def chat(
        self,
        request: NormalizedChatRequest,
        credentials: ProviderCredentialRecord | None = None,
    ) -> ChatResponse:
        if credentials is None:
            raise RuntimeError("Missing Gemini credentials. Run `opentoken login gemini` first.")
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
                provider="gemini",
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


def _parse_gemini_response(payload: str) -> str:
    """Parse Gemini response and extract text content.

    Gemini uses a custom binary-ish format wrapped in JSON arrays.
    """
    # Gemini responses are typically in format: ["prefix", "[[[\"content\"]]"]\n"]
    try:
        # Concatenate the text fragment from every `data:` line. Returning on
        # the first non-empty fragment (the old behavior) truncated multi-
        # fragment responses to their opening piece.
        fragments: list[str] = []
        for line in payload.splitlines():
            line = line.strip()
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if not data_str:
                continue

            # Gemini wraps response in nested arrays
            try:
                parsed = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if isinstance(parsed, list) and parsed:
                text = _extract_text_from_gemini_array(parsed)
                if text:
                    fragments.append(text)

        return "".join(fragments)
    except Exception:
        return ""


def _extract_text_from_gemini_array(arr: list) -> str:
    """Recursively extract text from Gemini's nested array response."""
    if not arr:
        return ""
    first = arr[0]
    if isinstance(first, str):
        return first
    if isinstance(first, list):
        return _extract_text_from_gemini_array(first)
    return ""


def _iter_gemini_response(lines: Iterator[str]) -> Iterator[str]:
    # Each Gemini SSE `data:` line carries one nested-array fragment. Emit each
    # fragment's text as the line arrives. The previous loop accumulated the
    # whole buffer and re-parsed it per line (O(n²)) AND relied on
    # _parse_gemini_response returning the cumulative text — but that helper
    # returns on the FIRST text-producing line, so the cumulative-diff could
    # never grow and every fragment after the first was silently dropped.
    for raw_line in lines:
        line = raw_line.strip()
        if not line.startswith("data: "):
            continue
        data_str = line[6:].strip()
        if not data_str:
            continue
        try:
            parsed = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list) and parsed:
            text = _extract_text_from_gemini_array(parsed)
            if text:
                yield text
