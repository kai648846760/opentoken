from __future__ import annotations

import json
import struct
from collections.abc import Callable
from collections.abc import Iterator

import httpx

from opentoken.gateway.normalized import NormalizedChatRequest
from opentoken.models.model_aliases import normalize_provider_model
from opentoken.models.provider_credentials import ProviderCredentialRecord
from opentoken.providers._client_cache import BoundedClientCache
from opentoken.providers.base import ChatResponse, ProviderAdapter, ProviderRateLimitError
from opentoken.providers.prompts import build_role_prompt
from opentoken.providers.web_tool_calling import (
    build_web_tool_prompt,
    complete_web_tool_roundtrip,
    parse_web_tool_response,
    request_uses_web_tools,
)


class KimiWebClient:
    def __init__(self, credentials: ProviderCredentialRecord) -> None:
        self._credentials = credentials
        self._base_url = "https://www.kimi.com"
        self._client = httpx.Client(timeout=60.0, trust_env=False)

    def _get_kimi_auth(self) -> str:
        cookie_str = self._credentials.cookie or ""
        for item in cookie_str.split(";"):
            if "=" in item:
                k, v = item.strip().split("=", 1)
                if k in ("kimi-auth", "access_token"):
                    return v
        raise RuntimeError("kimi-auth not found in cookies")

    def _build_request_buffer(self, *, message: str, model: str) -> bytes:
        scenario = "SCENARIO_K1" if "k1" in model else "SCENARIO_K2"
        req = {
            "scenario": scenario,
            "message": {
                "role": "user",
                "blocks": [{"message_id": "", "text": {"content": message}}],
                "scenario": scenario,
            },
            "options": {"thinking": False},
        }
        req_json = json.dumps(req).encode("utf-8")
        return struct.pack(">BI", 0, len(req_json)) + req_json

    def _build_headers(self) -> dict[str, str]:
        kimi_auth = self._get_kimi_auth()
        return {
            "Content-Type": "application/connect+json",
            "Connect-Protocol-Version": "1",
            "Authorization": f"Bearer {kimi_auth}",
            "Origin": self._base_url,
            "Referer": f"{self._base_url}/",
            "User-Agent": self._credentials.user_agent or "Mozilla/5.0",
        }

    def chat_completion(self, *, message: str, model: str) -> str:
        request_buffer = self._build_request_buffer(message=message, model=model)
        for attempt in range(2):
            response = self._client.post(
                f"{self._base_url}/apiv2/kimi.gateway.chat.v1.ChatService/Chat",
                headers=self._build_headers(),
                content=request_buffer,
            )
            response.raise_for_status()
            try:
                return self._parse_response(response.content)
            except RuntimeError as exc:
                if "empty response" not in str(exc).lower() or attempt == 1:
                    raise
        raise RuntimeError("Kimi returned empty response")

    def iter_chat_completion_text(self, *, message: str, model: str) -> Iterator[str]:
        with self._client.stream(
            "POST",
            f"{self._base_url}/apiv2/kimi.gateway.chat.v1.ChatService/Chat",
            headers=self._build_headers(),
            content=self._build_request_buffer(message=message, model=model),
        ) as response:
            response.raise_for_status()
            yield from self._iter_response_chunks(response.iter_bytes())

    def _parse_response(self, data: bytes) -> str:
        result = "".join(self._iter_response_chunks(iter((data,)))).strip()
        if not result:
            raise RuntimeError("Kimi returned empty response")
        return result

    def _iter_response_chunks(self, byte_chunks: Iterator[bytes]) -> Iterator[str]:
        buffer = b""
        emitted = ""
        for incoming in byte_chunks:
            if incoming:
                buffer += incoming
            while len(buffer) >= 5:
                length = struct.unpack(">I", buffer[1:5])[0]
                if len(buffer) < 5 + length:
                    break
                chunk = buffer[5 : 5 + length]
                buffer = buffer[5 + length :]
                payload = _decode_kimi_frame(chunk)
                if payload is None:
                    continue
                candidate = _extract_kimi_text_candidate(payload)
                suffix, emitted = _advance_streamed_text_state(emitted, candidate)
                if suffix:
                    yield suffix


class KimiWebAdapter(ProviderAdapter):
    def __init__(
        self, *, client_factory: Callable[[ProviderCredentialRecord], KimiWebClient] | None = None
    ) -> None:
        self._client_factory = client_factory or (lambda cred: KimiWebClient(cred))
        self._client_cache: BoundedClientCache[KimiWebClient] = BoundedClientCache()

    def _client_key(self, credentials: ProviderCredentialRecord) -> str:
        return f"{credentials.provider}:{credentials.cookie}:{credentials.user_agent}"

    def chat(
        self, request: NormalizedChatRequest, credentials: ProviderCredentialRecord | None = None
    ) -> ChatResponse:
        if credentials is None:
            raise RuntimeError("Missing Kimi credentials. Run `opentoken login kimi` first.")
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
                provider="kimi",
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


def _decode_kimi_frame(chunk: bytes) -> dict[str, object] | None:
    try:
        payload = json.loads(chunk.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("error"):
        error_detail = payload["error"]
        error_code = error_detail.get("code", "") if isinstance(error_detail, dict) else ""
        if error_code == "resource_exhausted":
            raise ProviderRateLimitError(f"Kimi rate limit: {error_detail}")
        raise RuntimeError(f"Kimi error: {error_detail}")
    return payload


def _extract_kimi_text_candidate(payload: dict[str, object]) -> str:
    op = payload.get("op", "")
    if op not in ("set", "append"):
        return ""
    block = payload.get("block", {})
    if not isinstance(block, dict):
        return ""
    text_obj = block.get("text", {})
    if not isinstance(text_obj, dict):
        return ""
    text = text_obj.get("content", "")
    return text if isinstance(text, str) else ""


def _advance_streamed_text_state(current: str, candidate: str) -> tuple[str, str]:
    if not candidate:
        return "", current
    if candidate.startswith(current):
        suffix = candidate[len(current) :]
        return suffix, candidate
    if current.startswith(candidate):
        return "", current
    return candidate, current + candidate
