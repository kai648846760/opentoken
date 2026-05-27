from __future__ import annotations

import json
from collections.abc import Callable
from collections.abc import Iterator
import re
from typing import Any
from uuid import uuid4

import httpx

from opentoken.gateway.normalized import NormalizedChatRequest
from opentoken.models.provider_credentials import ProviderCredentialRecord
from opentoken.providers._client_cache import BoundedClientCache, close_httpx_backed_client
from opentoken.providers.base import ChatResponse, ProviderAdapter, ProviderRateLimitError
from opentoken.providers.prompts import build_doubao_prompt

_DOUBAO_STATIC_QUERY_PARAMS = {
    "aid": "497858",
    "device_platform": "web",
    "language": "zh",
    "pkg_type": "release_version",
    "real_aid": "497858",
    "region": "CN",
    "samantha_web": "1",
    "sys_region": "CN",
    "use_olympus_account": "1",
    "version_code": "20800",
}


class DoubaoWebClient:
    def __init__(
        self,
        credentials: ProviderCredentialRecord,
        *,
        base_url: str = "https://www.doubao.com",
        client: httpx.Client | None = None,
    ) -> None:
        self._credentials = credentials
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=60.0, trust_env=False)
        self._config = {
            key: str(value)
            for key, value in resolve_doubao_query_params(credentials).items()
        }
        self._conversation_id: str | None = None

    def build_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": self._credentials.user_agent or "Mozilla/5.0",
            "Referer": "https://www.doubao.com/chat/",
            "Origin": "https://www.doubao.com",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Agw-js-conv": "str",
            "Cookie": self._credentials.cookie or "",
        }

    def build_query_params(self) -> dict[str, str]:
        params: dict[str, str] = {}
        for key, value in self._config.items():
            if value:
                params[key] = value
        return params

    def _refresh_query_params(self) -> None:
        """Refresh query params from credentials (e.g., after re-login)."""
        self._config = {
            key: str(value)
            for key, value in resolve_doubao_query_params(self._credentials).items()
        }

    def chat_completion(self, *, message: str, model: str) -> str:
        need_create_conversation = not self._conversation_id
        response = self._client.post(
            f"{self._base_url}/samantha/chat/completion",
            headers=self.build_headers(),
            params=self.build_query_params(),
            json={
                "messages": [
                    {
                        "content": json.dumps({"text": message}),
                        "content_type": 2001,
                        "attachments": [],
                        "references": [],
                    }
                ],
                "completion_option": {
                    "is_regen": False,
                    "with_suggest": True,
                    "need_create_conversation": need_create_conversation,
                    "launch_stage": 1,
                    "is_replace": False,
                    "is_delete": False,
                    "message_from": 0,
                    "event_id": "0",
                },
                "conversation_id": self._conversation_id or "0",
                "local_conversation_id": f"local_16{str(uuid4().int)[:14]}",
                "local_message_id": str(uuid4()),
                "model": model,
            },
        )

        # Handle 401 by re-validating and retrying
        if response.status_code == 401:
            self._refresh_query_params()
            response = self._client.post(
                f"{self._base_url}/samantha/chat/completion",
                headers=self.build_headers(),
                params=self.build_query_params(),
                json={
                    "messages": [
                        {
                            "content": json.dumps({"text": message}),
                            "content_type": 2001,
                            "attachments": [],
                            "references": [],
                        }
                    ],
                    "completion_option": {
                        "is_regen": False,
                        "with_suggest": True,
                        "need_create_conversation": need_create_conversation,
                        "launch_stage": 1,
                        "is_replace": False,
                        "is_delete": False,
                        "message_from": 0,
                        "event_id": "0",
                    },
                    "conversation_id": self._conversation_id or "0",
                    "local_conversation_id": f"local_16{str(uuid4().int)[:14]}",
                    "local_message_id": str(uuid4()),
                    "model": model,
                },
            )

        response.raise_for_status()
        content = _parse_doubao_response_text(response.text)

        # Extract conversation_id for future requests (session tracking)
        self._extract_conversation_id(response.text)

        if not content:
            raise RuntimeError("Doubao chat completion returned no text content.")
        return content

    def iter_chat_completion_text(self, *, message: str, model: str) -> Iterator[str]:
        yield from self._iter_chat_completion_text(message=message, model=model, allow_retry=True)

    def _iter_chat_completion_text(
        self,
        *,
        message: str,
        model: str,
        allow_retry: bool,
    ) -> Iterator[str]:
        need_create_conversation = not self._conversation_id
        payload = {
            "messages": [
                {
                    "content": json.dumps({"text": message}),
                    "content_type": 2001,
                    "attachments": [],
                    "references": [],
                }
            ],
            "completion_option": {
                "is_regen": False,
                "with_suggest": True,
                "need_create_conversation": need_create_conversation,
                "launch_stage": 1,
                "is_replace": False,
                "is_delete": False,
                "message_from": 0,
                "event_id": "0",
            },
            "conversation_id": self._conversation_id or "0",
            "local_conversation_id": f"local_16{str(uuid4().int)[:14]}",
            "local_message_id": str(uuid4()),
            "model": model,
        }

        with self._client.stream(
            "POST",
            f"{self._base_url}/samantha/chat/completion",
            headers=self.build_headers(),
            params=self.build_query_params(),
            json=payload,
        ) as response:
            if response.status_code == 401 and allow_retry:
                response.read()
                self._refresh_query_params()
                yield from self._iter_chat_completion_text(
                    message=message,
                    model=model,
                    allow_retry=False,
                )
                return

            response.raise_for_status()
            captured_lines: list[str] = []
            current_event: str | None = None
            current_data: str | None = None
            for raw_line in response.iter_lines():
                line = raw_line.strip()
                if raw_line:
                    captured_lines.append(raw_line)
                    # Detect a throttle event in the stream. The non-stream path
                    # raises on these codes, but the streaming path used to just
                    # produce no content chunks, so a rate-limited request looked
                    # like an empty/successful answer. Raise as soon as we see the
                    # signal (it arrives before any content), surfacing 429 to the
                    # caller instead of a silent empty stream.
                    if _doubao_payload_is_rate_limited(raw_line):
                        raise ProviderRateLimitError(_DOUBAO_RATE_LIMIT_MESSAGE)
                if not line:
                    if current_event and current_data:
                        try:
                            parsed = json.loads(current_data)
                        except json.JSONDecodeError:
                            parsed = None
                        if isinstance(parsed, dict):
                            for chunk in _extract_doubao_chunks_from_event(current_event, parsed):
                                if chunk:
                                    yield chunk
                    current_event = None
                    current_data = None
                    continue
                if line.startswith("id:") and " event: " in line and " data: " in line:
                    single = line.split(" event: ", 1)[1]
                    event_name, data = single.split(" data: ", 1)
                    try:
                        parsed = json.loads(data)
                    except json.JSONDecodeError:
                        parsed = None
                    if isinstance(parsed, dict):
                        for chunk in _extract_doubao_chunks_from_event(event_name.strip(), parsed):
                            if chunk:
                                yield chunk
                    continue
                data_line = line[6:].strip() if line.startswith("data: ") else line
                samantha_chunks = _extract_samantha_chunks(data_line)
                if samantha_chunks:
                    for chunk in samantha_chunks:
                        if chunk:
                            yield chunk
                    continue
                if line.startswith("event: "):
                    current_event = line[7:].strip()
                    continue
                if line.startswith("data: "):
                    current_data = line[6:].strip()

            if current_event and current_data:
                try:
                    parsed = json.loads(current_data)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    for chunk in _extract_doubao_chunks_from_event(current_event, parsed):
                        if chunk:
                            yield chunk

            self._extract_conversation_id("\n".join(captured_lines))

    def _extract_conversation_id(self, payload: str) -> None:
        """Extract conversation_id from response for session tracking."""
        try:
            for raw_line in payload.splitlines():
                line = raw_line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                if isinstance(data, dict):
                    conv_id = data.get("conversation_id") or data.get("conv_id")
                    if isinstance(conv_id, str) and conv_id:
                        self._conversation_id = conv_id
                        return
        except Exception:
            pass  # Ignore conversation extraction failures


class DoubaoWebAdapter(ProviderAdapter):
    def __init__(
        self,
        *,
        client_factory: Callable[[ProviderCredentialRecord], DoubaoWebClient] | None = None,
    ) -> None:
        self._client_factory = client_factory or (lambda credentials: DoubaoWebClient(credentials))
        self._client_cache: BoundedClientCache[DoubaoWebClient] = BoundedClientCache(closer=close_httpx_backed_client)

    def _get_client(self, credentials: ProviderCredentialRecord) -> DoubaoWebClient:
        key = f"{credentials.provider}:{credentials.cookie}:{credentials.user_agent}"
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
            raise RuntimeError("Missing Doubao credentials. Run `opentoken login doubao` first.")
        client = self._get_client(credentials)
        content = client.chat_completion(
            message=build_doubao_prompt(request),
            model=request.model.rsplit("/", 1)[-1],
        )
        return ChatResponse(model=request.model, content=content)

    def stream_chat(
        self,
        request: NormalizedChatRequest,
        credentials: ProviderCredentialRecord | None = None,
    ) -> Iterator[str] | None:
        if credentials is None:
            raise RuntimeError("Missing Doubao credentials. Run `opentoken login doubao` first.")
        client = self._get_client(credentials)
        stream_method = getattr(client, "iter_chat_completion_text", None)
        if not callable(stream_method):
            return None
        return stream_method(
            message=build_doubao_prompt(request),
            model=request.model.rsplit("/", 1)[-1],
        )


def _extract_doubao_chunks_from_event(event: str, data: dict[str, Any]) -> list[str]:
    chunks: list[str] = []
    if event == "CHUNK_DELTA":
        text = data.get("text")
        if isinstance(text, str) and text:
            chunks.append(text)
    elif event == "STREAM_CHUNK":
        patch_ops = data.get("patch_op")
        if isinstance(patch_ops, list):
            for patch in patch_ops:
                if not isinstance(patch, dict):
                    continue
                patch_value = patch.get("patch_value", {})
                if isinstance(patch_value, dict):
                    content_blocks = patch_value.get("content_block")
                    if isinstance(content_blocks, list):
                        for block in content_blocks:
                            if not isinstance(block, dict):
                                continue
                            block_content = block.get("content", {})
                            if isinstance(block_content, dict):
                                text_block = block_content.get("text_block", {})
                                if isinstance(text_block, dict) and isinstance(
                                    text_block.get("text"), str
                                ):
                                    chunks.append(text_block["text"])
    elif event == "STREAM_MSG_NOTIFY":
        content = data.get("content", {})
        if isinstance(content, dict):
            blocks = content.get("content_block")
            if isinstance(blocks, list):
                for block in blocks:
                    if not isinstance(block, dict):
                        continue
                    block_content = block.get("content", {})
                    if isinstance(block_content, dict):
                        text_block = block_content.get("text_block", {})
                        if isinstance(text_block, dict) and isinstance(text_block.get("text"), str):
                            chunks.append(text_block["text"])
    return chunks


def _extract_samantha_chunks(line: str) -> list[str]:
    chunks: list[str] = []
    try:
        raw = json.loads(line)
    except json.JSONDecodeError:
        return chunks
    if not isinstance(raw, dict):
        return chunks
    if raw.get("event_type") == 2003:
        return chunks
    if raw.get("event_type") != 2001:
        return chunks
    event_data = raw.get("event_data")
    if not isinstance(event_data, str) or not event_data:
        return chunks
    try:
        payload = json.loads(event_data)
    except json.JSONDecodeError:
        return chunks
    if not isinstance(payload, dict) or payload.get("is_finish"):
        return chunks
    message = payload.get("message", {})
    if not isinstance(message, dict):
        return chunks
    content = message.get("content")
    if not isinstance(content, str) or not content:
        return chunks
    try:
        parsed_content = json.loads(content)
    except json.JSONDecodeError:
        return chunks
    if isinstance(parsed_content, dict) and isinstance(parsed_content.get("text"), str):
        chunks.append(parsed_content["text"])
    return chunks


def _doubao_payload_is_rate_limited(payload: str) -> bool:
    """Shared predicate for the non-stream and streaming paths so the latter
    doesn't silently emit empty output when Doubao throttles a 200 stream."""
    lower_payload = payload.lower()
    return (
        "710022004" in payload
        or "710022002" in payload
        or ("rate limit" in lower_payload and ("stream_error" in lower_payload or '"event_type":2005' in lower_payload))
        or ("访问频繁" in payload and '"event_type":2005' in payload)
        or ("请稍后重试" in payload and '"event_type":2005' in payload)
        or ("\"message\":\"block\"" in lower_payload and '"event_type":2005' in lower_payload)
    )


_DOUBAO_RATE_LIMIT_MESSAGE = (
    "Doubao rate limit exceeded. Please wait 10-30 minutes or run "
    "`opentoken login doubao` to refresh credentials."
)


def _parse_doubao_response_text(payload: str) -> str:
    # Check for rate limit error first
    if _doubao_payload_is_rate_limited(payload):
        raise ProviderRateLimitError(_DOUBAO_RATE_LIMIT_MESSAGE)

    chunks: list[str] = []
    current_event: str | None = None
    current_data: str | None = None
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line:
            if current_event and current_data:
                try:
                    parsed = json.loads(current_data)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    chunks.extend(_extract_doubao_chunks_from_event(current_event, parsed))
            current_event = None
            current_data = None
            continue
        if line.startswith("id:") and " event: " in line and " data: " in line:
            single = line.split(" event: ", 1)[1]
            event_name, data = single.split(" data: ", 1)
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                chunks.extend(_extract_doubao_chunks_from_event(event_name.strip(), parsed))
            continue
        data_line = line[6:].strip() if line.startswith("data: ") else line
        samantha_chunks = _extract_samantha_chunks(data_line)
        if samantha_chunks:
            chunks.extend(samantha_chunks)
            continue
        if line.startswith("event: "):
            current_event = line[7:].strip()
            continue
        if line.startswith("data: "):
            current_data = line[6:].strip()
    if current_event and current_data:
        try:
            parsed = json.loads(current_data)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            chunks.extend(_extract_doubao_chunks_from_event(current_event, parsed))
    return "".join(chunks)


def resolve_doubao_query_params(credentials: ProviderCredentialRecord) -> dict[str, str]:
    params = dict(_DOUBAO_STATIC_QUERY_PARAMS)
    optional_fields = {
        "device_id": _resolve_doubao_metadata_value(credentials, "device_id", "device_id"),
        "fp": _resolve_doubao_metadata_value(credentials, "fp", "s_v_web_id"),
        "tea_uuid": _resolve_doubao_metadata_value(credentials, "tea_uuid", "tea_uuid"),
        "web_tab_id": _resolve_doubao_metadata_value(credentials, "web_tab_id", "web_tab_id"),
        "pc_version": _resolve_doubao_metadata_value(credentials, "pc_version", "pc_version"),
        "msToken": _resolve_doubao_metadata_value(credentials, "msToken", "msToken"),
        "a_bogus": _resolve_doubao_metadata_value(credentials, "a_bogus", "a_bogus"),
    }
    for key, value in optional_fields.items():
        if value:
            params[key] = value
    return params


def _resolve_doubao_metadata_value(
    credentials: ProviderCredentialRecord,
    metadata_key: str,
    cookie_name: str,
) -> str:
    saved = str(credentials.metadata.get(metadata_key, "")).strip()
    if saved:
        return saved
    return _extract_cookie_value(credentials.cookie or "", cookie_name)


def _extract_cookie_value(cookie_string: str, name: str) -> str:
    match = re.search(rf"{re.escape(name)}=([^;]+)", cookie_string)
    if match is None:
        return ""
    return match.group(1).strip()
