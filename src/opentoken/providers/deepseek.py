from __future__ import annotations

import base64
import json
from collections.abc import Callable
from collections.abc import Iterator
from functools import lru_cache
from hashlib import sha256
from math import floor, log2
import struct
from typing import Any, Protocol

import httpx

from opentoken.gateway.normalized import NormalizedChatRequest
from opentoken.models.model_aliases import normalize_provider_model
from opentoken.models.provider_credentials import ProviderCredentialRecord
from opentoken.providers._client_cache import BoundedClientCache, close_httpx_backed_client
from opentoken.providers.base import ChatResponse, ProviderAdapter, ProviderRateLimitError
from opentoken.providers.prompts import stringify_message_content
from opentoken.providers.web_tool_calling import (
    build_web_tool_prompt,
    complete_web_tool_roundtrip,
    parse_web_tool_response,
    request_uses_web_tools,
)
from opentoken.providers.deepseek_hash_v1_wasm import DEEPSEEK_HASH_V1_WASM_B64


class DeepSeekClientProtocol(Protocol):
    def create_chat_session(self) -> str: ...

    def chat_completion(self, *, session_id: str, message: str, model: str) -> str: ...


class DeepSeekWebClient:
    def __init__(
        self,
        credentials: ProviderCredentialRecord,
        *,
        base_url: str = "https://chat.deepseek.com",
        client: httpx.Client | None = None,
    ) -> None:
        self._credentials = credentials
        self._base_url = base_url.rstrip("/")
        # 120s matches NIM's reasoning-friendly default. DeepSeek Reasoner can
        # take >30s to produce a first byte on hard prompts; a 30s timeout cut
        # off legitimate slow reasoning responses with a confusing httpx
        # timeout error.
        self._client = client or httpx.Client(timeout=120.0, trust_env=False)
        self._session_id: str | None = None

    def build_headers(self) -> dict[str, str]:
        authorization = self._credentials.headers.get("authorization", "")
        headers = {
            "Cookie": self._credentials.cookie or "",
            "User-Agent": self._credentials.user_agent
            or "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
            "Content-Type": "application/json",
            "Accept": "*/*",
            "Referer": "https://chat.deepseek.com/",
            "Origin": "https://chat.deepseek.com",
            "x-client-platform": "web",
            "x-client-version": "1.7.0",
            "x-app-version": "20241129.1",
            "x-client-locale": "zh_CN",
            "x-client-timezone-offset": "28800",
        }
        if authorization:
            headers["Authorization"] = authorization
        return headers

    def create_chat_session(self) -> str:
        response = self._request_with_auth_retry(
            "POST",
            f"{self._base_url}/api/v0/chat_session/create",
            json={},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"DeepSeek session creation returned malformed response: {type(payload).__name__}")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise RuntimeError(f"DeepSeek session creation returned no data section.")
        session = data.get("biz_data", {})
        session_id = session.get("id") or session.get("chat_session_id") or ""
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError("DeepSeek session creation returned no session id.")
        return session_id

    def validate_credentials(self) -> str:
        response = self._client.get(
            f"{self._base_url}/api/v0/users/current",
            headers=self.build_headers(),
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"DeepSeek credential validation returned malformed response: {type(payload).__name__}")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise RuntimeError("DeepSeek credential validation returned no data section.")
        token = data.get("biz_data", {}).get("token", "")
        if not isinstance(token, str) or not token:
            raise RuntimeError("DeepSeek credential validation returned no token.")
        self._credentials.headers["authorization"] = f"Bearer {token}"
        return token

    def create_pow_challenge(self, target_path: str) -> dict[str, Any]:
        response = self._request_with_auth_retry(
            "POST",
            f"{self._base_url}/api/v0/chat/create_pow_challenge",
            json={"target_path": target_path},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("DeepSeek PoW challenge response was malformed.")
        challenge = (
            payload.get("data", {}).get("biz_data", {}).get("challenge")
            or payload.get("data", {}).get("challenge")
            or payload.get("challenge")
        )
        if not isinstance(challenge, dict):
            raise RuntimeError("DeepSeek PoW challenge response was malformed.")
        return challenge

    def solve_pow(self, challenge: dict[str, Any]) -> int | float:
        algorithm = str(challenge.get("algorithm", ""))
        target = str(challenge.get("challenge", ""))
        salt = str(challenge.get("salt", ""))
        difficulty = float(challenge.get("difficulty", 0))

        if algorithm == "sha256":
            target_difficulty = floor(log2(difficulty)) if difficulty > 1000 else int(difficulty)
            nonce = 0
            while nonce <= 1_000_000:
                digest = sha256(f"{salt}{target}{nonce}".encode("utf-8")).hexdigest()
                if _leading_zero_bits(digest) >= target_difficulty:
                    return nonce
                nonce += 1
            raise RuntimeError("DeepSeek sha256 PoW solving timed out.")

        if algorithm == "DeepSeekHashV1":
            expire_at = challenge.get("expire_at")
            if expire_at is None or not str(expire_at).strip():
                raise RuntimeError("DeepSeekHashV1 challenge was missing expire_at.")
            answer = _solve_deepseek_hash_v1_wasm(
                challenge=target,
                prefix=f"{salt}_{expire_at}_",
                difficulty=difficulty,
            )
            if answer is None:
                raise RuntimeError("DeepSeekHashV1 failed to find solution")
            return _normalize_deepseek_pow_answer(answer)

        raise RuntimeError(f"Unsupported DeepSeek PoW algorithm: {algorithm}.")

    def _build_completion_payload(self, *, message: str, model: str) -> dict[str, object]:
        return {
            "chat_session_id": self._session_id,
            "parent_message_id": None,
            "prompt": message,
            "ref_file_ids": [],
            "thinking_enabled": model != "deepseek-chat",
            "search_enabled": False,
            "preempt": False,
        }

    def _start_fresh_chat_session(self) -> str:
        try:
            self._session_id = self.create_chat_session()
        except Exception:
            self._session_id = self.create_chat_session()
        return self._session_id

    def _build_pow_headers(self, *, target_path: str) -> dict[str, str]:
        challenge = self.create_pow_challenge(target_path)
        answer = self.solve_pow(challenge)
        pow_response = base64.b64encode(
            json.dumps(
                {
                    **challenge,
                    "answer": answer,
                    "target_path": target_path,
                }
            ).encode("utf-8")
        ).decode("utf-8")
        return {
            **self.build_headers(),
            "x-ds-pow-response": pow_response,
        }

    def chat_completion(self, *, message: str, model: str) -> str:
        self._start_fresh_chat_session()

        target_path = "/api/v0/chat/completion"
        headers = self._build_pow_headers(target_path=target_path)
        response = self._client.post(
            f"{self._base_url}{target_path}",
            headers=headers,
            json=self._build_completion_payload(message=message, model=model),
        )
        if response.status_code == 401:
            self.validate_credentials()
            headers = self._build_pow_headers(target_path=target_path)
            response = self._client.post(
                f"{self._base_url}{target_path}",
                headers=headers,
                json=self._build_completion_payload(message=message, model=model),
            )
        if response.status_code >= 400:
            # Session may be invalid, create new one and retry
            self._session_id = None
            try:
                self._start_fresh_chat_session()
            except Exception:
                pass
            response.raise_for_status()
        _raise_for_deepseek_json_error(response)
        _raise_for_deepseek_sse_error(response.text)
        content = _parse_deepseek_sse_text(response.text)
        if not content:
            raise RuntimeError("DeepSeek chat completion returned no text content.")
        return content

    def iter_chat_completion_text(self, *, message: str, model: str) -> Iterator[str]:
        self._start_fresh_chat_session()
        target_path = "/api/v0/chat/completion"
        yield from self._iter_chat_completion_text(
            message=message,
            model=model,
            target_path=target_path,
            allow_retry=True,
        )

    def _iter_chat_completion_text(
        self,
        *,
        message: str,
        model: str,
        target_path: str,
        allow_retry: bool,
    ) -> Iterator[str]:
        headers = self._build_pow_headers(target_path=target_path)
        with self._client.stream(
            "POST",
            f"{self._base_url}{target_path}",
            headers=headers,
            json=self._build_completion_payload(message=message, model=model),
        ) as response:
            if response.status_code == 401 and allow_retry:
                self.validate_credentials()
                yield from self._iter_chat_completion_text(
                    message=message,
                    model=model,
                    target_path=target_path,
                    allow_retry=False,
                )
                return
            if response.status_code >= 400:
                self._session_id = None
                try:
                    self._session_id = self.create_chat_session()
                except Exception:
                    pass
                response.raise_for_status()

            # Incremental parser: process each SSE line once and emit its
            # delta directly. The previous loop accumulated raw_payload and
            # re-parsed the entire buffer per line (O(n²)) AND re-scanned it
            # for error events (O(n²)) AND diffed against `emitted`, which had
            # its own cascade bug on divergent snapshots. The stateful parser
            # below sidesteps all three by remembering active_fragment_type
            # and event_name across lines.
            parser = _DeepSeekStreamParser()
            for raw_line in response.iter_lines():
                chunk = parser.feed_line(raw_line)
                if chunk:
                    yield chunk
            tail = parser.close()
            if tail:
                yield tail

    def _request_with_auth_retry(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        response = self._client.request(
            method,
            url,
            headers=self.build_headers(),
            **kwargs,
        )
        if response.status_code != 401:
            return response
        self.validate_credentials()
        return self._client.request(
            method,
            url,
            headers=self.build_headers(),
            **kwargs,
        )


class DeepSeekWebAdapter(ProviderAdapter):
    def __init__(
        self,
        *,
        client_factory: Callable[[ProviderCredentialRecord], DeepSeekClientProtocol] | None = None,
    ) -> None:
        self._client_factory = client_factory or (lambda credentials: DeepSeekWebClient(credentials))
        self._client_cache: BoundedClientCache[DeepSeekClientProtocol] = BoundedClientCache(closer=close_httpx_backed_client)

    def _client_key(self, credentials: ProviderCredentialRecord) -> str:
        auth = credentials.headers.get("authorization", "")
        return f"{credentials.provider}:{credentials.cookie}:{auth}:{credentials.user_agent}"

    def chat(
        self,
        request: NormalizedChatRequest,
        credentials: ProviderCredentialRecord | None = None,
    ) -> ChatResponse:
        if credentials is None:
            raise RuntimeError("Missing DeepSeek credentials. Run `opentoken login deepseek` first.")
        key = self._client_key(credentials)
        client = self._client_cache.get(key)
        if client is None:
            client = self._client_factory(credentials)
            self._client_cache.set(key, client)
        model = normalize_provider_model(credentials.provider, request.model.rsplit("/", 1)[-1])
        if request_uses_web_tools(request):
            parsed_content, tool_calls, finish_reason = complete_web_tool_roundtrip(
                request,
                provider="deepseek",
                invoke=lambda message: client.chat_completion(
                    message=message,
                    model=model,
                ),
            )
        else:
            content = client.chat_completion(
                message=_build_deepseek_prompt(request),
                model=model,
            )
            parsed_content, tool_calls, finish_reason = content, [], "stop"
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
        model = normalize_provider_model(credentials.provider, request.model.rsplit("/", 1)[-1])
        stream_method = getattr(client, "iter_chat_completion_text", None)
        if not callable(stream_method):
            return None
        return stream_method(
            message=_build_deepseek_prompt(request),
            model=model,
        )


def _build_deepseek_prompt(request: NormalizedChatRequest) -> str:
    history: list[str] = []
    for message in request.messages:
        role = str(message.get("role", "user"))
        label = {
            "system": "System",
            "assistant": "Assistant",
            "tool": "Tool",
            "user": "User",
        }.get(role, role.title())
        content = _stringify_message_content(message.get("content", ""))
        if content:
            history.append(f"{label}: {content}")
    if not history:
        raise RuntimeError("DeepSeek requests require at least one message.")
    return "\n\n".join(history)


def _stringify_message_content(content: object) -> str:
    return stringify_message_content(content)


def _leading_zero_bits(hex_digest: str) -> int:
    total = 0
    for char in hex_digest:
        value = int(char, 16)
        if value == 0:
            total += 4
            continue
        total += 4 - value.bit_length()
        break
    return total


@lru_cache(maxsize=1)
def _load_deepseek_hash_v1_module() -> tuple[object, object]:
    try:
        from wasmtime import Engine, Module
    except ImportError as exc:
        raise RuntimeError(
            "wasmtime is required for DeepSeekHashV1. Run `uv sync` to install project dependencies."
        ) from exc

    engine = Engine()
    module = Module(engine, base64.b64decode(DEEPSEEK_HASH_V1_WASM_B64))
    return engine, module


def _solve_deepseek_hash_v1_wasm(
    *,
    challenge: str,
    prefix: str,
    difficulty: float,
) -> float | None:
    from wasmtime import Instance, Store

    engine, module = _load_deepseek_hash_v1_module()
    store = Store(engine)
    instance = Instance(store, module, [])
    exports = instance.exports(store)

    try:
        memory = exports["memory"]
        alloc = exports["__wbindgen_export_0"]
        add_to_stack_pointer = exports["__wbindgen_add_to_stack_pointer"]
        wasm_solve = exports["wasm_solve"]
    except KeyError as exc:
        raise RuntimeError("DeepSeekHashV1 wasm module exports were incomplete.") from exc

    challenge_ptr, challenge_len = _write_wasm_utf8(
        memory=memory,
        store=store,
        alloc=alloc,
        value=challenge,
    )
    prefix_ptr, prefix_len = _write_wasm_utf8(
        memory=memory,
        store=store,
        alloc=alloc,
        value=prefix,
    )

    retptr = add_to_stack_pointer(store, -16)
    try:
        wasm_solve(
            store,
            retptr,
            challenge_ptr,
            challenge_len,
            prefix_ptr,
            prefix_len,
            difficulty,
        )
        status = struct.unpack(
            "<i",
            bytes(memory.read(store, retptr, retptr + 4)),
        )[0]
        if status == 0:
            return None
        return struct.unpack(
            "<d",
            bytes(memory.read(store, retptr + 8, retptr + 16)),
        )[0]
    finally:
        add_to_stack_pointer(store, 16)


def _write_wasm_utf8(*, memory: object, store: object, alloc: object, value: str) -> tuple[int, int]:
    payload = value.encode("utf-8")
    ptr = int(alloc(store, len(payload), 1))
    memory.write(store, bytearray(payload), ptr)
    return ptr, len(payload)


def _parse_deepseek_sse_text(payload: str) -> str:
    return _parse_deepseek_sse_text_impl(payload, close_open_think=True)


def _parse_deepseek_sse_text_impl(payload: str, *, close_open_think: bool) -> str:
    chunks: list[str] = []
    active_fragment_type: str | None = None
    saw_fragments = False

    def _switch_fragment(fragment_type: str | None) -> None:
        nonlocal active_fragment_type
        normalized = (fragment_type or "").strip().upper() or None
        if normalized not in {"THINK", "RESPONSE"}:
            if active_fragment_type == "THINK":
                chunks.append("</think>")
            active_fragment_type = None
            return
        if normalized == active_fragment_type:
            return
        if active_fragment_type == "THINK":
            chunks.append("</think>")
        active_fragment_type = normalized
        if active_fragment_type == "THINK":
            chunks.append("<think>")

    def _append_fragment(fragment: dict[str, object]) -> None:
        fragment_type = fragment.get("type")
        content = fragment.get("content")
        if fragment_type is None and not isinstance(content, str):
            _switch_fragment(None)
            return
        if fragment_type is None:
            _switch_fragment("RESPONSE")
        else:
            _switch_fragment(str(fragment_type))
        if isinstance(content, str) and content:
            chunks.append(content)

    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line or not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if data == "[DONE]":
            continue
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            continue

        snapshot = parsed.get("v")
        if isinstance(snapshot, dict):
            response = snapshot.get("response", {})
            if isinstance(response, dict):
                fragments = response.get("fragments", [])
                if isinstance(fragments, list):
                    saw_fragments = True
                    for fragment in fragments:
                        if not isinstance(fragment, dict):
                            continue
                        _append_fragment(fragment)

        if parsed.get("p") == "response/fragments" and parsed.get("o") == "APPEND":
            appended_fragments = parsed.get("v", [])
            if isinstance(appended_fragments, list):
                saw_fragments = True
                for fragment in appended_fragments:
                    if not isinstance(fragment, dict):
                        continue
                    _append_fragment(fragment)
                continue

        if parsed.get("p") == "response" and parsed.get("o") == "BATCH":
            batch_items = parsed.get("v", [])
            if isinstance(batch_items, list):
                for item in batch_items:
                    if not isinstance(item, dict):
                        continue
                    if item.get("p") == "fragments" and item.get("o") == "APPEND":
                        appended_fragments = item.get("v", [])
                        if not isinstance(appended_fragments, list):
                            continue
                        saw_fragments = True
                        for fragment in appended_fragments:
                            if not isinstance(fragment, dict):
                                continue
                            _append_fragment(fragment)

        if isinstance(parsed.get("v"), str):
            path = str(parsed.get("p") or "")
            if not path and (active_fragment_type is not None or not saw_fragments):
                chunks.append(parsed["v"])
                continue
            if "choices" in path:
                _switch_fragment("RESPONSE")
                chunks.append(parsed["v"])
                continue
            if "content" in path and active_fragment_type is not None:
                chunks.append(parsed["v"])
                continue
        if parsed.get("type") == "text" and isinstance(parsed.get("content"), str):
            _switch_fragment("RESPONSE")
            chunks.append(parsed["content"])
            continue
        choice = (parsed.get("choices") or [{}])[0]
        if isinstance(choice, dict):
            delta = choice.get("delta", {})
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                _switch_fragment("RESPONSE")
                chunks.append(delta["content"])
    if close_open_think and active_fragment_type == "THINK":
        chunks.append("</think>")
    return "".join(chunks)


class _DeepSeekStreamParser:
    """Stateful, line-at-a-time DeepSeek SSE parser for the streaming path.

    Mirrors the fragment/think-tag and error-detection semantics of
    `_parse_deepseek_sse_text_impl` + `_raise_for_deepseek_sse_error`, but
    keeps `active_fragment_type` / `saw_fragments` / `event_name` as instance
    state so each line is processed exactly once. The full-buffer functions
    remain for the non-stream path (called a single time, so no O(n²)); keep
    the two in sync if the DeepSeek wire format changes.
    """

    def __init__(self) -> None:
        self._active_fragment_type: str | None = None
        self._saw_fragments = False
        self._event_name = ""

    def feed_line(self, raw_line: str) -> str:
        line = raw_line.strip()
        if not line:
            return ""
        if line.startswith("event: "):
            self._event_name = line[7:].strip()
            return ""
        if not line.startswith("data: "):
            return ""
        data = line[6:].strip()
        if data == "[DONE]" or not data:
            return ""
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            return ""
        if not isinstance(parsed, dict):
            return ""
        self._raise_if_error(parsed)
        chunks: list[str] = []
        self._extract(parsed, chunks)
        return "".join(chunks)

    def close(self) -> str:
        if self._active_fragment_type == "THINK":
            self._active_fragment_type = None
            return "</think>"
        return ""

    def _raise_if_error(self, parsed: dict[str, object]) -> None:
        if self._event_name not in {"hint", "error"}:
            return
        message = str(parsed.get("content") or parsed.get("msg") or parsed.get("message") or "").strip()
        finish_reason = str(parsed.get("finish_reason") or parsed.get("reason") or "").strip()
        parsed_type = str(parsed.get("type") or "").strip().lower()
        if not (message or finish_reason or parsed_type == "error"):
            return
        if _is_deepseek_rate_limit(message=message, finish_reason=finish_reason):
            raise ProviderRateLimitError(f"DeepSeek rate limit: {message or finish_reason}")
        raise RuntimeError(f"DeepSeek upstream error: {message or finish_reason or 'unknown stream error'}")

    def _switch_fragment(self, fragment_type: str | None, chunks: list[str]) -> None:
        normalized = (fragment_type or "").strip().upper() or None
        if normalized not in {"THINK", "RESPONSE"}:
            if self._active_fragment_type == "THINK":
                chunks.append("</think>")
            self._active_fragment_type = None
            return
        if normalized == self._active_fragment_type:
            return
        if self._active_fragment_type == "THINK":
            chunks.append("</think>")
        self._active_fragment_type = normalized
        if self._active_fragment_type == "THINK":
            chunks.append("<think>")

    def _append_fragment(self, fragment: dict[str, object], chunks: list[str]) -> None:
        fragment_type = fragment.get("type")
        content = fragment.get("content")
        if fragment_type is None and not isinstance(content, str):
            self._switch_fragment(None, chunks)
            return
        if fragment_type is None:
            self._switch_fragment("RESPONSE", chunks)
        else:
            self._switch_fragment(str(fragment_type), chunks)
        if isinstance(content, str) and content:
            chunks.append(content)

    def _extract(self, parsed: dict[str, object], chunks: list[str]) -> None:
        snapshot = parsed.get("v")
        if isinstance(snapshot, dict):
            response = snapshot.get("response", {})
            if isinstance(response, dict):
                fragments = response.get("fragments", [])
                if isinstance(fragments, list):
                    self._saw_fragments = True
                    for fragment in fragments:
                        if isinstance(fragment, dict):
                            self._append_fragment(fragment, chunks)

        if parsed.get("p") == "response/fragments" and parsed.get("o") == "APPEND":
            appended_fragments = parsed.get("v", [])
            if isinstance(appended_fragments, list):
                self._saw_fragments = True
                for fragment in appended_fragments:
                    if isinstance(fragment, dict):
                        self._append_fragment(fragment, chunks)
                return

        if parsed.get("p") == "response" and parsed.get("o") == "BATCH":
            batch_items = parsed.get("v", [])
            if isinstance(batch_items, list):
                for item in batch_items:
                    if not isinstance(item, dict):
                        continue
                    if item.get("p") == "fragments" and item.get("o") == "APPEND":
                        appended_fragments = item.get("v", [])
                        if not isinstance(appended_fragments, list):
                            continue
                        self._saw_fragments = True
                        for fragment in appended_fragments:
                            if isinstance(fragment, dict):
                                self._append_fragment(fragment, chunks)

        if isinstance(parsed.get("v"), str):
            path = str(parsed.get("p") or "")
            if not path and (self._active_fragment_type is not None or not self._saw_fragments):
                chunks.append(parsed["v"])
                return
            if "choices" in path:
                self._switch_fragment("RESPONSE", chunks)
                chunks.append(parsed["v"])
                return
            if "content" in path and self._active_fragment_type is not None:
                chunks.append(parsed["v"])
                return

        if parsed.get("type") == "text" and isinstance(parsed.get("content"), str):
            self._switch_fragment("RESPONSE", chunks)
            chunks.append(parsed["content"])
            return

        choice = (parsed.get("choices") or [{}])[0]
        if isinstance(choice, dict):
            delta = choice.get("delta", {})
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                self._switch_fragment("RESPONSE", chunks)
                chunks.append(delta["content"])


def _normalize_deepseek_pow_answer(answer: int | float) -> int | float:
    if isinstance(answer, float) and answer.is_integer():
        return int(answer)
    return answer


def _raise_for_deepseek_json_error(response: httpx.Response) -> None:
    content_type = response.headers.get("content-type", "").lower()
    if "application/json" not in content_type:
        return
    try:
        payload = response.json()
    except ValueError:
        return
    if not isinstance(payload, dict):
        return
    message = str(payload.get("msg", "")).strip() or str(payload.get("message", "")).strip()
    code = payload.get("code")
    if message:
        if _is_deepseek_rate_limit(message=message, finish_reason=""):
            raise ProviderRateLimitError(f"DeepSeek rate limit: {message}")
        if code is not None and str(code).strip():
            raise RuntimeError(f"DeepSeek upstream error {code}: {message}")
        raise RuntimeError(f"DeepSeek upstream error: {message}")


def _raise_for_deepseek_sse_error(payload: str) -> None:
    event_name = ""
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("event: "):
            event_name = line[7:].strip()
            continue
        if not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if data == "[DONE]":
            continue
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        if event_name not in {"hint", "error"}:
            continue
        message = str(parsed.get("content") or parsed.get("msg") or parsed.get("message") or "").strip()
        finish_reason = str(parsed.get("finish_reason") or parsed.get("reason") or "").strip()
        parsed_type = str(parsed.get("type") or "").strip().lower()
        if not (message or finish_reason or parsed_type == "error"):
            continue
        if _is_deepseek_rate_limit(message=message, finish_reason=finish_reason):
            raise ProviderRateLimitError(f"DeepSeek rate limit: {message or finish_reason}")
        raise RuntimeError(f"DeepSeek upstream error: {message or finish_reason or 'unknown stream error'}")


def _is_deepseek_rate_limit(*, message: str, finish_reason: str) -> bool:
    normalized_reason = finish_reason.strip().lower()
    normalized_message = message.strip().lower()
    if normalized_reason in {"rate_limit_reached", "rate_limited"}:
        return True
    return any(
        token in normalized_message
        for token in (
            "rate limit",
            "too many",
            "过于频繁",
            "稍后重试",
            "请求频繁",
            "有消息正在生成",
            "正在生成",
        )
    )
