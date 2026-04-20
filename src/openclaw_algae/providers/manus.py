from __future__ import annotations

import json
import time
from collections.abc import Callable
from collections.abc import Iterator

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


class ManusApiClient:
    def __init__(
        self,
        credentials: ProviderCredentialRecord,
        *,
        base_url: str = "https://api.manus.ai",
        client: httpx.Client | None = None,
        poll_interval_seconds: float = 2.0,
        max_poll_seconds: float = 120.0,
    ) -> None:
        self._credentials = credentials
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=60.0, trust_env=False)
        self._poll_interval_seconds = poll_interval_seconds
        self._max_poll_seconds = max_poll_seconds
        self._api_key = self._resolve_api_key()

    def _resolve_api_key(self) -> str:
        candidates = [
            self._credentials.headers.get("api_key", ""),
            self._credentials.headers.get("API_KEY", ""),
            self._credentials.metadata.get("api_key", ""),
        ]
        for candidate in candidates:
            value = str(candidate).strip()
            if value:
                return value
        raise RuntimeError("Manus API key is required. Run `openclaw-algae login manus --api-key ...`.")

    def _request(self, path: str, *, method: str = "GET", body: object | None = None) -> dict[str, object]:
        response = self._client.request(
            method,
            f"{self._base_url}{path}",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "API_KEY": self._api_key,
            },
            content=json.dumps(body).encode("utf-8") if body is not None else None,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Manus API returned malformed JSON.")
        return payload

    def create_task(self, *, message: str, model: str, conversation_id: str | None = None) -> str:
        payload = self._request(
            "/v1/tasks",
            method="POST",
            body={
                "prompt": message,
                "agentProfile": model,
                "taskMode": "chat",
                **({"taskId": conversation_id} if conversation_id else {}),
            },
        )
        task_id = payload.get("task_id") or payload.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise RuntimeError("Manus task creation returned no task id.")
        return task_id

    def get_task(self, task_id: str) -> dict[str, object]:
        return self._request(f"/v1/tasks/{task_id}")

    def chat_completion(self, *, message: str, model: str, conversation_id: str | None = None) -> str:
        task_id = self.create_task(message=message, model=model, conversation_id=conversation_id)
        deadline = time.time() + self._max_poll_seconds
        while time.time() < deadline:
            task = self.get_task(task_id)
            status = str(task.get("status", ""))
            if status == "completed":
                content = _extract_manus_output_text(task)
                if content:
                    return content
                raise RuntimeError("Manus task completed without text output.")
            if status == "failed":
                raise RuntimeError(str(task.get("error", "Manus task failed")))
            time.sleep(self._poll_interval_seconds)
        raise RuntimeError(f"Manus task timeout after {int(self._max_poll_seconds)}s")

    def iter_chat_completion_text(
        self,
        *,
        message: str,
        model: str,
        conversation_id: str | None = None,
    ) -> Iterator[str]:
        task_id = self.create_task(message=message, model=model, conversation_id=conversation_id)
        deadline = time.time() + self._max_poll_seconds
        emitted = ""
        while time.time() < deadline:
            task = self.get_task(task_id)
            status = str(task.get("status", ""))
            content = _extract_manus_output_text(task)
            suffix, emitted = _advance_streamed_text_state(emitted, content)
            if suffix:
                yield suffix
            if status == "completed":
                if emitted:
                    return
                raise RuntimeError("Manus task completed without text output.")
            if status == "failed":
                raise RuntimeError(str(task.get("error", "Manus task failed")))
            time.sleep(self._poll_interval_seconds)
        raise RuntimeError(f"Manus task timeout after {int(self._max_poll_seconds)}s")


class ManusApiAdapter(ProviderAdapter):
    def __init__(
        self,
        *,
        client_factory: Callable[[ProviderCredentialRecord], ManusApiClient] | None = None,
    ) -> None:
        self._client_factory = client_factory or (lambda credentials: ManusApiClient(credentials))
        self._client_cache: dict[str, ManusApiClient] = {}

    def _client_key(self, credentials: ProviderCredentialRecord) -> str:
        return (
            f"{credentials.provider}:{credentials.headers.get('api_key', '')}:"
            f"{credentials.metadata.get('api_key', '')}"
        )

    def chat(
        self,
        request: NormalizedChatRequest,
        credentials: ProviderCredentialRecord | None = None,
    ) -> ChatResponse:
        if credentials is None:
            raise RuntimeError(
                "Missing Manus credentials. Run `openclaw-algae login manus --api-key ...` first."
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
                provider="manus",
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


def _extract_manus_output_text(payload: dict[str, object]) -> str:
    texts: list[str] = []
    output = payload.get("output")
    if not isinstance(output, list):
        return ""
    for message in output:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if isinstance(text, str) and text:
                texts.append(text)
    return "\n\n".join(texts)


def _advance_streamed_text_state(current: str, candidate: str) -> tuple[str, str]:
    if not candidate:
        return "", current
    if candidate.startswith(current):
        suffix = candidate[len(current) :]
        return suffix, candidate
    if current.startswith(candidate):
        return "", current
    return candidate, current + candidate
