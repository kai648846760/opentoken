#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from openclaw_algae.config.app_config import load_or_create_app_config
from openclaw_algae.config.paths import resolve_app_config_path, resolve_providers_dir
from openclaw_algae.storage.provider_store import list_provider_credentials


THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>[\s\S]*?</think\s*>", re.IGNORECASE)
THINK_ONLY_RE = re.compile(r"^\s*(?:<think\b[^>]*>[\s\S]*?</think\s*>)+\s*$", re.IGNORECASE)
RATE_LIMIT_TEXT_RE = re.compile(r"rate[\s_-]*limit|too many|过于频繁|稍后重试|请求频繁|频率过高", re.IGNORECASE)
TRANSIENT_TEXT_RE = re.compile(
    r"connection error|fetch failed|temporarily unavailable|service unavailable|timed out|timeout|bad gateway|gateway timeout",
    re.IGNORECASE,
)


@dataclass(slots=True)
class LiveCaseResult:
    provider: str
    category: str
    name: str
    success: bool
    status_code: int
    latency_ms: float
    detail: str = ""
    preview: str = ""


@dataclass(slots=True)
class ProviderRunContext:
    provider: str
    primary_model: str
    think_model: str | None
    attachment_file_id: str


@dataclass(slots=True, frozen=True)
class ProviderExecutionPolicy:
    min_interval_seconds: float
    max_attempts: int
    retry_base_delay_seconds: float


DEFAULT_POLICY = ProviderExecutionPolicy(min_interval_seconds=0.25, max_attempts=4, retry_base_delay_seconds=4.0)
PROVIDER_POLICIES: dict[str, ProviderExecutionPolicy] = {
    "deepseek": ProviderExecutionPolicy(min_interval_seconds=2.0, max_attempts=6, retry_base_delay_seconds=10.0),
    "qwen-intl": ProviderExecutionPolicy(min_interval_seconds=0.25, max_attempts=5, retry_base_delay_seconds=4.0),
    "qwen-cn": ProviderExecutionPolicy(min_interval_seconds=0.25, max_attempts=5, retry_base_delay_seconds=4.0),
    "kimi": ProviderExecutionPolicy(min_interval_seconds=3.0, max_attempts=6, retry_base_delay_seconds=10.0),
    "doubao": ProviderExecutionPolicy(min_interval_seconds=0.5, max_attempts=6, retry_base_delay_seconds=6.0),
    "glm-cn": ProviderExecutionPolicy(min_interval_seconds=3.0, max_attempts=6, retry_base_delay_seconds=10.0),
}


class LiveProviderRunner:
    def __init__(self, *, base_url: str, api_key: str, timeout_seconds: float = 90.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            timeout=timeout_seconds,
            trust_env=False,
        )
        self.results: list[LiveCaseResult] = []
        self._last_request_finished_at = 0.0

    def close(self) -> None:
        self._client.close()

    def healthcheck(self) -> None:
        response = self._client.get("/health")
        response.raise_for_status()

    def discover_provider_models(self) -> dict[str, list[str]]:
        response = self._client.get("/v1/models")
        response.raise_for_status()
        payload = response.json()
        items = payload.get("data", [])
        models: dict[str, list[str]] = {}
        if not isinstance(items, list):
            return models
        for item in items:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id", ""))
            if not model_id.startswith("algae/"):
                continue
            parts = model_id.split("/", 2)
            if len(parts) != 3:
                continue
            provider = parts[1]
            models.setdefault(provider, []).append(model_id)
        return models

    def upload_attachment(self, *, provider: str) -> str:
        unique_token = f"ATTACHMENT_{provider.upper().replace('-', '_')}_{uuid4().hex[:10]}"
        files = {
            "file": (
                f"{provider}-attachment.txt",
                unique_token.encode("utf-8"),
                "text/plain",
            )
        }
        response = self._request_with_retry(
            provider=provider,
            send=lambda: self._client.post(
                "/v1/files",
                data={"purpose": "assistants"},
                files=files,
                headers={"Authorization": self._client.headers["Authorization"]},
            ),
        )
        response.raise_for_status()
        payload = response.json()
        file_id = str(payload.get("id", "")).strip()
        if not file_id:
            raise RuntimeError(f"provider {provider}: attachment upload returned no file id")
        return file_id

    def run_provider(self, *, provider: str, primary_model: str, think_model: str | None) -> None:
        context = ProviderRunContext(
            provider=provider,
            primary_model=primary_model,
            think_model=think_model,
            attachment_file_id=self.upload_attachment(provider=provider),
        )
        self._run_basic_chat_cases(context, total=40)
        self._run_system_chat_cases(context, total=20)
        self._run_chat_stream_cases(context, total=20)
        self._run_responses_cases(context, total=20)
        self._run_responses_stream_cases(context, total=20)
        self._run_tool_required_cases(context, total=20)
        self._run_tool_stream_cases(context, total=20)
        self._run_tool_followup_cases(context, total=20)
        self._run_attachment_cases(context, total=10)
        self._run_previous_response_cases(context, total=10)
        if think_model is not None:
            self._run_native_thinking_cases(context, total=20)
        else:
            self._run_tagged_thinking_contract_cases(context, total=20)

    def _record(
        self,
        *,
        provider: str,
        category: str,
        name: str,
        success: bool,
        status_code: int,
        started_at: float,
        detail: str = "",
        preview: str = "",
    ) -> None:
        self.results.append(
            LiveCaseResult(
                provider=provider,
                category=category,
                name=name,
                success=success,
                status_code=status_code,
                latency_ms=(time.perf_counter() - started_at) * 1000.0,
                detail=detail,
                preview=preview[:180],
            )
        )

    def _policy_for_provider(self, provider: str) -> ProviderExecutionPolicy:
        return PROVIDER_POLICIES.get(provider, DEFAULT_POLICY)

    def _respect_min_interval(self, provider: str) -> None:
        policy = self._policy_for_provider(provider)
        elapsed = time.perf_counter() - self._last_request_finished_at
        remaining = policy.min_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _mark_request_finished(self) -> None:
        self._last_request_finished_at = time.perf_counter()

    def _retry_delay(self, provider: str, attempt: int) -> float:
        policy = self._policy_for_provider(provider)
        return _compute_retry_delay(policy, attempt)

    def _request_with_retry(self, *, provider: str, send) -> httpx.Response:
        policy = self._policy_for_provider(provider)
        last_response: httpx.Response | None = None
        last_exc: Exception | None = None
        for attempt in range(1, policy.max_attempts + 1):
            self._respect_min_interval(provider)
            try:
                response = send()
            except Exception as exc:
                self._mark_request_finished()
                last_exc = exc
                if attempt >= policy.max_attempts or not _should_retry_exception(exc):
                    raise
                time.sleep(self._retry_delay(provider, attempt))
                continue
            self._mark_request_finished()
            payload = _safe_response_payload(response)
            should_retry, _ = _should_retry_http_result(response.status_code, payload)
            last_response = response
            if not should_retry or attempt >= policy.max_attempts:
                return response
            time.sleep(self._retry_delay(provider, attempt))
        if last_response is not None:
            return last_response
        assert last_exc is not None
        raise last_exc

    def _post_json(self, endpoint: str, payload: dict[str, object], *, provider: str) -> httpx.Response:
        return self._request_with_retry(
            provider=provider,
            send=lambda: self._client.post(endpoint, json=payload),
        )

    def _stream_lines(self, endpoint: str, payload: dict[str, object], *, provider: str) -> tuple[int, list[str]]:
        policy = self._policy_for_provider(provider)
        last_result: tuple[int, list[str]] | None = None
        last_exc: Exception | None = None
        for attempt in range(1, policy.max_attempts + 1):
            self._respect_min_interval(provider)
            try:
                with self._client.stream("POST", endpoint, json=payload) as response:
                    lines = [line for line in response.iter_lines() if line]
                    result = (response.status_code, lines)
            except Exception as exc:
                self._mark_request_finished()
                last_exc = exc
                if attempt >= policy.max_attempts or not _should_retry_exception(exc):
                    raise
                time.sleep(self._retry_delay(provider, attempt))
                continue
            self._mark_request_finished()
            status_code, lines = result
            should_retry, _ = _should_retry_stream_result(status_code, lines)
            last_result = result
            if not should_retry or attempt >= policy.max_attempts:
                return result
            time.sleep(self._retry_delay(provider, attempt))
        if last_result is not None:
            return last_result
        assert last_exc is not None
        raise last_exc

    def _run_basic_chat_cases(self, context: ProviderRunContext, *, total: int) -> None:
        prompts = [
            "请用一句话介绍 HTTP。",
            "什么是递归？请一句话回答。",
            "给我一个简短的鼓励句子。",
            "什么是缓存穿透？一句话说明。",
            "请用一句话解释单元测试。",
            "什么是 REST API？一句话回答。",
            "什么是向量数据库？一句话回答。",
            "什么是事务？一句话回答。",
            "什么是消息队列？一句话回答。",
            "什么是幂等性？一句话回答。",
        ]
        for index in range(total):
            prompt = f"{prompts[index % len(prompts)]} [basic-{index + 1}]"
            started = time.perf_counter()
            try:
                response = self._post_json(
                    "/v1/chat/completions",
                    {
                        "model": context.primary_model,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    provider=context.provider,
                )
                payload = response.json()
                text = _chat_text_from_payload(payload) if response.status_code == 200 else ""
                success = response.status_code == 200 and bool(text.strip())
                self._record(
                    provider=context.provider,
                    category="chat_basic",
                    name=f"basic-{index + 1}",
                    success=success,
                    status_code=response.status_code,
                    started_at=started,
                    detail="" if success else _error_preview(payload),
                    preview=text,
                )
            except Exception as exc:
                self._record(
                    provider=context.provider,
                    category="chat_basic",
                    name=f"basic-{index + 1}",
                    success=False,
                    status_code=0,
                    started_at=started,
                    detail=f"{type(exc).__name__}: {exc}",
                )

    def _run_system_chat_cases(self, context: ProviderRunContext, *, total: int) -> None:
        system_prompts = [
            "你是一个极简助手，只给一句话。",
            "你是一个中文助手，只能用中文。",
            "你是一个简洁老师，只能给一句定义。",
            "你是一个技术助手，只输出一句总结。",
        ]
        user_prompts = [
            "解释 HTTP。",
            "解释 TCP。",
            "解释数据库索引。",
            "解释限流。",
            "解释消息队列。",
        ]
        for index in range(total):
            started = time.perf_counter()
            try:
                response = self._post_json(
                    "/v1/chat/completions",
                    {
                        "model": context.primary_model,
                        "messages": [
                            {"role": "system", "content": system_prompts[index % len(system_prompts)]},
                            {"role": "user", "content": f"{user_prompts[index % len(user_prompts)]} [system-{index + 1}]"},
                        ],
                    },
                    provider=context.provider,
                )
                payload = response.json()
                text = _chat_text_from_payload(payload) if response.status_code == 200 else ""
                success = response.status_code == 200 and bool(text.strip())
                self._record(
                    provider=context.provider,
                    category="chat_system",
                    name=f"system-{index + 1}",
                    success=success,
                    status_code=response.status_code,
                    started_at=started,
                    detail="" if success else _error_preview(payload),
                    preview=text,
                )
            except Exception as exc:
                self._record(
                    provider=context.provider,
                    category="chat_system",
                    name=f"system-{index + 1}",
                    success=False,
                    status_code=0,
                    started_at=started,
                    detail=f"{type(exc).__name__}: {exc}",
                )

    def _run_chat_stream_cases(self, context: ProviderRunContext, *, total: int) -> None:
        prompts = [
            "请用至少二十个汉字解释 HTTP 是什么。",
            "请用至少二十个汉字解释什么是缓存。",
            "请用至少二十个汉字解释什么是函数。",
            "请用至少二十个汉字解释什么是数据库事务。",
        ]
        for index in range(total):
            started = time.perf_counter()
            try:
                status_code, lines = self._stream_lines(
                    "/v1/chat/completions",
                    {
                        "model": context.primary_model,
                        "stream": True,
                        "messages": [{"role": "user", "content": f"{prompts[index % len(prompts)]} [stream-{index + 1}]"}],
                    },
                    provider=context.provider,
                )
                text, delta_count, detail = _parse_chat_completion_stream(lines)
                success = status_code == 200 and bool(text.strip()) and delta_count >= 2 and detail == ""
                self._record(
                    provider=context.provider,
                    category="chat_stream",
                    name=f"chat-stream-{index + 1}",
                    success=success,
                    status_code=status_code,
                    started_at=started,
                    detail=detail or ("" if success else "stream validation failed"),
                    preview=text,
                )
            except Exception as exc:
                self._record(
                    provider=context.provider,
                    category="chat_stream",
                    name=f"chat-stream-{index + 1}",
                    success=False,
                    status_code=0,
                    started_at=started,
                    detail=f"{type(exc).__name__}: {exc}",
                )

    def _run_responses_cases(self, context: ProviderRunContext, *, total: int) -> None:
        prompts = [
            "请用一句话解释 HTTP。",
            "请用一句话解释单元测试。",
            "请用一句话解释数据库索引。",
            "请用一句话解释消息队列。",
        ]
        for index in range(total):
            started = time.perf_counter()
            try:
                response = self._post_json(
                    "/v1/responses",
                    {
                        "model": context.primary_model,
                        "input": f"{prompts[index % len(prompts)]} [responses-{index + 1}]",
                    },
                    provider=context.provider,
                )
                payload = response.json()
                text = _responses_text_from_payload(payload) if response.status_code == 200 else ""
                success = response.status_code == 200 and bool(text.strip())
                self._record(
                    provider=context.provider,
                    category="responses_basic",
                    name=f"responses-{index + 1}",
                    success=success,
                    status_code=response.status_code,
                    started_at=started,
                    detail="" if success else _error_preview(payload),
                    preview=text,
                )
            except Exception as exc:
                self._record(
                    provider=context.provider,
                    category="responses_basic",
                    name=f"responses-{index + 1}",
                    success=False,
                    status_code=0,
                    started_at=started,
                    detail=f"{type(exc).__name__}: {exc}",
                )

    def _run_responses_stream_cases(self, context: ProviderRunContext, *, total: int) -> None:
        prompts = [
            "请用至少二十个汉字解释 HTTP。",
            "请用至少二十个汉字解释什么是消息队列。",
            "请用至少二十个汉字解释什么是幂等性。",
            "请用至少二十个汉字解释什么是数据库事务。",
        ]
        for index in range(total):
            started = time.perf_counter()
            try:
                status_code, lines = self._stream_lines(
                    "/v1/responses",
                    {
                        "model": context.primary_model,
                        "stream": True,
                        "input": f"{prompts[index % len(prompts)]} [responses-stream-{index + 1}]",
                    },
                    provider=context.provider,
                )
                text, delta_count, detail = _parse_responses_stream(lines)
                success = status_code == 200 and bool(text.strip()) and delta_count >= 2 and detail == ""
                self._record(
                    provider=context.provider,
                    category="responses_stream",
                    name=f"responses-stream-{index + 1}",
                    success=success,
                    status_code=status_code,
                    started_at=started,
                    detail=detail or ("" if success else "responses stream validation failed"),
                    preview=text,
                )
            except Exception as exc:
                self._record(
                    provider=context.provider,
                    category="responses_stream",
                    name=f"responses-stream-{index + 1}",
                    success=False,
                    status_code=0,
                    started_at=started,
                    detail=f"{type(exc).__name__}: {exc}",
                )

    def _run_tool_required_cases(self, context: ProviderRunContext, *, total: int) -> None:
        tools = _weather_tools()
        for index in range(total):
            started = time.perf_counter()
            try:
                response = self._post_json(
                    "/v1/chat/completions",
                    {
                        "model": context.primary_model,
                        "messages": [
                            {
                                "role": "user",
                                "content": f"请必须调用 get_weather 查询 Tokyo 的天气，然后不要直接回答。[tool-required-{index + 1}]",
                            }
                        ],
                        "tools": tools,
                        "tool_choice": "required",
                    },
                    provider=context.provider,
                )
                payload = response.json()
                content = _chat_text_from_payload(payload) if response.status_code == 200 else ""
                tool_calls = _chat_tool_calls_from_payload(payload) if response.status_code == 200 else []
                content_ok = content in {"", None}
                success = (
                    response.status_code == 200
                    and _chat_finish_reason(payload) == "tool_calls"
                    and len(tool_calls) >= 1
                    and content_ok
                )
                self._record(
                    provider=context.provider,
                    category="tool_required",
                    name=f"tool-required-{index + 1}",
                    success=success,
                    status_code=response.status_code,
                    started_at=started,
                    detail="" if success else _error_preview(payload) or f"content={content!r}",
                    preview=json.dumps(tool_calls[:1], ensure_ascii=False),
                )
            except Exception as exc:
                self._record(
                    provider=context.provider,
                    category="tool_required",
                    name=f"tool-required-{index + 1}",
                    success=False,
                    status_code=0,
                    started_at=started,
                    detail=f"{type(exc).__name__}: {exc}",
                )

    def _run_tool_stream_cases(self, context: ProviderRunContext, *, total: int) -> None:
        tools = _weather_tools()
        for index in range(total):
            started = time.perf_counter()
            try:
                status_code, lines = self._stream_lines(
                    "/v1/chat/completions",
                    {
                        "model": context.primary_model,
                        "stream": True,
                        "messages": [
                            {
                                "role": "user",
                                "content": f"请必须调用 get_weather 查询 Tokyo 的天气，然后不要直接回答。[tool-stream-{index + 1}]",
                            }
                        ],
                        "tools": tools,
                        "tool_choice": "required",
                    },
                    provider=context.provider,
                )
                content, tool_calls, detail = _parse_chat_completion_tool_stream(lines)
                success = (
                    status_code == 200
                    and detail == ""
                    and content == ""
                    and len(tool_calls) >= 1
                    and bool(str(tool_calls[0].get("function", {}).get("name", "")).strip())
                    and bool(str(tool_calls[0].get("function", {}).get("arguments", "")).strip())
                )
                self._record(
                    provider=context.provider,
                    category="tool_stream",
                    name=f"tool-stream-{index + 1}",
                    success=success,
                    status_code=status_code,
                    started_at=started,
                    detail=detail or ("" if success else "tool stream validation failed"),
                    preview=json.dumps(tool_calls[:1], ensure_ascii=False),
                )
            except Exception as exc:
                self._record(
                    provider=context.provider,
                    category="tool_stream",
                    name=f"tool-stream-{index + 1}",
                    success=False,
                    status_code=0,
                    started_at=started,
                    detail=f"{type(exc).__name__}: {exc}",
                )

    def _run_tool_followup_cases(self, context: ProviderRunContext, *, total: int) -> None:
        tools = _weather_tools()
        for index in range(total):
            started = time.perf_counter()
            call_id = f"call_weather_{index + 1}"
            try:
                response = self._post_json(
                    "/v1/chat/completions",
                    {
                        "model": context.primary_model,
                        "messages": [
                            {"role": "user", "content": f"Tokyo 的天气怎么样？ [tool-followup-{index + 1}]"},
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": call_id,
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"location":"Tokyo"}',
                                        },
                                    }
                                ],
                            },
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": '{"temperature":22,"unit":"C","city":"Tokyo"}',
                            },
                        ],
                        "tools": tools,
                        "tool_choice": "auto",
                    },
                    provider=context.provider,
                )
                payload = response.json()
                text = _chat_text_from_payload(payload) if response.status_code == 200 else ""
                success = (
                    response.status_code == 200
                    and bool(text.strip())
                    and "<tool_call" not in text
                    and "<tool_calls" not in text
                )
                self._record(
                    provider=context.provider,
                    category="tool_followup",
                    name=f"tool-followup-{index + 1}",
                    success=success,
                    status_code=response.status_code,
                    started_at=started,
                    detail="" if success else _error_preview(payload),
                    preview=text,
                )
            except Exception as exc:
                self._record(
                    provider=context.provider,
                    category="tool_followup",
                    name=f"tool-followup-{index + 1}",
                    success=False,
                    status_code=0,
                    started_at=started,
                    detail=f"{type(exc).__name__}: {exc}",
                )

    def _run_attachment_cases(self, context: ProviderRunContext, *, total: int) -> None:
        for index in range(total):
            started = time.perf_counter()
            attachment_token = f"ATTACHMENT_{context.provider.upper().replace('-', '_')}"
            try:
                response = self._post_json(
                    "/v1/responses",
                    {
                        "model": context.primary_model,
                        "input": [
                            {
                                "type": "input_text",
                                "text": f"请只复述附件里的唯一 token。[attachment-{index + 1}]",
                            },
                            {
                                "type": "input_file",
                                "file_id": context.attachment_file_id,
                            },
                        ],
                    },
                    provider=context.provider,
                )
                payload = response.json()
                text = _responses_text_from_payload(payload) if response.status_code == 200 else ""
                success = response.status_code == 200 and attachment_token in text
                self._record(
                    provider=context.provider,
                    category="attachment",
                    name=f"attachment-{index + 1}",
                    success=success,
                    status_code=response.status_code,
                    started_at=started,
                    detail="" if success else _error_preview(payload),
                    preview=text,
                )
            except Exception as exc:
                self._record(
                    provider=context.provider,
                    category="attachment",
                    name=f"attachment-{index + 1}",
                    success=False,
                    status_code=0,
                    started_at=started,
                    detail=f"{type(exc).__name__}: {exc}",
                )

    def _run_previous_response_cases(self, context: ProviderRunContext, *, total: int) -> None:
        for index in range(total):
            started = time.perf_counter()
            token = f"MEM-{context.provider[:4].upper()}-{index + 1:02d}"
            try:
                first = self._post_json(
                    "/v1/responses",
                    {
                        "model": context.primary_model,
                        "input": f"记住这个代码：{token}。只回复 OK。",
                    },
                    provider=context.provider,
                )
                first_payload = first.json()
                response_id = str(first_payload.get("id", "")).strip()
                second = self._post_json(
                    "/v1/responses",
                    {
                        "model": context.primary_model,
                        "previous_response_id": response_id,
                        "input": "请输出刚才让你记住的代码，只输出代码本身。",
                    },
                    provider=context.provider,
                )
                payload = second.json()
                text = _responses_text_from_payload(payload) if second.status_code == 200 else ""
                success = first.status_code == 200 and second.status_code == 200 and token in text
                self._record(
                    provider=context.provider,
                    category="previous_response",
                    name=f"previous-response-{index + 1}",
                    success=success,
                    status_code=second.status_code,
                    started_at=started,
                    detail="" if success else _error_preview(payload) or _error_preview(first_payload),
                    preview=text,
                )
            except Exception as exc:
                self._record(
                    provider=context.provider,
                    category="previous_response",
                    name=f"previous-response-{index + 1}",
                    success=False,
                    status_code=0,
                    started_at=started,
                    detail=f"{type(exc).__name__}: {exc}",
                )

    def _run_native_thinking_cases(self, context: ProviderRunContext, *, total: int) -> None:
        assert context.think_model is not None
        half = total // 2
        for index in range(half):
            started = time.perf_counter()
            try:
                response = self._post_json(
                    "/v1/chat/completions",
                    {
                        "model": context.think_model,
                        "messages": [
                            {
                                "role": "user",
                                "content": f"请先思考再回答：13*17 等于多少？最后给一句结论。[native-think-{index + 1}]",
                            }
                        ],
                    },
                    provider=context.provider,
                )
                payload = response.json()
                text = _chat_text_from_payload(payload) if response.status_code == 200 else ""
                success = response.status_code == 200 and bool(text.strip()) and not _contains_well_formed_think(text)
                self._record(
                    provider=context.provider,
                    category="native_thinking",
                    name=f"native-think-{index + 1}",
                    success=success,
                    status_code=response.status_code,
                    started_at=started,
                    detail="" if success else _error_preview(payload),
                    preview=text,
                )
            except Exception as exc:
                self._record(
                    provider=context.provider,
                    category="native_thinking",
                    name=f"native-think-{index + 1}",
                    success=False,
                    status_code=0,
                    started_at=started,
                    detail=f"{type(exc).__name__}: {exc}",
                )

        for index in range(half, total):
            started = time.perf_counter()
            try:
                status_code, lines = self._stream_lines(
                    "/v1/chat/completions",
                    {
                        "model": context.think_model,
                        "stream": True,
                        "messages": [
                            {
                                "role": "user",
                                "content": f"请先思考再回答：19*23 等于多少？最后给一句结论。[native-think-stream-{index + 1}]",
                            }
                        ],
                    },
                    provider=context.provider,
                )
                text, delta_count, detail = _parse_chat_completion_stream(lines)
                success = (
                    status_code == 200
                    and delta_count >= 2
                    and bool(text.strip())
                    and not _contains_well_formed_think(text)
                    and detail == ""
                )
                self._record(
                    provider=context.provider,
                    category="native_thinking_stream",
                    name=f"native-think-stream-{index + 1}",
                    success=success,
                    status_code=status_code,
                    started_at=started,
                    detail=detail or ("" if success else "thinking stream validation failed"),
                    preview=text,
                )
            except Exception as exc:
                self._record(
                    provider=context.provider,
                    category="native_thinking_stream",
                    name=f"native-think-stream-{index + 1}",
                    success=False,
                    status_code=0,
                    started_at=started,
                    detail=f"{type(exc).__name__}: {exc}",
                )

    def _run_tagged_thinking_contract_cases(self, context: ProviderRunContext, *, total: int) -> None:
        tools = _weather_tools()
        half = total // 2
        for index in range(half):
            started = time.perf_counter()
            try:
                response = self._post_json(
                    "/v1/responses",
                    {
                        "model": context.primary_model,
                        "input": f"请先思考，再必须调用 get_weather 查询 Tokyo 的天气。[tagged-think-{index + 1}]",
                        "tools": tools,
                        "tool_choice": "required",
                    },
                    provider=context.provider,
                )
                payload = response.json()
                text = _responses_text_from_payload(payload) if response.status_code == 200 else ""
                success = (
                    response.status_code == 200
                    and _responses_has_function_call(payload)
                    and (text == "" or THINK_ONLY_RE.fullmatch(text) is not None)
                )
                self._record(
                    provider=context.provider,
                    category="tagged_thinking_contract",
                    name=f"tagged-think-{index + 1}",
                    success=success,
                    status_code=response.status_code,
                    started_at=started,
                    detail="" if success else _error_preview(payload),
                    preview=text,
                )
            except Exception as exc:
                self._record(
                    provider=context.provider,
                    category="tagged_thinking_contract",
                    name=f"tagged-think-{index + 1}",
                    success=False,
                    status_code=0,
                    started_at=started,
                    detail=f"{type(exc).__name__}: {exc}",
                )

        for index in range(half, total):
            started = time.perf_counter()
            try:
                status_code, lines = self._stream_lines(
                    "/v1/responses",
                    {
                        "model": context.primary_model,
                        "stream": True,
                        "input": f"请先思考，再必须调用 get_weather 查询 Tokyo 的天气。[tagged-think-stream-{index + 1}]",
                        "tools": tools,
                        "tool_choice": "required",
                    },
                    provider=context.provider,
                )
                text, delta_count, detail = _parse_responses_stream(lines)
                success = (
                    status_code == 200
                    and detail == ""
                    and delta_count >= 0
                    and (text == "" or THINK_ONLY_RE.fullmatch(text) is not None)
                )
                self._record(
                    provider=context.provider,
                    category="tagged_thinking_stream_contract",
                    name=f"tagged-think-stream-{index + 1}",
                    success=success,
                    status_code=status_code,
                    started_at=started,
                    detail=detail or ("" if success else "tagged thinking stream validation failed"),
                    preview=text,
                )
            except Exception as exc:
                self._record(
                    provider=context.provider,
                    category="tagged_thinking_stream_contract",
                    name=f"tagged-think-stream-{index + 1}",
                    success=False,
                    status_code=0,
                    started_at=started,
                    detail=f"{type(exc).__name__}: {exc}",
                )


def _safe_response_payload(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


def _should_retry_exception(exc: Exception) -> bool:
    if isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError)):
        return True
    return _looks_like_retryable_message(str(exc))


def _should_retry_http_result(status_code: int, payload: Any) -> tuple[bool, str]:
    detail = _extract_error_detail(payload)
    error_type = _extract_error_type(payload)
    if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True, detail
    if error_type in {"rate_limit_error", "api_error"}:
        return True, detail
    if _looks_like_retryable_message(detail):
        return True, detail
    return False, detail


def _should_retry_stream_result(status_code: int, lines: list[str]) -> tuple[bool, str]:
    if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True, f"http_status={status_code}"
    if not lines:
        return True, "empty stream"
    event_name = ""
    for line in lines:
        if line.startswith("event: "):
            event_name = line[7:].strip()
            continue
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        detail = _extract_error_detail(parsed)
        error_type = _extract_error_type(parsed)
        if error_type in {"rate_limit_error", "api_error"}:
            return True, detail
        if event_name == "error" and _looks_like_retryable_message(detail):
            return True, detail
        if _looks_like_retryable_message(detail):
            return True, detail
    if lines[-1] != "data: [DONE]":
        return True, f"missing DONE marker: {lines[-1]}"
    return False, ""


def _extract_error_type(payload: Any) -> str:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("type", "")).strip().lower()
        return str(payload.get("type", "")).strip().lower()
    return ""


def _extract_error_detail(payload: Any) -> str:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = str(error.get("message", "")).strip()
            if message:
                return message
        for key in ("message", "msg", "content", "detail"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(payload, str):
        return payload.strip()
    return _error_preview(payload)


def _compute_retry_delay(policy: ProviderExecutionPolicy, attempt: int) -> float:
    return min(policy.retry_base_delay_seconds * (2 ** max(attempt - 1, 0)), 20.0)


def _looks_like_retryable_message(message: str) -> bool:
    normalized = message.strip()
    if not normalized:
        return False
    return bool(RATE_LIMIT_TEXT_RE.search(normalized) or TRANSIENT_TEXT_RE.search(normalized))


def _weather_tools() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                    "required": ["location"],
                },
            },
        }
    ]


def _chat_text_from_payload(payload: dict[str, Any]) -> str:
    try:
        return str(payload["choices"][0]["message"]["content"] or "")
    except Exception:
        return ""


def _chat_tool_calls_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        tool_calls = payload["choices"][0]["message"].get("tool_calls", [])
    except Exception:
        return []
    return tool_calls if isinstance(tool_calls, list) else []


def _chat_finish_reason(payload: dict[str, Any]) -> str:
    try:
        return str(payload["choices"][0]["finish_reason"] or "")
    except Exception:
        return ""


def _responses_text_from_payload(payload: dict[str, Any]) -> str:
    output = payload.get("output", [])
    if not isinstance(output, list):
        return ""
    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "reasoning":
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            reasoning_parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "reasoning_text" and isinstance(block.get("text"), str):
                    reasoning_parts.append(block["text"])
            if reasoning_parts:
                parts.append(f"<think>{''.join(reasoning_parts)}</think>")
            continue
        if item_type != "message":
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
    return "".join(parts)


def _responses_has_function_call(payload: dict[str, Any]) -> bool:
    output = payload.get("output", [])
    if not isinstance(output, list):
        return False
    return any(isinstance(item, dict) and item.get("type") == "function_call" for item in output)


def _parse_chat_completion_stream(lines: list[str]) -> tuple[str, int, str]:
    if not lines:
        return "", 0, "empty stream"
    if lines[-1] != "data: [DONE]":
        return "", 0, f"missing DONE marker: {lines[-1]}"
    text_parts: list[str] = []
    delta_count = 0
    accumulated = ""
    for line in lines[:-1]:
        if not line.startswith("data: "):
            continue
        payload = json.loads(line[6:])
        choices = payload.get("choices", [])
        if not isinstance(choices, list) or not choices:
            continue
        delta = choices[0].get("delta", {})
        if not isinstance(delta, dict):
            continue
        content = delta.get("content")
        if isinstance(content, str) and content:
            if _looks_like_snapshot_delta(accumulated, content):
                return "", delta_count, f"snapshot-like duplicated content delta: {content[:120]!r}"
            text_parts.append(content)
            accumulated += content
            delta_count += 1
    return "".join(text_parts), delta_count, ""


def _parse_chat_completion_tool_stream(lines: list[str]) -> tuple[str, list[dict[str, Any]], str]:
    if not lines:
        return "", [], "empty stream"
    if lines[-1] != "data: [DONE]":
        return "", [], f"missing DONE marker: {lines[-1]}"
    text_parts: list[str] = []
    accumulated = ""
    streamed_tool_calls: list[dict[str, Any]] = []
    for line in lines[:-1]:
        if not line.startswith("data: "):
            continue
        payload = json.loads(line[6:])
        choices = payload.get("choices", [])
        if not isinstance(choices, list) or not choices:
            continue
        delta = choices[0].get("delta", {})
        if not isinstance(delta, dict):
            continue
        content = delta.get("content")
        if isinstance(content, str) and content:
            if _looks_like_snapshot_delta(accumulated, content):
                return "", [], f"snapshot-like duplicated content delta: {content[:120]!r}"
            text_parts.append(content)
            accumulated += content
        raw_tool_calls = delta.get("tool_calls")
        if not isinstance(raw_tool_calls, list):
            continue
        for item in raw_tool_calls:
            if not isinstance(item, dict):
                continue
            _merge_streamed_tool_call_delta(streamed_tool_calls, item)
    return "".join(text_parts), streamed_tool_calls, ""


def _parse_responses_stream(lines: list[str]) -> tuple[str, int, str]:
    if not lines:
        return "", 0, "empty stream"
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    delta_count = 0
    event_name = ""
    for line in lines:
        if line.startswith("event: "):
            event_name = line[7:].strip()
            continue
        if not line.startswith("data: "):
            continue
        data = json.loads(line[6:])
        if event_name == "error":
            return "", 0, _error_preview(data)
        if event_name == "response.output_text.delta":
            delta = data.get("delta")
            if isinstance(delta, str):
                text_parts.append(delta)
                delta_count += 1
        elif event_name == "response.reasoning_text.delta":
            delta = data.get("delta")
            if isinstance(delta, str):
                reasoning_parts.append(delta)
                delta_count += 1
    rendered_reasoning = f"<think>{''.join(reasoning_parts)}</think>" if reasoning_parts else ""
    return f"{rendered_reasoning}{''.join(text_parts)}", delta_count, ""


def _contains_well_formed_think(text: str) -> bool:
    return bool(text and THINK_BLOCK_RE.search(text))


def _looks_like_snapshot_delta(accumulated: str, delta: str) -> bool:
    if not accumulated or not delta:
        return False
    if delta.startswith(accumulated) or accumulated.startswith(delta):
        return True
    normalized_accumulated = _normalize_stream_delta_text(accumulated)
    normalized_delta = _normalize_stream_delta_text(delta)
    if len(normalized_accumulated) < 24 or len(normalized_delta) < 24:
        return False
    return normalized_delta.startswith(normalized_accumulated) or normalized_accumulated.startswith(normalized_delta)


def _normalize_stream_delta_text(text: str) -> str:
    collapsed = re.sub(r"\s+", "", text)
    collapsed = re.sub(r"[*_`~]+", "", collapsed)
    return collapsed


def _merge_streamed_tool_call_delta(target: list[dict[str, Any]], item: dict[str, Any]) -> None:
    index = int(item.get("index", len(target)))
    while len(target) <= index:
        target.append({"function": {}})
    merged = target[index]
    item_id = item.get("id")
    if isinstance(item_id, str) and item_id:
        merged["id"] = item_id
    item_type = item.get("type")
    if isinstance(item_type, str) and item_type:
        merged["type"] = item_type
    function = item.get("function")
    if not isinstance(function, dict):
        return
    merged_function = merged.setdefault("function", {})
    name = function.get("name")
    if isinstance(name, str) and name:
        merged_function["name"] = name
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        merged_function["arguments"] = str(merged_function.get("arguments", "")) + arguments


def _error_preview(payload: Any) -> str:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message", ""))[:180]
        return json.dumps(payload, ensure_ascii=False)[:180]
    return str(payload)[:180]


def _discover_requested_providers(raw: list[str] | None) -> list[str]:
    if raw:
        return raw
    providers = [
        record.provider
        for record in list_provider_credentials(resolve_providers_dir())
        if record.status == "valid"
    ]
    return providers


def _pick_primary_model(models: list[str]) -> str:
    candidates = [model for model in models if "/text-embedding-" not in model]
    if not candidates:
        return models[0]
    return min(candidates, key=_primary_model_sort_key)


def _pick_think_model(models: list[str]) -> str | None:
    for model in models:
        tail = model.rsplit("/", 1)[-1].lower()
        if "think" in tail or "reasoner" in tail:
            return model
    return None


def _primary_model_sort_key(model: str) -> tuple[int, str]:
    tail = model.rsplit("/", 1)[-1].lower()
    score = 100
    if "think" in tail or "reasoner" in tail:
        score += 1000
    if any(token in tail for token in ("max", "ultra", "opus", "128k", "preview")):
        score += 60
    if any(token in tail for token in ("omni", "vl", "coder")):
        score += 40
    if any(token in tail for token in ("flash", "lite", "haiku", "8k")):
        score -= 50
    return score, tail


def render_markdown(results: list[LiveCaseResult]) -> str:
    total = len(results)
    passed = sum(1 for result in results if result.success)
    failed = total - passed
    by_provider = Counter(result.provider for result in results)
    provider_passed = Counter(result.provider for result in results if result.success)
    lines = [
        "# Live Provider 200+ E2E Report",
        "",
        f"- Total cases: **{total}**",
        f"- Passed: **{passed}**",
        f"- Failed: **{failed}**",
        f"- Success rate: **{(passed / total * 100.0) if total else 0.0:.2f}%**",
        "",
        "## Provider Summary",
        "",
        "| Provider | Total | Passed | Failed |",
        "|---|---:|---:|---:|",
    ]
    for provider in sorted(by_provider):
        provider_total = by_provider[provider]
        provider_ok = provider_passed[provider]
        lines.append(f"| {provider} | {provider_total} | {provider_ok} | {provider_total - provider_ok} |")

    lines.extend(["", "## Failed Cases", ""])
    failures = [result for result in results if not result.success]
    if not failures:
        lines.append("None")
    else:
        for result in failures:
            lines.append(
                f"- **{result.provider} / {result.category} / {result.name}** — "
                f"HTTP {result.status_code} — {result.detail or result.preview}"
            )
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 200+ live E2E cases per provider.")
    parser.add_argument("--provider", action="append", help="Provider key. Repeatable.")
    parser.add_argument(
        "--out",
        default="live_provider_200_report.md",
        help="Markdown report output path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    config = load_or_create_app_config(resolve_app_config_path())
    base_url = f"http://{config['host']}:{config['port']}"
    providers = _discover_requested_providers(args.provider)
    if not providers:
        print("No valid providers available.", file=sys.stderr)
        return 1

    runner = LiveProviderRunner(base_url=base_url, api_key=str(config["api_key"]))
    try:
        runner.healthcheck()
        provider_models = runner.discover_provider_models()
        for provider in providers:
            models = provider_models.get(provider, [])
            if not models:
                runner.results.append(
                    LiveCaseResult(
                        provider=provider,
                        category="bootstrap",
                        name="missing-models",
                        success=False,
                        status_code=0,
                        latency_ms=0.0,
                        detail="no models discovered for provider",
                    )
                )
                continue
            primary_model = _pick_primary_model(models)
            think_model = _pick_think_model(models)
            runner.run_provider(provider=provider, primary_model=primary_model, think_model=think_model)
    finally:
        runner.close()

    report = render_markdown(runner.results)
    out_path = Path(args.out)
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport written to {out_path}")
    return 0 if all(result.success for result in runner.results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
