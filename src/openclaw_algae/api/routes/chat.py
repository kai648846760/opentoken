from __future__ import annotations

import json
from time import time
from uuid import uuid4

import httpx
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from openclaw_algae.api.errors import openai_error_response
from openclaw_algae.api.streaming import (
    ProtocolMarkupProjector,
    chunk_visible_text,
    sse_response_headers,
    strip_tool_protocol_markup,
)
from openclaw_algae.gateway.normalized import normalize_chat_completions_request
from openclaw_algae.gateway.router import get_default_router
from openclaw_algae.providers.base import ProviderRateLimitError

router = APIRouter()


@router.post("/v1/chat/completions")
def chat_completions(payload: dict[str, object]) -> dict[str, object]:
    try:
        request = normalize_chat_completions_request(payload)
    except ProviderRateLimitError as exc:
        return openai_error_response(
            status_code=429,
            message=str(exc),
            error_type="rate_limit_error",
        )
    except RuntimeError as exc:
        return openai_error_response(
            status_code=400,
            message=str(exc),
            error_type="invalid_request_error",
        )
    except ValidationError as exc:
        return openai_error_response(
            status_code=400,
            message=_format_validation_error(exc),
            error_type="invalid_request_error",
        )
    except httpx.HTTPError as exc:
        return openai_error_response(
            status_code=502,
            message=str(exc),
            error_type="api_error",
        )
    if request.stream:
        completion_id = f"chatcmpl-{uuid4().hex}"
        return StreamingResponse(
            _stream_chat_completion(
                request=request,
                completion_id=completion_id,
                created=int(time()),
            ),
            headers=sse_response_headers(),
            media_type="text/event-stream",
        )
    try:
        response = get_default_router().chat(request)
    except ProviderRateLimitError as exc:
        return openai_error_response(
            status_code=429,
            message=str(exc),
            error_type="rate_limit_error",
        )
    except RuntimeError as exc:
        return openai_error_response(
            status_code=400,
            message=str(exc),
            error_type="invalid_request_error",
        )
    except httpx.HTTPError as exc:
        return openai_error_response(
            status_code=502,
            message=str(exc),
            error_type="api_error",
        )
    created = int(time())
    message = _chat_message_payload(
        strip_tool_protocol_markup(response.content, include_think=False),
        response.tool_calls,
    )
    return {
        "id": f"chatcmpl-{uuid4().hex}",
        "object": "chat.completion",
        "created": created,
        "model": response.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": response.finish_reason,
            }
        ],
        "usage": _empty_chat_usage(),
    }


def _stream_chat_completion(
    *,
    request,
    completion_id: str,
    created: int,
):
    role_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(role_chunk, separators=(',', ':'))}\n\n"

    router = get_default_router()
    stream_method = getattr(router, "stream_chat", None)
    if callable(stream_method):
        try:
            content_stream = stream_method(request)
        except Exception as exc:
            yield f"data: {json.dumps(_stream_error_payload(exc), separators=(',', ':'))}\n\n"
            yield "data: [DONE]\n\n"
            return
        if content_stream is not None:
            include_think = _should_include_think_in_chat_stream(request)
            projector = ProtocolMarkupProjector(include_think=include_think)
            try:
                for raw_piece in content_stream:
                    if not raw_piece:
                        continue
                    piece = projector.push(str(raw_piece))
                    if not piece:
                        continue
                    for visible_piece in _chunk_stream_text(piece, include_think=include_think):
                        if not visible_piece:
                            continue
                        content_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": visible_piece},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(content_chunk, separators=(',', ':'))}\n\n"
                final_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(final_chunk, separators=(',', ':'))}\n\n"
                yield "data: [DONE]\n\n"
                return
            except Exception as exc:
                yield f"data: {json.dumps(_stream_error_payload(exc), separators=(',', ':'))}\n\n"
                yield "data: [DONE]\n\n"
                return

    try:
        response = router.chat(request)
    except Exception as exc:
        yield f"data: {json.dumps(_stream_error_payload(exc), separators=(',', ':'))}\n\n"
        yield "data: [DONE]\n\n"
        return

    if response.tool_calls:
        rendered_content = strip_tool_protocol_markup(response.content, include_think=False) or ""
        for piece in _chunk_stream_text(rendered_content):
            content_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": response.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": piece},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(content_chunk, separators=(',', ':'))}\n\n"
        for tool_delta in _iter_stream_tool_call_deltas(response.tool_calls):
            tool_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": response.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": tool_delta},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(tool_chunk, separators=(',', ':'))}\n\n"
    else:
        rendered_content = strip_tool_protocol_markup(response.content, include_think=False) or ""
        for piece in _chunk_stream_text(rendered_content):
            content_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": response.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": piece},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(content_chunk, separators=(',', ':'))}\n\n"

    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": response.model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": response.finish_reason,
            }
        ],
    }
    yield f"data: {json.dumps(final_chunk, separators=(',', ':'))}\n\n"
    yield "data: [DONE]\n\n"


def _chat_message_payload(content: str | None, tool_calls: list[dict[str, object]]) -> dict[str, object]:
    message: dict[str, object] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
        if content in {"", None}:
            message["content"] = None
    return message


def _format_validation_error(exc: ValidationError) -> str:
    first_error = exc.errors(include_url=False)[0]
    location = ".".join(str(part) for part in first_error.get("loc", ()))
    message = str(first_error.get("msg", "Invalid request."))
    if not location:
        return message
    return f"{location}: {message}"


def _empty_chat_usage() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def _chunk_stream_text(
    content: str,
    *,
    max_chunk_len: int = 12,
    include_think: bool = False,
) -> list[str]:
    return chunk_visible_text(content, max_chunk_len=max_chunk_len, include_think=include_think)


def _should_include_think_in_chat_stream(request) -> bool:
    if getattr(request, "tools", None):
        return False
    model = str(getattr(request, "model", "") or "").lower()
    return any(token in model for token in ("reasoner", "thinking", "-think"))


def _iter_stream_tool_call_deltas(
    tool_calls: list[dict[str, object]],
    *,
    max_arguments_chunk_len: int = 128,
):
    for index, tool_call in enumerate(tool_calls):
        function = tool_call.get("function", {})
        if not isinstance(function, dict):
            function = {}
        arguments = function.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        metadata_delta: dict[str, object] = {
            "index": index,
            "type": str(tool_call.get("type", "")).strip() or "function",
            "function": {
                "name": str(function.get("name", "")).strip(),
                "arguments": "",
            },
        }
        call_id = str(tool_call.get("id", "")).strip()
        if call_id:
            metadata_delta["id"] = call_id
        yield [metadata_delta]
        if not arguments:
            continue
        for start in range(0, len(arguments), max_arguments_chunk_len):
            yield [
                {
                    "index": index,
                    "function": {
                        "arguments": arguments[start : start + max_arguments_chunk_len],
                    },
                }
            ]


def _stream_error_payload(exc: Exception) -> dict[str, object]:
    if isinstance(exc, ProviderRateLimitError):
        error_type = "rate_limit_error"
    elif isinstance(exc, httpx.HTTPError):
        error_type = "api_error"
    else:
        error_type = "invalid_request_error"
    return {
        "error": {
            "message": str(exc),
            "type": error_type,
        }
    }
