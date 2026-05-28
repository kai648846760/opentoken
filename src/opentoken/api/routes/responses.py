import json
import re
from time import time
from uuid import uuid4

import httpx
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from opentoken.api.errors import classify_provider_runtime_error, openai_error_response
from opentoken.api.streaming import (
    ProtocolMarkupProjector,
    chunk_visible_text,
    sse_response_headers,
    strip_tool_protocol_markup,
)
from opentoken.api.usage import estimate_prompt_tokens, estimate_tokens
from opentoken.config.paths import resolve_state_dir
from opentoken.gateway.normalized import normalize_responses_request
from opentoken.gateway.router import get_default_router
from opentoken.providers.base import ChatResponse
from opentoken.providers.base import ProviderRateLimitError
from opentoken.providers.web_tool_calling import parse_web_tool_response
from opentoken.storage.response_store import (
    load_response_messages,
    save_response_messages,
)

router = APIRouter()
_THINK_BLOCK_PATTERN = re.compile(r"<think\b[^>]*>(?P<body>[\s\S]*?)</think\s*>", re.IGNORECASE)
_THINK_TAG_PATTERN = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)


@router.post("/v1/responses")
def responses(payload: dict[str, object]) -> dict[str, object]:
    try:
        request = _resolve_request_with_previous_response(payload)
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
        response_id = f"resp-{uuid4().hex}"
        created_at = int(time())
        return StreamingResponse(
            _stream_response_events(
                request=request,
                response_id=response_id,
                created_at=created_at,
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
        # Router failures get the shared classifier (missing creds -> 401,
        # unsupported model -> 400, anything else -> 502). The non-stream
        # /v1/responses path used to force-map all of these to 400.
        status_code, error_type = classify_provider_runtime_error(exc)
        return openai_error_response(
            status_code=status_code,
            message=str(exc),
            error_type=error_type,
        )
    except httpx.HTTPError as exc:
        return openai_error_response(
            status_code=502,
            message=str(exc),
            error_type="api_error",
        )
    response_id = f"resp-{uuid4().hex}"
    output = _build_response_output(response)
    _save_response_history(response_id=response_id, request=request, response=response)
    rendered_text = strip_tool_protocol_markup(response.content) or ""
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time()),
        "status": "incomplete" if response.tool_calls else "completed",
        "model": response.model,
        "output": output,
        "usage": _estimate_response_usage(request, rendered_text),
    }


def _stream_response_events(
    *,
    request,
    response_id: str,
    created_at: int,
):
    usage = _empty_response_usage()
    initial_response = _response_resource(
        response_id=response_id,
        created_at=created_at,
        status="in_progress",
        model=request.model,
        output=[],
        usage=usage,
    )

    yield from _sse_event("response.created", {"type": "response.created", "response": initial_response})
    yield from _sse_event(
        "response.in_progress",
        {"type": "response.in_progress", "response": initial_response},
    )

    router = get_default_router()
    stream_method = getattr(router, "stream_chat", None)
    if callable(stream_method):
        try:
            content_stream = stream_method(request)
        except Exception as exc:
            yield from _sse_event("error", _stream_error_payload(exc))
            return
        if content_stream is not None:
            completed_output: list[dict[str, object]] = []
            output_index = 0
            projector = ProtocolMarkupProjector()
            rendered_content = ""
            in_reasoning = False
            current_kind: str | None = None
            current_text = ""
            current_item_id: str | None = None
            try:
                for raw_piece in content_stream:
                    if not raw_piece:
                        continue
                    piece = projector.push(str(raw_piece))
                    if not piece:
                        continue
                    rendered_content += piece
                    segments, in_reasoning = _split_stream_piece_segments(piece, in_reasoning=in_reasoning)
                    for kind, segment_text in segments:
                        if not segment_text:
                            continue
                        if current_kind != kind:
                            if current_kind is not None and current_item_id is not None:
                                completed_item = yield from _yield_segment_done_events(
                                    current_kind,
                                    current_text,
                                    item_id=current_item_id,
                                    output_index=output_index,
                                )
                                completed_output.append(completed_item)
                                output_index += 1
                            current_kind = kind
                            current_text = ""
                            current_item_id = _build_stream_item_id(kind)
                            yield from _yield_segment_open_events(
                                current_kind,
                                item_id=current_item_id,
                                output_index=output_index,
                            )
                        current_text += segment_text
                        if current_item_id is None:
                            current_item_id = _build_stream_item_id(kind)
                            yield from _yield_segment_open_events(
                                kind,
                                item_id=current_item_id,
                                output_index=output_index,
                            )
                        yield from _yield_segment_delta_events(
                            kind,
                            segment_text,
                            item_id=current_item_id,
                            output_index=output_index,
                        )
            except Exception as exc:
                yield from _sse_event("error", _stream_error_payload(exc))
                return
            if current_kind is None:
                completed_item = yield from _yield_segment_events(
                    "message",
                    "",
                    output_index=output_index,
                )
                completed_output.append(completed_item)
                output_index += 1
            else:
                completed_item = yield from _yield_segment_done_events(
                    current_kind,
                    current_text,
                    item_id=str(current_item_id),
                    output_index=output_index,
                )
                completed_output.append(completed_item)
                output_index += 1
            # 流末尾从 projector.raw_text 重组 <tool_calls> —— projector 已经
            # 把这些标签从可见输出里隐藏掉,如果不在这里解析它们就完全丢失,客户
            # 端看到的是"模型空回答"。镜像非流式 fallback 路径的 function_call
            # event 协议（response.output_item.added → arguments.delta+done →
            # output_item.done）发射,并把 tool_calls 传给 _save_response_history
            # 让续聊历史保留这次工具调用。
            stream_tool_calls: list[dict[str, object]] = []
            try:
                _parsed_content, parsed_tool_calls, _finish = parse_web_tool_response(
                    projector.raw_text,
                    available_tools=request.tools if request.tools else None,
                    tool_choice=request.tool_choice,
                )
                if parsed_tool_calls:
                    stream_tool_calls = parsed_tool_calls
            except Exception:
                # 解析失败不应该污染已经流出的可见内容；当作没工具调用继续收尾。
                pass
            for tool_call in stream_tool_calls:
                item = _function_call_output_item(tool_call, status="in_progress", arguments_override="")
                completed_item = _function_call_output_item(tool_call, item_id=str(item["id"]), status="completed")
                yield from _sse_event(
                    "response.output_item.added",
                    {"type": "response.output_item.added", "output_index": output_index, "item": item},
                )
                function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                if not isinstance(function, dict):
                    function = {}
                arguments = function.get("arguments", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
                for piece in _chunk_function_call_arguments(arguments):
                    yield from _sse_event(
                        "response.function_call_arguments.delta",
                        {
                            "type": "response.function_call_arguments.delta",
                            "item_id": item["id"],
                            "output_index": output_index,
                            "delta": piece,
                        },
                    )
                yield from _sse_event(
                    "response.function_call_arguments.done",
                    {
                        "type": "response.function_call_arguments.done",
                        "item_id": item["id"],
                        "output_index": output_index,
                        "arguments": arguments,
                        "name": str(function.get("name", "")).strip(),
                    },
                )
                yield from _sse_event(
                    "response.output_item.done",
                    {"type": "response.output_item.done", "output_index": output_index, "item": completed_item},
                )
                completed_output.append(_function_call_output_item(tool_call, item_id=str(item["id"])))
                output_index += 1
            completed_response = _response_resource(
                response_id=response_id,
                created_at=created_at,
                status="incomplete" if stream_tool_calls else "completed",
                model=request.model,
                output=completed_output,
                usage=_estimate_response_usage(request, rendered_content),
            )
            _save_response_history(
                response_id=response_id,
                request=request,
                response=ChatResponse(
                    model=request.model,
                    content=rendered_content,
                    tool_calls=stream_tool_calls,
                ),
            )
            yield from _sse_event(
                "response.completed",
                {"type": "response.completed", "response": completed_response},
            )
            return

    try:
        response = router.chat(request)
    except Exception as exc:
        yield from _sse_event("error", _stream_error_payload(exc))
        return

    completed_output: list[dict[str, object]] = []
    output_index = 0
    rendered_content = strip_tool_protocol_markup(response.content) or ""
    tool_calls = response.tool_calls

    segments = _split_response_output_segments(rendered_content)
    if segments:
        for kind, text in segments:
            completed_item = yield from _yield_segment_events(kind, text, output_index=output_index)
            completed_output.append(completed_item)
            output_index += 1
    elif not tool_calls:
        completed_item = yield from _yield_segment_events("message", "", output_index=output_index)
        completed_output.append(completed_item)
        output_index += 1

    for tool_call in tool_calls:
        item = _function_call_output_item(tool_call, status="in_progress", arguments_override="")
        completed_item = _function_call_output_item(tool_call, item_id=str(item["id"]), status="completed")
        yield from _sse_event(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": item,
            },
        )
        function = tool_call.get("function", {})
        if not isinstance(function, dict):
            function = {}
        arguments = function.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        for piece in _chunk_function_call_arguments(arguments):
            yield from _sse_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": item["id"],
                    "output_index": output_index,
                    "delta": piece,
                },
            )
        yield from _sse_event(
            "response.function_call_arguments.done",
            {
                "type": "response.function_call_arguments.done",
                "item_id": item["id"],
                "output_index": output_index,
                "arguments": arguments,
                "name": str(function.get("name", "")).strip(),
            },
        )
        yield from _sse_event(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": completed_item,
            },
        )
        completed_output.append(_function_call_output_item(tool_call, item_id=str(item["id"])))
        output_index += 1

    completed_response = _response_resource(
        response_id=response_id,
        created_at=created_at,
        status="incomplete" if tool_calls else "completed",
        model=response.model,
        output=completed_output,
        usage=_estimate_response_usage(request, rendered_content),
    )
    _save_response_history(response_id=response_id, request=request, response=response)
    yield from _sse_event(
        "response.completed",
        {"type": "response.completed", "response": completed_response},
    )


def _format_validation_error(exc: ValidationError) -> str:
    first_error = exc.errors(include_url=False)[0]
    location = ".".join(str(part) for part in first_error.get("loc", ()))
    message = str(first_error.get("msg", "Invalid request."))
    if not location:
        return message
    return f"{location}: {message}"


def _assistant_output_item(
    content: str,
    *,
    item_id: str | None = None,
    status: str = "completed",
) -> dict[str, object]:
    return {
        "type": "message",
        "id": item_id or f"msg-{uuid4().hex}",
        "role": "assistant",
        "status": status,
        "content": [
            {
                "type": "output_text",
                "text": content,
            }
        ],
    }


def _build_response_output(response) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    rendered_content = strip_tool_protocol_markup(response.content) or ""
    segments = _split_response_output_segments(rendered_content)
    if segments:
        for kind, text in segments:
            if kind == "reasoning":
                output.append(_reasoning_output_item(text))
            else:
                output.append(_assistant_output_item(text))
    elif not response.tool_calls:
        output.append(_assistant_output_item(""))
    for tool_call in response.tool_calls:
        output.append(_function_call_output_item(tool_call))
    return output


def _function_call_output_item(
    tool_call: dict[str, object],
    *,
    item_id: str | None = None,
    arguments_override: str | None = None,
    status: str | None = None,
) -> dict[str, object]:
    function = tool_call.get("function", {})
    if not isinstance(function, dict):
        function = {}
    arguments = arguments_override if arguments_override is not None else function.get("arguments", "{}")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    item = {
        "type": "function_call",
        "id": item_id or f"fc-{uuid4().hex}",
        "call_id": str(tool_call.get("id", "")).strip() or f"call_{uuid4().hex[:8]}",
        "name": str(function.get("name", "")).strip(),
        "arguments": arguments,
    }
    if status is not None:
        item["status"] = status
    return item


def _reasoning_output_item(
    content: str,
    *,
    item_id: str | None = None,
    status: str = "completed",
) -> dict[str, object]:
    return {
        "type": "reasoning",
        "id": item_id or f"rs-{uuid4().hex}",
        "status": status,
        "summary": [],
        "content": [
            {
                "type": "reasoning_text",
                "text": content,
            }
        ],
    }


def _empty_response_usage() -> dict[str, object]:
    return _response_usage(input_tokens=0, output_tokens=0)


def _response_usage(*, input_tokens: int, output_tokens: int) -> dict[str, object]:
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens + output_tokens,
    }


def _estimate_response_usage(request, output_text: str) -> dict[str, object]:
    prompt_tokens = estimate_prompt_tokens(getattr(request, "messages", None))
    return _response_usage(
        input_tokens=prompt_tokens,
        output_tokens=estimate_tokens(output_text or ""),
    )


def _response_resource(
    *,
    response_id: str,
    created_at: int,
    status: str,
    model: str,
    output: list[dict[str, object]],
    usage: dict[str, object],
) -> dict[str, object]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "model": model,
        "output": output,
        "usage": usage,
    }


def _sse_event(event_name: str, payload: dict[str, object]):
    yield f"event: {event_name}\n"
    yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _chunk_stream_text(content: str, *, max_chunk_len: int = 12) -> list[str]:
    return chunk_visible_text(content, max_chunk_len=max_chunk_len)


def _split_response_output_segments(rendered_content: str) -> list[tuple[str, str]]:
    segments: list[tuple[str, str]] = []
    last_index = 0
    for match in _THINK_BLOCK_PATTERN.finditer(rendered_content):
        if match.start() > last_index:
            _append_output_segment(segments, "message", rendered_content[last_index : match.start()])
        _append_output_segment(segments, "reasoning", match.group("body"))
        last_index = match.end()
    if last_index < len(rendered_content):
        _append_output_segment(segments, "message", rendered_content[last_index:])
    return segments


def _split_stream_piece_segments(piece: str, *, in_reasoning: bool) -> tuple[list[tuple[str, str]], bool]:
    segments: list[tuple[str, str]] = []
    last_index = 0
    for match in _THINK_TAG_PATTERN.finditer(piece):
        if match.start() > last_index:
            _append_output_segment(
                segments,
                "reasoning" if in_reasoning else "message",
                piece[last_index : match.start()],
            )
        in_reasoning = not match.group(0).startswith("</")
        last_index = match.end()
    if last_index < len(piece):
        _append_output_segment(
            segments,
            "reasoning" if in_reasoning else "message",
            piece[last_index:],
        )
    return segments, in_reasoning


def _append_output_segment(segments: list[tuple[str, str]], kind: str, text: str) -> None:
    if not text:
        return
    if segments and segments[-1][0] == kind:
        previous_kind, previous_text = segments[-1]
        segments[-1] = (previous_kind, previous_text + text)
        return
    segments.append((kind, text))


def _yield_segment_events(kind: str, text: str, *, output_index: int):
    item_id = _build_stream_item_id(kind)
    yield from _yield_segment_open_events(kind, item_id=item_id, output_index=output_index)
    yield from _yield_segment_delta_events(kind, text, item_id=item_id, output_index=output_index)
    return (
        yield from _yield_segment_done_events(kind, text, item_id=item_id, output_index=output_index)
    )


def _build_stream_item_id(kind: str) -> str:
    prefix = "rs" if kind == "reasoning" else "msg"
    return f"{prefix}-{uuid4().hex}"


def _yield_segment_open_events(kind: str, *, item_id: str, output_index: int):
    if kind == "reasoning":
        in_progress_item = _reasoning_output_item("", item_id=item_id, status="in_progress")
        yield from _sse_event(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": in_progress_item,
            },
        )
        return

    in_progress_item = _assistant_output_item("", item_id=item_id, status="in_progress")
    yield from _sse_event(
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": in_progress_item,
        },
    )
    yield from _sse_event(
        "response.content_part.added",
        {
            "type": "response.content_part.added",
            "item_id": item_id,
            "output_index": output_index,
            "content_index": 0,
            "part": {"type": "output_text", "text": ""},
        },
    )


def _yield_segment_delta_events(kind: str, text: str, *, item_id: str, output_index: int):
    if kind == "reasoning":
        for piece in _chunk_stream_text(text):
            yield from _sse_event(
                "response.reasoning_text.delta",
                {
                    "type": "response.reasoning_text.delta",
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "delta": piece,
                },
            )
        return

    for piece in _chunk_stream_text(text):
        yield from _sse_event(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "delta": piece,
            },
        )


def _yield_segment_done_events(kind: str, text: str, *, item_id: str, output_index: int):
    if kind == "reasoning":
        yield from _sse_event(
            "response.reasoning_text.done",
            {
                "type": "response.reasoning_text.done",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "text": text,
            },
        )
        completed_item = _reasoning_output_item(text, item_id=item_id, status="completed")
        yield from _sse_event(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": completed_item,
            },
        )
        return completed_item

    yield from _sse_event(
        "response.output_text.done",
        {
            "type": "response.output_text.done",
            "item_id": item_id,
            "output_index": output_index,
            "content_index": 0,
            "text": text,
        },
    )
    yield from _sse_event(
        "response.content_part.done",
        {
            "type": "response.content_part.done",
            "item_id": item_id,
            "output_index": output_index,
            "content_index": 0,
            "part": {"type": "output_text", "text": text},
        },
    )
    completed_item = _assistant_output_item(text, item_id=item_id, status="completed")
    yield from _sse_event(
        "response.output_item.done",
        {
            "type": "response.output_item.done",
            "output_index": output_index,
            "item": completed_item,
        },
    )
    return completed_item


def _chunk_function_call_arguments(arguments: str, *, max_chunk_len: int = 24) -> list[str]:
    if not arguments:
        return []
    return [arguments[index : index + max_chunk_len] for index in range(0, len(arguments), max_chunk_len)]


def _resolve_request_with_previous_response(payload: dict[str, object]):
    request = normalize_responses_request(payload)
    previous_response_id = str(payload.get("previous_response_id", "")).strip()
    if not previous_response_id:
        return request
    previous_messages = load_response_messages(resolve_state_dir(), previous_response_id)
    if previous_messages is None:
        raise RuntimeError(f"Unknown previous_response_id: {previous_response_id}")
    # The new request's `instructions` are normalized into a leading system
    # message. When chaining onto a prior conversation, that system message must
    # stay at the FRONT of the model context — a plain `previous + new` concat
    # buried it after the entire prior conversation, so the active instructions
    # arrived last and models largely ignored them. Hoist leading system
    # messages to the top, then prior turns, then the new user input.
    new_messages = list(request.messages)
    leading_system: list[dict[str, object]] = []
    while new_messages and str(new_messages[0].get("role")) == "system":
        leading_system.append(new_messages.pop(0))
    merged = leading_system + list(previous_messages) + new_messages
    return request.model_copy(update={"messages": merged})


def _save_response_history(*, response_id: str, request, response: ChatResponse) -> None:
    messages = list(request.messages)
    # 保存历史时剥掉 <think> 标签和内部内容 —— 否则 previous_response_id 续聊
    # 会把模型自己的 reasoning 重新喂回去：(a) 上下文翻倍/三倍,推理成本爆炸；
    # (b) 模型被自己的草稿干扰。客户端如果想看推理过程,流式 SSE 里 think 标签
    # 还在,这里只是不写入持久化历史。
    rendered_content = strip_tool_protocol_markup(response.content, include_think=False)
    assistant_message: dict[str, object] = {
        "role": "assistant",
        "content": rendered_content,
    }
    if response.tool_calls:
        assistant_message["tool_calls"] = response.tool_calls
        if rendered_content in {"", None}:
            assistant_message["content"] = None
    messages.append(assistant_message)
    save_response_messages(
        resolve_state_dir(),
        response_id=response_id,
        model=response.model,
        messages=messages,
    )


def _stream_error_payload(exc: Exception) -> dict[str, object]:
    if isinstance(exc, ProviderRateLimitError):
        error_type = "rate_limit_error"
    elif isinstance(exc, RuntimeError):
        # Match the non-stream path: classify upstream RuntimeErrors so a 502 /
        # auth failure mid-stream isn't mislabeled invalid_request_error.
        _status, error_type = classify_provider_runtime_error(exc)
    elif isinstance(exc, httpx.HTTPError):
        error_type = "api_error"
    else:
        error_type = "api_error"
    # OpenAI Responses streaming "error" events carry flat top-level fields
    # (type/code/message/param) — NOT a nested "error" object (that nested shape
    # is the Chat Completions / non-stream convention). Clients parsing the
    # Responses SSE stream look for message/code at the top level.
    return {
        "type": "error",
        "code": error_type,
        "message": str(exc),
        "param": None,
    }
