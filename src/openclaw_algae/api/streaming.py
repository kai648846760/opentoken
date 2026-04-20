"""SSE streaming utilities for OpenAI-compatible responses."""
from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from time import time
from uuid import uuid4


_BASE_HIDDEN_PROTOCOL_TAGS = frozenset({"tool_calls", "tool_call"})
_THINK_PROTOCOL_TAG = "think"
_PROTOCOL_TAG_PREFIXES: tuple[tuple[str, str, bool], ...] = (
    ("</final_answer", "final_answer", True),
    ("</tool_calls", "tool_calls", True),
    ("</tool_call", "tool_call", True),
    ("</think", "think", True),
    ("<final_answer", "final_answer", False),
    ("<tool_calls", "tool_calls", False),
    ("<tool_call", "tool_call", False),
    ("<think", "think", False),
)
_PROTOCOL_TAG_PATTERNS: dict[tuple[str, bool], re.Pattern[str]] = {
    ("think", False): re.compile(r"<think\b[^>]*>", re.IGNORECASE),
    ("think", True): re.compile(r"</think\s*>", re.IGNORECASE),
    ("tool_calls", False): re.compile(r"<tool_calls\s*>", re.IGNORECASE),
    ("tool_calls", True): re.compile(r"</tool_calls\s*>", re.IGNORECASE),
    ("tool_call", False): re.compile(r"<tool_call\b[^>]*>", re.IGNORECASE),
    ("tool_call", True): re.compile(r"</tool_call\s*>", re.IGNORECASE),
    ("final_answer", False): re.compile(r"<final_answer\s*>", re.IGNORECASE),
    ("final_answer", True): re.compile(r"</final_answer\s*>", re.IGNORECASE),
}
_THINK_BLOCK_PATTERN = re.compile(r"(<think\b[^>]*>)([\s\S]*?)(</think\s*>)", re.IGNORECASE)


def sse_event(data: object, *, event_name: str | None = None) -> str:
    """Format an SSE event line."""
    lines = []
    if event_name:
        lines.append(f"event: {event_name}")
    if isinstance(data, str):
        lines.append(f"data: {data}")
    else:
        lines.append(f"data: {json.dumps(data, separators=(',', ':'))}")
    lines.append("")
    return "\n".join(lines)


def sse_comment(text: str) -> str:
    """Format an SSE comment (heartbeat)."""
    return f":{text}\n\n"


def sse_response_headers() -> dict[str, str]:
    """Headers that reduce intermediary/client buffering for SSE streams."""
    return {
        "Cache-Control": "no-cache, no-transform",
        "Pragma": "no-cache",
        "X-Accel-Buffering": "no",
    }


def strip_tool_protocol_markup(content: str | None, *, include_think: bool = True) -> str | None:
    if content is None:
        return None
    return _project_visible_protocol_text(content, include_think=include_think)


def chunk_visible_text(
    content: str,
    *,
    max_chunk_len: int = 12,
    include_think: bool = True,
) -> list[str]:
    text = _project_visible_protocol_text(content or "", include_think=include_think)
    if not text:
        return []

    if not include_think:
        return _chunk_plain_text(text, max_chunk_len=max_chunk_len)

    chunks: list[str] = []
    last_index = 0
    for match in _THINK_BLOCK_PATTERN.finditer(text):
        if match.start() > last_index:
            chunks.extend(_chunk_plain_text(text[last_index : match.start()], max_chunk_len=max_chunk_len))
        opening_tag, think_body, closing_tag = match.groups()
        chunks.append(opening_tag)
        chunks.extend(_chunk_plain_text(think_body, max_chunk_len=max_chunk_len))
        chunks.append(closing_tag)
        last_index = match.end()
    if last_index < len(text):
        chunks.extend(_chunk_plain_text(text[last_index:], max_chunk_len=max_chunk_len))
    return [chunk for chunk in chunks if chunk]


class ProtocolMarkupProjector:
    def __init__(self, *, include_think: bool = True) -> None:
        self._include_think = include_think
        self._raw_text = ""
        self._visible_text = ""

    @property
    def visible_text(self) -> str:
        return self._visible_text

    def push(self, piece: str) -> str:
        if not piece:
            return ""
        self._raw_text += piece
        projected = _project_visible_protocol_text(self._raw_text, include_think=self._include_think)
        common_prefix_len = _common_prefix_length(self._visible_text, projected)
        delta = projected[common_prefix_len:]
        self._visible_text = projected
        return delta


def _common_prefix_length(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _project_visible_protocol_text(content: str, *, include_think: bool = True) -> str:
    if not content:
        return ""

    visible_parts: list[str] = []
    stack: list[str] = []
    hidden_depth = 0
    index = 0
    visible_tags = {_THINK_PROTOCOL_TAG} if include_think else set()
    hidden_tags = set(_BASE_HIDDEN_PROTOCOL_TAGS)
    if not include_think:
        hidden_tags.add(_THINK_PROTOCOL_TAG)

    while index < len(content):
        if content[index] == "<":
            matched_tag = _match_protocol_tag(content, index)
            if matched_tag == "partial":
                break
            if matched_tag is not None:
                tag_name, is_closing, next_index, raw_tag = matched_tag
                was_hidden = hidden_depth > 0
                if is_closing:
                    for stack_index in range(len(stack) - 1, -1, -1):
                        if stack[stack_index] != tag_name:
                            continue
                        for popped in stack[stack_index:]:
                            if popped in hidden_tags:
                                hidden_depth -= 1
                        del stack[stack_index:]
                        break
                else:
                    stack.append(tag_name)
                    if tag_name in hidden_tags:
                        hidden_depth += 1
                if (
                    tag_name in visible_tags
                    and not was_hidden
                    and hidden_depth == 0
                ):
                    visible_parts.append(raw_tag)
                index = next_index
                continue
        if hidden_depth == 0:
            visible_parts.append(content[index])
        index += 1

    return "".join(visible_parts)


def _chunk_plain_text(content: str, *, max_chunk_len: int = 12) -> list[str]:
    text = content or ""
    if not text:
        return []

    wordish_parts = re.findall(r"\S+\s*", text)
    if len(wordish_parts) <= 1:
        return [text[i : i + max_chunk_len] for i in range(0, len(text), max_chunk_len)]

    chunks: list[str] = []
    current = ""
    for part in wordish_parts:
        if current and len(current) + len(part) > max_chunk_len:
            chunks.append(current)
            current = part
        else:
            current += part
    if current:
        chunks.append(current)
    return chunks


def _match_protocol_tag(
    content: str,
    start_index: int,
) -> tuple[str, bool, int, str] | str | None:
    remainder = content[start_index:].lower()
    matched: tuple[str, str, bool] | None = None

    for prefix, tag_name, is_closing in _PROTOCOL_TAG_PREFIXES:
        if remainder.startswith(prefix):
            matched = (prefix, tag_name, is_closing)
            break
        if prefix.startswith(remainder):
            return "partial"

    if matched is None:
        return None

    end_index = content.find(">", start_index)
    if end_index == -1:
        return "partial"

    _prefix, tag_name, is_closing = matched
    candidate = content[start_index : end_index + 1]
    pattern = _PROTOCOL_TAG_PATTERNS[(tag_name, is_closing)]
    if not pattern.fullmatch(candidate):
        return None
    return tag_name, is_closing, end_index + 1, candidate


def chat_completion_chunk(
    completion_id: str,
    model: str,
    content: str,
    *,
    finish_reason: str | None = None,
    created: int | None = None,
) -> str:
    """Format a chat completion chunk."""
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created or int(time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
    }
    return sse_event(chunk)


def chat_completion_done(completion_id: str, model: str, *, created: int | None = None) -> str:
    """Format the final chat completion chunk."""
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created or int(time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    return sse_event(chunk)


async def stream_with_heartbeat(
    content_stream: AsyncIterator[str],
    model: str,
    *,
    completion_id: str | None = None,
    heartbeat_interval: float = 3.0,
) -> AsyncIterator[str]:
    """Wrap a content stream with heartbeat comments to prevent timeout.

    Yields SSE chunks from the content stream, interspersed with
    heartbeat comments if no content arrives within the interval.
    """
    import asyncio

    completion_id = completion_id or f"chatcmpl-{uuid4().hex}"
    created = int(time())

    # Yield first chunk with role
    yield chat_completion_chunk(
        completion_id, model, "", finish_reason=None, created=created
    )

    last_activity = time()
    content_sent = False

    async for chunk_content in content_stream:
        # Check if we need a heartbeat
        while time() - last_activity > heartbeat_interval:
            yield sse_comment("keepalive")
            last_activity = time()

        if chunk_content:
            yield chat_completion_chunk(
                completion_id, model, chunk_content, created=created
            )
            content_sent = True
            last_activity = time()

    # Final chunk
    yield chat_completion_done(completion_id, model, created=created)
    yield sse_event("[DONE]")
