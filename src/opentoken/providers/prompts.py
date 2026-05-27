from __future__ import annotations

import base64
import ipaddress
import json
import mimetypes
import socket
import re
from urllib.parse import unquote_to_bytes, urlparse

import httpx

from opentoken.gateway.normalized import NormalizedChatRequest


_ROLE_LABELS = {
    "system": "System",
    "assistant": "Assistant",
    "tool": "Tool",
    "user": "User",
}
_ATTACHMENT_MAX_BYTES = 2 * 1024 * 1024
_ATTACHMENT_MAX_CHARS = 20_000
_TEXT_ATTACHMENT_MIME_TYPES = {
    "application/json",
    "application/xml",
    "text/csv",
    "text/html",
    "text/markdown",
    "text/plain",
    "text/xml",
}


def stringify_message_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Preserve the ORDER items appear in. The previous implementation
        # bucketed text and attachments into two lists and concatenated
        # text-then-attachments, which destroyed interleaving: an array like
        # [image, "what is in this image?"] rendered text-first, and
        # [text, img, text, img] collapsed to both texts then both images, so
        # "describe the second photo" lost which image it referred to.
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in {"text", "input_text", "output_text"} and isinstance(
                item.get("text"), str
            ):
                parts.append(item["text"])
                continue
            attachment_text = _stringify_attachment_item(item)
            if attachment_text:
                parts.append(attachment_text)
            attachment_content = _extract_attachment_content(item)
            if attachment_content:
                parts.append(attachment_content)
        return "\n".join(part for part in parts if part)
    return str(content)


def _stringify_attachment_item(item: dict[str, object]) -> str:
    item_type = str(item.get("type", "")).strip()
    if item_type in {"image_url", "input_image", "image"}:
        return _describe_image_item(item)
    if item_type in {"input_file", "file"}:
        return _describe_file_item(item)
    return ""


def _describe_image_item(item: dict[str, object]) -> str:
    raw_image = item.get("image_url") or item.get("url") or item.get("image")
    detail = _describe_resource_source(raw_image, fallback_label="image")
    if not detail:
        return "[Attached image]"
    return f"[Attached image: {detail}]"


def _describe_file_item(item: dict[str, object]) -> str:
    filename = str(item.get("filename") or item.get("name") or "").strip()
    source = item.get("file_url") or item.get("file_data") or item.get("url") or item.get("data")
    detail = _describe_resource_source(source, fallback_label="file")
    pieces = [piece for piece in (filename, detail) if piece]
    if not pieces:
        return "[Attached file]"
    return "[Attached file: " + " | ".join(pieces) + "]"


def _extract_attachment_content(item: dict[str, object]) -> str:
    item_type = str(item.get("type", "")).strip()
    if item_type not in {"input_file", "file"}:
        return ""
    filename = str(item.get("filename") or item.get("name") or "").strip()
    try:
        loaded = _load_attachment_payload(item, filename=filename)
    except Exception:
        return ""
    if loaded is None:
        return ""
    raw_bytes, mime_type = loaded
    if not _is_text_attachment(mime_type, filename=filename):
        return ""
    text = _decode_attachment_text(raw_bytes)
    normalized = text.strip()
    if not normalized:
        return ""
    if len(normalized) > _ATTACHMENT_MAX_CHARS:
        normalized = normalized[:_ATTACHMENT_MAX_CHARS].rstrip() + "\n...[truncated]"
    label = filename or "attachment"
    return f"[Attached file content: {label}]\n{normalized}"


def _load_attachment_payload(
    item: dict[str, object],
    *,
    filename: str,
) -> tuple[bytes, str | None] | None:
    source = item.get("file_data") or item.get("file_url") or item.get("data") or item.get("url")
    if isinstance(source, dict):
        source = source.get("data") or source.get("url")
    if not isinstance(source, str):
        return None
    value = source.strip()
    if not value:
        return None
    guessed_mime = mimetypes.guess_type(filename or value)[0]
    if value.startswith("data:"):
        return _decode_data_uri(value, fallback_mime=guessed_mime)
    if value.startswith("http://") or value.startswith("https://"):
        return _fetch_attachment_url(value, fallback_mime=guessed_mime)
    # file:// 与裸本地路径直接拒绝：网关面向公网时这会变成任意本地文件读取漏洞。
    # 客户端需要通过 /v1/files 上传后用 file_id 引用。
    return None


def _decode_data_uri(value: str, *, fallback_mime: str | None) -> tuple[bytes, str | None] | None:
    match = re.match(
        r"^data:(?P<mime>[^;,]+)?(?P<base64>;base64)?,(?P<data>.*)$",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None
    mime_type = (match.group("mime") or fallback_mime or "").strip().lower() or fallback_mime
    payload = match.group("data")
    try:
        if match.group("base64"):
            raw = base64.b64decode(payload, validate=False)
        else:
            raw = unquote_to_bytes(payload)
    except Exception:
        return None
    return _clamp_attachment_bytes(raw), mime_type


def _fetch_attachment_url(url: str, *, fallback_mime: str | None) -> tuple[bytes, str | None] | None:
    if not _is_public_http_url(url):
        return None
    with httpx.Client(timeout=10.0, trust_env=False, follow_redirects=False) as client:
        response = client.get(url)
        # 不允许重定向：上游可借 302 跨越 SSRF 防线指向内网或 file://。
        if response.status_code in {301, 302, 303, 307, 308}:
            return None
        response.raise_for_status()
        content_type = str(response.headers.get("content-type", "")).split(";", 1)[0].strip().lower()
        return _clamp_attachment_bytes(response.content), content_type or fallback_mime


def _is_public_http_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").strip()
    if not host:
        return False
    # 即便目标域名公开,也要把它解析出的所有 IP 一一校验,否则可被 DNS 重绑或公开域名指内网绕过。
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except OSError:
        return False
    for info in infos:
        try:
            addr = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_multicast
            or addr.is_reserved
            or addr.is_unspecified
        ):
            return False
    return True


def _clamp_attachment_bytes(raw: bytes) -> bytes:
    if len(raw) <= _ATTACHMENT_MAX_BYTES:
        return raw
    return raw[:_ATTACHMENT_MAX_BYTES]


def _is_text_attachment(mime_type: str | None, *, filename: str) -> bool:
    normalized = (mime_type or "").strip().lower()
    if normalized.startswith("text/"):
        return True
    if normalized in _TEXT_ATTACHMENT_MIME_TYPES:
        return True
    guessed = mimetypes.guess_type(filename)[0] or ""
    return guessed.startswith("text/") or guessed in _TEXT_ATTACHMENT_MIME_TYPES


def _decode_attachment_text(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "utf-16le", "utf-16be"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _describe_resource_source(source: object, *, fallback_label: str) -> str:
    if isinstance(source, dict):
        source = source.get("url") or source.get("data")
    if not isinstance(source, str):
        return ""
    value = source.strip()
    if not value:
        return ""
    if value.startswith("data:"):
        match = re.match(r"^data:(?P<mime>[^;,]+)", value, flags=re.IGNORECASE)
        mime = match.group("mime") if match is not None else fallback_label
        return f"data URI ({mime})"
    if len(value) > 256:
        return value[:253] + "..."
    return value


def build_role_prompt(request: NormalizedChatRequest) -> str:
    history: list[str] = []
    for message in request.messages:
        role = str(message.get("role", "user"))
        label = _ROLE_LABELS.get(role, role.title())
        content = stringify_message_content(message.get("content", ""))
        if content:
            history.append(f"{label}: {content}")
    if not history:
        raise RuntimeError("Requests require at least one message.")
    return "\n\n".join(history)


def build_doubao_prompt(request: NormalizedChatRequest) -> str:
    chunks: list[str] = []
    for message in request.messages:
        role = str(message.get("role", "user"))
        if role not in {"system", "assistant", "user"}:
            role = "user"
        content = stringify_message_content(message.get("content", ""))
        if not content:
            continue
        chunks.append(f"<|im_start|>{role}\n{content}\n")
    if not chunks:
        raise RuntimeError("Doubao requests require at least one message.")
    return "".join(chunks) + "<|im_end|>\n"


def build_qwen_prompt(request: NormalizedChatRequest) -> str:
    if not request.messages:
        raise RuntimeError("Qwen requests require at least one message.")

    history: list[str] = []
    system_parts: list[str] = []
    tool_name_by_id = _collect_tool_call_names(request.messages)

    for message in request.messages:
        role = str(message.get("role", "user"))
        if role in {"system", "developer"}:
            content = stringify_message_content(message.get("content", ""))
            if content:
                system_parts.append(content)

    tool_instructions = _build_qwen_tool_instructions(request)
    if tool_instructions:
        system_parts.append(tool_instructions)

    if system_parts:
        history.append(f"System: {'\n\n'.join(system_parts)}")

    for message in request.messages:
        role = str(message.get("role", "user"))
        if role in {"system", "developer"}:
            continue
        formatted = _format_qwen_message(message, tool_name_by_id)
        if not formatted:
            continue
        label = "User" if role == "tool" else _ROLE_LABELS.get(role, role.title())
        history.append(f"{label}: {formatted}")

    if tool_instructions and any(str(message.get("role", "")) == "tool" for message in request.messages):
        history.append(
            'System: [SYSTEM HINT]: Keep in mind your available tools. To use a tool, you MUST output the EXACT XML format: <tool_call id="unique_id" name="tool_name">{"arg": "value"}</tool_call>. Using plain text to describe your action will FAIL to execute the tool.'
        )

    if not history:
        raise RuntimeError("Qwen requests require at least one message.")
    return "\n\n".join(part for part in history if part.strip())


def build_browser_tool_prompt(request: NormalizedChatRequest) -> str:
    if not request.messages:
        raise RuntimeError("Browser tool requests require at least one message.")

    tool_instructions = _build_browser_tool_instructions(request)
    tool_name_by_id = _collect_tool_call_names(request.messages)
    messages = list(request.messages)
    last_message = messages[-1] if messages else {}
    last_role = str(last_message.get("role", "")).strip()

    if last_role == "tool":
        tool_call_id = str(last_message.get("tool_call_id", "")).strip() or "call_unknown"
        tool_name = tool_name_by_id.get(tool_call_id, "tool")
        content = stringify_message_content(last_message.get("content", ""))
        prompt = (
            f'\n<tool_response id="{tool_call_id}" name="{tool_name}">\n'
            f"{content}\n"
            "</tool_response>\n\n"
            "Please proceed based on this tool result."
        )
        if tool_instructions:
            prompt += f"\n\n{_browser_tool_system_hint()}"
        return prompt

    history: list[str] = []
    system_parts: list[str] = []

    for message in messages:
        role = str(message.get("role", "user"))
        if role in {"system", "developer"}:
            content = stringify_message_content(message.get("content", ""))
            if content:
                system_parts.append(content)

    if tool_instructions:
        system_parts.append(tool_instructions)

    if system_parts:
        history.append(f"System: {'\n\n'.join(system_parts)}")

    for message in messages:
        role = str(message.get("role", "user"))
        if role in {"system", "developer"}:
            continue
        formatted = _format_qwen_message(message, tool_name_by_id)
        if not formatted:
            continue
        label = "User" if role == "tool" else _ROLE_LABELS.get(role, role.title())
        history.append(f"{label}: {formatted}")

    if tool_instructions and any(str(message.get("role", "")) == "tool" for message in messages):
        history.append(f"System: {_browser_tool_system_hint()}")

    if not history:
        raise RuntimeError("Browser tool requests require at least one message.")
    return "\n\n".join(part for part in history if part.strip())


def build_browser_tool_repair_prompt(
    request: NormalizedChatRequest,
    *,
    invalid_response: str,
    error: str,
) -> str:
    repair_messages = list(request.messages)
    repair_messages.extend(
        [
            {"role": "assistant", "content": invalid_response},
            {
                "role": "user",
                "content": (
                    "Your previous response violated the required XML tool-calling format. "
                    f"Error: {error}. Rewrite the SAME turn now. "
                    "If you need a tool, output ONLY the exact XML form "
                    '<tool_call id="unique_id" name="tool_name">{"arg":"value"}</tool_call>. '
                    "Do not explain the error. Do not mention unavailable tools."
                ),
            },
        ]
    )
    return build_browser_tool_prompt(request.model_copy(update={"messages": repair_messages}))


def _build_qwen_tool_instructions(request: NormalizedChatRequest) -> str:
    if request.tool_choice == "none" or not request.tools:
        return ""

    lines = [
        "## Tool Use Instructions",
        "You are equipped with specialized tools to perform actions or retrieve information.",
        'To use a tool, output a specific XML tag: <tool_call id="unique_id" name="tool_name">{"arg": "value"}</tool_call>.',
        "Rules for tool use:",
        "1. ALWAYS think before calling a tool. Explain your reasoning inside <think> tags.",
        "2. The 'id' attribute should be a unique 8-character string for each call.",
        "3. Wait for the tool result before proceeding with further analysis.",
    ]

    requested_tool = _requested_tool_name(request.tool_choice)
    if request.tool_choice == "required":
        lines.append("You MUST call at least one tool before answering.")
    elif requested_tool:
        lines.append(f"You MUST call the tool `{requested_tool}` before answering.")

    lines.append("")
    lines.append("### Available Tools")

    for tool in request.tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name", "")).strip()
        if not name:
            continue
        description = str(function.get("description", "")).strip()
        parameters = function.get("parameters", {})
        lines.append(f"#### {name}")
        if description:
            lines.append(description)
        lines.append(
            f"Parameters: {json.dumps(parameters, ensure_ascii=False, separators=(',', ':'))}"
        )
        lines.append("")

    return "\n".join(lines).strip()


def _build_browser_tool_instructions(request: NormalizedChatRequest) -> str:
    if request.tool_choice == "none" or not request.tools:
        return ""

    lines = [
        "## Tool Use Instructions",
        "You have access to external tools.",
        'To call a tool, you MUST output the EXACT XML format: <tool_call id="unique_id" name="tool_name">{"arg":"value"}</tool_call>.',
        "Do not describe tool usage in plain text.",
        "Call only tools from the available tools list.",
        "After you receive a <tool_response ...> block, continue based on that tool result.",
    ]

    requested_tool = _requested_tool_name(request.tool_choice)
    if request.tool_choice == "required":
        lines.append("You MUST call at least one tool before answering.")
    elif requested_tool:
        lines.append(f"You MUST call only the tool `{requested_tool}` before answering.")

    lines.append("")
    lines.append("### Available Tools")

    for tool in request.tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name", "")).strip()
        if not name:
            continue
        description = str(function.get("description", "")).strip()
        parameters = function.get("parameters", {})
        lines.append(f"#### {name}")
        if description:
            lines.append(description)
        lines.append(
            f"Parameters: {json.dumps(parameters, ensure_ascii=False, separators=(',', ':'))}"
        )
        lines.append("")

    return "\n".join(lines).strip()


def _browser_tool_system_hint() -> str:
    return (
        '[SYSTEM HINT]: Keep in mind your available tools. To use a tool, you MUST output '
        'the EXACT XML format: <tool_call id="unique_id" name="tool_name">{"arg": "value"}</tool_call>. '
        "Using plain text to describe your action will FAIL to execute the tool."
    )


def _requested_tool_name(tool_choice: object) -> str | None:
    if not isinstance(tool_choice, dict):
        return None
    tool_type = str(tool_choice.get("type", "")).strip()
    if tool_type != "function":
        return None
    function = tool_choice.get("function")
    if not isinstance(function, dict):
        return None
    name = str(function.get("name", "")).strip()
    return name or None


def _collect_tool_call_names(messages: list[dict[str, object]]) -> dict[str, str]:
    names: dict[str, str] = {}
    for message in messages:
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("id", "")).strip()
            function = call.get("function")
            if not call_id or not isinstance(function, dict):
                continue
            name = str(function.get("name", "")).strip()
            if name:
                names[call_id] = name
    return names


def _format_qwen_message(
    message: dict[str, object],
    tool_name_by_id: dict[str, str],
) -> str:
    role = str(message.get("role", "user"))
    if role == "assistant":
        parts: list[str] = []
        content = stringify_message_content(message.get("content", ""))
        if content:
            parts.append(content)
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if not isinstance(function, dict):
                    continue
                name = str(function.get("name", "")).strip()
                if not name:
                    continue
                call_id = str(call.get("id", "")).strip() or "call_unknown"
                arguments = function.get("arguments", "{}")
                parts.append(
                    f'<tool_call id="{call_id}" name="{name}">{_stringify_jsonish(arguments)}</tool_call>'
                )
        return "".join(parts).strip()

    if role == "tool":
        tool_call_id = str(message.get("tool_call_id", "")).strip() or "call_unknown"
        tool_name = tool_name_by_id.get(tool_call_id, "tool")
        content = stringify_message_content(message.get("content", ""))
        return (
            f'\n<tool_response id="{tool_call_id}" name="{tool_name}">\n'
            f"{content}\n"
            f"</tool_response>\n"
        )

    return stringify_message_content(message.get("content", ""))


def _stringify_jsonish(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
