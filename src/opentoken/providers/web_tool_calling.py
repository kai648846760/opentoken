from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Callable
from uuid import uuid4

from opentoken.gateway.normalized import NormalizedChatRequest
from opentoken.providers.prompts import stringify_message_content


_FENCED_TOOL_REGEX = re.compile(r"```tool_json\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
_BARE_TOOL_REGEX = re.compile(
    r"\{\s*\"tool\"\s*:\s*\"(?P<tool>[^\"]+)\"\s*,\s*\"parameters\"\s*:\s*(?P<params>\{[\s\S]*?\})\s*\}"
)
_THINK_BLOCK_REGEX = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_PROTOCOL_MARKER_REGEX = re.compile(
    r"<(?:think|tool_call|tool_calls|final_answer)\b|```tool_json",
    re.IGNORECASE,
)
_TOOL_CALL_WITH_ATTRS_REGEX = re.compile(
    r"<tool_call\b(?P<attrs>[^>]*)>(?P<body>[\s\S]*?)</tool_call\s*>",
    re.IGNORECASE,
)
_BUILTIN_TOOL_FUNCTIONS: dict[str, dict[str, object]] = {
    "web_search": {
        "name": "web_search",
        "description": "Search the web and return relevant results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    "web_fetch": {
        "name": "web_fetch",
        "description": "Fetch and read a web page or URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
            },
            "required": ["url"],
        },
    },
}
_BUILTIN_TOOL_NAME_ALIASES: dict[str, str] = {
    "web_search": "web_search",
    "web_search_preview": "web_search",
    "web_search_preview_2025_03_11": "web_search",
    "web_fetch": "web_fetch",
    "web_fetch_preview": "web_fetch",
}
_TOOL_NAME_ALIAS_GROUPS: tuple[tuple[str, set[str]], ...] = (
    ("web_search", {"web_search", "web_search_preview", "search_web", "browser.search"}),
    ("web_fetch", {"web_fetch", "web_fetch_preview", "fetch_url", "browser.fetch"}),
    ("read", {"read", "read_file", "cat_file"}),
    ("write", {"write", "write_file"}),
    ("exec", {"exec", "run_shell_command", "exec_command", "shell"}),
    ("message", {"message", "send_message"}),
)

TAGGED_TOOL_PROMPT_PARALLEL = """You are a tool-capable assistant.

You must respond using only the following XML-like tags:
- <think>...</think>
- <tool_calls>[{"name":"ToolName","arguments":{...}}]</tool_calls>
- <final_answer>...</final_answer>

Rules:
- You may output one or more <think> blocks.
- You must then output exactly one terminal block: either <tool_calls> or <final_answer>.
- Do not output any text outside these tags.
- In <tool_calls>, the content must be a valid JSON array. Each item must be an object with keys "name" and "arguments".
- If you need only one tool, still use <tool_calls> with an array of length 1.
- In string values inside <tool_calls>, you must escape quotes, backslashes, and newlines exactly as JSON requires.
- After </tool_calls> or </final_answer>, stop immediately.
- Never generate Observation, tool results, or a second terminal block in the same response.
- Never output <observation>; the system will provide tool results in the next turn.
"""

STRICT_SUFFIX_PARALLEL = (
    "Use the strict tagged tool protocol when responding. "
    "Your final response must be optional <think> blocks followed by exactly one "
    "terminal block: <tool_calls> or <final_answer>."
)


@dataclass(frozen=True, slots=True)
class _NormalizedToolChoice:
    mode: str
    function_name: str | None = None


@dataclass(frozen=True, slots=True)
class _ParsedTaggedOutput:
    content: str | None
    tool_calls: list[dict[str, object]]
    finish_reason: str


def request_uses_web_tools(request: NormalizedChatRequest) -> bool:
    if request.tools:
        return True
    for message in request.messages:
        if str(message.get("role", "")) == "tool":
            return True
        if isinstance(message.get("tool_calls"), list):
            return True
    return False


def build_web_tool_prompt(request: NormalizedChatRequest, *, provider: str) -> str:
    if not request.messages:
        raise RuntimeError(f"{provider} requests require at least one message.")

    raw_tools = request.tools or []
    normalized_choice = _normalize_tool_choice(request.tool_choice)
    effective_tools = _select_tools_for_choice(raw_tools, normalized_choice)
    parts: list[str] = []

    if effective_tools or normalized_choice.mode != "auto":
        parts.append(
            _build_tagged_prompt(
                raw_tools,
                tool_choice=request.tool_choice,
            )
        )

    for message in request.messages:
        role = str(message.get("role", "user"))
        content = stringify_message_content(message.get("content", ""))

        if role in {"system", "developer"}:
            if content:
                parts.append(f"System:\n{content}")
            continue

        if role == "user":
            if not content:
                continue
            suffix_parts = [
                suffix
                for suffix in (
                    STRICT_SUFFIX_PARALLEL if raw_tools else "",
                    _tool_choice_user_suffix(normalized_choice),
                )
                if suffix
            ]
            if suffix_parts:
                parts.append(f"User:\n{content}\n\n" + "\n".join(suffix_parts))
            else:
                parts.append(f"User:\n{content}")
            continue

        if role == "assistant":
            if content:
                parts.append(f"Assistant:\n{content}")
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                normalized_calls = _normalize_tool_calls(tool_calls)
                call_ids = [call["id"] for call in normalized_calls if call.get("id")]
                if call_ids:
                    parts.append("Assistant tool calls: " + ", ".join(call_ids))
                if normalized_calls:
                    parts.append(_assistant_tool_block(normalized_calls))
            continue

        if role == "tool":
            if not content:
                continue
            call_id = str(message.get("tool_call_id", "")).strip()
            parts.append(
                f"Tool result for call_id={call_id}:\n"
                "<tool_result>\n"
                f"{content}\n"
                "</tool_result>\n\n"
                + _tool_result_followup_for_choice(tool_choice=request.tool_choice)
            )
            continue

        if content:
            parts.append(f"{role.title()}:\n{content}")

    prompt = "\n\n".join(part for part in parts if part.strip())
    if not prompt:
        raise RuntimeError(f"{provider} requests require at least one message.")
    return prompt


def parse_web_tool_response(
    payload: str,
    *,
    available_tools: list[dict[str, object]] | None = None,
    tool_choice: object = None,
) -> tuple[str | None, list[dict[str, object]], str]:
    normalized_choice = _normalize_tool_choice(tool_choice)
    effective_tools = _select_tools_for_choice(available_tools or [], normalized_choice)

    parsed: tuple[str | None, list[dict[str, object]], str] | None = None
    strict_error: RuntimeError | None = None

    try:
        strict_parsed = _parse_tagged_output(payload)
    except RuntimeError as exc:
        strict_error = exc
    else:
        if strict_parsed is not None:
            parsed = (
                strict_parsed.content,
                strict_parsed.tool_calls,
                strict_parsed.finish_reason,
            )

    if parsed is None:
        parsed = _parse_legacy_output(payload)

    if parsed is None:
        sanitized = _sanitize_plain_response(payload)
        if strict_error is not None or _contains_protocol_markup(payload):
            detail = str(strict_error) if strict_error is not None else "malformed strict tagged tool protocol output"
            raise RuntimeError(f"model returned malformed strict tagged tool protocol output: {detail}")
        parsed = (sanitized or None, [], "stop")

    content, tool_calls, finish_reason = parsed
    _rewrite_tool_call_names(tool_calls, effective_tools)
    _validate_parsed_output(
        content=content,
        tool_calls=tool_calls,
        available_tools=effective_tools,
        tool_choice=normalized_choice,
    )
    return content, tool_calls, finish_reason


def complete_web_tool_roundtrip(
    request: NormalizedChatRequest,
    *,
    provider: str,
    invoke: Callable[[str], str],
    max_repair_attempts: int = 2,
) -> tuple[str | None, list[dict[str, object]], str]:
    prompt = build_web_tool_prompt(request, provider=provider)
    payload = str(invoke(prompt))
    attempts = 0

    while True:
        try:
            parsed = parse_web_tool_response(
                payload,
                available_tools=request.tools,
                tool_choice=request.tool_choice,
            )
            if _should_force_tool_retry_for_explicit_request(request, parsed):
                if attempts >= max_repair_attempts:
                    return parsed
                attempts += 1
                payload = str(
                    invoke(
                        build_web_tool_repair_prompt(
                            request,
                            provider=provider,
                            invalid_response=payload,
                            error=(
                                "tool_choice=auto and the latest user message explicitly requested "
                                "an available tool, but the model returned a final answer instead"
                            ),
                        )
                    )
                )
                continue
            return parsed
        except RuntimeError as exc:
            if attempts >= max_repair_attempts:
                raise
            attempts += 1
            payload = str(
                invoke(
                    build_web_tool_repair_prompt(
                        request,
                        provider=provider,
                        invalid_response=payload,
                        error=str(exc),
                    )
                )
            )


def _should_force_tool_retry_for_explicit_request(
    request: NormalizedChatRequest,
    parsed: tuple[str | None, list[dict[str, object]], str],
) -> bool:
    _content, tool_calls, finish_reason = parsed
    if tool_calls or finish_reason != "stop":
        return False
    if _normalize_tool_choice(request.tool_choice).mode != "auto":
        return False
    markers = _explicit_tool_request_markers(request.tools or [])
    if not markers:
        return False
    latest_user_content = _latest_user_message_content(request.messages).lower()
    if not latest_user_content:
        return False
    return any(marker in latest_user_content for marker in markers)


def _latest_user_message_content(messages: list[dict[str, object]]) -> str:
    for message in reversed(messages):
        if str(message.get("role", "")).strip() != "user":
            continue
        content = stringify_message_content(message.get("content", ""))
        if content.strip():
            return content.strip()
    return ""


def _explicit_tool_request_markers(tools: list[dict[str, object]]) -> set[str]:
    markers: set[str] = set()
    for name in _allowed_tool_names(tools):
        lowered = name.lower()
        if lowered:
            markers.add(lowered)
        canonical = _canonical_builtin_tool_name(name)
        if canonical is None:
            continue
        markers.add(canonical.lower())
        for group_canonical, aliases in _TOOL_NAME_ALIAS_GROUPS:
            if group_canonical != canonical:
                continue
            markers.update(alias.lower() for alias in aliases)
            break
    return markers


def build_web_tool_repair_prompt(
    request: NormalizedChatRequest,
    *,
    provider: str,
    invalid_response: str,
    error: str,
) -> str:
    normalized_choice = _normalize_tool_choice(request.tool_choice)
    effective_tools = _select_tools_for_choice(request.tools or [], normalized_choice)
    allowed_tool_names = sorted(_allowed_tool_names(effective_tools))
    allowed_tool_line = (
        "Allowed tool names for this turn: " + ", ".join(allowed_tool_names) + ". "
        if allowed_tool_names
        else ""
    )
    repair_messages = list(request.messages)
    repair_messages.extend(
        [
            {"role": "assistant", "content": invalid_response},
            {
                "role": "user",
                "content": (
                    "Your previous response violated the strict tagged tool protocol. "
                    f"Error: {error}. Rewrite the SAME turn now. "
                    "Do not explain the error. Do not mention unavailable tools. "
                    + allowed_tool_line
                    + "Output only optional <think> blocks followed by exactly one terminal block: "
                    "<tool_calls>[...]</tool_calls> or <final_answer>...</final_answer>."
                ),
            },
        ]
    )
    repair_request = request.model_copy(update={"messages": repair_messages})
    return build_web_tool_prompt(repair_request, provider=provider)


def _parse_tagged_output(payload: str) -> _ParsedTaggedOutput | None:
    content = payload.strip()
    if not content:
        raise RuntimeError("empty tagged output")

    tool_call_matches = list(_TOOL_CALL_WITH_ATTRS_REGEX.finditer(content))
    if tool_call_matches:
        preserved_think = _extract_reasoning_markup(
            _remove_matched_ranges(content, [(match.start(), match.end()) for match in tool_call_matches])
        )
        if preserved_think is not None:
            return _ParsedTaggedOutput(
                content=preserved_think or None,
                tool_calls=[
                    _parse_single_tool_call_element(match.group("attrs"), match.group("body"))
                    for match in tool_call_matches
                ],
                finish_reason="tool_calls",
            )

    terminal_match: tuple[str, re.Match[str]] | None = None
    for tag, regex in (
        ("tool_calls", re.compile(r"<tool_calls>(?P<body>[\s\S]*?)</tool_calls\s*>", re.IGNORECASE)),
        ("tool_call", _TOOL_CALL_WITH_ATTRS_REGEX),
        ("final_answer", re.compile(r"<final_answer>(?P<body>[\s\S]*?)</final_answer\s*>", re.IGNORECASE)),
    ):
        for match in regex.finditer(content):
            if terminal_match is None or match.end() >= terminal_match[1].end():
                terminal_match = (tag, match)

    if terminal_match is None:
        if _contains_protocol_markup(content):
            raise RuntimeError("expected <tool_calls>, <tool_call>, or <final_answer>")
        return None

    tag, match = terminal_match
    if tag == "tool_calls":
        raw_tool_json = match.group("body").strip()
        preserved_think = _extract_reasoning_markup(content[: match.start()] + content[match.end() :])
        if preserved_think is None:
            raise RuntimeError("unexpected text outside <tool_calls> terminal block")
        return _ParsedTaggedOutput(
            content=preserved_think or None,
            tool_calls=_parse_parallel_tool_calls_block(raw_tool_json),
            finish_reason="tool_calls",
        )

    if tag == "tool_call":
        preserved_think = _extract_reasoning_markup(content[: match.start()] + content[match.end() :])
        if preserved_think is None:
            raise RuntimeError("unexpected text outside <tool_call> terminal block")
        return _ParsedTaggedOutput(
            content=preserved_think or None,
            tool_calls=[_parse_single_tool_call_element(match.group("attrs"), match.group("body"))],
            finish_reason="tool_calls",
        )

    if tag == "final_answer":
        answer = match.group("body").strip() or None
        preserved_think = _extract_reasoning_markup(content[: match.start()] + content[match.end() :])
        if preserved_think is None:
            raise RuntimeError("unexpected text outside <final_answer> terminal block")
        rendered = ((preserved_think or "") + (answer or "")).strip() or None
        return _ParsedTaggedOutput(content=rendered, tool_calls=[], finish_reason="stop")

    if _contains_protocol_markup(content):
        raise RuntimeError("expected <tool_calls>, <tool_call>, or <final_answer>")
    return None


def _parse_parallel_tool_calls_block(raw_json: str) -> list[dict[str, object]]:
    if not raw_json:
        raise RuntimeError("tool_calls payload must not be empty")
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        extracted = _extract_json_container_fragment(raw_json)
        if extracted is None:
            raise RuntimeError(f"invalid tool_calls json: {exc}") from exc
        try:
            payload = json.loads(extracted)
        except json.JSONDecodeError as nested_exc:
            raise RuntimeError(f"invalid tool_calls json: {nested_exc}") from nested_exc
    # Common model mistake: emitting a single tool-call object without the
    # surrounding array. Salvage it rather than hard-fail and force a repair
    # round-trip — a single object IS semantically "one tool call".
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("tool_calls payload must be a non-empty JSON array")
    counts: dict[str, int] = {}
    tool_calls: list[dict[str, object]] = []
    for item in payload:
        name, arguments = _parse_named_tool_payload(item)
        counts[name] = counts.get(name, 0) + 1
        tool_calls.append(
            {
                "id": f"call_{name}_{counts[name]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": _normalize_jsonish(arguments),
                },
            }
        )
    return tool_calls


def _parse_single_tool_call_element(attrs: str, body: str) -> dict[str, object]:
    name = _extract_tool_attr(attrs, "name")
    call_id = _extract_tool_attr(attrs, "id")
    if name:
        arguments = _normalize_xml_tool_body(body)
        return {
            "id": call_id or f"call_{name}_{uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments,
            },
        }

    parsed = _parse_tool_json(body)
    if parsed is None:
        raise RuntimeError("invalid tool_call json: expected name/arguments payload")
    tool_name = str(parsed["tool"])
    return {
        "id": call_id or f"call_{tool_name}_{uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": _normalize_jsonish(parsed["parameters"]),
        },
    }


def _parse_legacy_output(payload: str) -> tuple[str | None, list[dict[str, object]], str] | None:
    tool_calls: list[dict[str, object]] = []
    counts: dict[str, int] = {}

    def append_tool(name: str, parameters: object) -> None:
        counts[name] = counts.get(name, 0) + 1
        tool_calls.append(
            {
                "id": f"call_{name}_{counts[name]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": _normalize_jsonish(parameters),
                },
            }
        )

    for match in _FENCED_TOOL_REGEX.finditer(payload):
        parsed = _parse_tool_json(match.group(1))
        if parsed is not None:
            append_tool(parsed["tool"], parsed["parameters"])

    if not tool_calls:
        for match in _BARE_TOOL_REGEX.finditer(payload):
            append_tool(match.group("tool"), match.group("params"))

    if tool_calls:
        return None, tool_calls, "tool_calls"
    return None


def _parse_tool_json(raw: str) -> dict[str, object] | None:
    cleaned = raw.strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        opens = cleaned.count("{")
        closes = cleaned.count("}")
        if opens > closes:
            try:
                parsed = json.loads(cleaned + ("}" * (opens - closes)))
            except json.JSONDecodeError:
                return None
        else:
            return None

    if not isinstance(parsed, dict):
        return None
    if isinstance(parsed.get("tool"), str):
        return {
            "tool": parsed["tool"],
            "parameters": parsed.get("parameters", {}),
        }
    if isinstance(parsed.get("name"), str):
        return {
            "tool": parsed["name"],
            "parameters": parsed.get("arguments", {}),
        }
    return None


def _parse_named_tool_payload(payload: object) -> tuple[str, object]:
    if not isinstance(payload, dict):
        raise RuntimeError("tool call payload must be an object")
    name = str(payload.get("name", "")).strip()
    if not name:
        raise RuntimeError("tool call name must be a non-empty string")
    arguments = payload.get("arguments", {})
    if not isinstance(arguments, dict):
        raise RuntimeError("tool_call.arguments must be an object")
    return name, arguments


def _extract_json_container_fragment(raw: str) -> str | None:
    cleaned = raw.strip()
    if not cleaned:
        return None

    fenced_match = re.search(r"```(?:json)?\s*(?P<body>[\s\S]*?)```", cleaned, flags=re.IGNORECASE)
    if fenced_match is not None:
        candidate = _extract_json_container_fragment(fenced_match.group("body"))
        if candidate is not None:
            return candidate

    # Balanced-bracket scan: the previous find/rfind approach grabbed the
    # outermost brackets regardless of nesting or string content, so trailing
    # prose like `[{...}] note: see [docs]` would extend the candidate into
    # `[docs]` and produce invalid JSON. Scan forward, tracking depth and
    # respecting JSON string boundaries so brackets inside "..." are ignored.
    for opening, closing in (("[", "]"), ("{", "}")):
        candidate = _balanced_bracket_substring(cleaned, opening, closing)
        if candidate:
            return candidate.strip() or None
    return None


def _balanced_bracket_substring(text: str, open_char: str, close_char: str) -> str | None:
    start = text.find(open_char)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "[" or ch == "{":
            depth += 1
        elif ch == "]" or ch == "}":
            depth -= 1
            if depth == 0:
                # Outermost container closed; succeed only if it matches the
                # expected closer (otherwise the input had {} mixed where we
                # wanted [], or vice versa — caller falls through to the other
                # pair / returns None).
                if ch == close_char:
                    return text[start : index + 1]
                return None
    return None


def _normalize_tool_choice(tool_choice: object) -> _NormalizedToolChoice:
    if tool_choice is None or tool_choice == "auto":
        return _NormalizedToolChoice(mode="auto")
    if isinstance(tool_choice, str):
        if tool_choice in {"required", "none"}:
            return _NormalizedToolChoice(mode=tool_choice)
        raise RuntimeError(
            "tool_choice must be one of: auto, required, none, or a function selector"
        )
    if not isinstance(tool_choice, dict):
        raise RuntimeError(
            "tool_choice must be a string, a function selector object, or None"
        )
    if tool_choice.get("type") != "function":
        raise RuntimeError("tool_choice object must have type='function'")
    function = tool_choice.get("function")
    if isinstance(function, dict):
        name = function.get("name")
    else:
        name = tool_choice.get("name")
    function_name = str(name or "").strip()
    if not function_name:
        raise RuntimeError("tool_choice function selector must include a non-empty name")
    return _NormalizedToolChoice(mode="function", function_name=function_name)


def _select_tools_for_choice(
    tools: list[dict[str, object]],
    tool_choice: _NormalizedToolChoice,
) -> list[dict[str, object]]:
    if tool_choice.mode == "none":
        return []
    if tool_choice.mode == "required" and not tools:
        raise RuntimeError("tool_choice='required' requires at least one tool")
    if tool_choice.mode != "function":
        return list(tools)

    selected: list[dict[str, object]] = []
    for tool in tools:
        function = _tool_function(tool)
        if function is None:
            continue
        if str(function.get("name", "")).strip() == tool_choice.function_name:
            selected.append(tool)
    if not selected:
        raise RuntimeError(
            f"tool_choice selected function '{tool_choice.function_name}', but it is not present in tools"
        )
    return selected


def _tool_choice_prompt_guidance(tool_choice: _NormalizedToolChoice) -> str:
    if tool_choice.mode == "auto":
        return ""
    if tool_choice.mode == "none":
        return (
            "Tool choice for this response is fixed to none.\n"
            "You may output one or more <think> blocks.\n"
            "Your terminal block must be <final_answer>.\n"
            "Do not output <tool_calls> in this response."
        )
    if tool_choice.mode == "required":
        return (
            "Tool choice for this response is required.\n"
            "You may output one or more <think> blocks.\n"
            "Your terminal block must be <tool_calls>[...]</tool_calls>.\n"
            "Do not output <final_answer> in this response."
        )
    return (
        f"Tool choice for this response is fixed to function '{tool_choice.function_name}'.\n"
        "You may output one or more <think> blocks.\n"
        "Your terminal block must be <tool_calls>[...]</tool_calls>.\n"
        f"Every tool call in <tool_calls> must use the function name '{tool_choice.function_name}'.\n"
        "Do not output <final_answer> in this response."
    )


def _tool_choice_user_suffix(tool_choice: _NormalizedToolChoice) -> str:
    if tool_choice.mode == "auto":
        return ""
    if tool_choice.mode == "none":
        return "For this response, tool_choice is none. End with <final_answer>."
    if tool_choice.mode == "required":
        return (
            "For this response, tool_choice is required. "
            "End with <tool_calls>[...]</tool_calls>, not <final_answer>."
        )
    return (
        "For this response, tool_choice requires calling "
        f"'{tool_choice.function_name}'. Do not call any other function."
    )


def _build_tagged_prompt(
    tools: list[dict[str, object]],
    *,
    tool_choice: object,
) -> str:
    normalized_choice = _normalize_tool_choice(tool_choice)
    effective_tools = _select_tools_for_choice(tools, normalized_choice)
    prompt = TAGGED_TOOL_PROMPT_PARALLEL
    choice_guidance = _tool_choice_prompt_guidance(normalized_choice)
    if choice_guidance:
        prompt += "\n\n## Tool choice\n\n" + choice_guidance + "\n"
    tool_text = _format_tools_for_prompt(effective_tools)
    if not tool_text:
        return prompt
    return prompt + "\n\n---\n\n## Available tools\n\n" + tool_text + "\n"


def _format_tools_for_prompt(tools: list[dict[str, object]]) -> str:
    if not tools:
        return ""

    lines: list[str] = []
    for tool in tools:
        function = _tool_function(tool)
        if function is None:
            continue
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        description = str(function.get("description") or function.get("summary") or "")
        params = function.get("parameters") or function.get("input_schema") or {}
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = {}
        props_raw = params.get("properties") if isinstance(params, dict) else None
        props = props_raw if isinstance(props_raw, dict) else {}
        required = set(params.get("required") or []) if isinstance(params, dict) else set()
        args_desc = ", ".join(
            f"{key}: {value.get('type', 'any')}" + (" (required)" if key in required else "")
            for key, value in props.items()
            if isinstance(value, dict)
        )
        suffix = "..." if len(description) > 200 else ""
        lines.append(f"- {name}({args_desc}): {description[:200]}{suffix}".rstrip())
    return "\n".join(lines)


def _normalize_tool_calls(tool_calls: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for item in tool_calls:
        if not isinstance(item, dict):
            continue
        function = item.get("function") or {}
        if not isinstance(function, dict):
            continue
        raw_args = function.get("arguments", {})
        if isinstance(raw_args, str):
            try:
                arguments = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                arguments = {"raw": raw_args}
        elif isinstance(raw_args, dict):
            arguments = raw_args
        else:
            arguments = {"raw": str(raw_args)}
        normalized.append(
            {
                "id": str(item.get("id") or ""),
                "name": str(function.get("name") or ""),
                "arguments": arguments,
            }
        )
    return normalized


def _assistant_tool_block(tool_calls: list[dict[str, object]]) -> str:
    payload = [
        {"name": str(tool_call.get("name") or ""), "arguments": tool_call.get("arguments", {})}
        for tool_call in tool_calls
        if str(tool_call.get("name") or "").strip()
    ]
    return "<tool_calls>" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "</tool_calls>"


def _tool_result_followup_for_choice(*, tool_choice: object) -> str:
    guidance = (
        "Now output exactly one response using only the tagged protocol:\n"
        "- optional <think>...</think>\n"
        "- then exactly one <tool_calls>[...]</tool_calls> or <final_answer>...</final_answer>\n"
        "Do not output Observation.\n"
        "Do not output <tool_result>.\n"
        "Do not output tool results.\n"
        "Do not output a second terminal block.\n"
        "Stop immediately after </tool_calls> or </final_answer>.\n"
        "Inside <tool_calls>, the content must be valid JSON.\n"
        "If only one tool is needed, still use a JSON array with one item.\n"
        "If a string value contains quotes, backslashes, or newlines, escape them exactly as JSON requires."
    )
    choice_suffix = _tool_choice_user_suffix(_normalize_tool_choice(tool_choice))
    if not choice_suffix:
        return guidance
    return guidance + "\n" + choice_suffix


def _validate_parsed_output(
    *,
    content: str | None,
    tool_calls: list[dict[str, object]],
    available_tools: list[dict[str, object]],
    tool_choice: _NormalizedToolChoice,
) -> None:
    allowed_names = _allowed_tool_names(available_tools)
    if tool_calls:
        if tool_choice.mode == "none":
            raise RuntimeError("tool_choice='none' does not allow tool calls")
        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            if not isinstance(function, dict):
                continue
            name = str(function.get("name", "")).strip()
            if tool_choice.mode == "function" and name != tool_choice.function_name:
                raise RuntimeError(
                    f"tool_choice requires function '{tool_choice.function_name}', got '{name}'"
                )
            if allowed_names and name not in allowed_names:
                raise RuntimeError(f"tool call '{name}' is not present in available tools")
        return
    if tool_choice.mode in {"required", "function"}:
        raise RuntimeError("tool_choice requires a tool call, but the model returned a final answer")
    _ = content


def _allowed_tool_names(tools: list[dict[str, object]]) -> set[str]:
    allowed: set[str] = set()
    for tool in tools:
        function = _tool_function(tool)
        if function is None:
            continue
        name = str(function.get("name", "")).strip()
        if name:
            allowed.add(name)
    return allowed


def _tool_function(tool: dict[str, object]) -> dict[str, object] | None:
    if not isinstance(tool, dict):
        return None

    tool_type = str(tool.get("type", "")).strip()
    if tool_type == "function":
        function = tool.get("function")
        if isinstance(function, dict):
            return function
        if any(key in tool for key in ("name", "description", "parameters", "input_schema")):
            return tool
        return None

    builtin_function = _builtin_tool_function(tool)
    if builtin_function is not None:
        return builtin_function

    if any(key in tool for key in ("name", "description", "parameters", "input_schema")):
        return tool
    return None


def _builtin_tool_function(tool: dict[str, object]) -> dict[str, object] | None:
    tool_type = str(tool.get("type", "")).strip()
    canonical = _canonical_builtin_tool_name(tool_type)
    if canonical is None:
        return None
    template = dict(_BUILTIN_TOOL_FUNCTIONS[canonical])
    description = str(tool.get("description") or template.get("description") or "").strip()
    parameters = tool.get("parameters") or tool.get("input_schema") or template.get("parameters") or {}
    template["description"] = description or str(template.get("description") or "")
    template["parameters"] = parameters
    return template


def _rewrite_tool_call_names(
    tool_calls: list[dict[str, object]],
    available_tools: list[dict[str, object]],
) -> None:
    for tool_call in tool_calls:
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name", "")).strip()
        if not name:
            continue
        resolved = _resolve_available_tool_name(name, available_tools)
        if resolved is not None:
            function["name"] = resolved


def _resolve_available_tool_name(
    name: str,
    available_tools: list[dict[str, object]],
) -> str | None:
    allowed_names = _allowed_tool_names(available_tools)
    if not allowed_names:
        return None
    if name in allowed_names:
        return name

    lowered_allowed = {candidate.lower(): candidate for candidate in allowed_names}
    direct = lowered_allowed.get(name.lower())
    if direct is not None:
        return direct

    canonical_builtin = _canonical_builtin_tool_name(name)
    if canonical_builtin is not None:
        resolved = lowered_allowed.get(canonical_builtin.lower())
        if resolved is not None:
            return resolved

    for canonical, aliases in _TOOL_NAME_ALIAS_GROUPS:
        normalized_aliases = {alias.lower() for alias in aliases | {canonical}}
        if name.lower() not in normalized_aliases:
            continue
        for candidate in (canonical, *sorted(aliases)):
            resolved = lowered_allowed.get(candidate.lower())
            if resolved is not None:
                return resolved
    return None


def _canonical_builtin_tool_name(name: str) -> str | None:
    lowered = str(name or "").strip().lower()
    if not lowered:
        return None
    direct = _BUILTIN_TOOL_NAME_ALIASES.get(lowered)
    if direct is not None:
        return direct
    if lowered.startswith("web_search"):
        return "web_search"
    if lowered.startswith("web_fetch"):
        return "web_fetch"
    return None


def _sanitize_plain_response(payload: str) -> str:
    return payload.strip()


def _extract_only_think_markup(payload: str) -> str | None:
    cleaned = payload.strip()
    if not cleaned:
        return ""
    matches = list(_THINK_BLOCK_REGEX.finditer(cleaned))
    if not matches:
        return None
    remainder = _THINK_BLOCK_REGEX.sub("", cleaned).strip()
    if remainder:
        return None
    return "".join(match.group(0) for match in matches)


def _extract_reasoning_markup(payload: str) -> str | None:
    cleaned = payload.strip()
    if not cleaned:
        return ""
    pure_think = _extract_only_think_markup(cleaned)
    if pure_think is not None:
        return pure_think

    parts: list[str] = []
    cursor = 0
    matched_any = False
    for match in _THINK_BLOCK_REGEX.finditer(cleaned):
        matched_any = True
        prefix = _strip_loose_protocol_tags(cleaned[cursor:match.start()]).strip()
        if prefix:
            if "<" in prefix or ">" in prefix:
                return None
            parts.append(f"<think>{prefix}</think>")
        parts.append(match.group(0))
        cursor = match.end()
    suffix = _strip_loose_protocol_tags(cleaned[cursor:]).strip()
    if suffix:
        if "<" in suffix or ">" in suffix:
            return None
        parts.append(f"<think>{suffix}</think>")
    if matched_any or parts:
        return "".join(parts)
    normalized_cleaned = _strip_loose_protocol_tags(cleaned).strip()
    if not normalized_cleaned:
        return ""
    if "<" in normalized_cleaned or ">" in normalized_cleaned:
        return None
    return f"<think>{normalized_cleaned}</think>"


def _strip_loose_protocol_tags(payload: str) -> str:
    return re.sub(
        r"</?(?:think|tool_call|tool_calls|final_answer)\b[^>]*>",
        "",
        payload,
        flags=re.IGNORECASE,
    )


def _remove_matched_ranges(payload: str, ranges: list[tuple[int, int]]) -> str:
    if not ranges:
        return payload
    result: list[str] = []
    cursor = 0
    for start, end in ranges:
        result.append(payload[cursor:start])
        cursor = end
    result.append(payload[cursor:])
    return "".join(result)


def _contains_protocol_markup(payload: str) -> bool:
    return bool(_PROTOCOL_MARKER_REGEX.search(payload))


def _extract_tool_attr(attrs: str, name: str) -> str | None:
    match = re.search(rf'{name}\s*=\s*["\']([^"\']+)["\']', attrs, flags=re.IGNORECASE)
    if match is None:
        return None
    value = match.group(1).strip()
    return value or None


def _normalize_xml_tool_body(body: str) -> str:
    cleaned = body.strip() or "{}"
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return _normalize_jsonish(cleaned)
    if not isinstance(parsed, dict):
        # Body decoded to a non-object JSON value (a bare string, array, or
        # number). Treat it as raw arguments rather than rejecting the whole
        # tool call — the model still clearly intended a call.
        return _normalize_jsonish(cleaned)
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def _normalize_jsonish(value: object) -> str:
    if isinstance(value, str):
        try:
            json.loads(value)
            return value
        except json.JSONDecodeError:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
