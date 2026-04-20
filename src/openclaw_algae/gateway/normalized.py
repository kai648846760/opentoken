import base64
import copy
import json

from pydantic import BaseModel, Field

from openclaw_algae.config.paths import resolve_state_dir
from openclaw_algae.storage.file_store import read_file_content


class NormalizedChatRequest(BaseModel):
    model: str
    messages: list[dict[str, object]] = Field(default_factory=list)
    temperature: float | None = None
    stream: bool = False
    tools: list[dict[str, object]] | None = None
    tool_choice: object = None


def normalize_chat_completions_request(payload: dict[str, object]) -> NormalizedChatRequest:
    _require_model(payload)
    normalized_payload = dict(payload)
    messages = payload.get("messages", [])
    if isinstance(messages, list):
        normalized_payload["messages"] = _resolve_message_attachments(messages)
    return NormalizedChatRequest.model_validate(normalized_payload)


def normalize_responses_request(payload: dict[str, object]) -> NormalizedChatRequest:
    model = _require_model(payload)
    messages: list[dict[str, object]] = []
    instructions = payload.get("instructions")
    if instructions is not None and str(instructions).strip():
        messages.append({"role": "system", "content": str(instructions).strip()})
    input_value = payload.get("input", "")
    if isinstance(input_value, str):
        messages.append({"role": "user", "content": input_value})
    elif isinstance(input_value, list):
        normalized_items = _normalize_response_input_items(input_value)
        if normalized_items is None:
            messages.append({"role": "user", "content": input_value})
        else:
            messages.extend(normalized_items)
    else:
        messages.append({"role": "user", "content": str(input_value)})
    return NormalizedChatRequest(
        model=model,
        messages=_resolve_message_attachments(messages),
        stream=bool(payload.get("stream", False)),
        tools=_normalize_tools(payload.get("tools")),
        tool_choice=payload.get("tool_choice"),
    )


def _normalize_tools(value: object) -> list[dict[str, object]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise RuntimeError("tools must be a list.")
    normalized: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            raise RuntimeError("Each tool must be an object.")
        normalized.append(item)
    return normalized


def _normalize_response_input_items(
    items: list[object],
) -> list[dict[str, object]] | None:
    normalized: list[dict[str, object]] = []
    for item in items:
        if not isinstance(item, dict):
            return None
        item_type = str(item.get("type", "")).strip()
        if item_type == "message":
            role = str(item.get("role", "user")).strip() or "user"
            if role == "developer":
                role = "system"
            normalized.append({"role": role, "content": item.get("content", "")})
            continue
        if item_type == "function_call":
            call_id = str(item.get("call_id") or item.get("id") or "").strip()
            if not call_id:
                raise RuntimeError("function_call.call_id is required.")
            name = str(item.get("name", "")).strip()
            if not name:
                raise RuntimeError("function_call.name is required.")
            arguments = item.get("arguments", "{}")
            normalized.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": _stringify_tool_arguments(arguments),
                            },
                        }
                    ],
                }
            )
            continue
        if item_type == "function_call_output":
            call_id = str(item.get("call_id") or "").strip()
            if not call_id:
                raise RuntimeError("function_call_output.call_id is required.")
            output = item.get("output", "")
            normalized.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": _stringify_tool_arguments(output),
                }
            )
            continue
        if "role" in item:
            normalized.append(item)
            continue
        return None
    return normalized if normalized else None


def _resolve_message_attachments(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    resolved: list[dict[str, object]] = []
    for message in messages:
        normalized_message = copy.deepcopy(message)
        normalized_message["content"] = _resolve_content_attachments(normalized_message.get("content"))
        resolved.append(normalized_message)
    return resolved


def _resolve_content_attachments(content: object) -> object:
    if not isinstance(content, list):
        return content
    resolved: list[object] = []
    for item in content:
        if not isinstance(item, dict):
            resolved.append(item)
            continue
        item_type = str(item.get("type", "")).strip()
        file_id = str(item.get("file_id", "")).strip()
        if not file_id:
            resolved.append(copy.deepcopy(item))
            continue
        if item_type in {"input_file", "file"}:
            resolved.append(_inject_uploaded_file(item, file_id=file_id, as_image=False))
            continue
        if item_type in {"input_image", "image"}:
            resolved.append(_inject_uploaded_file(item, file_id=file_id, as_image=True))
            continue
        resolved.append(copy.deepcopy(item))
    return resolved


def _inject_uploaded_file(
    item: dict[str, object],
    *,
    file_id: str,
    as_image: bool,
) -> dict[str, object]:
    loaded = read_file_content(resolve_state_dir(), file_id)
    if loaded is None:
        raise RuntimeError(f"Unknown file_id: {file_id}")
    metadata, content = loaded
    mime_type = str(metadata.get("mime_type", "application/octet-stream"))
    filename = str(metadata.get("filename", ""))
    data_uri = _to_data_uri(content, mime_type=mime_type)
    normalized = copy.deepcopy(item)
    normalized.setdefault("filename", filename)
    normalized.setdefault("mime_type", mime_type)
    if as_image:
        normalized["image_url"] = {"url": data_uri}
    else:
        normalized["file_data"] = data_uri
    return normalized


def _to_data_uri(content: bytes, *, mime_type: str) -> str:
    payload = base64.b64encode(content).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def _stringify_tool_arguments(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _require_model(payload: dict[str, object]) -> str:
    model = str(payload.get("model", "")).strip()
    if not model:
        raise RuntimeError("Model is required.")
    return model
