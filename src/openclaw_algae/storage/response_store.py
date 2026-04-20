from __future__ import annotations

import copy
import json
from pathlib import Path


_DEFAULT_STORE: dict[str, object] = {
    "version": 1,
    "responses": {},
}


def load_response_messages(state_dir: Path, response_id: str) -> list[dict[str, object]] | None:
    responses = _load_store(_resolve_response_store_path(state_dir)).get("responses", {})
    if not isinstance(responses, dict):
        return None
    entry = responses.get(response_id)
    if not isinstance(entry, dict):
        return None
    messages = entry.get("messages")
    if not isinstance(messages, list):
        return None
    return copy.deepcopy([message for message in messages if isinstance(message, dict)])


def save_response_messages(
    state_dir: Path,
    *,
    response_id: str,
    model: str,
    messages: list[dict[str, object]],
) -> None:
    path = _resolve_response_store_path(state_dir)
    store = _load_store(path)
    responses = store.setdefault("responses", {})
    if not isinstance(responses, dict):
        responses = {}
        store["responses"] = responses
    responses[response_id] = {
        "model": model,
        "messages": copy.deepcopy(messages),
    }
    _save_store(path, store)


def _resolve_response_store_path(state_dir: Path) -> Path:
    return state_dir / "responses.json"


def _load_store(path: Path) -> dict[str, object]:
    if not path.exists():
        return copy.deepcopy(_DEFAULT_STORE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return copy.deepcopy(_DEFAULT_STORE)
    if not isinstance(payload, dict):
        return copy.deepcopy(_DEFAULT_STORE)
    store = copy.deepcopy(_DEFAULT_STORE)
    store.update(payload)
    return store


def _save_store(path: Path, store: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
