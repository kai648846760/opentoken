from __future__ import annotations

import copy
import json
import time
from collections import OrderedDict
from pathlib import Path

from opentoken.storage._atomic import file_lock, write_json_atomic


_DEFAULT_STORE: dict[str, object] = {
    "version": 1,
    "responses": {},
}

# Default retention: 7 days OR 1024 entries, whichever fires first. previous_response_id
# is meant for short-lived conversation context — the store used to grow forever, which
# eventually corrupted JSON on disk and made every load slower.
_DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60
_DEFAULT_MAX_ENTRIES = 1024


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
    ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    max_entries: int = _DEFAULT_MAX_ENTRIES,
) -> None:
    path = _resolve_response_store_path(state_dir)
    with file_lock(path):
        store = _load_store(path)
        responses_raw = store.get("responses", {})
        # Use an OrderedDict so we can evict LRU when over the cap. Re-inserting moves
        # to the end implicitly because we del-then-set below.
        if isinstance(responses_raw, dict):
            responses = OrderedDict(responses_raw)
        else:
            responses = OrderedDict()

        now = int(time.time())
        # Expire by age first.
        if ttl_seconds > 0:
            cutoff = now - ttl_seconds
            for stale_id in [
                key
                for key, entry in responses.items()
                if isinstance(entry, dict) and int(entry.get("updated_at", 0)) < cutoff
            ]:
                responses.pop(stale_id, None)

        responses.pop(response_id, None)
        responses[response_id] = {
            "model": model,
            "messages": copy.deepcopy(messages),
            "updated_at": now,
        }

        # Cap by count.
        while max_entries > 0 and len(responses) > max_entries:
            responses.popitem(last=False)

        store["responses"] = dict(responses)
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
    # sensitive=True (0600): this store holds the full conversation history
    # (user prompts + assistant responses) addressable by previous_response_id.
    # User prompts can contain secrets (API keys pasted into a question, PII,
    # work-in-progress code); 0644 would let any other local user on a shared
    # host read them.
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, store, sensitive=True)
