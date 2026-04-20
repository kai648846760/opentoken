from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import httpx

from openclaw_algae.config.paths import resolve_providers_dir
from openclaw_algae.models.discovery import load_model_catalog
from openclaw_algae.storage.provider_store import list_provider_credentials

DEFAULT_BASE_URL = "http://127.0.0.1:32117/v1"
DEFAULT_API_KEY = "test-e2e-key"
DEFAULT_PROMPT = "来一个3000字自我介绍"
DEFAULT_HEADERS = {"Authorization": f"Bearer {DEFAULT_API_KEY}"}
DEFAULT_READ_TIMEOUT = 20.0
DEFAULT_MAX_VISIBLE_CHUNKS = 12
DEFAULT_OBSERVATION_WINDOW_S = 2.5
PROVIDER_MIN_INTERVALS = {
    "deepseek": 2.0,
    "qwen-intl": 0.25,
    "qwen-cn": 0.25,
    "kimi": 3.0,
    "doubao": 0.5,
    "glm-cn": 3.0,
    "glm-intl": 0.5,
}


def iter_valid_models() -> list[str]:
    valid = {record.provider for record in list_provider_credentials(resolve_providers_dir()) if record.status == "valid"}
    return [entry.id for entry in load_model_catalog() if entry.id.split("/")[1] in valid]



def read_timeout_for(model: str) -> float:
    lowered = model.lower()
    if any(token in lowered for token in ("thinking", "reasoner", "-think")):
        return 55.0
    if lowered.startswith("algae/doubao/"):
        return 35.0
    if lowered.startswith("algae/glm-cn/"):
        return 30.0
    return DEFAULT_READ_TIMEOUT



def classify_probe_result(result: dict[str, Any]) -> str:
    visible = int(result.get("visible_chunks", 0) or 0)
    first_visible = result.get("first_visible_s")
    gap2 = result.get("gap_2_s")
    window5 = result.get("window_5_s")
    total_window = result.get("visible_window_s")
    if visible == 0:
        return "error" if result.get("error") else "no_visible"
    if visible == 1:
        return "single_chunk"
    if first_visible is not None and first_visible > 10 and (window5 or total_window or 0.0) < 0.25:
        return "burst_after_wait"
    if visible >= 5 and (window5 or 0.0) >= 0.45:
        return "incremental"
    if visible >= 3 and (total_window or 0.0) >= 0.35:
        return "coarse_incremental"
    if visible >= 3:
        return "burst"
    if gap2 is not None and gap2 >= 0.15:
        return "coarse_incremental"
    return "burst"



def analyze_stream_lines(
    lines: Iterable[str | bytes],
    *,
    mode: str,
    now_fn: Callable[[], float] | None = None,
    max_visible_chunks: int = DEFAULT_MAX_VISIBLE_CHUNKS,
    observation_window_s: float = DEFAULT_OBSERVATION_WINDOW_S,
) -> dict[str, Any]:
    del observation_window_s  # kept for call-site compatibility; analysis now drains to terminal event.
    clock = now_fn or time.perf_counter
    visible_times: list[float] = []
    preview_parts: list[str] = []
    raw_events = 0
    message_chunks = 0
    reasoning_chunks = 0
    error: str | None = None
    first_event_s: float | None = None
    done = False
    current_event: str | None = None

    for raw_line in lines:
        now = clock()
        line = _normalize_line(raw_line)
        if not line:
            continue
        raw_events += 1
        if first_event_s is None:
            first_event_s = now

        visible_piece = ""
        visible_kind = "message"

        if mode == "chat":
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ").strip()
            if data == "[DONE]":
                done = True
                break
            try:
                chunk = json.loads(data)
            except Exception:
                continue
            if isinstance(chunk, dict) and chunk.get("error"):
                error = json.dumps(chunk.get("error", {}), ensure_ascii=False)[:300]
                break
            try:
                visible_piece = str(chunk["choices"][0]["delta"].get("content", "") or "")
            except Exception:
                visible_piece = ""
        elif mode == "responses":
            if line.startswith("event: "):
                current_event = line.removeprefix("event: ").strip()
                continue
            if not line.startswith("data: "):
                continue
            try:
                payload = json.loads(line.removeprefix("data: ").strip())
            except Exception:
                payload = {}
            if current_event == "response.output_text.delta":
                visible_piece = str(payload.get("delta", "") or "")
                visible_kind = "message"
            elif current_event == "response.reasoning_text.delta":
                visible_piece = str(payload.get("delta", "") or "")
                visible_kind = "reasoning"
            elif current_event in {"response.failed", "error"}:
                error = _render_error_payload(payload)
                current_event = None
                break
            elif current_event == "response.completed":
                done = True
                current_event = None
                break
            current_event = None
        else:
            raise ValueError(f"unsupported mode: {mode}")

        if visible_piece and len(visible_times) < max_visible_chunks:
            visible_times.append(now)
            if len("".join(preview_parts)) < 240:
                preview_parts.append(visible_piece)
            if visible_kind == "reasoning":
                reasoning_chunks += 1
            else:
                message_chunks += 1

    first_visible = visible_times[0] if visible_times else None
    last_visible = visible_times[-1] if visible_times else None
    gap_2 = (visible_times[1] - visible_times[0]) if len(visible_times) >= 2 else None
    window_5 = (visible_times[4] - visible_times[0]) if len(visible_times) >= 5 else None
    return {
        "error": error,
        "visible_chunks": len(visible_times),
        "message_chunks": message_chunks,
        "reasoning_chunks": reasoning_chunks,
        "raw_events": raw_events,
        "first_event_s": first_event_s,
        "first_visible_s": first_visible,
        "last_visible_s": last_visible,
        "visible_window_s": (last_visible - first_visible) if first_visible is not None and last_visible is not None else None,
        "gap_2_s": gap_2,
        "window_5_s": window_5,
        "preview": "".join(preview_parts)[:240],
        "done": done,
    }



def probe_stream(
    client: httpx.Client,
    endpoint: str,
    payload: dict[str, Any],
    *,
    mode: str,
    max_visible_chunks: int = DEFAULT_MAX_VISIBLE_CHUNKS,
    observation_window_s: float = DEFAULT_OBSERVATION_WINDOW_S,
) -> dict[str, Any]:
    model = str(payload["model"])
    timeout = httpx.Timeout(connect=20.0, write=20.0, read=read_timeout_for(model), pool=20.0)
    start = time.perf_counter()
    status_code = 0
    try:
        with client.stream("POST", endpoint, json=payload, timeout=timeout) as response:
            status_code = response.status_code
            if response.status_code != 200:
                body = response.read().decode("utf-8", errors="replace")
                return {
                    "status_code": response.status_code,
                    "error": body[:300],
                    "visible_chunks": 0,
                    "message_chunks": 0,
                    "reasoning_chunks": 0,
                    "first_event_s": None,
                    "first_visible_s": None,
                    "visible_window_s": None,
                    "gap_2_s": None,
                    "window_5_s": None,
                    "preview": "",
                    "done": False,
                }
            result = analyze_stream_lines(
                response.iter_lines(),
                mode=mode,
                now_fn=lambda: time.perf_counter() - start,
                max_visible_chunks=max_visible_chunks,
                observation_window_s=observation_window_s,
            )
            result["status_code"] = status_code
            return result
    except Exception as exc:
        return {
            "status_code": status_code,
            "error": str(exc),
            "visible_chunks": 0,
            "message_chunks": 0,
            "reasoning_chunks": 0,
            "raw_events": 0,
            "first_event_s": None,
            "first_visible_s": None,
            "last_visible_s": None,
            "visible_window_s": None,
            "gap_2_s": None,
            "window_5_s": None,
            "preview": "",
            "done": False,
        }



def probe_chat(client: httpx.Client, model: str, *, prompt: str = DEFAULT_PROMPT) -> dict[str, Any]:
    return probe_stream(
        client,
        "/chat/completions",
        {
            "model": model,
            "stream": True,
            "messages": [{"role": "user", "content": prompt}],
        },
        mode="chat",
    )



def probe_responses(client: httpx.Client, model: str, *, prompt: str = DEFAULT_PROMPT) -> dict[str, Any]:
    return probe_stream(
        client,
        "/responses",
        {
            "model": model,
            "stream": True,
            "input": prompt,
        },
        mode="responses",
    )



def run_live_stream_regression(
    *,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    prompt: str = DEFAULT_PROMPT,
) -> list[dict[str, Any]]:
    client = httpx.Client(base_url=base_url, headers={"Authorization": f"Bearer {api_key}"}, trust_env=False)
    results: list[dict[str, Any]] = []
    last_finished: dict[str, float] = defaultdict(float)
    try:
        for model in iter_valid_models():
            provider = model.split("/")[1]
            min_interval = PROVIDER_MIN_INTERVALS.get(provider, 0.25)
            for endpoint_name, fn in (("chat/completions", probe_chat), ("responses", probe_responses)):
                elapsed = time.perf_counter() - last_finished[provider]
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
                started_at = time.perf_counter()
                result = fn(client, model, prompt=prompt)
                last_finished[provider] = time.perf_counter()
                result.update(
                    {
                        "provider": provider,
                        "model": model,
                        "endpoint": endpoint_name,
                        "elapsed_s": last_finished[provider] - started_at,
                    }
                )
                result["class"] = classify_probe_result(result)
                results.append(result)
        return results
    finally:
        client.close()



def write_regression_report(results: list[dict[str, Any]], *, out_dir: str | Path = "tmp") -> Path:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    out_path = path / f"live_batch_regression_v2_{time.strftime('%Y-%m-%d')}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path



def _normalize_line(raw_line: str | bytes) -> str:
    if isinstance(raw_line, bytes):
        return raw_line.decode("utf-8")
    return str(raw_line)



def _render_error_payload(payload: Any) -> str:
    if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
        return json.dumps(payload["error"], ensure_ascii=False)[:300]
    return json.dumps(payload, ensure_ascii=False)[:300]
