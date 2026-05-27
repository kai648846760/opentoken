#!/usr/bin/env python3
"""Fast end-to-end smoke test across logged-in providers.

For each provider that has saved credentials, runs:
  1. GET  /v1/models                   (sanity)
  2. POST /v1/chat/completions         (non-stream, one short prompt)
  3. POST /v1/chat/completions         (stream=True, same short prompt)

Prints a per-provider pass/fail table and writes a JSON report.

Usage:
    opentoken start &       # bring the gateway up first
    uv run python scripts/live_provider_smoke.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

from opentoken.config.app_config import load_or_create_app_config
from opentoken.config.paths import resolve_app_config_path, resolve_providers_dir
from opentoken.models.discovery import load_model_catalog
from opentoken.storage.provider_store import list_provider_credentials


SMOKE_PROMPT = "Say 'pong' once and stop."


def _pick_model_for_provider(provider: str) -> str | None:
    catalog = load_model_catalog(providers_dir=resolve_providers_dir())
    for entry in catalog:
        # load_model_catalog returns ModelCatalogEntry objects, not dicts.
        ident = getattr(entry, "id", "") or ""
        owned_by = (getattr(entry, "provider", "") or "").strip()
        if owned_by == "opentoken" and ident.startswith(f"algae/{provider}/"):
            return ident
    return None


def _ping_models(client: httpx.Client) -> tuple[bool, str]:
    try:
        response = client.get("/v1/models")
    except Exception as exc:
        return False, f"connection: {exc}"
    if response.status_code != 200:
        return False, f"http {response.status_code}: {response.text[:80]}"
    payload = response.json()
    data = payload.get("data") or []
    return True, f"{len(data)} models listed"


def _ping_chat_completion(client: httpx.Client, model: str) -> tuple[bool, str]:
    try:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": SMOKE_PROMPT}],
                "stream": False,
            },
            timeout=120.0,
        )
    except Exception as exc:
        return False, f"connection: {exc}"
    if response.status_code != 200:
        return False, f"http {response.status_code}: {response.text[:120]}"
    payload = response.json()
    choices = payload.get("choices") or []
    if not choices:
        return False, "no choices in response"
    content = choices[0].get("message", {}).get("content") or ""
    if not content.strip():
        return False, "empty content"
    return True, content.strip()[:80]


def _ping_chat_completion_stream(client: httpx.Client, model: str) -> tuple[bool, str]:
    try:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": SMOKE_PROMPT}],
                "stream": True,
            },
            timeout=120.0,
        ) as response:
            if response.status_code != 200:
                return False, f"http {response.status_code}"
            content_pieces: list[str] = []
            for raw_line in response.iter_lines():
                line = (raw_line or "").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if isinstance(piece, str) and piece:
                    content_pieces.append(piece)
    except Exception as exc:
        return False, f"connection: {exc}"

    text = "".join(content_pieces).strip()
    if not text:
        return False, "no content yielded"
    return True, text[:80]


def _ping_responses(client: httpx.Client, model: str) -> tuple[bool, str]:
    try:
        response = client.post(
            "/v1/responses",
            json={
                "model": model,
                "input": SMOKE_PROMPT,
                "stream": False,
            },
            timeout=120.0,
        )
    except Exception as exc:
        return False, f"connection: {exc}"
    if response.status_code != 200:
        return False, f"http {response.status_code}: {response.text[:120]}"
    payload = response.json()
    output = payload.get("output") or []
    texts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        for piece in item.get("content") or []:
            if isinstance(piece, dict) and isinstance(piece.get("text"), str):
                texts.append(piece["text"])
    joined = "".join(texts).strip()
    if not joined:
        return False, "empty responses output"
    return True, joined[:80]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", action="append", help="Provider key (repeatable)")
    parser.add_argument("--out", default="live_provider_smoke_report.json")
    parser.add_argument("--no-stream", action="store_true", help="Skip the streaming check")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    config = load_or_create_app_config(resolve_app_config_path())
    base_url = f"http://{config['host']}:{config['port']}"
    api_key = str(config["api_key"])
    records = list_provider_credentials(resolve_providers_dir())
    logged_in = {record.provider for record in records}
    requested = set(args.provider) if args.provider else logged_in
    targets = sorted(requested & logged_in)
    if not targets:
        print(f"No logged-in providers to smoke. Logged in: {sorted(logged_in)}", file=sys.stderr)
        return 1

    print(f"Smoke testing against {base_url} -> {targets}", file=sys.stderr)

    results: list[dict[str, object]] = []
    with httpx.Client(
        base_url=base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30.0,
        trust_env=False,
    ) as client:
        # /v1/models is provider-agnostic; run it once.
        models_ok, models_detail = _ping_models(client)
        print(f"[ALL] /v1/models -> {'pass' if models_ok else 'FAIL'} ({models_detail})", file=sys.stderr)

        for provider in targets:
            model = _pick_model_for_provider(provider)
            if model is None:
                results.append({
                    "provider": provider,
                    "status": "skip",
                    "reason": "no model in catalog",
                })
                print(f"[{provider}] no model in catalog — skipped", file=sys.stderr)
                continue

            started = time.perf_counter()
            non_stream_ok, non_stream_detail = _ping_chat_completion(client, model)
            non_stream_latency = (time.perf_counter() - started) * 1000

            if args.no_stream:
                stream_ok, stream_detail, stream_latency = True, "skipped", 0.0
            else:
                started = time.perf_counter()
                stream_ok, stream_detail = _ping_chat_completion_stream(client, model)
                stream_latency = (time.perf_counter() - started) * 1000

            started = time.perf_counter()
            responses_ok, responses_detail = _ping_responses(client, model)
            responses_latency = (time.perf_counter() - started) * 1000

            overall = non_stream_ok and stream_ok and responses_ok
            results.append({
                "provider": provider,
                "model": model,
                "non_stream": {
                    "ok": non_stream_ok,
                    "detail": non_stream_detail,
                    "latency_ms": round(non_stream_latency, 1),
                },
                "stream": {
                    "ok": stream_ok,
                    "detail": stream_detail,
                    "latency_ms": round(stream_latency, 1),
                },
                "responses": {
                    "ok": responses_ok,
                    "detail": responses_detail,
                    "latency_ms": round(responses_latency, 1),
                },
                "status": "pass" if overall else "fail",
            })
            print(
                f"[{provider}] chat-nonstream={'pass' if non_stream_ok else 'FAIL'} "
                f"({non_stream_latency:.0f}ms) | "
                f"chat-stream={'pass' if stream_ok else 'FAIL'} "
                f"({stream_latency:.0f}ms) | "
                f"responses={'pass' if responses_ok else 'FAIL'} "
                f"({responses_latency:.0f}ms)",
                file=sys.stderr,
            )

    report = {
        "base_url": base_url,
        "targets": targets,
        "models_endpoint": {"ok": models_ok, "detail": models_detail},
        "providers": results,
        "passes": sum(1 for r in results if r.get("status") == "pass"),
        "fails": sum(1 for r in results if r.get("status") == "fail"),
        "skips": sum(1 for r in results if r.get("status") == "skip"),
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary": {k: v for k, v in report.items() if k != "providers"}}, indent=2))
    return 0 if report["fails"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
