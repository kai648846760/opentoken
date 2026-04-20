from __future__ import annotations

import json
import re
import time
from pathlib import Path

from openclaw_algae.browser.common import (
    build_cookie_string,
    prepare_browser_state_dir,
    require_sync_playwright,
)

_MIMO_TOKEN_MARKERS = ("token", "session", "auth", "user")


def capture_mimo_browser_credentials(*, state_dir: Path) -> dict[str, str]:
    sync_playwright = require_sync_playwright()
    browser_state_dir = prepare_browser_state_dir(state_dir, "mimo")

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            str(browser_state_dir),
            headless=False,
        )
        try:
            page = context.pages[0] if context.pages else context.new_page()
            captured_token: dict[str, str] = {"value": ""}

            def on_request(request) -> None:
                if "xiaomimimo.com" not in request.url:
                    return
                auth = request.headers.get("authorization", "")
                if auth.lower().startswith("bearer "):
                    captured_token["value"] = auth[7:].strip()
                    return
                cookie = request.headers.get("cookie", "")
                if not cookie:
                    return
                match = re.search(r"(?:token|session|auth|user)[^=]*=([^;]+)", cookie, re.IGNORECASE)
                if match:
                    captured_token["value"] = match.group(1)

            def on_response(response) -> None:
                if "xiaomimimo.com" not in response.url or not response.ok:
                    return
                content_type = response.headers.get("content-type", "")
                if "application/json" not in content_type:
                    return
                try:
                    payload = response.json()
                except Exception:
                    return
                token = _extract_storage_token(payload)
                if token:
                    captured_token["value"] = token

            page.on("request", on_request)
            page.on("response", on_response)
            page.goto("https://aistudio.xiaomimimo.com/#/", wait_until="domcontentloaded")
            user_agent = page.evaluate("() => navigator.userAgent")

            deadline = time.time() + 300
            while time.time() < deadline:
                cookies = context.cookies(
                    [
                        "https://aistudio.xiaomimimo.com",
                        "https://xiaomimimo.com",
                    ]
                )
                cookie_string = build_cookie_string(cookies)
                token_cookie = next(
                    (
                        item
                        for item in cookies
                        if any(marker in str(item["name"]).lower() for marker in _MIMO_TOKEN_MARKERS)
                    ),
                    None,
                )

                if not captured_token["value"]:
                    try:
                        storage_entries = page.evaluate(
                            """
                            () => {
                              const data = {};
                              for (let i = 0; i < window.localStorage.length; i += 1) {
                                const key = window.localStorage.key(i);
                                if (key) {
                                  data[key] = window.localStorage.getItem(key) || "";
                                }
                              }
                              return data;
                            }
                            """
                        )
                    except Exception:
                        storage_entries = {}

                    if isinstance(storage_entries, dict):
                        token = _extract_storage_token(storage_entries)
                        if token:
                            captured_token["value"] = token

                if cookie_string and len(cookies) > 1 and (token_cookie is not None or captured_token["value"]):
                    return {
                        "cookie": cookie_string,
                        "user_agent": user_agent,
                    }

                time.sleep(2)
        finally:
            context.close()

    raise RuntimeError("Timed out waiting for Xiaomi MiMo browser login to complete.")


def _extract_storage_token(payload: object) -> str:
    if isinstance(payload, str):
        if len(payload.strip()) > 10:
            return payload.strip()
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return ""
        return _extract_storage_token(parsed)

    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized = str(key).lower()
            if any(marker in normalized for marker in _MIMO_TOKEN_MARKERS):
                extracted = _extract_storage_token(value)
                if extracted:
                    return extracted
        for value in payload.values():
            extracted = _extract_storage_token(value)
            if extracted:
                return extracted
        return ""

    if isinstance(payload, list):
        for item in payload:
            extracted = _extract_storage_token(item)
            if extracted:
                return extracted
        return ""

    return ""
