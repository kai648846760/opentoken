from __future__ import annotations

import re
import time
from pathlib import Path

from openclaw_algae.browser.common import (
    build_cookie_string,
    prepare_browser_state_dir,
    require_sync_playwright,
)

_CLAUDE_SESSION_PREFIXES = ("sk-ant-sid01-", "sk-ant-sid02-")


def capture_claude_browser_credentials(*, state_dir: Path) -> dict[str, object]:
    sync_playwright = require_sync_playwright()
    browser_state_dir = prepare_browser_state_dir(state_dir, "claude")

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            str(browser_state_dir),
            headless=False,
        )
        try:
            page = context.pages[0] if context.pages else context.new_page()
            captured_session_key: dict[str, str] = {"value": ""}

            def on_request(request) -> None:
                if "claude.ai" not in request.url:
                    return
                cookie = request.headers.get("cookie", "")
                if not cookie:
                    return
                match = re.search(r"sessionKey=([^;]+)", cookie)
                if match and match.group(1).startswith(_CLAUDE_SESSION_PREFIXES):
                    captured_session_key["value"] = match.group(1)

            page.on("request", on_request)
            page.goto("https://claude.ai/", wait_until="domcontentloaded")
            user_agent = page.evaluate("() => navigator.userAgent")

            deadline = time.time() + 300
            while time.time() < deadline:
                cookies = context.cookies(["https://claude.ai", "https://www.claude.ai"])
                cookie_string = build_cookie_string(cookies)

                session_key = captured_session_key["value"]
                if not session_key:
                    session_cookie = next(
                        (
                            item
                            for item in cookies
                            if item["name"] == "sessionKey"
                            or str(item["value"]).startswith(_CLAUDE_SESSION_PREFIXES)
                        ),
                        None,
                    )
                    if session_cookie is not None:
                        session_key = str(session_cookie["value"])

                if cookie_string and session_key.startswith(_CLAUDE_SESSION_PREFIXES):
                    return {
                        "cookie": cookie_string,
                        "user_agent": user_agent,
                        "metadata": {"session_key": session_key},
                    }

                time.sleep(2)
        finally:
            context.close()

    raise RuntimeError("Timed out waiting for Claude browser login to complete.")
