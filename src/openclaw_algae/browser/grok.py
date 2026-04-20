from __future__ import annotations

import time
from pathlib import Path

from openclaw_algae.browser.common import (
    build_cookie_string,
    prepare_browser_state_dir,
    require_sync_playwright,
)


def capture_grok_browser_credentials(*, state_dir: Path) -> dict[str, str]:
    sync_playwright = require_sync_playwright()
    browser_state_dir = prepare_browser_state_dir(state_dir, "grok")

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            str(browser_state_dir),
            headless=False,
        )
        try:
            page = context.pages[0] if context.pages else context.new_page()
            page.goto("https://grok.com", wait_until="domcontentloaded")
            user_agent = page.evaluate("() => navigator.userAgent")

            deadline = time.time() + 300
            while time.time() < deadline:
                cookies = context.cookies(["https://grok.com"])
                cookie_string = build_cookie_string(cookies)
                cookie_names = {str(item["name"]).lower() for item in cookies}
                has_auth_cookie = any("sso" in name for name in cookie_names)

                has_chat_input = False
                try:
                    has_chat_input = bool(
                        page.evaluate(
                            """
                            () => document.querySelector(
                              'textarea, [contenteditable="true"], div[role="textbox"]'
                            ) !== null
                            """
                        )
                    )
                except Exception:
                    has_chat_input = False

                if cookie_string and (has_auth_cookie or (len(cookies) > 1 and has_chat_input)):
                    return {
                        "cookie": cookie_string,
                        "user_agent": user_agent,
                    }

                time.sleep(2)
        finally:
            context.close()

    raise RuntimeError("Timed out waiting for Grok browser login to complete.")
