from __future__ import annotations

import time
from pathlib import Path

from openclaw_algae.browser.common import (
    build_cookie_string,
    prepare_browser_state_dir,
    require_sync_playwright,
)

_GEMINI_COOKIE_NAMES = {"SID", "__Secure-1PSID", "__Secure-3PSID"}


def capture_gemini_browser_credentials(*, state_dir: Path) -> dict[str, str]:
    sync_playwright = require_sync_playwright()
    browser_state_dir = prepare_browser_state_dir(state_dir, "gemini")

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            str(browser_state_dir),
            headless=False,
        )
        try:
            page = context.pages[0] if context.pages else context.new_page()
            page.goto("https://gemini.google.com/app", wait_until="domcontentloaded")
            user_agent = page.evaluate("() => navigator.userAgent")

            deadline = time.time() + 300
            while time.time() < deadline:
                cookies = context.cookies(
                    [
                        "https://gemini.google.com",
                        "https://accounts.google.com",
                        "https://www.google.com",
                    ]
                )
                cookie_string = build_cookie_string(cookies)
                if cookie_string and any(item["name"] in _GEMINI_COOKIE_NAMES for item in cookies):
                    return {
                        "cookie": cookie_string,
                        "user_agent": user_agent,
                    }
                time.sleep(2)
        finally:
            context.close()

    raise RuntimeError("Timed out waiting for Gemini browser login to complete.")
