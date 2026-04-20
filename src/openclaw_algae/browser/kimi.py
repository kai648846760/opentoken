from __future__ import annotations

import time
from pathlib import Path

from openclaw_algae.browser.common import (
    build_cookie_string,
    prepare_browser_state_dir,
    require_sync_playwright,
)


def capture_kimi_browser_credentials(*, state_dir: Path) -> dict[str, str]:
    """Capture Kimi credentials by waiting for user to close the browser.

    Kimi generates a guest 'kimi-auth' cookie on first page load, so we
    cannot detect login automatically. Instead we wait for the user to
    complete login and close the browser manually.
    """
    sync_playwright = require_sync_playwright()
    browser_state_dir = prepare_browser_state_dir(state_dir, 'kimi')

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            str(browser_state_dir),
            headless=False,
        )
        try:
            page = context.pages[0] if context.pages else context.new_page()
            page.goto('https://www.kimi.com/', wait_until='domcontentloaded')
            user_agent = page.evaluate('() => navigator.userAgent')

            print("Please login to Kimi and close the browser window when done.")
            print("Waiting for browser to close...")

            # Wait for browser to close (user closes the window)
            while True:
                try:
                    page.evaluate('() => document.title')
                    time.sleep(1)
                except Exception:
                    # Browser closed
                    break

            # Capture cookies after user closes
            cookies = context.cookies(['https://www.kimi.com', 'https://kimi.com'])
            cookie_string = build_cookie_string(cookies)
            if not cookie_string:
                raise RuntimeError("No cookies captured after Kimi login.")

            return {
                'cookie': cookie_string,
                'user_agent': user_agent,
            }
        finally:
            try:
                context.close()
            except Exception:
                pass

    raise RuntimeError('Failed to capture Kimi credentials.')
