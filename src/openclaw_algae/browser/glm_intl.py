from __future__ import annotations

import time
from pathlib import Path

from openclaw_algae.browser.common import (
    build_cookie_string,
    prepare_browser_state_dir,
    require_sync_playwright,
)

_GLM_INTL_COOKIE_MARKERS = (
    "chatglm_refresh_token",
    "refresh_token",
    "chatglm_token",
    "auth_token",
    "access_token",
)


def _glm_intl_has_auth_cookie(cookie_names: set[str]) -> bool:
    normalized = {name.strip().lower() for name in cookie_names if name and name.strip()}
    return any(marker in normalized for marker in _GLM_INTL_COOKIE_MARKERS)


def capture_glm_intl_browser_credentials(*, state_dir: Path) -> dict[str, str]:
    sync_playwright = require_sync_playwright()
    browser_state_dir = prepare_browser_state_dir(state_dir, "glm-intl")

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            str(browser_state_dir),
            headless=False,
        )
        try:
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(
                "https://chat.z.ai/",
                wait_until="domcontentloaded",
                timeout=120000,
            )
            user_agent = page.evaluate("() => navigator.userAgent")

            deadline = time.time() + 600
            while time.time() < deadline:
                cookies = context.cookies(["https://chat.z.ai"])
                cookie_string = build_cookie_string(cookies)
                cookie_names = {str(item["name"]).lower() for item in cookies}
                has_auth_cookie = _glm_intl_has_auth_cookie(cookie_names)

                has_chat_input = False
                logged_in_url = False
                has_sign_in = False
                try:
                    page_state = page.evaluate(
                        """
                        () => ({
                          href: window.location.href,
                          has_input: document.querySelector(
                            'textarea, [contenteditable="true"], .chat-input, .message-input'
                          ) !== null,
                          has_sign_in: [...document.querySelectorAll('button, a, [role="button"]')].some((node) => {
                            const text = String(node.innerText || node.textContent || node.getAttribute('aria-label') || '')
                              .trim()
                              .toLowerCase();
                            return ['sign in', 'log in', 'login', '登录'].some((keyword) => text.includes(keyword));
                          }),
                        })
                        """
                    )
                    href = str(page_state.get("href", ""))
                    has_chat_input = bool(page_state.get("has_input"))
                    has_sign_in = bool(page_state.get("has_sign_in"))
                    logged_in_url = "login" not in href and "auth" not in href
                except Exception:
                    has_chat_input = False
                    logged_in_url = False
                    has_sign_in = False

                if cookie_string and (has_auth_cookie or (has_chat_input and logged_in_url and not has_sign_in)):
                    return {
                        "cookie": cookie_string,
                        "user_agent": user_agent,
                    }

                time.sleep(2)
        finally:
            context.close()

    raise RuntimeError("Timed out waiting for GLM International browser login to complete.")
