from __future__ import annotations

import time
from pathlib import Path

from openclaw_algae.browser.common import (
    build_cookie_string,
    prepare_browser_state_dir,
    require_sync_playwright,
)


def capture_chatgpt_browser_credentials(*, state_dir: Path) -> dict[str, str]:
    sync_playwright = require_sync_playwright()
    browser_state_dir = prepare_browser_state_dir(state_dir, 'chatgpt')

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            str(browser_state_dir),
            headless=False,
        )
        try:
            page = context.pages[0] if context.pages else context.new_page()
            page.goto('https://chatgpt.com/', wait_until='domcontentloaded')
            user_agent = page.evaluate('() => navigator.userAgent')

            deadline = time.time() + 300
            while time.time() < deadline:
                cookies = context.cookies(['https://chatgpt.com', 'https://chat.openai.com'])
                cookie_string = build_cookie_string(cookies)

                access_token = ''
                session_cookie = next(
                    (item for item in cookies if item['name'] == '__Secure-next-auth.session-token'),
                    None,
                )
                if session_cookie is not None:
                    access_token = str(session_cookie['value'])
                else:
                    token_0 = next(
                        (item for item in cookies if item['name'] == '__Secure-next-auth.session-token.0'),
                        None,
                    )
                    token_1 = next(
                        (item for item in cookies if item['name'] == '__Secure-next-auth.session-token.1'),
                        None,
                    )
                    if token_0 is not None and token_1 is not None:
                        access_token = f"{token_0['value']}{token_1['value']}"

                if cookie_string and access_token:
                    return {
                        'cookie': cookie_string,
                        'access_token': access_token,
                        'user_agent': user_agent,
                    }

                time.sleep(2)
        finally:
            context.close()

    raise RuntimeError('Timed out waiting for ChatGPT browser login to complete.')
