from __future__ import annotations

import re
import time
from pathlib import Path

from openclaw_algae.browser.common import (
    build_cookie_string,
    prepare_browser_state_dir,
    require_sync_playwright,
)


def capture_qwen_browser_credentials(*, state_dir: Path) -> dict[str, str]:
    sync_playwright = require_sync_playwright()
    browser_state_dir = prepare_browser_state_dir(state_dir, 'qwen-intl')

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            str(browser_state_dir),
            headless=False,
        )
        try:
            page = context.pages[0] if context.pages else context.new_page()
            session_token: dict[str, str] = {'value': ''}

            def on_request(request) -> None:
                if 'qwen.ai' not in request.url:
                    return
                auth = request.headers.get('authorization', '')
                if auth.lower().startswith('bearer '):
                    session_token['value'] = auth[7:].strip()
                    return
                cookie = request.headers.get('cookie', '')
                if not cookie:
                    return
                match = re.search(r'(?:session|token|auth)[^=]*=([^;]+)', cookie, re.IGNORECASE)
                if match:
                    session_token['value'] = match.group(1)

            page.on('request', on_request)
            page.goto('https://chat.qwen.ai/', wait_until='domcontentloaded')
            user_agent = page.evaluate('() => navigator.userAgent')

            deadline = time.time() + 300
            while time.time() < deadline:
                cookies = context.cookies(['https://chat.qwen.ai', 'https://qwen.ai'])
                cookie_string = build_cookie_string(cookies)
                if cookie_string and session_token['value']:
                    return {
                        'cookie': cookie_string,
                        'session_token': session_token['value'],
                        'user_agent': user_agent,
                    }
                time.sleep(2)
        finally:
            context.close()

    raise RuntimeError('Timed out waiting for Qwen browser login to complete.')
