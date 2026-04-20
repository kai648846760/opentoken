from __future__ import annotations

import time
from pathlib import Path

from openclaw_algae.browser.common import (
    build_cookie_string,
    prepare_browser_state_dir,
    require_sync_playwright,
)


def capture_qwen_cn_browser_credentials(*, state_dir: Path) -> dict[str, object]:
    sync_playwright = require_sync_playwright()
    browser_state_dir = prepare_browser_state_dir(state_dir, 'qwen-cn')

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            str(browser_state_dir),
            headless=False,
        )
        try:
            page = context.pages[0] if context.pages else context.new_page()
            page.goto('https://www.qianwen.com/', wait_until='domcontentloaded')
            user_agent = page.evaluate('() => navigator.userAgent')

            deadline = time.time() + 180
            while time.time() < deadline:
                cookies = context.cookies(['https://www.qianwen.com', 'https://qianwen.com'])
                cookie_string = build_cookie_string(cookies)
                session_cookie = next(
                    (
                        item
                        for item in cookies
                        if item['name'] in {'tongyi_sso_ticket', 'login_aliyunid_ticket'}
                    ),
                    None,
                )
                if session_cookie is None:
                    time.sleep(2)
                    continue

                xsrf_cookie = next((item for item in cookies if item['name'] == 'XSRF-TOKEN'), None)
                ut_cookie = next((item for item in cookies if item['name'] == 'b-user-id'), None)
                return {
                    'cookie': cookie_string,
                    'user_agent': user_agent,
                    'metadata': {
                        'xsrf_token': str(xsrf_cookie['value']) if xsrf_cookie is not None else '',
                        'ut': str(ut_cookie['value']) if ut_cookie is not None else '',
                    },
                }
        finally:
            context.close()

    raise RuntimeError('Timed out waiting for Qwen China browser login to complete.')
