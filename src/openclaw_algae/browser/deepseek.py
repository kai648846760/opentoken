from __future__ import annotations

import json
import time
from pathlib import Path

from openclaw_algae.browser.common import (
    build_cookie_string,
    prepare_browser_state_dir,
    require_sync_playwright,
)


def capture_deepseek_browser_credentials(*, state_dir: Path) -> dict[str, str]:
    sync_playwright = require_sync_playwright()
    browser_state_dir = prepare_browser_state_dir(state_dir, 'deepseek')

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            str(browser_state_dir),
            headless=False,
        )
        try:
            page = context.pages[0] if context.pages else context.new_page()
            bearer: dict[str, str] = {'value': ''}

            def on_request(request) -> None:
                if '/api/v0/' not in request.url:
                    return
                auth = request.headers.get('authorization', '')
                if auth.startswith('Bearer '):
                    bearer['value'] = auth.removeprefix('Bearer ').strip()

            def on_response(response) -> None:
                if '/api/v0/users/current' not in response.url or not response.ok:
                    return
                try:
                    payload = response.json()
                except Exception:
                    return
                token = payload.get('data', {}).get('biz_data', {}).get('token', '')
                if isinstance(token, str) and token:
                    bearer['value'] = token

            page.on('request', on_request)
            page.on('response', on_response)
            page.goto('https://chat.deepseek.com', wait_until='domcontentloaded')
            user_agent = page.evaluate('() => navigator.userAgent')

            deadline = time.time() + 300
            while time.time() < deadline:
                cookies = context.cookies(['https://chat.deepseek.com', 'https://deepseek.com'])
                cookie_string = build_cookie_string(cookies)

                if not bearer['value']:
                    try:
                        storage_entries = page.evaluate(
                            """
                            () => {
                              const data = {};
                              for (let i = 0; i < window.localStorage.length; i += 1) {
                                const key = window.localStorage.key(i);
                                if (key) {
                                  data[key] = window.localStorage.getItem(key) || '';
                                }
                              }
                              return data;
                            }
                            """
                        )
                        if isinstance(storage_entries, dict):
                            for key, value in storage_entries.items():
                                if 'token' not in key.lower() and 'auth' not in key.lower():
                                    continue
                                if not isinstance(value, str) or not value:
                                    continue
                                try:
                                    parsed = json.loads(value)
                                except json.JSONDecodeError:
                                    parsed = value
                                if isinstance(parsed, dict):
                                    token = parsed.get('token', '')
                                    if isinstance(token, str) and token:
                                        bearer['value'] = token
                                        break
                                if isinstance(parsed, str) and parsed:
                                    bearer['value'] = parsed
                                    break
                    except Exception:
                        pass

                if cookie_string and bearer['value']:
                    return {
                        'cookie': cookie_string,
                        'bearer': bearer['value'],
                        'user_agent': user_agent,
                    }

                time.sleep(2)
        finally:
            context.close()

    raise RuntimeError('Timed out waiting for DeepSeek browser login to complete.')
