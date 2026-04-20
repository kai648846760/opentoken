from __future__ import annotations

import base64
import json
import time
from pathlib import Path

from openclaw_algae.browser.common import (
    build_cookie_string,
    prepare_browser_state_dir,
    require_sync_playwright,
)


def _is_glm_guest_token(token: str) -> bool:
    """Check if a GLM JWT token is a guest token by inspecting the is_guest claim."""
    if not token:
        return True
    try:
        parts = token.split(".")
        if len(parts) == 3:
            payload_b64 = parts[1] + "=" * ((4 - len(parts[1]) % 4) % 4)
            payload = json.loads(base64.b64decode(payload_b64))
            return bool(payload.get("is_guest", True))
    except Exception:
        pass
    # If we can't decode, treat as guest to be safe
    return True


def capture_glm_browser_credentials(*, state_dir: Path) -> dict[str, str]:
    """Capture GLM credentials.

    Strategy: Poll for valid auth cookies. GLM generates a guest token
    immediately on page load, so we distinguish real login by checking
    for the presence of BOTH chatglm_token AND chatglm_user_id with a
    non-zero-length value (guest user_id is often empty or missing).
    """
    sync_playwright = require_sync_playwright()
    browser_state_dir = prepare_browser_state_dir(state_dir, "glm-cn")

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            str(browser_state_dir),
            headless=False,
        )
        try:
            page = context.pages[0] if context.pages else context.new_page()
            page.goto("https://chatglm.cn/main/all", wait_until="domcontentloaded")
            user_agent = page.evaluate("() => navigator.userAgent")

            print("Please login to GLM (智谱清言)...")

            # Poll for valid login cookies
            deadline = time.time() + 600  # 10 min timeout
            last_token = ""
            stable_count = 0

            while time.time() < deadline:
                try:
                    cookies = context.cookies(["https://chatglm.cn"])
                    cookie_map = {c["name"]: c["value"] for c in cookies if "name" in c}

                    token = cookie_map.get("chatglm_token", "")
                    user_id = cookie_map.get("chatglm_user_id", "")

                    # Reject guest tokens via JWT is_guest claim
                    if token and _is_glm_guest_token(token):
                        stable_count = 0
                        last_token = token
                        continue

                    # Real login: non-guest token AND user_id is populated
                    if token and token != last_token and user_id:
                        stable_count += 1
                        # Require 2 consecutive checks with stable token
                        if stable_count >= 2:
                            cookie_string = build_cookie_string(cookies)
                            if cookie_string:
                                return {
                                    "cookie": cookie_string,
                                    "user_agent": user_agent,
                                }
                        last_token = token
                    elif token == last_token and token:
                        stable_count += 1
                        if stable_count >= 2 and user_id:
                            cookie_string = build_cookie_string(cookies)
                            if cookie_string:
                                return {
                                    "cookie": cookie_string,
                                    "user_agent": user_agent,
                                }
                    else:
                        stable_count = 0
                        if token:
                            last_token = token

                except Exception:
                    pass

                time.sleep(2)

        finally:
            try:
                context.close()
            except Exception:
                pass

    raise RuntimeError("Timed out waiting for GLM browser login to complete.")
