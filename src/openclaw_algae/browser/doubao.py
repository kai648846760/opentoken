from __future__ import annotations

import json
import re
import time
from pathlib import Path

from openclaw_algae.browser.common import (
    build_cookie_string,
    prepare_browser_state_dir,
    require_sync_playwright,
)


def capture_doubao_browser_credentials(*, state_dir: Path) -> dict[str, object]:
    browser_state_dir = prepare_browser_state_dir(state_dir, "doubao")
    sync_playwright = require_sync_playwright()
    return _capture_doubao_browser_credentials_with_factory(
        sync_playwright,
        browser_state_dir=browser_state_dir,
    )


def _capture_doubao_browser_credentials_with_factory(
    sync_playwright_factory,
    *,
    browser_state_dir: Path,
) -> dict[str, object]:
    with sync_playwright_factory() as playwright:
        context = playwright.chromium.launch_persistent_context(
            str(browser_state_dir),
            headless=False,
        )
        try:
            page = context.pages[0] if context.pages else context.new_page()
            captured_sessionid: dict[str, str] = {"value": ""}

            def on_request(request) -> None:
                if "doubao.com" not in request.url:
                    return
                cookie = request.headers.get("cookie", "")
                if not cookie:
                    return
                match = re.search(r"sessionid=([^;]+)", cookie)
                if match:
                    captured_sessionid["value"] = match.group(1)

            page.on("request", on_request)
            page.goto("https://www.doubao.com/chat/", wait_until="domcontentloaded")
            user_agent = page.evaluate("() => navigator.userAgent")

            deadline = time.time() + 300
            while time.time() < deadline:
                cookies = context.cookies(["https://www.doubao.com", "https://doubao.com"])
                cookie_string = build_cookie_string(cookies)
                sessionid_cookie = next((item for item in cookies if item["name"] == "sessionid"), None)
                ttwid_cookie = next((item for item in cookies if item["name"] == "ttwid"), None)
                s_v_web_id_cookie = next((item for item in cookies if item["name"] == "s_v_web_id"), None)
                sessionid = captured_sessionid["value"] or (
                    str(sessionid_cookie["value"]) if sessionid_cookie is not None else ""
                )

                if cookie_string and sessionid:
                    metadata = {
                        "sessionid": sessionid,
                        "ttwid": str(ttwid_cookie["value"]) if ttwid_cookie is not None else "",
                    }
                    if s_v_web_id_cookie:
                        metadata["fp"] = str(s_v_web_id_cookie["value"])

                    return {
                        "cookie": cookie_string,
                        "user_agent": user_agent,
                        "metadata": metadata,
                    }

                time.sleep(2)
        finally:
            context.close()

    raise RuntimeError("Timed out waiting for Doubao browser login to complete.")
