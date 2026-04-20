from __future__ import annotations

from pathlib import Path
from typing import Any

from openclaw_algae.browser.claude import capture_claude_browser_credentials
from openclaw_algae.browser.chatgpt import capture_chatgpt_browser_credentials
from openclaw_algae.browser.deepseek import capture_deepseek_browser_credentials
from openclaw_algae.browser.doubao import capture_doubao_browser_credentials
from openclaw_algae.browser.gemini import capture_gemini_browser_credentials
from openclaw_algae.browser.glm import capture_glm_browser_credentials
from openclaw_algae.browser.glm_intl import capture_glm_intl_browser_credentials
from openclaw_algae.browser.grok import capture_grok_browser_credentials
from openclaw_algae.browser.kimi import capture_kimi_browser_credentials
from openclaw_algae.browser.mimo import capture_mimo_browser_credentials
from openclaw_algae.browser.qwen import capture_qwen_browser_credentials
from openclaw_algae.browser.qwen_cn import capture_qwen_cn_browser_credentials


def capture_provider_browser_credentials(provider: str, *, state_dir: Path) -> dict[str, Any]:
    handlers = {
        "claude": capture_claude_browser_credentials,
        "chatgpt": capture_chatgpt_browser_credentials,
        "deepseek": capture_deepseek_browser_credentials,
        "doubao": capture_doubao_browser_credentials,
        "gemini": capture_gemini_browser_credentials,
        "glm-cn": capture_glm_browser_credentials,
        "glm-intl": capture_glm_intl_browser_credentials,
        "grok": capture_grok_browser_credentials,
        "kimi": capture_kimi_browser_credentials,
        "mimo": capture_mimo_browser_credentials,
        "qwen-cn": capture_qwen_cn_browser_credentials,
        "qwen-intl": capture_qwen_browser_credentials,
    }
    handler = handlers.get(provider)
    if handler is None:
        raise RuntimeError(f"Browser login is not implemented for {provider}")
    return handler(state_dir=state_dir)
