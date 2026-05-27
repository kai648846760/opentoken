from __future__ import annotations

import atexit
import base64
import hashlib
import inspect
import json
import queue
import re
import shutil
import time
import threading
from contextlib import contextmanager
from collections.abc import Iterator
from dataclasses import dataclass
from uuid import uuid4

import httpx

from opentoken.browser.common import (
    build_cookie_string,
    prepare_browser_state_dir,
    require_sync_playwright,
)
from opentoken.config.paths import resolve_state_dir
from opentoken.models.provider_credentials import ProviderCredentialRecord
from opentoken.providers.base import ProviderRateLimitError
from opentoken.providers.doubao import (
    DoubaoWebClient,
    _extract_doubao_chunks_from_event,
    _extract_samantha_chunks,
    _parse_doubao_response_text,
    resolve_doubao_query_params,
)
from opentoken.providers.glm import GLMApiClient, GLMIntlApiClient
from opentoken.providers.glm import _extract_glm_text_candidates, _glm_meta_data_for_model
from opentoken.providers.qwen import (
    QwenApiClient,
    _extract_qwen_intl_phased_segments,
    _qwen_feature_config_for_model,
)
from opentoken.storage.provider_store import save_provider_credentials
from opentoken.storage.provider_sessions import load_provider_session, save_provider_session


_GLM_SIGN_SECRET = "8a1317a7468aa3ad86e997d08f3f31cb"
_GLM_X_EXP_GROUPS = (
    "na_android_config:exp:NA,na_4o_config:exp:4o_A,tts_config:exp:tts_config_a,"
    "na_glm4plus_config:exp:open,mainchat_server_app:exp:A,mobile_history_daycheck:exp:a,"
    "desktop_toolbar:exp:A,chat_drawing_server:exp:A,drawing_server_cogview:exp:cogview4,"
    "app_welcome_v2:exp:A,chat_drawing_streamv2:exp:A,mainchat_rm_fc:exp:add,"
    "mainchat_dr:exp:open,chat_auto_entrance:exp:A,drawing_server_hi_dream:control:A,"
    "homepage_square:exp:close,assistant_recommend_prompt:exp:3,app_home_regular_user:exp:A,"
    "memory_common:exp:enable,mainchat_moe:exp:300,assistant_greet_user:exp:greet_user,"
    "app_welcome_personalize:exp:A,assistant_model_exp_group:exp:glm4.5,"
    "ai_wallet:exp:ai_wallet_enable"
)
_GLM_ASSISTANT_ID_MAP = {
    "glm-5": "65940acff94777010aa6b796",
    "glm-4-plus": "65940acff94777010aa6b796",
    "glm-4": "65940acff94777010aa6b796",
    "glm-4-think": "676411c38945bbc58a905d31",
    "glm-4-zero": "676411c38945bbc58a905d31",
}
_QWEN_CN_BASE_URL = "https://chat2.qianwen.com"
_QWEN_INTL_BASE_URL = "https://chat.qwen.ai"
_KIMI_BASE_URL = "https://www.kimi.com"
_CHATGPT_URL = "https://chatgpt.com/"
_GEMINI_URL = "https://gemini.google.com/app"
_GROK_URL = "https://grok.com"
_GLM_CN_URL = "https://chatglm.cn"
_GLM_INTL_URL = "https://chat.z.ai/"
_DOUBAO_URL = "https://www.doubao.com/chat/"
_GLM_INTL_API_STREAM_STARTUP_TIMEOUT_SECONDS = 4.0
_GLM_INTL_DOM_BOOTSTRAP_TIMEOUT_MS = 10000
_GLM_INTL_API_STREAM_HTTP_TIMEOUT = httpx.Timeout(
    connect=6.0,
    read=6.0,
    write=20.0,
    pool=20.0,
)
_QWEN_INTL_API_STREAM_STARTUP_TIMEOUT_SECONDS = 8.0
_QWEN_INTL_API_STREAM_MAX_ATTEMPTS = 2
_QWEN_INTL_BROWSER_BOOTSTRAP_TIMEOUT_MS = 10000
_QWEN_INTL_API_STREAM_HTTP_TIMEOUT = httpx.Timeout(
    connect=6.0,
    read=12.0,
    write=20.0,
    pool=20.0,
)
_DOUBAO_BROWSER_STREAM_STARTUP_TIMEOUT_SECONDS = 15.0
_DOUBAO_DOM_STREAM_STARTUP_TIMEOUT_SECONDS = 15.0
_DOUBAO_MODEL_MENU_NAME_MAP = {
    "doubao-seed-2.0": ("快速", "快速 适用于大部分情况"),
    "doubao-lite": ("快速", "快速 适用于大部分情况"),
    "doubao-thinking": ("思考", "思考 擅长解决更难的问题"),
    "doubao-pro": ("专家", "专家 新 研究级智能模型"),
}
_GLM_CN_AUTH_COOKIE_NAMES = frozenset(
    {
        "chatglm_token",
        "chatglm_refresh_token",
        "chatglm_user_id",
        "chatglm_device_id",
    }
)
_PROVIDER_SESSION_LOCKS: dict[str, threading.Lock] = {}
_PROVIDER_SESSION_LOCKS_GUARD = threading.Lock()
# Global per-provider sessions — protected by _PROVIDER_SESSION_LOCKS.
# A single session is reused across threads; after each use the playwright
# context stays open so the next call (possibly on a different thread) can
# reuse it.  We recreate only on errors or page-closed events.
_PROVIDER_GLOBAL_SESSIONS: dict[str, "_ProviderBrowserSession"] = {}
_PROVIDER_GLOBAL_SESSIONS_GUARD = threading.Lock()


@dataclass
class _ProviderBrowserSession:
    manager: object
    context: object
    page: object
    headless: bool
    owner_thread: threading.Thread
    metadata: dict[str, object]


class CamoufoxProviderClient:
    def __init__(
        self,
        provider: str,
        credentials: ProviderCredentialRecord,
        *,
        headless: bool = True,
    ) -> None:
        self._provider = provider
        self._credentials = credentials
        self._headless = headless
        self._state_dir = resolve_state_dir()

    def chat_completion(self, *, message: str, model: str) -> str:
        dispatch = {
            "doubao": self._chat_doubao,
            "qwen-intl": self._chat_qwen_intl,
            "qwen-cn": self._chat_qwen_cn,
            "kimi": self._chat_kimi,
            "chatgpt": self._chat_chatgpt,
            "gemini": self._chat_gemini,
            "grok": self._chat_grok,
            "glm-cn": self._chat_glm_cn,
            "glm-intl": self._chat_glm_intl,
        }
        handler = dispatch.get(self._provider)
        if handler is None:
            raise RuntimeError(f"Unsupported browser provider: {self._provider}")
        return handler(message=message, model=model)

    def stream_chat_completion(self, *, message: str, model: str) -> Iterator[str] | None:
        dispatch = {
            "doubao": self._stream_doubao,
            "qwen-intl": self._stream_qwen_intl,
            "glm-cn": self._stream_glm_cn,
            "glm-intl": self._stream_glm_intl,
        }
        handler = dispatch.get(self._provider)
        if handler is None:
            return None
        return handler(message=message, model=model)

    def tool_chat_completion(self, *, message: str, model: str) -> str:
        dispatch = {
            "doubao": self._tool_chat_doubao,
            "glm-cn": self._tool_chat_glm_cn,
            "glm-intl": self._tool_chat_glm_intl,
        }
        handler = dispatch.get(self._provider)
        if handler is None:
            return self.chat_completion(message=message, model=model)
        return handler(message=message, model=model)

    def _with_page(self, *, start_url: str, cookie_domains: tuple[str, ...], action):
        with _provider_session_lock(self._provider):
            session = _get_or_create_browser_session(
                provider=self._provider,
                state_dir=self._state_dir,
                headless=self._headless,
            )
            context = session.context
            page = session.page
            page_was_replaced = False
            try:
                self._inject_cookie_string(context, cookie_domains)
                if _page_is_closed(page):
                    page = context.pages[0] if getattr(context, "pages", []) else context.new_page()
                    session.page = page
                    page_was_replaced = True
                current_url = str(getattr(page, "url", ""))
                def _url_matches_domain(url: str, domain: str) -> bool:
                    clean_domain = domain.strip(".").lower()
                    clean_url = url.lower().split("?")[0].split("#")[0]
                    # Check if URL hostname ends with domain
                    if "://" in clean_url:
                        hostname = clean_url.split("://", 1)[1].split("/", 1)[0]
                    else:
                        hostname = clean_url.split("/", 1)[0]
                    return hostname == clean_domain or hostname.endswith("." + clean_domain)
                if not current_url or not any(
                    _url_matches_domain(current_url, domain) for domain in cookie_domains
                ):
                    page.goto(start_url, wait_until="domcontentloaded", timeout=120000)
                return action(context, page)
            except Exception:
                if not page_was_replaced:
                    _close_browser_session(self._provider)
                raise

    def _inject_cookie_string(self, context, domains: tuple[str, ...]) -> None:
        cookie_string = self._credentials.cookie or ""
        if not cookie_string:
            return
        skip_cookie_names: set[str] = set()
        if self._provider == "glm-cn":
            existing_cookie_map = self._cookie_map(
                context,
                [f"https://{domain.lstrip('.')}" for domain in domains],
            )
            saved_cookie_map = _parse_cookie_string(cookie_string)
            if (
                _glm_cookie_map_has_real_login(existing_cookie_map)
                and not _glm_cookie_map_has_real_login(saved_cookie_map)
            ):
                skip_cookie_names.update(_GLM_CN_AUTH_COOKIE_NAMES)
        cookies = []
        for item in cookie_string.split(";"):
            if "=" not in item:
                continue
            name, value = item.split("=", 1)
            name = name.strip()
            value = value.strip()
            if not name:
                continue
            if name in skip_cookie_names:
                continue
            for domain in domains:
                cookie = {
                    "name": name,
                    "value": value,
                    "domain": domain,
                    "path": "/",
                }
                if name.startswith("__Secure-") or name.startswith("__Host-"):
                    cookie["secure"] = True
                cookies.append(cookie)
        if not cookies:
            return
        try:
            context.add_cookies(cookies)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to inject cookies for {self._provider}: {exc}"
            ) from exc

    def _cookie_map(self, context, urls: list[str]) -> dict[str, str]:
        try:
            cookies = context.cookies(urls)
        except Exception:
            cookies = []
        return {
            str(cookie.get("name", "")): str(cookie.get("value", ""))
            for cookie in cookies
            if cookie.get("name")
        }

    def _sync_live_glm_cn_credentials(self, context, cookie_map: dict[str, str]) -> None:
        if self._provider != "glm-cn" or not _glm_cookie_map_has_real_login(cookie_map):
            return
        current_cookie_string = self._credentials.cookie or ""
        saved_cookie_map = _parse_cookie_string(current_cookie_string)
        if (
            _glm_cookie_map_has_real_login(saved_cookie_map)
            and saved_cookie_map.get("chatglm_token") == cookie_map.get("chatglm_token")
            and saved_cookie_map.get("chatglm_refresh_token") == cookie_map.get("chatglm_refresh_token")
            and saved_cookie_map.get("chatglm_user_id") == cookie_map.get("chatglm_user_id")
        ):
            return
        try:
            live_cookies = context.cookies([_GLM_CN_URL])
        except Exception:
            return
        live_cookie_string = build_cookie_string(live_cookies)
        if not live_cookie_string or live_cookie_string == current_cookie_string:
            return
        self._credentials = self._credentials.model_copy(update={"cookie": live_cookie_string})
        try:
            save_provider_credentials(self._state_dir / "providers", self._credentials)
        except Exception:
            pass

    def _chat_qwen_intl(self, *, message: str, model: str) -> str:
        def action(_context, page):
            created = page.evaluate(
                """
                async ({ baseUrl }) => {
                  const res = await fetch(`${baseUrl}/api/v2/chats/new`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({}),
                  });
                  if (!res.ok) {
                    return { ok: false, status: res.status, error: await res.text() };
                  }
                  const data = await res.json();
                  return { ok: true, chatId: data?.data?.id ?? data?.chat_id ?? data?.id ?? data?.chatId };
                }
                """,
                {"baseUrl": _QWEN_INTL_BASE_URL},
            )
            if not created.get("ok") or not created.get("chatId"):
                raise RuntimeError(
                    f"Qwen International chat creation failed: {created.get('status')} {created.get('error', '')}"
                )
            response = page.evaluate(
                """
                async ({ baseUrl, chatId, model, message, fid }) => {
                  const res = await fetch(`${baseUrl}/api/v2/chat/completions?chat_id=${chatId}`, {
                    method: "POST",
                    headers: {
                      "Content-Type": "application/json",
                      Accept: "text/event-stream",
                    },
                    body: JSON.stringify({
                      stream: true,
                      version: "2.1",
                      incremental_output: true,
                      chat_id: chatId,
                      chat_mode: "normal",
                      model,
                      parent_id: null,
                      messages: [{
                        fid,
                        parentId: null,
                        childrenIds: [],
                        role: "user",
                        content: message,
                        user_action: "chat",
                        files: [],
                        timestamp: Math.floor(Date.now() / 1000),
                        models: [model],
                        chat_type: "t2t",
                        feature_config: featureConfig,
                      }],
                    }),
                  });
                  if (!res.ok) {
                    return { ok: false, status: res.status, error: await res.text() };
                  }
                  return { ok: true, text: await res.text() };
                }
                """,
                {
                    "baseUrl": _QWEN_INTL_BASE_URL,
                    "chatId": created["chatId"],
                    "model": model,
                    "message": message,
                    "fid": str(uuid4()),
                    "featureConfig": _qwen_feature_config_for_model(model),
                },
            )
            if not response.get("ok"):
                raise RuntimeError(
                    f"Qwen International request failed: {response.get('status')} {response.get('error', '')}"
                )
            content = _parse_sse_text(str(response.get("text", "")))
            if not content:
                raise RuntimeError("Qwen International returned no text content.")
            return content

        return self._with_page(
            start_url=f"{_QWEN_INTL_BASE_URL}/",
            cookie_domains=(".qwen.ai",),
            action=action,
        )

    def _stream_qwen_intl(self, *, message: str, model: str) -> Iterator[str]:
        def iterator() -> Iterator[str]:
            # Prefer direct API SSE for qwen-intl streaming because it delivers
            # stable incremental tokens without browser/profile startup latency.
            api_error: Exception | None = None
            for attempt in range(_QWEN_INTL_API_STREAM_MAX_ATTEMPTS):
                try:
                    yielded_any = False
                    for piece in _iter_stream_with_startup_timeout(
                        lambda: _stream_qwen_intl_api_completion(
                            self._credentials,
                            message=message,
                            model=model,
                        ),
                        description="Qwen International API stream",
                        startup_timeout_seconds=_QWEN_INTL_API_STREAM_STARTUP_TIMEOUT_SECONDS,
                    ):
                        if not piece:
                            continue
                        yielded_any = True
                        yield piece
                    if yielded_any:
                        return
                except Exception as exc:
                    if yielded_any:
                        raise
                    api_error = exc
                    if (
                        attempt + 1 < _QWEN_INTL_API_STREAM_MAX_ATTEMPTS
                        and _is_qwen_intl_retryable_api_stream_error(exc)
                    ):
                        continue
                    break

            with _provider_session_lock(self._provider):
                session = _get_or_create_browser_session(
                    provider=self._provider,
                    state_dir=self._state_dir,
                    headless=self._headless,
                )
                context = session.context
                page = session.page
                page_was_replaced = False
                try:
                    self._inject_cookie_string(context, (".qwen.ai",))
                    if _page_is_closed(page):
                        page = context.pages[0] if getattr(context, "pages", []) else context.new_page()
                        session.page = page
                        page_was_replaced = True
                    current_url = str(getattr(page, "url", ""))
                    if not current_url or "chat.qwen.ai" not in current_url:
                        page.goto(
                            f"{_QWEN_INTL_BASE_URL}/",
                            wait_until="commit",
                            timeout=_QWEN_INTL_BROWSER_BOOTSTRAP_TIMEOUT_MS,
                        )

                    try:
                        yielded_any = False
                        for piece in _stream_qwen_intl_browser_completion(
                            page,
                            message=message,
                            model=model,
                        ):
                            if not piece:
                                continue
                            yielded_any = True
                            yield piece
                        if yielded_any:
                            return
                    except Exception:
                        if yielded_any:
                            raise

                    yield from _stream_qwen_intl_dom_completion(page, message=message, model=model)
                    return
                except Exception:
                    if not page_was_replaced:
                        _close_browser_session(self._provider)
                    raise

        return iterator()

    def _chat_qwen_cn(self, *, message: str, model: str) -> str:
        def action(context, page):
            current_url = str(getattr(page, "url", ""))
            if not current_url.startswith("https://www.qianwen.com"):
                page.goto("https://www.qianwen.com/", wait_until="domcontentloaded", timeout=120000)
                page.wait_for_timeout(3000)
            else:
                # Already on the right page — just make sure the composer is ready
                try:
                    page.locator('[contenteditable="true"][role="textbox"]').first.wait_for(timeout=5000)
                except Exception:
                    page.goto("https://www.qianwen.com/", wait_until="domcontentloaded", timeout=120000)
                    page.wait_for_timeout(3000)
            cookie_map = self._cookie_map(
                context,
                [
                    "https://www.qianwen.com",
                    "https://qianwen.com",
                    "https://chat2.qianwen.com",
                ],
            )
            auth = _resolve_qwen_cn_request_auth(self._credentials, cookie_map)
            return _fetch_qwen_cn_browser_completion(
                page,
                message=message,
                model=model,
                auth=auth,
            )

        return self._with_page(
            start_url="https://www.qianwen.com/",
            cookie_domains=(".qianwen.com",),
            action=action,
        )

    def _chat_doubao(self, *, message: str, model: str) -> str:
        with _provider_session_lock(self._provider):
            session = _get_or_create_browser_session(
                provider=self._provider,
                state_dir=self._state_dir,
                headless=self._headless,
            )
            context = session.context
            page = session.page
            page_was_replaced = False
            isolated_tool_context = bool(getattr(self, "_tool_conversation_isolation", False))
            prior_conversation_id = str(session.metadata.get("doubao_conversation_id") or "").strip()
            try:
                self._inject_cookie_string(context, (".doubao.com", "www.doubao.com"))
                if _page_is_closed(page):
                    page = context.pages[0] if getattr(context, "pages", []) else context.new_page()
                    session.page = page
                    page_was_replaced = True
                current_url = str(getattr(page, "url", ""))
                if (
                    not current_url
                    or "doubao.com" not in current_url
                    or "doubao-region-ban" in current_url
                ):
                    page.goto(_DOUBAO_URL, wait_until="domcontentloaded", timeout=120000)
                    page.wait_for_timeout(3000)
                session.metadata["doubao_conversation_id"] = "0"
                if isolated_tool_context:
                    self._suppress_doubao_conversation_persist = True
                session.metadata["doubao_request_params"] = resolve_doubao_query_params(self._credentials)
                try:
                    _select_doubao_model(page, model)
                    return _call_with_supported_kwargs(
                        _fetch_doubao_browser_completion,
                        page,
                        session=session,
                        client=self,
                        message=message,
                        model=model,
                    )
                except ProviderRateLimitError:
                    return _call_with_supported_kwargs(
                        _dom_send_and_wait_doubao,
                        page,
                        session=session,
                        client=self,
                        message=message,
                        model=model,
                    )
                except RuntimeError:
                    return _call_with_supported_kwargs(
                        _dom_send_and_wait_doubao,
                        page,
                        session=session,
                        client=self,
                        message=message,
                        model=model,
                    )
            except ProviderRateLimitError:
                raise
            except Exception:
                if not page_was_replaced:
                    _close_browser_session(self._provider)
                raise
            finally:
                if isolated_tool_context:
                    if prior_conversation_id:
                        session.metadata["doubao_conversation_id"] = prior_conversation_id
                    else:
                        session.metadata.pop("doubao_conversation_id", None)
                    self._suppress_doubao_conversation_persist = False

    def _stream_doubao(self, *, message: str, model: str) -> Iterator[str]:
        def iterator() -> Iterator[str]:
            with _provider_session_lock(self._provider):
                session = _get_or_create_browser_session(
                    provider=self._provider,
                    state_dir=self._state_dir,
                    headless=self._headless,
                )
                context = session.context
                page = session.page
                page_was_replaced = False
                try:
                    self._inject_cookie_string(context, (".doubao.com", "www.doubao.com"))
                    if _page_is_closed(page):
                        page = context.pages[0] if getattr(context, "pages", []) else context.new_page()
                        session.page = page
                        page_was_replaced = True
                    current_url = str(getattr(page, "url", ""))
                    if (
                        not current_url
                        or "doubao.com" not in current_url
                        or "doubao-region-ban" in current_url
                    ):
                        page.goto(_DOUBAO_URL, wait_until="domcontentloaded", timeout=120000)
                        page.wait_for_timeout(3000)
                    session.metadata["doubao_conversation_id"] = "0"
                    session.metadata["doubao_request_params"] = resolve_doubao_query_params(self._credentials)
                    _select_doubao_model(page, model)
                    prefer_dom_first = _prefer_dom_first_doubao_stream(model)

                    def _yield_stream(stream_factory) -> tuple[bool, Exception | None]:
                        emitted_any = False
                        try:
                            stream = stream_factory()
                            for piece in stream:
                                if not piece:
                                    continue
                                emitted_any = True
                                yield piece
                        except ProviderRateLimitError:
                            raise
                        except RuntimeError as exc:
                            if emitted_any:
                                raise
                            return emitted_any, exc
                        return emitted_any, None

                    if prefer_dom_first:
                        dom_result = _yield_stream(
                            lambda: _call_with_supported_kwargs(
                                _stream_doubao_dom_completion,
                                page,
                                session=session,
                                client=self,
                                message=message,
                                model=model,
                                startup_timeout_seconds=_DOUBAO_DOM_STREAM_STARTUP_TIMEOUT_SECONDS,
                            )
                        )
                        dom_emitted, dom_error = yield from dom_result
                        if dom_emitted:
                            return

                        browser_result = _yield_stream(
                            lambda: _call_with_supported_kwargs(
                                _stream_doubao_browser_completion,
                                page,
                                session=session,
                                client=self,
                                message=message,
                                model=model,
                                startup_timeout_seconds=_DOUBAO_BROWSER_STREAM_STARTUP_TIMEOUT_SECONDS,
                            )
                        )
                        browser_emitted, browser_error = yield from browser_result
                        if browser_emitted:
                            return
                        if browser_error is not None:
                            raise browser_error
                        if dom_error is not None:
                            raise dom_error
                        raise RuntimeError("Doubao stream returned no text content from DOM or browser path.")

                    browser_result = _yield_stream(
                        lambda: _call_with_supported_kwargs(
                            _stream_doubao_browser_completion,
                            page,
                            session=session,
                            client=self,
                            message=message,
                            model=model,
                            startup_timeout_seconds=_DOUBAO_BROWSER_STREAM_STARTUP_TIMEOUT_SECONDS,
                        )
                    )
                    browser_emitted, browser_error = yield from browser_result
                    if browser_emitted:
                        return

                    dom_result = _yield_stream(
                        lambda: _call_with_supported_kwargs(
                            _stream_doubao_dom_completion,
                            page,
                            session=session,
                            client=self,
                            message=message,
                            model=model,
                            startup_timeout_seconds=_DOUBAO_DOM_STREAM_STARTUP_TIMEOUT_SECONDS,
                        )
                    )
                    dom_emitted, dom_error = yield from dom_result
                    if dom_emitted:
                        return
                    if dom_error is not None:
                        raise dom_error
                    if browser_error is not None:
                        raise browser_error
                    raise RuntimeError("Doubao stream returned no text content from browser or DOM path.")
                except Exception:
                    if not page_was_replaced:
                        _close_browser_session(self._provider)
                    raise

        return iterator()

    def _tool_chat_doubao(self, *, message: str, model: str) -> str:
        self._tool_conversation_isolation = True
        try:
            return self._chat_doubao(message=message, model=model)
        finally:
            self._tool_conversation_isolation = False

    def _persist_doubao_conversation_id(self, conversation_id: str) -> None:
        if bool(getattr(self, "_suppress_doubao_conversation_persist", False)):
            return
        conversation_id = conversation_id.strip()
        if not conversation_id or conversation_id == "0":
            return
        save_provider_session(
            self._state_dir,
            provider=self._provider,
            credentials=self._credentials,
            state={"conversation_id": conversation_id},
        )

    def _load_glm_cn_session_state(self) -> dict[str, str]:
        return load_provider_session(
            self._state_dir,
            provider=self._provider,
            credentials=self._credentials,
        )

    def _persist_glm_cn_session_state(
        self,
        *,
        device_id: str | None = None,
        conversation_id: str | None = None,
    ) -> None:
        current = self._load_glm_cn_session_state()
        cleaned_device_id = str(device_id or "").strip()
        cleaned_conversation_id = str(conversation_id or "").strip()
        if cleaned_device_id:
            current["device_id"] = cleaned_device_id
        if (
            cleaned_conversation_id
            and cleaned_conversation_id != "0"
            and not bool(getattr(self, "_suppress_glm_cn_conversation_persist", False))
        ):
            current["conversation_id"] = cleaned_conversation_id
        if not current:
            return
        save_provider_session(
            self._state_dir,
            provider=self._provider,
            credentials=self._credentials,
            state=current,
        )

    def _chat_kimi(self, *, message: str, model: str) -> str:
        def action(context, page):
            # Ensure page is on kimi.com
            current_url = str(getattr(page, "url", ""))
            if not current_url.startswith(_KIMI_BASE_URL):
                page.goto(_KIMI_BASE_URL, wait_until="domcontentloaded", timeout=120000)
                page.wait_for_timeout(3000)

            cookie_map = self._cookie_map(context, [_KIMI_BASE_URL, "https://kimi.com"])
            kimi_auth = cookie_map.get("kimi-auth") or cookie_map.get("access_token") or ""
            if not kimi_auth:
                raise RuntimeError("Kimi auth cookie not found. Re-run `opentoken login kimi`.")
            scenario = (
                "SCENARIO_SEARCH"
                if "search" in model
                else "SCENARIO_RESEARCH"
                if "research" in model
                else "SCENARIO_K1"
                if "k1" in model
                else "SCENARIO_K2"
            )
            response = page.evaluate(
                """
                async ({ baseUrl, message, kimiAuthToken, scenario }) => {
                  const req = {
                    scenario,
                    message: {
                      role: "user",
                      blocks: [{ message_id: "", text: { content: message } }],
                      scenario,
                    },
                    options: { thinking: false },
                  };
                  const enc = new TextEncoder().encode(JSON.stringify(req));
                  const buf = new ArrayBuffer(5 + enc.byteLength);
                  const dv = new DataView(buf);
                  dv.setUint8(0, 0);
                  dv.setUint32(1, enc.byteLength, false);
                  new Uint8Array(buf, 5).set(enc);

                  const res = await fetch(`${baseUrl}/apiv2/kimi.gateway.chat.v1.ChatService/Chat`, {
                    method: "POST",
                    headers: {
                      "Content-Type": "application/connect+json",
                      "Connect-Protocol-Version": "1",
                      Accept: "*/*",
                      Origin: baseUrl,
                      Referer: `${baseUrl}/`,
                      "X-Language": "zh-CN",
                      "X-Msh-Platform": "web",
                      Authorization: `Bearer ${kimiAuthToken}`,
                    },
                    body: buf,
                  });
                  if (!res.ok) {
                    return { ok: false, status: res.status, error: await res.text() };
                  }
                  const arr = await res.arrayBuffer();
                  const u8 = new Uint8Array(arr);
                  const texts = [];
                  let offset = 0;
                  while (offset + 5 <= u8.length) {
                    const len = new DataView(u8.buffer, u8.byteOffset + offset + 1, 4).getUint32(0, false);
                    if (offset + 5 + len > u8.length) {
                      break;
                    }
                    const chunk = u8.slice(offset + 5, offset + 5 + len);
                    try {
                      const payload = JSON.parse(new TextDecoder().decode(chunk));
                      if (payload?.error) {
                        return {
                          ok: false,
                          status: 500,
                          error: payload.error.message || JSON.stringify(payload.error),
                        };
                      }
                      if (payload?.block?.text?.content && ["set", "append"].includes(payload.op || "")) {
                        texts.push(payload.block.text.content);
                      }
                    } catch (_err) {
                    }
                    offset += 5 + len;
                  }
                  return { ok: true, text: texts.join("") };
                }
                """,
                {
                    "baseUrl": _KIMI_BASE_URL,
                    "message": message,
                    "kimiAuthToken": kimi_auth,
                    "scenario": scenario,
                },
            )
            if not response.get("ok"):
                raise RuntimeError(f"Kimi request failed: {response.get('status')} {response.get('error', '')}")
            content = str(response.get("text", ""))
            if not content:
                raise RuntimeError("Kimi returned no text content.")
            return content

        return self._with_page(
            start_url=f"{_KIMI_BASE_URL}/",
            cookie_domains=(".kimi.com", ".moonshot.cn"),
            action=action,
        )

    def _chat_chatgpt(self, *, message: str, model: str) -> str:
        def action(_context, page):
            return _dom_send_and_wait_chatgpt(page, message)

        return self._with_page(
            start_url=_CHATGPT_URL,
            cookie_domains=(".chatgpt.com",),
            action=action,
        )

    def _chat_gemini(self, *, message: str, model: str) -> str:
        def action(_context, page):
            return _dom_send_and_wait_gemini(page, message)

        return self._with_page(
            start_url=_GEMINI_URL,
            cookie_domains=(".google.com",),
            action=action,
        )

    def _chat_grok(self, *, message: str, model: str) -> str:
        def action(_context, page):
            try:
                response = page.evaluate(
                    """
                    async ({ message }) => {
                      let convId = null;
                      const urls = [
                        "https://grok.com/rest/app-chat/conversations?limit=1",
                        "https://grok.com/rest/app-chat/conversations",
                      ];
                      for (const url of urls) {
                        const listRes = await fetch(url, { credentials: "include" });
                        if (listRes.ok) {
                          const list = await listRes.json();
                          convId = list?.conversations?.[0]?.conversationId ?? null;
                          if (convId) {
                            break;
                          }
                        }
                      }
                      if (!convId) {
                        const createRes = await fetch("https://grok.com/rest/app-chat/conversations", {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          credentials: "include",
                          body: JSON.stringify({}),
                        });
                        if (!createRes.ok) {
                          return { ok: false, status: createRes.status, error: await createRes.text() };
                        }
                        const createData = await createRes.json();
                        convId = createData?.conversationId ?? createData?.id ?? null;
                      }
                      if (!convId) {
                        return { ok: false, status: 500, error: "No conversation id" };
                      }
                      const res = await fetch(`https://grok.com/rest/app-chat/conversations/${convId}/responses`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        credentials: "include",
                        body: JSON.stringify({
                          message,
                          parentResponseId: globalThis.crypto?.randomUUID?.() || Math.random().toString(36).slice(2),
                          disableSearch: false,
                          enableImageGeneration: false,
                          imageAttachments: [],
                          returnImageBytes: false,
                          returnRawGrokInXaiRequest: false,
                          fileAttachments: [],
                          enableImageStreaming: false,
                          imageGenerationCount: 1,
                          forceConcise: false,
                          toolOverrides: {},
                          enableSideBySide: false,
                          sendFinalMetadata: true,
                          isReasoning: false,
                          metadata: { request_metadata: { mode: "auto" } },
                          disableTextFollowUps: false,
                          disableArtifact: false,
                          isFromGrokFiles: false,
                          disableMemory: false,
                          forceSideBySide: false,
                          modelMode: "MODEL_MODE_AUTO",
                          isAsyncChat: false,
                          skipCancelCurrentInflightRequests: false,
                          isRegenRequest: false,
                          disableSelfHarmShortCircuit: false,
                          deviceEnvInfo: {
                            darkModeEnabled: false,
                            devicePixelRatio: 1,
                            screenWidth: 2560,
                            screenHeight: 1440,
                            viewportWidth: 1440,
                            viewportHeight: 720,
                          },
                        }),
                      });
                      if (!res.ok) {
                        return { ok: false, status: res.status, error: await res.text() };
                      }
                      return { ok: true, text: await res.text() };
                    }
                    """,
                    {"message": message},
                )
                if response.get("ok"):
                    content = _parse_ndjson_text(str(response.get("text", "")))
                    if content:
                        return content
            except Exception:
                pass
            return _dom_send_and_wait_grok(page, message)

        return self._with_page(
            start_url=_GROK_URL,
            cookie_domains=(".grok.com",),
            action=action,
        )

    def _chat_glm_cn(self, *, message: str, model: str) -> str:
        def action(context, page):
            current_url = str(getattr(page, "url", ""))
            if "/main/" not in current_url:
                page.goto(f"{_GLM_CN_URL}/main/all", wait_until="domcontentloaded", timeout=120000)
                page.wait_for_timeout(3000)
            cookie_map = self._cookie_map(context, [_GLM_CN_URL])
            self._sync_live_glm_cn_credentials(context, cookie_map)
            access_token = cookie_map.get("chatglm_token", "")
            refresh_token = cookie_map.get("chatglm_refresh_token", "")
            session = _PROVIDER_GLOBAL_SESSIONS.get(self._provider)
            persisted_state = self._load_glm_cn_session_state()
            persisted_device_id = str(persisted_state.get("device_id") or "").strip()
            isolated_tool_context = bool(getattr(self, "_tool_conversation_isolation", False))
            if session is not None:
                if persisted_device_id and not str(session.metadata.get("glm_cn_device_id") or "").strip():
                    session.metadata["glm_cn_device_id"] = persisted_device_id
            stable_device_id = None
            conversation_id = ""
            if session is not None:
                stable_device_id = str(session.metadata.get("glm_cn_device_id") or "").strip() or None
                stable_device_id = stable_device_id or persisted_device_id or None
            if stable_device_id is None:
                stable_device_id = cookie_map.get("chatglm_device_id", "")
                if not stable_device_id:
                    token_for_device = access_token or refresh_token
                    if token_for_device:
                        try:
                            payload_part = token_for_device.split(".")[1]
                            payload_b64 = payload_part + "=" * ((4 - len(payload_part) % 4) % 4)
                            token_payload = json.loads(base64.b64decode(payload_b64))
                            stable_device_id = str(token_payload.get("device_id", ""))
                        except Exception:
                            stable_device_id = ""
                stable_device_id = stable_device_id or uuid4().hex
            if session is not None:
                session.metadata["glm_cn_device_id"] = stable_device_id
                session.metadata["glm_cn_conversation_id"] = ""
            self._suppress_glm_cn_conversation_persist = isolated_tool_context
            self._persist_glm_cn_session_state(
                device_id=stable_device_id,
            )
            # Refresh if access token is missing or expired
            if refresh_token and (not access_token or _is_jwt_expired(access_token)):
                refreshed = _refresh_glm_access_token(
                    page,
                    refresh_token=refresh_token,
                    base_url=_GLM_CN_URL,
                    device_id=stable_device_id,
                )
                if refreshed:
                    access_token = refreshed
                elif not access_token:
                    raise RuntimeError(
                        "GLM China: access token expired and refresh failed. "
                        "Run `opentoken login glm cn` to re-authenticate."
                    )
            sign = _generate_glm_sign()
            assistant_id = _GLM_ASSISTANT_ID_MAP.get(model, "65940acff94777010aa6b796")

            def do_request(token: str | None):
                return page.evaluate(
                    """
                    async ({ accessToken, body, deviceId, requestId, sign, xExpGroups, timeoutMs }) => {
                      const controller = new AbortController();
                      const timeout = setTimeout(() => controller.abort("timeout"), timeoutMs);
                      const headers = {
                        "Content-Type": "application/json",
                        Accept: "text/event-stream",
                        "App-Name": "chatglm",
                        Origin: "https://chatglm.cn",
                        "X-App-Platform": "pc",
                        "X-App-Version": "0.0.1",
                        "X-App-fr": "default",
                        "X-Device-Brand": "",
                        "X-Device-Id": deviceId,
                        "X-Device-Model": "",
                        "X-Exp-Groups": xExpGroups,
                        "X-Lang": "zh",
                        "X-Nonce": sign.nonce,
                        "X-Request-Id": requestId,
                        "X-Sign": sign.sign,
                        "X-Timestamp": sign.timestamp,
                      };
                      if (accessToken) {
                        headers["Authorization"] = "Bearer " + accessToken;
                      }
                      try {
                        const res = await fetch("https://chatglm.cn/chatglm/backend-api/assistant/stream", {
                          method: "POST",
                          headers,
                          credentials: "include",
                          body: JSON.stringify(body),
                          signal: controller.signal,
                        });
                        const rawText = await res.text();
                        return { ok: res.ok, status: res.status, error: res.ok ? "" : rawText, rawText };
                      } catch (error) {
                        const message = String(error || "");
                        if (message.toLowerCase().includes("abort") || message.toLowerCase().includes("timeout")) {
                          return {
                            ok: false,
                            status: 408,
                            error: `ChatGLM API request timed out after ${timeoutMs}ms`,
                            rawText: "",
                          };
                        }
                        return {
                          ok: false,
                          status: 500,
                          error: message,
                          rawText: "",
                        };
                      } finally {
                        clearTimeout(timeout);
                      }
                    }
                    """,
                    {
                        "accessToken": token or None,
                        "body": {
                            "assistant_id": assistant_id,
                            "conversation_id": conversation_id,
                            "project_id": "",
                            "chat_type": "user_chat",
                            "meta_data": _glm_meta_data_for_model(model),
                            "messages": [{"role": "user", "content": [{"type": "text", "text": message}]}],
                        },
                        "deviceId": stable_device_id,
                        "requestId": uuid4().hex,
                        "sign": sign,
                        "xExpGroups": _GLM_X_EXP_GROUPS,
                        "timeoutMs": 120000,
                    },
                )

            try:
                response = do_request(access_token)
                status = response.get("status")
                if status == 401 and refresh_token:
                    refreshed = _refresh_glm_access_token(
                        page,
                        refresh_token=refresh_token,
                        base_url=_GLM_CN_URL,
                        device_id=stable_device_id,
                    )
                    if refreshed:
                        response = do_request(refreshed)
                        status = response.get("status")
                if not response.get("ok"):
                    error_body = response.get("error", "")
                    if status == 401:
                        raise RuntimeError(
                            "GLM China authentication failed (401). "
                            "Run `opentoken login glm cn` to re-authenticate."
                        )
                    if status == 429 or "10061" in str(error_body):
                        raise RuntimeError(
                            f"GLM China rate limited (error 10061). "
                            f"Too many requests — wait a few minutes before retrying. "
                            f"Raw: {error_body[:200]}"
                        )
                    raise RuntimeError(f"GLM China request failed: {status} {error_body[:300]}")
                raw_text = str(response.get("rawText", ""))
                error_detail = _extract_glm_error_detail(raw_text)
                if error_detail:
                    if "请登录后继续使用" in error_detail:
                        raise RuntimeError(
                            "GLM China requires a logged-in account for continued use. "
                            "Run `opentoken login glm cn` to re-authenticate."
                        )
                    raise RuntimeError(f"GLM China request returned an application error: {error_detail}")

                content, returned_conversation_id = _parse_glm_sse_response(raw_text)
                if session is not None and returned_conversation_id:
                    session.metadata["glm_cn_conversation_id"] = returned_conversation_id
                if returned_conversation_id:
                    self._persist_glm_cn_session_state(
                        device_id=stable_device_id,
                    )
                if not content:
                    raise RuntimeError("GLM China returned no text content.")
                return content
            except RuntimeError:
                return _call_with_supported_kwargs(
                    _dom_send_and_wait_glm_cn,
                    page,
                    session=session,
                    client=self,
                    message=message,
                )
            finally:
                self._suppress_glm_cn_conversation_persist = False
                if isolated_tool_context and session is not None:
                    session.metadata.pop("glm_cn_conversation_id", None)

        return self._with_page(
            start_url=f"{_GLM_CN_URL}/main/all",
            cookie_domains=(".chatglm.cn",),
            action=action,
        )

    def _stream_glm_cn(self, *, message: str, model: str) -> Iterator[str]:
        def iterator() -> Iterator[str]:
            with _provider_session_lock(self._provider):
                session = _get_or_create_browser_session(
                    provider=self._provider,
                    state_dir=self._state_dir,
                    headless=self._headless,
                )
                context = session.context
                page = session.page
                page_was_replaced = False
                try:
                    self._inject_cookie_string(context, (".chatglm.cn",))
                    if _page_is_closed(page):
                        page = context.pages[0] if getattr(context, "pages", []) else context.new_page()
                        session.page = page
                        page_was_replaced = True
                    current_url = str(getattr(page, "url", ""))
                    if "/main/" not in current_url:
                        page.goto(f"{_GLM_CN_URL}/main/all", wait_until="domcontentloaded", timeout=120000)
                        page.wait_for_timeout(3000)

                    cookie_map = self._cookie_map(context, [_GLM_CN_URL])
                    self._sync_live_glm_cn_credentials(context, cookie_map)
                    access_token = cookie_map.get("chatglm_token", "")
                    refresh_token = cookie_map.get("chatglm_refresh_token", "")
                    persisted_state = self._load_glm_cn_session_state()
                    persisted_device_id = str(persisted_state.get("device_id") or "").strip()
                    stable_device_id = str(session.metadata.get("glm_cn_device_id") or "").strip()
                    stable_device_id = stable_device_id or persisted_device_id
                    if not stable_device_id:
                        stable_device_id = cookie_map.get("chatglm_device_id", "")
                        if not stable_device_id:
                            token_for_device = access_token or refresh_token
                            if token_for_device:
                                try:
                                    payload_part = token_for_device.split(".")[1]
                                    payload_b64 = payload_part + "=" * ((4 - len(payload_part) % 4) % 4)
                                    token_payload = json.loads(base64.b64decode(payload_b64))
                                    stable_device_id = str(token_payload.get("device_id", ""))
                                except Exception:
                                    stable_device_id = ""
                        stable_device_id = stable_device_id or uuid4().hex

                    session.metadata["glm_cn_device_id"] = stable_device_id
                    session.metadata["glm_cn_conversation_id"] = ""
                    self._persist_glm_cn_session_state(device_id=stable_device_id)

                    if refresh_token and (not access_token or _is_jwt_expired(access_token)):
                        refreshed = _refresh_glm_access_token(
                            page,
                            refresh_token=refresh_token,
                            base_url=_GLM_CN_URL,
                            device_id=stable_device_id,
                        )
                        if refreshed:
                            access_token = refreshed
                        elif not access_token:
                            raise RuntimeError(
                                "GLM China: access token expired and refresh failed. "
                                "Run `opentoken login glm cn` to re-authenticate."
                            )

                    stream_error: Exception | None = None
                    for attempt in range(2):
                        try:
                            yield from _stream_glm_cn_browser_completion(
                                page,
                                session=session,
                                client=self,
                                message=message,
                                model=model,
                                access_token=access_token or None,
                                device_id=stable_device_id,
                            )
                            self._persist_glm_cn_session_state(device_id=stable_device_id)
                            return
                        except RuntimeError as exc:
                            stream_error = exc
                            if attempt == 0 and refresh_token and "401" in str(exc):
                                refreshed = _refresh_glm_access_token(
                                    page,
                                    refresh_token=refresh_token,
                                    base_url=_GLM_CN_URL,
                                    device_id=stable_device_id,
                                )
                                if refreshed:
                                    access_token = refreshed
                                    continue
                            break

                    fallback_text = _call_with_supported_kwargs(
                        _dom_send_and_wait_glm_cn,
                        page,
                        session=session,
                        client=self,
                        message=message,
                    )
                    yielded = False
                    for piece in _iter_browser_text_chunks(fallback_text):
                        yielded = True
                        yield piece
                    if not yielded and stream_error is not None:
                        raise stream_error
                    self._persist_glm_cn_session_state(device_id=stable_device_id)
                except Exception:
                    if not page_was_replaced:
                        _close_browser_session(self._provider)
                    raise

        return iterator()

    def _tool_chat_glm_cn(self, *, message: str, model: str) -> str:
        self._tool_conversation_isolation = True
        try:
            return self._chat_glm_cn(message=message, model=model)
        finally:
            self._tool_conversation_isolation = False

    def _chat_glm_intl(self, *, message: str, model: str) -> str:
        try:
            return _chat_glm_intl_api_completion(
                self._credentials,
                message=message,
                model=model,
            )
        except Exception:
            pass

        def action(_context, page):
            return _dom_send_and_wait_glm_intl(page, message)

        return self._with_page(
            start_url=_GLM_INTL_URL,
            cookie_domains=(".z.ai",),
            action=action,
        )

    def _stream_glm_intl(self, *, message: str, model: str) -> Iterator[str]:
        def iterator() -> Iterator[str]:
            yielded_any = False
            try:
                for piece in _iter_stream_with_startup_timeout(
                    lambda: _stream_glm_intl_api_completion(
                        self._credentials,
                        message=message,
                        model=model,
                    ),
                    description="GLM International API stream",
                    startup_timeout_seconds=_GLM_INTL_API_STREAM_STARTUP_TIMEOUT_SECONDS,
                ):
                    if not piece:
                        continue
                    yielded_any = True
                    yield piece
                if yielded_any:
                    return
            except Exception:
                if yielded_any:
                    raise

            with _provider_session_lock(self._provider):
                session = _get_or_create_browser_session(
                    provider=self._provider,
                    state_dir=self._state_dir,
                    headless=self._headless,
                )
                context = session.context
                page = session.page
                page_was_replaced = False
                try:
                    self._inject_cookie_string(context, (".z.ai",))
                    if _page_is_closed(page):
                        page = context.pages[0] if getattr(context, "pages", []) else context.new_page()
                        session.page = page
                        page_was_replaced = True
                    current_url = str(getattr(page, "url", ""))
                    if "z.ai" not in current_url:
                        page.goto(
                            _GLM_INTL_URL,
                            wait_until="domcontentloaded",
                            timeout=_GLM_INTL_DOM_BOOTSTRAP_TIMEOUT_MS,
                        )
                    yield from _stream_glm_intl_dom_completion(page, message=message)
                except Exception:
                    if not page_was_replaced:
                        _close_browser_session(self._provider)
                    raise

        return iterator()

    def _tool_chat_glm_intl(self, *, message: str, model: str) -> str:
        try:
            return _chat_glm_intl_api_completion(
                self._credentials,
                message=message,
                model=model,
            )
        except Exception:
            return self._chat_glm_intl(message=message, model=model)


def _is_jwt_expired(token: str, buffer_seconds: int = 60) -> bool:
    """Return True if the JWT access token is expired or about to expire."""
    import base64
    import time as _time
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return True
        payload_b64 = parts[1] + "=" * ((4 - len(parts[1]) % 4) % 4)
        payload = json.loads(base64.b64decode(payload_b64))
        exp = payload.get("exp", 0)
        return _time.time() >= exp - buffer_seconds
    except Exception:
        return True


def _refresh_glm_access_token(page, *, refresh_token: str, base_url: str, device_id: str | None = None) -> str:
    sign = _generate_glm_sign()
    stable_device_id = device_id or uuid4().hex
    response = page.evaluate(
        """
        async ({ refreshToken, baseUrl, deviceId, requestId, sign }) => {
          const res = await fetch(`${baseUrl}/chatglm/user-api/user/refresh`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${refreshToken}`,
              "App-Name": "chatglm",
              "X-App-Platform": "pc",
              "X-App-Version": "0.0.1",
              "X-Device-Id": deviceId,
              "X-Request-Id": requestId,
              "X-Sign": sign.sign,
              "X-Nonce": sign.nonce,
              "X-Timestamp": sign.timestamp,
            },
            credentials: "include",
            body: JSON.stringify({}),
          });
          if (!res.ok) {
            return { ok: false, status: res.status, error: await res.text() };
          }
          const data = await res.json();
          return {
            ok: true,
            accessToken: data?.result?.access_token ?? data?.result?.accessToken ?? data?.accessToken ?? "",
          };
        }
        """,
        {
            "refreshToken": refresh_token,
            "baseUrl": base_url,
            "deviceId": stable_device_id,
            "requestId": uuid4().hex,
            "sign": sign,
        },
    )
    if not response.get("ok") or not response.get("accessToken"):
        return ""
    return str(response.get("accessToken", ""))


def _iter_stream_with_startup_timeout(
    stream_factory,
    *,
    description: str,
    startup_timeout_seconds: float,
) -> Iterator[str]:
    result_queue: queue.Queue[tuple[str, object | None]] = queue.Queue()
    stop_event = threading.Event()
    iterator = None

    def worker() -> None:
        nonlocal iterator
        terminal_sent = False
        try:
            iterator = iter(stream_factory())
            for piece in iterator:
                if stop_event.is_set():
                    break
                result_queue.put(("piece", str(piece)))
            result_queue.put(("done", None))
            terminal_sent = True
        except BaseException as exc:
            result_queue.put(("error", exc))
            terminal_sent = True
        finally:
            close_stream = getattr(iterator, "close", None) if iterator is not None else None
            if callable(close_stream):
                try:
                    close_stream()
                except BaseException as exc:
                    if not terminal_sent:
                        result_queue.put(("error", exc))
                        terminal_sent = True
            if not terminal_sent:
                result_queue.put(("done", None))

    threading.Thread(
        target=worker,
        name=f"opentoken-stream-startup-{description.lower().replace(' ', '-')}",
        daemon=True,
    ).start()

    saw_visible_piece = False
    deadline = time.monotonic() + max(0.0, startup_timeout_seconds)
    try:
        while True:
            timeout = None
            if not saw_visible_piece:
                timeout = max(0.0, deadline - time.monotonic())
            try:
                kind, payload = result_queue.get(timeout=timeout)
            except queue.Empty as exc:
                stop_event.set()
                raise RuntimeError(
                    f"{description} startup timed out after {startup_timeout_seconds:g}s"
                ) from exc
            if kind == "piece":
                piece = str(payload or "")
                if not piece:
                    continue
                saw_visible_piece = True
                yield piece
                continue
            if kind == "done":
                return
            if isinstance(payload, RuntimeError):
                raise payload
            if isinstance(payload, BaseException):
                raise RuntimeError(str(payload)) from payload
            raise RuntimeError(f"{description} failed without an exception payload.")
    finally:
        stop_event.set()


def _generate_glm_sign() -> dict[str, str]:
    now = str(int(time.time() * 1000))
    digits = [int(char) for char in now]
    replacement = (sum(digits) - digits[-2]) % 10
    timestamp = f"{now[:-2]}{replacement}{now[-1]}"
    nonce = uuid4().hex
    sign = hashlib.md5(f"{timestamp}-{nonce}-{_GLM_SIGN_SECRET}".encode("utf-8")).hexdigest()
    return {"timestamp": timestamp, "nonce": nonce, "sign": sign}


def _parse_glm_sse_response(raw_text: str) -> tuple[str, str]:
    """Parse GLM CN SSE response.

    Returns (content, conversation_id). Accepts both parts[].content[].text and
    top-level text/content/delta fallback shapes.
    """
    last_text = ""
    conversation_id = ""
    for line in raw_text.splitlines():
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if not data_str or data_str == "[DONE]":
            continue
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        cid = data.get("conversation_id")
        if isinstance(cid, str) and cid:
            conversation_id = cid
        parts = data.get("parts", [])
        for part in parts:
            if isinstance(part, dict):
                for item in part.get("content", []):
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text", "")
                        if text:
                            last_text = text
        fallback_text = data.get("text") or data.get("content") or data.get("delta")
        if isinstance(fallback_text, str) and fallback_text:
            last_text = fallback_text
    return last_text.strip(), conversation_id


def _extract_glm_error_detail(raw_text: str) -> str | None:
    stripped = raw_text.strip()
    if not stripped or stripped.startswith("data:"):
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    message = str(payload.get("message", "")).strip()
    return message or None


def _parse_cookie_string(cookie_string: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in cookie_string.split(";"):
        if "=" not in item:
            continue
        name, value = item.split("=", 1)
        name = name.strip()
        value = value.strip()
        if name:
            parsed[name] = value
    return parsed


def _decode_jwt_payload(token: str) -> dict[str, object] | None:
    if not token:
        return None
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        payload_b64 = parts[1] + "=" * ((4 - len(parts[1]) % 4) % 4)
        payload = json.loads(base64.b64decode(payload_b64))
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _is_glm_guest_token_value(token: str) -> bool:
    payload = _decode_jwt_payload(token)
    if payload is None:
        return True
    return bool(payload.get("is_guest", False))


def _glm_cookie_map_has_real_login(cookie_map: dict[str, str]) -> bool:
    token = str(cookie_map.get("chatglm_token", "") or "").strip()
    user_id = str(cookie_map.get("chatglm_user_id", "") or "").strip()
    return bool(token and user_id and not _is_glm_guest_token_value(token))


def _get_or_create_browser_session(*, provider: str, state_dir, headless: bool) -> "_ProviderBrowserSession":
    # Single global session per provider, protected by _PROVIDER_SESSION_LOCKS.
    # The caller must already hold the provider lock before calling this.
    # Playwright sync objects are thread-affine, so a cached session may only
    # be reused from the same owner thread that created it.
    with _PROVIDER_GLOBAL_SESSIONS_GUARD:
        session = _PROVIDER_GLOBAL_SESSIONS.get(provider)

    if (
        session is not None
        and session.headless == headless
        and session.owner_thread is threading.current_thread()
        and not _page_is_closed(session.page)
    ):
        return session

    # Close stale session
    if session is not None:
        with _PROVIDER_GLOBAL_SESSIONS_GUARD:
            _PROVIDER_GLOBAL_SESSIONS.pop(provider, None)
        try:
            session.context.close()
        except Exception:
            pass
        try:
            session.manager.__exit__(None, None, None)
        except Exception:
            pass

    sync_playwright = require_sync_playwright()
    browser_state_dir = prepare_browser_state_dir(state_dir, provider)
    launch_state_dir = _prepare_browser_launch_dir(browser_state_dir, headless=headless)
    manager, context = _launch_browser_context_with_profile_recovery(
        sync_playwright=sync_playwright,
        browser_state_dir=launch_state_dir,
        headless=headless,
    )
    page = context.pages[0] if getattr(context, "pages", []) else context.new_page()
    new_session = _ProviderBrowserSession(
        manager=manager,
        context=context,
        page=page,
        headless=headless,
        owner_thread=threading.current_thread(),
        metadata={},
    )
    with _PROVIDER_GLOBAL_SESSIONS_GUARD:
        _PROVIDER_GLOBAL_SESSIONS[provider] = new_session
    return new_session


def _sweep_stale_recovery_dirs(browser_state_dir, *, max_age_seconds: float) -> None:
    parent = browser_state_dir.parent
    prefix = f"{browser_state_dir.name}-recovery-"
    now = time.time()
    try:
        children = list(parent.iterdir())
    except OSError:
        return
    for child in children:
        if not child.name.startswith(prefix):
            continue
        try:
            if not child.is_dir():
                continue
            age = now - child.stat().st_mtime
        except OSError:
            continue
        if age < max_age_seconds:
            continue
        shutil.rmtree(child, ignore_errors=True)


def _launch_browser_context_with_profile_recovery(*, sync_playwright, browser_state_dir, headless: bool):
    manager = sync_playwright()
    manager_entered = False
    try:
        playwright = manager.__enter__()
        manager_entered = True
        try:
            context = playwright.chromium.launch_persistent_context(
                str(browser_state_dir),
                headless=headless,
            )
            return manager, context
        except Exception as exc:
            message = str(exc)
            if (
                "Only one copy of Firefox can be open at a time" not in message
                and "copy of Firefox is already open" not in message
            ):
                raise
            if manager_entered:
                try:
                    manager.__exit__(None, None, None)
                except Exception:
                    pass
                manager_entered = False

            # Sweep stale recovery dirs from previous retries before creating a
            # new one. Without this, every Firefox-already-open retry accumulates
            # a `{name}-recovery-{ms}` sibling forever and eventually fills the
            # disk. We keep anything younger than an hour (might be in active use
            # by a concurrent process) and delete the rest.
            _sweep_stale_recovery_dirs(browser_state_dir, max_age_seconds=3600)
            recovery_dir = browser_state_dir.parent / f"{browser_state_dir.name}-recovery-{int(time.time() * 1000)}"
            recovery_dir.mkdir(parents=True, exist_ok=True)

            recovery_manager = sync_playwright()
            recovery_entered = False
            try:
                recovery_playwright = recovery_manager.__enter__()
                recovery_entered = True
                context = recovery_playwright.chromium.launch_persistent_context(
                    str(recovery_dir),
                    headless=headless,
                )
                return recovery_manager, context
            except Exception:
                if recovery_entered:
                    try:
                        recovery_manager.__exit__(None, None, None)
                    except Exception:
                        pass
                raise
    except Exception:
        if manager_entered:
            try:
                manager.__exit__(None, None, None)
            except Exception:
                pass
        raise


def _prepare_browser_launch_dir(browser_state_dir, *, headless: bool):
    if not headless:
        return browser_state_dir

    runtime_dir = browser_state_dir.parent / f"{browser_state_dir.name}-runtime"
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir, ignore_errors=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    if not browser_state_dir.exists():
        return runtime_dir

    for child in browser_state_dir.iterdir():
        name = child.name.lower()
        if "lock" in name or name.startswith(".parentlock") or name.startswith("singleton"):
            continue
        destination = runtime_dir / child.name
        try:
            if child.is_dir():
                shutil.copytree(child, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(child, destination)
        except Exception:
            continue
    return runtime_dir


def _close_browser_session(provider: str) -> None:
    with _PROVIDER_GLOBAL_SESSIONS_GUARD:
        session = _PROVIDER_GLOBAL_SESSIONS.pop(provider, None)
    if session is not None:
        try:
            session.context.close()
        except Exception:
            pass
        try:
            session.manager.__exit__(None, None, None)
        except Exception:
            pass


def _close_all_browser_sessions() -> None:
    with _PROVIDER_GLOBAL_SESSIONS_GUARD:
        sessions = list(_PROVIDER_GLOBAL_SESSIONS.values())
        _PROVIDER_GLOBAL_SESSIONS.clear()
    for session in sessions:
        try:
            session.context.close()
        except Exception:
            pass
        try:
            session.manager.__exit__(None, None, None)
        except Exception:
            pass


def _page_is_closed(page) -> bool:
    if page is None:
        return True
    try:
        is_closed = getattr(page, "is_closed", False)
        if callable(is_closed):
            return bool(is_closed())
        return bool(is_closed)
    except Exception:
        error_msg = str(getattr(page, "url", "")).lower()
        if "closed" in error_msg or "crashed" in error_msg:
            return True
        return False


def _prefer_dom_first_doubao_stream(model: str) -> bool:
    _ = str(model or "").strip().lower()
    return False


def _stop_doubao_dom_generation(page) -> None:
    try:
        page.evaluate(
            """
            () => {
              const visible = (node) => !!node && !!node.offsetParent;
              const candidates = Array.from(document.querySelectorAll('button, [role="button"]'));
              const target = candidates.find((node) => {
                if (!visible(node)) {
                  return false;
                }
                if (node.disabled || node.getAttribute('aria-disabled') === 'true') {
                  return false;
                }
                const label = `${node.getAttribute('aria-label') || ''} ${node.textContent || ''} ${node.className || ''}`.toLowerCase();
                return label.includes('stop') || label.includes('停止');
              });
              if (!target) {
                return false;
              }
              target.click();
              return true;
            }
            """
        )
    except Exception:
        pass


def _detach_page_listener(page, event_name: str, callback) -> None:
    remover = getattr(page, "remove_listener", None) or getattr(page, "off", None)
    if callable(remover):
        try:
            remover(event_name, callback)
        except Exception:
            pass


@contextmanager
def _provider_session_lock(provider: str):
    with _PROVIDER_SESSION_LOCKS_GUARD:
        lock = _PROVIDER_SESSION_LOCKS.setdefault(provider, threading.Lock())
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


def _call_with_supported_kwargs(func, *args, **kwargs):
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return func(*args, **kwargs)
    supported_kwargs = {name: value for name, value in kwargs.items() if name in parameters}
    return func(*args, **supported_kwargs)


def _iter_browser_text_chunks(content: str, *, max_chunk_len: int = 24) -> Iterator[str]:
    text = content or ""
    if not text:
        return
    wordish_parts = re.findall(r"\S+\s*", text)
    if len(wordish_parts) <= 1:
        for i in range(0, len(text), max_chunk_len):
            yield text[i : i + max_chunk_len]
        return
    current = ""
    for part in wordish_parts:
        if current and len(current) + len(part) > max_chunk_len:
            yield current
            current = part
        else:
            current += part
    if current:
        yield current


atexit.register(_close_all_browser_sessions)


def _parse_sse_text(payload: str) -> str:
    chunks: list[str] = []
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("data: "):
            line = line[6:].strip()
        elif line.startswith("data:"):
            line = line[5:].strip()
        if line == "[DONE]":
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        fragment = _extract_text_fragment(parsed)
        if fragment:
            chunks.append(fragment)
    return "".join(chunks)


def _resolve_qwen_cn_request_auth(
    credentials: ProviderCredentialRecord,
    cookie_map: dict[str, str],
) -> dict[str, str]:
    live_xsrf = str(cookie_map.get("XSRF-TOKEN", "")).strip()
    live_ut = str(cookie_map.get("b-user-id", "")).strip()
    saved_xsrf = str(credentials.metadata.get("xsrf_token", "")).strip()
    saved_ut = str(credentials.metadata.get("ut", "")).strip()
    saved_device_id = str(credentials.metadata.get("device_id", "")).strip()

    ut = live_ut or saved_ut
    return {
        "xsrf_token": live_xsrf or saved_xsrf,
        "ut": ut,
        "device_id": saved_device_id or ut or ("random-" + uuid4().hex[:12]),
    }


def _fetch_qwen_cn_browser_completion(
    page,
    *,
    message: str,
    model: str,
    auth: dict[str, str],
) -> str:
    xsrf_token = str(auth.get("xsrf_token", "")).strip()
    ut = str(auth.get("ut", "")).strip()
    device_id = str(auth.get("device_id", "")).strip() or ut
    if not xsrf_token or not ut:
        raise RuntimeError(
            "Qwen China authentication metadata is incomplete. "
            "Run `opentoken login qwen-cn` again."
        )

    timestamp = int(time.time() * 1000)
    nonce = uuid4().hex[:12]
    session_id = uuid4().hex
    response = page.evaluate(
        """
        async ({ baseUrl, sessionId, model, message, ut, xsrfToken, deviceId, nonce, timestamp }) => {
          const params = new URLSearchParams({
            biz_id: "ai_qwen",
            chat_client: "h5",
            device: "pc",
            fr: "pc",
            pr: "qwen",
            nonce,
            timestamp: String(timestamp),
            ut,
          });
          const bodyObj = {
            model,
            messages: [
              {
                content: message,
                mime_type: "text/plain",
                meta_data: {
                  ori_query: message,
                },
              },
            ],
            session_id: sessionId,
            parent_req_id: "0",
            deep_search: "0",
            req_id: "req-" + Math.random().toString(36).slice(2),
            scene: "chat",
            sub_scene: "chat",
            temporary: false,
            from: "default",
            scene_param: "first_turn",
            chat_client: "h5",
            client_tm: String(timestamp),
            protocol_version: "v2",
            biz_id: "ai_qwen",
          };
          const res = await fetch(`${baseUrl}/api/v2/chat?${params.toString()}`, {
            method: "POST",
            credentials: "include",
            headers: {
              "Content-Type": "application/json",
              Accept: "text/event-stream, text/plain, */*",
              "x-xsrf-token": xsrfToken,
              "x-deviceid": deviceId,
              "x-platform": "pc_tongyi",
            },
            body: JSON.stringify(bodyObj),
          });
          return {
            ok: res.ok,
            status: res.status,
            text: await res.text(),
          };
        }
        """,
        {
            "baseUrl": _QWEN_CN_BASE_URL,
            "sessionId": session_id,
            "model": model,
            "message": message,
            "ut": ut,
            "xsrfToken": xsrf_token,
            "deviceId": device_id,
            "nonce": nonce,
            "timestamp": timestamp,
        },
    )
    payload = str(response.get("text", ""))
    content = _parse_qwen_cn_response_text(payload).strip()
    if content:
        return content

    status = int(response.get("status", 0) or 0)
    if not response.get("ok"):
        if status in {401, 403}:
            raise RuntimeError(
                "Qwen China authentication failed. "
                "Run `opentoken login qwen-cn` again."
            )
        raise RuntimeError(f"Qwen China request failed: {status} {payload[:500]}")
    raise RuntimeError("Qwen China request returned no text content.")


def _parse_qwen_cn_response_text(payload: str) -> str:
    best_full_text = ""
    delta_chunks: list[str] = []

    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line or line == "[DONE]":
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue

        candidate, is_full_text = _extract_qwen_cn_text_fragment(parsed)
        candidate = str(candidate).strip()
        if not candidate:
            continue

        if is_full_text:
            if not best_full_text or candidate.startswith(best_full_text) or len(candidate) >= len(
                best_full_text
            ):
                best_full_text = candidate
            continue

        if not delta_chunks or delta_chunks[-1] != candidate:
            delta_chunks.append(candidate)

    if best_full_text:
        return best_full_text
    return "".join(delta_chunks).strip()


def _extract_qwen_cn_text_fragment(payload: object) -> tuple[str, bool]:
    if not isinstance(payload, dict):
        return "", False

    data = payload.get("data")
    if isinstance(data, dict):
        messages = data.get("messages")
        if isinstance(messages, list):
            for item in reversed(messages):
                if not isinstance(item, dict):
                    continue
                for key in ("content", "text"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        return value, True
                value = item.get("delta")
                if isinstance(value, str) and value.strip():
                    return value, False
        for key in ("text", "content"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value, True
        value = data.get("delta")
        if isinstance(value, str) and value.strip():
            return value, False

    communication = payload.get("communication")
    if isinstance(communication, dict):
        for key in ("text", "content"):
            value = communication.get(key)
            if isinstance(value, str) and value.strip():
                return value, True
        value = communication.get("delta")
        if isinstance(value, str) and value.strip():
            return value, False

    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        delta = choices[0].get("delta")
        if isinstance(delta, dict):
            for key in ("content", "text"):
                value = delta.get(key)
                if isinstance(value, str) and value.strip():
                    return value, False

    for key in ("text", "content"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value, True
    value = payload.get("delta")
    if isinstance(value, str) and value.strip():
        return value, False
    return "", False


def _extract_qwen_cn_candidate_text(
    prompt: str,
    *,
    markdown_texts: list[str],
    message_texts: list[str],
) -> str:
    prompt_text = _clean_dom_text(prompt)
    cleaned_markdown = [_clean_dom_text(text) for text in markdown_texts]
    for text in reversed(cleaned_markdown):
        if text and text not in {"向千问提问", prompt_text}:
            return text

    cleaned_messages = [_clean_dom_text(text) for text in message_texts]
    for text in reversed(cleaned_messages):
        if not text:
            continue
        if prompt_text and text == prompt_text:
            continue
        if prompt_text and prompt_text in text:
            suffix = text.split(prompt_text, 1)[1].strip()
            if suffix:
                return suffix
            continue
        return text
    return ""


def _extract_doubao_dom_candidate_text(
    prompt: str,
    *,
    markdown_texts: list[str],
    message_texts: list[str],
) -> str:
    return _extract_qwen_cn_candidate_text(
        prompt,
        markdown_texts=markdown_texts,
        message_texts=message_texts,
    )


def _advance_text_stream_state(current: str, candidate: str) -> tuple[str, str]:
    if not candidate:
        return "", current
    if not current:
        return candidate, candidate
    if candidate.startswith(current):
        suffix = candidate[len(current) :]
        return suffix, candidate
    if current.startswith(candidate):
        return "", current
    relation, raw_boundary = _normalized_snapshot_relation(current, candidate)
    if relation == "candidate_extends_current":
        return candidate[raw_boundary:], candidate
    if relation == "equivalent":
        return "", candidate
    if relation == "current_extends_candidate":
        return "", current
    return candidate, current + candidate


def _advance_snapshot_stream_state(current: str, candidate: str) -> tuple[str, str]:
    suffix, updated = _advance_text_stream_state(current, candidate)
    if current and suffix == candidate and updated == current + candidate:
        return candidate, candidate
    return suffix, updated


def _is_reformatted_snapshot_duplicate(current: str, candidate: str) -> bool:
    relation, _raw_boundary = _normalized_snapshot_relation(current, candidate)
    if relation == "equivalent":
        return True
    if relation == "candidate_extends_current":
        return False
    if relation == "current_extends_candidate":
        return True
    current_fingerprint = _normalize_snapshot_fingerprint(current)
    candidate_fingerprint = _normalize_snapshot_fingerprint(candidate)
    if not current_fingerprint or not candidate_fingerprint:
        return False
    shorter, longer = sorted(
        (current_fingerprint, candidate_fingerprint),
        key=len,
    )
    if len(shorter) < 32:
        return False
    if shorter == longer:
        return True
    return shorter in longer and (len(shorter) / len(longer)) >= 0.92


def _normalize_snapshot_fingerprint(text: str) -> str:
    collapsed, _boundaries = _normalize_snapshot_with_boundaries(text)
    return collapsed


def _normalized_snapshot_relation(current: str, candidate: str) -> tuple[str | None, int]:
    current_fingerprint, _current_boundaries = _normalize_snapshot_with_boundaries(current)
    candidate_fingerprint, candidate_boundaries = _normalize_snapshot_with_boundaries(candidate)
    if not current_fingerprint or not candidate_fingerprint:
        return None, 0
    if candidate_fingerprint == current_fingerprint:
        return "equivalent", len(candidate)
    if candidate_fingerprint.startswith(current_fingerprint):
        return "candidate_extends_current", candidate_boundaries[len(current_fingerprint)]
    if current_fingerprint.startswith(candidate_fingerprint):
        return "current_extends_candidate", len(candidate)
    return None, 0


def _normalize_snapshot_with_boundaries(text: str) -> tuple[str, list[int]]:
    raw = str(text or "")
    normalized_parts: list[str] = []
    boundaries: list[int] = [0]
    index = 0
    while index < len(raw):
        char = raw[index]
        if char.isspace() or char in "*_`~":
            index += 1
            continue
        marker_end = _consume_snapshot_list_marker(raw, index, normalized_parts[-1] if normalized_parts else "")
        if marker_end is not None:
            index = marker_end
            continue
        normalized_parts.append(char)
        boundaries.append(index + 1)
        index += 1
    return "".join(normalized_parts), boundaries


def _consume_snapshot_list_marker(text: str, index: int, previous_normalized_char: str) -> int | None:
    if previous_normalized_char and previous_normalized_char not in "。！？.!?:：;；":
        return None
    end = index
    while end < len(text) and _is_snapshot_digit(text[end]):
        end += 1
    if end == index:
        return None
    if end >= len(text) or text[end] not in ".、．)）":
        return None
    return end + 1


def _is_snapshot_digit(char: str) -> bool:
    return ("0" <= char <= "9") or ("０" <= char <= "９")


def _clean_dom_text(value: object) -> str:
    text = str(value or "")
    return " ".join(text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "").split())


def _set_visible_textarea_value(page, selectors: list[str], value: str) -> bool:
    try:
        return bool(
            page.evaluate(
                """
                ({ selectors, value }) => {
                  const visible = (node) => !!node && (node.offsetParent !== null || node === document.activeElement);
                  for (const selector of selectors) {
                    const candidate = document.querySelector(selector);
                    if (!visible(candidate)) {
                      continue;
                    }
                    candidate.focus();
                    candidate.value = value;
                    candidate.dispatchEvent(new Event("input", { bubbles: true }));
                    candidate.dispatchEvent(new Event("change", { bubbles: true }));
                    return true;
                  }
                  return false;
                }
                """,
                {"selectors": selectors, "value": value},
            )
        )
    except Exception:
        return False


def _is_glm_retryable_dom_error(error_detail: str) -> bool:
    text = str(error_detail or "").strip().lower()
    if not text:
        return False
    return any(
        fragment in text
        for fragment in (
            "请求过于频繁",
            "请稍后再试",
            "too many requests",
            "rate limit",
            "10061",
        )
    )


def _dom_send_and_wait_doubao(page, session: _ProviderBrowserSession, client: CamoufoxProviderClient, message: str, model: str) -> str:
    for attempt in range(2):
        _select_doubao_model(page, model)

        composer = page.locator(
            'textarea[placeholder="发消息..."], textarea.semi-input-textarea'
        ).first
        composer.wait_for(timeout=120000)
        composer_focused = False
        try:
            composer.click(timeout=5000, force=True)
            composer_focused = True
        except TypeError:
            try:
                composer.click(timeout=5000)
                composer_focused = True
            except Exception:
                composer_focused = False
        except Exception:
            composer_focused = False
        page.wait_for_timeout(250)

        typed = False
        prefer_fill = "\n" in message or "\r" in message
        if composer_focused and not prefer_fill:
            try:
                page.keyboard.press("Meta+A")
                page.keyboard.press("Backspace")
                page.wait_for_timeout(150)
                page.keyboard.type(message, delay=55)
                typed = True
            except Exception:
                typed = False
        if not typed:
            try:
                composer.fill(message)
            except Exception:
                _set_visible_textarea_value(
                    page,
                    [
                        'textarea[placeholder="发消息..."]',
                        "textarea.semi-input-textarea",
                        "textarea",
                    ],
                    message,
                )
        page.wait_for_timeout(500)

        send_button = None
        try:
            send_button = page.locator(
                'button[class*="g-send-msg-btn-bg"], #input-engine-container button'
            ).first
            send_button.wait_for(timeout=30000)
        except Exception:
            send_button = None

        captured_response = None

        def handle_response(response):
            nonlocal captured_response
            if captured_response is not None:
                return
            if not _is_doubao_chat_completion_response(response, message):
                return
            headers = {}
            try:
                headers = response.headers
            except Exception:
                headers = {}
            content_type = str(headers.get("content-type", "")).lower()
            if content_type and "text/event-stream" not in content_type:
                return
            captured_response = response

        page.on("response", handle_response)
        try:
            try:
                clicked = bool(
                    page.evaluate(
                        """
                        () => {
                          const visible = (node) => !!node && !!node.offsetParent;
                          const candidates = Array.from(
                            document.querySelectorAll('#input-engine-container button, button')
                          ).filter((node) => {
                            if (!visible(node)) {
                              return false;
                            }
                            if (node.disabled || node.getAttribute('aria-disabled') === 'true') {
                              return false;
                            }
                            return true;
                          });
                          const preferred = candidates.find((node) =>
                            String(node.className || '').includes('g-send-msg-btn-bg')
                          ) || candidates.at(-1);
                          if (!preferred) {
                            return false;
                          }
                          preferred.click();
                          return true;
                        }
                        """
                    )
                )
                if not clicked:
                    raise RuntimeError("Doubao send button not found")
            except Exception:
                try:
                    if send_button is None:
                        raise RuntimeError("send button locator unavailable")
                    send_button.click(timeout=5000)
                except Exception:
                    page.keyboard.press("Enter")

            deadline = time.time() + 120
            while time.time() < deadline:
                if captured_response is not None:
                    break
                page.wait_for_timeout(250)
            if captured_response is None:
                raise RuntimeError("Doubao response listener timed out waiting for chat/completion.")
        finally:
            _detach_page_listener(page, "response", handle_response)

        response = captured_response
        if response is None:
            raise RuntimeError("Doubao response listener completed without a response object.")
        payload = response.text()
        conversation_id = _extract_doubao_conversation_id(payload)
        if conversation_id:
            session.metadata["doubao_conversation_id"] = conversation_id
            client._persist_doubao_conversation_id(conversation_id)
        content = _parse_doubao_response_text(payload).strip()
        if content:
            return content

        error_detail = _extract_doubao_stream_error(payload)
        if error_detail:
            raise RuntimeError(f"Doubao DOM send failed: {error_detail}")
        if attempt == 0:
            page.goto(_DOUBAO_URL, wait_until="domcontentloaded", timeout=120000)
            page.wait_for_timeout(5000)
            continue
        raise RuntimeError("Doubao DOM send completed but returned no text content.")

    raise RuntimeError("Doubao DOM send retry budget exhausted.")


def _fetch_doubao_browser_completion(
    page,
    *,
    session: _ProviderBrowserSession,
    client: CamoufoxProviderClient,
    message: str,
    model: str,
) -> str:
    conversation_id = str(session.metadata.get("doubao_conversation_id") or "").strip() or "0"
    request_body = {
        "messages": [
            {
                "content": json.dumps({"text": message}),
                "content_type": 2001,
                "attachments": [],
                "references": [],
            }
        ],
        "completion_option": {
            "is_regen": False,
            "with_suggest": True,
            "need_create_conversation": conversation_id == "0",
            "launch_stage": 1,
            "is_replace": False,
            "is_delete": False,
            "message_from": 0,
            "event_id": "0",
        },
        "conversation_id": conversation_id,
        "local_conversation_id": f"local_16{str(int(time.time() * 1000))[-14:]}",
        "local_message_id": str(uuid4()),
    }
    response = page.evaluate(
        """
        async ({ body, params }) => {
          const paramsObj = new URLSearchParams(params);
          const controller = new AbortController();
          const timeout = setTimeout(() => controller.abort("timeout"), 120000);
          try {
            const res = await fetch(`https://www.doubao.com/samantha/chat/completion?${paramsObj.toString()}`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                Accept: "text/event-stream",
                Referer: "https://www.doubao.com/chat/",
                Origin: "https://www.doubao.com",
                "Agw-js-conv": "str",
              },
              body: JSON.stringify(body),
              signal: controller.signal,
            });
            return { ok: res.ok, status: res.status, text: await res.text() };
          } catch (error) {
            return { ok: false, status: 599, text: String(error) };
          } finally {
            clearTimeout(timeout);
          }
        }
        """,
        {
            "body": request_body,
            "params": {
                key: str(value)
                for key, value in session.metadata.get("doubao_request_params", {}).items()
            },
        },
    )
    payload = str(response.get("text", ""))
    returned_conversation_id = _extract_doubao_conversation_id(payload)
    if returned_conversation_id:
        session.metadata["doubao_conversation_id"] = returned_conversation_id
        client._persist_doubao_conversation_id(returned_conversation_id)
    content = _parse_doubao_response_text(payload).strip()
    if content:
        return content
    status = response.get("status")
    if not response.get("ok"):
        raise RuntimeError(f"Doubao browser fetch failed: {status} {payload[:500]}")
    error_detail = _extract_doubao_stream_error(payload)
    if error_detail:
        raise RuntimeError(f"Doubao browser fetch failed: {error_detail}")
    raise RuntimeError("Doubao browser fetch returned no text content.")


def _stream_doubao_browser_completion(
    page,
    *,
    session: _ProviderBrowserSession,
    client: CamoufoxProviderClient,
    message: str,
    model: str,
    poll_interval_seconds: float = 0.1,
    timeout_seconds: float = 120.0,
    startup_timeout_seconds: float | None = None,
) -> Iterator[str]:
    conversation_id = str(session.metadata.get("doubao_conversation_id") or "").strip() or "0"
    request_body = {
        "messages": [
            {
                "content": json.dumps({"text": message}),
                "content_type": 2001,
                "attachments": [],
                "references": [],
            }
        ],
        "completion_option": {
            "is_regen": False,
            "with_suggest": True,
            "need_create_conversation": conversation_id == "0",
            "launch_stage": 1,
            "is_replace": False,
            "is_delete": False,
            "message_from": 0,
            "event_id": "0",
        },
        "conversation_id": conversation_id,
        "local_conversation_id": f"local_16{str(int(time.time() * 1000))[-14:]}",
        "local_message_id": str(uuid4()),
    }
    stream_id = f"__opentoken_doubao_stream_{uuid4().hex}"
    started = page.evaluate(
        """
        ({ streamId, body, params }) => {
          const controller = new AbortController();
          const state = {
            lines: [],
            done: false,
            error: "",
          };
          state.abort = (reason) => {
            try {
              controller.abort(reason || "opentoken-cleanup");
            } catch (_error) {}
          };
          window[streamId] = state;
          const paramsObj = new URLSearchParams(params);
          const timeout = setTimeout(() => controller.abort("timeout"), 120000);
          const appendLine = (value) => {
            if (typeof value !== "string") {
              return;
            }
            state.lines.push(value);
          };
          const finalize = (error) => {
            if (error) {
              state.error = String(error);
            }
            state.done = true;
            clearTimeout(timeout);
          };
          (async () => {
            try {
              const res = await fetch(`https://www.doubao.com/samantha/chat/completion?${paramsObj.toString()}`, {
                method: "POST",
                headers: {
                  "Content-Type": "application/json",
                  Accept: "text/event-stream",
                  Referer: "https://www.doubao.com/chat/",
                  Origin: "https://www.doubao.com",
                  "Agw-js-conv": "str",
                },
                body: JSON.stringify(body),
                signal: controller.signal,
              });
              if (!res.ok) {
                finalize(`${res.status} ${await res.text()}`);
                return;
              }
              const reader = res.body?.getReader();
              if (!reader) {
                finalize("No response body");
                return;
              }
              const decoder = new TextDecoder();
              let buffer = "";
              while (true) {
                const { done, value } = await reader.read();
                if (done) {
                  break;
                }
                buffer += decoder.decode(value, { stream: true });
                while (true) {
                  const newlineIndex = buffer.indexOf("\\n");
                  if (newlineIndex === -1) {
                    break;
                  }
                  let line = buffer.slice(0, newlineIndex);
                  buffer = buffer.slice(newlineIndex + 1);
                  if (line.endsWith("\\r")) {
                    line = line.slice(0, -1);
                  }
                  appendLine(line);
                }
              }
              if (buffer) {
                appendLine(buffer);
              }
              finalize("");
            } catch (error) {
              finalize(error);
            }
          })();
          return { ok: true };
        }
        """,
        {
            "streamId": stream_id,
            "body": request_body,
            "params": {
                key: str(value)
                for key, value in session.metadata.get("doubao_request_params", {}).items()
            },
        },
    )
    if not started.get("ok"):
        raise RuntimeError("Doubao browser stream failed to start.")

    captured_lines: list[str] = []
    current_event: str | None = None
    current_data: str | None = None
    deadline = time.time() + timeout_seconds
    startup_deadline = (
        time.time() + max(0.0, startup_timeout_seconds)
        if startup_timeout_seconds is not None
        else None
    )
    emitted = ""
    completed = False
    saw_visible_piece = False

    def emit_incremental_chunks(
        chunks: list[str],
        *,
        require_prefix_match: bool = False,
    ) -> Iterator[str]:
        nonlocal emitted, saw_visible_piece
        for chunk in chunks:
            if not chunk:
                continue
            relation, _raw_boundary = _normalized_snapshot_relation(emitted, chunk)
            if (
                require_prefix_match
                and emitted
                and not chunk.startswith(emitted)
                and relation not in {"candidate_extends_current", "equivalent"}
            ):
                continue
            if emitted and not chunk.startswith(emitted) and _is_reformatted_snapshot_duplicate(emitted, chunk):
                if relation == "equivalent":
                    emitted = chunk
                continue
            suffix, emitted = _advance_text_stream_state(emitted, chunk)
            if suffix:
                saw_visible_piece = True
                yield suffix

    try:
        while time.time() < deadline:
            snapshot = page.evaluate(
                """
                ({ streamId }) => {
                  const state = window[streamId];
                  if (!state) {
                    return { lines: [], done: true, error: "Doubao browser stream state missing" };
                  }
                  const lines = state.lines.splice(0, state.lines.length);
                  return { lines, done: !!state.done, error: state.error || "" };
                }
                """,
                {"streamId": stream_id},
            )
            for raw_line in list(snapshot.get("lines", [])):
                normalized_lines = str(raw_line).splitlines() or [""]
                for normalized_line in normalized_lines:
                    if normalized_line:
                        captured_lines.append(normalized_line)
                    line = str(normalized_line).strip()
                    if not line:
                        if current_event and current_data:
                            try:
                                parsed = json.loads(current_data)
                            except json.JSONDecodeError:
                                parsed = None
                            if isinstance(parsed, dict):
                                conversation = _extract_doubao_conversation_id(
                                    f'event: {current_event}\ndata: {current_data}\n'
                                )
                                if conversation:
                                    session.metadata["doubao_conversation_id"] = conversation
                                    client._persist_doubao_conversation_id(conversation)
                                yield from emit_incremental_chunks(
                                    _extract_doubao_chunks_from_event(current_event, parsed),
                                    require_prefix_match=current_event == "STREAM_MSG_NOTIFY",
                                )
                        current_event = None
                        current_data = None
                        continue
                    if line.startswith("id:") and " event: " in line and " data: " in line:
                        single = line.split(" event: ", 1)[1]
                        event_name, data = single.split(" data: ", 1)
                        try:
                            parsed = json.loads(data)
                        except json.JSONDecodeError:
                            parsed = None
                        if isinstance(parsed, dict):
                            conversation = _extract_doubao_conversation_id(
                                f'event: {event_name.strip()}\ndata: {data}\n'
                            )
                            if conversation:
                                session.metadata["doubao_conversation_id"] = conversation
                                client._persist_doubao_conversation_id(conversation)
                            event_name = event_name.strip()
                            yield from emit_incremental_chunks(
                                _extract_doubao_chunks_from_event(event_name, parsed),
                                require_prefix_match=event_name == "STREAM_MSG_NOTIFY",
                            )
                        continue
                    data_line = line[6:].strip() if line.startswith("data: ") else line
                    samantha_chunks = _extract_samantha_chunks(data_line)
                    if samantha_chunks:
                        conversation = _extract_doubao_conversation_id(f"data: {data_line}\n")
                        if conversation:
                            session.metadata["doubao_conversation_id"] = conversation
                            client._persist_doubao_conversation_id(conversation)
                        yield from emit_incremental_chunks(
                            samantha_chunks,
                            require_prefix_match=True,
                        )
                        continue
                    if line.startswith("event: "):
                        current_event = line[7:].strip()
                        continue
                    if line.startswith("data: "):
                        current_data = line[6:].strip()
            if not saw_visible_piece and captured_lines:
                partial_payload = "\n".join(captured_lines)
                _parse_doubao_response_text(partial_payload)
                error_detail = _extract_doubao_stream_error(partial_payload)
                if error_detail:
                    raise RuntimeError(f"Doubao browser stream failed: {error_detail}")
            if not saw_visible_piece and startup_deadline is not None and time.time() >= startup_deadline:
                raise RuntimeError(
                    f"Doubao browser stream startup timed out after {startup_timeout_seconds:g}s"
                )
            if snapshot.get("done"):
                break
            time.sleep(poll_interval_seconds)
        else:
            raise RuntimeError(f"Doubao browser stream timed out after {int(timeout_seconds)}s")
        if current_event and current_data:
            try:
                parsed = json.loads(current_data)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                conversation = _extract_doubao_conversation_id(
                    f'event: {current_event}\ndata: {current_data}\n'
                )
                if conversation:
                    session.metadata["doubao_conversation_id"] = conversation
                    client._persist_doubao_conversation_id(conversation)
                yield from emit_incremental_chunks(
                    _extract_doubao_chunks_from_event(current_event, parsed),
                    require_prefix_match=current_event == "STREAM_MSG_NOTIFY",
                )
        final_payload = "\n".join(captured_lines)
        conversation = _extract_doubao_conversation_id(final_payload)
        if conversation:
            session.metadata["doubao_conversation_id"] = conversation
            client._persist_doubao_conversation_id(conversation)
        error_detail = _extract_doubao_stream_error(final_payload)
        if error_detail:
            raise RuntimeError(f"Doubao browser stream failed: {error_detail}")
        final_error = str(snapshot.get("error") or "").strip()
        if final_error and not final_payload.strip():
            raise RuntimeError(f"Doubao browser stream failed: {final_error[:500]}")
        completed = True
    finally:
        try:
            page.evaluate(
                """
                ({ streamId, shouldAbort }) => {
                  try {
                    const state = window[streamId];
                    if (shouldAbort && state && !state.done && typeof state.abort === "function") {
                      try {
                        state.abort("opentoken-cleanup");
                      } catch (_error) {}
                    }
                    delete window[streamId];
                  } catch (_error) {}
                  return true;
                }
                """,
                {"streamId": stream_id, "shouldAbort": not completed},
            )
        except Exception:
            pass


def _stream_doubao_dom_completion(
    page,
    *,
    session,
    client,
    message: str,
    model: str,
    poll_interval_seconds: float = 0.2,
    timeout_seconds: float = 120.0,
    startup_timeout_seconds: float | None = None,
) -> Iterator[str]:
    _select_doubao_model(page, model)
    baseline = page.evaluate(
        """
        () => ({
          markdownCount: document.querySelectorAll('[class*="markdown"]').length,
          messageCount: document.querySelectorAll('[class*="message"]').length,
        })
        """
    )
    before_markdown_count = int(baseline.get("markdownCount", 0))
    before_message_count = int(baseline.get("messageCount", 0))

    composer = page.locator(
        'textarea[placeholder="发消息..."], textarea.semi-input-textarea'
    ).first
    composer.wait_for(timeout=120000)
    composer_focused = False
    try:
        composer.click(timeout=5000, force=True)
        composer_focused = True
    except TypeError:
        try:
            composer.click(timeout=5000)
            composer_focused = True
        except Exception:
            composer_focused = False
    except Exception:
        composer_focused = False
    page.wait_for_timeout(250)

    typed = False
    prefer_fill = "\n" in message or "\r" in message
    if composer_focused and not prefer_fill:
        try:
            page.keyboard.press("Meta+A")
            page.keyboard.press("Backspace")
            page.wait_for_timeout(150)
            page.keyboard.type(message, delay=55)
            typed = True
        except Exception:
            typed = False
    if not typed:
        try:
            composer.fill(message)
        except Exception:
            _set_visible_textarea_value(
                page,
                [
                    'textarea[placeholder="发消息..."]',
                    "textarea.semi-input-textarea",
                    "textarea",
                ],
                message,
            )
    page.wait_for_timeout(500)

    send_button = None
    try:
        send_button = page.locator(
            'button[class*="g-send-msg-btn-bg"], #input-engine-container button'
        ).first
        send_button.wait_for(timeout=30000)
    except Exception:
        send_button = None

    captured_response = None

    def handle_response(response):
        nonlocal captured_response
        if captured_response is not None:
            return
        if not _is_doubao_chat_completion_response(response, message):
            return
        headers = {}
        try:
            headers = response.headers
        except Exception:
            headers = {}
        content_type = str(headers.get("content-type", "")).lower()
        if content_type and "text/event-stream" not in content_type:
            return
        captured_response = response

    page.on("response", handle_response)
    emitted = ""
    last_text = ""
    stable_rounds = 0
    deadline = time.time() + timeout_seconds
    startup_deadline = (
        time.time() + max(0.0, startup_timeout_seconds)
        if startup_timeout_seconds is not None
        else None
    )
    try:
        try:
            clicked = bool(
                page.evaluate(
                    """
                    () => {
                      const visible = (node) => !!node && !!node.offsetParent;
                      const candidates = Array.from(
                        document.querySelectorAll('#input-engine-container button, button')
                      ).filter((node) => {
                        if (!visible(node)) {
                          return false;
                        }
                        if (node.disabled || node.getAttribute('aria-disabled') === 'true') {
                          return false;
                        }
                        return true;
                      });
                      const preferred = candidates.find((node) =>
                        String(node.className || '').includes('g-send-msg-btn-bg')
                      ) || candidates.at(-1);
                      if (!preferred) {
                        return false;
                      }
                      preferred.click();
                      return true;
                    }
                    """
                )
            )
            if not clicked:
                raise RuntimeError("Doubao send button not found")
        except Exception:
            try:
                if send_button is None:
                    raise RuntimeError("send button locator unavailable")
                send_button.click(timeout=5000)
            except Exception:
                page.keyboard.press("Enter")

        while time.time() < deadline:
            result = page.evaluate(
                """
                ({ beforeMarkdownCount, beforeMessageCount }) => {
                  const clean = (text) => (text || "").replace(/[\\u200B-\\u200D\\uFEFF]/g, " ").replace(/\\s+/g, " ").trim();
                  const isVisible = (node) => !!node && !!node.offsetParent;
                  const markdownTexts = Array.from(document.querySelectorAll('[class*="markdown"]'))
                    .slice(beforeMarkdownCount)
                    .filter(isVisible)
                    .map((node) => clean(node.innerText || node.textContent || ""))
                    .filter(Boolean);
                  const messageTexts = Array.from(document.querySelectorAll('[class*="message"]'))
                    .slice(beforeMessageCount)
                    .filter(isVisible)
                    .map((node) => clean(node.innerText || node.textContent || ""))
                    .filter(Boolean);
                  const composerBusy = Boolean(
                    document.querySelector(
                      '[aria-busy="true"], [class*="loading"], [class*="typing"], [class*="stream"], [class*="generat"], [class*="stop"]'
                    )
                  );
                  return { markdownTexts, messageTexts, composerBusy };
                }
                """,
                {
                    "beforeMarkdownCount": before_markdown_count,
                    "beforeMessageCount": before_message_count,
                },
            )
            text = _extract_doubao_dom_candidate_text(
                message,
                markdown_texts=list(result.get("markdownTexts", [])),
                message_texts=list(result.get("messageTexts", [])),
            )
            if text and text == last_text:
                stable_rounds += 1
            elif text:
                last_text = text
                stable_rounds = 0
            if text:
                if startup_deadline is not None:
                    startup_deadline = None
                suffix, emitted = _advance_snapshot_stream_state(emitted, text)
                if suffix:
                    yield suffix
            if not text and captured_response is not None and not result.get("composerBusy"):
                payload = captured_response.text()
                if payload:
                    conversation_id = _extract_doubao_conversation_id(payload)
                    if conversation_id:
                        session.metadata["doubao_conversation_id"] = conversation_id
                        client._persist_doubao_conversation_id(conversation_id)
                    final_text = _parse_doubao_response_text(payload).strip()
                    if final_text:
                        if startup_deadline is not None:
                            startup_deadline = None
                        if emitted and not final_text.startswith(emitted) and _is_reformatted_snapshot_duplicate(emitted, final_text):
                            return
                        suffix, emitted = _advance_snapshot_stream_state(emitted, final_text)
                        if suffix:
                            yield suffix
                        return
                    error_detail = _extract_doubao_stream_error(payload)
                    if error_detail:
                        raise RuntimeError(f"Doubao DOM stream failed: {error_detail}")
            if last_text and captured_response is not None and not result.get("composerBusy") and stable_rounds >= 1:
                break
            if startup_deadline is not None and time.time() >= startup_deadline:
                raise RuntimeError(
                    f"Doubao DOM stream startup timed out after {startup_timeout_seconds:g}s"
                )
            time.sleep(poll_interval_seconds)
        else:
            raise RuntimeError(f"Doubao DOM stream timed out after {int(timeout_seconds)}s")
    finally:
        _detach_page_listener(page, "response", handle_response)
        if not emitted and not last_text:
            _stop_doubao_dom_generation(page)

    payload = ""
    if captured_response is not None:
        payload = captured_response.text()
    conversation_id = _extract_doubao_conversation_id(payload)
    if conversation_id:
        session.metadata["doubao_conversation_id"] = conversation_id
        client._persist_doubao_conversation_id(conversation_id)
    final_text = _parse_doubao_response_text(payload).strip() if payload else ""
    if final_text:
        if emitted and not final_text.startswith(emitted) and _is_reformatted_snapshot_duplicate(emitted, final_text):
            return
        suffix, emitted = _advance_snapshot_stream_state(emitted, final_text)
        if suffix:
            yield suffix
        return
    if last_text:
        return
    error_detail = _extract_doubao_stream_error(payload)
    if error_detail:
        raise RuntimeError(f"Doubao DOM stream failed: {error_detail}")
    raise RuntimeError("Doubao DOM stream completed but returned no text content.")


def _stream_glm_cn_browser_completion(
    page,
    *,
    session,
    client,
    message: str,
    model: str,
    access_token: str | None = None,
    device_id: str | None = None,
    poll_interval_seconds: float = 0.05,
    timeout_seconds: float = 120.0,
) -> Iterator[str]:
    stable_device_id = str(device_id or session.metadata.get("glm_cn_device_id") or "").strip() or uuid4().hex
    session.metadata["glm_cn_device_id"] = stable_device_id
    session.metadata["glm_cn_conversation_id"] = ""
    assistant_id = _GLM_ASSISTANT_ID_MAP.get(model, "65940acff94777010aa6b796")
    sign = _generate_glm_sign()
    stream_id = f"__opentoken_glm_cn_stream_{uuid4().hex}"
    started = page.evaluate(
        """
        ({ streamId, accessToken, body, deviceId, requestId, sign, xExpGroups, timeoutMs }) => {
          const state = {
            lines: [],
            done: false,
            error: "",
          };
          window[streamId] = state;
          const appendLine = (value) => {
            if (typeof value === "string") {
              state.lines.push(value);
            }
          };
          const finalize = (error) => {
            if (error) {
              state.error = String(error);
            }
            state.done = true;
          };
          (async () => {
            let timeout = null;
            let controller = null;
            try {
              controller = new AbortController();
              timeout = setTimeout(() => controller.abort("timeout"), timeoutMs);
              const headers = {
                "Content-Type": "application/json",
                Accept: "text/event-stream",
                "App-Name": "chatglm",
                Origin: "https://chatglm.cn",
                "X-App-Platform": "pc",
                "X-App-Version": "0.0.1",
                "X-App-fr": "default",
                "X-Device-Brand": "",
                "X-Device-Id": deviceId,
                "X-Device-Model": "",
                "X-Exp-Groups": xExpGroups,
                "X-Lang": "zh",
                "X-Nonce": sign.nonce,
                "X-Request-Id": requestId,
                "X-Sign": sign.sign,
                "X-Timestamp": sign.timestamp,
              };
              if (accessToken) {
                headers["Authorization"] = "Bearer " + accessToken;
              }
              const res = await fetch("https://chatglm.cn/chatglm/backend-api/assistant/stream", {
                method: "POST",
                headers,
                credentials: "include",
                body: JSON.stringify(body),
                signal: controller.signal,
              });
              if (!res.ok) {
                finalize(`${res.status} ${await res.text()}`);
                return;
              }
              const reader = res.body?.getReader();
              if (!reader) {
                finalize("No response body");
                return;
              }
              const decoder = new TextDecoder();
              let buffer = "";
              while (true) {
                const { done, value } = await reader.read();
                if (done) {
                  break;
                }
                buffer += decoder.decode(value, { stream: true });
                while (true) {
                  const newlineIndex = buffer.indexOf("\\n");
                  if (newlineIndex === -1) {
                    break;
                  }
                  let line = buffer.slice(0, newlineIndex);
                  buffer = buffer.slice(newlineIndex + 1);
                  if (line.endsWith("\\r")) {
                    line = line.slice(0, -1);
                  }
                  appendLine(line);
                }
              }
              if (buffer) {
                appendLine(buffer);
              }
              finalize("");
            } catch (error) {
              finalize(error);
            } finally {
              if (timeout) {
                clearTimeout(timeout);
              }
            }
          })();
          return { ok: true };
        }
        """,
        {
            "streamId": stream_id,
            "accessToken": access_token or None,
            "body": {
                "assistant_id": assistant_id,
                "conversation_id": "",
                "project_id": "",
                "chat_type": "user_chat",
                "meta_data": _glm_meta_data_for_model(model),
                "messages": [{"role": "user", "content": [{"type": "text", "text": message}]}],
            },
            "deviceId": stable_device_id,
            "requestId": uuid4().hex,
            "sign": sign,
            "xExpGroups": _GLM_X_EXP_GROUPS,
            "timeoutMs": int(timeout_seconds * 1000),
        },
    )
    if not started.get("ok"):
        raise RuntimeError("GLM China browser stream failed to start.")

    deadline = time.time() + timeout_seconds
    snapshot: dict[str, object] = {"lines": [], "done": False, "error": ""}
    emitted = ""
    saw_any_piece = False

    try:
        while time.time() < deadline:
            snapshot = page.evaluate(
                """
                ({ streamId }) => {
                  const state = window[streamId];
                  if (!state) {
                    return { lines: [], done: true, error: "GLM China browser stream state missing" };
                  }
                  const lines = state.lines.splice(0, state.lines.length);
                  return { lines, done: !!state.done, error: state.error || "" };
                }
                """,
                {"streamId": stream_id},
            )
            for raw_line in list(snapshot.get("lines", [])):
                for normalized_line in str(raw_line).splitlines() or [""]:
                    line = str(normalized_line).strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str or data_str == "[DONE]":
                        continue
                    try:
                        payload = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    conversation_id = payload.get("conversation_id")
                    if isinstance(conversation_id, str) and conversation_id:
                        session.metadata["glm_cn_conversation_id"] = conversation_id
                    for candidate in _extract_glm_text_candidates(payload):
                        suffix, emitted = _advance_text_stream_state(emitted, candidate)
                        if not suffix:
                            continue
                        saw_any_piece = True
                        yield suffix
            if snapshot.get("done"):
                break
            time.sleep(poll_interval_seconds)
        else:
            raise RuntimeError(f"GLM China browser stream timed out after {int(timeout_seconds)}s")

        if saw_any_piece:
            client._persist_glm_cn_session_state(device_id=stable_device_id)

        final_error = str(snapshot.get("error") or "").strip()
        if final_error and not saw_any_piece:
            raise RuntimeError(f"GLM China browser stream failed: {final_error[:500]}")
        if not saw_any_piece:
            raise RuntimeError("GLM China browser stream returned no text content.")
    finally:
        try:
            page.evaluate(
                """
                ({ streamId }) => {
                  try {
                    delete window[streamId];
                  } catch (_error) {}
                  return true;
                }
                """,
                {"streamId": stream_id},
            )
        except Exception:
            pass


def _is_doubao_chat_completion_response(response, message: str) -> bool:
    if "chat/completion" not in str(getattr(response, "url", "")):
        return False

    request = getattr(response, "request", None)
    if request is None or str(getattr(request, "method", "")).upper() != "POST":
        return False

    post_data = getattr(request, "post_data", "") or ""
    if post_data:
        raw_post_data = str(post_data)
        if message in raw_post_data:
            return True
        message_text = _normalize_doubao_message_text(message)
        for candidate in _extract_doubao_request_text_candidates(raw_post_data):
            candidate_text = _normalize_doubao_message_text(candidate)
            if not candidate_text:
                continue
            if candidate_text == message_text:
                return True
            if message_text and (message_text in candidate_text or candidate_text in message_text):
                return True
        return False
    return True


def _normalize_doubao_message_text(value: str) -> str:
    text = str(value or "")
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n")
    return text.replace("\r\n", "\n").strip()


def _extract_doubao_request_text_candidates(post_data: str) -> list[str]:
    try:
        payload = json.loads(post_data)
    except json.JSONDecodeError:
        return []

    candidates: list[str] = []
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return candidates

    for item in messages:
        if not isinstance(item, dict):
            continue
        raw_content = item.get("content")
        if isinstance(raw_content, str):
            try:
                parsed_content = json.loads(raw_content)
            except json.JSONDecodeError:
                parsed_content = raw_content
            if isinstance(parsed_content, dict):
                text_value = parsed_content.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    candidates.append(text_value)
            elif isinstance(parsed_content, str) and parsed_content.strip():
                candidates.append(parsed_content)
        content_blocks = item.get("content_block")
        if isinstance(content_blocks, list):
            for block in content_blocks:
                if not isinstance(block, dict):
                    continue
                block_content = block.get("content")
                if not isinstance(block_content, dict):
                    continue
                text_block = block_content.get("text_block")
                if not isinstance(text_block, dict):
                    continue
                text_value = text_block.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    candidates.append(text_value)
    return candidates


def _extract_doubao_conversation_id(payload: str) -> str:
    for match in re.finditer(r'"conversation_id"\s*:\s*"([^"]+)"', payload):
        candidate = match.group(1).strip()
        if candidate and candidate != "0":
            return candidate
    return ""


def _extract_doubao_stream_error(payload: str) -> str:
    current_event = ""
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("event: "):
            current_event = line.removeprefix("event: ").strip()
            continue
        if not line.startswith("data: "):
            continue
        data_line = line.removeprefix("data: ").strip()
        try:
            parsed = json.loads(data_line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and parsed.get("event_type") == 2005:
            event_data = parsed.get("event_data")
            if isinstance(event_data, str):
                try:
                    event_data = json.loads(event_data)
                except json.JSONDecodeError:
                    event_data = None
            if isinstance(event_data, dict):
                error_code = event_data.get("code") or event_data.get("error_code")
                error_message = event_data.get("message") or event_data.get("error_msg")
                error_detail = event_data.get("error_detail")
                if isinstance(error_detail, dict):
                    detail_message = error_detail.get("message")
                    if isinstance(detail_message, str) and detail_message.strip():
                        if error_message and detail_message.strip() not in error_message:
                            error_message = f"{error_message}: {detail_message.strip()}"
                        elif not error_message:
                            error_message = detail_message.strip()
                if error_code or error_message:
                    return f"SAMANTHA_ERROR {error_code}: {error_message}".strip()
        if current_event == "STREAM_ERROR" and isinstance(parsed, dict):
            error_code = parsed.get("error_code")
            error_message = parsed.get("error_msg")
            if error_code or error_message:
                return f"STREAM_ERROR {error_code}: {error_message}".strip()
        if current_event == "gateway-error" and isinstance(parsed, dict):
            message = parsed.get("message")
            if isinstance(message, str) and message.strip():
                return f"gateway-error: {message.strip()}"
    return ""


def _select_doubao_model(page, model: str) -> None:
    candidate_labels = _DOUBAO_MODEL_MENU_NAME_MAP.get(model)
    if not candidate_labels:
        return

    try:
        current_label = str(
            page.evaluate(
                """
                () => {
                  const clean = (s) => (s || "").replace(/\\s+/g, " ").trim();
                  const labels = Array.from(document.querySelectorAll('button[aria-haspopup="menu"]'))
                    .map((el) => clean(el.innerText || el.textContent || ""))
                    .filter(Boolean);
                  return labels.find((text) => /^(快速|思考|专家)/.test(text)) || "";
                }
                """
            )
        ).strip()
        if current_label and any(current_label.startswith(label) for label in candidate_labels):
            return

        model_selector = page.locator('button[aria-haspopup="menu"]').filter(
            has_text=re.compile(r"^(快速|思考|专家)")
        ).first
        model_selector.wait_for(timeout=3000)
        model_selector.click(timeout=30000)
        for label in candidate_labels:
            try:
                menu_item = page.get_by_role("menuitem", name=re.compile(rf"^{re.escape(label)}"))
                menu_item.first.wait_for(timeout=3000)
                menu_item.first.click(timeout=30000)
                page.wait_for_timeout(300)
                return
            except Exception:
                continue
    except Exception:
        return


def _dom_send_and_wait_qwen_cn(page, message: str) -> str:
    baseline = page.evaluate(
        """
        () => ({
          markdownCount: document.querySelectorAll('[class*="markdown"]').length,
          messageCount: document.querySelectorAll('[class*="message"]').length,
        })
        """
    )
    before_markdown_count = int(baseline.get("markdownCount", 0))
    before_message_count = int(baseline.get("messageCount", 0))

    composer = page.locator('[contenteditable="true"][role="textbox"]').first
    composer.wait_for(timeout=120000)
    composer.click(timeout=30000)
    page.keyboard.press("Meta+A")
    page.keyboard.press("Backspace")
    page.keyboard.type(message, delay=20)
    page.keyboard.press("Enter")

    deadline = time.time() + 120
    stable_rounds = 0
    last_text = ""
    while time.time() < deadline:
        page.wait_for_timeout(2000)
        result = page.evaluate(
            """
            ({ beforeMarkdownCount, beforeMessageCount }) => {
              const clean = (text) => (text || "").replace(/[\\u200B-\\u200D\\uFEFF]/g, " ").replace(/\\s+/g, " ").trim();
              const isVisible = (node) => !!node && !!node.offsetParent;
              const markdownTexts = Array.from(document.querySelectorAll('[class*="markdown"]'))
                .slice(beforeMarkdownCount)
                .filter(isVisible)
                .map((node) => clean(node.innerText || node.textContent || ""))
                .filter(Boolean);
              const messageTexts = Array.from(document.querySelectorAll('[class*="message"]'))
                .slice(beforeMessageCount)
                .filter(isVisible)
                .map((node) => clean(node.innerText || node.textContent || ""))
                .filter(Boolean);
              const composerBusy = Boolean(
                document.querySelector('[aria-busy="true"], [class*="loading"], [class*="typing"], [class*="stream"]')
              );
              return { markdownTexts, messageTexts, composerBusy };
            }
            """,
            {
                "beforeMarkdownCount": before_markdown_count,
                "beforeMessageCount": before_message_count,
            },
        )
        text = _extract_qwen_cn_candidate_text(
            message,
            markdown_texts=list(result.get("markdownTexts", [])),
            message_texts=list(result.get("messageTexts", [])),
        )
        if text and text == last_text:
            stable_rounds += 1
        elif text:
            last_text = text
            stable_rounds = 0
        if last_text and ((not result.get("composerBusy") and stable_rounds >= 1) or stable_rounds >= 2):
            return last_text

    if last_text:
        return last_text
    raise RuntimeError("Qwen China DOM reply capture failed.")


class _QwenIntlStreamProjector:
    def __init__(self) -> None:
        self._in_think = False

    def push_event_payload(self, payload: dict[str, object]) -> list[str]:
        pieces: list[str] = []
        for phase, text in _extract_qwen_intl_phased_segments(payload):
            if not text:
                continue
            if phase == "think":
                if not self._in_think:
                    pieces.append("<think>")
                    self._in_think = True
                pieces.append(text)
                continue
            if self._in_think:
                pieces.append("</think>")
                self._in_think = False
            pieces.append(text)
        return pieces

    def finish(self) -> str:
        if not self._in_think:
            return ""
        self._in_think = False
        return "</think>"


_QWEN_INTL_MODEL_LABELS: dict[str, tuple[str, ...]] = {
    "qwen3.6-plus": ("Qwen3.6-Plus",),
    "qwen3.5-plus": ("Qwen3.5-Plus",),
    "qwen3.5-flash": ("Qwen3.5-Flash",),
    "qwen3.5-omni-plus": ("Qwen3.5-Omni-Plus",),
    "qwen-max-latest": ("Qwen2.5-Max", "Qwen-Max"),
    "qwen3.5-max-2026-03-08": ("Qwen3.5-Max",),
}


def _normalize_qwen_intl_model_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(label or "").lower())


def _qwen_intl_model_option_labels(model: str) -> tuple[str, ...]:
    labels = _QWEN_INTL_MODEL_LABELS.get(str(model or "").strip().lower())
    if labels:
        return labels
    text = str(model or "").strip()
    if not text:
        return ()
    if text.lower().startswith("qwen"):
        return (text,)
    return ()


def _prepare_qwen_intl_dom_page(page, *, model: str) -> None:
    current_url = str(getattr(page, "url", ""))
    if not current_url or "chat.qwen.ai" not in current_url:
        page.goto(f"{_QWEN_INTL_BASE_URL}/", wait_until="domcontentloaded", timeout=120000)
    page.wait_for_timeout(2500)
    if _qwen_intl_page_has_visible_messages(page):
        _click_qwen_intl_new_chat(page)
        page.wait_for_timeout(1000)
    _select_qwen_intl_dom_model(page, model=model)
    page.locator("textarea.message-input-textarea").first.wait_for(timeout=120000)


def _qwen_intl_page_has_visible_messages(page) -> bool:
    try:
        return bool(
            page.evaluate(
                """
                () => {
                  const clean = (s) => (s || "").replace(/\\s+/g, " ").trim();
                  const root = document.querySelector(".chat-messages");
                  return clean(root?.innerText || "").length > 0;
                }
                """
            )
        )
    except Exception:
        return False


def _click_qwen_intl_new_chat(page) -> bool:
    try:
        clicked = bool(
            page.evaluate(
                """
                () => {
                  const clean = (s) => (s || "").replace(/\\s+/g, " ").trim();
                  const visible = (node) => !!node && (node.offsetParent !== null || node === document.activeElement);
                  const candidates = Array.from(document.querySelectorAll("div, button, a, span")).filter((node) => {
                    if (!visible(node)) {
                      return false;
                    }
                    if (!String(node.className || "").includes("sidebar-entry-fixed-list")) {
                      return false;
                    }
                    return clean(node.innerText || node.textContent || "") === "New Chat";
                  });
                  const target = candidates[0];
                  if (!target) {
                    return false;
                  }
                  target.click();
                  return true;
                }
                """
            )
        )
        if clicked:
            page.wait_for_timeout(800)
            return True
    except Exception:
        pass
    try:
        locator = page.locator("div.sidebar-entry-fixed-list").filter(has_text=re.compile(r"^New Chat$")).first
        locator.wait_for(timeout=5000)
        locator.click(timeout=5000)
        page.wait_for_timeout(800)
        return True
    except Exception:
        try:
            clicked = bool(
                page.evaluate(
                    """
                    () => {
                      const clean = (s) => (s || "").replace(/\\s+/g, " ").trim();
                      const visible = (node) => !!node && (node.offsetParent !== null || node === document.activeElement);
                      const candidates = Array.from(document.querySelectorAll("div, button, a, span")).filter((node) => {
                        if (!visible(node)) {
                          return false;
                        }
                        if (!String(node.className || "").includes("sidebar-entry-fixed-list")) {
                          return false;
                        }
                        return clean(node.innerText || node.textContent || "") === "New Chat";
                      });
                      const target = candidates[0];
                      if (!target) {
                        return false;
                      }
                      target.click();
                      return true;
                    }
                    """
                )
            )
        except Exception:
            clicked = False
        if clicked:
            page.wait_for_timeout(800)
        return clicked


def _get_qwen_intl_selected_model_label(page) -> str:
    return str(
        page.evaluate(
            """
            () => {
              const clean = (s) => (s || "").replace(/\\s+/g, " ").trim();
              const direct = document.querySelector(".index-module__model-selector-text___XvWe0");
              if (direct) {
                return clean(direct.innerText || direct.textContent || "");
              }
              const trigger = document.querySelector("header .ant-dropdown-trigger");
              return clean(trigger?.innerText || trigger?.textContent || "");
            }
            """
        )
        or ""
    ).strip()


def _select_qwen_intl_dom_model(page, *, model: str) -> None:
    target_labels = _qwen_intl_model_option_labels(model)
    if not target_labels:
        return
    current_label = _get_qwen_intl_selected_model_label(page)
    normalized_targets = {_normalize_qwen_intl_model_label(label) for label in target_labels}
    if _normalize_qwen_intl_model_label(current_label) in normalized_targets:
        return
    opened = bool(
        page.evaluate(
            """
            () => {
              const visible = (node) => !!node && (node.offsetParent !== null || node === document.activeElement);
              const trigger = document.querySelector("header .ant-dropdown-trigger")
                || document.querySelector(".index-module__model-selector-text___XvWe0")?.closest(".ant-dropdown-trigger");
              if (!visible(trigger)) {
                return false;
              }
              trigger.click();
              return true;
            }
            """
        )
    )
    if not opened:
        return
    page.wait_for_timeout(300)
    for label in target_labels:
        clicked = bool(
            page.evaluate(
                """
                ({ label }) => {
                  const clean = (s) => (s || "").replace(/\\s+/g, " ").trim();
                  const visible = (node) => !!node && (node.offsetParent !== null || node === document.activeElement);
                  const dropdown = Array.from(document.querySelectorAll(".ant-dropdown"))
                    .filter((node) => visible(node))
                    .at(-1);
                  if (!dropdown) {
                    return false;
                  }
                  const candidates = Array.from(dropdown.querySelectorAll("div, span, button, a"))
                    .filter((node) => visible(node));
                  const target = candidates.find((node) => clean(node.innerText || node.textContent || "") === label);
                  if (!target) {
                    return false;
                  }
                  target.click();
                  return true;
                }
                """,
                {"label": label},
            )
        )
        if clicked:
            page.wait_for_timeout(500)
            return


def _send_qwen_intl_dom_message(page, *, message: str) -> None:
    composer = page.locator("textarea.message-input-textarea").first
    composer.wait_for(timeout=120000)
    composer.click(timeout=30000)
    filled = _set_visible_textarea_value(page, ["textarea.message-input-textarea", "textarea"], message)
    if not filled:
        try:
            page.keyboard.press("Meta+A")
            page.keyboard.press("Backspace")
        except Exception:
            pass
        typed = False
        try:
            page.keyboard.type(message, delay=12)
            typed = True
        except Exception:
            typed = False
        if not typed:
            composer.fill(message)
    page.wait_for_timeout(200)
    try:
        clicked = bool(
            page.evaluate(
                """
                () => {
                  const visible = (node) => !!node && (node.offsetParent !== null || node === document.activeElement);
                  const button = Array.from(document.querySelectorAll("button.send-button, button"))
                    .find((node) => visible(node) && String(node.className || "").includes("send-button") && !node.disabled);
                  if (!button) {
                    return false;
                  }
                  button.click();
                  return true;
                }
                """
            )
        )
        if clicked:
            return
    except Exception:
        pass
    send_button = page.locator("button.send-button").first
    send_button.wait_for(timeout=30000)
    send_button.click(timeout=30000)


def _capture_qwen_intl_dom_stream_state(page) -> dict[str, object]:
    snapshot = page.evaluate(
        """
        () => {
          const clean = (s) => (s || "")
            .replace(/[\\u200B-\\u200D\\uFEFF]/g, "")
            .replace(/\\s+/g, " ")
            .trim();
          const assistants = Array.from(document.querySelectorAll(".qwen-chat-message.qwen-chat-message-assistant"));
          const latest = assistants.at(-1) || null;
          const answerNode = latest?.querySelector(".response-message-content.t2t.phase-answer")
            || latest?.querySelector(".custom-qwen-markdown")
            || latest?.querySelector(".qwen-markdown");
          const thinkingNode = latest?.querySelector(".qwen-chat-thinking-status-card-title-text");
          const isStreaming = Boolean(
            document.querySelector("button.stop-button, [class*='stop-button'], [aria-label*='Stop'], [aria-label*='stop']")
          );
          return {
            answer_text: clean(answerNode?.innerText || answerNode?.textContent || ""),
            assistant_text: clean(latest?.innerText || latest?.textContent || ""),
            thinking_text: clean(thinkingNode?.innerText || thinkingNode?.textContent || ""),
            is_streaming: isStreaming,
          };
        }
        """
    )
    if not isinstance(snapshot, dict):
        return {
            "answer_text": "",
            "assistant_text": "",
            "thinking_text": "",
            "is_streaming": False,
        }
    return {
        "answer_text": str(snapshot.get("answer_text") or ""),
        "assistant_text": str(snapshot.get("assistant_text") or ""),
        "thinking_text": str(snapshot.get("thinking_text") or ""),
        "is_streaming": bool(snapshot.get("is_streaming")),
    }


def _stream_qwen_intl_dom_completion(
    page,
    *,
    message: str,
    model: str,
    poll_interval_seconds: float = 0.05,
    timeout_seconds: float = 120.0,
) -> Iterator[str]:
    _prepare_qwen_intl_dom_page(page, model=model)
    _send_qwen_intl_dom_message(page, message=message)

    emitted = ""
    last_answer = ""
    stable_rounds = 0
    saw_streaming = False
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        snapshot = _capture_qwen_intl_dom_stream_state(page)
        answer_text = str(snapshot.get("answer_text") or "").strip()
        if answer_text:
            suffix, emitted = _advance_text_stream_state(emitted, answer_text)
            if suffix:
                yield suffix
            if answer_text == last_answer:
                stable_rounds += 1
            else:
                last_answer = answer_text
                stable_rounds = 0
        if bool(snapshot.get("is_streaming")):
            saw_streaming = True
            stable_rounds = 0
        elif emitted and (saw_streaming or stable_rounds >= 1):
            return
        time.sleep(poll_interval_seconds)

    if emitted:
        return
    raise RuntimeError(f"Qwen International DOM stream timed out after {int(timeout_seconds)}s")


def _stream_qwen_intl_browser_completion(
    page,
    *,
    message: str,
    model: str,
    poll_interval_seconds: float = 0.05,
    timeout_seconds: float = 120.0,
) -> Iterator[str]:
    stream_id = f"__opentoken_qwen_intl_stream_{uuid4().hex}"
    started = page.evaluate(
        """
        ({ streamId, baseUrl, model, message, fid, featureConfig, timeoutMs }) => {
          const state = {
            lines: [],
            done: false,
            error: "",
          };
          window[streamId] = state;
          const appendLine = (value) => {
            if (typeof value === "string") {
              state.lines.push(value);
            }
          };
          const finalize = (error) => {
            if (error) {
              state.error = String(error);
            }
            state.done = true;
          };
          (async () => {
            let timeout = null;
            let controller = null;
            try {
              controller = new AbortController();
              timeout = setTimeout(() => controller.abort("timeout"), timeoutMs);
              const created = await fetch(`${baseUrl}/api/v2/chats/new`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({}),
                credentials: "include",
                signal: controller.signal,
              });
              if (!created.ok) {
                finalize(`${created.status} ${await created.text()}`);
                return;
              }
              const createdData = await created.json();
              const chatId = createdData?.data?.id ?? createdData?.data?.chat_id ?? createdData?.chat_id ?? createdData?.id ?? "";
              if (!chatId) {
                finalize("Qwen International chat creation returned no chat id");
                return;
              }
              const res = await fetch(`${baseUrl}/api/v2/chat/completions?chat_id=${chatId}`, {
                method: "POST",
                headers: {
                  "Content-Type": "application/json",
                  "Accept": "text/event-stream",
                },
                credentials: "include",
                signal: controller.signal,
                body: JSON.stringify({
                  stream: true,
                  version: "2.1",
                  incremental_output: true,
                  chat_id: chatId,
                  chat_mode: "normal",
                  model,
                  parent_id: null,
                  messages: [{
                    fid,
                    parentId: null,
                    childrenIds: [],
                    role: "user",
                    content: message,
                    user_action: "chat",
                    files: [],
                    timestamp: Math.floor(Date.now() / 1000),
                    models: [model],
                    chat_type: "t2t",
                    feature_config: featureConfig,
                  }],
                }),
              });
              if (!res.ok) {
                finalize(`${res.status} ${await res.text()}`);
                return;
              }
              const reader = res.body?.getReader();
              if (!reader) {
                finalize("No response body");
                return;
              }
              const decoder = new TextDecoder();
              let buffer = "";
              while (true) {
                const { done, value } = await reader.read();
                if (done) {
                  break;
                }
                buffer += decoder.decode(value, { stream: true });
                while (true) {
                  const newlineIndex = buffer.indexOf("\\n");
                  if (newlineIndex === -1) {
                    break;
                  }
                  let line = buffer.slice(0, newlineIndex);
                  buffer = buffer.slice(newlineIndex + 1);
                  if (line.endsWith("\\r")) {
                    line = line.slice(0, -1);
                  }
                  appendLine(line);
                }
              }
              if (buffer) {
                appendLine(buffer);
              }
              finalize("");
            } catch (error) {
              finalize(error);
            } finally {
              if (timeout) {
                clearTimeout(timeout);
              }
            }
          })();
          return { ok: true };
        }
        """,
        {
            "streamId": stream_id,
            "baseUrl": _QWEN_INTL_BASE_URL,
            "model": model,
            "message": message,
            "fid": str(uuid4()),
            "featureConfig": _qwen_feature_config_for_model(model),
            "timeoutMs": int(timeout_seconds * 1000),
        },
    )
    if not started.get("ok"):
        raise RuntimeError("Qwen International browser stream failed to start.")

    projector = _QwenIntlStreamProjector()
    deadline = time.time() + timeout_seconds
    snapshot: dict[str, object] = {"lines": [], "done": False, "error": ""}
    saw_any_piece = False

    try:
        while time.time() < deadline:
            snapshot = page.evaluate(
                """
                ({ streamId }) => {
                  const state = window[streamId];
                  if (!state) {
                    return { lines: [], done: true, error: "Qwen International browser stream state missing" };
                  }
                  const lines = state.lines.splice(0, state.lines.length);
                  return { lines, done: !!state.done, error: state.error || "" };
                }
                """,
                {"streamId": stream_id},
            )
            for raw_line in list(snapshot.get("lines", [])):
                for normalized_line in str(raw_line).splitlines() or [""]:
                    line = str(normalized_line).strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str or data_str == "[DONE]":
                        continue
                    try:
                        payload = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    for piece in projector.push_event_payload(payload):
                        if not piece:
                            continue
                        saw_any_piece = True
                        yield piece
            if snapshot.get("done"):
                break
            time.sleep(poll_interval_seconds)
        else:
            raise RuntimeError(f"Qwen International browser stream timed out after {int(timeout_seconds)}s")

        final_piece = projector.finish()
        if final_piece:
            saw_any_piece = True
            yield final_piece

        final_error = str(snapshot.get("error") or "").strip()
        if final_error and not saw_any_piece:
            raise RuntimeError(f"Qwen International browser stream failed: {final_error[:500]}")
    finally:
        try:
            page.evaluate(
                """
                ({ streamId }) => {
                  try {
                    delete window[streamId];
                  } catch (_error) {}
                  return true;
                }
                """,
                {"streamId": stream_id},
            )
        except Exception:
            pass


def _stream_qwen_intl_api_completion(
    credentials: ProviderCredentialRecord,
    *,
    message: str,
    model: str,
) -> Iterator[str]:
    client = QwenApiClient(
        credentials,
        base_url=_QWEN_INTL_BASE_URL,
        client=httpx.Client(
            timeout=_QWEN_INTL_API_STREAM_HTTP_TIMEOUT,
            trust_env=False,
        ),
    )
    yield from client.iter_chat_completion_text(message=message, model=model)


def _chat_glm_intl_api_completion(
    credentials: ProviderCredentialRecord,
    *,
    message: str,
    model: str,
) -> str:
    client = GLMIntlApiClient(credentials)
    return client.chat_completion(message=message, model=model)


def _stream_glm_intl_api_completion(
    credentials: ProviderCredentialRecord,
    *,
    message: str,
    model: str,
) -> Iterator[str]:
    client = GLMIntlApiClient(
        credentials,
        client=httpx.Client(
            timeout=_GLM_INTL_API_STREAM_HTTP_TIMEOUT,
            trust_env=False,
        ),
    )
    yield from client.iter_marked_chat_completion_text(message=message, model=model)


def _parse_ndjson_text(payload: str) -> str:
    chunks: list[str] = []
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        fragment = _extract_text_fragment(parsed)
        if fragment:
            chunks.append(fragment)
    return "".join(chunks)


def _is_qwen_intl_retryable_api_stream_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.ReadTimeout):
        return True
    detail = str(exc or "").lower()
    return "startup timed out" in detail or "timed out" in detail or "timeout" in detail


def _extract_text_fragment(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    if isinstance(payload.get("contentDelta"), str):
        return payload["contentDelta"]
    if isinstance(payload.get("textDelta"), str):
        return payload["textDelta"]
    if isinstance(payload.get("text"), str):
        return payload["text"]
    if isinstance(payload.get("content"), str):
        return payload["content"]
    if isinstance(payload.get("delta"), str):
        return payload["delta"]
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        delta = choices[0].get("delta", {})
        if isinstance(delta, dict):
            if isinstance(delta.get("content"), str):
                return delta["content"]
            if isinstance(delta.get("text"), str):
                return delta["text"]
    delta = payload.get("delta")
    if isinstance(delta, dict) and isinstance(delta.get("text"), str):
        return delta["text"]
    message = payload.get("message")
    if isinstance(message, dict):
        content = message.get("content", {})
        if isinstance(content, str) and content:
            return content
        if isinstance(content, dict):
            parts = content.get("parts")
            if isinstance(parts, list) and parts and isinstance(parts[-1], str):
                return parts[-1]
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict) and item.get("type") == "output_text" and isinstance(
                item.get("text"), str
            ):
                return item["text"]
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    return ""


def _dom_send_and_wait_chatgpt(page, message: str) -> str:
    page.evaluate(
        """
        (msg) => {
          const inputSelectors = [
            "#prompt-textarea",
            "textarea[placeholder]",
            "textarea",
            '[contenteditable="true"][data-placeholder]',
            '[contenteditable="true"]',
          ];
          let input = null;
          for (const selector of inputSelectors) {
            const candidate = document.querySelector(selector);
            if (candidate && candidate.offsetParent !== null) {
              input = candidate;
              break;
            }
          }
          if (!input) {
            throw new Error("ChatGPT input not found");
          }
          input.focus();
          if (input.tagName === "TEXTAREA" || input.tagName === "INPUT") {
            input.value = msg;
            input.dispatchEvent(new Event("input", { bubbles: true }));
          } else {
            input.textContent = msg;
            input.dispatchEvent(new Event("input", { bubbles: true }));
          }
          const sendSelectors = [
            "#composer-submit-button",
            'button[data-testid="send-button"]',
            'button[aria-label*="Send"]',
            'button[type="submit"]',
          ];
          for (const selector of sendSelectors) {
            const button = document.querySelector(selector);
            if (button && !button.disabled) {
              button.click();
              return;
            }
          }
          throw new Error("ChatGPT send button not found");
        }
        """,
        message,
    )
    deadline = time.time() + 90
    stable_rounds = 0
    last_text = ""
    while time.time() < deadline:
        time.sleep(2)
        result = page.evaluate(
            """
            () => {
              const clean = (text) => (text || "").replace(/[\\u200B-\\u200D\\uFEFF]/g, "").trim();
              const nodes = document.querySelectorAll(
                'div[data-message-author-role="assistant"], [class*="markdown"], [class*="assistant"]'
              );
              const latest = nodes.length > 0 ? nodes[nodes.length - 1] : null;
              const text = latest ? clean(latest.textContent) : "";
              const stopButton = document.querySelector('[aria-label*="Stop"], [aria-label*="stop"]');
              return { text, isStreaming: !!stopButton };
            }
            """
        )
        text = str(result.get("text", "")).strip()
        if text and text == last_text:
            stable_rounds += 1
        elif text:
            last_text = text
            stable_rounds = 0
        if last_text and not result.get("isStreaming") and stable_rounds >= 2:
            return last_text
    if last_text:
        return last_text
    raise RuntimeError("ChatGPT DOM reply capture failed.")


def _dom_send_and_wait_gemini(page, message: str) -> str:
    page.evaluate(
        """
        (msg) => {
          const inputSelectors = [
            '[placeholder*="Gemini"]',
            '[placeholder*="问问"]',
            '[contenteditable="true"]',
            'div[role="textbox"]',
            'textarea',
          ];
          let input = null;
          for (const selector of inputSelectors) {
            const candidate = document.querySelector(selector);
            if (candidate && candidate.offsetParent !== null) {
              input = candidate;
              break;
            }
          }
          if (!input) {
            throw new Error("Gemini input not found");
          }
          input.focus();
          if (input.tagName === "TEXTAREA" || input.tagName === "INPUT") {
            input.value = msg;
            input.dispatchEvent(new Event("input", { bubbles: true }));
          } else {
            input.innerText = msg;
            input.dispatchEvent(new Event("input", { bubbles: true }));
            input.dispatchEvent(new Event("change", { bubbles: true }));
          }
          const sendSelectors = [
            'button[aria-label*="Send"]',
            'button[aria-label*="send"]',
            'button[type="submit"]',
            '[aria-label*="Send message"]',
          ];
          for (const selector of sendSelectors) {
            const button = document.querySelector(selector);
            if (button && !button.disabled) {
              button.click();
              return;
            }
          }
          input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
        }
        """,
        message,
    )
    deadline = time.time() + 120
    stable_rounds = 0
    last_text = ""
    while time.time() < deadline:
        time.sleep(2)
        result = page.evaluate(
            """
            () => {
              const clean = (text) => (text || "").replace(/[\\u200B-\\u200D\\uFEFF]/g, "").trim();
              const selectors = [
                '[data-message-author="model"]',
                '[class*="assistant-message"]',
                '[class*="response-content"]',
                'article',
                '[class*="markdown"]',
              ];
              let text = "";
              for (const selector of selectors) {
                const nodes = document.querySelectorAll(selector);
                for (let idx = nodes.length - 1; idx >= 0; idx -= 1) {
                  const value = clean(nodes[idx].textContent);
                  if (value.length >= 40) {
                    text = value;
                    break;
                  }
                }
                if (text) break;
              }
              const stopButton = document.querySelector('[aria-label*="Stop"], [aria-label*="stop"]');
              return { text, isStreaming: !!stopButton };
            }
            """
        )
        text = str(result.get("text", "")).strip()
        if text and text == last_text:
            stable_rounds += 1
        elif text:
            last_text = text
            stable_rounds = 0
        if last_text and not result.get("isStreaming") and stable_rounds >= 2:
            return last_text
    if last_text:
        return last_text
    raise RuntimeError("Gemini DOM reply capture failed.")


def _dom_send_and_wait_grok(page, message: str) -> str:
    page.evaluate(
        """
        (msg) => {
          const inputSelectors = [
            '[contenteditable="true"]',
            'textarea[placeholder]',
            'textarea',
            'div[role="textbox"]',
          ];
          let input = null;
          for (const selector of inputSelectors) {
            const candidate = document.querySelector(selector);
            if (candidate && candidate.offsetParent !== null) {
              input = candidate;
              break;
            }
          }
          if (!input) {
            throw new Error("Grok input not found");
          }
          input.focus();
          if (input.tagName === "TEXTAREA" || input.tagName === "INPUT") {
            input.value = msg;
            input.dispatchEvent(new Event("input", { bubbles: true }));
          } else {
            input.innerText = msg;
            input.dispatchEvent(new Event("input", { bubbles: true }));
          }
          const sendSelectors = [
            'button[aria-label*="Send"]',
            'button[type="submit"]',
            '[class*="send"]',
          ];
          for (const selector of sendSelectors) {
            const button = document.querySelector(selector);
            if (button && !button.disabled) {
              button.click();
              return;
            }
          }
          input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
        }
        """,
        message,
    )
    deadline = time.time() + 90
    stable_rounds = 0
    last_text = ""
    while time.time() < deadline:
        time.sleep(2)
        result = page.evaluate(
            """
            () => {
              const clean = (text) => (text || "").replace(/[\\u200B-\\u200D\\uFEFF]/g, "").trim();
              const selectors = [
                '[data-role="assistant"]',
                '[class*="assistant"]',
                '[class*="response"]',
                '[class*="message"]',
                'article',
                '[class*="markdown"]',
              ];
              let text = "";
              for (const selector of selectors) {
                const nodes = document.querySelectorAll(selector);
                const latest = nodes.length > 0 ? nodes[nodes.length - 1] : null;
                if (latest) {
                  const value = clean(latest.textContent);
                  if (value.length > 20) {
                    text = value;
                    break;
                  }
                }
              }
              const stopButton = document.querySelector('[aria-label*="Stop"], [aria-label*="stop"]');
              return { text, isStreaming: !!stopButton };
            }
            """
        )
        text = str(result.get("text", "")).strip()
        if text and text == last_text:
            stable_rounds += 1
        elif text:
            last_text = text
            stable_rounds = 0
        if last_text and not result.get("isStreaming") and stable_rounds >= 2:
            return last_text
    if last_text:
        return last_text
    raise RuntimeError("Grok DOM reply capture failed.")
def _dom_send_and_wait_glm_cn(
    page,
    message: str,
    session: _ProviderBrowserSession | None = None,
    client: CamoufoxProviderClient | None = None,
) -> str:
    input_ready = False
    for attempt in range(2):
        input_ready = bool(page.evaluate("() => !!document.querySelector('textarea')"))
        if input_ready:
            break
        page.goto(f"{_GLM_CN_URL}/main/all", wait_until="domcontentloaded", timeout=120000)
        page.wait_for_timeout(5000)
    if not input_ready:
        raise RuntimeError("GLM China input not found")

    composer = page.locator("textarea").first
    composer.wait_for(timeout=120000)

    for attempt in range(2):
        composer.click(timeout=30000)
        typed = False
        prefer_fill = "\n" in message or "\r" in message
        if not prefer_fill:
            page.keyboard.press("Meta+A")
            page.keyboard.press("Backspace")
            page.keyboard.type(message, delay=35)
            typed = True
        if not typed:
            try:
                composer.fill(message)
            except Exception:
                _set_visible_textarea_value(page, ["textarea"], message)
        page.wait_for_timeout(300)

        captured_response = None

        def handle_response(response):
            nonlocal captured_response
            if captured_response is not None:
                return
            if "assistant/stream" not in str(getattr(response, "url", "")):
                return
            if int(getattr(response, "status", 0) or 0) < 200:
                return
            captured_response = response

        page.on("response", handle_response)
        try:
            page.keyboard.press("Enter")
            deadline = time.time() + 180
            while time.time() < deadline:
                if captured_response is not None:
                    break
                page.wait_for_timeout(250)
            if captured_response is None:
                raise RuntimeError("GLM China response listener timed out waiting for assistant/stream.")
        finally:
            _detach_page_listener(page, "response", handle_response)

        payload = captured_response.text()
        error_detail = _extract_glm_error_detail(payload)
        if error_detail:
            if attempt == 0 and _is_glm_retryable_dom_error(error_detail):
                page.wait_for_timeout(1500)
                continue
            raise RuntimeError(f"GLM China DOM send failed: {error_detail}")

        content, conversation_id = _parse_glm_sse_response(payload)
        if conversation_id and session is not None:
            session.metadata["glm_cn_conversation_id"] = conversation_id
        if conversation_id and client is not None:
            client._persist_glm_cn_session_state(
                device_id=str(session.metadata.get("glm_cn_device_id") or "").strip() if session else None,
            )
        if content:
            return content
        if attempt == 0:
            page.wait_for_timeout(1500)
            continue
        raise RuntimeError("GLM China DOM reply capture failed.")

    raise RuntimeError("GLM China DOM send retry budget exhausted.")


class _GLMIntlDomStreamProjector:
    def __init__(self) -> None:
        self._emitted_thinking = ""
        self._emitted_answer = ""
        self._in_think = False

    def push_snapshot(self, *, thinking_text: str, answer_text: str) -> list[str]:
        pieces: list[str] = []

        thinking_suffix, self._emitted_thinking = _advance_text_stream_state(
            self._emitted_thinking,
            thinking_text,
        )
        if thinking_suffix:
            if not self._in_think:
                pieces.append("<think>")
                self._in_think = True
            pieces.append(thinking_suffix)

        answer_suffix, self._emitted_answer = _advance_text_stream_state(
            self._emitted_answer,
            answer_text,
        )
        if answer_suffix:
            if self._in_think:
                pieces.append("</think>")
                self._in_think = False
            pieces.append(answer_suffix)

        return pieces

    def finish(self) -> str:
        if not self._in_think:
            return ""
        self._in_think = False
        return "</think>"


def _strip_glm_intl_think_markup(content: str) -> str:
    stripped = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL)
    stripped = re.sub(r"</?think>", "", stripped)
    return stripped.strip()


def _wait_for_glm_intl_input_ready(page, *, timeout_ms: int = 120000) -> None:
    page.wait_for_selector("#chat-input, textarea, [contenteditable='true'], input[type='text']", state="visible", timeout=timeout_ms)


def _prepare_glm_intl_dom_page(page) -> None:
    try:
        clicked = bool(
            page.evaluate(
                """
                () => {
                  const visible = (node) => !!node && (node.offsetParent !== null || node === document.activeElement);
                  const button = document.querySelector('#sidebar-new-chat-button');
                  if (!visible(button)) {
                    return false;
                  }
                  button.click();
                  return true;
                }
                """
            )
        )
        if clicked:
            page.wait_for_timeout(1200)
    except Exception:
        pass


def _send_glm_intl_dom_message(page, *, message: str) -> None:
    filled = _set_visible_textarea_value(page, ["#chat-input", "textarea", "input[type='text']"], message)
    if not filled:
        composer = page.locator("#chat-input, textarea").first
        composer.wait_for(state="visible", timeout=30000)
        composer.click(timeout=10000)
        try:
            composer.fill(message)
        except Exception:
            page.keyboard.press("Meta+A")
            page.keyboard.press("Backspace")
            page.keyboard.type(message, delay=12)
    page.wait_for_timeout(200)
    clicked = bool(
        page.evaluate(
            """
            () => {
              const visible = (node) => !!node && (node.offsetParent !== null || node === document.activeElement);
              const selectors = [
                '#send-message-button',
                'button.sendMessageButton',
                'button[aria-label*="Send"]',
                'button[type="submit"]',
              ];
              for (const selector of selectors) {
                const button = document.querySelector(selector);
                if (visible(button) && !button.disabled) {
                  button.click();
                  return true;
                }
              }
              return false;
            }
            """
        )
    )
    if clicked:
        return
    page.keyboard.press("Enter")


def _capture_glm_intl_dom_stream_state(page) -> dict[str, object]:
    snapshot = page.evaluate(
        """
        () => {
          const clean = (s) => (s || "")
            .replace(/[\\u200B-\\u200D\\uFEFF]/g, "")
            .replace(/\\s+/g, " ")
            .trim();
          const assistants = Array.from(document.querySelectorAll('.chat-assistant'));
          const latest = assistants.at(-1) || null;
          const latestParent = latest?.parentElement || null;
          const latestClone = latest ? latest.cloneNode(true) : null;
          if (latestClone) {
            latestClone.querySelectorAll('.thinking-chain-container, .thinking-block, blockquote').forEach((node) => node.remove());
          }
          const answerText = clean(latestClone?.innerText || latestClone?.textContent || "");
          const thinkingNode = latest?.querySelector('blockquote') || latest?.querySelector('.thinking-block');
          const thinkingText = clean(thinkingNode?.innerText || thinkingNode?.textContent || "");
          const regenerateVisible = !!latestParent?.querySelector('.regenerate-response-button');
          return {
            assistant_count: assistants.length,
            answer_text: answerText,
            thinking_text: thinkingText,
            regenerate_visible: regenerateVisible,
          };
        }
        """
    )
    if not isinstance(snapshot, dict):
        return {
            "assistant_count": 0,
            "answer_text": "",
            "thinking_text": "",
            "regenerate_visible": False,
        }
    return {
        "assistant_count": int(snapshot.get("assistant_count") or 0),
        "answer_text": str(snapshot.get("answer_text") or ""),
        "thinking_text": str(snapshot.get("thinking_text") or ""),
        "regenerate_visible": bool(snapshot.get("regenerate_visible")),
    }


def _stream_glm_intl_dom_completion(
    page,
    *,
    message: str,
    poll_interval_seconds: float = 0.05,
    timeout_seconds: float = 120.0,
) -> Iterator[str]:
    _prepare_glm_intl_dom_page(page)
    _wait_for_glm_intl_input_ready(page, timeout_ms=120000)
    before_count = int(page.evaluate("() => document.querySelectorAll('.chat-assistant').length || 0"))
    _send_glm_intl_dom_message(page, message=message)

    projector = _GLMIntlDomStreamProjector()
    last_answer = ""
    last_thinking = ""
    stable_rounds = 0
    saw_output = False
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        snapshot = _capture_glm_intl_dom_stream_state(page)
        assistant_count = int(snapshot.get("assistant_count") or 0)
        answer_text = str(snapshot.get("answer_text") or "").strip()
        thinking_text = str(snapshot.get("thinking_text") or "").strip()

        if assistant_count <= before_count and not answer_text and not thinking_text:
            time.sleep(poll_interval_seconds)
            continue

        pieces = projector.push_snapshot(thinking_text=thinking_text, answer_text=answer_text)
        if pieces:
            saw_output = True
            for piece in pieces:
                if piece:
                    yield piece

        if answer_text == last_answer and thinking_text == last_thinking:
            stable_rounds += 1
        else:
            last_answer = answer_text
            last_thinking = thinking_text
            stable_rounds = 0

        if (
            saw_output
            and answer_text
            and bool(snapshot.get("regenerate_visible"))
            and stable_rounds >= 1
        ):
            break
        if saw_output and answer_text and stable_rounds >= 4:
            break

        time.sleep(poll_interval_seconds)

    tail = projector.finish()
    if tail:
        yield tail
    if saw_output:
        return
    raise RuntimeError("GLM International DOM reply capture failed.")


def _dom_send_and_wait_glm_intl(page, message: str) -> str:
    return _strip_glm_intl_think_markup(
        "".join(_stream_glm_intl_dom_completion(page, message=message))
    )
