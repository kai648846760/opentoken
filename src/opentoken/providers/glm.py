from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import time
from collections.abc import Callable
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode
from uuid import uuid4

import httpx

from opentoken.gateway.normalized import NormalizedChatRequest
from opentoken.models.provider_credentials import ProviderCredentialRecord
from opentoken.providers._client_cache import BoundedClientCache, close_httpx_backed_client
from opentoken.providers.base import (
    ChatResponse,
    ProviderAdapter,
    ProviderRateLimitError,
    raise_for_provider_auth,
)


def _raise_for_glm_rate_limit(status_code: int, *, label: str) -> None:
    """Map 429 to ProviderRateLimitError so the gateway returns a proper
    rate_limit_error instead of letting raise_for_status surface a generic
    HTTPStatusError → 502 api_error. Callers invoke this between
    raise_for_provider_auth (401/403) and raise_for_status."""
    if status_code == 429:
        raise ProviderRateLimitError(
            f"{label} rate-limited. Please retry after a backoff."
        )
from opentoken.providers.prompts import build_role_prompt

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
_GLM_INTL_SIGNATURE_SECRET = "key-@@@@)))()((9))-xxxx&&&%%%%%"
_GLM_INTL_FRONTEND_VERSION_FALLBACK = "prod-fe-1.1.12"
_GLM_INTL_FRONTEND_VERSION_PATTERN = re.compile(r"/z-ai/frontend/([^/]+)/_app/immutable/")
_GLM_INTL_MODEL_CANDIDATES = {
    "glm-4-plus": ("GLM-5-Turbo", "glm-4.7", "glm-5", "GLM-5.1"),
    "glm-4-think": ("GLM-5.1", "zero", "glm-5", "GLM-5-Turbo"),
}


def _glm_chat_mode_for_model(model: str) -> str:
    model_lower = model.strip().lower()
    if "think" in model_lower or model_lower.endswith("-zero"):
        return "zero"
    return "normal"


def _glm_meta_data_for_model(model: str) -> dict[str, object]:
    return {
        "cogview": {"rm_label_watermark": False},
        "is_test": False,
        "input_question_type": "xxxx",
        "channel": "",
        "draft_id": "",
        "chat_mode": _glm_chat_mode_for_model(model),
        "is_networking": False,
        "quote_log_id": "",
        "platform": "pc",
    }


def _generate_glm_sign() -> dict[str, str]:
    """Generate GLM anti-bot signature headers."""
    ts = str(int(time.time() * 1000))
    digits = [int(c) for c in ts]
    digit_sum = sum(digits) - digits[-2]
    check = digit_sum % 10
    timestamp = ts[:-2] + str(check) + ts[-1:]
    nonce = uuid4().hex.replace("-", "")
    sign = hashlib.md5(f"{timestamp}-{nonce}-{_GLM_SIGN_SECRET}".encode()).hexdigest()
    return {"timestamp": timestamp, "nonce": nonce, "sign": sign}


def _compute_glm_intl_request_signature(
    *,
    sorted_payload: str,
    signature_prompt: str,
    timestamp_ms: str,
) -> str:
    prompt_b64 = base64.b64encode(signature_prompt.encode("utf-8")).decode("ascii")
    payload = f"{sorted_payload}|{prompt_b64}|{timestamp_ms}"
    bucket = str(int(int(timestamp_ms) / (5 * 60 * 1000)))
    bucket_secret = hmac.new(
        _GLM_INTL_SIGNATURE_SECRET.encode("utf-8"),
        bucket.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return hmac.new(
        bucket_secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _extract_glm_intl_frontend_version(html: str) -> str:
    match = _GLM_INTL_FRONTEND_VERSION_PATTERN.search(str(html or ""))
    if match:
        version = match.group(1).strip()
        if version:
            return version
    return _GLM_INTL_FRONTEND_VERSION_FALLBACK


def _extract_glm_intl_phased_segments(payload: dict[str, object]) -> list[tuple[str, str]]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    phase = str(data.get("phase") or data.get("data", {}).get("phase") or "").strip().lower()
    delta = data.get("delta_content")
    if not isinstance(delta, str):
        nested = data.get("data")
        if isinstance(nested, dict):
            nested_delta = nested.get("delta_content")
            if isinstance(nested_delta, str):
                delta = nested_delta
    if not isinstance(delta, str) or not delta:
        return []
    if phase == "thinking":
        return [("think", delta)]
    if phase == "answer":
        return [("answer", delta)]
    return []


def _extract_glm_intl_stream_error(payload: dict[str, object]) -> str:
    if not isinstance(payload, dict):
        return ""
    candidates = [payload]
    data = payload.get("data")
    if isinstance(data, dict):
        candidates.append(data)
        nested = data.get("data")
        if isinstance(nested, dict):
            candidates.append(nested)
    for candidate in candidates:
        error = candidate.get("error")
        if isinstance(error, str) and error.strip():
            return error.strip()
        if isinstance(error, dict):
            detail = error.get("detail") or error.get("message") or error.get("code")
            if detail:
                return str(detail).strip()
    return ""


class _GLMIntlMarkedStreamProjector:
    def __init__(self) -> None:
        self._in_think = False

    def push(self, *, phase: str, text: str) -> list[str]:
        if not text:
            return []
        pieces: list[str] = []
        if phase == "think":
            if not self._in_think:
                pieces.append("<think>")
                self._in_think = True
            pieces.append(text)
            return pieces
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


class GLMApiClient:
    """Real API client for GLM China (chatglm.cn)."""

    def __init__(
        self,
        credentials: ProviderCredentialRecord,
        *,
        base_url: str = "https://chatglm.cn",
        client: httpx.Client | None = None,
    ) -> None:
        self._credentials = credentials
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=60.0, trust_env=False)
        self._access_token: str = ""
        self._conversation_id: str = ""
        self._device_id: str = self._extract_device_id()

    def _extract_cookie_value(self, name: str) -> str:
        cookie = self._credentials.cookie or ""
        for item in cookie.split(";"):
            if "=" in item:
                k, v = item.strip().split("=", 1)
                if k == name:
                    return v
        return ""

    def _extract_device_id(self) -> str:
        cookie_device_id = self._extract_cookie_value("chatglm_device_id")
        if cookie_device_id:
            return cookie_device_id
        token = self._extract_cookie_value("chatglm_token") or self._extract_cookie_value("chatglm_refresh_token")
        if token:
            try:
                import base64
                parts = token.split(".")
                if len(parts) == 3:
                    payload_b64 = parts[1] + "=" * ((4 - len(parts[1]) % 4) % 4)
                    payload = json.loads(base64.b64decode(payload_b64))
                    device_id = payload.get("device_id", "")
                    if isinstance(device_id, str) and device_id:
                        return device_id
            except Exception:
                pass
        return uuid4().hex.replace("-", "")

    def build_headers(self) -> dict[str, str]:
        sign = _generate_glm_sign()
        auth_token = self._access_token or self._extract_cookie_value("chatglm_token")

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Cookie": self._credentials.cookie or "",
            "User-Agent": self._credentials.user_agent or "Mozilla/5.0",
            "Origin": self._base_url,
            "Referer": f"{self._base_url}/main/all",
            "App-Name": "chatglm",
            "X-App-Platform": "pc",
            "X-App-Version": "0.0.1",
            "X-App-fr": "default",
            "X-Device-Brand": "",
            "X-Device-Id": self._device_id,
            "X-Device-Model": "",
            "X-Exp-Groups": _GLM_X_EXP_GROUPS,
            "X-Lang": "zh",
            "X-Nonce": sign["nonce"],
            "X-Request-Id": uuid4().hex.replace("-", ""),
            "X-Sign": sign["sign"],
            "X-Timestamp": sign["timestamp"],
        }
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        return headers

    def _refresh_access_token(self) -> None:
        refresh_token = self._extract_cookie_value("chatglm_refresh_token")
        if not refresh_token:
            self._access_token = self._extract_cookie_value("chatglm_token")
            return
        sign = _generate_glm_sign()
        request_id = uuid4().hex.replace("-", "")
        response = self._client.post(
            f"{self._base_url}/chatglm/user-api/user/refresh",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {refresh_token}",
                "App-Name": "chatglm",
                "X-App-Platform": "pc",
                "X-App-Version": "0.0.1",
                "X-Device-Id": self._device_id,
                "X-Request-Id": request_id,
                "X-Sign": sign["sign"],
                "X-Nonce": sign["nonce"],
                "X-Timestamp": sign["timestamp"],
            },
            json={},
        )
        if response.status_code == 200:
            try:
                data = response.json()
                access_token = (
                    data.get("result", {}).get("access_token")
                    or data.get("result", {}).get("accessToken")
                    or data.get("accessToken")
                    or ""
                )
                if isinstance(access_token, str) and access_token:
                    self._access_token = access_token
                    return
            except Exception:
                pass
        self._access_token = self._extract_cookie_value("chatglm_token")


    def chat_completion(self, *, message: str, model: str) -> str:
        assistant_id = _GLM_ASSISTANT_ID_MAP.get(model, "65940acff94777010aa6b796")
        self._refresh_access_token()

        payload = {
            "assistant_id": assistant_id,
            "conversation_id": self._conversation_id,
            "project_id": "",
            "chat_type": "user_chat",
            "meta_data": _glm_meta_data_for_model(model),
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": message}]},
            ],
        }

        response = self._client.post(
            f"{self._base_url}/chatglm/backend-api/assistant/stream",
            headers=self.build_headers(),
            json=payload,
        )

        if response.status_code == 401:
            self._access_token = ""
            self._refresh_access_token()
            response = self._client.post(
                f"{self._base_url}/chatglm/backend-api/assistant/stream",
                headers=self.build_headers(),
                json=payload,
            )

        raise_for_provider_auth(
            response.status_code, provider="GLM", login_command="opentoken login glm china"
        )
        _raise_for_glm_rate_limit(response.status_code, label="GLM China")
        response.raise_for_status()
        content, conversation_id = _parse_glm_sse_response(response.text)
        if conversation_id:
            self._conversation_id = conversation_id
        if not content:
            raise RuntimeError("GLM chat completion returned no text content.")
        return content

    def iter_chat_completion_text(self, *, message: str, model: str) -> Iterator[str]:
        yield from self._iter_chat_completion_text(message=message, model=model, allow_retry=True)

    def _iter_chat_completion_text(
        self,
        *,
        message: str,
        model: str,
        allow_retry: bool,
    ) -> Iterator[str]:
        assistant_id = _GLM_ASSISTANT_ID_MAP.get(model, "65940acff94777010aa6b796")
        self._refresh_access_token()

        payload = {
            "assistant_id": assistant_id,
            "conversation_id": self._conversation_id,
            "project_id": "",
            "chat_type": "user_chat",
            "meta_data": _glm_meta_data_for_model(model),
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": message}]},
            ],
        }

        with self._client.stream(
            "POST",
            f"{self._base_url}/chatglm/backend-api/assistant/stream",
            headers=self.build_headers(),
            json=payload,
        ) as response:
            if response.status_code == 401 and allow_retry:
                response.read()
                self._access_token = ""
                yield from self._iter_chat_completion_text(
                    message=message,
                    model=model,
                    allow_retry=False,
                )
                return

            raise_for_provider_auth(
                response.status_code, provider="GLM", login_command="opentoken login glm china"
            )
            _raise_for_glm_rate_limit(response.status_code, label="GLM China")
            response.raise_for_status()
            emitted = ""
            for raw_line in response.iter_lines():
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                try:
                    data = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue
                cid = data.get("conversation_id")
                if isinstance(cid, str) and cid:
                    self._conversation_id = cid
                for candidate in _extract_glm_text_candidates(data):
                    suffix, emitted = _advance_streamed_text_state(emitted, candidate)
                    if suffix:
                        yield suffix


class GLMWebAdapter(ProviderAdapter):
    def __init__(
        self,
        *,
        client_factory: Callable[[ProviderCredentialRecord], GLMApiClient] | None = None,
    ) -> None:
        self._client_factory = client_factory or (
            lambda credentials: GLMApiClient(credentials)
        )
        self._client_cache: BoundedClientCache[GLMApiClient] = BoundedClientCache(closer=close_httpx_backed_client)

    def _client_key(self, credentials: ProviderCredentialRecord) -> str:
        return f"{credentials.provider}:{credentials.cookie}:{credentials.user_agent}"

    def chat(
        self,
        request: NormalizedChatRequest,
        credentials: ProviderCredentialRecord | None = None,
    ) -> ChatResponse:
        if credentials is None:
            raise RuntimeError("Missing GLM credentials. Run `opentoken login glm cn` first.")
        key = self._client_key(credentials)
        client = self._client_cache.get(key)
        if client is None:
            client = self._client_factory(credentials)
            self._client_cache.set(key, client)
        content = client.chat_completion(
            message=build_role_prompt(request),
            model=request.model.rsplit("/", 1)[-1],
        )
        return ChatResponse(model=request.model, content=content)

    def stream_chat(
        self,
        request: NormalizedChatRequest,
        credentials: ProviderCredentialRecord | None = None,
    ) -> Iterator[str] | None:
        if credentials is None:
            raise RuntimeError("Missing GLM credentials. Run `opentoken login glm cn` first.")
        key = self._client_key(credentials)
        client = self._client_cache.get(key)
        if client is None:
            client = self._client_factory(credentials)
            self._client_cache.set(key, client)
        stream_method = getattr(client, "iter_chat_completion_text", None)
        if not callable(stream_method):
            return None
        return stream_method(
            message=build_role_prompt(request),
            model=request.model.rsplit("/", 1)[-1],
        )


class GLMIntlApiClient(GLMApiClient):
    """API client for GLM International (chat.z.ai)."""

    def __init__(
        self,
        credentials: ProviderCredentialRecord,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(credentials, base_url="https://chat.z.ai", client=client)
        self._intl_auth_context: dict[str, str] | None = None
        self._frontend_version: str = ""
        self._site_model_ids: list[str] | None = None

    def build_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Cookie": self._credentials.cookie or "",
            "User-Agent": self._credentials.user_agent or "Mozilla/5.0",
            "Referer": "https://chat.z.ai/",
            "Origin": "https://chat.z.ai",
        }

    def _json_headers(
        self,
        *,
        accept: str = "application/json, text/plain, */*",
        authorization: str | None = None,
        frontend_version: str | None = None,
    ) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": accept,
            "Cookie": self._credentials.cookie or "",
            "User-Agent": self._credentials.user_agent or "Mozilla/5.0",
            "Referer": "https://chat.z.ai/",
            "Origin": "https://chat.z.ai",
        }
        if authorization:
            headers["Authorization"] = f"Bearer {authorization}"
        if frontend_version:
            headers["X-FE-Version"] = frontend_version
        return headers

    def _fetch_auth_context(self) -> dict[str, str]:
        if self._intl_auth_context is not None:
            return self._intl_auth_context
        response = self._client.get(
            f"{self._base_url}/api/v1/auths/",
            headers=self._json_headers(),
        )
        response.raise_for_status()
        payload = response.json()
        token = str(payload.get("token") or self._extract_cookie_value("token") or "").strip()
        user_id = str(payload.get("id") or "").strip()
        user_name = str(payload.get("name") or "User").strip() or "User"
        if not token or not user_id:
            raise RuntimeError("GLM Intl auth context is missing token or user id.")
        self._intl_auth_context = {
            "token": token,
            "user_id": user_id,
            "user_name": user_name,
        }
        return self._intl_auth_context

    def _fetch_frontend_version(self) -> str:
        if self._frontend_version:
            return self._frontend_version
        response = self._client.get(
            f"{self._base_url}/",
            headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Cookie": self._credentials.cookie or "",
                "User-Agent": self._credentials.user_agent or "Mozilla/5.0",
            },
        )
        response.raise_for_status()
        self._frontend_version = _extract_glm_intl_frontend_version(response.text)
        return self._frontend_version

    def _fetch_site_model_ids(self, *, token: str) -> list[str]:
        if self._site_model_ids is not None:
            return list(self._site_model_ids)
        response = self._client.get(
            f"{self._base_url}/api/models",
            headers=self._json_headers(authorization=token),
        )
        response.raise_for_status()
        payload = response.json()
        seen: list[str] = []
        for item in payload.get("data", []):
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id") or "").strip()
            if model_id and model_id not in seen:
                seen.append(model_id)
        self._site_model_ids = seen
        return list(seen)

    def _resolve_site_model_id(self, *, requested_model: str, token: str) -> str:
        available = self._fetch_site_model_ids(token=token)
        requested = str(requested_model or "").strip()
        if requested in available:
            return requested
        preferences = _GLM_INTL_MODEL_CANDIDATES.get(requested.lower())
        if preferences:
            for candidate in preferences:
                if candidate in available:
                    return candidate
        if requested.lower().endswith("think") or "think" in requested.lower():
            for candidate in ("GLM-5.1", "zero", "glm-5", "GLM-5-Turbo"):
                if candidate in available:
                    return candidate
        for candidate in ("GLM-5-Turbo", "glm-4.7", "glm-5", "GLM-5.1"):
            if candidate in available:
                return candidate
        return requested or (available[0] if available else "GLM-5-Turbo")

    def _resolve_site_chat_model_id(self, *, requested_model: str, token: str) -> str:
        available = self._fetch_site_model_ids(token=token)
        if "GLM-5.1" in available:
            return "GLM-5.1"
        return self._resolve_site_model_id(requested_model=requested_model, token=token)

    def _build_sorted_payload(self, *, timestamp_ms: str, request_id: str, user_id: str) -> str:
        entries = sorted(
            (
                ("timestamp", timestamp_ms),
                ("requestId", request_id),
                ("user_id", user_id),
            ),
            key=lambda item: item[0],
        )
        return ",".join(f"{key},{value}" for key, value in entries)

    def _build_query_params(
        self,
        *,
        timestamp_ms: str,
        request_id: str,
        user_id: str,
        token: str,
        chat_id: str,
    ) -> dict[str, str]:
        now = datetime.now().astimezone()
        timezone_name = str(now.tzinfo or "UTC")
        now_utc = now.astimezone(timezone.utc)
        user_agent = self._credentials.user_agent or "Mozilla/5.0"
        browser_name = "Firefox" if "Firefox" in user_agent else "Unknown"
        os_name = "Mac OS" if "Mac OS X" in user_agent or "Macintosh" in user_agent else "Unknown"
        return {
            "timestamp": timestamp_ms,
            "signature_timestamp": timestamp_ms,
            "requestId": request_id,
            "user_id": user_id,
            "version": "0.0.1",
            "platform": "web",
            "token": token,
            "user_agent": user_agent,
            "language": "en-US",
            "languages": "en-US,en",
            "timezone": timezone_name,
            "cookie_enabled": "true",
            "screen_width": "1920",
            "screen_height": "1080",
            "screen_resolution": "1920x1080",
            "viewport_height": "1019",
            "viewport_width": "1920",
            "viewport_size": "1920x1019",
            "color_depth": "30",
            "pixel_ratio": "1",
            "current_url": f"https://chat.z.ai/c/{chat_id}",
            "pathname": f"/c/{chat_id}",
            "search": "",
            "hash": "",
            "host": "chat.z.ai",
            "hostname": "chat.z.ai",
            "protocol": "https:",
            "referrer": "",
            "title": "Z.ai - Free AI Chatbot & Agent powered by GLM-5.1 & GLM-5",
            "timezone_offset": str(-int((now.utcoffset() or timezone.utc.utcoffset(now) or 0).total_seconds() / 60)),
            "local_time": now_utc.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "utc_time": now_utc.strftime("%a, %d %b %Y %H:%M:%S GMT"),
            "is_mobile": "false",
            "is_touch": "false",
            "max_touch_points": "0",
            "browser_name": browser_name,
            "os_name": os_name,
        }

    def _build_prompt_variables(self, *, user_name: str) -> dict[str, str]:
        now = datetime.now().astimezone()
        return {
            "{{USER_NAME}}": user_name,
            "{{USER_LOCATION}}": "Unknown",
            "{{CURRENT_DATETIME}}": now.strftime("%Y-%m-%d %H:%M:%S"),
            "{{CURRENT_DATE}}": now.strftime("%Y-%m-%d"),
            "{{CURRENT_TIME}}": now.strftime("%H:%M:%S"),
            "{{CURRENT_WEEKDAY}}": now.strftime("%A"),
            "{{CURRENT_TIMEZONE}}": str(now.tzinfo or "UTC"),
            "{{USER_LANGUAGE}}": "en-US",
        }

    def _create_chat(
        self,
        *,
        message: str,
        requested_model: str,
        site_model: str,
        token: str,
        frontend_version: str,
        user_message_id: str,
        timestamp_seconds: int,
        timestamp_ms: int,
    ) -> str:
        response = self._client.post(
            f"{self._base_url}/api/v1/chats/new",
            headers=self._json_headers(
                authorization=token,
                frontend_version=frontend_version,
            ),
            json={
                "chat": {
                    "id": "",
                    "title": "New Chat",
                    "models": [site_model],
                    "params": {},
                    "history": {
                        "messages": {
                            user_message_id: {
                                "id": user_message_id,
                                "parentId": None,
                                "childrenIds": [],
                                "role": "user",
                                "content": message,
                                "timestamp": timestamp_seconds,
                                "models": [site_model],
                            }
                        },
                        "currentId": user_message_id,
                    },
                    "tags": [],
                    "flags": [],
                    "features": [
                        {
                            "type": "tool_selector",
                            "server": "tool_selector_h",
                            "status": "hidden",
                        }
                    ],
                    "mcp_servers": [],
                    "enable_thinking": "think" in requested_model.lower() or requested_model.lower().endswith("-zero"),
                    "auto_web_search": False,
                    "message_version": 1,
                    "extra": {},
                    "timestamp": timestamp_ms,
                    "type": "default",
                }
            },
        )
        response.raise_for_status()
        payload = response.json()
        chat_id = str(payload.get("id") or payload.get("chat", {}).get("id") or "").strip()
        if not chat_id:
            raise RuntimeError("GLM Intl chat creation returned no chat id.")
        return chat_id

    def _completion_request_parts(
        self,
        *,
        message: str,
        model: str,
    ) -> tuple[str, dict[str, str], dict[str, str], dict[str, object]]:
        auth = self._fetch_auth_context()
        token = auth["token"]
        user_id = auth["user_id"]
        user_name = auth["user_name"]
        chat_model = self._resolve_site_chat_model_id(requested_model=model, token=token)
        site_model = self._resolve_site_model_id(requested_model=model, token=token)
        frontend_version = self._fetch_frontend_version()
        timestamp_ms = int(time.time() * 1000)
        timestamp_seconds = int(time.time())
        request_id = str(uuid4())
        user_message_id = str(uuid4())
        chat_id = self._create_chat(
            message=message,
            requested_model=model,
            site_model=chat_model,
            token=token,
            frontend_version=frontend_version,
            user_message_id=user_message_id,
            timestamp_seconds=timestamp_seconds,
            timestamp_ms=timestamp_ms,
        )
        url_params = self._build_query_params(
            timestamp_ms=str(timestamp_ms),
            request_id=request_id,
            user_id=user_id,
            token=token,
            chat_id=chat_id,
        )
        signature = _compute_glm_intl_request_signature(
            sorted_payload=self._build_sorted_payload(
                timestamp_ms=str(timestamp_ms),
                request_id=request_id,
                user_id=user_id,
            ),
            signature_prompt=message,
            timestamp_ms=str(timestamp_ms),
        )
        headers = self._json_headers(
            accept="text/event-stream",
            authorization=token,
            frontend_version=frontend_version,
        )
        headers["X-Signature"] = signature
        completion_payload: dict[str, object] = {
            "stream": True,
            "model": site_model,
            "messages": [{"role": "user", "content": message}],
            "signature_prompt": message,
            "params": {},
            "extra": {},
            "features": {
                "image_generation": False,
                "web_search": False,
                "auto_web_search": False,
                "preview_mode": True,
                "flags": [],
                "vlm_tools_enable": False,
                "vlm_web_search_enable": False,
                "vlm_website_mode": False,
                "enable_thinking": "think" in model.lower() or model.lower().endswith("-zero"),
            },
            "variables": self._build_prompt_variables(user_name=user_name),
            "chat_id": chat_id,
            "id": str(uuid4()),
            "current_user_message_id": user_message_id,
            "current_user_message_parent_id": None,
            "background_tasks": {
                "title_generation": True,
                "tags_generation": True,
            },
        }
        return site_model, url_params, headers, completion_payload

    def iter_marked_chat_completion_text(
        self,
        *,
        message: str,
        model: str,
    ) -> Iterator[str]:
        _site_model, url_params, headers, completion_payload = self._completion_request_parts(
            message=message,
            model=model,
        )
        projector = _GLMIntlMarkedStreamProjector()
        saw_any_piece = False
        with self._client.stream(
            "POST",
            f"{self._base_url}/api/v2/chat/completions?{urlencode(url_params)}",
            headers=headers,
            json=completion_payload,
        ) as stream_response:
            _raise_for_glm_rate_limit(stream_response.status_code, label="GLM International")
            stream_response.raise_for_status()
            for raw_line in stream_response.iter_lines():
                line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else str(raw_line)
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    payload = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                error_detail = _extract_glm_intl_stream_error(payload)
                if error_detail:
                    raise RuntimeError(error_detail)
                for phase, text in _extract_glm_intl_phased_segments(payload):
                    for piece in projector.push(phase=phase, text=text):
                        if piece:
                            saw_any_piece = True
                            yield piece
        tail = projector.finish()
        if tail:
            saw_any_piece = True
            yield tail
        if not saw_any_piece:
            raise RuntimeError("GLM Intl chat completion returned no streamed text content.")

    def iter_chat_completion_text(self, *, message: str, model: str) -> Iterator[str]:
        inside_think = False
        for piece in self.iter_marked_chat_completion_text(message=message, model=model):
            if piece == "<think>":
                inside_think = True
                continue
            if piece == "</think>":
                inside_think = False
                continue
            if inside_think:
                continue
            if piece:
                yield piece

    def chat_completion(self, *, message: str, model: str) -> str:
        content = "".join(self.iter_chat_completion_text(message=message, model=model)).strip()
        if not content:
            raise RuntimeError("GLM Intl chat completion returned no text content.")
        return content


class GLMIntlWebAdapter(ProviderAdapter):
    def __init__(
        self,
        *,
        client_factory: Callable[[ProviderCredentialRecord], GLMIntlApiClient] | None = None,
    ) -> None:
        self._client_factory = client_factory or (
            lambda credentials: GLMIntlApiClient(credentials)
        )
        self._client_cache: BoundedClientCache[GLMIntlApiClient] = BoundedClientCache(closer=close_httpx_backed_client)

    def _get_client(self, credentials: ProviderCredentialRecord) -> GLMIntlApiClient:
        key = f"{credentials.provider}:{credentials.cookie}:{credentials.user_agent}"
        client = self._client_cache.get(key)
        if client is None:
            client = self._client_factory(credentials)
            self._client_cache.set(key, client)
        return client

    def chat(
        self,
        request: NormalizedChatRequest,
        credentials: ProviderCredentialRecord | None = None,
    ) -> ChatResponse:
        if credentials is None:
            raise RuntimeError("Missing GLM Intl credentials. Run `opentoken login glm international` first.")
        client = self._get_client(credentials)
        content = client.chat_completion(
            message=build_role_prompt(request),
            model=request.model.rsplit("/", 1)[-1],
        )
        return ChatResponse(model=request.model, content=content)

    def stream_chat(
        self,
        request: NormalizedChatRequest,
        credentials: ProviderCredentialRecord | None = None,
    ) -> Iterator[str] | None:
        if credentials is None:
            raise RuntimeError("Missing GLM Intl credentials. Run `opentoken login glm international` first.")
        client = self._get_client(credentials)
        stream_method = getattr(client, "iter_chat_completion_text", None)
        if not callable(stream_method):
            return None
        return stream_method(
            message=build_role_prompt(request),
            model=request.model.rsplit("/", 1)[-1],
        )


def _extract_glm_text_candidates(data: dict[str, object]) -> list[str]:
    candidates: list[str] = []
    for part in data.get("parts", []):
        if not isinstance(part, dict):
            continue
        for item in part.get("content", []):
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if isinstance(text, str) and text:
                    candidates.append(text)
    text = data.get("text") or data.get("content") or data.get("delta")
    if isinstance(text, str) and text:
        candidates.append(text)
    return candidates


def _advance_streamed_text_state(current: str, candidate: str) -> tuple[str, str]:
    if not candidate:
        return "", current
    if candidate.startswith(current):
        suffix = candidate[len(current) :]
        return suffix, candidate
    if current.startswith(candidate):
        return "", current
    return candidate, current + candidate


def _parse_glm_sse_response(payload: str) -> tuple[str, str]:
    """Parse GLM SSE response.

    GLM sends incremental full-text updates in parts[].content[].text.
    We keep the latest text and capture returned conversation_id for reuse.
    """
    last_text = ""
    conversation_id = ""
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        try:
            data = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        cid = data.get("conversation_id")
        if isinstance(cid, str) and cid:
            conversation_id = cid
        for part in data.get("parts", []):
            if not isinstance(part, dict):
                continue
            for item in part.get("content", []):
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    if isinstance(text, str) and text:
                        last_text = text
        # Fallbacks if response shape varies
        text = data.get("text") or data.get("content") or data.get("delta")
        if isinstance(text, str) and text:
            last_text = text
    return last_text, conversation_id
