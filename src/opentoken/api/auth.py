import hmac
import json
import threading
from pathlib import Path

from fastapi import Request

from opentoken.api.errors import openai_error_response
from opentoken.config.paths import resolve_app_config_path


_CACHE_LOCK = threading.Lock()
_CACHED_API_KEY: str | None = None
_CACHED_MTIME_NS: int | None = None
_CACHED_PATH: Path | None = None


def maybe_require_api_key(request: Request):
    if request.url.path == "/health":
        return None

    expected_api_key, keyless_explicit = _get_expected_api_key(resolve_app_config_path())
    if expected_api_key is None:
        # The config file EXISTS but couldn't be parsed (corrupt/truncated).
        # Fail CLOSED: we must never serve authenticated provider sessions
        # unauthenticated just because we couldn't read the key.
        return openai_error_response(
            status_code=503,
            message="Gateway configuration is unreadable; cannot verify the API key.",
            error_type="api_error",
        )
    if not expected_api_key:
        # 空 / 缺失 api_key 不再隐式等于 "keyless 本地模式"。配置 rotation 中
        # 用户清空 key 准备换新的、或者无意把 key 设成空白,如果默认就放行,等于
        # 网关短暂全 open（fail-open 漏洞）。要开 keyless 必须显式声明
        # "keyless_local": true,这样意图明确。否则 fail-closed 503。
        # 配置文件本身不存在（首次启动前）仍然 keyless,那是开发场景。
        if keyless_explicit:
            return None
        return openai_error_response(
            status_code=503,
            message=(
                "Gateway has no api_key configured. Set api_key in config.json, "
                'or explicitly opt into keyless local mode with "keyless_local": true.'
            ),
            error_type="api_error",
        )

    authorization = request.headers.get("Authorization", "")
    expected_header = f"Bearer {expected_api_key}"
    # 常量时间比较：避免逐字节 timing 探测 API key 内容
    if not hmac.compare_digest(authorization.encode("utf-8"), expected_header.encode("utf-8")):
        return openai_error_response(
            status_code=401,
            message="Invalid or missing API key.",
            error_type="authentication_error",
        )
    return None


def _get_expected_api_key(config_path: Path) -> tuple[str | None, bool]:
    """Return (api_key, keyless_explicit).

      - (key, _)        when an api_key is configured
      - ("", True)      when the config file is absent — first-run dev path,
                        treated as keyless
      - ("", True)      when the config exists AND sets "keyless_local": true,
                        an explicit opt-in for keyless local mode
      - ("", False)     when the config exists but api_key is empty AND
                        keyless_local isn't set — caller fails closed (this
                        is the rotation / fat-finger scenario that used to
                        silently open the gateway)
      - (None, _)       when the config file exists but is corrupt / unreadable
                        — caller fails closed
    """
    global _CACHED_API_KEY, _CACHED_MTIME_NS, _CACHED_PATH
    try:
        stat = config_path.stat()
        mtime_ns = stat.st_mtime_ns
    except FileNotFoundError:
        with _CACHE_LOCK:
            _CACHED_API_KEY = None
            _CACHED_MTIME_NS = None
            _CACHED_PATH = config_path
        return "", True
    except OSError:
        # 任何其它 OSError（权限被撤销、FS 异常等）—— fail-closed 503。
        return None, False

    with _CACHE_LOCK:
        if (
            _CACHED_PATH == config_path
            and _CACHED_MTIME_NS == mtime_ns
            and _CACHED_API_KEY is not None
        ):
            # 缓存里的 keyless_explicit 用一个简单约定：缓存值为 "" 时,我们重读
            # payload 判定 keyless_explicit（fast-path 仅缓存非空 key）。
            return _CACHED_API_KEY, False

    payload = _load_existing_app_config(config_path)
    if payload is None:
        # 文件存在但 parse 失败 / 不是 JSON object —— fail-closed。
        return None, False
    api_key = str(payload.get("api_key", "")).strip()
    keyless_explicit = bool(payload.get("keyless_local", False))

    # 只缓存非空 key —— 空 key 路径每次重读,以便用户改了 keyless_local 立即生效。
    if api_key:
        with _CACHE_LOCK:
            _CACHED_API_KEY = api_key
            _CACHED_MTIME_NS = mtime_ns
            _CACHED_PATH = config_path
    return api_key, keyless_explicit


def reset_auth_cache() -> None:
    """Drop cached config; useful for tests / config rotation."""
    global _CACHED_API_KEY, _CACHED_MTIME_NS, _CACHED_PATH
    with _CACHE_LOCK:
        _CACHED_API_KEY = None
        _CACHED_MTIME_NS = None
        _CACHED_PATH = None


def _load_existing_app_config(config_path: Path) -> dict[str, object] | None:
    if not config_path.exists():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # Corrupt / truncated config must NOT crash the auth middleware on every
        # request (a 500 storm). Return None; the caller fails closed.
        return None
    if not isinstance(payload, dict):
        return None
    return payload
