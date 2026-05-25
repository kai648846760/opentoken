import logging
import time
import traceback
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from opentoken.api.auth import maybe_require_api_key
from opentoken.api.routes.chat import router as chat_router
from opentoken.api.routes.embeddings import router as embeddings_router
from opentoken.api.routes.files import router as files_router
from opentoken.api.routes.health import router as health_router
from opentoken.api.routes.models import router as models_router
from opentoken.api.routes.responses import router as responses_router
from opentoken.api.routes.uploads import router as uploads_router
from opentoken.config.gateway_config import load_gateway_config


logger = logging.getLogger("opentoken.api")


def create_app() -> FastAPI:
    app = FastAPI(title="OpenToken")
    _bootstrap_gateway_config()

    @app.middleware("http")
    async def assign_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
        request.state.request_id = request_id
        started_ns = time.perf_counter_ns()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter_ns() - started_ns) / 1e6
            logger.exception(
                "request_failed request_id=%s method=%s path=%s elapsed_ms=%.2f",
                request_id,
                request.method,
                request.url.path,
                elapsed_ms,
            )
            raise
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1e6
        response.headers["X-Request-Id"] = request_id
        logger.info(
            "request_complete request_id=%s method=%s path=%s status=%s elapsed_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @app.middleware("http")
    async def require_api_key(request: Request, call_next):
        rejection = maybe_require_api_key(request)
        if rejection is not None:
            return rejection
        return await call_next(request)

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        # 把异常完整堆栈写日志（含请求 id），但响应不带任何内部细节，避免 cookie / 路径 / PoW 等泄漏到客户端
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(
            "unhandled_exception request_id=%s path=%s exc=%s\n%s",
            request_id,
            request.url.path,
            type(exc).__name__,
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Internal server error.",
                    "type": "internal_server_error",
                    "code": "internal_server_error",
                    "request_id": request_id,
                }
            },
        )

    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(files_router)
    app.include_router(uploads_router)
    app.include_router(embeddings_router)
    app.include_router(chat_router)
    app.include_router(responses_router)
    return app


def _bootstrap_gateway_config() -> None:
    """Read the gateway YAML config so its presence is observable in logs.

    The pool-aware router currently does not wire worker.page into provider
    adapters, so v1 ships in single-worker mode even when the YAML asks for
    multiple workers — make that obvious in logs instead of silently ignoring
    the config.
    """
    config = load_gateway_config()
    if config is None:
        return
    pool = getattr(config, "pool", None)
    if pool is None:
        logger.info("gateway_config_loaded pool=disabled")
        return
    if getattr(pool, "enabled", False):
        logger.warning(
            "gateway_config_pool_requested_but_not_active: "
            "the pool/failover stack is loaded but worker.page is not yet "
            "threaded into the provider adapters, so v1 runs in single-worker "
            "mode. Pool config has been read but not activated."
        )
    else:
        logger.info("gateway_config_loaded pool=disabled")
