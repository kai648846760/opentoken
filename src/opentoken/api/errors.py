from fastapi.responses import JSONResponse


def classify_provider_runtime_error(exc: RuntimeError) -> tuple[int, str]:
    """Pick a sensible OpenAI-shaped (status, error_type) for a RuntimeError
    raised by the router while talking to a provider. Provider failures are not
    request-validation errors — mapping them all to 400 invalid_request_error
    misleads clients into retrying a request that the upstream provider, not the
    request itself, broke. Shared by the chat and responses routes."""
    lowered = str(exc).lower()
    # Request-validation errors the router happens to raise (client asked for
    # something the gateway can't route): 400.
    if "unsupported model" in lowered or "no route configured" in lowered or "no adapter" in lowered:
        return 400, "invalid_request_error"
    # Provider not logged in / session dead / key not configured: 401 so clients
    # hit their re-auth flow.
    if ("missing" in lowered or "required" in lowered) and (
        "credential" in lowered or "api key" in lowered or "token" in lowered
    ):
        return 401, "authentication_error"
    # Session / auth-token failures, including the ways providers signal them
    # in a body rather than via HTTP status:
    #   - "session/credentials/token expired"               (our own messages)
    #   - "re-login" / "refresh the session"                (our own messages)
    #   - "unauthenticated" / "invalid_auth_token"          (Kimi gRPC body)
    #   - "no chat id"                                      (Qwen: stale session
    #     can't create a chat — returns 200 with no id)
    #
    # NOTE: match "expired" only when paired with an auth subject. Bare
    # "expired" misclassifies genuine upstream failures like "upstream
    # certificate expired" or "cache entry expired" as 401, sending clients
    # into a pointless re-login loop instead of retrying a 502.
    _AUTH_SIGNALS = (
        "session expired",
        "credentials expired",
        "token expired",
        "re-login",
        "re-log in",
        "refresh the session",
        "unauthenticated",
        "invalid_auth_token",
        "invalid auth token",
        "no chat id",
        "run `opentoken login",
    )
    if any(signal in lowered for signal in _AUTH_SIGNALS):
        return 401, "authentication_error"
    # Worker failures, parse errors, empty upstream responses, … — gateway-side
    # or upstream failure, 502 not 400.
    return 502, "api_error"


def openai_error_response(
    *,
    status_code: int,
    message: str,
    error_type: str,
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    error: dict[str, object] = {
        "message": message,
        "type": error_type,
    }
    if param is not None:
        error["param"] = param
    if code is not None:
        error["code"] = code
    return JSONResponse(
        status_code=status_code,
        content={"error": error},
    )
