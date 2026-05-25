from fastapi.responses import JSONResponse


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
