from fastapi.responses import JSONResponse


def openai_error_response(
    *,
    status_code: int,
    message: str,
    error_type: str,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
            }
        },
    )
