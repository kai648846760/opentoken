from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import Response

from opentoken.api.errors import openai_error_response
from opentoken.config.paths import resolve_state_dir
from opentoken.storage.file_store import create_file, delete_file, get_file, list_files, read_file_content

router = APIRouter()


# OpenAI's /v1/files endpoint nominally caps at 512 MB but we set a smaller default
# because the underlying store buffers fully in memory before persisting.
_MAX_UPLOAD_BYTES = 100 * 1024 * 1024
_UPLOAD_CHUNK_SIZE = 1024 * 1024


@router.post("/v1/files")
async def files_create(
    file: UploadFile = File(...),
    purpose: str = Form(...),
):
    # Read in chunks so a single malicious request can't allocate an unbounded
    # amount of memory before we get to enforce the size cap.
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_UPLOAD_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > _MAX_UPLOAD_BYTES:
            return openai_error_response(
                status_code=413,
                message=(
                    f"Uploaded file exceeds the maximum size of "
                    f"{_MAX_UPLOAD_BYTES // (1024 * 1024)} MiB."
                ),
                error_type="invalid_request_error",
            )
        chunks.append(chunk)
    content = b"".join(chunks)
    created = create_file(
        resolve_state_dir(),
        filename=file.filename or "upload.bin",
        content=content,
        purpose=purpose,
        mime_type=file.content_type,
    )
    return created


@router.get("/v1/files")
def files_list() -> dict[str, object]:
    return {
        "object": "list",
        "data": list_files(resolve_state_dir()),
    }


@router.get("/v1/files/{file_id}")
def files_retrieve(file_id: str):
    entry = get_file(resolve_state_dir(), file_id)
    if entry is None:
        return openai_error_response(
            status_code=404,
            message=f"File not found: {file_id}",
            error_type="invalid_request_error",
        )
    return entry


@router.get("/v1/files/{file_id}/content")
def files_content(file_id: str):
    resolved = read_file_content(resolve_state_dir(), file_id)
    if resolved is None:
        return openai_error_response(
            status_code=404,
            message=f"File not found: {file_id}",
            error_type="invalid_request_error",
        )
    metadata, content = resolved
    return Response(content=content, media_type=str(metadata.get("mime_type", "application/octet-stream")))


@router.delete("/v1/files/{file_id}", response_model=None)
def files_delete(file_id: str):
    deleted = delete_file(resolve_state_dir(), file_id)
    if not deleted:
        return openai_error_response(
            status_code=404,
            message=f"File not found: {file_id}",
            error_type="invalid_request_error",
        )
    return {
        "id": file_id,
        "object": "file",
        "deleted": True,
    }
