from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import Response

from openclaw_algae.api.errors import openai_error_response
from openclaw_algae.config.paths import resolve_state_dir
from openclaw_algae.storage.file_store import create_file, delete_file, get_file, list_files, read_file_content

router = APIRouter()


@router.post("/v1/files")
async def files_create(
    file: UploadFile = File(...),
    purpose: str = Form(...),
):
    content = await file.read()
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
