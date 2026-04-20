from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel, Field

from openclaw_algae.api.errors import openai_error_response
from openclaw_algae.config.paths import resolve_state_dir
from openclaw_algae.storage.file_store import create_file
from openclaw_algae.storage.upload_store import add_upload_part, cancel_upload, complete_upload, create_upload

router = APIRouter()


class UploadCreateRequest(BaseModel):
    filename: str
    bytes: int = Field(ge=0)
    mime_type: str | None = None
    purpose: str


class UploadCompleteRequest(BaseModel):
    part_ids: list[str] | None = None


@router.post("/v1/uploads")
def uploads_create(payload: UploadCreateRequest) -> dict[str, object]:
    return create_upload(
        resolve_state_dir(),
        filename=payload.filename,
        expected_bytes=payload.bytes,
        mime_type=payload.mime_type,
        purpose=payload.purpose,
    )


@router.post("/v1/uploads/{upload_id}/parts")
async def uploads_add_part(upload_id: str, data: UploadFile = File(...)):
    created = add_upload_part(
        resolve_state_dir(),
        upload_id,
        content=await data.read(),
        content_type=data.content_type,
    )
    if created is None:
        return openai_error_response(
            status_code=404,
            message=f"Upload not found or unavailable: {upload_id}",
            error_type="invalid_request_error",
        )
    return created


@router.post("/v1/uploads/{upload_id}/complete")
def uploads_complete(upload_id: str, payload: UploadCompleteRequest | None = None):
    part_ids = payload.part_ids if payload is not None else None
    completed = complete_upload(resolve_state_dir(), upload_id, part_ids=part_ids)
    if completed is None:
        return openai_error_response(
            status_code=404,
            message=f"Upload not found or unavailable: {upload_id}",
            error_type="invalid_request_error",
        )
    upload, content = completed
    if int(upload.get("bytes", 0)) != len(content):
        return openai_error_response(
            status_code=400,
            message=(
                f"Upload {upload_id} expected {upload.get('bytes', 0)} bytes but received {len(content)} bytes"
            ),
            error_type="invalid_request_error",
        )
    return create_file(
        resolve_state_dir(),
        filename=str(upload.get("filename", "upload.bin")),
        content=content,
        purpose=str(upload.get("purpose", "assistants")),
        mime_type=str(upload.get("mime_type", "application/octet-stream")),
    )


@router.post("/v1/uploads/{upload_id}/cancel")
def uploads_cancel(upload_id: str):
    cancelled = cancel_upload(resolve_state_dir(), upload_id)
    if cancelled is None:
        return openai_error_response(
            status_code=404,
            message=f"Upload not found: {upload_id}",
            error_type="invalid_request_error",
        )
    return cancelled
