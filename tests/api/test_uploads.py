from __future__ import annotations

import io

from fastapi.testclient import TestClient

from opentoken.api.app import create_app
import opentoken.api.routes.uploads as uploads_route_module


def test_uploads_create_add_part_complete_and_cancel(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(uploads_route_module, "resolve_state_dir", lambda: tmp_path)
    client = TestClient(create_app())

    create_response = client.post(
        "/v1/uploads",
        json={
            "filename": "greeting.txt",
            "bytes": 11,
            "mime_type": "text/plain",
            "purpose": "assistants",
        },
    )

    assert create_response.status_code == 200
    upload = create_response.json()
    assert upload["object"] == "upload"
    assert upload["status"] == "created"
    upload_id = upload["id"]

    part_response = client.post(
        f"/v1/uploads/{upload_id}/parts",
        files={"data": ("part-1", io.BytesIO(b"hello world"), "application/octet-stream")},
    )
    assert part_response.status_code == 200
    part = part_response.json()
    assert part["object"] == "upload.part"
    assert part["bytes"] == 11

    complete_response = client.post(
        f"/v1/uploads/{upload_id}/complete",
        json={},
    )
    assert complete_response.status_code == 200
    completed = complete_response.json()
    assert completed["object"] == "file"
    assert completed["filename"] == "greeting.txt"
    assert completed["bytes"] == 11

    cancel_create_response = client.post(
        "/v1/uploads",
        json={
            "filename": "cancel.txt",
            "bytes": 5,
            "mime_type": "text/plain",
            "purpose": "assistants",
        },
    )
    cancel_upload_id = cancel_create_response.json()["id"]
    cancel_response = client.post(f"/v1/uploads/{cancel_upload_id}/cancel")
    assert cancel_response.status_code == 200
    assert cancel_response.json()["status"] == "cancelled"


def test_uploads_create_rejects_absurd_declared_size(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(uploads_route_module, "resolve_state_dir", lambda: tmp_path)
    client = TestClient(create_app())
    # 1 TB declared total exceeds the 8 GiB ceiling -> 422 (pydantic validation).
    response = client.post(
        "/v1/uploads",
        json={
            "filename": "huge.bin",
            "bytes": 1024 * 1024 * 1024 * 1024,
            "mime_type": "application/octet-stream",
            "purpose": "assistants",
        },
    )
    assert response.status_code == 422


def test_uploads_add_part_rejects_oversize_part(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(uploads_route_module, "resolve_state_dir", lambda: tmp_path)
    # Shrink the cap so we don't have to actually send 100 MiB in the test.
    monkeypatch.setattr(uploads_route_module, "_MAX_PART_BYTES", 1024)
    client = TestClient(create_app())

    create_response = client.post(
        "/v1/uploads",
        json={"filename": "f.bin", "bytes": 5000, "mime_type": "application/octet-stream", "purpose": "assistants"},
    )
    upload_id = create_response.json()["id"]

    response = client.post(
        f"/v1/uploads/{upload_id}/parts",
        files={"data": ("part", b"x" * 4096, "application/octet-stream")},
    )
    assert response.status_code == 413
    assert "maximum part size" in response.json()["error"]["message"]
