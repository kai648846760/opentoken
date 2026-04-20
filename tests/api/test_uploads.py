from __future__ import annotations

import io

from fastapi.testclient import TestClient

from openclaw_algae.api.app import create_app
import openclaw_algae.api.routes.uploads as uploads_route_module


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
