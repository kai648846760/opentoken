from __future__ import annotations

import io

from fastapi.testclient import TestClient

from openclaw_algae.api.app import create_app
import openclaw_algae.api.routes.files as files_route_module


def test_files_create_retrieve_list_content_and_delete(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(files_route_module, "resolve_state_dir", lambda: tmp_path)
    client = TestClient(create_app())

    create_response = client.post(
        "/v1/files",
        files={"file": ("hello.txt", io.BytesIO(b"hello files"), "text/plain")},
        data={"purpose": "assistants"},
    )

    assert create_response.status_code == 200
    created = create_response.json()
    assert created["object"] == "file"
    assert created["filename"] == "hello.txt"
    assert created["purpose"] == "assistants"
    assert created["bytes"] == 11

    file_id = created["id"]

    retrieve_response = client.get(f"/v1/files/{file_id}")
    assert retrieve_response.status_code == 200
    assert retrieve_response.json() == created

    list_response = client.get("/v1/files")
    assert list_response.status_code == 200
    listed = list_response.json()
    assert listed["object"] == "list"
    assert listed["data"] == [created]

    content_response = client.get(f"/v1/files/{file_id}/content")
    assert content_response.status_code == 200
    assert content_response.content == b"hello files"
    assert content_response.headers["content-type"].startswith("text/plain")

    delete_response = client.delete(f"/v1/files/{file_id}")
    assert delete_response.status_code == 200
    assert delete_response.json() == {
        "id": file_id,
        "object": "file",
        "deleted": True,
    }

    missing_response = client.get(f"/v1/files/{file_id}")
    assert missing_response.status_code == 404
    assert missing_response.json() == {
        "error": {
            "message": f"File not found: {file_id}",
            "type": "invalid_request_error",
        }
    }
