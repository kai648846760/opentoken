from opentoken.providers.prompts import stringify_message_content


def test_stringify_message_content_keeps_attachment_markers() -> None:
    content = stringify_message_content(
        [
            {"type": "input_text", "text": "看看这张图"},
            {
                "type": "input_image",
                "image_url": {
                    "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA",
                },
            },
            {
                "type": "input_file",
                "filename": "report.pdf",
                "file_url": "https://example.com/report.pdf",
            },
        ]
    )

    assert "看看这张图" in content
    assert "[Attached image: data URI (image/png)]" in content
    assert "[Attached file: report.pdf | https://example.com/report.pdf]" in content


def test_stringify_message_content_preserves_interleaved_order() -> None:
    """Items must render in the order they appear. Previously text was bucketed
    ahead of all attachments, so [image, text] came out text-first and
    positional references ('the second photo') were lost."""
    content = stringify_message_content(
        [
            {"type": "input_image", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "input_text", "text": "describe the image above"},
        ]
    )
    lines = content.splitlines()
    # The image marker must come BEFORE the text, matching input order.
    assert lines[0].startswith("[Attached image")
    assert lines[1] == "describe the image above"


def test_stringify_message_content_embeds_text_file_contents_from_data_uri() -> None:
    content = stringify_message_content(
        [
            {"type": "input_text", "text": "请阅读附件"},
            {
                "type": "input_file",
                "filename": "notes.txt",
                "file_data": "data:text/plain;base64,SGVsbG8gZnJvbSBhdHRhY2htZW50IQ==",
            },
        ]
    )

    assert "请阅读附件" in content
    assert "[Attached file: notes.txt | data URI (text/plain)]" in content
    assert "[Attached file content: notes.txt]" in content
    assert "Hello from attachment!" in content


def test_stringify_message_content_does_not_read_local_file_paths(tmp_path) -> None:
    file_path = tmp_path / "secret.md"
    file_path.write_text("SENSITIVE_CONTENTS_DO_NOT_LEAK", encoding="utf-8")

    content = stringify_message_content(
        [
            {
                "type": "input_file",
                "filename": "secret.md",
                "file_url": str(file_path),
            }
        ]
    )

    assert "SENSITIVE_CONTENTS_DO_NOT_LEAK" not in content
    assert "Attached file content" not in content


def test_stringify_message_content_does_not_follow_file_uri(tmp_path) -> None:
    file_path = tmp_path / "secret.md"
    file_path.write_text("FILE_URI_SECRET", encoding="utf-8")

    content = stringify_message_content(
        [
            {
                "type": "input_file",
                "filename": "secret.md",
                "file_url": f"file://{file_path}",
            }
        ]
    )

    assert "FILE_URI_SECRET" not in content


def test_stringify_message_content_rejects_private_http_url() -> None:
    content = stringify_message_content(
        [
            {
                "type": "input_file",
                "filename": "metadata",
                "file_url": "http://169.254.169.254/latest/meta-data/",
            }
        ]
    )

    # Private/metadata IPs must be blocked by SSRF guard — content must not include any fetched body.
    assert "Attached file content" not in content


def test_stringify_message_content_rejects_loopback_http_url() -> None:
    content = stringify_message_content(
        [
            {
                "type": "input_file",
                "filename": "loop",
                "file_url": "http://127.0.0.1:8080/secret",
            }
        ]
    )

    assert "Attached file content" not in content
