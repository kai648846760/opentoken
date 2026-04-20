from openclaw_algae.providers.prompts import stringify_message_content


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


def test_stringify_message_content_embeds_local_text_file_contents(tmp_path) -> None:
    file_path = tmp_path / "report.md"
    file_path.write_text("# Title\n\nlocal attachment body", encoding="utf-8")

    content = stringify_message_content(
        [
            {
                "type": "input_file",
                "filename": "report.md",
                "file_url": str(file_path),
            }
        ]
    )

    assert f"[Attached file: report.md | {file_path}]" in content
    assert "[Attached file content: report.md]" in content
    assert "local attachment body" in content
