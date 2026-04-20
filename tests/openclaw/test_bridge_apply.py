import json

from openclaw_algae.openclaw.bridge import (
    apply_algae_provider_patch,
    build_algae_provider_patch,
)


def test_apply_algae_provider_patch_preserves_unrelated_settings(tmp_path) -> None:
    config_path = tmp_path / "openclaw.json"
    config_path.write_text(
        json.dumps(
            {
                "channels": {"default": "stable"},
                "models": {
                    "providers": {
                        "other": {
                            "baseUrl": "http://example.test/v1",
                            "apiKey": "${OTHER_API_KEY}",
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    patch = build_algae_provider_patch(
        base_url="http://127.0.0.1:32117/v1",
        api_key="test-algae-key",
    )

    backup_path = apply_algae_provider_patch(config_path, patch)

    assert backup_path is not None
    assert backup_path.exists()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["channels"]["default"] == "stable"
    assert payload["models"]["providers"]["other"]["baseUrl"] == "http://example.test/v1"
    assert payload["models"]["providers"]["algae"]["apiKey"] == "test-algae-key"
