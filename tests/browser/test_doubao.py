from pathlib import Path

import openclaw_algae.browser.doubao as doubao_module


def test_capture_doubao_browser_credentials_raises_when_camoufox_launch_fails(
    monkeypatch, tmp_path: Path
) -> None:
    class BrokenPlaywright:
        chromium = None

        def __enter__(self):
            self.chromium = self
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def launch_persistent_context(self, user_data_dir: str, headless: bool):
            raise RuntimeError(
                "BrowserType.launch_persistent_context: Failed to launch the browser process."
            )

    monkeypatch.setattr(doubao_module, "prepare_browser_state_dir", lambda state_dir, provider: state_dir / provider)
    monkeypatch.setattr(doubao_module, "require_sync_playwright", lambda: (lambda: BrokenPlaywright()))
    try:
        doubao_module.capture_doubao_browser_credentials(state_dir=tmp_path)
    except RuntimeError as exc:
        assert "Failed to launch the browser process" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")
