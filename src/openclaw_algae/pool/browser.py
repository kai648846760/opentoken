"""Browser launcher: manages a single Camoufox/Playwright browser instance."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class BrowserLauncher:
    """Launch and manage a single browser instance."""

    def __init__(
        self,
        *,
        user_data_dir: Path,
        headless: bool = True,
        proxy: dict[str, str] | None = None,
    ) -> None:
        self._user_data_dir = user_data_dir
        self._headless = headless
        self._proxy = proxy
        self._browser = None
        self._playwright_manager = None

    @property
    def browser(self):
        return self._browser

    def launch(self):
        """Launch the browser instance."""
        from openclaw_algae.browser.common import require_sync_playwright

        sync_playwright = require_sync_playwright()
        self._playwright_manager = sync_playwright()
        playwright = self._playwright_manager.__enter__()

        launch_kwargs = {
            "headless": self._headless,
        }
        if self._proxy:
            launch_kwargs["proxy"] = self._proxy

        self._user_data_dir.mkdir(parents=True, exist_ok=True)
        self._browser = playwright.chromium.launch_persistent_context(
            str(self._user_data_dir),
            **launch_kwargs,
        )
        logger.info("pool: Browser launched at %s", self._user_data_dir)
        return self._browser

    def shutdown(self):
        """Gracefully shut down the browser."""
        if self._browser is not None:
            try:
                self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._playwright_manager is not None:
            try:
                self._playwright_manager.__exit__(None, None, None)
            except Exception:
                pass
            self._playwright_manager = None
        logger.info("pool: Browser shut down")

    @property
    def is_alive(self) -> bool:
        """Check if the browser is still running."""
        if self._browser is None:
            return False
        try:
            return not getattr(self._browser, "is_closed", True)
        except Exception:
            return False
