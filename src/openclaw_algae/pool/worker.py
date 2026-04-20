"""Browser worker: wraps a single browser tab for one provider type."""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from openclaw_algae.pool.types import WorkerIdentity, WorkerState

logger = logging.getLogger(__name__)


class BrowserWorker:
    """A single worker that hosts one provider type in a browser tab."""

    def __init__(
        self,
        identity: WorkerIdentity,
        *,
        supported_models: list[str],
        user_data_dir: Path,
    ) -> None:
        self._identity = identity
        self._supported_models = list(supported_models)
        self._user_data_dir = user_data_dir
        self._state = WorkerState.IDLE
        self._busy_count = 0
        self._busy_lock = threading.Lock()
        self._page = None
        self._browser_launcher = None  # Set by PoolManager during init

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def identity(self) -> WorkerIdentity:
        return self._identity

    @property
    def name(self) -> str:
        return self._identity.name

    @property
    def provider_type(self) -> str:
        return self._identity.provider_type

    @property
    def state(self) -> WorkerState:
        return self._state

    @property
    def busy_count(self) -> int:
        return self._busy_count

    # ── Model matching ──────────────────────────────────────────────────────

    def supports(self, model_id: str) -> bool:
        """Check if this worker supports the given model ID."""
        # Match exact model or provider/model prefix
        for supported in self._supported_models:
            if model_id == supported or model_id.endswith(f"/{supported}"):
                return True
        return False

    def get_models(self) -> list[str]:
        return list(self._supported_models)

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def set_browser_launcher(self, launcher) -> None:
        """Set the browser launcher (called by PoolManager)."""
        self._browser_launcher = launcher

    def set_page(self, page) -> None:
        """Set the page object (called by PoolManager)."""
        self._page = page

    @property
    def page(self):
        return self._page

    def acquire(self) -> None:
        """Mark this worker as busy."""
        with self._busy_lock:
            self._busy_count += 1
            self._state = WorkerState.BUSY

    def release(self) -> None:
        """Mark this worker as available."""
        with self._busy_lock:
            self._busy_count = max(0, self._busy_count - 1)
            if self._busy_count == 0 and self._state == WorkerState.BUSY:
                self._state = WorkerState.IDLE

    def mark_crashed(self) -> None:
        """Mark this worker as crashed."""
        self._state = WorkerState.CRASHED
        with self._busy_lock:
            self._busy_count = 0

    def shutdown(self) -> None:
        """Shut down this worker."""
        self._state = WorkerState.SHUTDOWN
        self._page = None

    def recreate_page(self) -> bool:
        """Recreate the page after a crash. Returns True on success."""
        if self._browser_launcher is None or not self._browser_launcher.is_alive:
            logger.warning("pool: Cannot recreate page — browser not alive for %s", self.name)
            return False

        try:
            browser = self._browser_launcher.browser
            self._page = browser.pages[0] if getattr(browser, "pages", []) else browser.new_page()
            self._state = WorkerState.IDLE
            logger.info("pool: Page recreated for %s", self.name)
            return True
        except Exception as exc:
            logger.error("pool: Failed to recreate page for %s: %s", self.name, exc)
            return False

    def __repr__(self) -> str:
        return f"BrowserWorker({self.name}, state={self._state.value}, busy={self._busy_count})"
