"""PoolManager: manages browser instances and their workers."""
from __future__ import annotations

import logging
from pathlib import Path

from openclaw_algae.pool.browser import BrowserLauncher
from openclaw_algae.pool.strategy import LoadBalancer, create_strategy
from openclaw_algae.pool.types import (
    BrowserConfig,
    InstanceConfig,
    WorkerConfig,
    WorkerIdentity,
    WorkerState,
)
from openclaw_algae.pool.worker import BrowserWorker

logger = logging.getLogger(__name__)


class PoolManager:
    """Manages browser instances and their workers.

    Workers sharing the same userDataDir share a single browser process.
    """

    def __init__(
        self,
        *,
        strategy: str = "least_busy",
        state_dir: Path | None = None,
        browser_config: BrowserConfig | None = None,
    ) -> None:
        self._strategy_name = strategy
        self._state_dir = state_dir or Path.home() / ".openclaw-algae"
        self._browser_config = browser_config or BrowserConfig()
        self._instances: dict[str, _BrowserInstance] = {}
        self._workers: dict[str, BrowserWorker] = {}
        self._strategy: LoadBalancer = create_strategy(strategy)
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self, instance_configs: list[InstanceConfig]) -> None:
        """Initialize all browser instances and workers."""
        if self._initialized:
            return

        for instance_config in instance_configs:
            self._initialize_instance(instance_config)

        if not self._workers:
            raise RuntimeError("No workers initialized. Check instance configurations.")

        self._initialized = True
        logger.info(
            "pool: Pool initialized with %d workers across %d browser instances",
            len(self._workers),
            len(self._instances),
        )

    def _initialize_instance(self, config: InstanceConfig) -> None:
        """Initialize a single browser instance with its workers."""
        user_data_dir = self._resolve_user_data_dir(config)
        user_data_dir.mkdir(parents=True, exist_ok=True)

        # Launch browser
        launcher = BrowserLauncher(
            user_data_dir=user_data_dir,
            headless=self._browser_config.headless,
            proxy=self._resolve_proxy(config),
        )
        try:
            browser = launcher.launch()
        except Exception as exc:
            logger.error("pool: Failed to launch browser for %s: %s", config.name, exc)
            return

        instance = _BrowserInstance(config=config, launcher=launcher, browser=browser)
        self._instances[config.name] = instance

        # Create workers for this instance
        for worker_config in config.workers:
            worker = BrowserWorker(
                identity=WorkerIdentity(
                    name=worker_config.name,
                    provider_type=worker_config.worker_type,
                    instance_name=config.name,
                ),
                supported_models=_get_models_for_provider(worker_config.worker_type),
                user_data_dir=user_data_dir,
            )
            worker.set_browser_launcher(launcher)

            # Create a page for this worker
            try:
                page = browser.pages[0] if getattr(browser, "pages", []) else browser.new_page()
                worker.set_page(page)
            except Exception as exc:
                logger.error("pool: Failed to create page for %s: %s", worker_config.name, exc)
                continue

            self._workers[worker_config.name] = worker
            logger.info(
                "pool: Worker %s (%s) ready on instance %s",
                worker_config.name,
                worker_config.worker_type,
                config.name,
            )

    def _resolve_user_data_dir(self, config: InstanceConfig) -> Path:
        """Resolve the user data directory for a browser instance."""
        base = self._state_dir / "browser"
        if config.user_data_mark:
            return base / f"camoufoxUserData_{config.user_data_mark}"
        return base / "camoufoxUserData"

    def _resolve_proxy(self, config: InstanceConfig) -> dict[str, str] | None:
        """Resolve proxy config for a browser instance."""
        proxy = config.proxy or self._browser_config.proxy
        if proxy and proxy.enable:
            proxy_url = f"{proxy.proxy_type}://"
            if proxy.user and proxy.passwd:
                proxy_url += f"{proxy.user}:{proxy.passwd}@"
            proxy_url += f"{proxy.host}:{proxy.port}"
            return {"server": proxy_url}
        return None

    def get_candidates(self, model_id: str) -> list[BrowserWorker]:
        """Get all workers that support the given model, sorted by strategy."""
        candidates = [
            w for w in self._workers.values()
            if w.state not in (WorkerState.SHUTDOWN, WorkerState.CRASHED)
            and w.supports(model_id)
        ]
        if not candidates:
            return []
        return self._strategy.sort(candidates)

    def get_worker(self, name: str) -> BrowserWorker | None:
        """Get a worker by name."""
        return self._workers.get(name)

    def get_status(self) -> dict[str, object]:
        """Get pool status for health/monitoring."""
        workers_status = []
        for name, worker in self._workers.items():
            workers_status.append({
                "name": name,
                "provider": worker.provider_type,
                "state": worker.state.value,
                "busy": worker.busy_count,
            })

        return {
            "initialized": self._initialized,
            "total_workers": len(self._workers),
            "total_instances": len(self._instances),
            "strategy": self._strategy_name,
            "workers": workers_status,
        }

    def get_models(self) -> list[str]:
        """Get all models supported by the pool."""
        models = set()
        for worker in self._workers.values():
            models.update(worker.get_models())
        return sorted(models)

    def shutdown(self) -> None:
        """Shut down all browser instances and workers."""
        for worker in self._workers.values():
            worker.shutdown()
        for instance in self._instances.values():
            try:
                instance.launcher.shutdown()
            except Exception as exc:
                logger.error("pool: Error shutting down browser %s: %s", instance.config.name, exc)
        self._workers.clear()
        self._instances.clear()
        self._initialized = False
        logger.info("pool: Pool shut down")


class _BrowserInstance:
    """Internal: a single browser process hosting multiple workers."""

    def __init__(self, config: InstanceConfig, launcher: BrowserLauncher, browser) -> None:
        self.config = config
        self.launcher = launcher
        self.browser = browser


# ── Model registry for worker creation ──────────────────────────────────────

_PROVIDER_MODELS: dict[str, list[str]] = {
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "claude": ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-6"],
    "qwen-intl": ["qwen3.5-plus", "qwen-max"],
    "qwen-cn": ["Qwen3.5-Plus", "Qwen3.5-Turbo"],
    "kimi": ["moonshot-v1-32k"],
    "doubao": ["doubao-seed-2.0", "doubao-pro", "doubao-lite"],
    "chatgpt": ["gpt-4"],
    "gemini": ["gemini-pro", "gemini-ultra"],
    "grok": ["grok-2"],
    "glm-cn": ["glm-4-plus", "glm-4", "glm-4-think", "glm-4-zero"],
    "glm-intl": ["glm-4-plus", "glm-4-think"],
    "mimo": ["mimo-v2-pro", "xiaomimo-chat"],
    "manus": ["manus-1.6"],
}


def _get_models_for_provider(provider_type: str) -> list[str]:
    """Get the model IDs supported by a provider type."""
    models = _PROVIDER_MODELS.get(provider_type, [])
    if models:
        return models
    # Fallback: try provider_type as-is (e.g., "qwen")
    for key, value in _PROVIDER_MODELS.items():
        if provider_type in key or key in provider_type:
            return value
    return []
