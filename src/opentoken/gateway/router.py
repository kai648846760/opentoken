"""Pool-aware router that integrates browser pool and failover."""
from __future__ import annotations

import logging
from pathlib import Path
from collections.abc import Iterator

from opentoken.config.paths import resolve_providers_dir
from opentoken.failover.errors import NonRetryableError, RetryableError, normalize_error
from opentoken.failover.executor import FailoverExecutor
from opentoken.gateway.normalized import NormalizedChatRequest
from opentoken.pool.manager import PoolManager
from opentoken.pool.worker import BrowserWorker
from opentoken.providers.base import ChatResponse, ProviderAdapter, ProviderRateLimitError
from opentoken.providers.browser import BrowserChatAdapter
from opentoken.providers.camoufox_clients import CamoufoxProviderClient
from opentoken.providers.chatgpt import ChatGPTWebAdapter
from opentoken.providers.claude import ClaudeWebAdapter
from opentoken.providers.deepseek import DeepSeekWebAdapter
from opentoken.providers.doubao import DoubaoWebAdapter
from opentoken.providers.gemini import GeminiWebAdapter
from opentoken.providers.glm import GLMIntlWebAdapter, GLMWebAdapter
from opentoken.providers.grok import GrokWebAdapter
from opentoken.providers.kimi import KimiWebAdapter
from opentoken.providers.manus import ManusApiAdapter
from opentoken.providers.mimo import MimoWebAdapter
from opentoken.providers.nim import NimChatAdapter
from opentoken.providers.qwen import QwenCnWebAdapter, QwenWebAdapter
from opentoken.failover.model_chain import chain_from_credentials, run_with_chain, stream_with_chain
from opentoken.models.openai_compat import resolve_requested_model
from opentoken.providers.registry import supported_provider_keys
from opentoken.storage.provider_store import load_provider_credentials

logger = logging.getLogger(__name__)

# Provider types that use direct HTTP API calls (bypass browser pool)
_HTTP_PROVIDERS = frozenset({
    "deepseek", "claude", "kimi", "manus", "mimo", "nim",
})

# Providers that support cross-model fallback when rate-limited. NIM is the
# initial member because its catalog has many equivalent-tier free models.
_CHAINABLE_PROVIDERS = frozenset({"nim"})

# Provider types that use the browser pool
_BROWSER_PROVIDERS = frozenset({
    "doubao", "qwen-intl", "qwen-cn", "chatgpt", "gemini", "grok", "glm-cn", "glm-intl",
})


class PoolAwareRouter:
    """Router that integrates browser pool for failover and load balancing.

    HTTP providers are called directly. Browser providers go through the pool
    with failover across multiple workers.
    """

    def __init__(
        self,
        *,
        pool_manager: PoolManager | None = None,
        adapters: dict[str, ProviderAdapter] | None = None,
        max_failover_retries: int = 2,
        providers_dir: Path | None = None,
    ) -> None:
        self._pool_manager = pool_manager
        self._http_adapters = adapters or {}
        self._adapters = self._http_adapters  # Backward compatibility alias
        self._max_failover_retries = max_failover_retries
        self._providers_dir = providers_dir or resolve_providers_dir()
        self._setup_default_adapters()

    def _setup_default_adapters(self) -> None:
        """Set up default HTTP adapters if not provided."""
        defaults = {
            "deepseek": DeepSeekWebAdapter(),
            "claude": ClaudeWebAdapter(),
            "kimi": KimiWebAdapter(),
            "manus": ManusApiAdapter(),
            "mimo": MimoWebAdapter(),
            "nim": NimChatAdapter(),
            "doubao": BrowserChatAdapter(
                provider_name="Doubao",
                login_hint="opentoken login doubao",
                client_factory=lambda credentials: CamoufoxProviderClient("doubao", credentials),
                fallback_to_non_stream_chat_on_stream_failure=False,
            ),
            "qwen-intl": QwenWebAdapter(
                base_url="https://chat.qwen.ai",
                stream_client_factory=lambda credentials: CamoufoxProviderClient("qwen-intl", credentials),
            ),
            "qwen-cn": QwenCnWebAdapter(),
            "chatgpt": ChatGPTWebAdapter(),
            "gemini": GeminiWebAdapter(),
            "grok": GrokWebAdapter(),
            "glm-cn": BrowserChatAdapter(
                provider_name="GLM China",
                login_hint="opentoken login glm-cn",
                client_factory=lambda credentials: CamoufoxProviderClient("glm-cn", credentials),
            ),
            "glm-intl": BrowserChatAdapter(
                provider_name="GLM International",
                login_hint="opentoken login glm-intl",
                client_factory=lambda credentials: CamoufoxProviderClient("glm-intl", credentials),
            ),
        }
        for key, adapter in defaults.items():
            if key not in self._http_adapters:
                self._http_adapters[key] = adapter

    def chat(self, request: NormalizedChatRequest) -> ChatResponse:
        """Route a chat request to the appropriate provider.

        For HTTP providers: call directly.
        For browser providers: use pool with failover.
        """
        resolved_model = resolve_requested_model(request.model, providers_dir=self._providers_dir)
        if resolved_model is None:
            raise RuntimeError(f"Unsupported model: {request.model}")

        provider_name = resolved_model.provider
        if provider_name not in supported_provider_keys():
            raise RuntimeError(f"Unsupported model: {request.model}")

        credentials = load_provider_credentials(self._providers_dir, provider_name)
        if credentials is None:
            raise RuntimeError(
                f"Missing {provider_name} credentials. Run `opentoken login {provider_name}` first."
            )

        if provider_name in _HTTP_PROVIDERS:
            return self._call_http_provider(provider_name, request, credentials)
        elif provider_name in _BROWSER_PROVIDERS:
            return self._call_browser_provider(
                provider_name,
                request,
                credentials,
                model_lookup_id=resolved_model.canonical_model,
            )
        else:
            raise RuntimeError(f"No route configured for provider: {provider_name}")

    def stream_chat(self, request: NormalizedChatRequest) -> Iterator[str] | None:
        resolved_model = resolve_requested_model(request.model, providers_dir=self._providers_dir)
        if resolved_model is None:
            raise RuntimeError(f"Unsupported model: {request.model}")
        provider_name = resolved_model.provider
        if provider_name not in supported_provider_keys():
            raise RuntimeError(f"Unsupported model: {request.model}")

        credentials = load_provider_credentials(self._providers_dir, provider_name)
        if credentials is None:
            raise RuntimeError(
                f"Missing {provider_name} credentials. Run `opentoken login {provider_name}` first."
            )

        adapter = self._http_adapters.get(provider_name)
        if adapter is None:
            return None
        stream_method = getattr(adapter, "stream_chat", None)
        if not callable(stream_method):
            return None
        if provider_name in _CHAINABLE_PROVIDERS:
            chain = chain_from_credentials(credentials if hasattr(credentials, "metadata") else None)
            return stream_with_chain(
                request,
                chain,
                lambda req: stream_method(req, credentials),
            )
        return stream_method(request, credentials)

    def _call_http_provider(
        self,
        provider_name: str,
        request: NormalizedChatRequest,
        credentials: object,
        *,
        model_lookup_id: str | None = None,
    ) -> ChatResponse:
        """Call an HTTP provider directly."""
        adapter = self._http_adapters.get(provider_name)
        if adapter is None:
            raise RuntimeError(f"No adapter registered for {provider_name}")
        if provider_name in _CHAINABLE_PROVIDERS:
            chain = chain_from_credentials(credentials if hasattr(credentials, "metadata") else None)
            return run_with_chain(
                request,
                chain,
                lambda req: adapter.chat(req, credentials),
            )
        return adapter.chat(request, credentials)

    def _call_browser_provider(
        self,
        provider_name: str,
        request: NormalizedChatRequest,
        credentials: object,
        *,
        model_lookup_id: str | None = None,
    ) -> ChatResponse:
        """Call a browser provider through the pool with failover."""
        if self._pool_manager is None or not self._pool_manager.initialized:
            # Fallback: use direct HTTP adapter if pool not available
            adapter = self._http_adapters.get(provider_name)
            if adapter is not None:
                return adapter.chat(request, credentials)
            raise RuntimeError(f"Browser pool not initialized for {provider_name}")

        model_id = model_lookup_id or request.model
        candidates = self._pool_manager.get_candidates(model_id)
        if not candidates:
            # Fallback to direct HTTP adapter
            adapter = self._http_adapters.get(provider_name)
            if adapter is not None:
                return adapter.chat(request, credentials)
            raise RuntimeError(f"No browser workers available for {provider_name}")

        executor = FailoverExecutor(
            max_retries=self._max_failover_retries,
            on_retry=lambda w, e: logger.warning("failover: Retrying on worker %s: %s", w.name, e),
        )

        def work_fn(worker: BrowserWorker) -> ChatResponse:
            worker.acquire()
            try:
                adapter = self._http_adapters.get(provider_name)
                if adapter is None:
                    raise RuntimeError(f"No adapter for {provider_name}")
                return adapter.chat(request, credentials)
            finally:
                worker.release()

        try:
            return executor.execute(candidates, work_fn)
        except (RetryableError, NonRetryableError) as exc:
            raise RuntimeError(f"All browser workers failed for {provider_name}: {exc}") from exc


# Backward compatibility alias
ProviderRouter = PoolAwareRouter

_DEFAULT_ROUTER: PoolAwareRouter | None = None


def get_default_router() -> PoolAwareRouter:
    """Get a singleton default pool-aware router.

    Reusing the same router instance across requests is important so provider
    adapters can keep long-lived client/session state instead of recreating a
    fresh session on every API call.
    """
    global _DEFAULT_ROUTER
    if _DEFAULT_ROUTER is None:
        _DEFAULT_ROUTER = PoolAwareRouter()
    return _DEFAULT_ROUTER
