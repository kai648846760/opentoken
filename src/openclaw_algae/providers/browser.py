from __future__ import annotations

from collections.abc import Callable
import queue
import re
import threading
from typing import Protocol

from openclaw_algae.gateway.normalized import NormalizedChatRequest
from openclaw_algae.models.model_aliases import normalize_provider_model
from openclaw_algae.models.provider_credentials import ProviderCredentialRecord
from openclaw_algae.providers.base import ChatResponse, ProviderAdapter
from openclaw_algae.providers.prompts import build_role_prompt
from openclaw_algae.providers.web_tool_calling import (
    complete_web_tool_roundtrip,
    parse_web_tool_response,
    request_uses_web_tools,
)


class BrowserProviderClient(Protocol):
    def chat_completion(self, *, message: str, model: str) -> str: ...


class BrowserChatAdapter(ProviderAdapter):
    def __init__(
        self,
        *,
        provider_name: str,
        login_hint: str,
        client_factory: Callable[[ProviderCredentialRecord], BrowserProviderClient],
        fallback_to_non_stream_chat_on_stream_failure: bool = True,
    ) -> None:
        self._provider_name = provider_name
        self._login_hint = login_hint
        self._client_factory = client_factory
        self._fallback_to_non_stream_chat_on_stream_failure = (
            fallback_to_non_stream_chat_on_stream_failure
        )

    def chat(
        self,
        request: NormalizedChatRequest,
        credentials: ProviderCredentialRecord | None = None,
    ) -> ChatResponse:
        if credentials is None:
            raise RuntimeError(
                f"Missing {self._provider_name} credentials. Run `{self._login_hint}` first."
            )
        client = self._client_factory(credentials)
        model = normalize_provider_model(
            credentials.provider,
            request.model.rsplit("/", 1)[-1],
        )
        if request_uses_web_tools(request):
            tool_invoke = getattr(client, "tool_chat_completion", None)
            parsed_content, tool_calls, finish_reason = complete_web_tool_roundtrip(
                request,
                provider=self._provider_name,
                invoke=(
                    (
                        lambda message: _run_browser_completion(
                            provider_name=self._provider_name,
                            invoke=lambda: str(
                                tool_invoke(
                                    message=message,
                                    model=model,
                                )
                            ),
                        )
                    )
                    if callable(tool_invoke)
                    else lambda message: _run_browser_completion(
                        provider_name=self._provider_name,
                        invoke=lambda: client.chat_completion(
                            message=message,
                            model=model,
                        ),
                    )
                ),
            )
        else:
            content = _run_browser_completion(
                provider_name=self._provider_name,
                invoke=lambda: client.chat_completion(
                    message=build_role_prompt(request),
                    model=model,
                ),
            )
            parsed_content, tool_calls, finish_reason = parse_web_tool_response(
                content,
                available_tools=request.tools,
                tool_choice=request.tool_choice,
            )
        return ChatResponse(
            model=request.model,
            content=parsed_content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

    def stream_chat(
        self,
        request: NormalizedChatRequest,
        credentials: ProviderCredentialRecord | None = None,
    ):
        if credentials is None or request_uses_web_tools(request):
            return None

        client = self._client_factory(credentials)
        model = normalize_provider_model(
            credentials.provider,
            request.model.rsplit("/", 1)[-1],
        )
        stream_method = getattr(client, "stream_chat_completion", None)
        if not callable(stream_method):
            content = _run_browser_completion(
                provider_name=self._provider_name,
                invoke=lambda: client.chat_completion(
                    message=build_role_prompt(request),
                    model=model,
                ),
            )
            return _iter_text_chunks(content)
        prompt = build_role_prompt(request)

        def stream_with_fallback():
            emitted_any = False
            stream_error: Exception | None = None
            try:
                for piece in _run_browser_stream(
                    provider_name=self._provider_name,
                    invoke=lambda: stream_method(
                        message=prompt,
                        model=model,
                    ),
                ):
                    if piece:
                        emitted_any = True
                        yield piece
            except Exception as exc:
                emitted_any = False
                stream_error = exc
            if emitted_any:
                return
            if not self._fallback_to_non_stream_chat_on_stream_failure:
                if stream_error is not None:
                    raise stream_error
                raise RuntimeError(f"{self._provider_name} browser stream returned no text content.")
            content = _run_browser_completion(
                provider_name=self._provider_name,
                invoke=lambda: client.chat_completion(
                    message=prompt,
                    model=model,
                ),
            )
            yield from _iter_text_chunks(content)

        return stream_with_fallback()


# ── Persistent per-provider browser worker ────────────────────────────────────
#
# Playwright (sync API) uses greenlets to dispatch IO; each greenlet has a
# "parent" back-pointer that is bound to the OS thread that created it.
# Calling a playwright page method from a *different* thread causes greenlet to
# raise "Cannot switch to a different thread".
#
# The fix: one long-lived daemon thread per provider.  All browser work for
# that provider is serialised through a task queue and executed on that one
# thread, so the playwright context is always accessed from its owner thread.
#
# _run_browser_completion submits work to the appropriate worker and blocks
# the caller (FastAPI's anyio worker thread) until the result arrives.

class _BrowserWorkerThread(threading.Thread):
    """A single long-lived daemon thread that serialises all browser calls for
    one provider.  Tasks arrive via *task_queue* as (invoke, result_queue)
    pairs; the result (True, value) or (False, exception) is placed on
    *result_queue* when done.
    """

    def __init__(self, provider_name: str) -> None:
        super().__init__(
            name=f"openclaw-browser-worker-{provider_name.lower().replace(' ', '-')}",
            daemon=True,
        )
        self.task_queue: queue.Queue[
            tuple[Callable[[], str], queue.Queue[tuple[bool, object]]] | None
        ] = queue.Queue()

    def run(self) -> None:
        while True:
            item = self.task_queue.get()
            if item is None:
                # Poison pill – shutdown requested
                return
            invoke, result_queue = item
            try:
                result_queue.put((True, invoke()))
            except BaseException as exc:
                result_queue.put((False, exc))


_WORKER_THREADS: dict[str, _BrowserWorkerThread] = {}
_WORKER_THREADS_LOCK = threading.Lock()


def _get_or_create_worker(provider_name: str) -> _BrowserWorkerThread:
    """Return the persistent worker thread for *provider_name*, creating and
    starting it on the first call."""
    key = provider_name.lower()
    with _WORKER_THREADS_LOCK:
        worker = _WORKER_THREADS.get(key)
        if worker is None or not worker.is_alive():
            worker = _BrowserWorkerThread(provider_name)
            worker.start()
            _WORKER_THREADS[key] = worker
    return worker


def _run_browser_completion(
    *,
    provider_name: str,
    invoke: Callable[[], str],
    timeout_seconds: float = 300.0,
) -> str:
    """Submit *invoke* to the persistent browser worker for *provider_name*
    and block until a result is available (or *timeout_seconds* elapses)."""
    worker = _get_or_create_worker(provider_name)
    result_queue: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)
    worker.task_queue.put((invoke, result_queue))

    try:
        ok, payload = result_queue.get(timeout=timeout_seconds)
    except queue.Empty:
        raise RuntimeError(
            f"{provider_name} browser call timed out after {int(timeout_seconds)}s"
        )

    if ok:
        return str(payload)
    if isinstance(payload, RuntimeError):
        raise payload
    if isinstance(payload, BaseException):
        raise RuntimeError(str(payload)) from payload
    raise RuntimeError(f"{provider_name} browser call failed without an exception payload.")


def _run_browser_stream(
    *,
    provider_name: str,
    invoke: Callable[[], object],
    timeout_seconds: float = 300.0,
):
    stream_queue: queue.Queue[tuple[str, object | None]] = queue.Queue()
    control_queue: queue.Queue[str] = queue.Queue()
    cancel_event = threading.Event()

    def task() -> str:
        iterator = None
        terminal_sent = False
        try:
            iterator = invoke()
            if iterator is None:
                raise RuntimeError(f"{provider_name} browser stream is unavailable.")
            iterator = iter(iterator)
            while not cancel_event.is_set():
                command = control_queue.get()
                if command == "cancel":
                    break
                if command != "next":
                    continue
                while not cancel_event.is_set():
                    try:
                        piece = next(iterator)
                    except StopIteration:
                        stream_queue.put(("done", None))
                        terminal_sent = True
                        return ""
                    if piece:
                        stream_queue.put(("piece", str(piece)))
                        break
        except BaseException as exc:
            stream_queue.put(("error", exc))
            terminal_sent = True
        finally:
            close_stream = getattr(iterator, "close", None) if iterator is not None else None
            if callable(close_stream):
                try:
                    close_stream()
                except BaseException as exc:
                    if not terminal_sent:
                        stream_queue.put(("error", exc))
                        terminal_sent = True
            if not terminal_sent:
                stream_queue.put(("done", None))
        return ""

    threading.Thread(
        target=task,
        name=f"openclaw-browser-stream-{provider_name.lower().replace(' ', '-')}",
        daemon=True,
    ).start()

    def generator():
        control_queue.put("next")
        try:
            while True:
                try:
                    kind, payload = stream_queue.get(timeout=timeout_seconds)
                except queue.Empty as exc:
                    raise RuntimeError(
                        f"{provider_name} browser stream timed out after {int(timeout_seconds)}s"
                    ) from exc
                if kind == "piece":
                    yield str(payload or "")
                    control_queue.put("next")
                    continue
                if kind == "error":
                    if isinstance(payload, RuntimeError):
                        raise payload
                    if isinstance(payload, BaseException):
                        raise RuntimeError(str(payload)) from payload
                    raise RuntimeError(f"{provider_name} browser stream failed.")
                return
        finally:
            cancel_event.set()
            control_queue.put("cancel")

    return generator()
def _iter_text_chunks(content: str, *, max_chunk_len: int = 16):
    text = content or ""
    if not text:
        return
    parts = re.findall(r"\S+\s*", text)
    if len(parts) <= 1:
        for i in range(0, len(text), max_chunk_len):
            yield text[i : i + max_chunk_len]
        return
    current = ""
    for part in parts:
        if current and len(current) + len(part) > max_chunk_len:
            yield current
            current = part
        else:
            current += part
    if current:
        yield current
