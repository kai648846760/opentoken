"""Bounded LRU client cache shared by provider adapters.

Provider adapters keep a per-credential httpx-backed client around to avoid
re-establishing the upstream session on every chat completion. The previous
implementation used an unbounded dict, which leaked memory + file descriptors
when callers rotated through many credentials. This module provides a small
LRU container with explicit cleanup hooks.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Generic, TypeVar


T = TypeVar("T")

DEFAULT_MAX_CLIENTS = 64


class BoundedClientCache(Generic[T]):
    """Thread-safe LRU cache for provider clients.

    When the cache exceeds `max_size`, the least-recently-used entry is evicted
    and, if a `closer` is configured, the evicted client gets a best-effort
    `close()` so we don't leak underlying httpx connection pools.
    """

    def __init__(
        self,
        *,
        max_size: int = DEFAULT_MAX_CLIENTS,
        closer: Callable[[T], None] | None = None,
    ) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self._max_size = max_size
        self._closer = closer
        self._lock = threading.Lock()
        self._items: OrderedDict[str, T] = OrderedDict()

    def get(self, key: str) -> T | None:
        with self._lock:
            value = self._items.get(key)
            if value is not None:
                self._items.move_to_end(key)
            return value

    def set(self, key: str, value: T) -> None:
        evicted: T | None = None
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                self._items[key] = value
            else:
                self._items[key] = value
                if len(self._items) > self._max_size:
                    _, evicted = self._items.popitem(last=False)
        if evicted is not None and self._closer is not None:
            try:
                self._closer(evicted)
            except Exception:
                pass

    def get_or_create(self, key: str, factory: Callable[[], T]) -> T:
        existing = self.get(key)
        if existing is not None:
            return existing
        created = factory()
        self.set(key, created)
        return created

    def clear(self) -> None:
        with self._lock:
            items = list(self._items.values())
            self._items.clear()
        if self._closer is None:
            return
        for item in items:
            try:
                self._closer(item)
            except Exception:
                pass

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
