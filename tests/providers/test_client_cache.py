"""BoundedClientCache: LRU semantics + closer hook."""
from __future__ import annotations

from opentoken.providers._client_cache import BoundedClientCache


def test_cache_returns_stored_values():
    cache: BoundedClientCache[int] = BoundedClientCache()
    cache.set("a", 1)
    assert cache.get("a") == 1
    assert cache.get("missing") is None


def test_cache_evicts_lru_when_over_capacity():
    cache: BoundedClientCache[int] = BoundedClientCache(max_size=2)
    cache.set("a", 1)
    cache.set("b", 2)
    # "a" is now the LRU; touching it via get bumps it back to MRU.
    cache.get("a")
    cache.set("c", 3)  # Should evict "b", not "a".

    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3


def test_cache_invokes_closer_on_eviction():
    closed: list[int] = []

    cache: BoundedClientCache[int] = BoundedClientCache(max_size=1, closer=closed.append)
    cache.set("a", 1)
    cache.set("b", 2)  # Evicts "a".

    assert closed == [1]


def test_cache_clear_closes_all_entries():
    closed: list[int] = []

    cache: BoundedClientCache[int] = BoundedClientCache(closer=closed.append)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.clear()

    assert sorted(closed) == [1, 2]
    assert cache.get("a") is None
    assert cache.get("b") is None


def test_cache_get_or_create_only_calls_factory_once():
    calls: list[int] = []

    def factory():
        calls.append(1)
        return "x"

    cache: BoundedClientCache[str] = BoundedClientCache()
    assert cache.get_or_create("k", factory) == "x"
    assert cache.get_or_create("k", factory) == "x"
    assert len(calls) == 1
