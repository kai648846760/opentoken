"""Tests for load balancing strategies."""
from openclaw_algae.pool.strategy import (
    LeastBusyStrategy,
    RandomStrategy,
    RoundRobinStrategy,
    Worker,
    create_strategy,
)


class _FakeWorker(Worker):
    def __init__(self, name: str, busy: int = 0) -> None:
        self._name = name
        self._busy = busy

    @property
    def busy_count(self) -> int:
        return self._busy

    @property
    def name(self) -> str:
        return self._name


def test_least_busy_sorts_by_load() -> None:
    strategy = LeastBusyStrategy()
    workers = [
        _FakeWorker("c", 5),
        _FakeWorker("a", 1),
        _FakeWorker("b", 3),
    ]
    result = strategy.sort(workers)
    assert [w.name for w in result] == ["a", "b", "c"]


def test_least_busy_empty_list() -> None:
    strategy = LeastBusyStrategy()
    assert strategy.sort([]) == []


def test_round_robin_first_selection() -> None:
    strategy = RoundRobinStrategy()
    workers = [_FakeWorker("a"), _FakeWorker("b"), _FakeWorker("c")]
    result = strategy.sort(workers)
    assert [w.name for w in result] == ["a", "b", "c"]


def test_round_robin_rotates() -> None:
    strategy = RoundRobinStrategy()
    workers = [_FakeWorker("a"), _FakeWorker("b"), _FakeWorker("c")]
    first = strategy.sort(workers)
    second = strategy.sort(workers)
    third = strategy.sort(workers)
    assert [w.name for w in first] == ["a", "b", "c"]
    assert [w.name for w in second] == ["b", "c", "a"]
    assert [w.name for w in third] == ["c", "a", "b"]


def test_random_shuffles() -> None:
    strategy = RandomStrategy()
    workers = [_FakeWorker(str(i)) for i in range(10)]
    result = strategy.sort(workers)
    assert len(result) == 10
    assert {w.name for w in result} == {str(i) for i in range(10)}


def test_create_strategy_valid_names() -> None:
    assert isinstance(create_strategy("least_busy"), LeastBusyStrategy)
    assert isinstance(create_strategy("round_robin"), RoundRobinStrategy)
    assert isinstance(create_strategy("random"), RandomStrategy)


def test_create_strategy_invalid_name() -> None:
    import pytest
    with pytest.raises(ValueError, match="Unknown strategy"):
        create_strategy("nonexistent")
