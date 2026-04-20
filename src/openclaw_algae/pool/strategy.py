"""Load balancing strategies for worker selection."""
from __future__ import annotations

import random
from abc import ABC, abstractmethod


class Worker(ABC):
    """Abstract worker that can be selected for work."""

    @property
    @abstractmethod
    def busy_count(self) -> int:
        """Number of concurrent tasks on this worker."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Worker name for logging."""


class LoadBalancer(ABC):
    """Abstract load balancer that sorts candidates."""

    @abstractmethod
    def sort(self, candidates: list[Worker]) -> list[Worker]:
        """Return candidates sorted by preference (best first)."""


class LeastBusyStrategy(LoadBalancer):
    """Select the worker with the fewest active tasks."""

    def sort(self, candidates: list[Worker]) -> list[Worker]:
        return sorted(candidates, key=lambda w: w.busy_count)


class RoundRobinStrategy(LoadBalancer):
    """Round-robin across candidates."""

    def __init__(self) -> None:
        self._index = 0

    def sort(self, candidates: list[Worker]) -> list[Worker]:
        idx = self._index % max(len(candidates), 1)
        self._index += 1
        return list(candidates[idx:]) + list(candidates[:idx])


class RandomStrategy(LoadBalancer):
    """Random selection across candidates."""

    def sort(self, candidates: list[Worker]) -> list[Worker]:
        result = list(candidates)
        random.shuffle(result)
        return result


def create_strategy(name: str) -> LoadBalancer:
    """Create a load balancer by name."""
    strategies = {
        "least_busy": LeastBusyStrategy,
        "round_robin": RoundRobinStrategy,
        "random": RandomStrategy,
    }
    cls = strategies.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies.keys())}")
    return cls()
