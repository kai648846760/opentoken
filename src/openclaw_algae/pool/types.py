"""Type definitions for the browser instance pool."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class WorkerState(str, Enum):
    """Lifecycle state of a browser worker."""

    IDLE = "idle"
    BUSY = "busy"
    CRASHED = "crashed"
    SHUTDOWN = "shutdown"
    INITIALIZING = "initializing"


@dataclass(frozen=True)
class WorkerIdentity:
    """Unique identity of a browser worker."""

    name: str
    provider_type: str
    instance_name: str  # Parent browser instance name


@dataclass
class SelectionResult:
    """Result of a worker selection operation."""

    worker: object | None = None
    reason: str = ""


@dataclass(frozen=True)
class ProxyConfig:
    """Proxy configuration for a browser instance."""

    enable: bool = False
    proxy_type: str = "http"  # http or socks5
    host: str = "127.0.0.1"
    port: int = 7890
    user: str | None = None
    passwd: str | None = None


@dataclass(frozen=True)
class BrowserConfig:
    """Browser launch configuration."""

    headless: bool = True
    humanize_cursor: bool = True
    fission: bool = True
    proxy: ProxyConfig = field(default_factory=ProxyConfig)


@dataclass(frozen=True)
class WorkerConfig:
    """Configuration for a single worker in an instance."""

    name: str
    worker_type: str  # Provider type (e.g., "doubao", "qwen-intl")


@dataclass(frozen=True)
class InstanceConfig:
    """Configuration for a browser instance (one browser process)."""

    name: str
    workers: list[WorkerConfig]
    user_data_mark: str | None = None  # Optional suffix for data directory
    proxy: ProxyConfig | None = None  # Instance-level proxy override
