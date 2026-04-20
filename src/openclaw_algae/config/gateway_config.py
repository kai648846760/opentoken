"""YAML gateway configuration loader."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ProxyConfig(BaseModel):
    enable: bool = False
    proxy_type: str = "http"
    host: str = "127.0.0.1"
    port: int = 7890
    user: str | None = None
    passwd: str | None = None


class BrowserConfig(BaseModel):
    headless: bool = True
    path: str | None = None
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)


class WorkerConfig(BaseModel):
    name: str
    worker_type: str


class InstanceConfig(BaseModel):
    name: str
    user_data_mark: str | None = None
    proxy: ProxyConfig | None = None
    workers: list[WorkerConfig] = Field(default_factory=list)


class FailoverConfig(BaseModel):
    enabled: bool = True
    max_retries: int = 2


class PoolConfig(BaseModel):
    strategy: str = "least_busy"
    failover: FailoverConfig = Field(default_factory=FailoverConfig)
    wait_timeout: int = 120000


class ModelFilterConfig(BaseModel):
    mode: str = "blacklist"  # whitelist or blacklist
    list: list[str] = Field(default_factory=list)


class AdapterConfig(BaseModel):
    model_filter: ModelFilterConfig | None = None


class GatewayConfig(BaseModel):
    server: dict[str, Any] = Field(default_factory=lambda: {"host": "127.0.0.1", "port": 32117})
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    pool: PoolConfig = Field(default_factory=PoolConfig)
    instances: list[InstanceConfig] = Field(default_factory=list)
    adapter: dict[str, AdapterConfig] = Field(default_factory=dict)


def load_gateway_config(config_path: Path | None = None) -> GatewayConfig | None:
    """Load gateway configuration from YAML file.

    Args:
        config_path: Path to config file. If None, searches default locations.

    Returns:
        GatewayConfig or None if no config file found.
    """
    if config_path is None:
        # Search default locations
        candidates = [
            Path("gateway.yaml"),
            Path("gateway.yml"),
            Path("config/gateway.yaml"),
            Path.home() / ".openclaw-algae" / "gateway.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = candidate
                break

    if config_path is None or not config_path.exists():
        return None

    try:
        import yaml
    except ImportError:
        # PyYAML not installed — create default config
        return GatewayConfig()

    raw = config_path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        return None

    return GatewayConfig(**data)


def create_default_gateway_config() -> str:
    """Generate a default gateway config YAML string."""
    try:
        import yaml
    except ImportError:
        return "# Install PyYAML: uv add pyyaml\n"

    config = GatewayConfig()
    return yaml.dump(config.model_dump(), default_flow_style=False, sort_keys=False)
