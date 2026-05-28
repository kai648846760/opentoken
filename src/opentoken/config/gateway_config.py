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
    # Field name `list` previously shadowed the builtin during forward-ref
    # evaluation on Python 3.14, breaking pydantic model rebuilds. Renamed
    # to `entries` with `list` kept as an alias for YAML backwards-compat.
    entries: list[str] = Field(default_factory=list, alias="list")

    model_config = {"populate_by_name": True}


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
            Path.home() / ".opentoken" / "gateway.yaml",
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

    # 整个 load 过程任何异常都不能传到 create_app —— 否则 yaml 语法错或字段验证
    # 失败会让 uvicorn worker 起不来,整个服务不可用。失败就记日志 + 当作没配置,
    # 让 opentoken 用默认行为。
    import logging
    logger = logging.getLogger(__name__)
    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("gateway_config_unreadable path=%s error=%s", config_path, exc)
        return None
    try:
        data = yaml.safe_load(raw)
    except Exception as exc:
        logger.warning("gateway_config_yaml_parse_failed path=%s error=%s", config_path, exc)
        return None
    if not isinstance(data, dict):
        return None
    try:
        return GatewayConfig(**data)
    except Exception as exc:
        logger.warning("gateway_config_validation_failed path=%s error=%s", config_path, exc)
        return None


def create_default_gateway_config() -> str:
    """Generate a default gateway config YAML string."""
    try:
        import yaml
    except ImportError:
        return "# Install PyYAML: uv add pyyaml\n"

    config = GatewayConfig()
    return yaml.dump(config.model_dump(), default_flow_style=False, sort_keys=False)
