"""Model filter: whitelist/blacklist for adapter models."""
from __future__ import annotations


class ModelFilter:
    """Filter models by whitelist or blacklist."""

    def __init__(self, mode: str = "blacklist", models: list[str] | None = None) -> None:
        self._mode = mode.lower()
        self._models = set(models or [])

    def check(self, model_id: str) -> bool:
        """Check if a model is allowed."""
        if self._mode == "whitelist":
            return model_id in self._models
        # blacklist mode (default)
        return model_id not in self._models

    def get_allowed_models(self, all_models: list[str]) -> list[str]:
        """Filter a list of model IDs to only allowed ones."""
        return [m for m in all_models if self.check(m)]

    @property
    def is_empty(self) -> bool:
        return not self._models

    @classmethod
    def from_config(cls, config: dict[str, object] | None) -> "ModelFilter":
        """Create a ModelFilter from a config dict."""
        if not config:
            return cls()
        mode = str(config.get("mode", "blacklist"))
        model_list = config.get("list", [])
        if isinstance(model_list, list):
            return cls(mode=mode, models=[str(m) for m in model_list])
        return cls()
