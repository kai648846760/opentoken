from openclaw_algae.models.catalog import ModelCatalogEntry
from openclaw_algae.models.discovery import load_model_catalog


class ModelRegistry:
    def __init__(self, entries: list[ModelCatalogEntry]) -> None:
        self._entries = entries

    def list_models(self) -> list[ModelCatalogEntry]:
        return list(self._entries)


def get_default_model_registry() -> ModelRegistry:
    return ModelRegistry(load_model_catalog())
