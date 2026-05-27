from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCatalogEntry:
    id: str
    provider: str
    name: str

    def to_openai_dict(self) -> dict[str, str]:
        return {
            'id': self.id,
            'object': 'model',
            'owned_by': self.provider,
        }

    def to_opentoken_dict(self) -> dict[str, str]:
        return {
            'id': self.id.removeprefix('algae/'),
            'name': self.name,
        }


# The catalog is intentionally empty. opentoken used to ship a long list of
# baked-in model ids per provider, which (a) went stale fast as providers
# rotated their lineup and (b) advertised models the user wasn't actually
# logged in to.
#
# `/v1/models` is now backed entirely by `opentoken.models.discovery`, which
# queries each logged-in provider's live model registry on demand. Providers
# without credentials simply don't contribute models. The fallback for
# providers whose discovery momentarily fails comes from the per-provider
# disk cache (see `model-catalog-cache.json` in the state directory), not
# from this module.


def default_catalog() -> list[ModelCatalogEntry]:
    return []
