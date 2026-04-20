import copy
import json
from datetime import UTC, datetime
from pathlib import Path

from openclaw_algae.models.catalog import default_catalog


def build_algae_provider_patch(base_url: str, api_key: str) -> dict[str, object]:
    return {
        'models': {
            'providers': {
                'algae': {
                    'baseUrl': base_url,
                    'apiKey': api_key,
                    'api': 'openai-completions',
                    'models': [entry.to_openclaw_dict() for entry in default_catalog()],
                }
            }
        }
    }


def apply_algae_provider_patch(config_path: Path, patch: dict[str, object]) -> Path | None:
    existing: dict[str, object] = {}
    backup_path: Path | None = None

    if config_path.exists():
        raw = config_path.read_text(encoding='utf-8')
        existing = json.loads(raw)
        timestamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
        backup_path = config_path.with_name(f'{config_path.name}.{timestamp}.bak')
        backup_path.write_text(raw, encoding='utf-8')

    merged = _merge_provider_patch(existing, patch)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(merged, indent=2), encoding='utf-8')
    return backup_path


def _merge_provider_patch(existing: dict[str, object], patch: dict[str, object]) -> dict[str, object]:
    merged = copy.deepcopy(existing)
    models = merged.setdefault('models', {})
    if not isinstance(models, dict):
        models = {}
        merged['models'] = models

    providers = models.setdefault('providers', {})
    if not isinstance(providers, dict):
        providers = {}
        models['providers'] = providers

    patch_models = patch.get('models', {})
    if not isinstance(patch_models, dict):
        return merged

    patch_providers = patch_models.get('providers', {})
    if not isinstance(patch_providers, dict):
        return merged

    for provider_name, provider_config in patch_providers.items():
        providers[provider_name] = copy.deepcopy(provider_config)

    return merged
