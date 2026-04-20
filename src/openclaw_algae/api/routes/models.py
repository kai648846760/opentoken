from fastapi import APIRouter

from openclaw_algae.gateway.model_registry import get_default_model_registry
from openclaw_algae.models.openai_compat import build_openai_model_objects

router = APIRouter()


@router.get('/v1/models')
def list_models() -> dict[str, object]:
    registry = get_default_model_registry()
    entries = registry.list_models()
    return {
        'object': 'list',
        'data': build_openai_model_objects(entries),
    }
