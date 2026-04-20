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

    def to_openclaw_dict(self) -> dict[str, str]:
        return {
            'id': self.id.removeprefix('algae/'),
            'name': self.name,
        }


_MODEL_DEFINITIONS: tuple[tuple[str, str, str], ...] = (
    ('deepseek', 'deepseek-chat', 'DeepSeek Chat'),
    ('deepseek', 'deepseek-reasoner', 'DeepSeek Reasoner'),
    ('qwen-intl', 'qwen3.6-plus', 'Qwen 3.6 Plus'),
    ('qwen-intl', 'qwen3.5-plus', 'Qwen 3.5 Plus'),
    ('qwen-intl', 'qwen3.5-flash', 'Qwen 3.5 Flash'),
    ('qwen-intl', 'qwen3.5-omni-plus', 'Qwen 3.5 Omni Plus'),
    ('qwen-intl', 'qwen-max-latest', 'Qwen 2.5 Max'),
    ('qwen-cn', 'Qwen3.5-千问', 'Qwen 3.5 千问'),
    ('qwen-cn', 'Qwen3.5-Flash', 'Qwen 3.5 Flash'),
    ('qwen-cn', 'Qwen3-Max', 'Qwen 3 Max'),
    ('qwen-cn', 'Qwen3-Max-Thinking', 'Qwen 3 Max Thinking'),
    ('qwen-cn', 'Qwen3-Coder', 'Qwen 3 Coder'),
    ('kimi', 'moonshot-v1-8k', 'Kimi 8K'),
    ('kimi', 'moonshot-v1-32k', 'Kimi 32K'),
    ('kimi', 'moonshot-v1-128k', 'Kimi 128K'),
    ('claude', 'claude-sonnet-4-6', 'Claude Sonnet 4.6'),
    ('claude', 'claude-opus-4-6', 'Claude Opus 4.6'),
    ('claude', 'claude-haiku-4-6', 'Claude Haiku 4.6'),
    ('doubao', 'doubao-seed-2.0', 'Doubao Seed 2.0'),
    ('doubao', 'doubao-pro', 'Doubao Pro'),
    ('chatgpt', 'gpt-4', 'GPT-4'),
    ('chatgpt', 'gpt-4-turbo', 'GPT-4 Turbo'),
    ('gemini', 'gemini-pro', 'Gemini Pro'),
    ('gemini', 'gemini-ultra', 'Gemini Ultra'),
    ('grok', 'grok-1', 'Grok 1'),
    ('grok', 'grok-2', 'Grok 2'),
    ('glm-cn', 'glm-4-plus', 'GLM-4 Plus'),
    ('glm-cn', 'glm-4-think', 'GLM-4 Think'),
    ('glm-intl', 'glm-4-plus', 'GLM-4 Plus International'),
    ('glm-intl', 'glm-4-think', 'GLM-4 Think International'),
    ('mimo', 'mimo-2.0', 'MiMo 2.0'),
    ('mimo', 'mimo-2.5-pro', 'MiMo 2.5 Pro'),
    ('manus', 'manus-1.6', 'Manus 1.6'),
    ('manus', 'manus-1.6-lite', 'Manus 1.6 Lite'),
)


def default_catalog() -> list[ModelCatalogEntry]:
    return [
        ModelCatalogEntry(
            id=f'algae/{provider}/{model}',
            provider='openclaw-algae',
            name=name,
        )
        for provider, model, name in _MODEL_DEFINITIONS
    ]
