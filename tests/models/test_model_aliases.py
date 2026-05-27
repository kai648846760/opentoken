"""Model alias normalization is case-insensitive.

Clients send model ids in arbitrary casing. The qwen-cn map already carried
both lowercase and capitalized keys by hand — proving the intent was case-
insensitive everywhere — but qwen-intl's lowercase-only keys silently passed
mixed-case ids straight through to the upstream API (which then 400s on the
non-canonical form).
"""
from __future__ import annotations

from opentoken.models.model_aliases import normalize_provider_model


def test_exact_match_normalizes() -> None:
    assert normalize_provider_model("qwen-intl", "qwen-3.5-turbo") == "qwen3.5-flash"
    assert normalize_provider_model("qwen-cn", "Qwen3.5-Turbo") == "Qwen3.5-Flash"


def test_uppercase_alias_resolves_via_lowered_index() -> None:
    """The previously-uncovered case: a capitalized id against qwen-intl."""
    assert normalize_provider_model("qwen-intl", "Qwen-3.5-Turbo") == "qwen3.5-flash"
    assert normalize_provider_model("qwen-intl", "QWEN-MAX") == "qwen-max-latest"


def test_unknown_alias_passes_through() -> None:
    assert normalize_provider_model("qwen-intl", "some-model-we-dont-alias") == "some-model-we-dont-alias"


def test_unknown_provider_passes_through() -> None:
    assert normalize_provider_model("unknown-provider", "any-model") == "any-model"


def test_mixed_case_qwen_cn_resolves_either_way() -> None:
    """qwen-cn explicitly carried both cases by hand; both still resolve."""
    assert normalize_provider_model("qwen-cn", "qwen3.5-turbo") == "Qwen3.5-Flash"
    assert normalize_provider_model("qwen-cn", "qwen3.5-Turbo") == "Qwen3.5-Flash"
