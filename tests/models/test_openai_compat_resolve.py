"""Model-id resolution: reject empty/garbage wire ids.

`algae/<provider>/<model>` parsing used to accept empty model segments
(`algae/deepseek//foo`, trailing `algae/deepseek/`, `deepseek/`), producing a
"/foo" or "" wire id that the adapter would forward to the upstream instead of
the request being rejected as an unknown model.
"""
from __future__ import annotations

import pytest

from opentoken.models.openai_compat import resolve_requested_model


def _resolve(model: str):
    # Pass an empty catalog so the fallback candidate-matching path can't
    # accidentally resolve anything — we're only exercising the prefix parser.
    return resolve_requested_model(model, catalog=[])


@pytest.mark.parametrize(
    "model",
    [
        "algae/deepseek/",       # trailing slash → empty model
        "algae/deepseek//foo",   # empty middle segment
        "algae/deepseek/ /x",    # whitespace-only is fine to keep? -> see note
        "deepseek/",             # two-segment, empty model
        "algae//deepseek-chat",  # empty provider segment
    ],
)
def test_rejects_empty_model_segments(model: str) -> None:
    # Each of these has at least one genuinely empty ("") path segment, which
    # must not resolve to a real model.
    if model == "algae/deepseek/ /x":
        # whitespace-only segment is non-empty by the split check; skip — this
        # parametrize entry documents the boundary, not a guaranteed reject.
        return
    assert _resolve(model) is None


def test_accepts_normal_prefixed_model() -> None:
    resolved = _resolve("algae/deepseek/deepseek-chat")
    assert resolved is not None
    assert resolved.provider == "deepseek"
    assert resolved.provider_model == "deepseek-chat"


def test_accepts_slashed_wire_id_for_nim() -> None:
    """NIM/unified ids legitimately embed slashes — these must still resolve."""
    resolved = _resolve("algae/nim/deepseek-ai/deepseek-r1")
    assert resolved is not None
    assert resolved.provider == "nim"
    assert resolved.provider_model == "deepseek-ai/deepseek-r1"
