"""Token usage estimation for OpenAI-compatible responses.

opentoken is a gateway over many providers, most of which never report
token counts. Returning {"prompt_tokens": 0, ...} confuses clients that
use usage to manage context windows or estimate cost. We approximate
with a deterministic char-based heuristic — accurate to within ~20% on
English/Chinese mixed text, which is good enough for downstream pacing.

Estimate: 1 token ≈ 4 characters for ASCII text, 1 token ≈ 1.5 characters
for CJK. We blend by counting non-ASCII separately, which is cheap.
"""
from __future__ import annotations


# Server-side identifier returned in OpenAI responses so cache-aware clients can
# distinguish backend versions. We don't currently rotate this; bump on breaking
# changes to backend protocol handling so client-side caches reset.
SYSTEM_FINGERPRINT = "fp_opentoken_v1"


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    ascii_count = 0
    other_count = 0
    for char in text:
        if ord(char) < 128:
            ascii_count += 1
        else:
            other_count += 1
    return max(1, round(ascii_count / 4 + other_count / 1.5))


def estimate_prompt_tokens(messages: list[dict[str, object]] | None) -> int:
    if not messages:
        return 0
    total = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
            continue
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        total += estimate_tokens(text)
    return total
