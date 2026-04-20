from collections.abc import Iterator
from dataclasses import dataclass, field

from openclaw_algae.gateway.normalized import NormalizedChatRequest
from openclaw_algae.models.provider_credentials import ProviderCredentialRecord


@dataclass(frozen=True)
class ChatResponse:
    model: str
    content: str | None = ""
    tool_calls: list[dict[str, object]] = field(default_factory=list)
    finish_reason: str = "stop"


class ProviderRateLimitError(RuntimeError):
    pass


class ProviderAdapter:
    def chat(
        self,
        request: NormalizedChatRequest,
        credentials: ProviderCredentialRecord | None = None,
    ) -> ChatResponse:
        return ChatResponse(model=request.model, content="stub response")

    def stream_chat(
        self,
        request: NormalizedChatRequest,
        credentials: ProviderCredentialRecord | None = None,
    ) -> Iterator[str] | None:
        return None
