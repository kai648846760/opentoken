from pydantic import BaseModel, Field


class ProviderCredentialRecord(BaseModel):
    provider: str
    kind: str
    cookie: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    user_agent: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    status: str
