from typing import Literal

from pydantic import BaseModel


class RerankerConfigUpdate(BaseModel):
    enabled: bool
    provider_type: Literal["openrouter"]
    model_id: str
    api_key: str | None = None
    test_attestation: str | None = None


class RerankerConfigView(BaseModel):
    enabled: bool
    provider_type: Literal["openrouter"] | None
    model_id: str | None
    api_key_configured: bool
    masked_api_key: str | None


class RerankerTestRequest(BaseModel):
    provider_type: Literal["openrouter"]
    model_id: str | None = None
    api_key: str | None = None


class RerankerTestResponse(BaseModel):
    success: bool
    test_attestation: str


class OpenRouterModelsRequest(BaseModel):
    api_key: str | None = None


class OpenRouterModelView(BaseModel):
    id: str
    name: str


class OpenRouterModelsResponse(BaseModel):
    models: list[OpenRouterModelView]
