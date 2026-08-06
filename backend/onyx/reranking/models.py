from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from onyx.context.search.models import InferenceChunk
from onyx.reranking.constants import (
    MAX_RERANK_CANDIDATES,
    MAX_RERANK_DOCUMENT_BYTES,
    MAX_RERANK_DOCUMENT_TOKENS,
    MAX_RERANK_TOTAL_BYTES,
    MAX_RERANK_TOTAL_TOKENS,
)


class RerankOutcome(StrEnum):
    SUCCESS = "success"
    DISABLED = "disabled"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    PROVIDER_ERROR = "provider_error"
    INVALID_RESPONSE = "invalid_response"
    CIRCUIT_OPEN = "circuit_open"


class RerankResult(BaseModel):
    ordered_chunks: list[InferenceChunk]
    scores_by_chunk: dict[tuple[str, int], float]
    submitted_count: int
    result_count: int
    outcome: RerankOutcome
    fallback_used: bool


class RerankScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    relevance_score: float


class RerankPayloadLimits(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_candidates: int = Field(default=MAX_RERANK_CANDIDATES, gt=0)
    max_document_bytes: int = Field(default=MAX_RERANK_DOCUMENT_BYTES, gt=0)
    max_document_tokens: int = Field(default=MAX_RERANK_DOCUMENT_TOKENS, gt=0)
    max_total_bytes: int = Field(default=MAX_RERANK_TOTAL_BYTES, gt=0)
    max_total_tokens: int = Field(default=MAX_RERANK_TOTAL_TOKENS, gt=0)


class SerializedRerankCandidates(BaseModel):
    documents: list[str]
    submitted_chunks: list[InferenceChunk]
    unsent_chunks: list[InferenceChunk]
    utf8_bytes: int
    estimated_tokens: int


class RerankCircuitKey(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: str
    config_fingerprint: str


class RerankError(Exception):
    pass


class RerankTimeout(RerankError):
    pass


class RerankRateLimited(RerankError):
    def __init__(self, *, retry_after_seconds: float | None) -> None:
        super().__init__()
        self.retry_after_seconds = retry_after_seconds


class RerankProviderError(RerankError):
    def __init__(
        self,
        *,
        status_code: int | None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__()
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


class RerankPayloadTooLarge(RerankError):
    pass


class InvalidRerankResponse(RerankError):
    pass
