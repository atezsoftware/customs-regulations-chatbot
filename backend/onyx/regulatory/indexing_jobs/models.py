from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from shared_configs.enums import EmbeddingProvider


class IndexingGatewayError(RuntimeError):
    """SDK-neutral error raised by regulatory indexing gateway boundaries."""


class IndexingGatewayTimeoutError(IndexingGatewayError):
    def __init__(self) -> None:
        super().__init__("Regulatory indexing gateway request timed out")


class IndexingGatewayConnectionError(IndexingGatewayError):
    def __init__(self) -> None:
        super().__init__("Regulatory indexing gateway connection failed")


class IndexingGatewayHTTPError(IndexingGatewayError):
    status_code: int

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(
            f"Regulatory indexing gateway returned HTTP status {status_code}"
        )


class IndexingGatewayIndeterminateSubmissionError(IndexingGatewayError):
    submission_key: str

    def __init__(self, submission_key: str) -> None:
        self.submission_key = submission_key
        super().__init__(
            "Regulatory indexing submission outcome is indeterminate for "
            f"{submission_key}"
        )


class IndexingPublicationIndeterminateError(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "Regulatory indexing publication visibility requires reconciliation"
        )


class RetryDisposition(StrEnum):
    RETRYABLE = "RETRYABLE"
    TERMINAL = "TERMINAL"


class RetryReason(StrEnum):
    TIMEOUT = "TIMEOUT"
    NETWORK = "NETWORK"
    HTTP_STATUS = "HTTP_STATUS"
    PUBLICATION_INDETERMINATE = "PUBLICATION_INDETERMINATE"
    UNKNOWN = "UNKNOWN"


class RetryDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    disposition: RetryDisposition
    reason: RetryReason
    status_code: int | None = None

    @property
    def retryable(self) -> bool:
        return self.disposition is RetryDisposition.RETRYABLE

    @property
    def error_code(self) -> str:
        if self.status_code is not None:
            return f"http_{self.status_code}"
        return self.reason.value.lower()


class VertexAuthenticationMode(StrEnum):
    SERVICE_ACCOUNT_JSON = "service_account_json"
    WORKLOAD_IDENTITY = "workload_identity"


class RegulatoryInputHashVersion(StrEnum):
    """Stable algorithms used to identify one immutable loader result."""

    LEGACY_V1 = "legacy-v1"
    CANONICAL_V2 = "canonical-v2"
    CHUNK_ROWS_V3 = "chunk-rows-v3"
    LEGACY_OR_CANONICAL = "legacy-or-canonical"


class VertexBatchConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    model_configuration_id: int = Field(gt=0)
    model_name: str = Field(min_length=1)
    project: str = Field(min_length=1)
    location: str = Field(min_length=1)
    authentication_mode: VertexAuthenticationMode
    gcs_uri: str = Field(pattern=r"^gs://[^/\s]+(?:/[^\s]*)?$")


class OpenRouterBatchConfig(BaseModel):
    """Non-secret provider contract captured with a durable embedding job."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    api_url: str = Field(pattern=r"^https?://[^\s]+/api/beta/batches/?$")
    model_name: str = Field(min_length=1)
    effective_dimension: int = Field(gt=0)
    request_input_size: int = Field(default=64, gt=0)
    max_requests: int = Field(default=1_000, gt=0)
    # OpenRouter's documented provider ceiling is 50,000 total inputs. Keep a
    # strict margin so config drift cannot submit at the hard boundary.
    max_inputs: int = Field(default=45_000, gt=0, lt=50_000)
    max_bytes: int = Field(default=32 * 1024 * 1024, gt=0)
    completion_horizon_seconds: int = Field(default=86_400, gt=0)


class RegulatoryIndexingConfigSnapshot(BaseModel):
    """Immutable, JSON-safe configuration captured for one indexing job."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    input_content_hash: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    input_hash_version: RegulatoryInputHashVersion
    chunk_generation_hash: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    search_settings_id: int = Field(gt=0)
    embedding_provider: EmbeddingProvider
    embedding_model_name: str = Field(min_length=1)
    model_dimension: int = Field(gt=0)
    reduced_dimension: int | None = Field(default=None, gt=0)
    effective_dimension: int = Field(gt=0)
    index_name: str = Field(min_length=1)
    vertex: VertexBatchConfig
    openrouter_batch: OpenRouterBatchConfig | None = None
    prompt_version: str = Field(min_length=1)
    prompt_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    max_attempts: int = Field(default=5, gt=0)
    retry_base_seconds: float = Field(default=15, gt=0, allow_inf_nan=False)
    retry_max_seconds: float = Field(default=900, gt=0, allow_inf_nan=False)
    poll_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)
    lease_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    embedding_request_size: int = Field(default=64, gt=0)
    context_request_size: int = Field(default=64, gt=0)
    context_jsonl_max_bytes: int = Field(default=8 * 1024 * 1024, gt=0)
    submission_reconcile_seconds: int = Field(default=300, gt=0)

    @model_validator(mode="after")
    def validate_derived_values(self) -> "RegulatoryIndexingConfigSnapshot":
        expected_dimension = self.reduced_dimension or self.model_dimension
        if self.effective_dimension != expected_dimension:
            raise ValueError(
                "effective_dimension must equal reduced_dimension or model_dimension"
            )
        if self.retry_max_seconds < self.retry_base_seconds:
            raise ValueError(
                "retry_max_seconds must be greater than or equal to retry_base_seconds"
            )
        if self.openrouter_batch is not None:
            if self.openrouter_batch.model_name != self.embedding_model_name:
                raise ValueError("OpenRouter Batch model must match embedding model")
            if self.openrouter_batch.effective_dimension != self.effective_dimension:
                raise ValueError(
                    "OpenRouter Batch dimension must match effective dimension"
                )
            if self.openrouter_batch.request_input_size != self.embedding_request_size:
                raise ValueError(
                    "OpenRouter Batch request size must match embedding request size"
                )
        return self
