from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from shared_configs.enums import EmbeddingProvider


class RetryDisposition(StrEnum):
    RETRYABLE = "RETRYABLE"
    TERMINAL = "TERMINAL"


class RetryReason(StrEnum):
    TIMEOUT = "TIMEOUT"
    NETWORK = "NETWORK"
    HTTP_STATUS = "HTTP_STATUS"
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


class RegulatoryIndexingConfigSnapshot(BaseModel):
    """Immutable, JSON-safe configuration captured for one indexing job."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    search_settings_id: int = Field(gt=0)
    embedding_provider: EmbeddingProvider
    embedding_model_name: str = Field(min_length=1)
    model_dimension: int = Field(gt=0)
    reduced_dimension: int | None = Field(default=None, gt=0)
    effective_dimension: int = Field(gt=0)
    index_name: str = Field(min_length=1)
    vertex: VertexBatchConfig
    prompt_version: str = Field(min_length=1)
    prompt_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    max_attempts: int = Field(default=5, gt=0)
    retry_base_seconds: float = Field(default=15, gt=0, allow_inf_nan=False)
    retry_max_seconds: float = Field(default=900, gt=0, allow_inf_nan=False)
    poll_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)
    lease_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    embedding_request_size: int = Field(default=64, gt=0)

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
        return self
