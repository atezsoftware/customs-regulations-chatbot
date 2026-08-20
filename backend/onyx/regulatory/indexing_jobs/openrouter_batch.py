from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from hashlib import sha256
from typing import cast
from uuid import UUID

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from onyx.regulatory.indexing_jobs.models import (
    IndexingGatewayConnectionError,
    IndexingGatewayHTTPError,
    IndexingGatewayIndeterminateSubmissionError,
    IndexingGatewayTimeoutError,
    OpenRouterBatchConfig,
)
from onyx.tracing.flows import LLMFlow
from onyx.tracing.llm_utils import traced_llm_call


class OpenRouterBatchContractError(ValueError):
    """Secret-safe violation of the OpenRouter Batch input/output contract."""


class OpenRouterBatchJobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class OpenRouterEmbeddingBatchRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    custom_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    inputs: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_inputs(self) -> "OpenRouterEmbeddingBatchRequest":
        if any(not value.strip() for value in self.inputs):
            raise ValueError("embedding inputs must not be blank")
        return self


class OpenRouterEmbeddingBatchResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    custom_id: str
    vectors: list[list[float]] | None = None
    error_code: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> "OpenRouterEmbeddingBatchResult":
        if (self.vectors is None) == (self.error_code is None):
            raise ValueError("batch result must contain exactly one outcome")
        return self


class OpenRouterBatchState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    remote_batch_id: str = Field(min_length=1)
    status: OpenRouterBatchJobStatus
    results: list[dict[str, object]] | None = None


def openrouter_batch_submission_key(
    requests: Sequence[OpenRouterEmbeddingBatchRequest],
    *,
    tenant_id: str,
    job_id: UUID,
    submission_attempt: int,
) -> str:
    if not tenant_id.strip() or not requests or submission_attempt < 1:
        raise OpenRouterBatchContractError(
            "OpenRouter submission identity is incomplete"
        )
    custom_ids = [request.custom_id for request in requests]
    if len(set(custom_ids)) != len(custom_ids):
        raise OpenRouterBatchContractError(
            "OpenRouter submission identity contains duplicate custom_ids"
        )
    identity = json.dumps(
        {
            "tenant_id": tenant_id,
            "job_id": str(job_id),
            "submission_attempt": submission_attempt,
            "custom_ids": custom_ids,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"regulatory-embedding-{sha256(identity.encode()).hexdigest()}"


_STATUS_MAP = {
    "validating": OpenRouterBatchJobStatus.PENDING,
    "in_progress": OpenRouterBatchJobStatus.RUNNING,
    "finalizing": OpenRouterBatchJobStatus.RUNNING,
    "completed": OpenRouterBatchJobStatus.SUCCEEDED,
    "failed": OpenRouterBatchJobStatus.FAILED,
    "cancelling": OpenRouterBatchJobStatus.CANCELLING,
    "cancelled": OpenRouterBatchJobStatus.CANCELLED,
    "expired": OpenRouterBatchJobStatus.EXPIRED,
}


def _batch_object(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise OpenRouterBatchContractError("OpenRouter Batch response is malformed")
    typed_payload = cast(dict[str, object], payload)
    nested = typed_payload.get("data")
    if isinstance(nested, dict):
        return cast(dict[str, object], nested)
    return typed_payload


def _batch_state(payload: object) -> OpenRouterBatchState:
    batch = _batch_object(payload)
    remote_id = batch.get("id")
    raw_status = batch.get("status")
    if not isinstance(remote_id, str) or not remote_id.strip():
        raise OpenRouterBatchContractError("OpenRouter Batch response has no id")
    if not isinstance(raw_status, str) or raw_status not in _STATUS_MAP:
        raise OpenRouterBatchContractError(
            "OpenRouter Batch response has unknown status"
        )
    raw_results = batch.get("results")
    if raw_results is not None and not isinstance(raw_results, list):
        raise OpenRouterBatchContractError("OpenRouter Batch results are malformed")
    results: list[dict[str, object]] | None = None
    if isinstance(raw_results, list):
        if any(not isinstance(result, dict) for result in raw_results):
            raise OpenRouterBatchContractError("OpenRouter Batch results are malformed")
        results = [cast(dict[str, object], result) for result in raw_results]
    return OpenRouterBatchState(
        remote_batch_id=remote_id,
        status=_STATUS_MAP[raw_status],
        results=results,
    )


def parse_openrouter_embedding_results(
    raw_results: Sequence[Mapping[str, object]],
    *,
    expected_custom_ids: set[str],
    expected_model: str,
    expected_dimension: int,
) -> dict[str, OpenRouterEmbeddingBatchResult]:
    """Validate inline results and map them by custom_id, independent of order."""

    parsed: dict[str, OpenRouterEmbeddingBatchResult] = {}
    for raw_result in raw_results:
        custom_id = raw_result.get("custom_id")
        if not isinstance(custom_id, str) or custom_id not in expected_custom_ids:
            raise OpenRouterBatchContractError(
                "OpenRouter Batch returned an unexpected custom_id"
            )
        if custom_id in parsed:
            raise OpenRouterBatchContractError(
                "OpenRouter Batch returned a duplicate custom_id"
            )
        response = raw_result.get("response")
        if not isinstance(response, dict):
            raise OpenRouterBatchContractError(
                "OpenRouter Batch result has no response"
            )
        typed_response = cast(dict[str, object], response)
        status_code = typed_response.get("status_code")
        if not isinstance(status_code, int):
            raise OpenRouterBatchContractError(
                "OpenRouter Batch result has no HTTP status"
            )
        if status_code < 200 or status_code >= 300:
            parsed[custom_id] = OpenRouterEmbeddingBatchResult(
                custom_id=custom_id,
                error_code=f"http_{status_code}",
            )
            continue
        body = typed_response.get("body")
        if not isinstance(body, dict):
            raise OpenRouterBatchContractError("embedding result body is malformed")
        typed_body = cast(dict[str, object], body)
        if typed_body.get("model") != expected_model:
            raise OpenRouterBatchContractError("embedding result model is mismatched")
        data = typed_body.get("data")
        if not isinstance(data, list) or not data:
            raise OpenRouterBatchContractError("embedding result data is empty")
        indexed_vectors: dict[int, list[float]] = {}
        for entry in data:
            if not isinstance(entry, dict):
                raise OpenRouterBatchContractError(
                    "embedding result entry is malformed"
                )
            typed_entry = cast(dict[str, object], entry)
            index = typed_entry.get("index")
            vector = typed_entry.get("embedding")
            if not isinstance(index, int) or index < 0 or index in indexed_vectors:
                raise OpenRouterBatchContractError("embedding result index is invalid")
            if (
                not isinstance(vector, list)
                or len(vector) != expected_dimension
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in vector
                )
            ):
                raise OpenRouterBatchContractError("embedding vector is invalid")
            indexed_vectors[index] = [
                float(cast(int | float, value)) for value in vector
            ]
        if set(indexed_vectors) != set(range(len(indexed_vectors))):
            raise OpenRouterBatchContractError(
                "embedding result indices are incomplete"
            )
        parsed[custom_id] = OpenRouterEmbeddingBatchResult(
            custom_id=custom_id,
            vectors=[indexed_vectors[index] for index in range(len(indexed_vectors))],
        )
    return parsed


def openrouter_embedding_payload(
    requests: Sequence[OpenRouterEmbeddingBatchRequest],
    *,
    config: OpenRouterBatchConfig,
    submission_key: str,
) -> dict[str, object]:
    return {
        "endpoint": "/v1/embeddings",
        "model": config.model_name,
        "requests": [
            {
                "custom_id": request.custom_id,
                "body": {
                    "input": request.inputs,
                    "dimensions": config.effective_dimension,
                },
            }
            for request in requests
        ],
        "metadata": {"submission_key": submission_key},
    }


def openrouter_embedding_payload_size(
    requests: Sequence[OpenRouterEmbeddingBatchRequest],
    *,
    config: OpenRouterBatchConfig,
    submission_key: str,
) -> int:
    payload = openrouter_embedding_payload(
        requests, config=config, submission_key=submission_key
    )
    return len(httpx.Request("POST", config.api_url, json=payload).content)


class HttpxOpenRouterBatchGateway:
    def __init__(
        self,
        *,
        config: OpenRouterBatchConfig,
        api_key_provider: Callable[[], str],
        client: httpx.Client | None = None,
    ) -> None:
        self._config = config
        self._api_key_provider = api_key_provider
        self._client = client or httpx.Client(timeout=60.0)

    def _headers(self) -> dict[str, str]:
        api_key = self._api_key_provider().strip()
        if not api_key:
            raise OpenRouterBatchContractError("OpenRouter credential is unavailable")
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, object] | None = None,
        indeterminate_key: str | None = None,
    ) -> object:
        try:
            response = self._client.request(
                method,
                url,
                headers=self._headers(),
                json=json_body,
            )
        except (httpx.ReadTimeout, httpx.WriteTimeout):
            if indeterminate_key is not None:
                raise IndexingGatewayIndeterminateSubmissionError(
                    indeterminate_key
                ) from None
            raise IndexingGatewayTimeoutError() from None
        except httpx.TimeoutException:
            raise IndexingGatewayTimeoutError() from None
        except httpx.RequestError:
            raise IndexingGatewayConnectionError() from None
        if response.status_code < 200 or response.status_code >= 300:
            raise IndexingGatewayHTTPError(response.status_code)
        try:
            return response.json()
        except ValueError:
            raise OpenRouterBatchContractError(
                "OpenRouter Batch response is not JSON"
            ) from None

    def submit(
        self,
        requests: Sequence[OpenRouterEmbeddingBatchRequest],
        *,
        submission_key: str,
    ) -> OpenRouterBatchState:
        if not requests:
            raise OpenRouterBatchContractError("OpenRouter Batch requires requests")
        custom_ids = [request.custom_id for request in requests]
        if len(set(custom_ids)) != len(custom_ids):
            raise OpenRouterBatchContractError(
                "OpenRouter Batch custom_ids are duplicate"
            )
        input_count = sum(len(request.inputs) for request in requests)
        if len(requests) > self._config.max_requests:
            raise OpenRouterBatchContractError(
                "OpenRouter Batch request limit exceeded"
            )
        if input_count > self._config.max_inputs:
            raise OpenRouterBatchContractError("OpenRouter Batch input limit exceeded")
        payload = openrouter_embedding_payload(
            requests, config=self._config, submission_key=submission_key
        )
        if (
            openrouter_embedding_payload_size(
                requests,
                config=self._config,
                submission_key=submission_key,
            )
            > self._config.max_bytes
        ):
            raise OpenRouterBatchContractError("OpenRouter Batch byte limit exceeded")
        with traced_llm_call(
            flow=LLMFlow.REGULATORY_EMBEDDING_BATCH,
            model=self._config.model_name,
            provider="openrouter",
            extra_config={
                "request_count": str(len(requests)),
                "input_count": str(input_count),
            },
        ):
            response = self._request(
                "POST",
                self._config.api_url,
                json_body=payload,
                indeterminate_key=submission_key,
            )
        return _batch_state(response)

    def get(self, remote_batch_id: str) -> OpenRouterBatchState:
        if re.fullmatch(r"[A-Za-z0-9_.:-]+", remote_batch_id) is None:
            raise OpenRouterBatchContractError("OpenRouter Batch id is invalid")
        response = self._request(
            "GET", f"{self._config.api_url.rstrip('/')}/{remote_batch_id}"
        )
        return _batch_state(response)

    def reconcile_submission(self, submission_key: str) -> OpenRouterBatchState | None:
        response = self._request("GET", self._config.api_url)
        if not isinstance(response, dict):
            raise OpenRouterBatchContractError("OpenRouter Batch list is malformed")
        raw_batches = cast(dict[str, object], response).get("data")
        if not isinstance(raw_batches, list):
            raise OpenRouterBatchContractError("OpenRouter Batch list is malformed")
        matches: list[dict[str, object]] = []
        for raw_batch in raw_batches:
            if not isinstance(raw_batch, dict):
                raise OpenRouterBatchContractError("OpenRouter Batch list is malformed")
            batch = cast(dict[str, object], raw_batch)
            metadata = batch.get("metadata")
            if (
                isinstance(metadata, dict)
                and cast(dict[str, object], metadata).get("submission_key")
                == submission_key
            ):
                matches.append(batch)
        if not matches:
            return None
        if len(matches) != 1:
            raise OpenRouterBatchContractError(
                "OpenRouter submission identity matched multiple batches"
            )
        return _batch_state(matches[0])

    def cancel(self, remote_batch_id: str) -> None:
        if re.fullmatch(r"[A-Za-z0-9_.:-]+", remote_batch_id) is None:
            raise OpenRouterBatchContractError("OpenRouter Batch id is invalid")
        self._request(
            "POST",
            f"{self._config.api_url.rstrip('/')}/{remote_batch_id}/cancel",
        )
