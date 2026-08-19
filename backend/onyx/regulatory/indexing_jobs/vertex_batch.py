from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Collection, Iterator, Sequence
from contextlib import contextmanager
from enum import StrEnum
from hashlib import sha256
from typing import TYPE_CHECKING, NoReturn, Protocol, cast

import google.auth
import httpx
import requests
from google.api_core import exceptions as google_exceptions
from google.auth import exceptions as google_auth_exceptions
from google.auth.credentials import Credentials
from google.cloud import storage
from google.oauth2 import service_account
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from onyx.regulatory.indexing_jobs.models import (
    IndexingGatewayConnectionError,
    IndexingGatewayError,
    IndexingGatewayHTTPError,
    IndexingGatewayIndeterminateSubmissionError,
    IndexingGatewayTimeoutError,
    VertexAuthenticationMode,
    VertexBatchConfig,
)
from onyx.tracing.flows import LLMFlow
from onyx.tracing.llm_utils import traced_llm_call

if TYPE_CHECKING:
    from google import genai
    from google.genai import types as genai_types

_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_SAFETY_FINISH_REASONS = frozenset(
    {
        "SAFETY",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "RECITATION",
        "IMAGE_SAFETY",
        "MODEL_ARMOR",
    }
)


class VertexBatchContractError(ValueError):
    """A secret-safe violation of the Vertex input or output contract."""


class VertexBatchSubmissionConflictError(VertexBatchContractError):
    submission_key: str
    match_count: int

    def __init__(self, submission_key: str, match_count: int) -> None:
        self.submission_key = submission_key
        self.match_count = match_count
        super().__init__(
            f"Vertex submission {submission_key} matched {match_count} remote jobs"
        )


class VertexBatchResultError(StrEnum):
    EMPTY = "empty_response"
    SAFETY = "safety_blocked"
    REMOTE_ERROR = "remote_error"
    MALFORMED = "malformed_response"


class VertexBatchJobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"


def _request_contents(prompt: str) -> dict[str, object]:
    return {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}


def _request_payload(prompt: str) -> dict[str, object]:
    return {
        **_request_contents(prompt),
        "generationConfig": {"temperature": 0, "maxOutputTokens": 256},
    }


def _canonical_request_hash(prompt: str) -> str:
    canonical_contents = json.dumps(
        _request_contents(prompt),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(canonical_contents.encode()).hexdigest()


class VertexBatchRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt: str = Field(min_length=1)

    @computed_field
    @property
    def request_hash(self) -> str:
        return _canonical_request_hash(self.prompt)


class VertexBatchResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request_hash: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    context: str | None = None
    error: VertexBatchResultError | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> "VertexBatchResult":
        if (self.context is None) == (self.error is None):
            raise ValueError("a result must contain exactly one outcome")
        return self


class VertexBatchState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    remote_job_name: str = Field(min_length=1)
    status: VertexBatchJobStatus
    input_uri: str | None = None
    output_uri: str | None = None
    error_code: int | None = None


class VertexBatchGateway(Protocol):
    def submit(self, requests: Sequence[VertexBatchRequest]) -> VertexBatchState: ...

    def get(self, remote_job_name: str) -> VertexBatchState: ...

    def reconcile_submission(self, submission_key: str) -> VertexBatchState | None: ...

    def read_results(self, output_uri: str) -> str: ...

    def cancel(self, remote_job_name: str) -> None: ...

    def cleanup(self, prefix: str) -> None: ...


def build_vertex_jsonl(requests: Sequence[VertexBatchRequest]) -> str:
    if not requests:
        raise VertexBatchContractError("Vertex batch requires at least one request")
    request_hashes = [request.request_hash for request in requests]
    if len(set(request_hashes)) != len(request_hashes):
        raise VertexBatchContractError("Vertex batch contains a duplicate request hash")
    lines = [
        json.dumps(
            {"request": _request_payload(request.prompt)},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for request in requests
    ]
    return "\n".join(lines) + "\n"


def _output_request_hash(value: object) -> str:
    if not isinstance(value, dict):
        raise VertexBatchContractError("Vertex output has no correlatable request")
    typed_value = cast(dict[str, object], value)
    contents = typed_value.get("contents")
    if not isinstance(contents, list) or len(contents) != 1:
        raise VertexBatchContractError("Vertex output has no correlatable request")
    content = contents[0]
    if not isinstance(content, dict):
        raise VertexBatchContractError("Vertex output has no correlatable request")
    typed_content = cast(dict[str, object], content)
    if typed_content.get("role") != "user":
        raise VertexBatchContractError("Vertex output has no correlatable request")
    parts = typed_content.get("parts")
    if not isinstance(parts, list) or len(parts) != 1:
        raise VertexBatchContractError("Vertex output has no correlatable request")
    part = parts[0]
    if not isinstance(part, dict):
        raise VertexBatchContractError("Vertex output has no correlatable request")
    typed_part = cast(dict[str, object], part)
    if not isinstance(typed_part.get("text"), str):
        raise VertexBatchContractError("Vertex output has no correlatable request")
    return _canonical_request_hash(cast(str, typed_part["text"]))


def _failure(request_hash: str, error: VertexBatchResultError) -> VertexBatchResult:
    return VertexBatchResult(request_hash=request_hash, error=error)


def _parse_output_result(
    value: dict[str, object], request_hash: str
) -> VertexBatchResult:
    status = value.get("status", "")
    if not isinstance(status, str):
        return _failure(request_hash, VertexBatchResultError.MALFORMED)
    if status:
        return _failure(request_hash, VertexBatchResultError.REMOTE_ERROR)

    response = value.get("response")
    if not isinstance(response, dict):
        return _failure(request_hash, VertexBatchResultError.MALFORMED)
    typed_response = cast(dict[str, object], response)
    prompt_feedback = typed_response.get("promptFeedback")
    if isinstance(prompt_feedback, dict):
        block_reason = cast(dict[str, object], prompt_feedback).get("blockReason")
        if isinstance(block_reason, str) and block_reason not in {
            "",
            "BLOCK_REASON_UNSPECIFIED",
        }:
            return _failure(request_hash, VertexBatchResultError.SAFETY)

    candidates = typed_response.get("candidates")
    if not isinstance(candidates, list):
        return _failure(request_hash, VertexBatchResultError.MALFORMED)
    if not candidates:
        return _failure(request_hash, VertexBatchResultError.EMPTY)
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        return _failure(request_hash, VertexBatchResultError.MALFORMED)
    typed_candidate = cast(dict[str, object], candidate)
    finish_reason = typed_candidate.get("finishReason")
    if isinstance(finish_reason, str) and finish_reason in _SAFETY_FINISH_REASONS:
        return _failure(request_hash, VertexBatchResultError.SAFETY)
    content = typed_candidate.get("content")
    if not isinstance(content, dict):
        return _failure(request_hash, VertexBatchResultError.MALFORMED)
    parts = cast(dict[str, object], content).get("parts")
    if not isinstance(parts, list) or not parts:
        return _failure(request_hash, VertexBatchResultError.MALFORMED)
    text_parts: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            return _failure(request_hash, VertexBatchResultError.MALFORMED)
        typed_part = cast(dict[str, object], part)
        if not isinstance(typed_part.get("text"), str):
            return _failure(request_hash, VertexBatchResultError.MALFORMED)
        text_parts.append(cast(str, typed_part["text"]))
    context = "".join(text_parts).strip()
    if not context:
        return _failure(request_hash, VertexBatchResultError.EMPTY)
    return VertexBatchResult(request_hash=request_hash, context=context)


def parse_vertex_jsonl_output(
    output: str,
    expected_request_hashes: Collection[str],
) -> dict[str, VertexBatchResult]:
    expected = set(expected_request_hashes)
    results: dict[str, VertexBatchResult] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            value: object = json.loads(line)
        except json.JSONDecodeError:
            _raise_secret_safe_public_error(
                VertexBatchContractError("Vertex output has no correlatable request")
            )
        if not isinstance(value, dict):
            raise VertexBatchContractError("Vertex output has no correlatable request")
        typed_value = cast(dict[str, object], value)
        request_hash = _output_request_hash(typed_value.get("request"))
        if request_hash not in expected:
            raise VertexBatchContractError(
                "Vertex output has an unexpected request hash"
            )
        if request_hash in results:
            raise VertexBatchContractError("Vertex output has a duplicate request hash")
        results[request_hash] = _parse_output_result(typed_value, request_hash)
    if results.keys() != expected:
        raise VertexBatchContractError("Vertex output is missing a request hash")
    return results


def _translated_gateway_error(error: Exception) -> IndexingGatewayError | None:
    from google.genai import errors as genai_errors

    if isinstance(
        error,
        (
            google_exceptions.DeadlineExceeded,
            requests.Timeout,
            httpx.TimeoutException,
            TimeoutError,
        ),
    ):
        return IndexingGatewayTimeoutError()
    if isinstance(
        error,
        (
            google_auth_exceptions.TransportError,
            requests.ConnectionError,
            httpx.NetworkError,
            ConnectionError,
        ),
    ):
        return IndexingGatewayConnectionError()
    if isinstance(error, google_exceptions.RetryError):
        cause = error.cause
        if isinstance(cause, Exception):
            return _translated_gateway_error(cause)
        return None
    if isinstance(error, genai_errors.APIError):
        return IndexingGatewayHTTPError(error.code)
    if isinstance(error, google_exceptions.GoogleAPICallError):
        raw_code = error.code
        if isinstance(raw_code, int):
            return IndexingGatewayHTTPError(raw_code)
        code_value = getattr(raw_code, "value", None)
        if isinstance(code_value, int):
            return IndexingGatewayHTTPError(code_value)
    return None


def _is_indeterminate_create_error(error: Exception) -> bool:
    translated = _translated_gateway_error(error)
    if isinstance(
        translated,
        (IndexingGatewayTimeoutError, IndexingGatewayConnectionError),
    ):
        return True
    return isinstance(translated, IndexingGatewayHTTPError) and (
        translated.status_code == 408 or 500 <= translated.status_code < 600
    )


def _raise_secret_safe_public_error(error: Exception) -> NoReturn:
    try:
        raise error from None
    except Exception:
        error.__context__ = None
        raise


@contextmanager
def _translate_gateway_errors() -> Iterator[None]:
    normalized_error: IndexingGatewayError | None = None
    try:
        yield
    except Exception as error:
        normalized_error = _translated_gateway_error(error)
        if normalized_error is None:
            raise
    if normalized_error is not None:
        _raise_secret_safe_public_error(normalized_error)


@contextmanager
def _translate_create_errors(submission_key: str) -> Iterator[None]:
    normalized_error: IndexingGatewayError | None = None
    try:
        yield
    except Exception as error:
        if _is_indeterminate_create_error(error):
            normalized_error = IndexingGatewayIndeterminateSubmissionError(
                submission_key
            )
        else:
            normalized_error = _translated_gateway_error(error)
        if normalized_error is None:
            raise
    if normalized_error is not None:
        _raise_secret_safe_public_error(normalized_error)


def _parse_gcs_uri(uri: str, *, require_object_path: bool = True) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise VertexBatchContractError("Vertex storage URI must use gs://")
    bucket, separator, object_name = uri[5:].partition("/")
    if not bucket or (require_object_path and (not separator or not object_name)):
        raise VertexBatchContractError("Vertex storage URI requires an object path")
    return bucket, object_name.rstrip("/")


_PENDING_JOB_STATES = frozenset(
    {
        "JOB_STATE_UNSPECIFIED",
        "JOB_STATE_QUEUED",
        "JOB_STATE_PENDING",
    }
)
_RUNNING_JOB_STATES = frozenset({"JOB_STATE_RUNNING", "JOB_STATE_UPDATING"})


def _job_status(value: genai_types.JobState | str | None) -> VertexBatchJobStatus:
    from google.genai import types as genai_types

    raw_value = value.value if isinstance(value, genai_types.JobState) else value
    if raw_value in _PENDING_JOB_STATES:
        return VertexBatchJobStatus.PENDING
    if raw_value in _RUNNING_JOB_STATES:
        return VertexBatchJobStatus.RUNNING
    if raw_value in {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}:
        return VertexBatchJobStatus.SUCCEEDED
    if raw_value == "JOB_STATE_CANCELLING":
        return VertexBatchJobStatus.CANCELLING
    if raw_value == "JOB_STATE_CANCELLED":
        return VertexBatchJobStatus.CANCELLED
    if raw_value in {"JOB_STATE_FAILED", "JOB_STATE_EXPIRED", "JOB_STATE_PAUSED"}:
        return VertexBatchJobStatus.FAILED
    raise VertexBatchContractError("Vertex batch job returned an unknown state")


def _batch_state(
    job: genai_types.BatchJob,
    *,
    input_uri: str | None = None,
    fallback_output_uri: str | None = None,
) -> VertexBatchState:
    if not job.name:
        raise VertexBatchContractError("Vertex batch job returned no resource name")
    output_uri = fallback_output_uri
    if job.output_info and job.output_info.gcs_output_directory:
        output_uri = job.output_info.gcs_output_directory
    elif job.dest and job.dest.gcs_uri:
        output_uri = job.dest.gcs_uri
    return VertexBatchState(
        remote_job_name=job.name,
        status=_job_status(job.state),
        input_uri=input_uri,
        output_uri=output_uri,
        error_code=job.error.code if job.error else None,
    )


class GoogleVertexBatchGateway:
    def __init__(
        self,
        *,
        config: VertexBatchConfig,
        object_prefix: str,
        credential_json_provider: Callable[[], str | None],
        request_timeout_seconds: float = 60,
    ) -> None:
        normalized_prefix = object_prefix.strip("/")
        if not normalized_prefix or ".." in normalized_prefix.split("/"):
            raise VertexBatchContractError("Vertex object prefix is invalid")
        if not math.isfinite(request_timeout_seconds) or request_timeout_seconds <= 0:
            raise VertexBatchContractError(
                "Vertex request timeout must be positive and finite"
            )
        self._config = config
        self._object_prefix = normalized_prefix
        self._credential_json_provider = credential_json_provider
        self._request_timeout_seconds = request_timeout_seconds
        self._request_timeout_milliseconds = math.ceil(request_timeout_seconds * 1000)

    def _credentials(self) -> Credentials:
        if (
            self._config.authentication_mode
            is VertexAuthenticationMode.WORKLOAD_IDENTITY
        ):
            credentials, _ = google.auth.default(scopes=[_CLOUD_PLATFORM_SCOPE])
            return credentials
        raw_credentials = self._credential_json_provider()
        if not raw_credentials:
            raise VertexBatchContractError(
                "Vertex service-account credential is unavailable"
            )
        try:
            parsed: object = json.loads(raw_credentials)
        except json.JSONDecodeError:
            _raise_secret_safe_public_error(
                VertexBatchContractError("Vertex service-account credential is invalid")
            )
        if not isinstance(parsed, dict):
            raise VertexBatchContractError(
                "Vertex service-account credential is invalid"
            )
        try:
            return service_account.Credentials.from_service_account_info(
                cast(dict[str, object], parsed),
                scopes=[_CLOUD_PLATFORM_SCOPE],
            )
        except (TypeError, ValueError):
            _raise_secret_safe_public_error(
                VertexBatchContractError("Vertex service-account credential is invalid")
            )

    def _storage_client(self, credentials: Credentials) -> storage.Client:
        return storage.Client(project=self._config.project, credentials=credentials)

    def _genai_client(self, credentials: Credentials) -> genai.Client:
        from google import genai
        from google.genai import types as genai_types

        return genai.Client(
            vertexai=True,
            project=self._config.project,
            location=self._config.location,
            credentials=credentials,
            http_options=genai_types.HttpOptions(
                api_version="v1",
                timeout=self._request_timeout_milliseconds,
            ),
        )

    def submit(self, requests: Sequence[VertexBatchRequest]) -> VertexBatchState:
        from google.genai import types as genai_types

        payload = build_vertex_jsonl(requests)
        request_set_hash = sha256(
            "\n".join(sorted(request.request_hash for request in requests)).encode()
        ).hexdigest()
        submission_key = f"regulatory-context-{request_set_hash}"
        base_uri = self._config.gcs_uri.rstrip("/")
        batch_prefix = f"{base_uri}/{self._object_prefix}/{request_set_hash}"
        input_uri = f"{batch_prefix}/input.jsonl"
        output_uri = f"{batch_prefix}/output"
        bucket_name, input_name = _parse_gcs_uri(input_uri)

        with _translate_gateway_errors():
            credentials = self._credentials()
            storage_client = self._storage_client(credentials)
            storage_client.bucket(bucket_name).blob(input_name).upload_from_string(
                payload,
                content_type="application/jsonl",
                timeout=self._request_timeout_seconds,
            )
            client = self._genai_client(credentials)
            with _translate_create_errors(submission_key):
                with traced_llm_call(
                    flow=LLMFlow.REGULATORY_CONTEXTUAL_BATCH,
                    model=self._config.model_name,
                    provider="vertex_ai",
                    extra_config={"request_count": str(len(requests))},
                ):
                    job = client.batches.create(
                        model=self._config.model_name,
                        src=input_uri,
                        config=genai_types.CreateBatchJobConfig(
                            display_name=submission_key,
                            dest=output_uri,
                        ),
                    )
        return _batch_state(
            job,
            input_uri=input_uri,
            fallback_output_uri=output_uri,
        )

    def get(self, remote_job_name: str) -> VertexBatchState:
        with _translate_gateway_errors():
            credentials = self._credentials()
            job = self._genai_client(credentials).batches.get(name=remote_job_name)
        return _batch_state(job)

    def reconcile_submission(self, submission_key: str) -> VertexBatchState | None:
        from google.genai import types as genai_types

        if re.fullmatch(r"regulatory-context-[0-9a-f]{64}", submission_key) is None:
            raise VertexBatchContractError("Vertex submission key is invalid")
        with _translate_gateway_errors():
            credentials = self._credentials()
            pager = self._genai_client(credentials).batches.list(
                config=genai_types.ListBatchJobsConfig(
                    page_size=2,
                    filter=f'displayName="{submission_key}"',
                )
            )
            matches = pager.page
        if not matches:
            return None
        if len(matches) != 1:
            raise VertexBatchSubmissionConflictError(submission_key, len(matches))
        return _batch_state(matches[0])

    def read_results(self, output_uri: str) -> str:
        bucket_name, output_prefix = _parse_gcs_uri(output_uri)
        with _translate_gateway_errors():
            credentials = self._credentials()
            storage_client = self._storage_client(credentials)
            blobs = sorted(
                (
                    blob
                    for blob in storage_client.list_blobs(
                        bucket_name,
                        prefix=f"{output_prefix}/",
                        timeout=self._request_timeout_seconds,
                    )
                    if blob.name.endswith(".jsonl")
                ),
                key=lambda blob: blob.name,
            )
            if not blobs:
                raise VertexBatchContractError("Vertex output contains no JSONL files")
            parts = [
                blob.download_as_text(timeout=self._request_timeout_seconds).rstrip(
                    "\n"
                )
                for blob in blobs
            ]
        return "\n".join(parts) + "\n"

    def cancel(self, remote_job_name: str) -> None:
        with _translate_gateway_errors():
            credentials = self._credentials()
            self._genai_client(credentials).batches.cancel(name=remote_job_name)

    def cleanup(self, prefix: str) -> None:
        bucket_name, object_prefix = _parse_gcs_uri(prefix)
        base_bucket, base_prefix = _parse_gcs_uri(
            self._config.gcs_uri, require_object_path=False
        )
        is_within_base = (
            object_prefix.startswith(f"{base_prefix}/")
            if base_prefix
            else bool(object_prefix)
        )
        if bucket_name != base_bucket or not is_within_base:
            raise VertexBatchContractError("Vertex cleanup prefix is outside its base")
        with _translate_gateway_errors():
            credentials = self._credentials()
            storage_client = self._storage_client(credentials)
            blobs = list(
                storage_client.list_blobs(
                    bucket_name,
                    prefix=f"{object_prefix}/",
                    timeout=self._request_timeout_seconds,
                )
            )
            if blobs:
                storage_client.bucket(bucket_name).delete_blobs(
                    blobs,
                    timeout=self._request_timeout_seconds,
                )
