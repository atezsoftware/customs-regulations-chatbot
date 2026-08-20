from __future__ import annotations

import json
import math
import re
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, NoReturn, cast

import google.auth
import httpx
import requests
from google.api_core import exceptions as google_exceptions
from google.auth import exceptions as google_auth_exceptions
from google.auth.credentials import Credentials
from google.cloud import storage
from google.oauth2 import service_account

from onyx.regulatory.indexing_jobs.models import (
    IndexingGatewayConnectionError,
    IndexingGatewayError,
    IndexingGatewayHTTPError,
    IndexingGatewayIndeterminateSubmissionError,
    IndexingGatewayTimeoutError,
    VertexAuthenticationMode,
    VertexBatchConfig,
)
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchContractError,
    VertexBatchJobStatus,
    VertexBatchRequest,
    VertexBatchState,
    VertexBatchSubmissionConflictError,
)
from onyx.tracing.flows import LLMFlow
from onyx.tracing.llm_utils import traced_llm_call

if TYPE_CHECKING:
    from google import genai
    from google.genai import types as genai_types

_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_PENDING_JOB_STATES = frozenset(
    {"JOB_STATE_UNSPECIFIED", "JOB_STATE_QUEUED", "JOB_STATE_PENDING"}
)
_RUNNING_JOB_STATES = frozenset({"JOB_STATE_RUNNING", "JOB_STATE_UPDATING"})


def _raise_secret_safe(error: Exception) -> NoReturn:
    try:
        raise error from None
    except Exception:
        error.__context__ = None
        raise


def _credential_identity(credentials: Credentials) -> str:
    for attribute in ("service_account_email", "signer_email"):
        value = getattr(credentials, attribute, None)
        if isinstance(value, str) and value.strip() and value.strip() != "default":
            return value.strip()
    raise VertexBatchContractError(
        "Vertex credential identity is unavailable for readiness verification"
    )


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
        return (
            _translated_gateway_error(cause) if isinstance(cause, Exception) else None
        )
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
        translated, (IndexingGatewayTimeoutError, IndexingGatewayConnectionError)
    ):
        return True
    return isinstance(translated, IndexingGatewayHTTPError) and (
        translated.status_code == 408 or 500 <= translated.status_code < 600
    )


@contextmanager
def _translate_gateway_errors() -> Iterator[None]:
    try:
        yield
    except Exception as error:
        translated = _translated_gateway_error(error)
        if translated is None:
            raise
        _raise_secret_safe(translated)


@contextmanager
def _translate_create_errors(submission_key: str) -> Iterator[None]:
    try:
        yield
    except Exception as error:
        if _is_indeterminate_create_error(error):
            _raise_secret_safe(
                IndexingGatewayIndeterminateSubmissionError(submission_key)
            )
        translated = _translated_gateway_error(error)
        if translated is None:
            raise
        _raise_secret_safe(translated)


def _parse_gcs_uri(uri: str, *, require_object_path: bool = True) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise VertexBatchContractError("Vertex storage URI must use gs://")
    bucket, separator, object_name = uri[5:].partition("/")
    if not bucket or (require_object_path and (not separator or not object_name)):
        raise VertexBatchContractError("Vertex storage URI requires an object path")
    return bucket, object_name.rstrip("/")


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


def _legacy_jsonl_line(request: VertexBatchRequest) -> bytes:
    payload = {
        "request": {
            "contents": [{"role": "user", "parts": [{"text": request.prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 256},
        }
    }
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


class GoogleVertexBatchGateway:
    """Compatibility gateway for GCS-backed jobs persisted before Files API rollout."""

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
        if config.gcs_uri is None:
            raise VertexBatchContractError("Legacy Vertex storage URI is unavailable")
        _parse_gcs_uri(config.gcs_uri, require_object_path=False)
        if not math.isfinite(request_timeout_seconds) or request_timeout_seconds <= 0:
            raise VertexBatchContractError(
                "Vertex request timeout must be positive and finite"
            )
        self._config = config
        self._gcs_uri = config.gcs_uri
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
            if not isinstance(parsed, dict):
                raise ValueError
            return service_account.Credentials.from_service_account_info(
                cast(dict[str, object], parsed), scopes=[_CLOUD_PLATFORM_SCOPE]
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            _raise_secret_safe(
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
                api_version="v1", timeout=self._request_timeout_milliseconds
            ),
        )

    @contextmanager
    def _managed_genai_client(self, credentials: Credentials) -> Iterator[genai.Client]:
        client = self._genai_client(credentials)
        try:
            yield client
        finally:
            client.close()

    @contextmanager
    def _managed_storage_client(
        self, credentials: Credentials
    ) -> Iterator[storage.Client]:
        client = self._storage_client(credentials)
        try:
            yield client
        finally:
            client.close()

    def submit(
        self,
        requests: Sequence[VertexBatchRequest],
        *,
        submission_key: str,
        max_jsonl_bytes: int,
    ) -> VertexBatchState:
        from google.genai import types as genai_types

        if re.fullmatch(r"regulatory-context-[0-9a-f]{64}", submission_key) is None:
            raise VertexBatchContractError("Vertex submission key is invalid")
        if max_jsonl_bytes < 1 or not requests:
            raise VertexBatchContractError("Vertex batch input is invalid")
        if len({request.request_hash for request in requests}) != len(requests):
            raise VertexBatchContractError(
                "Vertex batch contains a duplicate request hash"
            )
        request_set_hash = submission_key.removeprefix("regulatory-context-")
        batch_prefix = (
            f"{self._gcs_uri.rstrip('/')}/{self._object_prefix}/{request_set_hash}"
        )
        input_uri = f"{batch_prefix}/input.jsonl"
        output_uri = f"{batch_prefix}/output"
        bucket_name, input_name = _parse_gcs_uri(input_uri)
        with _translate_gateway_errors():
            credentials = self._credentials()
            with self._managed_storage_client(credentials) as storage_client:
                with tempfile.SpooledTemporaryFile(max_size=1024 * 1024) as payload:
                    used_bytes = 0
                    for request in requests:
                        line = _legacy_jsonl_line(request)
                        if used_bytes + len(line) > max_jsonl_bytes:
                            raise VertexBatchContractError(
                                "Vertex batch exceeds the configured JSONL byte limit"
                            )
                        payload.write(line)
                        used_bytes += len(line)
                    storage_client.bucket(bucket_name).blob(
                        input_name
                    ).upload_from_file(
                        payload,
                        content_type="application/jsonl",
                        rewind=True,
                        timeout=self._request_timeout_seconds,
                    )
            with self._managed_genai_client(credentials) as client:
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
                                display_name=submission_key, dest=output_uri
                            ),
                        )
        return _batch_state(job, input_uri=input_uri, fallback_output_uri=output_uri)

    def get(self, remote_job_name: str) -> VertexBatchState:
        with _translate_gateway_errors():
            credentials = self._credentials()
            with self._managed_genai_client(credentials) as client:
                job = client.batches.get(name=remote_job_name)
        return _batch_state(job)

    def reconcile_submission(self, submission_key: str) -> VertexBatchState | None:
        from google.genai import types as genai_types

        if re.fullmatch(r"regulatory-context-[0-9a-f]{64}", submission_key) is None:
            raise VertexBatchContractError("Vertex submission key is invalid")
        with _translate_gateway_errors():
            credentials = self._credentials()
            with self._managed_genai_client(credentials) as client:
                pager = client.batches.list(
                    config=genai_types.ListBatchJobsConfig(
                        page_size=2, filter=f'displayName="{submission_key}"'
                    )
                )
                matches = pager.page
        if not matches:
            return None
        if len(matches) != 1:
            raise VertexBatchSubmissionConflictError(submission_key, len(matches))
        return _batch_state(matches[0])

    def read_results(self, output_uri: str) -> Iterator[str]:
        bucket_name, output_prefix = _parse_gcs_uri(output_uri)

        def iter_lines() -> Iterator[str]:
            with _translate_gateway_errors():
                credentials = self._credentials()
                with self._managed_storage_client(credentials) as storage_client:
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
                        raise VertexBatchContractError(
                            "Vertex output contains no JSONL files"
                        )
                    for blob in blobs:
                        with blob.open(
                            "rt",
                            encoding="utf-8",
                            timeout=self._request_timeout_seconds,
                        ) as stream:
                            yield from stream

        return iter_lines()

    def cancel(self, remote_job_name: str) -> None:
        with _translate_gateway_errors():
            credentials = self._credentials()
            with self._managed_genai_client(credentials) as client:
                client.batches.cancel(name=remote_job_name)

    def delete(self, remote_job_name: str) -> None:
        with _translate_gateway_errors():
            credentials = self._credentials()
            with self._managed_genai_client(credentials) as client:
                client.batches.delete(name=remote_job_name)

    def cleanup(self, prefix: str) -> None:
        bucket_name, object_prefix = _parse_gcs_uri(prefix)
        base_bucket, base_prefix = _parse_gcs_uri(
            self._gcs_uri, require_object_path=False
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
            with self._managed_storage_client(credentials) as storage_client:
                bucket = storage_client.bucket(bucket_name)
                batch: list[object] = []
                for blob in storage_client.list_blobs(
                    bucket_name,
                    prefix=f"{object_prefix}/",
                    timeout=self._request_timeout_seconds,
                ):
                    batch.append(blob)
                    if len(batch) == 100:
                        bucket.delete_blobs(
                            batch, timeout=self._request_timeout_seconds
                        )
                        batch.clear()
                if batch:
                    bucket.delete_blobs(batch, timeout=self._request_timeout_seconds)
