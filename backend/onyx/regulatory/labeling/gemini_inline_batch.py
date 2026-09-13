from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import NoReturn, Protocol, cast

import requests as http_requests

from onyx.regulatory.indexing_jobs.models import (
    IndexingGatewayConnectionError,
    IndexingGatewayHTTPError,
    IndexingGatewayIndeterminateSubmissionError,
    IndexingGatewayTimeoutError,
    VertexBatchConfig,
)
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchContractError,
    VertexBatchGateway,
    VertexBatchJobStatus,
    VertexBatchRequest,
    VertexBatchResultError,
    VertexBatchState,
    VertexBatchSubmissionConflictError,
    VertexReadOnlyAccessProbe,
    build_vertex_jsonl,
)
from onyx.tracing.flows import LLMFlow
from onyx.tracing.llm_utils import traced_llm_call

_GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com"
_SUBMISSION_KEY_PATTERN = re.compile(r"regulatory-labeling-([0-9a-f]{64})")
_REMOTE_JOB_PATTERN = re.compile(r"batches/([A-Za-z0-9_-]+)")
_VIRTUAL_OUTPUT_PATTERN = re.compile(r"gemini-inline://batches/([A-Za-z0-9_-]+)")
_MODEL_NAME_PATTERN = re.compile(r"[A-Za-z0-9._-]+")
_REQUEST_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")

_INLINE_BATCH_BODY_LIMIT = 20_000_000
_MAX_INLINE_REQUESTS = 64
_MAX_BATCH_LIST_PAGES = 100
_MAX_METADATA_BYTES = 2 * 1024 * 1024
_MAX_PROBE_ERROR_BYTES = 64 * 1024

_SAFE_ACCESS_REASONS = frozenset(
    {
        "ACCESS_TOKEN_SCOPE_INSUFFICIENT",
        "API_KEY_ANDROID_APP_BLOCKED",
        "API_KEY_HTTP_REFERRER_BLOCKED",
        "API_KEY_INVALID",
        "API_KEY_IOS_APP_BLOCKED",
        "API_KEY_IP_ADDRESS_BLOCKED",
        "API_KEY_SERVICE_BLOCKED",
        "BILLING_DISABLED",
        "PERMISSION_DENIED",
        "RESOURCE_EXHAUSTED",
        "SERVICE_DISABLED",
        "UNAUTHENTICATED",
    }
)


class GeminiInlineBatchAccessError(VertexBatchContractError):
    """Secret-safe failure from the Gemini Developer API readiness probe."""

    status_code: int
    reason_code: str

    def __init__(self, status_code: int, reason_code: str) -> None:
        self.status_code = status_code
        self.reason_code = reason_code
        super().__init__(
            "Gemini Developer API access probe failed "
            f"(HTTP {status_code}, reason {reason_code})"
        )


class _HTTPResponse(Protocol):
    status_code: int
    headers: object
    content: bytes

    def iter_content(self, chunk_size: int) -> Iterator[bytes]: ...

    def close(self) -> None: ...


class _HTTPSession(Protocol):
    def request(self, method: str, url: str, **kwargs: object) -> _HTTPResponse: ...

    def close(self) -> None: ...


def _raise_secret_safe(error: Exception) -> NoReturn:
    try:
        raise error from None
    except Exception:
        error.__context__ = None
        raise


def _nested_dict(value: object, key: str) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    child = cast(dict[str, object], value).get(key)
    return cast(dict[str, object], child) if isinstance(child, dict) else {}


def _job_status(raw_state: object) -> VertexBatchJobStatus:
    if not isinstance(raw_state, str):
        raise VertexBatchContractError(
            "Gemini inline batch job returned an invalid state"
        )
    if raw_state in {
        "BATCH_STATE_UNSPECIFIED",
        "BATCH_STATE_PENDING",
        "JOB_STATE_UNSPECIFIED",
        "JOB_STATE_PENDING",
        "JOB_STATE_QUEUED",
    }:
        return VertexBatchJobStatus.PENDING
    if raw_state in {
        "BATCH_STATE_RUNNING",
        "JOB_STATE_RUNNING",
        "JOB_STATE_UPDATING",
    }:
        return VertexBatchJobStatus.RUNNING
    if raw_state in {
        "BATCH_STATE_SUCCEEDED",
        "JOB_STATE_PARTIALLY_SUCCEEDED",
        "JOB_STATE_SUCCEEDED",
    }:
        return VertexBatchJobStatus.SUCCEEDED
    if raw_state == "JOB_STATE_CANCELLING":
        return VertexBatchJobStatus.CANCELLING
    if raw_state in {"BATCH_STATE_CANCELLED", "JOB_STATE_CANCELLED"}:
        return VertexBatchJobStatus.CANCELLED
    if raw_state in {
        "BATCH_STATE_EXPIRED",
        "BATCH_STATE_FAILED",
        "JOB_STATE_EXPIRED",
        "JOB_STATE_FAILED",
        "JOB_STATE_PAUSED",
    }:
        return VertexBatchJobStatus.FAILED
    raise VertexBatchContractError("Gemini inline batch job returned an unknown state")


def _validate_submission_key(submission_key: str) -> None:
    if _SUBMISSION_KEY_PATTERN.fullmatch(submission_key) is None:
        raise VertexBatchContractError(
            "Gemini inline labeling submission key is invalid"
        )


def _validate_remote_job_name(remote_job_name: str) -> None:
    if _REMOTE_JOB_PATTERN.fullmatch(remote_job_name) is None:
        raise VertexBatchContractError("Gemini inline batch resource name is invalid")


def _virtual_output_uri(remote_job_name: str) -> str:
    _validate_remote_job_name(remote_job_name)
    return f"gemini-inline://{remote_job_name}"


def _remote_job_from_output_uri(output_uri: str) -> str:
    match = _VIRTUAL_OUTPUT_PATTERN.fullmatch(output_uri)
    if match is None:
        raise VertexBatchContractError("Gemini inline batch output URI is invalid")
    return f"batches/{match.group(1)}"


def _display_name(payload: object) -> object:
    metadata = _nested_dict(payload, "metadata")
    return metadata.get("displayName") or (
        cast(dict[str, object], payload).get("displayName")
        if isinstance(payload, dict)
        else None
    )


def _batch_state(
    payload: object,
    *,
    expected_display_name: str | None = None,
) -> VertexBatchState:
    if not isinstance(payload, dict):
        raise VertexBatchContractError(
            "Gemini inline batch job returned malformed metadata"
        )
    typed_payload = cast(dict[str, object], payload)
    name = typed_payload.get("name")
    if not isinstance(name, str):
        raise VertexBatchContractError(
            "Gemini inline batch job returned no resource name"
        )
    _validate_remote_job_name(name)
    if (
        expected_display_name is not None
        and _display_name(typed_payload) != expected_display_name
    ):
        raise VertexBatchContractError(
            "Gemini inline batch job returned a different display name"
        )
    metadata = _nested_dict(typed_payload, "metadata")
    status = _job_status(metadata.get("state") or typed_payload.get("state"))
    error = _nested_dict(typed_payload, "error") or _nested_dict(metadata, "error")
    error_code = error.get("code")
    return VertexBatchState(
        remote_job_name=name,
        status=status,
        output_uri=(
            _virtual_output_uri(name)
            if status is VertexBatchJobStatus.SUCCEEDED
            else None
        ),
        error_code=error_code if isinstance(error_code, int) else None,
    )


def _safe_access_reason(payload: object, status_code: int) -> str:
    error = _nested_dict(payload, "error")
    details = error.get("details")
    if isinstance(details, list):
        for detail in details:
            if not isinstance(detail, dict):
                continue
            reason = cast(dict[str, object], detail).get("reason")
            if isinstance(reason, str) and reason in _SAFE_ACCESS_REASONS:
                return reason
    status = error.get("status")
    if isinstance(status, str) and status in _SAFE_ACCESS_REASONS:
        return status
    return f"HTTP_{status_code}"


class LabelingGeminiInlineBatchGateway(VertexBatchGateway):
    def __init__(
        self,
        *,
        config: VertexBatchConfig,
        api_key_provider: Callable[[], str],
        request_timeout_seconds: float = 20,
        max_result_bytes: int = 64 * 1024 * 1024,
        max_reconciliation_seconds: float = 180,
        session_factory: Callable[[], _HTTPSession] | None = None,
    ) -> None:
        if not math.isfinite(request_timeout_seconds) or request_timeout_seconds <= 0:
            raise VertexBatchContractError(
                "Gemini inline request timeout must be positive and finite"
            )
        if max_result_bytes < 1:
            raise VertexBatchContractError(
                "Gemini inline result byte limit must be positive"
            )
        if (
            not math.isfinite(max_reconciliation_seconds)
            or max_reconciliation_seconds <= 0
        ):
            raise VertexBatchContractError(
                "Gemini inline reconciliation deadline must be positive and finite"
            )
        if _MODEL_NAME_PATTERN.fullmatch(config.model_name) is None:
            raise VertexBatchContractError("Gemini inline model name is invalid")
        self._config = config
        self._api_key_provider = api_key_provider
        self._request_timeout_seconds = request_timeout_seconds
        self._max_result_bytes = max_result_bytes
        self._max_reconciliation_seconds = max_reconciliation_seconds
        self._session_factory = session_factory or (
            lambda: cast(_HTTPSession, http_requests.Session())
        )

    def _api_key(self) -> str:
        try:
            value = self._api_key_provider()
        except Exception:
            _raise_secret_safe(
                VertexBatchContractError("Gemini authorization API key is unavailable")
            )
        if not isinstance(value, str):
            raise VertexBatchContractError(
                "Gemini authorization API key is unavailable"
            )
        stripped = value.strip()
        if (
            not stripped
            or len(stripped) > 4096
            or any(
                ord(character) < 33 or ord(character) > 126 for character in stripped
            )
        ):
            raise VertexBatchContractError(
                "Gemini authorization API key is unavailable"
            )
        return stripped

    @contextmanager
    def _session(self) -> Iterator[tuple[_HTTPSession, dict[str, str]]]:
        session = self._session_factory()
        try:
            yield session, {"x-goog-api-key": self._api_key()}
        finally:
            session.close()

    def _request(
        self,
        session: _HTTPSession,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        **kwargs: object,
    ) -> _HTTPResponse:
        try:
            return session.request(
                method,
                url,
                headers=headers,
                timeout=self._request_timeout_seconds,
                stream=True,
                **kwargs,
            )
        except http_requests.Timeout:
            _raise_secret_safe(IndexingGatewayTimeoutError())
        except http_requests.ConnectionError:
            _raise_secret_safe(IndexingGatewayConnectionError())
        except http_requests.RequestException:
            _raise_secret_safe(IndexingGatewayConnectionError())

    def _response_bytes(self, response: _HTTPResponse, *, max_bytes: int) -> bytes:
        header_get = getattr(response.headers, "get", lambda _key: None)
        content_length = header_get("Content-Length")
        if isinstance(content_length, str) and content_length.isdigit():
            if int(content_length) > max_bytes:
                response.close()
                raise VertexBatchContractError(
                    "Gemini inline response exceeds its size limit"
                )
        buffer = bytearray()
        try:
            iterator = getattr(response, "iter_content", None)
            chunks = (
                cast(Callable[..., Iterator[bytes]], iterator)(chunk_size=65536)
                if callable(iterator)
                else iter([response.content])
            )
            for chunk in chunks:
                if not isinstance(chunk, bytes):
                    raise VertexBatchContractError(
                        "Gemini inline response stream is malformed"
                    )
                if len(buffer) + len(chunk) > max_bytes:
                    raise VertexBatchContractError(
                        "Gemini inline response exceeds its size limit"
                    )
                buffer.extend(chunk)
        except http_requests.Timeout:
            _raise_secret_safe(IndexingGatewayTimeoutError())
        except (
            http_requests.ConnectionError,
            http_requests.exceptions.ChunkedEncodingError,
        ):
            _raise_secret_safe(IndexingGatewayConnectionError())
        except http_requests.RequestException:
            _raise_secret_safe(IndexingGatewayConnectionError())
        finally:
            response.close()
        return bytes(buffer)

    def _response_json(self, response: _HTTPResponse, *, max_bytes: int) -> object:
        raw = self._response_bytes(response, max_bytes=max_bytes)
        try:
            return json.loads(raw)
        except (UnicodeDecodeError, ValueError):
            _raise_secret_safe(
                VertexBatchContractError(
                    "Gemini inline gateway returned malformed JSON"
                )
            )

    def _json_request(
        self,
        session: _HTTPSession,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        max_bytes: int = _MAX_METADATA_BYTES,
        access_probe: bool = False,
        **kwargs: object,
    ) -> object:
        response = self._request(
            session,
            method,
            url,
            headers=headers,
            **kwargs,
        )
        if response.status_code >= 400:
            status_code = response.status_code
            if access_probe:
                try:
                    payload = self._response_json(
                        response, max_bytes=_MAX_PROBE_ERROR_BYTES
                    )
                except (IndexingGatewayConnectionError, IndexingGatewayTimeoutError):
                    raise
                except Exception:
                    payload = {}
                _raise_secret_safe(
                    GeminiInlineBatchAccessError(
                        status_code, _safe_access_reason(payload, status_code)
                    )
                )
            response.close()
            _raise_secret_safe(IndexingGatewayHTTPError(status_code))
        return self._response_json(response, max_bytes=max_bytes)

    def _empty_request(
        self,
        session: _HTTPSession,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        **kwargs: object,
    ) -> None:
        response = self._request(
            session,
            method,
            url,
            headers=headers,
            **kwargs,
        )
        status_code = response.status_code
        response.close()
        if status_code >= 400:
            _raise_secret_safe(IndexingGatewayHTTPError(status_code))

    def submit(
        self,
        requests: Sequence[VertexBatchRequest],
        *,
        submission_key: str,
        max_jsonl_bytes: int,
    ) -> VertexBatchState:
        _validate_submission_key(submission_key)
        if max_jsonl_bytes < 1:
            raise VertexBatchContractError(
                "Gemini inline JSONL byte limit must be positive"
            )
        if len(requests) > _MAX_INLINE_REQUESTS:
            raise VertexBatchContractError(
                "Gemini inline batch exceeds the request-count limit"
            )
        jsonl = build_vertex_jsonl(requests).encode("utf-8")
        if len(jsonl) > max_jsonl_bytes:
            raise VertexBatchContractError(
                "Gemini inline batch exceeds the configured JSONL byte limit"
            )
        body = {
            "batch": {
                "display_name": submission_key,
                "input_config": {
                    "requests": {
                        "requests": [
                            {
                                "request": request.to_generate_content_request(),
                                "metadata": {"key": request.request_hash},
                            }
                            for request in requests
                        ]
                    }
                },
            }
        }
        encoded_body = json.dumps(
            body,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded_body) >= _INLINE_BATCH_BODY_LIMIT:
            raise VertexBatchContractError(
                "Gemini inline batch must remain under the 20 MB API limit"
            )
        with self._session() as (session, api_key_headers):
            try:
                with traced_llm_call(
                    flow=LLMFlow.REGULATORY_LABELING_BATCH,
                    model=self._config.model_name,
                    provider="gemini_developer",
                    extra_config={"request_count": str(len(requests))},
                ):
                    payload = self._json_request(
                        session,
                        "POST",
                        f"{_GEMINI_API_BASE_URL}/v1beta/models/"
                        f"{self._config.model_name}:batchGenerateContent",
                        headers={
                            **api_key_headers,
                            "Content-Type": "application/json",
                        },
                        data=encoded_body,
                    )
            except (
                IndexingGatewayConnectionError,
                IndexingGatewayTimeoutError,
            ):
                _raise_secret_safe(
                    IndexingGatewayIndeterminateSubmissionError(submission_key)
                )
            except IndexingGatewayHTTPError as error:
                if error.status_code == 408 or 500 <= error.status_code < 600:
                    _raise_secret_safe(
                        IndexingGatewayIndeterminateSubmissionError(submission_key)
                    )
                raise
            try:
                return _batch_state(
                    payload,
                    expected_display_name=submission_key,
                )
            except VertexBatchContractError:
                _raise_secret_safe(
                    IndexingGatewayIndeterminateSubmissionError(submission_key)
                )

    def get(self, remote_job_name: str) -> VertexBatchState:
        _validate_remote_job_name(remote_job_name)
        with self._session() as (session, headers):
            payload = self._json_request(
                session,
                "GET",
                f"{_GEMINI_API_BASE_URL}/v1beta/{remote_job_name}",
                headers=headers,
                params={"fields": "name,metadata(displayName,state),error(code),done"},
            )
        return _batch_state(payload)

    def reconcile_submission(self, submission_key: str) -> VertexBatchState | None:
        _validate_submission_key(submission_key)
        deadline = time.monotonic() + self._max_reconciliation_seconds
        matches: list[object] = []
        page_token: str | None = None
        seen_page_tokens: set[str] = set()
        with self._session() as (session, headers):
            for _page_number in range(_MAX_BATCH_LIST_PAGES):
                if time.monotonic() >= deadline:
                    raise IndexingGatewayTimeoutError()
                params: dict[str, object] = {
                    "fields": (
                        "operations(name,metadata(displayName,state),"
                        "error(code),done),nextPageToken"
                    ),
                    "pageSize": 100,
                }
                if page_token is not None:
                    params["pageToken"] = page_token
                payload = self._json_request(
                    session,
                    "GET",
                    f"{_GEMINI_API_BASE_URL}/v1beta/batches",
                    headers=headers,
                    params=params,
                )
                if not isinstance(payload, dict):
                    raise VertexBatchContractError(
                        "Gemini inline batch list is malformed"
                    )
                typed_payload = cast(dict[str, object], payload)
                operations = typed_payload.get("operations", [])
                if not isinstance(operations, list):
                    raise VertexBatchContractError(
                        "Gemini inline batch list is malformed"
                    )
                matches.extend(
                    operation
                    for operation in operations
                    if _display_name(operation) == submission_key
                )
                if len(matches) > 1:
                    raise VertexBatchSubmissionConflictError(
                        submission_key, len(matches)
                    )
                next_page_token = typed_payload.get("nextPageToken")
                if next_page_token is None:
                    break
                if (
                    not isinstance(next_page_token, str)
                    or not next_page_token
                    or next_page_token in seen_page_tokens
                ):
                    raise VertexBatchContractError(
                        "Gemini inline batch list returned an invalid page token"
                    )
                seen_page_tokens.add(next_page_token)
                page_token = next_page_token
            else:
                raise VertexBatchContractError(
                    "Gemini inline batch reconciliation exceeded the page limit"
                )
        if not matches:
            return None
        return _batch_state(matches[0], expected_display_name=submission_key)

    def _inline_responses(self, payload: object) -> list[object]:
        state = _batch_state(payload)
        if state.status is not VertexBatchJobStatus.SUCCEEDED:
            raise VertexBatchContractError(
                "Gemini inline batch results are unavailable before success"
            )
        metadata = _nested_dict(payload, "metadata")
        response = _nested_dict(payload, "response")
        output = (
            _nested_dict(metadata, "output")
            or _nested_dict(response, "output")
            or response
        )
        if output.get("responsesFile") is not None:
            raise VertexBatchContractError(
                "Gemini inline batch unexpectedly returned a file result"
            )
        inlined: object = output.get("inlinedResponses")
        if isinstance(inlined, dict):
            inlined = cast(dict[str, object], inlined).get("inlinedResponses")
        if not isinstance(inlined, list) or not inlined:
            raise VertexBatchContractError(
                "Gemini inline batch returned malformed or empty results"
            )
        if len(inlined) > _MAX_INLINE_REQUESTS:
            raise VertexBatchContractError(
                "Gemini inline batch returned too many results"
            )
        batch_stats = _nested_dict(metadata, "batchStats") or _nested_dict(
            response, "batchStats"
        )
        request_count = batch_stats.get("requestCount")
        if request_count is not None:
            try:
                expected_count = int(cast(str | int, request_count))
            except (TypeError, ValueError):
                raise VertexBatchContractError(
                    "Gemini inline batch returned malformed request statistics"
                ) from None
            if expected_count != len(inlined):
                raise VertexBatchContractError(
                    "Gemini inline batch returned a truncated result set"
                )
        return cast(list[object], inlined)

    def read_results(self, output_uri: str) -> Iterator[str]:
        remote_job_name = _remote_job_from_output_uri(output_uri)

        def iter_lines() -> Iterator[str]:
            with self._session() as (session, headers):
                payload = self._json_request(
                    session,
                    "GET",
                    f"{_GEMINI_API_BASE_URL}/v1beta/{remote_job_name}",
                    headers=headers,
                    max_bytes=self._max_result_bytes,
                )
            used_bytes = 0
            for item in self._inline_responses(payload):
                if not isinstance(item, dict):
                    raise VertexBatchContractError(
                        "Gemini inline batch result row is malformed"
                    )
                typed_item = cast(dict[str, object], item)
                metadata = typed_item.get("metadata")
                if not isinstance(metadata, dict):
                    raise VertexBatchContractError(
                        "Gemini inline batch result has no request key"
                    )
                key = cast(dict[str, object], metadata).get("key")
                if (
                    not isinstance(key, str)
                    or _REQUEST_HASH_PATTERN.fullmatch(key) is None
                ):
                    raise VertexBatchContractError(
                        "Gemini inline batch result has no request key"
                    )
                response = typed_item.get("response")
                error = typed_item.get("error")
                if (response is None) == (error is None):
                    raise VertexBatchContractError(
                        "Gemini inline batch result must contain one outcome"
                    )
                if response is not None:
                    if not isinstance(response, dict):
                        raise VertexBatchContractError(
                            "Gemini inline batch response is malformed"
                        )
                    normalized: dict[str, object] = {
                        "key": key,
                        "response": response,
                    }
                else:
                    if not isinstance(error, dict):
                        raise VertexBatchContractError(
                            "Gemini inline batch error is malformed"
                        )
                    normalized = {
                        "key": key,
                        "error": VertexBatchResultError.REMOTE_ERROR.value,
                    }
                line = (
                    json.dumps(
                        normalized,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                line_bytes = len(line.encode("utf-8"))
                if used_bytes + line_bytes > self._max_result_bytes:
                    raise VertexBatchContractError(
                        "Gemini inline normalized results exceed their size limit"
                    )
                used_bytes += line_bytes
                yield line

        return iter_lines()

    def cancel(self, remote_job_name: str) -> None:
        _validate_remote_job_name(remote_job_name)
        with self._session() as (session, headers):
            self._empty_request(
                session,
                "POST",
                f"{_GEMINI_API_BASE_URL}/v1beta/{remote_job_name}:cancel",
                headers={**headers, "Content-Type": "application/json"},
                data=b"{}",
            )

    def delete(self, remote_job_name: str) -> None:
        _validate_remote_job_name(remote_job_name)
        with self._session() as (session, headers):
            self._empty_request(
                session,
                "DELETE",
                f"{_GEMINI_API_BASE_URL}/v1beta/{remote_job_name}",
                headers=headers,
            )

    def cleanup(self, prefix: str) -> None:
        _remote_job_from_output_uri(prefix)

    def probe_gemini_read_access(self) -> VertexReadOnlyAccessProbe:
        with self._session() as (session, headers):
            model_payload = self._json_request(
                session,
                "GET",
                f"{_GEMINI_API_BASE_URL}/v1beta/models/{self._config.model_name}",
                headers=headers,
                access_probe=True,
            )
            if not isinstance(model_payload, dict):
                raise VertexBatchContractError(
                    "Gemini inline model metadata is malformed"
                )
            supported_methods = cast(dict[str, object], model_payload).get(
                "supportedGenerationMethods"
            )
            if (
                not isinstance(supported_methods, list)
                or "batchGenerateContent" not in supported_methods
            ):
                raise VertexBatchContractError(
                    "Gemini model does not support batchGenerateContent"
                )
            batches_payload = self._json_request(
                session,
                "GET",
                f"{_GEMINI_API_BASE_URL}/v1beta/batches",
                headers=headers,
                params={"fields": "nextPageToken", "pageSize": 1},
                access_probe=True,
            )
            if not isinstance(batches_payload, dict):
                raise VertexBatchContractError("Gemini inline batch list is malformed")
        return VertexReadOnlyAccessProbe(credential_identity="gemini-authorization-key")
