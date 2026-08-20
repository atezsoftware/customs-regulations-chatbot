from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import NoReturn, Protocol, cast

import google.auth
import requests as http_requests
from google.auth import exceptions as google_auth_exceptions
from google.auth.credentials import Credentials
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import service_account

from onyx.regulatory.indexing_jobs.models import (
    IndexingGatewayConnectionError,
    IndexingGatewayHTTPError,
    IndexingGatewayIndeterminateSubmissionError,
    IndexingGatewayTimeoutError,
    VertexAuthenticationMode,
    VertexBatchConfig,
)
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchContractError,
    VertexBatchGateway,
    VertexBatchJobStatus,
    VertexBatchRequest,
    VertexBatchState,
    VertexBatchSubmissionConflictError,
    VertexReadOnlyAccessProbe,
    build_vertex_jsonl,
)
from onyx.tracing.flows import LLMFlow
from onyx.tracing.llm_utils import traced_llm_call

_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_GENERATIVE_LANGUAGE_SCOPE = (
    "https://www.googleapis.com/auth/generative-language.retriever"
)
_GEMINI_OAUTH_SCOPES = (_CLOUD_PLATFORM_SCOPE, _GENERATIVE_LANGUAGE_SCOPE)
_GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com"
_SUBMISSION_KEY_PATTERN = re.compile(r"regulatory-context-([0-9a-f]{64})")
_MAX_BATCH_LIST_PAGES = 100


def gemini_input_file_name(submission_key: str) -> str:
    match = _SUBMISSION_KEY_PATTERN.fullmatch(submission_key)
    if match is None:
        raise VertexBatchContractError("Gemini submission key is invalid")
    return f"files/regctx-{match.group(1)[:32]}"


class _HTTPResponse(Protocol):
    status_code: int
    headers: object
    content: bytes

    def json(self) -> object: ...


class _HTTPSession(Protocol):
    def request(self, method: str, url: str, **kwargs: object) -> _HTTPResponse: ...

    def close(self) -> None: ...


def _raise_secret_safe(error: Exception) -> NoReturn:
    try:
        raise error from None
    except Exception:
        error.__context__ = None
        raise


def _credentials_from_service_account_json(raw_credentials: str) -> Credentials:
    try:
        parsed: object = json.loads(raw_credentials)
    except json.JSONDecodeError:
        _raise_secret_safe(
            VertexBatchContractError("Gemini service-account credential is invalid")
        )
    if not isinstance(parsed, dict):
        raise VertexBatchContractError("Gemini service-account credential is invalid")
    try:
        return service_account.Credentials.from_service_account_info(
            cast(dict[str, object], parsed),
            scopes=_GEMINI_OAUTH_SCOPES,
        )
    except (TypeError, ValueError):
        _raise_secret_safe(
            VertexBatchContractError("Gemini service-account credential is invalid")
        )


def _credential_identity(credentials: Credentials) -> str:
    for attribute in ("service_account_email", "signer_email"):
        value = getattr(credentials, attribute, None)
        if isinstance(value, str) and value.strip() and value.strip() != "default":
            return value.strip()
    raise VertexBatchContractError(
        "Gemini credential identity is unavailable for readiness verification"
    )


def _job_status(raw_state: object) -> VertexBatchJobStatus:
    if raw_state in {
        "JOB_STATE_UNSPECIFIED",
        "JOB_STATE_QUEUED",
        "JOB_STATE_PENDING",
    }:
        return VertexBatchJobStatus.PENDING
    if raw_state in {"JOB_STATE_RUNNING", "JOB_STATE_UPDATING"}:
        return VertexBatchJobStatus.RUNNING
    if raw_state in {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}:
        return VertexBatchJobStatus.SUCCEEDED
    if raw_state == "JOB_STATE_CANCELLING":
        return VertexBatchJobStatus.CANCELLING
    if raw_state == "JOB_STATE_CANCELLED":
        return VertexBatchJobStatus.CANCELLED
    if raw_state in {"JOB_STATE_FAILED", "JOB_STATE_EXPIRED", "JOB_STATE_PAUSED"}:
        return VertexBatchJobStatus.FAILED
    raise VertexBatchContractError("Gemini batch job returned an unknown state")


def _nested_dict(value: object, key: str) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    child = cast(dict[str, object], value).get(key)
    return cast(dict[str, object], child) if isinstance(child, dict) else {}


def _batch_state(
    payload: object,
    *,
    fallback_input_uri: str | None = None,
) -> VertexBatchState:
    if not isinstance(payload, dict):
        raise VertexBatchContractError("Gemini batch job returned malformed metadata")
    typed_payload = cast(dict[str, object], payload)
    name = typed_payload.get("name")
    metadata = _nested_dict(typed_payload, "metadata")
    if not isinstance(name, str) or not name.startswith("batches/"):
        raise VertexBatchContractError("Gemini batch job returned no resource name")
    response = _nested_dict(typed_payload, "response")
    output = _nested_dict(metadata, "output")
    output_uri = response.get("responsesFile") or output.get("responsesFile")
    if output_uri is not None and (
        not isinstance(output_uri, str) or not output_uri.startswith("files/")
    ):
        raise VertexBatchContractError("Gemini batch job returned invalid output")
    error = _nested_dict(typed_payload, "error") or _nested_dict(metadata, "error")
    error_code = error.get("code")
    return VertexBatchState(
        remote_job_name=name,
        status=_job_status(metadata.get("state")),
        input_uri=fallback_input_uri,
        output_uri=output_uri,
        error_code=error_code if isinstance(error_code, int) else None,
    )


class GoogleGeminiFilesBatchGateway(VertexBatchGateway):
    def __init__(
        self,
        *,
        config: VertexBatchConfig,
        credential_json_provider: Callable[[], str | None],
        request_timeout_seconds: float = 60,
        credentials_factory: Callable[[str], Credentials] = (
            _credentials_from_service_account_json
        ),
        session_factory: Callable[[], _HTTPSession] | None = None,
    ) -> None:
        if not math.isfinite(request_timeout_seconds) or request_timeout_seconds <= 0:
            raise VertexBatchContractError(
                "Gemini request timeout must be positive and finite"
            )
        self._config = config
        self._credential_json_provider = credential_json_provider
        self._request_timeout_seconds = request_timeout_seconds
        self._credentials_factory = credentials_factory
        self._session_factory = session_factory or (
            lambda: cast(_HTTPSession, http_requests.Session())
        )

    def _credentials(self) -> Credentials:
        if (
            self._config.authentication_mode
            is VertexAuthenticationMode.WORKLOAD_IDENTITY
        ):
            credentials, _ = google.auth.default(scopes=_GEMINI_OAUTH_SCOPES)
            return credentials
        raw_credentials = self._credential_json_provider()
        if not raw_credentials:
            raise VertexBatchContractError(
                "Gemini service-account credential is unavailable"
            )
        return self._credentials_factory(raw_credentials)

    def _authorization_headers(self, credentials: Credentials) -> dict[str, str]:
        if not credentials.valid or not credentials.token:
            try:
                credentials.refresh(GoogleAuthRequest())
            except google_auth_exceptions.TransportError:
                _raise_secret_safe(IndexingGatewayConnectionError())
            except google_auth_exceptions.RefreshError as error:
                if error.retryable:
                    _raise_secret_safe(IndexingGatewayConnectionError())
                _raise_secret_safe(IndexingGatewayHTTPError(401))
            except Exception:
                _raise_secret_safe(
                    VertexBatchContractError("Gemini credential refresh failed")
                )
        headers: dict[str, str] = {}
        credentials.apply(headers)
        headers["x-goog-user-project"] = self._config.project
        return headers

    @contextmanager
    def _session(
        self,
    ) -> Iterator[tuple[_HTTPSession, dict[str, str], Credentials]]:
        session = self._session_factory()
        try:
            credentials = self._credentials()
            yield session, self._authorization_headers(credentials), credentials
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
            response = session.request(
                method,
                url,
                headers=headers,
                timeout=self._request_timeout_seconds,
                **kwargs,
            )
        except http_requests.Timeout:
            _raise_secret_safe(IndexingGatewayTimeoutError())
        except http_requests.ConnectionError:
            _raise_secret_safe(IndexingGatewayConnectionError())
        if response.status_code >= 400:
            _raise_secret_safe(IndexingGatewayHTTPError(response.status_code))
        return response

    def _response_json(self, response: _HTTPResponse) -> object:
        try:
            return response.json()
        except (TypeError, ValueError):
            _raise_secret_safe(
                VertexBatchContractError("Gemini gateway returned malformed JSON")
            )

    def _upload_jsonl(
        self,
        session: _HTTPSession,
        headers: dict[str, str],
        payload: bytes,
        submission_key: str,
    ) -> str:
        start_headers = {
            **headers,
            "Content-Type": "application/json",
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(len(payload)),
            "X-Goog-Upload-Header-Content-Type": "application/jsonl",
        }
        start_response = self._request(
            session,
            "POST",
            f"{_GEMINI_API_BASE_URL}/upload/v1beta/files",
            headers=start_headers,
            json={
                "file": {
                    "name": gemini_input_file_name(submission_key),
                    "displayName": submission_key,
                }
            },
        )
        upload_url = getattr(start_response.headers, "get", lambda _key: None)(
            "x-goog-upload-url"
        )
        if not isinstance(upload_url, str) or not upload_url.startswith("https://"):
            raise VertexBatchContractError("Gemini Files upload URL is unavailable")
        upload_response = self._request(
            session,
            "POST",
            upload_url,
            headers={
                **headers,
                "Content-Length": str(len(payload)),
                "Content-Type": "application/jsonl",
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            data=payload,
        )
        file_payload = _nested_dict(self._response_json(upload_response), "file")
        file_name = file_payload.get("name")
        if not isinstance(file_name, str) or not file_name.startswith("files/"):
            raise VertexBatchContractError("Gemini Files upload returned no resource")
        return file_name

    def submit(
        self,
        requests: Sequence[VertexBatchRequest],
        *,
        submission_key: str,
        max_jsonl_bytes: int,
    ) -> VertexBatchState:
        gemini_input_file_name(submission_key)
        payload = build_vertex_jsonl(requests).encode("utf-8")
        if len(payload) > max_jsonl_bytes:
            raise VertexBatchContractError(
                "Gemini batch exceeds the configured JSONL byte limit"
            )
        with self._session() as (session, headers, _credentials):
            input_file_name = self._upload_jsonl(
                session,
                headers,
                payload,
                submission_key,
            )
            try:
                with traced_llm_call(
                    flow=LLMFlow.REGULATORY_CONTEXTUAL_BATCH,
                    model=self._config.model_name,
                    provider="gemini_developer",
                    extra_config={"request_count": str(len(requests))},
                ):
                    response = self._request(
                        session,
                        "POST",
                        f"{_GEMINI_API_BASE_URL}/v1beta/models/"
                        f"{self._config.model_name}:batchGenerateContent",
                        headers={**headers, "Content-Type": "application/json"},
                        json={
                            "batch": {
                                "display_name": submission_key,
                                "input_config": {"file_name": input_file_name},
                            }
                        },
                    )
            except (
                IndexingGatewayTimeoutError,
                IndexingGatewayConnectionError,
            ):
                _raise_secret_safe(
                    IndexingGatewayIndeterminateSubmissionError(submission_key)
                )
            except IndexingGatewayHTTPError as error:
                if error.status_code == 408 or 500 <= error.status_code < 600:
                    _raise_secret_safe(
                        IndexingGatewayIndeterminateSubmissionError(submission_key)
                    )
                try:
                    self.cleanup(input_file_name)
                except IndexingGatewayHTTPError as cleanup_error:
                    if cleanup_error.status_code != 404:
                        _raise_secret_safe(
                            IndexingGatewayIndeterminateSubmissionError(submission_key)
                        )
                except Exception:
                    _raise_secret_safe(
                        IndexingGatewayIndeterminateSubmissionError(submission_key)
                    )
                raise
            return _batch_state(
                self._response_json(response),
                fallback_input_uri=input_file_name,
            )

    def get(self, remote_job_name: str) -> VertexBatchState:
        if re.fullmatch(r"batches/[A-Za-z0-9_-]+", remote_job_name) is None:
            raise VertexBatchContractError("Gemini batch resource name is invalid")
        with self._session() as (session, headers, _credentials):
            response = self._request(
                session,
                "GET",
                f"{_GEMINI_API_BASE_URL}/v1beta/{remote_job_name}",
                headers=headers,
            )
            return _batch_state(self._response_json(response))

    def reconcile_submission(self, submission_key: str) -> VertexBatchState | None:
        gemini_input_file_name(submission_key)
        matches: list[object] = []
        page_token: str | None = None
        seen_page_tokens: set[str] = set()
        with self._session() as (session, headers, _credentials):
            for _page_number in range(_MAX_BATCH_LIST_PAGES):
                params: dict[str, object] = {"pageSize": 100}
                if page_token is not None:
                    params["pageToken"] = page_token
                response = self._request(
                    session,
                    "GET",
                    f"{_GEMINI_API_BASE_URL}/v1beta/batches",
                    headers=headers,
                    params=params,
                )
                payload = self._response_json(response)
                if not isinstance(payload, dict):
                    raise VertexBatchContractError("Gemini batch list is malformed")
                typed_payload = cast(dict[str, object], payload)
                operations = typed_payload.get("operations", [])
                if not isinstance(operations, list):
                    raise VertexBatchContractError("Gemini batch list is malformed")
                for operation in operations:
                    metadata = _nested_dict(operation, "metadata")
                    if metadata.get("displayName") == submission_key:
                        matches.append(operation)
                next_page_token = typed_payload.get("nextPageToken")
                if next_page_token is None:
                    break
                if (
                    not isinstance(next_page_token, str)
                    or not next_page_token
                    or next_page_token in seen_page_tokens
                ):
                    raise VertexBatchContractError(
                        "Gemini batch list returned an invalid page token"
                    )
                seen_page_tokens.add(next_page_token)
                page_token = next_page_token
            else:
                raise VertexBatchContractError(
                    "Gemini batch reconciliation exceeded the page limit"
                )
        if not matches:
            return None
        if len(matches) != 1:
            raise VertexBatchSubmissionConflictError(submission_key, len(matches))
        return _batch_state(matches[0])

    def read_results(self, output_uri: str) -> Iterator[str]:
        if re.fullmatch(r"files/[A-Za-z0-9_-]+", output_uri) is None:
            raise VertexBatchContractError("Gemini output file resource is invalid")

        def iter_lines() -> Iterator[str]:
            with self._session() as (session, headers, _credentials):
                response = self._request(
                    session,
                    "GET",
                    f"{_GEMINI_API_BASE_URL}/download/v1beta/{output_uri}:download",
                    headers=headers,
                    params={"alt": "media"},
                )
                try:
                    output = response.content.decode("utf-8")
                except UnicodeDecodeError:
                    _raise_secret_safe(
                        VertexBatchContractError(
                            "Gemini output file is not valid UTF-8"
                        )
                    )
                yield from output.splitlines(keepends=True)

        return iter_lines()

    def cancel(self, remote_job_name: str) -> None:
        if re.fullmatch(r"batches/[A-Za-z0-9_-]+", remote_job_name) is None:
            raise VertexBatchContractError("Gemini batch resource name is invalid")
        with self._session() as (session, headers, _credentials):
            self._request(
                session,
                "POST",
                f"{_GEMINI_API_BASE_URL}/v1beta/{remote_job_name}:cancel",
                headers={**headers, "Content-Type": "application/json"},
                json={},
            )

    def delete(self, remote_job_name: str) -> None:
        if re.fullmatch(r"batches/[A-Za-z0-9_-]+", remote_job_name) is None:
            raise VertexBatchContractError("Gemini batch resource name is invalid")
        with self._session() as (session, headers, _credentials):
            self._request(
                session,
                "DELETE",
                f"{_GEMINI_API_BASE_URL}/v1beta/{remote_job_name}",
                headers=headers,
            )

    def cleanup(self, prefix: str) -> None:
        if re.fullmatch(r"files/[A-Za-z0-9_-]+", prefix) is None:
            raise VertexBatchContractError("Gemini file resource name is invalid")
        with self._session() as (session, headers, _credentials):
            self._request(
                session,
                "DELETE",
                f"{_GEMINI_API_BASE_URL}/v1beta/{prefix}",
                headers=headers,
            )

    def probe_gemini_read_access(self) -> VertexReadOnlyAccessProbe:
        with self._session() as (session, headers, credentials):
            model_response = self._request(
                session,
                "GET",
                f"{_GEMINI_API_BASE_URL}/v1beta/models/{self._config.model_name}",
                headers=headers,
            )
            model_payload = self._response_json(model_response)
            if not isinstance(model_payload, dict):
                raise VertexBatchContractError("Gemini model metadata is malformed")
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
            self._request(
                session,
                "GET",
                f"{_GEMINI_API_BASE_URL}/v1beta/batches",
                headers=headers,
                params={"pageSize": 1},
            )
            self._request(
                session,
                "GET",
                f"{_GEMINI_API_BASE_URL}/v1beta/files",
                headers=headers,
                params={"pageSize": 1},
            )
            identity = _credential_identity(credentials)
        return VertexReadOnlyAccessProbe(credential_identity=identity)
