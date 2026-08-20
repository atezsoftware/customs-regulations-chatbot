from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import pytest
import requests
from google.auth import exceptions as google_auth_exceptions
from google.auth.credentials import Credentials

from onyx.regulatory.indexing_jobs.gemini_files_batch import (
    GoogleGeminiFilesBatchGateway,
    _credentials_from_service_account_json,
    gemini_input_file_name,
)
from onyx.regulatory.indexing_jobs.models import (
    IndexingGatewayConnectionError,
    IndexingGatewayHTTPError,
    IndexingGatewayIndeterminateSubmissionError,
    VertexAuthenticationMode,
    VertexBatchConfig,
)
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchContractError,
    VertexBatchJobStatus,
    VertexBatchRequest,
    VertexBatchSubmissionConflictError,
    build_vertex_jsonl,
)

_SUBMISSION_KEY = "regulatory-context-" + "a" * 64


class _FakeCredentials(Credentials):
    service_account_email = "regulatory@example.iam.gserviceaccount.com"

    def __init__(self) -> None:
        super().__init__()
        self.token = "oauth-token"

    def refresh(self, request: object) -> None:
        del request

    def apply(self, headers: dict[str, str], token: str | None = None) -> None:
        headers["authorization"] = f"Bearer {token or self.token}"

    @property
    def expired(self) -> bool:
        return False

    @property
    def valid(self) -> bool:
        return True


class _FakeResponse:
    def __init__(
        self,
        status_code: int,
        *,
        payload: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
        content: bytes = b"",
    ) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}
        self.content = content

    def json(self) -> dict[str, object]:
        return self._payload


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse | Exception]) -> None:
        self._responses = iter(responses)
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.closed = False

    def request(self, method: str, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append((method, url, kwargs))
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self) -> None:
        self.closed = True


def _config() -> VertexBatchConfig:
    return VertexBatchConfig(
        model_configuration_id=73,
        model_name="gemini-3.6-flash",
        project="customs-prod",
        location="europe-west4",
        authentication_mode=VertexAuthenticationMode.SERVICE_ACCOUNT_JSON,
    )


def _gateway(
    session: _FakeSession,
    credentials_factory: Callable[[str], Credentials] | None = None,
) -> GoogleGeminiFilesBatchGateway:
    return GoogleGeminiFilesBatchGateway(
        config=_config(),
        credential_json_provider=lambda: '{"type":"service_account"}',
        credentials_factory=credentials_factory or (lambda _raw: _FakeCredentials()),
        session_factory=cast(Any, lambda: session),
    )


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_gateway_rejects_nonpositive_or_nonfinite_timeout(timeout: float) -> None:
    with pytest.raises(VertexBatchContractError, match="positive and finite"):
        GoogleGeminiFilesBatchGateway(
            config=_config(),
            credential_json_provider=lambda: '{"type":"service_account"}',
            request_timeout_seconds=timeout,
        )


def test_service_account_requests_documented_gemini_oauth_scopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_scopes: list[str] = []

    def fake_from_info(
        _info: dict[str, object], *, scopes: Sequence[str]
    ) -> Credentials:
        captured_scopes.extend(scopes)
        return _FakeCredentials()

    monkeypatch.setattr(
        "onyx.regulatory.indexing_jobs.gemini_files_batch.service_account.Credentials.from_service_account_info",
        fake_from_info,
    )

    _credentials_from_service_account_json('{"type":"service_account"}')

    assert captured_scopes == [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/generative-language.retriever",
    ]


def test_credential_refresh_failure_is_terminal_and_secret_safe() -> None:
    class _RejectedCredentials(_FakeCredentials):
        @property
        def valid(self) -> bool:
            return False

        def refresh(self, request: object) -> None:
            del request
            raise google_auth_exceptions.RefreshError("secret provider response")

    with pytest.raises(IndexingGatewayHTTPError) as caught:
        _gateway(
            _FakeSession([]),
            credentials_factory=lambda _raw: _RejectedCredentials(),
        ).probe_gemini_read_access()

    assert caught.value.status_code == 401
    assert "secret provider response" not in str(caught.value)
    assert caught.value.__context__ is None


def test_retryable_credential_refresh_failure_remains_retryable_and_secret_safe() -> (
    None
):
    class _RetryableCredentials(_FakeCredentials):
        @property
        def valid(self) -> bool:
            return False

        def refresh(self, request: object) -> None:
            del request
            raise google_auth_exceptions.RefreshError(
                "secret token endpoint response", retryable=True
            )

    with pytest.raises(IndexingGatewayConnectionError) as caught:
        _gateway(
            _FakeSession([]),
            credentials_factory=lambda _raw: _RetryableCredentials(),
        ).probe_gemini_read_access()

    assert "secret token endpoint response" not in str(caught.value)
    assert caught.value.__context__ is None


def test_submit_raises_secret_safe_indeterminate_error_after_ambiguous_create() -> None:
    session = _FakeSession(
        [
            _FakeResponse(
                200,
                headers={"x-goog-upload-url": "https://upload.example/session-1"},
            ),
            _FakeResponse(200, payload={"file": {"name": "files/input-1"}}),
            requests.Timeout("provider-secret-payload"),
        ]
    )

    with pytest.raises(IndexingGatewayIndeterminateSubmissionError) as caught:
        _gateway(session).submit(
            [VertexBatchRequest(prompt="first prompt")],
            submission_key=_SUBMISSION_KEY,
            max_jsonl_bytes=4096,
        )

    assert caught.value.submission_key == _SUBMISSION_KEY
    assert "provider-secret-payload" not in str(caught.value)
    assert caught.value.__context__ is None


def test_submit_deletes_uploaded_input_after_definite_create_rejection() -> None:
    session = _FakeSession(
        [
            _FakeResponse(
                200,
                headers={"x-goog-upload-url": "https://upload.example/session-1"},
            ),
            _FakeResponse(
                200,
                payload={"file": {"name": gemini_input_file_name(_SUBMISSION_KEY)}},
            ),
            _FakeResponse(400),
            _FakeResponse(200),
        ]
    )

    with pytest.raises(IndexingGatewayHTTPError) as caught:
        _gateway(session).submit(
            [VertexBatchRequest(prompt="first prompt")],
            submission_key=_SUBMISSION_KEY,
            max_jsonl_bytes=4096,
        )

    assert caught.value.status_code == 400
    assert session.calls[-1][0] == "DELETE"
    assert session.calls[-1][1].endswith(
        f"/v1beta/{gemini_input_file_name(_SUBMISSION_KEY)}"
    )


def test_submit_uploads_jsonl_and_creates_file_backed_batch_with_oauth() -> None:
    session = _FakeSession(
        [
            _FakeResponse(
                200,
                headers={"x-goog-upload-url": "https://upload.example/session-1"},
            ),
            _FakeResponse(200, payload={"file": {"name": "files/input-1"}}),
            _FakeResponse(
                200,
                payload={
                    "name": "batches/job-1",
                    "metadata": {
                        "displayName": _SUBMISSION_KEY,
                        "state": "JOB_STATE_PENDING",
                        "model": "models/gemini-3.6-flash",
                    },
                },
            ),
        ]
    )

    state = _gateway(session).submit(
        [VertexBatchRequest(prompt="first prompt")],
        submission_key=_SUBMISSION_KEY,
        max_jsonl_bytes=4096,
    )

    assert state.remote_job_name == "batches/job-1"
    assert state.status is VertexBatchJobStatus.PENDING
    assert state.input_uri == "files/input-1"
    assert state.output_uri is None
    assert session.closed
    assert [call[0] for call in session.calls] == ["POST", "POST", "POST"]
    assert session.calls[0][1].endswith("/upload/v1beta/files")
    assert session.calls[0][2]["json"] == {
        "file": {
            "name": gemini_input_file_name(_SUBMISSION_KEY),
            "displayName": _SUBMISSION_KEY,
        }
    }
    assert session.calls[1][1] == "https://upload.example/session-1"
    assert session.calls[2][1].endswith(
        "/v1beta/models/gemini-3.6-flash:batchGenerateContent"
    )
    assert session.calls[2][2]["json"] == {
        "batch": {
            "display_name": _SUBMISSION_KEY,
            "input_config": {"file_name": "files/input-1"},
        }
    }
    for _method, _url, kwargs in session.calls:
        headers = kwargs["headers"]
        assert headers["authorization"] == "Bearer oauth-token"
        assert headers["x-goog-user-project"] == "customs-prod"


def test_get_returns_generated_file_from_real_batch_operation_shape() -> None:
    session = _FakeSession(
        [
            _FakeResponse(
                200,
                payload={
                    "name": "batches/job-1",
                    "metadata": {
                        "displayName": _SUBMISSION_KEY,
                        "state": "JOB_STATE_SUCCEEDED",
                        "model": "models/gemini-3.6-flash",
                    },
                    "done": True,
                    "response": {"responsesFile": "files/output-1"},
                },
            )
        ]
    )

    state = _gateway(session).get("batches/job-1")

    assert state.status is VertexBatchJobStatus.SUCCEEDED
    assert state.output_uri == "files/output-1"
    assert session.calls[0][1].endswith("/v1beta/batches/job-1")
    assert session.closed


def test_reconcile_matches_exact_display_name_on_developer_batch_list() -> None:
    session = _FakeSession(
        [
            _FakeResponse(
                200,
                payload={
                    "operations": [
                        {
                            "name": "batches/other",
                            "metadata": {
                                "displayName": "regulatory-context-" + "b" * 64,
                                "state": "JOB_STATE_PENDING",
                            },
                        },
                        {
                            "name": "batches/job-1",
                            "metadata": {
                                "displayName": _SUBMISSION_KEY,
                                "state": "JOB_STATE_RUNNING",
                            },
                        },
                    ]
                },
            )
        ]
    )

    state = _gateway(session).reconcile_submission(_SUBMISSION_KEY)

    assert state is not None
    assert state.remote_job_name == "batches/job-1"
    assert state.status is VertexBatchJobStatus.RUNNING
    assert session.calls[0][2]["params"] == {"pageSize": 100}


def test_reconcile_follows_batch_list_pagination() -> None:
    session = _FakeSession(
        [
            _FakeResponse(
                200,
                payload={
                    "operations": [],
                    "nextPageToken": "next-page",
                },
            ),
            _FakeResponse(
                200,
                payload={
                    "operations": [
                        {
                            "name": "batches/job-1",
                            "metadata": {
                                "displayName": _SUBMISSION_KEY,
                                "state": "JOB_STATE_RUNNING",
                            },
                        }
                    ]
                },
            ),
        ]
    )

    state = _gateway(session).reconcile_submission(_SUBMISSION_KEY)

    assert state is not None
    assert state.remote_job_name == "batches/job-1"
    assert session.calls[0][2]["params"] == {"pageSize": 100}
    assert session.calls[1][2]["params"] == {
        "pageSize": 100,
        "pageToken": "next-page",
    }


def test_reconcile_rejects_duplicate_display_name_matches() -> None:
    operation = {
        "name": "batches/job-1",
        "metadata": {
            "displayName": _SUBMISSION_KEY,
            "state": "JOB_STATE_PENDING",
        },
    }
    duplicate = {
        **operation,
        "name": "batches/job-2",
    }
    session = _FakeSession(
        [_FakeResponse(200, payload={"operations": [operation, duplicate]})]
    )

    try:
        _gateway(session).reconcile_submission(_SUBMISSION_KEY)
    except VertexBatchSubmissionConflictError as error:
        assert error.submission_key == _SUBMISSION_KEY
        assert error.match_count == 2
    else:
        raise AssertionError("duplicate Gemini batch identity was accepted")


def test_read_results_downloads_generated_file_as_utf8_jsonl() -> None:
    session = _FakeSession(
        [_FakeResponse(200, content=b'{"response": {}}\n{"response": {}}\n')]
    )

    lines = list(_gateway(session).read_results("files/output-1"))

    assert lines == ['{"response": {}}\n', '{"response": {}}\n']
    assert session.calls[0][1].endswith("/download/v1beta/files/output-1:download")
    assert session.calls[0][2]["params"] == {"alt": "media"}
    assert session.closed


def test_cancel_delete_and_file_cleanup_use_bounded_resource_endpoints() -> None:
    session = _FakeSession(
        [
            _FakeResponse(200),
            _FakeResponse(200),
            _FakeResponse(200),
        ]
    )
    gateway = _gateway(session)

    gateway.cancel("batches/job-1")
    gateway.delete("batches/job-1")
    gateway.cleanup("files/input-1")

    assert [method for method, _url, _kwargs in session.calls] == [
        "POST",
        "DELETE",
        "DELETE",
    ]
    assert session.calls[0][1].endswith("/v1beta/batches/job-1:cancel")
    assert session.calls[1][1].endswith("/v1beta/batches/job-1")
    assert session.calls[2][1].endswith("/v1beta/files/input-1")


def test_readiness_probe_is_observational_and_returns_service_account_identity() -> (
    None
):
    session = _FakeSession(
        [
            _FakeResponse(
                200,
                payload={
                    "name": "models/gemini-3.6-flash",
                    "supportedGenerationMethods": [
                        "generateContent",
                        "batchGenerateContent",
                    ],
                },
            ),
            _FakeResponse(200, payload={"operations": []}),
            _FakeResponse(200, payload={"files": []}),
        ]
    )

    probe = _gateway(session).probe_gemini_read_access()

    assert probe.credential_identity == "regulatory@example.iam.gserviceaccount.com"
    assert [call[0] for call in session.calls] == ["GET", "GET", "GET"]
    assert session.calls[0][1].endswith("/v1beta/models/gemini-3.6-flash")
    assert session.calls[1][1].endswith("/v1beta/batches")
    assert session.calls[1][2]["params"] == {"pageSize": 1}
    assert session.calls[2][1].endswith("/v1beta/files")
    assert session.calls[2][2]["params"] == {"pageSize": 1}


def test_readiness_probe_rejects_model_without_batch_generation() -> None:
    session = _FakeSession(
        [
            _FakeResponse(
                200,
                payload={
                    "name": "models/gemini-3.6-flash",
                    "supportedGenerationMethods": ["generateContent"],
                },
            )
        ]
    )

    with pytest.raises(VertexBatchContractError, match="batchGenerateContent"):
        _gateway(session).probe_gemini_read_access()

    assert len(session.calls) == 1


def test_submit_rejects_whole_batch_when_jsonl_limit_would_truncate_requests() -> None:
    first = VertexBatchRequest(prompt="first prompt")
    second = VertexBatchRequest(prompt="second prompt")
    first_line_bytes = len(build_vertex_jsonl([first]).encode("utf-8"))
    session = _FakeSession([])

    with pytest.raises(VertexBatchContractError, match="exceeds"):
        _gateway(session).submit(
            [first, second],
            submission_key=_SUBMISSION_KEY,
            max_jsonl_bytes=first_line_bytes,
        )

    assert session.calls == []
