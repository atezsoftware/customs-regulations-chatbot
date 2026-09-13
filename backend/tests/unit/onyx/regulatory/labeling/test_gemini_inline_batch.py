from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, cast

import pytest
import requests

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
    VertexBatchJobStatus,
    VertexBatchRequest,
    VertexBatchSubmissionConflictError,
    build_vertex_jsonl,
)
from onyx.regulatory.labeling.gemini_inline_batch import (
    GeminiInlineBatchAccessError,
    LabelingGeminiInlineBatchGateway,
)
from onyx.regulatory.labeling.provider import parse_labeling_batch_output
from onyx.tracing.flows import LLMFlow

_SUBMISSION_KEY = "regulatory-labeling-" + "a" * 64


class _FakeResponse:
    def __init__(
        self,
        status_code: int,
        *,
        payload: object | None = None,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
        chunks: list[bytes | Exception] | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self.content = (
            content
            if content is not None
            else json.dumps(payload if payload is not None else {}).encode()
        )
        self._chunks = chunks
        self.closed = False
        self.chunks_read = 0

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        del chunk_size
        chunks: list[bytes | Exception]
        if self._chunks is None:
            chunks = [self.content]
        else:
            chunks = self._chunks
        for chunk in chunks:
            self.chunks_read += 1
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def close(self) -> None:
        self.closed = True


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
        model_name="gemini-3.8-flash",
        project="customs-dev",
        location="global",
        authentication_mode=VertexAuthenticationMode.SERVICE_ACCOUNT_JSON,
    )


def _gateway(
    session: _FakeSession,
    *,
    api_key_provider: Callable[[], str] = lambda: "test-auth-key",
    request_timeout_seconds: float = 20,
    max_result_bytes: int = 64 * 1024 * 1024,
    max_reconciliation_seconds: float = 180,
) -> LabelingGeminiInlineBatchGateway:
    return LabelingGeminiInlineBatchGateway(
        config=_config(),
        api_key_provider=api_key_provider,
        request_timeout_seconds=request_timeout_seconds,
        max_result_bytes=max_result_bytes,
        max_reconciliation_seconds=max_reconciliation_seconds,
        session_factory=cast(Any, lambda: session),
    )


def _request(prompt: str = "label this chunk") -> VertexBatchRequest:
    return VertexBatchRequest(
        prompt=prompt,
        generation_config={
            "responseMimeType": "application/json",
            "responseJsonSchema": {
                "type": "object",
                "properties": {"labels": {"type": "array"}},
            },
            "maxOutputTokens": 32768,
        },
        system_instruction="Return the required labeling JSON only.",
    )


def _operation(
    *,
    name: str = "batches/job-1",
    display_name: str = _SUBMISSION_KEY,
    state: str = "JOB_STATE_PENDING",
    response: object | None = None,
    error: object | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": name,
        "metadata": {"displayName": display_name, "state": state},
    }
    if response is not None:
        payload["response"] = response
    if error is not None:
        payload["error"] = error
    return payload


def _model_response() -> dict[str, object]:
    return {
        "name": "models/gemini-3.8-flash",
        "supportedGenerationMethods": ["generateContent", "batchGenerateContent"],
    }


def _candidate(text: str = '{"labels":[],"abstained":true}') -> dict[str, object]:
    return {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {"parts": [{"text": text}]},
            }
        ]
    }


def test_submit_preserves_full_request_and_uses_only_api_key_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession([_FakeResponse(200, payload=_operation())])
    gateway = _gateway(session)
    traces: list[dict[str, object]] = []

    @contextmanager
    def traced(**kwargs: object) -> Iterator[None]:
        traces.append(kwargs)
        yield

    monkeypatch.setattr(
        "onyx.regulatory.labeling.gemini_inline_batch.traced_llm_call", traced
    )
    request = _request()

    state = gateway.submit(
        [request], submission_key=_SUBMISSION_KEY, max_jsonl_bytes=8 * 1024 * 1024
    )

    assert state.remote_job_name == "batches/job-1"
    assert state.status is VertexBatchJobStatus.PENDING
    assert state.input_uri is None
    assert state.output_uri is None
    assert len(session.calls) == 1
    method, url, kwargs = session.calls[0]
    assert method == "POST"
    assert url.endswith("/v1beta/models/gemini-3.8-flash:batchGenerateContent")
    assert kwargs["headers"] == {
        "Content-Type": "application/json",
        "x-goog-api-key": "test-auth-key",
    }
    assert "json" not in kwargs
    body = json.loads(cast(bytes, kwargs["data"]))
    assert body == {
        "batch": {
            "display_name": _SUBMISSION_KEY,
            "input_config": {
                "requests": {
                    "requests": [
                        {
                            "metadata": {"key": request.request_hash},
                            "request": request.to_generate_content_request(),
                        }
                    ]
                }
            },
        }
    }
    assert traces == [
        {
            "flow": LLMFlow.REGULATORY_LABELING_BATCH,
            "model": "gemini-3.8-flash",
            "provider": "gemini_developer",
            "extra_config": {"request_count": "1"},
        }
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"request_timeout_seconds": 0},
        {"request_timeout_seconds": float("inf")},
        {"max_result_bytes": 0},
        {"max_reconciliation_seconds": float("nan")},
    ],
)
def test_constructor_rejects_unbounded_configuration(
    kwargs: dict[str, object],
) -> None:
    arguments: dict[str, object] = {
        "config": _config(),
        "api_key_provider": lambda: "key",
    }
    arguments.update(kwargs)
    with pytest.raises(VertexBatchContractError):
        LabelingGeminiInlineBatchGateway(**cast(Any, arguments))


def test_submit_rejects_invalid_duplicate_and_oversized_inputs_before_key_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _gateway(
        _FakeSession([]),
        api_key_provider=lambda: pytest.fail("invalid input must not read the API key"),
    )
    request = _request()

    with pytest.raises(VertexBatchContractError, match="duplicate"):
        gateway.submit(
            [request, request],
            submission_key=_SUBMISSION_KEY,
            max_jsonl_bytes=8 * 1024 * 1024,
        )
    with pytest.raises(VertexBatchContractError, match="JSONL byte limit"):
        gateway.submit([request], submission_key=_SUBMISSION_KEY, max_jsonl_bytes=10)
    with pytest.raises(VertexBatchContractError, match="submission key"):
        gateway.submit(
            [request],
            submission_key="regulatory-context-" + "a" * 64,
            max_jsonl_bytes=8 * 1024 * 1024,
        )

    monkeypatch.setattr(
        "onyx.regulatory.labeling.gemini_inline_batch._INLINE_BATCH_BODY_LIMIT", 20
    )
    with pytest.raises(VertexBatchContractError, match="20 MB"):
        gateway.submit(
            [request],
            submission_key=_SUBMISSION_KEY,
            max_jsonl_bytes=8 * 1024 * 1024,
        )


def test_submit_allows_inline_envelope_above_exact_jsonl_budget() -> None:
    request = _request()
    exact_jsonl_size = len(build_vertex_jsonl([request]).encode())
    session = _FakeSession([_FakeResponse(200, payload=_operation())])

    _gateway(session).submit(
        [request],
        submission_key=_SUBMISSION_KEY,
        max_jsonl_bytes=exact_jsonl_size,
    )

    assert len(cast(bytes, session.calls[0][2]["data"])) > exact_jsonl_size


@pytest.mark.parametrize(
    "error",
    [
        requests.Timeout("secret timeout"),
        requests.ConnectionError("secret connection"),
    ],
)
def test_submit_maps_ambiguous_transport_failure_to_indeterminate(
    error: Exception,
) -> None:
    session = _FakeSession([error])
    with pytest.raises(IndexingGatewayIndeterminateSubmissionError) as caught:
        _gateway(session).submit(
            [_request()],
            submission_key=_SUBMISSION_KEY,
            max_jsonl_bytes=8 * 1024 * 1024,
        )
    assert caught.value.submission_key == _SUBMISSION_KEY
    assert "secret" not in str(caught.value)
    assert caught.value.__context__ is None


@pytest.mark.parametrize("status", [408, 500, 503])
def test_submit_maps_ambiguous_http_failure_to_indeterminate(status: int) -> None:
    with pytest.raises(IndexingGatewayIndeterminateSubmissionError):
        _gateway(_FakeSession([_FakeResponse(status)])).submit(
            [_request()],
            submission_key=_SUBMISSION_KEY,
            max_jsonl_bytes=8 * 1024 * 1024,
        )


def test_submit_keeps_definite_rejection_and_malformed_success_distinct() -> None:
    with pytest.raises(IndexingGatewayHTTPError) as caught:
        _gateway(_FakeSession([_FakeResponse(400)])).submit(
            [_request()],
            submission_key=_SUBMISSION_KEY,
            max_jsonl_bytes=8 * 1024 * 1024,
        )
    assert caught.value.status_code == 400

    with pytest.raises(IndexingGatewayIndeterminateSubmissionError):
        _gateway(_FakeSession([_FakeResponse(200, payload={"name": "broken"})])).submit(
            [_request()],
            submission_key=_SUBMISSION_KEY,
            max_jsonl_bytes=8 * 1024 * 1024,
        )


def test_get_sets_virtual_output_only_for_successful_inline_job() -> None:
    session = _FakeSession(
        [
            _FakeResponse(200, payload=_operation(state="JOB_STATE_RUNNING")),
            _FakeResponse(200, payload=_operation(state="JOB_STATE_SUCCEEDED")),
        ]
    )
    gateway = _gateway(session)

    running = gateway.get("batches/job-1")
    succeeded = gateway.get("batches/job-1")

    assert running.status is VertexBatchJobStatus.RUNNING
    assert running.output_uri is None
    assert succeeded.status is VertexBatchJobStatus.SUCCEEDED
    assert succeeded.output_uri == "gemini-inline://batches/job-1"
    assert session.calls[0][2]["params"] == {
        "fields": "name,metadata(displayName,state),error(code),done"
    }


def test_reconcile_scans_bounded_pages_and_matches_exact_display_name() -> None:
    session = _FakeSession(
        [
            _FakeResponse(
                200,
                payload={
                    "operations": [_operation(display_name=_SUBMISSION_KEY + "-other")],
                    "nextPageToken": "next",
                },
            ),
            _FakeResponse(
                200,
                payload={
                    "operations": [_operation(state="JOB_STATE_PARTIALLY_SUCCEEDED")]
                },
            ),
        ]
    )

    state = _gateway(session).reconcile_submission(_SUBMISSION_KEY)

    assert state is not None
    assert state.status is VertexBatchJobStatus.SUCCEEDED
    assert state.output_uri == "gemini-inline://batches/job-1"
    assert session.calls[0][2]["params"]["pageSize"] == 100
    assert session.calls[0][2]["params"]["fields"] == (
        "operations(name,metadata(displayName,state),error(code),done),nextPageToken"
    )
    assert session.calls[1][2]["params"]["pageToken"] == "next"


def test_reconcile_rejects_duplicates_and_invalid_page_tokens() -> None:
    duplicate = _operation()
    with pytest.raises(VertexBatchSubmissionConflictError):
        _gateway(
            _FakeSession(
                [_FakeResponse(200, payload={"operations": [duplicate, duplicate]})]
            )
        ).reconcile_submission(_SUBMISSION_KEY)

    session = _FakeSession(
        [
            _FakeResponse(200, payload={"operations": [], "nextPageToken": "same"}),
            _FakeResponse(200, payload={"operations": [], "nextPageToken": "same"}),
        ]
    )
    with pytest.raises(VertexBatchContractError, match="page token"):
        _gateway(session).reconcile_submission(_SUBMISSION_KEY)


def test_reconciliation_deadline_bounds_list_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter([100.0, 100.0, 281.0])
    monkeypatch.setattr(
        "onyx.regulatory.labeling.gemini_inline_batch.time.monotonic",
        lambda: next(ticks),
    )
    session = _FakeSession(
        [_FakeResponse(200, payload={"operations": [], "nextPageToken": "next"})]
    )
    with pytest.raises(IndexingGatewayTimeoutError):
        _gateway(session).reconcile_submission(_SUBMISSION_KEY)
    assert len(session.calls) == 1


def test_read_results_normalizes_success_and_partial_failure_for_existing_parser() -> (
    None
):
    first = _request("first")
    second = _request("second")
    payload = _operation(
        state="JOB_STATE_SUCCEEDED",
        response={
            "inlinedResponses": {
                "inlinedResponses": [
                    {
                        "metadata": {"key": first.request_hash},
                        "response": _candidate(),
                    },
                    {
                        "metadata": {"key": second.request_hash},
                        "error": {"code": 429, "message": "never expose this"},
                    },
                ]
            }
        },
    )
    session = _FakeSession([_FakeResponse(200, payload=payload)])

    parsed = parse_labeling_batch_output(
        _gateway(session).read_results("gemini-inline://batches/job-1"),
        {first.request_hash, second.request_hash},
    )

    assert parsed[first.request_hash].context == '{"labels":[],"abstained":true}'
    assert parsed[second.request_hash].error is not None
    assert session.calls[0][2]["stream"] is True


def test_read_results_accepts_documented_metadata_output_shape() -> None:
    request = _request()
    payload = _operation(state="JOB_STATE_SUCCEEDED")
    metadata = cast(dict[str, object], payload["metadata"])
    metadata["output"] = {
        "inlinedResponses": {
            "inlinedResponses": [
                {
                    "metadata": {"key": request.request_hash},
                    "response": _candidate(),
                }
            ]
        }
    }

    parsed = parse_labeling_batch_output(
        _gateway(_FakeSession([_FakeResponse(200, payload=payload)])).read_results(
            "gemini-inline://batches/job-1"
        ),
        {request.request_hash},
    )

    assert parsed[request.request_hash].context == '{"labels":[],"abstained":true}'


@pytest.mark.parametrize(
    "inlined",
    [
        [],
        [{"metadata": {}, "response": _candidate()}],
        [
            {
                "metadata": {"key": "a" * 64},
                "response": _candidate(),
                "error": {"code": 500},
            }
        ],
        [{"metadata": {"key": "a" * 64}}],
        [{"metadata": {"key": "a" * 64}, "response": _candidate()} for _ in range(65)],
    ],
)
def test_read_results_fails_closed_on_malformed_inline_output(
    inlined: list[object],
) -> None:
    payload = _operation(
        state="JOB_STATE_SUCCEEDED",
        response={"inlinedResponses": {"inlinedResponses": inlined}},
    )
    with pytest.raises(VertexBatchContractError):
        list(
            _gateway(_FakeSession([_FakeResponse(200, payload=payload)])).read_results(
                "gemini-inline://batches/job-1"
            )
        )


def test_existing_parser_rejects_unknown_and_duplicate_inline_keys() -> None:
    request = _request()
    unknown_key = "b" * 64
    duplicate_payload = _operation(
        state="JOB_STATE_SUCCEEDED",
        response={
            "inlinedResponses": {
                "inlinedResponses": [
                    {"metadata": {"key": unknown_key}, "response": _candidate()},
                    {"metadata": {"key": unknown_key}, "response": _candidate()},
                ]
            }
        },
    )
    gateway = _gateway(_FakeSession([_FakeResponse(200, payload=duplicate_payload)]))
    with pytest.raises(VertexBatchContractError, match="unexpected request hash"):
        parse_labeling_batch_output(
            gateway.read_results("gemini-inline://batches/job-1"),
            {request.request_hash},
        )

    known_duplicate_payload = _operation(
        state="JOB_STATE_SUCCEEDED",
        response={
            "inlinedResponses": {
                "inlinedResponses": [
                    {
                        "metadata": {"key": request.request_hash},
                        "response": _candidate(),
                    },
                    {
                        "metadata": {"key": request.request_hash},
                        "response": _candidate(),
                    },
                ]
            }
        },
    )
    gateway = _gateway(
        _FakeSession([_FakeResponse(200, payload=known_duplicate_payload)])
    )
    with pytest.raises(VertexBatchContractError, match="duplicate request hash"):
        parse_labeling_batch_output(
            gateway.read_results("gemini-inline://batches/job-1"),
            {request.request_hash},
        )


def test_read_results_stops_at_byte_limit_and_sanitizes_stream_failures() -> None:
    oversized = _FakeResponse(200, chunks=[b"x" * 8, b"y" * 8])
    with pytest.raises(VertexBatchContractError, match="size limit"):
        list(
            _gateway(_FakeSession([oversized]), max_result_bytes=10).read_results(
                "gemini-inline://batches/job-1"
            )
        )
    assert oversized.chunks_read == 2
    assert oversized.closed

    broken = _FakeResponse(
        200,
        chunks=[
            b"{}",
            requests.exceptions.ContentDecodingError("secret stream detail"),
        ],
    )
    with pytest.raises(IndexingGatewayConnectionError) as caught:
        list(
            _gateway(_FakeSession([broken])).read_results(
                "gemini-inline://batches/job-1"
            )
        )
    assert "secret" not in str(caught.value)
    assert caught.value.__context__ is None


def test_cancel_delete_and_cleanup_never_use_files_or_storage() -> None:
    session = _FakeSession(
        [_FakeResponse(204, content=b""), _FakeResponse(204, content=b"")]
    )
    gateway = _gateway(session)

    gateway.cancel("batches/job-1")
    gateway.delete("batches/job-1")
    gateway.cleanup("gemini-inline://batches/job-1")

    assert [call[0] for call in session.calls] == ["POST", "DELETE"]
    assert session.calls[0][1].endswith("/v1beta/batches/job-1:cancel")
    assert session.calls[1][1].endswith("/v1beta/batches/job-1")
    assert all(
        "files" not in call[1] and "storage" not in call[1] for call in session.calls
    )


def test_read_only_probe_checks_model_and_batch_access_without_files() -> None:
    session = _FakeSession(
        [
            _FakeResponse(200, payload=_model_response()),
            _FakeResponse(200, payload={"operations": []}),
        ]
    )

    probe = _gateway(session).probe_gemini_read_access()

    assert probe.credential_identity == "gemini-authorization-key"
    assert [call[0] for call in session.calls] == ["GET", "GET"]
    assert session.calls[0][1].endswith("/v1beta/models/gemini-3.8-flash")
    assert session.calls[1][1].endswith("/v1beta/batches")
    assert session.calls[1][2]["params"] == {
        "fields": "nextPageToken",
        "pageSize": 1,
    }
    assert all("files" not in call[1] for call in session.calls)


@pytest.mark.parametrize(
    "payload,expected_reason",
    [
        (
            {
                "error": {
                    "code": 403,
                    "message": "secret provider detail",
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                            "reason": "SERVICE_DISABLED",
                        }
                    ],
                }
            },
            "SERVICE_DISABLED",
        ),
        ({"error": {"code": 401, "status": "UNAUTHENTICATED"}}, "UNAUTHENTICATED"),
        ({"error": {"message": "secret only"}}, "HTTP_403"),
    ],
)
def test_probe_errors_expose_only_safe_status_and_reason(
    payload: object, expected_reason: str
) -> None:
    with pytest.raises(GeminiInlineBatchAccessError) as caught:
        _gateway(
            _FakeSession([_FakeResponse(403, payload=payload)])
        ).probe_gemini_read_access()

    assert caught.value.status_code == 403
    assert caught.value.reason_code == expected_reason
    assert "secret" not in str(caught.value)
    assert "test-auth-key" not in str(caught.value)
    assert caught.value.__context__ is None


def test_missing_or_failing_key_provider_is_secret_safe() -> None:
    for provider in (
        lambda: "",
        lambda: "secret-prefix\nsecret-suffix",
        lambda: (_ for _ in ()).throw(RuntimeError("secret key manager detail")),
    ):
        with pytest.raises(VertexBatchContractError) as caught:
            _gateway(
                _FakeSession([]), api_key_provider=provider
            ).probe_gemini_read_access()
        assert "secret" not in str(caught.value)
        assert caught.value.__context__ is None


def test_unexpected_requests_header_failure_is_secret_safe() -> None:
    session = _FakeSession(
        [requests.exceptions.InvalidHeader("invalid secret-auth-key header")]
    )

    with pytest.raises(IndexingGatewayConnectionError) as caught:
        _gateway(session).probe_gemini_read_access()

    assert "secret-auth-key" not in str(caught.value)
    assert caught.value.__context__ is None
