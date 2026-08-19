import json
import logging
import traceback
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from google import genai
from google.api_core import exceptions as google_exceptions
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from onyx.regulatory.indexing_jobs.models import (
    IndexingGatewayHTTPError,
    IndexingGatewayIndeterminateSubmissionError,
    IndexingGatewayTimeoutError,
    VertexAuthenticationMode,
    VertexBatchConfig,
)
from onyx.regulatory.indexing_jobs.vertex_batch import (
    GoogleVertexBatchGateway,
    VertexBatchContractError,
    VertexBatchJobStatus,
    VertexBatchRequest,
    VertexBatchResultError,
    VertexBatchSubmissionConflictError,
    build_vertex_jsonl,
    parse_vertex_jsonl_output,
    vertex_batch_submission_key,
)

_FIRST_HASH = "27948fe650396b332c6e0b7073fbc4adf9cda51e33c0fc013fcd5b0be01a6f5f"
_SECOND_HASH = "5fa17eb7621a1e36adb1f59543cab32abd36396b57eac8f2482ba99d1b230e2f"


def _request_payload(prompt: str) -> dict[str, object]:
    return {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 256},
    }


def _output_line(
    prompt: str,
    *,
    text: str | None = "context",
    finish_reason: str = "STOP",
    status: str = "",
) -> str:
    response: dict[str, object]
    if text is None:
        response = {"candidates": []}
    else:
        response = {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": text}]},
                    "finishReason": finish_reason,
                }
            ]
        }
    return json.dumps(
        {
            "status": status,
            "request": {"contents": _request_payload(prompt)["contents"]},
            "response": response,
        }
    )


def _vertex_config(
    authentication_mode: VertexAuthenticationMode = (
        VertexAuthenticationMode.SERVICE_ACCOUNT_JSON
    ),
    gcs_uri: str = "gs://customs-indexing/regulatory",
) -> VertexBatchConfig:
    return VertexBatchConfig(
        model_configuration_id=73,
        model_name="gemini-3.1-flash-lite",
        project="customs-prod",
        location="europe-west4",
        authentication_mode=authentication_mode,
        gcs_uri=gcs_uri,
    )


def test_build_vertex_jsonl_has_stable_hash_and_exact_request_shape() -> None:
    first = VertexBatchRequest(prompt="first prompt")
    repeated = VertexBatchRequest(prompt="first prompt")

    encoded = build_vertex_jsonl([first])

    assert first.request_hash == _FIRST_HASH
    assert repeated.request_hash == _FIRST_HASH
    assert json.loads(encoded) == {"request": _request_payload("first prompt")}
    assert encoded.endswith("\n")


def test_build_vertex_jsonl_rejects_duplicate_request_hashes() -> None:
    with pytest.raises(VertexBatchContractError, match="duplicate request hash"):
        build_vertex_jsonl(
            [
                VertexBatchRequest(prompt="first prompt"),
                VertexBatchRequest(prompt="first prompt"),
            ]
        )


def test_submission_key_is_stable_across_request_order() -> None:
    first = VertexBatchRequest(prompt="first prompt")
    second = VertexBatchRequest(prompt="second prompt")

    assert vertex_batch_submission_key([first, second]) == (
        vertex_batch_submission_key([second, first])
    )
    assert vertex_batch_submission_key([first, second]).startswith(
        "regulatory-context-"
    )


def test_partial_parse_returns_only_correlated_available_results() -> None:
    results = parse_vertex_jsonl_output(
        _output_line("first prompt", text="first context"),
        {_FIRST_HASH, _SECOND_HASH},
        require_complete=False,
    )

    assert set(results) == {_FIRST_HASH}
    assert results[_FIRST_HASH].context == "first context"


def test_parse_correlates_shuffled_output_by_canonical_request_hash() -> None:
    output = "\n".join(
        [
            _output_line("second prompt", text="second context"),
            _output_line("first prompt", text="first context"),
        ]
    )

    results = parse_vertex_jsonl_output(output, {_FIRST_HASH, _SECOND_HASH})

    assert results[_FIRST_HASH].context == "first context"
    assert results[_SECOND_HASH].context == "second context"
    assert all(result.error is None for result in results.values())


@pytest.mark.parametrize(
    ("output", "expected_error"),
    [
        (_output_line("first prompt", text=None), VertexBatchResultError.EMPTY),
        (
            _output_line("first prompt", text="", finish_reason="SAFETY"),
            VertexBatchResultError.SAFETY,
        ),
        (
            _output_line("first prompt", status="Bad Request: hidden detail"),
            VertexBatchResultError.REMOTE_ERROR,
        ),
        (
            json.dumps(
                {
                    "status": "",
                    "request": {
                        "contents": _request_payload("first prompt")["contents"]
                    },
                    "response": {"candidates": [{"content": {"parts": [{}]}}]},
                }
            ),
            VertexBatchResultError.MALFORMED,
        ),
    ],
)
def test_parse_classifies_non_successful_outputs(
    output: str, expected_error: VertexBatchResultError
) -> None:
    result = parse_vertex_jsonl_output(output, {_FIRST_HASH})[_FIRST_HASH]

    assert result.context is None
    assert result.error is expected_error


@pytest.mark.parametrize(
    "output",
    [
        "\n".join([_output_line("first prompt"), _output_line("first prompt")]),
        _output_line("first prompt"),
        _output_line("second prompt"),
    ],
)
def test_parse_rejects_duplicate_missing_or_unexpected_hashes(output: str) -> None:
    with pytest.raises(VertexBatchContractError):
        parse_vertex_jsonl_output(output, {_FIRST_HASH, _SECOND_HASH})


def test_parse_rejects_json_without_a_correlatable_request() -> None:
    with pytest.raises(VertexBatchContractError, match="correlatable request"):
        parse_vertex_jsonl_output('{"response":{}}', {_FIRST_HASH})


def test_malformed_jsonl_failure_does_not_retain_the_full_line() -> None:
    with pytest.raises(VertexBatchContractError) as raised:
        parse_vertex_jsonl_output("LEGAL_JSONL_SENTINEL{", {_FIRST_HASH})

    assert "LEGAL_JSONL_SENTINEL" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_pure_contract_does_not_log_sensitive_payloads(
    caplog: pytest.LogCaptureFixture,
) -> None:
    prompt = "PROMPT_SENTINEL credential-json VECTOR_SENTINEL"
    request = VertexBatchRequest(prompt=prompt)

    with caplog.at_level(logging.DEBUG):
        build_vertex_jsonl([request])
        parse_vertex_jsonl_output(
            _output_line(prompt, text="LEGAL_OUTPUT_SENTINEL"),
            {request.request_hash},
        )

    assert "PROMPT_SENTINEL" not in caplog.text
    assert "credential-json" not in caplog.text
    assert "VECTOR_SENTINEL" not in caplog.text
    assert "LEGAL_OUTPUT_SENTINEL" not in caplog.text


class _FakeBlob:
    def __init__(self, name: str, downloads: dict[str, str]) -> None:
        self.name = name
        self._downloads = downloads
        self.uploaded: str | None = None
        self.content_type: str | None = None
        self.upload_timeout: float | None = None
        self.download_timeouts: list[float | None] = []

    def upload_from_string(
        self,
        data: str,
        *,
        content_type: str,
        timeout: float | None = None,
    ) -> None:
        self.uploaded = data
        self.content_type = content_type
        self.upload_timeout = timeout

    def download_as_text(self, *, timeout: float | None = None) -> str:
        self.download_timeouts.append(timeout)
        return self._downloads[self.name]


class _FakeBucket:
    def __init__(self, downloads: dict[str, str]) -> None:
        self._downloads = downloads
        self.blobs: dict[str, _FakeBlob] = {}
        self.deleted_names: list[str] = []
        self.delete_timeout: float | None = None

    def blob(self, name: str) -> _FakeBlob:
        return self.blobs.setdefault(name, _FakeBlob(name, self._downloads))

    def delete_blobs(
        self, blobs: list[_FakeBlob], *, timeout: float | None = None
    ) -> None:
        self.deleted_names.extend(blob.name for blob in blobs)
        self.delete_timeout = timeout


class _FakeStorageClient:
    def __init__(self, downloads: dict[str, str] | None = None) -> None:
        self.bucket_value = _FakeBucket(downloads or {})
        self.list_timeouts: list[float | None] = []

    def bucket(self, _bucket_name: str) -> _FakeBucket:
        return self.bucket_value

    def list_blobs(
        self,
        _bucket_name: str,
        *,
        prefix: str,
        timeout: float | None = None,
    ) -> list[_FakeBlob]:
        self.list_timeouts.append(timeout)
        return [
            self.bucket_value.blob(name)
            for name in self.bucket_value._downloads
            if name.startswith(prefix)
        ]


class _FakeBatches:
    def __init__(self) -> None:
        self.created: dict[str, Any] | None = None
        self.get_result: object = SimpleNamespace(
            name="jobs/remote-1",
            state="JOB_STATE_RUNNING",
            output_info=None,
            dest=None,
            error=None,
        )
        self.cancelled_name: str | None = None
        self.error: Exception | None = None
        self.create_error: Exception | None = None
        self.listed_config: object | None = None
        self.list_results: list[object] = []

    def create(self, **kwargs: Any) -> object:
        self.created = kwargs
        if self.create_error is not None:
            raise self.create_error
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            name="jobs/remote-1",
            state="JOB_STATE_PENDING",
            output_info=None,
            dest=SimpleNamespace(gcs_uri=kwargs["config"].dest),
            error=None,
        )

    def get(self, **_kwargs: Any) -> object:
        if self.error is not None:
            raise self.error
        return self.get_result

    def cancel(self, *, name: str) -> None:
        if self.error is not None:
            raise self.error
        self.cancelled_name = name

    def list(self, *, config: object) -> object:
        if self.error is not None:
            raise self.error
        self.listed_config = config
        return SimpleNamespace(page=self.list_results)


class _FakeGenAIClient:
    def __init__(self, batches: _FakeBatches) -> None:
        self.batches = batches


def _install_clients(
    monkeypatch: pytest.MonkeyPatch,
    storage_client: _FakeStorageClient,
    batches: _FakeBatches,
) -> list[dict[str, object]]:
    from onyx.regulatory.indexing_jobs import vertex_batch

    client_kwargs: list[dict[str, object]] = []

    def fake_storage_client(**_kwargs: object) -> _FakeStorageClient:
        return storage_client

    def fake_genai_client(**kwargs: object) -> _FakeGenAIClient:
        client_kwargs.append(kwargs)
        return _FakeGenAIClient(batches)

    def fake_service_account_credentials(
        _info: dict[str, object], *, scopes: list[str]
    ) -> object:
        assert scopes
        return object()

    monkeypatch.setattr(vertex_batch.storage, "Client", fake_storage_client)
    monkeypatch.setattr(genai, "Client", fake_genai_client)
    monkeypatch.setattr(
        vertex_batch.service_account.Credentials,
        "from_service_account_info",
        fake_service_account_credentials,
    )
    return client_kwargs


def _gateway(
    credential_provider: Callable[[], str | None],
    authentication_mode: VertexAuthenticationMode = (
        VertexAuthenticationMode.SERVICE_ACCOUNT_JSON
    ),
    request_timeout_seconds: float = 60,
) -> GoogleVertexBatchGateway:
    return GoogleVertexBatchGateway(
        config=_vertex_config(authentication_mode),
        object_prefix="tenant-a/job-42",
        credential_json_provider=credential_provider,
        request_timeout_seconds=request_timeout_seconds,
    )


def test_submit_resolves_service_account_fresh_and_uses_v1_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.regulatory.indexing_jobs import vertex_batch

    storage_client = _FakeStorageClient()
    batches = _FakeBatches()
    client_kwargs = _install_clients(monkeypatch, storage_client, batches)
    credential_infos: list[dict[str, object]] = []
    credential = object()

    def fake_from_info(info: dict[str, object], *, scopes: list[str]) -> object:
        assert scopes == ["https://www.googleapis.com/auth/cloud-platform"]
        credential_infos.append(info)
        return credential

    monkeypatch.setattr(
        vertex_batch.service_account.Credentials,
        "from_service_account_info",
        fake_from_info,
    )
    gateway = _gateway(
        lambda: json.dumps({"type": "service_account", "project_id": "customs-prod"}),
        request_timeout_seconds=12.5,
    )

    state = gateway.submit([VertexBatchRequest(prompt="first prompt")])

    assert credential_infos == [
        {"type": "service_account", "project_id": "customs-prod"}
    ]
    assert client_kwargs[0]["vertexai"] is True
    assert client_kwargs[0]["project"] == "customs-prod"
    assert client_kwargs[0]["location"] == "europe-west4"
    assert client_kwargs[0]["credentials"] is credential
    http_options = cast(genai_types.HttpOptions, client_kwargs[0]["http_options"])
    assert http_options.api_version == "v1"
    assert http_options.timeout == 12_500
    assert state.remote_job_name == "jobs/remote-1"
    assert state.status is VertexBatchJobStatus.PENDING
    assert state.input_uri is not None
    assert state.output_uri is not None
    assert state.input_uri.endswith("/input.jsonl")
    assert state.output_uri.endswith("/output")
    assert batches.created is not None
    assert batches.created["model"] == "gemini-3.1-flash-lite"
    assert batches.created["src"] == state.input_uri
    assert batches.created["config"].dest == state.output_uri
    uploaded = next(iter(storage_client.bucket_value.blobs.values()))
    assert json.loads(uploaded.uploaded or "") == {
        "request": _request_payload("first prompt")
    }
    assert uploaded.content_type == "application/jsonl"
    assert uploaded.upload_timeout == 12.5


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_gateway_rejects_a_non_positive_or_nonfinite_timeout(timeout: float) -> None:
    with pytest.raises(VertexBatchContractError):
        _gateway(lambda: None, request_timeout_seconds=timeout)


@pytest.mark.parametrize(
    "sdk_error",
    [
        httpx.ReadTimeout("CREATE_TIMEOUT_SECRET"),
        genai_errors.ServerError(503, {"message": "CREATE_SERVER_SECRET"}),
    ],
)
def test_submit_raises_secret_safe_indeterminate_error_after_ambiguous_create(
    monkeypatch: pytest.MonkeyPatch,
    sdk_error: Exception,
) -> None:
    batches = _FakeBatches()
    batches.create_error = sdk_error
    _install_clients(monkeypatch, _FakeStorageClient(), batches)
    gateway = _gateway(lambda: '{"type":"service_account"}')
    with pytest.raises(IndexingGatewayIndeterminateSubmissionError) as raised:
        gateway.submit([VertexBatchRequest(prompt="first prompt")])

    assert batches.created is not None
    submission_key = raised.value.submission_key
    assert batches.created["config"].display_name == submission_key
    assert submission_key == (
        "regulatory-context-"
        "98ee8349628a7adadfdc1b029bc3e3f5b93fc917934860a3b3bc25d79147e4dd"
    )
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    rendered = "".join(traceback.format_exception(raised.value))
    assert "CREATE_TIMEOUT_SECRET" not in rendered
    assert "CREATE_SERVER_SECRET" not in rendered


def test_reconcile_submission_uses_exact_identity_and_first_page_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches = _FakeBatches()
    batches.list_results = [
        SimpleNamespace(
            name="jobs/reconciled-1",
            state="JOB_STATE_PENDING",
            output_info=None,
            dest=None,
            error=None,
        )
    ]
    _install_clients(monkeypatch, _FakeStorageClient(), batches)
    gateway = _gateway(lambda: '{"type":"service_account"}')
    submission_key = f"regulatory-context-{'a' * 64}"

    state = gateway.reconcile_submission(submission_key)

    assert state is not None
    assert state.remote_job_name == "jobs/reconciled-1"
    listed_config = cast(genai_types.ListBatchJobsConfig, batches.listed_config)
    assert listed_config.page_size == 2
    assert listed_config.filter == f'displayName="{submission_key}"'


def test_reconcile_submission_returns_none_for_no_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches = _FakeBatches()
    _install_clients(monkeypatch, _FakeStorageClient(), batches)

    state = _gateway(lambda: '{"type":"service_account"}').reconcile_submission(
        f"regulatory-context-{'b' * 64}"
    )

    assert state is None


def test_reconcile_submission_raises_explicit_secret_safe_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches = _FakeBatches()
    batches.list_results = [object(), object()]
    _install_clients(monkeypatch, _FakeStorageClient(), batches)
    gateway = _gateway(lambda: '{"type":"service_account"}')
    submission_key = f"regulatory-context-{'c' * 64}"

    with pytest.raises(VertexBatchSubmissionConflictError) as raised:
        gateway.reconcile_submission(submission_key)

    assert raised.value.submission_key == submission_key
    assert raised.value.match_count == 2
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_workload_identity_uses_ambient_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.regulatory.indexing_jobs import vertex_batch

    storage_client = _FakeStorageClient()
    batches = _FakeBatches()
    client_kwargs = _install_clients(monkeypatch, storage_client, batches)
    ambient_credential = object()
    provider_called = False

    def fake_default(*, scopes: list[str]) -> tuple[object, str]:
        assert scopes == ["https://www.googleapis.com/auth/cloud-platform"]
        return ambient_credential, "ambient-project-must-not-override-snapshot"

    def credential_provider() -> str | None:
        nonlocal provider_called
        provider_called = True
        return "must-not-be-read"

    monkeypatch.setattr(vertex_batch.google.auth, "default", fake_default)
    gateway = _gateway(
        credential_provider,
        VertexAuthenticationMode.WORKLOAD_IDENTITY,
    )

    gateway.get("jobs/remote-1")

    assert provider_called is False
    assert client_kwargs[0]["credentials"] is ambient_credential
    assert client_kwargs[0]["project"] == "customs-prod"


@pytest.mark.parametrize("credential_json", ["CREDENTIAL_SENTINEL", "{}"])
def test_invalid_service_account_failure_does_not_retain_credential_details(
    monkeypatch: pytest.MonkeyPatch,
    credential_json: str,
) -> None:
    from onyx.regulatory.indexing_jobs import vertex_batch

    _install_clients(monkeypatch, _FakeStorageClient(), _FakeBatches())

    def reject_service_account(
        _info: dict[str, object], *, scopes: list[str]
    ) -> object:
        assert scopes
        raise ValueError(f"credential rejected: {credential_json}")

    monkeypatch.setattr(
        vertex_batch.service_account.Credentials,
        "from_service_account_info",
        reject_service_account,
    )
    gateway = _gateway(lambda: credential_json)

    with pytest.raises(VertexBatchContractError) as raised:
        gateway.get("jobs/remote-1")

    assert "CREDENTIAL_SENTINEL" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_read_results_combines_jsonl_blobs_in_deterministic_name_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_client = _FakeStorageClient(
        {
            "output/part-2.jsonl": "second\n",
            "output/part-1.jsonl": "first\n",
            "output/metadata.json": "ignored",
            "output-sibling/secret.jsonl": "SIBLING_SENTINEL\n",
        }
    )
    _install_clients(monkeypatch, storage_client, _FakeBatches())
    gateway = _gateway(lambda: '{"type":"service_account"}')

    output = gateway.read_results("gs://customs-indexing/output")

    assert output == "first\nsecond\n"
    assert storage_client.list_timeouts == [60]
    assert all(
        blob.download_timeouts == [60]
        for name, blob in storage_client.bucket_value.blobs.items()
        if name.startswith("output/") and name.endswith(".jsonl")
    )
    assert "output-sibling/secret.jsonl" not in storage_client.bucket_value.blobs


def test_get_cancel_and_cleanup_are_single_bounded_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_client = _FakeStorageClient(
        {
            "regulatory/tenant-a/job-42/input.jsonl": "input",
            "regulatory/tenant-a/job-420/input.jsonl": "keep",
            "other/file": "keep",
        }
    )
    batches = _FakeBatches()
    batches.get_result = SimpleNamespace(
        name="jobs/remote-1",
        state="JOB_STATE_SUCCEEDED",
        output_info=SimpleNamespace(gcs_output_directory="gs://bucket/output"),
        dest=None,
        error=None,
    )
    _install_clients(monkeypatch, storage_client, batches)
    gateway = _gateway(lambda: '{"type":"service_account"}')

    state = gateway.get("jobs/remote-1")
    gateway.cancel("jobs/remote-1")
    gateway.cleanup("gs://customs-indexing/regulatory/tenant-a/job-42")

    assert state.status is VertexBatchJobStatus.SUCCEEDED
    assert state.output_uri == "gs://bucket/output"
    assert batches.cancelled_name == "jobs/remote-1"
    assert storage_client.bucket_value.deleted_names == [
        "regulatory/tenant-a/job-42/input.jsonl"
    ]
    assert storage_client.list_timeouts == [60]
    assert storage_client.bucket_value.delete_timeout == 60


def test_cleanup_supports_a_bucket_root_configured_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_client = _FakeStorageClient(
        {"tenant-a/job-42/input.jsonl": "input", "other/file": "keep"}
    )
    _install_clients(monkeypatch, storage_client, _FakeBatches())
    gateway = GoogleVertexBatchGateway(
        config=_vertex_config(gcs_uri="gs://customs-indexing"),
        object_prefix="tenant-a/job-42",
        credential_json_provider=lambda: '{"type":"service_account"}',
    )

    gateway.cleanup("gs://customs-indexing/tenant-a/job-42")

    assert storage_client.bucket_value.deleted_names == ["tenant-a/job-42/input.jsonl"]


def test_get_maps_paused_job_to_terminal_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches = _FakeBatches()
    batches.get_result = SimpleNamespace(
        name="jobs/remote-1",
        state="JOB_STATE_PAUSED",
        output_info=None,
        dest=None,
        error=None,
    )
    _install_clients(monkeypatch, _FakeStorageClient(), batches)

    state = _gateway(lambda: '{"type":"service_account"}').get("jobs/remote-1")

    assert state.status is VertexBatchJobStatus.FAILED


@pytest.mark.parametrize(
    ("sdk_error", "expected_error"),
    [
        (
            google_exceptions.DeadlineExceeded("timeout detail"),
            IndexingGatewayTimeoutError,
        ),
        (
            genai_errors.ServerError(503, {"message": "secret detail"}),
            IndexingGatewayHTTPError,
        ),
    ],
)
def test_gateway_translates_typed_sdk_errors_without_leaking_details(
    monkeypatch: pytest.MonkeyPatch,
    sdk_error: Exception,
    expected_error: type[Exception],
) -> None:
    batches = _FakeBatches()
    batches.error = sdk_error
    _install_clients(monkeypatch, _FakeStorageClient(), batches)
    gateway = _gateway(lambda: '{"type":"service_account"}')

    with pytest.raises(expected_error) as raised:
        gateway.get("jobs/remote-1")

    assert raised.value.__cause__ is None
    if isinstance(raised.value, IndexingGatewayHTTPError):
        assert raised.value.status_code == 503
    rendered = "".join(traceback.format_exception(raised.value))
    assert "timeout detail" not in rendered
    assert "secret detail" not in rendered
