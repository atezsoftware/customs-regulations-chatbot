import json
import logging
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import pytest
from google import genai
from google.api_core import exceptions as google_exceptions
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from onyx.regulatory.indexing_jobs.models import (
    IndexingGatewayHTTPError,
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
    build_vertex_jsonl,
    parse_vertex_jsonl_output,
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

    def upload_from_string(self, data: str, *, content_type: str) -> None:
        self.uploaded = data
        self.content_type = content_type

    def download_as_text(self) -> str:
        return self._downloads[self.name]


class _FakeBucket:
    def __init__(self, downloads: dict[str, str]) -> None:
        self._downloads = downloads
        self.blobs: dict[str, _FakeBlob] = {}
        self.deleted_names: list[str] = []

    def blob(self, name: str) -> _FakeBlob:
        return self.blobs.setdefault(name, _FakeBlob(name, self._downloads))

    def delete_blobs(self, blobs: list[_FakeBlob]) -> None:
        self.deleted_names.extend(blob.name for blob in blobs)


class _FakeStorageClient:
    def __init__(self, downloads: dict[str, str] | None = None) -> None:
        self.bucket_value = _FakeBucket(downloads or {})

    def bucket(self, _bucket_name: str) -> _FakeBucket:
        return self.bucket_value

    def list_blobs(self, _bucket_name: str, *, prefix: str) -> list[_FakeBlob]:
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

    def create(self, **kwargs: Any) -> object:
        if self.error is not None:
            raise self.error
        self.created = kwargs
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
) -> GoogleVertexBatchGateway:
    return GoogleVertexBatchGateway(
        config=_vertex_config(authentication_mode),
        object_prefix="tenant-a/job-42",
        credential_json_provider=credential_provider,
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
        lambda: json.dumps({"type": "service_account", "project_id": "customs-prod"})
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


def test_read_results_combines_jsonl_blobs_in_deterministic_name_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_client = _FakeStorageClient(
        {
            "output/part-2.jsonl": "second\n",
            "output/part-1.jsonl": "first\n",
            "output/metadata.json": "ignored",
        }
    )
    _install_clients(monkeypatch, storage_client, _FakeBatches())
    gateway = _gateway(lambda: '{"type":"service_account"}')

    output = gateway.read_results("gs://customs-indexing/output")

    assert output == "first\nsecond\n"


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

    assert raised.value.__cause__ is sdk_error
    assert "secret detail" not in str(raised.value)
