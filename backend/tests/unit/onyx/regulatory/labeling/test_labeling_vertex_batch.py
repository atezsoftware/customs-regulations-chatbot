from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace
from typing import Any, cast

import pytest
import requests
from google.auth.credentials import Credentials

from onyx.regulatory.indexing_jobs.models import (
    IndexingGatewayIndeterminateSubmissionError,
    VertexAuthenticationMode,
    VertexBatchConfig,
)
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchContractError,
    VertexBatchJobStatus,
    VertexBatchRequest,
    VertexBatchSubmissionConflictError,
)
from onyx.regulatory.labeling.provider import parse_labeling_batch_output
from onyx.regulatory.labeling.vertex_batch import LabelingVertexBatchGateway
from onyx.tracing.flows import LLMFlow

_SUBMISSION_KEY = "regulatory-labeling-" + "a" * 64
_STAGING_URI = "gs://regulatory-batch/tenant-a/labeling-stage"


def _config() -> VertexBatchConfig:
    return VertexBatchConfig(
        model_configuration_id=7,
        model_name="gemini-3.8-flash",
        project="customs-dev",
        location="global",
        authentication_mode=VertexAuthenticationMode.SERVICE_ACCOUNT_JSON,
    )


def _request(prompt: str = "Label this chunk") -> VertexBatchRequest:
    return VertexBatchRequest(
        prompt=prompt,
        system_instruction="Return only labels.",
        generation_config={
            "responseMimeType": "application/json",
            "responseJsonSchema": {
                "type": "object",
                "properties": {"labels": {"type": "array"}},
            },
            "maxOutputTokens": 32768,
            "thinkingConfig": {"thinkingLevel": "medium"},
        },
    )


def _response(text: str = '{"labels":[],"abstained":true}') -> dict[str, object]:
    return {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "parts": [{"thought": True, "text": "private"}, {"text": text}]
                },
            }
        ]
    }


class _FakeBlob:
    def __init__(self, name: str, content: str = "") -> None:
        self.name = name
        self.content = content
        self.uploaded = ""
        self.size: int | None = len(content.encode())

    def upload_from_file(
        self,
        file_obj: object,
        *,
        content_type: str,
        rewind: bool,
        timeout: float,
    ) -> None:
        assert content_type == "application/jsonl"
        assert timeout == 20
        if rewind:
            cast(Any, file_obj).seek(0)
        raw = cast(Any, file_obj).read()
        self.uploaded = raw.decode() if isinstance(raw, bytes) else cast(str, raw)
        self.size = len(self.uploaded.encode())

    def open(self, mode: str, **kwargs: object) -> BytesIO:
        assert mode == "rb"
        assert kwargs == {"timeout": 20}
        return BytesIO(self.content.encode())

    def upload_from_string(
        self, data: bytes, *, content_type: str, timeout: float
    ) -> None:
        assert content_type == "application/json"
        assert timeout == 20
        self.uploaded = data.decode()
        self.size = len(data)

    def download_as_bytes(self, *, start: int, end: int, timeout: float) -> bytes:
        assert start == 0
        assert end == 1024 * 1024
        assert timeout == 20
        data = (self.uploaded or self.content).encode()
        return data[start : end + 1]


class _FakeBucket:
    def __init__(self) -> None:
        self.blobs: dict[str, _FakeBlob] = {}
        self.deleted_batches: list[list[str]] = []

    def blob(self, name: str) -> _FakeBlob:
        return self.blobs.setdefault(name, _FakeBlob(name))

    def delete_blobs(self, blobs: list[object], *, timeout: float) -> None:
        assert timeout == 20
        self.deleted_batches.append([cast(_FakeBlob, blob).name for blob in blobs])


class _FakeStorageClient:
    def __init__(self) -> None:
        self.bucket_value = _FakeBucket()
        self.list_calls: list[dict[str, object]] = []

    def bucket(self, bucket_name: str) -> _FakeBucket:
        assert bucket_name == "regulatory-batch"
        return self.bucket_value

    def list_blobs(self, bucket_name: str, **kwargs: object) -> list[_FakeBlob]:
        assert bucket_name == "regulatory-batch"
        self.list_calls.append(kwargs)
        prefix = cast(str, kwargs["prefix"])
        max_results = cast(int, kwargs["max_results"])
        return [
            blob
            for name, blob in self.bucket_value.blobs.items()
            if name.startswith(prefix)
        ][:max_results]


class _FakeBatches:
    def __init__(self) -> None:
        self.created: dict[str, Any] | None = None
        self.create_error: Exception | None = None
        self.list_results: list[object] = []
        self.list_config: object | None = None
        self.next_page_token: str | None = None
        self.cancelled: str | None = None
        self.deleted: str | None = None
        self.get_result: object | None = None

    def create(self, **kwargs: Any) -> object:
        self.created = kwargs
        if self.create_error is not None:
            raise self.create_error
        return SimpleNamespace(
            name="projects/customs-dev/locations/global/batchPredictionJobs/1",
            display_name=kwargs["config"].display_name,
            state="JOB_STATE_PENDING",
            output_info=None,
            dest=SimpleNamespace(gcs_uri=kwargs["config"].dest),
            error=None,
        )

    def get(self, *, name: str) -> object:
        if self.get_result is not None:
            return self.get_result
        submission_root = (
            "gs://regulatory-batch/tenant-a/labeling-stage/labeling/" + "a" * 64
        )
        return SimpleNamespace(
            name=name,
            display_name=_SUBMISSION_KEY,
            state="JOB_STATE_RUNNING",
            output_info=None,
            src=SimpleNamespace(gcs_uri=[f"{submission_root}/input.jsonl"]),
            dest=SimpleNamespace(gcs_uri=f"{submission_root}/output"),
            error=None,
        )

    def list(self, *, config: object) -> object:
        self.list_config = config
        return SimpleNamespace(
            page=self.list_results,
            config={"page_token": self.next_page_token},
        )

    def cancel(self, *, name: str) -> None:
        self.cancelled = name

    def delete(self, *, name: str) -> None:
        self.deleted = name


class _FakeGenAIClient:
    def __init__(self, batches: _FakeBatches) -> None:
        self.batches = batches
        self.models = SimpleNamespace(get=lambda *, model: {"name": model})


def _gateway(
    *,
    request_timeout_seconds: float = 20,
    max_result_bytes: int = 64 * 1024 * 1024,
    max_reconciliation_seconds: float = 180,
) -> LabelingVertexBatchGateway:
    return LabelingVertexBatchGateway(
        config=_config(),
        staging_uri=_STAGING_URI,
        credential_json_provider=lambda: '{"type":"service_account"}',
        request_timeout_seconds=request_timeout_seconds,
        max_result_bytes=max_result_bytes,
        max_reconciliation_seconds=max_reconciliation_seconds,
    )


def _install_clients(
    monkeypatch: pytest.MonkeyPatch,
    gateway: LabelingVertexBatchGateway,
    storage_client: _FakeStorageClient,
    batches: _FakeBatches,
) -> None:
    credentials = cast(Credentials, object())

    @contextmanager
    def managed_storage(_credentials: Credentials) -> Iterator[_FakeStorageClient]:
        yield storage_client

    @contextmanager
    def managed_genai(_credentials: Credentials) -> Iterator[_FakeGenAIClient]:
        yield _FakeGenAIClient(batches)

    monkeypatch.setattr(gateway, "_credentials", lambda: credentials)
    monkeypatch.setattr(gateway, "_managed_storage_client", managed_storage)
    monkeypatch.setattr(gateway, "_managed_genai_client", managed_genai)


def _correlation_hash(request: VertexBatchRequest) -> str:
    payload = request.to_generate_content_request()
    canonical = {"contents": payload["contents"]}
    return sha256(
        json.dumps(
            canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode()
    ).hexdigest()


def _install_manifest(
    storage_client: _FakeStorageClient,
    requests: list[VertexBatchRequest],
) -> None:
    root = f"tenant-a/labeling-stage/labeling/{'a' * 64}"
    storage_client.bucket_value.blobs[f"{root}/manifest.json"] = _FakeBlob(
        f"{root}/manifest.json",
        json.dumps(
            {
                "version": 1,
                "submission_key": _SUBMISSION_KEY,
                "requests": [
                    {
                        "correlation_hash": _correlation_hash(request),
                        "request_hash": request.request_hash,
                    }
                    for request in requests
                ],
            }
        ),
    )


def test_submit_preserves_full_request_and_uses_labeling_owned_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _gateway()
    storage_client = _FakeStorageClient()
    batches = _FakeBatches()
    _install_clients(monkeypatch, gateway, storage_client, batches)
    traces: list[dict[str, object]] = []

    @contextmanager
    def traced(**kwargs: object) -> Iterator[None]:
        traces.append(kwargs)
        yield

    monkeypatch.setattr("onyx.regulatory.labeling.vertex_batch.traced_llm_call", traced)
    request = _request()

    state = gateway.submit(
        [request], submission_key=_SUBMISSION_KEY, max_jsonl_bytes=1024 * 1024
    )

    assert state.status is VertexBatchJobStatus.PENDING
    expected_root = f"{_STAGING_URI}/labeling/{'a' * 64}"
    assert state.input_uri == f"{expected_root}/input.jsonl"
    assert state.output_uri == f"{expected_root}/output"
    uploaded = storage_client.bucket_value.blobs[
        f"tenant-a/labeling-stage/labeling/{'a' * 64}/input.jsonl"
    ]
    assert json.loads(uploaded.uploaded) == {
        "request": request.to_generate_content_request()
    }
    manifest = storage_client.bucket_value.blobs[
        f"tenant-a/labeling-stage/labeling/{'a' * 64}/manifest.json"
    ]
    assert json.loads(manifest.uploaded) == {
        "version": 1,
        "submission_key": _SUBMISSION_KEY,
        "requests": [
            {
                "correlation_hash": _correlation_hash(request),
                "request_hash": request.request_hash,
            }
        ],
    }
    assert batches.created is not None
    assert batches.created["model"] == "gemini-3.8-flash"
    assert batches.created["src"] == state.input_uri
    assert batches.created["config"].display_name == _SUBMISSION_KEY
    assert batches.created["config"].dest == state.output_uri
    assert traces == [
        {
            "flow": LLMFlow.REGULATORY_LABELING_BATCH,
            "model": "gemini-3.8-flash",
            "provider": "vertex_ai",
            "extra_config": {"request_count": "1"},
        }
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"staging_uri": "https://bucket/prefix"},
        {"staging_uri": "gs://bucket"},
        {"staging_uri": "gs://bucket/path/../other"},
        {"request_timeout_seconds": 0},
        {"max_result_bytes": 0},
        {"max_reconciliation_seconds": float("inf")},
    ],
)
def test_constructor_rejects_unbounded_or_unscoped_configuration(
    kwargs: dict[str, object],
) -> None:
    arguments: dict[str, object] = {
        "config": _config(),
        "staging_uri": _STAGING_URI,
        "credential_json_provider": lambda: None,
    }
    arguments.update(kwargs)
    with pytest.raises(VertexBatchContractError):
        LabelingVertexBatchGateway(**cast(Any, arguments))


def test_submit_rejects_invalid_or_oversized_input_before_cloud_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _gateway()
    monkeypatch.setattr(
        gateway,
        "_credentials",
        lambda: pytest.fail("invalid input must not resolve cloud credentials"),
    )
    request = _request()

    with pytest.raises(VertexBatchContractError, match="duplicate"):
        gateway.submit(
            [request, request],
            submission_key=_SUBMISSION_KEY,
            max_jsonl_bytes=1024 * 1024,
        )
    with pytest.raises(VertexBatchContractError, match="byte limit"):
        gateway.submit([request], submission_key=_SUBMISSION_KEY, max_jsonl_bytes=10)
    with pytest.raises(VertexBatchContractError, match="submission key"):
        gateway.submit(
            [request],
            submission_key="regulatory-context-" + "a" * 64,
            max_jsonl_bytes=1024 * 1024,
        )
    different_config = request.model_copy(
        update={"generation_config": {"maxOutputTokens": 12}}
    )
    with pytest.raises(VertexBatchContractError, match="ambiguous"):
        gateway.submit(
            [request, different_config],
            submission_key=_SUBMISSION_KEY,
            max_jsonl_bytes=1024 * 1024,
        )


def test_indeterminate_create_is_reconcilable_and_reconciliation_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _gateway()
    storage_client = _FakeStorageClient()
    batches = _FakeBatches()
    batches.create_error = requests.Timeout("secret response")
    _install_clients(monkeypatch, gateway, storage_client, batches)

    with pytest.raises(IndexingGatewayIndeterminateSubmissionError) as captured:
        gateway.submit(
            [_request()],
            submission_key=_SUBMISSION_KEY,
            max_jsonl_bytes=1024 * 1024,
        )
    assert captured.value.submission_key == _SUBMISSION_KEY
    assert "secret response" not in str(captured.value)

    batches.create_error = None
    batches.list_results = []
    assert gateway.reconcile_submission(_SUBMISSION_KEY) is None
    assert cast(Any, batches.list_config).page_size == 2
    assert cast(Any, batches.list_config).filter == f'display_name="{_SUBMISSION_KEY}"'
    assert cast(Any, batches.list_config).http_options.timeout == 20_000

    expected_root = f"{_STAGING_URI}/labeling/{'a' * 64}"
    batches.list_results = [
        SimpleNamespace(
            name="projects/customs-dev/locations/global/batchPredictionJobs/1",
            display_name=_SUBMISSION_KEY,
            state="JOB_STATE_PENDING",
            output_info=None,
            dest=None,
            error=None,
        )
    ]
    reconciled = gateway.reconcile_submission(_SUBMISSION_KEY)
    assert reconciled is not None
    assert reconciled.input_uri == f"{expected_root}/input.jsonl"
    assert reconciled.output_uri == f"{expected_root}/output"

    batches.next_page_token = "another-page"
    with pytest.raises(VertexBatchSubmissionConflictError, match="matched 2"):
        gateway.reconcile_submission(_SUBMISSION_KEY)
    batches.next_page_token = None

    batches.list_results = [
        SimpleNamespace(
            name=(f"projects/customs-dev/locations/global/batchPredictionJobs/{index}"),
            display_name=_SUBMISSION_KEY,
            state="JOB_STATE_PENDING",
            output_info=None,
            dest=None,
            error=None,
        )
        for index in range(2)
    ]
    with pytest.raises(VertexBatchSubmissionConflictError, match="matched 2"):
        gateway.reconcile_submission(_SUBMISSION_KEY)


def _vertex_output_line(request: VertexBatchRequest, *, model: str | None) -> str:
    echoed_request = deepcopy(request.to_generate_content_request())
    echoed_request.pop("generationConfig", None)
    echoed_request.pop("systemInstruction", None)
    contents = cast(list[object], echoed_request["contents"])
    content = cast(dict[str, object], contents[0])
    parts = cast(list[object], content["parts"])
    cast(dict[str, object], parts[0])["fileData"] = None
    if model is not None:
        echoed_request["model"] = model
    return json.dumps({"request": echoed_request, "response": _response()}) + "\n"


def test_results_are_bounded_and_correlated_from_full_echoed_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _gateway(max_result_bytes=64 * 1024)
    storage_client = _FakeStorageClient()
    batches = _FakeBatches()
    _install_clients(monkeypatch, gateway, storage_client, batches)
    first = _request("first")
    second = _request("second")
    output_prefix = (
        f"tenant-a/labeling-stage/labeling/{'a' * 64}/output/prediction-model"
    )
    storage_client.bucket_value.blobs = {
        f"{output_prefix}/part-00002.jsonl": _FakeBlob(
            f"{output_prefix}/part-00002.jsonl",
            _vertex_output_line(
                second, model="publishers/google/models/gemini-3.8-flash"
            ),
        ),
        f"{output_prefix}/part-00001.jsonl": _FakeBlob(
            f"{output_prefix}/part-00001.jsonl",
            _vertex_output_line(first, model=None),
        ),
        f"{output_prefix}/manifest.json": _FakeBlob(
            f"{output_prefix}/manifest.json", "{}"
        ),
    }
    _install_manifest(storage_client, [first, second])

    parsed = parse_labeling_batch_output(
        gateway.read_results(
            f"gs://regulatory-batch/{output_prefix.rsplit('/prediction-model', 1)[0]}"
        ),
        {first.request_hash, second.request_hash},
    )

    assert set(parsed) == {first.request_hash, second.request_hash}
    assert all(result.error is None for result in parsed.values())
    assert storage_client.list_calls == [
        {
            "prefix": f"tenant-a/labeling-stage/labeling/{'a' * 64}/output/",
            "max_results": 1025,
            "page_size": 100,
            "timeout": 20,
        }
    ]


def test_tampered_or_duplicate_result_echo_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _gateway()
    storage_client = _FakeStorageClient()
    batches = _FakeBatches()
    _install_clients(monkeypatch, gateway, storage_client, batches)
    request = _request()
    output_prefix = f"tenant-a/labeling-stage/labeling/{'a' * 64}/output"
    tampered = deepcopy(request.to_generate_content_request())
    contents = cast(list[object], tampered["contents"])
    content = cast(dict[str, object], contents[0])
    parts = cast(list[object], content["parts"])
    cast(dict[str, object], parts[0])["text"] = "different prompt"
    storage_client.bucket_value.blobs[f"{output_prefix}/part.jsonl"] = _FakeBlob(
        f"{output_prefix}/part.jsonl",
        json.dumps({"request": tampered, "response": _response()}) + "\n",
    )
    _install_manifest(storage_client, [request])

    with pytest.raises(VertexBatchContractError, match="unexpected request hash"):
        parse_labeling_batch_output(
            gateway.read_results(f"gs://regulatory-batch/{output_prefix}"),
            {request.request_hash},
        )

    duplicate = _vertex_output_line(request, model="gemini-3.8-flash")
    storage_client.bucket_value.blobs[f"{output_prefix}/part.jsonl"].content = (
        duplicate + duplicate
    )
    with pytest.raises(VertexBatchContractError, match="duplicate request hash"):
        parse_labeling_batch_output(
            gateway.read_results(f"gs://regulatory-batch/{output_prefix}"),
            {request.request_hash},
        )


def test_result_stream_rejects_wrong_model_and_total_byte_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _gateway(max_result_bytes=64 * 1024)
    storage_client = _FakeStorageClient()
    batches = _FakeBatches()
    _install_clients(monkeypatch, gateway, storage_client, batches)
    output_prefix = f"tenant-a/labeling-stage/labeling/{'a' * 64}/output"
    path = f"{output_prefix}/part.jsonl"
    storage_client.bucket_value.blobs[path] = _FakeBlob(
        path, _vertex_output_line(_request(), model="gemini-2.5-flash")
    )
    _install_manifest(storage_client, [_request()])

    with pytest.raises(VertexBatchContractError, match="different model"):
        list(gateway.read_results(f"gs://regulatory-batch/{output_prefix}"))

    gateway = _gateway(max_result_bytes=128)
    _install_clients(monkeypatch, gateway, storage_client, batches)
    storage_client.bucket_value.blobs[path] = _FakeBlob(path, "x" * 129)
    storage_client.bucket_value.blobs[path].size = None
    with pytest.raises(VertexBatchContractError, match="size limit"):
        list(gateway.read_results(f"gs://regulatory-batch/{output_prefix}"))


def test_result_and_cleanup_paths_are_confined_to_one_labeling_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _gateway()
    storage_client = _FakeStorageClient()
    batches = _FakeBatches()
    _install_clients(monkeypatch, gateway, storage_client, batches)

    for operation in (
        lambda: list(
            gateway.read_results("gs://regulatory-batch/tenant-a/labeling-stage/other")
        ),
        lambda: list(
            gateway.read_results(
                "gs://regulatory-batch/tenant-a/labeling-stage/labeling/"
                + "a" * 64
                + "/output-sibling"
            )
        ),
        lambda: gateway.cleanup(
            "gs://regulatory-batch/tenant-a/labeling-stage/labeling"
        ),
        lambda: gateway.cleanup(
            "gs://different/tenant-a/labeling-stage/labeling/" + "a" * 64
        ),
    ):
        with pytest.raises(VertexBatchContractError, match="outside"):
            operation()

    job_root = f"tenant-a/labeling-stage/labeling/{'a' * 64}"
    for index in range(205):
        name = f"{job_root}/output/part-{index:05}.jsonl"
        storage_client.bucket_value.blobs[name] = _FakeBlob(name)
    gateway.cleanup(f"gs://regulatory-batch/{job_root}/input.jsonl")

    assert [len(batch) for batch in storage_client.bucket_value.deleted_batches] == [
        100,
        100,
        5,
    ]
    assert all(
        name.startswith(f"{job_root}/")
        for batch in storage_client.bucket_value.deleted_batches
        for name in batch
    )


def test_get_cancel_and_delete_use_managed_vertex_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _gateway()
    storage_client = _FakeStorageClient()
    batches = _FakeBatches()
    _install_clients(monkeypatch, gateway, storage_client, batches)

    remote_job_name = "projects/123456789/locations/global/batchPredictionJobs/1"
    state = gateway.get(remote_job_name)
    gateway.cancel(state.remote_job_name)
    gateway.delete(state.remote_job_name)

    assert state.status is VertexBatchJobStatus.RUNNING
    assert batches.cancelled == state.remote_job_name
    assert batches.deleted == state.remote_job_name


@pytest.mark.parametrize(
    ("display_name", "input_uri", "output_uri"),
    [
        ("regulatory-labeling-" + "b" * 64, "owned", "owned"),
        (_SUBMISSION_KEY, "gs://regulatory-batch/foreign/input.jsonl", "owned"),
        (_SUBMISSION_KEY, "owned", "gs://regulatory-batch/foreign/output"),
    ],
)
def test_cancel_and_delete_require_owned_job_metadata_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    display_name: str,
    input_uri: str,
    output_uri: str,
) -> None:
    gateway = _gateway()
    storage_client = _FakeStorageClient()
    batches = _FakeBatches()
    _install_clients(monkeypatch, gateway, storage_client, batches)
    submission_root = (
        "gs://regulatory-batch/tenant-a/labeling-stage/labeling/" + "a" * 64
    )
    batches.get_result = SimpleNamespace(
        name="projects/123456789/locations/global/batchPredictionJobs/1",
        display_name=display_name,
        state="JOB_STATE_RUNNING",
        output_info=None,
        src=SimpleNamespace(
            gcs_uri=[f"{submission_root}/input.jsonl"]
            if input_uri == "owned"
            else [input_uri]
        ),
        dest=SimpleNamespace(
            gcs_uri=f"{submission_root}/output" if output_uri == "owned" else output_uri
        ),
        error=None,
    )
    remote_job_name = "projects/123456789/locations/global/batchPredictionJobs/1"

    for operation in (gateway.cancel, gateway.delete):
        with pytest.raises(VertexBatchContractError):
            operation(remote_job_name)

    assert batches.cancelled is None
    assert batches.deleted is None


def test_read_only_probe_checks_vertex_and_owned_storage_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _gateway()
    storage_client = _FakeStorageClient()
    batches = _FakeBatches()
    _install_clients(monkeypatch, gateway, storage_client, batches)
    credentials = SimpleNamespace(service_account_email="batch@example.com")
    monkeypatch.setattr(gateway, "_credentials", lambda: credentials)

    probe = gateway.probe_gemini_read_access()

    assert probe.credential_identity == "batch@example.com"
    assert storage_client.list_calls == [
        {
            "prefix": "tenant-a/labeling-stage/labeling/",
            "max_results": 1,
            "timeout": 20,
        }
    ]
    assert batches.list_config is not None
    assert batches.created is None
    assert storage_client.bucket_value.blobs == {}


def test_foreign_reconciliation_and_remote_job_names_fail_before_adoption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _gateway()
    storage_client = _FakeStorageClient()
    batches = _FakeBatches()
    _install_clients(monkeypatch, gateway, storage_client, batches)
    batches.list_results = [
        SimpleNamespace(
            name="projects/foreign/locations/global/batchPredictionJobs/1",
            display_name=_SUBMISSION_KEY,
            state="JOB_STATE_PENDING",
            output_info=None,
            dest=None,
            error=None,
        )
    ]

    with pytest.raises(VertexBatchContractError, match="project or location"):
        gateway.reconcile_submission(_SUBMISSION_KEY)

    foreign_name = "projects/foreign/locations/global/batchPredictionJobs/1"
    for operation in (
        lambda: gateway.get(foreign_name),
        lambda: gateway.cancel(foreign_name),
        lambda: gateway.delete(foreign_name),
    ):
        with pytest.raises(VertexBatchContractError, match="project or location"):
            operation()
    assert batches.cancelled is None
    assert batches.deleted is None


def test_reconciliation_rejects_a_foreign_destination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _gateway()
    storage_client = _FakeStorageClient()
    batches = _FakeBatches()
    _install_clients(monkeypatch, gateway, storage_client, batches)
    batches.list_results = [
        SimpleNamespace(
            name="projects/customs-dev/locations/global/batchPredictionJobs/1",
            display_name=_SUBMISSION_KEY,
            state="JOB_STATE_PENDING",
            output_info=None,
            dest=SimpleNamespace(gcs_uri="gs://regulatory-batch/unowned/output"),
            error=None,
        )
    ]

    with pytest.raises(VertexBatchContractError, match="owned prefix"):
        gateway.reconcile_submission(_SUBMISSION_KEY)
