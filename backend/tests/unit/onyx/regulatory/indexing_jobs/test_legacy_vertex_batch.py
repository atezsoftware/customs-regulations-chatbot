from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from io import StringIO
from types import SimpleNamespace
from typing import Any, cast

import pytest
import requests
from google.auth.credentials import Credentials

from onyx.regulatory.indexing_jobs.legacy_vertex_batch import (
    GoogleVertexBatchGateway,
)
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

_SUBMISSION_KEY = "regulatory-context-" + "a" * 64


def _config() -> VertexBatchConfig:
    return VertexBatchConfig.model_validate(
        {
            "model_configuration_id": 7,
            "model_name": "gemini-3.1-flash-lite",
            "project": "customs-prod",
            "location": "europe-west4",
            "authentication_mode": VertexAuthenticationMode.SERVICE_ACCOUNT_JSON,
            "gcs_uri": "gs://legacy-regulatory/jobs",
        }
    )


def _gateway() -> GoogleVertexBatchGateway:
    return GoogleVertexBatchGateway(
        config=_config(),
        object_prefix="tenants/tenant-a/jobs/job-42",
        credential_json_provider=lambda: '{"type":"service_account"}',
    )


class _FakeBlob:
    def __init__(self, name: str, content: str = "") -> None:
        self.name = name
        self.content = content
        self.uploaded = ""

    def upload_from_file(
        self,
        file_obj: object,
        *,
        content_type: str,
        rewind: bool,
        timeout: float,
    ) -> None:
        assert content_type == "application/jsonl"
        assert timeout == 60
        if rewind:
            cast(Any, file_obj).seek(0)
        raw = cast(Any, file_obj).read()
        self.uploaded = raw.decode() if isinstance(raw, bytes) else cast(str, raw)

    def open(self, _mode: str, **_kwargs: object) -> StringIO:
        return StringIO(self.content)


class _FakeBucket:
    def __init__(self) -> None:
        self.blobs: dict[str, _FakeBlob] = {}
        self.deleted: list[str] = []

    def blob(self, name: str) -> _FakeBlob:
        return self.blobs.setdefault(name, _FakeBlob(name))

    def delete_blobs(self, blobs: list[object], *, timeout: float) -> None:
        assert timeout == 60
        self.deleted.extend(cast(_FakeBlob, blob).name for blob in blobs)


class _FakeStorageClient:
    def __init__(self) -> None:
        self.bucket_value = _FakeBucket()

    def bucket(self, _bucket_name: str) -> _FakeBucket:
        return self.bucket_value

    def list_blobs(
        self, _bucket_name: str, *, prefix: str, timeout: float
    ) -> list[_FakeBlob]:
        assert timeout == 60
        return [
            blob
            for name, blob in self.bucket_value.blobs.items()
            if name.startswith(prefix)
        ]


class _FakeBatches:
    def __init__(self) -> None:
        self.created: dict[str, Any] | None = None
        self.create_error: Exception | None = None
        self.list_results: list[object] = []
        self.list_config: object | None = None
        self.cancelled: str | None = None
        self.deleted: str | None = None

    def create(self, **kwargs: Any) -> object:
        self.created = kwargs
        if self.create_error is not None:
            raise self.create_error
        return SimpleNamespace(
            name="projects/p/locations/l/batchJobs/1",
            state="JOB_STATE_PENDING",
            output_info=None,
            dest=SimpleNamespace(gcs_uri=kwargs["config"].dest),
            error=None,
        )

    def get(self, *, name: str) -> object:
        return SimpleNamespace(
            name=name,
            state="JOB_STATE_RUNNING",
            output_info=None,
            dest=None,
            error=None,
        )

    def list(self, *, config: object) -> object:
        self.list_config = config
        return SimpleNamespace(page=self.list_results)

    def cancel(self, *, name: str) -> None:
        self.cancelled = name

    def delete(self, *, name: str) -> None:
        self.deleted = name


class _FakeGenAIClient:
    def __init__(self, batches: _FakeBatches) -> None:
        self.batches = batches


def _install_clients(
    monkeypatch: pytest.MonkeyPatch,
    gateway: GoogleVertexBatchGateway,
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


def test_legacy_gateway_resolves_stored_service_account_at_execution_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[dict[str, object], list[str]]] = []
    credential = cast(Credentials, object())

    def fake_from_info(info: dict[str, object], *, scopes: list[str]) -> Credentials:
        captured.append((info, scopes))
        return credential

    monkeypatch.setattr(
        "onyx.regulatory.indexing_jobs.legacy_vertex_batch.service_account.Credentials.from_service_account_info",
        fake_from_info,
    )

    assert _gateway()._credentials() is credential
    assert captured == [
        (
            {"type": "service_account"},
            ["https://www.googleapis.com/auth/cloud-platform"],
        )
    ]


def test_legacy_submit_uses_gcs_vertex_contract_without_gemini_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _gateway()
    storage_client = _FakeStorageClient()
    batches = _FakeBatches()
    _install_clients(monkeypatch, gateway, storage_client, batches)

    state = gateway.submit(
        [VertexBatchRequest(prompt="first prompt")],
        submission_key=_SUBMISSION_KEY,
        max_jsonl_bytes=4096,
    )

    assert state.status is VertexBatchJobStatus.PENDING
    assert state.input_uri is not None and state.input_uri.startswith("gs://")
    assert state.output_uri is not None and state.output_uri.startswith("gs://")
    uploaded = next(iter(storage_client.bucket_value.blobs.values()))
    assert json.loads(uploaded.uploaded) == {
        "request": {
            "contents": [{"role": "user", "parts": [{"text": "first prompt"}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 256},
        }
    }
    assert batches.created is not None
    assert batches.created["src"] == state.input_uri
    assert batches.created["config"].display_name == _SUBMISSION_KEY


def test_legacy_submit_preserves_indeterminate_create_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _gateway()
    batches = _FakeBatches()
    batches.create_error = requests.Timeout("secret upstream response")
    _install_clients(monkeypatch, gateway, _FakeStorageClient(), batches)

    with pytest.raises(IndexingGatewayIndeterminateSubmissionError) as caught:
        gateway.submit(
            [VertexBatchRequest(prompt="first prompt")],
            submission_key=_SUBMISSION_KEY,
            max_jsonl_bytes=4096,
        )

    assert caught.value.submission_key == _SUBMISSION_KEY
    assert "secret upstream response" not in str(caught.value)
    assert caught.value.__context__ is None


def test_legacy_reconcile_filters_exact_identity_and_rejects_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _gateway()
    batches = _FakeBatches()
    match = SimpleNamespace(
        name="projects/p/locations/l/batchJobs/1",
        state="JOB_STATE_RUNNING",
        output_info=None,
        dest=None,
        error=None,
    )
    batches.list_results = [match]
    _install_clients(monkeypatch, gateway, _FakeStorageClient(), batches)

    state = gateway.reconcile_submission(_SUBMISSION_KEY)

    assert state is not None and state.remote_job_name == match.name
    assert getattr(batches.list_config, "filter") == (
        f'displayName="{_SUBMISSION_KEY}"'
    )
    batches.list_results = [match, match]
    with pytest.raises(VertexBatchSubmissionConflictError, match="matched 2"):
        gateway.reconcile_submission(_SUBMISSION_KEY)


def test_legacy_get_read_cancel_delete_and_cleanup_are_routed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _gateway()
    storage_client = _FakeStorageClient()
    first = "jobs/tenants/tenant-a/jobs/job-42/result-2.jsonl"
    second = "jobs/tenants/tenant-a/jobs/job-42/result-1.jsonl"
    storage_client.bucket_value.blobs = {
        first: _FakeBlob(first, "second\n"),
        second: _FakeBlob(second, "first\n"),
    }
    batches = _FakeBatches()
    _install_clients(monkeypatch, gateway, storage_client, batches)
    remote_name = "projects/p/locations/l/batchJobs/1"

    state = gateway.get(remote_name)
    output = list(
        gateway.read_results("gs://legacy-regulatory/jobs/tenants/tenant-a/jobs/job-42")
    )
    gateway.cancel(remote_name)
    gateway.delete(remote_name)
    gateway.cleanup("gs://legacy-regulatory/jobs/tenants/tenant-a/jobs/job-42")

    assert state.status is VertexBatchJobStatus.RUNNING
    assert output == ["first\n", "second\n"]
    assert batches.cancelled == remote_name
    assert batches.deleted == remote_name
    assert set(storage_client.bucket_value.deleted) == {first, second}


def test_legacy_cleanup_rejects_prefix_outside_configured_gcs_base() -> None:
    with pytest.raises(VertexBatchContractError, match="outside its base"):
        _gateway().cleanup("gs://other-bucket/jobs/tenant-a")
