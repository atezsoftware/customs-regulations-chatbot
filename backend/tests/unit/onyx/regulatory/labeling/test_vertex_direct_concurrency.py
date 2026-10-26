"""Opt-in direct concurrency retains durable receipts and the serial default."""

import json
from collections.abc import Callable
from contextvars import ContextVar
from threading import Barrier, Event, Lock, get_ident
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from google.auth.credentials import Credentials
from google.genai import errors as genai_errors

from onyx.file_store.file_store import FileStore
from onyx.regulatory.indexing_jobs.models import (
    IndexingGatewayHTTPError,
    VertexAuthenticationMode,
    VertexBatchConfig,
)
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchContractError
from onyx.regulatory.labeling.vertex_direct import LabelingVertexDirectGateway
from tests.unit.onyx.regulatory.labeling.test_vertex_direct import (
    _BINDING,
    _KEY,
    _request,
    _response,
    _Store,
)


def _gateway(
    store: _Store, generate: Callable[..., Any], **kwargs: Any
) -> tuple[LabelingVertexDirectGateway, list[MagicMock]]:
    clients: list[MagicMock] = []
    lock = Lock()

    def factory(_credentials: Credentials) -> MagicMock:
        client = MagicMock()
        client.models.generate_content.side_effect = generate
        with lock:
            clients.append(client)
        return client

    gateway = LabelingVertexDirectGateway(
        config=VertexBatchConfig(
            model_configuration_id=7,
            model_name="gemini-3.5-flash",
            project="test",
            location="global",
            authentication_mode=VertexAuthenticationMode.SERVICE_ACCOUNT_JSON,
        ),
        binding_sha256=_BINDING,
        credential_identity="test@example.com",
        credential_json_provider=lambda: None,
        credentials_factory=lambda: cast(Credentials, object()),
        client_factory=factory,
        store=cast(FileStore, store),
        **kwargs,
    )
    return gateway, clients


def test_default_remains_serial_with_one_sdk_client_and_unchanged_options() -> None:
    gateway, clients = _gateway(_Store(), lambda **_kwargs: _response())
    requests = [_request(f"Source {i}") for i in range(8)]
    gateway.submit(requests, submission_key=_KEY, max_jsonl_bytes=8 * 1024 * 1024)
    assert len(clients) == 1
    assert clients[0].models.generate_content.call_count == 8
    clients[0].close.assert_called_once()
    assert gateway._timeout == 120


def test_four_requests_use_distinct_clients_fresh_context_and_ordered_cached_results() -> (
    None
):
    tenant = ContextVar("direct_test_tenant", default="missing")
    barrier, lock = Barrier(4), Lock()
    active = maximum = 0
    seen: list[str] = []
    parent = get_ident()
    progress_threads: list[int] = []

    def generate(**kwargs: Any) -> Any:
        nonlocal active, maximum
        text = json.loads(kwargs["contents"][0].parts[0].text)["target"]["text"]
        with lock:
            active += 1
            maximum = max(maximum, active)
            seen.append(tenant.get())
        tenant.set("worker-local")
        if text in {f"Source {i}" for i in range(4)}:
            barrier.wait(timeout=5)
        with lock:
            active -= 1
        return _response()

    store = _Store()
    gateway, clients = _gateway(store, generate, max_concurrency=4)
    requests = [_request(f"Source {i}") for i in range(9)]
    token = tenant.set("tenant-and-trace")
    try:
        state = gateway.submit_stream(
            requests,
            submission_key=_KEY,
            max_jsonl_bytes=8 * 1024 * 1024,
            on_progress=lambda: progress_threads.append(get_ident()),
        )
        assert tenant.get() == "tenant-and-trace"
    finally:
        tenant.reset(token)
    assert maximum == 4 and len(clients) == 9
    assert seen == ["tenant-and-trace"] * 9
    assert progress_threads and set(progress_threads) == {parent}
    for client in clients:
        client.models.generate_content.assert_called_once()
        client.close.assert_called_once()
        config = client.models.generate_content.call_args.kwargs["config"]
        assert config.max_output_tokens == 32768
        assert config.thinking_config.thinking_level.value.lower() == "medium"
    assert state.output_uri
    assert [
        json.loads(line)["key"] for line in gateway.read_results(state.output_uri)
    ] == sorted(request.request_hash for request in requests)
    original_cache = dict(store.files)
    assert (
        gateway.submit(requests, submission_key=_KEY, max_jsonl_bytes=8 * 1024 * 1024)
        == state
    )
    assert len(clients) == 9 and store.files == original_cache


def test_failure_stops_new_dispatch_waits_for_acks_and_retry_reuses_successes() -> None:
    barrier, failed = Barrier(4), Event()
    first_attempt = True

    def generate(**kwargs: Any) -> Any:
        text = json.loads(kwargs["contents"][0].parts[0].text)["target"]["text"]
        if first_attempt:
            barrier.wait(timeout=5)
            if text == "Source 0":
                failed.set()
                raise genai_errors.ServerError(503, {"error": {"message": "private"}})
            assert failed.wait(timeout=5)
        return _response()

    store = _Store()
    gateway, clients = _gateway(store, generate, max_concurrency=4)
    requests = [_request(f"Source {i}") for i in range(10)]
    with pytest.raises(IndexingGatewayHTTPError) as failure:
        gateway.submit(requests, submission_key=_KEY, max_jsonl_bytes=8 * 1024 * 1024)
    assert failure.value.status_code == 503
    assert len(clients) == 4
    acknowledgments = {
        key: value
        for key, value in store.files.items()
        if not key.endswith("manifest.json")
    }
    assert len(acknowledgments) == 3
    assert gateway.reconcile_submission(_KEY) is None
    first_attempt = False
    gateway.submit(requests, submission_key=_KEY, max_jsonl_bytes=8 * 1024 * 1024)
    assert len(clients) == 11
    assert all(store.files[key] == value for key, value in acknowledgments.items())


def test_parent_cancellation_waits_for_active_acknowledgments() -> None:
    barrier, release = Barrier(4), Event()

    def generate(**_kwargs: Any) -> Any:
        barrier.wait(timeout=5)
        release.set()
        return _response()

    def progress() -> None:
        if release.is_set():
            raise RuntimeError("Cancelled locally")

    store = _Store()
    gateway, clients = _gateway(store, generate, max_concurrency=4)
    with pytest.raises(RuntimeError, match="Cancelled locally"):
        gateway.submit_stream(
            [_request(f"Source {i}") for i in range(10)],
            submission_key=_KEY,
            max_jsonl_bytes=8 * 1024 * 1024,
            on_progress=progress,
        )
    assert len(clients) == 4
    assert len([key for key in store.files if not key.endswith("manifest.json")]) == 4
    assert gateway.reconcile_submission(_KEY) is None
    gateway.cancel(f"vertex-direct/{_BINDING}/" + "b" * 64)
    before = dict(store.files)
    gateway.submit(
        [_request(f"Source {i}") for i in range(10)],
        submission_key=_KEY,
        max_jsonl_bytes=8 * 1024 * 1024,
    )
    assert len(clients) == 4 and store.files == before


def test_parallel_total_byte_budget_rejects_before_manifest_completion_and_on_resume() -> (
    None
):
    store = _Store()
    gateway, clients = _gateway(
        store, lambda **_kwargs: _response(), max_concurrency=4, max_result_bytes=1
    )
    requests = [_request(f"Source {i}") for i in range(8)]
    with pytest.raises(VertexBatchContractError, match="complete result budget"):
        gateway.submit(requests, submission_key=_KEY, max_jsonl_bytes=8 * 1024 * 1024)
    assert 1 <= len(clients) <= 4
    assert gateway.reconcile_submission(_KEY) is None
    count = len(clients)
    with pytest.raises(VertexBatchContractError, match="complete result budget"):
        gateway.submit(requests, submission_key=_KEY, max_jsonl_bytes=8 * 1024 * 1024)
    assert len(clients) == count


@pytest.mark.parametrize("concurrency", [0, 5, True, 1.5])
def test_invalid_concurrency_rejects_before_provider_calls(concurrency: Any) -> None:
    with pytest.raises(VertexBatchContractError, match="concurrency"):
        _gateway(_Store(), lambda **_kwargs: _response(), max_concurrency=concurrency)
