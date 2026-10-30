"""Metadata capacity waits are bounded, cancellable and never broaden retries."""

import json
from collections.abc import Callable
from threading import Event
from typing import cast
from unittest.mock import Mock

import pytest

from onyx.regulatory.document_intake import metadata, metadata_job
from onyx.regulatory.document_intake.capacity import (
    MetadataCapacityStopped,
    capacity_retry_delay_seconds,
    run_with_metadata_capacity_retry,
)
from onyx.regulatory.indexing_jobs.models import (
    IndexingGatewayConnectionError,
    IndexingGatewayHTTPError,
    IndexingGatewayTimeoutError,
)
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchGateway
from tests.unit.onyx.regulatory.test_intake_metadata_budget import _envelope
from tests.unit.onyx.regulatory.test_intake_metadata_scope_repair import _CachedGateway

_HASH = "a" * 64


def _fake_wait(
    monkeypatch: pytest.MonkeyPatch,
    event: Event,
    callback: Callable[[], None] = lambda: None,
) -> list[float]:
    waits: list[float] = []

    def wait(timeout: float) -> bool:
        waits.append(timeout)
        callback()
        return event.is_set()

    monkeypatch.setattr(event, "wait", wait)
    return waits


def test_capacity_rejection_then_ack_retries_exact_operation_with_heartbeats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = Event()
    waits = _fake_wait(monkeypatch, event)
    ack = object()
    operation = Mock(side_effect=[IndexingGatewayHTTPError(429), ack])
    heartbeat = Mock()
    assert (
        run_with_metadata_capacity_retry(
            operation, request_hash=_HASH, heartbeat=heartbeat, stopped=event
        )
        is ack
    )
    assert operation.call_count == 2
    assert sum(waits) == pytest.approx(capacity_retry_delay_seconds(_HASH, 1))
    assert all(0 < tick <= 15 for tick in waits)
    assert heartbeat.call_count == operation.call_count + len(waits)


def test_four_capacity_attempts_exhaust_with_the_original_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = Event()
    waits = _fake_wait(monkeypatch, event)
    error = IndexingGatewayHTTPError(429)
    operation = Mock(side_effect=error)
    with pytest.raises(IndexingGatewayHTTPError) as caught:
        run_with_metadata_capacity_retry(
            operation, request_hash=_HASH, heartbeat=Mock(), stopped=event
        )
    assert caught.value is error and operation.call_count == 4
    assert sum(waits) == pytest.approx(
        sum(capacity_retry_delay_seconds(_HASH, retry) for retry in (1, 2, 3))
    )
    assert len(waits) <= 14 and max(waits) <= 15


@pytest.mark.parametrize(
    "error",
    [
        IndexingGatewayHTTPError(400),
        IndexingGatewayHTTPError(401),
        IndexingGatewayHTTPError(403),
        IndexingGatewayHTTPError(408),
        IndexingGatewayHTTPError(500),
        IndexingGatewayConnectionError(),
        IndexingGatewayTimeoutError(),
        ValueError("Invalid metadata"),
    ],
)
def test_non_capacity_errors_fail_immediately_without_waiting(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    event = Event()
    waits = _fake_wait(monkeypatch, event)
    operation = Mock(side_effect=error)
    with pytest.raises(type(error)) as caught:
        run_with_metadata_capacity_retry(
            operation, request_hash=_HASH, heartbeat=Mock(), stopped=event
        )
    assert caught.value is error and operation.call_count == 1
    assert waits == []


@pytest.mark.parametrize("during_wait", [False, True])
def test_cancellation_before_or_during_capacity_wait_prevents_next_call(
    monkeypatch: pytest.MonkeyPatch, during_wait: bool
) -> None:
    event = Event()
    if not during_wait:
        event.set()
    waits = _fake_wait(monkeypatch, event, event.set)
    operation = Mock(side_effect=IndexingGatewayHTTPError(429))
    with pytest.raises(MetadataCapacityStopped):
        run_with_metadata_capacity_retry(
            operation, request_hash=_HASH, heartbeat=Mock(), stopped=event
        )
    assert operation.call_count == (1 if during_wait else 0)
    assert len(waits) == (1 if during_wait else 0)


def test_lease_failure_during_wait_prevents_the_next_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = Event()
    waits = _fake_wait(monkeypatch, event)
    error = RuntimeError("Metadata lease lost")
    heartbeat = Mock(side_effect=[None, None, error])
    operation = Mock(side_effect=IndexingGatewayHTTPError(429))
    with pytest.raises(RuntimeError) as caught:
        run_with_metadata_capacity_retry(
            operation, request_hash=_HASH, heartbeat=heartbeat, stopped=event
        )
    assert caught.value is error and operation.call_count == 1 and len(waits) == 1


def test_cancelled_heartbeat_prevents_first_provider_call() -> None:
    event = Event()
    operation = Mock()
    with pytest.raises(MetadataCapacityStopped):
        run_with_metadata_capacity_retry(
            operation, request_hash=_HASH, heartbeat=event.set, stopped=event
        )
    operation.assert_not_called()


def test_jitter_is_deterministic_bounded_and_differs_between_requests() -> None:
    first = [capacity_retry_delay_seconds(_HASH, retry) for retry in (1, 2, 3)]
    assert first == [capacity_retry_delay_seconds(_HASH, retry) for retry in (1, 2, 3)]
    assert all(
        ceiling / 2 <= delay <= ceiling for delay, ceiling in zip(first, (30, 60, 120))
    )
    assert first != [
        capacity_retry_delay_seconds("b" * 64, retry) for retry in (1, 2, 3)
    ]
    with pytest.raises(ValueError):
        capacity_retry_delay_seconds(_HASH, 4)
    with pytest.raises(ValueError):
        capacity_retry_delay_seconds("forged", 1)


@pytest.mark.parametrize("semantic_error", [False, True])
def test_metadata_request_retries_same_cache_key_and_never_retries_semantic_validation(
    monkeypatch: pytest.MonkeyPatch, semantic_error: bool
) -> None:
    sources = [
        metadata.MetadataChunk(
            chunk_id="target",
            text="MADDE 1 – Kural.",
            heading_path=["MADDE 1"],
            chunk_type="article",
        )
    ]
    request = metadata.build_metadata_request(sources, ["target"])
    fields = (
        [
            {
                "field": "article_no",
                "value": "1",
                "evidence_id": "f" * 64,
                "basis": "explicit",
                "reference_evidence_id": "",
            }
        ]
        if semantic_error
        else []
    )
    response = json.dumps(
        {"chunks": [{"chunk_id": "target", "fields": fields}], "warnings": []}
    )
    provider = Mock(
        side_effect=[
            IndexingGatewayHTTPError(429),
            _envelope(request, "STOP", response),
        ]
    )
    gateway = _CachedGateway(provider)
    submit = Mock(wraps=gateway.submit)
    monkeypatch.setattr(gateway, "submit", submit)
    event = Event()
    waits = _fake_wait(monkeypatch, event)

    def run() -> tuple[str | None, bool]:
        return metadata_job._run_metadata_request(
            gateway=cast(VertexBatchGateway, gateway),
            request=request,
            identity_prefix="sgm4-frozen",
            heartbeat=Mock(),
            stopped=event,
        )

    assert run() == (response, False)
    assert provider.call_count == 2 and waits
    if semantic_error:
        with pytest.raises(ValueError, match="not in this document catalog"):
            metadata.validate_metadata_transport_response(
                response, chunks=sources, target_ids=["target"]
            )
        assert provider.call_count == 2
    before = dict(gateway.lines)
    assert run() == (response, False)
    assert provider.call_count == 2 and gateway.lines == before
    assert submit.call_count == 3
    assert {call.kwargs["submission_key"] for call in submit.call_args_list} == {
        metadata_job._metadata_submission_key("sgm4-frozen", request)
    }
    assert all(call.args[0] == [request] for call in submit.call_args_list)
