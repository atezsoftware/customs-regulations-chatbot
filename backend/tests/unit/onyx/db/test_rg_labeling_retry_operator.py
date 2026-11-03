"""Incident retry retains normal GET/retry semantics and never stages label runs."""

import importlib.util
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db.models import (
    RegulatoryDocumentIntake,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
)


@pytest.fixture
def operator(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    root = Path(__file__).resolve().parents[4] / "onyx/db"
    module: ModuleType | None = None
    for name in ("_new_document_rg_operator", "_rg_labeling_retry_20261102"):
        spec = importlib.util.spec_from_file_location(name, root / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    assert module is not None
    return module


@pytest.fixture
def run(operator: ModuleType) -> RegulatoryLabelingRun:
    return RegulatoryLabelingRun(
        id=operator.scope.LABELING_RUN,
        document_set_id=15,
        file_ids=[operator.scope.MAIN_FILE, operator.scope.PDF_FILE],
        status="completed_with_errors",
        stage="finished",
        total_chunks=662,
        completed_chunks=619,
        failed_chunks=43,
        stale_chunks=0,
        derived_chunks=35,
        unresolved_derived_chunks=2,
        cancel_requested=False,
        lease_token=None,
        lease_expires_at=None,
    )


def test_current_operator_cannot_mutate_before_final_sha_is_pinned(
    operator: ModuleType, run: RegulatoryLabelingRun
) -> None:
    operator.require_failed_run(run)
    assert operator.RETRY_IMAGE_SHA is None


@pytest.mark.parametrize(
    "reason,count,content",
    [
        ("STOP", 1, True),
        ("MAX_TOKENS", 1, True),
        (None, 1, True),
        ("STOP", 2, True),
        ("STOP", 0, True),
        ("STOP", 1, False),
    ],
)
def test_ack_eligibility_requires_one_complete_stop_candidate(
    operator: ModuleType, reason: str | None, count: int, content: bool
) -> None:
    body = '{"labels":[],"abstained":true}'
    candidate = {
        "finishReason": reason,
        "content": {"parts": [{"text": body if content else ""}]},
    }
    line = json.dumps(
        {"key": "a" * 64, "response": {"candidates": [candidate] * count}}
    )
    if reason == "STOP" and count == 1 and content:
        assert operator.complete_ack_context(line, "a" * 64) == body
    else:
        with pytest.raises(RuntimeError, match="complete STOP"):
            operator.complete_ack_context(line, "a" * 64)


@pytest.mark.parametrize("storage", ["postgres", "s3", "oversize"])
def test_direct_ack_reads_use_postgres_even_with_an_s3_default_store(
    operator: ModuleType, monkeypatch: pytest.MonkeyPatch, storage: str
) -> None:
    session = MagicMock(spec=Session)
    raw = b"immutable cached ACK"
    record = {
        "bucket_name": "other-bucket" if storage == "s3" else "postgres",
        "file_size": 2_000_001 if storage == "oversize" else len(raw),
        "file_metadata": {
            "binding_sha256": "b" * 64,
            "sha256": sha256(raw).hexdigest(),
        },
    }
    monkeypatch.setattr(operator.scope, "rows", lambda *_args, **_kwargs: [record])
    default_read = MagicMock(
        side_effect=AssertionError("Default S3 store cannot read direct ACKs")
    )
    monkeypatch.setattr(operator.scope, "retained_source_stream", default_read)
    store = MagicMock()
    store.read_file.return_value = BytesIO(raw)
    monkeypatch.setattr(operator, "PostgresBackedFileStore", lambda: store)
    if storage == "postgres":
        assert operator.retained_ack_bytes(session, "exact-ack", "b" * 64) == raw
        store.read_file.assert_called_once_with("exact-ack", db_session=session)
    else:
        with pytest.raises(RuntimeError, match="PostgreSQL cache"):
            operator.retained_ack_bytes(session, "exact-ack", "b" * 64)
        store.read_file.assert_not_called()
    default_read.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", uuid4()),
        ("status", "running"),
        ("stage", "projecting"),
        ("file_ids", []),
        ("completed_chunks", 618),
        ("failed_chunks", 44),
        ("stale_chunks", 1),
        ("derived_chunks", 36),
        ("unresolved_derived_chunks", 1),
        ("cancel_requested", True),
        ("lease_token", uuid4()),
    ],
)
def test_failed_run_identity_counters_and_owner_are_exact(
    operator: ModuleType, run: RegulatoryLabelingRun, field: str, value: object
) -> None:
    setattr(run, field, value)
    with pytest.raises(RuntimeError, match="terminal labeling receipt"):
        operator.require_failed_run(run)


@pytest.mark.parametrize(
    "failure", [None, "get", "deliveries", "publication", "source"]
)
def test_retry_uses_normal_get_then_retry_and_keeps_publication_unchanged(
    operator: ModuleType, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    session = MagicMock(spec=Session)
    batch = RegulatoryDocumentIntake(
        id=operator.scope.BATCH,
        status="preparing",
        stage="labeling",
        labeling_run_id=operator.scope.LABELING_RUN,
    )
    calls: list[str] = []
    monkeypatch.setattr(
        operator,
        "retry_plan",
        lambda *_: {"verified": True, "all_canonical_source_sha256": "s"},
    )
    monkeypatch.setattr(
        operator,
        "canonical_source_sha256",
        lambda *_: "changed" if failure == "source" else "s",
    )
    epochs = MagicMock(
        side_effect=[[{"epoch": 0}], [{"epoch": 1 if failure == "publication" else 0}]]
    )
    monkeypatch.setattr(operator.scope, "unpublished_epoch_scope", epochs)
    verify = MagicMock()
    monkeypatch.setattr(operator.scope, "verify_scope", verify)

    def normal_get(*_args: object) -> dict[str, object]:
        calls.append("normal_get")
        batch.status = "partial"
        return {
            "status": "partial",
            "stage": "labeling",
            "labeling_failed_count": 42 if failure == "get" else 43,
        }

    def normal_retry(*_args: object) -> list[tuple[object, object]]:
        calls.append("normal_retry")
        batch.status, batch.stage, batch.error = "preparing", "chunking", None
        return [(uuid4(), uuid4())] if failure == "deliveries" else []

    monkeypatch.setattr(operator.intake, "intake_snapshot", normal_get)
    monkeypatch.setattr(operator.intake, "retry_intake", normal_retry)
    if failure:
        with pytest.raises(RuntimeError):
            operator.request_retry(session, batch)
        assert calls == (
            ["normal_get"] if failure == "get" else ["normal_get", "normal_retry"]
        )
    else:
        result = operator.request_retry(session, batch)
        assert calls == ["normal_get", "normal_retry"]
        assert result["operator_action"] == "normal_labeling_retry_requested"
        assert (
            result["delivery"]
            == "standard_enqueue_intake_batch_without_reprepare_request"
        )
        assert result["verified"] is True
        verify.assert_called_once_with(session, batch)
    session.commit.assert_not_called()


@pytest.mark.parametrize("failure", [None, "receipt", "late", "source"])
def test_observation_reads_worker_created_plan_and_exact_copied_receipts(
    operator: ModuleType,
    run: RegulatoryLabelingRun,
    monkeypatch: pytest.MonkeyPatch,
    failure: str | None,
) -> None:
    binding: dict[str, object] = {"model": "frozen"}
    run.provider_binding, run.taxonomy_id = binding, uuid4()
    replacement = RegulatoryLabelingRun(
        id=uuid4(),
        total_chunks=651,
        derived_chunks=46,
        completed_chunks=609,
        file_ids=run.file_ids,
        provider_binding=run.provider_binding,
        taxonomy_id=run.taxonomy_id,
    )
    batch = RegulatoryDocumentIntake(labeling_run_id=replacement.id)

    def item(index: int, *, completed: bool) -> RegulatoryLabelingItem:
        return RegulatoryLabelingItem(
            regulatory_chunk_id=f"source-{index}",
            status="completed" if completed else "pending",
            source_snapshot={},
            text_snapshot=str(index),
            context_snapshot="same-context",
            context_sha256="c" * 64,
            canonical_text_sha256="d" * 64,
            request_hash="a" * 64 if completed else None,
            request_payload={"original": index} if completed else None,
            labels=["supported"] if completed else [],
            assignments=[],
        )

    old_items = [item(index, completed=index < 619) for index in range(662)]
    for old in old_items[619:]:
        old.status = "failed"
    new_items = [item(index, completed=True) for index in range(609)] + [
        item(index, completed=False) for index in range(619, 661)
    ]
    if failure == "receipt":
        new_items[0].request_hash = "changed"
    elif failure == "late":
        new_items[-1].status = "submitted"
    session = MagicMock(spec=Session)
    session.get_one.side_effect = [batch, replacement, run]
    session.scalars.side_effect = [new_items, old_items]

    @contextmanager
    def current_session() -> Iterator[Session]:
        yield session

    monkeypatch.setattr(operator, "get_session_with_current_tenant", current_session)
    monkeypatch.setattr(
        operator,
        "canonical_source_sha256",
        lambda *_: "changed" if failure == "source" else "s",
    )
    monkeypatch.setattr(operator.scope, "unpublished_epoch_scope", MagicMock())
    if failure:
        with pytest.raises(RuntimeError):
            operator.observe_replacement("s")
    else:
        result = operator.observe_replacement("s")
        assert result["observed_exact_reused_receipts"] == 609
        assert result["observed_initial_pending_targets"] == 42
        assert result["replacement_run_id"] == str(replacement.id)
        assert result["provider_dispatched_by_operator"] is False
    session.commit.assert_not_called()
