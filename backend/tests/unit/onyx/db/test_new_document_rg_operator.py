"""Incident operations must fail closed before a standard reprepare is requested."""

import importlib.util
from io import BytesIO
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db import document_intake_reprepare as repository
from onyx.db.models import (
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeReprepare,
    RegulatoryLabelingRun,
)
from onyx.file_store import file_store
from onyx.file_store.file_store import S3BackedFileStore
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.regulatory.document_intake.models import IntakePreparationPolicy


@pytest.fixture
def operator() -> ModuleType:
    path = Path(__file__).resolve().parents[4] / "onyx/db/_new_document_rg_operator.py"
    spec = importlib.util.spec_from_file_location("rg_operator_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def batch(operator: ModuleType) -> RegulatoryDocumentIntake:
    return RegulatoryDocumentIntake(
        id=operator.BATCH,
        status="failed",
        stage="metadata",
        error="New-document metadata failed strict validation after bounded repair attempts",
        labeling_run_id=None,
        review_sha256=None,
    )


def test_reprepare_uses_reviewed_policy_and_source_guards(
    operator: ModuleType,
    batch: RegulatoryDocumentIntake,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = MagicMock(spec=Session)
    policy = IntakePreparationPolicy().model_dump(mode="json")
    documents = [operator.MAIN_DOCUMENT, operator.PDF_DOCUMENT]
    preview = {
        "expected_sha256": "a" * 64,
        "target_preparation_policy": policy,
        "documents": [{"document_id": value} for value in documents],
    }
    request = RegulatoryDocumentIntakeReprepare(
        id=uuid4(),
        status="queued",
        documents={
            value: {"target_chunking_version": "4", "target_preparation_policy": policy}
            for value in documents
        },
    )
    monkeypatch.setattr(repository, "latest_reprepare", MagicMock(return_value=None))
    monkeypatch.setattr(
        repository, "reprepare_snapshot", MagicMock(return_value=preview)
    )
    scope = MagicMock()
    monkeypatch.setattr(operator, "verify_scope", scope)
    monkeypatch.setattr(
        operator, "pdf_chunk_verification", lambda _: {"chunking_version": "3"}
    )

    def queue(*_args: object, **_kwargs: object) -> RegulatoryDocumentIntakeReprepare:
        batch.stage = "reprepare"
        return request

    queue_request = MagicMock(side_effect=queue)
    monkeypatch.setattr(repository, "request_reprepare", queue_request)
    report, request_id = operator.request_v4_reprepare(session, batch)

    queue_request.assert_called_once_with(
        session, batch, expected_sha256="a" * 64, idempotency_key=operator.REPREPARE_KEY
    )
    scope.assert_called_once_with(session, batch)
    assert request_id == request.id
    assert report["expected_review_sha256"] == "a" * 64
    assert report["publication_or_index_requested"] is False
    session.commit.assert_not_called()


@pytest.mark.parametrize("status", ["queued", "running", "prepared"])
def test_existing_request_is_inspected_without_duplicate_delivery(
    operator: ModuleType,
    batch: RegulatoryDocumentIntake,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    request = RegulatoryDocumentIntakeReprepare(
        id=uuid4(), idempotency_key=operator.REPREPARE_KEY, status=status
    )
    monkeypatch.setattr(repository, "latest_reprepare", MagicMock(return_value=request))
    queue = MagicMock()
    monkeypatch.setattr(repository, "request_reprepare", queue)
    report, delivery = operator.request_v4_reprepare(MagicMock(spec=Session), batch)
    assert report["operator_action"] == "existing_reprepare_inspected_only"
    assert delivery is None
    queue.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "preparing"),
        ("stage", "chunking"),
        ("error", "different"),
        ("review_sha256", "a" * 64),
        ("labeling_run_id", uuid4()),
    ],
)
def test_changed_incident_state_cannot_request_reprepare(
    operator: ModuleType,
    batch: RegulatoryDocumentIntake,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    setattr(batch, field, value)
    monkeypatch.setattr(repository, "latest_reprepare", MagicMock(return_value=None))
    preview = MagicMock()
    monkeypatch.setattr(repository, "reprepare_snapshot", preview)
    with pytest.raises(RuntimeError):
        operator.request_v4_reprepare(MagicMock(spec=Session), batch)
    preview.assert_not_called()


def test_canary_rejects_changed_wire_before_any_provider_call(
    operator: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "wire.txt"
    path.write_text("title: none | text: | changed source |")
    sessions = MagicMock()
    monkeypatch.setattr(operator, "get_session_with_current_tenant", sessions)
    with pytest.raises(RuntimeError, match="Fixture size or SHA changed"):
        operator.strict_v4_canary(path, "a" * 64)
    sessions.assert_not_called()


@pytest.mark.parametrize("change", ["policy", "documents", "review_sha"])
def test_changed_review_snapshot_cannot_queue(
    operator: ModuleType,
    batch: RegulatoryDocumentIntake,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    preview = {
        "expected_sha256": "a" * 64,
        "target_preparation_policy": IntakePreparationPolicy().model_dump(mode="json"),
        "documents": [
            {"document_id": operator.MAIN_DOCUMENT},
            {"document_id": operator.PDF_DOCUMENT},
        ],
    }
    if change == "policy":
        preview["target_preparation_policy"] = None
    elif change == "documents":
        preview["documents"] = [{"document_id": operator.PDF_DOCUMENT}]
    else:
        preview["expected_sha256"] = "invalid"
    monkeypatch.setattr(repository, "latest_reprepare", MagicMock(return_value=None))
    monkeypatch.setattr(
        repository, "reprepare_snapshot", MagicMock(return_value=preview)
    )
    monkeypatch.setattr(
        operator, "pdf_chunk_verification", lambda _: {"chunking_version": "3"}
    )
    queue = MagicMock()
    monkeypatch.setattr(repository, "request_reprepare", queue)
    with pytest.raises(RuntimeError):
        operator.request_v4_reprepare(MagicMock(spec=Session), batch)
    queue.assert_not_called()


@pytest.mark.parametrize("store_type", [S3BackedFileStore, PostgresBackedFileStore])
def test_retained_source_reuses_read_only_session(
    operator: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    store_type: type[S3BackedFileStore] | type[PostgresBackedFileStore],
) -> None:
    session = MagicMock(spec=Session)
    store = object.__new__(store_type)
    stream = BytesIO(b"retained source")
    read = MagicMock(return_value=stream)
    monkeypatch.setattr(store, "read_file", read)
    monkeypatch.setattr(file_store, "get_default_file_store", lambda: store)
    assert operator.retained_source_stream("retained-file", session) is stream
    read.assert_called_once_with("retained-file", db_session=session)


def test_worker_inspection_reports_only_the_exact_intake(
    operator: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    from onyx.background.celery.versioned_apps import client

    inspector = MagicMock()
    target = {"id": "task1", "kwargs": {"batch_id": str(operator.BATCH)}}
    other = {"id": "task2", "kwargs": {"batch_id": str(uuid4())}}
    inspector.active.return_value = {"regulatory_annex@worker": [target, other]}
    inspector.reserved.return_value = {}
    app = MagicMock()
    app.control.inspect.return_value = inspector
    monkeypatch.setattr(client, "app", app)
    report = operator.exact_worker_activity()
    assert [task["task_id"] for task in report["exact_intake_tasks"]] == ["task1"]
    assert report["exact_intake_tasks"][0]["state"] == "active"
    app.send_task.assert_not_called()


def test_normal_retry_preserves_the_original_frozen_request(
    operator: ModuleType,
    batch: RegulatoryDocumentIntake,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch.stage = "reprepare"
    batch.error = (
        "(psycopg2.errors.QueryCanceled) canceling statement due to user request\n"
        f"DELETE FROM regulatory_chunk WHERE user_file_id='{operator.PDF_FILE}'"
    )
    policy = IntakePreparationPolicy().model_dump(mode="json")
    request = RegulatoryDocumentIntakeReprepare(
        id=operator.UUID(operator.REPREPARE_REQUEST),
        intake_id=batch.id,
        idempotency_key=operator.REPREPARE_KEY,
        expected_sha256=operator.REPREPARE_REVIEW_SHA,
        status="failed",
        error=batch.error,
        documents={
            value: {
                "status": "prepared" if value == operator.MAIN_DOCUMENT else "pending",
                "target_chunking_version": "4",
                "target_preparation_policy": policy,
            }
            for value in [operator.MAIN_DOCUMENT, operator.PDF_DOCUMENT]
        },
    )
    monkeypatch.setattr(repository, "latest_reprepare", MagicMock(return_value=request))
    monkeypatch.setattr(
        repository,
        "reprepare_snapshot",
        MagicMock(
            return_value={
                "expected_sha256": request.expected_sha256,
                "target_preparation_policy": policy,
            }
        ),
    )
    monkeypatch.setattr(
        operator, "pdf_chunk_verification", lambda _: {"chunking_version": "3"}
    )
    monkeypatch.setattr(operator, "verify_scope", MagicMock())
    monkeypatch.setattr(
        operator,
        "validate_partial_checkpoint",
        MagicMock(return_value={"verified": True}),
    )

    def retry(*_args: object) -> list[object]:
        batch.status, batch.error = "preparing", None
        return []

    retry_call = MagicMock(side_effect=retry)
    monkeypatch.setattr(operator.repository, "retry_intake", retry_call)
    session = MagicMock(spec=Session)
    report, delivery = operator.retry_frozen_reprepare(session, batch)
    retry_call.assert_called_once_with(session, batch)
    assert delivery == request.id
    assert request.status == "failed"
    assert report["expected_review_sha256"] == operator.REPREPARE_REVIEW_SHA
    assert report["publication_or_index_requested"] is False
    session.commit.assert_not_called()


@pytest.mark.parametrize(
    "error", ["timeout", "canceling statement due to user request"]
)
def test_retry_does_not_accept_other_failure_signatures(
    operator: ModuleType,
    batch: RegulatoryDocumentIntake,
    monkeypatch: pytest.MonkeyPatch,
    error: str,
) -> None:
    batch.stage, batch.error = "reprepare", error
    request = RegulatoryDocumentIntakeReprepare(
        id=operator.UUID(operator.REPREPARE_REQUEST),
        intake_id=batch.id,
        idempotency_key=operator.REPREPARE_KEY,
        expected_sha256=operator.REPREPARE_REVIEW_SHA,
        status="failed",
        error=error,
    )
    monkeypatch.setattr(repository, "latest_reprepare", MagicMock(return_value=request))
    retry = MagicMock()
    monkeypatch.setattr(operator.repository, "retry_intake", retry)
    with pytest.raises(RuntimeError, match="cancellation signature changed"):
        operator.retry_frozen_reprepare(MagicMock(spec=Session), batch)
    retry.assert_not_called()


@pytest.mark.parametrize("drift", ["source", "checkpoint", "main_profile"])
def test_partial_resume_rejects_changed_source_or_checkpoint_before_file_reads(
    operator: ModuleType,
    batch: RegulatoryDocumentIntake,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    policy = IntakePreparationPolicy().model_dump(mode="json")
    documents = {
        operator.MAIN_DOCUMENT: {"status": "prepared", "canonical": []},
        operator.PDF_DOCUMENT: {"status": "pending", "canonical": []},
    }
    live = [{"document_id": key, "canonical": []} for key in documents]
    if drift == "source":
        live[0]["source_sha256"] = "changed"
    elif drift == "checkpoint":
        documents[operator.MAIN_DOCUMENT]["status"] = "pending"
    request = RegulatoryDocumentIntakeReprepare(documents=documents)
    monkeypatch.setattr(
        repository,
        "reprepare_snapshot",
        MagicMock(
            return_value={"target_preparation_policy": policy, "documents": live}
        ),
    )
    monkeypatch.setattr(
        operator, "pdf_chunk_verification", lambda _: {"chunking_version": "3"}
    )
    session = MagicMock(spec=Session)
    with pytest.raises(RuntimeError):
        operator.validate_partial_checkpoint(session, batch, request)
    session.get_one.assert_not_called()


@pytest.fixture
def completed_run(operator: ModuleType) -> RegulatoryLabelingRun:
    return RegulatoryLabelingRun(
        id=operator.LABELING_RUN,
        document_set_id=15,
        file_ids=[operator.MAIN_FILE, operator.PDF_FILE],
        status="completed",
        stage="finished",
        total_chunks=662,
        completed_chunks=662,
        failed_chunks=0,
        stale_chunks=0,
        derived_chunks=35,
        unresolved_derived_chunks=0,
        cancel_requested=False,
        error=None,
    )


def test_exact_completed_run_passes_refresh_guard(
    operator: ModuleType, completed_run: RegulatoryLabelingRun
) -> None:
    operator.require_completed_labeling(completed_run)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "running"),
        ("completed_chunks", 661),
        ("total_chunks", 663),
        ("failed_chunks", 1),
        ("stale_chunks", 1),
        ("unresolved_derived_chunks", 1),
        ("derived_chunks", 34),
        ("document_set_id", 16),
        ("cancel_requested", True),
        ("error", "failure"),
        ("file_ids", []),
    ],
)
def test_incomplete_or_changed_labels_block_refresh(
    operator: ModuleType,
    completed_run: RegulatoryLabelingRun,
    field: str,
    value: object,
) -> None:
    setattr(completed_run, field, value)
    with pytest.raises(RuntimeError):
        operator.require_completed_labeling(completed_run)


def test_review_refresh_calls_only_normal_snapshot_without_approval(
    operator: ModuleType,
    batch: RegulatoryDocumentIntake,
    completed_run: RegulatoryLabelingRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch.status, batch.stage, batch.error = "preparing", "labeling", None
    batch.labeling_run_id = operator.LABELING_RUN
    session = MagicMock(spec=Session)
    session.get_one.return_value = completed_run
    proof = MagicMock(return_value={"canonical_metadata_receipts": 662})
    monkeypatch.setattr(operator, "review_receipt_proof", proof)
    monkeypatch.setattr(operator, "unpublished_epoch_scope", MagicMock(return_value=[]))
    monkeypatch.setattr(operator, "verify_scope", MagicMock())
    monkeypatch.setattr(
        operator.repository, "review_digest", MagicMock(return_value="a" * 64)
    )
    approve = MagicMock()
    monkeypatch.setattr(operator.repository, "approve_intake", approve)

    def snapshot(*_args: object) -> dict[str, object]:
        batch.status, batch.stage, batch.review_sha256 = "ready", "review", "a" * 64
        return {
            "status": "ready",
            "stage": "review",
            "labeling_failed_count": 0,
            "active_document_count": 2,
            "edit_review_sha256": "b" * 64,
            "publication_review_sha256": "c" * 64,
            "documents": [
                {
                    "id": document_id,
                    "user_file_id": file_id,
                    "display_name": "source title",
                    "source_display_name": "source",
                    "source_url": "https://www.resmigazete.gov.tr/source.htm",
                    "role": role,
                    "parent_document_id": parent,
                    "status": "ready",
                    "chunk_count": count,
                    "labeling_failed_count": 0,
                    "document_links": (
                        [{"target_document_id": operator.PDF_DOCUMENT}]
                        if role == "root"
                        else []
                    ),
                }
                for document_id, file_id, role, parent, count in [
                    (operator.MAIN_DOCUMENT, operator.MAIN_FILE, "root", None, 365),
                    (
                        operator.PDF_DOCUMENT,
                        operator.PDF_FILE,
                        "annex",
                        operator.MAIN_DOCUMENT,
                        332,
                    ),
                ]
            ],
        }

    normal_get = MagicMock(side_effect=snapshot)
    monkeypatch.setattr(operator.repository, "intake_snapshot", normal_get)
    report = operator.refresh_completed_review(session, batch)
    proof.assert_called_once_with(session, batch, completed_run)
    normal_get.assert_called_once_with(session, batch)
    assert report["operator_action"] == "normal_get_review_refreshed"
    assert report["approval_or_publication_or_index_requested"] is False
    assert report["review_sha256"] == "a" * 64
    approve.assert_not_called()
    session.commit.assert_not_called()
