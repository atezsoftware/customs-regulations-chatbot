"""Busy label slots defer only the current unpublished reprepare checkpoint."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from threading import Event
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Engine, select, text
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import document_intake_queue as queue
from onyx.db import document_intake_reprepare as repository
from onyx.db.models import (
    DocumentSet,
    DocumentSet__UserFile,
    ModelConfiguration,
    RegulatoryCanonicalRevision,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryDocumentIntakeReprepare,
    RegulatoryFilePublication,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    User,
    UserFile,
)
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.regulatory.document_intake import reprepare_job
from tests.external_dependency_unit.regulatory.test_document_intake import _ENVIRONMENT
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_owner as intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_session as intake_session,
)
from tests.external_dependency_unit.regulatory.test_document_intake_markdown import (
    native_markdown_runtime as native_markdown_runtime,
)
from tests.external_dependency_unit.regulatory.test_document_intake_queue import (
    committed_queue as committed_queue,
)
from tests.external_dependency_unit.regulatory.test_document_intake_reprepare import (
    _fixture,
    _request,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)


def _active_run(
    session: Session, previous: RegulatoryLabelingRun, document_set_id: int
) -> RegulatoryLabelingRun:
    run = RegulatoryLabelingRun(
        document_set_id=document_set_id,
        taxonomy_id=previous.taxonomy_id,
        model_configuration_id=previous.model_configuration_id,
        requested_by_id=previous.requested_by_id,
        idempotency_key=uuid4(),
        model=previous.model,
        provider_binding=deepcopy(previous.provider_binding),
        file_ids=[],
        snapshot_hash="b" * 64,
        status="running",
        stage="submitting",
    )
    session.add(run)
    session.flush()
    return run


def _scope_state(session: Session, intake_id: UUID, request_id: UUID) -> str:
    members = list(
        session.scalars(
            select(RegulatoryDocumentIntakeFile).where(
                RegulatoryDocumentIntakeFile.intake_id == intake_id
            )
        )
    )
    ids = [member.user_file_id for member in members]
    records: dict[str, Any] = {}
    for model, condition in (
        (RegulatoryDocumentIntake, RegulatoryDocumentIntake.id == intake_id),
        (
            RegulatoryDocumentIntakeReprepare,
            RegulatoryDocumentIntakeReprepare.id == request_id,
        ),
        (
            RegulatoryDocumentIntakeFile,
            RegulatoryDocumentIntakeFile.intake_id == intake_id,
        ),
        (UserFile, UserFile.id.in_(ids)),
        (RegulatoryChunk, RegulatoryChunk.user_file_id.in_(ids)),
        (
            RegulatoryCanonicalRevision,
            RegulatoryCanonicalRevision.user_file_id.in_(ids),
        ),
        (RegulatoryLabelingItem, RegulatoryLabelingItem.user_file_id.in_(ids)),
        (RegulatoryFilePublication, RegulatoryFilePublication.user_file_id.in_(ids)),
        (DocumentSet__UserFile, DocumentSet__UserFile.user_file_id.in_(ids)),
    ):
        records[model.__tablename__] = sorted(
            [
                dict(row)
                for row in session.execute(
                    select(model.__table__).where(condition)
                ).mappings()
            ],
            key=str,
        )
    return json.dumps(records, default=str, sort_keys=True)


@pytest.mark.parametrize("foreign_document_set", [False, True])
def test_queued_reprepare_waits_only_for_its_document_set_and_keeps_request_identity(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
    foreign_document_set: bool,
) -> None:
    session = intake_session
    batch, previous_id = _fixture(
        session, intake_owner, native_markdown_runtime, monkeypatch
    )
    request = _request(session, batch)
    target_id = batch.document_set_id
    if foreign_document_set:
        target = DocumentSet(name=f"other-{uuid4()}", user_id=batch.created_by)
        session.add(target)
        session.flush()
        target_id = target.id
    active = _active_run(
        session, session.get_one(RegulatoryLabelingRun, previous_id), target_id
    )
    session.commit()
    frozen_documents = deepcopy(request.documents)
    before = _scope_state(session, batch.id, request.id)
    token = repository.claim_reprepare(session, batch.id, request.id)
    if foreign_document_set:
        assert token is not None
        assert request.status == "running" and request.documents == frozen_documents
        assert batch.lease_token == token
        return
    assert token is None
    assert _scope_state(session, batch.id, request.id) == before

    active.status, active.stage = "completed", "finished"
    now = datetime.now(timezone.utc)
    batch.updated_at = now - timedelta(seconds=931)
    session.commit()
    deliveries = queue.reserve_intake_preparation_deliveries(
        session, environment=_ENVIRONMENT, now=now
    )
    assert deliveries == [queue.IntakePreparationDelivery(batch.id, request.id)]
    session.commit()
    token = repository.claim_reprepare(session, batch.id, request.id)
    assert token is not None
    assert request.status == "running" and request.documents == frozen_documents
    assert repository.claim_reprepare(session, batch.id, request.id) is None
    assert batch.lease_token == token


@pytest.mark.parametrize(
    "case",
    [
        "stale_token",
        "expired",
        "batch_status",
        "stage",
        "request_status",
        "newer",
        "approved",
    ],
)
def test_busy_deferral_cannot_release_a_replaced_expired_or_reviewed_claim(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    session = intake_session
    batch, _previous_id = _fixture(
        session, intake_owner, native_markdown_runtime, monkeypatch
    )
    request = _request(session, batch)
    token = repository.claim_reprepare(session, batch.id, request.id)
    assert token is not None
    if case == "stale_token":
        token = uuid4()
    elif case == "expired":
        batch.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    elif case == "batch_status":
        batch.status = "failed"
    elif case == "stage":
        batch.stage = "metadata"
    elif case == "request_status":
        request.status = "failed"
    elif case == "newer":
        session.add(
            RegulatoryDocumentIntakeReprepare(
                intake_id=batch.id,
                idempotency_key=uuid4(),
                expected_sha256="c" * 64,
                status="queued",
                documents={},
                created_at=datetime.now(timezone.utc) + timedelta(seconds=1),
            )
        )
    else:
        intake.intake_files(session, batch)[0].approval_attempt_id = uuid4()
    batch.error, request.error = "keep batch error", "keep request error"
    session.flush()
    before = _scope_state(session, batch.id, request.id)
    assert not repository.defer_busy_reprepare(session, batch.id, request.id, token)
    assert _scope_state(session, batch.id, request.id) == before


@pytest.mark.parametrize("completed_before_busy", [0, 1])
def test_postclaim_labeling_race_defers_and_resumes_only_pending_file_checkpoints(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
    completed_before_busy: int,
) -> None:
    session = intake_session
    batch, previous_id = _fixture(
        session, intake_owner, native_markdown_runtime, monkeypatch
    )
    request = _request(session, batch)
    batch_id, request_id = batch.id, request.id
    immutable_receipts = {
        member.user_file_id: deepcopy(
            session.get_one(
                RegulatoryFilePublication, member.user_file_id
            ).original_ingestion_receipt
        )
        for member in intake.intake_files(session, batch)
    }
    original_prepare = reprepare_job.prepare_intake_file
    prepare_calls: list[UUID] = []
    active_ids: list[UUID] = []
    metadata_calls: list[UUID] = []

    def interleave(**kwargs: Any) -> None:
        prepare_calls.append(kwargs["user_file_id"])
        if len(prepare_calls) == completed_before_busy + 1:
            active = _active_run(
                session,
                session.get_one(RegulatoryLabelingRun, previous_id),
                batch.document_set_id,
            )
            active_ids.append(active.id)
            session.commit()
        original_prepare(**kwargs)

    monkeypatch.setattr(reprepare_job, "prepare_intake_file", interleave)
    monkeypatch.setattr(
        reprepare_job,
        "run_intake_metadata",
        lambda **kwargs: metadata_calls.append(kwargs["intake_id"]),
    )
    assert (
        reprepare_job.run_document_intake_reprepare(
            intake_id=batch_id, request_id=request_id, tenant_id="public"
        )
        is None
    )
    session.expire_all()
    batch = session.get_one(RegulatoryDocumentIntake, batch_id)
    request = session.get_one(RegulatoryDocumentIntakeReprepare, request_id)
    assert (batch.status, batch.stage, batch.error) == ("preparing", "reprepare", None)
    assert batch.lease_token is None and batch.lease_expires_at is None
    assert request.status == "queued" and request.error is None
    assert metadata_calls == []
    done = {
        key: deepcopy(entry)
        for key, entry in request.documents.items()
        if entry["status"] == "prepared"
    }
    assert len(done) == completed_before_busy
    pending_ids = {
        UUID(entry["user_file_id"])
        for entry in request.documents.values()
        if entry["status"] == "pending"
    }
    for entry in request.documents.values():
        if entry["status"] == "pending":
            assert [
                row.model_dump(mode="json")
                for row in repository.capture_canonical_scope(
                    session, UUID(entry["user_file_id"])
                )
            ] == entry["canonical"]
    assert {
        file_id: session.get_one(
            RegulatoryFilePublication, file_id
        ).original_ingestion_receipt
        for file_id in immutable_receipts
    } == immutable_receipts

    active = session.get_one(RegulatoryLabelingRun, active_ids[0])
    active.status, active.stage = "completed", "finished"
    batch.updated_at = datetime.now(timezone.utc) - timedelta(seconds=931)
    session.commit()
    assert queue.reserve_intake_preparation_deliveries(
        session, environment=_ENVIRONMENT
    ) == [queue.IntakePreparationDelivery(batch_id, request_id)]
    session.commit()
    resumed_calls: list[UUID] = []

    def resume(**kwargs: Any) -> None:
        resumed_calls.append(kwargs["user_file_id"])
        original_prepare(**kwargs)

    monkeypatch.setattr(reprepare_job, "prepare_intake_file", resume)
    run_id = reprepare_job.run_document_intake_reprepare(
        intake_id=batch_id, request_id=request_id, tenant_id="public"
    )
    assert run_id is not None and run_id != previous_id
    assert set(resumed_calls) == pending_ids and len(resumed_calls) == len(pending_ids)
    assert metadata_calls == [batch_id]
    session.expire_all()
    request = session.get_one(RegulatoryDocumentIntakeReprepare, request_id)
    assert request.status == "prepared"
    assert all(entry["status"] == "prepared" for entry in request.documents.values())
    assert all(request.documents[key] == entry for key, entry in done.items())
    assert {
        file_id: session.get_one(
            RegulatoryFilePublication, file_id
        ).original_ingestion_receipt
        for file_id in immutable_receipts
    } == immutable_receipts


@pytest.mark.parametrize("locked_row", ["batch", "request"])
def test_claim_skips_locked_rows_without_deadlocking_a_label_admission_lock_order(
    committed_queue: tuple[Engine, str, list[UUID]],
    locked_row: str,
) -> None:
    engine, _environment, ids = committed_queue
    with Session(engine) as setup:
        batch = setup.get_one(RegulatoryDocumentIntake, ids[0])
        batch.stage = "reprepare"
        request = RegulatoryDocumentIntakeReprepare(
            intake_id=batch.id,
            idempotency_key=uuid4(),
            expected_sha256="d" * 64,
            status="queued",
            documents={},
        )
        setup.add(request)
        setup.commit()
        request_id, target_id = request.id, batch.document_set_id
    locked, attempt_document_set = Event(), Event()
    owner_pids: list[int] = []

    def label_admission() -> None:
        with Session(engine) as owner:
            owner.execute(text("SET LOCAL lock_timeout = '4s'"))
            owner_pids.append(
                owner.execute(text("SELECT pg_backend_pid()")).scalar_one()
            )
            if locked_row == "batch":
                owner.get_one(RegulatoryDocumentIntake, ids[0], with_for_update=True)
            else:
                owner.get_one(
                    RegulatoryDocumentIntakeReprepare, request_id, with_for_update=True
                )
            locked.set()
            assert attempt_document_set.wait(timeout=5)
            owner.get_one(DocumentSet, target_id, with_for_update=True)
            owner.commit()

    with ThreadPoolExecutor(max_workers=1) as pool:
        waiter = pool.submit(label_admission)
        assert locked.wait(timeout=5)
        with Session(engine) as claimant:
            claimant.execute(text("SET LOCAL lock_timeout = '4s'"))
            claimant.get_one(DocumentSet, target_id, with_for_update=True)
            attempt_document_set.set()
            deadline = time.monotonic() + 3
            while not claimant.execute(
                text("SELECT cardinality(pg_blocking_pids(:pid)) > 0"),
                {"pid": owner_pids[0]},
            ).scalar_one():
                assert time.monotonic() < deadline, (
                    "Owner never contended on the document-set lock"
                )
                time.sleep(0.01)
            assert repository.claim_reprepare(claimant, ids[0], request_id) is None
            claimant.commit()
        waiter.result(timeout=5)
    with Session(engine) as resumed:
        token = repository.claim_reprepare(resumed, ids[0], request_id)
        assert token is not None
        resumed.commit()
        assert resumed.get_one(RegulatoryDocumentIntake, ids[0]).lease_token == token
