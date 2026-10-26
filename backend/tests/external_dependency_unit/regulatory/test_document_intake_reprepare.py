"""Reprepare replaces only unpublished intake rows and retains their audit history."""

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import document_intake_metadata
from onyx.db import document_intake_reprepare as repository
from onyx.db.document_intake_sources import create_markdown_source_package
from onyx.db.models import (
    DocumentSet,
    DocumentSet__UserFile,
    ModelConfiguration,
    RegulatoryCanonicalRevision,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeReprepare,
    RegulatoryFilePublication,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryLabelingShard,
    User,
    UserFile,
)
from onyx.file_processing.original_ingestion import LoadedUserFileDocuments
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.regulatory import writer_publication
from onyx.regulatory.amendments.annexes import job as source_job
from onyx.regulatory.document_intake import job as intake_job
from onyx.regulatory.document_intake import reprepare_job
from tests.external_dependency_unit.regulatory.test_document_intake import _ENVIRONMENT
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_owner as intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_session as intake_session,
)
from tests.external_dependency_unit.regulatory.test_document_intake_markdown import (
    _markdown_input,
)
from tests.external_dependency_unit.regulatory.test_document_intake_markdown import (
    native_markdown_runtime as native_markdown_runtime,
)


def _fixture(
    session: Session,
    owner: tuple[User, DocumentSet, ModelConfiguration],
    store: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
    *,
    legacy_chunks: bool = True,
) -> tuple[RegulatoryDocumentIntake, UUID]:
    user, _target, model = owner
    display_name, content, _originals = _markdown_input(True, True)
    package, _ = create_markdown_source_package(
        session,
        user=user,
        environment=_ENVIRONMENT,
        idempotency_key=uuid4(),
        display_name=display_name,
        content=content,
        store=store,
    )
    package_id = package.id
    session.commit()
    source_job.run_source_package(package_id=package_id, environment=_ENVIRONMENT)
    batch, _ = intake.create_intake(
        session,
        source_package_id=package_id,
        idempotency_key=uuid4(),
        labeling_model_configuration_id=model.id,
        user=user,
        environment=_ENVIRONMENT,
    )
    batch_id, model_id = batch.id, model.id
    session.commit()
    from types import SimpleNamespace

    monkeypatch.setattr(
        intake,
        "resolve_intake_labeling_provider_binding",
        lambda *_args, **_kwargs: SimpleNamespace(
            model="gemini-3.5-flash",
            model_dump=lambda **_options: {"model_configuration_id": model_id},
        ),
    )
    monkeypatch.setattr(
        "onyx.regulatory.document_intake.metadata_job.run_intake_metadata",
        lambda **_kwargs: None,
    )
    original = writer_publication.chunk_owned_file

    def legacy(file_id: UUID, tenant_id: str, documents: list[Any]) -> bool:
        assert isinstance(documents, LoadedUserFileDocuments)
        copies = []
        for doc in documents:
            copied = doc.model_copy(deep=True)
            copied.doc_metadata["regulatory_document_intake"].pop(
                "chunking_version", None
            )
            copies.append(copied)
        return original(
            file_id,
            tenant_id,
            LoadedUserFileDocuments(
                copies,
                file_id=documents.original_extraction.file_id,
                raw_sha256=documents.original_extraction.raw_sha256,
            ),
        )

    with monkeypatch.context() as initial:
        if legacy_chunks:
            initial.setattr(writer_publication, "chunk_owned_file", legacy)
        run_id = intake_job.run_document_intake(intake_id=batch_id, tenant_id="public")
    assert run_id is not None
    session.expire_all()
    batch = session.get_one(RegulatoryDocumentIntake, batch_id)
    run = session.get_one(RegulatoryLabelingRun, run_id)
    run.status, run.stage = "failed", "finished"
    batch.status, batch.stage = "partial", "labeling"
    session.commit()
    connection = session.connection()

    @contextmanager
    def shared_session(**_kwargs: object) -> Iterator[Session]:
        with Session(
            connection, join_transaction_mode="create_savepoint", expire_on_commit=False
        ) as shared:
            yield shared

    monkeypatch.setattr(repository, "get_session_with_tenant", shared_session)
    monkeypatch.setattr(
        reprepare_job, "get_session_with_current_tenant", shared_session
    )
    monkeypatch.setattr(reprepare_job, "get_default_file_store", lambda: store)
    monkeypatch.setattr(reprepare_job, "run_intake_metadata", lambda **_kwargs: None)
    return batch, run_id


def _request(
    session: Session, batch: RegulatoryDocumentIntake
) -> RegulatoryDocumentIntakeReprepare:
    view = repository.reprepare_snapshot(session, batch)
    request = repository.request_reprepare(
        session, batch, expected_sha256=view["expected_sha256"], idempotency_key=uuid4()
    )
    session.commit()
    return request


def _mark_ready(
    session: Session, batch: RegulatoryDocumentIntake, run_id: UUID
) -> None:
    run = session.get_one(RegulatoryLabelingRun, run_id)
    run.status = "completed"
    for item in session.scalars(
        select(RegulatoryLabelingItem).where(RegulatoryLabelingItem.run_id == run_id)
    ):
        item.status, item.labels = "completed", ["Gümrük"]
    batch.status, batch.stage = "ready", "review"
    for member in intake.intake_files(session, batch):
        member.status = "ready"
    session.flush()
    batch.review_sha256 = intake.review_digest(session, batch)
    session.commit()


@pytest.mark.parametrize("status", ["partial", "ready"])
def test_reprepare_same_files_retains_old_labels_receipts_and_canonical_revisions(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    session = intake_session
    batch, old_run_id = _fixture(
        session, intake_owner, native_markdown_runtime, monkeypatch
    )
    if status == "ready":
        _mark_ready(session, batch, old_run_id)
    previous_review_sha256 = batch.review_sha256
    files = intake.intake_files(session, batch)
    membership = list(session.execute(select(DocumentSet__UserFile.__table__)))
    all_ids = set(session.scalars(select(UserFile.id)))
    graph = [
        (
            m.id,
            m.user_file_id,
            m.source_asset_id,
            m.parent_document_id,
            deepcopy(m.relationships),
        )
        for m in files
    ]
    old_labels = list(
        session.execute(
            select(RegulatoryLabelingItem.__table__).where(
                RegulatoryLabelingItem.run_id == old_run_id
            )
        )
    )
    immutable_receipts = {
        m.user_file_id: deepcopy(
            session.get_one(
                RegulatoryFilePublication, m.user_file_id
            ).original_ingestion_receipt
        )
        for m in files
    }
    request = _request(session, batch)
    request_id, batch_id = request.id, batch.id
    assert request.previous_labeling_run_id == old_run_id
    assert batch.labeling_run_id is None
    assert all(entry["before_revision_ids"] for entry in request.documents.values())
    assert all(
        entry["previous_review_sha256"] == previous_review_sha256
        for entry in request.documents.values()
    )
    new_run_id = reprepare_job.run_document_intake_reprepare(
        intake_id=batch_id, request_id=request_id, tenant_id="public"
    )
    assert new_run_id is not None and new_run_id != old_run_id
    session.expire_all()
    request = session.get_one(RegulatoryDocumentIntakeReprepare, request_id)
    assert request.status == "prepared"
    assert {
        m.user_file_id: session.get_one(
            RegulatoryFilePublication, m.user_file_id
        ).original_ingestion_receipt
        for m in files
    } == immutable_receipts
    assert all(
        entry["status"] == "prepared" and entry["after_revision_ids"]
        for entry in request.documents.values()
    )
    for entry in request.documents.values():
        assert (
            entry["source_receipt"]["raw_sha256"]
            == entry["after_receipt"]["raw_sha256"]
        )
        assert (
            entry["source_receipt"]["plaintext_sha256"]
            == entry["after_receipt"]["plaintext_sha256"]
        )
        assert (
            entry["source_receipt"]["generation_hash"]
            != entry["after_receipt"]["generation_hash"]
        )
        for revision_id in entry["before_revision_ids"] + entry["after_revision_ids"]:
            assert (
                session.get(RegulatoryCanonicalRevision, UUID(revision_id)) is not None
            )
    assert set(session.scalars(select(UserFile.id))) == all_ids
    assert list(session.execute(select(DocumentSet__UserFile.__table__))) == membership
    assert [
        (m.id, m.user_file_id, m.source_asset_id, m.parent_document_id, m.relationships)
        for m in intake.intake_files(
            session, session.get_one(RegulatoryDocumentIntake, batch_id)
        )
    ] == graph
    assert (
        list(
            session.execute(
                select(RegulatoryLabelingItem.__table__).where(
                    RegulatoryLabelingItem.run_id == old_run_id
                )
            )
        )
        == old_labels
    )
    rows = list(
        session.scalars(
            select(RegulatoryChunk).where(
                RegulatoryChunk.user_file_id.in_([m.user_file_id for m in files])
            )
        )
    )
    assert all(row.chunk_metadata["intake_chunking_version"] == "1" for row in rows)
    assert all(not (row.text.startswith("##") and "\n" not in row.text) for row in rows)
    assert (
        reprepare_job.run_document_intake_reprepare(
            intake_id=batch_id, request_id=request_id, tenant_id="public"
        )
        is None
    )


def test_failed_file_reprepare_rolls_back_atomically_and_retry_reuses_prepared_files(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.regulatory import indexing

    session = intake_session
    batch, _old_run_id = _fixture(
        session, intake_owner, native_markdown_runtime, monkeypatch
    )
    request = _request(session, batch)
    batch_id, request_id = batch.id, request.id
    original = indexing.documents_to_regulatory_chunks
    calls = []

    def fail_second(*args: Any, **kwargs: Any) -> Any:
        calls.append(args[0][0].id)
        if len(calls) == 2:
            raise RuntimeError("synthetic rechunk failure")
        return original(*args, **kwargs)

    with monkeypatch.context() as failing:
        failing.setattr(indexing, "documents_to_regulatory_chunks", fail_second)
        with pytest.raises(RuntimeError, match="synthetic rechunk"):
            reprepare_job.run_document_intake_reprepare(
                intake_id=batch_id, request_id=request_id, tenant_id="public"
            )
    session.expire_all()
    request = session.get_one(RegulatoryDocumentIntakeReprepare, request_id)
    assert request.status == "failed"
    done = {
        key: deepcopy(value)
        for key, value in request.documents.items()
        if value["status"] == "prepared"
    }
    assert len(done) == 1
    for entry in request.documents.values():
        if entry["status"] == "pending":
            current = repository.capture_canonical_scope(
                session, UUID(entry["user_file_id"])
            )
            assert [row.model_dump(mode="json") for row in current] == entry[
                "canonical"
            ]
    batch = session.get_one(RegulatoryDocumentIntake, batch_id)
    intake.retry_intake(session, batch)
    session.commit()
    assert (
        reprepare_job.run_document_intake_reprepare(
            intake_id=batch_id, request_id=request_id, tenant_id="public"
        )
        is not None
    )
    session.expire_all()
    request = session.get_one(RegulatoryDocumentIntakeReprepare, request_id)
    assert all(request.documents[key] == value for key, value in done.items())


def test_metadata_failure_after_reprepare_retries_without_rechunking(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = intake_session
    batch, _old_run_id = _fixture(
        session, intake_owner, native_markdown_runtime, monkeypatch
    )
    request = _request(session, batch)
    batch_id, request_id = batch.id, request.id

    def metadata_failure(*, intake_id: UUID, lease_token: UUID, **_kwargs: Any) -> None:
        with reprepare_job.get_session_with_current_tenant() as shared:
            document_intake_metadata.load_metadata_work(shared, intake_id, lease_token)
            shared.commit()
        raise RuntimeError("synthetic metadata failure")

    monkeypatch.setattr(reprepare_job, "run_intake_metadata", metadata_failure)
    with pytest.raises(RuntimeError, match="synthetic metadata"):
        reprepare_job.run_document_intake_reprepare(
            intake_id=batch_id, request_id=request_id, tenant_id="public"
        )
    session.expire_all()
    request = session.get_one(RegulatoryDocumentIntakeReprepare, request_id)
    assert request.status == "prepared"
    batch = session.get_one(RegulatoryDocumentIntake, batch_id)
    assert batch.status == "failed" and batch.stage == "metadata"
    before = deepcopy(request.documents)
    intake.retry_intake(session, batch)
    session.commit()

    def forbidden(*_args: Any, **_kwargs: Any) -> bool:
        pytest.fail("A metadata retry must not replace canonical chunks again")

    monkeypatch.setattr(writer_publication, "chunk_owned_file", forbidden)
    monkeypatch.setattr(writer_publication, "reprepare_owned_intake_file", forbidden)
    assert (
        intake_job.run_document_intake(intake_id=batch_id, tenant_id="public")
        is not None
    )
    session.expire_all()
    assert (
        session.get_one(RegulatoryDocumentIntakeReprepare, request_id).documents
        == before
    )


@pytest.mark.parametrize(
    "block",
    [
        "manual",
        "manual_ready",
        "stale_ready",
        "manual_date",
        "completed",
        "published",
        "lease",
        "submitted",
        "shard",
    ],
)
def test_reprepare_refuses_reviewed_published_or_inflight_state(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
    block: str,
) -> None:
    session = intake_session
    batch, run_id = _fixture(
        session, intake_owner, native_markdown_runtime, monkeypatch
    )
    member = intake.intake_files(session, batch)[0]
    row = session.scalar(
        select(RegulatoryChunk).where(
            RegulatoryChunk.user_file_id == member.user_file_id
        )
    )
    assert row is not None
    if block in {"manual_ready", "stale_ready"}:
        _mark_ready(session, batch, run_id)
    if block in {"manual", "manual_ready"}:
        row.chunk_metadata = {
            **row.chunk_metadata,
            "intake_edit_revision": str(uuid4()),
        }
        if block == "manual_ready":
            session.flush()
            batch.review_sha256 = intake.review_digest(session, batch)
    elif block == "stale_ready":
        row.text += " changed outside the ready review"
    elif block == "manual_date":
        row.validity_start_date = datetime.now().date()
    elif block == "completed":
        from onyx.db.enums import UserFileStatus

        session.get_one(UserFile, member.user_file_id).status = UserFileStatus.COMPLETED
    elif block == "published":
        session.get_one(RegulatoryFilePublication, member.user_file_id).epoch = 1
    elif block == "lease":
        run = session.get_one(RegulatoryLabelingRun, run_id)
        run.lease_token, run.lease_expires_at = (
            uuid4(),
            datetime.now(timezone.utc) + timedelta(minutes=5),
        )
    elif block == "submitted":
        item = session.scalar(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == run_id
            )
        )
        assert item is not None
        item.status = "submitted"
    else:
        session.add(
            RegulatoryLabelingShard(
                run_id=run_id,
                ordinal=99,
                item_ids=[],
                status="reconcile_required",
                submission_key=uuid4().hex,
            )
        )
    session.flush()
    with pytest.raises(intake.IntakeConflictError):
        repository.reprepare_snapshot(session, batch)


def test_stale_request_does_not_detach_old_run_and_idempotency_preserves_request(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = intake_session
    batch, run_id = _fixture(
        session, intake_owner, native_markdown_runtime, monkeypatch
    )
    with pytest.raises(intake.IntakeConflictError, match="snapshot changed"):
        repository.request_reprepare(
            session, batch, expected_sha256="0" * 64, idempotency_key=uuid4()
        )
    assert batch.labeling_run_id == run_id
    request = _request(session, batch)
    assert (
        repository.request_reprepare(
            session,
            batch,
            expected_sha256=request.expected_sha256,
            idempotency_key=request.idempotency_key,
        ).id
        == request.id
    )
