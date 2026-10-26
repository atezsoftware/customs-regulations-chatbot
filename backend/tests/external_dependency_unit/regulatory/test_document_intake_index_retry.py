"""Failed initial publication retries retain the exact approved intake authority."""

import json
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db.enums import UserFileProjectionRepairStatus, UserFileStatus
from onyx.db.models import (
    DocumentSet,
    DocumentSet__UserFile,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryFilePublication,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryPublicationOrdinal,
    RegulatorySourceAsset,
    User,
    UserFile,
    UserFileProjectionRepair,
)
from onyx.db.regulatory_writer_publication import canonical_scope_digest
from onyx.db.user_file import (
    claim_user_file_projection_repair,
    finish_user_file_index_request,
    start_user_file_index_request,
)
from onyx.file_processing.original_ingestion import OriginalIngestionReceipt
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from tests.external_dependency_unit.regulatory.test_document_intake import (
    _admitted,
    _completed_labels,
    _existing_rows,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_owner as intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_session as intake_session,
)


def _rows(session: Session, *models: Any) -> str:
    return json.dumps(
        {
            model.__tablename__: sorted(
                (
                    dict(row)
                    for row in session.execute(select(model.__table__)).mappings()
                ),
                key=lambda row: str(row),
            )
            for model in models
        },
        sort_keys=True,
        default=str,
    )


def _failed_initial_publication(
    session: Session,
    owner: tuple[User, DocumentSet, ModelConfiguration],
    *,
    original_raw_sha256: str | None = None,
) -> tuple[
    RegulatoryDocumentIntake, list[RegulatoryDocumentIntakeFile], dict[UUID, UUID]
]:
    user, target, model = owner
    batch, members = _admitted(session, user, target, model)
    for member in members:
        file = session.get_one(UserFile, member.user_file_id)
        asset = session.get_one(RegulatorySourceAsset, member.source_asset_id)
        file.regulatory_chunk_generation_hash = "a" * 64
        receipt = OriginalIngestionReceipt(
            file_id=file.file_id,
            raw_sha256=original_raw_sha256 or asset.sha256,
            documents_sha256="b" * 64,
            plaintext_sha256=asset.text_sha256 or asset.sha256,
            canonical_sha256=canonical_scope_digest(session, file.id),
            generation_hash="a" * 64,
        )
        session.add(
            RegulatoryFilePublication(
                user_file_id=file.id,
                scope_key="intake-index-retry-regression",
                owner_id=uuid4(),
                fencing_token=1,
                lease_expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
                gate_closed=False,
                epoch=0,
                next_ordinal=1,
                original_ingestion_receipt=receipt.model_dump(mode="json"),
            )
        )
        session.flush()
        for chunk in session.scalars(
            select(RegulatoryChunk).where(RegulatoryChunk.user_file_id == file.id)
        ):
            session.add(
                RegulatoryPublicationOrdinal(
                    user_file_id=file.id,
                    allocation_key=f"canonical:{chunk.id}",
                    ordinal=chunk.projection_ordinal,
                )
            )
    _completed_labels(session, batch, members)
    intake.refresh_intake(session, batch)
    assert batch.status == "ready" and batch.review_sha256
    tokens = dict(intake.approve_intake(session, batch, batch.review_sha256))
    assert len(tokens) == 2
    session.commit()
    for file_id, token in tokens.items():
        assert start_user_file_index_request(session, file_id, token)
        assert finish_user_file_index_request(
            session, file_id, token, failure_code="context:ValueError"
        )
        assert session.get_one(UserFile, file_id).status == UserFileStatus.FAILED
        repair = session.get_one(UserFileProjectionRepair, file_id)
        assert repair.status == UserFileProjectionRepairStatus.FAILED
        assert repair.initial_index and repair.attempt_id == token
    intake.refresh_intake(session, batch)
    assert batch.status == "failed" and batch.stage == "indexing"
    session.commit()
    return batch, members, tokens


def test_failed_initial_projection_retry_creates_two_fresh_intents_without_relabeling(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
) -> None:
    session = intake_session
    user, target, model = intake_owner
    old_batch, old_members = _admitted(session, user, target, model)
    _completed_labels(session, old_batch, old_members)
    for member in old_members:
        session.get_one(UserFile, member.user_file_id).status = UserFileStatus.COMPLETED
        member.status = "completed"
    old_batch.status, old_batch.stage = "completed", "completed"
    session.flush()
    old_ids = [member.user_file_id for member in old_members]
    old_before = _existing_rows(session, old_ids)
    batch, members, previous = _failed_initial_publication(session, intake_owner)
    reviewed, run_id = batch.review_sha256, batch.labeling_run_id
    retained = _rows(
        session,
        RegulatoryChunk,
        RegulatoryLabelingItem,
        RegulatoryLabelingRun,
        RegulatorySourceAsset,
        RegulatoryFilePublication,
    )
    store = PostgresBackedFileStore()
    source_bytes: dict[str, bytes] = {}
    for member in members:
        file = session.get_one(UserFile, member.user_file_id)
        with store.read_file(file.file_id, db_session=session) as source:
            source_bytes[file.file_id] = source.read()
        # The general repair API must continue to reject FAILED files.
        assert (
            claim_user_file_projection_repair(session, file.id, initial_index=True)
            is None
        )
    fresh = dict(intake.retry_intake(session, batch))
    assert set(fresh) == set(previous)
    assert all(fresh[key] != previous[key] for key in previous)
    assert batch.status == "approving" and batch.stage == "indexing"
    assert batch.review_sha256 == reviewed and batch.labeling_run_id == run_id
    for member in members:
        file = session.get_one(UserFile, member.user_file_id)
        repair = session.get_one(UserFileProjectionRepair, file.id)
        assert file.status == UserFileStatus.CHUNKED
        assert member.status == "indexing" and member.error is None
        assert member.approval_attempt_id == fresh[file.id] == repair.attempt_id
        assert repair.status == UserFileProjectionRepairStatus.PENDING
        assert repair.initial_index and repair.failure_code is None
        with store.read_file(file.file_id, db_session=session) as source:
            assert source.read() == source_bytes[file.file_id]
        assert not start_user_file_index_request(session, file.id, previous[file.id])
    assert (
        _rows(
            session,
            RegulatoryChunk,
            RegulatoryLabelingItem,
            RegulatoryLabelingRun,
            RegulatorySourceAsset,
            RegulatoryFilePublication,
        )
        == retained
    )
    assert _existing_rows(session, old_ids) == old_before


@pytest.mark.parametrize(
    "change",
    [
        "review",
        "text",
        "source_pointer",
        "source_sha",
        "membership",
        "owner",
        "gate",
        "manifest",
        "lease",
        "published",
        "previous_intent",
        "ordinary_intent",
        "missing_intent",
    ],
)
def test_failed_initial_retry_rejects_changed_authority_before_any_mutation(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    change: str,
) -> None:
    session = intake_session
    batch, members, _previous = _failed_initial_publication(
        session,
        intake_owner,
        original_raw_sha256="0" * 64 if change == "source_sha" else None,
    )
    # Corrupt the last file so an earlier valid file cannot be retried first.
    member = intake.intake_files(session, batch)[-1]
    file = session.get_one(UserFile, member.user_file_id)
    publication = session.get_one(RegulatoryFilePublication, file.id)
    repair = session.get_one(UserFileProjectionRepair, file.id)
    if change == "review":
        batch.review_sha256 = "0" * 64
    elif change == "text":
        chunk = session.scalar(
            select(RegulatoryChunk).where(RegulatoryChunk.user_file_id == file.id)
        )
        assert chunk is not None
        chunk.text += " unreviewed mutation"
    elif change == "source_pointer":
        other = next(item for item in members if item.id != member.id)
        file.file_id = session.get_one(UserFile, other.user_file_id).file_id
    elif change == "source_sha":
        # Both source assets and receipts are immutable; seed their mismatch above.
        pass
    elif change == "membership":
        session.delete(
            session.get_one(DocumentSet__UserFile, (batch.document_set_id, file.id))
        )
    elif change == "owner":
        file.user_id = None
    elif change == "gate":
        publication.gate_closed = True
    elif change == "manifest":
        publication.writer_manifest = {"unfinished": True}
    elif change == "lease":
        publication.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=2)
    elif change == "published":
        publication.epoch = 1
    elif change == "previous_intent":
        repair.attempt_id = uuid4()
    elif change == "ordinary_intent":
        repair.initial_index = False
    else:
        session.delete(repair)
    session.flush()
    before = _rows(
        session,
        UserFile,
        RegulatoryDocumentIntake,
        RegulatoryDocumentIntakeFile,
        UserFileProjectionRepair,
        RegulatoryFilePublication,
        RegulatoryChunk,
        RegulatoryLabelingItem,
        RegulatorySourceAsset,
        DocumentSet__UserFile,
    )
    with pytest.raises(intake.IntakeConflictError):
        intake.retry_intake(session, batch)
    assert (
        _rows(
            session,
            UserFile,
            RegulatoryDocumentIntake,
            RegulatoryDocumentIntakeFile,
            UserFileProjectionRepair,
            RegulatoryFilePublication,
            RegulatoryChunk,
            RegulatoryLabelingItem,
            RegulatorySourceAsset,
            DocumentSet__UserFile,
        )
        == before
    )
