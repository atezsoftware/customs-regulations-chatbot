"""Legal readiness and its digest do not depend on background label progress."""

from datetime import date
from uuid import UUID

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import document_intake_metadata as metadata
from onyx.db.models import (
    DocumentSet,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryLabelingItem,
    User,
    UserFileProjectionRepair,
)
from onyx.regulatory.document_intake.metadata import (
    ChunkLegalMetadata,
    LegalMetadataResult,
    metadata_input_hash,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    _completed_labels,
)
from tests.external_dependency_unit.regulatory.test_document_intake_metadata import (
    _ready_metadata,
)
from tests.external_dependency_unit.regulatory.test_document_intake_metadata import (
    intake_owner as _owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake_metadata import (
    intake_session as _session,
)

intake_owner = _owner
intake_session = _session


def test_background_context_edit_freezes_scoped_refresh_then_returns_to_legal_review(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.db import document_intake_edit as editing
    from onyx.db import document_intake_edit_budget as edit_budget
    from onyx.natural_language_processing.utils import TiktokenTokenizer
    from onyx.regulatory.document_intake.metadata_checkpoint import MetadataCheckpoint
    from onyx.regulatory.document_intake.metadata_job import metadata_recipe_sha256
    from tests.external_dependency_unit.regulatory.test_intake_metadata_checkpoint import (
        ready_checkpoint,
    )

    batch, members, token, binding = ready_checkpoint(
        intake_session, intake_owner, scoped=True
    )
    batch.background_labeling_enabled = True
    for member in members:
        rows = list(
            intake_session.scalars(
                select(RegulatoryChunk)
                .where(RegulatoryChunk.user_file_id == member.user_file_id)
                .order_by(RegulatoryChunk.position)
            )
        )
        chunks = metadata.metadata_chunks(rows)
        metadata.apply_metadata_results(
            intake_session,
            intake_id=batch.id,
            lease_token=token,
            document_id=member.id,
            expected_input_sha256=metadata_input_hash(chunks),
            results=[
                (
                    "a" * 64,
                    LegalMetadataResult(
                        chunks=[ChunkLegalMetadata(chunk_id=c.chunk_id) for c in chunks]
                    ),
                )
            ],
        )
    _completed_labels(intake_session, batch, members)
    batch.status, batch.stage = "ready", "review"
    batch.lease_token = batch.lease_expires_at = None
    batch.background_label_status = "completed"
    batch.review_sha256 = intake.review_digest(intake_session, batch)
    batch.background_label_source_sha256 = batch.review_sha256
    monkeypatch.setattr(
        editing, "resolve_intake_labeling_provider_binding", lambda *_a, **_k: binding
    )
    monkeypatch.setattr(editing, "validate_intake_edit_budgets", lambda *_a: None)
    monkeypatch.setattr(
        edit_budget,
        "_embedding_tokenizer",
        lambda _s: TiktokenTokenizer("text-embedding-3-large"),
    )
    intake_session.flush()
    root = next(m for m in members if m.role == "root")
    row = intake_session.scalar(
        select(RegulatoryChunk)
        .where(RegulatoryChunk.user_file_id == root.user_file_id)
        .order_by(RegulatoryChunk.position)
    )
    assert row is not None
    receipt = editing.edit_review_digest(intake_session, batch)
    assert receipt is not None
    assert (
        editing.edit_intake_chunk(
            intake_session,
            batch,
            document_id=root.id,
            chunk_id=row.id,
            expected_review_sha256=receipt,
            updates={"text": row.text + "\nBelge ibraz edilir."},
        )
        is None
    )
    saved = MetadataCheckpoint.model_validate(batch.metadata_checkpoint)
    assert saved.extraction_mode == "scoped_v1"
    assert saved.recipe_sha256 == metadata_recipe_sha256("scoped_v1")
    assert (batch.status, batch.stage, batch.background_label_status) == (
        "preparing",
        "chunking",
        "cancelled",
    )
    token = intake.claim_intake(intake_session, batch.id)
    assert token is not None
    _batch, work = metadata.load_metadata_work(intake_session, batch.id, token)
    for document_id, chunks in work:
        metadata.apply_metadata_results(
            intake_session,
            intake_id=batch.id,
            lease_token=token,
            document_id=document_id,
            expected_input_sha256=metadata_input_hash(chunks),
            results=[
                (
                    "b" * 64,
                    LegalMetadataResult(
                        chunks=[ChunkLegalMetadata(chunk_id=c.chunk_id) for c in chunks]
                    ),
                )
            ],
        )
    assert intake.start_intake_labeling(intake_session, batch.id, token) is None
    assert (batch.status, batch.stage, batch.background_label_status) == (
        "ready",
        "review",
        "pending",
    )


@pytest.mark.parametrize("label_status", ["completed", "failed", "cancelled"])
def test_terminal_background_labels_allow_metadata_edit_without_blocking_review(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    label_status: str,
) -> None:
    from onyx.db import document_intake_edit as editing
    from onyx.db.models import RegulatoryLabelingRun
    from tests.external_dependency_unit.regulatory.test_document_intake_edit import (
        _ready,
        _root_chunk,
    )

    batch, members = _ready(intake_session, intake_owner, monkeypatch)
    batch.background_labeling_enabled = True
    run = intake_session.get_one(RegulatoryLabelingRun, batch.labeling_run_id)
    run.status = label_status
    batch.background_label_status = (
        "completed" if label_status == "completed" else "failed"
    )
    batch.review_sha256 = intake.review_digest(intake_session, batch)
    batch.background_label_source_sha256 = batch.review_sha256
    intake_session.flush()
    root, chunk = _root_chunk(intake_session, members)
    receipt = editing.edit_review_digest(intake_session, batch)
    assert receipt is not None
    monkeypatch.setattr(
        "onyx.db.document_intake_background_labels.validate_legal_source",
        lambda *_a: None,
    )
    result = editing.edit_intake_chunk(
        intake_session,
        batch,
        document_id=root.id,
        chunk_id=chunk.id,
        expected_review_sha256=receipt,
        updates={"validity_start_date": date(2025, 3, 1)},
    )
    assert result is None
    assert (batch.status, batch.stage, batch.background_label_status) == (
        "ready",
        "review",
        "pending",
    )
    assert (
        batch.review_sha256
        == batch.background_label_source_sha256
        == intake.review_digest(intake_session, batch)
    )
    assert batch.labeling_run_id == run.id


def test_metadata_ready_is_reviewable_while_background_labels_run_and_digest_is_stable(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.db import document_intake_background_labels as background

    batch, members, token, _article, _closing = _ready_metadata(
        intake_session, intake_owner
    )
    batch.metadata_extraction_mode = "scoped_v1"
    batch.background_labeling_enabled = True
    for member in members:
        rows = list(
            intake_session.scalars(
                select(RegulatoryChunk)
                .where(RegulatoryChunk.user_file_id == member.user_file_id)
                .order_by(RegulatoryChunk.position)
            )
        )
        chunks = metadata.metadata_chunks(rows)
        metadata.apply_metadata_results(
            intake_session,
            intake_id=batch.id,
            lease_token=token,
            document_id=member.id,
            expected_input_sha256=metadata_input_hash(chunks),
            results=[
                (
                    "a" * 64,
                    LegalMetadataResult(
                        chunks=[ChunkLegalMetadata(chunk_id=c.chunk_id) for c in chunks]
                    ),
                )
            ],
        )

    def create_run(_session: Session, _batch: object) -> UUID:
        run = _completed_labels(intake_session, batch, members)
        run.status, run.stage = "queued", "preparing"
        run.requested_by_id = batch.created_by
        batch.status, batch.stage = "ready", "review"
        return run.id

    monkeypatch.setattr(background, "_create_run", create_run)
    assert intake.start_intake_labeling(intake_session, batch.id, token) is None
    assert (batch.status, batch.stage, batch.background_label_status) == (
        "ready",
        "review",
        "pending",
    )
    run_id = background.start_background_labels(intake_session, batch)
    assert run_id is not None
    assert (batch.status, batch.stage) == ("ready", "review")
    assert batch.background_label_status == "running"
    assert batch.lease_token is None
    before = batch.review_sha256
    assert before == intake.review_digest(intake_session, batch)
    for item in intake_session.scalars(
        select(RegulatoryLabelingItem).where(RegulatoryLabelingItem.run_id == run_id)
    ):
        item.labels = ["Different completed label"]
    intake_session.flush()
    assert before == intake.review_digest(intake_session, batch)
    assert not list(
        intake_session.scalars(
            select(UserFileProjectionRepair).where(
                UserFileProjectionRepair.user_file_id.in_(
                    [m.user_file_id for m in members]
                )
            )
        )
    )
    from onyx.db.document_intake_labeling import is_current_intake_labeling_run
    from onyx.db.enums import UserFileStatus
    from onyx.db.models import RegulatoryLabelingRun, UserFile

    run = intake_session.get_one(RegulatoryLabelingRun, run_id)
    assert is_current_intake_labeling_run(intake_session, run)
    batch.status, batch.stage = "approving", "indexing"
    for member in members:
        intake_session.get_one(
            UserFile, member.user_file_id
        ).status = UserFileStatus.PROCESSING
    intake_session.flush()
    assert is_current_intake_labeling_run(intake_session, run)
    batch.status, batch.stage = "completed", "completed"
    for member in members:
        intake_session.get_one(
            UserFile, member.user_file_id
        ).status = UserFileStatus.COMPLETED
    run.status, run.stage, run.error = (
        "failed",
        "finished",
        "Label provider unavailable",
    )
    background.reconcile_background_labels(intake_session, batch)
    assert batch.background_label_status == "failed"
    assert (batch.status, batch.stage) == ("completed", "completed")
    assert intake.review_digest(intake_session, batch) == before


def test_incomplete_legal_metadata_cannot_become_ready_even_with_background_labels(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
) -> None:
    batch, _members, token, _article, _closing = _ready_metadata(
        intake_session, intake_owner
    )
    batch.metadata_extraction_mode = "scoped_v1"
    batch.background_labeling_enabled = True
    with pytest.raises(intake.IntakeConflictError, match="metadata"):
        intake.start_intake_labeling(intake_session, batch.id, token)


def test_legacy_completed_metadata_adopts_ready_state_and_recovers_label_intent(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.db import document_intake_background_labels as background

    batch, members, token, _article, _closing = _ready_metadata(
        intake_session, intake_owner
    )
    assert batch.metadata_extraction_mode is None
    for member in members:
        rows = list(
            intake_session.scalars(
                select(RegulatoryChunk)
                .where(RegulatoryChunk.user_file_id == member.user_file_id)
                .order_by(RegulatoryChunk.position)
            )
        )
        chunks = metadata.metadata_chunks(rows)
        metadata.apply_metadata_results(
            intake_session,
            intake_id=batch.id,
            lease_token=token,
            document_id=member.id,
            expected_input_sha256=metadata_input_hash(chunks),
            results=[
                (
                    "a" * 64,
                    LegalMetadataResult(
                        chunks=[ChunkLegalMetadata(chunk_id=c.chunk_id) for c in chunks]
                    ),
                )
            ],
        )
    batch.stage = "waiting_labeling"
    batch.lease_token = batch.lease_expires_at = None
    background.adopt_completed_legal_preparation(intake_session, batch)
    assert (batch.status, batch.stage, batch.background_label_status) == (
        "ready",
        "review",
        "pending",
    )

    def create_run(_session: Session, _batch: object) -> UUID:
        run = _completed_labels(intake_session, batch, members)
        run.status, run.stage = "queued", "preparing"
        run.requested_by_id = batch.created_by
        batch.status, batch.stage = "ready", "review"
        return run.id

    monkeypatch.setattr(background, "_create_run", create_run)
    deliveries = background.resume_background_labeling(
        intake_session, environment="document-intake-regression"
    )
    assert deliveries == [batch.labeling_run_id]
    assert batch.background_label_status == "running"
    assert (batch.status, batch.stage) == ("ready", "review")
