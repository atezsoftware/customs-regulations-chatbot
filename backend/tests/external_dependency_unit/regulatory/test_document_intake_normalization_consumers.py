"""Receipt consumers retain strict source authority across a proven formatter change."""

from copy import deepcopy
from datetime import date
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import document_intake_edit as editing
from onyx.db import document_intake_publication_recovery as recovery
from onyx.db import regulatory_labeling as labeling
from onyx.db.enums import UserFileStatus
from onyx.db.models import (
    DocumentSet,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryDocumentIntakeReprepare,
    RegulatoryFilePublication,
    RegulatoryLabelingRun,
    User,
    UserFile,
)
from onyx.db.regulatory_original_ingestion import original_attachment_receipt
from onyx.db.regulatory_writer_publication import canonical_scope_digest
from onyx.db.user_file import (
    finish_user_file_index_request,
    start_user_file_index_request,
)
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.regulatory.document_intake.metadata import METADATA_AUDIT_KEY
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_owner as intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_session as intake_session,
)
from tests.external_dependency_unit.regulatory.test_document_intake_markdown import (
    native_markdown_runtime as native_markdown_runtime,
)
from tests.external_dependency_unit.regulatory.test_document_intake_normalization import (
    _prepared_transition,
)


def _complete_labels(session: Session, batch: RegulatoryDocumentIntake) -> None:
    assert batch.labeling_run_id is not None
    run = session.get_one(RegulatoryLabelingRun, batch.labeling_run_id)
    lease = labeling.claim_run(
        session,
        run_id=run.id,
        expected_generation=run.lease_generation,
        lease_seconds=300,
    )
    assert lease is not None
    while page := labeling.prepare_next_item_page(session, lease, limit=128):
        for item in page:
            item.status, item.labels = "completed", ["Gümrük"]
            assignments: list[dict[str, object]] = [
                {"label_id": "Gümrük", "evidence_quote": item.text_snapshot}
            ]
            item.assignments = assignments
    run.completed_chunks = run.total_chunks
    while labeling.has_pending_derived_projections(session, lease):
        labeling.project_derived_labels(session, lease, limit=128)
    labeling.release_run(
        session, lease, status="completed", stage="finished", finished=True
    )
    session.flush()
    intake.refresh_intake(session, batch)
    assert batch.status == "ready" and batch.review_sha256
    session.commit()


def _reviewed_transition(
    session: Session,
    owner: tuple[User, DocumentSet, ModelConfiguration],
    store: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeReprepare,
    RegulatoryDocumentIntakeFile,
    dict[UUID, UUID],
    dict[UUID, dict[str, Any]],
]:
    batch, request = _prepared_transition(session, owner, store, monkeypatch)
    _complete_labels(session, batch)
    main_id = next(
        UUID(entry["user_file_id"])
        for entry in request.documents.values()
        if entry.get("normalization_derivation")
    )
    members = intake.intake_files(session, batch)
    main = next(member for member in members if member.user_file_id == main_id)
    chunk = session.scalar(
        select(RegulatoryChunk)
        .where(
            RegulatoryChunk.user_file_id == main_id,
            RegulatoryChunk.chunk_type != "hierarchical_aggregate",
            RegulatoryChunk.text.contains("Başvuru sırasında"),
        )
        .order_by(RegulatoryChunk.position)
    )
    assert chunk is not None and batch.review_sha256 is not None
    monkeypatch.setattr(
        editing,
        "resolve_intake_labeling_provider_binding",
        intake.resolve_intake_labeling_provider_binding,
    )
    assert editing.edit_intake_chunk(
        session,
        batch,
        document_id=main.id,
        chunk_id=chunk.id,
        expected_review_sha256=batch.review_sha256,
        updates={
            "text": chunk.text.replace("Başvuru sırasında", "Başvuru  sırasında"),
            "chunk_metadata": {"article_title": "Başvuru için belgeler"},
            "validity_start_date": date(2027, 1, 1),
        },
    )
    _complete_labels(session, batch)
    assert (
        chunk.chunk_metadata[METADATA_AUDIT_KEY]["fields"]["validity_start_date"][
            "source"
        ]
        == "manual"
    )
    assert "Başvuru  sırasında" in chunk.text
    assert chunk.chunk_metadata["article_title"] == "Başvuru için belgeler"
    assert (
        chunk.chunk_metadata[METADATA_AUDIT_KEY]["fields"]["article_title"]["source"]
        == "manual"
    )
    assert batch.review_sha256 is not None
    tokens = dict(intake.approve_intake(session, batch, batch.review_sha256))
    originals: dict[UUID, dict[str, Any]] = {}
    for member in members:
        original = session.get_one(
            RegulatoryFilePublication, member.user_file_id
        ).original_ingestion_receipt
        assert original is not None
        originals[member.user_file_id] = deepcopy(original)
    session.commit()
    return batch, request, main, tokens, originals


@pytest.mark.parametrize("consumer", ["original", "retry", "recovery"])
@pytest.mark.parametrize("forged", [False, True])
def test_normalized_original_review_retry_and_recovery_keep_the_same_source_authority(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
    consumer: str,
    forged: bool,
) -> None:
    session = intake_session
    batch, request, main, tokens, originals = _reviewed_transition(
        session, intake_owner, native_markdown_runtime, monkeypatch
    )
    before_canonical = {
        file_id: canonical_scope_digest(session, file_id) for file_id in tokens
    }
    reviewed = batch.review_sha256
    retained_run = batch.labeling_run_id
    for file_id, token in tokens.items():
        assert start_user_file_index_request(session, file_id, token)
        assert finish_user_file_index_request(
            session,
            file_id,
            token,
            failure_code=None if consumer == "original" else "context:ValueError",
        )
        if consumer == "original":
            session.get_one(UserFile, file_id).status = UserFileStatus.COMPLETED
            session.get_one(RegulatoryFilePublication, file_id).epoch = 1
    session.flush()
    intake.refresh_intake(session, batch)
    assert batch.status == ("completed" if consumer == "original" else "failed")
    session.commit()
    if forged:
        altered = deepcopy(request.documents)
        altered[str(main.id)]["normalization_derivation"]["witness_sha256"] = "0" * 64
        request.documents = altered
        session.flush()
    if consumer == "original":
        receipt = original_attachment_receipt(session, main.user_file_id)
        if forged:
            assert receipt is None
        else:
            assert receipt is not None
            assert receipt.raw_sha256 == originals[main.user_file_id]["raw_sha256"]
            assert (
                receipt.plaintext_sha256
                == request.documents[str(main.id)]["after_receipt"]["plaintext_sha256"]
            )
            assert (
                receipt.plaintext_sha256
                != originals[main.user_file_id]["plaintext_sha256"]
            )
            assert receipt.canonical_sha256 == before_canonical[main.user_file_id]
    elif consumer == "retry":
        if forged:
            with pytest.raises(intake.IntakeConflictError, match="source"):
                intake.retry_intake(session, batch)
            assert batch.status == "failed"
            assert all(
                session.get_one(UserFile, file_id).status == UserFileStatus.FAILED
                for file_id in tokens
            )
        else:
            fresh = dict(intake.retry_intake(session, batch))
            assert fresh.keys() == tokens.keys()
            assert all(fresh[file_id] != token for file_id, token in tokens.items())
    else:
        file = session.get_one(UserFile, main.user_file_id)
        publication = session.get_one(RegulatoryFilePublication, file.id)
        if forged:
            with pytest.raises(intake.IntakeConflictError, match="source"):
                recovery._validate_source(session, batch, main, file, publication)
        else:
            assert (
                recovery._validate_source(session, batch, main, file, publication)
                == before_canonical[file.id]
            )
    assert batch.review_sha256 == reviewed and batch.labeling_run_id == retained_run
    assert {
        file_id: canonical_scope_digest(session, file_id) for file_id in tokens
    } == before_canonical
    assert {
        file_id: session.get_one(
            RegulatoryFilePublication, file_id
        ).original_ingestion_receipt
        for file_id in tokens
    } == originals
