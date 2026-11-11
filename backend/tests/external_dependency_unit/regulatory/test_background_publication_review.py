"""Publication/parent review does not depend on optional label completion."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import document_intake_metadata as metadata
from onyx.db.models import (
    DocumentSet,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryFilePublication,
    RegulatorySourceAsset,
    User,
)
from onyx.regulatory.document_intake.metadata import (
    ChunkLegalMetadata,
    LegalMetadataResult,
    metadata_input_hash,
)
from tests.external_dependency_unit.regulatory.test_document_intake_review_acceptance import (
    _decision,
    _table_rows,
    _three_document_review,
)
from tests.external_dependency_unit.regulatory.test_document_intake_review_acceptance import (
    intake_owner as intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake_review_acceptance import (
    intake_session as intake_session,
)


@pytest.mark.parametrize(
    "state", ["pending", "waiting", "running", "completed", "failed"]
)
def test_publication_scope_can_change_without_reintroducing_label_gate(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    s = intake_session
    batch, members, previous, _ = _three_document_review(s, intake_owner, monkeypatch)
    batch.labeling_run_id = None
    batch.status, batch.stage = "preparing", "metadata"
    batch.lease_token = token = uuid4()
    batch.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)
    for m in members.values():
        m.status = "chunked"
        rows = list(
            s.scalars(
                select(RegulatoryChunk)
                .where(RegulatoryChunk.user_file_id == m.user_file_id)
                .order_by(RegulatoryChunk.position)
            )
        )
        chunks = metadata.metadata_chunks(rows)
        metadata.apply_metadata_results(
            s,
            intake_id=batch.id,
            lease_token=token,
            document_id=m.id,
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
    batch.lease_token = batch.lease_expires_at = None
    batch.background_labeling_enabled = True
    batch.background_label_status = state
    batch.labeling_run_id = None if state in {"pending", "waiting"} else previous.id
    previous.status = (
        "running"
        if state == "running"
        else "failed"
        if state == "failed"
        else "completed"
    )
    batch.status, batch.stage = "ready", "review"
    for m in members.values():
        m.status = "ready"
    batch.review_sha256 = intake.review_digest(s, batch)
    batch.background_label_source_sha256 = batch.review_sha256
    s.flush()
    original = _table_rows(
        s, RegulatoryChunk, RegulatorySourceAsset, RegulatoryFilePublication
    )
    assert (
        _decision(
            s,
            batch,
            members["alias"],
            intake_owner[0],
            disposition="source_evidence",
            duplicate=members["root"].id,
        )
        is None
    )
    assert (batch.status, batch.stage, batch.background_label_status) == (
        "ready",
        "review",
        "pending",
    )
    assert (
        batch.review_sha256
        == batch.background_label_source_sha256
        == intake.review_digest(s, batch)
    )
    assert members["alias"].status == "source_evidence"
    if state == "running":
        assert previous.cancel_requested
    assert (
        _decision(
            s,
            batch,
            members["alias"],
            intake_owner[0],
            disposition="source_evidence",
            duplicate=members["annex"].id,
        )
        is None
    )
    assert (
        batch.background_label_source_sha256
        == batch.review_sha256
        == intake.review_digest(s, batch)
    )
    assert _decision(s, batch, members["alias"], intake_owner[0]) is None
    assert members["alias"].status == "ready"
    assert (batch.status, batch.stage, batch.background_label_status) == (
        "ready",
        "review",
        "pending",
    )
    assert (
        _table_rows(
            s, RegulatoryChunk, RegulatorySourceAsset, RegulatoryFilePublication
        )
        == original
    )

    from uuid import UUID

    from onyx.db import document_intake_background_labels as background
    from onyx.db import document_intake_review as review
    from onyx.db import regulatory_labeling as labeling
    from onyx.db.models import RegulatoryDocumentIntake

    if previous.status == "running":
        previous.status = "cancelled"

    def create(session: Session, current: RegulatoryDocumentIntake) -> UUID:
        run, _ = labeling.create_labeling_run(
            session,
            document_set_id=current.document_set_id,
            taxonomy=previous.taxonomy,
            model_configuration_id=current.labeling_model_configuration_id,
            model=previous.model,
            provider_binding=previous.provider_binding,
            requested_by_id=current.created_by,
            idempotency_key=uuid4(),
            target_file_ids=[
                m.user_file_id for m in review.effective_intake_files(session, current)
            ],
        )
        return run.id

    monkeypatch.setattr(background, "_create_run", create)
    assert background.start_background_labels(s, batch) is not None
    assert (batch.status, batch.stage) == ("ready", "review")
