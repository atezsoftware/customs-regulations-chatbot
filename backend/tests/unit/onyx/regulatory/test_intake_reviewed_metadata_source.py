"""Reviewed edits bind exact sources, revisions and idle label ownership."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import document_intake_metadata_review as review
from onyx.db import regulatory_labeling as labeling
from onyx.db.document_intake_metadata import metadata_chunks
from onyx.db.models import RegulatoryChunk, RegulatoryDocumentIntake
from onyx.regulatory.document_intake.metadata import (
    METADATA_AUDIT_KEY,
    metadata_input_hash,
)
from onyx.regulatory.document_intake.metadata_review import (
    ReviewedMetadataDocument,
    ReviewedMetadataEdit,
)


@pytest.mark.parametrize(
    "change", ["none", "text", "revision", "file", "stale_evidence"]
)
def test_two_reviewed_graph_edits_bind_all54_missing_metadata_sources(
    change: str,
) -> None:
    fid = uuid4()
    rows = [
        RegulatoryChunk(
            id=f"graph-{i}",
            user_file_id=fid,
            text=f"Graph source row {i}.",
            heading_path=["EK 3"],
            chunk_type="free_text",
            position=i,
            chunk_metadata={},
        )
        for i in range(54)
    ]
    before = metadata_input_hash(metadata_chunks(rows))
    revisions = {}
    for i in (43, 44):
        rows[i].text = f"Reviewed graph relations and durations {i}."
        revisions[rows[i].id] = uuid4()
        rows[i].chunk_metadata = {"intake_edit_revision": str(revisions[rows[i].id])}
    document = ReviewedMetadataDocument(
        user_file_id=fid,
        previous_input_sha256=before,
        input_sha256=metadata_input_hash(metadata_chunks(rows)),
        canonical_source_sha256=review.canonical_metadata_source_digest(rows),
        revisions=revisions,
        refresh_metadata=True,
    )
    if change == "text":
        rows[43].text += " Unexpected drift."
    elif change == "revision":
        rows[44].chunk_metadata["intake_edit_revision"] = str(uuid4())
    elif change == "file":
        fid = uuid4()
    elif change == "stale_evidence":
        rows[10].chunk_metadata[METADATA_AUDIT_KEY] = {
            "input_sha256": before,
            "input_stale": True,
            "fields": {"document_title": {"source": "llm", "stale": False}},
        }
    if change == "none":
        review.require_reviewed_document(document, fid, rows)
        assert sum(METADATA_AUDIT_KEY in row.chunk_metadata for row in rows) == 0
        assert document.previous_input_sha256 != document.input_sha256
    else:
        with pytest.raises(intake.IntakeConflictError):
            review.require_reviewed_document(document, fid, rows)


@pytest.mark.parametrize(
    "change,allowed",
    [
        ("prelabel", True),
        ("terminal", True),
        ("disjoint", True),
        ("own_active", False),
        ("foreign_overlap", False),
        ("prior_running", False),
        ("prior_scope", False),
        ("prior_lease", False),
        ("wrong_owner", False),
    ],
)
def test_reviewed_metadata_rechecks_prior_owner_and_overlapping_label_runs(
    monkeypatch: pytest.MonkeyPatch, change: str, allowed: bool
) -> None:
    fid, member, previous_id = uuid4(), uuid4(), uuid4()
    batch = cast(
        RegulatoryDocumentIntake,
        SimpleNamespace(
            id=uuid4(),
            source_package_id=uuid4(),
            labeling_run_id=None if change == "prelabel" else previous_id,
            document_set_id=15,
        ),
    )
    document = ReviewedMetadataDocument(
        user_file_id=fid,
        previous_input_sha256="a" * 64,
        input_sha256="b" * 64,
        canonical_source_sha256="c" * 64,
        revisions={"source": uuid4()},
        refresh_metadata=True,
    )
    intent = ReviewedMetadataEdit(
        intake_id=batch.id,
        package_id=batch.source_package_id,
        previous_labeling_run_id=batch.labeling_run_id,
        expected_review_sha256="d" * 64,
        documents={member: document},
    )
    previous = SimpleNamespace(
        status="completed_with_errors",
        file_ids=[str(fid)],
        lease_token=None,
        lease_expires_at=None,
    )
    active_id = (
        previous_id
        if change == "own_active"
        else uuid4()
        if change in {"foreign_overlap", "disjoint"}
        else None
    )
    active = SimpleNamespace(file_ids=[str(uuid4() if change == "disjoint" else fid)])
    if change == "prior_running":
        previous.status = "running"
    elif change == "prior_scope":
        previous.file_ids = [str(uuid4())]
    elif change == "prior_lease":
        previous.lease_token, previous.lease_expires_at = (
            uuid4(),
            datetime.now(timezone.utc) + timedelta(minutes=1),
        )
    elif change == "wrong_owner":
        batch.labeling_run_id = uuid4()
    session = MagicMock(spec=Session)
    session.get_one.side_effect = lambda _model, identifier, **_kw: (
        previous if identifier == previous_id else active
    )
    monkeypatch.setattr(labeling, "get_active_run_id", lambda *_args: active_id)
    if allowed:
        review.require_reviewed_metadata_owner(session, batch, intent, {str(fid)})
    else:
        with pytest.raises(intake.IntakeConflictError):
            review.require_reviewed_metadata_owner(session, batch, intent, {str(fid)})
