"""Reviewed source changes refresh metadata before new label snapshots exist."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import document_intake_metadata as repository
from onyx.db import document_intake_metadata_review as review
from onyx.db import document_intake_review as publication_review
from onyx.db.models import RegulatoryChunk, RegulatoryDocumentIntake
from onyx.regulatory.document_intake.metadata import (
    METADATA_AUDIT_KEY,
    METADATA_MODEL,
    METADATA_VERSION,
    ChunkLegalMetadata,
    LegalMetadataResult,
    metadata_input_hash,
)
from onyx.regulatory.document_intake.metadata_checkpoint import MetadataCheckpoint
from onyx.regulatory.document_intake.metadata_review import (
    ReviewedMetadataDocument,
    ReviewedMetadataEdit,
)


def _work(
    monkeypatch: pytest.MonkeyPatch, *, labeled: bool, unchanged_missing: bool = False
) -> tuple[
    MagicMock,
    RegulatoryDocumentIntake,
    list[SimpleNamespace],
    list[RegulatoryChunk],
    list[RegulatoryChunk],
]:
    fid, other_fid, changed, other = uuid4(), uuid4(), uuid4(), uuid4()
    rows = [
        RegulatoryChunk(
            id=f"graph-{i}",
            user_file_id=fid,
            text=f"Source graph {i}.",
            heading_path=["EK 3"],
            chunk_type="free_text",
            position=i,
            chunk_metadata={},
        )
        for i in range(54)
    ]
    old_hash = metadata_input_hash(repository.metadata_chunks(rows))
    revisions = {}
    for i in (43, 44):
        rows[i].text += " Corrected relation."
        revisions[rows[i].id] = uuid4()
        rows[i].chunk_metadata = {"intake_edit_revision": str(revisions[rows[i].id])}
    if labeled:
        for row in rows:
            row.chunk_metadata.update(
                {
                    "document_title": "Old model title",
                    METADATA_AUDIT_KEY: {
                        "model": METADATA_MODEL,
                        "version": METADATA_VERSION,
                        "input_sha256": old_hash,
                        "input_stale": True,
                        "fields": {"document_title": {"source": "llm", "stale": True}},
                    },
                }
            )
    current_hash = metadata_input_hash(repository.metadata_chunks(rows))
    document = ReviewedMetadataDocument(
        user_file_id=fid,
        previous_input_sha256=old_hash,
        input_sha256=current_hash,
        canonical_source_sha256=review.canonical_metadata_source_digest(rows),
        revisions=revisions,
        refresh_metadata=True,
    )
    batch = RegulatoryDocumentIntake(
        id=uuid4(),
        source_package_id=uuid4(),
        status="preparing",
        stage="metadata",
        labeling_run_id=uuid4() if labeled else None,
        lease_token=uuid4(),
        lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
    )
    intent = ReviewedMetadataEdit(
        intake_id=batch.id,
        package_id=batch.source_package_id,
        previous_labeling_run_id=batch.labeling_run_id,
        expected_review_sha256="a" * 64,
        documents={changed: document},
    )
    batch.metadata_checkpoint = MetadataCheckpoint.build(
        recipe_sha256="b" * 64,
        provider_binding={},
        source_snapshot={"reviewed_edit": intent.model_dump(mode="json")},
        request_hashes={},
    ).model_dump(mode="json")
    unchanged = [
        RegulatoryChunk(
            id="unchanged",
            user_file_id=other_fid,
            text="(1) Unchanged source.",
            heading_path=["MADDE 1"],
            chunk_type="paragraph",
            position=0,
            chunk_metadata={},
        )
    ]
    if labeled and not unchanged_missing:
        unchanged[0].chunk_metadata[METADATA_AUDIT_KEY] = {
            "model": METADATA_MODEL,
            "version": METADATA_VERSION,
            "input_sha256": metadata_input_hash(repository.metadata_chunks(unchanged)),
            "input_stale": False,
            "fields": {},
        }
    members = [
        SimpleNamespace(id=changed, user_file_id=fid),
        SimpleNamespace(id=other, user_file_id=other_fid),
    ]
    session = MagicMock(spec=Session)
    session.get_one.return_value = batch
    monkeypatch.setattr(intake, "load_intake_job", lambda *_args: (batch, None, []))
    monkeypatch.setattr(intake, "intake_files", lambda *_args: members)
    monkeypatch.setattr(
        publication_review, "effective_intake_files", lambda *_args: members
    )
    monkeypatch.setattr(
        repository,
        "_load_rows",
        lambda _s, _b, mid: rows if mid == changed else unchanged,
    )
    return session, batch, members, rows, unchanged


@pytest.mark.parametrize("labeled", [False, True])
def test_reviewed_work_includes_missing_unchanged_files_only_before_labeling(
    monkeypatch: pytest.MonkeyPatch, labeled: bool
) -> None:
    session, batch, members, rows, unchanged = _work(monkeypatch, labeled=labeled)
    assert isinstance(batch.lease_token, UUID)
    before = deepcopy([r.chunk_metadata for r in rows + unchanged])
    _batch, work = repository.load_metadata_work(session, batch.id, batch.lease_token)
    assert [member for member, _chunks in work] == [
        m.id for m in members[: 1 if labeled else 2]
    ]
    assert len(work[0][1]) == 54
    assert all(
        "document_title" not in chunk.structural_metadata for chunk in work[0][1]
    )
    assert [r.chunk_metadata for r in rows + unchanged] == before


def test_unreviewed_post_label_missing_metadata_is_not_a_refresh_waiver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, batch, _members, _rows, _unchanged = _work(
        monkeypatch, labeled=True, unchanged_missing=True
    )
    assert isinstance(batch.lease_token, UUID)
    with pytest.raises(intake.IntakeConflictError, match="Unreviewed post-label"):
        repository.load_metadata_work(session, batch.id, batch.lease_token)


def test_validated_apply_clears_omitted_stale_model_fields_and_preserves_manual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, batch, members, rows, _unchanged = _work(monkeypatch, labeled=True)
    assert isinstance(batch.lease_token, UUID)
    manual = {
        "source": "manual",
        "basis": "user_review",
        "stale": False,
        "evidence_quote": None,
        "source_chunk_id": rows[43].id,
    }
    rows[43].chunk_metadata["annex_title"] = "Reviewed annex"
    rows[43].chunk_metadata[METADATA_AUDIT_KEY]["fields"]["annex_title"] = deepcopy(
        manual
    )
    expected = metadata_input_hash(repository.metadata_chunks(rows))
    results = [
        (
            f"{offset + 1:064x}",
            LegalMetadataResult(
                chunks=[
                    ChunkLegalMetadata(chunk_id=r.id, fields=[])
                    for r in rows[offset : offset + 8]
                ]
            ),
        )
        for offset in range(0, len(rows), 8)
    ]
    monkeypatch.setattr(repository, "refresh_intake_document_name", lambda *_args: None)
    monkeypatch.setattr(repository, "heartbeat_metadata", lambda *_args: None)
    repository.apply_metadata_results(
        session,
        intake_id=batch.id,
        lease_token=batch.lease_token,
        document_id=members[0].id,
        expected_input_sha256=expected,
        results=results,
    )
    assert all("document_title" not in r.chunk_metadata for r in rows)
    assert rows[43].chunk_metadata["annex_title"] == "Reviewed annex"
    assert (
        rows[43].chunk_metadata[METADATA_AUDIT_KEY]["fields"]["annex_title"] == manual
    )
    assert repository.metadata_receipts_current(rows, expected)


def test_old_source_ack_cannot_clear_stale_metadata_after_reviewed_edit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, batch, members, rows, _unchanged = _work(monkeypatch, labeled=True)
    assert isinstance(batch.lease_token, UUID)
    before = deepcopy([r.chunk_metadata for r in rows])
    with pytest.raises(intake.IntakeConflictError, match="Canonical document changed"):
        repository.apply_metadata_results(
            session,
            intake_id=batch.id,
            lease_token=batch.lease_token,
            document_id=members[0].id,
            expected_input_sha256="d" * 64,
            results=[],
        )
    assert [r.chunk_metadata for r in rows] == before
