"""Saved receipt validation cannot be bypassed by matching hashes or labeling."""

from copy import deepcopy
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import document_intake_metadata as repository
from onyx.db import document_intake_review as review
from onyx.db.models import RegulatoryChunk, RegulatoryDocumentIntake
from onyx.regulatory.document_intake.metadata import (
    METADATA_AUDIT_KEY,
    metadata_input_hash,
)


def _rows(*, invalid: bool, source: str = "llm") -> list[RegulatoryChunk]:
    file_id = uuid4()
    rows = [
        RegulatoryChunk(
            id=f"row-{uuid4()}",
            user_file_id=file_id,
            position=index,
            projection_ordinal=index,
            text="(1) Kaynak hüküm.",
            heading_path=[f"MADDE {number} - Başlık"],
            chunk_type="paragraph",
            chunk_metadata={},
            status="active",
            source="indexed",
        )
        for index, number in enumerate((33, 49))
    ]
    rows[0].chunk_metadata = {"article_no": "33"}
    input_hash = metadata_input_hash(repository.metadata_chunks(rows))
    for row in rows:
        row.chunk_metadata = {
            **row.chunk_metadata,
            METADATA_AUDIT_KEY: {
                "input_sha256": input_hash,
                "fields": {},
            },
        }
    evidence = rows[1] if invalid else rows[0]
    rows[0].chunk_metadata[METADATA_AUDIT_KEY]["fields"]["article_no"] = {
        "source": source,
        "basis": "explicit",
        "request_hash": "a" * 64,
        "source_chunk_id": evidence.id,
        "evidence_quote": evidence.heading_path[0],
    }
    return rows


@pytest.mark.parametrize("source", ["llm", "manual", "structural"])
def test_saved_article_receipts_revalidate_only_model_evidence(source: str) -> None:
    rows = _rows(invalid=True, source=source)
    before = deepcopy([row.chunk_metadata for row in rows])
    if source == "llm":
        with pytest.raises(ValueError, match="article locator contradicts"):
            repository.validate_metadata_receipt_evidence(
                rows, repository.metadata_chunks(rows)
            )
    else:
        repository.validate_metadata_receipt_evidence(
            rows, repository.metadata_chunks(rows)
        )
    assert [row.chunk_metadata for row in rows] == before


@pytest.mark.parametrize(
    "labeling,effective_invalid", [(False, True), (True, True), (True, False)]
)
def test_receipt_hash_skip_revalidates_and_preserves_other_completed_files(
    monkeypatch: pytest.MonkeyPatch,
    labeling: bool,
    effective_invalid: bool,
) -> None:
    valid, invalid = _rows(invalid=False), _rows(invalid=True)
    before = deepcopy([row.chunk_metadata for row in valid])
    batch = cast(
        RegulatoryDocumentIntake,
        SimpleNamespace(labeling_run_id=uuid4() if labeling else None),
    )
    members = [SimpleNamespace(id=uuid4()), SimpleNamespace(id=uuid4())]
    monkeypatch.setattr(intake, "load_intake_job", lambda *_args: (batch, None, []))
    monkeypatch.setattr(intake, "intake_files", lambda *_args: members)
    monkeypatch.setattr(
        review,
        "effective_intake_files",
        lambda *_args: members if effective_invalid else members[:1],
    )
    by_member = {members[0].id: valid, members[1].id: invalid}
    monkeypatch.setattr(
        repository, "_load_rows", lambda _session, _batch, member: by_member[member]
    )
    if labeling and effective_invalid:
        with pytest.raises(intake.IntakeConflictError, match="Saved article evidence"):
            repository.load_metadata_work(MagicMock(spec=Session), uuid4(), uuid4())
    else:
        _batch, work = repository.load_metadata_work(
            MagicMock(spec=Session), uuid4(), uuid4()
        )
        assert [member_id for member_id, _chunks in work] == (
            [] if labeling else [members[1].id]
        )
    assert [row.chunk_metadata for row in valid] == before


def test_approval_rejects_legacy_invalid_receipts_before_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = cast(
        RegulatoryDocumentIntake,
        SimpleNamespace(status="ready", review_sha256="review"),
    )
    monkeypatch.setattr(intake, "refresh_intake", lambda *_args: None)
    monkeypatch.setattr(intake, "review_digest", lambda *_args: "review")

    def reject(*_args: object) -> None:
        raise intake.IntakeConflictError(
            "Saved article evidence failed source validation"
        )

    monkeypatch.setattr(repository, "validate_intake_metadata_receipts", reject)
    session = MagicMock(spec=Session)
    with pytest.raises(intake.IntakeConflictError, match="Saved article evidence"):
        intake.approve_intake(session, batch, "review")
    session.get_one.assert_not_called()


def test_metadata_apply_locks_and_refreshes_before_checking_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = MagicMock(spec=Session)
    intake_id, lease, member = uuid4(), uuid4(), uuid4()

    def reject(*_args: object) -> None:
        session.get_one.assert_called_once_with(
            RegulatoryDocumentIntake,
            intake_id,
            with_for_update=True,
            populate_existing=True,
        )
        raise intake.IntakeConflictError("The intake preparation lease changed")

    monkeypatch.setattr(intake, "load_intake_job", reject)
    with pytest.raises(intake.IntakeConflictError, match="lease changed"):
        repository.apply_metadata_results(
            session,
            intake_id=intake_id,
            lease_token=lease,
            document_id=member,
            expected_input_sha256="a" * 64,
            results=[],
        )
    session.flush.assert_not_called()


@pytest.mark.parametrize("invalid", [False, True])
def test_approval_validates_only_effective_publication_members(
    monkeypatch: pytest.MonkeyPatch, invalid: bool
) -> None:
    active = SimpleNamespace(id=uuid4())
    batch = MagicMock(spec=RegulatoryDocumentIntake)
    monkeypatch.setattr(review, "effective_intake_files", lambda *_args: [active])
    monkeypatch.setattr(
        intake,
        "intake_files",
        lambda *_args: pytest.fail("Excluded source members are outside publication"),
    )

    def rows(
        _session: Session, _batch: RegulatoryDocumentIntake, member: object
    ) -> list[RegulatoryChunk]:
        assert member == active.id
        return _rows(invalid=invalid)

    monkeypatch.setattr(repository, "_load_rows", rows)
    if invalid:
        with pytest.raises(intake.IntakeConflictError, match="Saved article evidence"):
            repository.validate_intake_metadata_receipts(MagicMock(spec=Session), batch)
    else:
        repository.validate_intake_metadata_receipts(MagicMock(spec=Session), batch)
