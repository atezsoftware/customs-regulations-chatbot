"""One reviewed edit set replaces labels once and rolls back as a whole."""

import datetime
import json
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import document_intake_edit as editing
from onyx.db import regulatory_labeling as labeling
from onyx.db.models import (
    DocumentSet,
    ModelConfiguration,
    RegulatoryCanonicalRevision,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    User,
    UserFile,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_owner as intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_session as intake_session,
)
from tests.external_dependency_unit.regulatory.test_document_intake_edit import (
    _partial,
    _ready,
    _root_chunk,
)


def _snapshot(session: Session) -> str:
    session.flush()
    return json.dumps(
        {
            model.__tablename__: sorted(
                (
                    dict(row)
                    for row in session.execute(select(model.__table__)).mappings()
                ),
                key=str,
            )
            for model in (
                RegulatoryDocumentIntake,
                RegulatoryDocumentIntakeFile,
                UserFile,
                RegulatoryChunk,
                RegulatoryCanonicalRevision,
                RegulatoryLabelingRun,
                RegulatoryLabelingItem,
            )
        },
        default=str,
        sort_keys=True,
    )


def _items(session: Session, run_id: Any) -> list[RegulatoryLabelingItem]:
    return list(
        session.scalars(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == run_id
            )
        )
    )


def _annex(
    session: Session, members: list[RegulatoryDocumentIntakeFile]
) -> tuple[RegulatoryDocumentIntakeFile, RegulatoryChunk]:
    member = next(member for member in members if member.role == "annex")
    return member, session.scalars(
        select(RegulatoryChunk).where(
            RegulatoryChunk.user_file_id == member.user_file_id
        )
    ).one()


@pytest.mark.parametrize("partial", [False, True])
def test_bulk_changes_two_files_in_one_run_and_retains_original_outcomes(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    partial: bool,
) -> None:
    batch, members = (_partial if partial else _ready)(
        intake_session, intake_owner, monkeypatch
    )
    root, first = _root_chunk(intake_session, members)
    second = intake_session.scalars(
        select(RegulatoryChunk).where(
            RegulatoryChunk.user_file_id == root.user_file_id,
            RegulatoryChunk.position == 1,
        )
    ).one()
    annex, last = _annex(intake_session, members)
    previous = batch.labeling_run_id
    assert intake.intake_labeling_failed_counts(intake_session, batch) == (
        {root.user_file_id: 1} if partial else {}
    )
    previous_outcomes = {
        item.id: (
            item.status,
            item.labels,
            item.assignments,
            item.request_hash,
            item.text_snapshot,
        )
        for item in _items(intake_session, previous)
    }
    run_ids = set(intake_session.scalars(select(RegulatoryLabelingRun.id)))
    receipt = editing.edit_review_digest(intake_session, batch)
    assert receipt

    run_id = editing.edit_intake_chunks(
        intake_session,
        batch,
        expected_review_sha256=receipt,
        edits=[
            editing.IntakeChunkEdit(
                root.id,
                first.id,
                {"text": "MADDE 1 — Birlikte kaydedilmiş yeni hüküm."},
            ),
            editing.IntakeChunkEdit(
                root.id, second.id, {"heading_path": ["Ana mevzuat", "MADDE 2"]}
            ),
            editing.IntakeChunkEdit(
                annex.id, last.id, {"validity_start_date": datetime.date(2026, 11, 1)}
            ),
        ],
    )
    assert run_id and run_id != previous and batch.labeling_run_id == run_id
    assert set(intake_session.scalars(select(RegulatoryLabelingRun.id))) - run_ids == {
        run_id
    }
    replacement = intake_session.get_one(RegulatoryLabelingRun, run_id)
    assert set(replacement.file_ids) == {str(member.user_file_id) for member in members}
    assert replacement.total_chunks == 3 and replacement.completed_chunks == 0
    assert len(_items(intake_session, run_id)) == 3
    assert all(
        item.status == "pending" and not item.labels
        for item in _items(intake_session, run_id)
    )
    assert intake.intake_labeling_failed_counts(intake_session, batch) == {}
    assert {
        item.id: (
            item.status,
            item.labels,
            item.assignments,
            item.request_hash,
            item.text_snapshot,
        )
        for item in _items(intake_session, previous)
    } == previous_outcomes
    revisions = list(intake_session.scalars(select(RegulatoryCanonicalRevision)))
    assert len(revisions) == 6
    assert all(
        sum(revision.canonical_chunk_id == chunk.id for revision in revisions) == 2
        for chunk in (first, second, last)
    )
    assert last.validity_start_date == datetime.date(2026, 11, 1)
    assert batch.status == "preparing" and batch.review_sha256 is None
    with pytest.raises(intake.IntakeConflictError):
        intake.approve_intake(intake_session, batch, expected_review_sha256=receipt)


def test_bulk_reuses_unchanged_file_and_rebuilds_derived_rows_once(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, members = _ready(intake_session, intake_owner, monkeypatch, aggregate=True)
    root, first = _root_chunk(intake_session, members)
    second = intake_session.scalars(
        select(RegulatoryChunk).where(
            RegulatoryChunk.user_file_id == root.user_file_id,
            RegulatoryChunk.position == 1,
        )
    ).one()
    annex, unchanged = _annex(intake_session, members)
    previous = next(
        item
        for item in _items(intake_session, batch.labeling_run_id)
        if item.user_file_id == annex.user_file_id
    )
    old_result = (
        previous.labels,
        previous.assignments,
        previous.request_hash,
        previous.request_payload,
    )
    assert batch.review_sha256
    run_id = editing.edit_intake_chunks(
        intake_session,
        batch,
        expected_review_sha256=batch.review_sha256,
        edits=[
            editing.IntakeChunkEdit(
                root.id, first.id, {"text": "MADDE 1 — Yeni birinci hüküm."}
            ),
            editing.IntakeChunkEdit(
                root.id, second.id, {"text": "MADDE 2 — Yeni ikinci hüküm."}
            ),
            editing.IntakeChunkEdit(annex.id, unchanged.id, {"text": unchanged.text}),
        ],
    )
    assert run_id
    items = _items(intake_session, run_id)
    copied = next(item for item in items if item.user_file_id == annex.user_file_id)
    assert copied.status == "completed"
    assert (
        copied.labels,
        copied.assignments,
        copied.request_hash,
        copied.request_payload,
    ) == old_result
    run = intake_session.get_one(RegulatoryLabelingRun, run_id)
    assert run.completed_chunks == 1
    assert labeling.current_labeling_items_for_search(
        intake_session, run=run, items=[copied]
    ) == frozenset({unchanged.id})
    aggregate = intake_session.scalars(
        select(RegulatoryChunk).where(
            RegulatoryChunk.user_file_id == root.user_file_id,
            RegulatoryChunk.chunk_type == "hierarchical_aggregate",
        )
    ).one()
    assert first.text in aggregate.text and second.text in aggregate.text
    revisions = list(intake_session.scalars(select(RegulatoryCanonicalRevision)))
    assert len(revisions) == 6
    assert all(revision.canonical_chunk_id != unchanged.id for revision in revisions)
    assert (
        sum(revision.canonical_chunk_id == aggregate.id for revision in revisions) == 2
    )


def test_bulk_noop_keeps_receipt_run_and_audit_exact(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, members = _ready(intake_session, intake_owner, monkeypatch)
    root, first = _root_chunk(intake_session, members)
    annex, last = _annex(intake_session, members)
    assert batch.review_sha256
    before = _snapshot(intake_session)
    assert (
        editing.edit_intake_chunks(
            intake_session,
            batch,
            expected_review_sha256=batch.review_sha256,
            edits=[
                editing.IntakeChunkEdit(root.id, first.id, {"text": first.text}),
                editing.IntakeChunkEdit(
                    annex.id, last.id, {"validity_start_date": None}
                ),
            ],
        )
        is None
    )
    assert _snapshot(intake_session) == before


@pytest.mark.parametrize(
    "failure",
    [
        "invalid_last",
        "protected_last",
        "foreign_document",
        "wrong_chunk_owner",
        "duplicate",
        "stale",
        "active",
        "live_lease",
        "approved",
        "late_failure",
    ],
)
def test_bulk_invalid_or_stale_edit_rolls_back_the_whole_set(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    batch, members = _ready(intake_session, intake_owner, monkeypatch)
    root, first = _root_chunk(intake_session, members)
    annex, last = _annex(intake_session, members)
    receipt = batch.review_sha256
    assert receipt
    changes = [
        editing.IntakeChunkEdit(
            root.id, first.id, {"text": "Must roll back with the last edit"}
        ),
        editing.IntakeChunkEdit(annex.id, last.id, {"text": "Second corrected source"}),
    ]
    if failure == "invalid_last":
        changes[-1] = editing.IntakeChunkEdit(annex.id, last.id, {"text": " "})
    elif failure == "protected_last":
        changes[-1] = editing.IntakeChunkEdit(
            annex.id, last.id, {"chunk_metadata": {"source_asset_id": str(uuid4())}}
        )
    elif failure == "foreign_document":
        changes[-1] = editing.IntakeChunkEdit(uuid4(), last.id, {"text": "Foreign"})
    elif failure == "wrong_chunk_owner":
        changes[-1] = editing.IntakeChunkEdit(
            root.id, last.id, {"text": "Foreign chunk"}
        )
    elif failure == "duplicate":
        changes[-1] = editing.IntakeChunkEdit(root.id, first.id, {"text": "Duplicate"})
    elif failure == "stale":
        receipt = "0" * 64
    elif failure == "active":
        intake_session.get_one(
            RegulatoryLabelingRun, batch.labeling_run_id
        ).status = "running"
    elif failure == "live_lease":
        run = intake_session.get_one(RegulatoryLabelingRun, batch.labeling_run_id)
        run.lease_token = uuid4()
        run.lease_expires_at = datetime.datetime.now(
            datetime.timezone.utc
        ) + datetime.timedelta(minutes=5)
    elif failure == "approved":
        members[-1].approval_attempt_id = uuid4()
    elif failure == "late_failure":

        def unavailable(*_a: Any, **_k: Any) -> None:
            raise ValueError("Provider binding changed after validation")

        monkeypatch.setattr(
            editing, "resolve_intake_labeling_provider_binding", unavailable
        )
    before = _snapshot(intake_session)
    with pytest.raises((ValueError, intake.IntakeConflictError)):
        editing.edit_intake_chunks(
            intake_session, batch, expected_review_sha256=receipt, edits=changes
        )
    assert _snapshot(intake_session) == before


@pytest.mark.parametrize("count", [0, 257])
def test_bulk_collection_bounds_do_not_change_state(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    count: int,
) -> None:
    batch, members = _ready(intake_session, intake_owner, monkeypatch)
    root, _first = _root_chunk(intake_session, members)
    assert batch.review_sha256
    before = _snapshot(intake_session)
    with pytest.raises(ValueError):
        editing.edit_intake_chunks(
            intake_session,
            batch,
            expected_review_sha256=batch.review_sha256,
            edits=[
                editing.IntakeChunkEdit(
                    root.id, str(uuid4()), {"text": "out of bounds"}
                )
                for _ in range(count)
            ],
        )
    assert _snapshot(intake_session) == before
