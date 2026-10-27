"""A reviewed imported annex can refer to an existing main without changing it."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.configs.constants import FileOrigin
from onyx.db import document_intake as intake
from onyx.db import regulatory_labeling as labeling
from onyx.db.enums import UserFileStatus
from onyx.db.models import (
    AmendmentSourcePackage,
    DocumentSet,
    DocumentSet__UserFile,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryFilePublication,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatorySourceAsset,
    User,
    UserFile,
    UserFileProjectionRepair,
)
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from tests.external_dependency_unit.regulatory.test_document_intake import (
    _existing_rows,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_owner as intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_session as intake_session,
)
from tests.external_dependency_unit.regulatory.test_document_intake_review_acceptance import (
    _decision,
    _table_rows,
    _three_document_review,
)


def _completed_parent(
    session: Session, owner: tuple[User, DocumentSet, ModelConfiguration]
) -> UserFile:
    user, target, _model = owner
    content = b"%PDF-1.7\nExisting-original-source-must-not-change"
    file_id = PostgresBackedFileStore().save_file(
        BytesIO(content),
        "20220618-8.htm.pdf",
        FileOrigin.USER_FILE,
        "application/pdf",
        db_session=session,
    )
    file = UserFile(
        id=uuid4(),
        user_id=user.id,
        file_id=file_id,
        name="Tehlikeli Maddelerin Karayoluyla Taşınması Hakkında Yönetmelik",
        file_type="application/pdf",
        status=UserFileStatus.COMPLETED,
        regulatory_chunk_generation_hash="c" * 64,
    )
    session.add(file)
    session.flush()
    session.add(DocumentSet__UserFile(document_set_id=target.id, user_file_id=file.id))
    session.add(
        RegulatoryChunk(
            id=f"rc_existing_main_{uuid4().hex}",
            user_file_id=file.id,
            text="MADDE 1 — Mevcut ana belgenin yayımlanmış metni.",
            position=0,
            projection_ordinal=0,
            heading_path=["MADDE 1"],
            chunk_metadata={
                "document_title": file.name,
                "document_date": "2022-06-18",
                "raw_sha256": sha256(content).hexdigest(),
            },
            chunk_type="article",
            status="active",
            source="indexed",
        )
    )
    session.add(
        RegulatoryFilePublication(
            user_file_id=file.id,
            scope_key="existing-parent-regression",
            owner_id=uuid4(),
            fencing_token=1,
            lease_expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
            epoch=2,
            gate_closed=False,
            next_ordinal=1,
        )
    )
    session.flush()
    return file


def _parent_publication(session: Session, parent: UserFile) -> str:
    row = (
        session.execute(
            select(RegulatoryFilePublication.__table__).where(
                RegulatoryFilePublication.user_file_id == parent.id,
            )
        )
        .mappings()
        .one()
    )
    return json.dumps(dict(row), sort_keys=True, default=str)


def _complete_child_labels(
    session: Session, batch: RegulatoryDocumentIntake
) -> RegulatoryLabelingRun:
    run = session.get_one(RegulatoryLabelingRun, batch.labeling_run_id)
    lease = labeling.claim_run(
        session,
        run_id=run.id,
        expected_generation=run.lease_generation,
        lease_seconds=300,
    )
    assert lease
    labeling.prepare_next_item_page(session, lease, limit=128)
    items = list(
        session.scalars(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == run.id
            )
        )
    )
    assert items
    for item in items:
        item.status, item.labels, item.error = "completed", ["Gümrük"], None
        assignment: dict[str, object] = {
            "label": "Gümrük",
            "evidence": item.text_snapshot,
        }
        item.assignments = [assignment]
        item.request_hash = item.request_hash or uuid4().hex * 2
    run.status, run.stage = "completed", "finished"
    run.completed_chunks, run.failed_chunks = len(items), 0
    run.lease_token, run.lease_expires_at = None, None
    session.flush()
    intake.refresh_intake(session, batch)
    assert batch.status == "ready" and batch.review_sha256
    return run


def test_standalone_annex_labels_and_approves_only_new_file_with_frozen_external_parent(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.db import document_intake_review as review

    session = intake_session
    user, target, _model = intake_owner
    parent = _completed_parent(session, intake_owner)
    before = _existing_rows(session, [parent.id])
    parent_publication = _parent_publication(session, parent)
    batch, members, previous, _binding = _three_document_review(
        session, intake_owner, monkeypatch, standalone_annex=True
    )
    child = members["annex"]
    assert child.role == "root" and child.parent_document_id is None
    assert set(session.scalars(select(UserFile.id))) == {parent.id, child.user_file_id}
    graph = (child.role, child.parent_document_id, list(child.relationships))
    source = _table_rows(
        session, AmendmentSourcePackage, RegulatorySourceAsset, RegulatoryChunk
    )
    run_id = _decision(session, batch, child, user, existing_parent=parent.id)
    assert run_id and run_id != previous.id
    run = session.get_one(RegulatoryLabelingRun, run_id)
    assert run.file_ids == [str(child.user_file_id)]
    assert any(
        item.status != "completed"
        for item in session.scalars(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == run_id
            )
        )
    )
    _complete_child_labels(session, batch)
    selected = review.latest_publication_decisions(session, batch.id)[child.id]
    assert selected.existing_parent_file_id == parent.id
    assert selected.existing_parent_snapshot
    assert (
        review.file_attachment_relations(session, target.id, parent.id)["attachments"]
        == []
    )
    assert (
        review.file_attachment_relations(session, target.id, child.user_file_id)[
            "existing_parents"
        ]
        == []
    )
    assert _existing_rows(session, [parent.id]) == before
    assert (
        _table_rows(
            session, AmendmentSourcePackage, RegulatorySourceAsset, RegulatoryChunk
        )
        == source
    )
    assert (child.role, child.parent_document_id, child.relationships) == graph
    reviewed = batch.review_sha256
    assert reviewed
    deliveries = intake.approve_intake(session, batch, reviewed)
    assert {identifier for identifier, _token in deliveries} == {child.user_file_id}
    assert session.get(UserFileProjectionRepair, parent.id) is None
    assert _existing_rows(session, [parent.id]) == before
    assert session.get(DocumentSet__UserFile, (target.id, parent.id))
    session.get_one(UserFile, child.user_file_id).status = UserFileStatus.COMPLETED
    session.get_one(RegulatoryFilePublication, child.user_file_id).epoch = 2
    session.flush()
    intake.refresh_intake(session, batch)
    assert batch.status == "completed"
    assert _existing_rows(session, [parent.id]) == before
    assert _parent_publication(session, parent) == parent_publication
    links = review.file_attachment_relations(session, target.id, parent.id)
    assert [item["user_file_id"] for item in links["attachments"]] == [
        str(child.user_file_id)
    ]
    child_links = review.file_attachment_relations(
        session, target.id, child.user_file_id
    )
    assert [item["user_file_id"] for item in child_links["existing_parents"]] == [
        str(parent.id)
    ]


def test_completed_external_link_keeps_frozen_review_after_parent_later_changes(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.db import document_intake_review as review
    from onyx.db.models import RegulatoryDocumentLineageReview

    session = intake_session
    user, target, _model = intake_owner
    parent = _completed_parent(session, intake_owner)
    batch, members, _previous, _binding = _three_document_review(
        session, intake_owner, monkeypatch, standalone_annex=True
    )
    child = members["annex"]
    _decision(session, batch, child, user, existing_parent=parent.id)
    _complete_child_labels(session, batch)
    reviewed = batch.review_sha256
    assert reviewed
    intake.approve_intake(session, batch, reviewed)
    session.get_one(UserFile, child.user_file_id).status = UserFileStatus.COMPLETED
    session.get_one(RegulatoryFilePublication, child.user_file_id).epoch = 2
    session.flush()
    intake.refresh_intake(session, batch)
    decision = review.latest_publication_decisions(session, batch.id)[child.id]
    snapshot = json.dumps(decision.existing_parent_snapshot, sort_keys=True)
    parent.name += " güncel başlık"
    chunk = session.scalar(
        select(RegulatoryChunk).where(RegulatoryChunk.user_file_id == parent.id)
    )
    assert chunk
    chunk.text += " Ana belgenin sonraki geçerli değişikliği."
    session.flush()
    assert intake.review_digest(session, batch) == reviewed
    assert json.dumps(decision.existing_parent_snapshot, sort_keys=True) == snapshot
    assert review.file_attachment_relations(session, target.id, parent.id)[
        "attachments"
    ][0]["user_file_id"] == str(child.user_file_id)
    assert list(session.scalars(select(RegulatoryDocumentLineageReview))) == []


def test_ordinary_labeling_without_intake_keeps_current_completed_snapshot(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = intake_session
    user, target, model = intake_owner
    ordinary = _completed_parent(session, intake_owner)
    _batch, _members, seed, binding = _three_document_review(
        session, intake_owner, monkeypatch
    )
    run, created = labeling.create_labeling_run(
        session,
        document_set_id=target.id,
        taxonomy=seed.taxonomy,
        model_configuration_id=model.id,
        model=binding.model,
        provider_binding=binding.model_dump(mode="json"),
        requested_by_id=user.id,
        idempotency_key=uuid4(),
        target_file_ids=[ordinary.id],
    )
    assert created
    lease = labeling.claim_run(
        session, run_id=run.id, expected_generation=0, lease_seconds=300
    )
    assert lease
    items = labeling.prepare_next_item_page(session, lease, limit=8)
    assert len(items) == 1
    item = items[0]
    assert "intake_parent_identity" not in item.source_snapshot
    assert "Reviewed main-document identity" not in (item.context_snapshot or "")
    item.status, item.labels = "completed", ["Gümrük"]
    assignment: dict[str, object] = {"label": "Gümrük", "evidence": item.text_snapshot}
    item.assignments = [assignment]
    run.status, run.stage, run.completed_chunks = "completed", "finished", 1
    run.lease_token, run.lease_expires_at = None, None
    session.flush()
    assert labeling.current_labeling_items_for_search(
        session, run=run, items=items
    ) == {item.regulatory_chunk_id}


@pytest.mark.parametrize(
    "change",
    ["name", "canonical", "membership", "closed_gate", "incomplete", "source_file"],
)
def test_changed_existing_parent_blocks_approval_without_any_child_or_parent_index_request(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    from onyx.db import document_intake_review as review

    session = intake_session
    user, target, _model = intake_owner
    parent = _completed_parent(session, intake_owner)
    batch, members, _previous, _binding = _three_document_review(
        session, intake_owner, monkeypatch, standalone_annex=True
    )
    _decision(session, batch, members["annex"], user, existing_parent=parent.id)
    _complete_child_labels(session, batch)
    reviewed = batch.review_sha256
    assert reviewed
    if change == "name":
        parent.name += " changed"
    elif change == "canonical":
        chunk = session.scalar(
            select(RegulatoryChunk).where(RegulatoryChunk.user_file_id == parent.id)
        )
        assert chunk
        chunk.text += " changed"
    elif change == "membership":
        session.delete(session.get_one(DocumentSet__UserFile, (target.id, parent.id)))
    elif change == "closed_gate":
        session.get_one(RegulatoryFilePublication, parent.id).gate_closed = True
    elif change == "incomplete":
        parent.status = UserFileStatus.CHUNKED
    else:
        parent.file_id = "different-original-source"
    session.flush()
    before = _table_rows(
        session, UserFileProjectionRepair, UserFile, DocumentSet__UserFile
    )
    with pytest.raises((ValueError, intake.IntakeConflictError)):
        intake.approve_intake(session, batch, reviewed)
    assert (
        _table_rows(session, UserFileProjectionRepair, UserFile, DocumentSet__UserFile)
        == before
    )
    assert (
        review.latest_publication_decisions(session, batch.id)[
            members["annex"].id
        ].existing_parent_file_id
        == parent.id
    )


@pytest.mark.parametrize(
    "invalid", ["foreign_set", "new_intake_member", "same_batch_parent", "self"]
)
def test_existing_parent_selection_rejects_nonpublic_unpublished_or_conflicting_parent(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    invalid: str,
) -> None:
    session = intake_session
    user, target, _model = intake_owner
    parent = _completed_parent(session, intake_owner)
    batch, members, _previous, _binding = _three_document_review(
        session,
        intake_owner,
        monkeypatch,
        standalone_annex=invalid != "same_batch_parent",
    )
    member = members["annex"]
    if invalid == "foreign_set":
        session.delete(session.get_one(DocumentSet__UserFile, (target.id, parent.id)))
        other = DocumentSet(
            name=f"foreign-{uuid4()}", user_id=user.id, is_up_to_date=True
        )
        session.add(other)
        session.flush()
        session.add(
            DocumentSet__UserFile(document_set_id=other.id, user_file_id=parent.id)
        )
    elif invalid == "new_intake_member":
        _other, other_members, _run, _binding = _three_document_review(
            session, intake_owner, monkeypatch
        )
        parent = session.get_one(UserFile, other_members["root"].user_file_id)
    elif invalid == "self":
        parent = session.get_one(UserFile, member.user_file_id)
    session.flush()
    before = _table_rows(
        session,
        UserFileProjectionRepair,
        UserFile,
        RegulatoryChunk,
        RegulatoryLabelingRun,
    )
    with pytest.raises((ValueError, intake.IntakeConflictError)):
        _decision(session, batch, member, user, existing_parent=parent.id)
    assert (
        _table_rows(
            session,
            UserFileProjectionRepair,
            UserFile,
            RegulatoryChunk,
            RegulatoryLabelingRun,
        )
        == before
    )
