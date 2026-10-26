"""File succession preserves prior content and becomes effective after admission."""

import datetime
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db.document_lineage import (
    activate_intake_lineage,
    file_lineage_snapshot,
    freeze_intake_lineage,
    intake_lineage_selections,
    lineage_review_payload,
    predecessor_candidates,
    reconcile_completed_intake_file,
    reconcile_file_lineage,
    repealed_file_ids,
    set_predecessor_selection,
)
from onyx.db.enums import UserFileStatus
from onyx.db.models import (
    DocumentSet,
    DocumentSet__UserFile,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryDocumentLineageReview,
    User,
    UserFile,
)
from onyx.db.regulatory_chunks import (
    get_bounded_source_lexical_matches,
    get_regulatory_provision_heading_source,
    get_visible_regulatory_chunk_ids,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    _admitted,
    _completed_labels,
    _existing_rows,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_owner as _intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_session as _intake_session,
)

intake_owner = _intake_owner
intake_session = _intake_session


def _previous(
    session: Session,
    owner: User,
    target: DocumentSet,
    title: str = "4000 sayılı Gümrük Kanunu.pdf",
) -> UserFile:
    file = UserFile(
        id=uuid4(),
        user_id=owner.id,
        file_id=f"old-{uuid4()}",
        name=title,
        file_type="text/plain",
        status=UserFileStatus.COMPLETED,
    )
    session.add(file)
    session.flush()
    session.add(DocumentSet__UserFile(document_set_id=target.id, user_file_id=file.id))
    session.add(
        RegulatoryChunk(
            id=f"rc_lineage_old_{uuid4().hex}",
            user_file_id=file.id,
            text="Eski kanunun değişmeyen metni.",
            position=0,
            projection_ordinal=0,
            chunk_type="article",
            heading_path=["MADDE 1"],
            chunk_metadata={"historical": True},
            source="indexed",
            status="active",
        )
    )
    session.flush()
    return file


def test_title_candidate_ignores_number_but_does_not_create_a_relation(
    intake_session: Session, intake_owner: tuple[User, DocumentSet, ModelConfiguration]
) -> None:
    owner, target, model = intake_owner
    batch, members = _admitted(intake_session, owner, target, model)
    previous = _previous(intake_session, owner, target)
    _previous(intake_session, owner, target, "2026 İthalat Denetimi Tebliği.md")
    intake_session.get_one(
        UserFile, members[0].user_file_id
    ).name = "5000 Sayılı Gümrük Kanunu.md"
    result = predecessor_candidates(intake_session, batch, members[0].id)
    assert result[0]["user_file_id"] == str(previous.id)
    assert result[0]["score"] == 100
    assert result[0]["status"] == "current"
    assert "Hukuki ilişkiyi onaylayın" in result[0]["reason"]
    assert intake_lineage_selections(intake_session, batch) == {}
    assert predecessor_candidates(intake_session, batch, members[0].id, query="4000")[
        0
    ]["user_file_id"] == str(previous.id)
    assert (
        predecessor_candidates(intake_session, batch, members[0].id, query="987654")
        == []
    )


def test_repeal_activates_only_after_all_files_complete_and_is_reciprocal(
    intake_session: Session, intake_owner: tuple[User, DocumentSet, ModelConfiguration]
) -> None:
    owner, target, model = intake_owner
    batch, members = _admitted(intake_session, owner, target, model)
    _completed_labels(intake_session, batch, members)
    intake.refresh_intake(intake_session, batch)
    previous = _previous(intake_session, owner, target)
    before = _existing_rows(intake_session, [previous.id])
    effective = datetime.date(2030, 1, 1)
    set_predecessor_selection(
        intake_session,
        batch,
        members[0].id,
        predecessor_user_file_id=previous.id,
        relation="repeals",
        effective_date=effective,
        evidence="5000 sayılı Kanun MADDE 99: 4000 sayılı Kanun yürürlükten kaldırılmıştır.",
    )
    approved = lineage_review_payload(intake_session, batch)
    assert (
        intake_lineage_selections(intake_session, batch)[members[0].id]["status"]
        == "draft"
    )
    assert (
        file_lineage_snapshot(intake_session, target.id, previous.id)["status"]
        == "current"
    )
    freeze_intake_lineage(intake_session, batch)
    assert lineage_review_payload(intake_session, batch) == approved
    activate_intake_lineage(intake_session, batch)
    assert (
        intake_lineage_selections(intake_session, batch)[members[0].id]["status"]
        == "frozen"
    )
    batch.status = "completed"
    with pytest.raises(ValueError, match="tüm yeni dosyalar"):
        activate_intake_lineage(intake_session, batch)
    for member in members:
        intake_session.get_one(
            UserFile, member.user_file_id
        ).status = UserFileStatus.COMPLETED
    intake_session.flush()
    activate_intake_lineage(intake_session, batch)
    activate_intake_lineage(intake_session, batch)
    assert lineage_review_payload(intake_session, batch) == approved
    old = file_lineage_snapshot(
        intake_session, target.id, previous.id, as_of=datetime.date(2029, 12, 31)
    )
    assert old["status"] == "repeal_scheduled"
    assert old["successors"][0]["user_file_id"] == str(members[0].user_file_id)
    assert old["effective_date"] == "2030-01-01"
    assert (
        file_lineage_snapshot(intake_session, target.id, previous.id, as_of=effective)[
            "status"
        ]
        == "repealed"
    )
    new = file_lineage_snapshot(intake_session, target.id, members[0].user_file_id)
    assert new["predecessors"][0]["user_file_id"] == str(previous.id)
    assert new["predecessors"][0]["evidence"].startswith("5000 sayılı")
    assert _existing_rows(intake_session, [previous.id]) == before


@pytest.mark.parametrize(
    "effective,evidence",
    [
        (None, None),
        (datetime.date(2030, 1, 1), None),
        (None, "açık dayanak"),
        (datetime.date(2030, 1, 1), "  "),
    ],
)
def test_repeal_requires_explicit_date_and_evidence(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    effective: datetime.date | None,
    evidence: str | None,
) -> None:
    owner, target, model = intake_owner
    batch, members = _admitted(intake_session, owner, target, model)
    previous = _previous(intake_session, owner, target)
    batch.status, batch.stage = "ready", "review"
    with pytest.raises(ValueError, match="yürürlük tarihi ve açık dayanak"):
        set_predecessor_selection(
            intake_session,
            batch,
            members[0].id,
            predecessor_user_file_id=previous.id,
            relation="repeals",
            effective_date=effective,
            evidence=evidence,
        )
    assert intake_lineage_selections(intake_session, batch) == {}


def test_successor_without_repeal_keeps_old_document_current_and_selection_can_be_cleared(
    intake_session: Session, intake_owner: tuple[User, DocumentSet, ModelConfiguration]
) -> None:
    owner, target, model = intake_owner
    batch, members = _admitted(intake_session, owner, target, model)
    previous = _previous(intake_session, owner, target)
    batch.status, batch.stage = "ready", "review"
    set_predecessor_selection(
        intake_session,
        batch,
        members[0].id,
        predecessor_user_file_id=previous.id,
        relation="successor",
        effective_date=None,
        evidence=None,
    )
    set_predecessor_selection(
        intake_session,
        batch,
        members[0].id,
        predecessor_user_file_id=None,
        relation="successor",
        effective_date=None,
        evidence=None,
    )
    assert intake_lineage_selections(intake_session, batch) == {}
    set_predecessor_selection(
        intake_session,
        batch,
        members[0].id,
        predecessor_user_file_id=previous.id,
        relation="successor",
        effective_date=None,
        evidence=None,
    )
    freeze_intake_lineage(intake_session, batch)
    with pytest.raises(ValueError, match="Onaylanmış"):
        set_predecessor_selection(
            intake_session,
            batch,
            members[0].id,
            predecessor_user_file_id=None,
            relation="successor",
            effective_date=None,
            evidence=None,
        )
    for member in members:
        intake_session.get_one(
            UserFile, member.user_file_id
        ).status = UserFileStatus.COMPLETED
    batch.status = "completed"
    activate_intake_lineage(intake_session, batch)
    assert (
        file_lineage_snapshot(intake_session, target.id, previous.id)["status"]
        == "current"
    )


@pytest.mark.parametrize("change", ["name", "membership", "status"])
def test_changed_predecessor_blocks_freeze_and_changes_review_digest(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    change: str,
) -> None:
    owner, target, model = intake_owner
    batch, members = _admitted(intake_session, owner, target, model)
    previous = _previous(intake_session, owner, target)
    batch.status, batch.stage = "ready", "review"
    set_predecessor_selection(
        intake_session,
        batch,
        members[0].id,
        predecessor_user_file_id=previous.id,
        relation="successor",
        effective_date=None,
        evidence=None,
    )
    reviewed = lineage_review_payload(intake_session, batch)
    if change == "name":
        previous.name += " changed"
    elif change == "membership":
        intake_session.delete(
            intake_session.get_one(DocumentSet__UserFile, (target.id, previous.id))
        )
    else:
        previous.status = UserFileStatus.FAILED
    intake_session.flush()
    assert lineage_review_payload(intake_session, batch) != reviewed
    with pytest.raises(ValueError):
        freeze_intake_lineage(intake_session, batch)
    row = intake_session.get_one(RegulatoryDocumentLineageReview, members[0].id)
    assert row.status == "draft" and row.frozen_at is None


def test_foreign_file_or_own_new_file_cannot_be_predecessor(
    intake_session: Session, intake_owner: tuple[User, DocumentSet, ModelConfiguration]
) -> None:
    owner, target, model = intake_owner
    batch, members = _admitted(intake_session, owner, target, model)
    previous = _previous(intake_session, owner, target)
    batch.status, batch.stage = "ready", "review"
    membership = intake_session.get_one(DocumentSet__UserFile, (target.id, previous.id))
    intake_session.delete(membership)
    intake_session.flush()
    with pytest.raises(ValueError, match="PC Külliyatı"):
        set_predecessor_selection(
            intake_session,
            batch,
            members[0].id,
            predecessor_user_file_id=previous.id,
            relation="successor",
            effective_date=None,
            evidence=None,
        )
    intake_session.get_one(
        UserFile, members[0].user_file_id
    ).status = UserFileStatus.COMPLETED
    intake_session.flush()
    with pytest.raises(ValueError, match="başka bir dosya"):
        set_predecessor_selection(
            intake_session,
            batch,
            members[0].id,
            predecessor_user_file_id=members[0].user_file_id,
            relation="successor",
            effective_date=None,
            evidence=None,
        )
    assert (
        list(
            intake_session.scalars(
                select(RegulatoryDocumentLineageReview).where(
                    RegulatoryDocumentLineageReview.intake_document_id == members[0].id
                )
            )
        )
        == []
    )


def test_frozen_relation_retains_reviewed_identity_if_old_title_later_changes(
    intake_session: Session, intake_owner: tuple[User, DocumentSet, ModelConfiguration]
) -> None:
    owner, target, model = intake_owner
    batch, members = _admitted(intake_session, owner, target, model)
    previous = _previous(intake_session, owner, target)
    batch.status, batch.stage = "ready", "review"
    set_predecessor_selection(
        intake_session,
        batch,
        members[0].id,
        predecessor_user_file_id=previous.id,
        relation="successor",
        effective_date=None,
        evidence=None,
    )
    reviewed = lineage_review_payload(intake_session, batch)
    freeze_intake_lineage(intake_session, batch)
    previous.name = "4000 sayılı Gümrük Kanunu — arşiv"
    for member in members:
        intake_session.get_one(
            UserFile, member.user_file_id
        ).status = UserFileStatus.COMPLETED
    batch.status = "completed"
    activate_intake_lineage(intake_session, batch)
    assert lineage_review_payload(intake_session, batch) == reviewed
    assert (
        file_lineage_snapshot(intake_session, target.id, members[0].user_file_id)[
            "predecessors"
        ][0]["title"]
        == "4000 sayılı Gümrük Kanunu.pdf"
    )


def test_missing_predecessor_membership_blocks_activation_then_retry_only_restores_lineage(
    intake_session: Session, intake_owner: tuple[User, DocumentSet, ModelConfiguration]
) -> None:
    owner, target, model = intake_owner
    batch, members = _admitted(intake_session, owner, target, model)
    _completed_labels(intake_session, batch, members)
    intake.refresh_intake(intake_session, batch)
    previous = _previous(intake_session, owner, target)
    before = _existing_rows(intake_session, [previous.id])
    set_predecessor_selection(
        intake_session,
        batch,
        members[0].id,
        predecessor_user_file_id=previous.id,
        relation="repeals",
        effective_date=datetime.date(2026, 1, 1),
        evidence="Yürürlükten kaldırma maddesi.",
    )
    reviewed = intake.review_digest(intake_session, batch)
    batch.review_sha256 = reviewed
    intake_session.flush()
    intake.approve_intake(intake_session, batch, reviewed)
    for member in members:
        intake_session.get_one(
            UserFile, member.user_file_id
        ).status = UserFileStatus.COMPLETED
    intake_session.delete(
        intake_session.get_one(DocumentSet__UserFile, (target.id, previous.id))
    )
    intake_session.flush()
    intake.refresh_intake(intake_session, batch)
    assert batch.status == "blocked" and batch.stage == "lineage"
    assert batch.error and "PC Külliyatı" in batch.error
    assert (
        intake_lineage_selections(intake_session, batch)[members[0].id]["status"]
        == "frozen"
    )
    published_new = _existing_rows(
        intake_session, [member.user_file_id for member in members]
    )
    intake_session.add(
        DocumentSet__UserFile(document_set_id=target.id, user_file_id=previous.id)
    )
    intake_session.flush()
    assert intake.retry_intake(intake_session, batch) == []
    assert batch.status == "completed"
    assert (
        intake_lineage_selections(intake_session, batch)[members[0].id]["status"]
        == "active"
    )
    assert (
        _existing_rows(intake_session, [member.user_file_id for member in members])
        == published_new
    )
    assert _existing_rows(intake_session, [previous.id]) == before


def test_reading_old_file_reconciles_successful_indexing_without_intake_poll(
    intake_session: Session, intake_owner: tuple[User, DocumentSet, ModelConfiguration]
) -> None:
    owner, target, model = intake_owner
    batch, members = _admitted(intake_session, owner, target, model)
    _completed_labels(intake_session, batch, members)
    intake.refresh_intake(intake_session, batch)
    previous = _previous(intake_session, owner, target)
    set_predecessor_selection(
        intake_session,
        batch,
        members[0].id,
        predecessor_user_file_id=previous.id,
        relation="repeals",
        effective_date=datetime.date(2026, 1, 1),
        evidence="Açık yürürlükten kaldırma maddesi.",
    )
    reviewed = intake.review_digest(intake_session, batch)
    batch.review_sha256 = reviewed
    intake_session.flush()
    intake.approve_intake(intake_session, batch, reviewed)
    for member in members:
        intake_session.get_one(
            UserFile, member.user_file_id
        ).status = UserFileStatus.COMPLETED
    intake_session.flush()
    assert batch.status == "approving"
    reconcile_file_lineage(intake_session, target.id, previous.id)
    assert batch.status == "completed"
    assert (
        file_lineage_snapshot(intake_session, target.id, previous.id)["status"]
        == "repealed"
    )
    assert file_lineage_snapshot(intake_session, target.id, members[0].user_file_id)[
        "predecessors"
    ][0]["user_file_id"] == str(previous.id)


def _reviewed_repeal(
    session: Session, fixture: tuple[User, DocumentSet, ModelConfiguration]
) -> tuple[RegulatoryDocumentIntake, list[RegulatoryDocumentIntakeFile], UserFile]:
    owner, target, model = fixture
    batch, members = _admitted(session, owner, target, model)
    _completed_labels(session, batch, members)
    intake.refresh_intake(session, batch)
    previous = _previous(session, owner, target)
    set_predecessor_selection(
        session,
        batch,
        members[0].id,
        predecessor_user_file_id=previous.id,
        relation="repeals",
        effective_date=datetime.date(2030, 1, 1),
        evidence="Açık yürürlükten kaldırma maddesi.",
    )
    reviewed = intake.review_digest(session, batch)
    batch.review_sha256 = reviewed
    session.flush()
    intake.approve_intake(session, batch, reviewed)
    return batch, members, previous


def test_completion_hook_activates_only_own_intake_and_repeal_query_respects_date(
    intake_session: Session, intake_owner: tuple[User, DocumentSet, ModelConfiguration]
) -> None:
    batch, members, previous = _reviewed_repeal(intake_session, intake_owner)
    before = _existing_rows(intake_session, [previous.id])
    scope = tuple([previous.id, *(member.user_file_id for member in members)])
    assert (
        repealed_file_ids(intake_session, scope, as_of_date=datetime.date(2031, 1, 1))
        == frozenset()
    )
    reconcile_completed_intake_file(intake_session, previous.id)
    assert batch.status == "approving"
    for member in members[:-1]:
        intake_session.get_one(
            UserFile, member.user_file_id
        ).status = UserFileStatus.COMPLETED
        intake_session.flush()
        reconcile_completed_intake_file(intake_session, member.user_file_id)
    assert batch.status == "approving"
    intake_session.get_one(
        UserFile, members[-1].user_file_id
    ).status = UserFileStatus.COMPLETED
    intake_session.flush()
    reconcile_completed_intake_file(intake_session, members[-1].user_file_id)
    assert batch.status == "completed"
    assert (
        repealed_file_ids(intake_session, scope, as_of_date=datetime.date(2029, 12, 31))
        == frozenset()
    )
    assert repealed_file_ids(
        intake_session, scope, as_of_date=datetime.date(2030, 1, 1)
    ) == frozenset({previous.id})
    assert repealed_file_ids(
        intake_session, None, as_of_date=datetime.date(2031, 1, 1)
    ) >= {previous.id}
    assert _existing_rows(intake_session, [previous.id]) == before


def test_current_sql_visibility_and_expansion_hide_repealed_source_but_history_remains(
    intake_session: Session, intake_owner: tuple[User, DocumentSet, ModelConfiguration]
) -> None:
    batch, members, previous = _reviewed_repeal(intake_session, intake_owner)
    for member in members:
        intake_session.get_one(
            UserFile, member.user_file_id
        ).status = UserFileStatus.COMPLETED
    second = RegulatoryChunk(
        id=f"rc_lineage_old_{uuid4().hex}",
        user_file_id=previous.id,
        text="MADDE 2 — Eski gümrük kanunu metni.",
        position=1,
        projection_ordinal=1,
        chunk_type="article",
        heading_path=["MADDE 2"],
        chunk_metadata={},
        source="indexed",
        status="active",
    )
    intake_session.add(second)
    intake_session.flush()
    reconcile_completed_intake_file(intake_session, members[-1].user_file_id)
    assert batch.status == "completed"
    old_ids = list(
        intake_session.scalars(
            select(RegulatoryChunk.id).where(
                RegulatoryChunk.user_file_id == previous.id
            )
        )
    )
    new_ids = list(
        intake_session.scalars(
            select(RegulatoryChunk.id).where(
                RegulatoryChunk.user_file_id.in_(
                    [member.user_file_id for member in members]
                )
            )
        )
    )
    before = _existing_rows(intake_session, [previous.id])
    historic, current = datetime.date(2029, 12, 31), datetime.date(2030, 1, 1)
    assert get_visible_regulatory_chunk_ids(
        intake_session, old_ids + new_ids, as_of_date=historic
    ) == set(old_ids + new_ids)
    assert get_visible_regulatory_chunk_ids(
        intake_session, old_ids + new_ids, as_of_date=current
    ) == set(new_ids)
    assert (
        get_regulatory_provision_heading_source(
            intake_session, old_ids, as_of_date=historic
        )
        is not None
    )
    assert (
        get_regulatory_provision_heading_source(
            intake_session, old_ids, as_of_date=current
        )
        is None
    )
    assert get_bounded_source_lexical_matches(
        intake_session,
        user_file_id=previous.id,
        query="Eski kanunun metni",
        as_of_date=historic,
    )
    assert (
        get_bounded_source_lexical_matches(
            intake_session,
            user_file_id=previous.id,
            query="Eski kanunun metni",
            as_of_date=current,
        )
        == []
    )
    assert _existing_rows(intake_session, [previous.id]) == before
