from datetime import date
from uuid import uuid4

import pytest
from sqlalchemy import delete
from sqlalchemy.orm import Session

from onyx.db.enums import UserFileStatus
from onyx.db.models import (
    AmendmentBatch,
    AmendmentProposal,
    DocumentSet,
    RegulatoryChunk,
    User,
    UserFile,
)
from onyx.db.regulatory_amendments import approve_amendment_proposal
from onyx.db.regulatory_chunks import (
    get_current_chunks_by_ids,
    make_regulatory_chunk_id,
)
from tests.external_dependency_unit.conftest import create_test_user


def test_stale_chunk_identity_resolves_across_multiple_approved_versions(
    db_session: Session,
    tenant_context: None,  # noqa: ARG001
) -> None:
    user = create_test_user(db_session, "amendment_lineage")
    user_file_id = uuid4()
    user_file = UserFile(
        id=user_file_id,
        user_id=user.id,
        file_id=f"amendment_lineage_{uuid4().hex}",
        name="amendment-lineage.md",
        file_type="text/markdown",
        status=UserFileStatus.COMPLETED,
    )
    old = RegulatoryChunk(
        id=f"rc_{uuid4().hex}",
        user_file_id=user_file_id,
        text="(1) Toplam sayı sekizi geçemez.",
        position=132,
        chunk_type="paragraph",
        heading_path=["MADDE 20", "(1) Toplam sayı sekizi geçemez"],
        chunk_metadata={"article_no": "20", "paragraph_no": "1"},
        status="superseded",
        source="indexed",
        projection_ordinal=132,
        validity_end_date=date(2026, 7, 4),
    )
    middle = RegulatoryChunk(
        id=f"rc_{uuid4().hex}",
        user_file_id=user_file_id,
        text="(1) Toplam sayı yediyi geçemez.",
        position=132,
        chunk_type="paragraph",
        heading_path=["MADDE 20", "(1) Toplam sayı yediyi geçemez"],
        chunk_metadata={"article_no": "20", "paragraph_no": "1"},
        status="superseded",
        source="amendment",
        projection_ordinal=1_000_000_001,
        validity_start_date=date(2026, 7, 4),
        validity_end_date=date(2026, 8, 1),
        supersedes_chunk_id=old.id,
    )
    current = RegulatoryChunk(
        id=f"rc_{uuid4().hex}",
        user_file_id=user_file_id,
        text="(1) Toplam sayı sekizi geçemez.",
        position=132,
        chunk_type="paragraph",
        heading_path=["MADDE 20", "(1) Toplam sayı sekizi geçemez"],
        chunk_metadata={"article_no": "20", "paragraph_no": "1"},
        status="active",
        source="amendment",
        projection_ordinal=1_000_000_002,
        validity_start_date=date(2026, 8, 1),
        supersedes_chunk_id=middle.id,
    )
    old.superseded_by_chunk_id = middle.id
    middle.superseded_by_chunk_id = current.id
    db_session.add_all([user_file, old, middle, current])
    db_session.commit()

    try:
        resolved = get_current_chunks_by_ids(
            db_session, [old.id, middle.id, current.id]
        )

        assert resolved == {old.id: current, middle.id: current, current.id: current}
    finally:
        db_session.rollback()
        db_session.execute(
            delete(RegulatoryChunk).where(RegulatoryChunk.user_file_id == user_file_id)
        )
        db_session.execute(delete(UserFile).where(UserFile.id == user_file_id))
        persisted_user = db_session.get(User, user.id)
        if persisted_user is not None:
            db_session.delete(persisted_user)
        db_session.commit()


def test_approval_rechecks_descendants_added_after_analysis(
    db_session: Session,
    tenant_context: None,  # noqa: ARG001
) -> None:
    user = create_test_user(db_session, "amendment_descendant_guard")
    document_set = DocumentSet(
        name=f"amendment-descendant-guard-{uuid4().hex}",
        description="Approval-time descendant guard regression test",
        user_id=user.id,
        is_public=False,
        is_up_to_date=True,
    )
    user_file_id = uuid4()
    user_file = UserFile(
        id=user_file_id,
        user_id=user.id,
        file_id=f"amendment_descendant_guard_{uuid4().hex}",
        name="amendment-descendant-guard.md",
        file_type="text/markdown",
        status=UserFileStatus.COMPLETED,
    )
    parent_heading = ["MADDE 20", "(3) Yediden fazla idare için"]
    parent = RegulatoryChunk(
        id=f"rc_{uuid4().hex}",
        user_file_id=user_file_id,
        text="(3) Yediden fazla idare için aşağıdaki yöntemler uygulanır:",
        position=136,
        chunk_type="paragraph",
        heading_path=parent_heading,
        chunk_metadata={"article_no": "20", "paragraph_no": "3"},
        status="active",
        source="indexed",
        projection_ordinal=136,
    )
    db_session.add_all([document_set, user_file, parent])
    db_session.flush()
    batch = AmendmentBatch(
        document_set_id=document_set.id,
        raw_text="20 nci maddenin üçüncü fıkrası değiştirilmiştir.",
        user_file_ids=[str(user_file_id)],
        segmented_instructions=[],
        unmatched_instructions=[],
        status="analyzed",
        stage="finalizing",
        instruction_count=1,
        processed_instruction_count=1,
        processed_instruction_indices=[0],
    )
    db_session.add(batch)
    db_session.flush()
    instruction = (
        "20 nci maddenin üçüncü fıkrası aşağıdaki şekilde değiştirilmiştir. "
        "“(3) Sekizden fazla idare için aşağıdaki yöntemler uygulanır.”"
    )
    proposal = AmendmentProposal(
        batch_id=batch.id,
        instruction_index=0,
        instruction_text=instruction,
        instruction_indices=[0],
        instruction_texts=[instruction],
        old_chunk_id=parent.id,
        old_chunk_snapshot={
            "id": parent.id,
            "chunk_type": "paragraph",
            "heading_path": parent_heading,
            "metadata": {"article_no": "20", "paragraph_no": "3"},
        },
        new_chunk_draft={
            "user_file_id": str(user_file_id),
            "position": 136,
            "text": "(3) Sekizden fazla idare için aşağıdaki yöntemler uygulanır.",
            "chunk_type": "paragraph",
            "heading_path": parent_heading,
            "metadata": {"article_no": "20", "paragraph_no": "3"},
        },
        status="approving",
    )
    db_session.add(proposal)
    db_session.commit()

    child = RegulatoryChunk(
        id=f"rc_{uuid4().hex}",
        user_file_id=user_file_id,
        text="a) Birinci yöntem.",
        position=137,
        chunk_type="clause",
        heading_path=["MADDE 20", "(3) Eski başlık", "a) Birinci yöntem"],
        chunk_metadata={
            "article_no": "20",
            "paragraph_no": "3",
            "clause_label": "a",
        },
        status="active",
        source="indexed",
        projection_ordinal=137,
    )
    db_session.add(child)
    db_session.commit()

    try:
        with pytest.raises(ValueError, match="active descendant chunks"):
            approve_amendment_proposal(db_session, proposal, decided_by=user.id)
        db_session.rollback()
        persisted_parent = db_session.get(RegulatoryChunk, parent.id)
        persisted_proposal = db_session.get(AmendmentProposal, proposal.id)
        assert persisted_parent is not None
        assert persisted_parent.status == "active"
        assert persisted_proposal is not None
        assert persisted_proposal.applied_new_chunk_id is None
    finally:
        db_session.rollback()
        db_session.execute(
            delete(AmendmentProposal).where(AmendmentProposal.batch_id == batch.id)
        )
        db_session.execute(delete(AmendmentBatch).where(AmendmentBatch.id == batch.id))
        db_session.execute(
            delete(RegulatoryChunk).where(RegulatoryChunk.user_file_id == user_file_id)
        )
        db_session.execute(delete(UserFile).where(UserFile.id == user_file_id))
        db_session.execute(delete(DocumentSet).where(DocumentSet.id == document_set.id))
        persisted_user = db_session.get(User, user.id)
        if persisted_user is not None:
            db_session.delete(persisted_user)
        db_session.commit()


def test_approval_persists_same_text_version_and_retry_is_idempotent(
    db_session: Session,
    tenant_context: None,  # noqa: ARG001
) -> None:
    user = create_test_user(db_session, "amendment_same_text")
    document_set = DocumentSet(
        name=f"amendment-same-text-{uuid4().hex}",
        description="Same-text amendment approval regression test",
        user_id=user.id,
        is_public=False,
        is_up_to_date=True,
    )
    user_file_id = uuid4()
    user_file = UserFile(
        id=user_file_id,
        user_id=user.id,
        file_id=f"amendment_same_text_{uuid4().hex}",
        name="amendment-same-text.md",
        file_type="text/markdown",
        status=UserFileStatus.COMPLETED,
    )
    text = "Taşıt onay belgesi beş yıl süreyle geçerli olmak üzere düzenlenir."
    position = 34
    old_chunk_id = make_regulatory_chunk_id(user_file_id, position, text)
    old_chunk = RegulatoryChunk(
        id=old_chunk_id,
        user_file_id=user_file_id,
        text=text,
        position=position,
        chunk_type="article",
        heading_path=["MADDE 15"],
        chunk_metadata={"chunk_variant": "atomic"},
        status="active",
        source="indexed",
        projection_ordinal=position,
        validity_start_date=date(2020, 1, 1),
    )
    db_session.add_all([document_set, user_file, old_chunk])
    db_session.flush()

    batch = AmendmentBatch(
        document_set_id=document_set.id,
        raw_text="Geçerlilik başlangıç tarihi değiştirilmiştir.",
        user_file_ids=[str(user_file_id)],
        segmented_instructions=[],
        unmatched_instructions=[],
        status="analyzed",
        stage="finalizing",
        instruction_count=1,
        processed_instruction_count=1,
        processed_instruction_indices=[0],
    )
    db_session.add(batch)
    db_session.flush()
    proposal = AmendmentProposal(
        batch_id=batch.id,
        instruction_index=0,
        instruction_text=batch.raw_text,
        instruction_indices=[0],
        instruction_texts=[batch.raw_text],
        old_chunk_id=old_chunk_id,
        old_chunk_snapshot={},
        new_chunk_draft={
            "user_file_id": str(user_file_id),
            "position": position,
            "text": text,
            "chunk_type": "article",
            "heading_path": ["MADDE 15"],
            "metadata": {"chunk_variant": "atomic"},
            "effective_start_date": "2026-07-04",
            "effective_end_date": None,
        },
        status="approving",
    )
    db_session.add(proposal)
    db_session.commit()

    try:
        first_result = approve_amendment_proposal(
            db_session,
            proposal,
            decided_by=user.id,
        )
        new_chunk_id = first_result.new_chunk.id
        db_session.commit()
        db_session.expire_all()

        persisted_old = db_session.get(RegulatoryChunk, old_chunk_id)
        persisted_new = db_session.get(RegulatoryChunk, new_chunk_id)
        persisted_proposal = db_session.get(AmendmentProposal, proposal.id)
        assert persisted_old is not None
        assert persisted_new is not None
        assert persisted_proposal is not None
        assert new_chunk_id != old_chunk_id
        assert persisted_old.status == "superseded"
        assert persisted_old.validity_end_date == date(2026, 7, 4)
        assert persisted_old.superseded_by_chunk_id == new_chunk_id
        assert persisted_new.supersedes_chunk_id == old_chunk_id
        assert persisted_new.text == persisted_old.text
        assert persisted_proposal.applied_new_chunk_id == new_chunk_id

        retry_result = approve_amendment_proposal(
            db_session,
            persisted_proposal,
            decided_by=user.id,
        )
        db_session.commit()
        assert retry_result.new_chunk.id == new_chunk_id
        assert (
            db_session.query(RegulatoryChunk)
            .filter(RegulatoryChunk.user_file_id == user_file_id)
            .count()
            == 2
        )
    finally:
        db_session.rollback()
        db_session.execute(
            delete(AmendmentProposal).where(AmendmentProposal.batch_id == batch.id)
        )
        db_session.execute(delete(AmendmentBatch).where(AmendmentBatch.id == batch.id))
        db_session.execute(
            delete(RegulatoryChunk).where(RegulatoryChunk.user_file_id == user_file_id)
        )
        db_session.execute(delete(UserFile).where(UserFile.id == user_file_id))
        db_session.execute(delete(DocumentSet).where(DocumentSet.id == document_set.id))
        persisted_user = db_session.get(User, user.id)
        if persisted_user is not None:
            db_session.delete(persisted_user)
        db_session.commit()
