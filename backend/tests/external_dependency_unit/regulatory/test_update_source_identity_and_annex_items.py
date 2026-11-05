"""Whole-source insertion and own-opening identity against migrated PostgreSQL."""

from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.models import (
    AmendmentBatch,
    AmendmentProposal,
    DocumentSet,
    RegulatoryChunk,
    UserFile,
)
from onyx.db.regulatory_amendment_document_identity import (
    IDENTITY_KEY,
    load_document_identity,
    validate_document_identity,
)
from onyx.db.regulatory_amendment_order import (
    validate_insertion_order,
)
from onyx.db.regulatory_amendments import (
    approve_amendment_proposal,
    finalize_amendment_proposal_projection,
)
from onyx.db.regulatory_publication import PublicationStore
from onyx.regulatory.amendments.insertion_order import InsertionOrder
from onyx.regulatory.amendments.models import (
    AmendmentInstruction,
    ChunkFieldsDraft,
    DateResolution,
    DraftResult,
    MatchResult,
    ProposalDraft,
)
from onyx.regulatory.amendments.pipeline import (
    _build_proposal_draft,
    load_instruction_draft_context,
)
from onyx.regulatory.amendments.ranker import CandidateChunk
from tests.external_dependency_unit.regulatory.test_annex_baseline import _chunk, _file
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    LabelingDatabase,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)
from tests.unit.onyx.regulatory.indexing_jobs.owned_publication_test_helpers import (
    OwnedAuthority,
)

TEXT = "MADDE 2- Aynı Tebliğin Ek I’ine aşağıdaki madde eklenmiştir:\n“23. Kontrol ünitesi: Motoru çalıştıran üniteyi ifade eder.”"


def setup_list(
    session: Session,
) -> tuple[DocumentSet, UserFile, list[RegulatoryChunk], ProposalDraft]:
    group = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    session.add(group)
    session.flush()
    file = _file(session, group)
    header = _chunk(
        session,
        file,
        0,
        "Tanımlar\nEk II ila Ek VI için aşağıdaki tanımlar geçerlidir.",
    )
    header.chunk_type = "free_text"
    header.chunk_metadata = {"appendix_label": "EK I"}
    header.heading_path = ["Kaynak", "EK I"]
    rows = [header]
    for i in range(1, 23):
        row = _chunk(session, file, i, f"{i}. Mevcut tanım{i}.")
        row.chunk_type = "paragraph"
        row.chunk_metadata = {"appendix_label": "EK I", "paragraph_no": str(i)}
        row.heading_path = ["Kaynak", "EK I", row.text]
        rows.append(row)
    next_annex = _chunk(session, file, 23, "EK II - Ürünler")
    next_annex.chunk_metadata = {"appendix_label": "EK II"}
    next_annex.heading_path = ["Kaynak", "EK II"]
    rows.append(next_annex)
    session.flush()
    instruction = AmendmentInstruction(
        instruction_text=TEXT,
        target_source="Örnek Ürünler Tebliği",
        article_reference="Ek I Madde 23",
    )
    candidate = CandidateChunk(
        chunk_id=header.id,
        user_file_id=str(file.id),
        text=header.text,
        metadata={**header.chunk_metadata, "heading_path": header.heading_path},
        source_verified=True,
        structured_match=True,
    )
    match = MatchResult(
        outcome="new_provision",
        old_chunk_id=None,
        confidence=1,
        rationale="verified source",
    )
    context = load_instruction_draft_context(
        session, candidates=[candidate], match=match, instruction=instruction
    )
    assert context is not None and context.expected_new_article_no is None
    result = _build_proposal_draft(
        instruction_indices=[2],
        instructions=[instruction],
        matches=[match],
        context=context,
        draft=DraftResult(
            new_chunk=ChunkFieldsDraft(
                text="23. Kontrol ünitesi: Motoru çalıştıran üniteyi ifade eder.",
                chunk_type="paragraph",
            ),
            dates=DateResolution(
                effective_start_date="2026-10-16", rationale="published"
            ),
        ),
    )
    return group, file, rows, result


def test_numbered_annex_item_drafts_and_approves_as_a_new_record_after_22(
    labeling_database: LabelingDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    with Session(labeling_database.engine) as session:
        group, file, rows, draft = setup_list(session)
        before = [(r.id, r.text, r.projection_ordinal, r.status) for r in rows]
        assert draft.old_chunk_id is None and draft.new_chunk_draft["position"] == 23
        assert [(r.id, r.text, r.projection_ordinal, r.status) for r in rows] == before
        batch = AmendmentBatch(
            document_set_id=group.id, raw_text=TEXT, user_file_ids=[str(file.id)]
        )
        session.add(batch)
        session.flush()
        proposal = AmendmentProposal(
            batch_id=batch.id,
            instruction_index=2,
            instruction_text=TEXT,
            old_chunk_id=None,
            old_chunk_snapshot={},
            new_chunk_draft=draft.new_chunk_draft,
            status="approving",
        )
        session.add(proposal)
        session.flush()
        authority = OwnedAuthority(file.id)
        monkeypatch.setattr(PublicationStore, "lock_owned_snapshot", lambda *_: None)
        monkeypatch.setattr(
            PublicationStore,
            "allocate_in_session",
            lambda _store, _session, owner, key: authority.allocate(owner, key),
        )
        result = approve_amendment_proposal(
            session, proposal, publication_owner=authority.owner
        )
        assert finalize_amendment_proposal_projection(
            session, proposal_id=proposal.id, succeeded=True
        )
        session.flush()
        session.expire_all()
        active = list(
            session.scalars(
                select(RegulatoryChunk)
                .where(
                    RegulatoryChunk.user_file_id == file.id,
                    RegulatoryChunk.status == "active",
                )
                .order_by(RegulatoryChunk.position)
            )
        )
        assert [r.id for r in active] == [
            *(r.id for r in rows[:-1]),
            result.new_chunk.id,
            rows[-1].id,
        ]
        assert (
            active[23].chunk_metadata["paragraph_no"] == "23"
            and active[23].chunk_metadata.get("article_no") is None
        )
        assert [(r.id, r.text, r.projection_ordinal, r.status) for r in rows] == before


def test_approval_rejects_changed_annex_neighbors(
    labeling_database: LabelingDatabase,
) -> None:
    with Session(labeling_database.engine) as session:
        _, file, rows, draft = setup_list(session)
        order = InsertionOrder.model_validate(draft.new_chunk_draft["insertion_order"])
        rows[-2].text += " Changed after review."
        session.flush()
        with pytest.raises(ValueError, match="changed after review"):
            validate_insertion_order(session, file.id, order)


@pytest.mark.parametrize("change", ["value", "opening"])
def test_own_opening_identity_cannot_be_forged_or_reused_after_source_change(
    labeling_database: LabelingDatabase, change: str
) -> None:
    with Session(labeling_database.engine) as session:
        group = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
        session.add(group)
        session.flush()
        file = _file(session, group)
        title = "ÖRNEK KAYNAK USULLERİ HAKKINDA YÖNETMELİK"
        row = _chunk(
            session,
            file,
            0,
            f"**{title}**\n*Ticaret Bakanlığından:20.12.2025 tarih ve33113 sayılı R.G.*",
        )
        row.heading_path = [title]
        session.flush()
        identity = load_document_identity(session, file.id)
        assert identity is not None
        metadata = {**identity.fields, IDENTITY_KEY: identity.model_dump(mode="json")}
        validate_document_identity(session, file.id, metadata)
        if change == "value":
            metadata["document_number"] = "2017/10411"
        else:
            row.text += " Edited source."
            session.flush()
        with pytest.raises(ValueError, match="Document identity changed"):
            validate_document_identity(session, file.id, metadata)
