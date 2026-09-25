"""Canonical source and subunit lookup against PostgreSQL, rolled back per test."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db.models import DocumentSet
from onyx.db.regulatory_chunks import get_active_chunks_by_structural_reference
from onyx.regulatory.amendments import search_retriever
from onyx.regulatory.amendments.models import AmendmentInstruction
from tests.external_dependency_unit.regulatory.test_amendment_sources import (
    source_session as source_session,
)
from tests.external_dependency_unit.regulatory.test_annex_baseline import _chunk, _file


def test_real_annex_list_item_uses_verified_source_without_search_model(
    source_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = source_session
    group = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    session.add(group)
    session.flush()
    correct, wrong = _file(session, group), _file(session, group)
    correct.name = "2031-04_makina_guvenligi_tebligi.md"
    wrong.name = "2031-34_makina_guvenligi_tebligi.md"
    expected = _chunk(
        session,
        correct,
        0,
        "9. En az üç fotoğraf.",
        appendix_label="EK 7",
        paragraph_no="9",
    )
    expected.heading_path = ["Makina Güvenliği Tebliği (2031/4)", "EK 7", "9. Belgeler"]
    foreign = _chunk(
        session, wrong, 0, "9. Fotoğraflar.", appendix_label="EK 7", paragraph_no="9"
    )
    foreign.heading_path = ["Makina Güvenliği Tebliği (2031/34)", "EK 7", "9. Belgeler"]
    session.flush()

    @contextmanager
    def read_session():
        yield session

    monkeypatch.setattr(
        search_retriever, "get_session_with_current_tenant", read_session
    )
    monkeypatch.setattr(
        search_retriever, "get_current_search_settings", lambda _: MagicMock()
    )
    monkeypatch.setattr(
        search_retriever, "get_default_document_index", lambda *_: MagicMock()
    )
    monkeypatch.setattr(
        search_retriever,
        "get_tools",
        lambda _: [
            SimpleNamespace(id=1, in_code_tool_id=search_retriever.SEARCH_TOOL_ID)
        ],
    )
    search = MagicMock(
        side_effect=AssertionError("Exact canonical lookup must not call SearchTool")
    )
    monkeypatch.setattr(search_retriever, "SearchTool", search)
    retriever = search_retriever.build_amendment_search_retriever(
        session,
        document_set_id=group.id,
        created_by=correct.user_id,
        user_file_ids=[wrong.id, correct.id],
        llm=MagicMock(),
    )
    result = retriever.search(
        AmendmentInstruction(
            instruction_text="MADDE 12- Aynı Tebliğin Ek-7’sinde yer alan listenin 9 uncu maddesinde yer alan “fotoğraflar” ibaresi “en az üç fotoğraf” şeklinde değiştirilmiştir.",
            target_source="Makina Güvenliği Tebliği (2031/4)",
        )
    )
    assert [candidate.chunk_id for candidate in result] == [expected.id]
    assert result[0].source_verified and result[0].structured_match
    search.assert_not_called()


def test_null_parent_and_appended_duplicate_survive_real_lookup_shortlisting(
    source_session: Session,
) -> None:
    from dataclasses import replace

    from onyx.db.regulatory_amendment_targets import load_amendment_source_chunks
    from onyx.regulatory.amendments.ranker import CandidateChunk
    from onyx.regulatory.amendments.structural_target import (
        AmendmentStructuralTarget,
        deterministic_structural_candidate,
    )
    from onyx.regulatory.amendments.target_scope import reconcile_structural_candidates

    session = source_session
    group = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    session.add(group)
    session.flush()
    file = _file(session, group)
    file.name = "2031-04_makina_guvenligi_tebligi.md"
    definitions = [
        ("MADDE 8- (2) Tanımlar.", {"article_no": "8", "paragraph_no": "2"}),
        ("d) Fiili denetim: Belge kontrolü.", {"article_no": "8", "clause_label": "d"}),
        ("MADDE 9- (1) Başvurular.", {"article_no": "9", "paragraph_no": "1"}),
        (
            "d) Dijital kayıt: Elektronik bilgi.",
            {"article_no": "8", "paragraph_no": "2", "clause_label": "d"},
        ),
    ]
    rows = []
    for position, (text, metadata) in enumerate(definitions):
        item = _chunk(session, file, position, text)
        item.chunk_metadata = metadata
        rows.append(item)
    session.flush()
    narrow = get_active_chunks_by_structural_reference(
        session,
        user_file_ids=[file.id],
        article_no="8",
        paragraph_no="2",
        clause_label="d",
        appendix_label=None,
        source_name_hint=None,
    )
    assert [match.chunk.id for match in narrow] == [rows[3].id]
    candidates = [
        CandidateChunk(
            chunk_id=match.chunk.id,
            user_file_id=str(file.id),
            text=match.chunk.text,
            metadata=dict(match.chunk.chunk_metadata),
            source_verified=True,
        )
        for match in load_amendment_source_chunks(session, file.id)
    ]
    result = reconcile_structural_candidates(
        [replace(candidates[3], structured_match=True)],
        candidates,
        AmendmentStructuralTarget(article_no="8", paragraph_no="2", clause_label="d"),
    )
    assert [item.chunk_id for item in result[:2]] == [rows[1].id, rows[3].id]
    assert all(item.structure_conflict for item in result)
    assert (
        deterministic_structural_candidate(
            AmendmentInstruction(
                instruction_text="MADDE 2- Aynı Tebliğin 8 inci maddesinin ikinci fıkrasının (d) bendi değiştirilmiştir.",
                target_source="Makina Güvenliği Tebliği (2031/4)",
            ),
            result[:1],
        )
        is None
    )
    assert rows[1].chunk_metadata.get("paragraph_no") is None


def test_clause_lookup_keeps_parent_paragraph_and_turkish_identity(
    source_session: Session,
) -> None:
    session = source_session
    group = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    session.add(group)
    session.flush()
    file = _file(session, group)
    for position, (parent, clause) in enumerate(
        [(1, "g"), (1, "ğ"), (2, "ğ"), (1, "ı"), (1, "i")]
    ):
        row = _chunk(session, file, position, f"{clause}) Definition.")
        row.chunk_type = "clause"
        row.chunk_metadata = {
            "article_no": "8",
            "paragraph_no": str(parent),
            "clause_label": clause,
        }
    session.flush()
    for letter in ("ğ", "ı", "i"):
        rows = get_active_chunks_by_structural_reference(
            session,
            user_file_ids=[file.id],
            article_no="8",
            paragraph_no="1",
            clause_label=letter,
            appendix_label=None,
            source_name_hint=None,
        )
        assert len(rows) == 1
        assert rows[0].chunk.chunk_metadata == {
            "article_no": "8",
            "paragraph_no": "1",
            "clause_label": letter,
        }


def test_reviewed_insertion_preserves_ids_ordinals_and_historical_sibling_order(
    source_session: Session,
) -> None:
    from sqlalchemy import select

    from onyx.db.models import RegulatoryChunk
    from onyx.db.regulatory_amendment_order import (
        apply_insertion_order,
        load_amendment_order,
        validate_insertion_order,
    )
    from onyx.regulatory.amendments.insertion_order import plan_insertion

    session = source_session
    group = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    session.add(group)
    session.flush()
    file, other_file = _file(session, group), _file(session, group)
    rows = [
        _chunk(
            session,
            file,
            i,
            f"({i + 1}) Rule.",
            article_no="8",
            paragraph_no=str(i + 1),
        )
        for i in range(3)
    ]
    historical = _chunk(
        session, file, 3, "Historic next article.", article_no="9", paragraph_no="1"
    )
    historical.status = "superseded"
    current = _chunk(
        session, file, 4, "Current next article.", article_no="9", paragraph_no="1"
    )
    current.position = 3
    foreign = _chunk(session, other_file, 3, "Foreign source.")
    session.flush()
    before = {
        row.id: (row.text, row.projection_ordinal)
        for row in [*rows, historical, current, foreign]
    }
    order = plan_insertion(
        load_amendment_order(session, file.id),
        article_no="8",
        paragraph_no="4",
        clause_label=None,
    )
    assert order.position == 3 and order.after_chunk_id == rows[-1].id
    validate_insertion_order(session, file.id, order)
    original_text = rows[0].text
    rows[0].text = "Changed parent rule during review."
    session.flush()
    with pytest.raises(ValueError, match="changed after review"):
        validate_insertion_order(session, file.id, order)

    rows[0].text = original_text
    session.flush()
    added = _chunk(
        session, file, 100, "(4) Added rule.", article_no="8", paragraph_no="4"
    )
    added.position = order.position
    session.flush()
    apply_insertion_order(session, file.id, order, added.id)
    session.expire_all()
    assert added.position == 3 and current.position == historical.position == 4
    assert foreign.position == 3
    for identifier, expected in before.items():
        row = session.get(RegulatoryChunk, identifier)
        assert row is not None and (row.text, row.projection_ordinal) == expected
    assert [
        row.id
        for row in session.scalars(
            select(RegulatoryChunk)
            .where(
                RegulatoryChunk.user_file_id == file.id,
                RegulatoryChunk.status == "active",
            )
            .order_by(RegulatoryChunk.position)
        )
    ] == [*[row.id for row in rows], added.id, current.id]
    with pytest.raises(ValueError, match="already exists|changed after review"):
        validate_insertion_order(session, file.id, order)


@pytest.mark.parametrize(
    "opening,direct", [("MADDE 8- Tanımlar:", True), ("MADDE 8- (1) Tanımlar:", False)]
)
def test_direct_clause_order_requires_actual_source_opening(
    source_session: Session,
    opening: str,
    direct: bool,
) -> None:
    from onyx.db.regulatory_amendment_order import (
        load_amendment_order,
        validate_insertion_order,
    )
    from onyx.regulatory.amendments.insertion_order import plan_insertion

    session = source_session
    group = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    session.add(group)
    session.flush()
    file = _file(session, group)
    for position, (body, label) in enumerate(
        [(opening, None), ("a) Mevcut tanım.", "a"), ("b) Diğer tanım.", "b")]
    ):
        item = _chunk(session, file, position, body)
        item.chunk_metadata = {"article_no": "8", "clause_label": label}
    session.flush()
    members = load_amendment_order(session, file.id)
    assert members[0].direct_clause_parent is direct
    if direct:
        order = plan_insertion(
            members, article_no="8", paragraph_no=None, clause_label="c"
        )
        assert order.paragraph_no is None and order.position == 3
        validate_insertion_order(session, file.id, order)
    else:
        with pytest.raises(ValueError, match="parent"):
            plan_insertion(members, article_no="8", paragraph_no=None, clause_label="c")


def test_approval_rejects_retained_heading_review_that_recreates_removed_children(
    source_session: Session,
) -> None:
    from onyx.db.models import AmendmentProposal
    from onyx.db.regulatory_amendments import _approve_multi_chunk_proposal
    from onyx.regulatory.amendments.pipeline import _chunk_to_review_dict
    from tests.unit.onyx.regulatory.indexing_jobs.owned_publication_test_helpers import (
        OwnedAuthority,
    )

    session = source_session
    group = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    session.add(group)
    session.flush()
    file = _file(session, group)
    parent = _chunk(session, file, 0, "(1) Eski kural.")
    parent.chunk_type = "paragraph"
    parent.chunk_metadata = {"article_no": "8", "paragraph_no": "1"}
    parent.heading_path = ["Kaynak", "MADDE 8 - Eski başlık", "(1) Eski kural"]
    child = _chunk(session, file, 1, "a) Eski alt kural.")
    child.chunk_type = "clause"
    child.chunk_metadata = {**parent.chunk_metadata, "clause_label": "a"}
    child.heading_path = [*parent.heading_path, "a) Eski alt kural"]
    session.flush()
    changes = []
    for row in (parent, child):
        snapshot = _chunk_to_review_dict(row)
        snapshot["heading_change"] = {"article_no": "8", "title": "Yeni başlık"}
        changes.append(
            {
                "old_chunk_id": row.id,
                "old_chunk_snapshot": snapshot,
                "new_chunk_draft": {
                    "user_file_id": str(file.id),
                    "position": row.position,
                    "text": "(1) Yeni tam kural." if row is parent else row.text,
                    "chunk_type": row.chunk_type,
                    "heading_path": row.heading_path,
                    "metadata": row.chunk_metadata,
                    "effective_start_date": "2027-01-01",
                    "effective_end_date": None,
                },
                "instruction_indices": [0],
                "instruction_texts": [
                    "MADDE 1- Aynı Kanunun 8 inci maddesinin birinci fıkrası aşağıdaki şekilde değiştirilmiştir. “(1) Yeni tam kural.”"
                ],
            }
        )
    proposal = AmendmentProposal(
        chunk_changes=changes,
        old_chunk_snapshot={
            "heading_change_scope": {
                "article_no": "8",
                "chunk_ids": [parent.id, child.id],
            }
        },
    )
    with pytest.raises(ValueError, match="complete replacement.*descendant"):
        _approve_multi_chunk_proposal(
            session,
            proposal,
            publication_owner=OwnedAuthority(file.id).owner,
        )
    assert parent.status == child.status == "active"


def test_heading_and_full_parent_replacement_retire_consumed_children(
    source_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from onyx.db.models import AmendmentBatch, AmendmentProposal
    from onyx.db.regulatory_amendments import _approve_multi_chunk_proposal
    from onyx.db.regulatory_publication import PublicationStore
    from onyx.regulatory.amendments.compound_heading import attach_heading_changes
    from onyx.regulatory.amendments.models import ProposalDraft
    from onyx.regulatory.amendments.pipeline import _chunk_to_review_dict
    from tests.unit.onyx.regulatory.indexing_jobs.owned_publication_test_helpers import (
        OwnedAuthority,
    )

    session = source_session
    group = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    session.add(group)
    session.flush()
    file = _file(session, group)
    parent = _chunk(session, file, 0, "(1) Eski kural.")
    parent.chunk_type = "paragraph"
    parent.chunk_metadata = {"article_no": "8", "paragraph_no": "1"}
    parent.heading_path = ["Kaynak", "MADDE 8 - Eski başlık", "Fıkra 1"]
    child = _chunk(session, file, 1, "a) Eski alt kural.")
    child.chunk_type = "clause"
    child.chunk_metadata = {**parent.chunk_metadata, "clause_label": "a"}
    child.heading_path = [*parent.heading_path, "a) Eski alt kural"]
    sibling = _chunk(session, file, 2, "(2) Korunan kural.")
    sibling.chunk_type = "paragraph"
    sibling.chunk_metadata = {"article_no": "8", "paragraph_no": "2"}
    sibling.heading_path = ["Kaynak", "MADDE 8 - Eski başlık", "Fıkra 2"]
    session.flush()
    snapshots = [_chunk_to_review_dict(row) for row in (parent, child, sibling)]
    texts = [
        "MADDE 1- Kanunun 8 inci maddesinin başlığı “Yeni başlık” şeklinde değiştirilmiştir.",
        "MADDE 2- Kanunun 8 inci maddesinin birinci fıkrası aşağıdaki şekilde değiştirilmiştir. “(1) Yeni tam kural.”",
    ]
    reviewed = attach_heading_changes(
        ProposalDraft(
            instruction_index=0,
            instruction_indices=[0, 1],
            instruction_text=texts[0],
            instruction_texts=texts,
            old_chunk_id=parent.id,
            old_chunk_snapshot={**snapshots[0], "descendant_snapshots": [snapshots[1]]},
            new_chunk_draft={
                "user_file_id": str(file.id),
                "position": parent.position,
                "text": "(1) Yeni tam kural.",
                "chunk_type": "paragraph",
                "heading_path": parent.heading_path,
                "metadata": parent.chunk_metadata,
                "effective_start_date": "2027-01-01",
                "effective_end_date": None,
            },
            match_confidence=1,
            match_rationale="verified",
            date_rationale="explicit",
        ),
        snapshots=snapshots,
        article_no="8",
        title="Yeni başlık",
    )
    assert [change.old_chunk_id for change in reviewed.chunk_changes] == [
        parent.id,
        sibling.id,
    ]
    batch = AmendmentBatch(
        document_set_id=group.id,
        raw_text="owned",
        status="analyzed",
        user_file_ids=[str(file.id)],
    )
    session.add(batch)
    session.flush()
    proposal = AmendmentProposal(
        batch_id=batch.id, status="approving", **reviewed.model_dump(mode="json")
    )
    session.add(proposal)
    session.flush()
    authority = OwnedAuthority(file.id)
    monkeypatch.setattr(
        PublicationStore,
        "allocate_in_session",
        lambda _store, _session, owner, key: authority.allocate(owner, key),
    )
    result = _approve_multi_chunk_proposal(
        session, proposal, publication_owner=authority.owner
    )
    assert len(result.new_chunks) == 2
    assert parent.status == child.status == sibling.status == "superseded"
    assert child.superseded_by_chunk_id == result.new_chunks[0].id
    assert child.validity_end_date == result.new_chunks[0].validity_start_date
    assert [row.text for row in result.new_chunks] == [
        "(1) Yeni tam kural.",
        "(2) Korunan kural.",
    ]
    assert all(
        row.chunk_metadata["article_title"] == "Yeni başlık"
        for row in result.new_chunks
    )
    assert child.text == "a) Eski alt kural."


def test_already_applied_outcome_is_persistable_without_changing_chunks(
    source_session: Session,
) -> None:
    # Exercise the actual migration on a transaction-local copy, never ALTER
    # the live proposal table just to test an undeployed schema change.
    import importlib.util
    from datetime import date
    from pathlib import Path

    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import select, text

    from onyx.db.models import AmendmentBatch, AmendmentProposal
    from onyx.db.regulatory_amendments import persist_proposal_checkpoint
    from onyx.regulatory.amendments.models import ProposalDraft
    from onyx.regulatory.amendments.pipeline import _chunk_to_review_dict

    session = source_session
    session.execute(
        text(
            "CREATE TEMP TABLE amendment_proposal (LIKE public.amendment_proposal INCLUDING ALL) ON COMMIT DROP"
        )
    )
    spec = importlib.util.spec_from_file_location(
        "outcome_migration",
        Path(__file__).resolve().parents[3]
        / "alembic/versions/f6a1c9d27b40_amendment_already_applied_outcome.py",
    )
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    with Operations.context(MigrationContext.configure(session.connection())):
        migration.upgrade()
    group = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    session.add(group)
    session.flush()
    file = _file(session, group)
    parent = _chunk(session, file, 0, "Previous wording")
    current = _chunk(session, file, 1, "Current wording")
    parent.status = "superseded"
    parent.validity_end_date = date(2031, 1, 1)
    parent.superseded_by_chunk_id = current.id
    current.source = "amendment"
    current.validity_start_date = date(2031, 1, 1)
    current.supersedes_chunk_id = parent.id
    parent.position = current.position
    batch = AmendmentBatch(
        document_set_id=group.id,
        raw_text="Current wording",
        status="analyzing",
        stage="processing",
        lease_generation=1,
        instruction_count=1,
    )
    session.add(batch)
    session.flush()
    snapshot = _chunk_to_review_dict(current)
    draft = {
        key: snapshot[key]
        for key in (
            "user_file_id",
            "position",
            "text",
            "chunk_type",
            "heading_path",
            "metadata",
        )
    }
    draft.update(effective_start_date="2031-01-01", effective_end_date=None)
    proposal = ProposalDraft(
        instruction_index=0,
        instruction_text="Update wording",
        instruction_indices=[0],
        instruction_texts=["Update wording"],
        old_chunk_id=current.id,
        old_chunk_snapshot=snapshot,
        new_chunk_draft=draft,
    )
    assert persist_proposal_checkpoint(
        session, batch_id=batch.id, lease_generation=1, proposal=proposal
    )
    stored = session.scalar(
        select(AmendmentProposal).where(AmendmentProposal.batch_id == batch.id)
    )
    assert stored is not None and stored.status == "already_applied"
    assert stored.decided_by is None and stored.applied_new_chunk_id is None
    session.refresh(current)
    assert _chunk_to_review_dict(current) == snapshot
    from onyx.db.regulatory_amendments import _proposal_is_already_applied

    parent.superseded_by_chunk_id = None
    session.flush()
    assert not _proposal_is_already_applied(session, proposal)
