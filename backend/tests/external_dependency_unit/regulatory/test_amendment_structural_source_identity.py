"""A reordered own title remains usable without accepting cited instruments."""

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db.models import DocumentSet, RegulatoryChunk, UserFile
from onyx.db.regulatory_amendment_targets import (
    load_amendment_source_identities,
    load_amendment_structural_source_titles,
)
from onyx.regulatory.amendments import search_retriever
from onyx.regulatory.amendments.models import AmendmentInstruction
from tests.external_dependency_unit.regulatory.test_amendment_sources import (
    source_session as source_session,
)
from tests.external_dependency_unit.regulatory.test_annex_baseline import _chunk, _file

_TITLE = "TEHLİKELİ MADDELERİN DEMİRYOLU İLE TAŞINMASI HAKKINDA YÖNETMELİK"
_TARGET = "Tehlikeli Maddelerin Demiryolu ile Taşınması Hakkında Yönetmelik"


def _source(session: Session) -> tuple[DocumentSet, UserFile]:
    group = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    session.add(group)
    session.flush()
    file = _file(session, group)
    file.name = "Başbakanlık Mevzuatı Geliştirme ve Yayın Genel Müdürlüğü.md"
    first = _chunk(session, file, 0, "MADDE 1 - (1) Amaç.")
    first.heading_path = ["Amaç", "MADDE 1 - Amaç", "(1) Amaç."]
    first.chunk_metadata = {"article_no": "1", "paragraph_no": "1"}
    first.chunk_type = "paragraph"
    session.flush()
    return group, file


def _title_descendant(
    session: Session,
    file: UserFile,
    position: int,
    article: str,
    *,
    title: str = _TITLE,
    prefix: tuple[str, ...] = (),
) -> RegulatoryChunk:
    row = _chunk(session, file, position, f"MADDE {article} - (1) Kural.")
    row.heading_path = [
        "Amaç",
        *prefix,
        title,
        f"MADDE {article} - Kural",
        "(1) Kural.",
    ]
    row.chunk_type = "paragraph"
    row.chunk_metadata = {"article_no": article, "paragraph_no": "1"}
    session.flush()
    return row


def _retriever(
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
    group: DocumentSet,
    files: list[UserFile],
) -> tuple[search_retriever.AmendmentSearchRetriever, MagicMock, MagicMock]:
    @contextmanager
    def read_session() -> Iterator[Session]:
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
    title_loader = MagicMock(wraps=load_amendment_structural_source_titles)
    monkeypatch.setattr(
        search_retriever, "load_amendment_structural_source_titles", title_loader
    )
    search = MagicMock(
        side_effect=AssertionError("Exact target must not call SearchTool")
    )
    monkeypatch.setattr(search_retriever, "SearchTool", search)
    retriever = search_retriever.build_amendment_search_retriever(
        session,
        document_set_id=group.id,
        created_by=files[0].user_id,
        user_file_ids=[file.id for file in files],
        llm=MagicMock(),
    )
    return retriever, title_loader, search


@pytest.mark.parametrize(
    "case",
    [
        "own_title",
        "body_citation",
        "clause_heading",
        "numbered_annex",
        "bare_ek",
        "bare_annex",
        "competing_title",
        "same_article",
        "amendment_title",
        "superseded",
        "linked_supersession",
        "companion",
        "aggregate",
        "outside_scope",
    ],
)
def test_structural_identity_requires_one_current_own_title_above_distinct_articles(
    source_session: Session, case: str
) -> None:
    session = source_session
    _, file = _source(session)
    prefixes = {
        "clause_heading": ("MADDE 2", "(1) Dayanak", "a) Atıflar"),
        "numbered_annex": ("EK-1",),
        "bare_ek": ("EK",),
        "bare_annex": ("ANNEX",),
    }
    title = _TITLE
    if case == "amendment_title":
        title = _TITLE.replace(
            "YÖNETMELİK", "YÖNETMELİKTE DEĞİŞİKLİK YAPILMASINA DAİR YÖNETMELİK"
        )
    elif case == "body_citation":
        title = "BAŞKA BİR YÖNETMELİK"
    rows = [
        _title_descendant(
            session,
            file,
            position,
            "5" if case == "same_article" else str(position + 4),
            title=title,
            prefix=prefixes.get(case, ()),
        )
        for position in (1, 2)
    ]
    if case == "body_citation":
        for row in rows:
            row.text += f" {_TITLE} hükümlerine göre işlem yapılır."
    if case == "competing_title":
        _title_descendant(session, file, 3, "7", title="BAŞKA BİR YÖNETMELİK")
    for row in rows:
        if case == "superseded":
            row.status = "superseded"
        elif case == "linked_supersession":
            row.superseded_by_chunk_id = rows[0].id
        elif case == "companion":
            row.chunk_metadata = {
                **row.chunk_metadata,
                "bound_to_regulatory_chunk_id": rows[0].id,
            }
        elif case == "aggregate":
            row.chunk_type = "hierarchical_aggregate"
    session.flush()
    result = load_amendment_structural_source_titles(
        session, [] if case == "outside_scope" else [file.id], _TARGET
    )
    assert result == ({file.id: _TITLE} if case == "own_title" else {})


def test_batch_332_shape_recovers_exact_clause_and_caches_title_lookup(
    source_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = source_session
    group, file = _source(session)
    _title_descendant(session, file, 1, "5")
    paragraph = _title_descendant(session, file, 2, "6")
    paragraph.text = "MADDE 6 - (2) Manevra kuralları."
    paragraph.heading_path = [
        "Amaç",
        _TITLE,
        "MADDE 6 - Kurallar",
        "(2) Manevra kuralları.",
    ]
    paragraph.chunk_metadata = {"article_no": "6", "paragraph_no": "2"}
    target = _chunk(
        session, file, 3, "d) Manevralar gündüz saatleri içerisinde yapılacaktır."
    )
    target.heading_path = [*paragraph.heading_path, "d) Manevralar"]
    target.chunk_metadata = {
        "article_no": "6",
        "paragraph_no": "2",
        "clause_label": "d",
    }
    target.chunk_type = "clause"
    session.flush()
    identity = load_amendment_source_identities(session, [file.id])[0]
    assert identity.root_heading == "Amaç"

    retriever, title_loader, search = _retriever(session, monkeypatch, group, [file])
    instruction = AmendmentInstruction(
        instruction_text="MADDE 1- Aynı Yönetmeliğin 6 ncı maddesinin ikinci fıkrasının (d) bendi yürürlükten kaldırılmıştır.",
        target_source=_TARGET,
    )
    for _ in range(2):
        candidates = retriever.search(instruction)
        assert candidates and candidates[0].chunk_id == target.id
        assert candidates[0].source_verified and candidates[0].structured_match
        assert retriever.last_attention is None
        assert retriever.query_stats[0]["strategy"] == "verified_structure"
    title_loader.assert_called_once()
    search.assert_not_called()


def test_two_recovered_source_versions_still_block_an_ambiguous_addition(
    source_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = source_session
    group, first = _source(session)
    second = _file(session, group)
    second.name = "ikinci-belge.md"
    for file in (first, second):
        _title_descendant(session, file, 1, "5")
        _title_descendant(session, file, 2, "6")
    session.flush()
    retriever, title_loader, search = _retriever(
        session, monkeypatch, group, [first, second]
    )
    result = retriever.search(
        AmendmentInstruction(
            instruction_text="MADDE 1- Aynı Yönetmeliğin 6 ncı maddesinin ikinci fıkrasına aşağıdaki bent eklenmiştir: “ğ) Yeni kural.”",
            target_source=_TARGET,
        )
    )
    assert result == []
    assert "Multiple target source files" in (retriever.last_attention or "")
    title_loader.assert_called_once()
    search.assert_not_called()


@pytest.mark.parametrize(
    "filename, root, title, target",
    [
        (
            "2031-34_makina_guvenligi_tebligi.md",
            "Amaç",
            "Makina Güvenliği Tebliği (2031/4)",
            "Makina Güvenliği Tebliği (2031/4)",
        ),
        (
            "belge.md",
            "Makina Güvenliği Tebliği (2031/34)",
            "Makina Güvenliği Tebliği (2031/4)",
            "Makina Güvenliği Tebliği (2031/4)",
        ),
        (
            "4458_sayili_kanun.md",
            "Amaç",
            "5434 sayılı Kanun",
            "5434 sayılı Kanun",
        ),
        (
            "belge.md",
            "4458 sayılı Kanun",
            "5434 sayılı Kanun",
            "5434 sayılı Kanun",
        ),
    ],
)
def test_recovered_title_cannot_override_filename_or_root_identifiers(
    source_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    root: str,
    title: str,
    target: str,
) -> None:
    session = source_session
    group, file = _source(session)
    file.name = filename
    root_row = _title_descendant(session, file, 1, "2", title=root)
    root_row.position = -1
    root_row.heading_path = [root, "MADDE 2"]
    _title_descendant(session, file, 2, "5", title=title)
    _title_descendant(session, file, 3, "6", title=title)
    session.flush()
    retriever, title_loader, search = _retriever(session, monkeypatch, group, [file])
    result = retriever.search(
        AmendmentInstruction(
            instruction_text="MADDE 1- Aynı metnin 6 ncı maddesine aşağıdaki fıkra eklenmiştir: “(2) Yeni kural.”",
            target_source=target,
        )
    )
    assert result == []
    assert "could not be verified" in (retriever.last_attention or "")
    title_loader.assert_called_once()
    search.assert_not_called()


def test_labeled_clause_retry_recovers_parent_beyond_shortlist_and_drafts_insertion(
    source_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from onyx.db.amendment_match_checkpoints import (
        capture_match_evidence,
        load_match_checkpoint,
        persist_match_checkpoint,
    )
    from onyx.db.models import (
        AmendmentBatch,
        AmendmentMatchCheckpoint,
        AmendmentProposal,
    )
    from onyx.db.regulatory_amendment_order import validate_insertion_order
    from onyx.db.regulatory_amendments import (
        claim_batch_for_analysis,
        reset_batch_attention_for_retry,
    )
    from onyx.regulatory.amendments import drafter, match_checkpoint, pipeline
    from onyx.regulatory.amendments.insertion_order import InsertionOrder
    from onyx.regulatory.amendments.match_checkpoint import MatchedInstruction
    from onyx.regulatory.amendments.models import (
        ChunkFieldsDraft,
        DateResolution,
        DraftResult,
        MatchResult,
    )
    from onyx.regulatory.amendments.ranker import CandidateChunk
    from onyx.regulatory.amendments.structural_target import (
        parse_amendment_structural_target,
    )

    session = source_session
    group, file = _source(session)
    _title_descendant(session, file, 1, "5")
    article_path = ["Amaç", _TITLE, "MADDE 6 - Taşıma kuralları"]
    early_rows = []
    for position, paragraph, clause in [
        (2, "1", None),
        (3, "2", None),
        *[(position, "2", label) for position, label in enumerate("abcçdefgğh", 4)],
    ]:
        marker = f"{clause})" if clause else f"({paragraph})"
        row = _chunk(session, file, position, f"{marker} Mevcut manevra kuralı.")
        row.heading_path = [*article_path, f"({paragraph}) Manevra kuralları", marker]
        row.chunk_type = "clause" if clause else "paragraph"
        row.chunk_metadata = {"article_no": "6", "paragraph_no": paragraph}
        if clause:
            row.chunk_metadata = {**row.chunk_metadata, "clause_label": clause}
        if position == 2:
            row.text = "MADDE 6 - (1) Mevcut taşıma kuralları."
        early_rows.append(row)
    parent = _chunk(
        session, file, 14, "(3) Tehlikeli madde taşıyacak tren teşkili kuralları."
    )
    parent.heading_path = [*article_path, "(3) Tren teşkili kuralları"]
    parent.chunk_type = "paragraph"
    parent.chunk_metadata = {"article_no": "6", "paragraph_no": "3"}
    existing = _chunk(session, file, 15, "c) Mevcut emniyet vagonu kuralı.")
    existing.heading_path = [*parent.heading_path, "c) Emniyet vagonu"]
    existing.chunk_type = "clause"
    existing.chunk_metadata = {
        "article_no": "6",
        "paragraph_no": "3",
        "clause_label": "c",
    }
    following = _chunk(session, file, 16, "(4) Mevcut sevk kuralları.")
    following.heading_path = [*article_path, "(4) Sevk kuralları"]
    following.chunk_type = "paragraph"
    following.chunk_metadata = {"article_no": "6", "paragraph_no": "4"}
    session.flush()
    original_rows = {
        row.id: (row.text, row.position, row.projection_ordinal)
        for row in [*early_rows, parent, existing, following]
    }
    new_text = (
        "ç) Taşınan tehlikeli maddelerin sınıflarına göre gerekli olduğu durumlarda, "
        "vagonlar veya taşıma birimleri (büyük konteyner, portatif tank, tank konteyner, "
        "ÇEGK veya karayolu taşıtı) arasında RID Bölüm 7.5.3’te belirtildiği şekilde "
        "koruyucu mesafe bulunacaktır."
    )
    instruction = AmendmentInstruction(
        instruction_text=(
            "MADDE 1- 16/7/2015 tarihli ve 29418 sayılı Resmî Gazete’de yayımlanan "
            f"{_TARGET.replace('Yönetmelik', 'Yönetmeliğin')} 6 ncı maddesinin üçüncü "
            f"fıkrasına aşağıdaki (ç) bendi eklenmiştir: “{new_text}”"
        ),
        target_source=_TARGET,
    )
    result = MatchResult(
        outcome="new_provision",
        old_chunk_id=None,
        confidence=1,
        rationale="New clause.",
    )
    legacy_candidates = [
        CandidateChunk(
            chunk_id=row.id,
            user_file_id=str(file.id),
            text=row.text,
            metadata={**row.chunk_metadata, "heading_path": row.heading_path},
            source_verified=True,
        )
        for row in early_rows[:6]
    ]
    batch = AmendmentBatch(
        document_set_id=group.id,
        raw_text=instruction.instruction_text,
        status="analyzing",
        stage="processing",
        lease_generation=2,
        instruction_count=5,
        processed_instruction_count=5,
        processed_instruction_indices=list(range(5)),
        user_file_ids=[str(file.id)],
        segmented_instructions=[
            {"instruction_text": f"Completed {index}"} for index in range(4)
        ]
        + [instruction.model_dump(mode="json")],
    )
    session.add(batch)
    session.flush()
    completed = [
        AmendmentProposal(
            batch_id=batch.id,
            instruction_index=index,
            instruction_indices=[index],
            instruction_text=f"Completed {index}",
            instruction_texts=[f"Completed {index}"],
            old_chunk_snapshot={},
            new_chunk_draft={},
            status="pending",
        )
        for index in range(4)
    ]
    session.add_all(completed)
    inputs = {"raw_text": batch.raw_text, "user_file_ids": batch.user_file_ids}
    with monkeypatch.context() as legacy:
        legacy.setattr(match_checkpoint, "MATCH_CONTRACT_VERSION", 3)
        legacy_hash = match_checkpoint.match_input_sha256(**inputs)
    current_hash = match_checkpoint.match_input_sha256(**inputs)
    assert legacy_hash != current_hash
    checkpoint = MatchedInstruction(
        instruction_index=4,
        instruction=instruction,
        candidates=legacy_candidates,
        match=result,
        evidence=capture_match_evidence(
            session, batch.user_file_ids, legacy_candidates
        ),
    )
    assert persist_match_checkpoint(
        session,
        batch_id=batch.id,
        lease_generation=2,
        input_sha256=legacy_hash,
        checkpoint=checkpoint,
    )
    assert (
        load_match_checkpoint(
            session,
            batch_id=batch.id,
            instruction_index=4,
            input_sha256=legacy_hash,
        )
        == checkpoint
    )
    batch.status = "analyzed"
    batch.unmatched_instructions = [
        instruction.instruction_text + "\n\nAttention: Missing parent."
    ]
    session.commit()
    assert reset_batch_attention_for_retry(session, batch_id=batch.id) is not None
    assert batch.processed_instruction_indices == [0, 1, 2, 3]
    assert all(
        session.get(AmendmentProposal, proposal.id) is proposal
        for proposal in completed
    )
    assert (
        load_match_checkpoint(
            session,
            batch_id=batch.id,
            instruction_index=4,
            input_sha256=current_hash,
        )
        is None
    )
    stored = session.get(AmendmentMatchCheckpoint, (batch.id, 4))
    assert stored is not None and stored.input_sha256 == legacy_hash
    lease = claim_batch_for_analysis(session, batch_id=batch.id)
    assert lease is not None
    retriever, title_loader, search = _retriever(session, monkeypatch, group, [file])
    candidates = retriever.search(instruction)
    assert candidates[0].chunk_id == parent.id
    assert retriever.query_stats[0]["strategy"] == "verified_parent"
    target = parse_amendment_structural_target(instruction)
    assert target is not None and target.clause_label == "ç"
    title_loader.assert_called_once()
    search.assert_not_called()
    monkeypatch.setattr(pipeline, "confirm_match", MagicMock(return_value=result))
    match = pipeline.confirm_instruction_match(
        MagicMock(), instruction=instruction, candidates=candidates
    )
    assert match == result
    checkpoint = checkpoint.model_copy(
        update={
            "candidates": candidates,
            "evidence": capture_match_evidence(
                session, batch.user_file_ids, candidates
            ),
        }
    )
    assert persist_match_checkpoint(
        session,
        batch_id=batch.id,
        lease_generation=lease.generation,
        input_sha256=current_hash,
        checkpoint=checkpoint,
    )
    assert (
        load_match_checkpoint(
            session,
            batch_id=batch.id,
            instruction_index=4,
            input_sha256=current_hash,
        )
        == checkpoint
    )
    context = pipeline.load_instruction_draft_context(
        session,
        candidates=candidates,
        match=result,
        instruction=instruction,
    )
    assert context is not None and context.target_user_file_id == file.id
    assert context.old_chunk_snapshot == {} and context.insertion_members
    generated = MagicMock(
        return_value=DraftResult(
            new_chunk=ChunkFieldsDraft(
                text=new_text,
                chunk_type="clause",
                heading_path=article_path,
                metadata_changes={
                    "article_no": "6",
                    "paragraph_no": "2",
                    "clause_label": "c",
                },
            ),
            dates=DateResolution(
                effective_start_date="2023-02-03", rationale="Publication date."
            ),
        )
    )
    monkeypatch.setattr(drafter, "generate_structured", generated)
    proposal = pipeline.draft_instruction_group_proposal(
        MagicMock(),
        instruction_indices=[4],
        instructions=[instruction],
        matches=[result],
        reference_date="2023-02-03",
        context=context,
    )
    draft = proposal.new_chunk_draft
    assert proposal.old_chunk_id is None and proposal.instruction_indices == [4]
    assert draft["user_file_id"] == str(file.id) and draft["text"] == new_text
    assert draft["chunk_type"] == "clause"
    assert {
        key: draft["metadata"][key]
        for key in ("article_no", "paragraph_no", "clause_label")
    } == {
        "article_no": "6",
        "paragraph_no": "3",
        "clause_label": "ç",
    }
    order = InsertionOrder.model_validate(draft["insertion_order"])
    assert order.after_chunk_id == existing.id and order.before_chunk_id == following.id
    assert (
        order.paragraph_no == "3"
        and order.clause_label == "ç"
        and draft["position"] == following.position
    )
    assert validate_insertion_order(session, file.id, order) == order
    assert original_rows == {
        row.id: (row.text, row.position, row.projection_ordinal)
        for row in [*early_rows, parent, existing, following]
    }
    generated.assert_called_once()
