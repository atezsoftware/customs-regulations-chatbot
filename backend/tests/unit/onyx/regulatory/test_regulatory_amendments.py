from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from sqlalchemy.orm import Session

from onyx.db.models import AmendmentProposal, RegulatoryChunk
from onyx.db.regulatory_amendments import (
    approve_amendment_proposal,
    queue_amendment_proposal_approval,
    reject_proposal,
)
from onyx.db.regulatory_chunks import get_active_chunks_by_ids, get_next_chunk_position
from onyx.regulatory.amendments import candidate_finder, pipeline
from onyx.regulatory.amendments.drafter import DraftResult
from onyx.regulatory.amendments.models import (
    AmendmentInstruction,
    ChunkFieldsDraft,
    DateResolution,
    MatchResult,
    ReviewedAmendmentChunkDraft,
    SegmentationResult,
)
from onyx.regulatory.amendments.new_provision_policy import (
    explicitly_adds_top_level_provision,
)
from onyx.regulatory.amendments.ranker import CandidateChunk
from onyx.regulatory.amendments.segmenter import propagate_target_sources


def test_amendment_candidate_queries_exclude_hierarchical_aggregates() -> None:
    for statement in (
        candidate_finder._TEXT_TRGM_SQL,
        candidate_finder._HEADING_TRGM_SQL,
        candidate_finder._STRUCTURED_SQL,
    ):
        assert "chunk_type IS DISTINCT FROM 'hierarchical_aggregate'" in str(statement)


def test_review_snapshot_contains_the_complete_existing_chunk_state() -> None:
    chunk = cast(
        RegulatoryChunk,
        SimpleNamespace(
            id="old-chunk",
            user_file_id=UUID("00000000-0000-0000-0000-000000000123"),
            position=15,
            text="MADDE 15 eski metin",
            chunk_type="article",
            heading_path=["TIR İşlemleri", "MADDE 15"],
            chunk_metadata={"article_no": "15"},
            validity_start_date=date(2010, 12, 31),
            validity_end_date=None,
            status="active",
            source="indexed",
            supersedes_chunk_id=None,
            superseded_by_chunk_id=None,
            created_at=datetime(2010, 12, 31, tzinfo=timezone.utc),
            updated_at=datetime(2026, 8, 28, tzinfo=timezone.utc),
        ),
    )

    assert pipeline._chunk_to_review_dict(chunk) == {
        "id": "old-chunk",
        "user_file_id": "00000000-0000-0000-0000-000000000123",
        "position": 15,
        "text": "MADDE 15 eski metin",
        "chunk_type": "article",
        "heading_path": ["TIR İşlemleri", "MADDE 15"],
        "metadata": {"article_no": "15"},
        "validity_start_date": "2010-12-31",
        "validity_end_date": None,
        "status": "active",
        "source": "indexed",
        "supersedes_chunk_id": None,
        "superseded_by_chunk_id": None,
        "created_at": "2010-12-31T00:00:00+00:00",
        "updated_at": "2026-08-28T00:00:00+00:00",
    }


def test_exact_search_projection_ids_load_canonical_active_chunks() -> None:
    row = SimpleNamespace(id="chunk-1")
    scalars = MagicMock()
    scalars.all.return_value = [row]
    db_session = MagicMock(spec=Session)
    db_session.scalars.return_value = scalars

    chunks = get_active_chunks_by_ids(
        db_session,
        ["chunk-1", "chunk-1"],
    )

    assert chunks == {"chunk-1": row}
    statement = str(db_session.scalars.call_args.args[0])
    assert "regulatory_chunk.status" in statement
    assert "IS DISTINCT FROM" in statement


def test_candidate_lookup_keeps_large_dataset_out_of_llm_context() -> None:
    user_file_id = UUID("00000000-0000-0000-0000-000000000123")
    rows = [
        SimpleNamespace(
            id=f"chunk-{index}",
            user_file_id=user_file_id,
            text=f"MADDE {index} düzenlemesi",
            chunk_metadata={},
            score=1 - index / 100,
        )
        for index in range(15)
    ]
    result = MagicMock()
    result.all.return_value = rows
    db_session = MagicMock(spec=Session)
    db_session.execute.side_effect = [MagicMock(), result]

    candidates = candidate_finder.find_candidates(
        db_session,
        user_file_ids=[user_file_id],
        instruction=AmendmentInstruction(
            instruction_text="MADDE 1 düzenlemesi değiştirilmiştir."
        ),
    )

    assert len(candidates) == 5
    query_parameters = db_session.execute.call_args_list[1].args[1]
    assert query_parameters["limit"] == 15
    assert query_parameters["user_file_ids"] == [user_file_id]


def test_new_provision_position_uses_database_max_without_loading_chunks() -> None:
    db_session = MagicMock(spec=Session)
    db_session.scalar.return_value = 14_999

    position = get_next_chunk_position(
        db_session, UUID("00000000-0000-0000-0000-000000000123")
    )

    assert position == 15_000
    db_session.scalar.assert_called_once()


def test_analyze_instruction_without_candidates_skips_llm_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruction = AmendmentInstruction(instruction_text="MADDE 7 değiştirilmiştir.")
    monkeypatch.setattr(pipeline, "find_candidates", lambda *_args, **_kwargs: [])
    confirm_match = MagicMock()
    draft_new_chunk = MagicMock()
    monkeypatch.setattr(pipeline, "confirm_match", confirm_match)
    monkeypatch.setattr(pipeline, "draft_new_chunk", draft_new_chunk)

    proposal = pipeline.analyze_instruction(
        MagicMock(spec=Session),
        llm=MagicMock(),
        user_file_ids=[UUID("00000000-0000-0000-0000-000000000123")],
        instruction_index=0,
        instruction=instruction,
        reference_date=None,
    )

    assert proposal is None
    confirm_match.assert_not_called()
    draft_new_chunk.assert_not_called()


def test_amendment_approval_rejects_derived_aggregate_target() -> None:
    proposal = MagicMock()
    proposal.id = 41
    proposal.status = "approving"
    proposal.old_chunk_id = "aggregate-id"
    proposal.new_chunk_draft = {
        "user_file_id": str(UUID("00000000-0000-0000-0000-000000000123")),
        "position": 3,
        "text": "Yeni metin",
    }
    aggregate = MagicMock()
    aggregate.status = "active"
    aggregate.chunk_type = "hierarchical_aggregate"
    aggregate.chunk_metadata = {"chunk_variant": "hierarchical_aggregate"}
    db_session = MagicMock(spec=Session)
    db_session.scalar.side_effect = [proposal, aggregate]

    with pytest.raises(
        ValueError, match="Derived aggregate chunks cannot be amended directly"
    ):
        approve_amendment_proposal(
            db_session,
            proposal,
            decided_by=None,
        )

    db_session.add.assert_not_called()


def test_amendment_approval_locks_and_refreshes_stale_proposal_and_old_chunk_rows() -> (
    None
):
    submitted_proposal = cast(
        AmendmentProposal,
        SimpleNamespace(
            id=42,
            status="approving",
            old_chunk_id="old-chunk",
            new_chunk_draft={
                "user_file_id": "00000000-0000-0000-0000-000000000123",
                "position": 3,
                "text": "Yeni metin",
                "effective_start_date": "2026-08-15",
            },
        ),
    )
    locked_proposal = SimpleNamespace(
        id=42,
        status="approving",
        old_chunk_id="old-chunk",
        new_chunk_draft={
            "user_file_id": "00000000-0000-0000-0000-000000000123",
            "position": 3,
            "text": "Yeni metin",
            "effective_start_date": "2026-08-15",
        },
    )
    old_chunk = SimpleNamespace(
        id="old-chunk",
        status="active",
        chunk_type=None,
        chunk_metadata={},
    )
    db_session = MagicMock(spec=Session)
    db_session.scalar.side_effect = [locked_proposal, old_chunk]

    result = approve_amendment_proposal(
        db_session,
        submitted_proposal,
        decided_by=None,
    )

    assert result.proposal is locked_proposal
    assert old_chunk.status == "superseded"
    assert old_chunk.validity_end_date == date(2026, 8, 15)
    assert result.new_chunk.validity_start_date == date(2026, 8, 15)
    assert result.new_chunk.supersedes_chunk_id == "old-chunk"
    assert old_chunk.superseded_by_chunk_id == result.new_chunk.id
    lock_queries = [str(call.args[0]) for call in db_session.scalar.call_args_list]
    assert len(lock_queries) == 2
    assert all("FOR UPDATE" in query for query in lock_queries)
    assert all(
        call.args[0].get_execution_options()["populate_existing"] is True
        for call in db_session.scalar.call_args_list
    )


def test_amendment_approval_applies_reviewed_draft_atomically() -> None:
    stored_draft = {
        "user_file_id": "00000000-0000-0000-0000-000000000123",
        "position": 15,
        "text": "MADDE 15 eski öneri",
        "chunk_type": "article",
        "heading_path": ["TIR İşlemleri", "MADDE 15"],
        "metadata": {"article_no": "15"},
        "effective_start_date": None,
        "effective_end_date": None,
    }
    reviewed_draft = {
        **stored_draft,
        "text": "MADDE 15 - (2) a) (Mülga)",
        "effective_start_date": "2026-08-28",
    }
    submitted_proposal = cast(
        AmendmentProposal,
        SimpleNamespace(id=46, status="pending", old_chunk_id="old-chunk"),
    )
    locked_proposal = SimpleNamespace(
        id=46,
        status="pending",
        old_chunk_id="old-chunk",
        old_chunk_snapshot={"id": "old-chunk"},
        new_chunk_draft=stored_draft,
    )
    old_chunk = SimpleNamespace(
        id="old-chunk",
        status="active",
        chunk_type="article",
        chunk_metadata={"chunk_variant": "atomic"},
    )
    db_session = MagicMock(spec=Session)
    db_session.scalar.side_effect = [locked_proposal, locked_proposal, old_chunk]

    queue_amendment_proposal_approval(
        db_session,
        submitted_proposal,
        decided_by=None,
        reviewed_new_chunk_draft=reviewed_draft,
    )
    result = approve_amendment_proposal(
        db_session,
        submitted_proposal,
        decided_by=None,
    )

    assert locked_proposal.new_chunk_draft == reviewed_draft
    assert result.new_chunk.text == "MADDE 15 - (2) a) (Mülga)"
    assert result.new_chunk.validity_start_date == date(2026, 8, 28)
    assert old_chunk.validity_end_date == date(2026, 8, 28)


def test_amendment_approval_rejects_reviewed_draft_identity_changes() -> None:
    stored_draft = {
        "user_file_id": "00000000-0000-0000-0000-000000000123",
        "position": 15,
        "text": "MADDE 15 eski öneri",
    }
    reviewed_draft = {
        **stored_draft,
        "user_file_id": "00000000-0000-0000-0000-000000000999",
    }
    submitted_proposal = cast(
        AmendmentProposal,
        SimpleNamespace(id=47, status="pending", old_chunk_id="old-chunk"),
    )
    locked_proposal = SimpleNamespace(
        id=47,
        status="pending",
        old_chunk_id="old-chunk",
        old_chunk_snapshot={"id": "old-chunk"},
        new_chunk_draft=stored_draft,
    )
    db_session = MagicMock(spec=Session)
    db_session.scalar.return_value = locked_proposal

    with pytest.raises(ValueError, match="cannot change user_file_id"):
        queue_amendment_proposal_approval(
            db_session,
            submitted_proposal,
            decided_by=None,
            reviewed_new_chunk_draft=reviewed_draft,
        )

    db_session.add.assert_not_called()


@pytest.mark.parametrize(
    "effective_end_date",
    [
        date.today().isoformat(),
        (date.today() - timedelta(days=1)).isoformat(),
    ],
)
def test_amendment_approval_rejects_end_on_or_before_approval_date_fallback(
    effective_end_date: str,
) -> None:
    stored_draft = {
        "user_file_id": "00000000-0000-0000-0000-000000000123",
        "position": 15,
        "text": "MADDE 15 eski öneri",
    }
    submitted_proposal = cast(
        AmendmentProposal,
        SimpleNamespace(id=48, status="pending", old_chunk_id="old-chunk"),
    )
    locked_proposal = SimpleNamespace(
        id=48,
        status="pending",
        old_chunk_id="old-chunk",
        old_chunk_snapshot={"id": "old-chunk", "text": "Eski metin"},
        new_chunk_draft=stored_draft,
    )
    old_chunk = SimpleNamespace(
        id="old-chunk",
        text="Eski metin",
        status="active",
        chunk_type="article",
        chunk_metadata={"chunk_variant": "atomic"},
    )
    db_session = MagicMock(spec=Session)
    db_session.scalar.side_effect = [locked_proposal, locked_proposal, old_chunk]

    queue_amendment_proposal_approval(
        db_session,
        submitted_proposal,
        decided_by=None,
        reviewed_new_chunk_draft={
            **stored_draft,
            "effective_start_date": None,
            "effective_end_date": effective_end_date,
        },
    )
    db_session.add.reset_mock()
    with pytest.raises(ValueError, match="effective_end_date must be after"):
        approve_amendment_proposal(
            db_session,
            submitted_proposal,
            decided_by=None,
        )

    db_session.add.assert_not_called()


def test_amendment_approval_rejects_old_chunk_changed_after_analysis() -> None:
    stored_draft = {
        "user_file_id": "00000000-0000-0000-0000-000000000123",
        "position": 15,
        "text": "Yeni metin",
    }
    submitted_proposal = cast(
        AmendmentProposal,
        SimpleNamespace(id=49, status="approving", old_chunk_id="old-chunk"),
    )
    locked_proposal = SimpleNamespace(
        id=49,
        status="approving",
        old_chunk_id="old-chunk",
        old_chunk_snapshot={"id": "old-chunk", "text": "İncelenen eski metin"},
        new_chunk_draft=stored_draft,
    )
    changed_old_chunk = SimpleNamespace(
        id="old-chunk",
        text="Analizden sonra elle değiştirilmiş metin",
        status="active",
        chunk_type="article",
        chunk_metadata={"chunk_variant": "atomic"},
    )
    db_session = MagicMock(spec=Session)
    db_session.scalar.side_effect = [locked_proposal, changed_old_chunk]

    with pytest.raises(ValueError, match="changed after analysis"):
        approve_amendment_proposal(
            db_session,
            submitted_proposal,
            decided_by=None,
        )

    db_session.add.assert_not_called()


def test_reviewed_draft_rejects_string_position() -> None:
    with pytest.raises(ValueError):
        ReviewedAmendmentChunkDraft.model_validate(
            {
                "user_file_id": "00000000-0000-0000-0000-000000000123",
                "position": "15",
                "text": "MADDE 15 yeni metin",
            }
        )


def test_amendment_approval_rejects_target_deleted_after_analysis() -> None:
    proposal = cast(
        AmendmentProposal,
        SimpleNamespace(
            id=43,
            status="approving",
            old_chunk_id=None,
            old_chunk_snapshot={"id": "deleted-old-chunk", "text": "Eski metin"},
            new_chunk_draft={
                "user_file_id": "00000000-0000-0000-0000-000000000123",
                "position": 3,
                "text": "Yeni metin",
            },
        ),
    )
    db_session = MagicMock(spec=Session)
    db_session.scalar.return_value = proposal

    with pytest.raises(ValueError, match="deleted-old-chunk.*no longer exists"):
        approve_amendment_proposal(db_session, proposal, decided_by=None)

    db_session.add.assert_not_called()
    db_session.flush.assert_not_called()


def test_amendment_approval_keeps_true_new_provision_unlinked() -> None:
    proposal = cast(
        AmendmentProposal,
        SimpleNamespace(
            id=44,
            status="approving",
            old_chunk_id=None,
            old_chunk_snapshot={},
            new_chunk_draft={
                "user_file_id": "00000000-0000-0000-0000-000000000123",
                "position": 3,
                "text": "Yeni ek madde",
            },
        ),
    )
    db_session = MagicMock(spec=Session)
    db_session.scalar.return_value = proposal

    result = approve_amendment_proposal(db_session, proposal, decided_by=None)

    assert result.old_chunk is None
    assert result.new_chunk.supersedes_chunk_id is None
    assert result.proposal.status == "approving"


def test_amendment_approval_redelivery_reuses_applied_chunk() -> None:
    proposal = cast(
        AmendmentProposal,
        SimpleNamespace(
            id=50,
            status="approving",
            applied_new_chunk_id="new-chunk",
        ),
    )
    new_chunk = SimpleNamespace(
        id="new-chunk",
        supersedes_chunk_id="old-chunk",
    )
    old_chunk = SimpleNamespace(id="old-chunk")
    db_session = MagicMock(spec=Session)
    db_session.scalar.return_value = proposal
    db_session.get.side_effect = [new_chunk, old_chunk]

    result = approve_amendment_proposal(db_session, proposal, decided_by=None)

    assert result.new_chunk is new_chunk
    assert result.old_chunk is old_chunk
    db_session.add.assert_not_called()
    db_session.flush.assert_not_called()


def test_rejection_locks_and_refreshes_before_checking_pending_status() -> None:
    submitted_proposal = cast(
        AmendmentProposal,
        SimpleNamespace(id=45, status="pending"),
    )
    approved_while_waiting = cast(
        AmendmentProposal,
        SimpleNamespace(id=45, status="approved"),
    )
    db_session = MagicMock(spec=Session)
    db_session.scalar.return_value = approved_while_waiting

    with pytest.raises(ValueError, match="already approved"):
        reject_proposal(
            db_session,
            submitted_proposal,
            decided_by=None,
        )

    assert submitted_proposal.status == "pending"
    assert approved_while_waiting.status == "approved"
    lock_query = db_session.scalar.call_args.args[0]
    assert "FOR UPDATE" in str(lock_query)
    assert lock_query.get_execution_options()["populate_existing"] is True


def test_analyze_amendment_marks_match_id_outside_candidates_unmatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruction = AmendmentInstruction(instruction_text="MADDE 3 değiştirilmiştir.")
    monkeypatch.setattr(
        pipeline,
        "segment_amendment_text",
        lambda *_args: SegmentationResult(
            reference_date=None, instructions=[instruction]
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "find_candidates",
        lambda *_args, **_kwargs: [
            CandidateChunk(
                chunk_id="allowed",
                user_file_id="00000000-0000-0000-0000-000000000123",
                text="MADDE 3 eski metin.",
            )
        ],
    )
    monkeypatch.setattr(
        pipeline,
        "confirm_match",
        lambda *_args, **_kwargs: MatchResult(
            old_chunk_id="foreign", confidence=0.9, rationale="wrong chunk"
        ),
    )
    monkeypatch.setattr(pipeline, "get_chunk_by_id", lambda *_args: None)
    monkeypatch.setattr(pipeline, "get_next_chunk_position", lambda *_args: 0)
    monkeypatch.setattr(
        pipeline,
        "draft_new_chunk",
        lambda *_args, **_kwargs: DraftResult(
            new_chunk=ChunkFieldsDraft(text="MADDE 3 yeni metin."),
            dates=DateResolution(
                effective_start_date=None,
                effective_end_date=None,
                rationale="no date",
            ),
        ),
    )

    result = pipeline.analyze_amendment(
        MagicMock(spec=Session),
        llm=MagicMock(),
        user_file_ids=[UUID("00000000-0000-0000-0000-000000000123")],
        raw_text="MADDE 3 değiştirilmiştir.",
    )

    assert result.proposals == []
    assert result.unmatched_instructions == [instruction]


@pytest.mark.parametrize(
    "instruction_text",
    [
        "Aynı Tebliğin 34 üncü maddesine aşağıdaki fıkra eklenmiştir.",
        "Aynı Yönetmeliğin 8 inci maddesine aşağıdaki bent eklenmiştir.",
        "MADDE 7 aşağıdaki şekilde değiştirilmiştir.",
        "MADDE 9 yürürlükten kaldırılmıştır.",
        "Bu Tebliğ 17/11/2024 tarihinde yürürlüğe girer.",
        "Bu Tebliğ hükümlerini Ticaret Bakanı yürütür.",
    ],
)
def test_only_explicit_top_level_addition_can_be_new(
    instruction_text: str,
) -> None:
    assert explicitly_adds_top_level_provision(instruction_text) is False


@pytest.mark.parametrize(
    "instruction_text",
    [
        "Aynı Tebliğe aşağıdaki MADDE 46 eklenmiştir.",
        "Yönetmeliğe aşağıdaki geçici madde eklenmiştir: GEÇİCİ MADDE 3- ...",
        "MADDE 12- Aşağıdaki yeni madde Tebliğe eklenmiştir.",
    ],
)
def test_explicit_top_level_addition_is_allowed(instruction_text: str) -> None:
    assert explicitly_adds_top_level_provision(instruction_text) is True


def test_matcher_null_for_existing_update_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruction = AmendmentInstruction(
        instruction_text=(
            "Aynı Tebliğin 19 uncu maddesindeki DEC_REJ ibaresi "
            "DEP_REJ olarak değiştirilmiştir."
        ),
        article_reference="Madde 19",
        target_source="Gümrük Genel Tebliği (Transit Rejimi) (Seri No: 4)",
    )
    candidates = [
        CandidateChunk(
            chunk_id="candidate",
            user_file_id="00000000-0000-0000-0000-000000000123",
            text="MADDE 19 eski metin",
        )
    ]
    monkeypatch.setattr(
        pipeline,
        "confirm_match",
        lambda *_args, **_kwargs: MatchResult(
            old_chunk_id=None,
            confidence=0.2,
            rationale="No sufficiently related candidate",
        ),
    )

    assert (
        pipeline.confirm_instruction_match(
            MagicMock(), instruction=instruction, candidates=candidates
        )
        is None
    )


def test_matcher_null_for_explicit_new_article_is_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruction = AmendmentInstruction(
        instruction_text="Aynı Tebliğe aşağıdaki MADDE 46 eklenmiştir.",
        article_reference="Madde 46",
    )
    expected = MatchResult(
        old_chunk_id=None,
        confidence=0.95,
        rationale="Explicit top-level article addition",
    )
    monkeypatch.setattr(pipeline, "confirm_match", lambda *_args, **_kwargs: expected)

    actual = pipeline.confirm_instruction_match(
        MagicMock(),
        instruction=instruction,
        candidates=[
            CandidateChunk(
                chunk_id="sibling",
                user_file_id="00000000-0000-0000-0000-000000000123",
                text="MADDE 45 mevcut metin",
            )
        ],
    )

    assert actual == expected


def test_disappeared_matched_chunk_is_not_converted_to_new_provision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "get_chunk_by_id", lambda *_args: None)
    get_next_position = MagicMock()
    monkeypatch.setattr(pipeline, "get_next_chunk_position", get_next_position)

    context = pipeline.load_instruction_draft_context(
        MagicMock(spec=Session),
        candidates=[
            CandidateChunk(
                chunk_id="old",
                user_file_id="00000000-0000-0000-0000-000000000123",
                text="MADDE 19 eski metin",
            )
        ],
        match=MatchResult(
            old_chunk_id="old",
            confidence=0.9,
            rationale="matched",
        ),
    )

    assert context is None
    get_next_position.assert_not_called()


def test_source_and_structured_lanes_beat_unrelated_same_number_candidate() -> None:
    related = CandidateChunk(
        chunk_id="related",
        user_file_id="00000000-0000-0000-0000-000000000123",
        text="MADDE 19 transit beyanı",
        source_name="gumruk_genel_tebligi_transit_rejimi_seri_no_4.md",
    )
    unrelated = CandidateChunk(
        chunk_id="unrelated",
        user_file_id="00000000-0000-0000-0000-000000000124",
        text="MADDE 19 etil alkol düzenlemesi",
        source_name="etil_alkol_yonetmeligi.md",
    )

    fused = candidate_finder.fuse_candidate_lanes(
        [
            ([unrelated, related], 0.7),
            ([related, unrelated], 1.5),
            ([related], 1.2),
        ],
        limit=2,
    )

    assert [candidate.chunk_id for candidate in fused] == ["related", "unrelated"]


def test_structured_lane_prioritizes_normalized_source_name() -> None:
    statement = str(candidate_finder._STRUCTURED_SQL)

    assert "JOIN user_file" in statement
    assert "source_score" in statement
    assert "ORDER BY source_score DESC" in statement
    assert ">= :minimum_source_score" in str(candidate_finder._SOURCE_FILES_SQL)


def test_concrete_source_is_extracted_and_propagated_to_same_regulation() -> None:
    instructions = propagate_target_sources(
        [
            AmendmentInstruction(
                instruction_text=(
                    "Resmî Gazete’de yayımlanan Gümrük Genel Tebliği "
                    "(Transit Rejimi) (Seri No: 4)’nin 17 nci maddesi "
                    "değiştirilmiştir."
                ),
                article_reference="Madde 17",
            ),
            AmendmentInstruction(
                instruction_text="Aynı Tebliğin 19 uncu maddesi değiştirilmiştir.",
                article_reference="Madde 19",
                target_source="Aynı Tebliğ",
            ),
        ]
    )

    assert [item.target_source for item in instructions] == [
        "Gümrük Genel Tebliği (Transit Rejimi) (Seri No: 4)",
        "Gümrük Genel Tebliği (Transit Rejimi) (Seri No: 4)",
    ]
