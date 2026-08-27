from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from sqlalchemy.orm import Session

from onyx.db.regulatory_amendments import approve_amendment_proposal
from onyx.db.regulatory_chunks import get_next_chunk_position
from onyx.regulatory.amendments import candidate_finder, pipeline
from onyx.regulatory.amendments.drafter import DraftResult
from onyx.regulatory.amendments.models import (
    AmendmentInstruction,
    ChunkFieldsDraft,
    DateResolution,
    MatchResult,
    SegmentationResult,
)
from onyx.regulatory.amendments.ranker import CandidateChunk


def test_amendment_candidate_queries_exclude_hierarchical_aggregates() -> None:
    for statement in (
        candidate_finder._TEXT_TRGM_SQL,
        candidate_finder._HEADING_TRGM_SQL,
        candidate_finder._STRUCTURED_SQL,
    ):
        assert "chunk_type IS DISTINCT FROM 'hierarchical_aggregate'" in str(statement)


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
    proposal.status = "pending"
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
    db_session.get.return_value = aggregate

    with pytest.raises(
        ValueError, match="Derived aggregate chunks cannot be amended directly"
    ):
        approve_amendment_proposal(
            db_session,
            proposal,
            decided_by=None,
        )

    db_session.add.assert_not_called()


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
