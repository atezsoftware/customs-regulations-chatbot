from unittest.mock import MagicMock
from uuid import UUID

import pytest
from sqlalchemy.orm import Session

from onyx.db.regulatory_amendments import approve_amendment_proposal
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
    monkeypatch.setattr(pipeline, "get_chunks_for_file", lambda *_args: [])
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
