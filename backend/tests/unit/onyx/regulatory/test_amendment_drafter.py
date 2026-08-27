from unittest.mock import MagicMock
from uuid import UUID

import pytest

from onyx.regulatory.amendments import drafter, pipeline
from onyx.regulatory.amendments.models import (
    AmendmentInstruction,
    ChunkFieldsDraft,
    DateResolution,
    DraftResult,
    MatchResult,
)


def test_combined_draft_prompt_contains_each_instruction_and_returns_one_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instructions = [
        AmendmentInstruction(
            instruction_text="The amount is changed to 500 lira.",
            raw_date_phrase=" Yayımı   Tarihinden İtibaren ",
        ),
        AmendmentInstruction(
            instruction_text="The filing period is changed to 30 days.",
            raw_date_phrase="yayımı tarihinden itibaren",
        ),
    ]
    matches = [
        MatchResult(old_chunk_id="shared", confidence=0.91, rationale="amount row"),
        MatchResult(old_chunk_id="shared", confidence=0.73, rationale="period row"),
    ]
    context = pipeline.InstructionDraftContext(
        match=matches[0],
        old_chunk_snapshot={
            "id": "shared",
            "text": "The amount is 100 lira and the filing period is 10 days.",
            "chunk_type": "article",
            "heading_path": ["MADDE 4"],
            "metadata": {"article_no": "4"},
        },
        target_user_file_id=UUID("00000000-0000-0000-0000-000000000123"),
        target_position=4,
        sibling_reference=None,
        base_metadata={"article_no": "4"},
        base_heading_path=["MADDE 4"],
    )
    generated = DraftResult(
        new_chunk=ChunkFieldsDraft(
            text="The amount is 500 lira and the filing period is 30 days.",
            chunk_type="article",
        ),
        dates=DateResolution(
            effective_start_date="2026-08-27",
            effective_end_date=None,
            rationale="Both changes use the publication date.",
        ),
    )
    generate_structured = MagicMock(return_value=generated)
    monkeypatch.setattr(drafter, "generate_structured", generate_structured)

    proposal = pipeline.draft_instruction_group_proposal(
        MagicMock(),
        instruction_indices=[2, 5],
        instructions=instructions,
        matches=matches,
        reference_date="2026-08-27",
        context=context,
    )

    generate_structured.assert_called_once()
    system_prompt = generate_structured.call_args.kwargs["system_prompt"].lower()
    user_prompt = generate_structured.call_args.kwargs["user_prompt"]
    assert "one full replacement chunk" in system_prompt
    assert "The amount is changed to 500 lira." in user_prompt
    assert "The filing period is changed to 30 days." in user_prompt
    assert "Yayımı   Tarihinden İtibaren" in user_prompt
    assert "yayımı tarihinden itibaren" in user_prompt
    assert proposal.instruction_index == 2
    assert proposal.instruction_text == instructions[0].instruction_text
    assert proposal.instruction_indices == [2, 5]
    assert proposal.instruction_texts == [
        instruction.instruction_text for instruction in instructions
    ]
    assert proposal.new_chunk_draft["text"] == generated.new_chunk.text
    assert proposal.match_confidence == 0.73
    assert "amount row" in (proposal.match_rationale or "")
    assert "period row" in (proposal.match_rationale or "")
