from unittest.mock import MagicMock
from uuid import UUID

import pytest

from onyx.regulatory.amendments import pipeline
from onyx.regulatory.amendments.models import (
    AmendmentInstruction,
    ChunkFieldsDraft,
    DateResolution,
    DraftResult,
    MatchResult,
)

TEXT = 'MADDE 4- Aynı Kararın 10 uncu maddesinin başlığı "Yetki, denetim ve izleme" şeklinde değiştirilmiş ve aynı maddeye aşağıdaki fıkra eklenmiştir:\n"(4) Yatırımcı bilgi verir."'


def test_compound_proposal_keeps_heading_change_and_new_paragraph_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    match = MatchResult(
        old_chunk_id=None,
        outcome="new_provision",
        confidence=1,
        rationale="new paragraph",
    )
    file_id = UUID("00000000-0000-0000-0000-000000000123")
    old = {
        "id": "old",
        "user_file_id": str(file_id),
        "position": 1,
        "text": "(1) Existing rule.",
        "chunk_type": "paragraph",
        "heading_path": ["Source", "MADDE 10 - Yetki", "(1) Existing rule."],
        "metadata": {"article_no": "10", "paragraph_no": "1", "article_title": "Yetki"},
    }
    context = pipeline.InstructionDraftContext(
        match=match,
        old_chunk_snapshot={},
        target_user_file_id=file_id,
        target_position=3,
        sibling_reference={"metadata": {"article_no": "10"}},
        base_metadata={},
        base_heading_path=[],
        expected_new_article_no="10",
        heading_change_snapshots=[old],
    )
    monkeypatch.setattr(
        pipeline,
        "draft_combined_chunk",
        lambda *_args, **_kwargs: DraftResult(
            new_chunk=ChunkFieldsDraft(
                text="(4) Yatırımcı bilgi verir.",
                chunk_type="paragraph",
                heading_path=[
                    "Source",
                    "MADDE 10 - Yetki",
                    "(4) Yatırımcı bilgi verir.",
                ],
                metadata_changes={"article_no": "10", "paragraph_no": "4"},
            ),
            dates=DateResolution(
                effective_start_date=None, effective_end_date=None, rationale="unknown"
            ),
        ),
    )
    proposal = pipeline.draft_instruction_group_proposal(
        MagicMock(),
        instruction_indices=[10],
        instructions=[AmendmentInstruction(instruction_text=TEXT)],
        matches=[match],
        reference_date=None,
        context=context,
    )
    assert len(proposal.chunk_changes) == 2
    addition, heading = proposal.chunk_changes
    assert addition.old_chunk_id is None
    assert addition.new_chunk_draft["text"] == "(4) Yatırımcı bilgi verir."
    assert heading.old_chunk_id == "old"
    assert heading.old_chunk_snapshot["heading_change"] == {
        "article_no": "10",
        "title": "Yetki, denetim ve izleme",
    }
    assert heading.new_chunk_draft["text"] == old["text"]
    assert heading.new_chunk_draft["metadata"]["paragraph_no"] == "1"
    for change in proposal.chunk_changes:
        assert (
            change.new_chunk_draft["heading_path"][1]
            == "MADDE 10 - Yetki, denetim ve izleme"
        )
        assert change.instruction_indices == [10]


def test_review_validation_preserves_authorized_parent_heading_change() -> None:
    from onyx.db.regulatory_amendments import _validated_reviewed_chunk_draft

    snapshot = {
        "id": "old",
        "text": "(1) Existing rule.",
        "chunk_type": "paragraph",
        "metadata": {"article_no": "10", "paragraph_no": "1", "article_title": "Yetki"},
        "heading_path": ["Source", "MADDE 10 - Yetki", "(1) Existing rule."],
        "heading_change": {"article_no": "10", "title": "Yetki, denetim ve izleme"},
    }
    draft = {
        "user_file_id": "00000000-0000-0000-0000-000000000123",
        "position": 1,
        "text": snapshot["text"],
        "chunk_type": "paragraph",
        "metadata": snapshot["metadata"],
        "heading_path": [
            "Source",
            "MADDE 10 - Yetki, denetim ve izleme",
            "(1) Existing rule.",
        ],
        "effective_start_date": None,
        "effective_end_date": None,
    }
    validated = _validated_reviewed_chunk_draft(
        draft, draft, old_chunk_snapshot=snapshot
    )
    assert validated["heading_path"][1] == "MADDE 10 - Yetki, denetim ve izleme"
    assert validated["metadata"]["article_title"] == "Yetki, denetim ve izleme"
    assert validated["metadata"]["paragraph_no"] == "1"
