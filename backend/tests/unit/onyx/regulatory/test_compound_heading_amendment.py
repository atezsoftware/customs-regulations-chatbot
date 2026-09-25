from typing import Any
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
    ProposalChunkChange,
    ProposalDraft,
)

TEXT = 'MADDE 4- Aynı Kararın 10 uncu maddesinin başlığı "Yetki, denetim ve izleme" şeklinde değiştirilmiş ve aynı maddeye aşağıdaki fıkra eklenmiştir:\n"(4) Yatırımcı bilgi verir."'


def test_heading_full_parent_replacement_cannot_resurrect_old_children() -> None:
    from onyx.regulatory.amendments.compound_heading import attach_heading_changes

    file_id = "00000000-0000-0000-0000-000000000123"
    parent = {
        "id": "parent",
        "user_file_id": file_id,
        "position": 0,
        "text": "(1) Eski kurallar.",
        "chunk_type": "paragraph",
        "heading_path": ["Kaynak", "MADDE 8 - Eski başlık", "(1) Eski kurallar"],
        "metadata": {"article_no": "8", "paragraph_no": "1"},
    }
    child = {
        **parent,
        "id": "child",
        "position": 1,
        "chunk_type": "clause",
        "text": "a) Kaldırılan eski kural.",
        "metadata": {**parent["metadata"], "clause_label": "a"},
    }
    texts = [
        "MADDE 1- Aynı Kanunun 8 inci maddesinin başlığı “Yeni başlık” şeklinde değiştirilmiştir.",
        "MADDE 2- Aynı Kanunun 8 inci maddesinin birinci fıkrası aşağıdaki şekilde değiştirilmiştir.\n“(1) Tam yeni kural.”",
    ]
    proposal = ProposalDraft(
        instruction_index=0,
        instruction_text=texts[0],
        instruction_indices=[0, 1],
        instruction_texts=texts,
        old_chunk_id="parent",
        old_chunk_snapshot={**parent, "descendant_snapshots": [child]},
        new_chunk_draft={
            **{k: v for k, v in parent.items() if k != "id"},
            "text": "(1) Tam yeni kural.",
            "effective_start_date": None,
            "effective_end_date": None,
        },
        match_confidence=1,
        match_rationale="verified",
        date_rationale="unknown",
    )
    result = attach_heading_changes(
        proposal, snapshots=[parent, child], article_no="8", title="Yeni başlık"
    )
    assert len(result.chunk_changes) == 1
    change = result.chunk_changes[0]
    assert change.old_chunk_id == "parent"
    assert change.new_chunk_draft["text"] == "(1) Tam yeni kural."
    assert change.old_chunk_snapshot["descendant_snapshots"] == [child]
    assert change.new_chunk_draft["metadata"]["article_title"] == "Yeni başlık"


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
        sibling_reference={
            "metadata": {"article_no": "10"},
            "heading_path": old["heading_path"],
        },
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


def test_split_heading_operations_merge_without_losing_body_edits() -> None:
    from onyx.regulatory.amendments.compound_heading import attach_heading_changes

    snapshots: list[dict[str, Any]] = [
        {
            "id": f"p{number}",
            "user_file_id": "source",
            "position": number,
            "text": f"({number}) Original rule.",
            "chunk_type": "paragraph",
            "heading_path": [
                "Source",
                "MADDE 8 - Old title",
                f"({number}) Original rule",
            ],
            "metadata": {
                "article_no": "8",
                "paragraph_no": str(number),
                "article_title": "Old title",
            },
        }
        for number in (1, 2)
    ]
    changes = [
        ProposalChunkChange(
            old_chunk_id=snapshots[0]["id"],
            old_chunk_snapshot=snapshots[0],
            new_chunk_draft={
                **snapshots[0],
                "text": "(1) New body.",
                "effective_start_date": "2026-01-01",
            },
            instruction_indices=[3],
            instruction_texts=[
                'MADDE 2- Birinci fıkra aşağıdaki şekilde değiştirilmiştir. "(1) New body."'
            ],
            match_confidence=1,
            match_rationale="verified",
            date_rationale="explicit",
        )
    ]
    proposal = ProposalDraft(
        instruction_index=2,
        instruction_text='MADDE 2- Maddenin başlığı "New title" şeklinde değiştirilmiştir.',
        instruction_indices=[2, 3],
        instruction_texts=[
            'MADDE 2- Maddenin başlığı "New title" şeklinde değiştirilmiştir.',
            changes[0].instruction_texts[0],
        ],
        old_chunk_id="p1",
        old_chunk_snapshot=snapshots[0],
        new_chunk_draft=changes[0].new_chunk_draft,
        chunk_changes=changes,
        match_confidence=1,
        match_rationale="verified",
        date_rationale="explicit",
    )
    result = attach_heading_changes(
        proposal, snapshots=snapshots, article_no="8", title="New title"
    )
    assert len(result.chunk_changes) == 2
    assert result.chunk_changes[0].new_chunk_draft["text"] == "(1) New body."
    assert result.chunk_changes[1].new_chunk_draft["text"] == snapshots[1]["text"]
    assert result.chunk_changes[1].instruction_indices == [2]
    for change in result.chunk_changes:
        assert change.new_chunk_draft["heading_path"][1] == "MADDE 8 - New title"
        assert change.old_chunk_snapshot["heading_change"]["article_no"] == "8"
    assert result.old_chunk_snapshot["heading_change_scope"]["chunk_ids"] == [
        "p1",
        "p2",
    ]


@pytest.mark.parametrize(
    "ending",
    ["değiştirilmiştir.", "değiştirilmiş ve aynı maddeye aşağıdaki fıkra eklenmiştir."],
)
def test_heading_title_survives_atomic_instruction_segmentation(ending: str) -> None:
    from onyx.regulatory.amendments.compound_heading import article_heading_title

    assert (
        article_heading_title(
            f"MADDE 4- Kanunun ek 8 inci maddesinin başlığı “Yeni başlık” şeklinde {ending}"
        )
        == "Yeni başlık"
    )
    assert (
        article_heading_title(
            "MADDE 4- Kanunun 8 inci maddesinde yer alan “başlığı” ibaresi “etiketi” şeklinde değiştirilmiştir."
        )
        is None
    )


def test_separate_heading_and_addition_create_one_review_and_keep_existing_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_id = UUID("00000000-0000-0000-0000-000000000123")
    old = {
        "id": "p1",
        "user_file_id": str(file_id),
        "position": 1,
        "text": "(1) Existing rule.",
        "chunk_type": "paragraph",
        "heading_path": ["Source", "MADDE 8 - Old title", "(1) Existing rule"],
        "metadata": {
            "article_no": "8",
            "paragraph_no": "1",
            "article_title": "Old title",
        },
    }
    heading = AmendmentInstruction(
        instruction_text='MADDE 4- Kanunun 8 inci maddesinin başlığı "New title" şeklinde değiştirilmiştir.',
        article_reference="Madde 8 başlığı",
    )
    addition = AmendmentInstruction(
        instruction_text='MADDE 4- Kanunun 8 inci maddesine aşağıdaki fıkra eklenmiştir. "(2) New rule."',
        article_reference="Madde 8 fıkra 2",
    )
    matches = [
        MatchResult(old_chunk_id="p1", confidence=1, rationale="heading"),
        MatchResult(
            old_chunk_id=None,
            outcome="new_provision",
            confidence=1,
            rationale="addition",
        ),
    ]
    contexts = [
        pipeline.InstructionDraftContext(
            match=match,
            old_chunk_snapshot=old if match.old_chunk_id else {},
            target_user_file_id=file_id,
            target_position=1 if match.old_chunk_id else 2,
            sibling_reference=None
            if match.old_chunk_id
            else {"metadata": {"article_no": "8"}, "heading_path": old["heading_path"]},
            base_metadata=old["metadata"] if match.old_chunk_id else {},
            base_heading_path=old["heading_path"] if match.old_chunk_id else [],
            expected_new_article_no=None if match.old_chunk_id else "8",
            heading_change_snapshots=[old] if match.old_chunk_id else [],
        )
        for match in matches
    ]

    def draft(*_args: object, **kwargs: object) -> DraftResult:
        existing = kwargs["old_chunk"] is not None
        return DraftResult(
            new_chunk=ChunkFieldsDraft(
                text="LLM attempted to overwrite the old body."
                if existing
                else "(2) New rule.",
                chunk_type="paragraph",
                heading_path=[
                    "Source",
                    "MADDE 8 - Old title",
                    "(1) Existing rule" if existing else "(2) New rule",
                ],
                metadata_changes={
                    "article_no": "8",
                    "paragraph_no": "1" if existing else "2",
                },
            ),
            dates=DateResolution(
                effective_start_date=None, effective_end_date=None, rationale="unknown"
            ),
        )

    monkeypatch.setattr(pipeline, "draft_combined_chunk", draft)
    proposal = pipeline.draft_article_heading_group_proposal(
        MagicMock(),
        items=[
            (index, instruction, matches[index], contexts[index])
            for index, instruction in enumerate([heading, addition])
        ],
        reference_date=None,
    )
    assert proposal.instruction_indices == [0, 1]
    assert len(proposal.chunk_changes) == 2
    assert proposal.chunk_changes[0].new_chunk_draft["text"] == old["text"]
    assert proposal.chunk_changes[1].new_chunk_draft["text"] == "(2) New rule."
    assert proposal.chunk_changes[1].old_chunk_id is None
    for change in proposal.chunk_changes:
        assert change.new_chunk_draft["heading_path"][1] == "MADDE 8 - New title"
