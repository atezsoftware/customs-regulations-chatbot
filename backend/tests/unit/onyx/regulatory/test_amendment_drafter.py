from unittest.mock import MagicMock
from uuid import UUID

import pytest

from onyx.regulatory.amendments import drafter, pipeline
from onyx.regulatory.amendments.draft_integrity import (
    DraftIntegrityError,
    reconcile_existing_heading_path,
)
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


def _article_20_context(
    *, has_active_descendants: bool = False
) -> pipeline.InstructionDraftContext:
    heading_path = [
        "GÜMRÜK GENEL TEBLİĞİ (TIR İşlemleri) (Seri No: 1)",
        "ÜÇÜNCÜ BÖLÜM",
        "MADDE 20 - TIR karnesi himayesinde eşya taşıma yöntemleri",
        "(1) Bir TIR taşımasında hareket ve varış gümrük idarelerinin toplam sayısı yediyi geçemez",
    ]
    metadata = {
        "article_no": "20",
        "paragraph_no": "1",
        "heading_path": heading_path,
    }
    return pipeline.InstructionDraftContext(
        match=MatchResult(
            old_chunk_id="article-20-v2", confidence=1, rationale="exact"
        ),
        old_chunk_snapshot={
            "id": "article-20-v2",
            "text": "**MADDE 20 -** (1) Bir TIR taşımasında hareket ve varış gümrük idarelerinin toplam sayısı yediyi geçemez.",
            "chunk_type": "paragraph",
            "heading_path": heading_path,
            "metadata": metadata,
        },
        target_user_file_id=UUID("00000000-0000-0000-0000-000000000123"),
        target_position=132,
        sibling_reference=None,
        base_metadata=metadata,
        base_heading_path=heading_path,
        has_active_descendants=has_active_descendants,
    )


def test_explicit_replacement_draft_must_contain_authoritative_new_body() -> None:
    instruction = AmendmentInstruction(
        instruction_text=(
            "Aynı Tebliğin 20 nci maddesinin birinci fıkrası aşağıdaki şekilde "
            "değiştirilmiştir. “(1) Bir TIR taşımasında hareket ve varış "
            "gümrük idarelerinin toplam sayısı sekizi geçemez.”"
        )
    )
    unchanged_draft = DraftResult(
        new_chunk=ChunkFieldsDraft(
            text="**MADDE 20 -** (1) Bir TIR taşımasında hareket ve varış gümrük idarelerinin toplam sayısı yediyi geçemez."
        ),
        dates=DateResolution(rationale="publication date"),
    )

    with pytest.raises(DraftIntegrityError, match="explicit replacement"):
        pipeline._build_proposal_draft(
            instruction_indices=[0],
            instructions=[instruction],
            matches=[_article_20_context().match],
            context=_article_20_context(),
            draft=unchanged_draft,
        )


def test_existing_chunk_type_and_heading_follow_the_amended_text() -> None:
    instruction = AmendmentInstruction(
        instruction_text=(
            "Aynı Tebliğin 20 nci maddesinin birinci fıkrası aşağıdaki şekilde "
            "değiştirilmiştir. “(1) Bir TIR taşımasında hareket ve varış "
            "gümrük idarelerinin toplam sayısı sekizi geçemez.”"
        )
    )
    amended_text = (
        "**MADDE 20 -** (1) Bir TIR taşımasında hareket ve varış gümrük "
        "idarelerinin toplam sayısı sekizi geçemez."
    )
    draft = DraftResult(
        new_chunk=ChunkFieldsDraft(
            text=amended_text,
            chunk_type="clause",
            heading_path=["INVENTED DOCUMENT", "INVENTED ARTICLE"],
            metadata_changes={"clause_label": "a", "subclause_label": "i"},
        ),
        dates=DateResolution(rationale="publication date"),
    )

    proposal = pipeline._build_proposal_draft(
        instruction_indices=[0],
        instructions=[instruction],
        matches=[_article_20_context().match],
        context=_article_20_context(),
        draft=draft,
    )

    expected_heading = [
        "GÜMRÜK GENEL TEBLİĞİ (TIR İşlemleri) (Seri No: 1)",
        "ÜÇÜNCÜ BÖLÜM",
        "MADDE 20 - TIR karnesi himayesinde eşya taşıma yöntemleri",
        "(1) Bir TIR taşımasında hareket ve varış gümrük idarelerinin toplam sayısı sekizi geçemez",
    ]
    assert proposal.new_chunk_draft["chunk_type"] == "paragraph"
    assert "clause_label" not in proposal.new_chunk_draft["metadata"]
    assert "subclause_label" not in proposal.new_chunk_draft["metadata"]
    assert proposal.new_chunk_draft["heading_path"] == expected_heading
    assert proposal.new_chunk_draft["metadata"]["heading_path"] == expected_heading


def test_incomplete_full_replacement_of_parent_with_descendants_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruction = AmendmentInstruction(
        instruction_text=(
            "20 nci maddenin üçüncü fıkrası aşağıdaki şekilde değiştirilmiştir. "
            "“(3) Yediden fazla idare için: a) Birinci yöntem... b) İkinci yöntem...”"
        )
    )
    generate_structured = MagicMock()
    monkeypatch.setattr(drafter, "generate_structured", generate_structured)

    with pytest.raises(DraftIntegrityError, match="descendant chunks"):
        pipeline.draft_instruction_group_proposal(
            MagicMock(),
            instruction_indices=[0],
            instructions=[instruction],
            matches=[_article_20_context(has_active_descendants=True).match],
            reference_date="2026-07-04",
            context=_article_20_context(has_active_descendants=True),
        )

    generate_structured.assert_not_called()


@pytest.mark.parametrize(
    ("heading_path", "text", "chunk_type", "labels", "expected"),
    [
        (
            ["KANUN", "MADDE 20 - Eski başlık"],
            "MADDE 20 - Yeni hüküm.",
            "article",
            {"article_no": "20", "article_title": "Yeni başlık"},
            ["KANUN", "MADDE 20 - Yeni başlık"],
        ),
        (
            ["KANUN", "GEÇİCİ MADDE 1 - Eski başlık"],
            "GEÇİCİ MADDE 1 - Yeni hüküm.",
            "article",
            {"article_no": "GEÇİCİ 1", "article_title": "Yeni başlık"},
            ["KANUN", "GEÇİCİ MADDE 1 - Yeni başlık"],
        ),
        (
            ["KANUN", "MÜKERRER MADDE 2"],
            "MÜKERRER MADDE 2 - Yeni hüküm.",
            "article",
            {"article_no": "MÜKERRER 2"},
            ["KANUN", "MÜKERRER MADDE 2"],
        ),
        (
            ["MADDE 20", "(3) Yöntemler", "a) Eski yöntem"],
            "a) Yeni yöntem uygulanır.",
            "clause",
            {"clause_label": "a"},
            ["MADDE 20", "(3) Yöntemler", "a) Yeni yöntem uygulanır"],
        ),
        (
            ["MADDE 20", "a) Yöntem", "(i) Eski alt bent"],
            "(i) Yeni alt bent uygulanır.",
            "subclause",
            {"clause_label": "a", "subclause_label": "i"},
            ["MADDE 20", "a) Yöntem", "(i) Yeni alt bent uygulanır"],
        ),
        (
            [],
            "(1) Yeni paragraf uygulanır.",
            "paragraph",
            {"paragraph_no": "1"},
            ["(1) Yeni paragraf uygulanır"],
        ),
    ],
)
def test_reconcile_existing_heading_path_for_structural_types(
    heading_path: list[str],
    text: str,
    chunk_type: str,
    labels: dict[str, str],
    expected: list[str],
) -> None:
    assert (
        reconcile_existing_heading_path(
            heading_path,
            amended_text=text,
            chunk_type=chunk_type,
            article_no=labels.get("article_no"),
            article_title=labels.get("article_title"),
            paragraph_no=labels.get("paragraph_no"),
            clause_label=labels.get("clause_label"),
            subclause_label=labels.get("subclause_label"),
        )
        == expected
    )
