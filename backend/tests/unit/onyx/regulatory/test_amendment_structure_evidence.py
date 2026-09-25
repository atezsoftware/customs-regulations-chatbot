from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from onyx.regulatory.amendments.models import AmendmentInstruction
from onyx.regulatory.amendments.ranker import CandidateChunk
from onyx.regulatory.amendments.structural_target import (
    AmendmentStructuralTarget,
    deterministic_structural_candidate,
)
from onyx.regulatory.amendments.target_scope import reconcile_structural_candidates


def row(key: str, text: str, **metadata: str | None) -> CandidateChunk:
    return CandidateChunk(
        chunk_id=key,
        user_file_id="source",
        text=text,
        source_name="Makina Güvenliği Tebliği (2031/4)",
        metadata=metadata,
        source_verified=True,
    )


def test_appended_duplicate_cannot_hide_legacy_parent_or_win_exact_matching() -> None:
    opening = row("opening", "MADDE 8- (2) Tanımlar.", article_no="8", paragraph_no="2")
    siblings = [row(f"sibling-{i}", f"a) Diğer {i}", article_no="8") for i in range(8)]
    old = row(
        "old", "d) Fiili denetim: Belge kontrolü.", article_no="8", clause_label="d"
    )
    next_article = row("next", "MADDE 9- (1) Başvurular.", article_no="9")
    appended = replace(
        row(
            "appended",
            "d) Dijital kayıt: Elektronik bilgi.",
            article_no="8",
            paragraph_no="2",
            clause_label="d",
        ),
        structured_match=True,
    )
    rows = [opening, *siblings, old, next_article, appended]
    target = AmendmentStructuralTarget(
        article_no="8", paragraph_no="2", clause_label="d"
    )
    results = reconcile_structural_candidates([appended], rows, target)
    assert {item.chunk_id for item in results[:2]} == {"old", "appended"}
    assert all(item.structure_conflict for item in results)
    instruction = AmendmentInstruction(
        instruction_text="MADDE 2- Aynı Tebliğin 8 inci maddesinin ikinci fıkrasının (d) bendi değiştirilmiştir.",
        target_source=opening.source_name,
    )
    assert deterministic_structural_candidate(instruction, results[:1]) is None
    assert old.metadata.get("paragraph_no") is None


def test_different_paragraph_same_letter_is_not_a_conflict() -> None:
    one = row("one", "MADDE 8- (1) Tanımlar.", article_no="8", paragraph_no="1")
    a = row(
        "a", "ç) Birinci tanım.", article_no="8", paragraph_no="1", clause_label="ç"
    )
    two = row("two", "(2) Başka tanımlar.", article_no="8", paragraph_no="2")
    b = replace(
        row(
            "b", "ç) İkinci tanım.", article_no="8", paragraph_no="2", clause_label="ç"
        ),
        structured_match=True,
    )
    target = AmendmentStructuralTarget(
        article_no="8", paragraph_no="2", clause_label="ç"
    )
    result = reconcile_structural_candidates([b], [one, a, two, b], target)
    assert result[0].chunk_id == "b"
    assert result[0].structure_conflict is None and result[0].structured_match


def test_folded_letter_cannot_override_the_actual_enumerator() -> None:
    opening = row("opening", "MADDE 8- (1) Tanımlar.", article_no="8", paragraph_no="1")
    c = row("c", "c) Üçüncü tanım.", article_no="8", clause_label="c")
    folded = replace(
        row(
            "folded",
            "ç) Dördüncü tanım.",
            article_no="8",
            paragraph_no="1",
            clause_label="c",
        ),
        structured_match=True,
    )
    result = reconcile_structural_candidates(
        [folded],
        [opening, c, folded],
        AmendmentStructuralTarget(article_no="8", paragraph_no="1", clause_label="c"),
    )
    assert result[0].chunk_id == "c"
    assert all(item.structure_conflict for item in result)


def test_one_subunit_in_multiple_canonical_pieces_is_not_automatically_exact() -> None:
    opening = row("opening", "MADDE 8- (1) Tanımlar.", article_no="8", paragraph_no="1")
    first = replace(
        row(
            "first",
            "d) Birinci satır.",
            article_no="8",
            paragraph_no="1",
            clause_label="d",
        ),
        structured_match=True,
    )
    continuation = row(
        "continued", "Tanım devamı.", article_no="8", paragraph_no="1", clause_label="d"
    )
    result = reconcile_structural_candidates(
        [first],
        [opening, first, continuation],
        AmendmentStructuralTarget(article_no="8", paragraph_no="1", clause_label="d"),
    )
    assert result[0].structure_conflict


def test_model_cannot_override_a_verified_source_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.regulatory.amendments import pipeline
    from onyx.regulatory.amendments.models import MatchResult

    old = replace(
        row("old", "d) Denetim.", article_no="8", clause_label="d"),
        resolved_article_no="8",
        structure_conflict="Several units claim d.",
    )
    appended = replace(
        row(
            "appended", "d) Kayıt.", article_no="8", paragraph_no="2", clause_label="d"
        ),
        structure_conflict="Several units claim d.",
    )
    instruction = AmendmentInstruction(
        instruction_text="MADDE 2- Aynı Tebliğin 8 inci maddesinin ikinci fıkrasının (d) bendi değiştirilmiştir.",
        target_source=old.source_name,
    )
    for selected, expected in [(appended, False), (old, True)]:
        match = MatchResult(
            old_chunk_id=selected.chunk_id, confidence=1, rationale="metadata"
        )
        monkeypatch.setattr(pipeline, "confirm_match", lambda *_args, **_kwargs: match)
        decisions: list[dict[str, object]] = []
        result = pipeline.confirm_instruction_match(
            MagicMock(),
            instruction=instruction,
            candidates=[appended, old],
            decisions=decisions,
        )
        assert (result is not None) == expected
        if not expected:
            assert "boundary" in str(decisions[-1]["rationale"])
