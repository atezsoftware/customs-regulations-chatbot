from dataclasses import replace

from onyx.regulatory.amendments.models import AmendmentInstruction
from onyx.regulatory.amendments.ranker import CandidateChunk
from onyx.regulatory.amendments.target_scope import (
    article_scope_candidates,
    validated_addition_anchor,
)


def candidate(chunk_id: str, text: str, **metadata: str) -> CandidateChunk:
    return CandidateChunk(
        chunk_id=chunk_id,
        user_file_id="source",
        text=text,
        source_name="3713_sayili_terorle_mucadele_kanunu.md",
        metadata=metadata,
        source_verified=True,
    )


def test_legacy_table_inherits_only_its_actual_article_scope() -> None:
    rows = [
        candidate("previous", "**EK MADDE 2-** önceki", article_no="22"),
        candidate("heading", "**EK MADDE 3-** Birinci fıkra", article_no="22"),
        candidate("table", "| 22.382 | 20.821 |", article_no="22"),
        candidate("paragraph", "Üçüncü fıkra metni", article_no="22"),
        candidate("next", "**EK MADDE 4-** sonraki", article_no="22"),
        candidate("citation", "Ek 3 üncü madde hükümleri uygulanır.", article_no="22"),
    ]
    found = article_scope_candidates(rows, "EK 3")
    assert [row.chunk_id for row in found] == ["heading", "table", "paragraph"]
    assert all(row.resolved_article_no == "EK 3" for row in found)
    assert all(row.metadata["article_no"] == "22" for row in found)
    assert "heading" in (found[1].scope_evidence or "")


def test_quoted_heading_is_not_an_existing_article_boundary() -> None:
    rows = [candidate("quote", "“EK MADDE 3- Yeni hüküm.”", article_no="22")]
    assert article_scope_candidates(rows, "EK 3") == []


def test_new_paragraph_requires_verified_source_and_parent() -> None:
    instruction = AmendmentInstruction(
        instruction_text="3713 sayılı Kanunun ek 3 üncü maddesine aşağıdaki fıkra eklenmiştir.",
        target_source="3713 sayılı Terörle Mücadele Kanunu",
    )
    wrong = candidate("wrong", "Başka madde", article_no="22")
    assert validated_addition_anchor(instruction, [wrong]) is None
    resolved = replace(wrong, resolved_article_no="EK 3", scope_evidence="heading")
    assert validated_addition_anchor(instruction, [resolved]) == resolved
    assert (
        validated_addition_anchor(
            instruction,
            [
                replace(
                    resolved, source_verified=False, source_name="6111_sayili_kanun.md"
                )
            ],
        )
        is None
    )
    assert (
        validated_addition_anchor(
            instruction, [resolved, replace(resolved, user_file_id="ambiguous")]
        )
        is None
    )


def test_new_top_level_article_does_not_require_existing_target_article() -> None:
    instruction = AmendmentInstruction(
        instruction_text="3713 sayılı Kanuna aşağıdaki geçici madde eklenmiştir.\n“GEÇİCİ MADDE 20- Yeni metin.”",
        target_source="3713 sayılı Terörle Mücadele Kanunu",
    )
    anchor = candidate("last", "GEÇİCİ MADDE 19- Eski metin", article_no="GEÇİCİ 19")
    assert validated_addition_anchor(instruction, [anchor]) == anchor


def test_matcher_keeps_late_old_wording_evidence() -> None:
    from onyx.regulatory.amendments.matcher import _bounded_candidate_text

    text = "Başlık\n" + "a" * 12000 + "eski özel ibare" + "z" * 1000
    bounded = _bounded_candidate_text(
        text, '"eski özel ibare" ibaresi "yeni" olarak değiştirilmiştir.'
    )
    assert "eski özel ibare" in bounded
    assert "offset" in bounded
    assert len(bounded) < 6500


def test_legacy_scope_refuses_a_chunk_containing_two_articles() -> None:
    rows = [
        candidate("mixed", "EK MADDE 3- Birinci\nEK MADDE 4- İkinci", article_no="22")
    ]
    assert article_scope_candidates(rows, "EK 3") == []


def test_legacy_scope_stops_at_nonconsolidated_supplement() -> None:
    rows = [
        candidate("heading", "GEÇİCİ MADDE 19- Eski", article_no="19"),
        candidate("supplement", "KANUNA İŞLENEMEYEN HÜKÜMLER", article_no="19"),
        candidate("otherlaw", "4126 sayılı Kanunun geçici maddesi", article_no="19"),
    ]
    assert [row.chunk_id for row in article_scope_candidates(rows, "GEÇİCİ 19")] == [
        "heading"
    ]
