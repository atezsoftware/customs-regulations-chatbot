from onyx.regulatory.amendments.models import AmendmentInstruction
from onyx.regulatory.amendments.ranker import CandidateChunk
from onyx.regulatory.amendments.structural_target import (
    appendix_replacement_attention_message,
    parse_amendment_structural_target,
    source_identity_distinguishing_tokens,
    source_identity_matches,
)


def test_parses_batch10_article_and_clause_without_requiring_paragraph_metadata() -> (
    None
):
    instruction = AmendmentInstruction(
        instruction_text=(
            "Gümrük Genel Tebliği (TIR İşlemleri) (Seri No: 1)'nin 18 inci "
            "maddesinin altıncı fıkrasının (b) bendinde yer alan “Gümrük ve "
            "Ticaret Bölge Müdürlüklerine” ibaresi “Gümrük ve Dış Ticaret "
            "Bölge Müdürlüklerine” şeklinde değiştirilmiştir."
        ),
        article_reference="Madde 18",
    )

    target = parse_amendment_structural_target(instruction)

    assert target is not None
    assert target.article_no == "18"
    assert target.clause_label == "b"
    assert target.appendix_label is None


def test_structural_candidates_require_the_explicit_instrument_identity() -> None:
    target_source = "Gümrük Genel Tebliği (TIR İşlemleri) (Seri No: 1)"

    assert source_identity_matches(
        target_source,
        "gumruk_genel_tebligi_x1tir_islemlerix2_x1seri_no_1x2.docx",
    )
    assert not source_identity_matches(
        target_source,
        "gumruk_genel_tebligi_x1transit_rejimix2_seri_no_8.docx",
    )
    assert source_identity_distinguishing_tokens(target_source) == (
        "1",
        "islemleri",
        "tir",
    )


def test_appendix_target_without_attached_body_reports_found_not_unmatched() -> None:
    instruction = AmendmentInstruction(
        instruction_text=(
            "Gümrük Genel Tebliği (TIR İşlemleri) (Seri No: 1)'nin "
            "EK-10’u ekteki şekilde değiştirilmiştir."
        )
    )
    candidates = [
        CandidateChunk(
            chunk_id=f"ek-10-{index}",
            user_file_id="00000000-0000-0000-0000-000000000123",
            text=f"EK-10 part {index}",
            metadata={"appendix_label": "EK-10"},
        )
        for index in range(3)
    ]

    message = appendix_replacement_attention_message(instruction, candidates)

    assert message is not None
    assert "Target found: EK-10 (3 chunks)" in message
    assert "replacement appendix content" in message
    assert instruction.instruction_text in message


def test_appendix_instruction_with_inline_replacement_body_can_continue() -> None:
    instruction = AmendmentInstruction(
        instruction_text=(
            "EK-3’ü ekteki şekilde değiştirilmiştir.\n"
            "EK-3\n"
            "Sıra No | Ülke | Teminat tutarı\n"
            "1 | Türkiye | 100.000 EUR"
        )
    )
    candidates = [
        CandidateChunk(
            chunk_id="ek-3",
            user_file_id="00000000-0000-0000-0000-000000000123",
            text="old appendix",
            metadata={"appendix_label": "EK-3"},
        )
    ]

    assert appendix_replacement_attention_message(instruction, candidates) is None


def test_multi_chunk_appendix_with_inline_body_still_refuses_partial_update() -> None:
    instruction = AmendmentInstruction(
        instruction_text=(
            "EK-4’ü ekteki şekilde değiştirilmiştir.\n"
            "EK-4\n"
            "Sıra No | Açıklama | Tutar\n"
            "1 | Yeni düzenleme | 100.000 EUR"
        )
    )
    candidates = [
        CandidateChunk(
            chunk_id=f"ek-4-{index}",
            user_file_id="00000000-0000-0000-0000-000000000123",
            text=f"old appendix part {index}",
            metadata={"appendix_label": "EK-4"},
        )
        for index in range(2)
    ]

    message = appendix_replacement_attention_message(instruction, candidates)

    assert message is not None
    assert "atomic multi-chunk replacement" in message
    assert "no partial proposal" in message


def test_trailing_boilerplate_is_not_treated_as_attached_appendix_body() -> None:
    instruction = AmendmentInstruction(
        instruction_text=(
            "EK-3’ü ekteki şekilde değiştirilmiştir.\n"
            "Bu Tebliğ yayımı tarihinde yürürlüğe girer.\n"
            "Bu Tebliğ hükümlerini Bakan yürütür."
        )
    )
    candidates = [
        CandidateChunk(
            chunk_id="ek-3",
            user_file_id="00000000-0000-0000-0000-000000000123",
            text="old appendix",
            metadata={"appendix_label": "EK-3"},
        )
    ]

    message = appendix_replacement_attention_message(instruction, candidates)

    assert message is not None
    assert "No replacement appendix content" in message
