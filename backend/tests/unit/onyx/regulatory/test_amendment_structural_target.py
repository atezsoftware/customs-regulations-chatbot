"""Target resolution for real Resmî Gazete amendment instructions.

Every instruction here is taken verbatim from Tebliğ 2026/85, which amends
Tebliğ 2026/2. They share the shape that defeats naive parsing: the sentence
opens with the amending article's own number and only then names the provision
it changes.
"""

import pytest

from onyx.regulatory.amendments.models import AmendmentInstruction
from onyx.regulatory.amendments.new_provision_policy import (
    added_subordinate_unit_kind,
    explicitly_adds_top_level_provision,
)
from onyx.regulatory.amendments.ranker import CandidateChunk
from onyx.regulatory.amendments.structural_target import (
    appendix_replacement_attention_message,
    canonical_structural_query_anchor,
    deterministic_structural_candidate,
    parse_amendment_structural_target,
)

_REPLACE_PARAGRAPH = (
    "MADDE 1- 31/12/2025 tarihli ve 33124 (4. Mükerrer) sayılı Resmî "
    "Gazete’de yayımlanan Karayolu Dışında Kullanılan Hareketli Makinaların "
    "İthalat Denetimi Tebliği (Ürün Güvenliği ve Denetimi: 2026/2)’nin 1 inci "
    "maddesinin ikinci fıkrası aşağıdaki şekilde değiştirilmiştir."
)
_REPLACE_CLAUSE = (
    "MADDE 2- Aynı Tebliğin 3 üncü maddesinin birinci fıkrasının (d) bendi "
    "aşağıdaki şekilde değiştirilmiştir."
)
_ADD_CLAUSE = (
    "MADDE 3- Aynı Tebliğin 3 üncü maddesinin birinci fıkrasına (ç) bendinden "
    "sonra gelmek üzere aşağıdaki bent eklenmiş ve diğer bentler buna göre "
    "teselsül ettirilmiştir."
)
_ADD_PARAGRAPH = "MADDE 5- Aynı Tebliğin 7 nci maddesine aşağıdaki fıkra eklenmiştir."
_REPEAL_PARAGRAPH = (
    "MADDE 10- Aynı Tebliğin 11 inci maddesinin üçüncü fıkrası yürürlükten "
    "kaldırılmıştır."
)
_ADD_ARTICLE_CONJOINED = (
    "MADDE 13- Aynı Tebliğe 15 inci maddeden sonra gelmek üzere aşağıdaki "
    "madde eklenmiş ve diğer maddeler buna göre teselsül ettirilmiştir."
)
_ADD_ARTICLE_PLAIN = "MADDE 14- Aynı Tebliğe aşağıdaki geçici madde eklenmiştir."
_ADD_ANNEX_ROW = (
    "MADDE 17- Aynı Tebliğin Ek-2’sinde yer alan listeye aşağıdaki sıra eklenmiştir."
)


@pytest.mark.parametrize(
    ("instruction_text", "article_no", "paragraph_no", "clause_label"),
    [
        (_REPLACE_PARAGRAPH, "1", "2", None),
        (_REPLACE_PARAGRAPH.replace("ikinci", "İkinci"), "1", "2", None),
        (_REPLACE_PARAGRAPH.replace("ikinci", "İKİNCİ"), "1", "2", None),
        (_REPLACE_PARAGRAPH.replace("ikinci", "IKINCI"), "1", "2", None),
        (_REPLACE_CLAUSE, "3", "1", "d"),
        (_ADD_CLAUSE, "3", "1", "ç"),
        (_ADD_PARAGRAPH, "7", None, None),
        (_REPEAL_PARAGRAPH, "11", "3", None),
    ],
)
def test_target_is_the_amended_provision_not_the_amending_article(
    instruction_text: str,
    article_no: str,
    paragraph_no: str | None,
    clause_label: str | None,
) -> None:
    target = parse_amendment_structural_target(
        AmendmentInstruction(instruction_text=instruction_text)
    )

    assert target is not None
    assert target.article_no == article_no
    assert target.paragraph_no == paragraph_no
    assert target.clause_label == clause_label


def test_multiple_amended_paragraphs_stay_at_article_scope() -> None:
    """Two paragraphs of one article change; neither may narrow the target."""

    target = parse_amendment_structural_target(
        AmendmentInstruction(
            instruction_text=(
                "MADDE 8- Aynı Tebliğin 10 uncu maddesinin birinci fıkrasında "
                "yer alan “yirmi iş günü” ibaresi “otuz iş günü”, ikinci "
                "fıkrasında yer alan “Sanayi ve Teknoloji Bakanlığından” "
                "ibaresi “Sanayi ve Teknoloji Bakanlığından veya Bakanlıkça "
                "yetkilendirilen kuruluştan” şeklinde değiştirilmiştir."
            )
        )
    )

    assert target is not None
    assert target.article_no == "10"
    assert target.paragraph_no is None


def test_annex_instruction_resolves_to_its_appendix() -> None:
    target = parse_amendment_structural_target(
        AmendmentInstruction(instruction_text=_ADD_ANNEX_ROW)
    )

    assert target is not None
    assert target.appendix_label == "EK-2"
    assert canonical_structural_query_anchor(target) == "EK-2"


def test_exact_paragraph_repeal_does_not_depend_on_llm_confirmation() -> None:
    instruction = AmendmentInstruction(
        instruction_text=_REPEAL_PARAGRAPH,
        target_source=(
            "Karayolu Dışında Kullanılan Hareketli Makinaların İthalat Denetimi "
            "Tebliği (Ürün Güvenliği ve Denetimi: 2026/2)"
        ),
    )
    exact = CandidateChunk(
        chunk_id="article-11-paragraph-3",
        user_file_id="00000000-0000-0000-0000-000000000001",
        text="(3) Mevcut metin.",
        source_name=(
            "2026-02_ugd_karayolu_disinda_kullanilan_hareketli_makinalarin_"
            "ithalat_denetimi_tebligi.md"
        ),
        metadata={"article_no": "11", "paragraph_no": "3"},
        structured_match=True,
    )

    assert deterministic_structural_candidate(instruction, [exact]) == exact


def test_annex_row_addition_anchors_to_existing_canonical_scope() -> None:
    instruction = AmendmentInstruction(
        instruction_text=_ADD_ANNEX_ROW,
        target_source="Karayolu Dışında Kullanılan Hareketli Makinalar Tebliği",
    )
    exact = CandidateChunk(
        chunk_id="ek-2-part-1",
        user_file_id="00000000-0000-0000-0000-000000000001",
        text="EK-2 mevcut liste",
        source_name="karayolu_disinda_kullanilan_hareketli_makinalar_tebligi.md",
        metadata={"appendix_label": "EK-2"},
        structured_match=True,
    )

    assert deterministic_structural_candidate(instruction, [exact]) == exact


def test_query_anchor_uses_the_forward_designator_retrieval_indexes() -> None:
    """The exact-provision boost only recognizes ``madde N``, never ``N inci``."""

    target = parse_amendment_structural_target(
        AmendmentInstruction(instruction_text=_REPEAL_PARAGRAPH)
    )

    anchor = canonical_structural_query_anchor(target)
    assert anchor is not None
    assert anchor.startswith("madde 11")
    assert "madde 10" not in anchor


def test_surgical_appendix_change_is_not_rejected_for_physical_chunk_count() -> None:
    instruction = AmendmentInstruction(
        instruction_text=(
            "MADDE 18- Aynı Tebliğin Ek-2’sinde yer alan listenin 3 üncü "
            "sırasının MADDE ADI sütununda yer alan “Gantri vinçler” ibaresi "
            "“Portal ve yarı portal gantri vinçler” şeklinde değiştirilmiştir."
        )
    )
    candidates = [
        CandidateChunk(
            chunk_id=f"chunk-{index}",
            user_file_id="00000000-0000-0000-0000-000000000001",
            text=f"EK-2 bölüm {index}",
            metadata={"appendix_label": "EK-2"},
        )
        for index in (1, 2)
    ]

    assert appendix_replacement_attention_message(instruction, candidates) is None


def test_missing_full_appendix_body_still_blocks_partial_replacement() -> None:
    instruction = AmendmentInstruction(
        instruction_text="MADDE 18- Aynı Tebliğin Ek-2’si ekteki şekilde değiştirilmiştir."
    )
    candidate = CandidateChunk(
        chunk_id="chunk-1",
        user_file_id="00000000-0000-0000-0000-000000000001",
        text="EK-2",
        metadata={"appendix_label": "EK-2"},
    )

    message = appendix_replacement_attention_message(instruction, [candidate])

    assert message is not None
    assert "No replacement appendix content" in message


@pytest.mark.parametrize(
    ("instruction_text", "top_level", "unit_kind"),
    [
        (_ADD_ARTICLE_CONJOINED, True, None),
        (_ADD_ARTICLE_PLAIN, True, None),
        (_ADD_CLAUSE, False, "clause"),
        (_ADD_PARAGRAPH, False, "paragraph"),
        (_REPLACE_CLAUSE, False, None),
        (_REPEAL_PARAGRAPH, False, None),
        (_ADD_ANNEX_ROW, False, None),
    ],
)
def test_addition_classification_covers_conjoined_official_drafting(
    instruction_text: str, top_level: bool, unit_kind: str | None
) -> None:
    """``eklenmiş ve ... ettirilmiştir`` is as official as ``eklenmiştir``."""

    assert explicitly_adds_top_level_provision(instruction_text) is top_level
    assert added_subordinate_unit_kind(instruction_text) == unit_kind
