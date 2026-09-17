"""What one instruction is allowed to learn from the rest of its amendment.

Instructions are matched and drafted one at a time, so the article stating when
the amendment enters into force reaches them only through this context.
"""

from onyx.regulatory.amendments.amendment_context import (
    build_amendment_context,
    commencement_provisions,
)

_AMENDMENT = """GÜMRÜK GENEL TEBLİĞİNDE DEĞİŞİKLİK YAPILMASINA DAİR TEBLİĞ

MADDE 1- 30/12/2020 tarihli Gümrük Genel Tebliği'nin 4 üncü maddesinin birinci \
fıkrası aşağıdaki şekilde değiştirilmiştir.

MADDE 2- Aynı Tebliğ'in 7 nci maddesine aşağıdaki fıkra eklenmiştir.
"(6) Bu fıkra kapsamındaki izin, ilgili genelgenin yürürlüğe girdiği tarihten \
itibaren uygulanır."

MADDE 22- Bu Tebliğin 3 üncü maddesi 1/1/2027 tarihinde, diğer maddeleri yayımı \
tarihinde yürürlüğe girer.

MADDE 23- Bu Tebliğ hükümlerini Ticaret Bakanı yürütür."""


def test_the_commencement_article_is_found_among_the_other_articles() -> None:
    assert commencement_provisions(_AMENDMENT) == [
        "MADDE 22- Bu Tebliğin 3 üncü maddesi 1/1/2027 tarihinde, diğer "
        "maddeleri yayımı tarihinde yürürlüğe girer."
    ]


def test_an_entry_into_force_inside_amended_wording_is_not_the_commencement() -> None:
    """MADDE 2 quotes a provision that speaks about some *other* text taking
    effect. Treating that as this amendment's commencement would date every
    instruction from it."""

    assert not any(
        "7 nci maddesine" in provision
        for provision in commencement_provisions(_AMENDMENT)
    )


def test_the_context_carries_the_whole_amendment_and_its_commencement() -> None:
    context = build_amendment_context(_AMENDMENT)

    assert context is not None
    section = context.prompt_section()
    assert "MADDE 1-" in section
    assert "yürürlüğe girer" in section
    assert "never apply a change that appears here but is not in them" in section


def test_empty_text_has_no_context_to_share() -> None:
    assert build_amendment_context("   \n  ") is None
