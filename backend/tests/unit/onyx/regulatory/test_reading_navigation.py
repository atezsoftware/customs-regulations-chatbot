from pathlib import Path

import pytest

from onyx.regulatory.reading_navigation import remove_reading_navigation

FIXTURE = Path(__file__).parent / "fixtures" / "customs_law_introduction.md"
PROTOCOL_BODY = (
    "**BAŞLIK I**\n\n**GENEL HÜKÜMLER**\n\n"
    "**Madde 1**\n\n**Tanımlar**\n\n"
    "Bu Protokol bakımından aşağıdaki tanımlar uygulanır.\n\n"
    "**Madde 2**\n\n**Genel koşullar**\n\n"
    "Bu Protokol kapsamında gerekli genel koşullar sağlanır.\n\n"
    "**Madde 3**\n\n**Kümülasyon**\n\n"
    "Taraflar arasındaki kümülasyon hükümleri uygulanır.\n"
)
CONTENTS = (
    "**İÇİNDEKİLER**\n\n**BAŞLIK I GENEL HÜKÜMLER**\n\n"
    "Madde 1\tTanımlar\n\nMadde 2\tGenel koşullar\n\n"
    "Madde 3\tKümülasyon\n\n"
)


def test_customs_law_unlabelled_navigation_preserves_exact_prefix_and_body() -> None:
    source = FIXTURE.read_text()
    prefix = source[: source.index("**I.Kısım - I. Bölüm**")]
    body = source[source.index("#######\n", len(prefix)) :]
    assert remove_reading_navigation(source) == prefix + body
    assert "**BİRİNCİ BÖLÜM**" in body
    assert "**MADDE 1-** Bu Kanunun amacı" in body


def test_customs_law_with_article_heading_separate_from_body() -> None:
    source = FIXTURE.read_text()
    navigation_start = source.index("**I.Kısım - I. Bölüm**")
    body_start = source.index("#######\n", navigation_start)
    # The library can expose normalized Markdown headings on their own lines.
    body = (
        source[body_start:]
        .replace("####### ", "## ")
        .replace("**MADDE 1-** ", "### MADDE 1-\n\n")
    )
    assert remove_reading_navigation(source[:body_start] + body) == (
        source[:navigation_start] + body
    )


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("embedded", [False, True])
def test_headed_contents_preserves_preamble_and_protocol_body(
    newline: str, embedded: bool
) -> None:
    prefix = "# Anlaşma\n\nYayın tarihi: 01.01.2020\n\n"
    if embedded:
        prefix += "**Madde 1**\n\nEkli Protokol değiştirilmiştir.\n\n# Protokol\n\n"
    source = (prefix + CONTENTS + PROTOCOL_BODY).replace("\n", newline)
    expected = (prefix + PROTOCOL_BODY).replace("\n", newline)
    assert remove_reading_navigation(source) == expected
    assert remove_reading_navigation(expected) == expected


def test_linked_bulleted_contents_and_inline_article_titles() -> None:
    contents = (
        "## İçindekiler\n\n"
        "- [Madde 1: Tanımlar](#madde-1)\n"
        "- [Madde 2: Genel koşullar](#madde-2)\n"
        "- [Madde 3: Kümülasyon](#madde-3)\n\n"
    )
    body = (
        "**Madde 1. Tanımlar**\n\nTanımlar bu Protokol kapsamında geçerlidir.\n\n"
        "**Madde 2. Genel koşullar**\n\nGenel koşullar bu Protokolde belirlenir.\n\n"
        "**Madde 3. Kümülasyon**\n\nKümülasyon hükümleri taraflar arasında uygulanır.\n"
    )
    assert remove_reading_navigation(contents + body) == body


def test_preserves_operative_preamble_after_contents() -> None:
    preamble = "Taraflar aşağıdaki hükümler üzerinde anlaşmışlardır.\n\n"
    assert remove_reading_navigation(CONTENTS + preamble + PROTOCOL_BODY) == (
        preamble + PROTOCOL_BODY
    )


def test_contents_includes_wrapped_article_and_annex_navigation() -> None:
    contents = (
        CONTENTS + "Madde 4 Önceden düzenlenmiş bir belgeye\n\n"
        "istinaden sertifika düzenlenmesi\n\n"
        "Madde 4 a Tedarikçi beyanı\n\n"
        "**Eklerin Listesi**\n\n"
        "Ek I: Giriş notları\n\nEk III a: Belge örnekleri\n\n"
        "**Ortak Deklarasyon**\n\n"
    )
    body = PROTOCOL_BODY + (
        "**Madde 4**\n\n"
        "**Önceden düzenlenmiş bir belgeye istinaden sertifika düzenlenmesi**\n\n"
        "Bu madde kapsamında sertifika düzenlenmesine izin verilir.\n\n"
        "**Madde 4 a**\n\n**Tedarikçi beyanı**\n\n"
        "Bu madde kapsamında tedarikçi beyanı düzenlenir.\n"
    )
    assert remove_reading_navigation(contents + body) == body


def test_wrapped_long_heading_can_have_a_different_suffix_in_body() -> None:
    prefix = "Önceden düzenlenmiş veya hazırlanmış bir menşe ispat belgesine"
    contents = CONTENTS + (
        f"Madde 4 {prefix}\n\nistinaden sertifika düzenlenmesi\n\nMadde 5 Diğer koşullar\n\n"
    )
    body = PROTOCOL_BODY + (
        f"**Madde 4**\n\n**{prefix} istinaden sertifikalarının düzenlenmesi**\n\n"
        "Bu madde kapsamında sertifikalar düzenlenir.\n\n"
        "**Madde 5**\n\n**Diğer koşullar**\n\nDiğer koşullar uygulanır.\n"
    )
    assert remove_reading_navigation(contents + body) == body


def test_legal_footnote_is_not_a_wrapped_title_suffix() -> None:
    title = "Önceden düzenlenmiş veya hazırlanmış bir menşe ispat belgesine istinaden sertifika düzenlenmesi"
    contents = CONTENTS + (
        f"Madde 4 {title}\n\n(*) Bu maddenin kapsamına serbest bölgeler dahil değildir\n\n"
        "Madde 5 Diğer koşullar\n\n"
    )
    body = PROTOCOL_BODY + f"**Madde 4**\n\n**{title}**\n\nBu hüküm uygulanır.\n"
    assert remove_reading_navigation(contents + body) == contents + body


def test_dotted_navigation_stops_before_operative_prose() -> None:
    body = "# Amaç\n# Kapsam\n# Dayanak\n\n"
    contents = "İÇİNDEKİLER\n\nAmaç ... 1\nKapsam ... 2\nDayanak ... 3\n\n"
    prose = "Bu bölümde aşağıdaki kurallar uygulanır.\n\nDiğer ... 4\n"
    assert remove_reading_navigation(body + contents + prose) == body + "\n" + prose


def test_uncorroborated_continuation_does_not_leave_a_partial_contents_list() -> None:
    source = (
        CONTENTS
        + "Madde 4 Bir başlık\n\nTanınmayan devam\n\nMadde 5 Diğer başlık\n\n"
        + PROTOCOL_BODY
    )
    assert remove_reading_navigation(source) == source


def test_body_structure_absent_from_contents_is_preserved() -> None:
    contents = CONTENTS.replace("**BAŞLIK I GENEL HÜKÜMLER**\n\n", "")
    assert remove_reading_navigation(contents + PROTOCOL_BODY) == PROTOCOL_BODY


def test_unlabelled_article_navigation_is_also_removed() -> None:
    contents = CONTENTS.replace("**İÇİNDEKİLER**\n\n", "")
    assert remove_reading_navigation(contents + PROTOCOL_BODY) == PROTOCOL_BODY


def test_mixed_roman_and_arabic_navigation_is_removed_completely() -> None:
    contents = (
        "**I.Kısım - I. Bölüm**\n\n**I.Kısım - II. Bölüm**\n\n"
        "**II.Kısım - I. Bölüm**\n\n**II.Kısım - 2. Bölüm**\n\n"
    )
    body = "## BİRİNCİ KISIM\n\n### MADDE 1-\n\nBu Kanunun genel hükümleri ve kuralları uygulanır.\n"
    assert remove_reading_navigation(contents + body) == body


def test_unknown_navigation_variant_keeps_the_whole_block() -> None:
    contents = (
        "**I.Kısım - I. Bölüm**\n\n**I.Kısım - II. Bölüm**\n\n"
        "**II.Kısım - I. Bölüm**\n\n**II.Kısım - Özel Bölüm**\n\n"
    )
    body = "## BİRİNCİ KISIM\n\n### MADDE 1-\n\nBu Kanunun genel hükümleri ve kuralları uygulanır.\n"
    assert remove_reading_navigation(contents + body) == contents + body


def test_punctuated_unrecognized_entry_keeps_the_whole_contents() -> None:
    source = CONTENTS + "Madde 4 Diğer başlık.\n\n" + PROTOCOL_BODY
    assert remove_reading_navigation(source) == source


def test_wrapped_annex_navigation_is_removed_but_following_preamble_survives() -> None:
    annexes = (
        "**Eklerin Listesi**\n\nEk II: Menşe statüsü için menşeli\n\n"
        "olmayan maddelere uygulanacak işlemler listesi\n\n"
        "Ek III: Başvuru formu örnekleri\n\n"
    )
    preamble = "Taraflar aşağıdaki hükümlerde anlaşmıştır\n\n"
    assert remove_reading_navigation(CONTENTS + annexes + preamble + PROTOCOL_BODY) == (
        preamble + PROTOCOL_BODY
    )


def test_footnote_between_contents_and_annex_list_is_preserved() -> None:
    footnote = "1 1998 tarihli Ortak Komite Kararı ile değiştirilmiştir\n\n"
    annexes = "**Eklerin Listesi**\n\nEk I: Giriş notları\n\n"
    assert remove_reading_navigation(CONTENTS + footnote + annexes + PROTOCOL_BODY) == (
        footnote + PROTOCOL_BODY
    )


def test_contents_structure_with_missing_ocr_number() -> None:
    contents = CONTENTS.replace("BAŞLIK I GENEL", "BAŞLIK GENEL")
    assert remove_reading_navigation(contents + PROTOCOL_BODY) == PROTOCOL_BODY


def test_tabular_contents_is_removed_without_changing_body_tables() -> None:
    contents = (
        "**İÇİNDEKİLER**\n\n| BAŞLIK I | GENEL HÜKÜMLER |\n"
        "| --- | --- |\n| Madde 1 | Tanımlar |\n"
        "| Madde 2 | Genel koşullar |\n| Madde 3 | Kümülasyon |\n\n"
    )
    body = PROTOCOL_BODY + "\n| Eşya | Vergi |\n| --- | --- |\n| A | 5 |\n"
    assert remove_reading_navigation(contents + body) == body


def test_trailing_dotted_contents_is_corroborated_against_earlier_headings() -> None:
    body = "# Amaç\n\nBu kurallar uygulanır.\n\n# Kapsam\n\nEşya kapsamdadır.\n\n# Dayanak\n\nKanuna dayanır.\n\n"
    contents = (
        "**İÇİNDEKİLER**\n\n**BİRİNCİ BÖLÜM**\n\n"
        "Amaç ……………………… 1 1\n\nKapsam ............ 2 1\n\n"
        "Dayanak ……………………… 3 1\n\n"
    )
    assert remove_reading_navigation(body + contents + "İmza\n") == body + "\nİmza\n"


def test_uncorroborated_dotted_text_is_preserved() -> None:
    source = "İÇİNDEKİLER\n\nAmaç ... 1\nKapsam ... 2\nDayanak ... 3\n"
    assert remove_reading_navigation(source) == source


def test_dotted_contents_does_not_consume_a_following_fee_schedule() -> None:
    body = "# Amaç\n# Kapsam\n# Dayanak\n\n"
    contents = "İÇİNDEKİLER\n\nAmaç ... 1\nKapsam ... 2\nDayanak ... 3\n\n"
    fees = (
        "**EK ÜCRET TARİFESİ**\n\nUygulanacak ücretler\n\n"
        "Başvuru bedeli .... 100\n\n**MADDE 1-** Bu tarifede gösterilen ücretler uygulanır.\n"
    )
    assert remove_reading_navigation(body + contents + fees) == body + "\n" + fees


@pytest.mark.parametrize(
    "source",
    [
        "",
        "# Kanun\n\n**BİRİNCİ KISIM**\n\n**BİRİNCİ BÖLÜM**\n\n"
        "**Amaç**\n\n**MADDE 1-** Bu Kanunun hükümleri uygulanır.\n",
        "**İçindekiler**\n\nMadde 1 Tanımlar\n\nMadde 2 Koşullar\n",
        "İçindekiler ilgililerden başkasına açıklanamaz.\n" + PROTOCOL_BODY,
        CONTENTS + "Bu belgenin asıl metni henüz mevcut değildir.\n",
        CONTENTS + "**Madde 1-** Bu hükümler uygulanır.\n",
        "**I.Kısım - I. Bölüm**\n\n**I.Kısım - II. Bölüm**\n\n"
        "**II.Kısım - I. Bölüm**\n\nAçıklama metni.\n",
        "# Kanun\n\n**MADDE 1-** Bu hükümler uygulanır.\n\n"
        "**I.Kısım - I. Bölüm**\n\n**I.Kısım - II. Bölüm**\n\n"
        "**II.Kısım - I. Bölüm**\n\n**BİRİNCİ KISIM**\n\n"
        "**MADDE 1-** Bu diğer hükümler uygulanır.\n",
        "```markdown\n" + CONTENTS + PROTOCOL_BODY + "```\n",
        "**İÇİNDEKİLER**\n\n| Madde | Başlık |\n| --- | --- |\n"
        "| 1 | Tanımlar |\n\n" + PROTOCOL_BODY,
    ],
)
def test_leaves_ordinary_or_uncorroborated_text_unchanged(source: str) -> None:
    assert remove_reading_navigation(source) == source


def test_preserves_operative_sentence_even_when_followed_by_real_contents() -> None:
    operative = "**İçindekiler**\n\nMadde 1- Bu Kanunun hükümleri uygulanır.\n\n"
    assert remove_reading_navigation(operative + CONTENTS + PROTOCOL_BODY) == (
        operative + PROTOCOL_BODY
    )
