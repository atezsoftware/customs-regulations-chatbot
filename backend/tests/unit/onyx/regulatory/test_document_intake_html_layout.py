from hashlib import sha256
from uuid import uuid4

import pytest
from bs4 import BeautifulSoup

from onyx.file_processing.html_utils import parse_html_page_basic
from onyx.natural_language_processing.utils import TiktokenTokenizer
from onyx.regulatory.chunker import RegulatoryChunker, _markdown_to_blocks
from onyx.regulatory.document_intake.chunking import IntakeRegulatoryChunker
from onyx.regulatory.document_intake.headings import promote_plain_article_titles
from onyx.regulatory.document_intake.html import canonical_legal_html
from onyx.regulatory.document_intake.models import IntakeSourceAsset
from onyx.regulatory.document_intake.preparation import _canonical_source_text
from onyx.regulatory.document_intake.text import prepare_intake_text
from onyx.regulatory.document_intake.versions import reprepare_target_versions

# Native Word HTML uses one cell for the entire legal body, not a data grid.
_HTML = """<body><table><tr><td><table>
<tr><td><p>18 Haziran 2022 CUMARTESİ</p></td>
<td><p>Resmî Gazete</p></td><td><p>Sayı : 31870</p></td></tr>
<tr><td colspan="3"><p>YÖNETMELİK</p></td></tr>
<tr><td colspan="3">
<p>Ulaştırma ve Altyapı Bakanlığından:</p>
<p>TEHLİKELİ MADDE GÜVENLİK DANIŞMANLIĞI HİZMETLERİ HAKKINDA YÖNETMELİK</p>
<p>BİRİNCİ BÖLÜM</p><p>Başlangıç Hükümleri</p>
<p>Dayanak</p><p><b>MADDE 3-</b> (1) Bu Yönetmelik, 26/9/2011 tarihli ve
655 sayılı Kanun Hükmünde Kararnamenin 28 inci maddesine dayanılarak hazırlanmıştır.</p>
<p>Tanımlar</p><p><b>MADDE 4-</b> (1) Bu Yönetmelikte geçen;</p>
<p>a) ADR: Tehlikeli Malların Karayolu ile Uluslararası Taşımacılığına İlişkin Anlaşmayı,</p>
<p>Yürürlük</p><p><b>MADDE 27-</b> (1) Bu Yönetmelik yayımı tarihinde yürürlüğe girer.</p>
<p>Yürütme</p><p><b>MADDE 28-</b> (1) Bu Yönetmelik hükümlerini Ulaştırma ve Altyapı Bakanı yürütür.</p>
</td></tr></table></td></tr></table></body>"""


def _render(html: str) -> str:
    return canonical_legal_html(BeautifulSoup(html, "html.parser"))


def test_actual_word_container_restores_article_boundaries_without_changing_words() -> (
    None
):
    baseline = parse_html_page_basic(_HTML)
    assert "Tanımlar MADDE 4-" in baseline
    assert not any(
        chunk.metadata.article_no
        for chunk in RegulatoryChunker().chunk_text(baseline).chunks
    )
    restored = _render(_HTML)
    assert "Tanımlar\nMADDE 4-" in restored
    assert "Yürürlük\nMADDE 27-" in restored
    assert "Yürütme\nMADDE 28-" in restored
    assert "".join(restored.split()) == "".join(baseline.split())
    assert _render(_HTML) == restored
    normalized = prepare_intake_text(restored)
    result = IntakeRegulatoryChunker(
        tokenizer=TiktokenTokenizer("text-embedding-3-small"), content_token_limit=312
    ).chunk_text(normalized)
    atomic = [row for row in result.chunks if row.metadata.chunk_variant == "atomic"]
    assert {row.metadata.article_no for row in atomic} >= {"3", "4", "27", "28"}
    assert (
        next(
            row for row in atomic if row.metadata.article_no == "4"
        ).metadata.article_title
        == "Tanımlar"
    )
    for row in atomic:
        if row.metadata.article_no == "27":
            assert "Bakanı yürütür" not in row.text
        if row.metadata.article_no == "28":
            assert "yayımı tarihinde" not in row.text
    assert all(
        len(TiktokenTokenizer("text-embedding-3-small").encode(row.text)) <= 312
        for row in result.chunks
    )


def test_real_table_grid_and_following_article_keep_their_own_boundaries() -> None:
    grid = "<table><tr><td>Kod</td><td>Oran</td></tr><tr><td>0101</td><td>%5</td></tr></table>"
    html = _HTML.replace("<p>Tanımlar</p>", grid + "<p>Tanımlar</p>")
    restored = _render(html)
    assert "Kod\tOran\n\t0101\t%5\nTanımlar\nMADDE 4-" in restored
    assert "".join(restored.split()) == "".join(parse_html_page_basic(html).split())


def test_comparison_grid_and_midprose_references_are_not_layout_containers() -> None:
    html = """<table><tr><th>Mevcut hüküm</th><th>Yeni hüküm</th></tr>
    <tr><td><p>MADDE 1- (1) Beyan edilir.</p><p>MADDE 2- (1) Saklanır.</p></td>
    <td><p>Yürürlükten kaldırılmıştır.</p></td></tr></table>
    <p>Bu işlem MADDE 5 kapsamında değerlendirilir.</p>"""
    assert _render(html) == parse_html_page_basic(html)


def test_legacy_html_formatter_and_common_update_remain_exact() -> None:
    asset = IntakeSourceAsset(
        id=uuid4(),
        package_id=uuid4(),
        sha256=sha256(_HTML.encode()).hexdigest(),
        file_id="raw",
        text_file_id="prepared",
        mime_type="text/html",
        display_name="source",
        byte_count=len(_HTML.encode()),
    )
    assert _canonical_source_text(
        asset, _HTML.encode(), "frozen", formatter_version=3
    ) == parse_html_page_basic(_HTML)
    assert _canonical_source_text(
        asset, _HTML.encode(), "frozen", formatter_version=4
    ) == _render(_HTML)
    assert "Tanımlar MADDE 4-" in parse_html_page_basic(_HTML)
    assert reprepare_target_versions({"canonical": []}) == ("2", 3)
    assert reprepare_target_versions(
        {"target_chunking_version": "3", "target_formatter_version": 4}
    ) == ("3", 4)


@pytest.mark.parametrize(
    "entry",
    [
        {"target_chunking_version": "3"},
        {"target_formatter_version": 4},
        {"target_chunking_version": "3", "target_formatter_version": True},
        {"target_chunking_version": "9", "target_formatter_version": 4},
        {"target_chunking_version": "3", "target_formatter_version": 99},
    ],
)
def test_reprepare_target_version_audit_rejects_unknown_or_partial_values(
    entry: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="target versions"):
        reprepare_target_versions(entry)


@pytest.mark.parametrize(
    "source,expected",
    [
        (
            "GEÇİCİ\nMADDE 1-\n(1) Bu hüküm uygulanır.",
            "GEÇİCİ MADDE 1-\n(1) Bu hüküm uygulanır.",
        ),
        ("MADDE 10\n–\n(1) Bu hüküm uygulanır.", "MADDE 10–\n(1) Bu hüküm uygulanır."),
    ],
)
def test_wrapped_definition_restores_only_whitespace(
    source: str, expected: str
) -> None:
    actual = prepare_intake_text(source)
    assert actual == expected
    assert "".join(actual.split()) == "".join(source.split())
    assert prepare_intake_text(actual) == actual


def test_wrapped_midprose_reference_and_markdown_code_do_not_become_definitions() -> (
    None
):
    for source in [
        "Bu hüküm GEÇİCİ\nMADDE 1-\n(1) uyarınca uygulanır.",
        "MADDE 10\n– uyarınca gerekli belgeler sunulur.",
        "```text\nGEÇİCİ\nMADDE 1-\n(1) Örnek metindir.\n```",
        "| MADDE 10 | – | (1) Açıklama |",
    ]:
        assert prepare_intake_text(source) == source


@pytest.mark.parametrize(
    "prose",
    [
        "Bu hükümler uygulanır",
        "Başvurular elektronik ortamda verilir",
        "İşletme bu belgeyi saklamak zorundadır",
        "Bu işlem MADDE 5 kapsamında değerlendirilir.",
    ],
)
def test_operative_tail_before_next_article_does_not_become_its_title(
    prose: str,
) -> None:
    source = f"{prose}\nMADDE 2-\n(1) Ayrı bir hüküm uygulanır."
    blocks = _markdown_to_blocks(source)
    assert (
        promote_plain_article_titles(blocks, source_text=source)[0].kind == "paragraph"
    )


def test_plain_title_promotion_preserves_fenced_examples() -> None:
    source = "```text\nTanımlar\nMADDE 2-\n(1) Örnek metindir.\n```"
    blocks = _markdown_to_blocks(source)
    assert promote_plain_article_titles(blocks, source_text=source) == blocks
