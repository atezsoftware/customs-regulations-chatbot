"""Markdown conversion wrappers cannot erase a provision's legal boundaries."""

import re

import pytest

from onyx.natural_language_processing.utils import TiktokenTokenizer
from onyx.regulatory.chunker import (
    ChunkedDocument,
    RegulatoryChunker,
    _markdown_to_blocks,
)
from onyx.regulatory.document_intake.chunking import IntakeRegulatoryChunker
from onyx.regulatory.document_intake.text import require_chunk_source_coverage


def _intake(source: str) -> ChunkedDocument:
    return IntakeRegulatoryChunker(
        tokenizer=TiktokenTokenizer("text-embedding-3-large"),
        content_token_limit=512,
        table_content_token_limit=2048,
        chunking_version="5",
        retain_markdown_headings=True,
    ).chunk_text(source, source_file="regulation.md")


@pytest.mark.parametrize("prefix", ["- ", "+ ", "* ", "4. ", "4) "])
def test_markdown_wrapped_clause_keeps_its_article_and_paragraph(prefix: str) -> None:
    source = f"# Örnek Yönetmelik\n\nMADDE 4- (1) Tanımlar;\n\n{prefix}ç) Kapasite tanımıdır.\n\nd) Bakanlık tanımıdır."
    result = _intake(source)
    clauses = [
        chunk for chunk in result.chunks if chunk.metadata.chunk_variant == "atomic"
    ]
    assert [
        (
            chunk.metadata.clause_label,
            chunk.metadata.article_no,
            chunk.metadata.paragraph_no,
        )
        for chunk in clauses
    ] == [("ç", "4", "1"), ("d", "4", "1")]
    assert prefix + "ç) Kapasite tanımıdır." in clauses[0].text


def test_double_letter_legal_clauses_have_independent_identity() -> None:
    source = "# Örnek Yönetmelik\n\nMADDE 4- (1) Tanımlar;\n\naa) Birinci tanım.\n\nbb) İkinci tanım.\n\ncc) Üçüncü tanım.\n\nçç) Dördüncü tanım."
    result = _intake(source)
    clauses = [
        chunk for chunk in result.chunks if chunk.metadata.chunk_variant == "atomic"
    ]
    assert [chunk.metadata.clause_label for chunk in clauses] == [
        "aa",
        "bb",
        "cc",
        "çç",
    ]
    assert all(
        chunk.metadata.article_no == "4" and chunk.metadata.paragraph_no == "1"
        for chunk in clauses
    )


def test_consecutive_inline_double_letter_clauses_keep_exact_source_spans() -> None:
    source = "# Örnek Yönetmelik\n\nMADDE 4- (1) Tanımlar;\n\ncc) İlk tanımı, çç) İkinci tanımı, dd) Üçüncü tanımı, ifade eder."
    result = _intake(source)
    clauses = [
        chunk for chunk in result.chunks if chunk.metadata.chunk_variant == "atomic"
    ]
    assert [chunk.metadata.clause_label for chunk in clauses] == ["cc", "çç", "dd"]
    assert clauses[1].text == "çç) İkinci tanımı,"
    assert clauses[2].text == "dd) Üçüncü tanımı, ifade eder."
    for chunk in clauses:
        metadata = chunk.metadata
        assert re.sub(
            r"\s+", "", source[metadata.source_start_char : metadata.source_end_char]
        ) == re.sub(r"\s+", "", chunk.text)
    require_chunk_source_coverage(source, retain_markdown_headings=True)


_HEADER = """## Resmî Gazete

Ulaştırma ve Altyapı Bakanlığından:

## TEHLİKELİ MADDELERİN KARAYOLUYLA TAŞINMASI HAKKINDA YÖNETMELİK

## BİRİNCİ BÖLÜM

## Başlangıç Hükümleri

## YÖNETMELİK"""

_SOURCE = f"""## Amaç

MADDE 1- (1) Güvenli taşımayı düzenler.

## Tanımlar

MADDE 4- (1) Bu Yönetmelikte geçen;

- a) ADR: Anlaşmayı,
- b) Belge: Geçerli belgeyi,
- c) Alıcı: Eşyayı teslim alanı,
4. ç) Asgari kapasite: Taşıma kapasitesini,
- d) Bakanlık: Yetkili bakanlığı,
- e) Boşaltan: Eşyayı indiren işletmeyi,
- f) Bölge Müdürlüğü: Yetkili bölge müdürlüğünü,

{_HEADER}

g) Dolduran: Tankı dolduran işletmeyi,

ğ) Elektronik denetim: Uzaktan denetimi,

z) Taşıt belgesi: Geçerli belgeyi,

aa) Tehlikeli madde: ADR'deki maddeleri,

bb) Faaliyet belgesi: İşletmenin belgesini,

cc) Faaliyet raporu: İşletme raporunu, çç) Danışman: Yetkili kişiyi, dd) Danışmanlık: Sunulan hizmeti, ifade eder.

(2) Diğer tanımlar için ADR esas alınır.

## Genel kurallar

MADDE 5- (1) Yasak maddeler taşınamaz.
"""


def test_displaced_publication_header_does_not_break_article4_scope() -> None:
    result = _intake(_SOURCE)
    clauses = [
        chunk
        for chunk in result.chunks
        if chunk.metadata.chunk_variant == "atomic" and chunk.metadata.clause_label
    ]
    assert [chunk.metadata.clause_label for chunk in clauses] == [
        "a",
        "b",
        "c",
        "ç",
        "d",
        "e",
        "f",
        "g",
        "ğ",
        "z",
        "aa",
        "bb",
        "cc",
        "çç",
        "dd",
    ]
    assert all(
        chunk.metadata.article_no == "4" and chunk.metadata.paragraph_no == "1"
        for chunk in clauses
    )
    assert all("MADDE 4 - Tanımlar" in chunk.metadata.heading_path for chunk in clauses)
    assert all(
        "BİRİNCİ BÖLÜM" in " ".join(chunk.metadata.heading_path) for chunk in clauses
    )
    assert (
        result.metadata.title
        == "TEHLİKELİ MADDELERİN KARAYOLUYLA TAŞINMASI HAKKINDA YÖNETMELİK"
    )
    assert not any(
        "Bakanlığından:" in chunk.text or "## BİRİNCİ BÖLÜM" in chunk.text
        for chunk in result.chunks
    )
    require_chunk_source_coverage(_SOURCE, retain_markdown_headings=True)
    assert result.content == _SOURCE.strip()


@pytest.mark.parametrize("replacement", ["h) Dolduran", "> g) Dolduran"])
def test_uncertain_header_continuation_is_not_silently_removed(
    replacement: str,
) -> None:
    source = _SOURCE.replace("g) Dolduran", replacement)
    blocks = _markdown_to_blocks(source)
    assert all(block.kind != "document_header" for block in blocks)
    require_chunk_source_coverage(source, retain_markdown_headings=True)


def test_quoted_inline_double_labels_are_not_new_legal_units() -> None:
    source = "# Örnek Yönetmelik\n\nMADDE 4- (1) Tanımlar;\n\ncc) Metinde “çç) örnek, dd) örnek” ifadeleri korunur."
    result = _intake(source)
    assert [
        chunk.metadata.clause_label
        for chunk in result.chunks
        if chunk.metadata.chunk_variant == "atomic"
    ] == ["cc"]
    assert "“çç) örnek, dd) örnek”" in result.chunks[0].text


def test_table_and_fenced_header_are_not_reinterpreted() -> None:
    source = f"```markdown\n{_SOURCE}\n```\n\n| Kod | Tanım |\n| --- | --- |\n| cc) a | çç) b dd) c |"
    blocks = _markdown_to_blocks(source)
    assert all(block.kind != "document_header" for block in blocks)
    table = next(block for block in blocks if block.kind == "table")
    assert "| cc) a | çç) b dd) c |" in table.text


def test_parenthesized_roman_marker_keeps_existing_interpretation() -> None:
    result = RegulatoryChunker().chunk_text(
        "MADDE 4-\n\n(i) İlk alt bent.\n\n(ii) İkinci alt bent."
    )
    assert not any(chunk.metadata.chunk_type == "clause" for chunk in result.chunks)


def test_unclosed_quote_cannot_create_inline_legal_units() -> None:
    source = "MADDE 4- (1) Tanımlar;\n\ncc) “çç) Örnek, dd) Örnek"
    result = _intake(source)
    assert [
        chunk.metadata.clause_label
        for chunk in result.chunks
        if chunk.metadata.chunk_variant == "atomic"
    ] == ["cc"]


def test_multiple_publication_headers_are_not_assumed_to_be_one_document() -> None:
    source = _SOURCE + "\n\n" + _HEADER
    assert all(block.kind != "document_header" for block in _markdown_to_blocks(source))


def test_supported_page_marker_does_not_shift_verified_header_values() -> None:
    source = _SOURCE.replace("Bakanlığından:\n", "Bakanlığından:\n\n- 587 -\n")
    result = _intake(source)
    assert (
        result.metadata.title
        == "TEHLİKELİ MADDELERİN KARAYOLUYLA TAŞINMASI HAKKINDA YÖNETMELİK"
    )
    assert all(
        "BİRİNCİ BÖLÜM — Başlangıç Hükümleri" in chunk.metadata.heading_path
        for chunk in result.chunks
        if chunk.metadata.article_no == "4"
    )
    require_chunk_source_coverage(source, retain_markdown_headings=True)
    assert result.content == source.strip()


@pytest.mark.parametrize(
    "literal",
    [
        "`bentler çç) örnek, dd) örnek`",
        "``bentler ` çç) örnek, dd) örnek``",
        "‘bentler çç) örnek, dd) örnek’",
    ],
)
def test_inline_literal_examples_are_not_new_legal_clauses(literal: str) -> None:
    source = f"MADDE 4- (1) Tanımlar;\n\ncc) Metinde {literal} ifadeleri korunur."
    result = _intake(source)
    assert [
        chunk.metadata.clause_label
        for chunk in result.chunks
        if chunk.metadata.chunk_variant == "atomic"
    ] == ["cc"]
    assert literal in result.chunks[0].text
