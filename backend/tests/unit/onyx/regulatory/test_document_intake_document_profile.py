"""Intake identity comes from the document header, never browser or cited IDs."""

import re
from dataclasses import asdict

import pytest

from onyx.natural_language_processing.utils import TiktokenTokenizer
from onyx.regulatory.chunker import RegulatoryChunker, _markdown_to_blocks
from onyx.regulatory.document_intake.chunking import IntakeRegulatoryChunker

_URL = "resmigazete.gov.tr/eskiler/2025/01/20250105-3.htm"
_PRINT_HEADER = f"23.10.2026 11:13 {_URL}"
_PRINT_FOOTER = f"https://www.{_URL} 1/3"
_TITLE_LINES = [
    "EV VE BÜRO TİPİ ELEKTRİK-ELEKTRONİK CİHAZLARIN HAZIRDA",
    "BEKLEME KONUMLARININ ELEKTRİK ENERJİSİ TÜKETİMİ İLE",
    "İLGİLİ ÇEVREYE DUYARLI TASARIM GEREKLİLİKLERİNE DAİR TEBLİĞ",
]


def _chunker(limit: int = 505) -> IntakeRegulatoryChunker:
    return IntakeRegulatoryChunker(
        tokenizer=TiktokenTokenizer("text-embedding-3-large"),
        content_token_limit=limit,
    )


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def test_printed_legal_header_controls_identity_and_preserves_publication_source() -> (
    None
):
    publication = "5 Ocak 2025 PAZAR\nResmî Gazete\nSayı : 32773"
    heading = "\n".join(_TITLE_LINES)
    first = "MADDE 1- (1) Bu Tebliğ, 2023/2533 sayılı düzenlemeye atıf yapar."
    second = "(2) Belgenin bütün koşulları ve istisnaları korunur."
    source = (
        f"{_PRINT_HEADER}\n\n{publication}\n\nTEBLİĞ\n\n"
        f"Sanayi ve Teknoloji Bakanlığından:\n{heading}\n"
        f"(2023/826/AB) (SGM-2024/5)\n\nAmaç ve kapsam\n{first}\n"
        f"{_PRINT_FOOTER}\n\n{_PRINT_HEADER}\n\n{second}"
    )
    original_blocks = _markdown_to_blocks(source)
    result = _chunker().chunk_text(
        source, source_path="frozen-main", source_file="2025-01-05.pdf"
    )
    assert result.content == source
    assert [asdict(block) for block in result.blocks] == [
        asdict(block) for block in original_blocks
    ]
    assert result.metadata.title == " ".join(_TITLE_LINES)
    assert result.structure[0].label == result.metadata.title
    assert result.metadata.document_number == "SGM-2024/5"
    assert result.metadata.document_date is None
    assert result.metadata.file_date is None
    atomic = [c for c in result.chunks if c.metadata.chunk_variant == "atomic"]
    canonical = "\n".join(c.text for c in atomic)
    assert publication in canonical
    assert heading in canonical
    assert "(2023/826/AB) (SGM-2024/5)" in canonical
    assert first in canonical and second in canonical
    assert _PRINT_HEADER not in canonical and _PRINT_FOOTER not in canonical
    for chunk in result.chunks:
        assert chunk.metadata.document_number == "SGM-2024/5"
        assert chunk.metadata.heading_path[0] == result.metadata.title
        span = source[chunk.metadata.source_start_char : chunk.metadata.source_end_char]
        assert all(_compact(line) in _compact(span) for line in chunk.text.splitlines())
    # Existing Update chunking retains its previous inference and source behavior.
    legacy = RegulatoryChunker().chunk_text(source, source_file="2025-01-05.pdf")
    assert legacy.metadata.document_number == "2025/01"
    assert legacy.metadata.title == _PRINT_HEADER


@pytest.mark.parametrize(
    "masthead",
    [
        "Sayfa : 44 RESMÎ GAZETE 5 Ocak 2025 – Sayı : 32773",
        "5 Ocak 2025 – Sayı : 32773 RESMÎ GAZETE Sayfa : 45",
    ],
)
def test_annex_mastheads_never_name_document_or_promote_cited_identity(
    masthead: str,
) -> None:
    definitions = [
        "1. Ağ cihazı: Özgün teknik koşullara tabi cihazı ifade eder.",
        "2. Ev tipi cihaz: 2023/2533/AB ve (SGM-2024/3) kapsamındaki cihazı ifade eder.",
    ]
    source = (
        f"{masthead}\n\nEk I\nTanımlar\n"
        "Aşağıdaki tanımlar bu ek bakımından geçerlidir.\n"
        f"{definitions[0]}\n\n{masthead}\n\n{definitions[1]}"
    )
    result = _chunker().chunk_text(source, source_file="ekler.pdf")
    assert result.content == source
    assert result.metadata.title is None
    assert result.metadata.document_number is None
    assert result.metadata.document_date is None
    assert result.structure[0].label == "ekler.pdf"
    canonical = "\n".join(
        chunk.text
        for chunk in result.chunks
        if chunk.metadata.chunk_variant == "atomic"
    )
    assert masthead not in canonical
    assert all(definition in canonical for definition in definitions)
    assert all(chunk.metadata.document_number is None for chunk in result.chunks)
    assert len([c for c in result.chunks if c.metadata.chunk_type == "paragraph"]) == 2


@pytest.mark.parametrize(
    ("header", "number"),
    [
        ("# 5000 SAYILI GÜMRÜK KANUNU", "5000"),
        ("# Gümrük Kanunu\nKanun No: 5000", "5000"),
        ("# Başvuru Tebliği\nTebliğ No: SGM-2024/5", "SGM-2024/5"),
        ("# Başvuru Tebliği\n\nMADDE 1- 2023/2533 sayılı düzenleme uygulanır.", None),
        ("# Başvuru Tebliği\nAtıf yapılan belge (SGM-2024/3) uygulanır.", None),
        ("# Başvuru Tebliği\n(SGM-2024/5) (SGM-2024/3)", None),
        ("# Ek açıklaması\n\n1. Atıf: 2023/2533 sayılı düzenleme uygulanır.", None),
    ],
)
def test_only_own_header_identity_is_accepted(header: str, number: str | None) -> None:
    result = _chunker().chunk_text(header + "\n\nBelgenin özgün metni korunur.")
    assert result.metadata.document_number == number


def test_prose_and_table_resembling_print_chrome_are_preserved() -> None:
    sentence = f"Başvuru {_PRINT_HEADER} kaydını içeriyorsa doğrulanır."
    table = f"| Kayıt | {_PRINT_HEADER} |\n| --- | --- |\n| Sayfa | 44 |"
    source = f"# Başvuru düzenlemesi\n\n{sentence}\n\n{table}"
    result = _chunker().chunk_text(source)
    canonical = "\n".join(chunk.text for chunk in result.chunks)
    assert sentence in canonical and table in canonical


def test_short_parent_introduction_joins_first_child_without_losing_later_scope() -> (
    None
):
    introduction = "MADDE 2- (1) Bu Tebliğin amaçları doğrultusunda;"
    first = "a) AB: Avrupa Birliğini,"
    second = "b) Bakanlık: Sanayi ve Teknoloji Bakanlığını,"
    source = f"# Başvuru Tebliği\n\n{introduction}\n{first}\n{second}"
    result = _chunker().chunk_text(source)
    atomic = [c for c in result.chunks if c.metadata.chunk_variant == "atomic"]
    assert len(atomic) == 2
    assert atomic[0].text == f"{introduction}\n{first}"
    assert atomic[1].text == second
    assert _compact("".join(c.text for c in atomic)) == _compact(
        introduction + first + second
    )
    assert all(
        "(1) Bu Tebliğin amaçları doğrultusunda;" in c.metadata.heading_path
        for c in atomic
    )


def test_parent_introduction_stays_separate_when_combination_exceeds_budget() -> None:
    introduction = "MADDE 2- (1) Bu Tebliğin amaçları doğrultusunda;"
    first = "a) Başvuru: " + "özgün koşulu " * 16
    source = f"# Başvuru Tebliği\n\n{introduction}\n{first}"
    tokenizer = TiktokenTokenizer("text-embedding-3-large")
    limit = len(tokenizer.encode(first))
    result = _chunker(limit).chunk_text(source)
    atomic = [c for c in result.chunks if c.metadata.chunk_variant == "atomic"]
    assert atomic[0].text == introduction
    assert atomic[1].text == first.rstrip()
    assert all(len(tokenizer.encode(c.text)) <= limit for c in result.chunks)
