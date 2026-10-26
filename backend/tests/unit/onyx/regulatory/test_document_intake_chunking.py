"""Intake splitting must retain source wording and avoid artificial heading chunks."""

import re

import pytest

from onyx.natural_language_processing.utils import TiktokenTokenizer
from onyx.regulatory.chunker import RegulatoryChunk, RegulatoryChunker
from onyx.regulatory.document_intake.chunking import IntakeRegulatoryChunker


@pytest.fixture
def tokenizer() -> TiktokenTokenizer:
    return TiktokenTokenizer("text-embedding-3-large")


def _atomic(chunks: list[RegulatoryChunk]) -> list[RegulatoryChunk]:
    return [chunk for chunk in chunks if chunk.metadata.chunk_variant == "atomic"]


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _assert_source_coverage(
    source: str,
    expected: str,
    chunks: list[RegulatoryChunk],
    tokenizer: TiktokenTokenizer,
    limit: int,
) -> None:
    atomic = _atomic(chunks)
    assert _compact("".join(chunk.text for chunk in atomic)) == _compact(expected)
    for chunk in chunks:
        assert len(tokenizer.encode(chunk.text)) <= limit
    previous_end = 0
    for chunk in atomic:
        start, end = chunk.metadata.source_start_char, chunk.metadata.source_end_char
        assert previous_end <= start < end <= len(source)
        assert _compact(source[start:end]) == _compact(chunk.text)
        previous_end = end


@pytest.mark.parametrize("limit", [128, 256, 512])
def test_long_article_keeps_heading_with_first_body_and_preserves_exact_source(
    tokenizer: TiktokenTokenizer, limit: int
) -> None:
    first_heading = "## MADDE 1 — Başvuru ve kayıtların hazırlanması"
    second_heading = "## MADDE 2 — Belge düzeni"
    first_body = " ".join(
        f"Başvuru sahibi {index} numaralı beyanın menşe ülkesini ve eşyanın miktarını kayıt altına alır."
        for index in range(18)
    )
    second_body = "Belge görevlisi özgün faturanın tarihini korur."
    expected = f"{first_heading}\n\n{first_body}\n\n{second_heading}\n\n{second_body}"
    source = f"# Yerel test düzenlemesi\n\n{expected}"
    chunker = IntakeRegulatoryChunker(
        tokenizer=tokenizer, content_token_limit=limit, max_chunk_chars=500
    )
    result = chunker.chunk_text(source, source_path="source", source_file="test.md")
    atomic = _atomic(result.chunks)
    first = next(chunk for chunk in atomic if chunk.metadata.article_no == "1")
    assert first.text.startswith(first_heading + "\n")
    assert "Başvuru sahibi" in first.text
    assert all(chunk.text not in {first_heading, second_heading} for chunk in atomic)
    assert sum(first_heading in chunk.text for chunk in atomic) == 1
    assert all("Part " not in " ".join(chunk.metadata.heading_path) for chunk in atomic)
    _assert_source_coverage(source, expected, result.chunks, tokenizer, limit)
    assert result == chunker.chunk_text(
        source, source_path="source", source_file="test.md"
    )


@pytest.mark.parametrize("difference", [-1, 0, 1])
def test_exact_embedding_boundary_is_measured_with_heading_and_separators(
    tokenizer: TiktokenTokenizer, difference: int
) -> None:
    expected = "## MADDE 1 — Usul\nBaşvuru sahibi özgün belgeyi ibraz eder."
    source = f"# Yerel test\n\n{expected}"
    tokens = len(tokenizer.encode(expected))
    limit = tokens + difference
    result = IntakeRegulatoryChunker(
        tokenizer=tokenizer, content_token_limit=limit
    ).chunk_text(source)
    _assert_source_coverage(source, expected, result.chunks, tokenizer, limit)
    if difference >= 0:
        assert len(_atomic(result.chunks)) == 1
    else:
        assert len(_atomic(result.chunks)) > 1
        assert "Başvuru" in _atomic(result.chunks)[0].text


def test_giant_single_word_unicode_is_sliced_without_token_decode_or_word_loss(
    tokenizer: TiktokenTokenizer,
) -> None:
    heading = "## MADDE 1 — Kaynak"
    body = "İğüşöç🙂漢字" * 180
    source = f"# Yerel test\n\n{heading}\n\n{body}"
    limit = 96
    result = IntakeRegulatoryChunker(
        tokenizer=tokenizer, content_token_limit=limit
    ).chunk_text(source)
    _assert_source_coverage(
        source, f"{heading}\n\n{body}", result.chunks, tokenizer, limit
    )
    assert len(_atomic(result.chunks)) > 2
    assert "İğüşöç" in _atomic(result.chunks)[0].text


def test_tables_preserve_whole_rows_without_repeating_or_fabricating_headers(
    tokenizer: TiktokenTokenizer,
) -> None:
    table = "\n".join(
        ["| Kod | Açıklama |", "| --- | --- |"]
        + [f"| {index} | Özgün belge {index} ibraz edilir. |" for index in range(16)]
    )
    source = f"# EK-1 Yerel test\n\n{table}"
    limit = 72
    result = IntakeRegulatoryChunker(
        tokenizer=tokenizer, content_token_limit=limit
    ).chunk_text(source)
    _assert_source_coverage(source, table, result.chunks, tokenizer, limit)
    atomic = _atomic(result.chunks)
    assert len(atomic) > 1
    assert all(chunk.metadata.chunk_type == "table" for chunk in atomic)
    assert all(
        line.startswith("|") and line.endswith("|")
        for chunk in atomic
        for line in chunk.text.splitlines()
    )
    assert sum("| Kod | Açıklama |" in chunk.text for chunk in atomic) == 1


def test_oversized_single_table_row_fails_explicitly_instead_of_truncating(
    tokenizer: TiktokenTokenizer,
) -> None:
    source = (
        "# EK-1\n\n| Kod | Açıklama |\n| --- | --- |\n| 1 | " + "özgün " * 200 + "|"
    )
    with pytest.raises(ValueError, match="table row exceeds"):
        IntakeRegulatoryChunker(tokenizer=tokenizer, content_token_limit=64).chunk_text(
            source
        )


def test_standalone_heading_is_preserved_without_invented_operative_text(
    tokenizer: TiktokenTokenizer,
) -> None:
    source = "## MADDE 1 — Başvuru"
    result = IntakeRegulatoryChunker(
        tokenizer=tokenizer, content_token_limit=64
    ).chunk_text(source)
    assert len(result.chunks) == 1 and result.chunks[0].text == source


def test_oversized_heading_fails_without_artificial_body(
    tokenizer: TiktokenTokenizer,
) -> None:
    source = (
        "# Yerel test\n\n## MADDE 1 — " + "başlık " * 100 + "\n\nBelge ibraz edilir."
    )
    with pytest.raises(ValueError, match="heading exceeds"):
        IntakeRegulatoryChunker(tokenizer=tokenizer, content_token_limit=64).chunk_text(
            source
        )


def test_source_offsets_remain_exact_with_indented_unicode_paragraphs(
    tokenizer: TiktokenTokenizer,
) -> None:
    heading = "## MADDE 1 — Usul"
    body = "İlgili belge özgün biçimde korunur. " * 35
    source = f"# Yerel test\n\n{heading}\n\n   {body}  \n"
    result = IntakeRegulatoryChunker(
        tokenizer=tokenizer, content_token_limit=80
    ).chunk_text(source)
    _assert_source_coverage(
        source.strip(), f"{heading}\n{body}", result.chunks, tokenizer, 80
    )


def test_legacy_chunker_keeps_existing_split_behavior() -> None:
    heading = "## MADDE 1 — Başvuru"
    source = f"# Yerel test\n\n{heading}\n\n" + "Belge kontrol edilir. " * 60
    result = RegulatoryChunker(max_chunk_chars=500).chunk_text(source)
    assert _atomic(result.chunks)[0].text == heading


@pytest.mark.parametrize("heading", ["MADDE 1 — Başvuru", "MADDE 1 —\nBaşvuru"])
def test_plain_converted_article_locator_and_title_stay_with_body(
    tokenizer: TiktokenTokenizer, heading: str
) -> None:
    body = "Başvuru sahibi özgün faturanın tarihini kontrol eder. " * 28
    source = f"# Yerel test\n\n{heading}\n\n{body}"
    result = IntakeRegulatoryChunker(
        tokenizer=tokenizer, content_token_limit=96
    ).chunk_text(source)
    atomic = _atomic(result.chunks)
    assert "özgün faturanın" in atomic[0].text
    _assert_source_coverage(source, f"{heading}\n{body}", result.chunks, tokenizer, 96)


@pytest.mark.parametrize("heading", ["## MADDE 1 — Genel", "MADDE 1 —\nGenel"])
def test_structural_paragraphs_keep_article_heading_and_dense_aggregate_positions(
    tokenizer: TiktokenTokenizer, heading: str
) -> None:
    expected = (
        f"{heading}\n\n(1) "
        + "Belge ibraz edilir. " * 30
        + "\n\n(2) "
        + "Başvuru kaydedilir. " * 30
        + "\n\n## MADDE 2 — Özel\n\n(1) Fatura sunulur.\n\n(2) Tarih kaydedilir."
    )
    source = f"# Yerel test\n\n{expected}"
    result = IntakeRegulatoryChunker(
        tokenizer=tokenizer, content_token_limit=128
    ).chunk_text(source)
    _assert_source_coverage(source, expected, result.chunks, tokenizer, 128)
    atomic = _atomic(result.chunks)
    assert "Belge ibraz edilir." in atomic[0].text
    assert atomic[0].metadata.chunk_type == "paragraph"
    assert atomic[0].metadata.paragraph_no == "1"
    assert [chunk.metadata.chunk_order for chunk in result.chunks] == list(
        range(len(result.chunks))
    )
    aggregates = [
        chunk
        for chunk in result.chunks
        if chunk.metadata.chunk_variant == "hierarchical_aggregate"
    ]
    assert aggregates and all(
        chunk.metadata.source_chunk_orders for chunk in aggregates
    )
    assert all(
        all(order < len(atomic) for order in chunk.metadata.source_chunk_orders)
        for chunk in aggregates
    )


def test_explicit_markdown_titles_override_document_title_for_all_article_continuations(
    tokenizer: TiktokenTokenizer,
) -> None:
    parts = ["# Külliyat kaynak belgesi"]
    for number in range(1, 4):
        parts.append(f"## MADDE {number} — Başvuru koşulları {number}")
        parts.append(f"İlgili kişi {number} numaralı özgün belgeyi sunar. " * 22)
    source = "\n\n".join(parts)
    result = IntakeRegulatoryChunker(
        tokenizer=tokenizer, content_token_limit=96
    ).chunk_text(source)
    atomic = _atomic(result.chunks)
    for number in range(1, 4):
        article = [
            chunk for chunk in atomic if chunk.metadata.article_no == str(number)
        ]
        assert len(article) > 1
        assert all(
            chunk.metadata.article_title == f"Başvuru koşulları {number}"
            for chunk in article
        )
        assert all(
            chunk.metadata.heading_path[-1]
            == f"MADDE {number} - Başvuru koşulları {number}"
            for chunk in article
        )
    nodes = {node.node_id: node for node in result.structure}
    for chunk in result.chunks:
        assert chunk.metadata.parent_path == [
            nodes[identifier].label for identifier in chunk.metadata.parent_ids
        ]
        assert chunk.metadata.heading_path == chunk.metadata.parent_path
    _assert_source_coverage(
        source, "\n\n".join(parts[1:]), result.chunks, tokenizer, 96
    )


@pytest.mark.parametrize(
    "standalone_title", [None, "Başvuru koşulları", "### Başvuru koşulları"]
)
def test_bare_article_does_not_inherit_document_title_and_keeps_actual_separate_title(
    tokenizer: TiktokenTokenizer, standalone_title: str | None
) -> None:
    body = "Başvuru sahibi özgün belgeyi sunar. " * 20
    source = "# Külliyat kaynak belgesi\n\n## MADDE 1\n\n"
    if standalone_title:
        source += standalone_title + "\n\n"
    source += body
    result = IntakeRegulatoryChunker(
        tokenizer=tokenizer, content_token_limit=96
    ).chunk_text(source)
    expected_title = "Başvuru koşulları" if standalone_title else None
    expected_label = "MADDE 1 - Başvuru koşulları" if standalone_title else "MADDE 1"
    for chunk in result.chunks:
        assert chunk.metadata.article_title == expected_title
        assert expected_label in chunk.metadata.heading_path
        assert "MADDE 1 - Külliyat kaynak belgesi" not in chunk.metadata.heading_path


def test_corrected_article_nodes_and_hierarchical_aggregate_use_same_title_and_ids(
    tokenizer: TiktokenTokenizer,
) -> None:
    source = (
        "# Külliyat kaynak belgesi\n\n## Konu başlığı\n\n"
        "## MADDE 1 — Açık madde başlığı\n\n(1) Fatura sunulur.\n\n(2) Tarih kaydedilir."
    )
    result = IntakeRegulatoryChunker(
        tokenizer=tokenizer, content_token_limit=128
    ).chunk_text(source)
    nodes = {node.node_id: node for node in result.structure}
    article_node = next(
        node for node in result.structure if node.node_type == "article"
    )
    assert article_node.label == "MADDE 1 - Açık madde başlığı"
    aggregates = [
        chunk
        for chunk in result.chunks
        if chunk.metadata.chunk_variant == "hierarchical_aggregate"
    ]
    assert aggregates
    for chunk in result.chunks:
        assert chunk.metadata.article_title == "Açık madde başlığı"
        assert article_node.node_id in chunk.metadata.parent_ids
        assert chunk.metadata.parent_path == [
            nodes[identifier].label for identifier in chunk.metadata.parent_ids
        ]
    for chunk in aggregates:
        assert chunk.metadata.hierarchy_root_id in nodes
        assert (
            chunk.metadata.hierarchy_root_path[-1]
            == nodes[chunk.metadata.hierarchy_root_id].label
        )
    assert all(
        node.parent_id is None or node.parent_id in nodes for node in result.structure
    )


def test_plain_inline_and_no_title_article_semantics_remain_unchanged(
    tokenizer: TiktokenTokenizer,
) -> None:
    for source in (
        "MADDE 1 — Başvuru sahibi belgeyi sunar.",
        "MADDE 1\nBaşvuru sahibi belgeyi sunar.",
        "# Yalnız belge başlığı",
    ):
        expected = RegulatoryChunker().chunk_text(source)
        actual = IntakeRegulatoryChunker(
            tokenizer=tokenizer, content_token_limit=256
        ).chunk_text(source)
        assert actual == expected


@pytest.mark.parametrize(
    "following",
    [
        "## MADDE 2 — Sonraki madde",
        "## İKİNCİ BÖLÜM",
        "### 1.1.1 Belge düzeni",
        "(1) Belge sunulur.",
    ],
)
def test_bare_article_does_not_steal_following_structural_unit_as_title(
    tokenizer: TiktokenTokenizer, following: str
) -> None:
    source = f"# Kaynak belge\n\n## MADDE 1\n\n{following}\n\nİlgili belge korunur."
    result = IntakeRegulatoryChunker(
        tokenizer=tokenizer, content_token_limit=128
    ).chunk_text(source)
    first_article = next(
        node for node in result.structure if node.node_type == "article"
    )
    assert first_article.label == "MADDE 1"
    assert all(
        chunk.metadata.article_title is None
        for chunk in result.chunks
        if chunk.metadata.article_no == "1"
    )
