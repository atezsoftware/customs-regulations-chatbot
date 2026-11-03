"""Real retained matrix rows remain source-exact tables instead of legal clauses."""

import json
from pathlib import Path

import pytest

from onyx.natural_language_processing.utils import TiktokenTokenizer
from onyx.regulatory.document_intake.chunking import IntakeRegulatoryChunker
from onyx.regulatory.document_intake.logical_tables import (
    scan_logical_tables,
)

TOKENIZER = TiktokenTokenizer("text-embedding-3-large")


def chunk(source: str, *, version: str = "5", budget: int = 1848):
    return IntakeRegulatoryChunker(
        tokenizer=TOKENIZER,
        content_token_limit=312,
        table_content_token_limit=budget,
        chunking_version=version,
        max_chunk_chars=1200,
    ).chunk_text(source, source_file="annex.pdf")


def retained_pages() -> list[dict[str, object]]:
    return json.loads(
        (Path(__file__).parent / "fixtures/rg_matrix_retained_pages.json").read_text()
    )["pages"]


def test_real_multiline_matrix_pages_keep_all_rows_cells_and_source_characters() -> (
    None
):
    for page in retained_pages():
        source = str(page["text"])
        parsed = chunk(source)
        atomic = [c for c in parsed.chunks if c.metadata.chunk_variant == "atomic"]
        assert not [c for c in atomic if c.metadata.chunk_type == "clause"]
        tables = scan_logical_tables(source)
        assert len(tables) == 1
        table = tables[0]
        assert len({r.row_index for r in table.rows}) == len(table.rows)
        for row in table.rows:
            assert any(
                c.metadata.source_start_char <= row.start_char
                and c.metadata.source_end_char >= row.end_char
                for c in atomic
            )
            assert all(
                source[cell.start_char : cell.end_char]
                == source[row.start_char : row.end_char][
                    cell.start_char - row.start_char : cell.end_char - row.start_char
                ]
                for cell in row.cells
            )
        table_chunks = [c for c in atomic if c.metadata.chunk_type == "table"]
        assert len(table_chunks) == 1
        assert len(TOKENIZER.encode(table_chunks[0].text)) <= 1848
        assert (
            source[
                table_chunks[0].metadata.source_start_char : table_chunks[
                    0
                ].metadata.source_end_char
            ]
            == table_chunks[0].text
        )
        assert "intake_logical_table" in table_chunks[0].metadata.to_storage_dict()
    first = chunk(str(retained_pages()[0]["text"]))
    assert "Tablo 2:" in first.chunks[0].metadata.heading_path[-1]


def test_v4_is_frozen_and_ordinary_clauses_outside_table_are_unchanged() -> None:
    source = "# Kaynak\n\nMADDE 1- (1) Hüküm.\n\na) Olağan bent.\n\n| Ülke | Tarih |\n|---|---|\n| TR | (C) 1.1.2024\n(R) 1.1.2025 |\n\nb) Sonraki bent."
    current, legacy = chunk(source), chunk(source, version="4")
    assert any(
        c.text.startswith("(R)") and c.metadata.chunk_type == "clause"
        for c in legacy.chunks
    )
    assert not any(
        c.text.startswith("(R)") and c.metadata.chunk_type == "clause"
        for c in current.chunks
    )
    for text in ("a) Olağan bent.", "b) Sonraki bent."):
        assert [
            (c.text, c.metadata.chunk_type)
            for c in current.chunks
            if text in c.text and c.metadata.chunk_variant == "atomic"
        ] == [
            (c.text, c.metadata.chunk_type)
            for c in legacy.chunks
            if text in c.text and c.metadata.chunk_variant == "atomic"
        ]
    assert all(
        "intake_logical_table" not in c.metadata.to_storage_dict()
        for c in legacy.chunks
    )


def test_caption_starts_independent_scope_and_preserves_exact_source_span() -> None:
    source = "# EK V\n\n(3) Önceki tablonun yalnızca eski kurallara ilişkin dipnotu.\n\nTablo 2: Taraf ülkeler arası geçerlik tarihleri\n\n| Ülke | Tarih |\n|---|---|\n| TR | (C) 1.1.2024\n(R) 1.1.2025 |\n\n| Ülke | Tarih |\n|---|---|\n| AB | (R) 1.1.2025 |"
    parsed = chunk(source)
    tables = [c for c in parsed.chunks if c.metadata.chunk_type == "table"]
    assert len(tables) == 2
    assert all(
        c.metadata.heading_path[-1] == "Tablo 2: Taraf ülkeler arası geçerlik tarihleri"
        for c in tables
    )
    assert all(
        not any("Önceki tablonun" in part for part in c.metadata.heading_path)
        for c in tables
    )
    assert all(
        c.metadata.paragraph_no is None and c.metadata.clause_label is None
        for c in tables
    )
    assert (
        source[
            tables[0].metadata.source_start_char : tables[0].metadata.source_end_char
        ]
        == tables[0].text
    )
    assert any("Önceki tablonun" in c.text for c in parsed.chunks)


def test_large_table_splits_only_at_logical_rows_and_exposes_literal_header_context() -> (
    None
):
    rows = [f"| TR{i} | (C) 1.1.2024\n(R) 1.1.2025 |" for i in range(16)]
    source = "Tablo 2: Menşe tarihleri\n\n| Ülke | Tarih |\n|---|---|\n" + "\n".join(
        rows
    )
    parsed = chunk(source, budget=90)
    tables = [c for c in parsed.chunks if c.metadata.chunk_type == "table"]
    assert len(tables) > 1
    for row in rows:
        assert sum(row in c.text for c in tables) == 1
    for c in tables[1:]:
        assert "| Ülke | Tarih |" in c.metadata.heading_path
        assert len(TOKENIZER.encode(c.text)) <= 90


@pytest.mark.parametrize("header", ["| | Date |", "| Group |"])
def test_oversized_multiline_grid_with_ambiguous_axes_fails_closed(header: str) -> None:
    source = (
        header
        + "\n|---|---|\n"
        + "\n".join(f"| TR{i} | (C) 1.1.2024\n(R) 1.1.2025 |" for i in range(12))
    )
    with pytest.raises(ValueError, match="complete source axes"):
        chunk(source, budget=90)
