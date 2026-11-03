"""Logical scanner boundaries preserve literal source cells and surrounding prose."""

import pytest

from onyx.regulatory.chunker import SourceBlock
from onyx.regulatory.document_intake.logical_tables import (
    logical_markdown_blocks,
    scan_logical_tables,
)


@pytest.mark.parametrize(
    "source",
    [
        "| A | B |\n|---|---|\n| TR | (C) 1.1.2024\n(R) 1.1.2025",
        "| A | B |\n|---|---|\n| TR | unterminated\n\nOrdinary body",
        "| A | B |\n|---|---|\n| TR | missing\n| Next | row |",
    ],
)
def test_truncated_or_ambiguous_multiline_tables_fail_closed(source: str) -> None:
    with pytest.raises(ValueError, match="incomplete|ambiguous"):
        logical_markdown_blocks(source)


def test_escaped_pipes_and_ragged_headers_keep_literal_cell_coordinates() -> None:
    source = "| Groupe |\n|---|---|\n| A | B | C |\n| TR | escaped \\| literal\ncontinued | x |"
    tables = scan_logical_tables(source)
    assert [len(r.cells) for r in tables[0].rows] == [1, 2, 3, 3]
    assert (
        "escaped \\| literal\ncontinued"
        in source[
            tables[0].rows[-1].cells[1].start_char : tables[0]
            .rows[-1]
            .cells[1]
            .end_char
        ]
    )


def test_replacing_table_overlap_retains_external_source_material(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.regulatory.document_intake import logical_tables

    source = "Prose before\n| A | B |\n|---|---|\n| x | first\nsecond |\nProse after"
    real = logical_tables._markdown_to_blocks
    monkeypatch.setattr(
        logical_tables,
        "_markdown_to_blocks",
        lambda text: [SourceBlock(0, "paragraph", text, 0, len(text))],
    )
    # The synthetic legacy block spans both boundaries; the scanner cannot eat prose.
    fixed = source.replace("|\nProse after", "|\n\nProse after")
    blocks, _ = logical_tables.logical_markdown_blocks(fixed)
    assert blocks[0].text == "Prose before" and blocks[-1].text == "Prose after"
    assert "".join(b.text for b in blocks).replace("\n", "").replace(
        " ", ""
    ) == fixed.replace("\n", "").replace(" ", "")
    monkeypatch.setattr(logical_tables, "_markdown_to_blocks", real)


def test_literal_fenced_code_does_not_create_logical_table_proof() -> None:
    source = "```markdown\n| A | B |\n|---|---|\n| x | first\nsecond |\n```"
    assert scan_logical_tables(source) == []
