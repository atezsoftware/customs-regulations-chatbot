"""Keep OCR list numbering separate from legal provision locators."""

from pathlib import Path

import pytest

from onyx.regulatory.chunker import RegulatoryChunker
from onyx.regulatory.document_intake.metadata import MetadataChunk
from onyx.regulatory.document_intake.metadata_structure import (
    verified_structural_metadata,
)


@pytest.mark.parametrize(("prefix", "label"), [("4.", "ç"), ("9.", "ğ"), ("11.", "ı")])
def test_actual_docling_numbered_clauses_keep_their_real_paragraph(
    prefix: str, label: str
) -> None:
    text = (
        Path(__file__).parent / "fixtures/docling_railway_ocr_clauses.md"
    ).read_text()
    document = RegulatoryChunker().chunk_text(text)
    original_line = next(line for line in text.splitlines() if line.startswith(prefix))
    clause = next(chunk for chunk in document.chunks if original_line in chunk.text)

    assert clause.metadata.article_no == "4"
    assert clause.metadata.paragraph_no == "1"
    assert clause.metadata.clause_label == label
    evidence = MetadataChunk(
        chunk_id=clause.metadata.chunk_id,
        text=clause.text,
        heading_path=clause.metadata.heading_path,
        chunk_type=clause.metadata.chunk_type,
        structural_metadata={
            "article_no": clause.metadata.article_no,
            "paragraph_no": clause.metadata.paragraph_no,
            "clause_label": clause.metadata.clause_label,
        },
    )
    facts = verified_structural_metadata([evidence])[evidence.chunk_id]
    assert facts["paragraph_no"].value == "1"
    assert facts["clause_label"].value == label
    assert document.content == text.strip()
    assert all(
        not line.strip() or any(line in chunk.text for chunk in document.chunks)
        for line in text.strip().splitlines()
    )
    assert clause.text in document.content
    assert (
        original_line
        == document.content[
            clause.metadata.source_start_char : clause.metadata.source_end_char
        ].splitlines()[0]
    )


@pytest.mark.parametrize("paragraph", ["4", "9", "11"])
def test_old_ocr_numbered_clause_cannot_verify_a_false_paragraph(
    paragraph: str,
) -> None:
    text = f"{paragraph}. ç) Bakan: Ulaştırma ve Altyapı Bakanını,"
    chunk = MetadataChunk(
        chunk_id="old-ocr-chunk",
        text=text,
        heading_path=["MADDE 4", text],
        chunk_type="paragraph",
        structural_metadata={"article_no": "4", "paragraph_no": paragraph},
    )

    assert "paragraph_no" not in verified_structural_metadata([chunk])[chunk.chunk_id]


@pytest.mark.parametrize("marker", ["(4)", "4.", "4)"])
def test_genuine_numbered_paragraph_still_has_its_own_locator(marker: str) -> None:
    document = RegulatoryChunker().chunk_text(
        f"MADDE 1- (1) Başvuru esasları.\n\n{marker} Başvuru elektronik ortamda yapılır."
    )
    paragraph = next(c for c in document.chunks if c.text.startswith(marker))

    assert paragraph.metadata.paragraph_no == "4"
    assert paragraph.metadata.clause_label is None


def test_numbered_clause_without_parent_does_not_invent_a_paragraph() -> None:
    document = RegulatoryChunker().chunk_text(
        "4. ç) Bakan: Ulaştırma ve Altyapı Bakanını,"
    )
    assert document.chunks[0].metadata.paragraph_no is None
    assert document.chunks[0].metadata.clause_label == "ç"
