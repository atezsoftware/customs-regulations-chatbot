"""Rendering regulatory documents and chunks as PDF.

The documents are Turkish, so the interesting property is not "bytes were
produced" but that the text survives into the PDF intact — ReportLab's built-in
fonts are Latin-1 and silently mangle characters like ı, ş and ğ.
"""

import datetime
from io import BytesIO

from pypdf import PdfReader

from onyx.regulatory.pdf import render_chunk_pdf, render_document_pdf

TURKISH_SAMPLE = "Gümrük müşavirliği şirketlerinin yükümlülüğü ıslak imza gerektirir."


def _extract_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def test_document_pdf_is_a_readable_pdf() -> None:
    pdf = render_document_pdf(
        name="mevzuat/tir_sozlesmesi.md",
        markdown=f"# TIR Sozlesmesi\n\n{TURKISH_SAMPLE}\n",
    )

    assert pdf.startswith(b"%PDF-")
    text = _extract_text(pdf)
    assert "TIR Sozlesmesi" in text


def test_document_pdf_preserves_turkish_characters() -> None:
    pdf = render_document_pdf(name="t.md", markdown=TURKISH_SAMPLE)

    text = _extract_text(pdf)

    for character in ("ü", "ş", "ı", "ğ"):
        assert character in text, f"{character!r} missing from {text!r}"


def test_document_pdf_keeps_the_document_name_as_a_heading() -> None:
    pdf = render_document_pdf(name="mevzuat/1975_tir_sozlesmesi.md", markdown="Metin")

    assert "1975_tir_sozlesmesi" in _extract_text(pdf)


def test_chunk_pdf_carries_the_context_needed_to_read_it_alone() -> None:
    pdf = render_chunk_pdf(
        text=TURKISH_SAMPLE,
        heading_path=["BİRİNCİ BÖLÜM", "Madde 5"],
        validity_start_date=datetime.date(2024, 1, 1),
        validity_end_date=None,
        position=4,
    )

    text = _extract_text(pdf)

    assert "Madde 5" in text
    assert "BİRİNCİ BÖLÜM" in text
    assert "2024-01-01" in text
    assert "ş" in text


def test_chunk_pdf_omits_validity_when_the_chunk_has_none() -> None:
    pdf = render_chunk_pdf(
        text="Metin",
        heading_path=[],
        validity_start_date=None,
        validity_end_date=None,
        position=0,
    )

    assert "Validity" not in _extract_text(pdf)


def test_pdf_renders_the_non_latin_passages_these_treaties_contain() -> None:
    """The TIR convention carries Russian and Greek text alongside Turkish.

    A Latin-1 font drops these without erroring, so the check is that they come
    back out of the rendered PDF.
    """

    multilingual = "Türkçe metin. Русский текст. Ελληνικό κείμενο."

    text = _extract_text(render_document_pdf(name="t.md", markdown=multilingual))

    assert "Русский" in text
    assert "Ελληνικό" in text
    assert "Türkçe" in text


def test_markdown_structure_becomes_readable_layout() -> None:
    pdf = render_document_pdf(
        name="t.md",
        markdown=(
            "# Birinci Bölüm\n\n"
            "Giriş paragrafı.\n\n"
            "## Madde 1\n\n"
            "- ilk bent\n"
            "- ikinci bent\n"
        ),
    )

    text = _extract_text(pdf)

    for fragment in ("Birinci Bölüm", "Giriş paragrafı", "Madde 1", "ikinci bent"):
        assert fragment in text, f"{fragment!r} missing"
    # Heading markers are layout instructions, not content.
    assert "#" not in text


def test_markup_in_the_source_is_not_treated_as_layout_instructions() -> None:
    """Platypus reads a small XML dialect, so raw angle brackets must survive."""

    pdf = render_document_pdf(name="t.md", markdown="Koşul: a < b & c > d")

    assert "a < b & c > d" in _extract_text(pdf)
