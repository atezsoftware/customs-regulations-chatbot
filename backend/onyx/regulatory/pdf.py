"""Renders regulatory documents and individual chunks as PDF.

Operators review these documents on screen and hand them to colleagues, so the
same bytes serve both: the browser displays them inline and saves them
unchanged.

Font choice is not cosmetic. The corpus is Turkish but the treaties inside it
carry Russian and Greek passages, and ReportLab's built-in fonts are Latin-1 --
they would drop those characters silently, producing a PDF that looks fine
until someone reads the missing text. DejaVu Sans covers every character in the
corpus and ships alongside this module rather than coming from the host, so the
output is identical in development, CI and production.
"""

import datetime
import re
from io import BytesIO
from pathlib import Path

from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.platypus.flowables import Flowable

from onyx.utils.logger import setup_logger

logger = setup_logger()

_FONT_NAME = "DejaVuSans"
_FONT_PATH = Path(__file__).parent / "fonts" / "DejaVuSans.ttf"

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_PATTERN = re.compile(r"^\s*([-*+]|\d+[.)])\s+(.*)$")
_fonts_registered = False


def _register_font() -> str:
    """Register the bundled DejaVu face once.

    There is no fallback on purpose: a Latin-1 substitute would silently drop
    Cyrillic and Greek passages, and a PDF that is quietly missing text is worse
    than a failed request.
    """

    global _fonts_registered
    if not _fonts_registered:
        pdfmetrics.registerFont(TTFont(_FONT_NAME, str(_FONT_PATH)))
        _fonts_registered = True
    return _FONT_NAME


def _styles() -> dict[str, ParagraphStyle]:
    # One weight: headings separate from body by size, which is enough hierarchy
    # for reading and avoids shipping a second megabyte of font.
    font = font_bold = _register_font()
    base = getSampleStyleSheet()["BodyText"]
    return {
        "title": ParagraphStyle(
            "RegulatoryTitle",
            parent=base,
            fontName=font_bold,
            fontSize=15,
            leading=19,
            spaceAfter=10,
        ),
        "heading": ParagraphStyle(
            "RegulatoryHeading",
            parent=base,
            fontName=font_bold,
            fontSize=11.5,
            leading=15,
            spaceBefore=9,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "RegulatoryBody",
            parent=base,
            fontName=font,
            fontSize=9.5,
            leading=13.5,
            alignment=TA_LEFT,
            spaceAfter=5,
        ),
        "meta": ParagraphStyle(
            "RegulatoryMeta",
            parent=base,
            fontName=font,
            fontSize=8.5,
            leading=11,
            textColor="#555555",
            spaceAfter=3,
        ),
        "list": ParagraphStyle(
            "RegulatoryList",
            parent=base,
            fontName=font,
            fontSize=9.5,
            leading=13.5,
            leftIndent=10,
            spaceAfter=3,
        ),
    }


def _escape(text: str) -> str:
    """Platypus reads a small XML dialect, so raw markup has to be neutralized."""

    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _markdown_flowables(
    markdown: str, styles: dict[str, ParagraphStyle]
) -> list[Flowable]:
    """Lay markdown out for reading.

    Deliberately not a markdown engine: headings, lists and paragraphs carry
    almost all of the structure in these documents, and anything else reads
    acceptably as plain text.
    """

    flowables: list[Flowable] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            flowables.append(Paragraph(_escape(" ".join(paragraph)), styles["body"]))
            paragraph.clear()

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            flush()
            continue

        heading = _HEADING_PATTERN.match(line)
        if heading is not None:
            flush()
            flowables.append(Paragraph(_escape(heading.group(2)), styles["heading"]))
            continue

        list_item = _LIST_PATTERN.match(line)
        if list_item is not None:
            flush()
            flowables.append(
                Paragraph(
                    f"{_escape(list_item.group(1))} {_escape(list_item.group(2))}",
                    styles["list"],
                )
            )
            continue

        paragraph.append(line.strip())

    flush()
    return flowables


def _build(flowables: list[Flowable], title: str) -> bytes:
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        title=title,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )
    document.build(flowables)
    return buffer.getvalue()


def render_document_pdf(name: str, markdown: str) -> bytes:
    """Render a whole regulatory document, as uploaded, for reading."""

    styles = _styles()
    # The stored name carries the archive path; the document is the last part.
    display_name = name.rsplit("/", 1)[-1]
    flowables: list[Flowable] = [
        Paragraph(_escape(display_name), styles["title"]),
        Spacer(1, 4),
    ]
    flowables.extend(_markdown_flowables(markdown, styles))
    return _build(flowables, display_name)


def render_chunk_pdf(
    *,
    text: str,
    heading_path: list[str],
    validity_start_date: datetime.date | None,
    validity_end_date: datetime.date | None,
    position: int,
) -> bytes:
    """Render one chunk with the context needed to read it on its own."""

    styles = _styles()
    title = " › ".join(heading_path) if heading_path else f"Chunk {position + 1}"
    flowables: list[Flowable] = [Paragraph(_escape(title), styles["title"])]

    if validity_start_date is not None or validity_end_date is not None:
        start = validity_start_date.isoformat() if validity_start_date else "—"
        end = validity_end_date.isoformat() if validity_end_date else "open"
        flowables.append(
            Paragraph(_escape(f"Validity: {start} → {end}"), styles["meta"])
        )

    flowables.append(Spacer(1, 6))
    flowables.extend(_markdown_flowables(text, styles))
    return _build(flowables, title)
