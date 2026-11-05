"""Bounded, local text extraction for private chat attachments."""

import csv
import json
from collections.abc import Iterator
from io import BytesIO, StringIO
from pathlib import PurePosixPath
from typing import cast
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile
from zlib import error as ZlibError

from defusedxml.ElementTree import fromstring
from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException
from openpyxl.worksheet._read_only import ReadOnlyWorksheet

CHAT_ATTACHMENT_EXTENSIONS = frozenset(
    {".pdf", ".docx", ".xlsx", ".csv", ".txt", ".md"}
)
MAX_CHAT_ATTACHMENT_BYTES = 20 * 1024 * 1024
MAX_CHAT_ATTACHMENT_TEXT_CHARS = 200_000
MAX_CHAT_ATTACHMENT_ZIP_BYTES = 64 * 1024 * 1024
MAX_CHAT_ATTACHMENT_XML_BYTES = 8 * 1024 * 1024
MAX_CHAT_ATTACHMENT_ZIP_MEMBERS = 2_048
MAX_CHAT_ATTACHMENT_PAGES = 500
MAX_CHAT_ATTACHMENT_ROWS = 20_000
MAX_CHAT_ATTACHMENT_CELLS = 200_000
MAX_CHAT_ATTACHMENT_SHEETS = 100

_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class _TextBuilder:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.length = 0

    def append(self, text: str) -> None:
        self.length += len(text) + 1
        if self.length > MAX_CHAT_ATTACHMENT_TEXT_CHARS:
            raise ValueError(
                f"Extracted text exceeds {MAX_CHAT_ATTACHMENT_TEXT_CHARS:,} characters. "
                "Split the document into smaller files; no content was truncated."
            )
        self.parts.append(text)

    def render(self) -> str:
        return "\n".join(self.parts).strip()


def _decode_text(content: bytes) -> str:
    encoding = (
        "utf-16" if content.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    )
    try:
        text = content.decode(encoding)
    except UnicodeDecodeError as exc:
        raise ValueError(
            "Text and CSV files must use UTF-8 or BOM-marked UTF-16 encoding."
        ) from exc
    if "\x00" in text:
        raise ValueError(
            "The attachment contains binary data instead of readable text."
        )
    return text


def _validate_office_archive(content: bytes) -> None:
    with ZipFile(BytesIO(content)) as archive:
        members = archive.infolist()
        if len(members) > MAX_CHAT_ATTACHMENT_ZIP_MEMBERS:
            raise ValueError("The Office file contains too many archive entries.")
        if len({member.filename for member in members}) != len(members):
            raise ValueError("The Office file contains duplicate archive entries.")
        if any(member.flag_bits & 1 for member in members):
            raise ValueError("Password-protected Office files are not supported.")
        if sum(member.file_size for member in members) > MAX_CHAT_ATTACHMENT_ZIP_BYTES:
            raise ValueError("The expanded Office file exceeds the 64 MiB limit.")
        if any(
            member.filename.endswith(".xml")
            and member.file_size > MAX_CHAT_ATTACHMENT_XML_BYTES
            for member in members
        ):
            raise ValueError("An Office XML part exceeds the 8 MiB parsing limit.")


def _pdf_text(content: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(content), strict=True)
    if reader.is_encrypted:
        raise ValueError("Password-protected PDF files are not supported.")
    if len(reader.pages) > MAX_CHAT_ATTACHMENT_PAGES:
        raise ValueError(f"PDF exceeds the {MAX_CHAT_ATTACHMENT_PAGES}-page limit.")
    builder = _TextBuilder()
    builder.append(
        "[PDF text extraction only; images and scanned content require OCR.]"
    )
    has_text = False
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        has_text = has_text or bool(text.strip())
        builder.append(f"[Page {number}]")
        builder.append(
            text or "[No extractable text on this page; scanned content requires OCR.]"
        )
    if not has_text:
        raise ValueError(
            "PDF has no extractable text. Run OCR on scanned pages before uploading."
        )
    return builder.render()


def _paragraph_text(element: ElementTree.Element) -> str:
    return "".join(
        node.text or ""
        if node.tag == f"{_WORD_NS}t"
        else "\t"
        if node.tag == f"{_WORD_NS}tab"
        else "\n"
        for node in element.iter()
        if node.tag
        in {f"{_WORD_NS}t", f"{_WORD_NS}tab", f"{_WORD_NS}br", f"{_WORD_NS}cr"}
    )


def _word_blocks(element: ElementTree.Element, builder: _TextBuilder) -> bool:
    has_text = False
    for child in element:
        if child.tag == f"{_WORD_NS}p":
            text = _paragraph_text(child)
            builder.append(text)
            has_text = has_text or bool(text.strip())
        elif child.tag == f"{_WORD_NS}tbl":
            builder.append("[Table]")
            for row in child.findall(f"{_WORD_NS}tr"):
                cells = [
                    "\n".join(
                        _paragraph_text(paragraph)
                        for paragraph in cell.iter(f"{_WORD_NS}p")
                    )
                    for cell in row.findall(f"{_WORD_NS}tc")
                ]
                builder.append(json.dumps(cells, ensure_ascii=False))
                has_text = has_text or any(cell.strip() for cell in cells)
        else:
            # Includes content controls, footnotes, endnotes, headers and footers.
            has_text = _word_blocks(child, builder) or has_text
    return has_text


def _docx_text(content: bytes) -> str:
    builder = _TextBuilder()
    builder.append("[DOCX text extraction only; embedded images require OCR.]")
    has_text = False
    with ZipFile(BytesIO(content)) as archive:
        if "word/document.xml" not in archive.namelist():
            raise ValueError("The file is not a valid DOCX document.")
        extra_parts = sorted(
            name
            for name in archive.namelist()
            if name.startswith(("word/header", "word/footer"))
            and name.endswith(".xml")
            or name in {"word/footnotes.xml", "word/endnotes.xml"}
        )
        for part in ["word/document.xml", *extra_parts]:
            xml = archive.read(part)
            if b"<!DOCTYPE" in xml.upper() or b"<!ENTITY" in xml.upper():
                raise ValueError(
                    "Office XML declarations with entities are not supported."
                )
            root = fromstring(xml)
            builder.append(f"[{part}]")
            has_text = _word_blocks(root, builder) or has_text
    if not has_text:
        raise ValueError(
            "DOCX has no extractable text. Images require OCR before uploading."
        )
    return builder.render()


def _xlsx_rows(sheet: ReadOnlyWorksheet) -> Iterator[tuple[object, ...]]:
    try:
        yield from sheet.iter_rows(values_only=True)
    except TypeError as error:
        # openpyxl reports invalid typed XML attributes as TypeError.
        raise ValueError("The XLSX contains malformed worksheet XML.") from error


def _xlsx_text(content: bytes) -> str:
    try:
        workbook = load_workbook(
            BytesIO(content), read_only=True, data_only=False, keep_links=False
        )
    except TypeError as error:
        raise ValueError("The XLSX contains malformed workbook XML.") from error
    try:
        if len(workbook.worksheets) > MAX_CHAT_ATTACHMENT_SHEETS:
            raise ValueError(
                f"Workbook exceeds the {MAX_CHAT_ATTACHMENT_SHEETS}-sheet limit."
            )
        builder = _TextBuilder()
        builder.append(
            "[Workbook formulas are preserved as formulas; they have not been calculated.]"
        )
        total_rows = 0
        total_cells = 0
        has_content = False
        for sheet in workbook.worksheets:
            builder.append(f"[Sheet: {sheet.title}]")
            if (sheet.max_row or 0) > MAX_CHAT_ATTACHMENT_ROWS:
                raise ValueError(
                    f"Sheet exceeds the {MAX_CHAT_ATTACHMENT_ROWS:,}-row limit."
                )
            if (sheet.max_row or 0) * (
                sheet.max_column or 0
            ) > MAX_CHAT_ATTACHMENT_CELLS:
                raise ValueError(
                    f"Sheet exceeds the {MAX_CHAT_ATTACHMENT_CELLS:,}-cell limit."
                )
            # Producer-declared dimensions can be too small; scan actual rows
            # rather than omitting cells beyond that claimed boundary.
            cast(ReadOnlyWorksheet, sheet).reset_dimensions()
            for number, row in enumerate(
                _xlsx_rows(cast(ReadOnlyWorksheet, sheet)), start=1
            ):
                total_rows += 1
                total_cells += len(row)
                if (
                    total_rows > MAX_CHAT_ATTACHMENT_ROWS
                    or total_cells > MAX_CHAT_ATTACHMENT_CELLS
                ):
                    raise ValueError(
                        "Workbook exceeds the total row or cell limit; split it into smaller files."
                    )
                has_content = has_content or any(
                    value is not None and str(value).strip() for value in row
                )
                builder.append(
                    f"Row {number}: {json.dumps(row, ensure_ascii=False, default=str)}"
                )
        if not has_content:
            raise ValueError("The workbook contains no readable cell values.")
        return builder.render()
    finally:
        workbook.close()


def _csv_text(content: bytes) -> str:
    text = _decode_text(content)
    try:
        dialect = csv.Sniffer().sniff(text[:8_192], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    builder = _TextBuilder()
    total_cells = 0
    has_content = False
    for number, row in enumerate(
        csv.reader(StringIO(text), dialect=dialect, strict=True), start=1
    ):
        total_cells += len(row)
        if number > MAX_CHAT_ATTACHMENT_ROWS or total_cells > MAX_CHAT_ATTACHMENT_CELLS:
            raise ValueError(
                "CSV exceeds the total row or cell limit; split it into smaller files."
            )
        has_content = has_content or any(value.strip() for value in row)
        builder.append(f"Row {number}: {json.dumps(row, ensure_ascii=False)}")
    if not has_content:
        raise ValueError("The CSV contains no readable values.")
    return builder.render()


def extract_chat_attachment_text(filename: str, content: bytes) -> str:
    """Extract complete text or reject explicitly; never silently truncate content.

    These private files use local parsers independent of document import, indexing,
    and model providers. PDF images and Office embedded images are not OCRed.
    """
    from pypdf.errors import PyPdfError

    extension = PurePosixPath(filename).suffix.lower()
    if extension not in CHAT_ATTACHMENT_EXTENSIONS:
        raise ValueError(
            "Unsupported attachment type. Use PDF, DOCX, XLSX, CSV, TXT or MD."
        )
    if len(content) > MAX_CHAT_ATTACHMENT_BYTES:
        raise ValueError("The attachment exceeds the 20 MiB file size limit.")
    if not content:
        raise ValueError("The attachment is empty.")
    try:
        if extension in {".docx", ".xlsx"}:
            _validate_office_archive(content)
        if extension == ".pdf":
            return _pdf_text(content)
        if extension == ".docx":
            return _docx_text(content)
        if extension == ".xlsx":
            return _xlsx_text(content)
        if extension == ".csv":
            return _csv_text(content)
        builder = _TextBuilder()
        text = _decode_text(content)
        if not text.strip():
            raise ValueError("The attachment contains no readable text.")
        builder.append(text)
        return builder.render()
    except (
        BadZipFile,
        ElementTree.ParseError,
        KeyError,
        csv.Error,
        RecursionError,
        InvalidFileException,
        PyPdfError,
        ZlibError,
        NotImplementedError,
    ) as exc:
        raise ValueError(
            "The attachment is malformed or cannot be read safely."
        ) from exc
