from io import BytesIO
from typing import cast
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from docx import Document
from openpyxl import Workbook
from openpyxl.worksheet.worksheet import Worksheet
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from onyx.file_processing import chat_attachments
from onyx.file_processing.chat_attachments import extract_chat_attachment_text


def _pdf(*texts: str, password: str | None = None) -> bytes:
    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=200, height=200)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = stream
    if password:
        writer.encrypt(password)
    content = BytesIO()
    writer.write(content)
    return content.getvalue()


def _xlsx() -> bytes:
    workbook = Workbook()
    sheet = cast(Worksheet, workbook.active)
    sheet.title = "Ürünler"
    sheet.append(["GTIP", "Tutar"])
    sheet.append(["001234", 42.5])
    sheet.append(["Toplam", "=SUM(B2:B2)"])
    workbook.create_sheet("İstisnalar").append(["Koşul", "AB menşeli"])
    content = BytesIO()
    workbook.save(content)
    workbook.close()
    return content.getvalue()


def test_pdf_preserves_every_page_in_order() -> None:
    text = extract_chat_attachment_text(
        "test.PDF", _pdf("First rule", "Second exception")
    )
    assert "[Page 1]\nFirst rule" in text
    assert "[Page 2]\nSecond exception" in text
    assert text.index("First rule") < text.index("Second exception")


def test_pdf_encrypted_and_scanned_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="Password-protected"):
        extract_chat_attachment_text("private.pdf", _pdf("Secret", password="secret"))
    with pytest.raises(ValueError, match="OCR"):
        extract_chat_attachment_text("scanned.pdf", _pdf(""))


def test_docx_preserves_body_table_and_footer() -> None:
    document = Document()
    document.add_paragraph("Ön koşul")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "İstisna"
    table.cell(0, 1).text = "Gümrük vergisi"
    document.add_paragraph("Son koşul")
    document.sections[0].footer.paragraphs[0].text = "Ek açıklama"
    content = BytesIO()
    document.save(content)
    text = extract_chat_attachment_text("terms.docx", content.getvalue())
    assert '["İstisna", "Gümrük vergisi"]' in text
    assert text.index("Ön koşul") < text.index("İstisna") < text.index("Son koşul")
    assert "Ek açıklama" in text


def test_xlsx_preserves_sheet_row_values_and_formula_identity() -> None:
    text = extract_chat_attachment_text("tarife.xlsx", _xlsx())
    assert "[Sheet: Ürünler]" in text
    assert 'Row 2: ["001234", 42.5]' in text
    assert "=SUM(B2:B2)" in text
    assert "have not been calculated" in text
    assert "[Sheet: İstisnalar]" in text
    assert "AB menşeli" in text


def test_xlsx_does_not_trust_incorrectly_small_dimensions() -> None:
    content = BytesIO()
    with ZipFile(BytesIO(_xlsx())) as original, ZipFile(content, "w") as changed:
        for member in original.infolist():
            payload = original.read(member.filename)
            if member.filename == "xl/worksheets/sheet1.xml":
                payload = payload.replace(
                    b'<dimension ref="A1:B3"', b'<dimension ref="A1"'
                )
            changed.writestr(member, payload)
    text = extract_chat_attachment_text("incorrect-dimensions.xlsx", content.getvalue())
    assert 'Row 2: ["001234", 42.5]' in text
    assert "=SUM(B2:B2)" in text


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16"])
def test_csv_preserves_turkish_quoted_values_and_row_numbers(encoding: str) -> None:
    content = 'GTIP;Açıklama\n001234;"İstisna; koşul"\n'.encode(encoding)
    text = extract_chat_attachment_text("tarife.csv", content)
    assert 'Row 1: ["GTIP", "Açıklama"]' in text
    assert 'Row 2: ["001234", "İstisna; koşul"]' in text


@pytest.mark.parametrize("filename", ["notes.txt", "notes.md"])
def test_plain_text_is_kept_complete(filename: str) -> None:
    content = "Başlık\nTüm koşullar ve istisnalar."
    assert extract_chat_attachment_text(filename, content.encode()) == content


@pytest.mark.parametrize("extension", ["pdf", "docx", "xlsx"])
def test_malformed_documents_fail_explicitly(extension: str) -> None:
    with pytest.raises(ValueError, match="malformed|valid"):
        extract_chat_attachment_text(f"bad.{extension}", b"malformed document")


@pytest.mark.parametrize(
    ("filename", "content", "reason"),
    [
        ("empty.txt", b"", "empty"),
        ("blank.csv", b"\n\n", "no readable"),
        ("binary.txt", b"hello\x00world", "binary"),
        ("latin.csv", b"\xff\xff\x80", "UTF-8"),
        ("macro.exe", b"arbitrary", "Unsupported"),
    ],
)
def test_rejected_content_has_clear_reason(
    filename: str, content: bytes, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        extract_chat_attachment_text(filename, content)


def test_limits_reject_instead_of_truncating(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_attachments, "MAX_CHAT_ATTACHMENT_TEXT_CHARS", 10)
    with pytest.raises(ValueError, match="no content was truncated"):
        extract_chat_attachment_text("large.txt", b"12345678901")
    monkeypatch.setattr(chat_attachments, "MAX_CHAT_ATTACHMENT_BYTES", 2)
    with pytest.raises(ValueError, match="file size limit"):
        extract_chat_attachment_text("large.txt", b"123")


def test_pdf_page_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_attachments, "MAX_CHAT_ATTACHMENT_PAGES", 1)
    with pytest.raises(ValueError, match="page limit"):
        extract_chat_attachment_text("two.pdf", _pdf("one", "two"))


def test_spreadsheet_limits_reject_complete_workbook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chat_attachments, "MAX_CHAT_ATTACHMENT_ROWS", 2)
    with pytest.raises(ValueError, match="row limit"):
        extract_chat_attachment_text("tarife.xlsx", _xlsx())
    with pytest.raises(ValueError, match="row or cell limit"):
        extract_chat_attachment_text("many.csv", b"header\none\ntwo\n")


def test_office_expansion_limit_and_entities(monkeypatch: pytest.MonkeyPatch) -> None:
    content = BytesIO()
    with ZipFile(content, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", b"x" * 1_000)
    monkeypatch.setattr(chat_attachments, "MAX_CHAT_ATTACHMENT_ZIP_BYTES", 100)
    with pytest.raises(ValueError, match="expanded"):
        extract_chat_attachment_text("expanded.docx", content.getvalue())

    monkeypatch.setattr(chat_attachments, "MAX_CHAT_ATTACHMENT_ZIP_BYTES", 1_000)
    content = BytesIO()
    with ZipFile(content, "w") as archive:
        archive.writestr(
            "word/document.xml",
            b'<!DOCTYPE root [<!ENTITY text "x">]><root>&text;</root>',
        )
    with pytest.raises(ValueError, match="entities"):
        extract_chat_attachment_text("entities.docx", content.getvalue())
