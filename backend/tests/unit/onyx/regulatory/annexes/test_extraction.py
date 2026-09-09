from io import BytesIO
from unittest.mock import MagicMock

import pytest
from PIL import Image


def test_html_rows_cells_and_footnotes_preserve_anchors() -> None:
    from onyx.regulatory.amendments.annexes.extraction import extract_annex_structure

    old = b'<h1>EK-1</h1><table><tr><th>Code</th><th>Rate</th></tr><tr><td>A</td><td>5</td></tr><tr><td>B</td><td>8</td></tr></table><aside role="doc-footnote" id="n1">Note</aside>'
    new = old.replace(b"<tr><td>B", b"<tr><td>X</td><td>6</td></tr><tr><td>B")
    before = extract_annex_structure(old, "text/html")
    after = extract_annex_structure(new, "text/html")
    old_b = next(e for e in before.elements if e.text == "8")
    new_b = next(e for e in after.elements if e.text == "8")
    assert old_b.semantic_key == new_b.semantic_key
    assert old_b.locator.row == 3 and new_b.locator.row == 4
    assert any(e.kind == "footnote" and e.text == "Note" for e in before.elements)
    assert [e.text for e in before.elements if e.kind == "table_row"] == [
        "Code | Rate",
        "A | 5",
        "B | 8",
    ]


def test_html_retains_legal_text_in_document_order_without_container_duplicates() -> (
    None
):
    from onyx.regulatory.amendments.annexes.extraction import extract_annex_structure

    result = extract_annex_structure(
        b"<h1>EK-1</h1><div>Before <strong>table</strong>"
        b"<ul><li>Exception <em>A</em><ul><li>Nested exception</li></ul></li></ul>"
        b"<div>Condition</div>Direct text</div>"
        b"<table><tr><td>A</td><td>5%</td></tr></table>"
        b'<aside role="doc-footnote" id="n1"><p>Note <b>one</b></p>'
        b"<div>Note two</div></aside><p>After table</p>"
        b"<svg><text>Diagram condition</text></svg>",
        "text/html",
    )
    atomic = [element for element in result.elements if not element.aggregate]
    assert [element.text for element in atomic] == [
        "EK-1",
        "Before table",
        "Exception A",
        "Nested exception",
        "Condition",
        "Direct text",
        "A",
        "5%",
        "Note one",
        "Note two",
        "After table",
        "Diagram condition",
    ]
    assert [element.text for element in atomic if element.kind == "footnote"] == [
        "Note one",
        "Note two",
    ]
    assert atomic[-1].status == "uncertain"
    assert "unsupported_html_structure:svg" in atomic[-1].issues


def test_html_unsupported_table_content_retains_caption_and_image_evidence() -> None:
    from onyx.regulatory.amendments.annexes.extraction import extract_annex_structure

    result = extract_annex_structure(
        b"<table><caption>Rates <b>and exceptions</b></caption>"
        b'<tr><td>A</td><td><img alt="Rate diagram"/>5%</td></tr></table>',
        "text/html",
    )
    atomic = [element for element in result.elements if not element.aggregate]
    assert [element.text for element in atomic] == ["Rates and exceptions", "A", "5%"]
    assert atomic[-1].status == "uncertain"
    assert "unsupported_html_structure:img" in atomic[-1].issues
    assert atomic[-1].semantic_key is None


@pytest.mark.parametrize(
    "layout", ["gridSpan", "vMerge", "hMerge", "gridBefore", "gridAfter"]
)
def test_docx_unsupported_grid_disables_false_column_anchors(layout: str) -> None:
    from zipfile import ZipFile

    from onyx.regulatory.amendments.annexes.extraction import extract_annex_structure

    row_properties = (
        f'<w:trPr><w:{layout} w:val="1"/></w:trPr>'
        if layout.startswith("gridB") or layout == "gridAfter"
        else ""
    )
    cell_value = "restart" if layout in {"vMerge", "hMerge"} else "2"
    cell_properties = (
        f'<w:tcPr><w:{layout} w:val="{cell_value}"/></w:tcPr>'
        if not row_properties
        else ""
    )
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>',
        )
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:tbl>'
            "<w:tr><w:tc><w:p><w:t>Code</w:t></w:p></w:tc><w:tc><w:p><w:t>Description</w:t></w:p></w:tc><w:tc><w:p><w:t>Rate</w:t></w:p></w:tc></w:tr>"
            f"<w:tr>{row_properties}<w:tc>{cell_properties}<w:p><w:t>A</w:t></w:p></w:tc><w:tc><w:p><w:t>5%</w:t></w:p></w:tc></w:tr>"
            "</w:tbl></w:body></w:document>",
        )
    result = extract_annex_structure(
        buffer.getvalue(),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    rate = next(element for element in result.elements if element.text == "5%")
    assert rate.locator.row == 2
    assert rate.locator.column is None
    assert all(element.semantic_key is None for element in result.elements)
    assert all(element.status == "uncertain" for element in result.elements)
    assert all(
        "docx_table_grid_unsupported" in element.issues for element in result.elements
    )


@pytest.mark.parametrize("corrected", [True, False])
def test_vision_rejects_row_plus_cells_and_retries_with_atomic_cells(
    corrected: bool,
) -> None:
    import json

    from onyx.llm.model_response import Choice, Message, ModelResponse
    from onyx.regulatory.amendments.annexes.extraction import extract_annex_structure

    stream = BytesIO()
    Image.new("RGB", (200, 100), "white").save(stream, format="PNG")
    row = {
        "kind": "table_row",
        "text": "A | 5%",
        "box": [0, 0, 1, 1],
        "status": "readable",
    }
    cells = [
        {
            "kind": "table_cell",
            "text": "A",
            "box": [0, 0, 0.5, 1],
            "status": "readable",
        },
        {
            "kind": "table_cell",
            "text": "5%",
            "box": [0.5, 0, 1, 1],
            "status": "readable",
        },
    ]
    llm = MagicMock()
    llm.config.model_name = "configured-model"
    llm.config.model_provider = "configured-provider"
    llm.invoke.side_effect = [
        ModelResponse(
            id="vision",
            created="2026-01-01",
            choice=Choice(message=Message(content=json.dumps({"elements": elements}))),
        )
        for elements in [[row, *cells], cells if corrected else [row, *cells]]
    ]
    if not corrected:
        with pytest.raises(
            ValueError,
            match="LLM failed to produce valid AnnexVisionResult after 2 attempts",
        ):
            extract_annex_structure(stream.getvalue(), "image/png", vision_llm=llm)
        return
    result = extract_annex_structure(stream.getvalue(), "image/png", vision_llm=llm)
    assert [
        (element.kind, element.text, element.aggregate) for element in result.elements
    ] == [
        ("table_cell", "A", False),
        ("table_cell", "5%", False),
    ]
    assert llm.invoke.call_count == 2


def test_xlsx_retains_formula_and_cached_value() -> None:
    from openpyxl import Workbook

    from onyx.regulatory.amendments.annexes.extraction import extract_annex_structure

    workbook = Workbook()
    sheet = workbook.active
    from openpyxl.worksheet.worksheet import Worksheet

    assert isinstance(sheet, Worksheet)
    sheet.title = "EK-1"
    sheet.append(["Code", "Rate"])
    sheet.append(["A", "=1+4"])
    stream = BytesIO()
    workbook.save(stream)
    result = extract_annex_structure(
        stream.getvalue(),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    formula = next(e for e in result.elements if e.locator.cell == "B2")
    assert formula.formula == "=1+4"
    assert formula.value is None
    assert formula.locator.sheet == "EK-1"
    assert "formula_cache_missing" in formula.issues


def test_image_vision_extracts_real_pixels_and_retains_uncertainty() -> None:
    from onyx.llm.model_response import Choice, Message, ModelResponse
    from onyx.regulatory.amendments.annexes.extraction import extract_annex_structure

    stream = BytesIO()
    Image.new("RGB", (200, 100), "white").save(stream, format="PNG")
    llm = MagicMock()
    llm.config.model_name = "configured-model"
    llm.config.model_provider = "configured-provider"
    llm.invoke.return_value = ModelResponse(
        id="vision",
        created="2026-01-01",
        choice=Choice(
            message=Message(
                content='{"elements":[{"kind":"image_region","text":"unclear","box":[0.1,0.2,0.8,0.9],"status":"uncertain","issues":["illegible rate"]}]}'
            )
        ),
    )
    result = extract_annex_structure(stream.getvalue(), "image/png", vision_llm=llm)
    element = result.elements[0]
    assert element.status == "uncertain"
    assert element.locator.original_box == (20.0, 20.0, 160.0, 90.0)
    assert element.locator.normalized_box == (0.1, 0.2, 0.8, 0.9)
    assert element.evidence_kind == "original"
    assert result.model_snapshot is not None
    assert result.model_snapshot.model_name == "configured-model"
    messages = llm.invoke.call_args.args[0]
    assert messages[1].content[1].image_url.url.startswith("data:image/png;base64,")


def test_missing_vision_is_explicit_unreadable() -> None:
    from onyx.regulatory.amendments.annexes.extraction import extract_annex_structure

    stream = BytesIO()
    Image.new("RGB", (20, 10)).save(stream, format="PNG")
    result = extract_annex_structure(stream.getvalue(), "image/png")
    assert result.elements[0].status == "unreadable"
    assert result.issues == ["vision_model_unavailable"]


def test_multpage_pdf_retains_native_positions_and_preview_provenance() -> None:
    from pathlib import Path

    from onyx.regulatory.amendments.annexes.extraction import extract_annex_structure

    content = (Path(__file__).parent / "fixtures/two_page_annex.pdf").read_bytes()
    result = extract_annex_structure(content, "application/pdf")
    assert {e.locator.page for e in result.elements} == {1, 2}
    native = [e for e in result.elements if e.kind == "text"]
    assert [e.text for e in native] == ["EK-1 page one", "EK-1 page two"]
    assert all(
        e.locator.original_box
        and e.locator.coordinate_system == "top_left_points"
        and e.evidence_kind == "original"
        for e in native
    )
    assert all(
        e.evidence_kind == "rendered_preview"
        for e in result.elements
        if e.kind == "image_region"
    )


def test_docx_table_and_footnote_structure() -> None:
    from zipfile import ZipFile

    from onyx.regulatory.amendments.annexes.extraction import extract_annex_structure

    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>',
        )
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:t>EK-1</w:t></w:p><w:tbl><w:tr><w:tc><w:p><w:t>Code</w:t></w:p></w:tc><w:tc><w:p><w:t>Rate</w:t></w:p></w:tc></w:tr><w:tr><w:tc><w:p><w:t>A</w:t></w:p></w:tc><w:tc><w:p><w:t>5</w:t></w:p></w:tc></w:tr></w:tbl></w:body></w:document>',
        )
        archive.writestr(
            "word/footnotes.xml",
            '<w:footnotes xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:footnote w:id="1"><w:p><w:t>Approved footnote</w:t></w:p></w:footnote></w:footnotes>',
        )
    result = extract_annex_structure(
        buffer.getvalue(),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    assert [e.text for e in result.elements if e.kind == "table_cell"] == [
        "Code",
        "Rate",
        "A",
        "5",
    ]
    assert (
        next(e for e in result.elements if e.kind == "footnote").semantic_key
        == "footnote:1"
    )


def test_duplicate_row_labels_do_not_claim_stable_semantic_identity() -> None:
    from onyx.regulatory.amendments.annexes.extraction import extract_annex_structure

    result = extract_annex_structure(
        b"<table><tr><th>Code</th><th>Rate</th></tr><tr><td>A</td><td>5</td></tr><tr><td>A</td><td>8</td></tr></table>",
        "text/html",
    )
    assert all(
        e.semantic_key is None and "ambiguous_semantic_key" in e.issues
        for e in result.elements
        if e.locator.row in [2, 3]
    )


def test_tiff_frames_remain_previews_with_full_frame_coverage() -> None:
    from onyx.regulatory.amendments.annexes.extraction import extract_annex_structure

    buffer = BytesIO()
    Image.new("RGB", (30, 20)).save(
        buffer,
        format="TIFF",
        save_all=True,
        append_images=[Image.new("RGB", (30, 20), "white")],
    )
    result = extract_annex_structure(buffer.getvalue(), "image/tiff")
    assert [e.locator.page for e in result.elements] == [1, 2]
    assert all(e.evidence_kind == "rendered_preview" for e in result.elements)


def test_native_resource_budget_blocks_instead_of_truncating() -> None:
    import pytest
    from pypdf import PdfWriter

    from onyx.regulatory.amendments.annexes.extraction import extract_annex_structure

    writer = PdfWriter()
    writer.add_blank_page(width=10000, height=10000)
    buffer = BytesIO()
    writer.write(buffer)
    with pytest.raises(ValueError, match="annex_render_limit"):
        extract_annex_structure(buffer.getvalue(), "application/pdf")


def test_oriented_jpeg_uses_explicit_preview_coordinates() -> None:
    from onyx.llm.model_response import Choice, Message, ModelResponse
    from onyx.regulatory.amendments.annexes.extraction import extract_annex_structure

    buffer = BytesIO()
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (200, 100)).save(buffer, format="JPEG", exif=exif)
    llm = MagicMock()
    llm.config.model_name = "configured-model"
    llm.config.model_provider = "configured-provider"
    llm.invoke.return_value = ModelResponse(
        id="test",
        created="2026-01-01",
        choice=Choice(
            message=Message(
                content='{"elements":[{"kind":"image_region","text":"region","box":[0,0,1,1],"status":"readable","issues":[]}]}'
            )
        ),
    )
    result = extract_annex_structure(buffer.getvalue(), "image/jpeg", vision_llm=llm)
    assert result.elements[0].evidence_kind == "rendered_preview"
    assert result.elements[0].locator.original_box == (0, 0, 200, 100)
    assert (
        llm.invoke.call_args.args[0][1]
        .content[1]
        .image_url.url.startswith("data:image/png;")
    )


def test_rotated_pdf_preserves_native_coordinates_without_false_normalization() -> None:
    from pathlib import Path

    from pypdf import PdfReader, PdfWriter

    from onyx.regulatory.amendments.annexes.extraction import extract_annex_structure

    reader = PdfReader(Path(__file__).parent / "fixtures/two_page_annex.pdf")
    writer = PdfWriter()
    writer.add_page(reader.pages[0].rotate(90))
    buffer = BytesIO()
    writer.write(buffer)
    result = extract_annex_structure(buffer.getvalue(), "application/pdf")
    native = next(element for element in result.elements if element.kind == "text")
    assert native.locator.coordinate_system == "pdf_user_space"
    assert native.locator.normalized_box is None
    assert native.status == "uncertain"
    assert "rotated_pdf_native_coordinates" in native.issues
