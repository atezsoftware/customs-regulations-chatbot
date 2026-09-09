"""Structural source extraction, without downloads or canonical writes."""

import base64
import hashlib
from collections import Counter
from io import BytesIO
from typing import cast
from uuid import UUID
from zipfile import ZipFile

from bs4 import BeautifulSoup, Tag
from defusedxml import ElementTree as etree
from sqlalchemy.orm import Session

from onyx.file_store.file_store import FileStore
from onyx.llm.interfaces import LLM
from onyx.llm.models import ImageContentPart, ImageUrlDetail
from onyx.prompts.regulatory.annex_extraction import ANNEX_STRUCTURE_PROMPT
from onyx.regulatory.amendments.annexes.models import (
    AnnexExtraction,
    AnnexLocator,
    AnnexModelSnapshot,
    AnnexRenderedPage,
    AnnexVisionResult,
    ExtractedAnnexElement,
)
from onyx.regulatory.amendments.annexes.rendering import render_annex_pages
from onyx.regulatory.amendments.annexes.sources import inspect_source
from onyx.regulatory.structured_llm import generate_structured
from onyx.tracing.flows import LLMFlow
from onyx.utils.process_isolation import run_in_isolated_process


def _table_elements(
    rows: list[list[str]], scope: str, *, sheet: str | None = None
) -> list[ExtractedAnnexElement]:
    elements: list[ExtractedAnnexElement] = []
    for row_index, cells in enumerate(rows, 1):
        key = f"{scope}/row:{cells[0]}" if cells and cells[0] else None
        elements.append(
            ExtractedAnnexElement(
                kind="table_row",
                text=" | ".join(cells),
                semantic_key=key,
                aggregate=True,
                locator=AnnexLocator(row=row_index, sheet=sheet, path=scope),
            )
        )
        for column, text in enumerate(cells, 1):
            header = (
                rows[0][column - 1] if rows and len(rows[0]) >= column else str(column)
            )
            elements.append(
                ExtractedAnnexElement(
                    kind="table_cell",
                    text=text,
                    semantic_key=f"{key}/column:{header}" if key else None,
                    locator=AnnexLocator(
                        row=row_index, column=column, sheet=sheet, path=scope
                    ),
                )
            )
    return elements


def _html_elements(content: bytes) -> list[ExtractedAnnexElement]:
    soup = BeautifulSoup(content, "html.parser")
    elements: list[ExtractedAnnexElement] = []
    for index, table in enumerate(soup.find_all("table"), 1):
        heading = table.find_previous(["h1", "h2", "h3", "h4"])
        caption = table.find("caption")
        scope = f"{heading.get_text(' ', strip=True) if heading else ''}/table:{caption.get_text(' ', strip=True) if caption else index}"
        rows = [
            [
                cell.get_text(" ", strip=True)
                for cell in row.find_all(["th", "td"], recursive=False)
            ]
            for row in table.find_all("tr")
            if row.find_parent("table") is table
        ]
        table_elements = _table_elements(rows, scope)
        if table.find(attrs={"rowspan": True}) or table.find(attrs={"colspan": True}):
            for element in table_elements:
                element.status = "uncertain"
                element.issues.append("merged_table_cells")
        elements.extend(table_elements)
    for tag in soup.find_all(["p", "h1", "h2", "h3", "h4", "aside"]):
        if tag.find_parent("table") is not None:
            continue
        text = tag.get_text(" ", strip=True)
        if text:
            footnote = tag.get("role") == "doc-footnote"
            elements.append(
                ExtractedAnnexElement(
                    kind="footnote" if footnote else "text",
                    text=text,
                    semantic_key=str(tag.get("id")) if tag.get("id") else None,
                    locator=AnnexLocator(path=cast(Tag, tag).name),
                )
            )
    return elements


def _docx_elements(content: bytes) -> list[ExtractedAnnexElement]:
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    elements: list[ExtractedAnnexElement] = []
    with ZipFile(BytesIO(content)) as archive:
        root = etree.fromstring(archive.read("word/document.xml"))
        body = root.find("w:body", namespace)
        if body is None:
            raise ValueError("docx_body_missing")
        table_number = 0
        heading = ""
        for index, node in enumerate(body):
            text = " ".join(
                [part.text or "" for part in node.findall(".//w:t", namespace)]
            )
            if node.tag.endswith("}tbl"):
                table_number += 1
                rows = [
                    [
                        " ".join(
                            [
                                part.text or ""
                                for part in cell.findall(".//w:t", namespace)
                            ]
                        )
                        for cell in row.findall("w:tc", namespace)
                    ]
                    for row in node.findall("w:tr", namespace)
                ]
                elements.extend(
                    _table_elements(rows, f"{heading}/table:{table_number}")
                )
            elif text:
                if any(
                    (style.get(f"{{{namespace['w']}}}val") or "").startswith("Heading")
                    for style in node.findall(".//w:pStyle", namespace)
                ):
                    heading = text
                elements.append(
                    ExtractedAnnexElement(
                        kind="text",
                        text=text,
                        locator=AnnexLocator(path=f"body/{index}"),
                    )
                )
        if "word/footnotes.xml" in archive.namelist():
            notes = etree.fromstring(archive.read("word/footnotes.xml"))
            for note in notes:
                text = " ".join(
                    [part.text or "" for part in note.findall(".//w:t", namespace)]
                )
                if text:
                    key = note.get(f"{{{namespace['w']}}}id")
                    elements.append(
                        ExtractedAnnexElement(
                            kind="footnote",
                            text=text,
                            semantic_key=f"footnote:{key}",
                            locator=AnnexLocator(path=f"word/footnotes.xml/{key}"),
                        )
                    )
    return elements


def _xlsx_elements(content: bytes) -> list[ExtractedAnnexElement]:
    from openpyxl import load_workbook
    from openpyxl.utils.cell import get_column_letter

    formulas = load_workbook(BytesIO(content), read_only=True, data_only=False)
    values = load_workbook(BytesIO(content), read_only=True, data_only=True)
    elements: list[ExtractedAnnexElement] = []
    try:
        for sheet in formulas:
            if (
                sheet.max_row is None
                or sheet.max_column is None
                or sheet.max_row * sheet.max_column > 20000
            ):
                raise ValueError("annex_cell_limit")
            rows = list(sheet.iter_rows())
            table_elements = _table_elements(
                [
                    [str(cell.value) if cell.value is not None else "" for cell in row]
                    for row in rows
                ],
                f"sheet:{sheet.title}",
                sheet=sheet.title,
            )
            for element in table_elements:
                if element.kind != "table_cell":
                    continue
                assert (
                    element.locator.row is not None
                    and element.locator.column is not None
                )
                coordinate = (
                    f"{get_column_letter(element.locator.column)}{element.locator.row}"
                )
                cell = sheet.cell(element.locator.row, element.locator.column)
                element.locator.cell = coordinate
                value = (
                    values[sheet.title]
                    .cell(element.locator.row, element.locator.column)
                    .value
                )
                element.value = (
                    value
                    if isinstance(value, (str, int, float, bool))
                    else str(value)
                    if value is not None
                    else None
                )
                if cell.data_type == "f":
                    element.formula = str(cell.value)
                    if value is None:
                        element.issues.append("formula_cache_missing")
                        element.status = "uncertain"
            elements.extend(table_elements)
    finally:
        formulas.close()
        values.close()
    return elements


def _native_structure(
    content: bytes, mime_type: str
) -> tuple[list[ExtractedAnnexElement], list[AnnexRenderedPage]]:
    from onyx.regulatory.amendments.annexes.source_parser import (
        apply_source_process_limits,
    )

    apply_source_process_limits()
    inspect_source(content, mime_type)
    if mime_type == "text/html":
        elements = _html_elements(content)
    elif mime_type.endswith("wordprocessingml.document"):
        elements = _docx_elements(content)
    elif mime_type.endswith("spreadsheetml.sheet"):
        elements = _xlsx_elements(content)
    elif mime_type == "application/pdf" or mime_type.startswith("image/"):
        return [], render_annex_pages(content, mime_type)
    else:
        elements = [ExtractedAnnexElement(kind="text", text=content.decode("utf-8"))]
    if (
        len(elements) > 20000
        or sum(len(element.text) for element in elements) > 2_000_000
    ):
        raise ValueError("annex_structure_limit")
    return elements, []


def extract_annex_structure(
    content: bytes, mime_type: str, *, vision_llm: LLM | None = None
) -> AnnexExtraction:
    elements, pages = run_in_isolated_process(
        _native_structure, content, mime_type, timeout=30
    )
    result = AnnexExtraction(
        source_sha256=hashlib.sha256(content).hexdigest(),
        mime_type=mime_type,
        elements=elements,
    )
    if pages:
        if vision_llm is not None:
            result.model_snapshot = AnnexModelSnapshot(
                model_provider=vision_llm.config.model_provider,
                model_name=vision_llm.config.model_name,
            )
        else:
            result.issues.append("vision_model_unavailable")
        original_image_input = (
            mime_type in ("image/png", "image/jpeg", "image/webp")
            and len(pages) == 1
            and pages[0].original_orientation == 1
        )
        for page in pages:
            evidence_kind = "original" if original_image_input else "rendered_preview"
            image_mime = mime_type if original_image_input else "image/png"
            image_bytes = content if original_image_input else page.png
            result.elements.extend(page.text_elements)
            if vision_llm is None:
                result.elements.append(
                    ExtractedAnnexElement(
                        kind="image_region",
                        text="",
                        locator=AnnexLocator(page=page.page),
                        evidence_kind=evidence_kind,
                        status="unreadable",
                        issues=["vision_model_unavailable"],
                    )
                )
                continue
            response = generate_structured(
                vision_llm,
                flow=LLMFlow.REGULATORY_ANNEX_EXTRACTION,
                system_prompt=ANNEX_STRUCTURE_PROMPT,
                user_prompt=f"Page {page.page}. Native text (evidence): "
                + "\n".join(element.text for element in page.text_elements),
                image_parts=[
                    ImageContentPart(
                        image_url=ImageUrlDetail(
                            url=f"data:{image_mime};base64,"
                            + base64.b64encode(image_bytes).decode()
                        )
                    )
                ],
                response_model=AnnexVisionResult,
                timeout_override=60,
                max_tokens=12000,
            )
            if not response.elements:
                result.issues.append(f"page_{page.page}_no_visual_structure")
            # Native PDF text remains source evidence but is not double counted
            # as atomic structure when a complete visual transcription exists.
            for native in page.text_elements:
                native.aggregate = True
            for item in response.elements:
                left, top, right, bottom = item.box
                if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
                    raise ValueError("invalid_vision_coordinates")
                result.elements.append(
                    ExtractedAnnexElement(
                        kind=item.kind,
                        text=item.text,
                        status=item.status,
                        issues=item.issues,
                        evidence_kind=evidence_kind,
                        locator=AnnexLocator(
                            page=page.page,
                            original_box=(
                                left * page.width,
                                top * page.height,
                                right * page.width,
                                bottom * page.height,
                            ),
                            normalized_box=item.box,
                            original_width=page.width,
                            original_height=page.height,
                            coordinate_system="top_left_points"
                            if mime_type == "application/pdf"
                            else "top_left_pixels",
                        ),
                    )
                )
    if (
        len(result.elements) > 20000
        or sum(len(element.text) for element in result.elements) > 2_000_000
    ):
        raise ValueError("annex_structure_limit")
    counts = Counter(
        element.semantic_key for element in result.elements if element.semantic_key
    )
    for element in result.elements:
        if element.semantic_key and counts[element.semantic_key] > 1:
            element.issues.append("ambiguous_semantic_key")
            element.semantic_key = None
    return result


def extract_source_asset(
    session: Session,
    store: FileStore,
    *,
    package_id: UUID,
    asset_id: UUID,
    document_set_id: int,
    environment: str,
    vision_llm: LLM | None = None,
) -> AnnexExtraction:
    """Consume a trusted package asset; model output never selects a download."""
    from onyx.db.amendment_sources import get_source_asset, require_ready_source_package

    require_ready_source_package(
        session,
        package_id=package_id,
        document_set_id=document_set_id,
        environment=environment,
    )
    asset = get_source_asset(session, package_id=package_id, asset_id=asset_id)
    if asset is None:
        raise ValueError("annex asset scope mismatch")
    with store.read_file(asset.file_id) as stream:
        content = stream.read(25 * 1024 * 1024 + 1)
    if (
        len(content) != asset.byte_count
        or hashlib.sha256(content).hexdigest() != asset.sha256
    ):
        raise ValueError("annex source integrity mismatch")
    result = extract_annex_structure(content, asset.mime_type, vision_llm=vision_llm)
    for element in result.elements:
        element.source_asset_id = str(asset.id)
    return result
