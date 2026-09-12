"""Structural source extraction, without downloads or canonical writes."""

import base64
import hashlib
import time
from collections import Counter
from io import BytesIO
from typing import TypedDict
from uuid import UUID
from zipfile import ZipFile

from bs4 import BeautifulSoup, Comment, NavigableString, Tag
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
    AnnexVisionWireResult,
    ExtractedAnnexElement,
)
from onyx.regulatory.amendments.annexes.rendering import render_annex_pages
from onyx.regulatory.amendments.annexes.sources import inspect_source
from onyx.regulatory.structured_llm import generate_structured
from onyx.tracing.flows import LLMFlow
from onyx.utils.process_isolation import run_in_isolated_process


class _SourceRetryOptions(TypedDict, total=False):
    provider_max_attempts: int
    deadline: float


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
    blocks = {
        "[document]",
        "html",
        "body",
        "article",
        "section",
        "main",
        "header",
        "footer",
        "nav",
        "div",
        "p",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "aside",
        "ul",
        "ol",
        "li",
        "dl",
        "dt",
        "dd",
        "blockquote",
        "pre",
        "figure",
        "figcaption",
        "address",
        "details",
        "summary",
        "table",
    }
    inline = {
        "a",
        "span",
        "strong",
        "b",
        "em",
        "i",
        "u",
        "s",
        "small",
        "sup",
        "sub",
        "abbr",
        "cite",
        "q",
        "code",
        "time",
        "mark",
        "del",
        "ins",
        "br",
        "hr",
        "wbr",
    }
    ignored = {"head", "script", "style", "template"}
    table_number = 0
    heading = ""

    def visit(tag: Tag, footnote: bool = False, issues: tuple[str, ...] = ()) -> None:
        nonlocal table_number, heading
        if tag.name in ignored:
            return
        footnote = footnote or tag.get("role") == "doc-footnote"
        if tag.name not in blocks | inline:
            issues = (*issues, f"unsupported_html_structure:{tag.name}")
        if tag.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            heading = tag.get_text(" ", strip=True)
        if tag.name == "table":
            table_number += 1
            caption = tag.find("caption")
            scope = f"{heading}/table:{caption.get_text(' ', strip=True) if caption else table_number}"
            # Flatten unsupported nested grids once, without inventing cell anchors.
            if tag.find("table") is not None:
                elements.append(
                    ExtractedAnnexElement(
                        kind="text",
                        text=tag.get_text(" ", strip=True),
                        status="uncertain",
                        issues=[*issues, "nested_html_table_unsupported"],
                        locator=AnnexLocator(path=scope),
                    )
                )
                return
            rows = tag.find_all("tr")
            unsupported = {
                f"unsupported_html_structure:{child.name}"
                for child in tag.find_all(True)
                if child.name
                not in blocks
                | inline
                | {
                    "caption",
                    "col",
                    "colgroup",
                    "thead",
                    "tbody",
                    "tfoot",
                    "tr",
                    "th",
                    "td",
                }
                | ignored
            }
            table_issues = [*issues, *sorted(unsupported)]
            table_elements = _table_elements(
                [
                    [
                        cell.get_text(" ", strip=True)
                        for cell in row.find_all(["th", "td"], recursive=False)
                    ]
                    for row in rows
                ],
                scope,
            )
            if tag.find(attrs={"rowspan": True}) or tag.find(attrs={"colspan": True}):
                for element in table_elements:
                    element.status = "uncertain"
                    element.issues.append("merged_table_cells")
                    element.semantic_key = None
                    element.locator.column = None
            by_row: dict[int, list[ExtractedAnnexElement]] = {}
            for element in table_elements:
                assert element.locator.row is not None
                by_row.setdefault(element.locator.row, []).append(element)
                if table_issues:
                    element.status = "uncertain"
                    element.issues.extend(table_issues)
                    element.semantic_key = None
            row_index = 0
            for descendant in tag.descendants:
                if isinstance(descendant, Tag) and descendant.name == "tr":
                    row_index += 1
                    elements.extend(by_row[row_index])
                elif isinstance(descendant, Tag) and descendant.name == "caption":
                    elements.append(
                        ExtractedAnnexElement(
                            kind="footnote" if footnote else "text",
                            text=descendant.get_text(" ", strip=True),
                            status="uncertain" if table_issues else "readable",
                            issues=table_issues,
                            locator=AnnexLocator(path=scope),
                        )
                    )
                elif isinstance(descendant, NavigableString) and not isinstance(
                    descendant, Comment
                ):
                    if (
                        descendant.find_parent(["td", "th", "caption", *ignored])
                        is None
                        and descendant.strip()
                    ):
                        elements.append(
                            ExtractedAnnexElement(
                                kind="footnote" if footnote else "text",
                                text=str(descendant).strip(),
                                status="uncertain",
                                issues=[
                                    *table_issues,
                                    "unsupported_html_table_content",
                                ],
                                locator=AnnexLocator(path=scope),
                            )
                        )
            return
        pending: list[str] = []

        def flush() -> None:
            text = " ".join(" ".join(pending).split())
            if text:
                elements.append(
                    ExtractedAnnexElement(
                        kind="footnote" if footnote else "text",
                        text=text,
                        semantic_key=str(tag.get("id")) if tag.get("id") else None,
                        status="uncertain" if issues else "readable",
                        issues=list(dict.fromkeys(issues)),
                        locator=AnnexLocator(path=tag.name),
                    )
                )
            pending.clear()

        def collect(node: Tag | NavigableString) -> None:
            if isinstance(node, Comment):
                return
            if isinstance(node, NavigableString):
                pending.append(str(node))
            elif node.name in ignored:
                return
            elif (
                node.name in blocks
                or node.name not in inline
                or node.get("role") == "doc-footnote"
            ):
                flush()
                visit(node, footnote, issues)
            else:
                for child in node.children:
                    if isinstance(child, (Tag, NavigableString)):
                        collect(child)

        for child in tag.children:
            if isinstance(child, (Tag, NavigableString)):
                collect(child)
        flush()
        if tag.name not in blocks | inline and not tag.get_text(strip=True):
            elements.append(
                ExtractedAnnexElement(
                    kind="image_region",
                    text=str(tag.get("alt") or ""),
                    status="unreadable",
                    issues=list(issues),
                    locator=AnnexLocator(path=tag.name),
                )
            )

    visit(soup)
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
                table_elements = _table_elements(
                    rows, f"{heading}/table:{table_number}"
                )
                if any(
                    node.find(f".//w:{property_name}", namespace) is not None
                    for property_name in (
                        "gridSpan",
                        "vMerge",
                        "hMerge",
                        "gridBefore",
                        "gridAfter",
                    )
                ):
                    for element in table_elements:
                        element.status = "uncertain"
                        element.issues.append("docx_table_grid_unsupported")
                        element.semantic_key = None
                        element.locator.column = None
                elements.extend(table_elements)
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
    content: bytes,
    mime_type: str,
    *,
    vision_llm: LLM | None = None,
    vision_deadline: float | None = None,
) -> AnnexExtraction:
    remaining = (
        vision_deadline - time.monotonic() if vision_deadline is not None else 30
    )
    if remaining <= 0:
        raise TimeoutError("pdf_vision_preparation_deadline")
    elements, pages = run_in_isolated_process(
        _native_structure, content, mime_type, timeout=min(30, remaining)
    )
    result = AnnexExtraction(
        source_sha256=hashlib.sha256(content).hexdigest(),
        mime_type=mime_type,
        elements=[
            element.model_copy(update={"extraction_method": "native"})
            for element in elements
        ],
    )
    if pages:
        if vision_llm is not None:
            result.model_snapshot = AnnexModelSnapshot(
                model_provider=vision_llm.config.model_provider,
                model_name=vision_llm.config.model_name,
            )
        else:
            result.issues.append("vision_model_unavailable")
        result.page_count = len(pages)
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
            bounded_options: _SourceRetryOptions = {}
            if vision_deadline is not None:
                remaining = int(vision_deadline - time.monotonic())
                if remaining <= 0:
                    raise TimeoutError("pdf_vision_preparation_deadline")
                bounded_options = {
                    "provider_max_attempts": 3,
                    "deadline": vision_deadline,
                }
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
                response_model=AnnexVisionWireResult,
                timeout_override=min(45, int(remaining))
                if vision_deadline is not None
                else 60,
                max_tokens=12000,
                **bounded_options,
            )
            if vision_deadline is not None and time.monotonic() >= vision_deadline:
                raise TimeoutError("pdf_vision_preparation_deadline")
            if not response.elements:
                result.issues.append(f"page_{page.page}_no_visual_structure")
            # Native PDF text remains source evidence but is not double counted
            # as atomic structure when a complete visual transcription exists.
            for native in page.text_elements:
                native.aggregate = True
            for item in response.elements:
                box = item.box.as_tuple()
                left, top, right, bottom = box
                result.elements.append(
                    ExtractedAnnexElement(
                        kind=item.kind,
                        table_role=item.table_role,
                        extraction_method="vision",
                        text=item.text,
                        status=item.status,
                        issues=list(item.issues),
                        evidence_kind=evidence_kind,
                        locator=AnnexLocator(
                            page=page.page,
                            original_box=(
                                left * page.width,
                                top * page.height,
                                right * page.width,
                                bottom * page.height,
                            ),
                            normalized_box=box,
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

    package = require_ready_source_package(
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
    result = None
    if (
        asset.mime_type == "application/pdf"
        and package.manifest_file_id
        and package.manifest_sha256
    ):
        from onyx.regulatory.amendments.pdf_vision import load_frozen_pdf_asset

        result = load_frozen_pdf_asset(
            store,
            manifest_file_id=package.manifest_file_id,
            manifest_sha256=package.manifest_sha256,
            source_sha256=asset.sha256,
        )
    if result is None:
        result = extract_annex_structure(
            content, asset.mime_type, vision_llm=vision_llm
        )
    for element in result.elements:
        element.source_asset_id = str(asset.id)
    return result
