"""Verified original-file, page and table research tools; never publication writes."""

import base64
import csv
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from hashlib import sha256
from io import StringIO
from threading import local
from typing import cast
from uuid import UUID

from pydantic import JsonValue

from onyx.asv3.corpus_tools import (
    SOURCE_FIELD,
    CorpusBroker,
    guarded,
    json_value,
    schema,
)
from onyx.asv3.llm_adapter import model_slot
from onyx.asv3.models import (
    Artifact,
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolSpec,
)
from onyx.db.asv3_corpus import CorpusScopeUnavailable, original_source_record
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.file_processing.original_attachment import require_original_attachment_bytes
from onyx.file_store.file_store import get_default_file_store
from onyx.regulatory.amendments.annexes.extraction import extract_annex_structure
from onyx.regulatory.amendments.annexes.models import AnnexExtraction
from onyx.regulatory.amendments.annexes.rendering import render_annex_pages
from onyx.tracing.flows import LLMFlow
from onyx.tracing.llm_utils import llm_generation_span
from onyx.utils.process_isolation import run_in_isolated_process

MAX_ORIGINAL_BYTES = 25 * 1024 * 1024
MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_EXTRACT_CHARS = 64_000
_source_slots = local()


@contextmanager
def source_slot(context: RunContext, *, research: bool = True) -> Iterator[None]:
    """Bound original-file memory for the entire operation, including parsing."""
    check = context.check_research_active if research else context.check_active
    check()
    held: set[int] = getattr(_source_slots, "held", set())
    key = id(context.budget)
    if key in held:
        yield
        check()
        return
    while not context.budget.source_slots.acquire(timeout=0.05):
        check()
    held.add(key)
    _source_slots.held = held
    try:
        check()
        yield
        check()
    finally:
        held.remove(key)
        context.budget.source_slots.release()


def bounded_source_operation(
    handler: Callable[[dict[str, JsonValue], RunContext], ToolOutcome],
) -> Callable[[dict[str, JsonValue], RunContext], ToolOutcome]:
    def run(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        with source_slot(context):
            return handler(args, context)

    return run


def extract_source_vision(
    broker: CorpusBroker, content: bytes, mime: str, context: RunContext
) -> AnnexExtraction:
    if broker.vision_llm is None:
        raise CorpusScopeUnavailable("No source vision model is configured.")
    context.consume_research_decision()
    with (
        model_slot(context, research=True),
        llm_generation_span(broker.vision_llm, LLMFlow.ASV3_SOURCE_VISION),
    ):
        result = extract_annex_structure(
            content,
            mime,
            vision_llm=broker.vision_llm,
            vision_deadline=context.research_deadline,
        )
    context.check_research_active()
    return result


def read_verified_original(
    broker: CorpusBroker, source_id: str, context: RunContext
) -> tuple[bytes, str, str]:
    context.check_active()
    with get_session_with_current_tenant() as session:
        source, mime = original_source_record(
            session, user=broker.user, filters=broker.filters, source_id=UUID(source_id)
        )
    store = broker.file_store or get_default_file_store()
    with store.read_file(source.file_id, mode="b") as stream:
        content = stream.read(MAX_ORIGINAL_BYTES + 1)
    if len(content) > MAX_ORIGINAL_BYTES:
        raise CorpusScopeUnavailable(
            "Original exceeds the bounded research file limit."
        )
    require_original_attachment_bytes(source.file_id, content)
    broker.source(source_id, context)
    return content, mime, source.name


def original_evidence(
    source_id: str, text: str, digest: str, locator: dict[str, JsonValue]
) -> EvidenceItem:
    return EvidenceItem(
        source_id=source_id,
        text=text,
        metadata={
            "source_sha256": digest,
            "extraction": "original_native",
            "locator": locator,
            "canonical_identity": "not_assigned",
            "derived": True,
        },
    )


def selected_pdf(content: bytes, page_numbers: tuple[int, ...]) -> bytes:
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(__import__("io").BytesIO(content))
    if (
        not page_numbers
        or min(page_numbers) < 1
        or max(page_numbers) > len(reader.pages)
    ):
        raise ValueError("Selected PDF page is outside the original document.")
    writer = PdfWriter()
    for number in page_numbers:
        writer.add_page(reader.pages[number - 1])
    output = __import__("io").BytesIO()
    writer.write(output)
    return output.getvalue()


def build_source_specs(broker: CorpusBroker) -> list[ToolSpec]:
    def open_file(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        source_id = str(args["source_id"])
        content, mime, name = read_verified_original(broker, source_id, context)
        digest = sha256(content).hexdigest()
        if mime.startswith("text/") or mime in {"application/json", "application/xml"}:
            start = int(cast(int, args.get("start", 0)))
            limit = int(cast(int, args.get("limit", 16000)))
            text = content.decode("utf-8")
            selected = text[start : start + limit]
            more = start + limit < len(text)
            return ToolOutcome(
                status=OutcomeStatus.PARTIAL if more else OutcomeStatus.FOUND,
                summary="Verified original text range.",
                data={
                    "name": name,
                    "mime_type": mime,
                    "source_sha256": digest,
                    "byte_count": len(content),
                    "next_char": start + len(selected),
                    "has_more": more,
                },
                evidence=[
                    original_evidence(
                        source_id,
                        selected,
                        digest,
                        {"start_char": start, "end_char": start + len(selected)},
                    )
                ],
            )
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Original bytes verified; use page/table inspection or stage this source in the research sandbox.",
            data={
                "name": name,
                "mime_type": mime,
                "byte_count": len(content),
                "source_sha256": digest,
            },
            artifacts=[
                Artifact(
                    artifact_id=f"original:{source_id}:{digest}",
                    name=name,
                    media_type=mime,
                    source_ids=[source_id],
                    metadata={
                        "source_sha256": digest,
                        "byte_count": len(content),
                        "original": True,
                    },
                )
            ],
        )

    def page(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        source_id = str(args["source_id"])
        content, mime, _ = read_verified_original(broker, source_id, context)
        number = int(cast(int, args["page"]))
        if mime != "application/pdf" and not mime.startswith("image/"):
            return ToolOutcome(
                status=OutcomeStatus.INVALID,
                summary="Page inspection requires PDF or image source.",
            )
        remaining = context.deadline - time.monotonic()
        pages = run_in_isolated_process(
            render_annex_pages,
            content,
            mime,
            page_numbers=(number,),
            timeout=min(30, max(0.1, remaining)),
        )
        if not pages:
            return ToolOutcome(
                status=OutcomeStatus.UNAVAILABLE,
                summary="Source page could not be rendered.",
            )
        rendered = pages[0]
        digest = sha256(content).hexdigest()
        text = "\n".join(item.text for item in rendered.text_elements)
        artifacts = []
        if len(rendered.png) <= MAX_IMAGE_BYTES:
            artifacts.append(
                Artifact(
                    artifact_id=f"page:{source_id}:{digest}:{number}",
                    name=f"page-{number}.png",
                    media_type="image/png",
                    source_ids=[source_id],
                    metadata={
                        "source_sha256": digest,
                        "page": number,
                        "base64": base64.b64encode(rendered.png).decode("ascii"),
                    },
                )
            )
        partial = len(rendered.png) > MAX_IMAGE_BYTES or len(text) > MAX_EXTRACT_CHARS
        result = ToolOutcome(
            status=OutcomeStatus.PARTIAL if partial else OutcomeStatus.FOUND,
            summary="Verified source page with native text, locations and bounded page image.",
            data={
                "source_sha256": digest,
                "page": number,
                "image_available": bool(artifacts),
                "native_locations": [
                    cast(dict[str, JsonValue], json_value(item.locator.model_dump()))
                    for item in rendered.text_elements[:100]
                ],
                "text_truncated": len(text) > MAX_EXTRACT_CHARS,
            },
            evidence=[
                original_evidence(
                    source_id, text[:MAX_EXTRACT_CHARS], digest, {"page": number}
                )
            ]
            if text
            else [],
            artifacts=artifacts,
        )
        if args.get("vision"):
            if broker.vision_llm is None:
                result.status = OutcomeStatus.UNAVAILABLE
                result.summary = (
                    "Native page is available, but no vision model is configured."
                )
                return result
            payload = (
                selected_pdf(content, (number,))
                if mime == "application/pdf"
                else content
            )
            extraction = extract_source_vision(broker, payload, mime, context)
            result.data["vision_issues"] = list(extraction.issues)
            size = 0
            for item in extraction.elements:
                if item.text:
                    if size + len(item.text) > MAX_EXTRACT_CHARS:
                        result.status = OutcomeStatus.PARTIAL
                        result.data["vision_truncated"] = True
                        break
                    size += len(item.text)
                    result.evidence.append(
                        original_evidence(
                            source_id,
                            item.text,
                            digest,
                            {
                                "page": number,
                                "extraction_method": item.extraction_method,
                                "position": cast(
                                    dict[str, JsonValue],
                                    json_value(item.locator.model_dump()),
                                ),
                            },
                        )
                    )
        context.check_active()
        return result

    def table(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        source_id = str(args["source_id"])
        content, mime, _ = read_verified_original(broker, source_id, context)
        digest = sha256(content).hexdigest()
        page_number = args.get("page")
        if mime == "application/pdf":
            if page_number is None:
                return ToolOutcome(
                    status=OutcomeStatus.INVALID,
                    summary="Select a PDF page to bound table extraction.",
                )
            if broker.vision_llm is None:
                return ToolOutcome(
                    status=OutcomeStatus.UNAVAILABLE,
                    summary="PDF table reconstruction requires a configured vision model; inspect native source page meanwhile.",
                )
            payload = selected_pdf(content, (int(cast(int, page_number)),))
            extraction = extract_source_vision(broker, payload, mime, context)
        elif mime in {"text/csv", "application/csv"}:
            text = content.decode("utf-8-sig")
            rows = []
            evidence = []
            first = int(cast(int, args.get("start_row", 1)))
            limit = int(cast(int, args.get("limit", 50)))
            reader = csv.reader(StringIO(text))
            header = next(reader, [])
            more = False
            for number, cells in enumerate(reader, 2):
                if number < first:
                    continue
                if len(rows) >= limit:
                    more = True
                    break
                rows.append({"row": number, "cells": cells})
                evidence.append(
                    original_evidence(
                        source_id,
                        " | ".join(cells),
                        digest,
                        {"row": number, "headers": header},
                    )
                )
            return ToolOutcome(
                status=OutcomeStatus.PARTIAL if more else OutcomeStatus.FOUND,
                summary="Native CSV rows with original row numbers and header context.",
                data={"header": header, "rows": rows, "has_more": more},
                evidence=evidence,
            )
        else:
            extraction = extract_annex_structure(
                content, mime, vision_deadline=context.deadline
            )
        sheet = args.get("sheet")
        first = int(cast(int, args.get("start_row", 1)))
        limit = int(cast(int, args.get("limit", 50)))
        headers = [
            item
            for item in extraction.elements
            if item.kind == "table_row"
            and item.locator.row == 1
            and (sheet is None or item.locator.sheet == sheet)
        ]
        selected = [
            item
            for item in extraction.elements
            if item.kind in {"table_row", "table_cell", "footnote"}
            and (sheet is None or item.locator.sheet == sheet)
            and (item.locator.row is None or item.locator.row >= first)
        ]
        evidence = []
        size = 0
        rows: list[JsonValue] = []
        for item in selected[:limit]:
            if size + len(item.text) > MAX_EXTRACT_CHARS:
                break
            locator = cast(dict[str, JsonValue], json_value(item.locator.model_dump()))
            if page_number is not None:
                locator["page"] = cast(JsonValue, page_number)
            evidence.append(original_evidence(source_id, item.text, digest, locator))
            rows.append(
                {
                    "text": item.text,
                    "kind": item.kind,
                    "locator": locator,
                    "extraction_method": item.extraction_method,
                    "issues": list(item.issues),
                }
            )
            size += len(item.text)
        partial = (
            len(evidence) < len(selected)
            or bool(extraction.issues)
            or any(item.issues for item in selected[:limit])
        )
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL
            if partial
            else OutcomeStatus.FOUND
            if evidence
            else OutcomeStatus.NOT_FOUND,
            summary="Source-located table elements; extraction issues remain explicit.",
            data={
                "elements": rows,
                "headers": [
                    {
                        "text": item.text,
                        "locator": json_value(item.locator.model_dump()),
                    }
                    for item in headers[:20]
                ],
                "issues": list(extraction.issues),
                "source_sha256": digest,
                "derived": True,
                "has_more": len(evidence) < len(selected),
            },
            evidence=evidence,
        )

    return [
        ToolSpec(
            name="open_source_file",
            description="Verify authorized current original bytes; dated/version-invalid originals are denied. Read text ranges or obtain a source manifest.",
            parameters=schema(
                {
                    "source_id": SOURCE_FIELD,
                    "start": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 64000},
                },
                ["source_id"],
            ),
            handler=guarded(bounded_source_operation(open_file)),
        ),
        ToolSpec(
            name="inspect_source_page",
            description="Render one verified PDF/image page with native text and image; optional vision must be explicitly configured.",
            parameters=schema(
                {
                    "source_id": SOURCE_FIELD,
                    "page": {"type": "integer", "minimum": 1},
                    "vision": {"type": "boolean"},
                },
                ["source_id", "page"],
            ),
            handler=guarded(bounded_source_operation(page)),
        ),
        ToolSpec(
            name="extract_table",
            description="Extract source-located CSV/XLSX/HTML tables or one PDF page; preserve extraction uncertainty.",
            parameters=schema(
                {
                    "source_id": SOURCE_FIELD,
                    "page": {"type": "integer", "minimum": 1},
                    "sheet": {"type": "string"},
                    "start_row": {"type": "integer", "minimum": 1},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                ["source_id"],
            ),
            handler=guarded(bounded_source_operation(table)),
        ),
    ]
