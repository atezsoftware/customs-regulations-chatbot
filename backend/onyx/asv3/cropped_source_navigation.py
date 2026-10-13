"""Prospective structural navigation for delegated Cropped source pagination."""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Iterator
from typing import Protocol, cast

from pydantic import JsonValue

from onyx.asv3.cropped_research_limit import CroppedResearchLimit
from onyx.asv3.models import OutcomeStatus, RunContext, ToolOutcome
from onyx.db.asv3_corpus import CorpusChunk, CorpusSource
from onyx.regulatory.heading_path import parse_regulatory_article_heading


class SourceNavigationScan(Protocol):
    source: CorpusSource
    truncated: bool
    next_position: int

    def __iter__(self) -> Iterator[CorpusChunk]: ...


def _kind(chunk: CorpusChunk) -> str | None:
    value = chunk.metadata.get("document_type")
    if not isinstance(value, str):
        return None
    folded = "".join(
        char
        for char in unicodedata.normalize("NFKD", value.casefold().strip())
        if not unicodedata.combining(char)
    ).replace("ı", "i")
    return folded if folded in {"kanun", "yonetmelik", "teblig"} else None


def _heading(chunk: CorpusChunk) -> tuple[tuple[str, ...], str, str | None] | None:
    for index in range(len(chunk.heading_path) - 1, -1, -1):
        parsed = parse_regulatory_article_heading(chunk.heading_path[index])
        if parsed is not None:
            return (
                chunk.heading_path[: index + 1],
                parsed.article_no,
                parsed.qualifier,
            )
    return None


def _dates(chunk: CorpusChunk) -> dict[str, JsonValue]:
    return {
        "validity_start": chunk.validity_start.isoformat()
        if chunk.validity_start
        else None,
        "validity_end": chunk.validity_end.isoformat() if chunk.validity_end else None,
        "status": chunk.status,
    }


def _position(row: dict[str, JsonValue], chunk: CorpusChunk) -> None:
    ranges = cast(list[JsonValue], row["observed_position_ranges"])
    last = cast(list[JsonValue], ranges[-1])
    if chunk.position == last[1]:
        last[1] = chunk.position + 1
    else:
        ranges.append([chunk.position, chunk.position + 1])
    row["chunk_count"] = cast(int, row["chunk_count"]) + 1
    intervals = cast(list[JsonValue], row["observed_validity"])
    dates = _dates(chunk)
    if dates not in intervals:
        intervals.append(dates)


def prospective_cropped_source_navigation(
    *,
    source: CorpusSource,
    page: list[CorpusChunk],
    has_more: bool,
    start: int,
    context: RunContext,
    scan_source: Callable[[], SourceNavigationScan],
) -> ToolOutcome | None:
    """Return unread navigation only; None preserves the ordinary original read."""
    controller = context.services.get("cropped_research_controller")
    if (
        context.services.get("asv3_exp_cropped") is not True
        or context.services.get("asv3_exp_cheap") is True
        or context.services.get("independent_question") is True
        or not isinstance(controller, CroppedResearchLimit)
        or not has_more
        or not page
    ):
        return None
    controller.check_context(context)
    if not controller.has_delegated_research:
        return None
    kind = _kind(page[0])
    if (
        kind is None
        or any(chunk.source_id != source.id for chunk in page)
        or any(_kind(chunk) != kind for chunk in page)
        or not any(_heading(chunk) is not None for chunk in page)
    ):
        return None

    scan = scan_source()
    if scan.source.id != source.id:
        raise PermissionError("Source navigation differs from the authorized source.")
    rows: list[dict[str, JsonValue]] = []
    unowned: list[dict[str, JsonValue]] = []
    current_key: tuple[tuple[str, ...], str, str | None] | None = None
    current: dict[str, JsonValue] | None = None
    previous_position = start - 1
    scanned = 0
    for chunk in scan:
        context.check_active()
        if chunk.source_id != source.id:
            raise PermissionError("Source navigation contains a foreign source.")
        if (
            type(chunk.position) is not int
            or chunk.position < start
            or chunk.position < previous_position
        ):
            raise ValueError("Source navigation positions are not strictly ordered.")
        if chunk.position == previous_position:
            return None
        # An uncertain structural classification keeps the original read available.
        if _kind(chunk) != kind:
            return None
        previous_position = chunk.position
        scanned += 1
        heading = _heading(chunk)
        if heading is not None and heading == current_key and current is not None:
            _position(current, chunk)
            continue
        current_key = heading
        current = {
            "heading_path": list(heading[0] if heading else chunk.heading_path),
            "anchor_chunk_id": chunk.id,
            "observed_position_ranges": [[chunk.position, chunk.position + 1]],
            "chunk_count": 1,
            "observed_validity": [_dates(chunk)],
        }
        if heading is None:
            unowned.append(current)
            current = None
        else:
            current["article"] = heading[1]
            current["qualifier"] = heading[2]
            rows.append(current)
    if not scanned:
        return None
    return ToolOutcome(
        status=OutcomeStatus.PARTIAL,
        summary=(
            "The requested whole-source page was not delivered. Actual source headings "
            "are navigation; choose complete operative provisions and their continuations."
        ),
        data={
            "source_id": str(source.id),
            "source_name": scan.source.name,
            "document_type": kind,
            "requested_start": start,
            "requested_read_executed": False,
            "not_read": True,
            "absence_proven": False,
            "legal_approval": False,
            "source_text_delivered": False,
            "article_closure_complete": False,
            "navigation_only": True,
            "navigation_scan_start": start,
            "navigation_scanned_chunks": scanned,
            "headings": cast(list[JsonValue], rows),
            "unowned_headings": cast(list[JsonValue], unowned),
            "has_more": scan.truncated,
            "next_position": scan.next_position,
            "instruction": (
                "Inspect every article and unowned heading needed for the assigned "
                "outcomes. Read the complete operative provisions with read_provision, "
                "including prerequisites, exceptions, material references and clipped "
                "article continuations. Use owned chunk/context reads for necessary "
                "annex or unowned material. These positions and dates are observed "
                "navigation, not a legal completeness or validity finding. Retain "
                "unresolved conditions; no source text was delivered by this result."
            ),
        },
    )
