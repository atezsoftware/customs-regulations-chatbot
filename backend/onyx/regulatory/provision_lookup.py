"""Bounded structural reads from the authorized, published search index."""

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from onyx.context.search.models import IndexFilters, InferenceChunk
from onyx.document_index.interfaces_new import DocumentIndex
from onyx.regulatory.article_scope import SourceFragment, article_scope_indices
from onyx.regulatory.provision_identity import article_identity
from onyx.regulatory.source_identity import (
    named_law_number,
    source_identity_distinguishing_tokens,
    source_identity_matches,
)

MAX_CANDIDATES = 128
MAX_RESULT_CHUNKS = 24
MAX_RESULT_CHARS = 48000
_OPENING = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*{1,2})?\s*(?:(?:ek|geçici|gecici|mükerrer|mukerrer)\s+)?madde\s+\d+[a-z]?(?=\s*[-–—:.])",
    re.IGNORECASE,
)
_PARAGRAPH = re.compile(r"^(?P<indent>[ \t]*)\((?P<number>\d+)\)(?:\s|$)", re.MULTILINE)
_CLAUSE = re.compile(r"^\s*\(?([a-zçğıöşü])\)(?:\s|$)", re.MULTILINE | re.IGNORECASE)
_QUOTED = re.compile(r'"[^"]*"|“[^”]*”', re.DOTALL)


class ProvisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    source: str = Field(
        min_length=3,
        max_length=400,
        description="Instrument title and/or official number, with year/series when known. Do not infer missing identifiers.",
    )
    article_number: str = Field(
        pattern=r"^\d{1,5}[a-zA-Z]?$",
        description="Article number, without the MADDE prefix.",
    )
    article_kind: Literal["regular", "additional", "temporary", "repeated"] = "regular"
    paragraph: int | None = Field(default=None, ge=1, le=9999)
    clause: str | None = Field(default=None, pattern=r"^[a-zçğıöşü]$")
    subclause: str | None = Field(default=None, pattern=r"^[0-9ivxlcdm]+$")
    as_of_date: date | None = None

    @property
    def heading(self) -> str:
        prefix = {
            "regular": "",
            "additional": "EK ",
            "temporary": "GEÇİCİ ",
            "repeated": "MÜKERRER ",
        }[self.article_kind]
        return f"{prefix}MADDE {self.article_number.upper()}"

    @property
    def label(self) -> str:
        return " / ".join(
            str(v)
            for v in (
                self.source,
                self.heading,
                self.paragraph,
                self.clause,
                self.subclause,
            )
            if v is not None
        )


LookupStatus = Literal[
    "found",
    "partial",
    "ambiguous_source",
    "ambiguous_unit",
    "not_found_in_scope",
    "unavailable",
    "invalid_reference",
]


@dataclass
class ProvisionLookupResult:
    status: LookupStatus
    detail: str
    chunks: list[InferenceChunk] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)
    as_of_date: date | None = None


def _source_matches(source: str, row: InferenceChunk) -> bool:
    # Never verify source identity against article text or descendant headings.
    names = [row.title or row.semantic_identifier.split(" — ", 1)[0]]
    root = (row.heading_path or [""])[0]
    if root and article_identity(root) is None:
        names.append(root)
    requested_number = named_law_number(source)
    numbers = {number for name in names if (number := named_law_number(name))}
    if requested_number and numbers and numbers != {requested_number}:
        return False
    tokens = source_identity_distinguishing_tokens(source)
    if requested_number or tokens:
        title_tokens = set(tokens) - {requested_number}
        return any(
            source_identity_matches(source, name)
            and title_tokens <= set(source_identity_distinguishing_tokens(name))
            for name in names
        )
    # Generic titles still need actual identity; the update helper deliberately
    # treats an empty distinguishing-token set as unconstrained.
    return any(source.casefold() == name.casefold() for name in names)


def _article_path(row: InferenceChunk) -> tuple[str | None, list[str]]:
    path = row.heading_path or []
    for i, heading in enumerate(path):
        if re.match(
            r"^\s*(?:(?:ek|geçici|gecici|mükerrer|mukerrer)\s+)?madde\s+\d",
            heading,
            re.IGNORECASE,
        ):
            return article_identity(heading), path[i + 1 :]
    return None, []


def _unit_status(request: ProvisionRequest, rows: list[InferenceChunk]) -> LookupStatus:
    # Validate a subunit across the ordered canonical fragments; selecting each
    # fragment independently drops unnumbered continuations.
    text = "\n".join(row.content for row in rows)
    text = _QUOTED.sub(lambda match: re.sub(r"[^\n]", " ", match.group()), text)
    opening = _OPENING.match(text)
    if opening:
        text = text[opening.end() :].lstrip("-–—:.* \n")
    if request.paragraph is not None:
        markers = list(_PARAGRAPH.finditer(text))
        if markers:
            indent = min(len(marker.group("indent").expandtabs()) for marker in markers)
            markers = [
                marker
                for marker in markers
                if len(marker.group("indent").expandtabs()) == indent
            ]
        matches = [
            i
            for i, marker in enumerate(markers)
            if int(marker.group("number")) == request.paragraph
        ]
        if len(matches) > 1:
            return "ambiguous_unit"
        if not matches:
            return "partial"
        index = matches[0]
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        text = text[markers[index].end() : end]
    if request.clause:
        markers = list(_CLAUSE.finditer(text))
        matches = [
            i
            for i, marker in enumerate(markers)
            if marker.group(1).casefold() == request.clause
        ]
        if len(matches) > 1:
            return "ambiguous_unit"
        if not matches:
            return "partial"
        index = matches[0]
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        # A later paragraph is a boundary even when no paragraph was requested.
        following_paragraph = _PARAGRAPH.search(text, markers[index].end())
        if following_paragraph:
            end = min(end, following_paragraph.start())
        text = text[markers[index].end() : end]
    if request.subclause:
        if not request.clause:
            return "invalid_reference"
        matches = list(
            re.finditer(
                rf"^\s*\(?{re.escape(request.subclause)}[).]\s", text, re.MULTILINE
            )
        )
        if len(matches) != 1:
            return "ambiguous_unit" if matches else "partial"
    return "found"


def lookup_provision(
    request: ProvisionRequest, *, document_index: DocumentIndex, filters: IndexFilters
) -> ProvisionLookupResult:
    if (
        request.as_of_date
        and filters.as_of_date
        and request.as_of_date != filters.as_of_date
    ):
        return ProvisionLookupResult(
            "invalid_reference",
            "Requested date conflicts with the active search scope.",
            as_of_date=filters.as_of_date,
        )
    effective_date = filters.as_of_date or request.as_of_date or date.today()
    source_hint = named_law_number(request.source) or request.source
    scoped = filters.model_copy(
        update={
            "regulatory_chunks_only": True,
            "regulatory_source_hint": source_hint,
            "regulatory_lookup": True,
            "regulatory_lookup_heading": request.heading,
            "as_of_date": effective_date,
        }
    )
    rows = document_index.keyword_retrieval(
        query=request.heading, filters=scoped, num_to_retrieve=MAX_CANDIDATES + 1
    )
    exhausted = len(rows) <= MAX_CANDIDATES
    verified = [
        row
        for row in rows[:MAX_CANDIDATES]
        if row.regulatory_chunk_id and _source_matches(request.source, row)
    ]
    sources = {
        row.document_id: {
            "source_id": row.document_id,
            "title": row.title or row.semantic_identifier,
        }
        for row in verified
    }
    if len(sources) > 1:
        return ProvisionLookupResult(
            "ambiguous_source",
            "Several authorized sources match; distinguish title, year or official number before answering.",
            sources=list(sources.values())[:8],
            as_of_date=effective_date,
        )
    identity = article_identity(request.heading)
    selected = []
    for row in verified:
        heading_identity, _ = _article_path(row)
        opening = _OPENING.match(row.content)
        actual_identity = article_identity(opening.group()) if opening else None
        if actual_identity and heading_identity and actual_identity != heading_identity:
            continue
        if heading_identity == identity or actual_identity == identity:
            selected.append(row)
    # Check the entire bounded source even if one heading hit was found;
    # siblings with stale/missing headings would otherwise be silently omitted.
    # Legacy resolution requires a complete bounded source, never a truncated window.
    source_rows = document_index.keyword_retrieval(
        query=request.source,
        filters=scoped.model_copy(update={"regulatory_lookup_heading": None}),
        num_to_retrieve=MAX_CANDIDATES + 1,
    )
    source_complete = len(source_rows) <= MAX_CANDIDATES
    source_rows = [
        row
        for row in source_rows[:MAX_CANDIDATES]
        if row.regulatory_chunk_id and _source_matches(request.source, row)
    ]
    sources = {
        row.document_id: {
            "source_id": row.document_id,
            "title": row.title or row.semantic_identifier,
        }
        for row in source_rows
    }
    if len(sources) > 1:
        return ProvisionLookupResult(
            "ambiguous_source",
            "Several authorized sources match; distinguish title, year or official number.",
            sources=list(sources.values())[:8],
            as_of_date=effective_date,
        )
    if source_complete and identity is not None:
        source_rows.sort(
            key=lambda row: (
                row.structural_position
                if row.structural_position is not None
                else row.chunk_id,
                row.chunk_id,
            )
        )
        indices = article_scope_indices(
            [SourceFragment(row.document_id, row.content) for row in source_rows],
            identity,
            stop_at_annex=True,
        )
        if indices:
            selected = [source_rows[index] for index in indices]
        else:
            # A heading-only result remains usable context, but no completeness
            # claim is made if the actual article boundaries are not established.
            exhausted = False
    else:
        exhausted = False
    if not selected:
        return ProvisionLookupResult(
            "not_found_in_scope",
            "No verified unit in the searched scope. Other internal search modes may find additional evidence; this is not proof of absence.",
            as_of_date=effective_date,
        )
    unique = {row.regulatory_chunk_id: row for row in selected}
    selected = sorted(
        unique.values(),
        key=lambda row: (
            row.structural_position
            if row.structural_position is not None
            else row.chunk_id,
            row.chunk_id,
        ),
    )
    status = _unit_status(request, selected)
    detail = "Verified source and structural unit; article context retains canonical citation identities."
    if status == "partial":
        detail = "The article was found, but the requested subunit cannot be verified. Results are article context, not an exact subunit match."
    elif status == "ambiguous_unit":
        return ProvisionLookupResult(
            status,
            "The requested unit occurs more than once. Narrow the paragraph or clause; do not select an occurrence by guesswork.",
            sources=list(sources.values()),
            as_of_date=effective_date,
        )
    elif status == "invalid_reference":
        return ProvisionLookupResult(
            status, "A subclause requires its parent clause.", as_of_date=effective_date
        )
    bounded = []
    chars = 0
    for row in selected:
        if (
            len(bounded) >= MAX_RESULT_CHUNKS
            or chars + len(row.content) > MAX_RESULT_CHARS
        ):
            exhausted = False
            break
        bounded.append(row)
        chars += len(row.content)
    if not exhausted:
        status = "partial"
        detail = "The bounded lookup could not establish complete coverage. Use other internal search or narrow the reference; do not present this as the complete unit."
    return ProvisionLookupResult(
        status, detail, bounded, list(sources.values()), effective_date
    )
