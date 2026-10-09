"""Resolve source-reading ordinals only through the actual delivered witness catalogue."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated

from pydantic import Field, JsonValue, ValidationError

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.witnesses import original_witness_spans
from onyx.legal_composite.acquisition import InvalidSourceAction
from onyx.legal_composite.models import (
    GapResolution,
    IssueResearchStep,
    MaterialDependencyRequest,
    SourceAction,
    SpanSupport,
    StrictModel,
)
from onyx.legal_composite.requirements import canonicalize_source_support

PositiveInteger = Annotated[int, Field(gt=0, strict=True)]


class ReadingSupport(StrictModel):
    citation: PositiveInteger
    span_number: PositiveInteger


class ReadingRequirement(StrictModel):
    supersedes_requirement_ids: list[str] = Field(default_factory=list)
    requirement_id: str = Field(min_length=1)
    need_id: str = Field(min_length=1)
    dimension: str = Field(min_length=1)
    rule: str = Field(min_length=1)
    application: str = Field(min_length=1)
    supports: list[ReadingSupport] = Field(min_length=1)
    missing_user_facts: list[str] = Field(default_factory=list)


class IssueReadingResponse(StrictModel):
    actions: list[SourceAction]
    ready_to_answer: bool
    remaining_gaps: list[str]
    related_citations: list[int] = Field(default_factory=list)
    reconsider_citations: list[PositiveInteger] = Field(default_factory=list)
    gap_resolutions: list[GapResolution] = Field(default_factory=list)
    requirements: list[ReadingRequirement] = Field(default_factory=list)
    material_dependencies: list[MaterialDependencyRequest] = Field(default_factory=list)
    issue_gaps: dict[str, list[str]] = Field(default_factory=dict)


@dataclass(frozen=True)
class ReadingWitness:
    citation: int
    span_number: int
    source_id: str
    chunk_id: str | None
    text_hash: str
    witness_id: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class ReadingWitnessManifest:
    call_id: str
    witnesses: tuple[ReadingWitness, ...]


def _invalid_catalogue() -> InvalidSourceAction:
    return InvalidSourceAction("Reading witness catalogue is not an exact original")


def number_reading_witnesses(records: list[JsonValue]) -> list[JsonValue]:
    """Add local ordinals without changing originals or interpreting literal metadata."""
    numbered = deepcopy(records)
    for record in numbered:
        if not isinstance(record, dict):
            raise _invalid_catalogue()
        spans = record.get("witness_spans")
        if not isinstance(spans, list):
            raise _invalid_catalogue()
        for number, span in enumerate(spans, 1):
            if not isinstance(span, dict):
                raise _invalid_catalogue()
            existing = span.get("span_number")
            if "span_number" in span and (
                type(existing) is not int or existing != number
            ):
                raise _invalid_catalogue()
            span["span_number"] = number
    return numbered


def reading_witness_manifest(
    records: Iterable[dict[str, JsonValue]], *, call_id: str
) -> ReadingWitnessManifest:
    """Freeze only fitted, complete originals and their exact numbered selectors."""
    if not isinstance(call_id, str) or not call_id:
        raise _invalid_catalogue()
    witnesses: list[ReadingWitness] = []
    seen: set[int] = set()
    for record in records:
        citation = record.get("citation")
        source_id, chunk_id = record.get("source_id"), record.get("chunk_id")
        text, text_hash = record.get("text"), record.get("text_hash")
        spans = record.get("witness_spans")
        if (
            type(citation) is not int
            or citation <= 0
            or citation in seen
            or not isinstance(source_id, str)
            or not source_id
            or (chunk_id is not None and not isinstance(chunk_id, str))
            or not isinstance(text, str)
            or not text
            or not isinstance(text_hash, str)
            or hashlib.sha256(text.encode()).hexdigest() != text_hash
            or record.get("truncated") is not False
            or record.get("start_char", 0) != 0
            or not isinstance(spans, list)
        ):
            raise _invalid_catalogue()
        seen.add(citation)
        expected = original_witness_spans(citation, text)
        if len(spans) != len(expected):
            raise _invalid_catalogue()
        for number, (span, original) in enumerate(zip(spans, expected), 1):
            if not isinstance(span, dict) or any(
                span.get(key) != original[key]
                for key in ("witness_id", "start_char", "end_char")
            ):
                raise _invalid_catalogue()
            if (
                type(span.get("span_number")) is not int
                or span["span_number"] != number
                or type(span.get("start_char")) is not int
                or type(span.get("end_char")) is not int
            ):
                raise _invalid_catalogue()
            witnesses.append(
                ReadingWitness(
                    citation=citation,
                    span_number=number,
                    source_id=source_id,
                    chunk_id=chunk_id,
                    text_hash=text_hash,
                    witness_id=original["witness_id"],
                    start_char=original["start_char"],
                    end_char=original["end_char"],
                )
            )
    return ReadingWitnessManifest(call_id=call_id, witnesses=tuple(witnesses))


def resolve_issue_reading(
    value: IssueReadingResponse,
    ledger: EvidenceLedger,
    delivered: set[int],
    manifest: ReadingWitnessManifest,
    *,
    call_id: str,
) -> IssueResearchStep:
    """Translate provided ordinals to existing supports without admitting legal truth."""
    try:
        value = IssueReadingResponse.model_validate(
            value.model_dump(mode="python"), strict=True
        )
    except ValidationError as error:
        raise InvalidSourceAction(
            "Reading response failed the transport schema"
        ) from error
    if not call_id or call_id != manifest.call_id:
        raise InvalidSourceAction(
            "Reading witness manifest does not match the active delivery"
        )
    index = {(entry.citation, entry.span_number): entry for entry in manifest.witnesses}
    if len(index) != len(manifest.witnesses):
        raise _invalid_catalogue()
    actual_delivery = ledger.completely_delivered(call_id)
    converted = value.model_dump(mode="python")
    converted_requirements = []
    for requirement in value.requirements:
        supports: list[SpanSupport] = []
        for support in requirement.supports:
            entry = index.get((support.citation, support.span_number))
            if (
                entry is None
                or support.citation not in delivered
                or support.citation not in actual_delivery
            ):
                raise InvalidSourceAction(
                    "Reading support does not identify a provided delivered witness"
                )
            item = ledger.get(support.citation)
            if (
                item is None
                or item.source_id != entry.source_id
                or item.chunk_id != entry.chunk_id
                or item.text_hash != entry.text_hash
                or hashlib.sha256(item.text.encode()).hexdigest() != entry.text_hash
            ):
                raise InvalidSourceAction(
                    "Reading witness manifest no longer matches its original"
                )
            expected = original_witness_spans(support.citation, item.text)
            if type(entry.span_number) is not int or not 1 <= entry.span_number <= len(
                expected
            ):
                raise _invalid_catalogue()
            original = expected[entry.span_number - 1]
            if (
                entry.witness_id != original["witness_id"]
                or entry.start_char != original["start_char"]
                or entry.end_char != original["end_char"]
            ):
                raise _invalid_catalogue()
            supports.append(
                canonicalize_source_support(
                    SpanSupport(citation=support.citation, span_id=entry.witness_id),
                    ledger,
                    delivered,
                )
            )
        fields = requirement.model_dump(mode="python")
        fields["supports"] = [support.model_dump(mode="python") for support in supports]
        converted_requirements.append(fields)
    converted["requirements"] = converted_requirements
    return IssueResearchStep.model_validate(converted, strict=True)
