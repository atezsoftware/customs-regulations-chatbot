"""Read-only article identity from an actual contiguous canonical source read."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from onyx.asv3.authority import folded
from onyx.asv3.canonical_authority import (
    _AMENDING_PASSAGE,
    _OWN_ARTICLE_OPENING,
    ArticleParagraphBinding,
    _own_article_witness,
    canonical_identity_ledger,
    canonical_named_authority_gap,
)
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    ToolReceipt,
    model_evidence_metadata,
)
from onyx.regulatory.heading_path import parse_regulatory_article_heading

_FIELD = "asv3_ordered_source_read"
_METHODS = {"read_source_range", "read_provision", "read_named_provision"}
_PARAGRAPH = re.compile(r"^\s*(?:\((\d{1,3})\)|(\d{1,3})\.)\s+\S")
_ACQUISITION_FIELDS = {
    "retrieval_method",
    "section_context",
    "additional_context",
    "article_closure_complete",
    "article_closure_continuation",
    "article_closure_remaining_count",
    "source_outline_truncated",
    "follow_context_tool",
    "retrieved_projection_ordinal",
    "retrieved_center",
}


def without_ordered_source_read_data(data: dict[str, JsonValue]) -> None:
    """Remove host proofs only from tool data and known composed outcome data."""
    data.pop(_FIELD, None)
    for key in ("steps", "results"):
        outcomes = data.get(key)
        rows = (
            list(outcomes.values())
            if isinstance(outcomes, dict)
            else outcomes
            if isinstance(outcomes, list)
            else []
        )
        for outcome in rows:
            nested = outcome.get("data") if isinstance(outcome, dict) else None
            if isinstance(nested, dict):
                without_ordered_source_read_data(nested)


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _metadata_hash(metadata: dict[str, JsonValue]) -> str:
    return _hash({k: v for k, v in metadata.items() if k not in _ACQUISITION_FIELDS})


class _Member(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    chunk_id: str = Field(min_length=1)
    text_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    position: int = Field(ge=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    metadata_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class _Read(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    version: int
    run_id: str
    scope_hash: str
    source_id: str
    members: list[_Member]


class _RecordedRead(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    call_id: str = Field(min_length=1)
    method: str
    read: _Read
    citations: list[int]
    digest: str


def ordered_source_read_data(
    evidence: list[EvidenceItem], context: RunContext
) -> dict[str, JsonValue]:
    """Emit positional provenance only from the canonical corpus handler."""
    state = context.services.get("tuned_article_bindings")
    if not isinstance(state, ArticleBindings) or not evidence:
        return {}
    state._check(context)
    members: list[_Member] = []
    source_id = evidence[0].source_id
    for item in evidence:
        position = item.metadata.get("position")
        if (
            item.source_id != source_id
            or not item.chunk_id
            or type(position) is not int
        ):
            raise ValueError("Canonical ordered source read lacks its actual position")
        members.append(
            _Member(
                chunk_id=item.chunk_id,
                text_hash=item.text_hash,
                position=position,
                start_char=0,
                end_char=len(item.text),
                metadata_hash=_metadata_hash(item.metadata),
            )
        )
    read = _Read(
        version=1,
        run_id=context.run_id,
        scope_hash=_hash(context.scope),
        source_id=source_id,
        members=members,
    )
    body = read.model_dump(mode="json")
    return {_FIELD: {**body, "digest": _hash(body)}}


class ArticleBindings:
    def __init__(self, context: RunContext, request: str) -> None:
        self._run_id = context.run_id
        self._scope_hash = _hash(context.scope)
        self._request_hash = _hash(request)
        self._reads: dict[str, _RecordedRead] = {}
        self._lock = threading.RLock()
        self._check(context)

    def _check(self, context: RunContext) -> None:
        if (
            context.services.get("asv3_workflow_variant") != "asv3_tuned"
            or context.run_id != self._run_id
            or _hash(context.scope) != self._scope_hash
        ):
            raise ValueError("Foreign Experimental article-binding context")

    def _validate(
        self, record: _RecordedRead, ledger: EvidenceLedger
    ) -> list[EvidenceItem]:
        read = record.read
        if (
            type(read.version) is not int
            or read.version != 1
            or read.run_id != self._run_id
            or read.scope_hash != self._scope_hash
            or record.method not in _METHODS
            or not read.members
            or len(record.citations) != len(read.members)
            or len(set(record.citations)) != len(record.citations)
            or any(type(n) is not int or n <= 0 for n in record.citations)
            or record.digest != _hash(read.model_dump(mode="json"))
        ):
            raise ValueError("Invalid Experimental ordered source witness")
        items: list[EvidenceItem] = []
        for number, member in zip(record.citations, read.members, strict=True):
            item = ledger.get(number)
            metadata = model_evidence_metadata(item.metadata) if item else {}
            canonical = item.metadata.get("canonical_metadata") if item else None
            if (
                item is None
                or item.source_id != read.source_id
                or item.chunk_id != member.chunk_id
                or item.text_hash != member.text_hash
                or hashlib.sha256(item.text.encode()).hexdigest() != member.text_hash
                or type(item.metadata.get("position")) is not int
                or item.metadata["position"] != member.position
                or member.start_char != 0
                or member.end_char != len(item.text)
                or _metadata_hash(item.metadata) != member.metadata_hash
                or item.search_doc is None
                or item.search_doc.document_id != read.source_id
                or item.search_doc.metadata.get("regulatory_chunk_id")
                != member.chunk_id
                or item.metadata.get("truncated") is True
                or (isinstance(canonical, dict) and canonical.get("truncated") is True)
                or any(
                    metadata.get(k)
                    for k in ("truncated", "untrusted", "derived", "external")
                )
            ):
                raise ValueError(
                    "Ordered source witness does not match its canonical original"
                )
            items.append(item)
        return items

    def _bindings(
        self, record: _RecordedRead, ledger: EvidenceLedger
    ) -> dict[int, ArticleParagraphBinding]:
        items = self._validate(record, ledger)
        rows = {row["citation"]: row for row in ledger.authority_metadata()}
        result: dict[int, ArticleParagraphBinding] = {}
        active = None
        previous_position = None
        previous_paragraph = None
        for number, member, item in zip(
            record.citations, record.read.members, items, strict=True
        ):
            if (
                previous_position is not None
                and member.position != previous_position + 1
            ):
                active = None
                previous_paragraph = None
            previous_position = member.position
            row = rows[number]
            own = _own_article_witness(number, item, row)
            stripped = item.text.lstrip()
            if own is not None:
                active = own
                opening = _OWN_ARTICLE_OPENING.match(stripped)
                assert opening is not None
                marker = _PARAGRAPH.match(
                    stripped[opening.end("heading") :].lstrip(".-–—: ")
                )
            else:
                marker = _PARAGRAPH.match(stripped)
                if any(
                    parse_regulatory_article_heading(line.replace("**", ""))
                    for line in stripped.splitlines()
                ):
                    active = None
            metadata = model_evidence_metadata(item.metadata)
            if (
                active is None
                or marker is None
                or _AMENDING_PASSAGE.search(folded(item.text))
                or metadata.get("document_type")
                not in {"kanun", "law", "act", "statute"}
            ):
                active = None
                previous_paragraph = None
                continue
            paragraph = marker[1] or marker[2]
            identifiers = metadata.get("provision_identifiers")
            declared_articles = (
                [
                    parsed
                    for value in identifiers
                    if isinstance(value, str)
                    if (parsed := parse_regulatory_article_heading(value)) is not None
                ]
                if isinstance(identifiers, list)
                else []
            )
            if (
                metadata.get("paragraph_no") not in (None, paragraph)
                or row.get("article_no") not in (None, active.article_no)
                or (
                    row.get("article_no") is not None
                    and row.get("article_qualifier") != active.qualifier
                )
                or any(
                    (parsed.article_no, parsed.qualifier)
                    != (active.article_no, active.qualifier)
                    for parsed in declared_articles
                )
                or (
                    previous_paragraph is not None
                    and own is None
                    and int(paragraph) != previous_paragraph + 1
                )
            ):
                active = None
                previous_paragraph = None
                continue
            previous_paragraph = int(paragraph)
            result[number] = ArticleParagraphBinding(
                number,
                item.source_id,
                member.chunk_id,
                item.text_hash,
                active.article_no,
                active.qualifier,
                paragraph,
            )
        return result

    def record_receipt(
        self, receipt: ToolReceipt, context: RunContext, ledger: EvidenceLedger
    ) -> int:
        self._check(context)
        if receipt.call.name not in _METHODS:
            return 0
        raw = receipt.outcome.data.get(_FIELD)
        if raw is None:
            return 0
        if receipt.outcome.status not in {OutcomeStatus.FOUND, OutcomeStatus.PARTIAL}:
            raise ValueError("Ordered source witness is not a canonical read receipt")
        if not isinstance(raw, dict):
            raise ValueError("Invalid ordered source witness")
        digest = raw.get("digest")
        if not isinstance(digest, str):
            raise ValueError("Ordered source witness has no digest")
        read = _Read.model_validate({k: v for k, v in raw.items() if k != "digest"})
        record = _RecordedRead(
            call_id=receipt.call.call_id,
            method=receipt.call.name,
            read=read,
            citations=receipt.evidence_ids,
            digest=digest,
        )
        actual_source = receipt.call.arguments.get(
            "source_id"
        ) or receipt.outcome.data.get("source_id")
        if actual_source != read.source_id:
            raise ValueError("Ordered source witness crosses the requested source")
        with self._lock:
            bindings = self._bindings(record, ledger)
            existing = self._reads.get(record.call_id)
            if existing is not None and existing != record:
                raise ValueError("Changed ordered source receipt")
            self._reads[record.call_id] = record
            return len(bindings)

    def consume_receipt(
        self, receipt: ToolReceipt, context: RunContext, ledger: EvidenceLedger
    ) -> None:
        """Retain validated host provenance without adding it to model context."""
        self.record_receipt(receipt, context, ledger)
        without_ordered_source_read_data(receipt.outcome.data)

    def identity_bindings(
        self, context: RunContext, ledger: EvidenceLedger
    ) -> dict[int, ArticleParagraphBinding]:
        self._check(context)
        with self._lock:
            result: dict[int, ArticleParagraphBinding] = {}
            for read in self._reads.values():
                for number, binding in self._bindings(read, ledger).items():
                    if number in result and result[number] != binding:
                        raise ValueError("Conflicting ordered article identity")
                    result[number] = binding
            return result

    def export(self) -> dict[str, JsonValue]:
        with self._lock:
            return {
                "version": 1,
                "run_id": self._run_id,
                "scope_hash": self._scope_hash,
                "request_hash": self._request_hash,
                "reads": [r.model_dump(mode="json") for r in self._reads.values()],
            }

    def restore(
        self, saved: dict[str, JsonValue], context: RunContext, ledger: EvidenceLedger
    ) -> None:
        self._check(context)
        if (
            set(saved) != {"version", "run_id", "scope_hash", "request_hash", "reads"}
            or type(saved.get("version")) is not int
            or saved["version"] != 1
            or saved.get("run_id") != self._run_id
            or saved.get("scope_hash") != self._scope_hash
            or saved.get("request_hash") != self._request_hash
            or not isinstance(saved.get("reads"), list)
        ):
            raise ValueError("Foreign or invalid article-binding checkpoint")
        restored: dict[str, _RecordedRead] = {}
        for raw in cast(list[JsonValue], saved["reads"]):
            record = _RecordedRead.model_validate(raw)
            self._bindings(record, ledger)
            if record.call_id in restored:
                raise ValueError("Duplicate ordered source receipt")
            restored[record.call_id] = record
        with self._lock:
            self._reads = restored


def bound_identity_ledger(
    ledger: EvidenceLedger, context: RunContext
) -> EvidenceLedger:
    state = context.services.get("tuned_article_bindings")
    bindings = (
        state.identity_bindings(context, ledger)
        if isinstance(state, ArticleBindings)
        else None
    )
    return canonical_identity_ledger(ledger, article_bindings=bindings)


def bound_named_authority_gap(
    answer: str,
    ledger: EvidenceLedger,
    context: RunContext,
    *,
    strict_reference_boundaries: bool = False,
    syntactic_reference_binding: bool = False,
    resolve_defined_abbreviations: bool = False,
    strict_numeric_paragraphs: bool = False,
) -> dict[str, JsonValue] | None:
    return canonical_named_authority_gap(
        answer,
        bound_identity_ledger(ledger, context),
        strict_reference_boundaries=strict_reference_boundaries,
        syntactic_reference_binding=syntactic_reference_binding,
        resolve_defined_abbreviations=resolve_defined_abbreviations,
        strict_numeric_paragraphs=strict_numeric_paragraphs,
    )
