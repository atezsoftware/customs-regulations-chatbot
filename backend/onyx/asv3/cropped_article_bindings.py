"""Read-only article identity from an actual contiguous canonical source read."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from onyx.asv3.authority import _native_original_rows, folded
from onyx.asv3.cheap_authority import (
    _AMENDING_PASSAGE,
    _OWN_ARTICLE_OPENING,
    ArticleParagraphBinding,
    _own_article_witness,
    canonical_identity_ledger,
    canonical_named_authority_gap,
)
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    CapabilityCall,
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolReceipt,
    model_evidence_metadata,
)
from onyx.regulatory.heading_path import parse_regulatory_article_heading

_FIELD = "cropped_ordered_source_read"
_METHODS = {
    "read_source_range",
    "read_provision",
    "read_named_provision",
    "read_chunk_context",
    "follow_reference",
}
_PARAGRAPH = re.compile(r"^\s*(?:\((\d{1,3})\)|(\d{1,3})\.)\s+\S")
_CONTINUING_PARAGRAPH = re.compile(r"^\s*\((\d{1,3})\)\s+\S")
_UNNUMBERED_ARTICLE = re.compile(
    r"(?m)^(?P<heading>(?:\*\*)?(?:(?:GEÇİCİ|MÜKERRER|EK)\s+)?MADDE\s+"
    r"\d{1,4}(?:/[A-ZÇĞİÖŞÜ])?\s*\.\s*[-–—](?:\*\*)?)\s+(?P<body>\S)",
    re.IGNORECASE,
)
_STRUCTURAL_MARKER = re.compile(r"(?m)^\s*(?:\(?\d{1,3}[.)]|\(?[a-zçğıöşü]\))\s+")
_SOURCE_RULE = re.compile(r"\A\s*[-–—]{3,}\s*\n")
_INSERTED_ARTICLE_OPENING = re.compile(
    r"\A(?P<heading>\*\*MADDE\s+\d{1,4}/[A-ZÇĞİÖŞÜ]\s*\.?\s*[-–—:]?\*\*)\s+(?P<body>\S)",
    re.IGNORECASE,
)
_INSERTED_CLAUSE = re.compile(r"\A\s*([a-zçğıöşü])\)\s+\S", re.IGNORECASE)
_CLAUSE_ORDER = "abcçdefgğhıijklmnoöprsştuüvyz"


@dataclass(frozen=True)
class _InsertedArticleIdentity:
    article_no: str
    base_article_no: str


def _inserted_article_metadata(
    item: EvidenceItem,
    row: dict[str, JsonValue],
    identity: _InsertedArticleIdentity,
) -> bool:
    metadata = model_evidence_metadata(item.metadata)
    canonical = item.metadata.get("canonical_metadata")
    canonical = canonical if isinstance(canonical, dict) else item.metadata
    identifiers = metadata.get("provision_identifiers")
    allowed = {identity.article_no, identity.base_article_no}
    return bool(
        folded(str(metadata.get("document_type") or ""))
        in {"kanun", "law", "act", "statute"}
        and not any(
            metadata.get(key)
            for key in ("truncated", "untrusted", "derived", "external")
        )
        and not _AMENDING_PASSAGE.search(folded(item.text))
        and not any(
            canonical.get(key) is not None
            for key in (
                "target_article_no",
                "target_id",
                "asv3_navigation_position_authority",
            )
        )
        and row.get("article_no") in allowed
        and row.get("article_qualifier") is None
        and isinstance(identifiers, list)
        and identifiers
        and all(
            isinstance(value, str)
            and (parsed := parse_regulatory_article_heading(value)) is not None
            and parsed.article_no in allowed
            and parsed.qualifier is None
            for value in identifiers
        )
        and not (
            isinstance(original := canonical.get("original_heading_path"), list)
            and any(
                isinstance(value, str)
                and (parsed := parse_regulatory_article_heading(value)) is not None
                and (parsed.article_no not in allowed or parsed.qualifier is not None)
                for value in original
            )
        )
    )


def _inserted_article_anchor(
    item: EvidenceItem, row: dict[str, JsonValue]
) -> _InsertedArticleIdentity | None:
    # A separator is source formatting; preceding operative prose is a boundary.
    text = _SOURCE_RULE.sub("", item.text.lstrip(), count=1)
    opening = _INSERTED_ARTICLE_OPENING.match(text)
    if opening is None:
        return None
    parsed = parse_regulatory_article_heading(opening["heading"].replace("**", ""))
    if parsed is None:
        return None
    identity = _InsertedArticleIdentity(
        parsed.article_no, parsed.article_no.split("/")[0]
    )
    metadata = model_evidence_metadata(item.metadata)
    identifiers = metadata.get("provision_identifiers")
    if (
        not _inserted_article_metadata(item, row, identity)
        or metadata.get("paragraph_no") is not None
        or metadata.get("clause_label") is not None
        or not isinstance(identifiers, list)
        or not any(
            isinstance(value, str)
            and (declared := parse_regulatory_article_heading(value)) is not None
            and declared.article_no == identity.article_no
            and declared.qualifier is None
            for value in identifiers
        )
        or _STRUCTURAL_MARKER.search(text[opening.start("body") :])
        or text[opening.start("body") :].startswith(('"', "“", "'", "‘"))
        or any(
            parse_regulatory_article_heading(line.replace("**", ""))
            for line in text.splitlines()[1:]
        )
    ):
        return None
    return identity


def _unnumbered_article_anchor(
    item: EvidenceItem, row: dict[str, JsonValue]
) -> tuple[str, str | None, int] | None:
    """An actual law's literal operative heading, corroborated by its identifiers."""
    metadata = model_evidence_metadata(item.metadata)
    canonical = item.metadata.get("canonical_metadata")
    canonical = canonical if isinstance(canonical, dict) else item.metadata
    if (
        any(
            metadata.get(key)
            for key in ("truncated", "untrusted", "derived", "external")
        )
        or _AMENDING_PASSAGE.search(folded(item.text))
        or any(
            canonical.get(key) is not None for key in ("target_article_no", "target_id")
        )
    ):
        return None
    headings = list(_UNNUMBERED_ARTICLE.finditer(item.text))
    if not headings:
        return None
    heading = headings[-1]
    parsed = parse_regulatory_article_heading(heading["heading"].replace("**", ""))
    identifiers = metadata.get("provision_identifiers")
    if parsed is None or not isinstance(identifiers, list):
        return None
    declared = {
        (identity.article_no, identity.qualifier)
        for value in identifiers
        if isinstance(value, str)
        if (identity := parse_regulatory_article_heading(value)) is not None
    }
    body = item.text[heading.start("body") :]
    original_path = canonical.get("original_heading_path")
    if (
        declared != {(parsed.article_no, parsed.qualifier)}
        or row.get("article_no") not in (None, parsed.article_no)
        or (
            row.get("article_no") is not None
            and row.get("article_qualifier") != parsed.qualifier
        )
        or _STRUCTURAL_MARKER.search(body)
        or body.startswith(('"', "“", "'", "‘"))
        or not body.rstrip().endswith(".")
        or (
            isinstance(original_path, list)
            and any(
                isinstance(value, str)
                and (identity := parse_regulatory_article_heading(value)) is not None
                and (identity.article_no, identity.qualifier)
                != (parsed.article_no, parsed.qualifier)
                for value in original_path
            )
        )
    ):
        return None
    return parsed.article_no, parsed.qualifier, heading.start()


def _unnumbered_binding(
    number: int, item: EvidenceItem, article: str, qualifier: str | None
) -> ArticleParagraphBinding:
    assert item.chunk_id is not None
    return ArticleParagraphBinding(
        number, item.source_id, item.chunk_id, item.text_hash, article, qualifier, None
    )


def _authority_row_map(ledger: EvidenceLedger) -> dict[int, dict[str, JsonValue]]:
    return {
        number: row
        for row in ledger.authority_metadata()
        if type(number := row.get("citation")) is int
    }


def _source_row_map(ledger: EvidenceLedger) -> dict[int, dict[str, JsonValue]]:
    return {
        number: row
        for row in _native_original_rows(ledger)
        if type(number := row.get("citation")) is int
    }


_ACQUISITION_FIELDS = frozenset(
    {
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
)


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


def _source_metadata_hash(metadata: dict[str, JsonValue]) -> str:
    """Bind source provenance; outer acquisition annotations vary on exact rereads."""
    return _hash(
        {
            key: value
            for key, value in metadata.items()
            if key not in _ACQUISITION_FIELDS
        }
    )


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


@dataclass(frozen=True)
class _ComposedReadOwner:
    state: CroppedArticleBindings
    task_id: str
    depth: int
    call_id: str


_COMPOSED_OWNER: ContextVar[_ComposedReadOwner | None] = ContextVar(
    "cropped_composed_read_owner", default=None
)


@contextmanager
def isolated_cropped_read_capture() -> Iterator[None]:
    """Start a worker without inheriting its caller's compose ownership."""
    token = _COMPOSED_OWNER.set(None)
    try:
        yield
    finally:
        _COMPOSED_OWNER.reset(token)


def _owner(context: RunContext) -> tuple[str, int]:
    return str(context.services.get("task_id") or "root"), context.depth


@contextmanager
def cropped_composed_read_capture(
    call: CapabilityCall, context: RunContext
) -> Iterator[None]:
    """Carry the executed parent call through the composer's copied thread context."""
    state = context.services.get("cropped_article_bindings")
    if (
        call.name != "compose_tool_calls"
        or context.services.get("asv3_exp_cropped") is not True
        or context.services.get("asv3_exp_cheap") is True
        or not isinstance(state, CroppedArticleBindings)
    ):
        yield
        return
    state._check(context)
    task_id, depth = _owner(context)
    token = _COMPOSED_OWNER.set(_ComposedReadOwner(state, task_id, depth, call.call_id))
    try:
        yield
    except BaseException:
        state.discard_composed_read(call, context)
        raise
    finally:
        _COMPOSED_OWNER.reset(token)


def capture_cropped_dispatched_read(
    call: CapabilityCall, outcome: ToolOutcome, context: RunContext
) -> None:
    """Capture only a real completed registry subcall, never aggregate model data."""
    owner = _COMPOSED_OWNER.get()
    if owner is not None:
        if call.name == "compose_tool_calls" and call.call_id == owner.call_id:
            if outcome.status not in {OutcomeStatus.FOUND, OutcomeStatus.PARTIAL}:
                owner.state.discard_composed_read(call, context)
            return
        owner.state.capture_dispatched_read(call, outcome, context, owner)


def ordered_source_read_data(
    evidence: list[EvidenceItem], context: RunContext
) -> dict[str, JsonValue]:
    """Emit positional provenance only from the canonical corpus handler."""
    state = context.services.get("cropped_article_bindings")
    if (
        context.services.get("asv3_exp_cropped") is not True
        or context.services.get("asv3_exp_cheap") is True
        or not isinstance(state, CroppedArticleBindings)
        or not evidence
    ):
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
                metadata_hash=_source_metadata_hash(item.metadata),
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


class CroppedArticleBindings:
    def __init__(self, context: RunContext, request: str) -> None:
        self._run_id = context.run_id
        self._scope_hash = _hash(context.scope)
        self._request_hash = _hash(request)
        self._reads: dict[str, _RecordedRead] = {}
        self._pending: dict[tuple[str, int, str], dict[str, _RecordedRead]] = {}
        self._lock = threading.RLock()
        self._check(context)

    def _check(self, context: RunContext) -> None:
        if (
            context.services.get("asv3_exp_cropped") is not True
            or context.services.get("asv3_exp_cheap") is True
            or context.run_id != self._run_id
            or _hash(context.scope) != self._scope_hash
        ):
            raise ValueError("Foreign Cropped article-binding context")

    def _validate(
        self, record: _RecordedRead, ledger: EvidenceLedger
    ) -> list[EvidenceItem]:
        return self._validate_members(
            record, [ledger.get(number) for number in record.citations]
        )

    def _validate_members(
        self, record: _RecordedRead, originals: list[EvidenceItem | None]
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
            or len(originals) != len(read.members)
            or len(set(record.citations)) != len(record.citations)
            or any(type(n) is not int or n <= 0 for n in record.citations)
            or record.digest != _hash(read.model_dump(mode="json"))
        ):
            raise ValueError("Invalid Cropped ordered source witness")
        items: list[EvidenceItem] = []
        for member, item in zip(read.members, originals, strict=True):
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
                or _source_metadata_hash(item.metadata) != member.metadata_hash
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

    def capture_dispatched_read(
        self,
        call: CapabilityCall,
        outcome: ToolOutcome,
        context: RunContext,
        owner: _ComposedReadOwner,
    ) -> None:
        self._check(context)
        if (
            owner.state is not self
            or context.services.get("cropped_article_bindings") is not self
            or _owner(context) != (owner.task_id, owner.depth)
        ):
            raise ValueError("Foreign Cropped composed-read owner")
        if outcome.status not in {OutcomeStatus.FOUND, OutcomeStatus.PARTIAL}:
            return
        record = self._receipt_read(
            ToolReceipt(call=call, outcome=outcome, evidence_ids=[], elapsed_seconds=0)
        )
        if record is None:
            return
        provisional = record.model_copy(
            update={"citations": list(range(1, len(outcome.evidence) + 1))}
        )
        self._validate_members(provisional, list(outcome.evidence))
        key = owner.task_id, owner.depth, owner.call_id
        with self._lock:
            pending = self._pending.setdefault(key, {})
            existing = pending.get(record.call_id)
            if existing is not None and existing != record:
                raise ValueError("Changed executed Cropped subcall")
            pending[record.call_id] = record

    def _commit_composed_read(
        self, receipt: ToolReceipt, context: RunContext, ledger: EvidenceLedger
    ) -> int:
        task_id, depth = _owner(context)
        with self._lock:
            pending = self._pending.pop((task_id, depth, receipt.call.call_id), {})
            if receipt.outcome.status not in {
                OutcomeStatus.FOUND,
                OutcomeStatus.PARTIAL,
            }:
                return 0
            actual = {item.identity for item in receipt.outcome.evidence}
            numbers = {
                item.identity: number
                for number in receipt.evidence_ids
                if (item := ledger.get(number)) is not None and item.identity in actual
            }
            committed = []
            for captured in pending.values():
                identities = [
                    (captured.read.source_id, member.chunk_id, member.text_hash)
                    for member in captured.read.members
                ]
                if any(identity not in numbers for identity in identities):
                    raise ValueError("Executed subcall originals were not committed")
                record = captured.model_copy(
                    update={"citations": [numbers[identity] for identity in identities]}
                )
                bindings = self._bindings(record, ledger)
                previous = self._reads.get(record.call_id)
                if previous is not None and previous != record:
                    raise ValueError("Changed ordered source receipt")
                committed.append((record, len(bindings)))
            for record, _count in committed:
                self._reads[record.call_id] = record
            return sum(count for _record, count in committed)

    def discard_composed_read(self, call: CapabilityCall, context: RunContext) -> None:
        task_id, depth = _owner(context)
        with self._lock:
            self._pending.pop((task_id, depth, call.call_id), None)

    def _bindings(
        self,
        record: _RecordedRead,
        ledger: EvidenceLedger,
        *,
        authority_rows: dict[int, dict[str, JsonValue]] | None = None,
        source_rows: dict[int, dict[str, JsonValue]] | None = None,
    ) -> dict[int, ArticleParagraphBinding]:
        items = self._validate(record, ledger)
        rows = (
            authority_rows if authority_rows is not None else _authority_row_map(ledger)
        )
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
            body = stripped
            if own is not None:
                active = own
                opening = _OWN_ARTICLE_OPENING.match(stripped)
                assert opening is not None
                body = stripped[opening.end("heading") :].lstrip(".-–—: ")
                marker = _PARAGRAPH.match(body)
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
            paragraphs = [int(paragraph)]
            for line in body[marker.end() :].splitlines()[1:]:
                following = _CONTINUING_PARAGRAPH.match(line)
                if following is not None:
                    paragraphs.append(int(following[1]))
            if any(
                following != previous + 1
                for previous, following in zip(paragraphs, paragraphs[1:])
            ):
                active = None
                previous_paragraph = None
                continue
            previous_paragraph = paragraphs[-1]
            # Existing literal identities must not be narrowed to one paragraph.
            if (
                own is not None
                or row.get("article_no") is not None
                or len(paragraphs) > 1
            ):
                continue
            result[number] = ArticleParagraphBinding(
                number,
                item.source_id,
                member.chunk_id,
                item.text_hash,
                active.article_no,
                active.qualifier,
                paragraph,
            )
        for number, binding in self._unnumbered_bindings(
            record,
            ledger,
            items=items,
            source_rows=source_rows,
        ).items():
            result.setdefault(number, binding)
        for number, binding in self._inserted_bindings(
            record, ledger, items=items, source_rows=source_rows
        ).items():
            result.setdefault(number, binding)
        return result

    def _inserted_bindings(
        self,
        record: _RecordedRead,
        ledger: EvidenceLedger,
        *,
        items: list[EvidenceItem],
        source_rows: dict[int, dict[str, JsonValue]] | None,
    ) -> dict[int, ArticleParagraphBinding]:
        """Bind a literal inserted heading and its ordered clauses, not its base."""
        rows = source_rows if source_rows is not None else _source_row_map(ledger)
        result: dict[int, ArticleParagraphBinding] = {}
        active: _InsertedArticleIdentity | None = None
        previous_position: int | None = None
        next_clause = 0
        for number, member, item in zip(
            record.citations, record.read.members, items, strict=True
        ):
            if (
                previous_position is not None
                and member.position != previous_position + 1
            ):
                active = None
            previous_position = member.position
            row = rows.get(number)
            if row is None:
                active = None
                continue
            anchor = _inserted_article_anchor(item, row)
            if anchor is not None:
                active, next_clause = anchor, 0
                result[number] = _unnumbered_binding(
                    number, item, anchor.article_no, None
                )
                continue
            clause = _INSERTED_CLAUSE.match(item.text)
            metadata = model_evidence_metadata(item.metadata)
            if (
                active is None
                or clause is None
                or next_clause >= len(_CLAUSE_ORDER)
                or clause[1].lower() != _CLAUSE_ORDER[next_clause]
                or metadata.get("clause_label") != clause[1].lower()
                or metadata.get("paragraph_no") is not None
                or not _inserted_article_metadata(item, row, active)
                or any(
                    parse_regulatory_article_heading(line.replace("**", ""))
                    for line in item.text.splitlines()
                )
            ):
                active = None
                continue
            assert item.chunk_id is not None
            result[number] = ArticleParagraphBinding(
                number,
                item.source_id,
                item.chunk_id,
                item.text_hash,
                active.article_no,
                None,
                None,
                clause[1].lower(),
            )
            next_clause += 1
        return result

    def _unnumbered_bindings(
        self,
        record: _RecordedRead,
        ledger: EvidenceLedger,
        *,
        items: list[EvidenceItem],
        source_rows: dict[int, dict[str, JsonValue]] | None,
    ) -> dict[int, ArticleParagraphBinding]:
        rows = source_rows if source_rows is not None else _source_row_map(ledger)
        result: dict[int, ArticleParagraphBinding] = {}
        active: tuple[str, str | None] | None = None
        previous_position: int | None = None
        for number, member, item in zip(
            record.citations, record.read.members, items, strict=True
        ):
            if (
                previous_position is not None
                and member.position != previous_position + 1
            ):
                active = None
            previous_position = member.position
            row = rows.get(number)
            metadata = model_evidence_metadata(item.metadata)
            if row is None or _AMENDING_PASSAGE.search(folded(item.text)):
                active = None
                continue
            anchor = _unnumbered_article_anchor(item, row)
            if anchor is not None:
                article, qualifier, start = anchor
                active = (article, qualifier)
                # A chunk crossing an article boundary cannot inherit the last
                # heading for its preceding text; only subsequent chunks can.
                if (
                    start == len(item.text) - len(item.text.lstrip())
                    and row.get("article_no") is None
                ):
                    result[number] = _unnumbered_binding(
                        number, item, article, qualifier
                    )
                continue
            if (
                active is None
                or any(
                    parse_regulatory_article_heading(line.replace("**", ""))
                    for line in item.text.splitlines()
                )
                or _STRUCTURAL_MARKER.search(item.text)
                or item.text.lstrip().startswith(('"', "“", "'", "‘"))
                or not item.text.rstrip().endswith(".")
                or metadata.get("paragraph_no") is not None
                or metadata.get("provision_identifiers")
                or row.get("article_no") not in (None, active[0])
                or (
                    row.get("article_no") is not None
                    and row.get("article_qualifier") != active[1]
                )
            ):
                active = None
                continue
            if row.get("article_no") is None:
                result[number] = _unnumbered_binding(number, item, *active)
        return result

    def record_receipt(
        self, receipt: ToolReceipt, context: RunContext, ledger: EvidenceLedger
    ) -> int:
        self._check(context)
        if receipt.call.name == "compose_tool_calls":
            return self._commit_composed_read(receipt, context, ledger)
        record = self._receipt_read(receipt)
        if record is None:
            return 0
        with self._lock:
            bindings = self._bindings(record, ledger)
            existing = self._reads.get(record.call_id)
            if existing is not None and existing != record:
                raise ValueError("Changed ordered source receipt")
            self._reads[record.call_id] = record
            return len(bindings)

    @staticmethod
    def _receipt_read(receipt: ToolReceipt) -> _RecordedRead | None:
        if receipt.call.name not in _METHODS:
            return None
        if receipt.call.name == "follow_reference" and not receipt.call.arguments.get(
            "target_source_id"
        ):
            return None
        raw = receipt.outcome.data.get(_FIELD)
        if raw is None:
            return None
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
        actual_source = (
            receipt.call.arguments.get("target_source_id")
            if receipt.call.name == "follow_reference"
            else receipt.call.arguments.get("source_id")
            or receipt.outcome.data.get("source_id")
        )
        if actual_source != read.source_id:
            raise ValueError("Ordered source witness crosses the requested source")
        return record

    def identity_bindings(
        self,
        context: RunContext,
        ledger: EvidenceLedger,
        *,
        source_rows: dict[int, dict[str, JsonValue]] | None = None,
    ) -> dict[int, ArticleParagraphBinding]:
        self._check(context)
        with self._lock:
            result: dict[int, ArticleParagraphBinding] = {}
            authority_rows = _authority_row_map(ledger)
            source_rows = (
                source_rows if source_rows is not None else _source_row_map(ledger)
            )
            for read in self._reads.values():
                for number, binding in self._bindings(
                    read, ledger, authority_rows=authority_rows, source_rows=source_rows
                ).items():
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
            self._pending.clear()


def without_ordered_source_read_data(
    data: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Keep positional verification in host state rather than model tool history."""
    result = copy.deepcopy(data)
    result.pop(_FIELD, None)

    def clean_outcome(value: JsonValue) -> None:
        if not isinstance(value, dict):
            return
        child = value.get("data")
        if (
            isinstance(value.get("status"), str)
            and isinstance(value.get("summary"), str)
            and isinstance(child, dict)
        ):
            value["data"] = without_ordered_source_read_data(child)
        nested = value.get("outcome")
        if isinstance(nested, dict):
            clean_outcome(nested)

    steps = result.get("steps")
    if isinstance(steps, dict):
        for value in steps.values():
            clean_outcome(value)
    results = result.get("results")
    if isinstance(results, list):
        for value in results:
            clean_outcome(value)
    return result


def cropped_identity_ledger(
    ledger: EvidenceLedger, context: RunContext
) -> EvidenceLedger:
    state = context.services.get("cropped_article_bindings")
    source_rows = _source_row_map(ledger)
    bindings = (
        state.identity_bindings(context, ledger, source_rows=source_rows)
        if isinstance(state, CroppedArticleBindings)
        else {}
    )
    # A complete literal own heading can identify its own unnumbered passage
    # without inferring a neighbor or pretending a paragraph number was read.
    for row in source_rows.values():
        number = row.get("citation")
        item = ledger.get(number) if type(number) is int else None
        if item is None or type(number) is not int or row.get("article_no") is not None:
            continue
        anchor = _unnumbered_article_anchor(item, row)
        if anchor is not None and anchor[2] == len(item.text) - len(item.text.lstrip()):
            bindings.setdefault(number, _unnumbered_binding(number, item, *anchor[:2]))
    return canonical_identity_ledger(ledger, article_bindings=bindings)


def cropped_named_authority_gap(
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
        cropped_identity_ledger(ledger, context),
        strict_reference_boundaries=strict_reference_boundaries,
        syntactic_reference_binding=syntactic_reference_binding,
        resolve_defined_abbreviations=resolve_defined_abbreviations,
        strict_numeric_paragraphs=strict_numeric_paragraphs,
        allow_negative_attribution_disclosure=True,
    )
