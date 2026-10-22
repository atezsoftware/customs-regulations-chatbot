"""Bound automatic source-reference reads; unread navigation never proves a rule."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from onyx.asv3.authority import (
    cited_lower_statute_gap,
    cited_lower_statute_references,
    explicit_reference_leads,
    folded,
    known_statute_source_ids,
    native_named_statute_references,
    statute_references,
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

_SERVICE = "cropped_cross_references"


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def _owner(context: RunContext) -> str:
    return str(context.services.get("task_id") or "root")


def _canonical(item: EvidenceItem) -> bool:
    metadata = model_evidence_metadata(item.metadata)
    return bool(
        item.chunk_id
        and item.search_doc is not None
        and item.search_doc.document_id == item.source_id
        and item.search_doc.metadata.get("regulatory_chunk_id") == item.chunk_id
        and not any(
            metadata.get(key)
            for key in ("derived", "external", "untrusted", "truncated")
        )
        and hashlib.sha256(item.text.encode()).hexdigest() == item.text_hash
    )


def _judicial_original(item: EvidenceItem) -> bool:
    """Recover court identity, not the meaning or applicability of its holding."""
    if not _canonical(item):
        return False
    metadata = model_evidence_metadata(item.metadata)
    kind = folded(str(metadata.get("document_type") or ""))
    if kind in {"court_decision", "judicial_decision", "judgment"}:
        return True
    if kind not in {"", "unknown"}:
        return False
    headings = metadata.get("heading_path")
    root = folded(str(headings[0])) if isinstance(headings, list) and headings else ""
    # Spaced institutional headings occur in canonical judicial documents.
    institution = re.sub(r"\s+", "", root)
    numbers = metadata.get("decision_numbers")
    return (
        institution in {"danistay", "yargitay", "anayasamahkemesi"}
        and isinstance(numbers, list)
        and len(numbers) >= 2
        and all(
            isinstance(number, str) and re.fullmatch(r"\d{4}/\d+", number)
            for number in numbers
        )
    )


class _Source(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    citation: int = Field(ge=1)
    source_id: str
    chunk_id: str
    text_hash: str
    metadata_hash: str
    depth: Literal[0, 1]


class _Pending(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    owner: str = Field(min_length=1)
    owner_depth: int = Field(ge=0)
    call: CapabilityCall
    target_key: str
    reference: dict[str, JsonValue]
    depth: Literal[0, 1]


class _Lead(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    reference: dict[str, JsonValue]
    depth: Literal[0, 1, 2]


class _Snapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal[1]
    run_id: str
    scope_hash: str
    request_hash: str
    sources: list[_Source]
    pending: list[_Pending]
    targets: dict[str, Literal[0, 1]]
    leads: list[_Lead]
    skipped: list[dict[str, JsonValue]]
    integrity: str


def _source(number: int, ledger: EvidenceLedger, depth: Literal[0, 1]) -> _Source:
    item = ledger.get(number)
    if item is None or not _canonical(item):
        raise ValueError(
            "Cross-reference provenance needs an actual canonical original"
        )
    assert item.chunk_id is not None
    return _Source(
        citation=number,
        source_id=item.source_id,
        chunk_id=item.chunk_id,
        text_hash=item.text_hash,
        metadata_hash=_hash(model_evidence_metadata(item.metadata)),
        depth=depth,
    )


def _key(reference: dict[str, JsonValue]) -> str:
    number = reference.get("instrument_number")
    formal = reference.get("formal_name")
    article = reference.get("article")
    if (
        not (isinstance(number, str) and number or isinstance(formal, str) and formal)
        or not isinstance(article, str)
        or not article
    ):
        raise ValueError("Cross-reference needs an exact instrument and article")
    return _hash(
        {
            "instrument": number if number else folded(cast(str, formal)),
            "article": article.upper(),
            "qualifier": reference.get("qualifier", reference.get("article_qualifier")),
            "paragraph": reference.get("paragraph"),
            "clause": reference.get("clause"),
        }
    )


class CroppedCrossReferencePolicy:
    """Track host-acquired first-hop originals separately from independent reads."""

    def __init__(self, context: RunContext, request: str) -> None:
        if (
            context.services.get("asv3_exp_cropped") is not True
            or context.services.get("asv3_exp_cheap") is True
        ):
            raise ValueError("Cross-reference policy requires the Cropped workflow")
        self.context = context
        self.run_id = context.run_id
        self.scope_hash = _hash(context.scope)
        self.request = request
        self.request_hash = _hash(request)
        self._lock = threading.RLock()
        self._sources: dict[int, _Source] = {}
        self._pending: dict[tuple[str, int, str], _Pending] = {}
        self._targets: dict[str, Literal[0, 1]] = {}
        self._leads: dict[str, _Lead] = {}
        self._skipped: dict[str, dict[str, JsonValue]] = {}

    def check_context(self, context: RunContext) -> None:
        if (
            context.services.get("asv3_exp_cropped") is not True
            or context.services.get("asv3_exp_cheap") is True
            or context.services.get(_SERVICE) is not self
            or (context.run_id, _hash(context.scope)) != (self.run_id, self.scope_hash)
        ):
            raise ValueError("Cross-reference policy run, scope or host owner changed")

    def _direct(self, reference: dict[str, JsonValue], ledger: EvidenceLedger) -> bool:
        for user in native_named_statute_references(self.request, ledger):
            if (
                user["instrument_number"] != reference.get("instrument_number")
                or not user["instrument_number"]
                and user["formal_name"] != reference.get("formal_name")
            ):
                continue
            if user["article"] is None:
                return True
            if (
                str(user["article"]).upper() == str(reference.get("article")).upper()
                and user["qualifier"]
                == reference.get("qualifier", reference.get("article_qualifier"))
                and (
                    user["paragraph"] is None
                    or reference.get("paragraph") is None
                    or user["paragraph"] == reference.get("paragraph")
                )
                and (
                    user["clause"] is None
                    or reference.get("clause") is None
                    or user["clause"] == reference.get("clause")
                )
            ):
                return True
        return False

    def _origins(
        self, reference: dict[str, JsonValue], ledger: EvidenceLedger
    ) -> list[int]:
        numbers = reference.get("origin_citations")
        if (
            not isinstance(numbers, list)
            or not numbers
            or any(type(number) is not int or number < 1 for number in numbers)
        ):
            raise ValueError("Cross-reference dependency lacks actual origin citations")
        selected = cast(list[int], numbers)
        for number in selected:
            known = self._sources.get(number)
            _source(number, ledger, known.depth if known else 0)
            if known and _source(number, ledger, known.depth) != known:
                raise ValueError("Cross-reference original version changed")
        return selected

    def reference_depth(
        self, reference: dict[str, JsonValue], ledger: EvidenceLedger
    ) -> int:
        self.check_context(self.context)
        with self._lock:
            origins = self._origins(reference, ledger)
            if self._direct(reference, ledger):
                return 0
            target = self._targets.get(_key(reference))
            if target is not None:
                # Re-reading the same provision/its actual continuations is not a new hop.
                return target
            return 1 + min(
                self._sources[number].depth if number in self._sources else 0
                for number in origins
            )

    def filter_references(
        self,
        answer: str | list[dict[str, JsonValue]],
        ledger: EvidenceLedger,
        context: RunContext | None = None,
    ) -> list[dict[str, JsonValue]]:
        self.check_context(context or self.context)
        references = (
            cited_lower_statute_references(answer, ledger)
            if isinstance(answer, str)
            else answer
        )
        allowed: list[dict[str, JsonValue]] = []
        for reference in references:
            origins = self._origins(reference, ledger)
            if all(
                (item := ledger.get(number)) is not None and _judicial_original(item)
                for number in origins
            ):
                reason = "judicial_reference_navigation_not_automatic_dependency"
            elif self.reference_depth(reference, ledger) > 1:
                reason = "nested_cross_reference_not_read"
            else:
                allowed.append(copy.deepcopy(reference))
                continue
            skipped: dict[str, JsonValue] = {
                "reference": copy.deepcopy(reference),
                "reason": reason,
                "unread": True,
                "absence_proven": False,
                "legal_approval": False,
            }
            with self._lock:
                self._skipped[_hash(skipped)] = skipped
        return allowed

    def publication_gap(
        self, answer: str, ledger: EvidenceLedger, delivered: set[int]
    ) -> dict[str, JsonValue] | None:
        references = self.filter_references(
            cited_lower_statute_references(answer, ledger), ledger
        )
        return cited_lower_statute_gap(answer, ledger, delivered, references=references)

    def reserve_acquisition(
        self,
        call: CapabilityCall,
        reference: dict[str, JsonValue],
        context: RunContext,
        ledger: EvidenceLedger,
    ) -> bool:
        """Reserve the actual host-generated call before its ordinary tool dispatch."""
        self.check_context(context)
        if not self.filter_references([reference], ledger):
            return False
        depth = self.reference_depth(reference, ledger)
        if depth not in {0, 1} or call.argument_error is not None:
            raise ValueError("Cross-reference acquisition is not a first-hop call")
        pending = _Pending(
            owner=_owner(context),
            owner_depth=context.depth,
            call=call.model_copy(deep=True),
            target_key=_key(reference),
            reference=copy.deepcopy(reference),
            depth=cast(Literal[0, 1], depth),
        )
        identity = (_owner(context), context.depth, call.call_id)
        with self._lock:
            previous = self._pending.get(identity)
            if previous is not None and previous != pending:
                raise ValueError("Reserved cross-reference call changed")
            self._pending[identity] = pending
        return True

    def _remember_origin_leads(self, number: int, ledger: EvidenceLedger) -> None:
        source = self._sources.get(number)
        item = ledger.get(number)
        if item is None or _judicial_original(item):
            return
        rows = cast(
            list[dict[str, JsonValue]],
            json.loads(
                ledger.serialize_records([number], required=[number], max_chars=None)
            ),
        )
        for reference in explicit_reference_leads(
            ledger, rows, syntactic_reference_binding=True
        ):
            depth = (
                0
                if self._direct(reference, ledger)
                else 1 + (source.depth if source else 0)
            )
            lead = _Lead(reference=reference, depth=depth)
            key = _key(reference)
            previous = self._leads.get(key)
            if previous is None or depth < previous.depth:
                self._leads[key] = lead

    def _call_reference(
        self, call: CapabilityCall, ledger: EvidenceLedger
    ) -> dict[str, JsonValue] | None:
        if call.name not in {
            "read_provision",
            "read_named_provision",
            "follow_reference",
        }:
            return None
        article = call.arguments.get("article")
        if not isinstance(article, str):
            return None
        match = re.fullmatch(
            r"\s*(?:(gecici|geçici|mukerrer|mükerrer|ek)\s+)?(\d{1,4}(?:/[a-zA-Z])?)(?:/(\d+))?\s*",
            article,
        )
        if match is None:
            return None
        qualifier, identity, paragraph = match.groups()
        source = (
            call.arguments.get("target_source_id")
            if call.name == "follow_reference"
            else call.arguments.get("source_id")
        )
        name = call.arguments.get("source_name")
        for lead in self._leads.values():
            reference = lead.reference
            if str(reference["article"]).upper() != identity.upper() or (
                reference.get("qualifier") != (folded(qualifier) if qualifier else None)
            ):
                continue
            number, formal = (
                reference.get("instrument_number"),
                reference.get("formal_name"),
            )
            source_name = (
                f"{number} sayılı Kanun"
                if isinstance(number, str) and number
                else str(formal or "")
            )
            named = isinstance(name, str) and (
                name.strip() == number
                or any(
                    ref.number == number
                    for ref in statute_references(
                        name, strict_reference_boundaries=True
                    )
                )
                or folded(name) == formal
            )
            if not named and source not in known_statute_source_ids(
                source_name, ledger
            ):
                continue
            return {
                **copy.deepcopy(reference),
                "paragraph": call.arguments.get("paragraph", paragraph),
                "clause": call.arguments.get("clause"),
            }
        return None

    def admit(
        self, call: CapabilityCall, context: RunContext, ledger: EvidenceLedger
    ) -> ToolOutcome | None:
        """Explicit follow_reference origins carry depth even without automatic reads."""
        self.check_context(context)
        if call.argument_error is not None:
            return None
        with self._lock:
            identity = (_owner(context), context.depth, call.call_id)
            reserved = self._pending.get(identity)
            if reserved is not None:
                if reserved.call != call:
                    raise ValueError("Reserved cross-reference call changed")
                return None
            reference = self._call_reference(call, ledger)
            if reference is not None:
                if self._direct(reference, ledger):
                    return None
                key = _key({**reference, "paragraph": None, "clause": None})
                lead = self._leads.get(key)
                if (
                    lead is not None
                    and lead.depth > 1
                    and not self._direct(reference, ledger)
                ):
                    return self._blocked(reference.get("origin_citations", []))
            if call.name != "follow_reference":
                return None
        source, chunk = call.arguments.get("source_id"), call.arguments.get("chunk_id")
        origins = [
            number
            for number in ledger.citation_numbers()
            if (item := ledger.get(number)) is not None
            and item.source_id == source
            and item.chunk_id == chunk
        ]
        if len(origins) != 1:
            return None  # The existing canonical origin/RBAC gate handles unknown identity.
        with self._lock:
            known = self._sources.get(origins[0])
            _source(origins[0], ledger, known.depth if known else 0)
            self._remember_origin_leads(origins[0], ledger)
            if known is None or known.depth == 0:
                return None
            target_source = call.arguments.get("target_source_id")
            article = call.arguments.get("article")
            for user in statute_references(
                self.request,
                strict_reference_boundaries=True,
                syntactic_reference_binding=True,
            ):
                if (
                    user.article is None or user.article == article
                ) and target_source in known_statute_source_ids(
                    f"{user.number} sayılı Kanun", ledger
                ):
                    return None
            if isinstance(target_source, str) and isinstance(article, str):
                for pending in self._pending.values():
                    if (
                        pending.call.arguments.get("source_id") == target_source
                        and pending.call.arguments.get("article") == article
                    ):
                        return None
        return self._blocked(cast(list[JsonValue], origins))

    @staticmethod
    def _blocked(origins: JsonValue) -> ToolOutcome:
        return ToolOutcome(
            status=OutcomeStatus.DENIED,
            summary="This would follow a second source-reference hop; the target remains unread.",
            data={
                "cropped_nested_cross_reference_blocked": True,
                "origin_citations": origins,
                "requested_read_executed": False,
                "not_read": True,
                "absence_proven": False,
                "legal_approval": False,
            },
        )

    def observe(
        self, receipt: ToolReceipt, context: RunContext, ledger: EvidenceLedger
    ) -> None:
        """Record real acquired originals, never synthetic model delivery receipts."""
        self.check_context(context)
        identity = (_owner(context), context.depth, receipt.call.call_id)
        with self._lock:
            numbers = set(receipt.evidence_ids) | {
                read.citation for read in receipt.outcome.original_reads
            }
            if not numbers or receipt.outcome.status not in {
                OutcomeStatus.FOUND,
                OutcomeStatus.PARTIAL,
                OutcomeStatus.TRUNCATED,
                OutcomeStatus.AMBIGUOUS,
            }:
                return
            pending = self._pending.get(identity)
            if pending is not None and pending.call != receipt.call:
                raise ValueError(
                    "Cross-reference receipt differs from its reserved call"
                )
            depth: Literal[0, 1] = pending.depth if pending else 0
            independent = False
            referenced = pending is not None
            if (
                pending is None
                and receipt.call.name == "follow_reference"
                and (
                    receipt.outcome.data.get("origin_chunk_id") is not None
                    and receipt.outcome.status != OutcomeStatus.AMBIGUOUS
                )
            ):
                if self.admit(receipt.call, context, ledger) is not None:
                    raise ValueError("A blocked nested reference was executed")
                source, chunk = (
                    receipt.call.arguments.get("source_id"),
                    receipt.call.arguments.get("chunk_id"),
                )
                if any(
                    (item := ledger.get(number)) is not None
                    and item.source_id == source
                    and item.chunk_id == chunk
                    for number in ledger.citation_numbers()
                ):
                    direct_reference = self._call_reference(receipt.call, ledger)
                    independent = direct_reference is not None and self._direct(
                        direct_reference, ledger
                    )
                    depth = 0 if independent else 1
                    referenced = True
            elif pending is None:
                reference = self._call_reference(receipt.call, ledger)
                if reference is not None:
                    key = _key({**reference, "paragraph": None, "clause": None})
                    lead = self._leads.get(key)
                    if lead is not None:
                        if lead.depth > 1 and not self._direct(reference, ledger):
                            raise ValueError(
                                "A blocked nested target was read through another tool"
                            )
                        depth = (
                            0
                            if self._direct(reference, ledger)
                            else cast(Literal[0, 1], lead.depth)
                        )
                        independent = self._direct(reference, ledger)
                        referenced = True
            from onyx.asv3.cheap_authority import canonical_identity_ledger

            identities = {
                row["citation"]: (row.get("article_no"), row.get("article_qualifier"))
                for row in canonical_identity_ledger(ledger).authority_metadata()
            }
            for number in numbers:
                previous = self._sources.get(number)
                item = ledger.get(number)
                if item is None or not _canonical(item):
                    if pending is not None:
                        raise ValueError(
                            "Reserved cross-reference original is not canonical"
                        )
                    continue
                article = identities.get(number)
                inherited = [
                    row.depth
                    for row in self._sources.values()
                    if row.source_id == item.source_id
                    and (
                        article is None
                        or article[0] is None
                        or identities.get(row.citation) == article
                    )
                ]
                actual_depth = (
                    depth
                    if referenced or independent
                    else (
                        previous.depth
                        if previous
                        else max(inherited)
                        if inherited
                        else depth
                    )
                )
                actual = _source(number, ledger, actual_depth)
                if previous:
                    if actual.model_copy(update={"depth": previous.depth}) != previous:
                        raise ValueError("Cross-reference provenance original changed")
                    actual = actual.model_copy(
                        update={"depth": 0 if independent else previous.depth}
                    )
                self._sources[number] = actual
                self._remember_origin_leads(number, ledger)
            if pending is not None:
                self._targets[pending.target_key] = depth

    def referenced_source_ids(self, context: RunContext) -> list[str]:
        """Identify acquired first-hop sources without exposing mutable policy state."""
        self.check_context(context)
        with self._lock:
            return sorted(
                {row.source_id for row in self._sources.values() if row.depth > 0}
            )

    def view(self, context: RunContext | None = None) -> dict[str, JsonValue]:
        self.check_context(context or self.context)
        with self._lock:
            return {
                "maximum_automatic_reference_hops": 1,
                "unread_reference_navigation": copy.deepcopy(
                    list(self._skipped.values())
                ),
                "legal_approval": False,
                "instruction": "Only one source-reference hop is acquired automatically. Nested and incidental judicial references remain unread navigation. A direct user target is independent. Do not assert unread legal effects or treat missing reads as absent law; preserve complete operative originals and actual continuations.",
            }

    def export(self) -> dict[str, JsonValue]:
        with self._lock:
            data: dict[str, JsonValue] = {
                "version": 1,
                "run_id": self.run_id,
                "scope_hash": self.scope_hash,
                "request_hash": self.request_hash,
                "sources": [
                    row.model_dump(mode="json") for row in self._sources.values()
                ],
                "pending": [
                    row.model_dump(mode="json") for row in self._pending.values()
                ],
                "targets": dict(self._targets),
                "leads": [row.model_dump(mode="json") for row in self._leads.values()],
                "skipped": copy.deepcopy(list(self._skipped.values())),
            }
            return {**data, "integrity": _hash(data)}

    def restore(
        self, saved: dict[str, JsonValue], context: RunContext, ledger: EvidenceLedger
    ) -> None:
        self.check_context(context)
        snapshot = _Snapshot.model_validate(saved)
        data = dict(saved)
        data.pop("integrity")
        if snapshot.integrity != _hash(data) or (
            snapshot.run_id,
            snapshot.scope_hash,
            snapshot.request_hash,
        ) != (self.run_id, self.scope_hash, self.request_hash):
            raise ValueError("Cross-reference checkpoint scope or integrity changed")
        sources = {row.citation: row for row in snapshot.sources}
        pending = {
            (row.owner, row.owner_depth, row.call.call_id): row
            for row in snapshot.pending
        }
        if len(sources) != len(snapshot.sources) or len(pending) != len(
            snapshot.pending
        ):
            raise ValueError("Duplicate cross-reference checkpoint provenance")
        for row in sources.values():
            if _source(row.citation, ledger, row.depth) != row:
                raise ValueError("Cross-reference checkpoint original changed")
        for row in pending.values():
            if row.target_key != _key(row.reference):
                raise ValueError("Cross-reference checkpoint target changed")
            for number in cast(list[int], row.reference.get("origin_citations", [])):
                _source(
                    number, ledger, sources[number].depth if number in sources else 0
                )
        for lead in snapshot.leads:
            origins = lead.reference.get("origin_citations")
            if (
                not isinstance(origins, list)
                or not origins
                or any(
                    type(number) is not int or number not in sources
                    for number in origins
                )
            ):
                raise ValueError(
                    "Cross-reference checkpoint lead lacks its real origin"
                )
            numbers = cast(list[int], origins)
            expected = (
                0
                if self._direct(lead.reference, ledger)
                else 1 + min(sources[number].depth for number in numbers)
            )
            if lead.depth != expected:
                raise ValueError("Cross-reference checkpoint lead depth changed")
            literal = cast(
                list[dict[str, JsonValue]],
                json.loads(
                    ledger.serialize_records(numbers, required=numbers, max_chars=None)
                ),
            )
            if not any(
                _key(actual) == _key(lead.reference)
                for actual in explicit_reference_leads(
                    ledger, literal, syntactic_reference_binding=True
                )
            ):
                raise ValueError(
                    "Cross-reference checkpoint lead is not in its original"
                )
        if any(
            not any(row.target_key == key for row in pending.values())
            for key in snapshot.targets
        ):
            raise ValueError(
                "Cross-reference checkpoint lacks its acquisition provenance"
            )
        with self._lock:
            self._sources, self._pending, self._targets = (
                sources,
                pending,
                dict(snapshot.targets),
            )
            self._leads = {_key(row.reference): row for row in snapshot.leads}
            self._skipped = {_hash(row): copy.deepcopy(row) for row in snapshot.skipped}
