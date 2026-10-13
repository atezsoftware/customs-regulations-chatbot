"""Source-bound working packets; retained candidates are never legal approval."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from collections.abc import Sequence
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import CapabilityCall, HarnessView, RunContext, ToolReceipt
from onyx.asv3.outcome_map import OutcomeMap
from onyx.asv3.research_state import ResearchFinding, ResearchNeed, ResearchState

_SERVICE = "cropped_task_packet_store"
_TEXT_REF = "cropped_original_text_ref"
_RESTORED = "cropped_restored_topic_context"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _enabled(context: RunContext) -> bool:
    return context.services.get("asv3_exp_cropped") is True and (
        context.services.get("asv3_exp_cheap") is not True
    )


def _owner(context: RunContext) -> str:
    return str(context.services.get("task_id") or "root")


class CroppedTaskPacketPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    cumulative_input_tokens: int = Field(default=1_000_000, ge=1)
    original_records: int = Field(default=256, ge=1)


class _Source(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    citation: int = Field(ge=1)
    source_id: str
    chunk_id: str | None
    text_hash: str
    metadata_hash: str


def _source(number: int, ledger: EvidenceLedger) -> _Source:
    item = ledger.get(number)
    if item is None or hashlib.sha256(item.text.encode()).hexdigest() != item.text_hash:
        raise ValueError("Task packet original identity or literal text changed")
    return _Source(
        citation=number,
        source_id=item.source_id,
        chunk_id=item.chunk_id,
        text_hash=item.text_hash,
        metadata_hash=_digest(item.metadata),
    )


class _Receipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    owner: str
    call_id: str
    need_ids: list[str]
    outcome_ids: list[str]
    sources: list[_Source]
    raw: dict[str, JsonValue]
    composed_steps: list[_ComposedStep] = Field(default_factory=list)


class _ComposedStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    step_id: str
    child_call_id: str
    need_ids: list[str]
    outcome_ids: list[str]
    citations: list[int]
    sources: list[_Source]
    parent_step_hash: str
    captured_call: dict[str, JsonValue]
    captured_originals: list[dict[str, JsonValue]]
    continuation: dict[str, JsonValue]
    partitionable: bool
    capture_hash: str = ""
    references: list[_ResearchReference] = Field(default_factory=list)


class _ResearchReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    source: _Source
    need_id: str
    start_char: int | None = None
    end_char: int | None = None


def _research_references(
    tool_name: str,
    status: str,
    data: dict[str, JsonValue],
    ledger: EvidenceLedger,
    known_needs: set[str] | None = None,
) -> list[_ResearchReference]:
    """Validate retained finding references without recording another original read."""
    if tool_name != "update_research" or status != "found":
        return []
    result: list[_ResearchReference] = []
    for table in ("needs", "findings"):
        rows = data.get(table, [])
        if not isinstance(rows, list):
            raise ValueError("Retained research reference rows changed")
        for raw in rows:
            if not isinstance(raw, dict):
                raise ValueError("Retained research reference row changed")
            need_id = raw.get("need_id")
            if (
                not isinstance(need_id, str)
                or not need_id
                or known_needs is not None
                and need_id not in known_needs
            ):
                raise ValueError("Retained research reference need changed")
            if table == "needs":
                numbers = raw.get("evidence_numbers", [])
                if not isinstance(numbers, list) or any(
                    type(n) is not int for n in numbers
                ):
                    raise ValueError("Retained research reference numbers changed")
                result.extend(
                    _ResearchReference(
                        source=_source(cast(int, n), ledger), need_id=need_id
                    )
                    for n in numbers
                )
            else:
                finding = ResearchFinding.model_validate(raw)
                for witness in finding.witnesses:
                    source = _source(witness.citation, ledger)
                    item = ledger.get(witness.citation)
                    if (
                        item is None
                        or not 0
                        <= witness.start_char
                        < witness.end_char
                        <= len(item.text)
                    ):
                        raise ValueError("Retained research witness range changed")
                    result.append(
                        _ResearchReference(
                            source=source,
                            need_id=need_id,
                            start_char=witness.start_char,
                            end_char=witness.end_char,
                        )
                    )
    return result


def _step_need_ids(step: _ComposedStep) -> set[str]:
    return {*step.need_ids, *(reference.need_id for reference in step.references)}


def _captured_source(source: _Source) -> dict[str, JsonValue]:
    return {
        "source_id": source.source_id,
        "chunk_id": source.chunk_id,
        "text_hash": source.text_hash,
    }


def _capture_hash(step: _ComposedStep, owner: str, parent_id: str) -> str:
    return _digest(
        {
            "owner": owner,
            "parent_call_id": parent_id,
            "step_id": step.step_id,
            "parent_step_hash": step.parent_step_hash,
            "call": step.captured_call,
            "originals": step.captured_originals,
            "continuation": step.continuation,
            "partitionable": step.partitionable,
            **(
                {"references": [r.model_dump(mode="json") for r in step.references]}
                if step.references
                else {}
            ),
        }
    )


def _receipt_need_numbers(row: _Receipt, need_id: str) -> set[int]:
    if row.composed_steps:
        return {
            n
            for step in row.composed_steps
            if need_id in step.need_ids
            for n in step.citations
        } | {
            reference.source.citation
            for step in row.composed_steps
            for reference in step.references
            if reference.need_id == need_id
        }
    return {s.citation for s in row.sources} if need_id in row.need_ids else set()


def _composed_step_status(row: _Receipt, step: _ComposedStep) -> str:
    outcome = cast(dict[str, JsonValue], row.raw["outcome"])
    data = cast(dict[str, JsonValue], outcome["data"])
    steps = cast(dict[str, JsonValue], data["steps"])
    return str(cast(dict[str, JsonValue], steps[step.step_id])["status"])


def _unbound_numbers(row: _Receipt) -> set[int]:
    if row.composed_steps:
        assigned = {
            n
            for step in row.composed_steps
            for n in [*step.citations, *(r.source.citation for r in step.references)]
        }
        return {s.citation for s in row.sources if s.citation not in assigned} | {
            n
            for step in row.composed_steps
            if not step.need_ids
            for n in step.citations
        }
    return {s.citation for s in row.sources} if not row.need_ids else set()


def _project_composed_receipt(
    row: _Receipt, selected: set[str]
) -> dict[str, JsonValue]:
    """A labelled view of actual step outputs; the complete parent stays in the journal."""
    raw = copy.deepcopy(row.raw)
    if not row.composed_steps:
        return raw
    included = {
        step.step_id
        for step in row.composed_steps
        if not step.partitionable
        or not step.need_ids
        or step.captured_call.get("name") == "update_research"
        or selected.intersection(step.need_ids)
    }
    omitted = {step.step_id for step in row.composed_steps} - included
    if not omitted:
        return raw
    outcome = cast(dict[str, JsonValue], raw["outcome"])
    data = cast(dict[str, JsonValue], outcome["data"])
    steps = cast(dict[str, JsonValue], data["steps"])
    data["steps"] = {key: value for key, value in steps.items() if key in included}
    data["cropped_composed_topic_projection"] = {
        "parent_call_id": row.call_id,
        "parent_receipt_hash": _digest(row.raw),
        "included_step_ids": sorted(included),
        "parked_step_ids": sorted(omitted),
        "pending_passthrough_step_ids": [
            step.step_id for step in row.composed_steps if not step.partitionable
        ],
        "notice": "Only these actual step outputs are shown. Other outputs and originals remain in the immutable parent journal and return when their topic is activated or for final writing. No step was executed again; no parked work is approved or completed.",
    }
    numbers = {
        n
        for step in row.composed_steps
        if step.step_id in included
        for n in step.citations
    }
    raw["evidence_ids"] = sorted(numbers)
    outcome["original_reads"] = [
        read
        for read in cast(list[dict[str, JsonValue]], outcome.get("original_reads", []))
        if read.get("citation") in numbers
    ]
    return raw


class _Seal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    owner: str
    need_id: str
    facts_hash: str
    state_hash: str
    assessment_hash: str


class _Checkpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal[2]
    run_id: str
    scope_hash: str
    request_hash: str
    policy: CroppedTaskPacketPolicy
    revision: int = Field(ge=0)
    input_tokens: dict[str, int]
    owner_requests: dict[str, str]
    receipts: list[_Receipt]
    seals: list[_Seal]
    invalidated: list[_Seal]
    integrity: str


def _receipt_numbers(receipt: ToolReceipt) -> set[int]:
    numbers = {
        *receipt.evidence_ids,
        *(r.citation for r in receipt.outcome.original_reads),
    }

    def visit(value: JsonValue) -> None:
        if isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            for key, child in value.items():
                if key == "citation" and type(child) is int and child > 0:
                    numbers.add(child)
                elif key in {"evidence_numbers", "evidence_ids"} and isinstance(
                    child, list
                ):
                    numbers.update(n for n in child if type(n) is int and n > 0)
                else:
                    visit(child)

    visit(receipt.outcome.data)
    if any(type(number) is not int or number < 1 for number in numbers):
        raise ValueError("Task packet receipt has invalid original numbers")
    return numbers


def _outcome_ids(arguments: dict[str, JsonValue]) -> list[str]:
    result: set[str] = set()
    declared = arguments.get("outcome_ids")
    if isinstance(declared, list):
        result.update(value for value in declared if isinstance(value, str))
    outcome_rows = arguments.get("_outcomes")
    for row in outcome_rows if isinstance(outcome_rows, list) else []:
        if isinstance(row, dict) and isinstance(row.get("outcome_id"), str):
            result.add(cast(str, row["outcome_id"]))
    coverage = arguments.get("_coverage")
    if isinstance(coverage, dict):
        for row in (
            coverage.get("resolutions", [])
            if isinstance(coverage.get("resolutions"), list)
            else []
        ):
            if isinstance(row, dict) and isinstance(row.get("outcome_id"), str):
                result.add(cast(str, row["outcome_id"]))
        for row in (
            coverage.get("conditions", [])
            if isinstance(coverage.get("conditions"), list)
            else []
        ):
            ids = row.get("outcome_ids") if isinstance(row, dict) else None
            if isinstance(ids, list):
                result.update(value for value in ids if isinstance(value, str))
    return sorted(result)


def _state_rows(
    state: ResearchState,
) -> tuple[dict[str, JsonValue], dict[str, ResearchNeed], list[ResearchFinding]]:
    snapshot = state.export()
    needs = {
        row.need_id: row
        for row in (
            ResearchNeed.model_validate(item)
            for item in cast(list[JsonValue], snapshot["needs"])
        )
    }
    findings = [
        ResearchFinding.model_validate(item)
        for item in cast(list[JsonValue], snapshot["findings"])
    ]
    return snapshot, needs, findings


class CroppedTaskPacketStore:
    """Share immutable receipt bindings across owners with the same source fence."""

    def __init__(
        self,
        context: RunContext,
        request: str,
        *,
        policy: CroppedTaskPacketPolicy | None = None,
    ) -> None:
        if not _enabled(context):
            raise ValueError("Task packets require the isolated Cropped workflow")
        self.run_id = context.run_id
        self.scope_hash = _digest(context.scope)
        self.request_hash = _digest(request)
        self.request = request
        self.research_scope_hash = hashlib.sha256(
            json.dumps(context.scope, sort_keys=True).encode()
        ).hexdigest()
        self.policy = policy or CroppedTaskPacketPolicy()
        self._inputs: dict[str, int] = {}
        self._owner_requests: dict[str, str] = {}
        self._receipts: dict[tuple[str, str], _Receipt] = {}
        self._seals: dict[tuple[str, str], _Seal] = {}
        self._invalidated: dict[tuple[str, str], _Seal] = {}
        self._lock = threading.RLock()
        from onyx.asv3.cropped_composed_topics import ComposedTopicCapture

        self.composed_capture = ComposedTopicCapture(self)
        self._committed_compositions: dict[
            tuple[str, str], tuple[str, list[_ComposedStep]]
        ] = {}
        self.revision = 0

    def commit_composed_topics(
        self, receipt: ToolReceipt, context: RunContext, ledger: EvidenceLedger
    ) -> None:
        """Bind dispatched children only after their parent enters the host journal."""
        children = self.composed_capture.take(receipt, context)
        if not children:
            return
        from onyx.asv3.cropped_composed_topics import (
            topic_step_continuation,
            topic_step_partitionable,
        )

        if _outcome_ids(receipt.call.arguments) or receipt.call.arguments.get(
            "_coverage"
        ):
            return
        state = context.services.get("research_state")
        if not isinstance(state, ResearchState):
            return
        _, needs, _ = _state_rows(state)
        sources = {
            _source(n, ledger).citation: ledger.get(n)
            for n in _receipt_numbers(receipt)
        }
        steps: list[_ComposedStep] = []
        for child in children:
            numbers: set[int] = set()
            captured_originals: dict[str, dict[str, JsonValue]] = {}
            for item in child.outcome.evidence:
                matches = [
                    n
                    for n, actual in sources.items()
                    if actual is not None
                    and actual.identity == item.identity
                    and actual.text == item.text
                ]
                if len(matches) != 1:
                    return
                numbers.add(matches[0])
                original: dict[str, JsonValue] = {
                    "source_id": item.source_id,
                    "chunk_id": item.chunk_id,
                    "text_hash": item.text_hash,
                }
                captured_originals[_digest(original)] = original
            for read in child.outcome.original_reads:
                item = sources.get(read.citation)
                if (
                    item is None
                    or item.text_hash != read.text_hash
                    or not 0 <= read.start_char < read.end_char <= len(item.text)
                ):
                    return
                numbers.add(read.citation)
                original = {
                    "source_id": item.source_id,
                    "chunk_id": item.chunk_id,
                    "text_hash": read.text_hash,
                }
                captured_originals[_digest(original)] = original
            binding = child.call.arguments.get("_need_id")
            data_steps = receipt.outcome.data.get("steps")
            if (
                not isinstance(data_steps, dict)
                or child.step_id not in data_steps
                or not isinstance(data_steps[child.step_id], dict)
            ):
                return
            if data_steps[child.step_id].get("status") != child.outcome.status.value:
                return
            continuation = topic_step_continuation(child.outcome.data)
            saved_data = data_steps[child.step_id].get("data")
            if not isinstance(saved_data, dict) or any(
                continuation.get(key) != value
                for key, value in topic_step_continuation(saved_data).items()
            ):
                return
            try:
                references = _research_references(
                    child.call.name,
                    child.outcome.status.value,
                    child.outcome.data,
                    ledger,
                    set(needs),
                )
                if references != _research_references(
                    child.call.name,
                    child.outcome.status.value,
                    saved_data,
                    ledger,
                    set(needs),
                ):
                    return
            except ValueError:
                return
            step = _ComposedStep(
                step_id=child.step_id,
                child_call_id=child.call.call_id,
                need_ids=[binding]
                if isinstance(binding, str) and binding in needs
                else [],
                outcome_ids=_outcome_ids(child.call.arguments),
                citations=sorted(numbers),
                sources=[_source(n, ledger) for n in sorted(numbers)],
                parent_step_hash=_digest(data_steps[child.step_id]),
                captured_call=child.call.model_dump(mode="json"),
                captured_originals=[
                    captured_originals[key] for key in sorted(captured_originals)
                ],
                continuation=continuation,
                partitionable=topic_step_partitionable(
                    child.outcome.status.value, continuation, child.call.name
                ),
                references=references,
            )
            steps.append(
                step.model_copy(
                    update={
                        "capture_hash": _capture_hash(
                            step, _owner(context), receipt.call.call_id
                        )
                    }
                )
            )
        delivered = {
            *receipt.evidence_ids,
            *(read.citation for read in receipt.outcome.original_reads),
        }
        # Retained references cannot stand in for an actual child source delivery.
        if {n for step in steps for n in step.citations} != delivered or {
            n
            for step in steps
            for n in [*step.citations, *(r.source.citation for r in step.references)]
        } != set(sources):
            return
        raw_steps = receipt.call.arguments.get("steps")
        if not isinstance(raw_steps, list) or any(
            isinstance(row, dict) and (row.get("depends_on") or row.get("when"))
            for row in raw_steps
        ):
            return
        key = _owner(context), receipt.call.call_id
        value = _digest(receipt.model_dump(mode="json")), steps
        with self._lock:
            previous = self._committed_compositions.get(key)
            if previous is not None and previous != value:
                raise ValueError("Committed composed topic proof changed")
            self._committed_compositions[key] = value
        self.record_receipts(key[0], [receipt], state, ledger, context=context)

    def check_context(self, context: RunContext, request: str) -> None:
        if not _enabled(context) or (
            context.run_id,
            _digest(context.scope),
            _digest(request),
        ) != (self.run_id, self.scope_hash, self.request_hash):
            raise ValueError("Task packet run, request or authorized scope changed")

    def record_input(self, owner: str, input_tokens: int) -> None:
        if not owner or type(input_tokens) is not int or input_tokens < 0:
            raise ValueError(
                "Task packet input accounting needs a real owner and count"
            )
        with self._lock:
            self._inputs[owner] = self._inputs.get(owner, 0) + input_tokens

    def record_receipts(
        self,
        owner: str,
        receipts: Sequence[ToolReceipt],
        state: ResearchState,
        ledger: EvidenceLedger,
        *,
        context: RunContext | None = None,
    ) -> None:
        if not owner or (state.run_id, state.scope_hash) != (
            self.run_id,
            self.research_scope_hash,
        ):
            raise ValueError("Task packet receipt state or owner is not bound")
        _, needs, _ = _state_rows(state)
        focus: object = None
        if context is not None:
            self.check_context(context, _request(context, self.request))
            if _owner(context) != owner:
                raise ValueError("Task packet receipt owner changed")
            focus = context.services.get("task_need_ids")
        pending: list[_Receipt] = []
        for receipt in receipts:
            binding = receipt.call.arguments.get("_need_id")
            declared = receipt.call.arguments.get("need_ids")
            if receipt.call.name == "update_research" and not isinstance(binding, str):
                metadata_ids: list[str] = []
                for table in ("needs", "findings"):
                    supplied = receipt.call.arguments.get(table)
                    if isinstance(supplied, list):
                        metadata_ids.extend(
                            cast(str, row["need_id"])
                            for row in supplied
                            if isinstance(row, dict)
                            and isinstance(row.get("need_id"), str)
                        )
                declared = list(dict.fromkeys(metadata_ids))
            ids = (
                [binding]
                if isinstance(binding, str)
                else declared
                if isinstance(declared, list)
                else focus
            )
            selected = (
                list(dict.fromkeys(cast(list[str], ids)))
                if isinstance(ids, list)
                and ids
                and all(isinstance(key, str) and key in needs for key in ids)
                else []
            )
            raw = receipt.model_dump(mode="json")
            sources = [_source(n, ledger) for n in sorted(_receipt_numbers(receipt))]
            prior = self._receipts.get((owner, receipt.call.call_id))
            composition = self._committed_compositions.get(
                (owner, receipt.call.call_id)
            )
            steps = (
                composition[1]
                if composition is not None and composition[0] == _digest(raw)
                else prior.composed_steps
                if prior is not None and prior.raw == raw
                else []
            )
            if steps:
                selected = sorted(
                    {key for step in steps for key in _step_need_ids(step)}
                )
            pending.append(
                _Receipt(
                    owner=owner,
                    call_id=receipt.call.call_id,
                    need_ids=selected,
                    outcome_ids=_outcome_ids(receipt.call.arguments),
                    sources=sources,
                    raw=raw,
                    composed_steps=steps,
                )
            )
        with self._lock:
            owner_request = _digest(list(state.questions))
            if self._owner_requests.get(owner, owner_request) != owner_request:
                raise ValueError("Task packet owner request changed")
            for row in pending:
                previous = self._receipts.get((owner, row.call_id))
                if previous is not None and (
                    previous.raw != row.raw or previous.sources != row.sources
                ):
                    raise ValueError(
                        "An actual task receipt changed after being retained"
                    )
            additions = {
                (owner, row.call_id): row
                for row in pending
                if self._receipts.get((owner, row.call_id)) != row
            }
            if additions:
                self._receipts.update(additions)
                self.revision += 1
            self._owner_requests[owner] = owner_request

    def _need_numbers(
        self, owner: str, need: ResearchNeed, findings: Sequence[ResearchFinding]
    ) -> set[int]:
        return {
            *need.evidence_numbers,
            *(
                w.citation
                for f in findings
                if f.need_id == need.need_id
                for w in f.witnesses
            ),
            *(
                number
                for row in self._receipts.values()
                if row.owner == owner
                for number in _receipt_need_numbers(row, need.need_id)
            ),
        }

    def _need_hash(
        self,
        owner: str,
        need: ResearchNeed,
        needs: dict[str, ResearchNeed],
        findings: Sequence[ResearchFinding],
        ledger: EvidenceLedger,
        outcomes: object = None,
    ) -> str:
        closure = _closure({need.need_id}, needs)
        outcome_state: dict[str, JsonValue] = {}
        if isinstance(outcomes, OutcomeMap):
            linked = {
                o
                for row in self._receipts.values()
                if row.owner == owner and closure.intersection(row.need_ids)
                for o in row.outcome_ids
            }
            known = set(outcomes.outcome_ids())
            outcome_state = outcomes.view(
                outcome_ids=sorted(linked) if linked and linked <= known else None
            )
        return _digest(
            {
                "outcome_dependencies": outcome_state,
                "needs": [
                    needs[key].model_dump(mode="json") for key in sorted(closure)
                ],
                "findings": [
                    f.model_dump(mode="json") for f in findings if f.need_id in closure
                ],
                "sources": [
                    _source(n, ledger).model_dump(mode="json")
                    for n in sorted(
                        {
                            n
                            for key in closure
                            for n in self._need_numbers(owner, needs[key], findings)
                        }
                    )
                ],
                "receipts": [
                    row.model_dump(mode="json")
                    for row in self._receipts.values()
                    if row.owner == owner and closure.intersection(row.need_ids)
                ],
            }
        )

    def _assessment_hash(
        self,
        owner: str,
        need: ResearchNeed,
        needs: dict[str, ResearchNeed],
        findings: Sequence[ResearchFinding],
    ) -> str:
        closure = _closure({need.need_id}, needs)
        return _digest(
            {
                "needs": [
                    needs[key].model_dump(mode="json") for key in sorted(closure)
                ],
                "findings": [
                    finding.model_dump(mode="json")
                    for finding in findings
                    if finding.need_id in closure
                ],
                "assessments": [
                    row.raw
                    for row in self._receipts.values()
                    if row.owner == owner
                    and closure.intersection(row.need_ids)
                    and cast(dict[str, JsonValue], row.raw["call"]).get("name")
                    == "update_research"
                    and cast(dict[str, JsonValue], row.raw["outcome"]).get("status")
                    == "found"
                ],
            }
        )

    def seal_need_context(
        self,
        need_id: str,
        state: ResearchState,
        context: RunContext,
        facts: Sequence[str],
        ledger: EvidenceLedger,
    ) -> bool:
        """Retain bound topic context without completing or approving its research."""
        request = _request(context, self.request)
        self.check_context(context, request)
        if (state.run_id, state.scope_hash) != (self.run_id, self.research_scope_hash):
            raise ValueError("Task packet completion state scope changed")
        _, needs, findings = _state_rows(state)
        need = needs.get(need_id)
        if (
            need is None
            or need.status not in {"open", "researching", "candidate"}
            or need.status == "candidate"
            and not any(f.need_id == need_id for f in findings)
        ):
            return False
        owner = _owner(context)
        with self._lock:
            rows = [
                r
                for r in self._receipts.values()
                if r.owner == owner and need_id in r.need_ids
            ]
            numbers = self._need_numbers(owner, need, findings)
            bound = {s.citation for r in rows for s in r.sources}
            if not rows or not numbers or numbers - bound:
                return False
            # A candidate with outstanding acquisition cannot be treated as closed.
            # Open topics keep their exact pending operations when reactivated.
            for row in rows:
                if row.composed_steps:
                    if need.status == "candidate" and any(
                        need_id in step.need_ids and not step.partitionable
                        for step in row.composed_steps
                    ):
                        return False
                    continue
                raw_outcome = row.raw.get("outcome")
                if not isinstance(raw_outcome, dict):
                    return False
                data = raw_outcome.get("data")
                if need.status == "candidate" and (
                    raw_outcome.get("status") != "found"
                    or (
                        isinstance(data, dict)
                        and (
                            data.get("has_more") is True or data.get("not_read") is True
                        )
                    )
                ):
                    return False
            signature = self._need_hash(
                owner,
                need,
                needs,
                findings,
                ledger,
                context.services.get("outcome_map"),
            )
            invalidated = self._invalidated.get((owner, need_id))
            assessment = self._assessment_hash(owner, need, needs, findings)
            if invalidated is not None and invalidated.assessment_hash == assessment:
                return False
            seal = _Seal(
                owner=owner,
                need_id=need_id,
                facts_hash=_digest(list(facts)),
                state_hash=signature,
                assessment_hash=assessment,
            )
            if self._seals.get((owner, need_id)) != seal:
                self._seals[owner, need_id] = seal
                self._invalidated.pop((owner, need_id), None)
                self.revision += 1
            return True

    def seal_eligible(
        self,
        state: ResearchState,
        context: RunContext,
        facts: Sequence[str],
        ledger: EvidenceLedger,
    ) -> list[str]:
        _, needs, _ = _state_rows(state)
        return [
            key
            for key in needs
            if self.seal_need_context(key, state, context, facts, ledger)
        ]

    def export(self) -> dict[str, JsonValue]:
        with self._lock:
            content: dict[str, JsonValue] = {
                "version": 2,
                "run_id": self.run_id,
                "scope_hash": self.scope_hash,
                "request_hash": self.request_hash,
                "policy": self.policy.model_dump(mode="json"),
                "revision": self.revision,
                "input_tokens": dict(self._inputs),
                "owner_requests": dict(self._owner_requests),
                "receipts": [
                    r.model_dump(mode="json") for r in self._receipts.values()
                ],
                "seals": [r.model_dump(mode="json") for r in self._seals.values()],
                "invalidated": [
                    r.model_dump(mode="json") for r in self._invalidated.values()
                ],
            }
            return {**content, "integrity": _digest(content)}

    def restore(self, payload: dict[str, JsonValue], ledger: EvidenceLedger) -> None:
        saved = _Checkpoint.model_validate(payload)
        if saved.integrity != _digest(
            {k: v for k, v in payload.items() if k != "integrity"}
        ) or (saved.run_id, saved.scope_hash, saved.request_hash, saved.policy) != (
            self.run_id,
            self.scope_hash,
            self.request_hash,
            self.policy,
        ):
            raise ValueError("Task packet checkpoint fence or integrity changed")
        if any(type(n) is not int or n < 0 for n in saved.input_tokens.values()):
            raise ValueError("Task packet checkpoint input counts changed")
        rows = {(r.owner, r.call_id): r for r in saved.receipts}
        seals = {(r.owner, r.need_id): r for r in saved.seals}
        invalidated = {(r.owner, r.need_id): r for r in saved.invalidated}
        if (
            len(rows) != len(saved.receipts)
            or len(seals) != len(saved.seals)
            or len(invalidated) != len(saved.invalidated)
            or seals.keys() & invalidated.keys()
        ):
            raise ValueError("Task packet checkpoint has duplicate records")
        for row in rows.values():
            raw_call = row.raw.get("call")
            if not isinstance(raw_call, dict) or raw_call.get("call_id") != row.call_id:
                raise ValueError("Task packet checkpoint receipt binding changed")
            actual = ToolReceipt.model_validate(row.raw)
            if _receipt_numbers(actual) != {s.citation for s in row.sources} or (
                _outcome_ids(actual.call.arguments) != row.outcome_ids
            ):
                raise ValueError(
                    "Task packet checkpoint receipt source binding changed"
                )
            for source in row.sources:
                if _source(source.citation, ledger) != source:
                    raise ValueError("Task packet checkpoint original version changed")
            if row.composed_steps:
                raw_steps = actual.call.arguments.get("steps")
                results = actual.outcome.data.get("steps")
                if (
                    actual.call.name != "compose_tool_calls"
                    or not isinstance(raw_steps, list)
                    or not isinstance(results, dict)
                    or _outcome_ids(actual.call.arguments)
                    or actual.call.arguments.get("_coverage")
                ):
                    raise ValueError("Task packet composed parent shape changed")
                definitions = {
                    step.get("id"): step for step in raw_steps if isinstance(step, dict)
                }
                if (
                    len(definitions) != len(raw_steps)
                    or definitions.keys()
                    != {step.step_id for step in row.composed_steps}
                    or len({step.child_call_id for step in row.composed_steps})
                    != len(row.composed_steps)
                ):
                    raise ValueError("Task packet composed step identity changed")
                from onyx.asv3.cropped_composed_topics import (
                    _has_ref,
                    source_free_invalid_data,
                    topic_step_continuation,
                    topic_step_partitionable,
                )

                for step in row.composed_steps:
                    definition = definitions[step.step_id]
                    arguments = definition.get("arguments", {})
                    captured_call = CapabilityCall.model_validate(step.captured_call)
                    result = results.get(step.step_id)
                    result_data = (
                        result.get("data") if isinstance(result, dict) else None
                    )
                    source_free_invalid = (
                        isinstance(result, dict)
                        and result.get("status") == "invalid"
                        and isinstance(result_data, dict)
                        and source_free_invalid_data(result_data)
                        and not result.get("original_reads")
                        and not step.citations
                        and not step.sources
                        and not step.captured_originals
                        and not step.references
                    )
                    if (
                        not isinstance(arguments, dict)
                        or _has_ref(definition)
                        or definition.get("depends_on")
                        or definition.get("when")
                        or definition.get("tool") == "compose_tool_calls"
                        or captured_call.name != definition.get("tool")
                        or captured_call.call_id != step.child_call_id
                        or captured_call.arguments != arguments
                        or captured_call.argument_error is not None
                        or not isinstance(result, dict)
                        or not isinstance(result_data, dict)
                        or (
                            not source_free_invalid
                            and result.get("status")
                            not in {"found", "partial", "not_found", "ambiguous"}
                        )
                        or any(
                            step.continuation.get(key) != value
                            for key, value in topic_step_continuation(
                                result_data
                            ).items()
                        )
                        or step.partitionable
                        and not topic_step_partitionable(
                            str(result.get("status")),
                            step.continuation,
                            captured_call.name,
                        )
                        or step.capture_hash
                        != _capture_hash(step, row.owner, row.call_id)
                        or {_digest(original) for original in step.captured_originals}
                        != {
                            _digest(_captured_source(source)) for source in step.sources
                        }
                        or _digest(results.get(step.step_id)) != step.parent_step_hash
                        or (
                            step.need_ids
                            and step.need_ids != [arguments.get("_need_id")]
                        )
                        or _outcome_ids(arguments) != step.outcome_ids
                        or step.citations
                        != [source.citation for source in step.sources]
                        or any(
                            _source(source.citation, ledger) != source
                            for source in step.sources
                        )
                        or step.references
                        != _research_references(
                            captured_call.name,
                            str(result.get("status")),
                            result_data,
                            ledger,
                        )
                    ):
                        raise ValueError("Task packet composed step binding changed")
                if (
                    {n for step in row.composed_steps for n in step.citations}
                    != {
                        *actual.evidence_ids,
                        *(read.citation for read in actual.outcome.original_reads),
                    }
                    or {
                        n
                        for step in row.composed_steps
                        for n in [
                            *step.citations,
                            *(r.source.citation for r in step.references),
                        ]
                    }
                    != {source.citation for source in row.sources}
                    or sorted(
                        {
                            key
                            for step in row.composed_steps
                            for key in _step_need_ids(step)
                        }
                    )
                    != row.need_ids
                ):
                    raise ValueError("Task packet composed source ownership changed")
        with self._lock:
            self._receipts, self._seals = rows, seals
            self._invalidated = invalidated
            self._inputs = dict(saved.input_tokens)
            self._owner_requests = dict(saved.owner_requests)
            self.revision = saved.revision
            self._committed_compositions.clear()
            self.composed_capture.clear()


def _request(context: RunContext, fallback: str) -> str:
    request = context.services.get("cropped_user_context")
    return request if isinstance(request, str) and request else fallback


def _closure(focus: set[str], needs: dict[str, ResearchNeed]) -> set[str]:
    selected = set(focus)
    while True:
        extended = selected | {d for key in selected for d in needs[key].depends_on}
        if extended - needs.keys():
            raise ValueError("Task packet dependency references an unknown need")
        if extended == selected:
            return selected
        selected = extended


def _receipt_text_transport(
    receipts: list[dict[str, JsonValue]],
    numbers: set[int],
    ledger: EvidenceLedger,
) -> tuple[list[dict[str, JsonValue]], dict[str, JsonValue] | None]:
    """Factor only original-bound text; unknown payload fields remain literal."""

    def collision(value: JsonValue) -> bool:
        if isinstance(value, dict):
            return _TEXT_REF in value or any(collision(v) for v in value.values())
        return isinstance(value, list) and any(collision(v) for v in value)

    if any(collision(row) for row in receipts):
        return copy.deepcopy(receipts), None
    items = {n: ledger.get(n) for n in numbers}
    count = 0

    def visit(value: JsonValue) -> JsonValue:
        nonlocal count
        if isinstance(value, list):
            return [visit(child) for child in value]
        if not isinstance(value, dict):
            return value
        result = {key: visit(child) for key, child in value.items()}
        citation = value.get("citation")
        matches = (
            [citation]
            if type(citation) is int and citation in items
            else [
                n
                for n, item in items.items()
                if item is not None
                and all(
                    value.get(key) == getattr(item, key)
                    for key in ("source_id", "chunk_id", "text_hash")
                )
            ]
        )
        if len(matches) != 1:
            return result
        number = cast(int, matches[0])
        item = items[number]
        assert item is not None
        start = value.get("start_char", 0)
        for key in ("text", "page_content", "content"):
            text = value.get(key)
            if (
                isinstance(text, str)
                and text
                and type(start) is int
                and start >= 0
                and item.text[start : start + len(text)] == text
            ):
                result[key] = {_TEXT_REF: [number, start, start + len(text)]}
                count += 1
        return result

    projected = [cast(dict[str, JsonValue], visit(row)) for row in receipts]
    return projected, {
        "version": 1,
        "reference_count": count,
        "instruction": (
            "Within local_receipts only, cropped_original_text_ref:[citation,start_char,end_char] "
            "means the exact character slice of this same message's literal original_evidence "
            "row. All referenced original bodies are actually included in this packet. "
            "The marker replaces only that text field; preserve every other field, status, "
            "cursor and source identity. It does not mark an article complete or approve law."
        ),
    } if count else None


def expand_task_packet_receipts(
    manifest: dict[str, JsonValue], originals: Sequence[dict[str, JsonValue]]
) -> list[dict[str, JsonValue]]:
    """Reconstruct from the actual same-message originals, never a hidden ledger."""
    raw = manifest.get("local_receipts")
    hashes = manifest.get("local_receipt_hashes")
    if (
        not isinstance(raw, list)
        or not isinstance(hashes, list)
        or len(raw) != len(hashes)
    ):
        raise ValueError("Task packet receipt proof is missing")
    texts: dict[int, str] = {}
    for row in originals:
        number, text = row.get("citation"), row.get("text")
        if (
            type(number) is not int
            or not isinstance(text, str)
            or row.get("text_hash") != hashlib.sha256(text.encode()).hexdigest()
            or number in texts
        ):
            raise ValueError("Task packet actual original text is not canonical")
        texts[number] = text
    header = manifest.get("receipt_text_transport")
    count = 0
    if header is not None and (
        not isinstance(header, dict)
        or set(header) != {"version", "reference_count", "instruction"}
        or type(header.get("version")) is not int
        or header["version"] != 1
        or type(header.get("reference_count")) is not int
        or not isinstance(header.get("instruction"), str)
    ):
        raise ValueError("Task packet receipt transport is invalid")

    def visit(value: JsonValue) -> JsonValue:
        nonlocal count
        if isinstance(value, list):
            return [visit(child) for child in value]
        if not isinstance(value, dict):
            return value
        if header is not None and _TEXT_REF in value:
            ref = value[_TEXT_REF]
            if (
                set(value) != {_TEXT_REF}
                or not isinstance(ref, list)
                or len(ref) != 3
                or any(type(n) is not int for n in ref)
            ):
                raise ValueError("Task packet original reference is invalid")
            number, start, end = cast(list[int], ref)
            text = texts.get(number)
            if text is None or not 0 <= start < end <= len(text):
                raise ValueError(
                    "Task packet original reference is unread or out of range"
                )
            count += 1
            return text[start:end]
        return {key: visit(child) for key, child in value.items()}

    result: list[dict[str, JsonValue]] = []
    for row, proof in zip(raw, hashes, strict=True):
        expanded = visit(row)
        if (
            not isinstance(expanded, dict)
            or not isinstance(proof, dict)
            or set(proof) != {"call_id", "sha256"}
            or _digest(expanded) != proof.get("sha256")
        ):
            raise ValueError("Task packet reconstructed receipt changed")
        parsed = ToolReceipt.model_validate(expanded)
        if parsed.call.call_id != proof.get("call_id"):
            raise ValueError("Task packet reconstructed receipt owner changed")
        result.append(expanded)
    if isinstance(header, dict) and count != header["reference_count"]:
        raise ValueError("Task packet original reference count changed")
    return result


def restore_cropped_topic_context(
    view: HarnessView, context: RunContext
) -> HarnessView:
    """Restore actual parked observations for writing, without approving a topic."""
    packet = view.research_state.get("cropped_task_packet")
    if not _enabled(context) or not isinstance(packet, dict):
        return view
    store = context.services.get(_SERVICE)
    ledger = context.services.get("evidence")
    if not isinstance(store, CroppedTaskPacketStore) or not isinstance(
        ledger, EvidenceLedger
    ):
        raise ValueError("Parked topic restoration requires its actual host store")
    store.check_context(context, _request(context, view.request))
    if (
        packet.get("packet_hash")
        != _digest({k: v for k, v in packet.items() if k != "packet_hash"})
        or packet.get("run_id") != store.run_id
        or packet.get("scope_hash") != store.scope_hash
        or packet.get("request_hash") != store.request_hash
        or packet.get("owner") != _owner(context)
    ):
        raise ValueError("Parked topic restoration scope or packet changed")
    with store._lock:
        rows = [
            row.model_copy(deep=True)
            for row in store._receipts.values()
            if not context.depth or row.owner == _owner(context)
        ]
    retained_calls = {row.call_id for row in rows if row.owner == _owner(context)}
    for receipt in view.receipts:
        if receipt.call.call_id not in retained_calls:
            rows.append(
                _Receipt(
                    owner=_owner(context),
                    call_id=receipt.call.call_id,
                    need_ids=[],
                    outcome_ids=_outcome_ids(receipt.call.arguments),
                    sources=[
                        _source(n, ledger) for n in sorted(_receipt_numbers(receipt))
                    ],
                    raw=receipt.model_dump(mode="json"),
                )
            )
            retained_calls.add(receipt.call.call_id)
    raw = [copy.deepcopy(row.raw) for row in rows]
    originals = set(view.required_evidence_numbers)
    for row in rows:
        parsed = ToolReceipt.model_validate(row.raw)
        if parsed.call.call_id != row.call_id:
            raise ValueError("Parked receipt identity changed")
        for source in row.sources:
            if _source(source.citation, ledger) != source:
                raise ValueError("Parked receipt source version changed")
            originals.add(source.citation)
    observations = packet.get("assistant_observations", [])
    if not isinstance(observations, list) or any(
        not isinstance(row, dict) for row in observations
    ):
        raise ValueError("Parked assistant observations are invalid")
    observations = copy.deepcopy(observations)
    observed = {_digest(row) for row in observations}
    receipt_ids = {row.call_id for row in rows}
    for turn in view.turns:
        calls = turn.assistant.tool_calls or []
        observation: dict[str, JsonValue] = {
            "assistant": turn.assistant.model_dump(mode="json"),
            "unmatched_results": [
                result.model_dump(mode="json")
                for call, result in zip(calls, turn.results, strict=True)
                if call.id not in receipt_ids
            ],
        }
        if _digest(observation) not in observed:
            observations.append(observation)
            observed.add(_digest(observation))
    restored: dict[str, JsonValue] = {
        "version": 1,
        "run_id": store.run_id,
        "scope_hash": store.scope_hash,
        "request_hash": store.request_hash,
        "owner": _owner(context),
        "receipt_owners": [
            {"owner": row.owner, "call_id": row.call_id} for row in rows
        ],
        "local_receipts": raw,
        "local_receipt_hashes": [
            {"call_id": row.call_id, "sha256": _digest(row.raw)} for row in rows
        ],
        "receipt_text_transport": None,
        "assistant_observations": copy.deepcopy(observations),
        "legal_approval": False,
        "instruction": (
            "These are actual restored observations and unapproved assistant reasoning. "
            "An open topic remains open; acquisition status is not legal approval, full "
            "coverage or corpus absence. Preserve every recorded limitation, failure, "
            "pending continuation and source identity when assessing the whole question."
        ),
    }
    restored["context_hash"] = _digest(restored)
    state = copy.deepcopy(view.research_state)
    state.pop("cropped_task_packet", None)
    state[_RESTORED] = restored
    return view.model_copy(
        deep=True,
        update={
            "receipts": [ToolReceipt.model_validate(row) for row in raw],
            "research_state": state,
            "required_evidence_numbers": sorted(originals),
        },
    )


def restored_cropped_topic_observations(
    view: HarnessView,
) -> list[dict[str, JsonValue]] | None:
    restored = view.research_state.get(_RESTORED)
    return (
        [{_RESTORED: copy.deepcopy(restored)}] if isinstance(restored, dict) else None
    )


def project_restored_cropped_observations(
    observations: list[dict[str, JsonValue]],
    originals: Sequence[dict[str, JsonValue]],
    context: RunContext,
) -> list[dict[str, JsonValue]]:
    """Reference source bodies only when this physical packet includes them in full."""
    if not _enabled(context):
        return observations
    ledger = context.services.get("evidence")
    store = context.services.get(_SERVICE)
    if not isinstance(ledger, EvidenceLedger):
        raise ValueError("Restored topic observations require their canonical ledger")
    complete: dict[int, dict[str, JsonValue]] = {}
    for row in originals:
        number = row.get("citation")
        item = ledger.get(number) if type(number) is int else None
        if (
            item is not None
            and type(number) is int
            and row.get("text") == item.text
            and row.get("text_hash") == item.text_hash
            and hashlib.sha256(item.text.encode()).hexdigest() == item.text_hash
            and type(row.get("start_char", 0)) is int
            and row.get("start_char", 0) == 0
            and type(row.get("end_char", len(item.text))) is int
            and row.get("end_char", len(item.text)) == len(item.text)
            and row.get("source_id") == item.source_id
            and row.get("chunk_id") == item.chunk_id
        ):
            complete[number] = row
    result = copy.deepcopy(observations)
    for observation in result:
        restored = observation.get(_RESTORED)
        if not isinstance(restored, dict):
            continue
        if (
            not isinstance(store, CroppedTaskPacketStore)
            or restored.get("run_id") != context.run_id
            or restored.get("scope_hash") != _digest(context.scope)
            or restored.get("request_hash") != store.request_hash
            or restored.get("owner") != _owner(context)
        ):
            raise ValueError("Restored topic observation scope changed")
        store.check_context(context, _request(context, store.request))
        if restored.get("context_hash") != _digest(
            {k: v for k, v in restored.items() if k != "context_hash"}
        ):
            raise ValueError("Restored topic context changed")
        raw = restored.get("local_receipts")
        if not isinstance(raw, list) or any(not isinstance(row, dict) for row in raw):
            raise ValueError("Restored topic receipt context is invalid")
        projected, transport = _receipt_text_transport(
            cast(list[dict[str, JsonValue]], raw), set(complete), ledger
        )
        restored["local_receipts"] = projected
        restored["receipt_text_transport"] = transport
        if expand_task_packet_receipts(restored, list(complete.values())) != raw:
            raise ValueError(
                "Restored topic receipt text transport changed observations"
            )
    return result


def packet_for_view(
    view: HarnessView,
    full_receipts: Sequence[ToolReceipt],
    state: ResearchState,
    context: RunContext,
    ledger: EvidenceLedger,
) -> tuple[HarnessView, dict[str, JsonValue]] | None:
    """Partition bound independent topics; pending work stays open and addressable."""
    if (
        not _enabled(context)
        or view.draft_to_repair is not None
        or view.publication_gap is not None
    ):
        return None
    request = _request(context, view.request)
    store = context.services.get(_SERVICE)
    if store is None:
        store = CroppedTaskPacketStore(context, request)
        context.services[_SERVICE] = store
    if not isinstance(store, CroppedTaskPacketStore):
        raise ValueError("Cropped task packet store has an invalid type")
    store.check_context(context, request)
    # Match the ResearchState's separately captured source fence.
    if (state.run_id, state.scope_hash) != (
        context.run_id,
        hashlib.sha256(json.dumps(context.scope, sort_keys=True).encode()).hexdigest(),
    ):
        raise ValueError("Task packet research state scope changed")
    owner = _owner(context)
    store.record_receipts(owner, full_receipts, state, ledger, context=context)
    snapshot, needs, findings = _state_rows(state)
    explicit = context.services.get("task_need_ids")
    focus = explicit if isinstance(explicit, list) else snapshot["active_need_ids"]
    if (
        not isinstance(focus, list)
        or not focus
        or any(not isinstance(k, str) or k not in needs for k in focus)
    ):
        return None
    with store._lock:
        rows = [r for r in store._receipts.values() if r.owner == owner]
        candidates = {s.citation for r in rows for s in r.sources}
        for row in view.original_evidence:
            number, text, start = (
                row.get("citation"),
                row.get("text"),
                row.get("start_char", 0),
            )
            if (
                type(number) is not int
                or not isinstance(text, str)
                or type(start) is not int
            ):
                raise ValueError("Task packet working original has an invalid range")
            item = ledger.get(number)
            if (
                item is None
                or start < 0
                or item.text[start : start + len(text)] != text
                or any(
                    row.get(k) != getattr(item, k)
                    for k in ("source_id", "chunk_id", "text_hash")
                )
            ):
                raise ValueError("Task packet working original identity changed")
            candidates.add(number)
        if (
            sum(store._inputs.values()) < store.policy.cumulative_input_tokens
            and len(candidates) < store.policy.original_records
        ):
            return None
        reopened: set[str] = set()
        for key, seal in list(store._seals.items()):
            if key[0] != owner:
                continue
            need = needs.get(seal.need_id)
            if (
                need is None
                or seal.facts_hash != _digest(view.facts)
                or seal.state_hash
                != store._need_hash(
                    owner,
                    need,
                    needs,
                    findings,
                    ledger,
                    context.services.get("outcome_map"),
                )
            ):
                store._seals.pop(key)
                store._invalidated[key] = seal.model_copy(
                    update={
                        "state_hash": store._need_hash(
                            owner,
                            need,
                            needs,
                            findings,
                            ledger,
                            context.services.get("outcome_map"),
                        )
                        if need
                        else seal.state_hash,
                        "facts_hash": _digest(view.facts),
                        "assessment_hash": store._assessment_hash(
                            owner, need, needs, findings
                        )
                        if need
                        else seal.assessment_hash,
                    }
                )
                reopened.add(seal.need_id)
        if reopened:
            store.revision += 1
        store.seal_eligible(state, context, view.facts, ledger)
        sealed = {key for (bound_owner, key) in store._seals if bound_owner == owner}
        selected = _closure(
            set(cast(list[str], focus)) | (needs.keys() - sealed), needs
        )
        by_need = {
            key: store._need_numbers(owner, need, findings)
            for key, need in needs.items()
        }
        unbound = {number for row in rows for number in _unbound_numbers(row)}
        pending_passthrough = {
            number
            for row in rows
            for step in row.composed_steps
            if not step.partitionable
            for number in step.citations
        }
        reference_passthrough = {
            reference.source.citation
            for row in rows
            for step in row.composed_steps
            for reference in step.references
        }
        bound = {n for values in by_need.values() for n in values}
        protected = {
            *view.required_evidence_numbers,
            *state.source_conditions.citations(),
            *unbound,
            *(candidates - bound),
        }
        outcomes = context.services.get("outcome_map")
        selected_outcomes = {
            o for r in rows if selected.intersection(r.need_ids) for o in r.outcome_ids
        }
        assigned = context.services.get("task_outcome_ids")
        if isinstance(assigned, list):
            selected_outcomes.update(o for o in assigned if isinstance(o, str))
        outcome_view: dict[str, JsonValue] = {}
        known: set[str] = set()
        if isinstance(outcomes, OutcomeMap):
            known = set(outcomes.outcome_ids())
            if selected_outcomes - known:
                selected = set(needs)
                selected_outcomes = known
            # Unbound outcome obligations cannot be detached by question-level guessing.
            bound_outcomes = {o for r in rows for o in r.outcome_ids}
            selected_outcomes.update(known - bound_outcomes)
        while True:
            if isinstance(outcomes, OutcomeMap):
                selected_outcomes.update(
                    o
                    for r in rows
                    if selected.intersection(r.need_ids)
                    for o in r.outcome_ids
                    if o in known
                )
                previous: set[str] = set()
                while selected_outcomes != previous:
                    previous = set(selected_outcomes)
                    exported = outcomes.export()
                    for raw in cast(list[dict[str, JsonValue]], exported["conditions"]):
                        condition = raw["condition"]
                        ids = (
                            condition.get("outcome_ids")
                            if isinstance(condition, dict)
                            else None
                        )
                        if isinstance(ids, list) and selected_outcomes.intersection(
                            cast(list[str], ids)
                        ):
                            selected_outcomes.update(cast(list[str], ids))
                protected.update(
                    outcomes.preferred_citations(outcome_ids=sorted(selected_outcomes))
                )
                outcome_view = outcomes.view(outcome_ids=sorted(selected_outcomes))
            numbers = protected | {n for key in selected for n in by_need[key]}
            extended = _closure(
                selected
                | {key for key in needs if by_need[key].intersection(numbers)}
                | {
                    key
                    for row in rows
                    if set(row.outcome_ids).intersection(selected_outcomes)
                    for key in row.need_ids
                },
                needs,
            )
            if extended == selected:
                break
            selected = extended
        detached = set(needs) - selected
        if not detached:
            return None
        detached_calls = {
            r.call_id
            for r in rows
            if r.need_ids
            and set(r.need_ids) <= detached
            and not any(
                not step.need_ids
                or not step.partitionable
                or step.captured_call.get("name") == "update_research"
                for step in r.composed_steps
            )
        }
        # Visibility of unfinished child work alone does not activate its whole archive.
        # Required, active/shared, unbound and condition sources already closed above.
        numbers.update(
            unbound | (candidates - bound) | pending_passthrough | reference_passthrough
        )
        for row in rows:
            for source in row.sources:
                if _source(source.citation, ledger) != source:
                    raise ValueError("Task packet retained source version changed")
        originals = cast(
            list[dict[str, JsonValue]],
            json.loads(
                ledger.serialize_records(
                    sorted(numbers), required=sorted(numbers), max_chars=None
                )
            ),
        )
        observations: list[JsonValue] = []
        for turn in view.turns:
            calls = turn.assistant.tool_calls or []
            observations.append(
                {
                    "assistant": turn.assistant.model_dump(mode="json"),
                    "unmatched_results": [
                        result.model_dump(mode="json")
                        for call, result in zip(calls, turn.results, strict=True)
                        if call.id not in {r.call_id for r in rows}
                    ],
                }
            )
        local_receipts = [
            _project_composed_receipt(r, selected)
            for r in rows
            if r.call_id not in detached_calls
        ]
        model_receipts, receipt_transport = _receipt_text_transport(
            local_receipts, numbers, ledger
        )
        index = [
            {
                "need": needs[key].model_dump(mode="json"),
                "findings": [
                    f.model_dump(mode="json") for f in findings if f.need_id == key
                ],
                "original_ids": sorted(by_need[key]),
                "receipt_ids": [r.call_id for r in rows if key in r.need_ids],
                "status": (
                    "retained_candidate_not_legal_approval"
                    if needs[key].status == "candidate"
                    else "retained_open_need_not_completed"
                ),
                "pending_operations": [
                    {
                        "call_id": r.call_id,
                        "call": copy.deepcopy(r.raw["call"]),
                        "status": cast(dict[str, JsonValue], r.raw["outcome"]).get(
                            "status"
                        ),
                        "continuation": {
                            marker: value
                            for marker, value in cast(
                                dict[str, JsonValue],
                                cast(dict[str, JsonValue], r.raw["outcome"]).get(
                                    "data", {}
                                ),
                            ).items()
                            if marker
                            in {
                                "has_more",
                                "not_read",
                                "absence_proven",
                                "cursor",
                                "next_position",
                                "evidence_next_position",
                                "next_offset",
                                "scan_truncated",
                                "evidence_truncated",
                                "article_closure_complete",
                            }
                        },
                    }
                    for r in rows
                    if key in r.need_ids and needs[key].status != "candidate"
                ],
                "pending_composed_operations": [
                    {
                        "parent_call_id": row.call_id,
                        "step_id": step.step_id,
                        "child_call_id": step.child_call_id,
                        "need_ids": step.need_ids,
                        "status": _composed_step_status(row, step),
                        "continuation": step.continuation,
                        "original_ids": step.citations,
                        "visible_in_local_receipt": True,
                    }
                    for row in rows
                    for step in row.composed_steps
                    if key in step.need_ids and not step.partitionable
                ],
                "delivered_original_ids": sorted(by_need[key] & numbers),
                "unread_original_ids": sorted(by_need[key] - numbers),
                "originals_unread_this_packet": not bool(by_need[key] & numbers),
                "reopen": "read_evidence / read_research_state or activate this need",
            }
            for key in sorted(detached)
        ]
        manifest: dict[str, JsonValue] = {
            "version": 1,
            "run_id": context.run_id,
            "scope_hash": store.scope_hash,
            "request_hash": store.request_hash,
            "owner": owner,
            "facts_hash": _digest(view.facts),
            "research_revision": state.revision,
            "packet_revision": store.revision,
            "full_question": request,
            "common_facts": list(view.facts),
            "active_need_ids": sorted(selected),
            "selected_outcome_ids": sorted(selected_outcomes),
            "selected_original_ids": sorted(numbers),
            "pending_passthrough_original_ids": sorted(pending_passthrough),
            "reference_passthrough_original_ids": sorted(reference_passthrough),
            "unbound_original_ids": sorted(unbound | (candidates - bound)),
            "detached_need_index": index,
            "reopened_need_ids": sorted(reopened),
            "local_receipts": model_receipts,
            "receipt_text_transport": receipt_transport,
            "local_receipt_hashes": [
                {
                    "call_id": cast(dict[str, JsonValue], mapping["call"])["call_id"],
                    "sha256": _digest(mapping),
                }
                for mapping in local_receipts
            ],
            "assistant_observations": observations,
            "legal_approval": False,
            "topic_scope_instruction": (
                "Work on the active topics and their shared dependencies. Other open topics "
                "remain unfinished in detached_need_index; a parked topic is not answered, "
                "absent, irrelevant or approved. Activate its need ID to restore all its "
                "literal originals and complete operation results, including pending cursors. "
                "Do not close the whole question from this local packet. The final writer "
                "receives the retained originals and must assess every requested outcome."
            ),
        }
        manifest["packet_hash"] = _digest(manifest)
        if expand_task_packet_receipts(manifest, originals) != local_receipts:
            raise ValueError(
                "Task packet receipt transport changed actual source semantics"
            )
        local_state: dict[str, JsonValue] = {
            **snapshot,
            "needs": [
                needs[key].model_dump(mode="json") for key in needs if key in selected
            ],
            "findings": [
                f.model_dump(mode="json") for f in findings if f.need_id in selected
            ],
            "active_need_ids": sorted(selected),
            "cropped_task_packet": manifest,
            "outcome_map": outcome_view,
        }
        selected_view = view.model_copy(
            deep=True,
            update={
                "turns": [],
                "receipts": [ToolReceipt.model_validate(raw) for raw in local_receipts],
                "research_state": local_state,
                "original_evidence": originals,
                "original_evidence_omitted": [
                    {
                        "citation": n,
                        "reason": "retained_independent_topic",
                        "unread_this_packet": True,
                        "reopen": "read_evidence",
                    }
                    for n in sorted(candidates - numbers)
                ],
                "required_evidence_numbers": sorted(numbers),
                "evidence": [
                    row for row in view.evidence if row.get("citation") in numbers
                ],
            },
        )
        if state.export() != snapshot:
            raise ValueError("Task packet research state changed during projection")
        return selected_view, manifest
