"""Request-bound research needs and source witnesses, independent of tool history."""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from onyx.asv3.condition_memory import SourceConditionMemory
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import OutcomeStatus, RunContext, ToolOutcome, ToolSpec
from onyx.asv3.scenario import question_determinations


class ResearchNeed(BaseModel):
    model_config = ConfigDict(extra="forbid")
    need_id: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")
    question_ids: list[str] = Field(min_length=1, max_length=20)
    determination_ids: list[str] = Field(
        default_factory=list,
        max_length=40,
        description="Requested determination IDs addressed by this need. Independent outcomes in one numbered question remain distinct; choose related groupings yourself.",
    )
    purpose: str = Field(min_length=1, max_length=1000)
    completion_test: str = Field(min_length=1, max_length=1000)
    kind: Literal[
        "governing_basis",
        "special_rule",
        "procedure",
        "condition",
        "calculation",
        "other",
    ] = "other"
    depends_on: list[str] = Field(default_factory=list, max_length=20)
    status: Literal["open", "researching", "candidate", "blocked", "out_of_scope"] = (
        "open"
    )
    gap: str = Field(default="", max_length=1500)
    material: bool = True
    evidence_numbers: list[Annotated[int, Field(strict=True, ge=1)]] = Field(
        default_factory=list, max_length=40
    )


class SourceWitness(BaseModel):
    model_config = ConfigDict(extra="forbid")
    citation: int = Field(strict=True, ge=1)
    start_char: int = Field(default=0, strict=True, ge=0)
    end_char: int = Field(strict=True, ge=1)


class ResearchFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    finding_id: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")
    need_id: str
    statement: str = Field(min_length=1, max_length=2000)
    kind: Literal["rule", "application", "procedure", "limitation"]
    witnesses: list[SourceWitness] = Field(min_length=1, max_length=12)


class ResearchUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    needs: list[ResearchNeed] = Field(default_factory=list, max_length=20)
    findings: list[ResearchFinding] = Field(default_factory=list, max_length=24)
    active_need_ids: list[str] | None = Field(default=None, max_length=20)


class ResearchCheckpointUpdate(ResearchUpdate):
    needs: list[ResearchNeed] = Field(default_factory=list, max_length=64)
    findings: list[ResearchFinding] = Field(default_factory=list, max_length=128)


class ResearchState:
    def __init__(
        self,
        questions: list[str],
        context: RunContext,
        *,
        require_need_bindings: bool = False,
    ) -> None:
        self.run_id = context.run_id
        self.scope_hash = hashlib.sha256(
            json.dumps(context.scope, sort_keys=True).encode()
        ).hexdigest()
        self.questions = tuple(questions)
        self.source_conditions = SourceConditionMemory(
            self.run_id, self.scope_hash, questions
        )
        self.require_need_bindings = require_need_bindings
        self._lock = threading.RLock()
        self._needs: OrderedDict[str, ResearchNeed] = OrderedDict()
        self._findings: OrderedDict[str, ResearchFinding] = OrderedDict()
        self._active: list[str] = []
        self.revision = 0

    def question_ids(self, need_id: str) -> list[str]:
        with self._lock:
            need = self._needs.get(need_id)
            return list(need.question_ids) if need else []

    def has_need(self, need_id: str) -> bool:
        with self._lock:
            return need_id in self._needs

    def action_binding_gap(self, need_id: JsonValue) -> ToolOutcome | None:
        if not self.require_need_bindings:
            return None
        with self._lock:
            need = self._needs.get(need_id) if isinstance(need_id, str) else None
            if need is not None and need.material and need.status != "out_of_scope":
                return None
        return ToolOutcome(
            status=OutcomeStatus.INVALID,
            summary="Research needs an existing material information need; no source action was executed.",
            data={
                "research_binding_error": True,
                "instruction": "Use update_research to record the unresolved question, purpose and observable completion test, then bind source actions with _need_id or delegation with need_ids. The update and independent actions may share one decision. Choose the methods and queries yourself.",
                "questions": [
                    {"question_id": f"q{i}", "question": text}
                    for i, text in enumerate(self.questions)
                ],
            },
        )

    def uncovered_questions(self) -> list[str]:
        with self._lock:
            covered = {
                question_id
                for need in self._needs.values()
                if need.material and need.status != "out_of_scope"
                for question_id in need.question_ids
            }
            return [
                f"q{i}" for i in range(len(self.questions)) if f"q{i}" not in covered
            ]

    def update(self, update: ResearchUpdate, ledger: EvidenceLedger) -> None:
        with self._lock:
            needs = OrderedDict(self._needs)
            findings = OrderedDict(self._findings)
            known_questions = {f"q{i}" for i in range(len(self.questions))}
            for need in update.needs:
                if need.status == "out_of_scope" and not need.gap.strip():
                    raise ValueError(
                        "Excluded needs require an applicability explanation"
                    )
                if set(need.question_ids) - known_questions:
                    raise ValueError(
                        "Research needs must bind to original question IDs"
                    )
                bound_determinations = {
                    item["determination_id"]
                    for item in question_determinations(list(self.questions))
                    if item["question_id"] in need.question_ids
                }
                if set(need.determination_ids) - bound_determinations:
                    raise ValueError(
                        "Research determinations must belong to their original question IDs"
                    )
                previous = needs.get(need.need_id)
                if previous and previous.question_ids != need.question_ids:
                    raise ValueError(
                        "An existing need cannot move to different questions"
                    )
                if any(ledger.get(n) is None for n in need.evidence_numbers):
                    raise ValueError(
                        "Research need references unknown original evidence"
                    )
                needs[need.need_id] = need.model_copy(deep=True)
            if len(needs) > 64:
                raise ValueError("Research need capacity exceeded")
            settled: set[str] = set()
            remaining = set(needs)
            while remaining:
                ready = {
                    key for key in remaining if set(needs[key].depends_on) <= settled
                }
                if not ready:
                    raise ValueError("Research dependencies are unknown or cyclic")
                settled.update(ready)
                remaining -= ready
            for finding in update.findings:
                previous_finding = findings.get(finding.finding_id)
                if previous_finding and previous_finding.need_id != finding.need_id:
                    raise ValueError("An existing finding cannot move to another need")
                if finding.need_id not in needs:
                    raise ValueError("Finding must bind to an existing research need")
                for witness in finding.witnesses:
                    item = ledger.get(witness.citation)
                    if item is None or not (
                        witness.start_char < witness.end_char <= len(item.text)
                    ):
                        raise ValueError(
                            "Finding witness is outside recorded original text"
                        )
                findings[finding.finding_id] = finding.model_copy(deep=True)
            if len(findings) > 128:
                raise ValueError("Research finding capacity exceeded")
            active = (
                self._active
                if update.active_need_ids is None
                else update.active_need_ids
            )
            if set(active) - needs.keys():
                raise ValueError("Active research need is unknown")
            if (
                needs != self._needs
                or findings != self._findings
                or active != self._active
            ):
                self._needs, self._findings, self._active = (
                    needs,
                    findings,
                    list(active),
                )
                self.revision += 1

    def attach(self, need_id: str, numbers: list[int]) -> None:
        with self._lock:
            need = self._needs.get(need_id)
            if need is None:
                return
            merged = list(dict.fromkeys([*need.evidence_numbers, *numbers]))[:40]
            if merged != need.evidence_numbers:
                self._needs[need_id] = need.model_copy(
                    update={"evidence_numbers": merged}
                )
                self.revision += 1

    def preferred_citations(self, focus: list[str] | None = None) -> list[int]:
        with self._lock:
            ordered = [
                *(key for key in focus or [] if key in self._needs),
                *self._active,
                *(key for key, need in self._needs.items() if need.material),
                *self._needs,
            ]
            return list(
                dict.fromkeys(
                    [
                        *self.source_conditions.citations(),
                        *(
                            n
                            for key in dict.fromkeys(ordered)
                            for n in [
                                *(
                                    w.citation
                                    for f in self._findings.values()
                                    if f.need_id == key
                                    for w in f.witnesses
                                ),
                                *self._needs[key].evidence_numbers,
                            ]
                        ),
                    ]
                )
            )

    def view(self, *, max_chars: int = 18000) -> dict[str, JsonValue]:
        with self._lock:
            result: dict[str, JsonValue] = {
                "revision": self.revision,
                "need_bindings_required": self.require_need_bindings,
                "questions": [
                    {"question_id": f"q{i}", "question": q}
                    for i, q in enumerate(self.questions)
                ],
                "active_need_ids": list(self._active),
                "determinations": [
                    cast(dict[str, JsonValue], dict(item))
                    for item in question_determinations(list(self.questions))
                ],
                "needs": [],
                "findings": [],
                "source_conditions": [],
                "source_conditions_omitted": 0,
                "needs_omitted": 0,
                "findings_omitted": 0,
                "reopen": "inspect_research with a larger max_chars; originals use global citation numbers",
                "notice": "Findings are source-bound candidates, not verified law. Original witnesses must be delivered and evaluated before publication.",
            }
            order = list(dict.fromkeys([*self._active, *self._needs]))
            need_rows = [self._needs[key] for key in order]
            finding_rows = [
                f for key in order for f in self._findings.values() if f.need_id == key
            ]
            # Requirements survive context pressure ahead of descriptive findings.
            condition_rows = self.source_conditions.required_conditions()
            for field, rows in (
                ("source_conditions", condition_rows),
                ("needs", need_rows),
                ("findings", finding_rows),
            ):
                selected: list[JsonValue] = []
                for row in rows:
                    candidate = (
                        row.model_dump(mode="json")
                        if isinstance(row, (ResearchNeed, ResearchFinding))
                        else row
                    )
                    trial = {**result, field: [*selected, candidate]}
                    if len(json.dumps(trial, ensure_ascii=False)) > max_chars:
                        break
                    selected.append(candidate)
                result[field] = selected
                result[field + "_omitted"] = len(rows) - len(selected)
            return result

    def export(self) -> dict[str, JsonValue]:
        with self._lock:
            return {
                "version": 1,
                "run_id": self.run_id,
                "scope_hash": self.scope_hash,
                "questions": list(self.questions),
                "revision": self.revision,
                "needs": [n.model_dump(mode="json") for n in self._needs.values()],
                "findings": [
                    f.model_dump(mode="json") for f in self._findings.values()
                ],
                "active_need_ids": list(self._active),
                "source_conditions": self.source_conditions.export(),
            }

    def restore(self, data: dict[str, JsonValue], ledger: EvidenceLedger) -> None:
        if (
            data.get("version"),
            data.get("run_id"),
            data.get("scope_hash"),
            data.get("questions"),
        ) != (1, self.run_id, self.scope_hash, list(self.questions)):
            raise ValueError("Research state identity or request mismatch")
        revision = data.get("revision", 0)
        if type(revision) is not int or revision < 0:
            raise ValueError("Invalid research revision")
        self.update(
            ResearchCheckpointUpdate.model_validate(
                {
                    key: data.get(key, [])
                    for key in ("needs", "findings", "active_need_ids")
                }
            ),
            ledger,
        )
        self.revision = revision
        conditions = data.get("source_conditions")
        if isinstance(conditions, dict):
            self.source_conditions.restore(conditions, ledger)


class EvidenceWorkingSet:
    """Keep exact locators and hydrate them within the caller's context policy."""

    def __init__(self, *, max_ranges: int | None = 128) -> None:
        if max_ranges is not None and max_ranges < 1:
            raise ValueError("Evidence working set capacity must be positive")
        self.max_ranges = max_ranges
        self._ranges: OrderedDict[tuple[int, int, int], None] = OrderedDict()

    def remember(self, number: int, start: int, end: int) -> None:
        if end <= start:
            return
        covered = next(
            (
                key
                for key in self._ranges
                if key[0] == number and key[1] <= start and key[2] >= end
            ),
            None,
        )
        if covered:
            self._ranges.move_to_end(covered)
            return
        for key in list(self._ranges):
            if key[0] == number and start <= key[1] and end >= key[2]:
                del self._ranges[key]
        self._ranges[number, start, end] = None
        while self.max_ranges is not None and len(self._ranges) > self.max_ranges:
            self._ranges.popitem(last=False)

    def view(
        self,
        ledger: EvidenceLedger,
        *,
        preferred: list[int],
        required: list[int] | None = None,
        max_chars: int | None = 32000,
    ) -> dict[str, JsonValue]:
        from onyx.asv3.models import model_evidence_metadata

        ranges = list(reversed(self._ranges))
        ordered = list(
            dict.fromkeys(
                [
                    *(
                        (n, 0, len(item.text))
                        for n in required or []
                        if (item := ledger.get(n)) is not None
                    ),
                    *ranges,
                    *(
                        (n, 0, len(item.text))
                        for n in preferred
                        if (item := ledger.get(n)) is not None
                    ),
                ]
            )
        )
        records: list[JsonValue] = []
        omitted: list[JsonValue] = []
        included: list[tuple[int, int, int]] = []
        used = 2
        for number, start, end in ordered:
            if any(n == number and a <= start and b >= end for n, a, b in included):
                continue
            item = ledger.get(number)
            if item is None:
                continue
            record: dict[str, JsonValue] = {
                "citation": number,
                "source_id": item.source_id,
                "chunk_id": item.chunk_id,
                "text_hash": item.text_hash,
                "text": item.text[start:end],
                "start_char": start,
                "end_char": end,
                "total_chars": len(item.text),
                "truncated": start != 0 or end != len(item.text),
                "metadata": model_evidence_metadata(item.metadata),
            }
            if max_chars is not None:
                cost = len(json.dumps(record, ensure_ascii=False))
                if used + cost > max_chars:
                    omitted.append(
                        {"citation": number, "start_char": start, "end_char": end}
                    )
                    continue
                used += cost
            records.append(record)
            included.append((number, start, end))
        return {"records": records, "omitted": omitted, "reopen": "read_evidence"}

    def export(self) -> list[JsonValue]:
        return [[n, start, end] for n, start, end in self._ranges]

    def restore(self, rows: list[JsonValue], ledger: EvidenceLedger) -> None:
        if self.max_ranges is not None and len(rows) > self.max_ranges:
            raise ValueError("Evidence working set capacity exceeded")
        for row in rows:
            if (
                not isinstance(row, list)
                or len(row) != 3
                or not all(type(v) is int for v in row)
            ):
                raise ValueError("Invalid evidence working range")
            number, start, end = (int(str(value)) for value in row)
            item = ledger.get(number)
            if item is None or not 0 <= start < end <= len(item.text):
                raise ValueError("Evidence working range is not an original locator")
            self.remember(number, start, end)


def build_research_specs(
    state: ResearchState, ledger: EvidenceLedger
) -> list[ToolSpec]:
    def update(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        context.check_active()
        try:
            state.update(ResearchUpdate.model_validate(args), ledger)
        except ValueError as error:
            return ToolOutcome(status=OutcomeStatus.INVALID, summary=str(error))
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Research needs and source witnesses retained",
            data=state.view(max_chars=9000),
        )

    def inspect(args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Request-bound research state; findings are not verification",
            data=state.view(max_chars=int(str(args.get("max_chars", 24000)))),
        )

    return [
        ToolSpec(
            name="update_research",
            orchestrates=True,
            description="Update stable question-bound information needs and source-witnessed findings together. Record prerequisites, exceptions, procedures and governing references; choose methods yourself. Candidate findings do not constitute verification.",
            parameters=ResearchUpdate.model_json_schema(),
            handler=update,
        ),
        ToolSpec(
            name="inspect_research",
            orchestrates=True,
            description="Inspect retained research needs, dependencies, gaps and source-witnessed findings, without replaying tool history.",
            parameters={
                "type": "object",
                "properties": {
                    "max_chars": {"type": "integer", "minimum": 1000, "maximum": 100000}
                },
                "additionalProperties": False,
            },
            handler=inspect,
        ),
    ]
