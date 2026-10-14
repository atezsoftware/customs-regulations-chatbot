"""One focused evidence attempt per deficiency, followed by owned answer integration."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from onyx.asv3.artifacts import compact_json
from onyx.asv3.assertions import (
    AssertionWitness,
    assertion_inventory,
    assertion_witness_valid,
)
from onyx.asv3.authority import known_statute_source_ids
from onyx.asv3.cheap_authority import canonical_named_authority_gap
from onyx.asv3.cropped_article_bindings import cropped_identity_ledger
from onyx.asv3.cropped_repair_model import fresh_model, integrate_repair
from onyx.asv3.cropped_repair_state import (
    CroppedRepairState,
    CroppedRepairTarget,
    extract_repair_targets,
)
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    OutcomeStatus,
    RunStopped,
)
from onyx.prompts.asv3_cropped_repair import (
    CROPPED_REPAIR_CHECK,
    CROPPED_REPAIR_RESEARCH,
)
from onyx.tracing.answer_graph import graph_step
from onyx.tracing.flows import LLMFlow

if TYPE_CHECKING:
    from onyx.asv3.harness import Harness
    from onyx.asv3.llm_adapter import ResearchModel


_READ_CAPABILITIES = frozenset(
    {
        "read_named_provision",
        "read_provision",
        "read_source_range",
        "resolve_source",
        "search_source_text",
        "search_corpus",
        "query_corpus",
        "read_chunk_context",
        "read_evidence",
        "open_source_file",
    }
)


class _Action(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    arguments: dict[str, JsonValue]


class _ResearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    actions: list[_Action] = Field(max_length=4)
    reason: str = Field(min_length=1)


class _EvidenceCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["found", "not_found", "incomplete"]
    finding: str = Field(min_length=1)
    witnesses: list[AssertionWitness] = Field(max_length=32)
    limitations: list[str] = Field(max_length=32)


def source_gap_sentence(target: CroppedRepairTarget, status: str) -> str:
    name = (
        f"{target.instrument_number} sayılı Kanun"
        if target.instrument_number
        else target.formal_name or target.source_name or ""
    )
    reference = f"{name} {target.qualifier + ' ' if target.qualifier else ''}{target.article}. maddesinin"
    if status == "not_found":
        return f"{reference} özgün metni elde edilemedi."
    return f"Bu denemede {reference} dosyaya uygulanabilirliği doğrulanamadı."


def authority_answer_with_cropped_disclosures(
    answer: str,
    state: CroppedRepairState,
) -> str:
    """Accept only exact standalone host notices for completed unresolved attempts."""
    notices: set[str] = set()
    records = state.export().get("records", [])
    for record in records if isinstance(records, list) else []:
        if not isinstance(record, dict) or record.get("status") not in {
            "not_found",
            "incomplete",
            "failed",
        }:
            continue
        target = CroppedRepairTarget.model_validate(record["target"])
        if target.check_kind == "governing_original" and target.article:
            notices.add(source_gap_sentence(target, str(record["status"])))
    return "\n\n".join(
        unit["text"]
        for unit in assertion_inventory(answer)
        if unit["text"] not in notices
    )


def unresolved_governing_projection(
    draft: str,
    targets: list[CroppedRepairTarget],
    records: list[dict[str, JsonValue]],
) -> str | None:
    """Replace only mapped, exhausted source assertions; never approve the result."""
    units = assertion_inventory(draft)
    known = {unit["unit_id"] for unit in units}
    by_key = {record.get("key"): record for record in records}
    notices: dict[str, str] = {}
    for target in targets:
        record = by_key.get(target.key)
        if (
            target.check_kind != "governing_original"
            or not target.article
            or not target.unit_ids
            or not set(target.unit_ids).issubset(known)
            or record is None
            or record.get("status") not in {"not_found", "incomplete", "failed"}
        ):
            return None
        notice = source_gap_sentence(target, str(record["status"]))
        for identifier in target.unit_ids:
            if identifier in notices and notices[identifier] != notice:
                return None
            notices[identifier] = notice
    if not notices or not any(
        unit["unit_id"] not in notices and unit["evidence_numbers"] for unit in units
    ):
        return None
    emitted: set[str] = set()
    result: list[str] = []
    for unit in units:
        notice = notices.get(unit["unit_id"])
        if notice is None:
            result.append(unit["text"])
        elif notice not in emitted:
            result.append(notice)
            emitted.add(notice)
    return "\n\n".join(result)


def _check_description(value: JsonValue) -> JsonValue:
    if isinstance(value, list):
        return [_check_description(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        key: _check_description(item)
        for key, item in value.items()
        if key
        not in {
            "answer",
            "answer_text",
            "answer_units",
            "draft",
            "draft_to_repair",
            "unit_texts",
            "unit_id",
            "unit_ids",
            "answer_unit_ids",
        }
    }


class CroppedFocusedRepair:
    def __init__(self, state: CroppedRepairState) -> None:
        self.state = state

    @staticmethod
    def _source_name(target: CroppedRepairTarget) -> str:
        return (
            f"{target.instrument_number} sayılı Kanun"
            if target.instrument_number
            else target.formal_name or target.source_name or ""
        )

    @staticmethod
    def _paragraphs(target: CroppedRepairTarget) -> list[str]:
        rows = target.details.get("gap_entries", [])
        paragraphs: set[str] = set()
        for row in rows if isinstance(rows, list) else []:
            entry = row.get("entry") if isinstance(row, dict) else None
            paragraph = entry.get("paragraph") if isinstance(entry, dict) else None
            if isinstance(paragraph, str) and paragraph.isdecimal():
                paragraphs.add(paragraph)
        return sorted(paragraphs, key=lambda number: int(number))

    def _governing_gap(
        self, target: CroppedRepairTarget, harness: Harness, numbers: list[int]
    ) -> dict[str, JsonValue] | None:
        reference = self._source_name(target)
        if target.article:
            reference += f" {target.qualifier + ' ' if target.qualifier else ''}m.{target.article}"
        paragraphs = self._paragraphs(target)
        references = (
            [reference + "/" + paragraph for paragraph in paragraphs]
            if paragraphs
            else [reference]
        )
        for name in references:
            gap = canonical_named_authority_gap(
                name + " uygulanır " + " ".join(f"[{n}]" for n in numbers) + ".",
                cropped_identity_ledger(harness.evidence, harness.context),
                strict_reference_boundaries=True,
                syntactic_reference_binding=True,
                strict_numeric_paragraphs=True,
            )
            if gap is not None:
                return gap
        return None

    def _governing_numbers(
        self, target: CroppedRepairTarget, harness: Harness
    ) -> list[int]:
        source_ids = (
            [target.source_id]
            if target.source_id
            else known_statute_source_ids(self._source_name(target), harness.evidence)
        )
        numbers = []
        for row in cropped_identity_ledger(
            harness.evidence, harness.context
        ).authority_metadata():
            number = row.get("citation")
            item = harness.evidence.get(number) if type(number) is int else None
            if (
                item is not None
                and item.source_id in source_ids
                and row.get("article_no") == target.article
                and row.get("article_qualifier") == target.qualifier
                and type(number) is int
            ):
                numbers.append(number)
        return numbers

    def _paragraph_context_call(
        self, target: CroppedRepairTarget, harness: Harness
    ) -> CapabilityCall | None:
        paragraphs = self._paragraphs(target)
        if not paragraphs:
            return None
        source_ids = (
            [target.source_id]
            if target.source_id
            else known_statute_source_ids(self._source_name(target), harness.evidence)
        )
        numbers = set(target.evidence_numbers)
        rows = target.details.get("gap_entries", [])
        for row in rows if isinstance(rows, list) else []:
            entry = row.get("entry") if isinstance(row, dict) else None
            if isinstance(entry, dict):
                inline = entry.get("inline_evidence")
                numbers.update(
                    n
                    for n in (inline if isinstance(inline, list) else [])
                    if type(n) is int
                )
        candidates = []
        for number in sorted(numbers):
            item = harness.evidence.get(number)
            if item is None or item.source_id not in source_ids or not item.chunk_id:
                continue
            marker = re.match(r"^\s*\((\d{1,3})\)\s+\S", item.text)
            if marker is not None and marker[1] in paragraphs:
                candidates.append(item)
        if len(candidates) != 1:
            return None
        item = candidates[0]
        return CapabilityCall(
            name="read_chunk_context",
            arguments={"source_id": item.source_id, "chunk_id": item.chunk_id},
        )

    def _acquire(
        self, target: CroppedRepairTarget, harness: Harness, model: ResearchModel
    ) -> tuple[list[int], list[dict[str, JsonValue]]]:
        ledger = harness.evidence
        numbers = set(target.evidence_numbers)
        calls: list[CapabilityCall] = []
        governing = target.check_kind == "governing_original" and target.article
        if governing:
            numbers.update(self._governing_numbers(target, harness))
        if governing and self._governing_gap(target, harness, sorted(numbers)):
            source_name = self._source_name(target)
            source_ids = (
                [target.source_id]
                if target.source_id
                else known_statute_source_ids(source_name, ledger)
            )
            article = " ".join(filter(None, (target.qualifier, target.article)))
            if source_name or source_ids:
                name = (
                    "read_provision" if len(source_ids) == 1 else "read_named_provision"
                )
                arguments: dict[str, JsonValue] = {"article": article}
                arguments["source_id" if len(source_ids) == 1 else "source_name"] = (
                    source_ids[0] if len(source_ids) == 1 else source_name
                )
                paragraphs = self._paragraphs(target)
                if len(paragraphs) == 1:
                    arguments["paragraph"] = paragraphs[0]
                context_call = self._paragraph_context_call(target, harness)
                calls = [context_call or CapabilityCall(name=name, arguments=arguments)]
        elif target.source_id and not numbers:
            calls = [
                CapabilityCall(
                    name="read_source_range", arguments={"source_id": target.source_id}
                )
            ]
        elif not numbers:
            definitions = [
                spec.definition()
                for name in sorted(_READ_CAPABILITIES)
                if (spec := harness.registry.get(name)) is not None
                and not spec.external
            ]
            planner = fresh_model(model)
            payload: dict[str, JsonValue] = {
                "request": self.state.request,
                "deficiency": target.model_dump(mode="json"),
                "capabilities": cast(JsonValue, definitions),
            }
            text = planner.invoke_text(
                CROPPED_REPAIR_RESEARCH,
                json.dumps(payload, ensure_ascii=False),
                LLMFlow.ASV3_TARGETED_RESEARCH,
                max_tokens=1800,
                response_model_override=_ResearchPlan,
            )
            plan = _ResearchPlan.model_validate_json(text)
            if any(action.name not in _READ_CAPABILITIES for action in plan.actions):
                raise ValueError(
                    "Focused research can execute only read-only corpus capabilities"
                )
            calls = [
                CapabilityCall(name=action.name, arguments=action.arguments)
                for action in plan.actions
            ]
        observations: list[dict[str, JsonValue]] = []
        if calls:
            for receipt in harness.dispatch_repair_reads(calls):
                numbers.update(receipt.evidence_ids)
                observations.append(
                    {
                        "call_id": receipt.call.call_id,
                        "name": receipt.call.name,
                        "arguments": receipt.call.arguments,
                        "status": receipt.outcome.status.value,
                        "summary": receipt.outcome.summary,
                        "result": compact_json(receipt.outcome.data, max_chars=1500),
                    }
                )
        if governing:
            matching = self._governing_numbers(target, harness)
            if matching:
                numbers = set(target.evidence_numbers) | set(matching)
        return sorted(numbers), observations

    def _check(
        self,
        target: CroppedRepairTarget,
        harness: Harness,
        model: ResearchModel,
        numbers: list[int],
        observations: list[dict[str, JsonValue]],
    ) -> None:
        ledger = harness.evidence
        usable = [
            number
            for number in numbers
            if (item := ledger.get(number)) is not None
            and item.search_doc is not None
            and item.metadata.get("truncated") is not True
            and not (
                isinstance(canonical := item.metadata.get("canonical_metadata"), dict)
                and canonical.get("truncated") is True
            )
        ]
        if not usable:
            failed = any(
                row["status"]
                in {
                    OutcomeStatus.ERROR.value,
                    OutcomeStatus.INVALID.value,
                    OutcomeStatus.DENIED.value,
                    OutcomeStatus.CANCELLED.value,
                    OutcomeStatus.TRUNCATED.value,
                }
                for row in observations
            )
            self.state.finish(
                target.key,
                "failed" if failed else "not_found",
                [],
                json.dumps(
                    {"absence_proven": False, "observations": observations},
                    ensure_ascii=False,
                ),
            )
            return
        originals = json.loads(
            ledger.serialize_records(
                usable,
                required=usable,
                max_chars=None,
                include_witness_spans=True,
            )
        )
        checker = fresh_model(model)
        payload: dict[str, JsonValue] = {
            "request": self.state.request,
            "deficiency": {
                key: value
                for key, value in target.model_dump(mode="json").items()
                if key not in {"unit_ids", "unit_texts", "details"}
            },
            "requested_check": _check_description(
                target.details.get("gap_entries", [])
            ),
            "evidence": originals,
            "research_results": cast(JsonValue, observations),
        }

        def validate(text: str) -> None:
            check = _EvidenceCheck.model_validate_json(text)
            if check.status != "found":
                return
            if not check.witnesses or checker.last_call_id is None:
                raise ValueError(
                    "A verified finding requires delivered original witnesses"
                )
            delivered = ledger.completely_delivered(checker.last_call_id)
            by_number = {
                number: item.text
                for number in usable
                if (item := ledger.get(number)) is not None
            }
            if any(
                witness.citation not in delivered
                or not assertion_witness_valid(witness, by_number)
                for witness in check.witnesses
            ):
                raise ValueError(
                    "Focused finding has no actual delivered original witness"
                )
            if target.check_kind == "governing_original":
                if (
                    self._governing_gap(
                        target, harness, [w.citation for w in check.witnesses]
                    )
                    is not None
                ):
                    raise ValueError(
                        "The governing-original check requires its own matching original"
                    )

        text = checker.invoke_text(
            CROPPED_REPAIR_CHECK,
            json.dumps(payload, ensure_ascii=False),
            LLMFlow.ASV3_TARGETED_EVIDENCE_CHECK,
            max_tokens=3000,
            response_model_override=_EvidenceCheck,
            response_validator=validate,
        )
        result = _EvidenceCheck.model_validate_json(text)
        witnessed = (
            sorted({w.citation for w in result.witnesses})
            if result.status == "found"
            else []
        )
        if witnessed and checker.last_call_id:
            ledger.pin_delivery(checker.last_call_id)
        status = result.status
        if status == "not_found" and target.check_kind == "governing_original":
            if self._governing_gap(target, harness, usable) is None:
                status = "incomplete"
        self.state.finish(
            target.key,
            status,
            witnessed,
            json.dumps(
                {
                    "assessment": result.model_dump(mode="json"),
                    "check_call_id": checker.last_call_id,
                    "absence_proven": False,
                    "observations": observations,
                },
                ensure_ascii=False,
            ),
        )

    def decide(self, harness: Harness, model: ResearchModel) -> Decision:
        previous = harness.context.services.get("cropped_cost_phase")
        harness.context.services["cropped_cost_phase"] = "repair"
        try:
            return self._decide(harness, model)
        finally:
            if previous is None:
                harness.context.services.pop("cropped_cost_phase", None)
            else:
                harness.context.services["cropped_cost_phase"] = previous

    def _exhausted_decision(self, harness: Harness) -> Decision | None:
        """A closed repair can still publish an independently guarded retained answer."""
        if not harness.last_draft or harness.draft_guard is None:
            return None
        gap = harness.draft_guard(harness.last_draft)
        if gap is None:
            previous_terminal = next(
                (
                    receipt.call.name
                    for receipt in reversed(harness.receipts)
                    if receipt.call.name in {"submit_answer", "submit_partial_answer"}
                ),
                None,
            )
            if (
                not harness.context.depth
                and previous_terminal == "submit_partial_answer"
            ):
                return Decision(
                    calls=[
                        CapabilityCall(
                            name="submit_partial_answer",
                            arguments={"answer": harness.last_draft},
                        )
                    ]
                )
            return Decision(answer=harness.last_draft)
        if harness.context.depth:
            return None
        targets = extract_repair_targets(gap.data, harness.last_draft, harness.evidence)
        candidate = unresolved_governing_projection(
            harness.last_draft, targets, self.state.records_for(targets)
        )
        if candidate is None or harness.draft_guard(candidate) is not None:
            return None
        return Decision(
            calls=[
                CapabilityCall(
                    name="submit_partial_answer", arguments={"answer": candidate}
                )
            ]
        )

    def _decide(self, harness: Harness, model: ResearchModel) -> Decision:
        from onyx.asv3.cropped_cost_budget import (
            CroppedCostBudget,
            CroppedRepairBudgetReached,
        )

        budget = harness.context.services.get("cropped_cost_budget")
        if isinstance(budget, CroppedCostBudget) and not budget.repair_allowed:
            finished = self._exhausted_decision(harness)
            if finished is not None:
                return finished
            raise CroppedRepairBudgetReached(
                "Cropped repair cost limit reached; rejected draft was not published"
            )
        view = harness.view()
        if not view.publication_gap or not view.draft_to_repair:
            raise ValueError("Focused repair requires an owned rejected draft")
        targets = extract_repair_targets(
            view.publication_gap, view.draft_to_repair, harness.evidence
        )
        if not targets:
            raise RunStopped(
                "The rejected draft has no focused repair target; originals retained"
            )
        for target in targets:
            if isinstance(budget, CroppedCostBudget) and not budget.repair_allowed:
                raise CroppedRepairBudgetReached(
                    "Cropped repair cost limit reached; no further target was acquired"
                )
            if self.state.reserve(target):
                harness.save_repair_checkpoint()
                with graph_step(
                    "asv3.cropped_focused_research",
                    {"target": target.model_dump(mode="json")},
                ) as step:
                    try:
                        numbers, observations = self._acquire(target, harness, model)
                        self._check(target, harness, model, numbers, observations)
                    except RunStopped:
                        self.state.finish(
                            target.key,
                            "failed",
                            [],
                            "Focused attempt stopped before a verified finding; absence not proven.",
                        )
                        harness.context.check_active()
                        harness.context.check_research_active()
                    except Exception:
                        logging.getLogger(__name__).exception(
                            "Cropped focused evidence attempt failed"
                        )
                        self.state.finish(
                            target.key,
                            "failed",
                            [],
                            "Focused attempt failed before a verified finding; absence not proven.",
                        )
                    step.output_value = {"records": self.state.records_for([target])}
                harness.save_repair_checkpoint()
        owner = str(harness.context.services.get("task_id") or "coordinator")
        if not self.state.reserve_integration(targets, owner):
            finished = self._exhausted_decision(harness)
            if finished is not None:
                return finished
            raise RunStopped(
                "This deficiency already received its focused attempt and answer integration; originals retained"
            )
        harness.save_repair_checkpoint()
        records = self.state.records_for(targets)
        for target, record in zip(targets, records, strict=True):
            if (
                record.get("status") != "found"
                and target.check_kind == "governing_original"
                and target.article
            ):
                record["source_gap_sentence"] = source_gap_sentence(
                    target, str(record["status"])
                )
        return integrate_repair(model, harness.view(), records)
