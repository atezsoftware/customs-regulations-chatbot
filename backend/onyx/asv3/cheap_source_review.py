"""Cheap-only independent source obligations with actual original-delivery receipts."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from onyx.asv3.assertions import AssertionWitness, assertion_inventory
from onyx.asv3.condition_memory import SourceConditionMemory
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import (
    MaterialSourceOmission,
    ResearchModel,
    SourceConditionAuditResult,
    SourceConditionCheck,
    SourceConditionResolution,
    StructuredOutputError,
)
from onyx.asv3.models import OutcomeStatus, RunContext, RunStopped, ToolOutcome
from onyx.asv3.quotations import normalized
from onyx.asv3.scenario import question_determinations
from onyx.asv3.source_conditions import condition_omissions, condition_review_defects
from onyx.asv3.witnesses import original_witness_text
from onyx.prompts.asv3.research import (
    SOURCE_CONDITION_PROMPT,
    SOURCE_REQUIREMENT_PROMPT,
)
from onyx.tracing.flows import LLMFlow


class CompactSourceRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    citation: int = Field(strict=True, ge=1)
    source_quote: str = Field(min_length=1, max_length=800)
    determination_ids: list[str] = Field(min_length=1, max_length=40)
    detail: str = Field(min_length=1, max_length=600)
    applicability: str = Field(min_length=1, max_length=600)

    def materialize(self, ledger: EvidenceLedger) -> MaterialSourceOmission:
        original = ledger.get(self.citation)
        if original is None or original.text.count(self.source_quote) != 1:
            raise ValueError(
                "A compact requirement needs a unique exact literal quote in its original"
            )
        return MaterialSourceOmission(
            witness=AssertionWitness(
                citation=self.citation, source_quote=self.source_quote
            ),
            determination_ids=self.determination_ids,
            detail=self.detail,
            applicability=self.applicability,
        )


class CheapSourceRequirementInventory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_scope_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    whole_scope_completed: bool = Field(strict=True)
    requirements: list[CompactSourceRequirement]


class MissingSourceDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_quote: str = Field(min_length=1, max_length=320)
    detail: str = Field(min_length=1, max_length=300)


class CheapConditionCheck(SourceConditionCheck):
    missing_details: list[MissingSourceDetail] = Field(
        default_factory=list, max_length=8
    )
    affected_answer_unit_ids: list[str] = Field(default_factory=list, max_length=32)


class CheapConditionResolution(SourceConditionResolution):
    missing_details: list[MissingSourceDetail] = Field(
        default_factory=list, max_length=8
    )
    affected_answer_unit_ids: list[str] = Field(default_factory=list, max_length=32)


class CheapConditionAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_scope_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    whole_scope_completed: bool = Field(strict=True)
    conditions: list[CheapConditionCheck] = Field(max_length=256)
    resolutions: list[CheapConditionResolution] = Field(
        default_factory=list, max_length=256
    )

    def base(self, examined_citations: list[int]) -> SourceConditionAuditResult:
        excluded = {"missing_details", "affected_answer_unit_ids"}
        return SourceConditionAuditResult(
            examined_citations=examined_citations,
            conditions=[
                SourceConditionCheck.model_validate(row.model_dump(exclude=excluded))
                for row in self.conditions
            ],
            resolutions=[
                SourceConditionResolution.model_validate(
                    row.model_dump(exclude=excluded)
                )
                for row in self.resolutions
            ],
        )


class CheapSourceReviewCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    scenario_hash: str
    repair_requested: bool = Field(strict=True)
    memory: dict[str, JsonValue]
    source_bindings: list[dict[str, JsonValue]] = Field(max_length=256)


def _identity(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


_CATALOGUE_NOTE = """
original_metadata_catalogue preserves each citation's canonical source identity and metadata.
When original_metadata_codec is supplied, use its versioned fields, layouts and value pool
to read each original's identity_ref according to that codec. original_evidence
contains the unchanged literal originals; catalogue metadata is navigation, never legal proof.
No witness_spans IDs are supplied in this assessment: if the schema has witness_id, set it
to null. Select a
short exact contiguous source_quote of at most 800 characters from the actual original.
Never calculate or invent witness IDs. An exact original quote establishes the witness.
"""

_INVENTORY_NOTE = """
For this inventory, declare the supplied source_scope_id exactly and whole_scope_completed=true
only after assessing ALL supplied complete originals. Delivery alone does not prove assessment.
If the whole scope was not assessed, declare false; partial processing cannot approve inventory.
Use the compact requirement schema with flat citation and source_quote fields. Prefer the
shortest unique exact contiguous quote establishing each material condition in its original.
Keep detail/applicability concise but preserve ALL material actor, trigger, scope, exception and
procedural qualifications; do not omit conditions to fit an output limit or assume them from a
headline result. Do not calculate witness IDs, repeat source lists or recopy full passages.
"""

_REVIEW_SCOPE_NOTE = """
Declare the supplied source_scope_id exactly and whole_scope_completed=true only after
assessing ALL supplied complete originals against this exact candidate and retained conditions.
Delivery alone does not prove assessment. Declare false if any supplied source was not assessed;
partial processing cannot approve this review. Do not repeat a list of examined citations.
"""


def _cheap_prompt(prompt: str) -> str:
    return (
        prompt.replace(
            "Return examined_citations covering exactly every supplied original citation.",
            "",
        )
        .replace(
            "Return examined_citations covering exactly every supplied original citation, and conditions",
            "Return conditions",
        )
        .replace(
            "For each requirement select its actual supplied witness_id and determination_ids.",
            "For each requirement select its actual literal source_quote and determination_ids.",
        )
        .replace(
            "Select the supplied witness_id for the actual operative condition.",
            "Select an exact literal source_quote for the actual operative condition.",
        )
    )


_DELTA_NOTE = """
condition_unit_navigation lists current units with each retained original's inline citation.
This is navigation only, never proof of legal coverage. Assess the complete union of those
units and the full candidate before calling a requirement absent; a missing detail in one
summary may be stated in its detailed unit. Keep every immutable retained requirement intact.
For EACH omitted or uncertain finding, give missing_details identifying ONLY its actual open
components, with short exact source subquotes from its retained or selected operative witness.
Select affected_answer_unit_ids from the current candidate: the units needing correction or
the current relevant block where a missing branch should be inserted. Preserve already covered
components and explain the precise missing actor, proof, procedure, trigger, exception or fact.
Do not repeat the whole rule when only a component is absent. Covered/not_applicable findings
must have empty missing_details and affected_answer_unit_ids. A reference alone cannot prove
the contents of unread governing provisions or create additional requirements from memory.
"""


class CheapSourceReviewPolicy:
    def __init__(
        self, context: RunContext, scenario: str, questions: list[str]
    ) -> None:
        if context.services.get("asv3_exp_cheap") is not True or context.depth:
            raise ValueError(
                "Independent Cheap source review requires its root workflow"
            )
        self.context = context
        self.scenario = scenario
        self.questions = tuple(questions)
        self._scenario_hash = _identity(scenario)
        self._scope_hash = _identity(context.scope)
        self.memory = SourceConditionMemory(context.run_id, self._scope_hash, questions)
        self._reviews: dict[str, tuple[CheapConditionAudit, str]] = {}
        self._repair_requested = False
        self._ledger: EvidenceLedger | None = None

    def mark_repair_requested(self) -> None:
        self._repair_requested = True

    def _bound(self, model: ResearchModel, questions: list[str] | None = None) -> None:
        if (
            model.context is not self.context
            or self.context.services.get("asv3_exp_cheap") is not True
            or self.context.depth
            or self.context.run_id != self.memory.run_id
            or _identity(self.context.scope) != self._scope_hash
            or _identity(self.scenario) != self._scenario_hash
            or self.questions != self.memory.questions
            or (questions is not None and tuple(questions) != self.questions)
        ):
            raise ValueError("Cheap source review request or scope changed")
        self.context.check_active()

    def _reviewer(self, model: ResearchModel) -> ResearchModel:
        isolated = copy.copy(self.context)
        isolated.services = dict(self.context.services)
        reviewer = ResearchModel(
            model.llm,
            isolated,
            user_identity=model.user_identity,
            reasoning_effort=model.reasoning_effort,
            token_counter=model.token_counter,
            lean_native_mode=model.lean_native_mode,
        )
        reviewer._native_output_capacity = model._native_output_capacity
        return reviewer

    def _payload(self, ledger: EvidenceLedger) -> tuple[dict[str, JsonValue], set[int]]:
        if self.context.services.get("evidence") is not ledger:
            raise ValueError("Cheap source review requires its owned original ledger")
        self._ledger = ledger
        numbers = set(ledger.citation_mapping())
        if not numbers or set(self.memory.citations()) - numbers:
            raise ValueError(
                "Complete citable originals for retained conditions are missing"
            )
        canonical = cast(
            list[dict[str, JsonValue]],
            json.loads(ledger.serialize_records(sorted(numbers), max_chars=None)),
        )
        catalogue: list[dict[str, JsonValue]] = []
        originals: list[dict[str, JsonValue]] = []
        for row in canonical:
            metadata = row.get("metadata")
            text = row.get("text")
            if (
                row.get("citable") is not True
                or row.get("truncated") is not False
                or not isinstance(text, str)
                or not text
                or not isinstance(metadata, dict)
                or metadata.get("truncated")
                or metadata.get("untrusted")
            ):
                raise ValueError(
                    "Source-condition review requires complete trusted originals"
                )
            catalogue.append(
                {
                    key: row[key]
                    for key in (
                        "citation",
                        "source_id",
                        "chunk_id",
                        "text_hash",
                        "metadata",
                    )
                }
            )
            originals.append(
                {
                    "citation": row["citation"],
                    "identity_ref": row["citation"],
                    "text": text,
                    "start_char": 0,
                    "end_char": len(text),
                    "source_text_length": len(text),
                    "citable": True,
                    "truncated": False,
                }
            )
        payload: dict[str, JsonValue] = {
            "language": self.context.language,
            "scenario": self.scenario,
            "determinations": cast(
                JsonValue, question_determinations(list(self.questions))
            ),
            "original_metadata_catalogue": cast(JsonValue, catalogue),
            "original_evidence": cast(JsonValue, originals),
            "required_evidence_numbers": cast(JsonValue, sorted(numbers)),
        }
        payload["source_scope_id"] = _identity(payload)
        return payload, numbers

    @staticmethod
    def _encoded(payload: dict[str, JsonValue], model: ResearchModel) -> str:
        from onyx.asv3.cheap_metadata_transport import serialize_cheap_payload

        return serialize_cheap_payload(payload, model._tokens)

    def required_conditions(self) -> list[dict[str, JsonValue]]:
        return self.memory.required_conditions()

    @staticmethod
    def _requirement(item: MaterialSourceOmission) -> MaterialSourceOmission:
        return MaterialSourceOmission.model_validate(
            item.model_dump(include=set(MaterialSourceOmission.model_fields))
        )

    @staticmethod
    def _requirement_signature(item: MaterialSourceOmission) -> str:
        return _identity(
            [
                item.witness.model_dump(),
                sorted(item.determination_ids),
                normalized(item.detail),
            ]
        )

    def _navigation(self, answer: str) -> list[dict[str, JsonValue]]:
        units = assertion_inventory(answer)
        result: list[dict[str, JsonValue]] = []
        for record in self.required_conditions():
            item = MaterialSourceOmission.model_validate(record["requirement"])
            result.append(
                {
                    "condition_id": record["condition_id"],
                    "citation": item.witness.citation,
                    "inline_answer_unit_ids": cast(
                        JsonValue,
                        [
                            unit["unit_id"]
                            for unit in units
                            if not unit["presentation_only"]
                            and item.witness.citation in unit["evidence_numbers"]
                        ],
                    ),
                }
            )
        return result

    def _diagnostic_defects(
        self, audit: CheapConditionAudit, answer: str, ledger: EvidenceLedger
    ) -> list[str]:
        units = {unit["unit_id"]: unit for unit in assertion_inventory(answer)}
        retained = {
            str(row["condition_id"]): MaterialSourceOmission.model_validate(
                row["requirement"]
            )
            for row in self.required_conditions()
        }
        signatures = {
            self._requirement_signature(item): item for item in retained.values()
        }
        defects: list[str] = []
        pairs: list[
            tuple[CheapConditionCheck | CheapConditionResolution, AssertionWitness]
        ] = []
        for item in audit.conditions:
            requirement = self._requirement(item)
            original_requirement = signatures.get(
                self._requirement_signature(requirement)
            )
            if original_requirement is not None and requirement != original_requirement:
                defects.append(
                    "Repeated source requirements cannot change immutable detail or applicability."
                )
            pairs.append((item, item.witness))
        for resolution in audit.resolutions:
            original_requirement = retained.get(resolution.condition_id)
            if original_requirement is not None:
                pairs.append(
                    (resolution, resolution.witness or original_requirement.witness)
                )
        for item, witness in pairs:
            if item.disposition not in {"omitted", "uncertain"}:
                if item.missing_details or item.affected_answer_unit_ids:
                    defects.append(
                        "A closed condition cannot carry open repair deltas."
                    )
                continue
            if not item.missing_details or not item.affected_answer_unit_ids:
                defects.append(
                    "An omitted or uncertain condition needs precise source-bound missing details and affected current units."
                )
            selected = item.affected_answer_unit_ids
            if len(set(selected)) != len(selected) or set(selected) - units.keys():
                defects.append(
                    "Repair deltas must select unique exact current answer units."
                )
            original = ledger.get(witness.citation)
            passage = (
                original_witness_text(
                    witness.citation, original.text, witness.witness_id
                )
                if original is not None and witness.witness_id is not None
                else witness.source_quote
            )
            for detail in item.missing_details:
                if (
                    not detail.detail.strip()
                    or not detail.source_quote.strip()
                    or original is None
                    or detail.source_quote not in original.text
                    or not passage
                    or detail.source_quote not in passage
                ):
                    defects.append(
                        "Each missing detail needs an exact literal subquote within its retained or selected operative witness."
                    )
        return defects

    def _repair_deltas(
        self, audit: CheapConditionAudit, answer: str
    ) -> list[dict[str, JsonValue]]:
        identities = {
            self._requirement_signature(
                MaterialSourceOmission.model_validate(row["requirement"])
            ): str(row["condition_id"])
            for row in self.required_conditions()
        }
        items: list[
            tuple[str, CheapConditionCheck | CheapConditionResolution, int]
        ] = []
        retained = {
            str(row["condition_id"]): MaterialSourceOmission.model_validate(
                row["requirement"]
            )
            for row in self.required_conditions()
        }
        for condition in audit.conditions:
            items.append(
                (
                    identities[self._requirement_signature(condition)],
                    condition,
                    condition.witness.citation,
                )
            )
        for resolution in audit.resolutions:
            witness = resolution.witness or retained[resolution.condition_id].witness
            items.append((resolution.condition_id, resolution, witness.citation))
        return [
            {
                "condition_id": identity,
                "disposition": item.disposition,
                "citation": citation,
                "candidate_hash": _identity(answer),
                "affected_answer_unit_ids": cast(
                    JsonValue, list(item.affected_answer_unit_ids)
                ),
                "missing_details": cast(
                    JsonValue,
                    [detail.model_dump(mode="json") for detail in item.missing_details],
                ),
            }
            for identity, item, citation in items
            if item.disposition in {"omitted", "uncertain"}
        ]

    @staticmethod
    def _gap(detail: str, **data: JsonValue) -> ToolOutcome:
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Independent original-source condition assessment remains incomplete.",
            data={
                "source_condition_review": detail,
                "instruction": "Preserve the witnessed requirements and repair their exact omissions or resolve their precise source/fact gaps before publication.",
                **data,
            },
        )

    def ensure_inventory(
        self,
        model: ResearchModel,
        ledger: EvidenceLedger,
        questions: list[str] | None = None,
    ) -> ToolOutcome | None:
        self.mark_repair_requested()
        try:
            self._bound(model, questions)
            payload, numbers = self._payload(ledger)
            identity = _identity(payload)
            cached = self.memory.inventory_receipt(identity)
            if (
                cached
                and ledger.delivery_flow(cached) == LLMFlow.ASV3_SOURCE_INVENTORY.value
                and ledger.completely_delivered(cached) == numbers
            ):
                return None
            reviewer = self._reviewer(model)

            def materialize(
                inventory: CheapSourceRequirementInventory,
            ) -> list[MaterialSourceOmission]:
                if (
                    not inventory.whole_scope_completed
                    or inventory.source_scope_id != payload["source_scope_id"]
                    or reviewer.last_call_id is None
                    or ledger.delivery_flow(reviewer.last_call_id)
                    != LLMFlow.ASV3_SOURCE_INVENTORY.value
                    or ledger.completely_delivered(reviewer.last_call_id) != numbers
                ):
                    raise ValueError(
                        "The model must declare the exact whole source scope assessed with its actual complete inventory receipt"
                    )
                return [item.materialize(ledger) for item in inventory.requirements]

            def validate(text: str) -> None:
                inventory = CheapSourceRequirementInventory.model_validate_json(text)
                requirements = materialize(inventory)
                audit = SourceConditionAuditResult(
                    examined_citations=sorted(numbers),
                    conditions=[
                        SourceConditionCheck(**item.model_dump(), disposition="omitted")
                        for item in requirements
                    ],
                )
                defects = condition_review_defects(
                    audit,
                    "",
                    self.scenario,
                    list(self.questions),
                    ledger,
                    reviewer.last_call_id,
                    expected_citations=numbers,
                    expected_flow=LLMFlow.ASV3_SOURCE_INVENTORY,
                )
                if defects:
                    raise ValueError(" ".join(defects))

            text = reviewer.invoke_text(
                _cheap_prompt(SOURCE_REQUIREMENT_PROMPT)
                + _CATALOGUE_NOTE
                + _INVENTORY_NOTE,
                self._encoded(payload, reviewer),
                LLMFlow.ASV3_SOURCE_INVENTORY,
                max_tokens=24000,
                response_model_override=CheapSourceRequirementInventory,
                response_validator=validate,
            )
            validate(text)
            if reviewer.last_response_truncated or reviewer.last_call_id is None:
                raise StructuredOutputError(
                    "Source inventory lacks a complete original-delivery receipt"
                )
            inventory = CheapSourceRequirementInventory.model_validate_json(text)
            self.memory.remember(materialize(inventory), ledger)
            self.memory.retain_inventory_receipt(identity, reviewer.last_call_id)
            ledger.pin_delivery(reviewer.last_call_id)
            return None
        except RunStopped as error:
            self.context.check_active()
            return self._gap(str(error))
        except (StructuredOutputError, ValueError, ValidationError) as error:
            return self._gap(str(error)[:800])

    def publication_gap(
        self, answer: str, model: ResearchModel, ledger: EvidenceLedger
    ) -> ToolOutcome | None:
        if not self._repair_requested:
            return None
        gap = self.ensure_inventory(model, ledger)
        if gap is not None:
            return gap
        try:
            self._bound(model)
            payload, numbers = self._payload(ledger)
            payload["answer_units"] = cast(JsonValue, assertion_inventory(answer))
            payload["candidate_hash"] = _identity(answer)
            payload["retained_conditions"] = cast(JsonValue, self.required_conditions())
            payload["condition_unit_navigation"] = cast(
                JsonValue, self._navigation(answer)
            )
            payload["allow_explicit_gaps"] = False
            identity = _identity(payload)
            cached = self._reviews.get(identity)

            def base(
                audit: CheapConditionAudit, call_id: str | None
            ) -> SourceConditionAuditResult:
                if (
                    not audit.whole_scope_completed
                    or audit.source_scope_id != payload["source_scope_id"]
                    or call_id is None
                    or ledger.delivery_flow(call_id)
                    != LLMFlow.ASV3_CONDITION_REVIEW.value
                    or ledger.completely_delivered(call_id) != numbers
                ):
                    raise ValueError(
                        "The model must declare the exact whole source scope assessed with its actual complete condition-review receipt"
                    )
                return audit.base(sorted(numbers))

            if (
                cached is not None
                and ledger.delivery_flow(cached[1])
                == LLMFlow.ASV3_CONDITION_REVIEW.value
                and ledger.completely_delivered(cached[1]) == numbers
            ):
                audit, call_id = cached
            else:
                reviewer = self._reviewer(model)

                def validate(text: str) -> None:
                    audit = CheapConditionAudit.model_validate_json(text)
                    effective, defects = self.memory.materialize(
                        base(audit, reviewer.last_call_id), ledger
                    )
                    defects += self._diagnostic_defects(audit, answer, ledger)
                    defects += condition_review_defects(
                        effective,
                        answer,
                        self.scenario,
                        list(self.questions),
                        ledger,
                        reviewer.last_call_id,
                        expected_citations=numbers,
                    )
                    if defects:
                        raise ValueError(" ".join(defects))

                text = reviewer.invoke_text(
                    _cheap_prompt(SOURCE_CONDITION_PROMPT)
                    + _CATALOGUE_NOTE
                    + _REVIEW_SCOPE_NOTE
                    + _DELTA_NOTE,
                    self._encoded(payload, reviewer),
                    LLMFlow.ASV3_CONDITION_REVIEW,
                    max_tokens=16000,
                    response_model_override=CheapConditionAudit,
                    response_validator=validate,
                )
                validate(text)
                if reviewer.last_response_truncated or reviewer.last_call_id is None:
                    raise StructuredOutputError(
                        "Condition assessment lacks a complete original-delivery receipt"
                    )
                audit = CheapConditionAudit.model_validate_json(text)
                call_id = reviewer.last_call_id
                ledger.pin_delivery(call_id)
            base_audit = base(audit, call_id)
            effective, defects = self.memory.materialize(base_audit, ledger)
            defects += self._diagnostic_defects(audit, answer, ledger)
            defects += condition_review_defects(
                effective,
                answer,
                self.scenario,
                list(self.questions),
                ledger,
                call_id,
                expected_citations=numbers,
            )
            if defects:
                raise StructuredOutputError(" ".join(defects))
            self.memory.remember(base_audit.conditions, ledger)
            effective, defects = self.memory.materialize(base_audit, ledger)
            if defects:
                raise StructuredOutputError(" ".join(defects))
            payload["retained_conditions"] = cast(JsonValue, self.required_conditions())
            payload["condition_unit_navigation"] = cast(
                JsonValue, self._navigation(answer)
            )
            self._reviews = {_identity(payload): (audit.model_copy(deep=True), call_id)}
            omissions = condition_omissions(effective, answer)
            uncertain = [
                row for row in effective.conditions if row.disposition == "uncertain"
            ]
            if omissions or uncertain:
                return self._gap(
                    "Material source conditions were omitted or remain uncertain.",
                    omitted_material_source_details=cast(
                        JsonValue, [row.model_dump(mode="json") for row in omissions]
                    ),
                    uncertain_source_conditions=cast(
                        JsonValue, [row.model_dump(mode="json") for row in uncertain]
                    ),
                    retained_source_conditions=cast(
                        JsonValue, self.required_conditions()
                    ),
                    repair_deltas=cast(JsonValue, self._repair_deltas(audit, answer)),
                )
            return None
        except RunStopped as error:
            self.context.check_active()
            return self._gap(str(error))
        except (StructuredOutputError, ValueError, ValidationError) as error:
            return self._gap(str(error)[:800])

    def export(self) -> dict[str, JsonValue]:
        return CheapSourceReviewCheckpoint(
            scenario_hash=self._scenario_hash,
            repair_requested=self._repair_requested,
            memory=self.memory.export(),
            source_bindings=self._source_bindings(self._ledger),
        ).model_dump(mode="json")

    def restore(self, snapshot: dict[str, JsonValue], ledger: EvidenceLedger) -> None:
        saved = CheapSourceReviewCheckpoint.model_validate(snapshot)
        if (
            saved.scenario_hash != self._scenario_hash
            or _identity(self.context.scope) != self._scope_hash
        ):
            raise ValueError("Cheap source condition scenario or scope mismatch")
        restored = SourceConditionMemory(
            self.context.run_id, self._scope_hash, list(self.questions)
        )
        restored.restore(saved.memory, ledger)
        if not saved.repair_requested and restored.required_conditions():
            raise ValueError("Retained Cheap repair requirements cannot be deactivated")
        if saved.source_bindings != self._source_bindings(ledger, memory=restored):
            raise ValueError(
                "Cheap retained condition canonical source identity changed"
            )
        self.memory = restored
        self._ledger = ledger
        self._repair_requested = saved.repair_requested
        self._reviews.clear()

    def _source_bindings(
        self,
        ledger: EvidenceLedger | None,
        *,
        memory: SourceConditionMemory | None = None,
    ) -> list[dict[str, JsonValue]]:
        numbers = sorted((memory or self.memory).citations())
        if not numbers:
            return []
        if ledger is None:
            raise ValueError("Retained conditions have no canonical source ledger")
        rows = cast(
            list[dict[str, JsonValue]],
            json.loads(
                ledger.serialize_records(numbers, required=numbers, max_chars=None)
            ),
        )
        return [
            {
                key: row[key]
                for key in ("citation", "source_id", "chunk_id", "text_hash")
            }
            | {"metadata_hash": _identity(row["metadata"])}
            for row in rows
        ]
