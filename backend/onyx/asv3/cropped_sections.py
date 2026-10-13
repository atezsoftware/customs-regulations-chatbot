"""Bounded source sections with exact writer provenance and host-only assembly."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from onyx.asv3.assertions import assertion_inventory
from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.cropped_completion import (
    CroppedCompletionContext,
    CroppedRequiredFollowup,
    CroppedValidatedFollowups,
    build_cropped_completion_context,
    validate_cropped_required_followups,
)
from onyx.asv3.cropped_transport import CROPPED_INPUT_TOKEN_LIMIT
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    EvidenceItem,
    HarnessView,
    OutcomeStatus,
    RunContext,
    RunStopped,
    ToolOutcome,
)
from onyx.asv3.outcome_map import OutcomeMap
from onyx.asv3.scenario import question_determinations

_Citation = Annotated[int, Field(strict=True, ge=1)]
_Hash = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
_SEPARATOR = "\n\n"
_MATERIAL_CONTEXT = "cropped_section_material_context"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _body_hash(body: str) -> str:
    return hashlib.sha256(body.encode()).hexdigest()


def _require_cropped(context: RunContext) -> None:
    if (
        context.services.get("asv3_exp_cropped") is not True
        or context.services.get("asv3_exp_cheap") is True
    ):
        raise ValueError("Section assembly requires the isolated Cropped workflow")


def _outcomes(context: RunContext) -> dict[str, JsonValue]:
    outcomes = context.services.get("outcome_map")
    return outcomes.view() if isinstance(outcomes, OutcomeMap) else {}


@dataclass(frozen=True)
class CroppedSectionMaterialContext:
    """Immutable actual observations shared by fitting, writing and checkpointing."""

    run_id: str
    scope_hash: str
    request_hash: str
    owner_hash: str
    tasks_json: str
    observations_json: str

    @classmethod
    def capture(
        cls,
        context: RunContext,
        request: str,
        *,
        research_tasks: list[dict[str, JsonValue]],
        repair_observations: list[dict[str, JsonValue]],
    ) -> CroppedSectionMaterialContext:
        _require_cropped(context)
        return cls(
            context.run_id,
            _digest(context.scope),
            _digest(request),
            _owner(context),
            json.dumps(research_tasks, ensure_ascii=False, allow_nan=False),
            json.dumps(repair_observations, ensure_ascii=False, allow_nan=False),
        )

    def validate(self, context: RunContext, request: str) -> None:
        _require_cropped(context)
        if (self.run_id, self.scope_hash, self.request_hash, self.owner_hash) != (
            context.run_id,
            _digest(context.scope),
            _digest(request),
            _owner(context),
        ):
            raise ValueError("Cropped section material context owner or scope changed")
        for value in (self.research_tasks(), self.repair_observations()):
            if any(not isinstance(row, dict) for row in value):
                raise ValueError("Cropped section material observations are invalid")

    def research_tasks(self) -> list[dict[str, JsonValue]]:
        value = json.loads(self.tasks_json)
        if not isinstance(value, list):
            raise ValueError("Cropped section material tasks are invalid")
        return cast(list[dict[str, JsonValue]], value)

    def repair_observations(self) -> list[dict[str, JsonValue]]:
        value = json.loads(self.observations_json)
        if not isinstance(value, list):
            raise ValueError("Cropped section material observations are invalid")
        return cast(list[dict[str, JsonValue]], value)

    def payload(self) -> dict[str, JsonValue]:
        return {
            "research_tasks": cast(list[JsonValue], self.research_tasks()),
            "repair_research_observations": cast(
                list[JsonValue], self.repair_observations()
            ),
        }

    def export(self) -> dict[str, JsonValue]:
        data: dict[str, JsonValue] = {
            "run_id": self.run_id,
            "scope_hash": self.scope_hash,
            "request_hash": self.request_hash,
            "owner_hash": self.owner_hash,
            "tasks_json": self.tasks_json,
            "observations_json": self.observations_json,
        }
        return {**data, "integrity": _digest(data)}

    @property
    def digest(self) -> str:
        return _digest(self.export())

    @classmethod
    def restore(
        cls, data: dict[str, JsonValue], context: RunContext, request: str
    ) -> CroppedSectionMaterialContext:
        keys = {
            "run_id",
            "scope_hash",
            "request_hash",
            "owner_hash",
            "tasks_json",
            "observations_json",
        }
        if (
            set(data) != keys | {"integrity"}
            or any(not isinstance(data[key], str) for key in keys)
            or data["integrity"]
            != _digest({key: data[key] for key in data if key != "integrity"})
        ):
            raise ValueError("Cropped section material checkpoint changed")
        saved = cls(**{key: cast(str, data[key]) for key in keys})
        saved.validate(context, request)
        return saved

    def original_dependencies(
        self, context: RunContext, request: str, ledger: EvidenceLedger
    ) -> set[int]:
        from onyx.asv3.cropped_worker_handoff import (
            WORKER_HANDOFF_REASON,
            expand_worker_handoff,
        )

        self.validate(context, request)
        dependencies: set[int] = set()

        def citation(value: JsonValue) -> EvidenceItem:
            if type(value) is not int or value <= 0:
                raise RunStopped(
                    "Cropped section context has an unknown original reference"
                )
            item = ledger.get(value)
            if item is None:
                raise RunStopped("Cropped section context has an unavailable original")
            dependencies.add(value)
            return item

        def visit(value: JsonValue) -> None:
            if isinstance(value, str):
                for number in extract_citation_numbers(value):
                    citation(number)
                return
            if isinstance(value, list):
                for child in value:
                    visit(child)
                return
            if not isinstance(value, dict):
                return
            if value.get("kind") == WORKER_HANDOFF_REASON:
                actual = expand_worker_handoff(value, ledger=ledger, context=context)
                bindings = value.get("source_bindings")
                assert isinstance(bindings, list)
                for binding in bindings:
                    assert isinstance(binding, dict)
                    citation(binding.get("citation"))
                visit(actual)
                return
            if "cropped_worker_handoff" in value:
                handoff = value["cropped_worker_handoff"]
                if (
                    not isinstance(handoff, dict)
                    or handoff.get("kind") != WORKER_HANDOFF_REASON
                ):
                    raise RunStopped(
                        "Cropped section context has an opaque worker handoff"
                    )
            item = citation(value["citation"]) if "citation" in value else None
            if item is None and all(
                key in value for key in ("source_id", "chunk_id", "text_hash")
            ):
                matching = [
                    number
                    for number in ledger.citation_numbers()
                    if (candidate := ledger.get(number)) is not None
                    and all(
                        value[key] == getattr(candidate, key)
                        for key in ("source_id", "chunk_id", "text_hash")
                    )
                ]
                if len(matching) != 1:
                    raise RunStopped(
                        "Cropped section context original identity is unavailable"
                    )
                item = citation(matching[0])
            if "identity_ref" in value:
                referenced = citation(value["identity_ref"])
                if item is not None and any(
                    getattr(item, key) != getattr(referenced, key)
                    for key in ("source_id", "chunk_id", "text_hash")
                ):
                    raise RunStopped(
                        "Cropped section context original identity conflicts"
                    )
                item = referenced
            if item is not None and any(
                key in value and value[key] != getattr(item, key)
                for key in ("source_id", "chunk_id", "text_hash")
            ):
                raise RunStopped("Cropped section context original identity changed")
            if (
                item is not None
                and "text" in value
                and any(
                    key in value for key in ("source_id", "text_hash", "identity_ref")
                )
            ):
                text, start = value["text"], value.get("start_char", 0)
                if (
                    not isinstance(text, str)
                    or type(start) is not int
                    or start < 0
                    or item.text[start : start + len(text)] != text
                    or (
                        "end_char" in value
                        and (
                            type(value["end_char"]) is not int
                            or value["end_char"] != start + len(text)
                        )
                    )
                ):
                    raise RunStopped("Cropped section context original range changed")
            for key, child in value.items():
                if key in {"evidence_numbers", "required_evidence_numbers"}:
                    if not isinstance(child, list):
                        raise RunStopped(
                            "Cropped section context original references are invalid"
                        )
                    for number in child:
                        citation(number)
                if key in {"arguments_json", "content"} and isinstance(child, str):
                    try:
                        structured = json.loads(child)
                    except ValueError:
                        structured = None
                    if isinstance(structured, (dict, list)):
                        visit(structured)
                visit(child)

        visit(self.payload())
        return dependencies

    def require_assignment(
        self,
        context: RunContext,
        request: str,
        ledger: EvidenceLedger,
        evidence_numbers: list[int],
    ) -> None:
        missing = self.original_dependencies(context, request, ledger) - set(
            evidence_numbers
        )
        if missing:
            raise RunStopped(
                "Cropped section material context requires originals outside its assignment: "
                + ",".join(map(str, sorted(missing)))
            )


def _material(
    context: RunContext, request: str
) -> CroppedSectionMaterialContext | None:
    value = context.services.get(_MATERIAL_CONTEXT)
    if value is None:
        return None
    if not isinstance(value, CroppedSectionMaterialContext):
        raise ValueError("Cropped section material context is invalid")
    value.validate(context, request)
    return value


def _requirements(context: RunContext, ledger: EvidenceLedger) -> dict[str, JsonValue]:
    from onyx.asv3.authority_requirements import AuthorityRequirements
    from onyx.asv3.legal_source_reviews import LegalSourceReviews
    from onyx.asv3.research_state import ResearchState

    requirements = _outcomes(context)
    state = context.services.get("research_state")
    if isinstance(state, ResearchState):
        requirements["retained_source_conditions"] = (
            state.source_conditions.required_conditions()
        )
    authority = context.services.get("authority_requirements")
    if isinstance(authority, AuthorityRequirements):
        requirements["governing_source_requirements"] = authority.view(context)
    reviews = context.services.get("legal_source_reviews")
    if isinstance(reviews, LegalSourceReviews):
        requirements["related_source_requirements"] = reviews.cropped_requirements(
            context, ledger
        )
    material = context.services.get(_MATERIAL_CONTEXT)
    if isinstance(material, CroppedSectionMaterialContext):
        requirements["actual_material_context"] = material.payload()
    return requirements


def _state(view: HarnessView, context: RunContext) -> dict[str, JsonValue]:
    state: dict[str, JsonValue] = {
        "request": view.request,
        "questions": list(view.questions),
        "facts": list(view.facts),
        "publication_gap": copy.deepcopy(view.publication_gap),
        "research_state": copy.deepcopy(view.research_state),
        "user_context": cast(JsonValue, context.services.get("cropped_user_context")),
    }
    material = _material(context, view.request)
    if material is not None:
        state["material_context_hash"] = material.digest
    return state


def _owner(context: RunContext) -> str:
    return _digest(
        {
            "depth": context.depth,
            "task_id": context.services.get("task_id"),
            "user_id": context.services.get("user_id"),
        }
    )


class CroppedSectionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    section_id: str = Field(min_length=1)
    ordinal: int = Field(strict=True, ge=0)
    evidence_numbers: list[_Citation] = Field(min_length=1)
    outcome_ids: list[str] = Field(default_factory=list)
    question_ids: list[str] = Field(default_factory=list)
    retained_unit_ids: list[str] = Field(default_factory=list)
    source_state_hash: _Hash


class CroppedSectionGeneration(BaseModel):
    """The host callback supplies the actual invocation ID, never the model."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    answer: str = Field(min_length=1)
    model_call_id: str = Field(min_length=1)
    status: Literal[OutcomeStatus.FOUND, OutcomeStatus.PARTIAL] = OutcomeStatus.FOUND


class CroppedAssemblyReviewInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    answer: str
    answer_hash: _Hash
    request: str
    questions: list[str]
    facts: list[str]
    determinations: list[dict[str, JsonValue]]
    sections: list[dict[str, JsonValue]]
    retained_requirements: dict[str, JsonValue]
    publication_gap: dict[str, JsonValue] | None
    user_context: str | None
    completion_context: dict[str, JsonValue] | None = None


class CroppedAssemblyAssessment(BaseModel):
    """A consistency assessment cannot replace or extend any section's prose."""

    model_config = ConfigDict(extra="forbid")
    review_scope: Literal["section_coverage_and_consistency"]
    assessed_section_ids: list[str]
    assessed_determination_ids: list[str]
    complete: bool
    consistent: bool
    missing: list[str]
    contradictions: list[str]
    required_followups: list[CroppedRequiredFollowup] = Field(default_factory=list)


class CroppedAssemblyReview(CroppedAssemblyAssessment):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_call_id: str = Field(min_length=1)
    answer_hash: _Hash
    completion_snapshot: dict[str, JsonValue] | None = None


class _OriginalBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    citation: _Citation
    source_id: str
    chunk_id: str | None
    text_hash: _Hash
    metadata_hash: _Hash
    end_char: int = Field(strict=True, ge=1)


class _SectionReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    receipt_id: str
    plan: CroppedSectionPlan
    generation: CroppedSectionGeneration
    answer_hash: _Hash
    originals: list[_OriginalBinding]
    delivery_flow: str | None


class _AssemblyReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    receipt_id: str
    section_receipt_ids: list[str] = Field(min_length=2)
    answer: str
    answer_hash: _Hash
    source_state_hash: _Hash
    requirements_hash: _Hash
    review: CroppedAssemblyReview


class _Checkpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    run_id: str
    user_id: str | None
    scope_hash: str
    request_hash: str
    owner_hash: _Hash
    material_context: dict[str, JsonValue] | None = None
    sections: list[_SectionReceipt]
    assemblies: list[_AssemblyReceipt]
    integrity: str


PublicationGap = Callable[[str, str], ToolOutcome | None]
CountSectionInput = Callable[[HarnessView, CroppedSectionPlan], int]
WriteSection = Callable[[HarnessView, CroppedSectionPlan], CroppedSectionGeneration]
ReviewAssembly = Callable[[CroppedAssemblyReviewInput], CroppedAssemblyReview]


def _canonical_originals(
    view: HarnessView, context: RunContext, ledger: EvidenceLedger
) -> list[dict[str, JsonValue]]:
    from onyx.asv3.research_state import ResearchState

    numbers = set(view.required_evidence_numbers) | set(
        extract_citation_numbers(view.draft_to_repair or "")
    )
    outcome_map = context.services.get("outcome_map")
    if isinstance(outcome_map, OutcomeMap):
        numbers.update(outcome_map.preferred_citations())
    research_state = context.services.get("research_state")
    if isinstance(research_state, ResearchState):
        numbers.update(research_state.source_conditions.citations())
    related = _requirements(context, ledger).get("related_source_requirements")
    rows = related.get("reviews") if isinstance(related, dict) else None
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict):
            numbers.update(_numbers(row.get("required_evidence_numbers")))
    material = _material(context, view.request)
    if material is not None:
        numbers.update(material.original_dependencies(context, view.request, ledger))
    for omitted in view.original_evidence_omitted:
        if not isinstance(omitted, dict) or type(omitted.get("citation")) is not int:
            raise RunStopped("Cropped section planning received an invalid omission")
        numbers.add(cast(int, omitted["citation"]))
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
            raise RunStopped("Cropped section planning received an invalid original")
        item = ledger.get(number)
        if (
            item is None
            or start < 0
            or item.text[start : start + len(text)] != text
            or any(
                row.get(key) != getattr(item, key)
                for key in ("source_id", "chunk_id", "text_hash")
            )
        ):
            raise RunStopped("Cropped section planning original identity changed")
        numbers.add(number)
    if not numbers:
        raise RunStopped("Cropped section planning needs recorded originals")
    for number in numbers:
        item = ledger.get(number)
        canonical = item.metadata.get("canonical_metadata") if item else None
        if (
            item is None
            or not item.text
            or item.search_doc is None
            or item.metadata.get("truncated") is True
            or (isinstance(canonical, dict) and canonical.get("truncated") is True)
        ):
            raise RunStopped(
                "Cropped section planning requires complete citable originals"
            )
    ordered = sorted(numbers)
    return cast(
        list[dict[str, JsonValue]],
        json.loads(ledger.serialize_records(ordered, required=ordered, max_chars=None)),
    )


def _numbers(value: JsonValue) -> list[int]:
    return (
        [number for number in value if type(number) is int and number > 0]
        if isinstance(value, list)
        else []
    )


def _groups(
    originals: list[dict[str, JsonValue]],
    view: HarnessView,
    context: RunContext,
    ledger: EvidenceLedger,
) -> list[list[int]]:
    from onyx.asv3.legal_source_reviews import LegalSourceReviews

    numbers = [cast(int, row["citation"]) for row in originals]
    parents = dict.fromkeys(numbers, 0)
    for number in numbers:
        parents[number] = number

    def root(number: int) -> int:
        while parents[number] != number:
            parents[number] = parents[parents[number]]
            number = parents[number]
        return number

    def connect(selected: list[int]) -> None:
        selected = [number for number in selected if number in parents]
        if selected:
            anchor = min(root(number) for number in selected)
            for number in selected:
                parents[root(number)] = anchor

    # Missing article locators cannot establish an independent provision boundary.
    sources: dict[str, list[dict[str, JsonValue]]] = {}
    for row in originals:
        sources.setdefault(cast(str, row["source_id"]), []).append(row)
    for rows in sources.values():
        articles: dict[str, list[int]] = {}
        uncertain = False
        for row in rows:
            metadata = row.get("metadata")
            article = metadata.get("article_no") if isinstance(metadata, dict) else None
            if (
                not isinstance(article, (str, int))
                or isinstance(article, bool)
                or not str(article).strip()
            ):
                uncertain = True
                break
            articles.setdefault(str(article), []).append(cast(int, row["citation"]))
        if uncertain:
            connect([cast(int, row["citation"]) for row in rows])
        else:
            for citations in articles.values():
                connect(citations)
    outcome_sources: dict[str, list[int]] = {}
    outcomes = _outcomes(context)
    conditions = outcomes.get("conditions", [])
    for condition in conditions if isinstance(conditions, list) else []:
        if not isinstance(condition, dict):
            continue
        witnesses = condition.get("witnesses", [])
        citations = (
            [
                cast(int, row["citation"])
                for row in witnesses
                if isinstance(row, dict) and type(row.get("citation")) is int
            ]
            if isinstance(witnesses, list)
            else []
        )
        ids = condition.get("outcome_ids", [])
        for identity in ids if isinstance(ids, list) else []:
            if isinstance(identity, str):
                outcome_sources.setdefault(identity, []).extend(citations)
    resolutions = outcomes.get("resolutions", [])
    for resolution in resolutions if isinstance(resolutions, list) else []:
        identity = (
            resolution.get("outcome_id") if isinstance(resolution, dict) else None
        )
        if isinstance(resolution, dict) and isinstance(identity, str):
            outcome_sources.setdefault(identity, []).extend(
                _numbers(resolution.get("evidence_numbers"))
            )
    for citations in outcome_sources.values():
        connect(citations)
    reviews = context.services.get("legal_source_reviews")
    if isinstance(reviews, LegalSourceReviews):
        rows = reviews.cropped_requirements(context, ledger).get("reviews")
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, dict):
                connect(_numbers(row.get("required_evidence_numbers")))
    for unit in assertion_inventory(view.draft_to_repair or ""):
        connect(unit["evidence_numbers"])
    material = _material(context, view.request)
    if material is not None:
        connect(sorted(material.original_dependencies(context, view.request, ledger)))
    grouped: dict[int, list[int]] = {}
    for number in numbers:
        grouped.setdefault(root(number), []).append(number)
    return sorted(grouped.values(), key=lambda group: min(group))


def _plan(
    citations: list[int],
    ordinal: int,
    state_hash: str,
    view: HarnessView,
    context: RunContext,
) -> CroppedSectionPlan:
    selected = set(citations)
    outcome_map = context.services.get("outcome_map")
    outcome_ids: list[str] = []
    question_ids: list[str] = []
    if isinstance(outcome_map, OutcomeMap):
        for identity in outcome_map.outcome_ids():
            if selected.intersection(
                outcome_map.preferred_citations(outcome_ids=[identity])
            ):
                outcome_ids.append(identity)
        snapshot = outcome_map.view(outcome_ids=outcome_ids)
        outcomes = snapshot.get("outcomes", [])
        for outcome in outcomes if isinstance(outcomes, list) else []:
            ids = outcome.get("question_ids", []) if isinstance(outcome, dict) else []
            question_ids.extend(
                identity for identity in ids if isinstance(identity, str)
            )
    unit_ids = [
        unit["unit_id"]
        for unit in assertion_inventory(view.draft_to_repair or "")
        if selected.intersection(unit["evidence_numbers"])
        or (ordinal == 0 and not unit["evidence_numbers"])
    ]
    identity = {"ordinal": ordinal, "citations": citations, "state": state_hash}
    return CroppedSectionPlan(
        section_id="cropped_section_" + _digest(identity),
        ordinal=ordinal,
        evidence_numbers=citations,
        outcome_ids=outcome_ids,
        question_ids=list(dict.fromkeys(question_ids)),
        retained_unit_ids=unit_ids,
        source_state_hash=state_hash,
    )


def _scoped_view(
    view: HarnessView,
    plan: CroppedSectionPlan,
    originals: list[dict[str, JsonValue]],
    context: RunContext,
) -> HarnessView:
    from onyx.asv3.research_state import ResearchState

    selected = set(plan.evidence_numbers)
    units = assertion_inventory(view.draft_to_repair or "")
    draft = (
        _SEPARATOR.join(
            unit["text"] for unit in units if unit["unit_id"] in plan.retained_unit_ids
        )
        or None
    )
    prepared = view.model_copy(deep=True)
    prepared.turns = []
    prepared.receipts = []
    prepared.original_evidence = [
        copy.deepcopy(row) for row in originals if row["citation"] in selected
    ]
    prepared.original_evidence_omitted = []
    prepared.required_evidence_numbers = list(plan.evidence_numbers)
    prepared.evidence = [
        copy.deepcopy(row) for row in view.evidence if row.get("citation") in selected
    ]
    prepared.draft_to_repair = draft
    prepared.tools = [
        tool
        for tool in prepared.tools
        if isinstance(function := tool.get("function"), dict)
        and function.get("name") in {"submit_answer", "submit_partial_answer"}
    ]
    prepared.research_state = {
        **prepared.research_state,
        "cropped_section_assignment": plan.model_dump(mode="json"),
    }
    state = context.services.get("research_state")
    if isinstance(state, ResearchState):
        prepared.research_state["source_conditions"] = [
            row
            for row in state.source_conditions.required_conditions()
            if isinstance(requirement := row.get("requirement"), dict)
            and isinstance(witness := requirement.get("witness"), dict)
            and witness.get("citation") in selected
        ]
        prepared.research_state["source_conditions_omitted"] = 0
        prepared.research_state["cropped_source_condition_scope"] = (
            "This section's complete assigned-source requirements; the host retains "
            "the other sections' requirements for final assembly."
        )
    return prepared


def plan_cropped_sections(
    view: HarnessView,
    context: RunContext,
    ledger: EvidenceLedger,
    count_input: CountSectionInput,
) -> list[tuple[CroppedSectionPlan, HarnessView]]:
    """Keep connected source obligations intact; fitting never deletes a required atom."""
    _require_cropped(context)
    originals = _canonical_originals(view, context, ledger)
    state_hash = _digest(_state(view, context))
    atoms = _groups(originals, view, context, ledger)
    planned: list[tuple[CroppedSectionPlan, HarnessView]] = []
    pending: list[int] = []

    def counted(prepared: HarnessView, plan: CroppedSectionPlan) -> int:
        material = _material(context, view.request)
        if material is not None:
            material.require_assignment(
                context, view.request, ledger, list(plan.evidence_numbers)
            )
        count = count_input(prepared, plan)
        if type(count) is not int or count < 0:
            raise ValueError("Cropped section input counter returned an invalid count")
        return count

    for atom in atoms:
        candidate = sorted([*pending, *atom])
        plan = _plan(candidate, len(planned), state_hash, view, context)
        prepared = _scoped_view(view, plan, originals, context)
        count = counted(prepared, plan)
        if count <= CROPPED_INPUT_TOKEN_LIMIT:
            pending = candidate
            continue
        if pending:
            prior = _plan(pending, len(planned), state_hash, view, context)
            planned.append((prior, _scoped_view(view, prior, originals, context)))
        pending = sorted(atom)
        plan = _plan(pending, len(planned), state_hash, view, context)
        prepared = _scoped_view(view, plan, originals, context)
        if counted(prepared, plan) > CROPPED_INPUT_TOKEN_LIMIT:
            raise RunStopped(
                "Cropped connected source obligation exceeds the bounded writer input: "
                + ",".join(map(str, atom))
            )
    if pending:
        plan = _plan(pending, len(planned), state_hash, view, context)
        planned.append((plan, _scoped_view(view, plan, originals, context)))
    if len(planned) < 2:
        raise RunStopped(
            "Cropped section fallback requires at least two independently fitting source groups"
        )
    return planned


class CroppedSectionStore:
    """Receipts prove exact bytes and delivery; source guards still establish admissibility."""

    def __init__(
        self, context: RunContext, request: str, *, user_id: str | None
    ) -> None:
        _require_cropped(context)
        self.run_id, self.user_id = context.run_id, user_id
        self.scope_hash, self.request_hash = _digest(context.scope), _digest(request)
        self.owner_hash = _owner(context)
        self.material_context = _material(context, request)
        self._sections: dict[str, _SectionReceipt] = {}
        self._assemblies: dict[str, _AssemblyReceipt] = {}
        self._lock = threading.RLock()

    def _fence(self, context: RunContext, request: str) -> None:
        _require_cropped(context)
        if (
            context.run_id,
            _digest(context.scope),
            _digest(request),
            _owner(context),
        ) != (
            self.run_id,
            self.scope_hash,
            self.request_hash,
            self.owner_hash,
        ):
            raise ValueError("Cropped section owner, scope or request changed")
        material = _material(context, request)
        if (material.digest if material else None) != (
            self.material_context.digest if self.material_context else None
        ):
            raise ValueError("Cropped section material context changed")

    def owns_assembly(self, answer: str) -> bool:
        """Exact text lookup only; callers must still verify owner and actual sources."""
        with self._lock:
            return any(
                receipt.answer == answer for receipt in self._assemblies.values()
            )

    def assembly_status(self, answer: str) -> OutcomeStatus:
        with self._lock:
            receipt = next(
                (row for row in self._assemblies.values() if row.answer == answer), None
            )
            if receipt is None:
                raise ValueError("Cropped assembly is unowned")
            statuses = [
                self._sections[identity].generation.status
                for identity in receipt.section_receipt_ids
            ]
            required_followups = bool(receipt.review.required_followups)
        return (
            OutcomeStatus.PARTIAL
            if required_followups or OutcomeStatus.PARTIAL in statuses
            else OutcomeStatus.FOUND
        )

    def assembly_completion_followups(
        self,
        context: RunContext,
        view: HarnessView,
        answer: str,
        ledger: EvidenceLedger,
    ) -> CroppedValidatedFollowups | None:
        """Recover model-reported questions only after the real assembly guard."""
        self._fence(context, view.request)
        with self._lock:
            receipt = next(
                (row for row in self._assemblies.values() if row.answer == answer), None
            )
            if receipt is None:
                raise ValueError("Cropped assembly is unowned")
            if receipt.review.completion_snapshot is None:
                if receipt.review.required_followups:
                    raise ValueError(
                        "Assembly followups lack their actual review snapshot"
                    )
                return None
            if (
                receipt.source_state_hash != _digest(_state(view, context))
                or receipt.requirements_hash != _digest(_requirements(context, ledger))
                or receipt.answer_hash != _body_hash(answer)
            ):
                raise ValueError("Assembly followup scope or sources changed")
            for identity in receipt.section_receipt_ids:
                self._validate_section(self._sections[identity], ledger)
            snapshot = CroppedCompletionContext.restore(
                receipt.review.completion_snapshot
            )
            return validate_cropped_required_followups(
                [
                    row.model_dump(mode="json")
                    for row in receipt.review.required_followups
                ],
                context,
                view.request,
                snapshot,
                answer=answer,
                facts=list(view.facts),
                publication_gap=view.publication_gap,
            )

    def assembly_original_numbers(self, answer: str) -> set[int]:
        """Use only after verify_assembly rechecks every owned writer receipt."""
        with self._lock:
            receipt = next(
                (row for row in self._assemblies.values() if row.answer == answer), None
            )
            if receipt is None:
                raise ValueError("Cropped assembly is unowned")
            return {
                original.citation
                for identity in receipt.section_receipt_ids
                for original in self._sections[identity].originals
            }

    def retained_section(
        self,
        context: RunContext,
        view: HarnessView,
        plan: CroppedSectionPlan,
        ledger: EvidenceLedger,
        publication_gap: PublicationGap,
    ) -> str | None:
        self._fence(context, view.request)
        if plan.source_state_hash != _digest(_state(view, context)):
            raise ValueError("Cropped section scenario changed")
        with self._lock:
            receipt = next(
                (row for row in self._sections.values() if row.plan == plan), None
            )
        if receipt is None:
            return None
        self._validate_section(receipt, ledger)
        if (
            publication_gap(receipt.generation.answer, receipt.generation.model_call_id)
            is not None
        ):
            raise ValueError("Cropped section publication guard rejected its body")
        return receipt.receipt_id

    @staticmethod
    def _originals(
        plan: CroppedSectionPlan, call_id: str, ledger: EvidenceLedger
    ) -> list[_OriginalBinding]:
        bindings: list[_OriginalBinding] = []
        delivered = ledger.completely_delivered(call_id)
        for number in plan.evidence_numbers:
            item = ledger.get(number)
            if item is None or item.search_doc is None or number not in delivered:
                raise ValueError(
                    "Cropped section needs actual complete original delivery"
                )
            deliveries = ledger.inspect(number).get("deliveries")
            matched = (
                any(
                    isinstance(delivery, dict)
                    and delivery.get("call_id") == call_id
                    and isinstance(records := delivery.get("records"), list)
                    and any(
                        isinstance(row, dict)
                        and row.get("citation") == number
                        and row.get("source_id") == item.source_id
                        and row.get("chunk_id") == item.chunk_id
                        and row.get("text_hash") == item.text_hash
                        and row.get("passage_hash") == item.text_hash
                        and type(row.get("start_char")) is int
                        and row.get("start_char") == 0
                        and type(row.get("end_char")) is int
                        and row.get("end_char") == len(item.text)
                        and row.get("complete") is True
                        for row in records
                    )
                    for delivery in deliveries
                )
                if isinstance(deliveries, list)
                else False
            )
            if not matched:
                raise ValueError("Cropped section original receipt changed")
            bindings.append(
                _OriginalBinding(
                    citation=number,
                    source_id=item.source_id,
                    chunk_id=item.chunk_id,
                    text_hash=item.text_hash,
                    metadata_hash=_digest(item.metadata),
                    end_char=len(item.text),
                )
            )
        return bindings

    def seal_section(
        self,
        context: RunContext,
        view: HarnessView,
        plan: CroppedSectionPlan,
        generation: CroppedSectionGeneration,
        ledger: EvidenceLedger,
        publication_gap: PublicationGap,
    ) -> str:
        self._fence(context, view.request)
        if (
            plan.source_state_hash != _digest(_state(view, context))
            or context.services.get("last_model_call_id") != generation.model_call_id
        ):
            raise ValueError(
                "Cropped section needs its current owned writer invocation"
            )
        if set(extract_citation_numbers(generation.answer)) - set(
            plan.evidence_numbers
        ):
            raise ValueError(
                "Cropped section cites a source outside its owned assignment"
            )
        if (
            not generation.answer.strip()
            or generation.answer != generation.answer.strip()
            or publication_gap(generation.answer, generation.model_call_id) is not None
        ):
            raise ValueError("Cropped section publication guard rejected its body")
        originals = self._originals(plan, generation.model_call_id, ledger)
        receipt = _SectionReceipt(
            receipt_id="pending",
            plan=plan,
            generation=generation,
            answer_hash=_body_hash(generation.answer),
            originals=originals,
            delivery_flow=ledger.delivery_flow(generation.model_call_id),
        )
        receipt = receipt.model_copy(
            update={
                "receipt_id": "cropped_section_receipt_"
                + _digest(receipt.model_dump(mode="json", exclude={"receipt_id"}))
            }
        )
        with self._lock:
            if any(
                saved.generation.model_call_id == generation.model_call_id
                and saved != receipt
                for saved in self._sections.values()
            ):
                raise ValueError(
                    "Cropped sections require distinct owned writer invocations"
                )
            ledger.pin_delivery(generation.model_call_id)
            self._sections[receipt.receipt_id] = receipt
        return receipt.receipt_id

    def _validate_section(
        self, receipt: _SectionReceipt, ledger: EvidenceLedger
    ) -> None:
        if (
            receipt.receipt_id
            != "cropped_section_receipt_"
            + _digest(receipt.model_dump(mode="json", exclude={"receipt_id"}))
            or receipt.answer_hash != _body_hash(receipt.generation.answer)
            or receipt.originals
            != self._originals(receipt.plan, receipt.generation.model_call_id, ledger)
            or receipt.delivery_flow
            != ledger.delivery_flow(receipt.generation.model_call_id)
        ):
            raise ValueError(
                "Cropped section bytes, original version or receipt changed"
            )

    def assemble(
        self,
        context: RunContext,
        view: HarnessView,
        receipt_ids: list[str],
        ledger: EvidenceLedger,
        publication_gap: PublicationGap,
        review_assembly: ReviewAssembly,
    ) -> str:
        self._fence(context, view.request)
        with self._lock:
            receipts = [self._sections[identity] for identity in receipt_ids]
        state_hash = _digest(_state(view, context))
        if (
            len(set(receipt_ids)) != len(receipt_ids)
            or [receipt.plan.ordinal for receipt in receipts]
            != list(range(len(receipts)))
            or any(receipt.plan.source_state_hash != state_hash for receipt in receipts)
        ):
            raise ValueError(
                "Cropped assembly requires its complete ordered section set"
            )
        expected = {
            cast(int, row["citation"])
            for row in _canonical_originals(view, context, ledger)
        }
        assigned = [
            number for receipt in receipts for number in receipt.plan.evidence_numbers
        ]
        if set(assigned) != expected or len(set(assigned)) != len(assigned):
            raise ValueError("Cropped assembly omitted or duplicated an owned original")
        # New source-bound requirements may be retained by a section terminal. They
        # may not introduce a dependency spanning separately assessed writers.
        originals = _canonical_originals(view, context, ledger)
        assignments = [set(receipt.plan.evidence_numbers) for receipt in receipts]
        for connected in _groups(originals, view, context, ledger):
            if not any(
                set(connected).issubset(assignment) for assignment in assignments
            ):
                raise ValueError(
                    "Cropped assembly acquired a cross-section source obligation"
                )
        for receipt in receipts:
            self._validate_section(receipt, ledger)
            if (
                publication_gap(
                    receipt.generation.answer, receipt.generation.model_call_id
                )
                is not None
            ):
                raise ValueError("Cropped section publication guard rejected its body")
        answer = _SEPARATOR.join(receipt.generation.answer for receipt in receipts)
        determinations = cast(
            list[dict[str, JsonValue]],
            [dict(row) for row in question_determinations(view.questions)],
        )
        requirements = _requirements(context, ledger)
        requirements_hash = _digest(requirements)
        completion_snapshot = build_cropped_completion_context(
            context,
            view.request,
            facts=list(view.facts),
            publication_gap=view.publication_gap,
        )
        review_input = CroppedAssemblyReviewInput(
            answer=answer,
            answer_hash=_body_hash(answer),
            request=view.request,
            questions=list(view.questions),
            facts=list(view.facts),
            determinations=determinations,
            sections=[
                {
                    "plan": receipt.plan.model_dump(mode="json"),
                    "answer_hash": receipt.answer_hash,
                    "status": receipt.generation.status.value,
                }
                for receipt in receipts
            ],
            retained_requirements=requirements,
            publication_gap=copy.deepcopy(view.publication_gap),
            user_context=context.services.get("cropped_user_context")
            if isinstance(context.services.get("cropped_user_context"), str)
            else None,
            completion_context=completion_snapshot.writer_view(),
        )
        context.check_active()
        review = review_assembly(review_input)
        if review.required_followups:
            if (
                review.completion_snapshot is None
                or CroppedCompletionContext.restore(review.completion_snapshot)
                != completion_snapshot
            ):
                raise RunStopped(
                    "Cropped assembly followups lack their actual review context"
                )
            validated = validate_cropped_required_followups(
                [row.model_dump(mode="json") for row in review.required_followups],
                context,
                view.request,
                completion_snapshot,
                answer=answer,
                facts=list(view.facts),
                publication_gap=view.publication_gap,
            )
            review = review.model_copy(
                update={"required_followups": list(validated.items)}
            )
        expected_sections = [receipt.plan.section_id for receipt in receipts]
        expected_determinations = [
            cast(str, row["determination_id"]) for row in determinations
        ]
        if (
            review.answer_hash != review_input.answer_hash
            or context.services.get("last_model_call_id") != review.model_call_id
            or review.model_call_id
            in {receipt.generation.model_call_id for receipt in receipts}
            or review.assessed_section_ids != expected_sections
            or review.assessed_determination_ids != expected_determinations
            or not review.complete
            or not review.consistent
            or review.missing
            or review.contradictions
            or requirements_hash != _digest(_requirements(context, ledger))
            or state_hash != _digest(_state(view, context))
        ):
            raise RunStopped(
                "Cropped assembled answer failed completeness or consistency assessment"
            )
        for section in receipts:
            self._validate_section(section, ledger)
        receipt = _AssemblyReceipt(
            receipt_id="pending",
            section_receipt_ids=receipt_ids,
            answer=answer,
            answer_hash=_body_hash(answer),
            source_state_hash=state_hash,
            requirements_hash=requirements_hash,
            review=review,
        )
        receipt = receipt.model_copy(
            update={
                "receipt_id": "cropped_assembly_"
                + _digest(receipt.model_dump(mode="json", exclude={"receipt_id"}))
            }
        )
        with self._lock:
            self._assemblies[receipt.receipt_id] = receipt
        return answer

    def verify_assembly(
        self,
        context: RunContext,
        view: HarnessView,
        answer: str,
        ledger: EvidenceLedger,
        publication_gap: PublicationGap,
    ) -> None:
        self._fence(context, view.request)
        with self._lock:
            receipt = next(
                (row for row in self._assemblies.values() if row.answer == answer), None
            )
            sections = (
                [self._sections[identity] for identity in receipt.section_receipt_ids]
                if receipt
                else []
            )
        if (
            receipt is None
            or receipt.source_state_hash != _digest(_state(view, context))
            or receipt.requirements_hash != _digest(_requirements(context, ledger))
            or receipt.answer_hash != _body_hash(answer)
            or receipt.answer
            != _SEPARATOR.join(row.generation.answer for row in sections)
        ):
            raise ValueError("Cropped assembly is unowned or changed")
        for section in sections:
            self._validate_section(section, ledger)
            if (
                publication_gap(
                    section.generation.answer, section.generation.model_call_id
                )
                is not None
            ):
                raise ValueError("Cropped section publication guard rejected its body")
        if receipt.requirements_hash != _digest(_requirements(context, ledger)):
            raise ValueError("Cropped assembly requirements changed during validation")

    def export(self) -> dict[str, JsonValue]:
        with self._lock:
            content: dict[str, JsonValue] = {
                "version": 1,
                "run_id": self.run_id,
                "user_id": self.user_id,
                "scope_hash": self.scope_hash,
                "request_hash": self.request_hash,
                "owner_hash": self.owner_hash,
                "sections": [
                    row.model_dump(mode="json") for row in self._sections.values()
                ],
                "assemblies": [
                    row.model_dump(mode="json") for row in self._assemblies.values()
                ],
            }
            if self.material_context is not None:
                content["material_context"] = self.material_context.export()
        return {**content, "integrity": _digest(content)}

    def restore(
        self,
        snapshot: dict[str, JsonValue],
        context: RunContext,
        request: str,
        ledger: EvidenceLedger,
    ) -> None:
        _require_cropped(context)
        saved = _Checkpoint.model_validate(snapshot)
        saved_content = saved.model_dump(mode="json", exclude={"integrity"})
        if "material_context" not in snapshot:
            saved_content.pop("material_context", None)
        if (
            saved.run_id,
            saved.user_id,
            saved.scope_hash,
            saved.request_hash,
            saved.owner_hash,
        ) != (
            self.run_id,
            self.user_id,
            self.scope_hash,
            self.request_hash,
            self.owner_hash,
        ) or saved.integrity != _digest(saved_content):
            raise ValueError(
                "Cropped section checkpoint ownership or integrity changed"
            )
        restored_material = (
            CroppedSectionMaterialContext.restore(
                saved.material_context, context, request
            )
            if saved.material_context is not None
            else None
        )
        current_material = _material(context, request)
        if current_material is not None and (
            restored_material is None
            or current_material.digest != restored_material.digest
        ):
            raise ValueError("Cropped section checkpoint material context changed")
        sections: dict[str, _SectionReceipt] = {}
        for receipt in saved.sections:
            self._validate_section(receipt, ledger)
            if receipt.receipt_id in sections:
                raise ValueError("Duplicate Cropped section receipt")
            sections[receipt.receipt_id] = receipt
        assemblies: dict[str, _AssemblyReceipt] = {}
        for receipt in saved.assemblies:
            ordered = [
                sections[identity]
                for identity in receipt.section_receipt_ids
                if identity in sections
            ]
            if (
                receipt.receipt_id
                != "cropped_assembly_"
                + _digest(receipt.model_dump(mode="json", exclude={"receipt_id"}))
                or any(
                    identity not in sections for identity in receipt.section_receipt_ids
                )
                or receipt.answer_hash != _body_hash(receipt.answer)
                or receipt.answer
                != _SEPARATOR.join(
                    sections[identity].generation.answer
                    for identity in receipt.section_receipt_ids
                )
                or receipt.receipt_id in assemblies
                or len(set(receipt.section_receipt_ids))
                != len(receipt.section_receipt_ids)
                or [row.plan.ordinal for row in ordered] != list(range(len(ordered)))
                or any(
                    row.plan.source_state_hash != receipt.source_state_hash
                    for row in ordered
                )
                or not receipt.review.complete
                or not receipt.review.consistent
                or receipt.review.missing
                or receipt.review.contradictions
                or receipt.review.answer_hash != receipt.answer_hash
                or receipt.review.model_call_id
                in {row.generation.model_call_id for row in ordered}
                or receipt.review.assessed_section_ids
                != [
                    sections[identity].plan.section_id
                    for identity in receipt.section_receipt_ids
                ]
            ):
                raise ValueError("Cropped assembly checkpoint changed")
            assemblies[receipt.receipt_id] = receipt
        with self._lock:
            if self._sections or self._assemblies:
                raise ValueError("Cannot restore an active Cropped section store")
            self.material_context = restored_material
            if restored_material is not None:
                context.services[_MATERIAL_CONTEXT] = restored_material
            for receipt in sections.values():
                ledger.pin_delivery(receipt.generation.model_call_id)
            self._sections, self._assemblies = sections, assemblies


def assemble_cropped_sections(
    view: HarnessView,
    context: RunContext,
    ledger: EvidenceLedger,
    *,
    count_input: CountSectionInput,
    write_section: WriteSection,
    review_assembly: ReviewAssembly,
    publication_gap: PublicationGap,
    store: CroppedSectionStore | None = None,
) -> Decision:
    """Sequential bounded writes, one assessment, then exact host concatenation."""
    _require_cropped(context)
    if store is None:
        store = CroppedSectionStore(
            context,
            view.request,
            user_id=context.services.get("user_id")
            if isinstance(context.services.get("user_id"), str)
            else None,
        )
    context.services["cropped_section_store"] = store
    context.services["cropped_section_publication_view"] = view.model_copy(deep=True)
    plans = plan_cropped_sections(view, context, ledger, count_input)
    receipts: list[str] = []
    for plan, prepared in plans:
        context.check_active()
        retained = store.retained_section(context, view, plan, ledger, publication_gap)
        if retained is not None:
            receipts.append(retained)
            continue
        generation = write_section(prepared, plan)
        receipts.append(
            store.seal_section(context, view, plan, generation, ledger, publication_gap)
        )
    answer = store.assemble(
        context, view, receipts, ledger, publication_gap, review_assembly
    )
    context.services["cropped_section_store"] = store
    context.services["cropped_section_publication_view"] = view.model_copy(deep=True)
    status = store.assembly_status(answer)
    name = (
        "submit_partial_answer" if status == OutcomeStatus.PARTIAL else "submit_answer"
    )
    arguments: dict[str, JsonValue] = {"answer": answer}
    followups = store.assembly_completion_followups(context, view, answer, ledger)
    if followups is not None and followups.items:
        context.services["cropped_completion_context"] = followups.snapshot
        arguments["_required_followups"] = [
            row.model_dump(mode="json") for row in followups.items
        ]
    if status == OutcomeStatus.FOUND:
        arguments["basis"] = "originals"
    return Decision(calls=[CapabilityCall(name=name, arguments=arguments)])
