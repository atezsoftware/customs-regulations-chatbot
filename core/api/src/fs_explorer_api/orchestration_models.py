"""Typed contracts and deterministic validation for multi-agent research.

The contracts in this module are intentionally independent from chat history
and the legacy ``Action`` union.  They are compact, versioned artifacts passed
between isolated orchestration roles.  LLM-facing models forbid undeclared
fields so provider-side structured output and server-side parsing use the same
portable JSON schema.

Version 3 models a scenario as a small decision graph instead of a list of
topic headings.  Stable IDs connect answer requirements, user-provided facts,
material unknowns, evidence needs, task outputs, and grounded conclusions.
Semantic validation remains server-owned and deterministic.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class _StrictArtifact(BaseModel):
    """Base for exact structured-output artifacts."""

    model_config = ConfigDict(extra="forbid")


class PlanMode(str, Enum):
    """How the global planner routes a user question."""

    DIRECT = "direct"
    DECOMPOSED = "decomposed"


class ExecutionStrategy(str, Enum):
    """How much task-local orchestration a validated plan requires."""

    SINGLE_PASS = "single_pass"
    ADAPTIVE = "adaptive"


class ProblemType(str, Enum):
    """Top-level shape inferred by the planner in its existing single call."""

    LOOKUP = "lookup"
    COMPARISON = "comparison"
    SCENARIO_APPLICATION = "scenario_application"
    PROCEDURE = "procedure"
    MIXED = "mixed"


class TaskKind(str, Enum):
    """Execution primitive used for one node in the task DAG."""

    EVIDENCE = "evidence"
    APPLICATION = "application"
    INTEGRATION = "integration"


class AnswerRequirementKind(str, Enum):
    """User-visible result a plan must ultimately satisfy."""

    RULE = "rule"
    OUTCOME = "outcome"
    COMPARISON = "comparison"
    PROCEDURE = "procedure"
    CALCULATION = "calculation"
    FORMAT = "format"
    LIMITATION = "limitation"


class EvidenceRequirementKind(str, Enum):
    """Legal or factual support an evidence task must retrieve."""

    GOVERNING_RULE = "governing_rule"
    DEFINITION = "definition"
    THRESHOLD = "threshold"
    EXCEPTION = "exception"
    PROCEDURE = "procedure"
    DATE_VALIDITY = "date_validity"
    CROSS_REFERENCE = "cross_reference"
    CONFLICT_CHECK = "conflict_check"


class WorkerStatus(str, Enum):
    """Outcome of one search worker assignment."""

    SUCCESS = "success"
    NO_EVIDENCE = "no_evidence"
    FAILED = "failed"


class TaskStatus(str, Enum):
    """Coverage state of one task artifact."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class EvidenceConfidence(str, Enum):
    """Qualitative support strength for an extracted claim or conclusion."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AnswerRequirement(_StrictArtifact):
    """One explicit user-visible obligation of the final answer."""

    requirement_id: str = Field(description="Stable unique requirement ID")
    kind: AnswerRequirementKind = Field(description="Required answer shape")
    description: str = Field(description="Observable obligation to satisfy")
    required: bool = Field(description="Whether omission makes the answer incomplete")


class ScenarioFact(_StrictArtifact):
    """One user-provided fact; it is input, not externally verified evidence."""

    fact_id: str = Field(description="Stable unique scenario fact ID")
    description: str = Field(description="Fact stated or explicitly confirmed by user")
    requirement_ids: list[str] = Field(
        description="Answer requirements materially affected by this fact"
    )


class MaterialUnknown(_StrictArtifact):
    """Missing fact that can change a scenario outcome."""

    unknown_id: str = Field(description="Stable unique material-unknown ID")
    description: str = Field(description="Fact that remains unknown")
    why_material: str = Field(description="How the unknown can change the result")
    requirement_ids: list[str] = Field(
        description="Answer requirements affected by this unknown"
    )


class DecisionBranch(_StrictArtifact):
    """One explicit conditional path that final synthesis must preserve."""

    branch_id: str = Field(description="Stable unique decision-branch ID")
    condition: str = Field(description="Condition selecting this branch")
    consequence: str = Field(description="Outcome or issue if the condition holds")
    requirement_ids: list[str] = Field(
        description="Answer requirements this branch helps resolve"
    )


class ScenarioSpec(_StrictArtifact):
    """Bounded scenario inputs shared only through explicit task references."""

    jurisdiction: str | None = Field(
        description="Applicable jurisdiction when supplied or safely resolved"
    )
    law_as_of_date: str | None = Field(
        description="YYYY-MM-DD legal cut-off date when material, otherwise null"
    )
    facts: list[ScenarioFact] = Field(description="User-provided scenario facts")
    material_unknowns: list[MaterialUnknown] = Field(
        description="Missing facts that can alter an outcome"
    )
    decision_branches: list[DecisionBranch] = Field(
        description="Conditional paths caused by material unknowns or legal tests"
    )


class EvidenceRequirement(_StrictArtifact):
    """One typed support target inside an evidence task."""

    evidence_requirement_id: str = Field(
        description="Stable unique evidence requirement ID within the plan"
    )
    kind: EvidenceRequirementKind = Field(description="Support category")
    description: str = Field(description="Provision or fact evidence must establish")
    requirement_ids: list[str] = Field(
        description="Answer requirements supported when this evidence is found"
    )


class TaskOutputRef(_StrictArtifact):
    """Reference to one declared output of a dependency task."""

    task_id: str = Field(description="Dependency task ID")
    output_id: str = Field(description="Output ID declared by that dependency")


class TaskOutput(_StrictArtifact):
    """One typed data product made available to downstream tasks."""

    output_id: str = Field(description="Stable output ID unique within the task")
    description: str = Field(description="What the downstream artifact provides")


class TaskSpec(_StrictArtifact):
    """One bounded evidence, application, or integration node in the DAG."""

    task_id: str = Field(description="Stable unique ID within this plan")
    kind: TaskKind = Field(description="Server execution primitive")
    issue: str = Field(description="Standalone issue this task must resolve")
    search_question: str | None = Field(
        description="Standalone indexed query for evidence tasks, otherwise null"
    )
    requirement_ids: list[str] = Field(
        description="Answer requirements this task contributes to"
    )
    fact_ids: list[str] = Field(
        description="Scenario facts this task may receive; [] outside scenarios"
    )
    unknown_ids: list[str] = Field(
        description="Material unknowns this task must preserve"
    )
    branch_ids: list[str] = Field(
        description="Decision branches this task must evaluate"
    )
    evidence_requirements: list[EvidenceRequirement] = Field(
        description="Typed retrieval targets for evidence tasks; [] otherwise"
    )
    consumes: list[TaskOutputRef] = Field(
        description="Declared dependency outputs available to this task"
    )
    produces: list[TaskOutput] = Field(
        description="Declared outputs made available to downstream tasks"
    )
    required: bool = Field(
        description="Whether final output is incomplete if this task fails"
    )
    as_of_date: str | None = Field(
        description="YYYY-MM-DD for historical evidence retrieval, otherwise null"
    )
    filters: str | None = Field(
        description="Explicit reliable metadata filter, otherwise null"
    )

    @property
    def question(self) -> str:
        """Compatibility read for code that previously used ``question``."""

        return self.search_question or self.issue

    @property
    def depends_on(self) -> list[str]:
        """Unique dependency task IDs derived from typed consumed outputs."""

        return list(dict.fromkeys(reference.task_id for reference in self.consumes))

    @property
    def purpose(self) -> str:
        """Compatibility read for compact progress and diagnostics."""

        return self.issue

    @property
    def expected_output(self) -> str:
        """Compatibility read for coordinator objectives."""

        return "; ".join(output.description for output in self.produces)

    @property
    def success_criteria(self) -> list[str]:
        """Compatibility read; v3 execution should prefer stable IDs."""

        return [
            requirement.description for requirement in self.evidence_requirements
        ] or list(self.requirement_ids)


class GlobalPlan(_StrictArtifact):
    """Single-call routing and typed decision/research plan."""

    version: Literal["3"] = Field(description="Orchestration contract version")
    problem_type: ProblemType = Field(
        description="Question shape inferred in this call"
    )
    mode: PlanMode = Field(description="direct for one task; decomposed for a DAG")
    execution_strategy: ExecutionStrategy = Field(
        description="single_pass only for one precise lookup; adaptive otherwise"
    )
    normalized_question: str = Field(
        description="Standalone question with only necessary references resolved"
    )
    answer_requirements: list[AnswerRequirement] = Field(
        description="Every explicit user-visible obligation"
    )
    scenario: ScenarioSpec | None = Field(
        description="Scenario decision inputs, otherwise null"
    )
    tasks: list[TaskSpec] = Field(
        description="Non-overlapping nodes that collectively satisfy requirements"
    )
    synthesis_requirements: list[str] = Field(
        description="Cross-task comparison, ordering, or output requirements"
    )
    assumptions: list[str] = Field(
        description="Minimal assumptions made because input was ambiguous"
    )


class SearchAssignment(_StrictArtifact):
    """One independent, bounded assignment for a search worker."""

    assignment_id: str = Field(description="Unique assignment ID within the task")
    task_id: str = Field(description="Owning evidence TaskSpec ID")
    query: str = Field(description="Standalone indexed-search query")
    objective: str = Field(description="Specific evidence question to resolve")
    evidence_requirements: list[str] = Field(
        description="Stable evidence_requirement_ids this search targets"
    )
    excluded_queries: list[str] = Field(
        description="Already-run queries this assignment must not duplicate"
    )
    as_of_date: str | None = Field(
        description="YYYY-MM-DD for historical retrieval, otherwise null"
    )
    filters: str | None = Field(
        description="Explicit reliable metadata filter, otherwise null"
    )


class SearchAssignmentBatch(_StrictArtifact):
    """Coordinator dispatch for one bounded worker wave."""

    task_id: str = Field(description="Evidence task being coordinated")
    stop: bool = Field(description="True when no further search is warranted")
    stop_reason: str | None = Field(
        description="Concise coverage or no-novelty reason when stopping"
    )
    assignments: list[SearchAssignment] = Field(
        description="Independent non-duplicate assignments for this wave"
    )


class EvidenceClaim(_StrictArtifact):
    """One atomic proposition supported by one exact indexed chunk."""

    claim_id: str = Field(description="Stable claim ID copied by downstream tasks")
    claim: str = Field(description="Atomic factual or legal proposition")
    document_id: str = Field(description="Exact indexed document ID")
    chunk_id: str = Field(description="Exact indexed chunk ID")
    readable_title: str = Field(description="Human-readable document title")
    locator: str = Field(description="Article, section, paragraph, or heading locator")
    evidence_excerpt: str = Field(
        description="Minimal verbatim excerpt directly supporting the claim"
    )
    requirement_ids: list[str] = Field(
        description="Answer requirements supported by this claim"
    )
    evidence_requirement_ids: list[str] = Field(
        description="Task evidence requirements supported by this claim"
    )
    fact_ids: list[str] = Field(
        description="Scenario fact IDs this evidence directly bears on"
    )
    confidence: EvidenceConfidence = Field(description="Support strength")
    effective_start_date: str | None = Field(
        description="Evidence validity start date when stated, otherwise null"
    )
    effective_end_date: str | None = Field(
        description="Evidence validity end date when stated, otherwise null"
    )

    @property
    def supports_success_criteria(self) -> list[str]:
        """Compatibility read for the v2 criterion-oriented executor."""

        return list(self.evidence_requirement_ids)


class WorkerArtifact(_StrictArtifact):
    """Compact output of one isolated search worker."""

    task_id: str = Field(description="Owning evidence TaskSpec ID")
    assignment_id: str = Field(description="Completed SearchAssignment ID")
    worker_id: str = Field(description="Runtime worker ID")
    status: WorkerStatus = Field(description="Worker outcome")
    searches_run: list[str] = Field(description="Exact search queries executed")
    claims: list[EvidenceClaim] = Field(
        description="Atomic claims extracted from supplied indexed hits"
    )
    gaps: list[str] = Field(description="Unsupported evidence requirement IDs")
    cross_references: list[str] = Field(
        description="Explicit unresolved references named by supplied evidence"
    )
    error_code: str | None = Field(
        description="Stable execution error code only for failed status"
    )
    error_message: str | None = Field(
        description="Sanitized execution failure summary, otherwise null"
    )


class DerivedConclusion(_StrictArtifact):
    """Scenario/application finding whose provenance can be checked by server."""

    conclusion_id: str = Field(description="Stable unique finding ID in this task")
    finding: str = Field(description="User-relevant applied conclusion")
    requirement_ids: list[str] = Field(
        description="Answer requirements resolved by this finding"
    )
    fact_ids: list[str] = Field(
        description="Assigned user-provided facts applied in this finding"
    )
    branch_ids: list[str] = Field(
        description="Assigned conditional branches addressed by this finding"
    )
    supporting_claim_ids: list[str] = Field(
        description="Verified dependency claim IDs supporting the legal rule"
    )
    dependency_refs: list[TaskOutputRef] = Field(
        description="Declared dependency outputs used in this finding"
    )
    confidence: EvidenceConfidence = Field(
        description="Confidence based only on supplied facts and artifacts"
    )
    limitations: list[str] = Field(
        description="Material unknowns or evidence limits qualifying the result"
    )


class TaskArtifact(_StrictArtifact):
    """Validated evidence or application result consumed by the DAG."""

    task_id: str = Field(description="Completed TaskSpec ID")
    status: TaskStatus = Field(description="Task coverage state")
    answer_fragment: str | None = Field(
        description="Grounded task conclusion, or null when unsupported"
    )
    covered_requirement_ids: list[str] = Field(
        description="Task answer requirements fully supported"
    )
    uncovered_requirement_ids: list[str] = Field(
        description="Task answer requirements not fully supported"
    )
    claims: list[EvidenceClaim] = Field(
        description="Verified claims retained by evidence tasks"
    )
    application_findings: list[DerivedConclusion] = Field(
        description="Reference-grounded findings from application/integration tasks"
    )
    conflicts: list[str] = Field(
        description="Material source conflicts final synthesis must disclose"
    )
    gaps: list[str] = Field(
        description="Material unanswered issues or missing evidence/facts"
    )
    contributing_worker_ids: list[str] = Field(
        description="Workers whose verified claims contribute to this artifact"
    )

    @property
    def covered_success_criteria(self) -> list[str]:
        """Compatibility read for v2 UI/progress code."""

        return list(self.covered_requirement_ids)

    @property
    def uncovered_success_criteria(self) -> list[str]:
        """Compatibility read for v2 UI/progress code."""

        return list(self.uncovered_requirement_ids)


class PlanValidationError(ValueError):
    """One or more deterministic plan-integrity failures."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("Invalid global plan: " + "; ".join(self.errors))


class ArtifactValidationError(ValueError):
    """One or more deterministic task-artifact grounding failures."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("Invalid task artifact: " + "; ".join(self.errors))


def _validate_id(
    value: str,
    *,
    label: str,
    errors: list[str],
) -> None:
    if not value.strip():
        errors.append(f"{label} must not be blank")
    elif value != value.strip():
        errors.append(f"{label} must not have surrounding whitespace: {value!r}")
    elif _ID_PATTERN.fullmatch(value) is None:
        errors.append(f"{label} must be a 1-64 character safe identifier: {value!r}")


def _duplicates(values: Sequence[str]) -> list[str]:
    return sorted(value for value in set(values) if values.count(value) > 1)


def _validate_refs(
    values: Sequence[str],
    *,
    known: set[str],
    label: str,
    errors: list[str],
) -> None:
    duplicate_values = _duplicates(values)
    if duplicate_values:
        errors.append(f"{label} repeats IDs: {', '.join(duplicate_values)}")
    unknown = sorted(set(values) - known)
    if unknown:
        errors.append(f"{label} references unknown IDs: {', '.join(unknown)}")


def validate_global_plan(
    plan: GlobalPlan,
    *,
    max_tasks: int | None = 5,
    max_list_items: int | None = 12,
) -> GlobalPlan:
    """Validate coverage, scenario semantics, typed dataflow, and acyclicity."""

    if max_tasks is not None and max_tasks < 1:
        raise ValueError("max_tasks must be at least 1")
    if max_list_items is not None and max_list_items < 1:
        raise ValueError("max_list_items must be at least 1")

    errors: list[str] = []
    task_count = len(plan.tasks)
    if plan.mode == PlanMode.DIRECT and task_count != 1:
        errors.append("direct mode must contain exactly one task")
    if plan.mode == PlanMode.DECOMPOSED:
        if plan.execution_strategy != ExecutionStrategy.ADAPTIVE:
            errors.append("decomposed mode must use adaptive execution")
        if task_count < 2:
            errors.append("decomposed mode must contain at least two tasks")
    if max_tasks is not None and task_count > max_tasks:
        errors.append(f"plan exceeds the maximum of {max_tasks} tasks")

    if not plan.normalized_question.strip():
        errors.append("normalized_question must not be blank")
    if not plan.answer_requirements:
        errors.append("answer_requirements must not be empty")
    if max_list_items is not None and len(plan.answer_requirements) > max_list_items:
        errors.append(
            f"answer_requirements exceeds the maximum of {max_list_items} items"
        )
    if max_list_items is not None and len(plan.synthesis_requirements) > max_list_items:
        errors.append(
            f"synthesis_requirements exceeds the maximum of {max_list_items} items"
        )
    if max_list_items is not None and len(plan.assumptions) > max_list_items:
        errors.append(f"assumptions exceeds the maximum of {max_list_items} items")
    if any(not requirement.strip() for requirement in plan.synthesis_requirements):
        errors.append("synthesis_requirements must not contain blank items")
    if any(not assumption.strip() for assumption in plan.assumptions):
        errors.append("assumptions must not contain blank items")

    requirement_ids = [
        requirement.requirement_id for requirement in plan.answer_requirements
    ]
    requirement_id_set = set(requirement_ids)
    duplicate_requirement_ids = _duplicates(requirement_ids)
    if duplicate_requirement_ids:
        errors.append(
            "answer requirement IDs must be unique: "
            + ", ".join(duplicate_requirement_ids)
        )
    for requirement in plan.answer_requirements:
        _validate_id(
            requirement.requirement_id,
            label="requirement_id",
            errors=errors,
        )
        if not requirement.description.strip():
            errors.append(
                f"requirement {requirement.requirement_id!r} description must not be blank"
            )

    scenario = plan.scenario
    fact_ids: set[str] = set()
    unknown_ids: set[str] = set()
    branch_ids: set[str] = set()
    if scenario is not None:
        if not scenario.facts:
            errors.append("scenario must contain at least one user-provided fact")
        if scenario.jurisdiction is not None and not scenario.jurisdiction.strip():
            errors.append("scenario jurisdiction must be null or non-blank")
        for collection, field, label in (
            (scenario.facts, "fact_id", "scenario fact"),
            (scenario.material_unknowns, "unknown_id", "material unknown"),
            (scenario.decision_branches, "branch_id", "decision branch"),
        ):
            if max_list_items is not None and len(collection) > max_list_items:
                errors.append(
                    f"{label} list exceeds the maximum of {max_list_items} items"
                )
            ids = [getattr(item, field) for item in collection]
            duplicate_ids = _duplicates(ids)
            if duplicate_ids:
                errors.append(f"{label} IDs must be unique: {', '.join(duplicate_ids)}")
            for item in collection:
                item_id = getattr(item, field)
                _validate_id(item_id, label=f"{label} ID", errors=errors)
                _validate_refs(
                    item.requirement_ids,
                    known=requirement_id_set,
                    label=f"{label} {item_id!r} requirement_ids",
                    errors=errors,
                )
                if not item.requirement_ids:
                    errors.append(
                        f"{label} {item_id!r} must map at least one answer requirement"
                    )
        for fact in scenario.facts:
            if not fact.description.strip():
                errors.append(
                    f"scenario fact {fact.fact_id!r} description must not be blank"
                )
        for unknown in scenario.material_unknowns:
            if not unknown.description.strip() or not unknown.why_material.strip():
                errors.append(
                    f"material unknown {unknown.unknown_id!r} description and "
                    "why_material must not be blank"
                )
        for branch in scenario.decision_branches:
            if not branch.condition.strip() or not branch.consequence.strip():
                errors.append(
                    f"decision branch {branch.branch_id!r} condition and "
                    "consequence must not be blank"
                )
        fact_ids = {fact.fact_id for fact in scenario.facts}
        unknown_ids = {unknown.unknown_id for unknown in scenario.material_unknowns}
        branch_ids = {branch.branch_id for branch in scenario.decision_branches}

    if plan.problem_type == ProblemType.SCENARIO_APPLICATION and scenario is None:
        errors.append("scenario_application plans must define scenario inputs")
    if scenario is not None and plan.problem_type not in {
        ProblemType.SCENARIO_APPLICATION,
        ProblemType.MIXED,
    }:
        errors.append(
            "scenario inputs require scenario_application or mixed problem type"
        )
    if plan.execution_strategy == ExecutionStrategy.SINGLE_PASS and (
        plan.problem_type != ProblemType.LOOKUP
        or plan.mode != PlanMode.DIRECT
        or task_count != 1
    ):
        errors.append("single_pass is allowed only for one direct lookup task")
    if plan.problem_type == ProblemType.LOOKUP and (
        plan.mode != PlanMode.DIRECT or task_count != 1
    ):
        errors.append("lookup plans must use one direct task")

    task_ids = [task.task_id for task in plan.tasks]
    task_id_set = set(task_ids)
    task_kind_by_id = {task.task_id: task.kind for task in plan.tasks}
    duplicate_task_ids = _duplicates(task_ids)
    if duplicate_task_ids:
        errors.append(f"task IDs must be unique: {', '.join(duplicate_task_ids)}")

    outputs_by_task: dict[str, set[str]] = {}
    evidence_requirement_ids: set[str] = set()
    evidence_query_owners: dict[tuple[str, str, str], str] = {}
    for task in plan.tasks:
        label = task.task_id or "<blank>"
        _validate_id(task.task_id, label="task_id", errors=errors)
        if not task.issue.strip():
            errors.append(f"task {label!r} issue must not be blank")
        if not task.requirement_ids:
            errors.append(f"task {label!r} must map at least one answer requirement")
        _validate_refs(
            task.requirement_ids,
            known=requirement_id_set,
            label=f"task {label!r} requirement_ids",
            errors=errors,
        )
        _validate_refs(
            task.fact_ids,
            known=fact_ids,
            label=f"task {label!r} fact_ids",
            errors=errors,
        )
        _validate_refs(
            task.unknown_ids,
            known=unknown_ids,
            label=f"task {label!r} unknown_ids",
            errors=errors,
        )
        _validate_refs(
            task.branch_ids,
            known=branch_ids,
            label=f"task {label!r} branch_ids",
            errors=errors,
        )
        for field_name, values in (
            ("requirement_ids", task.requirement_ids),
            ("fact_ids", task.fact_ids),
            ("unknown_ids", task.unknown_ids),
            ("branch_ids", task.branch_ids),
            ("evidence_requirements", task.evidence_requirements),
            ("consumes", task.consumes),
            ("produces", task.produces),
        ):
            if max_list_items is not None and len(values) > max_list_items:
                errors.append(
                    f"task {label!r} {field_name} exceeds the maximum of "
                    f"{max_list_items} items"
                )

        output_ids = [output.output_id for output in task.produces]
        outputs_by_task[task.task_id] = set(output_ids)
        if not output_ids:
            errors.append(f"task {label!r} must declare at least one output")
        duplicate_outputs = _duplicates(output_ids)
        if duplicate_outputs:
            errors.append(
                f"task {label!r} output IDs must be unique: "
                + ", ".join(duplicate_outputs)
            )
        for output in task.produces:
            _validate_id(
                output.output_id,
                label=f"task {label!r} output_id",
                errors=errors,
            )
            if not output.description.strip():
                errors.append(
                    f"task {label!r} output {output.output_id!r} "
                    "description must not be blank"
                )

        if task.kind == TaskKind.EVIDENCE:
            if not (task.search_question or "").strip():
                errors.append(f"evidence task {label!r} requires search_question")
            if not task.evidence_requirements:
                errors.append(f"evidence task {label!r} requires evidence requirements")
            query_key = (
                " ".join((task.search_question or "").casefold().split()),
                " ".join((task.filters or "").casefold().split()),
                (task.as_of_date or "").strip(),
            )
            previous_owner = evidence_query_owners.get(query_key)
            if query_key[0] and previous_owner is not None:
                errors.append(
                    f"evidence tasks {previous_owner!r} and {label!r} repeat the "
                    "same search question; consolidate shared evidence"
                )
            elif query_key[0]:
                evidence_query_owners[query_key] = task.task_id
        else:
            if task.search_question is not None:
                errors.append(f"{task.kind.value} task {label!r} must not search")
            if task.evidence_requirements:
                errors.append(
                    f"{task.kind.value} task {label!r} must not define retrieval targets"
                )
            if not task.consumes:
                errors.append(
                    f"{task.kind.value} task {label!r} must consume dependency output"
                )
        if task.kind == TaskKind.APPLICATION:
            if scenario is None:
                errors.append(f"application task {label!r} requires a scenario")
            elif not task.fact_ids:
                errors.append(
                    f"application task {label!r} must reference scenario facts"
                )

        local_evidence_ids = [
            requirement.evidence_requirement_id
            for requirement in task.evidence_requirements
        ]
        duplicate_evidence_ids = _duplicates(local_evidence_ids)
        if duplicate_evidence_ids:
            errors.append(
                f"task {label!r} evidence requirement IDs must be unique: "
                + ", ".join(duplicate_evidence_ids)
            )
        for requirement in task.evidence_requirements:
            _validate_id(
                requirement.evidence_requirement_id,
                label=f"task {label!r} evidence_requirement_id",
                errors=errors,
            )
            if requirement.evidence_requirement_id in evidence_requirement_ids:
                errors.append(
                    "evidence requirement IDs must be plan-global unique: "
                    f"{requirement.evidence_requirement_id!r}"
                )
            evidence_requirement_ids.add(requirement.evidence_requirement_id)
            if not requirement.description.strip():
                errors.append(
                    f"evidence requirement {requirement.evidence_requirement_id!r} "
                    "description must not be blank"
                )
            _validate_refs(
                requirement.requirement_ids,
                known=set(task.requirement_ids),
                label=(
                    f"evidence requirement "
                    f"{requirement.evidence_requirement_id!r} requirement_ids"
                ),
                errors=errors,
            )

    for task in plan.tasks:
        seen_consumes: set[tuple[str, str]] = set()
        for reference in task.consumes:
            key = (reference.task_id, reference.output_id)
            if key in seen_consumes:
                errors.append(
                    f"task {task.task_id!r} repeats dependency output "
                    f"{reference.task_id!r}/{reference.output_id!r}"
                )
                continue
            seen_consumes.add(key)
            if reference.task_id not in task_id_set:
                errors.append(
                    f"task {task.task_id!r} consumes unknown task {reference.task_id!r}"
                )
            elif reference.task_id == task.task_id:
                errors.append(f"task {task.task_id!r} cannot depend on itself")
            elif reference.output_id not in outputs_by_task.get(
                reference.task_id, set()
            ):
                errors.append(
                    f"task {task.task_id!r} consumes unknown output "
                    f"{reference.task_id!r}/{reference.output_id!r}"
                )
        if task.kind == TaskKind.APPLICATION and not any(
            reference.task_id in task_id_set
            and task_kind_by_id.get(reference.task_id) == TaskKind.EVIDENCE
            for reference in task.consumes
        ):
            errors.append(
                f"application task {task.task_id!r} must consume evidence task output"
            )

    required_ids = {
        requirement.requirement_id
        for requirement in plan.answer_requirements
        if requirement.required and requirement.kind != AnswerRequirementKind.FORMAT
    }
    mapped_required_ids = {
        requirement_id for task in plan.tasks for requirement_id in task.requirement_ids
    }
    missing_required = sorted(required_ids - mapped_required_ids)
    if missing_required:
        errors.append(
            "required answer requirements are not mapped to tasks: "
            + ", ".join(missing_required)
        )
    required_format_ids = {
        requirement.requirement_id
        for requirement in plan.answer_requirements
        if requirement.required and requirement.kind == AnswerRequirementKind.FORMAT
    }
    if required_format_ids and not plan.synthesis_requirements:
        errors.append(
            "required format requirements must be represented in synthesis_requirements"
        )

    if scenario is not None:
        application_tasks = [
            task for task in plan.tasks if task.kind == TaskKind.APPLICATION
        ]
        if not application_tasks:
            errors.append("scenario plans require at least one application task")
        for item, item_id, label in (
            *((fact, fact.fact_id, "scenario fact") for fact in scenario.facts),
            *(
                (unknown, unknown.unknown_id, "material unknown")
                for unknown in scenario.material_unknowns
            ),
            *(
                (branch, branch.branch_id, "decision branch")
                for branch in scenario.decision_branches
            ),
        ):
            id_field = (
                "fact_ids"
                if label == "scenario fact"
                else "unknown_ids"
                if label == "material unknown"
                else "branch_ids"
            )
            receiving_tasks = [
                task for task in application_tasks if item_id in getattr(task, id_field)
            ]
            if not receiving_tasks:
                errors.append(
                    f"{label} {item_id!r} is not mapped to an application task"
                )
                continue
            missing_item_requirements = [
                requirement_id
                for requirement_id in item.requirement_ids
                if not any(
                    requirement_id in task.requirement_ids for task in receiving_tasks
                )
            ]
            if missing_item_requirements:
                errors.append(
                    f"{label} {item_id!r} requirement_ids are not carried by "
                    "its application task: " + ", ".join(missing_item_requirements)
                )

    dependency_map = {
        task.task_id: [
            dependency
            for dependency in task.depends_on
            if dependency in task_id_set and dependency != task.task_id
        ]
        for task in plan.tasks
    }
    visiting: set[str] = set()
    visited: set[str] = set()
    cycle_found = False

    def visit(task_id: str) -> None:
        nonlocal cycle_found
        if task_id in visited or cycle_found:
            return
        if task_id in visiting:
            cycle_found = True
            return
        visiting.add(task_id)
        for dependency in dependency_map.get(task_id, []):
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in task_ids:
        visit(task_id)
    if cycle_found:
        errors.append("task dependencies must form an acyclic graph")

    if errors:
        raise PlanValidationError(errors)
    return plan


def validate_task_artifact(
    artifact: TaskArtifact,
    *,
    plan: GlobalPlan,
    dependency_artifacts: Sequence[TaskArtifact] = (),
) -> TaskArtifact:
    """Reject references a task result could not have learned at its boundary."""

    errors: list[str] = []
    specs = {task.task_id: task for task in plan.tasks}
    spec = specs.get(artifact.task_id)
    if spec is None:
        raise ArtifactValidationError(
            [f"artifact references unknown task {artifact.task_id!r}"]
        )

    task_requirement_ids = set(spec.requirement_ids)
    task_evidence_ids = {
        requirement.evidence_requirement_id
        for requirement in spec.evidence_requirements
    }
    task_fact_ids = set(spec.fact_ids)
    task_branch_ids = set(spec.branch_ids)
    dependency_by_id = {
        dependency.task_id: dependency for dependency in dependency_artifacts
    }
    allowed_dependency_refs = {
        (reference.task_id, reference.output_id) for reference in spec.consumes
    }
    allowed_claim_ids = {
        claim.claim_id
        for dependency in dependency_artifacts
        for claim in dependency.claims
    } | {
        claim_id
        for dependency in dependency_artifacts
        for finding in dependency.application_findings
        for claim_id in finding.supporting_claim_ids
    }

    claim_ids = [claim.claim_id for claim in artifact.claims]
    duplicate_claim_ids = _duplicates(claim_ids)
    if duplicate_claim_ids:
        errors.append(f"claim IDs must be unique: {', '.join(duplicate_claim_ids)}")
    for claim in artifact.claims:
        _validate_id(claim.claim_id, label="claim_id", errors=errors)
        if not claim.claim.strip() or not claim.evidence_excerpt.strip():
            errors.append(f"claim {claim.claim_id!r} must contain claim and excerpt")
        if not claim.document_id.strip() or not claim.chunk_id.strip():
            errors.append(f"claim {claim.claim_id!r} must reference an indexed source")
        _validate_refs(
            claim.requirement_ids,
            known=task_requirement_ids,
            label=f"claim {claim.claim_id!r} requirement_ids",
            errors=errors,
        )
        _validate_refs(
            claim.evidence_requirement_ids,
            known=task_evidence_ids,
            label=f"claim {claim.claim_id!r} evidence_requirement_ids",
            errors=errors,
        )
        _validate_refs(
            claim.fact_ids,
            known=task_fact_ids,
            label=f"claim {claim.claim_id!r} fact_ids",
            errors=errors,
        )

    finding_ids = [finding.conclusion_id for finding in artifact.application_findings]
    duplicate_finding_ids = _duplicates(finding_ids)
    if duplicate_finding_ids:
        errors.append(
            f"application finding IDs must be unique: {', '.join(duplicate_finding_ids)}"
        )
    if spec.kind == TaskKind.EVIDENCE and artifact.application_findings:
        errors.append("evidence task artifacts must not contain application findings")
    if spec.kind != TaskKind.EVIDENCE and artifact.claims:
        errors.append(
            "application/integration artifacts must not create evidence claims"
        )

    for finding in artifact.application_findings:
        _validate_id(
            finding.conclusion_id,
            label="conclusion_id",
            errors=errors,
        )
        if not finding.finding.strip():
            errors.append(
                f"application finding {finding.conclusion_id!r} must not be blank"
            )
        _validate_refs(
            finding.requirement_ids,
            known=task_requirement_ids,
            label=f"finding {finding.conclusion_id!r} requirement_ids",
            errors=errors,
        )
        _validate_refs(
            finding.fact_ids,
            known=task_fact_ids,
            label=f"finding {finding.conclusion_id!r} fact_ids",
            errors=errors,
        )
        _validate_refs(
            finding.branch_ids,
            known=task_branch_ids,
            label=f"finding {finding.conclusion_id!r} branch_ids",
            errors=errors,
        )
        _validate_refs(
            finding.supporting_claim_ids,
            known=allowed_claim_ids,
            label=f"finding {finding.conclusion_id!r} supporting_claim_ids",
            errors=errors,
        )
        reference_keys = [
            (reference.task_id, reference.output_id)
            for reference in finding.dependency_refs
        ]
        if len(set(reference_keys)) != len(reference_keys):
            errors.append(f"finding {finding.conclusion_id!r} repeats dependency refs")
        for reference in finding.dependency_refs:
            key = (reference.task_id, reference.output_id)
            if key not in allowed_dependency_refs:
                errors.append(
                    f"finding {finding.conclusion_id!r} references unavailable "
                    f"dependency output {reference.task_id!r}/{reference.output_id!r}"
                )
            elif reference.task_id not in dependency_by_id:
                errors.append(
                    f"finding {finding.conclusion_id!r} references missing "
                    f"dependency artifact {reference.task_id!r}"
                )
        if spec.kind == TaskKind.APPLICATION and not finding.fact_ids:
            errors.append(
                f"application finding {finding.conclusion_id!r} must apply a fact"
            )
        if spec.kind == TaskKind.APPLICATION and not finding.dependency_refs:
            errors.append(
                f"application finding {finding.conclusion_id!r} must reference "
                "a dependency output"
            )
        if spec.kind == TaskKind.INTEGRATION and not finding.supporting_claim_ids:
            errors.append(
                f"integration finding {finding.conclusion_id!r} must reference "
                "an upstream supporting claim"
            )
        if spec.kind == TaskKind.INTEGRATION and not finding.dependency_refs:
            errors.append(
                f"integration finding {finding.conclusion_id!r} must reference "
                "a dependency output"
            )
        if (
            spec.kind == TaskKind.APPLICATION
            and spec.unknown_ids
            and not finding.limitations
        ):
            errors.append(
                f"application finding {finding.conclusion_id!r} must preserve "
                "material-unknown limitations"
            )
        if not finding.supporting_claim_ids and not finding.dependency_refs:
            errors.append(
                f"finding {finding.conclusion_id!r} has no grounded dependency support"
            )

    supported_requirement_ids = {
        requirement_id
        for claim in artifact.claims
        for requirement_id in claim.requirement_ids
    } | {
        requirement_id
        for finding in artifact.application_findings
        for requirement_id in finding.requirement_ids
    }
    expected_covered = [
        requirement_id
        for requirement_id in spec.requirement_ids
        if requirement_id in supported_requirement_ids
    ]
    expected_uncovered = [
        requirement_id
        for requirement_id in spec.requirement_ids
        if requirement_id not in supported_requirement_ids
    ]
    if artifact.covered_requirement_ids != expected_covered:
        errors.append("covered_requirement_ids do not match grounded artifact support")
    if artifact.uncovered_requirement_ids != expected_uncovered:
        errors.append(
            "uncovered_requirement_ids do not match grounded artifact support"
        )
    expected_status = (
        TaskStatus.COMPLETE
        if supported_requirement_ids and not expected_uncovered
        else TaskStatus.PARTIAL
        if supported_requirement_ids
        else TaskStatus.FAILED
    )
    if artifact.status != expected_status:
        errors.append(
            f"task status must be {expected_status.value!r} for grounded coverage"
        )
    if spec.kind == TaskKind.APPLICATION and artifact.status == TaskStatus.COMPLETE:
        applied_fact_ids = {
            fact_id
            for finding in artifact.application_findings
            for fact_id in finding.fact_ids
        }
        evaluated_branch_ids = {
            branch_id
            for finding in artifact.application_findings
            for branch_id in finding.branch_ids
        }
        if set(spec.fact_ids) - applied_fact_ids:
            errors.append("complete application artifact omits assigned scenario facts")
        if set(spec.branch_ids) - evaluated_branch_ids:
            errors.append(
                "complete application artifact omits assigned decision branches"
            )

    if errors:
        raise ArtifactValidationError(errors)
    return artifact
