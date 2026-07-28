"""Typed contracts and deterministic validation for multi-agent research.

The models in this module are deliberately independent from the legacy
``Action`` union.  They are small artifacts passed between orchestration
roles, not chat transcripts.  LLM-facing models forbid undeclared fields so
provider-side structured output and server-side parsing share one contract.

Semantic plan rules (task counts, dependency integrity, and acyclicity) live
in :func:`validate_global_plan` rather than JSON Schema.  This keeps the schema
portable across model providers and makes fallback behavior deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class _StrictArtifact(BaseModel):
    """Base for exact structured-output artifacts."""

    model_config = ConfigDict(extra="forbid")


class PlanMode(str, Enum):
    """How the global planner routes a user question."""

    DIRECT = "direct"
    DECOMPOSED = "decomposed"


class WorkerStatus(str, Enum):
    """Outcome of one search worker assignment."""

    SUCCESS = "success"
    NO_EVIDENCE = "no_evidence"
    FAILED = "failed"


class TaskStatus(str, Enum):
    """Coverage state of one research task."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class EvidenceConfidence(str, Enum):
    """Qualitative support strength for an extracted evidence claim."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskSpec(_StrictArtifact):
    """One bounded research problem in the planner-generated task DAG."""

    task_id: str = Field(
        description="Stable unique ID within this plan, such as task_1"
    )
    question: str = Field(description="Standalone research question for this task only")
    purpose: str = Field(
        description="Why this task is necessary for the original answer"
    )
    expected_output: str = Field(
        description="Concise description of the artifact this task must produce"
    )
    success_criteria: list[str] = Field(
        description="Observable evidence requirements that define task completion"
    )
    depends_on: list[str] = Field(
        description="Task IDs whose artifacts are required before this task starts"
    )
    required: bool = Field(
        description="Whether the final answer is incomplete if this task fails"
    )
    as_of_date: str | None = Field(
        description="YYYY-MM-DD for a historical-law task, otherwise null"
    )
    filters: str | None = Field(
        description="Explicit reliable metadata filter, otherwise null"
    )


class GlobalPlan(_StrictArtifact):
    """Single-call routing and research plan produced by the global planner."""

    version: Literal["1"] = Field(description="Orchestration contract version")
    mode: PlanMode = Field(
        description="direct for one coherent task; decomposed for a task DAG"
    )
    normalized_question: str = Field(
        description="Standalone user question with only necessary references resolved"
    )
    answer_requirements: list[str] = Field(
        description="Every explicit requirement the final answer must satisfy"
    )
    tasks: list[TaskSpec] = Field(
        description="Non-overlapping tasks that collectively satisfy the question"
    )
    synthesis_requirements: list[str] = Field(
        description="Cross-task comparisons or formatting required in the final answer"
    )
    assumptions: list[str] = Field(
        description="Minimal explicit assumptions made because input was ambiguous"
    )


class SearchAssignment(_StrictArtifact):
    """One independent, bounded assignment for a search worker."""

    assignment_id: str = Field(description="Unique assignment ID within the task")
    task_id: str = Field(description="Owning TaskSpec ID")
    query: str = Field(description="Standalone indexed-search query")
    objective: str = Field(
        description="Specific evidence question this assignment must resolve"
    )
    evidence_requirements: list[str] = Field(
        description="Facts, provisions, exceptions, or dates evidence must establish"
    )
    excluded_queries: list[str] = Field(
        description="Already-run queries that this assignment must not duplicate"
    )
    as_of_date: str | None = Field(
        description="YYYY-MM-DD for historical retrieval, otherwise null"
    )
    filters: str | None = Field(
        description="Explicit reliable metadata filter, otherwise null"
    )


class SearchAssignmentBatch(_StrictArtifact):
    """Coordinator dispatch for one bounded worker wave."""

    task_id: str = Field(description="Task being coordinated")
    stop: bool = Field(
        description="True when no further worker assignment is warranted"
    )
    stop_reason: str | None = Field(
        description="Concise coverage or no-novelty reason when stop is true"
    )
    assignments: list[SearchAssignment] = Field(
        description="Independent non-duplicate assignments for this wave"
    )


class EvidenceClaim(_StrictArtifact):
    """One atomic claim supported by one exact indexed chunk."""

    claim: str = Field(description="Atomic factual or legal proposition")
    document_id: str = Field(description="Exact indexed document ID")
    chunk_id: str = Field(description="Exact indexed chunk ID")
    readable_title: str = Field(description="Human-readable document title")
    locator: str = Field(
        description="Human-readable article, section, paragraph, or heading locator"
    )
    evidence_excerpt: str = Field(
        description="Minimal verbatim excerpt that directly supports the claim"
    )
    supports_success_criteria: list[str] = Field(
        description="Task success criteria supported by this claim"
    )
    confidence: EvidenceConfidence = Field(
        description="Support strength based only on the supplied chunk"
    )
    effective_start_date: str | None = Field(
        description="Evidence validity start date when stated, otherwise null"
    )
    effective_end_date: str | None = Field(
        description="Evidence validity end date when stated, otherwise null"
    )


class WorkerArtifact(_StrictArtifact):
    """Compact output of one isolated search worker."""

    task_id: str = Field(description="Owning TaskSpec ID")
    assignment_id: str = Field(description="Completed SearchAssignment ID")
    worker_id: str = Field(description="Runtime worker ID supplied by the scheduler")
    status: WorkerStatus = Field(description="Worker outcome with strict semantics")
    searches_run: list[str] = Field(
        description="Exact search queries actually executed"
    )
    claims: list[EvidenceClaim] = Field(
        description="Atomic evidence claims extracted from supplied indexed hits"
    )
    gaps: list[str] = Field(
        description="Assignment requirements not supported by the supplied evidence"
    )
    cross_references: list[str] = Field(
        description="Explicit unresolved references named by supplied evidence"
    )
    error_code: str | None = Field(
        description="Stable execution error code only when status is failed"
    )
    error_message: str | None = Field(
        description="Sanitized execution failure summary, otherwise null"
    )


class TaskArtifact(_StrictArtifact):
    """Reviewed task result consumed by dependencies and final synthesis."""

    task_id: str = Field(description="Completed TaskSpec ID")
    status: TaskStatus = Field(description="Task coverage state")
    answer_fragment: str | None = Field(
        description="Evidence-grounded task conclusion, or null when none is supportable"
    )
    covered_success_criteria: list[str] = Field(
        description="Task criteria fully supported by verified claims"
    )
    uncovered_success_criteria: list[str] = Field(
        description="Task criteria not fully supported"
    )
    claims: list[EvidenceClaim] = Field(
        description="Verified claims retained for final synthesis"
    )
    conflicts: list[str] = Field(
        description="Material source conflicts that synthesis must disclose"
    )
    gaps: list[str] = Field(
        description="Material unanswered issues or missing evidence"
    )
    contributing_worker_ids: list[str] = Field(
        description="Workers whose verified claims contribute to this artifact"
    )


class PlanValidationError(ValueError):
    """One or more deterministic plan-integrity failures."""

    def __init__(self, errors: list[str] | tuple[str, ...]) -> None:
        self.errors = tuple(errors)
        super().__init__("Invalid global plan: " + "; ".join(self.errors))


@dataclass(frozen=True)
class PlanResolution:
    """Validated planner output or a deterministic direct-mode fallback."""

    plan: GlobalPlan
    used_fallback: bool
    validation_errors: tuple[str, ...]


def validate_global_plan(
    plan: GlobalPlan,
    *,
    max_tasks: int = 5,
) -> GlobalPlan:
    """Validate routing, task IDs, dependency integrity, and DAG acyclicity.

    The validated object is returned unchanged for convenient use at the
    orchestration boundary.  All discovered errors are reported together.
    """

    if max_tasks < 1:
        raise ValueError("max_tasks must be at least 1")

    errors: list[str] = []
    task_count = len(plan.tasks)

    if plan.mode == PlanMode.DIRECT and task_count != 1:
        errors.append("direct mode must contain exactly one task")
    if plan.mode == PlanMode.DECOMPOSED:
        if task_count < 2:
            errors.append("decomposed mode must contain at least two tasks")
        if task_count > max_tasks:
            errors.append(f"decomposed mode exceeds the maximum of {max_tasks} tasks")
    if task_count > max_tasks:
        errors.append(f"plan exceeds the maximum of {max_tasks} tasks")

    if not plan.normalized_question.strip():
        errors.append("normalized_question must not be blank")
    if not plan.answer_requirements:
        errors.append("answer_requirements must not be empty")

    task_ids = [task.task_id for task in plan.tasks]
    known_ids = set(task_ids)
    duplicate_ids = sorted(
        task_id for task_id in known_ids if task_ids.count(task_id) > 1
    )
    if duplicate_ids:
        errors.append(f"task IDs must be unique: {', '.join(duplicate_ids)}")

    for task in plan.tasks:
        label = task.task_id or "<blank>"
        if not task.task_id.strip():
            errors.append("task_id must not be blank")
        elif task.task_id != task.task_id.strip():
            errors.append(f"task_id must not have surrounding whitespace: {label!r}")
        elif _TASK_ID_PATTERN.fullmatch(task.task_id) is None:
            errors.append(
                f"task_id must be a 1-64 character safe identifier: {label!r}"
            )
        if not task.question.strip():
            errors.append(f"task {label!r} question must not be blank")
        if not task.purpose.strip():
            errors.append(f"task {label!r} purpose must not be blank")
        if not task.expected_output.strip():
            errors.append(f"task {label!r} expected_output must not be blank")
        if not task.success_criteria:
            errors.append(f"task {label!r} must define success criteria")
        elif any(not criterion.strip() for criterion in task.success_criteria):
            errors.append(f"task {label!r} contains a blank success criterion")

        duplicate_dependencies = sorted(
            dependency
            for dependency in set(task.depends_on)
            if task.depends_on.count(dependency) > 1
        )
        if duplicate_dependencies:
            errors.append(
                f"task {label!r} repeats dependencies: "
                + ", ".join(duplicate_dependencies)
            )
        for dependency in task.depends_on:
            if dependency not in known_ids:
                errors.append(f"task {label!r} depends on unknown task {dependency!r}")
            elif dependency == task.task_id:
                errors.append(f"task {label!r} cannot depend on itself")

    dependency_map = {
        task.task_id: [
            dependency
            for dependency in task.depends_on
            if dependency in known_ids and dependency != task.task_id
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


def make_direct_fallback_plan(original_question: str) -> GlobalPlan:
    """Create the safe one-task plan used when planning or validation fails."""

    question = original_question.strip()
    if not question:
        raise ValueError("original_question must not be blank")

    return GlobalPlan(
        version="1",
        mode=PlanMode.DIRECT,
        normalized_question=question,
        answer_requirements=["Answer every explicit part of the original question."],
        tasks=[
            TaskSpec(
                task_id="task_1",
                question=question,
                purpose="Resolve the original question without decomposition.",
                expected_output=(
                    "A complete answer grounded in verified indexed evidence."
                ),
                success_criteria=[
                    "Every explicit question requirement is addressed.",
                    "Every factual or legal claim is supported by verified evidence.",
                ],
                depends_on=[],
                required=True,
                as_of_date=None,
                filters=None,
            )
        ],
        synthesis_requirements=[],
        assumptions=[],
    )


def resolve_global_plan(
    candidate: GlobalPlan | None,
    *,
    original_question: str,
    max_tasks: int = 5,
    planner_error: str | None = None,
) -> PlanResolution:
    """Return a valid plan, falling back to one direct task on any plan failure."""

    if candidate is None:
        error = planner_error or "planner returned no structured plan"
        return PlanResolution(
            plan=make_direct_fallback_plan(original_question),
            used_fallback=True,
            validation_errors=(error,),
        )

    try:
        validated = validate_global_plan(candidate, max_tasks=max_tasks)
    except PlanValidationError as exc:
        return PlanResolution(
            plan=make_direct_fallback_plan(original_question),
            used_fallback=True,
            validation_errors=exc.errors,
        )

    return PlanResolution(
        plan=validated,
        used_fallback=False,
        validation_errors=(),
    )
