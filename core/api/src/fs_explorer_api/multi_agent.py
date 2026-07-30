"""Context-isolated, model-controlled multi-agent research orchestration.

The indexed workflow builds a task DAG, delegates each evidence need, and lets
persistent search agents decide their next tool action from the latest verified
tool result. Every LLM role receives a fresh one-turn history and returns a
typed artifact; chat transcripts and hidden reasoning are never passed between
roles.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import inspect
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Awaitable, Callable, Literal, Protocol, Sequence, cast

from .llm import ChatTurn, LLMClient, LLMUsage
from .orchestration_models import (
    ArtifactValidationError,
    DerivedConclusion,
    EvidenceClaim,
    EvidenceRequirement,
    ExecutionStrategy,
    GlobalPlan,
    PlanMode,
    PlanValidationError,
    SearchAssignment,
    SearchAssignmentBatch,
    TaskArtifact,
    TaskKind,
    TaskSpec,
    TaskStatus,
    WorkerArtifact,
    WorkerStatus,
    validate_global_plan,
    validate_task_artifact,
)
from .orchestration_prompts import (
    APPLICATION_TASK_SYSTEM_PROMPT,
    EVIDENCE_WORKER_SYSTEM_PROMPT,
    INTEGRATION_TASK_SYSTEM_PROMPT,
    TASK_REVIEW_SYSTEM_PROMPT,
    WORKER_SEARCH_CONTINUATION_PROMPT,
    build_global_planner_prompt,
    build_task_coordinator_prompt,
)
from .search import IndexedQueryEngine, RankedDocument, SearchHit

logger = logging.getLogger(__name__)

ProgressStatus = Literal["started", "completed", "failed"]
AgentRole = Literal["planner", "task", "worker", "final"]
_FINAL_SYNTHESIS_CALL_RESERVE = 1


class SearchResult(Protocol):
    """Structural result returned by the existing indexed search function."""

    query: str
    hits: list[SearchHit]
    error: str | None


SearchRunner = Callable[..., SearchResult | Awaitable[SearchResult]]
LLMUsageObserver = Callable[
    [AgentRole, str, LLMClient, LLMUsage, str | None, str | None, int],
    Awaitable[None],
]
RetrievalObserver = Callable[
    [int, int, str | None, str | None, int],
    None,
]


@dataclass(frozen=True, slots=True)
class ResearchLimits:
    """Legacy bounded-test policy plus production operational safeguards."""

    max_tasks: int = 5
    max_parallel_tasks: int = 3
    max_assignments_per_wave: int = 3
    max_worker_rounds: int = 4
    max_total_workers: int = 12
    max_parallel_llm_calls: int = 4
    max_parallel_retrievals: int = 8
    worker_hit_limit: int = 6
    worker_hit_chars: int = 6_000
    review_hit_chars: int = 2_500
    final_evidence_limit: int = 16
    final_chunk_chars: int = 8_000
    max_claims_per_worker: int = 8
    max_claims_per_task: int = 12
    max_artifact_list_items: int = 12
    max_artifact_text_chars: int = 1_500
    max_artifact_context_chars: int = 16_000
    max_question_chars: int = 8_000
    max_planner_context_chars: int = 16_000
    max_final_context_chars: int = 48_000
    max_query_chars: int = 1_000
    max_search_attempts: int = 2
    search_timeout_seconds: float = 20.0
    llm_timeout_seconds: float = 120.0
    max_total_llm_calls: int = 24

    @classmethod
    def from_env(cls) -> "ResearchLimits":
        """Load only concurrency, retry, and timeout safeguards for production."""

        defaults = cls()

        def positive(name: str, default: int) -> int:
            raw = os.getenv(name)
            if raw is None:
                return default
            try:
                value = int(raw)
            except ValueError as exc:
                raise ValueError(f"{name} must be an integer.") from exc
            if value < 1:
                raise ValueError(f"{name} must be at least 1.")
            return value

        def positive_float(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None:
                return default
            try:
                value = float(raw)
            except ValueError as exc:
                raise ValueError(f"{name} must be a number.") from exc
            if value <= 0:
                raise ValueError(f"{name} must be greater than 0.")
            return value

        return cls(
            max_parallel_tasks=positive(
                "FS_EXPLORER_MULTI_AGENT_MAX_PARALLEL_TASKS",
                defaults.max_parallel_tasks,
            ),
            max_parallel_llm_calls=positive(
                "FS_EXPLORER_MULTI_AGENT_LLM_CONCURRENCY",
                defaults.max_parallel_llm_calls,
            ),
            max_parallel_retrievals=positive(
                "FS_EXPLORER_MULTI_AGENT_RETRIEVAL_CONCURRENCY",
                defaults.max_parallel_retrievals,
            ),
            search_timeout_seconds=positive_float(
                "FS_EXPLORER_MULTI_AGENT_SEARCH_TIMEOUT_SECONDS",
                defaults.search_timeout_seconds,
            ),
            llm_timeout_seconds=positive_float(
                "FS_EXPLORER_MULTI_AGENT_LLM_TIMEOUT_SECONDS",
                defaults.llm_timeout_seconds,
            ),
        )


@dataclass(frozen=True, slots=True)
class ResearchProgress:
    """One correlated lifecycle update emitted by the orchestrator."""

    event_id: str
    kind: str
    sequence: int
    agent_role: AgentRole
    status: ProgressStatus
    label: str
    detail: str
    task_id: str | None = None
    agent_id: str | None = None


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """Server-owned source record; models may reference but never mutate it."""

    evidence_id: str
    document_id: str
    chunk_id: str
    readable_title: str
    locator: str
    text: str
    score: float
    metadata: dict[str, object]

    def source_payload(self) -> dict[str, object]:
        return {
            "title": self.readable_title,
            "snippet": self.text[:1200],
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "score": self.score,
            "locator": self.locator,
        }


@dataclass(frozen=True, slots=True)
class _RenderedEvidence:
    """The exact records and bounded text exposed across an LLM boundary."""

    text: str
    records: tuple[EvidenceRecord, ...]


@dataclass(frozen=True, slots=True)
class MultiAgentResearchResult:
    """Final artifact bundle consumed by the existing answer streamer."""

    plan: GlobalPlan
    task_artifacts: tuple[TaskArtifact, ...]
    final_context: str
    evidence_sources: tuple[dict[str, object], ...]
    incomplete: bool
    unresolved_information: tuple[str, ...] = ()


def _normalized_query(value: str) -> str:
    return " ".join(value.casefold().split())


def _query_token_overlap(left: str, right: str) -> float:
    """Return a cheap duplicate-search score without another model call."""

    left_tokens = set(re.findall(r"\w+", left.casefold()))
    right_tokens = set(re.findall(r"\w+", right.casefold()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(
        len(left_tokens),
        len(right_tokens),
    )


def _is_near_duplicate_query(
    query: str,
    previous_queries: Sequence[str],
    *,
    threshold: float = 0.8,
) -> bool:
    normalized = _normalized_query(query)
    return any(
        normalized == _normalized_query(previous)
        or _query_token_overlap(normalized, previous) >= threshold
        for previous in previous_queries
    )


def _safe_identifier(value: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.:-]+", "_", value.strip())
    normalized = normalized.strip("._:-")[:80]
    return normalized or fallback


def _claim_identifier(task_id: str, assignment_id: str, index: int) -> str:
    """Build a stable contract-safe ID without losing uniqueness to truncation."""

    raw = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        f"{task_id}_{assignment_id}_claim_{index}",
    ).strip("_")
    if len(raw) <= 64:
        return raw
    digest = hashlib.sha256(
        f"{task_id}\0{assignment_id}\0{index}".encode()
    ).hexdigest()[:10]
    suffix = f"_{digest}"
    prefix = raw[: 64 - len(suffix)].rstrip("_-")
    return f"{prefix}{suffix}"


def _bounded_error(value: object, limit: int = 300) -> str:
    return " ".join(str(value).split())[:limit]


def _bounded_text(value: object, limit: int) -> str:
    return " ".join(str(value).split())[:limit]


def _validated_as_of_date(value: str | None) -> str | None:
    if value is None:
        return None
    candidate = value.strip()
    try:
        parsed = date.fromisoformat(candidate)
    except ValueError:
        return None
    return candidate if parsed.isoformat() == candidate else None


def _is_transient_search_error(value: str | None) -> bool:
    if not value:
        return False
    normalized = value.casefold()
    return any(
        marker in normalized
        for marker in (
            "timeout",
            "timed out",
            "temporar",
            "connection",
            "deadlock",
            "rate limit",
            "unavailable",
        )
    )


def _current_question(task: str) -> str:
    marker = "\n\nCurrent question:\n"
    return task.rsplit(marker, 1)[-1].strip() or task.strip()


def _bounded_planner_input(
    task: str,
    *,
    max_question_chars: int,
    max_context_chars: int,
) -> str:
    """Preserve the current question first, then fit only recent context."""

    marker = "\n\nCurrent question:\n"
    current = _bounded_text(
        _current_question(task),
        min(max_question_chars, max_context_chars),
    )
    prior = task.rsplit(marker, 1)[0] if marker in task else ""
    question_block = f"CURRENT QUESTION:\n{current}"
    if not prior:
        return question_block[:max_context_chars]
    context_label = "\n\nRECENT CONVERSATION CONTEXT:\n"
    remaining = max(max_context_chars - len(question_block) - len(context_label), 0)
    if remaining == 0:
        return question_block[:max_context_chars]
    recent_context = _bounded_text(prior[-remaining:], remaining)
    return f"{question_block}{context_label}{recent_context}"[:max_context_chars]


def _consume_background_task_exception(task: asyncio.Task[object]) -> None:
    """Retrieve background exceptions while preserving them for later awaits."""

    if task.cancelled():
        return
    try:
        task.exception()
    except BaseException:
        # Calling exception() is solely to avoid an un-retrieved-task warning.
        # The cached task still raises the same exception to every awaiter.
        pass


def _readable_title(hit: SearchHit) -> str:
    raw_path = hit.relative_path or hit.absolute_path
    title = Path(raw_path).name
    title = re.sub(r"^\d+-", "", title)
    title = re.sub(r"\.[a-zA-Z0-9]+$", "", title)
    title = title.replace("_x1", "(").replace("x2_", ")_").replace("x2", ")")
    title = re.sub(r"\s+", " ", title.replace("_", " ")).strip()
    return title or "Indexed document"


def _readable_locator(metadata: dict[str, object], position: int | None) -> str:
    parts: list[str] = []
    article = metadata.get("article_no")
    if article not in (None, ""):
        parts.append(f"Article {article}")
    paragraph = metadata.get("paragraph_no")
    if paragraph not in (None, ""):
        parts.append(f"paragraph {paragraph}")
    clause = metadata.get("clause_label") or metadata.get("subclause_label")
    if clause not in (None, ""):
        parts.append(f"clause {clause}")
    heading_path = metadata.get("heading_path")
    if isinstance(heading_path, list) and heading_path:
        parts.append(" > ".join(str(item) for item in heading_path[-3:]))
    if parts:
        return ", ".join(parts)
    return f"Section {position}" if position is not None else "Relevant section"


def _normalize_evidence_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def _claim_supported(claim: EvidenceClaim, record: EvidenceRecord) -> bool:
    excerpt = _normalize_evidence_text(claim.evidence_excerpt)
    return bool(excerpt) and excerpt in _normalize_evidence_text(record.text)


def _effective_date(metadata: dict[str, object], *keys: str) -> str | None:
    for key in keys:
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return None


class MultiAgentResearchOrchestrator:
    """Planner → task strategists → persistent search agents → synthesis."""

    def __init__(
        self,
        *,
        planner_llm: LLMClient,
        task_llm: LLMClient,
        worker_llm: LLMClient,
        search_runner: SearchRunner,
        limits: ResearchLimits | None = None,
        on_llm_usage: LLMUsageObserver | None = None,
        on_retrieval: RetrievalObserver | None = None,
        on_progress: Callable[[ResearchProgress], None] | None = None,
        search_runner_in_thread: bool = True,
    ) -> None:
        self._planner_llm = planner_llm
        self._task_llm = task_llm
        self._worker_llm = worker_llm
        self._search_runner = search_runner
        # Explicit limits are retained only for deterministic unit tests and
        # callers that deliberately opt into the legacy bounded policy.
        # Production passes no limits: research stops on model decisions and
        # evidence coverage, never on worker/round/token counters.
        self._unlimited_research = limits is None
        self._limits = limits or ResearchLimits.from_env()
        self._on_llm_usage = on_llm_usage
        self._on_retrieval = on_retrieval
        self._on_progress = on_progress
        self._search_runner_in_thread = search_runner_in_thread
        self._llm_semaphore = asyncio.Semaphore(self._limits.max_parallel_llm_calls)
        self._retrieval_semaphore = asyncio.Semaphore(
            self._limits.max_parallel_retrievals
        )
        self._task_semaphore = asyncio.Semaphore(self._limits.max_parallel_tasks)
        self._llm_budget_lock = asyncio.Lock()
        self._run_lock = asyncio.Lock()
        self._sequence = 0
        self._llm_call_count = 0
        self._structured_llm_call_limit = (
            None
            if self._unlimited_research
            else max(
                self._limits.max_total_llm_calls - _FINAL_SYNTHESIS_CALL_RESERVE,
                0,
            )
        )
        self._llm_tasks: dict[str, asyncio.Task[tuple[object, LLMUsage]]] = {}
        self._search_tasks: dict[tuple[str, str, str], asyncio.Task[SearchResult]] = {}
        self._task_rerank_tasks: dict[
            str, asyncio.Task[list[tuple[RankedDocument, float]]]
        ] = {}
        self._worker_count = 0
        self._worker_count_by_task: dict[str, int] = {}
        self._worker_closed_tasks: set[str] = set()
        self._next_round_by_task: dict[str, int] = {}
        self._plan: GlobalPlan | None = None
        self._task_artifacts: dict[str, TaskArtifact] = {}
        self._worker_artifacts: dict[str, dict[str, WorkerArtifact]] = {}
        self._evidence: dict[str, EvidenceRecord] = {}
        self._evidence_by_source: dict[tuple[str, str], EvidenceRecord] = {}
        self._task_evidence: dict[str, dict[str, EvidenceRecord]] = {}
        self._searches_by_task: dict[str, set[str]] = {}
        self._search_attempts_by_task: dict[str, dict[str, set[str]]] = {}
        self._gap_progress_rounds: set[tuple[str, int]] = set()
        self._material_unknowns_announced = False

    @property
    def has_progress(self) -> bool:
        return self._plan is not None or bool(self._worker_artifacts)

    @property
    def completed(self) -> bool:
        return bool(self._plan) and len(self._task_artifacts) == len(self._plan.tasks)

    def set_progress_observer(
        self, observer: Callable[[ResearchProgress], None] | None
    ) -> None:
        """Replace the connection-local progress sink when resuming a run."""

        self._on_progress = observer

    def _emit(
        self,
        *,
        event_id: str,
        kind: str,
        role: AgentRole,
        status: ProgressStatus,
        label: str,
        detail: str,
        task_id: str | None = None,
        agent_id: str | None = None,
    ) -> int:
        self._sequence += 1
        if self._on_progress is not None:
            self._on_progress(
                ResearchProgress(
                    event_id=event_id,
                    kind=kind,
                    sequence=self._sequence,
                    agent_role=role,
                    status=status,
                    label=label,
                    detail=detail,
                    task_id=task_id,
                    agent_id=agent_id,
                )
            )
        return self._sequence

    async def _generate_structured(
        self,
        *,
        client: LLMClient,
        role: AgentRole,
        purpose: str,
        prompt: str,
        user_text: str,
        operation_id: str,
        schema: type[GlobalPlan]
        | type[SearchAssignmentBatch]
        | type[WorkerArtifact]
        | type[TaskArtifact],
        task_id: str | None = None,
        agent_id: str | None = None,
    ) -> tuple[object, LLMUsage]:
        """Run one stable structured operation at most once across reconnects."""

        async with self._llm_budget_lock:
            operation = self._llm_tasks.get(operation_id)
            if operation is None:
                if (
                    self._structured_llm_call_limit is not None
                    and self._llm_call_count >= self._structured_llm_call_limit
                ):
                    raise RuntimeError(
                        "Multi-agent LLM call budget exhausted after reserving "
                        "the final synthesis call."
                    )
                self._llm_call_count += 1

                async def invoke() -> tuple[object, LLMUsage]:
                    async with self._llm_semaphore:
                        result, usage = await client.generate_structured(
                            [ChatTurn(role="user", text=user_text)],
                            prompt,
                            schema,
                        )
                    self._sequence += 1
                    sequence = self._sequence
                    if self._on_llm_usage is not None:
                        try:
                            await self._on_llm_usage(
                                role,
                                purpose,
                                client,
                                usage,
                                task_id,
                                agent_id,
                                sequence,
                            )
                        except Exception:
                            logger.exception(
                                "Multi-agent LLM telemetry callback failed"
                            )
                    return result, usage

                operation = asyncio.create_task(
                    invoke(),
                    name=f"multi-agent-llm:{operation_id}",
                )
                operation.add_done_callback(_consume_background_task_exception)
                self._llm_tasks[operation_id] = operation

        # The provider call is shielded so a WebSocket disconnect cannot
        # discard a paid response. A resumed run awaits this same operation.
        try:
            return await asyncio.wait_for(
                asyncio.shield(operation),
                timeout=self._limits.llm_timeout_seconds,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                f"Multi-agent {purpose} exceeded the LLM stage timeout."
            ) from exc

    async def run(self, task: str) -> MultiAgentResearchResult:
        """Run or resume a model-controlled research plan from persisted artifacts."""

        async with self._run_lock:
            if self._plan is None:
                self._plan = await self._create_plan(task)
            self._announce_material_unknowns(self._plan)

            required_evidence_tasks = sum(
                spec.required and spec.kind == TaskKind.EVIDENCE
                for spec in self._plan.tasks
            )
            if (
                not self._unlimited_research
                and required_evidence_tasks > self._limits.max_total_workers
            ):
                raise RuntimeError(
                    "The worker budget cannot reserve one worker for every "
                    "required evidence task."
                )
            await self._run_task_dag(self._plan)
            ordered_artifacts = tuple(
                self._task_artifacts[spec.task_id] for spec in self._plan.tasks
            )
            selected_evidence = self._select_final_evidence(ordered_artifacts)
            final_context, rendered_evidence = self._render_final_context(
                original_question=_current_question(task),
                plan=self._plan,
                artifacts=ordered_artifacts,
                evidence=selected_evidence,
            )
            incomplete = any(
                spec.required
                and self._task_artifacts[spec.task_id].status != TaskStatus.COMPLETE
                for spec in self._plan.tasks
            ) or len(rendered_evidence) < len(selected_evidence)
            incomplete = incomplete or not self._all_finding_support_rendered(
                ordered_artifacts,
                rendered_evidence,
            )
            unresolved_information = self._unresolved_information(
                self._plan,
                ordered_artifacts,
            )
            incomplete = incomplete or any(
                artifact.uncovered_requirement_ids
                for spec, artifact in zip(self._plan.tasks, ordered_artifacts)
                if spec.required
            )
            sources = tuple(record.source_payload() for record in rendered_evidence)
            return MultiAgentResearchResult(
                plan=self._plan,
                task_artifacts=ordered_artifacts,
                final_context=final_context,
                evidence_sources=sources,
                incomplete=incomplete,
                unresolved_information=unresolved_information,
            )

    async def _create_plan(self, task: str) -> GlobalPlan:
        event_id = "global-plan"
        self._emit(
            event_id=event_id,
            kind="plan_created",
            role="planner",
            status="started",
            label="Planning research",
            detail="Deciding whether the question needs one or several tasks",
            agent_id="global-planner",
        )
        max_tasks = None if self._unlimited_research else self._limits.max_tasks
        max_list_items = (
            None if self._unlimited_research else self._limits.max_artifact_list_items
        )
        planner_input = (
            task
            if self._unlimited_research
            else _bounded_planner_input(
                task,
                max_question_chars=self._limits.max_question_chars,
                max_context_chars=self._limits.max_planner_context_chars,
            )
        )
        attempt = 1
        correction_feedback: str | None = None
        while True:
            user_text = (
                f"CURRENT DATE: {date.today().isoformat()}\n\n"
                f"ORIGINAL REQUEST AND CONTEXT:\n{planner_input}"
            )
            if correction_feedback is not None:
                user_text += (
                    "\n\nYOUR PREVIOUS PLAN FAILED SERVER VALIDATION:\n"
                    f"{correction_feedback}\n\n"
                    "Return a corrected complete plan. Preserve every requested "
                    "answer heading; do not replace it with a simplified graph."
                )
            raw, _usage = await self._generate_structured(
                client=self._planner_llm,
                role="planner",
                purpose="global_plan",
                prompt=build_global_planner_prompt(
                    max_tasks=max_tasks,
                    max_list_items=max_list_items,
                ),
                user_text=user_text,
                operation_id=f"global-plan:{attempt}",
                schema=GlobalPlan,
                agent_id="global-planner",
            )
            assert isinstance(raw, GlobalPlan)
            try:
                plan = (
                    validate_global_plan(
                        raw,
                        max_tasks=None,
                        max_list_items=None,
                    )
                    if self._unlimited_research
                    else self._bounded_plan(raw)
                )
            except PlanValidationError as exc:
                correction_feedback = json.dumps(
                    {
                        "validation_errors": list(exc.errors),
                        "previous_plan": raw.model_dump(mode="json"),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                self._emit(
                    event_id=f"{event_id}-revision-{attempt}",
                    kind="plan_revising",
                    role="planner",
                    status="started",
                    label="Research plan is being corrected",
                    detail=(
                        "The planner is correcting its own invalid dependency "
                        "or coverage decision."
                    ),
                    agent_id="global-planner",
                )
                attempt += 1
                continue
            break

        detail = (
            f"{plan.problem_type.value}/{plan.mode.value}/"
            f"{plan.execution_strategy.value}: {len(plan.tasks)} task(s)"
        )
        self._emit(
            event_id=event_id,
            kind="plan_created",
            role="planner",
            status="completed",
            label="Research plan ready",
            detail=detail,
            agent_id="global-planner",
        )
        return plan

    def _bounded_plan(self, plan: GlobalPlan) -> GlobalPlan:
        """Bound free text without mutating the validated graph structure."""

        text_limit = self._limits.max_artifact_text_chars
        tasks = []
        for task in plan.tasks:
            tasks.append(
                task.model_copy(
                    update={
                        "issue": _bounded_text(task.issue, text_limit),
                        "search_question": (
                            _bounded_text(task.search_question, text_limit)
                            if task.search_question is not None
                            else None
                        ),
                        "evidence_requirements": [
                            requirement.model_copy(
                                update={
                                    "description": _bounded_text(
                                        requirement.description,
                                        text_limit,
                                    )
                                }
                            )
                            for requirement in task.evidence_requirements
                        ],
                        "produces": [
                            output.model_copy(
                                update={
                                    "description": _bounded_text(
                                        output.description,
                                        text_limit,
                                    )
                                }
                            )
                            for output in task.produces
                        ],
                        "as_of_date": _validated_as_of_date(task.as_of_date),
                        "filters": self._bounded_filter(task.filters),
                    }
                )
            )

        scenario = plan.scenario
        if scenario is not None:
            scenario = scenario.model_copy(
                update={
                    "jurisdiction": (
                        _bounded_text(scenario.jurisdiction, text_limit)
                        if scenario.jurisdiction is not None
                        else None
                    ),
                    "law_as_of_date": _validated_as_of_date(scenario.law_as_of_date),
                    "facts": [
                        fact.model_copy(
                            update={
                                "description": _bounded_text(
                                    fact.description,
                                    text_limit,
                                )
                            }
                        )
                        for fact in scenario.facts
                    ],
                    "material_unknowns": [
                        unknown.model_copy(
                            update={
                                "description": _bounded_text(
                                    unknown.description,
                                    text_limit,
                                ),
                                "why_material": _bounded_text(
                                    unknown.why_material,
                                    text_limit,
                                ),
                            }
                        )
                        for unknown in scenario.material_unknowns
                    ],
                    "decision_branches": [
                        branch.model_copy(
                            update={
                                "condition": _bounded_text(
                                    branch.condition,
                                    text_limit,
                                ),
                                "consequence": _bounded_text(
                                    branch.consequence,
                                    text_limit,
                                ),
                            }
                        )
                        for branch in scenario.decision_branches
                    ],
                }
            )

        bounded = plan.model_copy(
            update={
                "normalized_question": _bounded_text(
                    plan.normalized_question,
                    text_limit,
                ),
                "answer_requirements": [
                    requirement.model_copy(
                        update={
                            "description": _bounded_text(
                                requirement.description,
                                text_limit,
                            )
                        }
                    )
                    for requirement in plan.answer_requirements
                ],
                "scenario": scenario,
                "tasks": tasks,
                "synthesis_requirements": [
                    _bounded_text(requirement, text_limit)
                    for requirement in plan.synthesis_requirements
                ],
                "assumptions": [
                    _bounded_text(assumption, text_limit)
                    for assumption in plan.assumptions
                ],
            }
        )
        return validate_global_plan(
            bounded,
            max_tasks=self._limits.max_tasks,
            max_list_items=self._limits.max_artifact_list_items,
        )

    async def _run_task_dag(self, plan: GlobalPlan) -> None:
        specs = {task.task_id: task for task in plan.tasks}
        pending = set(specs) - set(self._task_artifacts)
        while pending:
            ready = [
                spec
                for spec in plan.tasks
                if spec.task_id in pending
                if all(
                    dependency in self._task_artifacts for dependency in spec.depends_on
                )
            ]
            if not ready:
                # Validation should make this unreachable; keep a deterministic
                # failure artifact instead of allowing an infinite scheduler loop.
                for task_id in sorted(pending):
                    self._task_artifacts[task_id] = self._failed_task_artifact(
                        specs[task_id],
                        "Dependency scheduling could not make progress.",
                    )
                return

            results = await asyncio.gather(
                *(self._run_task_bounded(spec) for spec in ready),
                return_exceptions=True,
            )
            for spec, result in zip(ready, results):
                if isinstance(result, BaseException):
                    logger.error(
                        "Task %s failed outside its isolated branch",
                        spec.task_id,
                        exc_info=(
                            type(result),
                            result,
                            result.__traceback__,
                        ),
                    )
                    artifact = self._failed_task_artifact(spec, _bounded_error(result))
                    self._emit(
                        event_id=f"task-{spec.task_id}",
                        kind="task_failed",
                        role="task",
                        status="failed",
                        label="Research task failed",
                        detail="The isolated task branch could not complete.",
                        task_id=spec.task_id,
                        agent_id=f"task-coordinator-{spec.task_id}",
                    )
                else:
                    artifact = result
                self._task_artifacts[spec.task_id] = artifact
                pending.discard(spec.task_id)

    async def _run_task_bounded(self, spec: TaskSpec) -> TaskArtifact:
        async with self._task_semaphore:
            return await self._run_task(spec)

    def _available_worker_slots(self, task_id: str) -> int:
        """Keep one global worker slot reserved for every untouched required task."""

        if self._unlimited_research:
            return 2**31 - 1
        global_remaining = self._limits.max_total_workers - self._worker_count
        if global_remaining <= 0 or self._plan is None:
            return max(global_remaining, 0)
        reserved_for_others = sum(
            1
            for candidate in self._plan.tasks
            if candidate.task_id != task_id
            and candidate.required
            and candidate.kind == TaskKind.EVIDENCE
            and candidate.task_id not in self._worker_closed_tasks
            and candidate.task_id not in self._task_artifacts
            and self._worker_count_by_task.get(candidate.task_id, 0) == 0
        )
        return max(global_remaining - reserved_for_others, 0)

    async def _run_task(self, spec: TaskSpec) -> TaskArtifact:
        if spec.kind == TaskKind.EVIDENCE:
            return await self._run_evidence_task(spec)
        return await self._run_application_task(spec)

    async def _run_evidence_task(self, spec: TaskSpec) -> TaskArtifact:
        """Run the model-controlled strategist/search/reviewer evidence path."""

        event_id = f"task-{spec.task_id}"
        self._emit(
            event_id=event_id,
            kind="task_started",
            role="task",
            status="started",
            label="Researching a task",
            detail=spec.question,
            task_id=spec.task_id,
            agent_id=f"task-coordinator-{spec.task_id}",
        )
        dependency_artifacts = [
            self._task_artifacts[task_id]
            for task_id in spec.depends_on
            if task_id in self._task_artifacts
        ]
        worker_by_assignment = self._worker_artifacts.setdefault(spec.task_id, {})
        searches_run = self._searches_by_task.setdefault(spec.task_id, set())
        attempts_by_requirement = self._search_attempts_by_task.setdefault(
            spec.task_id,
            {},
        )
        single_pass = (
            self._plan is not None
            and self._plan.mode == PlanMode.DIRECT
            and self._plan.execution_strategy == ExecutionStrategy.SINGLE_PASS
            and len(self._plan.tasks) == 1
            and not dependency_artifacts
        )
        single_pass_complete = False

        round_index = self._next_round_by_task.get(spec.task_id, 1)
        while self._unlimited_research or round_index <= self._limits.max_worker_rounds:
            if (
                not self._unlimited_research
                and self._worker_count >= self._limits.max_total_workers
            ):
                break
            uncovered_evidence = self._uncovered_evidence_requirements(
                spec,
                list(worker_by_assignment.values()),
            )
            if round_index > 1 and uncovered_evidence:
                self._emit_gap_recovery_progress(
                    spec,
                    uncovered_evidence,
                    round_index=round_index,
                )
            assignments: list[SearchAssignment] | None = None
            if single_pass and round_index == 1 and not worker_by_assignment:
                # The global planner has already established that this is one
                # precise lookup. Reuse its standalone TaskSpec as the search
                # assignment instead of paying for a redundant coordinator
                # call. Any coverage miss automatically falls through to the
                # normal adaptive second wave below.
                dispatch = self._single_pass_dispatch(
                    spec,
                    searches_run=searches_run,
                )
            else:
                coordinator_decision = 1
                rejection: str | None = None
                while True:
                    try:
                        dispatch = await self._coordinate_task(
                            spec=spec,
                            dependency_artifacts=dependency_artifacts,
                            worker_artifacts=list(worker_by_assignment.values()),
                            searches_run=sorted(searches_run),
                            round_index=round_index,
                            uncovered_evidence=uncovered_evidence,
                            attempts_by_requirement=attempts_by_requirement,
                            decision_index=coordinator_decision,
                            rejection=rejection,
                        )
                    except Exception:
                        raise
                    assignments = self._validated_assignments(
                        spec,
                        dispatch,
                        searches_run=searches_run,
                        existing_assignment_ids=set(worker_by_assignment),
                        allowed_evidence_requirement_ids={
                            requirement.evidence_requirement_id
                            for requirement in uncovered_evidence
                        },
                    )
                    if not self._unlimited_research:
                        break
                    if assignments or (dispatch.stop and worker_by_assignment):
                        break
                    rejection = (
                        "Your decision did not produce an executable action. "
                        "If no search agent has reported yet, delegate at least "
                        "one search. Otherwise either request a materially "
                        "different search or explicitly stop based on the "
                        "agents' latest reports."
                    )
                    coordinator_decision += 1

            if assignments is None:
                assignments = self._validated_assignments(
                    spec,
                    dispatch,
                    searches_run=searches_run,
                    existing_assignment_ids=set(worker_by_assignment),
                    allowed_evidence_requirement_ids={
                        requirement.evidence_requirement_id
                        for requirement in uncovered_evidence
                    },
                )
            remaining_slots = self._available_worker_slots(spec.task_id)
            if not self._unlimited_research:
                assignments = assignments[:remaining_slots]
            if not assignments:
                self._worker_closed_tasks.add(spec.task_id)
                self._next_round_by_task[spec.task_id] = (
                    self._limits.max_worker_rounds + 1
                )
                break

            self._worker_count += len(assignments)
            self._worker_count_by_task[spec.task_id] = self._worker_count_by_task.get(
                spec.task_id, 0
            ) + len(assignments)
            # Reserve query identity before starting I/O. If the parent run is
            # cancelled, unfinished reservations are rolled back below while
            # the underlying search/LLM operation remains cached and shielded.
            for assignment in assignments:
                searches_run.add(_normalized_query(assignment.query))
            try:
                results = await asyncio.gather(
                    *(
                        self._run_worker(
                            spec=spec,
                            assignment=assignment,
                            worker_id=(
                                f"worker-{spec.task_id}-{assignment.assignment_id}"
                            ),
                        )
                        for assignment in assignments
                    ),
                    return_exceptions=True,
                )
            except asyncio.CancelledError:
                unfinished = [
                    assignment
                    for assignment in assignments
                    if assignment.assignment_id not in worker_by_assignment
                ]
                for assignment in unfinished:
                    searches_run.discard(_normalized_query(assignment.query))
                self._worker_count -= len(unfinished)
                self._worker_count_by_task[spec.task_id] = max(
                    self._worker_count_by_task.get(spec.task_id, 0) - len(unfinished),
                    0,
                )
                raise
            for assignment, result in zip(assignments, results):
                if isinstance(result, BaseException):
                    worker_id = f"worker-{spec.task_id}-{assignment.assignment_id}"
                    artifact = self._failed_worker_artifact(
                        spec=spec,
                        assignment=assignment,
                        worker_id=worker_id,
                        error=_bounded_error(result),
                    )
                    self._emit(
                        event_id=f"worker-{spec.task_id}-{assignment.assignment_id}",
                        kind="worker_completed",
                        role="worker",
                        status="failed",
                        label="Evidence worker failed",
                        detail="The isolated worker branch could not complete.",
                        task_id=spec.task_id,
                        agent_id=worker_id,
                    )
                else:
                    artifact = result
                worker_by_assignment[assignment.assignment_id] = artifact
                if artifact.status == WorkerStatus.FAILED:
                    for query in artifact.searches_run:
                        searches_run.discard(_normalized_query(query))
                else:
                    searches_run.update(
                        _normalized_query(query)
                        for query in artifact.searches_run
                        if query.strip()
                    )
                    self._record_search_angles(
                        attempts_by_requirement,
                        assignment,
                        searches=artifact.searches_run,
                    )
            self._next_round_by_task[spec.task_id] = round_index + 1
            supported_criteria = {
                evidence_requirement_id
                for worker in worker_by_assignment.values()
                for claim in worker.claims
                for evidence_requirement_id in claim.evidence_requirement_ids
            }
            coverage_complete = all(
                requirement.evidence_requirement_id in supported_criteria
                for requirement in spec.evidence_requirements
            )
            if coverage_complete:
                single_pass_complete = single_pass and round_index == 1
                self._worker_closed_tasks.add(spec.task_id)
                self._next_round_by_task[spec.task_id] = (
                    self._limits.max_worker_rounds + 1
                )
                break
            round_index += 1

        self._worker_closed_tasks.add(spec.task_id)
        if single_pass_complete:
            # IDs, excerpts, criteria and source membership were already
            # verified server-side in `_run_worker`. A second model reviewer
            # cannot add evidence for this deliberately simple route.
            artifact = self._assemble_verified_task_artifact(
                spec,
                worker_artifacts=list(worker_by_assignment.values()),
            )
        else:
            artifact = await self._review_task(
                spec=spec,
                dependency_artifacts=dependency_artifacts,
                worker_artifacts=list(worker_by_assignment.values()),
            )
        status: ProgressStatus = (
            "completed" if artifact.status != TaskStatus.FAILED else "failed"
        )
        self._emit(
            event_id=event_id,
            kind=("task_completed" if status == "completed" else "task_failed"),
            role="task",
            status=status,
            label=(
                "Research task complete"
                if status == "completed"
                else "Research task failed"
            ),
            detail=(
                f"{len(artifact.claims)} verified claim(s); "
                f"{len(artifact.uncovered_requirement_ids)} gap(s)"
            ),
            task_id=spec.task_id,
            agent_id=f"task-coordinator-{spec.task_id}",
        )
        if artifact.uncovered_requirement_ids:
            descriptions_by_id = (
                {
                    requirement.requirement_id: requirement.description
                    for requirement in self._plan.answer_requirements
                }
                if self._plan is not None
                else {}
            )
            descriptions = [
                _bounded_text(
                    descriptions_by_id.get(requirement_id, requirement_id),
                    220,
                )
                for requirement_id in artifact.uncovered_requirement_ids[:3]
            ]
            distinct_searches = len(
                {
                    query
                    for queries in attempts_by_requirement.values()
                    for query in queries
                }
            )
            self._emit(
                event_id=f"gap-unresolved-{spec.task_id}",
                kind="gap_recovery_exhausted",
                role="task",
                status="completed",
                label="Kanıt açığı kaldı",
                detail=(
                    f"{distinct_searches} farklı hedefli aramaya rağmen şu "
                    "bölüm tamamlanamadı: "
                    f"{'; '.join(descriptions)}. "
                    "Yanıt mevcut kanıtla sınırlandırılacak ve eksik açıkça belirtilecek."
                ),
                task_id=spec.task_id,
                agent_id=f"task-coordinator-{spec.task_id}",
            )
        return artifact

    def _announce_material_unknowns(self, plan: GlobalPlan) -> None:
        """Surface user-owned missing facts without guessing or extra searches."""

        scenario = plan.scenario
        if (
            self._material_unknowns_announced
            or scenario is None
            or not scenario.material_unknowns
        ):
            return
        self._material_unknowns_announced = True
        unknowns = [
            _bounded_text(unknown.description, 220)
            for unknown in scenario.material_unknowns[:3]
        ]
        suffix = (
            f" (+{len(scenario.material_unknowns) - len(unknowns)} ek bilgi)"
            if len(scenario.material_unknowns) > len(unknowns)
            else ""
        )
        self._emit(
            event_id="material-unknowns",
            kind="information_needed",
            role="planner",
            status="completed",
            label="Eksik kullanıcı bilgisi",
            detail=(
                f"Sonucu etkileyebilecek bilgi eksik: {'; '.join(unknowns)}{suffix}. "
                "Bu bilgi tahmin edilmeyecek; mevcut kanıtla koşullu sonuç verilecek."
            ),
            agent_id="global-planner",
        )

    @staticmethod
    def _supported_evidence_requirement_ids(
        worker_artifacts: Sequence[WorkerArtifact],
    ) -> set[str]:
        return {
            evidence_requirement_id
            for worker in worker_artifacts
            for claim in worker.claims
            for evidence_requirement_id in claim.evidence_requirement_ids
        }

    def _uncovered_evidence_requirements(
        self,
        spec: TaskSpec,
        worker_artifacts: Sequence[WorkerArtifact],
    ) -> list[EvidenceRequirement]:
        supported = self._supported_evidence_requirement_ids(worker_artifacts)
        return [
            requirement
            for requirement in spec.evidence_requirements
            if requirement.evidence_requirement_id not in supported
        ]

    def _emit_gap_recovery_progress(
        self,
        spec: TaskSpec,
        requirements: Sequence[EvidenceRequirement],
        *,
        round_index: int,
    ) -> None:
        """Tell the client exactly what the bounded follow-up will target."""

        progress_key = (spec.task_id, round_index)
        if progress_key in self._gap_progress_rounds:
            return
        self._gap_progress_rounds.add(progress_key)
        descriptions = [
            _bounded_text(requirement.description, 220)
            for requirement in requirements[:3]
        ]
        suffix = (
            f" (+{len(requirements) - len(descriptions)} ek madde)"
            if len(requirements) > len(descriptions)
            else ""
        )
        self._emit(
            event_id=f"gap-recovery-{spec.task_id}-{round_index}",
            kind="gap_recovery_started",
            role="task",
            status="started",
            label="Eksik kanıt tamamlanıyor",
            detail=(
                "Yanıt için eksik kalan kanıt: "
                f"{'; '.join(descriptions)}{suffix}. "
                "Yalnızca bu eksikleri tamamlamak için hedefli arama yapıyorum."
            ),
            task_id=spec.task_id,
            agent_id=f"task-coordinator-{spec.task_id}",
        )

    def _record_search_angles(
        self,
        attempts_by_requirement: dict[str, set[str]],
        assignment: SearchAssignment,
        *,
        searches: Sequence[str] | None = None,
    ) -> None:
        for requirement_id in assignment.evidence_requirements:
            attempts_by_requirement.setdefault(requirement_id, set()).update(
                _normalized_query(query) for query in (searches or [assignment.query])
            )

    def _unresolved_information(
        self,
        plan: GlobalPlan,
        artifacts: Sequence[TaskArtifact],
    ) -> tuple[str, ...]:
        """Build bounded, user-readable gaps while keeping internal IDs private."""

        answer_descriptions = {
            requirement.requirement_id: requirement.description
            for requirement in plan.answer_requirements
        }
        task_specs = {spec.task_id: spec for spec in plan.tasks}
        values: list[str] = []
        for artifact in artifacts:
            spec = task_specs.get(artifact.task_id)
            if spec is None:
                continue
            supported_evidence = {
                evidence_requirement_id
                for claim in artifact.claims
                for evidence_requirement_id in claim.evidence_requirement_ids
            }
            values.extend(
                requirement.description
                for requirement in spec.evidence_requirements
                if requirement.evidence_requirement_id not in supported_evidence
            )
            values.extend(
                answer_descriptions.get(requirement_id, "")
                for requirement_id in artifact.uncovered_requirement_ids
            )
            values.extend(
                reference
                for worker in self._worker_artifacts.get(artifact.task_id, {}).values()
                for reference in worker.cross_references
            )
        if plan.scenario is not None:
            values.extend(
                unknown.description for unknown in plan.scenario.material_unknowns
            )

        selected: list[str] = []
        seen: set[str] = set()
        for value in values:
            bounded = value if self._unlimited_research else _bounded_text(value, 500)
            key = bounded.casefold()
            if not bounded or key in seen:
                continue
            seen.add(key)
            selected.append(bounded)
            if (
                not self._unlimited_research
                and len(selected) >= self._limits.max_artifact_list_items
            ):
                break
        return tuple(selected)

    def _scenario_context_for_task(self, spec: TaskSpec) -> dict[str, object] | None:
        """Return only the scenario inputs explicitly assigned to one task."""

        scenario = self._plan.scenario if self._plan is not None else None
        if scenario is None:
            return None
        fact_ids = set(spec.fact_ids)
        unknown_ids = set(spec.unknown_ids)
        branch_ids = set(spec.branch_ids)
        return {
            "jurisdiction": (
                (
                    scenario.jurisdiction
                    if self._unlimited_research
                    else _bounded_text(scenario.jurisdiction, 300)
                )
                if scenario.jurisdiction is not None
                else None
            ),
            "law_as_of_date": scenario.law_as_of_date,
            "facts": [
                {
                    "fact_id": fact.fact_id,
                    "description": (
                        fact.description
                        if self._unlimited_research
                        else _bounded_text(fact.description, 600)
                    ),
                    "requirement_ids": fact.requirement_ids,
                }
                for fact in scenario.facts
                if fact.fact_id in fact_ids
            ],
            "material_unknowns": [
                {
                    "unknown_id": unknown.unknown_id,
                    "description": (
                        unknown.description
                        if self._unlimited_research
                        else _bounded_text(unknown.description, 600)
                    ),
                    "why_material": (
                        unknown.why_material
                        if self._unlimited_research
                        else _bounded_text(unknown.why_material, 600)
                    ),
                    "requirement_ids": unknown.requirement_ids,
                }
                for unknown in scenario.material_unknowns
                if unknown.unknown_id in unknown_ids
            ],
            "decision_branches": [
                {
                    "branch_id": branch.branch_id,
                    "condition": (
                        branch.condition
                        if self._unlimited_research
                        else _bounded_text(branch.condition, 600)
                    ),
                    "consequence": (
                        branch.consequence
                        if self._unlimited_research
                        else _bounded_text(branch.consequence, 600)
                    ),
                    "requirement_ids": branch.requirement_ids,
                }
                for branch in scenario.decision_branches
                if branch.branch_id in branch_ids
            ],
        }

    async def _run_application_task(self, spec: TaskSpec) -> TaskArtifact:
        """Apply verified dependency outputs without opening a retrieval boundary."""

        event_id = f"task-{spec.task_id}"
        dependency_artifacts = [
            self._task_artifacts[task_id]
            for task_id in spec.depends_on
            if task_id in self._task_artifacts
        ]
        self._emit(
            event_id=event_id,
            kind="task_started",
            role="task",
            status="started",
            label=(
                "Applying evidence to a scenario"
                if spec.kind == TaskKind.APPLICATION
                else "Integrating grounded task results"
            ),
            detail=spec.issue,
            task_id=spec.task_id,
            agent_id=f"task-application-{spec.task_id}",
        )

        has_grounded_dependency = any(
            artifact.claims or artifact.application_findings
            for artifact in dependency_artifacts
        )
        if not has_grounded_dependency:
            artifact = self._failed_task_artifact(
                spec,
                "No grounded dependency output was available for application.",
            )
        else:
            try:
                raw, _usage = await self._generate_structured(
                    client=self._task_llm,
                    role="task",
                    purpose=(
                        "scenario_application"
                        if spec.kind == TaskKind.APPLICATION
                        else "task_integration"
                    ),
                    prompt=(
                        APPLICATION_TASK_SYSTEM_PROMPT
                        if spec.kind == TaskKind.APPLICATION
                        else INTEGRATION_TASK_SYSTEM_PROMPT
                    ),
                    user_text=(
                        f"TASK SPEC:\n{spec.model_dump_json()}\n\n"
                        "ASSIGNED SCENARIO INPUTS:\n"
                        f"{json.dumps(self._scenario_context_for_task(spec), ensure_ascii=False, separators=(',', ':'))}\n\n"
                        "DECLARED DEPENDENCY ARTIFACTS:\n"
                        f"{self._task_reports_for_parent(dependency_artifacts)}"
                    ),
                    operation_id=f"task-application:{spec.task_id}",
                    schema=TaskArtifact,
                    task_id=spec.task_id,
                    agent_id=f"task-application-{spec.task_id}",
                )
                assert isinstance(raw, TaskArtifact)
                artifact = self._verify_application_artifact(
                    raw,
                    spec=spec,
                    dependency_artifacts=dependency_artifacts,
                )
            except Exception as exc:
                logger.exception("Application task failed for %s", spec.task_id)
                artifact = self._failed_task_artifact(spec, _bounded_error(exc))

        status: ProgressStatus = (
            "completed" if artifact.status != TaskStatus.FAILED else "failed"
        )
        self._emit(
            event_id=event_id,
            kind=("task_completed" if status == "completed" else "task_failed"),
            role="task",
            status=status,
            label=(
                "Scenario application complete"
                if status == "completed"
                else "Scenario application failed"
            ),
            detail=(
                f"{len(artifact.application_findings)} grounded finding(s); "
                f"{len(artifact.uncovered_requirement_ids)} gap(s)"
            ),
            task_id=spec.task_id,
            agent_id=f"task-application-{spec.task_id}",
        )
        return artifact

    def _verify_application_artifact(
        self,
        candidate: TaskArtifact,
        *,
        spec: TaskSpec,
        dependency_artifacts: Sequence[TaskArtifact],
    ) -> TaskArtifact:
        """Keep only conclusions whose complete reference set is available."""

        plan = self._plan
        if plan is None:
            return self._failed_task_artifact(
                spec,
                "The validated global plan was unavailable.",
            )
        allowed_requirement_ids = set(spec.requirement_ids)
        allowed_fact_ids = set(spec.fact_ids)
        allowed_branch_ids = set(spec.branch_ids)
        allowed_claim_ids = {
            claim.claim_id
            for artifact in dependency_artifacts
            for claim in artifact.claims
        } | {
            claim_id
            for artifact in dependency_artifacts
            for finding in artifact.application_findings
            for claim_id in finding.supporting_claim_ids
        }
        present_dependency_ids = {artifact.task_id for artifact in dependency_artifacts}
        allowed_dependency_refs = {
            (reference.task_id, reference.output_id)
            for reference in spec.consumes
            if reference.task_id in present_dependency_ids
        }

        scenario = plan.scenario
        assigned_unknowns = []
        if scenario is not None:
            assigned_unknowns = [
                unknown
                for unknown in scenario.material_unknowns
                if unknown.unknown_id in set(spec.unknown_ids)
            ]
        allowed_limitations = [
            text
            for unknown in assigned_unknowns
            for text in (unknown.description, unknown.why_material)
        ]
        for artifact in dependency_artifacts:
            allowed_limitations.extend(artifact.gaps)
            allowed_limitations.extend(artifact.conflicts)
        limitation_by_key = {
            _normalize_evidence_text(value).casefold(): (
                value
                if self._unlimited_research
                else _bounded_text(
                    value,
                    self._limits.max_artifact_text_chars,
                )
            )
            for value in allowed_limitations
            if _normalize_evidence_text(value)
        }

        findings: list[DerivedConclusion] = []
        seen_ids: set[str] = set()
        candidate_findings = list(candidate.application_findings)
        if not self._unlimited_research:
            candidate_findings = candidate_findings[
                : self._limits.max_artifact_list_items
            ]
        for candidate_finding in candidate_findings:
            conclusion_id = re.sub(
                r"[^A-Za-z0-9_-]+",
                "_",
                candidate_finding.conclusion_id.strip(),
            ).strip("_")[:64]
            if not conclusion_id:
                conclusion_id = f"{spec.task_id}_finding_{len(findings) + 1}"[:64]
            if conclusion_id in seen_ids:
                continue
            requirement_ids = [
                requirement_id
                for requirement_id in spec.requirement_ids
                if requirement_id in set(candidate_finding.requirement_ids)
                and requirement_id in allowed_requirement_ids
            ]
            fact_ids = [
                fact_id
                for fact_id in spec.fact_ids
                if fact_id in set(candidate_finding.fact_ids)
                and fact_id in allowed_fact_ids
            ]
            branch_ids = [
                branch_id
                for branch_id in spec.branch_ids
                if branch_id in set(candidate_finding.branch_ids)
                and branch_id in allowed_branch_ids
            ]
            supporting_claim_ids = [
                claim_id
                for claim_id in dict.fromkeys(candidate_finding.supporting_claim_ids)
                if claim_id in allowed_claim_ids
            ]
            dependency_refs = [
                reference
                for reference in candidate_finding.dependency_refs
                if (reference.task_id, reference.output_id) in allowed_dependency_refs
            ]
            if not self._unlimited_research:
                supporting_claim_ids = supporting_claim_ids[
                    : self._limits.max_artifact_list_items
                ]
                dependency_refs = dependency_refs[
                    : self._limits.max_artifact_list_items
                ]
            limitations = [
                limitation_by_key[key]
                for key in dict.fromkeys(
                    _normalize_evidence_text(value).casefold()
                    for value in candidate_finding.limitations
                )
                if key in limitation_by_key
            ]
            # Assigned material unknowns are always carried forward even when
            # the model omits them. User-visible conditionality is not an
            # optional stylistic choice.
            for value in limitation_by_key.values():
                if value not in limitations:
                    limitations.append(value)
                if (
                    not self._unlimited_research
                    and len(limitations) >= self._limits.max_artifact_list_items
                ):
                    break

            finding_text = (
                candidate_finding.finding.strip()
                if self._unlimited_research
                else _bounded_text(
                    candidate_finding.finding,
                    self._limits.max_artifact_text_chars,
                )
            )
            if (
                not finding_text
                or not requirement_ids
                or (spec.kind == TaskKind.APPLICATION and not fact_ids)
                or (spec.kind == TaskKind.APPLICATION and not supporting_claim_ids)
                or (spec.kind == TaskKind.APPLICATION and not dependency_refs)
                or (
                    spec.kind == TaskKind.INTEGRATION
                    and (not supporting_claim_ids or not dependency_refs)
                )
            ):
                continue
            seen_ids.add(conclusion_id)
            findings.append(
                candidate_finding.model_copy(
                    update={
                        "conclusion_id": conclusion_id,
                        "finding": finding_text,
                        "requirement_ids": requirement_ids,
                        "fact_ids": fact_ids,
                        "branch_ids": branch_ids,
                        "supporting_claim_ids": supporting_claim_ids,
                        "dependency_refs": dependency_refs,
                        "limitations": limitations,
                    }
                )
            )

        supported_ids = {
            requirement_id
            for finding in findings
            for requirement_id in finding.requirement_ids
        }
        covered = [
            requirement_id
            for requirement_id in spec.requirement_ids
            if requirement_id in supported_ids
        ]
        uncovered = [
            requirement_id
            for requirement_id in spec.requirement_ids
            if requirement_id not in supported_ids
        ]
        status = (
            TaskStatus.COMPLETE
            if findings and not uncovered
            else TaskStatus.PARTIAL
            if findings
            else TaskStatus.FAILED
        )
        gaps = list(uncovered)
        for value in limitation_by_key.values():
            if value not in gaps:
                gaps.append(value)
            if (
                not self._unlimited_research
                and len(gaps) >= self._limits.max_artifact_list_items
            ):
                break
        conflicts = list(
            dict.fromkeys(
                conflict
                for artifact in dependency_artifacts
                for conflict in artifact.conflicts
            )
        )
        if not self._unlimited_research:
            conflicts = conflicts[: self._limits.max_artifact_list_items]
        verified = TaskArtifact(
            task_id=spec.task_id,
            status=status,
            answer_fragment=(
                " ".join(finding.finding for finding in findings) if findings else None
            ),
            covered_requirement_ids=covered,
            uncovered_requirement_ids=uncovered,
            claims=[],
            application_findings=findings,
            conflicts=conflicts,
            gaps=gaps,
            contributing_worker_ids=[],
        )
        try:
            return validate_task_artifact(
                verified,
                plan=plan,
                dependency_artifacts=dependency_artifacts,
            )
        except ArtifactValidationError:
            logger.exception(
                "Application artifact failed deterministic reference validation"
            )
            return self._failed_task_artifact(
                spec,
                "Application conclusions failed reference validation.",
            )

    async def _coordinate_task(
        self,
        *,
        spec: TaskSpec,
        dependency_artifacts: Sequence[TaskArtifact],
        worker_artifacts: Sequence[WorkerArtifact],
        searches_run: Sequence[str],
        round_index: int,
        uncovered_evidence: Sequence[EvidenceRequirement],
        attempts_by_requirement: dict[str, set[str]],
        decision_index: int = 1,
        rejection: str | None = None,
    ) -> SearchAssignmentBatch:
        agent_id = f"task-coordinator-{spec.task_id}"
        wave_label = (
            f"{round_index}/model-controlled"
            if self._unlimited_research
            else f"{round_index}/{self._limits.max_worker_rounds}"
        )
        raw, _usage = await self._generate_structured(
            client=self._task_llm,
            role="task",
            purpose="task_coordination",
            prompt=build_task_coordinator_prompt(
                max_assignments_per_wave=(
                    None
                    if self._unlimited_research
                    else self._limits.max_assignments_per_wave
                ),
                max_worker_rounds=(
                    None if self._unlimited_research else self._limits.max_worker_rounds
                ),
            ),
            user_text=(
                f"WAVE: {wave_label}\n\n"
                f"TASK SPEC:\n{spec.model_dump_json()}\n\n"
                "DEPENDENCY ARTIFACTS:\n"
                f"{self._task_reports_for_parent(dependency_artifacts)}\n\n"
                "WORKER ARTIFACTS SO FAR:\n"
                f"{self._worker_reports_for_parent(worker_artifacts)}\n\n"
                "UNRESOLVED EVIDENCE REQUIREMENTS:\n"
                f"{json.dumps([requirement.model_dump(mode='json') for requirement in uncovered_evidence], ensure_ascii=False, separators=(',', ':'))}\n\n"
                "SEARCH ATTEMPTS BY EVIDENCE REQUIREMENT:\n"
                f"{json.dumps({requirement.evidence_requirement_id: sorted(attempts_by_requirement.get(requirement.evidence_requirement_id, set())) for requirement in uncovered_evidence}, ensure_ascii=False, separators=(',', ':'))}\n\n"
                "SEARCHES ALREADY RUN:\n"
                f"{json.dumps(list(searches_run), ensure_ascii=False)}\n\n"
                "REJECTED PREVIOUS DECISION:\n"
                f"{rejection or 'none'}"
            ),
            operation_id=(
                f"task-coordinate:{spec.task_id}:{round_index}:{decision_index}"
            ),
            schema=SearchAssignmentBatch,
            task_id=spec.task_id,
            agent_id=agent_id,
        )
        assert isinstance(raw, SearchAssignmentBatch)
        return raw

    def _single_pass_dispatch(
        self,
        spec: TaskSpec,
        *,
        searches_run: set[str],
    ) -> SearchAssignmentBatch:
        """Create the one deterministic search authorized by a simple plan."""

        return SearchAssignmentBatch(
            task_id=spec.task_id,
            stop=False,
            stop_reason=None,
            assignments=[
                SearchAssignment(
                    assignment_id=f"{spec.task_id}_single_pass",
                    task_id=spec.task_id,
                    query=spec.question,
                    objective=spec.expected_output,
                    evidence_requirements=[
                        requirement.evidence_requirement_id
                        for requirement in spec.evidence_requirements
                    ],
                    excluded_queries=sorted(searches_run),
                    as_of_date=_validated_as_of_date(spec.as_of_date),
                    filters=spec.filters,
                )
            ],
        )

    def _validated_assignments(
        self,
        spec: TaskSpec,
        dispatch: SearchAssignmentBatch,
        *,
        searches_run: set[str],
        existing_assignment_ids: set[str],
        allowed_evidence_requirement_ids: set[str] | None = None,
    ) -> list[SearchAssignment]:
        if dispatch.task_id != spec.task_id or dispatch.stop:
            return []
        known_evidence_ids = {
            requirement.evidence_requirement_id
            for requirement in spec.evidence_requirements
        }
        accepted: list[SearchAssignment] = []
        accepted_queries: set[str] = set()
        accepted_ids: set[str] = set()
        for candidate in dispatch.assignments:
            query = (
                candidate.query.strip()
                if self._unlimited_research
                else _bounded_text(
                    candidate.query,
                    self._limits.max_query_chars,
                )
            )
            normalized = _normalized_query(query)
            assignment_id = _safe_identifier(
                candidate.assignment_id,
                fallback=f"assignment_{len(accepted) + 1}",
            )
            prior_queries = [*searches_run, *accepted_queries]
            if (
                candidate.task_id != spec.task_id
                or not query
                or not assignment_id
                or assignment_id in existing_assignment_ids
                or assignment_id in accepted_ids
                or _is_near_duplicate_query(query, prior_queries)
            ):
                continue
            evidence_requirements = [
                requirement_id
                for requirement_id in dict.fromkeys(candidate.evidence_requirements)
                if requirement_id in known_evidence_ids
                and (
                    allowed_evidence_requirement_ids is None
                    or requirement_id in allowed_evidence_requirement_ids
                )
            ]
            if not self._unlimited_research:
                evidence_requirements = evidence_requirements[
                    : self._limits.max_artifact_list_items
                ]
            if not evidence_requirements:
                continue
            accepted.append(
                candidate.model_copy(
                    update={
                        "assignment_id": assignment_id,
                        "query": query,
                        "objective": (
                            candidate.objective.strip()
                            if self._unlimited_research
                            else _bounded_text(
                                candidate.objective,
                                self._limits.max_artifact_text_chars,
                            )
                        ),
                        "evidence_requirements": evidence_requirements,
                        "excluded_queries": (
                            list(candidate.excluded_queries)
                            if self._unlimited_research
                            else [
                                _bounded_text(
                                    excluded,
                                    self._limits.max_query_chars,
                                )
                                for excluded in candidate.excluded_queries[
                                    : self._limits.max_artifact_list_items
                                ]
                            ]
                        ),
                        "as_of_date": _validated_as_of_date(
                            candidate.as_of_date or spec.as_of_date
                        ),
                        "filters": self._bounded_filter(
                            candidate.filters or spec.filters
                        ),
                    }
                )
            )
            accepted_queries.add(normalized)
            accepted_ids.add(assignment_id)
            if (
                not self._unlimited_research
                and len(accepted) >= self._limits.max_assignments_per_wave
            ):
                break
        return accepted

    def _bounded_filter(self, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split()).strip()
        if not normalized:
            return None
        if (
            not self._unlimited_research
            and len(normalized) > self._limits.max_query_chars
        ):
            return None
        return normalized

    @staticmethod
    def _search_key(assignment: SearchAssignment) -> tuple[str, str, str]:
        return (
            _normalized_query(assignment.query),
            _normalized_query(assignment.filters or ""),
            (assignment.as_of_date or "").strip(),
        )

    async def _perform_search(self, assignment: SearchAssignment) -> SearchResult:
        raw: SearchResult | Awaitable[SearchResult]
        if self._search_runner_in_thread:
            raw = await asyncio.to_thread(
                self._search_runner,
                query=assignment.query,
                filters=assignment.filters,
                as_of_date=assignment.as_of_date,
            )
        else:
            raw = self._search_runner(
                query=assignment.query,
                filters=assignment.filters,
                as_of_date=assignment.as_of_date,
            )
        if inspect.isawaitable(raw):
            return await cast(Awaitable[SearchResult], raw)
        return cast(SearchResult, raw)

    async def _execute_search(
        self,
        assignment: SearchAssignment,
    ) -> SearchResult:
        """Await one cached search with a hard caller-visible timeout."""

        key = self._search_key(assignment)
        operation = self._search_tasks.get(key)
        if operation is None:
            operation = asyncio.create_task(
                self._perform_search(assignment),
                name=f"multi-agent-search:{assignment.assignment_id}",
            )
            operation.add_done_callback(_consume_background_task_exception)
            self._search_tasks[key] = operation
        try:
            return await asyncio.wait_for(
                asyncio.shield(operation),
                timeout=self._limits.search_timeout_seconds,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                "Indexed search exceeded the multi-agent timeout."
            ) from exc
        except asyncio.CancelledError:
            # Preserve the shielded operation for a resumed worker.
            raise
        except Exception:
            # A completed failed operation may be retried once by the bounded
            # worker loop. In-flight timeouts remain cached and are re-awaited.
            if operation.done() and self._search_tasks.get(key) is operation:
                self._search_tasks.pop(key, None)
            raise

    def _forget_completed_search(self, assignment: SearchAssignment) -> None:
        key = self._search_key(assignment)
        operation = self._search_tasks.get(key)
        if operation is not None and operation.done():
            self._search_tasks.pop(key, None)

    async def _run_worker(
        self,
        *,
        spec: TaskSpec,
        assignment: SearchAssignment,
        worker_id: str,
    ) -> WorkerArtifact:
        """Let one search agent iterate until it finds support or self-stops."""

        if not self._unlimited_research:
            return await self._run_worker_once(
                spec=spec,
                assignment=assignment,
                worker_id=worker_id,
            )

        searches_run: list[str] = []
        completed_queries: set[str] = set()
        claims: list[EvidenceClaim] = []
        reported_gaps: list[str] = []
        cross_references: list[str] = []
        current_assignment = assignment
        decision_index = 0
        last_artifact: WorkerArtifact | None = None
        successful_search_seen = False
        rejection: str | None = None

        while True:
            branch_worker_id = (
                worker_id
                if not searches_run
                else f"{worker_id}-search-{len(searches_run) + 1}"
            )
            artifact = await self._run_worker_once(
                spec=spec,
                assignment=current_assignment,
                worker_id=branch_worker_id,
                persist_artifact=False,
            )
            last_artifact = artifact
            successful_search_seen = (
                successful_search_seen or artifact.status != WorkerStatus.FAILED
            )
            searches_run.extend(artifact.searches_run)
            if artifact.status != WorkerStatus.FAILED:
                completed_queries.update(
                    _normalized_query(query)
                    for query in artifact.searches_run
                    if query.strip()
                )
            claims.extend(artifact.claims)
            reported_gaps.extend(artifact.gaps)
            cross_references.extend(artifact.cross_references)

            supported = {
                evidence_requirement_id
                for claim in claims
                for evidence_requirement_id in claim.evidence_requirement_ids
            }
            unresolved = [
                requirement_id
                for requirement_id in assignment.evidence_requirements
                if requirement_id not in supported
            ]

            decision_index += 1
            dispatch = await self._continue_worker_search(
                spec=spec,
                assignment=assignment,
                worker_id=worker_id,
                searches_run=searches_run,
                claims=claims,
                cross_references=cross_references,
                unresolved=unresolved,
                last_artifact=last_artifact,
                decision_index=decision_index,
                rejection=rejection,
            )
            if dispatch.stop:
                return self._aggregate_persistent_worker_artifact(
                    spec=spec,
                    assignment=assignment,
                    worker_id=worker_id,
                    searches_run=searches_run,
                    claims=claims,
                    reported_gaps=reported_gaps,
                    cross_references=cross_references,
                    unresolved=unresolved,
                    last_artifact=last_artifact,
                    successful_search_seen=successful_search_seen,
                )

            next_assignments = self._validated_assignments(
                spec,
                dispatch,
                searches_run=completed_queries,
                existing_assignment_ids=set(),
                allowed_evidence_requirement_ids=set(
                    unresolved or assignment.evidence_requirements
                ),
            )
            if len(next_assignments) == 1:
                current_assignment = next_assignments[0]
                rejection = None
                self._emit(
                    event_id=f"worker-search-loop-{spec.task_id}-{assignment.assignment_id}",
                    kind="worker_search_continued",
                    role="worker",
                    status="started",
                    label="Search agent is trying another approach",
                    detail=current_assignment.query,
                    task_id=spec.task_id,
                    agent_id=worker_id,
                )
                continue
            rejection = (
                "The proposed continuation must contain exactly one valid, "
                "materially distinct next tool call. Review the latest result, "
                "then choose one next query or explicitly stop."
            )

    async def _continue_worker_search(
        self,
        *,
        spec: TaskSpec,
        assignment: SearchAssignment,
        worker_id: str,
        searches_run: Sequence[str],
        claims: Sequence[EvidenceClaim],
        cross_references: Sequence[str],
        unresolved: Sequence[str],
        last_artifact: WorkerArtifact | None,
        decision_index: int,
        rejection: str | None,
    ) -> SearchAssignmentBatch:
        raw, _usage = await self._generate_structured(
            client=self._worker_llm,
            role="worker",
            purpose="search_continuation",
            prompt=WORKER_SEARCH_CONTINUATION_PROMPT,
            user_text=(
                f"TASK SPEC:\n{spec.model_dump_json()}\n\n"
                f"ORIGINAL ASSIGNMENT:\n{assignment.model_dump_json()}\n\n"
                "UNRESOLVED EVIDENCE REQUIREMENT IDS:\n"
                f"{json.dumps(list(unresolved), ensure_ascii=False)}\n\n"
                "SEARCHES ALREADY RUN:\n"
                f"{json.dumps(list(searches_run), ensure_ascii=False)}\n\n"
                "EXACT VERIFIED EVIDENCE EXCERPTS FOUND SO FAR:\n"
                f"{json.dumps([claim.evidence_excerpt for claim in claims], ensure_ascii=False)}\n\n"
                "EXPLICIT CROSS-REFERENCES FOUND SO FAR:\n"
                f"{json.dumps(list(dict.fromkeys(cross_references)), ensure_ascii=False)}\n\n"
                "LAST INDEXED_SEARCH TOOL RESULT:\n"
                f"{self._worker_reports_for_parent([last_artifact]) if last_artifact else 'none'}\n\n"
                "REJECTED PREVIOUS CONTINUATION:\n"
                f"{rejection or 'none'}"
            ),
            operation_id=(
                f"worker-search-decision:{spec.task_id}:"
                f"{assignment.assignment_id}:{decision_index}"
            ),
            schema=SearchAssignmentBatch,
            task_id=spec.task_id,
            agent_id=worker_id,
        )
        assert isinstance(raw, SearchAssignmentBatch)
        return raw

    @staticmethod
    def _aggregate_persistent_worker_artifact(
        *,
        spec: TaskSpec,
        assignment: SearchAssignment,
        worker_id: str,
        searches_run: Sequence[str],
        claims: Sequence[EvidenceClaim],
        reported_gaps: Sequence[str],
        cross_references: Sequence[str],
        unresolved: Sequence[str],
        last_artifact: WorkerArtifact | None,
        successful_search_seen: bool,
    ) -> WorkerArtifact:
        unique_claims = list(
            {
                (
                    claim.document_id,
                    claim.chunk_id,
                    _normalize_evidence_text(claim.evidence_excerpt),
                ): claim
                for claim in claims
            }.values()
        )
        status = (
            WorkerStatus.SUCCESS
            if unique_claims
            else WorkerStatus.FAILED
            if last_artifact is not None
            and last_artifact.status == WorkerStatus.FAILED
            and not successful_search_seen
            else WorkerStatus.NO_EVIDENCE
        )
        gaps = list(dict.fromkeys([*reported_gaps, *unresolved]))
        return WorkerArtifact(
            task_id=spec.task_id,
            assignment_id=assignment.assignment_id,
            worker_id=worker_id,
            status=status,
            searches_run=list(searches_run),
            claims=unique_claims,
            gaps=gaps,
            cross_references=list(dict.fromkeys(cross_references)),
            error_code=(
                last_artifact.error_code
                if status == WorkerStatus.FAILED and last_artifact is not None
                else None
            ),
            error_message=(
                last_artifact.error_message
                if status == WorkerStatus.FAILED and last_artifact is not None
                else None
            ),
        )

    async def _run_worker_once(
        self,
        *,
        spec: TaskSpec,
        assignment: SearchAssignment,
        worker_id: str,
        persist_artifact: bool = True,
    ) -> WorkerArtifact:
        event_id = f"worker-{spec.task_id}-{assignment.assignment_id}"
        self._emit(
            event_id=event_id,
            kind="worker_started",
            role="worker",
            status="started",
            label="Searching indexed evidence",
            detail=assignment.objective,
            task_id=spec.task_id,
            agent_id=worker_id,
        )
        search_sequence = self._emit(
            event_id=f"search-{spec.task_id}-{assignment.assignment_id}",
            kind="search_started",
            role="worker",
            status="started",
            label="Running search",
            detail=assignment.query,
            task_id=spec.task_id,
            agent_id=worker_id,
        )
        result: SearchResult | None = None
        search_error: str | None = None
        automatic_attempts = (
            1 if self._unlimited_research else self._limits.max_search_attempts
        )
        async with self._retrieval_semaphore:
            for attempt in range(1, automatic_attempts + 1):
                try:
                    result = await self._execute_search(assignment)
                except Exception as exc:
                    search_error = _bounded_error(exc)
                else:
                    search_error = (
                        _bounded_error(result.error) if result.error else None
                    )
                if attempt >= automatic_attempts or not _is_transient_search_error(
                    search_error
                ):
                    break
                self._forget_completed_search(assignment)
        if _is_transient_search_error(search_error):
            # A failed tool call is returned to the owning search agent. Do not
            # cache it: the agent, rather than a hidden retry counter, decides
            # whether to request this exact tool call again.
            self._forget_completed_search(assignment)

        hits = list(result.hits if result is not None else [])
        if not self._unlimited_research:
            hits = hits[: self._limits.worker_hit_limit]
        records = self._register_evidence(spec, assignment, hits)
        rendered_bundle = self._render_evidence_bundle(
            records,
            max_chars=(
                None if self._unlimited_research else self._limits.worker_hit_chars
            ),
        )
        rendered = rendered_bundle.text
        exposed_records = list(rendered_bundle.records)
        task_pool = self._task_evidence.setdefault(spec.task_id, {})
        for record in exposed_records:
            task_pool[record.chunk_id] = record
        if self._on_retrieval is not None:
            try:
                self._on_retrieval(
                    len(exposed_records),
                    len(rendered),
                    spec.task_id,
                    worker_id,
                    search_sequence,
                )
            except Exception:
                logger.exception("Multi-agent retrieval telemetry callback failed")
        self._emit(
            event_id=f"search-{spec.task_id}-{assignment.assignment_id}",
            kind="search_completed",
            role="worker",
            status="failed" if search_error and not exposed_records else "completed",
            label=(
                "Search complete"
                if exposed_records
                else "Search returned no usable evidence"
            ),
            detail=(
                f"{len(exposed_records)} candidate chunk(s)"
                if not search_error
                else (
                    f"{len(exposed_records)} chunk(s); {_bounded_error(search_error)}"
                )
            ),
            task_id=spec.task_id,
            agent_id=worker_id,
        )

        if search_error and not exposed_records:
            artifact = self._failed_worker_artifact(
                spec=spec,
                assignment=assignment,
                worker_id=worker_id,
                error=search_error,
            )
        elif not exposed_records:
            artifact = WorkerArtifact(
                task_id=spec.task_id,
                assignment_id=assignment.assignment_id,
                worker_id=worker_id,
                status=WorkerStatus.NO_EVIDENCE,
                searches_run=[assignment.query],
                claims=[],
                gaps=list(assignment.evidence_requirements),
                cross_references=[],
                error_code=None,
                error_message=None,
            )
        else:
            try:
                raw, _usage = await self._generate_structured(
                    client=self._worker_llm,
                    role="worker",
                    purpose="evidence_extraction",
                    prompt=EVIDENCE_WORKER_SYSTEM_PROMPT,
                    user_text=(
                        "TASK ANSWER REQUIREMENT IDS:\n"
                        f"{json.dumps(spec.requirement_ids, ensure_ascii=False)}\n\n"
                        "TYPED EVIDENCE REQUIREMENTS:\n"
                        f"{json.dumps([item.model_dump(mode='json') for item in spec.evidence_requirements], ensure_ascii=False)}\n\n"
                        f"SEARCH ASSIGNMENT:\n{assignment.model_dump_json()}\n\n"
                        "RETRIEVED HITS:\n"
                        f"{rendered}"
                    ),
                    operation_id=(
                        f"worker-extract:{spec.task_id}:{assignment.assignment_id}"
                    ),
                    schema=WorkerArtifact,
                    task_id=spec.task_id,
                    agent_id=worker_id,
                )
                assert isinstance(raw, WorkerArtifact)
                artifact = self._verify_worker_artifact(
                    raw,
                    spec=spec,
                    assignment=assignment,
                    worker_id=worker_id,
                    records=exposed_records,
                    search_error=search_error,
                )
            except Exception as exc:
                logger.exception("Evidence worker %s failed", worker_id)
                artifact = self._failed_worker_artifact(
                    spec=spec,
                    assignment=assignment,
                    worker_id=worker_id,
                    error=_bounded_error(exc),
                )

        if persist_artifact:
            self._worker_artifacts.setdefault(spec.task_id, {})[
                assignment.assignment_id
            ] = artifact
        self._emit(
            event_id=event_id,
            kind="worker_completed",
            role="worker",
            status=(
                "failed" if artifact.status == WorkerStatus.FAILED else "completed"
            ),
            label=(
                "Evidence worker complete"
                if artifact.status != WorkerStatus.FAILED
                else "Evidence worker failed"
            ),
            detail=(
                f"{len(artifact.claims)} verified claim(s); "
                f"status={artifact.status.value}"
            ),
            task_id=spec.task_id,
            agent_id=worker_id,
        )
        return artifact

    def _register_evidence(
        self,
        spec: TaskSpec,
        assignment: SearchAssignment,
        hits: Sequence[SearchHit],
    ) -> list[EvidenceRecord]:
        records: list[EvidenceRecord] = []
        seen: set[str] = set()
        for index, hit in enumerate(hits, start=1):
            if not hit.chunk_id or hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            source_key = (hit.doc_id, hit.chunk_id)
            record = self._evidence_by_source.get(source_key)
            if record is None:
                evidence_id = (
                    f"{spec.task_id}:{assignment.assignment_id}:evidence_{index}"
                )
                record = EvidenceRecord(
                    evidence_id=evidence_id,
                    document_id=hit.doc_id,
                    chunk_id=hit.chunk_id,
                    readable_title=_readable_title(hit),
                    locator=_readable_locator(hit.metadata, hit.position),
                    text=html.unescape(hit.text).strip(),
                    score=float(hit.score),
                    metadata=dict(hit.metadata),
                )
                self._evidence[evidence_id] = record
                self._evidence_by_source[source_key] = record
            elif float(hit.score) > record.score:
                record = EvidenceRecord(
                    evidence_id=record.evidence_id,
                    document_id=record.document_id,
                    chunk_id=record.chunk_id,
                    readable_title=record.readable_title,
                    locator=record.locator,
                    text=record.text,
                    score=float(hit.score),
                    metadata=record.metadata,
                )
                self._evidence[record.evidence_id] = record
                self._evidence_by_source[source_key] = record
            records.append(record)
        return records

    def _verify_worker_artifact(
        self,
        candidate: WorkerArtifact,
        *,
        spec: TaskSpec,
        assignment: SearchAssignment,
        worker_id: str,
        records: Sequence[EvidenceRecord],
        search_error: str | None,
    ) -> WorkerArtifact:
        by_source = {
            (record.document_id, record.chunk_id): record for record in records
        }
        evidence_requirements = {
            requirement.evidence_requirement_id: requirement
            for requirement in spec.evidence_requirements
        }
        verified: list[EvidenceClaim] = []
        seen_claims: set[tuple[str, str, str]] = set()
        rejected = 0
        candidate_claims = list(candidate.claims)
        if not self._unlimited_research:
            candidate_claims = candidate_claims[: self._limits.max_claims_per_worker]
        for claim in candidate_claims:
            record = by_source.get((claim.document_id, claim.chunk_id))
            claim_key = (
                claim.document_id,
                claim.chunk_id,
                _normalize_evidence_text(claim.claim).casefold(),
            )
            if claim_key in seen_claims:
                continue
            if record is None or not _claim_supported(claim, record):
                rejected += 1
                continue
            seen_claims.add(claim_key)
            supported_evidence_ids = [
                requirement_id
                for requirement_id in dict.fromkeys(claim.evidence_requirement_ids)
                if requirement_id in evidence_requirements
            ]
            mapped_requirement_ids = {
                requirement_id
                for evidence_requirement_id in supported_evidence_ids
                for requirement_id in evidence_requirements[
                    evidence_requirement_id
                ].requirement_ids
            }
            supported_requirement_ids = [
                requirement_id
                for requirement_id in spec.requirement_ids
                if requirement_id in mapped_requirement_ids
            ]
            claim_id = _claim_identifier(
                spec.task_id,
                assignment.assignment_id,
                len(verified) + 1,
            )
            verified.append(
                claim.model_copy(
                    update={
                        "claim_id": claim_id,
                        "document_id": record.document_id,
                        "chunk_id": record.chunk_id,
                        "readable_title": record.readable_title,
                        "locator": record.locator,
                        "claim": (
                            claim.evidence_excerpt
                            if self._unlimited_research
                            else _bounded_text(
                                claim.claim,
                                self._limits.max_artifact_text_chars,
                            )
                        ),
                        "evidence_excerpt": (
                            claim.evidence_excerpt
                            if self._unlimited_research
                            else _bounded_text(
                                claim.evidence_excerpt,
                                self._limits.max_artifact_text_chars,
                            )
                        ),
                        "requirement_ids": supported_requirement_ids,
                        "evidence_requirement_ids": supported_evidence_ids,
                        "fact_ids": [
                            fact_id
                            for fact_id in spec.fact_ids
                            if fact_id in set(claim.fact_ids)
                        ],
                        "effective_start_date": _effective_date(
                            record.metadata,
                            "effective_start_date",
                            "valid_from",
                        ),
                        "effective_end_date": _effective_date(
                            record.metadata,
                            "effective_end_date",
                            "valid_to",
                        ),
                    }
                )
            )
        supported = {
            requirement_id
            for claim in verified
            for requirement_id in claim.evidence_requirement_ids
        }
        gaps = [
            (
                requirement
                if self._unlimited_research
                else _bounded_text(
                    requirement,
                    self._limits.max_artifact_text_chars,
                )
            )
            for requirement in assignment.evidence_requirements
            if requirement not in supported
        ]
        if not self._unlimited_research:
            gaps = gaps[: self._limits.max_artifact_list_items]
        if rejected:
            gaps.append(
                f"{rejected} unsupported claim(s) were rejected by source validation."
            )
        if search_error:
            gaps.append(f"Search warning: {search_error}")
        status = WorkerStatus.SUCCESS if verified else WorkerStatus.NO_EVIDENCE
        normalized_evidence = " ".join(record.text.casefold() for record in records)
        cross_references = [
            (
                reference
                if self._unlimited_research
                else _bounded_text(
                    reference,
                    self._limits.max_artifact_text_chars,
                )
            )
            for reference in candidate.cross_references
            if _normalize_evidence_text(reference).casefold()
            in _normalize_evidence_text(normalized_evidence)
        ]
        if not self._unlimited_research:
            cross_references = cross_references[: self._limits.max_artifact_list_items]
        return WorkerArtifact(
            task_id=spec.task_id,
            assignment_id=assignment.assignment_id,
            worker_id=worker_id,
            status=status,
            searches_run=[assignment.query],
            claims=verified,
            gaps=gaps,
            cross_references=cross_references,
            error_code=None,
            error_message=None,
        )

    async def _review_task(
        self,
        *,
        spec: TaskSpec,
        dependency_artifacts: Sequence[TaskArtifact],
        worker_artifacts: Sequence[WorkerArtifact],
    ) -> TaskArtifact:
        agent_id = f"task-reviewer-{spec.task_id}"
        task_records = await self._ranked_task_evidence(spec)
        if self._unlimited_research:
            claimed_sources = {
                (claim.document_id, claim.chunk_id)
                for artifact in worker_artifacts
                for claim in artifact.claims
            }
            task_records = [
                record
                for record in task_records
                if (record.document_id, record.chunk_id) in claimed_sources
            ]
        review_evidence = self._render_evidence_bundle(
            task_records,
            max_chars=(
                None if self._unlimited_research else self._limits.review_hit_chars
            ),
        )
        exposed_records = list(review_evidence.records)
        if (
            not exposed_records
            and not any(artifact.claims for artifact in worker_artifacts)
            and not any(artifact.claims for artifact in dependency_artifacts)
        ):
            # With no verified material at any boundary, a reviewer can only
            # restate gaps. Preserve those deterministically and spend no
            # additional model tokens.
            return self._assemble_verified_task_artifact(
                spec,
                worker_artifacts=worker_artifacts,
            )
        try:
            raw, _usage = await self._generate_structured(
                client=self._task_llm,
                role="task",
                purpose="task_review",
                prompt=TASK_REVIEW_SYSTEM_PROMPT,
                user_text=(
                    f"TASK SPEC:\n{spec.model_dump_json()}\n\n"
                    "DEPENDENCY ARTIFACTS:\n"
                    f"{self._task_reports_for_parent(dependency_artifacts)}\n\n"
                    "WORKER ARTIFACTS:\n"
                    f"{self._worker_reports_for_parent(worker_artifacts)}\n\n"
                    "SERVER-VERIFIED EVIDENCE:\n"
                    f"{review_evidence.text}"
                ),
                operation_id=f"task-review:{spec.task_id}",
                schema=TaskArtifact,
                task_id=spec.task_id,
                agent_id=agent_id,
            )
            assert isinstance(raw, TaskArtifact)
            return self._verify_task_artifact(
                raw,
                spec=spec,
                worker_artifacts=worker_artifacts,
                records=exposed_records,
            )
        except Exception:
            logger.exception("Task reviewer failed for %s", spec.task_id)
            raise

    async def _ranked_task_evidence(self, spec: TaskSpec) -> list[EvidenceRecord]:
        """Globally rerank cross-assignment candidates against the TaskSpec."""

        records = list(self._task_evidence.get(spec.task_id, {}).values())
        by_source = {
            (record.document_id, record.chunk_id): record for record in records
        }
        candidates = [
            RankedDocument(
                doc_id=record.document_id,
                relative_path=record.readable_title,
                absolute_path=record.readable_title,
                position=None,
                text=record.text,
                semantic_score=record.score,
                metadata_score=0,
                chunk_id=record.chunk_id,
                chunk_type=str(record.metadata.get("chunk_type") or "text"),
                metadata=record.metadata,
            )
            for record in records
        ]
        ranked_records: list[EvidenceRecord]
        if len(candidates) > 1:
            operation = self._task_rerank_tasks.get(spec.task_id)
            if operation is None:

                async def rerank() -> list[tuple[RankedDocument, float]]:
                    if self._search_runner_in_thread:
                        return await asyncio.to_thread(
                            IndexedQueryEngine.rank_candidates,
                            query=spec.question,
                            documents=candidates,
                            limit=len(candidates),
                            diversify=False,
                        )
                    return IndexedQueryEngine.rank_candidates(
                        query=spec.question,
                        documents=candidates,
                        limit=len(candidates),
                        diversify=False,
                    )

                operation = asyncio.create_task(
                    rerank(),
                    name=f"multi-agent-rerank:{spec.task_id}",
                )
                operation.add_done_callback(_consume_background_task_exception)
                self._task_rerank_tasks[spec.task_id] = operation
            try:
                ranked_pairs = await asyncio.wait_for(
                    asyncio.shield(operation),
                    timeout=self._limits.search_timeout_seconds,
                )
            except TimeoutError:
                logger.warning(
                    "Task-global rerank timed out for %s; using retrieval scores",
                    spec.task_id,
                )
                ranked_pairs = []
            except Exception:
                logger.exception(
                    "Task-global rerank failed for %s; using retrieval scores",
                    spec.task_id,
                )
                ranked_pairs = []
            ranked_records = [
                by_source[(document.doc_id, document.chunk_id)]
                for document, _score in ranked_pairs
                if document.chunk_id
                and (document.doc_id, document.chunk_id) in by_source
            ]
            ranked_ids = {record.chunk_id for record in ranked_records}
            ranked_records.extend(
                sorted(
                    (record for record in records if record.chunk_id not in ranked_ids),
                    key=lambda record: record.score,
                    reverse=True,
                )
            )
        elif candidates:
            # Hosted reranking cannot change a one-item ordering.
            ranked_records = sorted(
                records, key=lambda record: record.score, reverse=True
            )
        else:
            ranked_records = []

        diversified: list[EvidenceRecord] = []
        deferred: list[EvidenceRecord] = []
        seen_documents: set[str] = set()
        for record in ranked_records:
            if record.document_id in seen_documents:
                deferred.append(record)
                continue
            seen_documents.add(record.document_id)
            diversified.append(record)
        diversified.extend(deferred)
        if self._unlimited_research:
            return diversified
        pool_limit = max(
            self._limits.max_claims_per_task * 2,
            self._limits.worker_hit_limit,
        )
        return diversified[:pool_limit]

    def _verify_task_artifact(
        self,
        candidate: TaskArtifact,
        *,
        spec: TaskSpec,
        worker_artifacts: Sequence[WorkerArtifact],
        records: Sequence[EvidenceRecord],
    ) -> TaskArtifact:
        by_source = {
            (record.document_id, record.chunk_id): record for record in records
        }
        allowed_worker_claims = {
            (
                claim.document_id,
                claim.chunk_id,
                _normalize_evidence_text(claim.evidence_excerpt),
            ): claim
            for artifact in worker_artifacts
            for claim in artifact.claims
        }
        claims: list[EvidenceClaim] = []
        seen: set[tuple[str, str, str]] = set()
        candidate_claims = list(candidate.claims)
        if not self._unlimited_research:
            candidate_claims = candidate_claims[: self._limits.max_claims_per_task]
        for claim in candidate_claims:
            record = by_source.get((claim.document_id, claim.chunk_id))
            allowed = allowed_worker_claims.get(
                (
                    claim.document_id,
                    claim.chunk_id,
                    _normalize_evidence_text(claim.evidence_excerpt),
                )
            )
            key = (
                claim.document_id,
                claim.chunk_id,
                _normalize_evidence_text(claim.evidence_excerpt),
            )
            if (
                record is None
                or allowed is None
                or key in seen
                or not _claim_supported(allowed, record)
            ):
                continue
            seen.add(key)
            claims.append(
                allowed.model_copy(
                    update={
                        "readable_title": record.readable_title,
                        "locator": record.locator,
                    }
                )
            )

        if not claims:
            for key, allowed in allowed_worker_claims.items():
                record = by_source.get((allowed.document_id, allowed.chunk_id))
                if (
                    record is None
                    or key in seen
                    or not _claim_supported(allowed, record)
                ):
                    continue
                seen.add(key)
                claims.append(
                    allowed.model_copy(
                        update={
                            "readable_title": record.readable_title,
                            "locator": record.locator,
                        }
                    )
                )
                if (
                    not self._unlimited_research
                    and len(claims) >= self._limits.max_claims_per_task
                ):
                    break

        supported_evidence_ids = {
            evidence_requirement_id
            for claim in claims
            for evidence_requirement_id in claim.evidence_requirement_ids
        }
        evidence_ids_by_requirement = {
            requirement_id: {
                requirement.evidence_requirement_id
                for requirement in spec.evidence_requirements
                if requirement_id in requirement.requirement_ids
            }
            for requirement_id in spec.requirement_ids
        }
        covered = [
            requirement_id
            for requirement_id in spec.requirement_ids
            if evidence_ids_by_requirement[requirement_id]
            and evidence_ids_by_requirement[requirement_id].issubset(
                supported_evidence_ids
            )
        ]
        uncovered = [
            requirement_id
            for requirement_id in spec.requirement_ids
            if requirement_id not in covered
        ]
        covered_set = set(covered)
        claims = [
            claim.model_copy(
                update={
                    "requirement_ids": [
                        requirement_id
                        for requirement_id in claim.requirement_ids
                        if requirement_id in covered_set
                    ]
                }
            )
            for claim in claims
        ]
        if claims and not uncovered:
            status = TaskStatus.COMPLETE
        elif claims:
            status = TaskStatus.PARTIAL
        else:
            status = TaskStatus.FAILED
        retained_sources = {(claim.document_id, claim.chunk_id) for claim in claims}
        contributors = [
            artifact.worker_id
            for artifact in worker_artifacts
            if any(
                (claim.document_id, claim.chunk_id) in retained_sources
                for claim in artifact.claims
            )
        ]
        verified_gap_inputs: list[str] = list(uncovered)
        for worker in worker_artifacts:
            verified_gap_inputs.extend(worker.gaps)
            verified_gap_inputs.extend(worker.cross_references)
        allowed_gap_text = {
            _normalize_evidence_text(gap).casefold()
            for gap in verified_gap_inputs
            if _normalize_evidence_text(gap)
        }
        for gap in candidate.gaps:
            normalized_gap = _normalize_evidence_text(gap).casefold()
            if normalized_gap in allowed_gap_text:
                verified_gap_inputs.append(gap)
        gaps: list[str] = []
        seen_gaps: set[str] = set()
        for gap in verified_gap_inputs:
            bounded = (
                gap
                if self._unlimited_research
                else _bounded_text(
                    gap,
                    self._limits.max_artifact_text_chars,
                )
            )
            key = bounded.casefold()
            if not bounded or key in seen_gaps:
                continue
            seen_gaps.add(key)
            gaps.append(bounded)
            if (
                not self._unlimited_research
                and len(gaps) >= self._limits.max_artifact_list_items
            ):
                break

        exact_claim_texts = {
            _normalize_evidence_text(claim.evidence_excerpt).casefold()
            for claim in claims
        }
        conflicts: list[str] = []
        for conflict in candidate.conflicts:
            normalized_conflict = _normalize_evidence_text(conflict).casefold()
            anchored_claims = sum(
                claim_text in normalized_conflict
                for claim_text in exact_claim_texts
                if claim_text
            )
            if anchored_claims < 2:
                continue
            conflicts.append(
                conflict
                if self._unlimited_research
                else _bounded_text(
                    conflict,
                    self._limits.max_artifact_text_chars,
                )
            )
            if (
                not self._unlimited_research
                and len(conflicts) >= self._limits.max_artifact_list_items
            ):
                break

        contributing_worker_ids = list(dict.fromkeys(contributors))
        if not self._unlimited_research:
            contributing_worker_ids = contributing_worker_ids[
                : self._limits.max_artifact_list_items
            ]

        verified = TaskArtifact(
            task_id=spec.task_id,
            status=status,
            answer_fragment=(
                " ".join(claim.claim for claim in claims) if claims else None
            ),
            covered_requirement_ids=covered,
            uncovered_requirement_ids=uncovered,
            claims=claims,
            application_findings=[],
            conflicts=conflicts,
            gaps=gaps,
            contributing_worker_ids=contributing_worker_ids,
        )
        if self._plan is None:
            return verified
        try:
            return validate_task_artifact(verified, plan=self._plan)
        except ArtifactValidationError:
            logger.exception(
                "Evidence artifact failed deterministic reference validation"
            )
            return self._failed_task_artifact(
                spec,
                "Evidence artifact failed reference validation.",
            )

    def _assemble_verified_task_artifact(
        self,
        spec: TaskSpec,
        *,
        worker_artifacts: Sequence[WorkerArtifact],
    ) -> TaskArtifact:
        claims: list[EvidenceClaim] = []
        seen: set[tuple[str, str, str]] = set()
        for artifact in worker_artifacts:
            for claim in artifact.claims:
                key = (claim.document_id, claim.chunk_id, claim.claim.casefold())
                if key not in seen:
                    seen.add(key)
                    claims.append(claim)
                    if (
                        not self._unlimited_research
                        and len(claims) >= self._limits.max_claims_per_task
                    ):
                        break
            if (
                not self._unlimited_research
                and len(claims) >= self._limits.max_claims_per_task
            ):
                break
        supported_evidence_ids = {
            evidence_requirement_id
            for claim in claims
            for evidence_requirement_id in claim.evidence_requirement_ids
        }
        evidence_ids_by_requirement = {
            requirement_id: {
                requirement.evidence_requirement_id
                for requirement in spec.evidence_requirements
                if requirement_id in requirement.requirement_ids
            }
            for requirement_id in spec.requirement_ids
        }
        covered = [
            requirement_id
            for requirement_id in spec.requirement_ids
            if evidence_ids_by_requirement[requirement_id]
            and evidence_ids_by_requirement[requirement_id].issubset(
                supported_evidence_ids
            )
        ]
        uncovered = [
            requirement_id
            for requirement_id in spec.requirement_ids
            if requirement_id not in covered
        ]
        covered_set = set(covered)
        claims = [
            claim.model_copy(
                update={
                    "requirement_ids": [
                        requirement_id
                        for requirement_id in claim.requirement_ids
                        if requirement_id in covered_set
                    ]
                }
            )
            for claim in claims
        ]
        status = (
            TaskStatus.COMPLETE
            if claims and not uncovered
            else TaskStatus.PARTIAL
            if claims
            else TaskStatus.FAILED
        )
        gap_inputs: list[str] = list(uncovered)
        for worker in worker_artifacts:
            gap_inputs.extend(worker.gaps)
            gap_inputs.extend(worker.cross_references)
        gaps: list[str] = []
        seen_gaps: set[str] = set()
        for gap in gap_inputs:
            bounded = (
                gap
                if self._unlimited_research
                else _bounded_text(
                    gap,
                    self._limits.max_artifact_text_chars,
                )
            )
            key = bounded.casefold()
            if not bounded or key in seen_gaps:
                continue
            seen_gaps.add(key)
            gaps.append(bounded)
            if (
                not self._unlimited_research
                and len(gaps) >= self._limits.max_artifact_list_items
            ):
                break
        contributing_worker_ids = list(
            dict.fromkeys(
                artifact.worker_id for artifact in worker_artifacts if artifact.claims
            )
        )
        if not self._unlimited_research:
            contributing_worker_ids = contributing_worker_ids[
                : self._limits.max_artifact_list_items
            ]
        verified = TaskArtifact(
            task_id=spec.task_id,
            status=status,
            answer_fragment=(
                " ".join(claim.claim for claim in claims) if claims else None
            ),
            covered_requirement_ids=covered,
            uncovered_requirement_ids=uncovered,
            claims=claims,
            application_findings=[],
            conflicts=[],
            gaps=gaps,
            contributing_worker_ids=contributing_worker_ids,
        )
        if self._plan is None:
            return verified
        try:
            return validate_task_artifact(verified, plan=self._plan)
        except ArtifactValidationError:
            logger.exception(
                "Fallback evidence artifact failed deterministic validation"
            )
            return self._failed_task_artifact(
                spec,
                "Verified worker claims failed reference validation.",
            )

    def _failed_worker_artifact(
        self,
        *,
        spec: TaskSpec,
        assignment: SearchAssignment,
        worker_id: str,
        error: str,
    ) -> WorkerArtifact:
        return WorkerArtifact(
            task_id=spec.task_id,
            assignment_id=assignment.assignment_id,
            worker_id=worker_id,
            status=WorkerStatus.FAILED,
            searches_run=[assignment.query],
            claims=[],
            gaps=list(assignment.evidence_requirements),
            cross_references=[],
            error_code="worker_execution_failed",
            error_message=_bounded_error(error),
        )

    def _failed_task_artifact(self, spec: TaskSpec, error: str) -> TaskArtifact:
        return TaskArtifact(
            task_id=spec.task_id,
            status=TaskStatus.FAILED,
            answer_fragment=None,
            covered_requirement_ids=[],
            uncovered_requirement_ids=list(spec.requirement_ids),
            claims=[],
            application_findings=[],
            conflicts=[],
            gaps=[_bounded_error(error)],
            contributing_worker_ids=[],
        )

    def _select_final_evidence(
        self, artifacts: Sequence[TaskArtifact]
    ) -> list[EvidenceRecord]:
        by_source = dict(self._evidence_by_source)
        claims_by_id = {
            claim.claim_id: claim for artifact in artifacts for claim in artifact.claims
        }
        selected: list[EvidenceRecord] = []
        seen: set[str] = set()

        # Scenario/application outputs explicitly name the claims they rely
        # on. Reserve the evidence budget for those sources before adding
        # general supporting detail.
        required_claim_ids = list(
            dict.fromkeys(
                claim_id
                for artifact in artifacts
                for finding in artifact.application_findings
                for claim_id in finding.supporting_claim_ids
            )
        )
        for claim_id in required_claim_ids:
            claim = claims_by_id.get(claim_id)
            if claim is None:
                continue
            record = by_source.get((claim.document_id, claim.chunk_id))
            if record is None or record.chunk_id in seen:
                continue
            selected.append(record)
            seen.add(record.chunk_id)
            if (
                not self._unlimited_research
                and len(selected) >= self._limits.final_evidence_limit
            ):
                return selected

        per_task: list[list[EvidenceRecord]] = []
        for artifact in artifacts:
            task_records: list[EvidenceRecord] = []
            task_seen: set[str] = set()
            for claim in artifact.claims:
                record = by_source.get((claim.document_id, claim.chunk_id))
                if record is None or record.chunk_id in task_seen:
                    continue
                task_seen.add(record.chunk_id)
                task_records.append(record)
            per_task.append(task_records)

        max_depth = max((len(records) for records in per_task), default=0)
        for depth in range(max_depth):
            for task_records in per_task:
                if depth >= len(task_records):
                    continue
                record = task_records[depth]
                if record.chunk_id not in seen:
                    selected.append(record)
                    seen.add(record.chunk_id)
                    if (
                        not self._unlimited_research
                        and len(selected) >= self._limits.final_evidence_limit
                    ):
                        return selected
        return selected

    @staticmethod
    def _all_finding_support_rendered(
        artifacts: Sequence[TaskArtifact],
        rendered_evidence: Sequence[EvidenceRecord],
    ) -> bool:
        """Require every claim behind a retained conclusion at final boundary."""

        claims_by_id = {
            claim.claim_id: claim for artifact in artifacts for claim in artifact.claims
        }
        rendered_sources = {
            (record.document_id, record.chunk_id) for record in rendered_evidence
        }
        return all(
            finding.supporting_claim_ids
            and all(
                claim_id in claims_by_id
                and (
                    claims_by_id[claim_id].document_id,
                    claims_by_id[claim_id].chunk_id,
                )
                in rendered_sources
                for claim_id in finding.supporting_claim_ids
            )
            for artifact in artifacts
            for finding in artifact.application_findings
        )

    @staticmethod
    def _plan_summary_payload(
        plan: GlobalPlan,
        *,
        text_limit: int,
        evidence_context_truncated: bool,
    ) -> dict[str, object]:
        scenario = plan.scenario
        return {
            "version": plan.version,
            "problem_type": plan.problem_type.value,
            "mode": plan.mode.value,
            "execution_strategy": plan.execution_strategy.value,
            "normalized_question": _bounded_text(
                plan.normalized_question,
                text_limit,
            ),
            "answer_requirements": [
                {
                    "requirement_id": requirement.requirement_id,
                    "kind": requirement.kind.value,
                    "description": _bounded_text(
                        requirement.description,
                        text_limit,
                    ),
                    "required": requirement.required,
                }
                for requirement in plan.answer_requirements
            ],
            "scenario": (
                {
                    "jurisdiction": (
                        _bounded_text(scenario.jurisdiction, text_limit)
                        if scenario.jurisdiction is not None
                        else None
                    ),
                    "law_as_of_date": scenario.law_as_of_date,
                    "facts": [
                        {
                            "fact_id": fact.fact_id,
                            "description": _bounded_text(
                                fact.description,
                                text_limit,
                            ),
                            "requirement_ids": fact.requirement_ids,
                        }
                        for fact in scenario.facts
                    ],
                    "material_unknowns": [
                        {
                            "unknown_id": unknown.unknown_id,
                            "description": _bounded_text(
                                unknown.description,
                                text_limit,
                            ),
                            "why_material": _bounded_text(
                                unknown.why_material,
                                text_limit,
                            ),
                            "requirement_ids": unknown.requirement_ids,
                        }
                        for unknown in scenario.material_unknowns
                    ],
                    "decision_branches": [
                        {
                            "branch_id": branch.branch_id,
                            "condition": _bounded_text(
                                branch.condition,
                                text_limit,
                            ),
                            "consequence": _bounded_text(
                                branch.consequence,
                                text_limit,
                            ),
                            "requirement_ids": branch.requirement_ids,
                        }
                        for branch in scenario.decision_branches
                    ],
                }
                if scenario is not None
                else None
            ),
            "tasks": [
                {
                    "task_id": task.task_id,
                    "kind": task.kind.value,
                    "issue": _bounded_text(task.issue, text_limit),
                    "requirement_ids": task.requirement_ids,
                    "fact_ids": task.fact_ids,
                    "unknown_ids": task.unknown_ids,
                    "branch_ids": task.branch_ids,
                    "consumes": [
                        reference.model_dump(mode="json") for reference in task.consumes
                    ],
                    "produces": [output.output_id for output in task.produces],
                }
                for task in plan.tasks
            ],
            "synthesis_requirements": [
                _bounded_text(requirement, text_limit)
                for requirement in plan.synthesis_requirements
            ],
            "assumptions": [
                _bounded_text(assumption, text_limit) for assumption in plan.assumptions
            ],
            "evidence_context_truncated": evidence_context_truncated,
        }

    def _render_plan_summary(
        self,
        plan: GlobalPlan,
        *,
        max_chars: int,
        evidence_context_truncated: bool,
    ) -> str:
        """Fit valid plan JSON by reducing prose while retaining typed IDs."""

        for text_limit in (600, 300, 180, 120, 80, 40):
            rendered = json.dumps(
                self._plan_summary_payload(
                    plan,
                    text_limit=text_limit,
                    evidence_context_truncated=evidence_context_truncated,
                ),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if len(rendered) <= max_chars:
                return rendered
        minimal = json.dumps(
            {
                "version": plan.version,
                "problem_type": plan.problem_type.value,
                "mode": plan.mode.value,
                "answer_requirement_ids": [
                    requirement.requirement_id
                    for requirement in plan.answer_requirements
                ],
                "task_ids": [task.task_id for task in plan.tasks],
                "scenario_present": plan.scenario is not None,
                "summary_truncated": True,
                "evidence_context_truncated": evidence_context_truncated,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if len(minimal) <= max_chars:
            return minimal
        return "{}" if max_chars >= 2 else ""

    def _render_final_context(
        self,
        *,
        original_question: str,
        plan: GlobalPlan,
        artifacts: Sequence[TaskArtifact],
        evidence: Sequence[EvidenceRecord],
    ) -> tuple[str, tuple[EvidenceRecord, ...]]:
        if self._unlimited_research:
            rendered_evidence = self._render_evidence_bundle(
                evidence,
                max_chars=None,
            )
            context = (
                f"ORIGINAL QUESTION:\n{original_question}\n\n"
                f"GLOBAL PLAN:\n{plan.model_dump_json()}\n\n"
                "SERVER-VERIFIED FULL EVIDENCE REPORTS "
                "(claims are verbatim source excerpts, "
                "not paraphrases; presentation citations are intentionally "
                "omitted):\n"
                f"{self._task_reports_for_parent(artifacts)}"
            )
            return context, rendered_evidence.records

        headers = (
            "ORIGINAL QUESTION:\n",
            "\n\nGLOBAL PLAN SUMMARY:\n",
            "\n\nTASK ARTIFACTS:\n",
            "\n\nSERVER-VERIFIED FULL EVIDENCE:\n",
        )
        total_limit = self._limits.max_final_context_chars
        available = max(total_limit - sum(len(header) for header in headers), 0)
        question_budget = min(self._limits.max_question_chars, available // 6)
        evidence_budget = min(self._limits.final_chunk_chars, available // 5)
        artifact_budget = min(
            self._limits.max_artifact_context_chars,
            available // 3,
        )
        rendered_evidence = self._render_evidence_bundle(
            evidence,
            max_chars=evidence_budget,
        )
        artifact_context = self._compact_task_artifacts(
            artifacts,
            allowed_sources={
                (record.document_id, record.chunk_id)
                for record in rendered_evidence.records
            },
            max_chars=artifact_budget,
        )
        bounded_question = _bounded_text(original_question, question_budget)
        plan_budget = max(
            available
            - len(bounded_question)
            - len(artifact_context)
            - len(rendered_evidence.text),
            0,
        )
        plan_summary = self._render_plan_summary(
            plan,
            max_chars=plan_budget,
            evidence_context_truncated=(len(rendered_evidence.records) < len(evidence)),
        )
        context = (
            f"{headers[0]}{bounded_question}"
            f"{headers[1]}{plan_summary}"
            f"{headers[2]}{artifact_context}"
            f"{headers[3]}{rendered_evidence.text}"
        )
        return context[:total_limit], rendered_evidence.records

    def _claim_payload(self, claim: EvidenceClaim) -> dict[str, object]:
        payload = claim.model_dump(mode="json")
        payload["claim"] = _bounded_text(
            claim.claim,
            min(self._limits.max_artifact_text_chars, 600),
        )
        payload["evidence_excerpt"] = _bounded_text(
            claim.evidence_excerpt,
            min(self._limits.max_artifact_text_chars, 400),
        )
        payload["readable_title"] = _bounded_text(claim.readable_title, 300)
        payload["locator"] = _bounded_text(claim.locator, 300)
        for field in (
            "requirement_ids",
            "evidence_requirement_ids",
            "fact_ids",
        ):
            value = payload.get(field)
            if isinstance(value, list):
                payload[field] = value[: self._limits.max_artifact_list_items]
        return payload

    def _dump_artifact_payloads(
        self,
        payloads: list[dict[str, object]],
    ) -> str:
        return json.dumps(
            payloads,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def _append_artifact_value(
        self,
        payloads: list[dict[str, object]],
        *,
        payload_index: int,
        field: str,
        value: object,
        max_chars: int | None = None,
    ) -> bool:
        target = payloads[payload_index].get(field)
        if not isinstance(target, list):
            raise TypeError(f"{field} is not an artifact list")
        target_list = cast(list[object], target)
        target_list.append(value)
        context_limit = (
            self._limits.max_artifact_context_chars
            if max_chars is None
            else max(max_chars, 0)
        )
        if len(self._dump_artifact_payloads(payloads)) <= context_limit:
            return True
        target_list.pop()
        return False

    def _set_artifact_value(
        self,
        payloads: list[dict[str, object]],
        *,
        payload_index: int,
        field: str,
        value: object,
        max_chars: int | None = None,
    ) -> bool:
        previous = payloads[payload_index].get(field)
        payloads[payload_index][field] = value
        context_limit = (
            self._limits.max_artifact_context_chars
            if max_chars is None
            else max(max_chars, 0)
        )
        if len(self._dump_artifact_payloads(payloads)) <= context_limit:
            return True
        payloads[payload_index][field] = previous
        return False

    def _worker_reports_for_parent(
        self,
        artifacts: Sequence[WorkerArtifact],
    ) -> str:
        """Pass complete verbatim findings upward without presentation citations."""

        if not self._unlimited_research:
            return self._compact_worker_artifacts(artifacts)
        payloads = [
            {
                "task_id": artifact.task_id,
                "assignment_id": artifact.assignment_id,
                "worker_id": artifact.worker_id,
                "status": artifact.status.value,
                "searches_run": list(artifact.searches_run),
                "exact_evidence": [
                    {
                        "claim_id": claim.claim_id,
                        "document_id": claim.document_id,
                        "chunk_id": claim.chunk_id,
                        "text": claim.evidence_excerpt,
                        "requirement_ids": list(claim.requirement_ids),
                        "evidence_requirement_ids": list(
                            claim.evidence_requirement_ids
                        ),
                        "fact_ids": list(claim.fact_ids),
                        "effective_start_date": claim.effective_start_date,
                        "effective_end_date": claim.effective_end_date,
                    }
                    for claim in artifact.claims
                ],
                "gaps": list(artifact.gaps),
                "cross_references": list(artifact.cross_references),
                "error_code": artifact.error_code,
                "error_message": artifact.error_message,
            }
            for artifact in artifacts
        ]
        return json.dumps(payloads, ensure_ascii=False, separators=(",", ":"))

    def _task_reports_for_parent(
        self,
        artifacts: Sequence[TaskArtifact],
    ) -> str:
        """Pass complete task reports upward without source-title citations."""

        if not self._unlimited_research:
            return self._compact_task_artifacts(artifacts)
        payloads = [
            {
                "task_id": artifact.task_id,
                "status": artifact.status.value,
                "answer_fragment": artifact.answer_fragment,
                "covered_requirement_ids": list(artifact.covered_requirement_ids),
                "uncovered_requirement_ids": list(artifact.uncovered_requirement_ids),
                "exact_evidence": [
                    {
                        "claim_id": claim.claim_id,
                        "text": claim.evidence_excerpt,
                        "requirement_ids": list(claim.requirement_ids),
                        "evidence_requirement_ids": list(
                            claim.evidence_requirement_ids
                        ),
                        "fact_ids": list(claim.fact_ids),
                        "effective_start_date": claim.effective_start_date,
                        "effective_end_date": claim.effective_end_date,
                    }
                    for claim in artifact.claims
                ],
                "application_findings": [
                    finding.model_dump(mode="json")
                    for finding in artifact.application_findings
                ],
                "conflicts": list(artifact.conflicts),
                "gaps": list(artifact.gaps),
                "contributing_worker_ids": list(artifact.contributing_worker_ids),
            }
            for artifact in artifacts
        ]
        return json.dumps(payloads, ensure_ascii=False, separators=(",", ":"))

    def _compact_worker_artifacts(
        self,
        artifacts: Sequence[WorkerArtifact],
    ) -> str:
        selected = list(artifacts[: self._limits.max_total_workers])
        payloads: list[dict[str, object]] = [
            {
                "task_id": artifact.task_id,
                "assignment_id": artifact.assignment_id,
                "worker_id": artifact.worker_id,
                "status": artifact.status.value,
                "searches_run": [],
                "claims": [],
                "gaps": [],
                "cross_references": [],
                "error_code": artifact.error_code,
                "error_message": (
                    _bounded_error(artifact.error_message)
                    if artifact.error_message
                    else None
                ),
            }
            for artifact in selected
        ]

        # Preserve negative evidence and unresolved references before claims:
        # silently losing a limitation is more dangerous than omitting one
        # additional supported detail from a bounded downstream context.
        for field, extractor in (
            ("gaps", lambda item: item.gaps),
            ("cross_references", lambda item: item.cross_references),
        ):
            max_items = max(
                (len(extractor(artifact)) for artifact in selected),
                default=0,
            )
            for depth in range(min(max_items, self._limits.max_artifact_list_items)):
                for index, artifact in enumerate(selected):
                    values = extractor(artifact)
                    if depth >= len(values):
                        continue
                    self._append_artifact_value(
                        payloads,
                        payload_index=index,
                        field=field,
                        value=_bounded_text(values[depth], 500),
                    )

        claim_count = 0
        max_depth = max((len(artifact.claims) for artifact in selected), default=0)
        for depth in range(max_depth):
            for index, artifact in enumerate(selected):
                if (
                    depth >= len(artifact.claims)
                    or claim_count >= self._limits.max_claims_per_task
                ):
                    continue
                if self._append_artifact_value(
                    payloads,
                    payload_index=index,
                    field="claims",
                    value=self._claim_payload(artifact.claims[depth]),
                ):
                    claim_count += 1

        for field, extractor, text_limit in (
            ("searches_run", lambda item: item.searches_run, 500),
        ):
            max_items = max(
                (len(extractor(artifact)) for artifact in selected),
                default=0,
            )
            for depth in range(min(max_items, self._limits.max_artifact_list_items)):
                for index, artifact in enumerate(selected):
                    values = extractor(artifact)
                    if depth >= len(values):
                        continue
                    self._append_artifact_value(
                        payloads,
                        payload_index=index,
                        field=field,
                        value=_bounded_text(values[depth], text_limit),
                    )

        rendered = self._dump_artifact_payloads(payloads)
        if len(rendered) > self._limits.max_artifact_context_chars:
            return "[]"
        return rendered

    def _compact_task_artifacts(
        self,
        artifacts: Sequence[TaskArtifact],
        *,
        allowed_sources: set[tuple[str, str]] | None = None,
        max_chars: int | None = None,
    ) -> str:
        context_limit = (
            self._limits.max_artifact_context_chars
            if max_chars is None
            else max(max_chars, 0)
        )
        selected = list(artifacts[: self._limits.max_tasks])
        specs = (
            {task.task_id: task for task in self._plan.tasks}
            if self._plan is not None
            else {}
        )
        claims_by_task = {
            artifact.task_id: [
                claim
                for claim in artifact.claims
                if allowed_sources is None
                or (claim.document_id, claim.chunk_id) in allowed_sources
            ]
            for artifact in selected
        }
        allowed_claim_ids = {
            claim.claim_id for claims in claims_by_task.values() for claim in claims
        }
        findings_by_task: dict[str, list[DerivedConclusion]] = {}
        for artifact in selected:
            findings: list[DerivedConclusion] = []
            for finding in artifact.application_findings:
                supporting_claim_ids = (
                    list(finding.supporting_claim_ids)
                    if allowed_sources is None
                    else [
                        claim_id
                        for claim_id in finding.supporting_claim_ids
                        if claim_id in allowed_claim_ids
                    ]
                )
                if allowed_sources is not None and (
                    len(supporting_claim_ids) != len(finding.supporting_claim_ids)
                    or not supporting_claim_ids
                ):
                    continue
                findings.append(
                    finding.model_copy(
                        update={
                            "supporting_claim_ids": supporting_claim_ids,
                            "finding": _bounded_text(
                                finding.finding,
                                self._limits.max_artifact_text_chars,
                            ),
                            "limitations": [
                                _bounded_text(value, 500)
                                for value in finding.limitations[
                                    : self._limits.max_artifact_list_items
                                ]
                            ],
                        }
                    )
                )
            findings_by_task[artifact.task_id] = findings

        all_citations = {
            _normalize_evidence_text(
                f"[{claim.readable_title}, {claim.locator}]"
            ).casefold()
            for claims in claims_by_task.values()
            for claim in claims
        }
        conflicts_by_task: dict[str, list[str]] = {}
        for artifact in selected:
            if allowed_sources is None:
                conflicts_by_task[artifact.task_id] = artifact.conflicts
            else:
                conflicts_by_task[artifact.task_id] = [
                    conflict
                    for conflict in artifact.conflicts
                    if sum(
                        citation in _normalize_evidence_text(conflict).casefold()
                        for citation in all_citations
                    )
                    >= 2
                ]

        payloads: list[dict[str, object]] = []
        for artifact in selected:
            spec = specs.get(artifact.task_id)
            requirement_ids = (
                list(spec.requirement_ids)
                if spec is not None
                else list(
                    dict.fromkeys(
                        [
                            *artifact.covered_requirement_ids,
                            *artifact.uncovered_requirement_ids,
                        ]
                    )
                )
            )
            payloads.append(
                {
                    "task_id": artifact.task_id,
                    "kind": spec.kind.value if spec is not None else None,
                    "status": TaskStatus.FAILED.value,
                    "answer_fragment": None,
                    "covered_requirement_ids": [],
                    # Reserve room for limitations first. IDs are short and
                    # deterministic, so retaining all is safer than silently
                    # presenting a truncated artifact as complete.
                    "uncovered_requirement_ids": requirement_ids,
                    "claims": [],
                    "application_findings": [],
                    "conflicts": [],
                    "gaps": [],
                    "contributing_worker_ids": [],
                }
            )

        gaps_by_task = {
            artifact.task_id: list(
                dict.fromkeys(
                    [
                        *artifact.gaps,
                        *(
                            ["Verified claims were omitted by the context budget."]
                            if len(claims_by_task[artifact.task_id])
                            < len(artifact.claims)
                            else []
                        ),
                        *(
                            [
                                "Grounded application findings were omitted "
                                "because their evidence was outside the context budget."
                            ]
                            if len(findings_by_task[artifact.task_id])
                            < len(artifact.application_findings)
                            else []
                        ),
                    ]
                )
            )
            for artifact in selected
        }

        # Preserve limitations before supported detail.
        for field, extractor, text_limit in (
            ("conflicts", lambda item: conflicts_by_task[item.task_id], 600),
            ("gaps", lambda item: gaps_by_task[item.task_id], 500),
        ):
            max_items = max(
                (len(extractor(artifact)) for artifact in selected),
                default=0,
            )
            for depth in range(min(max_items, self._limits.max_artifact_list_items)):
                for index, artifact in enumerate(selected):
                    values = extractor(artifact)
                    if depth < len(values):
                        self._append_artifact_value(
                            payloads,
                            payload_index=index,
                            field=field,
                            value=_bounded_text(values[depth], text_limit),
                            max_chars=context_limit,
                        )

        included_findings: dict[str, list[DerivedConclusion]] = {
            artifact.task_id: [] for artifact in selected
        }
        max_finding_depth = max(
            (len(findings_by_task[artifact.task_id]) for artifact in selected),
            default=0,
        )
        finding_count = 0
        for depth in range(max_finding_depth):
            for index, artifact in enumerate(selected):
                findings = findings_by_task[artifact.task_id]
                if (
                    depth >= len(findings)
                    or finding_count >= self._limits.max_artifact_list_items
                ):
                    continue
                finding = findings[depth]
                if self._append_artifact_value(
                    payloads,
                    payload_index=index,
                    field="application_findings",
                    value=finding.model_dump(mode="json"),
                    max_chars=context_limit,
                ):
                    included_findings[artifact.task_id].append(finding)
                    finding_count += 1

        included_claims: dict[str, list[EvidenceClaim]] = {
            artifact.task_id: [] for artifact in selected
        }
        max_claim_depth = max(
            (len(claims_by_task[artifact.task_id]) for artifact in selected),
            default=0,
        )
        claim_count = 0
        for depth in range(max_claim_depth):
            for index, artifact in enumerate(selected):
                claims = claims_by_task[artifact.task_id]
                if (
                    depth >= len(claims)
                    or claim_count >= self._limits.max_claims_per_task
                ):
                    continue
                claim = claims[depth]
                if self._append_artifact_value(
                    payloads,
                    payload_index=index,
                    field="claims",
                    value=self._claim_payload(claim),
                    max_chars=context_limit,
                ):
                    included_claims[artifact.task_id].append(claim)
                    claim_count += 1

        for index, artifact in enumerate(selected):
            spec = specs.get(artifact.task_id)
            requirement_ids = (
                list(spec.requirement_ids)
                if spec is not None
                else list(
                    dict.fromkeys(
                        [
                            *artifact.covered_requirement_ids,
                            *artifact.uncovered_requirement_ids,
                        ]
                    )
                )
            )
            supported_ids = {
                requirement_id
                for claim in included_claims[artifact.task_id]
                for requirement_id in claim.requirement_ids
            } | {
                requirement_id
                for finding in included_findings[artifact.task_id]
                for requirement_id in finding.requirement_ids
            }
            covered = [
                requirement_id
                for requirement_id in requirement_ids
                if requirement_id in supported_ids
            ]
            uncovered = [
                requirement_id
                for requirement_id in requirement_ids
                if requirement_id not in supported_ids
            ]
            status = (
                TaskStatus.COMPLETE
                if supported_ids and not uncovered
                else TaskStatus.PARTIAL
                if supported_ids
                else TaskStatus.FAILED
            )
            payloads[index]["status"] = status.value
            payloads[index]["covered_requirement_ids"] = covered
            payloads[index]["uncovered_requirement_ids"] = uncovered
            fragments = [
                finding.finding for finding in included_findings[artifact.task_id]
            ] or [claim.claim for claim in included_claims[artifact.task_id]]
            if fragments:
                self._set_artifact_value(
                    payloads,
                    payload_index=index,
                    field="answer_fragment",
                    value=_bounded_text(
                        " ".join(fragments),
                        min(self._limits.max_artifact_text_chars, 800),
                    ),
                    max_chars=context_limit,
                )
            for worker_id in artifact.contributing_worker_ids[
                : self._limits.max_artifact_list_items
            ]:
                self._append_artifact_value(
                    payloads,
                    payload_index=index,
                    field="contributing_worker_ids",
                    value=_bounded_text(worker_id, 120),
                    max_chars=context_limit,
                )

        rendered = self._dump_artifact_payloads(payloads)
        return (
            rendered
            if len(rendered) <= context_limit
            else "[]"
            if context_limit >= 2
            else ""
        )

    @staticmethod
    def _render_evidence_bundle(
        records: Sequence[EvidenceRecord],
        *,
        max_chars: int | None,
    ) -> _RenderedEvidence:
        """Share one total budget fairly across the records that actually fit."""

        if not records:
            return _RenderedEvidence("No indexed evidence was found.", ())

        prefixes: list[str] = []
        suffix = "\nEND_UNTRUSTED_SOURCE_TEXT"
        for record in records:
            prefix = "\n".join(
                [
                    f"EVIDENCE_ID: {record.evidence_id}",
                    f"document_id: {record.document_id}",
                    f"chunk_id: {record.chunk_id}",
                    f"title: {record.readable_title}",
                    f"locator: {record.locator}",
                    f"relevance: {record.score:.4f}",
                    "BEGIN_UNTRUSTED_SOURCE_TEXT",
                    "",
                ]
            )
            prefixes.append(prefix)

        if max_chars is None:
            blocks = [
                prefix + record.text + suffix
                for prefix, record in zip(prefixes, records)
            ]
            return _RenderedEvidence("\n\n".join(blocks), tuple(records))

        # Include as many ordered sources as possible while reserving a useful
        # excerpt for each. The input order is already task/source diversified.
        minimum_excerpt = 64
        included_count = 0
        overhead = 0
        reserved_text = 0
        for index, prefix in enumerate(prefixes):
            separator_chars = 2 if index else 0
            required = separator_chars + len(prefix) + len(suffix)
            text_reserve = min(minimum_excerpt, len(records[index].text))
            if overhead + required + reserved_text + text_reserve > max_chars:
                break
            overhead += required
            reserved_text += text_reserve
            included_count += 1

        if included_count == 0:
            message = "Indexed evidence existed but exceeded the context budget."
            return _RenderedEvidence(message[:max_chars], ())

        included = list(records[:included_count])
        remaining_text = max(max_chars - overhead, 0)
        allocations = [0] * included_count
        active = {index for index, record in enumerate(included) if record.text}
        # Water-fill the excerpt budget so a long first record cannot crowd
        # later tasks or sources out of the reviewer/final context.
        while remaining_text > 0 and active:
            share = max(remaining_text // len(active), 1)
            progressed = False
            for index in list(active):
                capacity = len(included[index].text) - allocations[index]
                if capacity <= 0:
                    active.discard(index)
                    continue
                granted = min(capacity, share, remaining_text)
                allocations[index] += granted
                remaining_text -= granted
                progressed = progressed or granted > 0
                if allocations[index] >= len(included[index].text):
                    active.discard(index)
                if remaining_text <= 0:
                    break
            if not progressed:
                break

        blocks = [
            prefixes[index] + record.text[: allocations[index]] + suffix
            for index, record in enumerate(included)
        ]
        rendered = "\n\n".join(blocks)
        return _RenderedEvidence(rendered[:max_chars], tuple(included))

    @staticmethod
    def _render_evidence(
        records: Sequence[EvidenceRecord],
        *,
        max_chars: int,
    ) -> str:
        """Compatibility wrapper used by focused rendering tests."""

        return MultiAgentResearchOrchestrator._render_evidence_bundle(
            records,
            max_chars=max_chars,
        ).text
