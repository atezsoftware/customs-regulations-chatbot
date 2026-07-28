"""Bounded, context-isolated multi-agent research orchestration.

The legacy indexed workflow performs one LLM retrieval plan followed by
deterministic parallel searches.  This module adds a higher-level planner and
task DAG while deliberately reusing that deterministic retrieval seam.  Every
LLM role receives a fresh one-turn history and returns a typed artifact; chat
transcripts and hidden reasoning are never passed between roles.
"""

from __future__ import annotations

import asyncio
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
    EvidenceClaim,
    ExecutionStrategy,
    GlobalPlan,
    PlanMode,
    SearchAssignment,
    SearchAssignmentBatch,
    TaskArtifact,
    TaskSpec,
    TaskStatus,
    WorkerArtifact,
    WorkerStatus,
    resolve_global_plan,
)
from .orchestration_prompts import (
    EVIDENCE_WORKER_SYSTEM_PROMPT,
    TASK_REVIEW_SYSTEM_PROMPT,
    build_global_planner_prompt,
    build_task_coordinator_prompt,
)
from .search import IndexedQueryEngine, RankedDocument, SearchHit

logger = logging.getLogger(__name__)

ProgressStatus = Literal["started", "completed", "failed"]
AgentRole = Literal["planner", "task", "worker", "final"]


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
    """Server-enforced fan-out and context budgets for one research run."""

    max_tasks: int = 5
    max_parallel_tasks: int = 3
    max_assignments_per_wave: int = 3
    max_worker_rounds: int = 2
    max_total_workers: int = 8
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
    max_query_chars: int = 1_000
    max_search_attempts: int = 2
    search_timeout_seconds: float = 20.0
    llm_timeout_seconds: float = 120.0
    max_total_llm_calls: int = 24

    @classmethod
    def from_env(cls) -> "ResearchLimits":
        """Load positive integer limits, failing fast on unsafe config."""

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
            max_tasks=positive("FS_EXPLORER_MULTI_AGENT_MAX_TASKS", defaults.max_tasks),
            max_parallel_tasks=positive(
                "FS_EXPLORER_MULTI_AGENT_MAX_PARALLEL_TASKS",
                defaults.max_parallel_tasks,
            ),
            max_assignments_per_wave=positive(
                "FS_EXPLORER_MULTI_AGENT_MAX_WORKERS_PER_TASK",
                defaults.max_assignments_per_wave,
            ),
            max_worker_rounds=positive(
                "FS_EXPLORER_MULTI_AGENT_MAX_WORKER_ROUNDS",
                defaults.max_worker_rounds,
            ),
            max_total_workers=positive(
                "FS_EXPLORER_MULTI_AGENT_MAX_TOTAL_WORKERS",
                defaults.max_total_workers,
            ),
            max_parallel_llm_calls=positive(
                "FS_EXPLORER_MULTI_AGENT_LLM_CONCURRENCY",
                defaults.max_parallel_llm_calls,
            ),
            max_parallel_retrievals=positive(
                "FS_EXPLORER_MULTI_AGENT_RETRIEVAL_CONCURRENCY",
                defaults.max_parallel_retrievals,
            ),
            worker_hit_limit=positive(
                "FS_EXPLORER_MULTI_AGENT_WORKER_HIT_LIMIT",
                defaults.worker_hit_limit,
            ),
            worker_hit_chars=positive(
                "FS_EXPLORER_MULTI_AGENT_WORKER_HIT_CHARS",
                defaults.worker_hit_chars,
            ),
            review_hit_chars=positive(
                "FS_EXPLORER_MULTI_AGENT_REVIEW_HIT_CHARS",
                defaults.review_hit_chars,
            ),
            final_evidence_limit=positive(
                "FS_EXPLORER_MULTI_AGENT_FINAL_EVIDENCE_LIMIT",
                defaults.final_evidence_limit,
            ),
            final_chunk_chars=positive(
                "FS_EXPLORER_MULTI_AGENT_FINAL_CHUNK_CHARS",
                defaults.final_chunk_chars,
            ),
            max_claims_per_worker=positive(
                "FS_EXPLORER_MULTI_AGENT_MAX_CLAIMS_PER_WORKER",
                defaults.max_claims_per_worker,
            ),
            max_claims_per_task=positive(
                "FS_EXPLORER_MULTI_AGENT_MAX_CLAIMS_PER_TASK",
                defaults.max_claims_per_task,
            ),
            max_artifact_list_items=positive(
                "FS_EXPLORER_MULTI_AGENT_MAX_ARTIFACT_ITEMS",
                defaults.max_artifact_list_items,
            ),
            max_artifact_text_chars=positive(
                "FS_EXPLORER_MULTI_AGENT_MAX_ARTIFACT_TEXT_CHARS",
                defaults.max_artifact_text_chars,
            ),
            max_artifact_context_chars=positive(
                "FS_EXPLORER_MULTI_AGENT_MAX_ARTIFACT_CONTEXT_CHARS",
                defaults.max_artifact_context_chars,
            ),
            max_query_chars=positive(
                "FS_EXPLORER_MULTI_AGENT_MAX_QUERY_CHARS",
                defaults.max_query_chars,
            ),
            max_search_attempts=positive(
                "FS_EXPLORER_MULTI_AGENT_MAX_SEARCH_ATTEMPTS",
                defaults.max_search_attempts,
            ),
            search_timeout_seconds=positive_float(
                "FS_EXPLORER_MULTI_AGENT_SEARCH_TIMEOUT_SECONDS",
                defaults.search_timeout_seconds,
            ),
            llm_timeout_seconds=positive_float(
                "FS_EXPLORER_MULTI_AGENT_LLM_TIMEOUT_SECONDS",
                defaults.llm_timeout_seconds,
            ),
            max_total_llm_calls=positive(
                "FS_EXPLORER_MULTI_AGENT_MAX_LLM_CALLS",
                defaults.max_total_llm_calls,
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
    used_plan_fallback: bool


def _normalized_query(value: str) -> str:
    return " ".join(value.casefold().split())


def _safe_identifier(value: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.:-]+", "_", value.strip())
    normalized = normalized.strip("._:-")[:80]
    return normalized or fallback


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
    """Planner → isolated task DAG → bounded workers → artifact synthesis."""

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
        self._used_plan_fallback = False
        self._task_artifacts: dict[str, TaskArtifact] = {}
        self._worker_artifacts: dict[str, dict[str, WorkerArtifact]] = {}
        self._evidence: dict[str, EvidenceRecord] = {}
        self._evidence_by_source: dict[tuple[str, str], EvidenceRecord] = {}
        self._task_evidence: dict[str, dict[str, EvidenceRecord]] = {}
        self._searches_by_task: dict[str, set[str]] = {}

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
                if self._llm_call_count >= self._limits.max_total_llm_calls:
                    raise RuntimeError("Multi-agent LLM call budget exhausted.")
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
        """Run or resume a bounded research plan from persisted artifacts."""

        async with self._run_lock:
            if self._plan is None:
                self._plan = await self._create_plan(task)

            required_tasks = sum(spec.required for spec in self._plan.tasks)
            if required_tasks > self._limits.max_total_workers:
                raise RuntimeError(
                    "The worker budget cannot reserve one worker for every "
                    "required research task."
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
            sources = tuple(record.source_payload() for record in rendered_evidence)
            return MultiAgentResearchResult(
                plan=self._plan,
                task_artifacts=ordered_artifacts,
                final_context=final_context,
                evidence_sources=sources,
                incomplete=incomplete,
                used_plan_fallback=self._used_plan_fallback,
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
        candidate: GlobalPlan | None = None
        planner_error: str | None = None
        try:
            raw, _usage = await self._generate_structured(
                client=self._planner_llm,
                role="planner",
                purpose="global_plan",
                prompt=build_global_planner_prompt(max_tasks=self._limits.max_tasks),
                user_text=(
                    f"CURRENT DATE: {date.today().isoformat()}\n\n"
                    "CURRENT QUESTION AND BOUNDED CONVERSATION CONTEXT:\n"
                    f"{task}"
                ),
                operation_id="global-plan",
                schema=GlobalPlan,
                agent_id="global-planner",
            )
            candidate = raw if isinstance(raw, GlobalPlan) else None
        except Exception as exc:
            planner_error = _bounded_error(exc)
            logger.exception(
                "Multi-agent global planner failed; using a direct fallback plan"
            )

        resolution = resolve_global_plan(
            candidate,
            original_question=_current_question(task),
            max_tasks=self._limits.max_tasks,
            planner_error=planner_error,
        )
        plan = self._bounded_plan(resolution.plan)
        self._used_plan_fallback = resolution.used_fallback
        detail = (
            f"{plan.mode.value}/{plan.execution_strategy.value}: "
            f"{len(plan.tasks)} task(s)"
        )
        if resolution.used_fallback:
            detail += " (safe direct fallback)"
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
        """Apply server-owned artifact budgets after semantic validation."""

        tasks = [
            task.model_copy(
                update={
                    "question": _bounded_text(
                        task.question,
                        self._limits.max_artifact_text_chars,
                    ),
                    "purpose": _bounded_text(
                        task.purpose,
                        self._limits.max_artifact_text_chars,
                    ),
                    "expected_output": _bounded_text(
                        task.expected_output,
                        self._limits.max_artifact_text_chars,
                    ),
                    "success_criteria": [
                        _bounded_text(
                            criterion,
                            self._limits.max_artifact_text_chars,
                        )
                        for criterion in task.success_criteria[
                            : self._limits.max_artifact_list_items
                        ]
                    ],
                    "as_of_date": _validated_as_of_date(task.as_of_date),
                    "filters": self._bounded_filter(task.filters),
                }
            )
            for task in plan.tasks
        ]
        return plan.model_copy(
            update={
                "normalized_question": _bounded_text(
                    plan.normalized_question,
                    self._limits.max_artifact_text_chars,
                ),
                "answer_requirements": [
                    _bounded_text(
                        requirement,
                        self._limits.max_artifact_text_chars,
                    )
                    for requirement in plan.answer_requirements[
                        : self._limits.max_artifact_list_items
                    ]
                ],
                "tasks": tasks,
                "synthesis_requirements": [
                    _bounded_text(
                        requirement,
                        self._limits.max_artifact_text_chars,
                    )
                    for requirement in plan.synthesis_requirements[
                        : self._limits.max_artifact_list_items
                    ]
                ],
                "assumptions": [
                    _bounded_text(
                        assumption,
                        self._limits.max_artifact_text_chars,
                    )
                    for assumption in plan.assumptions[
                        : self._limits.max_artifact_list_items
                    ]
                ],
            }
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

        global_remaining = self._limits.max_total_workers - self._worker_count
        if global_remaining <= 0 or self._plan is None:
            return max(global_remaining, 0)
        reserved_for_others = sum(
            1
            for candidate in self._plan.tasks
            if candidate.task_id != task_id
            and candidate.required
            and candidate.task_id not in self._worker_closed_tasks
            and candidate.task_id not in self._task_artifacts
            and self._worker_count_by_task.get(candidate.task_id, 0) == 0
        )
        return max(global_remaining - reserved_for_others, 0)

    async def _run_task(self, spec: TaskSpec) -> TaskArtifact:
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
        single_pass = (
            self._plan is not None
            and self._plan.mode == PlanMode.DIRECT
            and self._plan.execution_strategy == ExecutionStrategy.SINGLE_PASS
            and len(self._plan.tasks) == 1
            and not dependency_artifacts
        )
        single_pass_complete = False

        first_round = self._next_round_by_task.get(spec.task_id, 1)
        for round_index in range(
            first_round,
            self._limits.max_worker_rounds + 1,
        ):
            if self._worker_count >= self._limits.max_total_workers:
                break
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
                try:
                    dispatch = await self._coordinate_task(
                        spec=spec,
                        dependency_artifacts=dependency_artifacts,
                        worker_artifacts=list(worker_by_assignment.values()),
                        searches_run=sorted(searches_run),
                        round_index=round_index,
                    )
                except Exception:
                    logger.exception(
                        "Task coordinator failed for %s; using bounded fallback",
                        spec.task_id,
                    )
                    dispatch = self._fallback_dispatch(
                        spec, searches_run=searches_run, round_index=round_index
                    )

            assignments = self._validated_assignments(
                spec,
                dispatch,
                searches_run=searches_run,
                existing_assignment_ids=set(worker_by_assignment),
            )
            remaining_slots = self._available_worker_slots(spec.task_id)
            assignments = assignments[:remaining_slots]
            if dispatch.stop or not assignments:
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
            self._next_round_by_task[spec.task_id] = round_index + 1
            supported_criteria = {
                criterion
                for worker in worker_by_assignment.values()
                for claim in worker.claims
                for criterion in claim.supports_success_criteria
            }
            coverage_complete = all(
                criterion in supported_criteria for criterion in spec.success_criteria
            )
            if coverage_complete:
                single_pass_complete = single_pass and round_index == 1
                self._worker_closed_tasks.add(spec.task_id)
                self._next_round_by_task[spec.task_id] = (
                    self._limits.max_worker_rounds + 1
                )
                break

        self._worker_closed_tasks.add(spec.task_id)
        if single_pass_complete:
            # IDs, excerpts, criteria and source membership were already
            # verified server-side in `_run_worker`. A second model reviewer
            # cannot add evidence for this deliberately simple route.
            artifact = self._fallback_task_artifact(
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
                f"{len(artifact.uncovered_success_criteria)} gap(s)"
            ),
            task_id=spec.task_id,
            agent_id=f"task-coordinator-{spec.task_id}",
        )
        return artifact

    async def _coordinate_task(
        self,
        *,
        spec: TaskSpec,
        dependency_artifacts: Sequence[TaskArtifact],
        worker_artifacts: Sequence[WorkerArtifact],
        searches_run: Sequence[str],
        round_index: int,
    ) -> SearchAssignmentBatch:
        agent_id = f"task-coordinator-{spec.task_id}"
        raw, _usage = await self._generate_structured(
            client=self._task_llm,
            role="task",
            purpose="task_coordination",
            prompt=build_task_coordinator_prompt(
                max_assignments_per_wave=self._limits.max_assignments_per_wave,
                max_worker_rounds=self._limits.max_worker_rounds,
            ),
            user_text=(
                f"WAVE: {round_index}/{self._limits.max_worker_rounds}\n\n"
                f"TASK SPEC:\n{spec.model_dump_json()}\n\n"
                "DEPENDENCY ARTIFACTS:\n"
                f"{self._compact_task_artifacts(dependency_artifacts)}\n\n"
                "WORKER ARTIFACTS SO FAR:\n"
                f"{self._compact_worker_artifacts(worker_artifacts)}\n\n"
                "SEARCHES ALREADY RUN:\n"
                f"{json.dumps(list(searches_run), ensure_ascii=False)}"
            ),
            operation_id=f"task-coordinate:{spec.task_id}:{round_index}",
            schema=SearchAssignmentBatch,
            task_id=spec.task_id,
            agent_id=agent_id,
        )
        assert isinstance(raw, SearchAssignmentBatch)
        return raw

    def _fallback_dispatch(
        self,
        spec: TaskSpec,
        *,
        searches_run: set[str],
        round_index: int,
    ) -> SearchAssignmentBatch:
        normalized = _normalized_query(spec.question)
        if round_index > 1 or normalized in searches_run:
            return SearchAssignmentBatch(
                task_id=spec.task_id,
                stop=True,
                stop_reason="The safe fallback query has already run.",
                assignments=[],
            )
        return SearchAssignmentBatch(
            task_id=spec.task_id,
            stop=False,
            stop_reason=None,
            assignments=[
                SearchAssignment(
                    assignment_id=f"{spec.task_id}_fallback",
                    task_id=spec.task_id,
                    query=spec.question,
                    objective=spec.expected_output,
                    evidence_requirements=list(spec.success_criteria),
                    excluded_queries=sorted(searches_run),
                    as_of_date=_validated_as_of_date(spec.as_of_date),
                    filters=spec.filters,
                )
            ],
        )

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
                    evidence_requirements=list(spec.success_criteria),
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
    ) -> list[SearchAssignment]:
        if dispatch.task_id != spec.task_id or dispatch.stop:
            return []
        accepted: list[SearchAssignment] = []
        accepted_queries: set[str] = set()
        accepted_ids: set[str] = set()
        for candidate in dispatch.assignments:
            query = _bounded_text(
                candidate.query,
                self._limits.max_query_chars,
            )
            normalized = _normalized_query(query)
            assignment_id = _safe_identifier(
                candidate.assignment_id,
                fallback=f"assignment_{len(accepted) + 1}",
            )
            if (
                candidate.task_id != spec.task_id
                or not query
                or not assignment_id
                or assignment_id in existing_assignment_ids
                or assignment_id in accepted_ids
                or normalized in searches_run
                or normalized in accepted_queries
            ):
                continue
            accepted.append(
                candidate.model_copy(
                    update={
                        "assignment_id": assignment_id,
                        "query": query,
                        "objective": _bounded_text(
                            candidate.objective,
                            self._limits.max_artifact_text_chars,
                        ),
                        "evidence_requirements": [
                            _bounded_text(
                                requirement,
                                self._limits.max_artifact_text_chars,
                            )
                            for requirement in candidate.evidence_requirements[
                                : self._limits.max_artifact_list_items
                            ]
                        ],
                        "excluded_queries": [
                            _bounded_text(
                                excluded,
                                self._limits.max_query_chars,
                            )
                            for excluded in candidate.excluded_queries[
                                : self._limits.max_artifact_list_items
                            ]
                        ],
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
            if len(accepted) >= self._limits.max_assignments_per_wave:
                break
        return accepted

    def _bounded_filter(self, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split()).strip()
        if not normalized or len(normalized) > self._limits.max_query_chars:
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
        async with self._retrieval_semaphore:
            for attempt in range(1, self._limits.max_search_attempts + 1):
                try:
                    result = await self._execute_search(assignment)
                except Exception as exc:
                    search_error = _bounded_error(exc)
                else:
                    search_error = (
                        _bounded_error(result.error) if result.error else None
                    )
                if (
                    attempt >= self._limits.max_search_attempts
                    or not _is_transient_search_error(search_error)
                ):
                    break
                self._forget_completed_search(assignment)

        hits = list(result.hits if result is not None else [])
        hits = hits[: self._limits.worker_hit_limit]
        records = self._register_evidence(spec, assignment, hits)
        rendered_bundle = self._render_evidence_bundle(
            records, max_chars=self._limits.worker_hit_chars
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
                f"{len(exposed_records)} bounded candidate chunk(s)"
                if not search_error
                else (
                    f"{len(exposed_records)} bounded chunk(s); "
                    f"{_bounded_error(search_error)}"
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
                        f"TASK SUCCESS CRITERIA:\n"
                        f"{json.dumps(spec.success_criteria, ensure_ascii=False)}\n\n"
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
        verified: list[EvidenceClaim] = []
        seen_claims: set[tuple[str, str, str]] = set()
        rejected = 0
        for claim in candidate.claims[: self._limits.max_claims_per_worker]:
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
            supported_criteria = [
                criterion
                for criterion in claim.supports_success_criteria
                if criterion in spec.success_criteria
            ]
            verified.append(
                claim.model_copy(
                    update={
                        "document_id": record.document_id,
                        "chunk_id": record.chunk_id,
                        "readable_title": record.readable_title,
                        "locator": record.locator,
                        "claim": _bounded_text(
                            claim.claim,
                            self._limits.max_artifact_text_chars,
                        ),
                        "evidence_excerpt": _bounded_text(
                            claim.evidence_excerpt,
                            self._limits.max_artifact_text_chars,
                        ),
                        "supports_success_criteria": supported_criteria,
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
            criterion
            for claim in verified
            for criterion in claim.supports_success_criteria
        }
        gaps = [
            _bounded_text(
                requirement,
                self._limits.max_artifact_text_chars,
            )
            for requirement in assignment.evidence_requirements
            if requirement not in supported
        ][: self._limits.max_artifact_list_items]
        if rejected:
            gaps.append(
                f"{rejected} unsupported claim(s) were rejected by source validation."
            )
        if search_error:
            gaps.append(f"Search warning: {search_error}")
        status = WorkerStatus.SUCCESS if verified else WorkerStatus.NO_EVIDENCE
        normalized_evidence = " ".join(record.text.casefold() for record in records)
        cross_references = [
            _bounded_text(
                reference,
                self._limits.max_artifact_text_chars,
            )
            for reference in candidate.cross_references
            if _normalize_evidence_text(reference).casefold()
            in _normalize_evidence_text(normalized_evidence)
        ][: self._limits.max_artifact_list_items]
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
        review_evidence = self._render_evidence_bundle(
            task_records,
            max_chars=self._limits.review_hit_chars,
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
            return self._fallback_task_artifact(
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
                    f"{self._compact_task_artifacts(dependency_artifacts)}\n\n"
                    "WORKER ARTIFACTS:\n"
                    f"{self._compact_worker_artifacts(worker_artifacts)}\n\n"
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
            logger.exception(
                "Task reviewer failed for %s; using verified worker artifacts",
                spec.task_id,
            )
            return self._fallback_task_artifact(spec, worker_artifacts=worker_artifacts)

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
        for claim in candidate.claims[: self._limits.max_claims_per_task]:
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
                if len(claims) >= self._limits.max_claims_per_task:
                    break

        supported = {
            criterion
            for claim in claims
            for criterion in claim.supports_success_criteria
        }
        covered = [
            criterion for criterion in spec.success_criteria if criterion in supported
        ]
        uncovered = [
            criterion
            for criterion in spec.success_criteria
            if criterion not in supported
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
            bounded = _bounded_text(
                gap,
                self._limits.max_artifact_text_chars,
            )
            key = bounded.casefold()
            if not bounded or key in seen_gaps:
                continue
            seen_gaps.add(key)
            gaps.append(bounded)
            if len(gaps) >= self._limits.max_artifact_list_items:
                break

        source_citations = {
            _normalize_evidence_text(
                f"[{claim.readable_title}, {claim.locator}]"
            ).casefold()
            for claim in claims
        }
        conflicts: list[str] = []
        for conflict in candidate.conflicts:
            normalized_conflict = _normalize_evidence_text(conflict).casefold()
            anchored_sources = sum(
                citation in normalized_conflict for citation in source_citations
            )
            if anchored_sources < 2:
                continue
            conflicts.append(
                _bounded_text(
                    conflict,
                    self._limits.max_artifact_text_chars,
                )
            )
            if len(conflicts) >= self._limits.max_artifact_list_items:
                break

        return TaskArtifact(
            task_id=spec.task_id,
            status=status,
            answer_fragment=(
                " ".join(claim.claim for claim in claims) if claims else None
            ),
            covered_success_criteria=covered,
            uncovered_success_criteria=uncovered,
            claims=claims,
            conflicts=conflicts,
            gaps=gaps,
            contributing_worker_ids=list(dict.fromkeys(contributors))[
                : self._limits.max_artifact_list_items
            ],
        )

    def _fallback_task_artifact(
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
                    if len(claims) >= self._limits.max_claims_per_task:
                        break
            if len(claims) >= self._limits.max_claims_per_task:
                break
        supported = {
            criterion
            for claim in claims
            for criterion in claim.supports_success_criteria
            if criterion in spec.success_criteria
        }
        covered = [
            criterion for criterion in spec.success_criteria if criterion in supported
        ]
        uncovered = [
            criterion
            for criterion in spec.success_criteria
            if criterion not in supported
        ]
        status = (
            TaskStatus.COMPLETE
            if claims and not uncovered
            else TaskStatus.PARTIAL
            if claims
            else TaskStatus.FAILED
        )
        fallback_gap_inputs: list[str] = list(uncovered)
        for worker in worker_artifacts:
            fallback_gap_inputs.extend(worker.gaps)
            fallback_gap_inputs.extend(worker.cross_references)
        gaps: list[str] = []
        seen_gaps: set[str] = set()
        for gap in fallback_gap_inputs:
            bounded = _bounded_text(
                gap,
                self._limits.max_artifact_text_chars,
            )
            key = bounded.casefold()
            if not bounded or key in seen_gaps:
                continue
            seen_gaps.add(key)
            gaps.append(bounded)
            if len(gaps) >= self._limits.max_artifact_list_items:
                break
        return TaskArtifact(
            task_id=spec.task_id,
            status=status,
            answer_fragment=(
                " ".join(claim.claim for claim in claims) if claims else None
            ),
            covered_success_criteria=covered,
            uncovered_success_criteria=uncovered,
            claims=claims,
            conflicts=[],
            gaps=gaps,
            contributing_worker_ids=list(
                dict.fromkeys(
                    artifact.worker_id
                    for artifact in worker_artifacts
                    if artifact.claims
                )
            )[: self._limits.max_artifact_list_items],
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
            covered_success_criteria=[],
            uncovered_success_criteria=list(spec.success_criteria),
            claims=[],
            conflicts=[],
            gaps=[_bounded_error(error)],
            contributing_worker_ids=[],
        )

    def _select_final_evidence(
        self, artifacts: Sequence[TaskArtifact]
    ) -> list[EvidenceRecord]:
        by_source = dict(self._evidence_by_source)
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

        selected: list[EvidenceRecord] = []
        seen: set[str] = set()
        max_depth = max((len(records) for records in per_task), default=0)
        for depth in range(max_depth):
            for task_records in per_task:
                if depth >= len(task_records):
                    continue
                record = task_records[depth]
                if record.chunk_id not in seen:
                    selected.append(record)
                    seen.add(record.chunk_id)
                    if len(selected) >= self._limits.final_evidence_limit:
                        return selected
        return selected

    def _render_final_context(
        self,
        *,
        original_question: str,
        plan: GlobalPlan,
        artifacts: Sequence[TaskArtifact],
        evidence: Sequence[EvidenceRecord],
    ) -> tuple[str, tuple[EvidenceRecord, ...]]:
        rendered_evidence = self._render_evidence_bundle(
            evidence,
            max_chars=self._limits.final_chunk_chars,
        )
        plan_summary = {
            "mode": plan.mode.value,
            "normalized_question": plan.normalized_question,
            "answer_requirements": plan.answer_requirements,
            "synthesis_requirements": plan.synthesis_requirements,
            "assumptions": plan.assumptions,
            "evidence_context_truncated": (
                len(rendered_evidence.records) < len(evidence)
            ),
        }
        artifact_context = self._compact_task_artifacts(
            artifacts,
            allowed_sources={
                (record.document_id, record.chunk_id)
                for record in rendered_evidence.records
            },
        )
        context = (
            f"ORIGINAL QUESTION:\n{original_question}\n\n"
            "GLOBAL PLAN SUMMARY:\n"
            f"{json.dumps(plan_summary, ensure_ascii=False)}\n\n"
            "TASK ARTIFACTS:\n"
            f"{artifact_context}\n\n"
            "SERVER-VERIFIED FULL EVIDENCE:\n"
            f"{rendered_evidence.text}"
        )
        return context, rendered_evidence.records

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
        payload["supports_success_criteria"] = [
            _bounded_text(criterion, 300)
            for criterion in claim.supports_success_criteria[
                : min(self._limits.max_artifact_list_items, 6)
            ]
        ]
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
    ) -> bool:
        target = payloads[payload_index].get(field)
        if not isinstance(target, list):
            raise TypeError(f"{field} is not an artifact list")
        target_list = cast(list[object], target)
        target_list.append(value)
        if (
            len(self._dump_artifact_payloads(payloads))
            <= self._limits.max_artifact_context_chars
        ):
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
    ) -> bool:
        previous = payloads[payload_index].get(field)
        payloads[payload_index][field] = value
        if (
            len(self._dump_artifact_payloads(payloads))
            <= self._limits.max_artifact_context_chars
        ):
            return True
        payloads[payload_index][field] = previous
        return False

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
    ) -> str:
        selected = list(artifacts[: self._limits.max_tasks])
        claims_by_task = {
            artifact.task_id: [
                claim
                for claim in artifact.claims
                if allowed_sources is None
                or (claim.document_id, claim.chunk_id) in allowed_sources
            ]
            for artifact in selected
        }
        supported_by_task = {
            artifact.task_id: {
                criterion
                for claim in claims_by_task[artifact.task_id]
                for criterion in claim.supports_success_criteria
            }
            for artifact in selected
        }
        covered_by_task = {
            artifact.task_id: (
                artifact.covered_success_criteria
                if allowed_sources is None
                else [
                    criterion
                    for criterion in artifact.covered_success_criteria
                    if criterion in supported_by_task[artifact.task_id]
                ]
            )
            for artifact in selected
        }
        uncovered_by_task = {
            artifact.task_id: (
                artifact.uncovered_success_criteria
                if allowed_sources is None
                else list(
                    dict.fromkeys(
                        [
                            *artifact.uncovered_success_criteria,
                            *(
                                criterion
                                for criterion in artifact.covered_success_criteria
                                if criterion not in supported_by_task[artifact.task_id]
                            ),
                        ]
                    )
                )
            )
            for artifact in selected
        }
        conflicts_by_task: dict[str, list[str]] = {}
        for artifact in selected:
            if allowed_sources is None:
                conflicts_by_task[artifact.task_id] = artifact.conflicts
                continue
            citations = {
                _normalize_evidence_text(
                    f"[{claim.readable_title}, {claim.locator}]"
                ).casefold()
                for claim in claims_by_task[artifact.task_id]
            }
            conflicts_by_task[artifact.task_id] = [
                conflict
                for conflict in artifact.conflicts
                if sum(
                    citation in _normalize_evidence_text(conflict).casefold()
                    for citation in citations
                )
                >= 2
            ]
        gaps_by_task = {
            artifact.task_id: [
                *artifact.gaps,
                *(
                    ["Verified claims were omitted by the final context budget."]
                    if allowed_sources is not None
                    and len(claims_by_task[artifact.task_id]) < len(artifact.claims)
                    else []
                ),
            ]
            for artifact in selected
        }
        payloads: list[dict[str, object]] = [
            {
                "task_id": artifact.task_id,
                "status": (
                    artifact.status.value
                    if allowed_sources is None
                    or len(claims_by_task[artifact.task_id]) == len(artifact.claims)
                    else (
                        TaskStatus.PARTIAL.value
                        if claims_by_task[artifact.task_id]
                        else TaskStatus.FAILED.value
                    )
                ),
                "answer_fragment": None,
                "covered_success_criteria": [],
                "uncovered_success_criteria": [],
                "claims": [],
                "conflicts": [],
                "gaps": [],
                "contributing_worker_ids": [],
            }
            for artifact in selected
        ]

        # Limitations are first-class final-synthesis input. Reserve the
        # shared budget for them before adding supported detail.
        for field, extractor, text_limit in (
            (
                "uncovered_success_criteria",
                lambda item: uncovered_by_task[item.task_id],
                400,
            ),
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
                    if depth >= len(values):
                        continue
                    self._append_artifact_value(
                        payloads,
                        payload_index=index,
                        field=field,
                        value=_bounded_text(values[depth], text_limit),
                    )

        claim_count = 0
        max_depth = max(
            (len(claims_by_task[artifact.task_id]) for artifact in selected),
            default=0,
        )
        for depth in range(max_depth):
            for index, artifact in enumerate(selected):
                task_claims = claims_by_task[artifact.task_id]
                if (
                    depth >= len(task_claims)
                    or claim_count >= self._limits.max_claims_per_task
                ):
                    continue
                if self._append_artifact_value(
                    payloads,
                    payload_index=index,
                    field="claims",
                    value=self._claim_payload(task_claims[depth]),
                ):
                    claim_count += 1

        for index, artifact in enumerate(selected):
            answer_fragment = (
                artifact.answer_fragment
                if allowed_sources is None
                else " ".join(claim.claim for claim in claims_by_task[artifact.task_id])
            )
            if answer_fragment:
                self._set_artifact_value(
                    payloads,
                    payload_index=index,
                    field="answer_fragment",
                    value=_bounded_text(
                        answer_fragment,
                        min(self._limits.max_artifact_text_chars, 600),
                    ),
                )

        for field, extractor, text_limit in (
            (
                "covered_success_criteria",
                lambda item: covered_by_task[item.task_id],
                400,
            ),
            (
                "contributing_worker_ids",
                lambda item: item.contributing_worker_ids,
                120,
            ),
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

    @staticmethod
    def _render_evidence_bundle(
        records: Sequence[EvidenceRecord],
        *,
        max_chars: int,
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
    def _render_evidence(records: Sequence[EvidenceRecord], *, max_chars: int) -> str:
        """Compatibility wrapper used by focused rendering tests."""

        return MultiAgentResearchOrchestrator._render_evidence_bundle(
            records,
            max_chars=max_chars,
        ).text
