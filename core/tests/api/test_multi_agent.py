"""Tests for context-isolated, model-controlled multi-agent research."""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Callable

import pytest

from fs_explorer_api.llm import ChatTurn, LLMClient, LLMUsage
from fs_explorer_api.llm.profile import LLMRole
from fs_explorer_api.multi_agent import (
    ContextBudget,
    EvidenceRecord,
    MultiAgentResearchOrchestrator,
    ResearchLimits,
    ResearchProgress,
    _bounded_planner_input,
    _claim_identifier,
    _is_near_duplicate_query,
)
from fs_explorer_api.orchestration_models import (
    AnswerRequirement,
    AnswerRequirementKind,
    DerivedConclusion,
    EvidenceClaim,
    EvidenceConfidence,
    EvidenceRequirement,
    EvidenceRequirementKind,
    ExecutionStrategy,
    GlobalPlan,
    PlanMode,
    ProblemType,
    SearchAssignment,
    SearchAssignmentBatch,
    TaskArtifact,
    TaskKind,
    TaskOutput,
    TaskOutputRef,
    TaskSpec,
    TaskStatus,
    WorkerArtifact,
    WorkerStatus,
)
from fs_explorer_api.search import IndexedQueryEngine, SearchHit


@pytest.fixture(autouse=True)
def _isolate_task_reranking(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests deterministic and independent of a hosted reranker."""

    def rank_candidates(*, query, documents, limit, diversify=True):
        del query, diversify
        return [
            (document, document.combined_score)
            for document in documents[: max(limit, 1)]
        ]

    monkeypatch.setattr(
        IndexedQueryEngine,
        "rank_candidates",
        staticmethod(rank_candidates),
    )


def _task(
    task_id: str,
    question: str,
    *,
    depends_on: list[str] | None = None,
) -> TaskSpec:
    criterion_id = f"criterion-{task_id}"
    return TaskSpec(
        task_id=task_id,
        kind=TaskKind.EVIDENCE,
        issue=f"Resolve {task_id}",
        search_question=question,
        requirement_ids=[criterion_id],
        fact_ids=[],
        unknown_ids=[],
        branch_ids=[],
        evidence_requirements=[
            EvidenceRequirement(
                evidence_requirement_id=criterion_id,
                kind=EvidenceRequirementKind.GOVERNING_RULE,
                description=criterion_id,
                requirement_ids=[criterion_id],
            )
        ],
        consumes=[
            TaskOutputRef(task_id=dependency, output_id=f"output-{dependency}")
            for dependency in depends_on or []
        ],
        produces=[
            TaskOutput(
                output_id=f"output-{task_id}",
                description=f"Supported conclusion for {task_id}",
            )
        ],
        required=True,
        as_of_date=None,
        filters=None,
    )


def _plan(
    *tasks: TaskSpec,
    execution_strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE,
) -> GlobalPlan:
    requirements = [
        AnswerRequirement(
            requirement_id=task.requirement_ids[0],
            kind=AnswerRequirementKind.OUTCOME,
            description=f"Answer the requirement for {task.task_id}.",
            required=True,
        )
        for task in tasks
    ]
    return GlobalPlan(
        version="3",
        problem_type=(ProblemType.LOOKUP if len(tasks) == 1 else ProblemType.MIXED),
        mode=PlanMode.DIRECT if len(tasks) == 1 else PlanMode.DECOMPOSED,
        execution_strategy=execution_strategy,
        normalized_question="Normalized question",
        answer_requirements=requirements,
        scenario=None,
        tasks=list(tasks),
        synthesis_requirements=[],
        assumptions=[],
    )


def _claim(task_id: str, *, excerpt: str | None = None) -> EvidenceClaim:
    return EvidenceClaim(
        claim_id=f"claim-{task_id}",
        claim=f"Supported claim for {task_id}",
        document_id=f"doc-{task_id}",
        chunk_id=f"chunk-{task_id}",
        readable_title=f"Model title {task_id}",
        locator="Model locator",
        evidence_excerpt=excerpt or f"Evidence text for {task_id}.",
        requirement_ids=[f"criterion-{task_id}"],
        evidence_requirement_ids=[f"criterion-{task_id}"],
        fact_ids=[],
        confidence=EvidenceConfidence.HIGH,
        effective_start_date=None,
        effective_end_date=None,
    )


def _hit(task_id: str) -> SearchHit:
    return SearchHit(
        doc_id=f"doc-{task_id}",
        relative_path=f"01-{task_id}_regulation.pdf",
        absolute_path=f"/private/{task_id}.pdf",
        position=7,
        text=f"Evidence text for {task_id}. Additional source detail.",
        semantic_score=0.92,
        metadata_score=0,
        score=0.92,
        matched_by="semantic",
        chunk_id=f"chunk-{task_id}",
        chunk_type="text",
        metadata={"article_no": "7"},
    )


def _source_hit(
    *,
    document_id: str,
    chunk_id: str,
    relative_path: str,
    article: str,
    text: str,
    score: float,
) -> SearchHit:
    return SearchHit(
        doc_id=document_id,
        relative_path=relative_path,
        absolute_path=f"/private/{relative_path}",
        position=int(article),
        text=text,
        semantic_score=score,
        metadata_score=0,
        score=score,
        matched_by="semantic",
        chunk_id=chunk_id,
        chunk_type="text",
        metadata={"article_no": article},
    )


def _source_claim(
    task_id: str,
    *,
    document_id: str,
    chunk_id: str,
    excerpt: str,
    title: str = "Model-provided title",
    locator: str = "Model-provided locator",
) -> EvidenceClaim:
    return EvidenceClaim(
        claim_id=f"claim-{task_id}-{chunk_id}",
        claim=f"Supported claim from {chunk_id}",
        document_id=document_id,
        chunk_id=chunk_id,
        readable_title=title,
        locator=locator,
        evidence_excerpt=excerpt,
        requirement_ids=[f"criterion-{task_id}"],
        evidence_requirement_ids=[f"criterion-{task_id}"],
        fact_ids=[],
        confidence=EvidenceConfidence.HIGH,
        effective_start_date=None,
        effective_end_date=None,
    )


@dataclass(frozen=True)
class _SearchResult:
    query: str
    hits: list[SearchHit]
    error: str | None = None


class _ScriptedClient:
    def __init__(
        self,
        role: str,
        responder: Callable[[type, str], object],
    ) -> None:
        self.role = role
        self.provider = "openrouter"
        self.model = f"test/{role}"
        self._responder = responder
        self.calls: list[tuple[type, list[ChatTurn], str]] = []

    async def generate_structured(
        self,
        history,
        system_prompt,
        schema,
        *,
        thinking_level=None,
    ):
        self.calls.append((schema, list(history), system_prompt))
        return self._responder(schema, history[0].text), LLMUsage(
            input_tokens=20,
            output_tokens=5,
        )

    async def stream_text(self, history, system_prompt, *, thinking_level=None):
        yield "unused"

    def last_stream_usage(self):
        return None


def _task_id_from_text(text: str) -> str:
    positions: list[tuple[int, str]] = []
    for task_id in ("task_1", "task_2", "task_3"):
        compact = text.find(f'"task_id":"{task_id}"')
        spaced = text.find(f'"task_id": "{task_id}"')
        candidates = [position for position in (compact, spaced) if position >= 0]
        if candidates:
            positions.append((min(candidates), task_id))
    if positions:
        return min(positions)[1]
    raise AssertionError(f"No task id in:\n{text}")


def _task_responder(schema: type, text: str) -> object:
    task_id = _task_id_from_text(text)
    if schema is SearchAssignmentBatch:
        if "WORKER ARTIFACTS SO FAR:\n[]" not in text:
            return SearchAssignmentBatch(
                task_id=task_id,
                stop=True,
                stop_reason="The task criteria are covered.",
                assignments=[],
            )
        return SearchAssignmentBatch(
            task_id=task_id,
            stop=False,
            stop_reason=None,
            assignments=[
                SearchAssignment(
                    assignment_id=f"assignment-{task_id}",
                    task_id=task_id,
                    query=f"query-{task_id}",
                    objective=f"Find evidence for {task_id}",
                    evidence_requirements=[f"criterion-{task_id}"],
                    excluded_queries=[],
                    as_of_date=None,
                    filters=None,
                )
            ],
        )
    assert schema is TaskArtifact
    return TaskArtifact(
        task_id=task_id,
        status=TaskStatus.COMPLETE,
        answer_fragment=f"Conclusion for {task_id}",
        covered_requirement_ids=[f"criterion-{task_id}"],
        uncovered_requirement_ids=[],
        claims=[_claim(task_id)],
        application_findings=[],
        conflicts=[],
        gaps=[],
        contributing_worker_ids=[f"worker-{task_id}-r1-1"],
    )


def _worker_responder(schema: type, text: str) -> object:
    assert schema is WorkerArtifact
    task_id = _task_id_from_text(text)
    return WorkerArtifact(
        task_id=task_id,
        assignment_id=f"assignment-{task_id}",
        worker_id=f"worker-{task_id}-r1-1",
        status=WorkerStatus.SUCCESS,
        searches_run=[f"query-{task_id}"],
        claims=[_claim(task_id)],
        gaps=[],
        cross_references=[],
        error_code=None,
        error_message=None,
    )


def _unlimited_worker_responder(schema: type, text: str) -> object:
    if schema is SearchAssignmentBatch:
        assert "LAST INDEXED_SEARCH TOOL RESULT" in text
        return SearchAssignmentBatch(
            task_id=_task_id_from_text(text),
            stop=True,
            stop_reason="The latest verified result sufficiently supports the need.",
            assignments=[],
        )
    return _worker_responder(schema, text)


def _limits() -> ResearchLimits:
    return ResearchLimits(
        max_tasks=5,
        max_parallel_tasks=3,
        max_assignments_per_wave=2,
        max_worker_rounds=2,
        max_total_workers=8,
        max_parallel_llm_calls=4,
        max_parallel_retrievals=8,
        worker_hit_limit=4,
        worker_hit_chars=2_000,
        review_hit_chars=2_000,
        final_evidence_limit=8,
        final_chunk_chars=4_000,
    )


@pytest.mark.asyncio
async def test_direct_question_stays_one_task_and_uses_verified_source() -> None:
    planner = _ScriptedClient(
        "planner",
        lambda schema, _text: _plan(_task("task_1", "Question one")),
    )
    task_client = _ScriptedClient("task", _task_responder)
    worker = _ScriptedClient("worker", _worker_responder)
    progress: list[ResearchProgress] = []

    orchestrator = MultiAgentResearchOrchestrator(
        planner_llm=planner,
        task_llm=task_client,
        worker_llm=worker,
        search_runner=lambda **kwargs: _SearchResult(
            query=kwargs["query"], hits=[_hit("task_1")]
        ),
        limits=_limits(),
        on_progress=progress.append,
        search_runner_in_thread=False,
    )

    result = await orchestrator.run("Question one")

    assert result.plan.mode == PlanMode.DIRECT
    assert [artifact.task_id for artifact in result.task_artifacts] == ["task_1"]
    assert result.task_artifacts[0].status == TaskStatus.COMPLETE
    assert result.incomplete is False
    assert result.evidence_sources[0]["chunk_id"] == "chunk-task_1"
    assert result.evidence_sources[0]["title"] == "task 1 regulation"
    assert "Article 7" in result.final_context
    assert "/private/" not in result.final_context
    assert all(
        len(history) == 1
        for client in (planner, task_client, worker)
        for _schema, history, _prompt in client.calls
    )
    assert [event.sequence for event in progress] == sorted(
        event.sequence for event in progress
    )


@pytest.mark.asyncio
async def test_unlimited_search_agent_self_corrects_until_it_finds_evidence() -> None:
    task = _task("task_1", "initial wrong terminology")
    planner = _ScriptedClient("planner", lambda _schema, _text: _plan(task))
    searches: list[str] = []
    continuation_inputs: list[str] = []

    def task_responder(schema: type, _text: str) -> object:
        if schema is SearchAssignmentBatch:
            return SearchAssignmentBatch(
                task_id="task_1",
                stop=False,
                stop_reason=None,
                assignments=[
                    SearchAssignment(
                        assignment_id="initial_search",
                        task_id="task_1",
                        query="initial wrong terminology",
                        objective="Test the initial terminology.",
                        evidence_requirements=["criterion-task_1"],
                        excluded_queries=[],
                        as_of_date=None,
                        filters=None,
                    )
                ],
            )
        assert schema is TaskArtifact
        return TaskArtifact(
            task_id="task_1",
            status=TaskStatus.COMPLETE,
            answer_fragment=None,
            covered_requirement_ids=[],
            uncovered_requirement_ids=["criterion-task_1"],
            claims=[],
            application_findings=[],
            conflicts=[],
            gaps=[],
            contributing_worker_ids=[],
        )

    def worker_responder(schema: type, text: str) -> object:
        if schema is SearchAssignmentBatch:
            continuation_inputs.append(text)
            if "Evidence text for task_1." in text:
                return SearchAssignmentBatch(
                    task_id="task_1",
                    stop=True,
                    stop_reason=(
                        "The latest evidence supports the assigned requirement "
                        "and exposes no unresolved exception or cross-reference."
                    ),
                    assignments=[],
                )
            query = (
                "correct instrument terminology"
                if len(continuation_inputs) == 1
                else "cross referenced procedure phrase"
            )
            return SearchAssignmentBatch(
                task_id="task_1",
                stop=False,
                stop_reason=None,
                assignments=[
                    SearchAssignment(
                        assignment_id=f"revised_{len(continuation_inputs)}",
                        task_id="task_1",
                        query=query,
                        objective=(
                            "The prior result shows the terminology was wrong; "
                            "switch to the governing instrument wording."
                        ),
                        evidence_requirements=["criterion-task_1"],
                        excluded_queries=list(searches),
                        as_of_date=None,
                        filters=None,
                    )
                ],
            )
        assert schema is WorkerArtifact
        return WorkerArtifact(
            task_id="task_1",
            assignment_id="revised_2",
            worker_id="model-worker",
            status=WorkerStatus.SUCCESS,
            searches_run=["cross referenced procedure phrase"],
            claims=[_claim("task_1")],
            gaps=[],
            cross_references=[],
            error_code=None,
            error_message=None,
        )

    def search(**kwargs):
        query = kwargs["query"]
        searches.append(query)
        return _SearchResult(
            query=query,
            hits=[_hit("task_1")]
            if query == "cross referenced procedure phrase"
            else [],
        )

    result = await MultiAgentResearchOrchestrator(
        planner_llm=planner,
        task_llm=_ScriptedClient("task", task_responder),
        worker_llm=_ScriptedClient("worker", worker_responder),
        search_runner=search,
        search_runner_in_thread=False,
    ).run("Find the rule even if the first terminology is wrong.")

    assert searches == [
        "initial wrong terminology",
        "correct instrument terminology",
        "cross referenced procedure phrase",
    ]
    assert len(continuation_inputs) == 3
    assert "initial wrong terminology" in continuation_inputs[0]
    assert "correct instrument terminology" in continuation_inputs[1]
    assert "Evidence text for task_1." in continuation_inputs[2]
    assert result.task_artifacts[0].status == TaskStatus.COMPLETE
    assert result.incomplete is False
    assert "Evidence text for task_1." in result.final_context
    assert "doc-task_1" not in result.final_context
    assert "Model title" not in result.final_context


@pytest.mark.asyncio
async def test_unlimited_search_stops_only_after_search_agent_reports_exhaustion() -> (
    None
):
    task = _task("task_1", "rare corpus phrase")
    planner = _ScriptedClient("planner", lambda _schema, _text: _plan(task))

    def task_responder(schema: type, text: str) -> object:
        assert schema is SearchAssignmentBatch
        if "WORKER ARTIFACTS SO FAR:\n[]" not in text:
            return SearchAssignmentBatch(
                task_id="task_1",
                stop=True,
                stop_reason="The owning search agent reported corpus exhaustion.",
                assignments=[],
            )
        return SearchAssignmentBatch(
            task_id="task_1",
            stop=False,
            stop_reason=None,
            assignments=[
                SearchAssignment(
                    assignment_id="rare_search",
                    task_id="task_1",
                    query="rare corpus phrase",
                    objective="Find the rare governing phrase.",
                    evidence_requirements=["criterion-task_1"],
                    excluded_queries=[],
                    as_of_date=None,
                    filters=None,
                )
            ],
        )

    worker = _ScriptedClient(
        "worker",
        lambda schema, _text: (
            SearchAssignmentBatch(
                task_id="task_1",
                stop=True,
                stop_reason=(
                    "No distinct terminology, instrument, date, exception, or "
                    "cross-reference search remains."
                ),
                assignments=[],
            )
            if schema is SearchAssignmentBatch
            else pytest.fail("An empty search must not run evidence extraction.")
        ),
    )
    searches: list[str] = []
    result = await MultiAgentResearchOrchestrator(
        planner_llm=planner,
        task_llm=_ScriptedClient("task", task_responder),
        worker_llm=worker,
        search_runner=lambda **kwargs: (
            searches.append(kwargs["query"])
            or _SearchResult(query=kwargs["query"], hits=[])
        ),
        search_runner_in_thread=False,
    ).run("Find the rare rule.")

    assert searches == ["rare corpus phrase"]
    assert len(worker.calls) == 1
    assert result.incomplete is True
    assert result.task_artifacts[0].status == TaskStatus.FAILED


@pytest.mark.asyncio
async def test_task_coordinator_repairs_non_executable_stop() -> None:
    task = _task("task_1", "governing instrument phrase")
    planner = _ScriptedClient("planner", lambda _schema, _text: _plan(task))
    coordinator_calls = 0

    def task_responder(schema: type, text: str) -> object:
        nonlocal coordinator_calls
        if schema is TaskArtifact:
            return TaskArtifact(
                task_id="task_1",
                status=TaskStatus.COMPLETE,
                answer_fragment=None,
                covered_requirement_ids=[],
                uncovered_requirement_ids=["criterion-task_1"],
                claims=[],
                application_findings=[],
                conflicts=[],
                gaps=[],
                contributing_worker_ids=[],
            )
        coordinator_calls += 1
        if coordinator_calls == 1:
            return SearchAssignmentBatch(
                task_id="task_1",
                stop=True,
                stop_reason="Stopped before delegating.",
                assignments=[],
            )
        assert "REJECTED PREVIOUS DECISION" in text
        assert "delegate at least one search" in text
        return SearchAssignmentBatch(
            task_id="task_1",
            stop=False,
            stop_reason=None,
            assignments=[
                SearchAssignment(
                    assignment_id="agent_chosen_search",
                    task_id="task_1",
                    query="governing instrument phrase",
                    objective="Search the governing instrument wording.",
                    evidence_requirements=["criterion-task_1"],
                    excluded_queries=[],
                    as_of_date=None,
                    filters=None,
                )
            ],
        )

    result = await MultiAgentResearchOrchestrator(
        planner_llm=planner,
        task_llm=_ScriptedClient("task", task_responder),
        worker_llm=_ScriptedClient("worker", _unlimited_worker_responder),
        search_runner=lambda **kwargs: _SearchResult(
            query=kwargs["query"],
            hits=[_hit("task_1")],
        ),
        search_runner_in_thread=False,
    ).run("Find the governing rule.")

    assert coordinator_calls == 2
    assert result.task_artifacts[0].status == TaskStatus.COMPLETE


@pytest.mark.asyncio
async def test_search_agent_may_retry_same_tool_call_after_transient_failure() -> None:
    task = _task("task_1", "exact governing phrase")
    planner = _ScriptedClient("planner", lambda _schema, _text: _plan(task))
    search_calls = 0

    def task_responder(schema: type, text: str) -> object:
        if schema is SearchAssignmentBatch:
            return SearchAssignmentBatch(
                task_id="task_1",
                stop=False,
                stop_reason=None,
                assignments=[
                    SearchAssignment(
                        assignment_id="initial_exact",
                        task_id="task_1",
                        query="exact governing phrase",
                        objective="Run the exact indexed search.",
                        evidence_requirements=["criterion-task_1"],
                        excluded_queries=[],
                        as_of_date=None,
                        filters=None,
                    )
                ],
            )
        return TaskArtifact(
            task_id="task_1",
            status=TaskStatus.COMPLETE,
            answer_fragment=None,
            covered_requirement_ids=[],
            uncovered_requirement_ids=["criterion-task_1"],
            claims=[],
            application_findings=[],
            conflicts=[],
            gaps=[],
            contributing_worker_ids=[],
        )

    def worker_responder(schema: type, text: str) -> object:
        if schema is SearchAssignmentBatch:
            if "Evidence text for task_1." in text:
                return SearchAssignmentBatch(
                    task_id="task_1",
                    stop=True,
                    stop_reason="The successful retry supports the requirement.",
                    assignments=[],
                )
            assert "worker_execution_failed" in text
            return SearchAssignmentBatch(
                task_id="task_1",
                stop=False,
                stop_reason=None,
                assignments=[
                    SearchAssignment(
                        assignment_id="retry_exact",
                        task_id="task_1",
                        query="exact governing phrase",
                        objective=(
                            "The tool failed transiently; retry the same valid "
                            "search instead of treating it as corpus exhaustion."
                        ),
                        evidence_requirements=["criterion-task_1"],
                        excluded_queries=[],
                        as_of_date=None,
                        filters=None,
                    )
                ],
            )
        return _worker_responder(schema, text)

    def search(**kwargs):
        nonlocal search_calls
        search_calls += 1
        if search_calls <= 2:
            return _SearchResult(
                query=kwargs["query"],
                hits=[],
                error="temporary connection timeout",
            )
        return _SearchResult(query=kwargs["query"], hits=[_hit("task_1")])

    result = await MultiAgentResearchOrchestrator(
        planner_llm=planner,
        task_llm=_ScriptedClient("task", task_responder),
        worker_llm=_ScriptedClient("worker", worker_responder),
        search_runner=search,
        search_runner_in_thread=False,
    ).run("Find the exact rule.")

    assert search_calls == 3
    assert result.task_artifacts[0].status == TaskStatus.COMPLETE


@pytest.mark.asyncio
async def test_exhaustive_scenario_planner_keeps_all_requested_legal_headings() -> None:
    headings = [
        ("tir_rejection", "Why the TIR Carnet was rejected"),
        ("driver_vehicle", "Driver and vehicle certification requirements"),
        ("guarantee", "Validity of UND and DAC global guarantees"),
        ("accident_debt", "Accident, leaked goods, and customs debt"),
        ("basel", "Basel Convention and illegal traffic"),
        ("debt_recovery", "Turkey Italy mutual assistance and debt recovery"),
        ("authorizations", "Authorized Consignor and guarantee reduction impact"),
    ]
    question = "\n".join(description for _task_id, description in headings)
    planned_tasks = [_task(task_id, description) for task_id, description in headings]

    def planner_responder(_schema: type, text: str) -> object:
        for _task_id, description in headings:
            assert description in text
        return _plan(*planned_tasks)

    orchestrator = MultiAgentResearchOrchestrator(
        planner_llm=_ScriptedClient("planner", planner_responder),
        task_llm=_ScriptedClient(
            "task",
            lambda _schema, _text: pytest.fail("Planning must not run a task."),
        ),
        worker_llm=_ScriptedClient(
            "worker",
            lambda _schema, _text: pytest.fail("Planning must not run a worker."),
        ),
        search_runner=lambda **_kwargs: pytest.fail("Planning must not search."),
        search_runner_in_thread=False,
    )

    plan = await orchestrator._create_plan(question)

    assert len(plan.answer_requirements) == 7
    assert [task.task_id for task in plan.tasks] == [
        task_id for task_id, _description in headings
    ]
    assert orchestrator._unlimited_research is True


@pytest.mark.asyncio
async def test_single_pass_direct_skips_redundant_task_llm_calls() -> None:
    planner = _ScriptedClient(
        "planner",
        lambda schema, _text: _plan(
            _task("task_1", "Question one"),
            execution_strategy=ExecutionStrategy.SINGLE_PASS,
        ),
    )
    task_client = _ScriptedClient(
        "task",
        lambda _schema, _text: pytest.fail(
            "single-pass coverage must not call coordinator or reviewer"
        ),
    )
    worker = _ScriptedClient("worker", _worker_responder)
    searches: list[str] = []

    def search(**kwargs):
        searches.append(kwargs["query"])
        return _SearchResult(query=kwargs["query"], hits=[_hit("task_1")])

    result = await MultiAgentResearchOrchestrator(
        planner_llm=planner,
        task_llm=task_client,
        worker_llm=worker,
        search_runner=search,
        limits=_limits(),
        search_runner_in_thread=False,
    ).run("Question one")

    assert result.plan.execution_strategy == ExecutionStrategy.SINGLE_PASS
    assert result.task_artifacts[0].status == TaskStatus.COMPLETE
    assert searches == ["Question one"]
    assert len(planner.calls) == 1
    assert len(worker.calls) == 1
    assert task_client.calls == []


@pytest.mark.asyncio
async def test_semantically_invalid_plan_is_returned_to_planner_for_correction() -> (
    None
):
    original_question = "What is the original filing requirement?"
    invalid = _plan(
        _task("one", "First duplicate part"),
        _task("two", "Second duplicate part"),
    ).model_copy(update={"problem_type": ProblemType.LOOKUP})
    corrected = _plan(_task("task_1", original_question))
    responses = iter([invalid, corrected])
    planner = _ScriptedClient("planner", lambda _schema, _text: next(responses))
    orchestrator = MultiAgentResearchOrchestrator(
        planner_llm=planner,
        task_llm=_ScriptedClient(
            "task",
            lambda _schema, _text: pytest.fail("Planning must not run tasks."),
        ),
        worker_llm=_ScriptedClient(
            "worker",
            lambda _schema, _text: pytest.fail("Planning must not run workers."),
        ),
        search_runner=lambda **_kwargs: pytest.fail("Planning must not search."),
        search_runner_in_thread=False,
    )

    plan = await orchestrator._create_plan(original_question)

    assert plan is corrected
    assert len(planner.calls) == 2
    correction_text = planner.calls[1][1][0].text
    assert "YOUR PREVIOUS PLAN FAILED SERVER VALIDATION" in correction_text
    assert "lookup plans must use one direct task" in correction_text
    assert "do not replace it with a simplified graph" in correction_text


@pytest.mark.asyncio
async def test_single_pass_coverage_miss_upgrades_to_adaptive_wave() -> None:
    planner = _ScriptedClient(
        "planner",
        lambda schema, _text: _plan(
            _task("task_1", "Question one"),
            execution_strategy=ExecutionStrategy.SINGLE_PASS,
        ),
    )

    def task_responder(schema: type, text: str) -> object:
        if schema is SearchAssignmentBatch:
            return SearchAssignmentBatch(
                task_id="task_1",
                stop=False,
                stop_reason=None,
                assignments=[
                    SearchAssignment(
                        assignment_id="adaptive-follow-up",
                        task_id="task_1",
                        query="query-task_1",
                        objective="Fill the missing direct-search evidence",
                        evidence_requirements=["criterion-task_1"],
                        excluded_queries=["Question one"],
                        as_of_date=None,
                        filters=None,
                    )
                ],
            )
        return _task_responder(schema, text)

    task_client = _ScriptedClient("task", task_responder)
    worker = _ScriptedClient("worker", _worker_responder)
    searches: list[str] = []
    progress: list[ResearchProgress] = []

    def search(**kwargs):
        query = kwargs["query"]
        searches.append(query)
        return _SearchResult(
            query=query,
            hits=[] if query == "Question one" else [_hit("task_1")],
        )

    result = await MultiAgentResearchOrchestrator(
        planner_llm=planner,
        task_llm=task_client,
        worker_llm=worker,
        search_runner=search,
        limits=_limits(),
        on_progress=progress.append,
        search_runner_in_thread=False,
    ).run("Question one")

    assert result.task_artifacts[0].status == TaskStatus.COMPLETE
    assert searches == ["Question one", "query-task_1"]
    assert [schema for schema, _history, _prompt in task_client.calls] == [
        SearchAssignmentBatch,
        TaskArtifact,
    ]
    coordinator_input = task_client.calls[0][1][0].text
    assert "UNRESOLVED EVIDENCE REQUIREMENTS" in coordinator_input
    assert "criterion-task_1" in coordinator_input
    gap_events = [event for event in progress if event.kind == "gap_recovery_started"]
    assert len(gap_events) == 1
    assert "criterion-task_1" in gap_events[0].detail


def test_near_duplicate_queries_are_rejected_without_an_llm_call() -> None:
    assert _is_near_duplicate_query(
        "transit süresi aşımı cezası yönetmelik",
        ["yönetmelik transit süresi aşımı cezası"],
    )
    assert not _is_near_duplicate_query(
        "transit süresi aşımı istisnaları",
        ["teminat iadesi başvuru usulü"],
    )


def test_natural_language_metadata_filter_is_discarded() -> None:
    orchestrator = MultiAgentResearchOrchestrator(
        planner_llm=_ScriptedClient("planner", lambda _schema, _text: None),
        task_llm=_ScriptedClient("task", lambda _schema, _text: None),
        worker_llm=_ScriptedClient("worker", lambda _schema, _text: None),
        search_runner=lambda **_kwargs: _SearchResult(query="", hits=[]),
        search_runner_in_thread=False,
    )

    assert orchestrator._bounded_filter("Öncelikle yürürlükteki resmî mevzuat") is None
    assert orchestrator._bounded_filter("document_type=tebliğ") == (
        "document_type=tebliğ"
    )


@pytest.mark.asyncio
async def test_production_lookup_searches_without_planner_prose_filter() -> None:
    task = _task("task_1", "TIR karnesi ek teminat").model_copy(
        update={"filters": "Öncelikle yürürlükteki resmî mevzuat"}
    )
    planner = _ScriptedClient(
        "planner",
        lambda _schema, _text: _plan(
            task,
            execution_strategy=ExecutionStrategy.SINGLE_PASS,
        ),
    )
    seen_filters: list[str | None] = []

    def search(**kwargs):
        seen_filters.append(kwargs["filters"])
        return _SearchResult(query=kwargs["query"], hits=[_hit("task_1")])

    result = await MultiAgentResearchOrchestrator(
        planner_llm=planner,
        task_llm=_ScriptedClient("task", _task_responder),
        worker_llm=_ScriptedClient("worker", _unlimited_worker_responder),
        search_runner=search,
        search_runner_in_thread=False,
    ).run("TIR karnesi ek teminat")

    assert result.task_artifacts[0].status == TaskStatus.COMPLETE
    assert seen_filters == [None]


@pytest.mark.asyncio
async def test_production_structured_stage_has_no_wall_clock_timeout() -> None:
    class SlowClient(_ScriptedClient):
        async def generate_structured(
            self,
            history,
            system_prompt,
            schema,
            *,
            thinking_level=None,
        ):
            await asyncio.sleep(0.03)
            return await super().generate_structured(
                history,
                system_prompt,
                schema,
                thinking_level=thinking_level,
            )

    plan = _plan(_task("task_1", "Question one"))
    planner = SlowClient("planner", lambda _schema, _text: plan)
    orchestrator = MultiAgentResearchOrchestrator(
        planner_llm=planner,
        task_llm=_ScriptedClient("task", lambda _schema, _text: None),
        worker_llm=_ScriptedClient("worker", lambda _schema, _text: None),
        search_runner=lambda **_kwargs: _SearchResult(query="", hits=[]),
        search_runner_in_thread=False,
    )
    object.__setattr__(orchestrator._limits, "llm_timeout_seconds", 0.001)

    result = await orchestrator._generate_structured(
        client=planner,
        role="planner",
        purpose="global_plan",
        prompt="system",
        user_text="Question one",
        operation_id="no-timeout",
        schema=GlobalPlan,
    )

    assert result[0] is plan


@pytest.mark.asyncio
async def test_independent_tasks_search_in_parallel_without_sibling_context() -> None:
    task_1 = _task("task_1", "Question one")
    task_2 = _task("task_2", "Question two")
    planner = _ScriptedClient("planner", lambda schema, _text: _plan(task_1, task_2))
    task_client = _ScriptedClient("task", _task_responder)
    worker = _ScriptedClient("worker", _worker_responder)
    active = 0
    maximum_active = 0

    async def search(**kwargs):
        nonlocal active, maximum_active
        task_id = kwargs["query"].removeprefix("query-")
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0.05)
        active -= 1
        return _SearchResult(query=kwargs["query"], hits=[_hit(task_id)])

    result = await MultiAgentResearchOrchestrator(
        planner_llm=planner,
        task_llm=task_client,
        worker_llm=worker,
        search_runner=search,
        limits=_limits(),
        search_runner_in_thread=False,
    ).run("Compare question one and question two")

    assert maximum_active == 2
    assert {artifact.task_id for artifact in result.task_artifacts} == {
        "task_1",
        "task_2",
    }
    for schema, history, _prompt in task_client.calls:
        if schema is not SearchAssignmentBatch:
            continue
        text = history[0].text
        if '"task_id":"task_1"' in text:
            assert "Question two" not in text
        if '"task_id":"task_2"' in text:
            assert "Question one" not in text


@pytest.mark.asyncio
async def test_identical_cross_task_worker_query_reuses_one_inflight_search() -> None:
    task_1 = _task("task_1", "Find the governing rule.")
    task_2 = _task("task_2", "Find the effective-date rule.")
    planner = _ScriptedClient("planner", lambda _schema, _text: _plan(task_1, task_2))
    shared_query = "shared indexed query"

    def task_responder(schema: type, text: str) -> object:
        task_id = _task_id_from_text(text)
        if schema is SearchAssignmentBatch:
            return SearchAssignmentBatch(
                task_id=task_id,
                stop=False,
                stop_reason=None,
                assignments=[
                    SearchAssignment(
                        assignment_id=f"shared_{task_id}",
                        task_id=task_id,
                        query=shared_query,
                        objective=f"Resolve {task_id}.",
                        evidence_requirements=[f"criterion-{task_id}"],
                        excluded_queries=[],
                        as_of_date=None,
                        filters=None,
                    )
                ],
            )
        assert schema is TaskArtifact
        claim = _source_claim(
            task_id,
            document_id="doc-shared",
            chunk_id="chunk-shared",
            excerpt="The shared provision supplies both required rules.",
        )
        return TaskArtifact(
            task_id=task_id,
            status=TaskStatus.COMPLETE,
            answer_fragment=claim.claim,
            covered_requirement_ids=[f"criterion-{task_id}"],
            uncovered_requirement_ids=[],
            claims=[claim],
            application_findings=[],
            conflicts=[],
            gaps=[],
            contributing_worker_ids=[f"worker-{task_id}-shared_{task_id}"],
        )

    def worker_responder(schema: type, text: str) -> object:
        assert schema is WorkerArtifact
        task_id = _task_id_from_text(text)
        return WorkerArtifact(
            task_id=task_id,
            assignment_id=f"shared_{task_id}",
            worker_id=f"worker-{task_id}-shared_{task_id}",
            status=WorkerStatus.SUCCESS,
            searches_run=[shared_query],
            claims=[
                _source_claim(
                    task_id,
                    document_id="doc-shared",
                    chunk_id="chunk-shared",
                    excerpt="The shared provision supplies both required rules.",
                )
            ],
            gaps=[],
            cross_references=[],
            error_code=None,
            error_message=None,
        )

    search_calls = 0

    async def search(**kwargs):
        nonlocal search_calls
        search_calls += 1
        await asyncio.sleep(0.05)
        return _SearchResult(
            query=kwargs["query"],
            hits=[
                _source_hit(
                    document_id="doc-shared",
                    chunk_id="chunk-shared",
                    relative_path="shared_regulation.pdf",
                    article="7",
                    text="The shared provision supplies both required rules.",
                    score=1.0,
                )
            ],
        )

    result = await MultiAgentResearchOrchestrator(
        planner_llm=planner,
        task_llm=_ScriptedClient("task", task_responder),
        worker_llm=_ScriptedClient("worker", worker_responder),
        search_runner=search,
        limits=_limits(),
        search_runner_in_thread=False,
    ).run("Answer both distinct requirements.")

    assert search_calls == 1
    assert all(
        artifact.status == TaskStatus.COMPLETE for artifact in result.task_artifacts
    )


@pytest.mark.asyncio
async def test_dependency_task_waits_for_required_artifact() -> None:
    task_1 = _task("task_1", "Question one")
    task_2 = _task("task_2", "Question two", depends_on=["task_1"])
    planner = _ScriptedClient("planner", lambda schema, _text: _plan(task_1, task_2))
    task_client = _ScriptedClient("task", _task_responder)
    worker = _ScriptedClient("worker", _worker_responder)
    search_order: list[str] = []

    def search(**kwargs):
        search_order.append(kwargs["query"])
        task_id = kwargs["query"].removeprefix("query-")
        return _SearchResult(query=kwargs["query"], hits=[_hit(task_id)])

    await MultiAgentResearchOrchestrator(
        planner_llm=planner,
        task_llm=task_client,
        worker_llm=worker,
        search_runner=search,
        limits=_limits(),
        search_runner_in_thread=False,
    ).run("Dependent question")

    assert search_order == ["query-task_1", "query-task_2"]
    task_2_inputs = [
        history[0].text
        for schema, history, _prompt in task_client.calls
        if '"task_id":"task_2"' in history[0].text
    ]
    assert any("Supported claim for task_1" in text for text in task_2_inputs)


@pytest.mark.asyncio
async def test_failed_required_task_isolated_and_marks_run_incomplete() -> None:
    task_1 = _task("task_1", "Question one")
    task_2 = _task("task_2", "Question two")
    planner = _ScriptedClient("planner", lambda schema, _text: _plan(task_1, task_2))

    def task_responder(schema: type, text: str) -> object:
        task_id = _task_id_from_text(text)
        if schema is SearchAssignmentBatch:
            return _task_responder(schema, text)
        if task_id == "task_2":
            return TaskArtifact(
                task_id=task_id,
                status=TaskStatus.FAILED,
                answer_fragment=None,
                covered_requirement_ids=[],
                uncovered_requirement_ids=[f"criterion-{task_id}"],
                claims=[],
                application_findings=[],
                conflicts=[],
                gaps=["retrieval unavailable"],
                contributing_worker_ids=[],
            )
        return _task_responder(schema, text)

    def search(**kwargs):
        task_id = kwargs["query"].removeprefix("query-")
        if task_id == "task_2":
            return _SearchResult(
                query=kwargs["query"],
                hits=[],
                error="database timeout",
            )
        return _SearchResult(query=kwargs["query"], hits=[_hit(task_id)])

    result = await MultiAgentResearchOrchestrator(
        planner_llm=planner,
        task_llm=_ScriptedClient("task", task_responder),
        worker_llm=_ScriptedClient("worker", _worker_responder),
        search_runner=search,
        limits=_limits(),
        search_runner_in_thread=False,
    ).run("Two-part question")

    by_id = {artifact.task_id: artifact for artifact in result.task_artifacts}
    assert by_id["task_1"].status == TaskStatus.COMPLETE
    assert by_id["task_2"].status == TaskStatus.FAILED
    assert result.incomplete is True
    assert "Answer the requirement for task_2." in result.unresolved_information
    assert "database timeout" not in result.final_context


@pytest.mark.asyncio
async def test_hallucinated_excerpt_is_rejected_before_task_artifact() -> None:
    planner = _ScriptedClient(
        "planner",
        lambda schema, _text: _plan(_task("task_1", "Question one")),
    )

    def invalid_worker(schema: type, text: str) -> object:
        artifact = _worker_responder(schema, text)
        assert isinstance(artifact, WorkerArtifact)
        return artifact.model_copy(
            update={"claims": [_claim("task_1", excerpt="Invented source text")]}
        )

    def invalid_review(schema: type, text: str) -> object:
        if schema is SearchAssignmentBatch:
            return _task_responder(schema, text)
        artifact = _task_responder(schema, text)
        assert isinstance(artifact, TaskArtifact)
        return artifact.model_copy(
            update={"claims": [_claim("task_1", excerpt="Invented source text")]}
        )

    result = await MultiAgentResearchOrchestrator(
        planner_llm=planner,
        task_llm=_ScriptedClient("task", invalid_review),
        worker_llm=_ScriptedClient("worker", invalid_worker),
        search_runner=lambda **kwargs: _SearchResult(
            query=kwargs["query"], hits=[_hit("task_1")]
        ),
        limits=_limits(),
        search_runner_in_thread=False,
    ).run("Question one")

    assert result.task_artifacts[0].status == TaskStatus.FAILED
    assert result.task_artifacts[0].claims == []
    assert result.incomplete is True


@pytest.mark.asyncio
async def test_transient_search_error_is_retried_once() -> None:
    planner = _ScriptedClient(
        "planner",
        lambda schema, _text: _plan(_task("task_1", "Question one")),
    )
    attempts = 0

    def search(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return _SearchResult(
                query=kwargs["query"],
                hits=[],
                error="temporary database timeout",
            )
        return _SearchResult(query=kwargs["query"], hits=[_hit("task_1")])

    result = await MultiAgentResearchOrchestrator(
        planner_llm=planner,
        task_llm=_ScriptedClient("task", _task_responder),
        worker_llm=_ScriptedClient("worker", _worker_responder),
        search_runner=search,
        limits=_limits(),
        search_runner_in_thread=False,
    ).run("Question one")

    assert attempts == 2
    assert result.task_artifacts[0].status == TaskStatus.COMPLETE


def test_evidence_renderer_enforces_one_total_character_budget() -> None:
    records = [
        EvidenceRecord(
            evidence_id=f"evidence-{index}",
            document_id=f"doc-{index}",
            chunk_id=f"chunk-{index}",
            readable_title=f"Regulation {index}",
            locator=f"Article {index}",
            text="x" * 2_000,
            score=0.9,
            metadata={},
        )
        for index in range(3)
    ]

    rendered = MultiAgentResearchOrchestrator._render_evidence(
        records,
        max_chars=750,
    )

    assert len(rendered) <= 750
    assert 1 <= rendered.count("BEGIN_UNTRUSTED_SOURCE_TEXT") <= len(records)


def test_claim_identifier_retains_uniqueness_when_long_ids_are_truncated() -> None:
    task_id = "task_" + ("a" * 59)
    first = _claim_identifier(task_id, "assignment_" + ("x" * 70), 1)
    second = _claim_identifier(task_id, "assignment_" + ("y" * 70), 1)
    third = _claim_identifier(task_id, "assignment_" + ("x" * 70), 2)

    assert len(first) <= 64
    assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", first)
    assert len({first, second, third}) == 3
    assert _claim_identifier("task_1", "single_pass", 1) == (
        "task_1_single_pass_claim_1"
    )


def test_planner_input_bounds_history_without_dropping_current_question() -> None:
    current = "CURRENT-QUESTION-" + ("q" * 4_000)
    rendered = _bounded_planner_input(
        f"{'old ' * 10_000}\n\nCurrent question:\n{current}",
        max_question_chars=1_000,
        max_context_chars=1_500,
    )

    assert len(rendered) <= 1_500
    assert rendered.startswith("CURRENT QUESTION:\nCURRENT-QUESTION-")
    assert current[:1_000] in rendered
    assert "RECENT CONVERSATION CONTEXT" in rendered


def test_compact_artifacts_are_valid_json_with_one_total_context_budget() -> None:
    limits = replace(_limits(), max_artifact_context_chars=3_000)
    orchestrator = MultiAgentResearchOrchestrator(
        planner_llm=_ScriptedClient("planner", lambda _schema, _text: None),
        task_llm=_ScriptedClient("task", lambda _schema, _text: None),
        worker_llm=_ScriptedClient("worker", lambda _schema, _text: None),
        search_runner=lambda **_kwargs: _SearchResult(query="", hits=[]),
        limits=limits,
        search_runner_in_thread=False,
    )
    long_claim = _claim("task_1").model_copy(
        update={
            "claim": "claim " * 1_000,
            "evidence_excerpt": "evidence " * 1_000,
            "requirement_ids": ["criterion-task_1"],
            "evidence_requirement_ids": ["criterion-task_1"],
        }
    )
    workers = [
        WorkerArtifact(
            task_id="task_1",
            assignment_id=f"assignment-{index}",
            worker_id=f"worker-{index}",
            status=WorkerStatus.SUCCESS,
            searches_run=["query " * 500] * 12,
            claims=[long_claim],
            gaps=["MATERIAL_WORKER_GAP", *(["gap " * 500] * 11)],
            cross_references=[
                "MATERIAL_CROSS_REFERENCE",
                *(["reference " * 500] * 11),
            ],
            error_code=None,
            error_message=None,
        )
        for index in range(8)
    ]
    tasks = [
        TaskArtifact(
            task_id=f"task_{index}",
            status=TaskStatus.PARTIAL,
            answer_fragment="answer " * 1_000,
            covered_requirement_ids=["covered"],
            uncovered_requirement_ids=[
                "MATERIAL_UNCOVERED_CRITERION",
                *(f"uncovered-{item}" for item in range(11)),
            ],
            claims=[long_claim],
            application_findings=[],
            conflicts=["MATERIAL_CONFLICT", *(["conflict " * 500] * 11)],
            gaps=["MATERIAL_TASK_GAP", *(["gap " * 500] * 11)],
            contributing_worker_ids=["worker " * 500] * 12,
        )
        for index in range(1, 6)
    ]

    worker_context = orchestrator._compact_worker_artifacts(workers)
    task_context = orchestrator._compact_task_artifacts(tasks)

    assert len(worker_context) <= limits.max_artifact_context_chars
    assert len(task_context) <= limits.max_artifact_context_chars
    assert isinstance(json.loads(worker_context), list)
    assert isinstance(json.loads(task_context), list)
    assert "MATERIAL_WORKER_GAP" in worker_context
    assert "MATERIAL_CROSS_REFERENCE" in worker_context
    assert "MATERIAL_UNCOVERED_CRITERION" in task_context
    assert "MATERIAL_CONFLICT" in task_context
    assert "MATERIAL_TASK_GAP" in task_context


def test_final_boundary_drops_conclusion_when_any_supporting_claim_is_omitted() -> None:
    first_claim = _claim("task_1")
    second_claim = _source_claim(
        "task_1",
        document_id="doc-second",
        chunk_id="chunk-second",
        excerpt="Second required source.",
    )
    evidence_artifact = TaskArtifact(
        task_id="task_1",
        status=TaskStatus.COMPLETE,
        answer_fragment="Both claims are required.",
        covered_requirement_ids=["criterion-task_1"],
        uncovered_requirement_ids=[],
        claims=[first_claim, second_claim],
        application_findings=[],
        conflicts=[],
        gaps=[],
        contributing_worker_ids=["worker-1"],
    )
    finding_text = "The multi-source conclusion applies."
    application_artifact = TaskArtifact(
        task_id="apply",
        status=TaskStatus.COMPLETE,
        answer_fragment=finding_text,
        covered_requirement_ids=["criterion-task_1"],
        uncovered_requirement_ids=[],
        claims=[],
        application_findings=[
            DerivedConclusion(
                conclusion_id="conclusion_1",
                finding=finding_text,
                requirement_ids=["criterion-task_1"],
                fact_ids=["fact_1"],
                branch_ids=[],
                supporting_claim_ids=[
                    first_claim.claim_id,
                    second_claim.claim_id,
                ],
                dependency_refs=[TaskOutputRef(task_id="task_1", output_id="result")],
                confidence=EvidenceConfidence.HIGH,
                limitations=[],
            )
        ],
        conflicts=[],
        gaps=[],
        contributing_worker_ids=[],
    )
    first_record = EvidenceRecord(
        evidence_id="evidence-1",
        document_id=first_claim.document_id,
        chunk_id=first_claim.chunk_id,
        readable_title=first_claim.readable_title,
        locator=first_claim.locator,
        text=first_claim.evidence_excerpt,
        score=1.0,
        metadata={},
    )
    orchestrator = MultiAgentResearchOrchestrator(
        planner_llm=_ScriptedClient("planner", lambda _schema, _text: None),
        task_llm=_ScriptedClient("task", lambda _schema, _text: None),
        worker_llm=_ScriptedClient("worker", lambda _schema, _text: None),
        search_runner=lambda **_kwargs: _SearchResult(query="", hits=[]),
        limits=_limits(),
        search_runner_in_thread=False,
    )

    compact = orchestrator._compact_task_artifacts(
        [evidence_artifact, application_artifact],
        allowed_sources={(first_record.document_id, first_record.chunk_id)},
    )

    assert finding_text not in compact
    assert "findings were omitted" in compact
    assert (
        orchestrator._all_finding_support_rendered(
            [evidence_artifact, application_artifact],
            [first_record],
        )
        is False
    )


@pytest.mark.asyncio
async def test_final_evidence_budget_fair_and_sources_match_rendered_records() -> None:
    task_1 = _task("task_1", "Question one")
    task_2 = _task("task_2", "Question two")
    planner = _ScriptedClient("planner", lambda schema, _text: _plan(task_1, task_2))

    def search(**kwargs):
        task_id = kwargs["query"].removeprefix("query-")
        hit = _hit(task_id)
        return _SearchResult(
            query=kwargs["query"],
            hits=[
                replace(
                    hit,
                    text=(f"Evidence text for {task_id}. " + task_id[-1] * 2_000),
                )
            ],
        )

    result = await MultiAgentResearchOrchestrator(
        planner_llm=planner,
        task_llm=_ScriptedClient("task", _task_responder),
        worker_llm=_ScriptedClient("worker", _worker_responder),
        search_runner=search,
        limits=replace(
            _limits(),
            final_chunk_chars=650,
            max_final_context_chars=4_000,
        ),
        search_runner_in_thread=False,
    ).run("Compare both questions")

    rendered_evidence = result.final_context.split(
        "SERVER-VERIFIED FULL EVIDENCE:\n", 1
    )[1]
    rendered_chunk_ids = re.findall(r"(?m)^chunk_id: (.+)$", rendered_evidence)

    assert rendered_chunk_ids == ["chunk-task_1", "chunk-task_2"]
    assert rendered_evidence.count("BEGIN_UNTRUSTED_SOURCE_TEXT") == 2
    assert len(rendered_evidence) <= 650
    assert {str(source["chunk_id"]) for source in result.evidence_sources} == set(
        rendered_chunk_ids
    )
    assert len(result.final_context) <= 4_000


@pytest.mark.asyncio
async def test_total_worker_budget_is_shared_fairly_across_ready_tasks() -> None:
    tasks = tuple(_task(f"task_{index}", f"Question {index}") for index in range(1, 4))
    planner = _ScriptedClient("planner", lambda schema, _text: _plan(*tasks))

    def fair_task_responder(schema: type, text: str) -> object:
        task_id = _task_id_from_text(text)
        if schema is TaskArtifact:
            return _task_responder(schema, text)
        assert schema is SearchAssignmentBatch
        if "WORKER ARTIFACTS SO FAR:\n[]" not in text:
            return SearchAssignmentBatch(
                task_id=task_id,
                stop=True,
                stop_reason="Enough evidence was collected.",
                assignments=[],
            )
        return SearchAssignmentBatch(
            task_id=task_id,
            stop=False,
            stop_reason=None,
            assignments=[
                SearchAssignment(
                    assignment_id=f"{task_id}-assignment-{index}",
                    task_id=task_id,
                    query=f"query-{task_id}-{index}",
                    objective=f"Find evidence path {index} for {task_id}",
                    evidence_requirements=[f"criterion-{task_id}"],
                    excluded_queries=[],
                    as_of_date=None,
                    filters=None,
                )
                for index in range(1, 3)
            ],
        )

    searches: list[str] = []

    def search(**kwargs):
        searches.append(kwargs["query"])
        match = re.search(r"(task_\d+)", kwargs["query"])
        assert match is not None
        return _SearchResult(
            query=kwargs["query"],
            hits=[_hit(match.group(1))],
        )

    result = await MultiAgentResearchOrchestrator(
        planner_llm=planner,
        task_llm=_ScriptedClient("task", fair_task_responder),
        worker_llm=_ScriptedClient("worker", _worker_responder),
        search_runner=search,
        limits=replace(
            _limits(),
            max_worker_rounds=1,
            max_total_workers=3,
            max_assignments_per_wave=2,
        ),
        search_runner_in_thread=False,
    ).run("Answer all three parts")

    searched_tasks: list[str] = []
    for query in searches:
        match = re.search(r"(task_\d+)", query)
        assert match is not None
        searched_tasks.append(match.group(1))
    assert sorted(searched_tasks) == ["task_1", "task_2", "task_3"]
    assert all(
        artifact.status == TaskStatus.COMPLETE for artifact in result.task_artifacts
    )


@pytest.mark.asyncio
async def test_cancel_resume_reuses_completed_search_and_structured_calls() -> None:
    planner = _ScriptedClient(
        "planner",
        lambda schema, _text: _plan(_task("task_1", "Question one")),
    )
    task_client = _ScriptedClient("task", _task_responder)
    worker = _ScriptedClient("worker", _worker_responder)
    provider_result_observed = asyncio.Event()
    finish_telemetry = asyncio.Event()
    search_count = 0

    async def pause_after_worker_provider_result(
        role,
        purpose,
        client,
        usage,
        task_id,
        agent_id,
        sequence,
    ):
        del client, usage, task_id, agent_id, sequence
        if role == "worker" and purpose == "evidence_extraction":
            provider_result_observed.set()
            await finish_telemetry.wait()

    def search(**kwargs):
        nonlocal search_count
        search_count += 1
        return _SearchResult(query=kwargs["query"], hits=[_hit("task_1")])

    orchestrator = MultiAgentResearchOrchestrator(
        planner_llm=planner,
        task_llm=task_client,
        worker_llm=worker,
        search_runner=search,
        limits=_limits(),
        on_llm_usage=pause_after_worker_provider_result,
        search_runner_in_thread=False,
    )

    interrupted = asyncio.create_task(orchestrator.run("Question one"))
    await asyncio.wait_for(provider_result_observed.wait(), timeout=1)
    interrupted.cancel()
    with suppress(asyncio.CancelledError):
        await interrupted
    finish_telemetry.set()

    result = await asyncio.wait_for(orchestrator.run("Question one"), timeout=1)

    assert result.task_artifacts[0].status == TaskStatus.COMPLETE
    assert search_count == 1
    assert len(planner.calls) == 1
    assert sum(schema is SearchAssignmentBatch for schema, *_ in task_client.calls) == 1
    assert sum(schema is TaskArtifact for schema, *_ in task_client.calls) == 1
    assert sum(schema is WorkerArtifact for schema, *_ in worker.calls) == 1


@pytest.mark.asyncio
async def test_task_global_rerank_uses_task_question_and_reviewer_gets_that_order(
    monkeypatch,
) -> None:
    from fs_explorer_api.search.query import IndexedQueryEngine

    spec = _task("task_1", "Common task-level rerank question")
    planner = _ScriptedClient("planner", lambda schema, _text: _plan(spec))
    claims = {
        "chunk-a": _source_claim(
            "task_1",
            document_id="doc-a",
            chunk_id="chunk-a",
            excerpt="Evidence A.",
        ),
        "chunk-b": _source_claim(
            "task_1",
            document_id="doc-b",
            chunk_id="chunk-b",
            excerpt="Evidence B.",
        ),
    }

    def task_responder(schema: type, text: str) -> object:
        if schema is SearchAssignmentBatch:
            if "WORKER ARTIFACTS SO FAR:\n[]" not in text:
                return SearchAssignmentBatch(
                    task_id="task_1",
                    stop=True,
                    stop_reason="Both searches completed.",
                    assignments=[],
                )
            return SearchAssignmentBatch(
                task_id="task_1",
                stop=False,
                stop_reason=None,
                assignments=[
                    SearchAssignment(
                        assignment_id=f"assignment-{suffix}",
                        task_id="task_1",
                        query=f"query-{suffix}",
                        objective=f"Find source {suffix}",
                        evidence_requirements=["criterion-task_1"],
                        excluded_queries=[],
                        as_of_date=None,
                        filters=None,
                    )
                    for suffix in ("a", "b")
                ],
            )
        assert schema is TaskArtifact
        return TaskArtifact(
            task_id="task_1",
            status=TaskStatus.COMPLETE,
            answer_fragment="Both sources support the conclusion.",
            covered_requirement_ids=["criterion-task_1"],
            uncovered_requirement_ids=[],
            claims=[claims["chunk-a"], claims["chunk-b"]],
            application_findings=[],
            conflicts=[],
            gaps=[],
            contributing_worker_ids=[],
        )

    def worker_responder(schema: type, text: str) -> object:
        assert schema is WorkerArtifact
        suffix = "a" if '"assignment_id":"assignment-a"' in text else "b"
        return WorkerArtifact(
            task_id="task_1",
            assignment_id=f"assignment-{suffix}",
            worker_id=f"model-worker-{suffix}",
            status=WorkerStatus.SUCCESS,
            searches_run=[f"query-{suffix}"],
            claims=[claims[f"chunk-{suffix}"]],
            gaps=[],
            cross_references=[],
            error_code=None,
            error_message=None,
        )

    rerank_queries: list[str] = []

    def fake_rank_candidates(*, query, documents, limit, diversify=True):
        del limit, diversify
        rerank_queries.append(query)
        by_chunk = {document.chunk_id: document for document in documents}
        return [(by_chunk["chunk-b"], 0.99), (by_chunk["chunk-a"], 0.98)]

    monkeypatch.setattr(
        IndexedQueryEngine,
        "rank_candidates",
        staticmethod(fake_rank_candidates),
    )

    def search(**kwargs):
        suffix = kwargs["query"].removeprefix("query-")
        return _SearchResult(
            query=kwargs["query"],
            hits=[
                _source_hit(
                    document_id=f"doc-{suffix}",
                    chunk_id=f"chunk-{suffix}",
                    relative_path=f"regulation_{suffix}.pdf",
                    article="7",
                    text=f"Evidence {suffix.upper()}.",
                    score=0.99 if suffix == "a" else 0.10,
                )
            ],
        )

    task_client = _ScriptedClient("task", task_responder)
    await MultiAgentResearchOrchestrator(
        planner_llm=planner,
        task_llm=task_client,
        worker_llm=_ScriptedClient("worker", worker_responder),
        search_runner=search,
        limits=replace(_limits(), max_worker_rounds=1),
        search_runner_in_thread=False,
    ).run("Use both searches")

    reviewer_text = next(
        history[0].text
        for schema, history, _prompt in task_client.calls
        if schema is TaskArtifact
    )
    verified_evidence = reviewer_text.split("SERVER-VERIFIED EVIDENCE:\n", 1)[1]

    assert rerank_queries == [spec.question]
    assert verified_evidence.index("chunk_id: chunk-b") < verified_evidence.index(
        "chunk_id: chunk-a"
    )


@pytest.mark.asyncio
async def test_task_review_preserves_verified_gap_and_exact_conflict_only() -> None:
    verified_gap = "missing-detail"
    spec = _task("task_1", "Compare the rule and exception")
    planner = _ScriptedClient("planner", lambda schema, _text: _plan(spec))
    claims = {
        "a": _source_claim(
            "task_1",
            document_id="doc-a",
            chunk_id="chunk-a",
            excerpt="Rule A applies.",
        ),
        "b": _source_claim(
            "task_1",
            document_id="doc-b",
            chunk_id="chunk-b",
            excerpt="Rule B creates an exception.",
        ),
    }
    verified_reference = "See Regulation B Article 9."
    sourced_conflict = "Rule A applies. Rule B creates an exception."
    unsourced_conflict = "Another authority silently overrides both provisions."

    def task_responder(schema: type, text: str) -> object:
        if schema is SearchAssignmentBatch:
            if "WORKER ARTIFACTS SO FAR:\n[]" not in text:
                return SearchAssignmentBatch(
                    task_id="task_1",
                    stop=True,
                    stop_reason="The relevant provisions were found.",
                    assignments=[],
                )
            return SearchAssignmentBatch(
                task_id="task_1",
                stop=False,
                stop_reason=None,
                assignments=[
                    SearchAssignment(
                        assignment_id=f"assignment-{suffix}",
                        task_id="task_1",
                        query=f"query-{suffix}",
                        objective=f"Find provision {suffix}",
                        evidence_requirements=[
                            "criterion-task_1",
                            verified_gap,
                        ],
                        excluded_queries=[],
                        as_of_date=None,
                        filters=None,
                    )
                    for suffix in ("a", "b")
                ],
            )
        assert schema is TaskArtifact
        return TaskArtifact(
            task_id="task_1",
            status=TaskStatus.COMPLETE,
            answer_fragment="The rule and exception must be reconciled.",
            covered_requirement_ids=["criterion-task_1"],
            uncovered_requirement_ids=[],
            claims=[claims["a"], claims["b"]],
            application_findings=[],
            conflicts=[sourced_conflict, unsourced_conflict],
            gaps=[verified_reference, verified_gap, "invented gap"],
            contributing_worker_ids=[],
        )

    def worker_responder(schema: type, text: str) -> object:
        assert schema is WorkerArtifact
        suffix = "a" if '"assignment_id":"assignment-a"' in text else "b"
        return WorkerArtifact(
            task_id="task_1",
            assignment_id=f"assignment-{suffix}",
            worker_id=f"model-worker-{suffix}",
            status=WorkerStatus.SUCCESS,
            searches_run=[f"query-{suffix}"],
            claims=[claims[suffix]],
            gaps=[verified_gap],
            cross_references=(
                [verified_reference, verified_gap] if suffix == "a" else []
            ),
            error_code=None,
            error_message=None,
        )

    def search(**kwargs):
        suffix = kwargs["query"].removeprefix("query-")
        if suffix == "a":
            hit = _source_hit(
                document_id="doc-a",
                chunk_id="chunk-a",
                relative_path="regulation_a.pdf",
                article="7",
                text=f"Rule A applies. {verified_reference} {verified_gap}",
                score=0.95,
            )
        else:
            hit = _source_hit(
                document_id="doc-b",
                chunk_id="chunk-b",
                relative_path="regulation_b.pdf",
                article="9",
                text="Rule B creates an exception.",
                score=0.94,
            )
        return _SearchResult(query=kwargs["query"], hits=[hit])

    result = await MultiAgentResearchOrchestrator(
        planner_llm=planner,
        task_llm=_ScriptedClient("task", task_responder),
        worker_llm=_ScriptedClient("worker", worker_responder),
        search_runner=search,
        limits=replace(_limits(), max_worker_rounds=1),
        search_runner_in_thread=False,
    ).run("Compare the rule and exception")

    artifact = result.task_artifacts[0]
    assert artifact.conflicts == [sourced_conflict]
    assert verified_reference in artifact.gaps
    assert verified_gap in artifact.gaps
    assert "invented gap" not in artifact.gaps
    assert unsourced_conflict not in artifact.conflicts


@pytest.mark.asyncio
async def test_workflow_streams_multi_agent_progress_and_prepares_final_context(
    monkeypatch,
) -> None:
    import fs_explorer_api.agent as agent_module

    from fs_explorer_api.workflow import (
        InputEvent,
        ResearchProgressEvent,
        get_run_agent,
        new_workflow,
    )

    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")

    async def inline_to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", inline_to_thread)
    monkeypatch.setattr(
        agent_module,
        "_run_planned_index_search",
        lambda **kwargs: _SearchResult(
            query=kwargs["query"],
            hits=[_hit("task_1")],
        ),
    )

    roles: dict[LLMRole, LLMClient] = {
        "planner": _ScriptedClient(
            "planner",
            lambda schema, _text: _plan(_task("task_1", "Question one")),
        ),
        "task": _ScriptedClient("task", _task_responder),
        "worker": _ScriptedClient("worker", _worker_responder),
        "final": _ScriptedClient("final", lambda schema, _text: object()),
    }
    workflow, resources = new_workflow(role_clients=roles)
    handler = workflow.run(
        start_event=InputEvent(
            task="Question one",
            folder="indexed",
            use_index=True,
            enable_semantic=True,
        )
    )
    events = [event async for event in handler.stream_events()]
    result = await handler
    agent = get_run_agent(resources)

    assert any(isinstance(event, ResearchProgressEvent) for event in events)
    assert result.error is None
    assert agent.prepared_indexed_evidence is True
    assert len(agent._chat_history) == 1
    assert "SERVER-VERIFIED FULL EVIDENCE" in agent._chat_history[0].text
    assert agent._chat_history[0].text.count("ORIGINAL QUESTION") == 1
    assert agent.final_model == "test/final"


class TestContextBudgetAndRunGuardrails:
    """Prompt size and runaway-cost ceilings that survive model-controlled depth."""

    def test_from_limits_mirrors_the_legacy_bounded_policy(self) -> None:
        limits = _limits()
        budget = ContextBudget.from_limits(limits)

        assert budget.worker_hit_chars == limits.worker_hit_chars
        assert budget.review_hit_chars == limits.review_hit_chars
        assert budget.final_evidence_limit == limits.final_evidence_limit
        assert budget.final_context_chars == limits.max_final_context_chars
        assert budget.artifact_text_chars == limits.max_artifact_text_chars

    def test_env_overrides_are_read_for_model_controlled_runs(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("FS_EXPLORER_MULTI_AGENT_MAX_FINAL_CONTEXT_CHARS", "1234")
        monkeypatch.setenv("FS_EXPLORER_MULTI_AGENT_WORKER_HIT_CHARS", "777")

        budget = ContextBudget.from_env()

        assert budget.final_context_chars == 1234
        assert budget.worker_hit_chars == 777

    def test_oversized_prompts_are_truncated_rather_than_rejected(self) -> None:
        orchestrator = MultiAgentResearchOrchestrator(
            planner_llm=_ScriptedClient("planner", lambda schema, _text: object()),
            task_llm=_ScriptedClient("task", _task_responder),
            worker_llm=_ScriptedClient("worker", _worker_responder),
            search_runner=lambda **kwargs: _SearchResult(
                query=kwargs["query"], hits=[]
            ),
            search_runner_in_thread=False,
        )
        object.__setattr__(
            orchestrator,
            "_context",
            replace(orchestrator._context, max_prompt_chars=500),
        )

        fitted = orchestrator._fit_prompt(
            "x" * 10_000,
            system_prompt="system",
            role="worker",
            purpose="worker_report",
        )

        assert len(fitted) <= 500
        assert "TRUNCATED" in fitted

    def test_parent_reports_shed_evidence_instead_of_overflowing(self) -> None:
        orchestrator = MultiAgentResearchOrchestrator(
            planner_llm=_ScriptedClient("planner", lambda schema, _text: object()),
            task_llm=_ScriptedClient("task", _task_responder),
            worker_llm=_ScriptedClient("worker", _worker_responder),
            search_runner=lambda **kwargs: _SearchResult(
                query=kwargs["query"], hits=[]
            ),
            search_runner_in_thread=False,
        )
        artifact = TaskArtifact(
            task_id="task_1",
            status=TaskStatus.COMPLETE,
            answer_fragment="Fragment",
            covered_requirement_ids=["criterion-task_1"],
            uncovered_requirement_ids=[],
            claims=[
                _source_claim(
                    "task_1",
                    document_id="doc-task_1",
                    chunk_id=f"chunk-{index}",
                    excerpt="Evidence excerpt. " * 40,
                )
                for index in range(12)
            ],
            application_findings=[],
            conflicts=[],
            gaps=[],
            contributing_worker_ids=["worker-task_1"],
        )

        rendered = orchestrator._task_reports_for_parent([artifact], max_chars=2_000)

        assert len(rendered) <= 2_000
        assert "withheld_evidence_count" in rendered

    @pytest.mark.asyncio
    async def test_final_context_respects_the_configured_ceiling(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("FS_EXPLORER_MULTI_AGENT_MAX_FINAL_CONTEXT_CHARS", "1500")
        planner = _ScriptedClient(
            "planner",
            lambda schema, _text: _plan(_task("task_1", "Question one")),
        )

        result = await MultiAgentResearchOrchestrator(
            planner_llm=planner,
            task_llm=_ScriptedClient("task", _task_responder),
            worker_llm=_ScriptedClient("worker", _worker_responder),
            search_runner=lambda **kwargs: _SearchResult(
                query=kwargs["query"], hits=[_hit("task_1")]
            ),
            search_runner_in_thread=False,
        ).run("Question one")

        assert len(result.final_context) <= 1_500

    @pytest.mark.asyncio
    async def test_token_budget_stops_research_but_still_answers(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("FS_EXPLORER_MULTI_AGENT_RUN_TOKEN_BUDGET", "1")
        progress: list[ResearchProgress] = []
        searches: list[str] = []
        planner = _ScriptedClient(
            "planner",
            lambda schema, _text: _plan(_task("task_1", "Question one")),
        )

        result = await MultiAgentResearchOrchestrator(
            planner_llm=planner,
            task_llm=_ScriptedClient("task", _task_responder),
            worker_llm=_ScriptedClient("worker", _worker_responder),
            search_runner=lambda **kwargs: (
                searches.append(kwargs["query"])
                or _SearchResult(query=kwargs["query"], hits=[_hit("task_1")])
            ),
            on_progress=progress.append,
            search_runner_in_thread=False,
        ).run("Question one")

        assert searches == []
        assert any(event.kind == "budget_exhausted" for event in progress)
        assert result.incomplete is True
        assert result.plan is not None

    @pytest.mark.asyncio
    async def test_coordinator_rejection_loop_is_bounded(self, monkeypatch) -> None:
        monkeypatch.setenv("FS_EXPLORER_MULTI_AGENT_MAX_COORDINATOR_DECISIONS", "3")
        planner = _ScriptedClient(
            "planner",
            lambda schema, _text: _plan(_task("task_1", "Question one")),
        )

        def never_executable(schema: type, text: str) -> object:
            # A coordinator that keeps returning a dispatch that is neither
            # executable nor an explicit stop used to loop forever, paying for
            # a larger prompt on every turn.
            assert schema is SearchAssignmentBatch
            return SearchAssignmentBatch(
                task_id=_task_id_from_text(text),
                stop=False,
                stop_reason=None,
                assignments=[],
            )

        task_client = _ScriptedClient("task", never_executable)

        result = await MultiAgentResearchOrchestrator(
            planner_llm=planner,
            task_llm=task_client,
            worker_llm=_ScriptedClient("worker", _worker_responder),
            search_runner=lambda **kwargs: _SearchResult(
                query=kwargs["query"], hits=[]
            ),
            search_runner_in_thread=False,
        ).run("Question one")

        coordinator_calls = [
            call for call in task_client.calls if call[0] is SearchAssignmentBatch
        ]
        assert len(coordinator_calls) == 3
        assert result.incomplete is True
