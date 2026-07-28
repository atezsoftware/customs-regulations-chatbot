"""Tests for bounded, context-isolated multi-agent indexed research."""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Callable

import pytest

from fs_explorer_api.llm import ChatTurn, LLMUsage
from fs_explorer_api.multi_agent import (
    EvidenceRecord,
    MultiAgentResearchOrchestrator,
    ResearchLimits,
    ResearchProgress,
)
from fs_explorer_api.orchestration_models import (
    EvidenceClaim,
    EvidenceConfidence,
    GlobalPlan,
    PlanMode,
    SearchAssignment,
    SearchAssignmentBatch,
    TaskArtifact,
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
    return TaskSpec(
        task_id=task_id,
        question=question,
        purpose=f"Resolve {task_id}",
        expected_output=f"Supported conclusion for {task_id}",
        success_criteria=[f"criterion-{task_id}"],
        depends_on=list(depends_on or []),
        required=True,
        as_of_date=None,
        filters=None,
    )


def _plan(*tasks: TaskSpec) -> GlobalPlan:
    return GlobalPlan(
        version="1",
        mode=PlanMode.DIRECT if len(tasks) == 1 else PlanMode.DECOMPOSED,
        normalized_question="Normalized question",
        answer_requirements=["Answer all requested parts."],
        tasks=list(tasks),
        synthesis_requirements=[],
        assumptions=[],
    )


def _claim(task_id: str, *, excerpt: str | None = None) -> EvidenceClaim:
    return EvidenceClaim(
        claim=f"Supported claim for {task_id}",
        document_id=f"doc-{task_id}",
        chunk_id=f"chunk-{task_id}",
        readable_title=f"Model title {task_id}",
        locator="Model locator",
        evidence_excerpt=excerpt or f"Evidence text for {task_id}.",
        supports_success_criteria=[f"criterion-{task_id}"],
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
        claim=f"Supported claim from {chunk_id}",
        document_id=document_id,
        chunk_id=chunk_id,
        readable_title=title,
        locator=locator,
        evidence_excerpt=excerpt,
        supports_success_criteria=[f"criterion-{task_id}"],
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
        covered_success_criteria=[f"criterion-{task_id}"],
        uncovered_success_criteria=[],
        claims=[_claim(task_id)],
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
                covered_success_criteria=[],
                uncovered_success_criteria=[f"criterion-{task_id}"],
                claims=[],
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
            "supports_success_criteria": ["criterion " * 500] * 12,
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
            covered_success_criteria=["covered " * 500] * 12,
            uncovered_success_criteria=[
                "MATERIAL_UNCOVERED_CRITERION",
                *(["uncovered " * 500] * 11),
            ],
            claims=[long_claim],
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
        limits=replace(_limits(), final_chunk_chars=650),
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

    searched_tasks = [
        re.search(r"(task_\d+)", query).group(1)  # type: ignore[union-attr]
        for query in searches
    ]
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
            covered_success_criteria=["criterion-task_1"],
            uncovered_success_criteria=[],
            claims=[claims["chunk-a"], claims["chunk-b"]],
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
async def test_task_review_preserves_verified_gap_and_sourced_conflict_only() -> None:
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
    verified_gap = "missing-detail"
    sourced_conflict = (
        "The provisions conflict [regulation a, Article 7] [regulation b, Article 9]."
    )
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
            covered_success_criteria=["criterion-task_1"],
            uncovered_success_criteria=[],
            claims=[claims["a"], claims["b"]],
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
            cross_references=[verified_reference] if suffix == "a" else [],
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
                text=f"Rule A applies. {verified_reference}",
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

    roles = {
        "planner": _ScriptedClient(
            "planner",
            lambda schema, _text: _plan(_task("task_1", "Question one")),
        ),
        "task": _ScriptedClient("task", _task_responder),
        "worker": _ScriptedClient("worker", _worker_responder),
        "final": _ScriptedClient("final", lambda schema, _text: object()),
    }
    workflow, resources = new_workflow(
        role_clients=roles,
        multi_agent_enabled=True,
    )
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
    assert "SERVER-VERIFIED FULL EVIDENCE" in agent._chat_history[1].text
    assert agent.final_model == "test/final"
