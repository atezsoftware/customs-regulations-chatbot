"""End-to-end scenario regressions for typed multi-agent research."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Callable

import pytest

from fs_explorer_api.agent import FsExplorerAgent
from fs_explorer_api.llm import ChatTurn, LLMUsage
from fs_explorer_api.multi_agent import (
    MultiAgentResearchOrchestrator,
    MultiAgentResearchResult,
    ResearchLimits,
    ResearchProgress,
)
from fs_explorer_api.orchestration_models import (
    AnswerRequirement,
    AnswerRequirementKind,
    ArtifactValidationError,
    DecisionBranch,
    DerivedConclusion,
    EvidenceClaim,
    EvidenceConfidence,
    EvidenceRequirement,
    EvidenceRequirementKind,
    ExecutionStrategy,
    GlobalPlan,
    MaterialUnknown,
    PlanMode,
    PlanValidationError,
    ProblemType,
    ScenarioFact,
    ScenarioSpec,
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
    validate_global_plan,
    validate_task_artifact,
)
from fs_explorer_api.orchestration_prompts import (
    SCENARIO_FINAL_SYNTHESIS_SYSTEM_PROMPT,
)
from fs_explorer_api.search import IndexedQueryEngine, SearchHit


@pytest.fixture(autouse=True)
def _deterministic_reranking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        IndexedQueryEngine,
        "rank_candidates",
        staticmethod(
            lambda *, query, documents, limit, diversify=True: [
                (document, document.combined_score) for document in documents[:limit]
            ]
        ),
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
        *,
        streamed_text: str = "final answer",
    ) -> None:
        self.role = role
        self.provider = "openrouter"
        self.model = f"test/{role}"
        self._responder = responder
        self._streamed_text = streamed_text
        self.structured_calls: list[tuple[type, str, str]] = []
        self.stream_calls: list[tuple[list[ChatTurn], str]] = []

    async def generate_structured(
        self, history, system_prompt, schema, *, thinking_level=None
    ):
        del thinking_level
        text = history[0].text
        self.structured_calls.append((schema, text, system_prompt))
        return self._responder(schema, text), LLMUsage(input_tokens=20, output_tokens=5)

    async def stream_text(self, history, system_prompt, *, thinking_level=None):
        del thinking_level
        self.stream_calls.append((list(history), system_prompt))
        yield self._streamed_text

    def last_stream_usage(self):
        return LLMUsage(input_tokens=20, output_tokens=5)


def _requirement(
    requirement_id: str,
    description: str,
    *,
    kind: AnswerRequirementKind = AnswerRequirementKind.OUTCOME,
) -> AnswerRequirement:
    return AnswerRequirement(
        requirement_id=requirement_id,
        kind=kind,
        description=description,
        required=True,
    )


def _evidence_task(
    task_id: str,
    question: str,
    requirement_ids: list[str],
    evidence: list[tuple[str, EvidenceRequirementKind, str]],
    *,
    output_id: str | None = None,
) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        kind=TaskKind.EVIDENCE,
        issue=question,
        search_question=question,
        requirement_ids=requirement_ids,
        fact_ids=[],
        unknown_ids=[],
        branch_ids=[],
        evidence_requirements=[
            EvidenceRequirement(
                evidence_requirement_id=evidence_id,
                kind=kind,
                description=description,
                requirement_ids=requirement_ids,
            )
            for evidence_id, kind, description in evidence
        ],
        consumes=[],
        produces=[
            TaskOutput(
                output_id=output_id or f"out_{task_id}",
                description=f"Verified evidence produced by {task_id}.",
            )
        ],
        required=True,
        as_of_date=None,
        filters=None,
    )


def _application_task(
    task_id: str,
    requirement_ids: list[str],
    *,
    fact_ids: list[str],
    unknown_ids: list[str],
    branch_ids: list[str],
    evidence_task_id: str = "shared_rule",
    evidence_output_id: str = "rule_evidence",
) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        kind=TaskKind.APPLICATION,
        issue=f"Apply verified law to {', '.join(fact_ids)}.",
        search_question=None,
        requirement_ids=requirement_ids,
        fact_ids=fact_ids,
        unknown_ids=unknown_ids,
        branch_ids=branch_ids,
        evidence_requirements=[],
        consumes=[
            TaskOutputRef(task_id=evidence_task_id, output_id=evidence_output_id)
        ],
        produces=[
            TaskOutput(
                output_id=f"out_{task_id}",
                description=f"Applied conclusion for {task_id}.",
            )
        ],
        required=True,
        as_of_date=None,
        filters=None,
    )


def _plan(
    question: str,
    requirements: list[AnswerRequirement],
    tasks: list[TaskSpec],
    *,
    problem_type: ProblemType = ProblemType.LOOKUP,
    strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE,
    scenario: ScenarioSpec | None = None,
) -> GlobalPlan:
    return GlobalPlan(
        version="3",
        problem_type=problem_type,
        mode=PlanMode.DIRECT if len(tasks) == 1 else PlanMode.DECOMPOSED,
        execution_strategy=strategy,
        normalized_question=question,
        answer_requirements=requirements,
        scenario=scenario,
        tasks=tasks,
        synthesis_requirements=[],
        assumptions=[],
    )


def _claim(
    task_id: str,
    requirement_ids: list[str],
    evidence_requirement_ids: list[str],
    key: str,
    excerpt: str,
    *,
    claim_id: str | None = None,
) -> EvidenceClaim:
    return EvidenceClaim(
        claim_id=claim_id or f"claim_{task_id}_{key}",
        claim=f"Supported proposition for {task_id}.",
        document_id=f"doc-{key}",
        chunk_id=f"chunk-{key}",
        readable_title=f"{key} regulation",
        locator="Article 7",
        evidence_excerpt=excerpt,
        requirement_ids=requirement_ids,
        evidence_requirement_ids=evidence_requirement_ids,
        fact_ids=[],
        confidence=EvidenceConfidence.HIGH,
        effective_start_date=None,
        effective_end_date=None,
    )


def _artifact(
    task_id: str,
    requirement_ids: list[str],
    *,
    claims: list[EvidenceClaim] | None = None,
    findings: list[DerivedConclusion] | None = None,
    uncovered: list[str] | None = None,
    gaps: list[str] | None = None,
    workers: list[str] | None = None,
) -> TaskArtifact:
    claims = claims or []
    findings = findings or []
    uncovered = uncovered or []
    return TaskArtifact(
        task_id=task_id,
        status=(
            TaskStatus.COMPLETE
            if requirement_ids and not uncovered
            else TaskStatus.PARTIAL
            if claims or findings
            else TaskStatus.FAILED
        ),
        answer_fragment=(
            " ".join([claim.claim for claim in claims])
            or " ".join(finding.finding for finding in findings)
            or None
        ),
        covered_requirement_ids=[
            requirement_id
            for requirement_id in requirement_ids
            if requirement_id not in uncovered
        ],
        uncovered_requirement_ids=uncovered,
        claims=claims,
        application_findings=findings,
        conflicts=[],
        gaps=gaps or [],
        contributing_worker_ids=workers or [],
    )


def _hit(key: str, text: str) -> SearchHit:
    return SearchHit(
        doc_id=f"doc-{key}",
        relative_path=f"{key}_regulation.pdf",
        absolute_path=f"/private/{key}.pdf",
        position=7,
        text=text,
        semantic_score=0.95,
        metadata_score=0,
        score=0.95,
        matched_by="semantic",
        chunk_id=f"chunk-{key}",
        chunk_type="text",
        metadata={"article_no": "7"},
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


def _task_id(text: str) -> str:
    match = re.search(r'TASK SPEC:\s*\{"task_id":"([^"]+)"', text)
    if not match:
        raise AssertionError(f"No TaskSpec ID in prompt:\n{text}")
    return match.group(1)


@pytest.mark.asyncio
async def test_precise_lookup_uses_direct_single_pass_three_call_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The simple path is planner + one worker + final synthesis."""

    import fs_explorer_api.agent as agent_module

    question = "What is the filing deadline?"
    requirement = _requirement("req_deadline", "State the filing deadline.")
    task = _evidence_task(
        "lookup",
        question,
        ["req_deadline"],
        [
            (
                "ev_deadline",
                EvidenceRequirementKind.GOVERNING_RULE,
                requirement.description,
            )
        ],
    )
    plan = _plan(
        question,
        [requirement],
        [task],
        strategy=ExecutionStrategy.SINGLE_PASS,
    )
    planner = _ScriptedClient("planner", lambda schema, text: plan)
    claim = _claim(
        "lookup",
        ["req_deadline"],
        ["ev_deadline"],
        "deadline",
        "The filing deadline is 30 days.",
    )
    worker = _ScriptedClient(
        "worker",
        lambda schema, text: WorkerArtifact(
            task_id="lookup",
            assignment_id="lookup_single_pass",
            worker_id="worker-lookup-lookup_single_pass",
            status=WorkerStatus.SUCCESS,
            searches_run=[question],
            claims=[claim],
            gaps=[],
            cross_references=[],
            error_code=None,
            error_message=None,
        ),
    )
    task_client = _ScriptedClient(
        "task",
        lambda schema, text: pytest.fail(
            "Covered single-pass lookup called a coordinator or reviewer"
        ),
    )
    final = _ScriptedClient(
        "final",
        lambda schema, text: pytest.fail("Final call must stream"),
        streamed_text="The deadline is 30 days. [deadline regulation, Article 7]",
    )

    async def inline_to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", inline_to_thread)
    monkeypatch.setenv("FS_EXPLORER_MULTI_AGENT_MAX_LLM_CALLS", "3")
    monkeypatch.setattr(
        agent_module,
        "_run_planned_index_search",
        lambda **kwargs: _SearchResult(
            query=kwargs["query"],
            hits=[_hit("deadline", "The filing deadline is 30 days.")],
        ),
    )
    agent = FsExplorerAgent(
        llm_client=final,
        role_clients={
            "planner": planner,
            "task": task_client,
            "worker": worker,
            "final": final,
        },
    )
    result = await agent.prepare_multi_agent_indexed(question)
    answer = "".join([chunk async for chunk in agent.stream_final_answer()])

    assert result.plan.mode == PlanMode.DIRECT
    assert result.plan.execution_strategy == ExecutionStrategy.SINGLE_PASS
    assert result.task_artifacts[0].status == TaskStatus.COMPLETE
    assert task_client.structured_calls == []
    assert answer.startswith("The deadline is 30 days.")
    assert (
        len(planner.structured_calls)
        + len(worker.structured_calls)
        + len(final.stream_calls)
        == 3
    )


def _two_scenario_plan() -> tuple[GlobalPlan, dict[str, str]]:
    markers = {
        "shared_rule": "SHARED-RULE-MARKER",
        "apply_a": "SCENARIO-A-ONLY",
        "apply_b": "SCENARIO-B-ONLY",
    }
    requirements = [
        _requirement(
            "req_rule", "Establish the shared rule.", kind=AnswerRequirementKind.RULE
        ),
        _requirement("req_a", "Apply the rule to scenario A."),
        _requirement("req_b", "Apply the rule to scenario B."),
    ]
    scenario = ScenarioSpec(
        jurisdiction="TR",
        law_as_of_date=None,
        facts=[
            ScenarioFact(
                fact_id="fact_a",
                description=markers["apply_a"],
                requirement_ids=["req_a"],
            ),
            ScenarioFact(
                fact_id="fact_b",
                description=markers["apply_b"],
                requirement_ids=["req_b"],
            ),
        ],
        material_unknowns=[],
        decision_branches=[
            DecisionBranch(
                branch_id="branch_a",
                condition="Scenario A facts apply.",
                consequence="Resolve scenario A.",
                requirement_ids=["req_a"],
            ),
            DecisionBranch(
                branch_id="branch_b",
                condition="Scenario B facts apply.",
                consequence="Resolve scenario B.",
                requirement_ids=["req_b"],
            ),
        ],
    )
    shared = _evidence_task(
        "shared_rule",
        f"Find {markers['shared_rule']}",
        ["req_rule"],
        [("ev_rule", EvidenceRequirementKind.GOVERNING_RULE, "Find the shared rule.")],
        output_id="rule_evidence",
    )
    tasks = [
        shared,
        _application_task(
            "apply_a",
            ["req_a"],
            fact_ids=["fact_a"],
            unknown_ids=[],
            branch_ids=["branch_a"],
        ),
        _application_task(
            "apply_b",
            ["req_b"],
            fact_ids=["fact_b"],
            unknown_ids=[],
            branch_ids=["branch_b"],
        ),
    ]
    return (
        _plan(
            "Evaluate scenario A and scenario B.",
            requirements,
            tasks,
            problem_type=ProblemType.SCENARIO_APPLICATION,
            scenario=scenario,
        ),
        markers,
    )


@pytest.mark.asyncio
async def test_scenario_result_selects_scenario_final_synthesis_prompt() -> None:
    plan, _markers = _two_scenario_plan()
    final = _ScriptedClient(
        "final",
        lambda schema, text: pytest.fail("Final synthesis must stream text."),
        streamed_text="Conditional scenario answer.",
    )
    agent = FsExplorerAgent(
        llm_client=final,
        role_clients={
            "planner": final,
            "task": final,
            "worker": final,
            "final": final,
        },
    )
    agent._multi_agent_result = MultiAgentResearchResult(
        plan=plan,
        task_artifacts=(),
        final_context="Validated scenario artifacts.",
        evidence_sources=(),
        incomplete=False,
        used_plan_fallback=False,
    )
    agent._multi_agent_final_llm = final
    agent._chat_history = [
        ChatTurn(role="user", text="Original scenario question."),
        ChatTurn(role="user", text="Validated scenario artifacts."),
    ]

    answer = "".join([chunk async for chunk in agent.stream_final_answer()])

    assert answer == "Conditional scenario answer."
    assert final.stream_calls[0][1] == SCENARIO_FINAL_SYNTHESIS_SYSTEM_PROMPT
    assert "conditional branch" in final.stream_calls[0][0][-1].text


def test_scenario_fallback_keeps_separate_evidence_only_deliverables() -> None:
    plan, _markers = _two_scenario_plan()
    claim = _claim(
        "shared_rule",
        ["req_rule"],
        ["ev_rule"],
        "shared",
        "The shared governing rule applies.",
        claim_id="claim_shared",
    )
    finding = DerivedConclusion(
        conclusion_id="scenario_a",
        finding="The rule produces the scenario A outcome.",
        requirement_ids=["req_a"],
        fact_ids=["fact_a"],
        branch_ids=["branch_a"],
        supporting_claim_ids=[claim.claim_id],
        dependency_refs=[
            TaskOutputRef(task_id="shared_rule", output_id="rule_evidence")
        ],
        confidence=EvidenceConfidence.HIGH,
        limitations=[],
    )
    final = _ScriptedClient("final", lambda _schema, _text: None)
    agent = FsExplorerAgent(llm_client=final)
    agent._multi_agent_result = MultiAgentResearchResult(
        plan=plan,
        task_artifacts=(
            _artifact("shared_rule", ["req_rule"], claims=[claim]),
            _artifact("apply_a", ["req_a"], findings=[finding]),
        ),
        final_context="",
        evidence_sources=(
            {
                "title": claim.readable_title,
                "snippet": claim.evidence_excerpt,
                "document_id": claim.document_id,
                "chunk_id": claim.chunk_id,
                "score": 1.0,
                "locator": claim.locator,
            },
        ),
        incomplete=False,
        used_plan_fallback=False,
    )

    answer = agent._multi_agent_fallback_answer()

    assert answer is not None
    assert finding.finding in answer
    assert claim.claim in answer
    assert "Additional verified rules and procedures" in answer


@pytest.mark.asyncio
async def test_shared_evidence_fans_out_to_isolated_application_nodes() -> None:
    plan, markers = _two_scenario_plan()
    planner = _ScriptedClient("planner", lambda schema, text: plan)
    shared_claim = _claim(
        "shared_rule",
        ["req_rule"],
        ["ev_rule"],
        "shared",
        f"Evidence for {markers['shared_rule']}.",
        claim_id="claim_shared_rule",
    )

    def task_response(schema: type, text: str) -> object:
        task_id = _task_id(text)
        if schema is SearchAssignmentBatch:
            assert task_id == "shared_rule"
            return SearchAssignmentBatch(
                task_id=task_id,
                stop=False,
                stop_reason=None,
                assignments=[
                    SearchAssignment(
                        assignment_id="search-shared",
                        task_id=task_id,
                        query="query-shared_rule",
                        objective="Find the shared rule.",
                        evidence_requirements=["ev_rule"],
                        excluded_queries=[],
                        as_of_date=None,
                        filters=None,
                    )
                ],
            )
        assert schema is TaskArtifact
        if task_id == "shared_rule":
            return _artifact(
                task_id,
                ["req_rule"],
                claims=[shared_claim],
                workers=["worker-shared_rule-search-shared"],
            )
        suffix = task_id[-1]
        finding = DerivedConclusion(
            conclusion_id=f"conclusion_{suffix}",
            finding=f"Applied conclusion for scenario {suffix.upper()}.",
            requirement_ids=[f"req_{suffix}"],
            fact_ids=[f"fact_{suffix}"],
            branch_ids=[f"branch_{suffix}"],
            supporting_claim_ids=["shared_rule_search-shared_claim_1"],
            dependency_refs=[
                TaskOutputRef(task_id="shared_rule", output_id="rule_evidence")
            ],
            confidence=EvidenceConfidence.HIGH,
            limitations=[],
        )
        return _artifact(task_id, [f"req_{suffix}"], findings=[finding])

    worker = _ScriptedClient(
        "worker",
        lambda schema, text: WorkerArtifact(
            task_id="shared_rule",
            assignment_id="search-shared",
            worker_id="worker-shared_rule-search-shared",
            status=WorkerStatus.SUCCESS,
            searches_run=["query-shared_rule"],
            claims=[shared_claim],
            gaps=[],
            cross_references=[],
            error_code=None,
            error_message=None,
        ),
    )
    task_client = _ScriptedClient("task", task_response)
    searches: list[str] = []

    def search(**kwargs):
        searches.append(kwargs["query"])
        return _SearchResult(
            query=kwargs["query"],
            hits=[_hit("shared", f"Evidence for {markers['shared_rule']}.")],
        )

    result = await MultiAgentResearchOrchestrator(
        planner_llm=planner,
        task_llm=task_client,
        worker_llm=worker,
        search_runner=search,
        limits=_limits(),
        search_runner_in_thread=False,
    ).run(plan.normalized_question)

    assert searches == ["query-shared_rule"]
    assert {artifact.task_id for artifact in result.task_artifacts} == {
        "shared_rule",
        "apply_a",
        "apply_b",
    }
    assert all(
        artifact.status == TaskStatus.COMPLETE for artifact in result.task_artifacts
    )
    assert all(
        artifact.application_findings
        for artifact in result.task_artifacts
        if artifact.task_id.startswith("apply_")
    )
    inputs = [text for _schema, text, _prompt in task_client.structured_calls]
    apply_a_inputs = [text for text in inputs if '"task_id":"apply_a"' in text]
    apply_b_inputs = [text for text in inputs if '"task_id":"apply_b"' in text]
    assert apply_a_inputs and apply_b_inputs
    assert all(
        markers["shared_rule"] in text for text in apply_a_inputs + apply_b_inputs
    )
    assert all(markers["apply_b"] not in text for text in apply_a_inputs)
    assert all(markers["apply_a"] not in text for text in apply_b_inputs)


@pytest.mark.asyncio
async def test_exception_cross_reference_prevents_single_pass_early_completion() -> (
    None
):
    requirement = _requirement("req_scope", "Resolve the rule and exception.")
    question = "Determine the rule and its Article 9 exception."
    task = _evidence_task(
        "scope",
        question,
        ["req_scope"],
        [
            ("ev_main", EvidenceRequirementKind.GOVERNING_RULE, "Find the main rule."),
            ("ev_exception", EvidenceRequirementKind.EXCEPTION, "Resolve Article 9."),
        ],
    )
    plan = _plan(
        question,
        [requirement],
        [task],
        strategy=ExecutionStrategy.SINGLE_PASS,
    )
    planner = _ScriptedClient("planner", lambda schema, text: plan)
    main_claim = _claim(
        "scope", ["req_scope"], ["ev_main"], "main", "The main rule applies."
    )
    exception_claim = _claim(
        "scope",
        ["req_scope"],
        ["ev_exception"],
        "exception",
        "Article 9 creates an exception.",
    )

    def task_response(schema: type, text: str) -> object:
        if schema is SearchAssignmentBatch:
            return SearchAssignmentBatch(
                task_id="scope",
                stop=False,
                stop_reason=None,
                assignments=[
                    SearchAssignment(
                        assignment_id="article-9-follow-up",
                        task_id="scope",
                        query="Article 9 exception text",
                        objective="Resolve the incorporated exception.",
                        evidence_requirements=["ev_exception"],
                        excluded_queries=[question],
                        as_of_date=None,
                        filters=None,
                    )
                ],
            )
        return _artifact(
            "scope",
            ["req_scope"],
            claims=[main_claim, exception_claim],
            workers=[
                "worker-scope-scope_single_pass",
                "worker-scope-article-9-follow-up",
            ],
        )

    def worker_response(schema: type, text: str) -> object:
        follow_up = "article-9-follow-up" in text
        return WorkerArtifact(
            task_id="scope",
            assignment_id="article-9-follow-up" if follow_up else "scope_single_pass",
            worker_id=(
                "worker-scope-article-9-follow-up"
                if follow_up
                else "worker-scope-scope_single_pass"
            ),
            status=WorkerStatus.SUCCESS,
            searches_run=["Article 9 exception text" if follow_up else question],
            claims=[exception_claim if follow_up else main_claim],
            gaps=[] if follow_up else ["ev_exception"],
            cross_references=[] if follow_up else ["Article 9"],
            error_code=None,
            error_message=None,
        )

    task_client = _ScriptedClient("task", task_response)
    searches: list[str] = []

    def search(**kwargs):
        query = kwargs["query"]
        searches.append(query)
        return _SearchResult(
            query=query,
            hits=(
                [_hit("main", "The main rule applies. See Article 9.")]
                if query == question
                else [_hit("exception", "Article 9 creates an exception.")]
            ),
        )

    result = await MultiAgentResearchOrchestrator(
        planner_llm=planner,
        task_llm=task_client,
        worker_llm=_ScriptedClient("worker", worker_response),
        search_runner=search,
        limits=_limits(),
        search_runner_in_thread=False,
    ).run(question)

    assert searches == [question, "Article 9 exception text"]
    assert [schema for schema, _text, _prompt in task_client.structured_calls] == [
        SearchAssignmentBatch,
        TaskArtifact,
    ]
    assert result.task_artifacts[0].status == TaskStatus.COMPLETE


def test_topic_outline_scenario_plan_is_rejected() -> None:
    """Research headings alone are not a scenario decision graph."""

    valid, _markers = _two_scenario_plan()
    topic_outline = valid.model_copy(
        update={
            "tasks": [
                valid.tasks[0],
                _evidence_task(
                    "exceptions_topic",
                    "List exceptions.",
                    ["req_a", "req_b"],
                    [
                        (
                            "ev_exceptions",
                            EvidenceRequirementKind.EXCEPTION,
                            "List exceptions as a topic.",
                        )
                    ],
                ),
            ]
        }
    )

    with pytest.raises(
        PlanValidationError,
        match="scenario plans require at least one application task",
    ):
        validate_global_plan(topic_outline)


def test_every_required_answer_requirement_must_be_mapped() -> None:
    requirement = _requirement("req_rule", "State the rule.")
    omitted = _requirement("req_deadline", "State the deadline.")
    task = _evidence_task(
        "lookup",
        "Find the rule.",
        ["req_rule"],
        [("ev_rule", EvidenceRequirementKind.GOVERNING_RULE, "Find the rule.")],
    )
    invalid = _plan("Find the rule and deadline.", [requirement, omitted], [task])

    with pytest.raises(
        PlanValidationError,
        match="required answer requirements are not mapped to tasks: req_deadline",
    ):
        validate_global_plan(invalid)


def test_invalid_derived_conclusion_references_are_rejected() -> None:
    plan, _markers = _two_scenario_plan()
    dependency_claim = _claim(
        "shared_rule",
        ["req_rule"],
        ["ev_rule"],
        "shared",
        "The shared rule applies.",
        claim_id="claim_shared_rule",
    )
    dependency = _artifact(
        "shared_rule", ["req_rule"], claims=[dependency_claim], workers=["worker-1"]
    )
    invalid_finding = DerivedConclusion(
        conclusion_id="conclusion_a",
        finding="Scenario A qualifies.",
        requirement_ids=["req_a"],
        fact_ids=["fact_a"],
        branch_ids=["branch_a"],
        supporting_claim_ids=["invented_claim"],
        dependency_refs=[
            TaskOutputRef(task_id="shared_rule", output_id="invented_output")
        ],
        confidence=EvidenceConfidence.HIGH,
        limitations=[],
    )
    artifact = _artifact("apply_a", ["req_a"], findings=[invalid_finding])

    with pytest.raises(ArtifactValidationError) as exc_info:
        validate_task_artifact(
            artifact,
            plan=plan,
            dependency_artifacts=[dependency],
        )

    assert "supporting_claim_ids references unknown IDs: invented_claim" in str(
        exc_info.value
    )
    assert "references unavailable dependency output" in str(exc_info.value)


@pytest.mark.asyncio
async def test_missing_material_fact_yields_conditional_limitation_in_final_context() -> (
    None
):
    requirements = [
        _requirement(
            "req_rule", "State the exemption rule.", kind=AnswerRequirementKind.RULE
        ),
        _requirement("req_outcome", "Apply the exemption conditionally."),
        _requirement(
            "req_limit",
            "State the missing material fact.",
            kind=AnswerRequirementKind.LIMITATION,
        ),
    ]
    unknown_text = "Whether the shipment is non-commercial remains unknown."
    scenario = ScenarioSpec(
        jurisdiction="TR",
        law_as_of_date=None,
        facts=[
            ScenarioFact(
                fact_id="fact_shipment",
                description="The user has a shipment.",
                requirement_ids=["req_outcome"],
            )
        ],
        material_unknowns=[
            MaterialUnknown(
                unknown_id="unknown_noncommercial",
                description=unknown_text,
                why_material="It determines whether the exemption applies.",
                requirement_ids=["req_outcome", "req_limit"],
            )
        ],
        decision_branches=[
            DecisionBranch(
                branch_id="branch_noncommercial",
                condition="The shipment is non-commercial.",
                consequence="The exemption may apply.",
                requirement_ids=["req_outcome"],
            ),
            DecisionBranch(
                branch_id="branch_commercial",
                condition="The shipment is commercial.",
                consequence="The exemption does not apply.",
                requirement_ids=["req_outcome"],
            ),
        ],
    )
    evidence = _evidence_task(
        "shared_rule",
        "Find the shipment exemption.",
        ["req_rule"],
        [("ev_rule", EvidenceRequirementKind.GOVERNING_RULE, "Find the exemption.")],
        output_id="rule_evidence",
    )
    application = _application_task(
        "apply_a",
        ["req_outcome", "req_limit"],
        fact_ids=["fact_shipment"],
        unknown_ids=["unknown_noncommercial"],
        branch_ids=["branch_noncommercial", "branch_commercial"],
    )
    plan = _plan(
        "Can this shipment use the exemption?",
        requirements,
        [evidence, application],
        problem_type=ProblemType.SCENARIO_APPLICATION,
        scenario=scenario,
    )
    planner = _ScriptedClient("planner", lambda schema, text: plan)
    rule_claim = _claim(
        "shared_rule",
        ["req_rule"],
        ["ev_rule"],
        "conditional",
        "The exemption is limited to non-commercial shipments.",
        claim_id="claim_exemption",
    )
    conditional_finding = DerivedConclusion(
        conclusion_id="conclusion_conditional",
        finding=(
            "If the shipment is non-commercial the exemption may apply; "
            "if commercial it does not apply."
        ),
        requirement_ids=["req_outcome", "req_limit"],
        fact_ids=["fact_shipment"],
        branch_ids=["branch_noncommercial", "branch_commercial"],
        supporting_claim_ids=["shared_rule_rule-search_claim_1"],
        dependency_refs=[
            TaskOutputRef(task_id="shared_rule", output_id="rule_evidence")
        ],
        confidence=EvidenceConfidence.MEDIUM,
        limitations=[unknown_text],
    )

    def task_response(schema: type, text: str) -> object:
        task_id = _task_id(text)
        if schema is SearchAssignmentBatch:
            return SearchAssignmentBatch(
                task_id="shared_rule",
                stop=False,
                stop_reason=None,
                assignments=[
                    SearchAssignment(
                        assignment_id="rule-search",
                        task_id="shared_rule",
                        query="shipment exemption rule",
                        objective="Find the exemption condition.",
                        evidence_requirements=["ev_rule"],
                        excluded_queries=[],
                        as_of_date=None,
                        filters=None,
                    )
                ],
            )
        if task_id == "shared_rule":
            return _artifact(
                "shared_rule",
                ["req_rule"],
                claims=[rule_claim],
                workers=["worker-shared_rule-rule-search"],
            )
        return _artifact(
            "apply_a",
            ["req_outcome", "req_limit"],
            findings=[conditional_finding],
            gaps=[unknown_text],
        )

    worker = _ScriptedClient(
        "worker",
        lambda schema, text: WorkerArtifact(
            task_id="shared_rule",
            assignment_id="rule-search",
            worker_id="worker-shared_rule-rule-search",
            status=WorkerStatus.SUCCESS,
            searches_run=["shipment exemption rule"],
            claims=[rule_claim],
            gaps=[],
            cross_references=[],
            error_code=None,
            error_message=None,
        ),
    )
    progress: list[ResearchProgress] = []
    result = await MultiAgentResearchOrchestrator(
        planner_llm=planner,
        task_llm=_ScriptedClient("task", task_response),
        worker_llm=worker,
        search_runner=lambda **kwargs: _SearchResult(
            query=kwargs["query"],
            hits=[
                _hit(
                    "conditional",
                    "The exemption is limited to non-commercial shipments.",
                )
            ],
        ),
        limits=_limits(),
        on_progress=progress.append,
        search_runner_in_thread=False,
    ).run(plan.normalized_question)

    application_artifact = next(
        artifact for artifact in result.task_artifacts if artifact.task_id == "apply_a"
    )
    assert application_artifact.application_findings, application_artifact
    finding = application_artifact.application_findings[0]
    assert finding.finding.startswith("If the shipment is non-commercial")
    assert unknown_text in finding.limitations
    assert "conclusion_conditional" in result.final_context
    assert unknown_text in result.unresolved_information
    information_events = [
        event for event in progress if event.kind == "information_needed"
    ]
    assert len(information_events) == 1
    assert unknown_text in information_events[0].detail
    assert finding.finding in result.final_context
    assert unknown_text in result.final_context
