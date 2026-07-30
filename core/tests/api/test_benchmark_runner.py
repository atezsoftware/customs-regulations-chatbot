"""Tests for the headless benchmark runner and LLM-judge scoring."""

from decimal import Decimal

import pytest

from fs_explorer_api import benchmark_runner as benchmark_runner_mod
from fs_explorer_api.agent import PlannedSearchResult, _index_tools_available
from fs_explorer_api.benchmark_runner import (
    _benchmark_llm_profile,
    _build_plan_trace,
    _resolve_profile_mode,
    _role_usage_breakdown,
    _sum_call_cost,
    judge_answer,
    run_agentic_session,
)
from fs_explorer_api.agent import LLMCallStats
from fs_explorer_api.llm.base import LLMUsage
from fs_explorer_api.llm.profile import DEFAULT_LLM_PROFILE, LLMProfile
from fs_explorer_api.models import JudgmentResult
from fs_explorer_api.orchestration_models import (
    AnswerRequirement,
    AnswerRequirementKind,
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
    TaskSpec,
    TaskStatus,
    WorkerArtifact,
    WorkerStatus,
)
from fs_explorer_api.search import SearchHit


class _FakeStorage:
    """Stand-in for `PostgresStorage`: no real Postgres connection made."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.closed = False

    def get_corpus_id(self, _folder: str) -> str | None:
        return "fake-corpus-id"

    def get_document(self, *, doc_id: str) -> dict | None:
        return None

    def list_documents(self, *, corpus_id: str, include_deleted: bool = False) -> list:
        return []

    def close(self) -> None:
        self.closed = True


class _EmptyCorpusStorage(_FakeStorage):
    """No folder resolves to a corpus — mirrors an un-indexed directory."""

    def get_corpus_id(self, _folder: str) -> str | None:
        return None


def _single_task_plan(
    task_id: str,
    question: str,
    *,
    execution_strategy: ExecutionStrategy = ExecutionStrategy.SINGLE_PASS,
) -> GlobalPlan:
    """A minimal one-task plan for the single, always-on multi-agent flow."""

    criterion_id = f"criterion-{task_id}"
    task = TaskSpec(
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
        consumes=[],
        produces=[TaskOutput(output_id=f"output-{task_id}", description="Conclusion")],
        required=True,
        as_of_date=None,
        filters=None,
    )
    return GlobalPlan(
        version="3",
        problem_type=ProblemType.LOOKUP,
        mode=PlanMode.DIRECT,
        execution_strategy=execution_strategy,
        normalized_question=question,
        answer_requirements=[
            AnswerRequirement(
                requirement_id=criterion_id,
                kind=AnswerRequirementKind.OUTCOME,
                description="Answer requirement",
                required=True,
            )
        ],
        scenario=None,
        tasks=[task],
        synthesis_requirements=[],
        assumptions=[],
    )


def _claim(task_id: str) -> EvidenceClaim:
    return EvidenceClaim(
        claim_id=f"claim-{task_id}",
        claim="Supported claim",
        document_id=f"doc-{task_id}",
        chunk_id=f"chunk-{task_id}",
        readable_title="Model title",
        locator="Model locator",
        evidence_excerpt=f"Evidence text for {task_id}.",
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


def _worker_artifact(task_id: str) -> WorkerArtifact:
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


class _MultiAgentStopClient:
    """Fake `LLMClient` driving the single always-on multi-agent flow end to
    end for one single-pass task: a plan, one worker round, then a streamed
    final answer. `get_llm_client` is patched to return this same instance
    for every role (planner/task/worker/final), so it must handle whichever
    schema each role's call requests."""

    model = "fake/model"

    def __init__(self) -> None:
        self.structured_calls = 0
        self.stream_calls = 0

    async def generate_structured(
        self, history, system_prompt, schema, *, thinking_level=None
    ):
        self.structured_calls += 1
        if schema is GlobalPlan:
            return (
                _single_task_plan("task_1", "What is the transit penalty?"),
                LLMUsage(
                    input_tokens=100,
                    output_tokens=20,
                    billed_cost_usd=Decimal("0.0010"),
                    cost_source="provider",
                ),
            )
        assert schema is WorkerArtifact
        return (
            _worker_artifact("task_1"),
            LLMUsage(
                input_tokens=100,
                output_tokens=20,
                billed_cost_usd=Decimal("0.0010"),
                cost_source="provider",
            ),
        )

    async def stream_text(self, history, system_prompt, *, thinking_level=None):
        self.stream_calls += 1
        yield "benchmark answer"

    def last_stream_usage(self):
        return LLMUsage(
            input_tokens=50,
            output_tokens=10,
            billed_cost_usd=Decimal("0.0005"),
            cost_source="provider",
        )


def _two_requirement_task_plan(task_id: str, question: str) -> GlobalPlan:
    """An ADAPTIVE one-task plan with two evidence requirements, so a single
    worker round covering only one of them forces a genuine second
    coordinator round instead of the deterministic single-pass shortcut."""

    requirement_ids = [f"{task_id}-a", f"{task_id}-b"]
    task = TaskSpec(
        task_id=task_id,
        kind=TaskKind.EVIDENCE,
        issue=f"Resolve {task_id}",
        search_question=question,
        requirement_ids=requirement_ids,
        fact_ids=[],
        unknown_ids=[],
        branch_ids=[],
        evidence_requirements=[
            EvidenceRequirement(
                evidence_requirement_id=requirement_id,
                kind=EvidenceRequirementKind.GOVERNING_RULE,
                description=requirement_id,
                requirement_ids=[requirement_id],
            )
            for requirement_id in requirement_ids
        ],
        consumes=[],
        produces=[TaskOutput(output_id=f"output-{task_id}", description="Conclusion")],
        required=True,
        as_of_date=None,
        filters=None,
    )
    return GlobalPlan(
        version="3",
        problem_type=ProblemType.LOOKUP,
        mode=PlanMode.DIRECT,
        execution_strategy=ExecutionStrategy.ADAPTIVE,
        normalized_question=question,
        answer_requirements=[
            AnswerRequirement(
                requirement_id=requirement_id,
                kind=AnswerRequirementKind.OUTCOME,
                description="Answer requirement",
                required=True,
            )
            for requirement_id in requirement_ids
        ],
        scenario=None,
        tasks=[task],
        synthesis_requirements=[],
        assumptions=[],
    )


def _round_claim(task_id: str, requirement_id: str) -> EvidenceClaim:
    return EvidenceClaim(
        claim_id=f"claim-{requirement_id}",
        claim=f"Supported claim for {requirement_id}",
        document_id=f"doc-{task_id}",
        chunk_id=f"chunk-{task_id}",
        readable_title="Model title",
        locator="Model locator",
        evidence_excerpt=f"Evidence text for {task_id}.",
        requirement_ids=[requirement_id],
        evidence_requirement_ids=[requirement_id],
        fact_ids=[],
        confidence=EvidenceConfidence.HIGH,
        effective_start_date=None,
        effective_end_date=None,
    )


class _AdaptiveRoundClient:
    """Fake `LLMClient` for one ADAPTIVE task whose coordinator (task role)
    requests a second worker round before both evidence requirements are
    covered."""

    model = "fake/model"

    def __init__(self) -> None:
        self.structured_calls = 0
        self.stream_calls = 0
        self.rounds = 0

    async def generate_structured(
        self, history, system_prompt, schema, *, thinking_level=None
    ):
        self.structured_calls += 1
        if schema is GlobalPlan:
            return (
                _two_requirement_task_plan(
                    "task_1", "What is the transit penalty and its exception?"
                ),
                LLMUsage(input_tokens=100, output_tokens=20),
            )
        if schema is SearchAssignmentBatch:
            self.rounds += 1
            requirement_id = f"task_1-{'a' if self.rounds == 1 else 'b'}"
            return (
                SearchAssignmentBatch(
                    task_id="task_1",
                    stop=False,
                    stop_reason=None,
                    assignments=[
                        SearchAssignment(
                            assignment_id=f"assignment-task_1-r{self.rounds}",
                            task_id="task_1",
                            query=f"query-task_1-r{self.rounds}",
                            objective=f"Find evidence for {requirement_id}",
                            evidence_requirements=[requirement_id],
                            excluded_queries=[],
                            as_of_date=None,
                            filters=None,
                        )
                    ],
                ),
                LLMUsage(input_tokens=100, output_tokens=20),
            )
        if schema is TaskArtifact:
            return (
                TaskArtifact(
                    task_id="task_1",
                    status=TaskStatus.COMPLETE,
                    answer_fragment="Conclusion for task_1",
                    covered_requirement_ids=["task_1-a", "task_1-b"],
                    uncovered_requirement_ids=[],
                    claims=[
                        _round_claim("task_1", "task_1-a"),
                        _round_claim("task_1", "task_1-b"),
                    ],
                    application_findings=[],
                    conflicts=[],
                    gaps=[],
                    contributing_worker_ids=[
                        "worker-task_1-assignment-task_1-r1",
                        "worker-task_1-assignment-task_1-r2",
                    ],
                ),
                LLMUsage(input_tokens=100, output_tokens=20),
            )
        assert schema is WorkerArtifact
        requirement_id = f"task_1-{'a' if self.rounds == 1 else 'b'}"
        return (
            WorkerArtifact(
                task_id="task_1",
                assignment_id=f"assignment-task_1-r{self.rounds}",
                worker_id=f"worker-task_1-r{self.rounds}",
                status=WorkerStatus.SUCCESS,
                searches_run=[f"query-task_1-r{self.rounds}"],
                claims=[_round_claim("task_1", requirement_id)],
                gaps=[],
                cross_references=[],
                error_code=None,
                error_message=None,
            ),
            LLMUsage(input_tokens=100, output_tokens=20),
        )

    async def stream_text(self, history, system_prompt, *, thinking_level=None):
        self.stream_calls += 1
        yield "benchmark answer"

    def last_stream_usage(self):
        return None


@pytest.fixture(autouse=True)
def _clear_context_before_and_after():
    from fs_explorer_api.agent import clear_index_context

    clear_index_context()
    yield
    clear_index_context()


class TestRunAgenticSession:
    @pytest.mark.asyncio
    async def test_candidate_profile_reaches_every_role_and_interrupted_run_is_cancelled(
        self, monkeypatch
    ) -> None:
        handler = _InterruptedHandler()
        captured_workflow_args: dict[str, object] = {}

        class _InterruptedWorkflow:
            def run(self, **_kwargs):
                return handler

        def fake_new_workflow(**kwargs):
            captured_workflow_args.update(kwargs)
            return _InterruptedWorkflow(), object()

        monkeypatch.setattr(benchmark_runner_mod, "PostgresStorage", _FakeStorage)
        monkeypatch.setattr(benchmark_runner_mod, "new_workflow", fake_new_workflow)
        monkeypatch.setattr(
            benchmark_runner_mod,
            "get_run_agent",
            lambda _resource_manager: object(),
        )
        monkeypatch.setenv("FS_EXPLORER_PLANNER_REASONING", "high")
        monkeypatch.setenv("FS_EXPLORER_TASK_REASONING", "medium")
        monkeypatch.setenv("FS_EXPLORER_WORKER_REASONING", "low")
        monkeypatch.setenv("FS_EXPLORER_FINAL_REASONING", "minimal")

        with pytest.raises(RuntimeError, match="benchmark stream interrupted"):
            await run_agentic_session(
                task="x",
                index_folders=["virtual://corpus-1"],
                database_url="postgresql://test/test",
                provider="openrouter",
                model="candidate/model",
            )

        profile = captured_workflow_args["llm_profile"]
        assert isinstance(profile, LLMProfile)
        assert {
            role: (
                profile.for_role(role).provider,
                profile.for_role(role).model,
                profile.for_role(role).reasoning_effort,
            )
            for role in ("planner", "task", "worker", "final")
        } == {
            "planner": ("openrouter", "candidate/model", "high"),
            "task": ("openrouter", "candidate/model", "medium"),
            "worker": ("openrouter", "candidate/model", "low"),
            "final": ("openrouter", "candidate/model", "minimal"),
        }
        assert handler.cancelled is True

    @pytest.mark.asyncio
    async def test_production_profile_reaches_workflow_without_candidate_override(
        self, monkeypatch
    ) -> None:
        handler = _InterruptedHandler()
        captured_workflow_args: dict[str, object] = {}

        class _InterruptedWorkflow:
            def run(self, **_kwargs):
                return handler

        def fake_new_workflow(**kwargs):
            captured_workflow_args.update(kwargs)
            return _InterruptedWorkflow(), object()

        monkeypatch.setattr(benchmark_runner_mod, "PostgresStorage", _FakeStorage)
        monkeypatch.setattr(benchmark_runner_mod, "new_workflow", fake_new_workflow)
        monkeypatch.setattr(
            benchmark_runner_mod,
            "get_run_agent",
            lambda _resource_manager: object(),
        )
        monkeypatch.setenv("FS_EXPLORER_PLANNER_MODEL", "planner/model")
        monkeypatch.setenv("FS_EXPLORER_TASK_MODEL", "task/model")
        monkeypatch.setenv("FS_EXPLORER_WORKER_MODEL", "worker/model")
        monkeypatch.setenv("FS_EXPLORER_FINAL_MODEL", "final/model")

        with pytest.raises(RuntimeError, match="benchmark stream interrupted"):
            await run_agentic_session(
                task="x",
                index_folders=["virtual://corpus-1"],
                database_url="postgresql://test/test",
                profile_mode="production_roles",
            )

        profile = captured_workflow_args["llm_profile"]
        assert isinstance(profile, LLMProfile)
        assert {
            role: profile.for_role(role).model
            for role in ("planner", "task", "worker", "final")
        } == {
            "planner": "planner/model",
            "task": "task/model",
            "worker": "worker/model",
            "final": "final/model",
        }
        assert captured_workflow_args["provider"] is None
        assert captured_workflow_args["model"] is None
        assert handler.cancelled is True

    @pytest.mark.asyncio
    async def test_returns_stats_shape_and_pools_call_cost(self, monkeypatch) -> None:
        monkeypatch.setattr(benchmark_runner_mod, "PostgresStorage", _FakeStorage)
        monkeypatch.setattr("fs_explorer_api.agent.PostgresStorage", _FakeStorage)
        monkeypatch.setattr(
            "fs_explorer_api.agent.get_llm_client",
            lambda **_kwargs: _MultiAgentStopClient(),
        )
        monkeypatch.setattr(
            "fs_explorer_api.agent._run_planned_index_search",
            lambda **kwargs: PlannedSearchResult(
                query=kwargs["query"], hits=[_hit("task_1")]
            ),
        )

        result = await run_agentic_session(
            task="What is the transit penalty?",
            index_folders=["virtual://corpus-1"],
            database_url="postgresql://test/test",
            provider="openrouter",
            model="test/model",
        )

        assert result.error is None
        assert result.incomplete is False
        assert result.final_result == "benchmark answer"
        assert set(result.stats) >= {
            "steps",
            "api_calls",
            "prompt_tokens",
            "completion_tokens",
            "thinking_tokens",
            "total_tokens",
            "duration_ms",
            "cost_usd",
            "cost_source",
            "profile_mode",
            "plan_trace",
            "role_usage",
        }
        # One plan + one single-pass worker round + one final answer.
        assert result.stats["api_calls"] == 3
        assert result.stats["cost_source"] == "provider"
        assert result.stats["cost_usd"] == "0.0025"
        assert result.stats["profile_mode"] == "candidate_all_roles"
        assert result.plan_trace == result.stats["plan_trace"]
        assert result.role_usage == result.stats["role_usage"]
        assert sum(row["calls"] for row in result.role_usage) == 3

    @pytest.mark.asyncio
    async def test_raises_when_no_folder_is_indexed(self, monkeypatch) -> None:
        monkeypatch.setattr(
            benchmark_runner_mod, "PostgresStorage", _EmptyCorpusStorage
        )
        monkeypatch.setattr(
            "fs_explorer_api.agent.get_llm_client",
            lambda **_kwargs: _MultiAgentStopClient(),
        )

        with pytest.raises(ValueError, match="No index found"):
            await run_agentic_session(
                task="x",
                index_folders=["virtual://missing"],
                database_url="postgresql://test/test",
            )

        assert _index_tools_available() is False

    @pytest.mark.asyncio
    async def test_runs_a_second_adaptive_worker_round(self, monkeypatch) -> None:
        client = _AdaptiveRoundClient()
        monkeypatch.setattr(benchmark_runner_mod, "PostgresStorage", _FakeStorage)
        monkeypatch.setattr("fs_explorer_api.agent.PostgresStorage", _FakeStorage)
        monkeypatch.setattr(
            "fs_explorer_api.agent.get_llm_client", lambda **_kwargs: client
        )
        monkeypatch.setattr(
            "fs_explorer_api.agent._run_planned_index_search",
            lambda **kwargs: PlannedSearchResult(
                query=kwargs["query"], hits=[_hit("task_1")]
            ),
        )

        result = await run_agentic_session(
            task="What is the transit penalty and its exception?",
            index_folders=["virtual://corpus-1"],
            database_url="postgresql://test/test",
        )

        assert result.error is None
        assert result.final_result == "benchmark answer"
        assert client.rounds == 2

    @pytest.mark.asyncio
    async def test_clears_index_context_when_the_run_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(benchmark_runner_mod, "PostgresStorage", _FakeStorage)
        monkeypatch.setattr("fs_explorer_api.agent.PostgresStorage", _FakeStorage)

        async def raise_during_research(self, task, *, on_progress=None):
            raise RuntimeError("boom")

        monkeypatch.setattr(
            "fs_explorer_api.agent.FsExplorerAgent.prepare_multi_agent_indexed",
            raise_during_research,
        )

        with pytest.raises(RuntimeError, match="boom"):
            await run_agentic_session(
                task="x",
                index_folders=["virtual://corpus-1"],
                database_url="postgresql://test/test",
            )

        assert _index_tools_available() is False


class _InterruptedHandler:
    def __init__(self) -> None:
        self.cancelled = False

    async def stream_events(self):
        raise RuntimeError("benchmark stream interrupted")
        yield  # pragma: no cover - makes this an async generator

    def is_done(self) -> bool:
        return False

    async def cancel_run(self) -> None:
        self.cancelled = True


class TestSumCallCost:
    def test_pools_provider_costs(self) -> None:
        calls = [
            LLMCallStats(
                purpose="action",
                model="m",
                prompt_tokens=1,
                completion_tokens=1,
                thinking_tokens=0,
                duration_ms=1,
                billed_cost_usd="0.0010",
                cost_source="provider",
            ),
            LLMCallStats(
                purpose="final_answer",
                model="m",
                prompt_tokens=1,
                completion_tokens=1,
                thinking_tokens=0,
                duration_ms=1,
                billed_cost_usd="0.0005",
                cost_source="provider",
            ),
        ]
        cost_usd, cost_source = _sum_call_cost(calls)
        assert cost_usd == "0.0015"
        assert cost_source == "provider"

    def test_marks_estimated_if_any_call_is_estimated(self) -> None:
        calls = [
            LLMCallStats(
                purpose="action",
                model="m",
                prompt_tokens=1,
                completion_tokens=1,
                thinking_tokens=0,
                duration_ms=1,
                billed_cost_usd="0.0010",
                cost_source="provider",
            ),
            LLMCallStats(
                purpose="final_answer",
                model="m",
                prompt_tokens=1,
                completion_tokens=1,
                thinking_tokens=0,
                duration_ms=1,
                billed_cost_usd="0.0002",
                cost_source="estimated",
            ),
        ]
        cost_usd, cost_source = _sum_call_cost(calls)
        assert cost_usd == "0.0012"
        assert cost_source == "estimated"

    def test_returns_none_when_no_call_reports_cost(self) -> None:
        calls = [
            LLMCallStats(
                purpose="action",
                model="m",
                prompt_tokens=1,
                completion_tokens=1,
                thinking_tokens=0,
                duration_ms=1,
            )
        ]
        assert _sum_call_cost(calls) == (None, None)


class TestBenchmarkObservability:
    def test_legacy_optional_arguments_resolve_to_the_same_profile_modes(self) -> None:
        assert (
            _resolve_profile_mode(
                profile_mode=None,
                provider="openrouter",
                model="candidate/model",
            )
            == "candidate_all_roles"
        )
        assert (
            _resolve_profile_mode(
                profile_mode=None,
                provider=None,
                model=None,
            )
            == "production_roles"
        )

    def test_production_profile_snapshots_heterogeneous_role_models(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("FS_EXPLORER_PLANNER_MODEL", "planner/model")
        monkeypatch.setenv("FS_EXPLORER_TASK_MODEL", "task/model")
        monkeypatch.setenv("FS_EXPLORER_WORKER_MODEL", "worker/model")
        monkeypatch.setenv("FS_EXPLORER_FINAL_MODEL", "final/model")

        profile = _benchmark_llm_profile(
            provider=None,
            model=None,
            profile_mode="production_roles",
        )

        assert {
            role: profile.for_role(role).model
            for role in ("planner", "task", "worker", "final")
        } == {
            "planner": "planner/model",
            "task": "task/model",
            "worker": "worker/model",
            "final": "final/model",
        }

    def test_role_usage_groups_calls_by_role_purpose_provider_and_model(self) -> None:
        calls = [
            LLMCallStats(
                purpose="evidence_extraction",
                model="worker/model",
                provider="openrouter",
                agent_role="worker",
                prompt_tokens=10,
                completion_tokens=4,
                thinking_tokens=1,
                cached_input_tokens=2,
                cache_write_tokens=3,
                duration_ms=12.5,
                billed_cost_usd="0.001",
                cost_source="provider",
            ),
            LLMCallStats(
                purpose="evidence_extraction",
                model="worker/model",
                provider="openrouter",
                agent_role="worker",
                prompt_tokens=20,
                completion_tokens=5,
                thinking_tokens=2,
                cached_input_tokens=4,
                cache_write_tokens=1,
                duration_ms=7.5,
                billed_cost_usd="0.002",
                cost_source="estimated",
            ),
        ]

        assert _role_usage_breakdown(calls) == [
            {
                "role": "worker",
                "purpose": "evidence_extraction",
                "provider": "openrouter",
                "model": "worker/model",
                "calls": 2,
                "prompt_tokens": 30,
                "completion_tokens": 9,
                "thinking_tokens": 3,
                "total_tokens": 42,
                "cached_input_tokens": 6,
                "cache_write_tokens": 4,
                "duration_ms": 20.0,
                "cost_usd": "0.003",
                "cost_source": "estimated",
            }
        ]

    def test_plan_trace_uses_agent_hook_and_runtime_fingerprints(self) -> None:
        class _Agent:
            benchmark_plan_trace = {
                "schema_version": 1,
                "contract_version": "2",
                "plan": {"mode": "direct"},
                "task_artifacts": [{"task_id": "task_1", "claim_count": 2}],
            }

        trace = _build_plan_trace(
            agent=_Agent(),
            profile_mode="production_roles",
            profile=DEFAULT_LLM_PROFILE,
        )

        assert trace["profile_mode"] == "production_roles"
        assert trace["role_profile"]["planner"]["model"] == "openai/gpt-5.6-sol"
        assert trace["execution"]["contract_version"] == "2"
        assert trace["runtime"]["contract"]["plan_contract_version"] == "2"
        assert trace["runtime"]["contract"]["plan_fields"] == ["mode"]
        assert trace["runtime"]["prompts"]["global_planner"]["sha256"]
        assert trace["runtime"]["prompts"]["application_task"]["sha256"]
        assert trace["runtime"]["prompts"]["integration_task"]["sha256"]
        assert trace["runtime"]["prompts"]["scenario_final_synthesis"]["sha256"]


class _JudgeClient:
    def __init__(self, judgment: JudgmentResult) -> None:
        self._judgment = judgment
        self.seen_prompt: str | None = None
        self.seen_system_prompt: str | None = None

    async def generate_structured(
        self, history, system_prompt, schema, *, thinking_level=None
    ):
        self.seen_prompt = history[0].text
        self.seen_system_prompt = system_prompt
        assert schema is JudgmentResult
        return self._judgment, LLMUsage()

    async def stream_text(self, *args, **kwargs):  # pragma: no cover - unused here
        yield ""

    def last_stream_usage(self):
        return None


class TestJudgeAnswer:
    @pytest.mark.asyncio
    async def test_computes_weighted_overall_score(self, monkeypatch) -> None:
        judgment = JudgmentResult(
            correctness=5,
            groundedness=5,
            completeness=5,
            clarity=5,
            rationale="fully correct and well cited",
        )
        client = _JudgeClient(judgment)
        monkeypatch.setattr(
            benchmark_runner_mod, "get_llm_client", lambda **_kwargs: client
        )

        result = await judge_answer(
            question="Transit süresi aşılırsa ceza uygulanır mı?",
            reference_answer="Evet, madde 241 uyarınca ceza uygulanır.",
            expected_facts=["241", "ceza"],
            rubric_notes=None,
            candidate_answer="Evet, Madde 241 uyarınca kademeli ceza uygulanır.",
            cited_sources=["Gümrük Kanunu Madde 241"],
            judge_provider="openrouter",
            judge_model="test/judge",
            cited_evidence=[
                {
                    "title": "Gümrük Kanunu",
                    "locator": "Madde 241",
                    "snippet": "Süre aşımında kademeli usulsüzlük cezası uygulanır.",
                }
            ],
        )

        assert result["overall_score"] == 100
        assert "241" in client.seen_prompt
        assert "Madde 241" in client.seen_prompt
        assert "Süre aşımında kademeli usulsüzlük" in client.seen_prompt
        assert "UNTRUSTED SOURCE TEXT" in client.seen_prompt

    @pytest.mark.asyncio
    async def test_partial_scores_yield_partial_weighted_total(
        self, monkeypatch
    ) -> None:
        judgment = JudgmentResult(
            correctness=1,
            groundedness=1,
            completeness=1,
            clarity=1,
            rationale="wrong and uncited",
        )
        client = _JudgeClient(judgment)
        monkeypatch.setattr(
            benchmark_runner_mod, "get_llm_client", lambda **_kwargs: client
        )

        result = await judge_answer(
            question="q",
            reference_answer=None,
            expected_facts=None,
            rubric_notes=None,
            candidate_answer="a",
            cited_sources=[],
            judge_provider="openrouter",
            judge_model="test/judge",
        )

        # All 1s -> 100 * (1/5) == 20.
        assert result["overall_score"] == 20

    def test_schema_has_no_numeric_bounds_openrouter_strict_mode_rejects(self) -> None:
        # Regression test: `ge=`/`le=` on a Pydantic field become JSON
        # Schema `minimum`/`maximum`, which OpenRouter's `strict: true`
        # structured-output mode rejects the whole request for — this is
        # exactly what broke every judge call before this test existed.
        schema = JudgmentResult.model_json_schema()

        def assert_no_bounds(node: object) -> None:
            if isinstance(node, dict):
                assert "minimum" not in node, node
                assert "maximum" not in node, node
                for value in node.values():
                    assert_no_bounds(value)
            elif isinstance(node, list):
                for item in node:
                    assert_no_bounds(item)

        assert_no_bounds(schema)

    @pytest.mark.asyncio
    async def test_clamps_out_of_rubric_scores_from_a_misbehaving_judge(
        self, monkeypatch
    ) -> None:
        # JudgmentResult has no schema-level bounds (see the regression test
        # above), so a judge model could return a value outside 1-5 despite
        # the rubric asking for it — judge_answer must clamp rather than
        # let it corrupt the weighted overall_score or violate the DB's
        # 1-5 CHECK constraint.
        judgment = JudgmentResult(
            correctness=9,
            groundedness=0,
            completeness=3,
            clarity=5,
            rationale="out of range",
        )
        client = _JudgeClient(judgment)
        monkeypatch.setattr(
            benchmark_runner_mod, "get_llm_client", lambda **_kwargs: client
        )

        result = await judge_answer(
            question="q",
            reference_answer=None,
            expected_facts=None,
            rubric_notes=None,
            candidate_answer="a",
            cited_sources=[],
            judge_provider="openrouter",
            judge_model="test/judge",
        )

        assert result["correctness"] == 5
        assert result["groundedness"] == 1
        assert 0 <= result["overall_score"] <= 100
