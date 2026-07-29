"""Tests for the headless benchmark runner and LLM-judge scoring."""

from decimal import Decimal

import pytest

from fs_explorer_api import benchmark_runner as benchmark_runner_mod
from fs_explorer_api.agent import PlannedSearchResult, _index_tools_available
from fs_explorer_api.benchmark_runner import (
    _benchmark_llm_profile,
    _benchmark_multi_agent_enabled,
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
from fs_explorer_api.models import (
    JudgmentResult,
    ResearchDecision,
    RetrievalPlan,
    RetrievalQuery,
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


class _StopActionClient:
    """Fake `LLMClient`: plans once, then streams a fixed final answer."""

    model = "fake/model"

    def __init__(self) -> None:
        self.structured_calls = 0
        self.stream_calls = 0

    async def generate_structured(
        self, history, system_prompt, schema, *, thinking_level=None
    ):
        self.structured_calls += 1
        if schema is ResearchDecision:
            return (
                ResearchDecision(
                    enough_evidence=True,
                    reason="Enough for the benchmark.",
                    additional_searches=[],
                ),
                LLMUsage(
                    input_tokens=100,
                    output_tokens=20,
                    billed_cost_usd=Decimal("0.0010"),
                    cost_source="provider",
                ),
            )
        return (
            RetrievalPlan(
                research_question="What is the transit penalty?",
                searches=[RetrievalQuery(query="transit penalty")],
            ),
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


class _RaisingClient:
    async def generate_structured(self, *args, **kwargs):
        raise RuntimeError("boom")

    async def stream_text(self, *args, **kwargs):
        raise RuntimeError("boom")
        yield ""  # pragma: no cover - never reached, keeps this an async generator

    def last_stream_usage(self):
        return None


class _AdaptiveRoundClient(_StopActionClient):
    def __init__(self) -> None:
        super().__init__()
        self.reviews = 0

    async def generate_structured(
        self, history, system_prompt, schema, *, thinking_level=None
    ):
        if schema is ResearchDecision:
            self.structured_calls += 1
            self.reviews += 1
            return (
                ResearchDecision(
                    enough_evidence=self.reviews >= 2,
                    reason="covered" if self.reviews >= 2 else "exception missing",
                    additional_searches=[]
                    if self.reviews >= 2
                    else [RetrievalQuery(query="force majeure exception")],
                ),
                LLMUsage(input_tokens=100, output_tokens=20),
            )
        return await super().generate_structured(
            history, system_prompt, schema, thinking_level=thinking_level
        )


@pytest.fixture(autouse=True)
def _clear_context_before_and_after(monkeypatch):
    from fs_explorer_api.agent import clear_index_context

    # Most legacy runner tests exercise the established stateless path. The
    # production API default is covered separately and can still be disabled
    # through the same operational kill switch.
    monkeypatch.setenv("FS_EXPLORER_MULTI_AGENT_ENABLED", "false")
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
            lambda **_kwargs: _StopActionClient(),
        )
        monkeypatch.setattr(
            "fs_explorer_api.agent._run_planned_index_search",
            lambda **kwargs: PlannedSearchResult(query=kwargs["query"], hits=[]),
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
        # One planner + one bounded evidence review + one final answer.
        assert result.stats["api_calls"] == 3
        assert result.stats["cost_source"] == "provider"
        assert result.stats["cost_usd"] == "0.0025"
        assert result.stats["profile_mode"] == "candidate_all_roles"
        assert result.plan_trace == result.stats["plan_trace"]
        assert result.role_usage == result.stats["role_usage"]
        assert result.cited_evidence == []
        assert sum(row["calls"] for row in result.role_usage) == 3

    @pytest.mark.asyncio
    async def test_raises_when_no_folder_is_indexed(self, monkeypatch) -> None:
        monkeypatch.setattr(
            benchmark_runner_mod, "PostgresStorage", _EmptyCorpusStorage
        )
        monkeypatch.setattr(
            "fs_explorer_api.agent.get_llm_client",
            lambda **_kwargs: _StopActionClient(),
        )

        with pytest.raises(ValueError, match="No index found"):
            await run_agentic_session(
                task="x",
                index_folders=["virtual://missing"],
                database_url="postgresql://test/test",
            )

        assert _index_tools_available() is False

    @pytest.mark.asyncio
    async def test_runs_an_adaptive_second_parallel_round(self, monkeypatch) -> None:
        client = _AdaptiveRoundClient()
        monkeypatch.setattr(benchmark_runner_mod, "PostgresStorage", _FakeStorage)
        monkeypatch.setattr("fs_explorer_api.agent.PostgresStorage", _FakeStorage)
        monkeypatch.setattr(
            "fs_explorer_api.agent.get_llm_client", lambda **_kwargs: client
        )

        def fake_search(**kwargs):
            query = kwargs["query"]
            return PlannedSearchResult(
                query=query,
                hits=[
                    SearchHit(
                        doc_id=f"doc_{len(query)}",
                        relative_path=f"{len(query)}.md",
                        absolute_path=f"/{len(query)}.md",
                        position=0,
                        text=f"evidence for {query}",
                        semantic_score=0.8,
                        metadata_score=0,
                        score=0.8,
                        matched_by="semantic",
                        chunk_id=f"chunk_{len(query)}",
                    )
                ],
            )

        monkeypatch.setattr(
            "fs_explorer_api.agent._run_planned_index_search", fake_search
        )

        result = await run_agentic_session(
            task="What is the transit penalty and its exception?",
            index_folders=["virtual://corpus-1"],
            database_url="postgresql://test/test",
        )

        assert result.error is None
        assert result.final_result == "benchmark answer"
        assert client.reviews == 2
        assert result.stats["api_calls"] == 4
        assert result.stats["steps"] == 2

    @pytest.mark.asyncio
    async def test_clears_index_context_when_the_run_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(benchmark_runner_mod, "PostgresStorage", _FakeStorage)
        monkeypatch.setattr("fs_explorer_api.agent.PostgresStorage", _FakeStorage)
        monkeypatch.setattr(
            "fs_explorer_api.agent.get_llm_client", lambda **_kwargs: _RaisingClient()
        )

        async def raise_during_collection(self, task, calls=None):
            raise RuntimeError("boom")

        monkeypatch.setattr(
            "fs_explorer_api.agent.FsExplorerAgent.collect_parallel_indexed_evidence",
            raise_during_collection,
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
    def test_benchmark_uses_server_multi_agent_default_and_kill_switch(
        self, monkeypatch
    ) -> None:
        monkeypatch.delenv("FS_EXPLORER_MULTI_AGENT_ENABLED", raising=False)
        assert _benchmark_multi_agent_enabled() is True

        monkeypatch.setenv("FS_EXPLORER_MULTI_AGENT_ENABLED", "false")
        assert _benchmark_multi_agent_enabled() is False

        monkeypatch.setenv("FS_EXPLORER_MULTI_AGENT_ENABLED", "invalid")
        with pytest.raises(ValueError, match="must be a boolean"):
            _benchmark_multi_agent_enabled()

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
