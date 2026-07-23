"""Tests for the headless benchmark runner and LLM-judge scoring."""

from decimal import Decimal

import pytest

from fs_explorer_api import benchmark_runner as benchmark_runner_mod
from fs_explorer_api.agent import _index_tools_available
from fs_explorer_api.benchmark_runner import (
    _sum_call_cost,
    judge_answer,
    run_agentic_session,
)
from fs_explorer_api.agent import LLMCallStats
from fs_explorer_api.llm.base import LLMUsage
from fs_explorer_api.models import Action, JudgmentResult, StopAction


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
    """Fake `LLMClient`: stops immediately, reports a fixed provider cost."""

    model = "fake/model"

    def __init__(self) -> None:
        self.structured_calls = 0
        self.stream_calls = 0

    async def generate_structured(
        self, history, system_prompt, schema, *, thinking_level=None
    ):
        self.structured_calls += 1
        return (
            Action(reason="done", action=StopAction(final_result="benchmark answer")),
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


@pytest.fixture(autouse=True)
def _clear_context_before_and_after():
    from fs_explorer_api.agent import clear_index_context

    clear_index_context()
    yield
    clear_index_context()


class TestRunAgenticSession:
    @pytest.mark.asyncio
    async def test_returns_stats_shape_and_pools_call_cost(self, monkeypatch) -> None:
        monkeypatch.setattr(benchmark_runner_mod, "PostgresStorage", _FakeStorage)
        monkeypatch.setattr("fs_explorer_api.agent.PostgresStorage", _FakeStorage)
        monkeypatch.setattr(
            "fs_explorer_api.agent.get_llm_client",
            lambda **_kwargs: _StopActionClient(),
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
        }
        # One structured (action) call + one streamed final-answer call.
        assert result.stats["api_calls"] == 2
        assert result.stats["cost_source"] == "provider"
        assert result.stats["cost_usd"] == "0.0015"

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
    async def test_clears_index_context_when_the_run_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(benchmark_runner_mod, "PostgresStorage", _FakeStorage)
        monkeypatch.setattr("fs_explorer_api.agent.PostgresStorage", _FakeStorage)
        monkeypatch.setattr(
            "fs_explorer_api.agent.get_llm_client", lambda **_kwargs: _RaisingClient()
        )

        with pytest.raises(RuntimeError, match="boom"):
            await run_agentic_session(
                task="x",
                index_folders=["virtual://corpus-1"],
                database_url="postgresql://test/test",
            )

        assert _index_tools_available() is False


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
        )

        assert result["overall_score"] == 100
        assert "241" in client.seen_prompt
        assert "Madde 241" in client.seen_prompt

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
