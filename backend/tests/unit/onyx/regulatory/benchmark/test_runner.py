import datetime
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from onyx.db.enums import BenchmarkRunItemStatus, BenchmarkRunStatus
from onyx.db.models import BenchmarkRun
from onyx.llm.cost import ModelPrice
from onyx.regulatory.benchmark.models import BenchmarkJudgeResult
from onyx.regulatory.benchmark.runner import (
    _citation_metrics,
    _mark_unfinished_items_error,
    _recover_interrupted_items,
    _run_item,
    _usage_cost,
    _usage_snapshots,
    run_benchmark,
)
from onyx.regulatory.benchmark.usage_capture import LLMCallUsage


def test_judge_schema_is_openrouter_provider_compatible() -> None:
    schema = BenchmarkJudgeResult.model_json_schema()
    serialized = str(schema)

    assert "minimum" not in serialized
    assert "maximum" not in serialized
    assert "propertyNames" not in serialized


def test_judge_scores_remain_application_validated() -> None:
    with pytest.raises(ValueError, match="criterion score"):
        BenchmarkJudgeResult.model_validate(
            {
                "correctness_score": 6,
                "groundedness_score": 5,
                "completeness_score": 5,
                "clarity_score": 5,
                "overall_score": 100,
                "rationale": "rationale",
                "summary": "summary",
                "criteria": {},
                "strengths": [],
                "weaknesses": [],
                "fact_assessments": [],
                "citation_assessments": [],
            }
        )


def test_judge_normalizes_models_that_return_overall_on_five_point_scale() -> None:
    result = BenchmarkJudgeResult.model_validate(
        {
            "correctness_score": 4,
            "groundedness_score": 2,
            "completeness_score": 3,
            "clarity_score": 4,
            "overall_score": 2,
            "rationale": "rationale",
            "summary": "summary",
            "criteria": {
                name: {"score": score, "rationale": "rationale"}
                for name, score in {
                    "correctness": 4,
                    "groundedness": 2,
                    "completeness": 3,
                    "clarity": 4,
                }.items()
            },
            "strengths": [],
            "weaknesses": [],
            "fact_assessments": [],
            "citation_assessments": [],
        }
    )

    assert result.overall_score == 40


def test_usage_cost_sums_each_actual_llm_call() -> None:
    usage_calls = [
        LLMCallUsage("candidate", "provider-a", 100, 20, 0),
        LLMCallUsage("query-rephraser", "provider-b", 30, 4, 5),
    ]

    with (
        patch(
            "onyx.regulatory.benchmark.runner.compute_cost_cents",
            side_effect=[(0.1, 0.2), (0.03, 0.04)],
        ) as compute_cost,
        patch(
            "onyx.regulatory.benchmark.runner.get_model_price_per_million",
            return_value=ModelPrice(
                model="priced",
                provider="provider",
                input_per_mtok=1.0,
                output_per_mtok=2.0,
                cache_per_mtok=None,
            ),
        ),
    ):
        input_tokens, output_tokens, cost_cents, cost_source = _usage_cost(
            MagicMock(), usage_calls
        )

    assert input_tokens == 130
    assert output_tokens == 24
    assert cost_cents == pytest.approx(0.37)
    assert cost_source == "measured"
    assert [call.args[:2] for call in compute_cost.call_args_list] == [
        ("candidate", "provider-a"),
        ("query-rephraser", "provider-b"),
    ]


def test_usage_cost_marks_unknown_pricing_unavailable() -> None:
    usage_calls = [LLMCallUsage("unknown", None, 10, 2, 0)]
    with (
        patch(
            "onyx.regulatory.benchmark.runner.compute_cost_cents",
            return_value=(0.0, 0.0),
        ),
        patch(
            "onyx.regulatory.benchmark.runner.get_model_price_per_million",
            return_value=ModelPrice(
                model="unknown",
                provider=None,
                input_per_mtok=None,
                output_per_mtok=None,
                cache_per_mtok=None,
            ),
        ),
    ):
        _, _, cost_cents, cost_source = _usage_cost(MagicMock(), usage_calls)

    assert cost_cents is None
    assert cost_source == "unavailable"


def test_required_citation_metrics_use_regulatory_chunk_ids() -> None:
    question = {
        "expected_citations": [
            {"chunk_id": "chunk-a", "requirement": "required"},
            {"chunk_id": "chunk-b", "requirement": "required"},
            {"chunk_id": "chunk-c", "requirement": "supporting"},
        ]
    }
    cited_sources: list[dict[str, object]] = [
        {"regulatory_chunk_id": "chunk-a"},
        {"regulatory_chunk_id": "chunk-unexpected"},
    ]

    recall, precision = _citation_metrics(question, cited_sources)

    assert recall == 0.5
    assert precision == 0.5


def test_usage_snapshots_keep_every_production_chat_llm_cycle() -> None:
    calls = [
        LLMCallUsage("query-model", "OpenRouter", 20, 5, 0, "query_expansion"),
        LLMCallUsage("answer-model", "OpenRouter", 100, 40, 10, "chat_response"),
    ]

    assert _usage_snapshots(calls, phase="answer") == [
        {
            "sequence": 1,
            "phase": "answer",
            "provider": "OpenRouter",
            "model": "query-model",
            "input_tokens": 20,
            "output_tokens": 5,
            "cache_read_tokens": 0,
            "flow": "query_expansion",
        },
        {
            "sequence": 2,
            "phase": "answer",
            "provider": "OpenRouter",
            "model": "answer-model",
            "input_tokens": 100,
            "output_tokens": 40,
            "cache_read_tokens": 10,
            "flow": "chat_response",
        },
    ]


def test_redelivery_recovers_running_items_without_discarding_saved_answer() -> None:
    recovered_at = datetime.datetime.now(datetime.timezone.utc)
    answer_saved = SimpleNamespace(
        status=BenchmarkRunItemStatus.RUNNING.value,
        judgment=None,
        final_result="persisted canonical-chat answer",
        completed_at=recovered_at,
    )
    judgment_saved = SimpleNamespace(
        status=BenchmarkRunItemStatus.RUNNING.value,
        judgment=object(),
        completed_at=None,
    )
    run = SimpleNamespace(items=[answer_saved, judgment_saved])

    recovered = _recover_interrupted_items(
        cast(BenchmarkRun, run), recovered_at=recovered_at
    )

    assert recovered == 2
    assert answer_saved.status == BenchmarkRunItemStatus.PENDING.value
    assert answer_saved.final_result == "persisted canonical-chat answer"
    assert answer_saved.completed_at is None
    assert judgment_saved.status == BenchmarkRunItemStatus.COMPLETED.value
    assert judgment_saved.completed_at == recovered_at


def test_resumed_item_skips_answer_generation_and_continues_with_judge() -> None:
    item = MagicMock()
    item.id = 8
    item.status = BenchmarkRunItemStatus.PENDING.value
    item.started_at = None
    item.completed_at = None
    item.duration_ms = 15
    item.judgment = None
    item.final_result = "already persisted"
    run = MagicMock(id=3, status=BenchmarkRunStatus.RUNNING.value)
    db_session = MagicMock()

    with (
        patch(
            "onyx.regulatory.benchmark.runner._generate_item_answer"
        ) as generate_answer,
        patch("onyx.regulatory.benchmark.runner._judge_completed_item") as judge_item,
    ):
        _run_item(
            db_session,
            run=run,
            item=item,
            user=MagicMock(),
            persona=None,
        )

    generate_answer.assert_not_called()
    judge_item.assert_called_once()
    assert item.status == BenchmarkRunItemStatus.COMPLETED.value


def test_unfinished_items_are_terminal_errors_not_false_completions() -> None:
    completed_at = datetime.datetime.now(datetime.timezone.utc)
    pending = SimpleNamespace(
        status=BenchmarkRunItemStatus.PENDING.value,
        error_message=None,
        completed_at=None,
    )
    running = SimpleNamespace(
        status=BenchmarkRunItemStatus.RUNNING.value,
        error_message=None,
        completed_at=None,
    )
    completed = SimpleNamespace(
        status=BenchmarkRunItemStatus.COMPLETED.value,
        error_message=None,
        completed_at=completed_at,
    )
    run = SimpleNamespace(items=[pending, running, completed])

    assert (
        _mark_unfinished_items_error(cast(BenchmarkRun, run), completed_at=completed_at)
        == 2
    )
    assert pending.status == BenchmarkRunItemStatus.ERROR.value
    assert running.status == BenchmarkRunItemStatus.ERROR.value
    assert completed.status == BenchmarkRunItemStatus.COMPLETED.value


def test_run_with_any_error_is_never_marked_completed() -> None:
    completed_item = SimpleNamespace(
        status=BenchmarkRunItemStatus.COMPLETED.value,
        judgment=None,
    )
    error_item = SimpleNamespace(
        status=BenchmarkRunItemStatus.ERROR.value,
        judgment=None,
    )
    run = SimpleNamespace(
        id=17,
        status=BenchmarkRunStatus.RUNNING.value,
        created_by="user-id",
        items=[completed_item, error_item],
        total_items=2,
        completed_items=0,
        failed_items=0,
        started_at=None,
        completed_at=None,
    )
    db_session = MagicMock()
    db_session.get.return_value = MagicMock()

    def refresh_counts(_db_session: object, target: BenchmarkRun) -> None:
        target.completed_items = sum(
            item.status == BenchmarkRunItemStatus.COMPLETED.value
            for item in target.items
        )
        target.failed_items = sum(
            item.status == BenchmarkRunItemStatus.ERROR.value for item in target.items
        )

    with (
        patch(
            "onyx.regulatory.benchmark.runner.get_benchmark_run",
            return_value=cast(BenchmarkRun, run),
        ),
        patch(
            "onyx.regulatory.benchmark.runner.get_best_persona_id_for_user",
            return_value=None,
        ),
        patch(
            "onyx.regulatory.benchmark.runner.refresh_benchmark_run_counts",
            side_effect=refresh_counts,
        ),
        patch("onyx.regulatory.benchmark.runner._generate_run_report"),
    ):
        run_benchmark(db_session, 17)

    assert run.status == BenchmarkRunStatus.ERROR.value
    assert run.completed_items == 1
    assert run.failed_items == 1
