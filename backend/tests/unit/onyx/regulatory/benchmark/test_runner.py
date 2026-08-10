import datetime
import time
from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from onyx.db.enums import BenchmarkRunItemStatus, BenchmarkRunStatus
from onyx.db.models import BenchmarkRun, BenchmarkRunItem, User
from onyx.db.regulatory_benchmark import mark_benchmark_run_failed
from onyx.llm.cost import ModelPrice
from onyx.llm.override_models import LLMOverride
from onyx.regulatory.benchmark.models import BenchmarkJudgeResult, BenchmarkRunReport
from onyx.regulatory.benchmark.runner import (
    _citation_metrics,
    _cited_sources,
    _generate_item_answer,
    _get_item_persona,
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


def test_run_report_schema_preserves_provider_identity() -> None:
    model_report_schema = BenchmarkRunReport.model_json_schema()["$defs"][
        "BenchmarkModelReport"
    ]
    provider_id_schema = model_report_schema["properties"]["provider_id"]

    assert "anyOf" in provider_id_schema
    assert "provider_id" in model_report_schema["required"]


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


def test_cited_sources_match_the_exact_citation_chunk_identity() -> None:
    earlier_chunk = SimpleNamespace(
        document_id="connector-document",
        chunk_ind=2,
        semantic_identifier="Customs Act",
        blurb="earlier chunk",
        score=0.8,
        link=None,
    )
    cited_chunk = SimpleNamespace(
        document_id="connector-document",
        chunk_ind=7,
        semantic_identifier="Customs Act — Article 46",
        blurb="the cited provision",
        score=0.9,
        link=None,
    )
    response = SimpleNamespace(
        citation_info=[
            SimpleNamespace(
                document_id="connector-document",
                chunk_ind=7,
                citation_number=1,
            )
        ],
        top_documents=[earlier_chunk, cited_chunk],
    )

    sources = _cited_sources(MagicMock(), response)

    assert len(sources) == 1
    assert sources[0]["chunk_index"] == 7
    assert sources[0]["excerpt"] == "the cited provision"


def test_cited_sources_do_not_positionally_fallback_for_unknown_chunk_identity() -> (
    None
):
    response = SimpleNamespace(
        citation_info=[
            SimpleNamespace(
                document_id="connector-document",
                chunk_ind=99,
                citation_number=1,
            )
        ],
        top_documents=[
            SimpleNamespace(
                document_id="connector-document",
                chunk_ind=2,
                semantic_identifier="Customs Act",
                blurb="different chunk",
                score=0.8,
                link=None,
            )
        ],
    )

    assert _cited_sources(MagicMock(), response) == []


def test_cited_sources_keep_legacy_positional_fallback_without_chunk_identity() -> None:
    documents = [
        SimpleNamespace(
            document_id="connector-document",
            chunk_ind=chunk_ind,
            semantic_identifier="Customs Act",
            blurb=f"chunk {chunk_ind}",
            score=0.8,
            link=None,
        )
        for chunk_ind in (2, 7)
    ]
    response = SimpleNamespace(
        citation_info=[
            SimpleNamespace(document_id="connector-document", citation_number=1),
            SimpleNamespace(document_id="connector-document", citation_number=2),
        ],
        top_documents=documents,
    )

    sources = _cited_sources(MagicMock(), response)

    assert [source["chunk_index"] for source in sources] == [2, 7]


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


def test_answer_generation_scopes_search_to_document_set_without_project() -> None:
    chat_session_id = uuid4()
    response = SimpleNamespace(
        error_msg=None,
        answer="scoped answer",
        pre_answer_reasoning=None,
        message_id=uuid4(),
    )
    item = SimpleNamespace(
        id=42,
        provider="OpenRouter",
        model_id="candidate-model",
        question=SimpleNamespace(
            prompt="fallback prompt",
            document_set=SimpleNamespace(name="Current Regulations"),
        ),
        question_snapshot={
            "prompt": "Which rule applies?",
            "as_of_date": "2026-08-06",
            "expected_citations": [],
        },
    )
    run = SimpleNamespace(id=7, deep_research=False)
    user = SimpleNamespace(id=uuid4())
    db_session = MagicMock()

    with (
        patch(
            "onyx.regulatory.benchmark.runner.create_chat_session",
            return_value=SimpleNamespace(id=chat_session_id),
        ) as create_chat_session,
        patch(
            "onyx.regulatory.benchmark.runner.benchmark_usage_capture",
            return_value=nullcontext([]),
        ),
        patch(
            "onyx.regulatory.benchmark.runner.handle_stream_message_objects",
            return_value=iter(()),
        ) as handle_stream,
        patch(
            "onyx.regulatory.benchmark.runner.gather_stream_full",
            return_value=response,
        ),
        patch(
            "onyx.regulatory.benchmark.runner._usage_cost",
            return_value=(0, 0, None, "unavailable"),
        ),
    ):
        _generate_item_answer(
            db_session,
            run=cast(BenchmarkRun, run),
            item=cast(BenchmarkRunItem, item),
            user=cast(User, user),
            persona=None,
            started=time.monotonic(),
        )

    assert "project_id" not in create_chat_session.call_args.kwargs
    expected_override = LLMOverride(
        model_provider="OpenRouter",
        model_provider_type="openrouter",
        model_version="candidate-model",
        temperature=0,
    )
    assert create_chat_session.call_args.kwargs["llm_override"] == expected_override
    request = handle_stream.call_args.args[0]
    assert request.llm_override == expected_override
    assert request.internal_search_filters is not None
    assert request.internal_search_filters.document_set == ["Current Regulations"]
    assert request.internal_search_filters.as_of_date == datetime.date(2026, 8, 6)
    db_session.commit.assert_called_once_with()


def test_nameless_openrouter_selector_becomes_a_typed_provider_override() -> None:
    chat_session_id = uuid4()
    response = SimpleNamespace(
        error_msg=None,
        answer="answer",
        pre_answer_reasoning=None,
        message_id=uuid4(),
        citation_info=[],
        top_documents=[],
        tool_calls=[],
    )
    item = SimpleNamespace(
        id=43,
        provider="openrouter",
        provider_id=8,
        model_id="candidate-model",
        question=SimpleNamespace(
            prompt="prompt",
            document_set=SimpleNamespace(name="Current Regulations"),
        ),
        question_snapshot={"prompt": "prompt", "expected_citations": []},
    )
    run = SimpleNamespace(id=7, deep_research=False)
    user = SimpleNamespace(id=uuid4())

    with (
        patch(
            "onyx.regulatory.benchmark.runner.create_chat_session",
            return_value=SimpleNamespace(id=chat_session_id),
        ) as create_chat_session,
        patch(
            "onyx.regulatory.benchmark.runner.benchmark_usage_capture",
            return_value=nullcontext([]),
        ),
        patch(
            "onyx.regulatory.benchmark.runner.handle_stream_message_objects",
            return_value=iter(()),
        ),
        patch(
            "onyx.regulatory.benchmark.runner.gather_stream_full",
            return_value=response,
        ),
        patch(
            "onyx.regulatory.benchmark.runner._usage_cost",
            return_value=(0, 0, None, "unavailable"),
        ),
    ):
        _generate_item_answer(
            MagicMock(),
            run=cast(BenchmarkRun, run),
            item=cast(BenchmarkRunItem, item),
            user=cast(User, user),
            persona=MagicMock(),
            started=time.monotonic(),
        )

    override = create_chat_session.call_args.kwargs["llm_override"]
    assert override.model_provider is None
    assert override.model_provider_type == "openrouter"
    assert override.model_provider_id == 8


def test_named_openrouter_provider_uses_persisted_provider_id() -> None:
    chat_session_id = uuid4()
    response = SimpleNamespace(
        error_msg=None,
        answer="answer",
        pre_answer_reasoning=None,
        message_id=uuid4(),
        citation_info=[],
        top_documents=[],
        tool_calls=[],
    )
    item = SimpleNamespace(
        id=44,
        provider="openrouter",
        provider_id=7,
        model_id="candidate-model",
        question=SimpleNamespace(
            prompt="prompt",
            document_set=SimpleNamespace(name="Current Regulations"),
        ),
        question_snapshot={"prompt": "prompt", "expected_citations": []},
    )

    with (
        patch(
            "onyx.regulatory.benchmark.runner.create_chat_session",
            return_value=SimpleNamespace(id=chat_session_id),
        ) as create_chat_session,
        patch(
            "onyx.regulatory.benchmark.runner.benchmark_usage_capture",
            return_value=nullcontext([]),
        ),
        patch(
            "onyx.regulatory.benchmark.runner.handle_stream_message_objects",
            return_value=iter(()),
        ),
        patch(
            "onyx.regulatory.benchmark.runner.gather_stream_full",
            return_value=response,
        ),
        patch(
            "onyx.regulatory.benchmark.runner._usage_cost",
            return_value=(0, 0, None, "unavailable"),
        ),
    ):
        _generate_item_answer(
            MagicMock(),
            run=cast(BenchmarkRun, SimpleNamespace(id=7, deep_research=False)),
            item=cast(BenchmarkRunItem, item),
            user=cast(User, SimpleNamespace(id=uuid4())),
            persona=MagicMock(),
            started=time.monotonic(),
        )

    override = create_chat_session.call_args.kwargs["llm_override"]
    assert override.model_provider is None
    assert override.model_provider_type == "openrouter"
    assert override.model_provider_id == 7


def test_benchmark_items_select_persona_for_their_own_document_set() -> None:
    db_session = MagicMock()
    user = MagicMock()
    first_persona = MagicMock(id=101)
    second_persona = MagicMock(id=202)
    db_session.get.side_effect = lambda _model, persona_id: {
        101: first_persona,
        202: second_persona,
    }[persona_id]
    first_item = cast(
        BenchmarkRunItem, SimpleNamespace(question_snapshot={"document_set_id": 11})
    )
    second_item = cast(
        BenchmarkRunItem, SimpleNamespace(question_snapshot={"document_set_id": 22})
    )

    with patch(
        "onyx.regulatory.benchmark.runner.get_best_persona_id_for_user",
        side_effect=[101, 202],
    ) as select_persona:
        assert (
            _get_item_persona(db_session, user=user, item=first_item) is first_persona
        )
        assert (
            _get_item_persona(db_session, user=user, item=second_item) is second_persona
        )

    assert [
        call.kwargs["document_set_id"] for call in select_persona.call_args_list
    ] == [
        11,
        22,
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


def test_runner_initial_claim_locks_the_dispatch_row() -> None:
    run = cast(
        BenchmarkRun,
        SimpleNamespace(id=31, status=BenchmarkRunStatus.ERROR.value),
    )
    db_session = MagicMock()

    def locked_scalar_result(statement: object) -> MagicMock:
        if getattr(statement, "_for_update_arg", None) is None:
            raise AssertionError("runner claim must wait for the dispatch row lock")
        result = MagicMock()
        result.one_or_none.return_value = run
        return result

    db_session.scalars.side_effect = locked_scalar_result

    run_benchmark(db_session, run.id)


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
            "onyx.regulatory.benchmark.runner.get_benchmark_run_for_update",
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


def test_creator_missing_startup_error_is_completed_by_task_recovery() -> None:
    pending_item = SimpleNamespace(
        status=BenchmarkRunItemStatus.PENDING.value,
        error_message=None,
        completed_at=None,
    )
    run = SimpleNamespace(
        id=23,
        status=BenchmarkRunStatus.RUNNING.value,
        created_by="missing-user-id",
        report_error=None,
        items=[pending_item],
        completed_items=0,
        failed_items=0,
        completed_at=None,
    )
    db_session = MagicMock()
    db_session.get.return_value = None

    with (
        patch(
            "onyx.regulatory.benchmark.runner.get_benchmark_run_for_update",
            return_value=cast(BenchmarkRun, run),
        ),
        pytest.raises(ValueError, match="creator no longer exists"),
    ):
        run_benchmark(db_session, run.id)

    assert run.status == BenchmarkRunStatus.ERROR.value
    assert run.report_error is None
    assert pending_item.status == BenchmarkRunItemStatus.PENDING.value

    with patch(
        "onyx.db.regulatory_benchmark.get_benchmark_run_for_update",
        return_value=cast(BenchmarkRun, run),
    ):
        mark_benchmark_run_failed(
            db_session, run.id, "Benchmark run creator no longer exists"
        )

    assert run.report_error == "Benchmark run creator no longer exists"
    assert pending_item.status == BenchmarkRunItemStatus.ERROR.value
    assert pending_item.error_message == "Benchmark run creator no longer exists"
    assert pending_item.completed_at == run.completed_at
    assert run.failed_items == 1
