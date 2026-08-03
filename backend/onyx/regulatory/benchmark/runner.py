import datetime
import time
from collections import defaultdict
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from onyx.chat.chat_state import ChatStateContainer
from onyx.chat.process_message import gather_stream_full, handle_stream_message_objects
from onyx.context.search.models import BaseFilters, SearchDoc
from onyx.db.chat import create_chat_session
from onyx.db.enums import (
    BenchmarkCostSource,
    BenchmarkRunItemStatus,
    BenchmarkRunStatus,
)
from onyx.db.models import BenchmarkRun, BenchmarkRunItem, Persona, User, UserFile
from onyx.db.persona import get_best_persona_id_for_user
from onyx.db.regulatory_benchmark import (
    add_benchmark_judgment,
    get_benchmark_run,
    refresh_benchmark_run_counts,
)
from onyx.db.regulatory_chunks import get_chunks_for_file
from onyx.llm.cost import compute_cost_cents, get_model_price_per_million
from onyx.llm.factory import get_llm_for_persona
from onyx.llm.override_models import LLMOverride
from onyx.regulatory.benchmark.judge import (
    generate_benchmark_run_report,
    judge_benchmark_answer,
)
from onyx.regulatory.benchmark.usage_capture import (
    LLMCallUsage,
    benchmark_usage_capture,
)
from onyx.server.query_and_chat.models import SendMessageRequest
from onyx.tracing.framework.create import ensure_trace
from onyx.utils.logger import setup_logger

logger = setup_logger()

_MAX_TOOL_RESULT_CHARS = 20_000
_MAX_REASONING_CHARS = 30_000


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _usage_cost(
    db_session: Session, usage_calls: list[LLMCallUsage]
) -> tuple[int, int, float | None, str]:
    input_tokens = sum(call.input_tokens for call in usage_calls)
    output_tokens = sum(call.output_tokens for call in usage_calls)
    total_cost_cents = 0.0
    pricing_available = bool(usage_calls)
    for call in usage_calls:
        input_cost, output_cost = compute_cost_cents(
            call.model,
            call.provider,
            call.input_tokens,
            call.output_tokens,
            call.cache_read_tokens,
            db_session=db_session,
        )
        total_cost_cents += input_cost + output_cost
        price = get_model_price_per_million(
            call.model, call.provider, db_session=db_session
        )
        pricing_available = pricing_available and (
            price.input_per_mtok is not None or price.output_per_mtok is not None
        )
    if not pricing_available:
        return (
            input_tokens,
            output_tokens,
            None,
            BenchmarkCostSource.UNAVAILABLE.value,
        )
    return (
        input_tokens,
        output_tokens,
        total_cost_cents,
        BenchmarkCostSource.MEASURED.value,
    )


def _usage_snapshots(
    usage_calls: list[LLMCallUsage], *, phase: str
) -> list[dict[str, object]]:
    return [
        {
            "sequence": index + 1,
            "phase": phase,
            **call.to_dict(),
        }
        for index, call in enumerate(usage_calls)
    ]


def _source_snapshot(
    db_session: Session,
    document: SearchDoc,
    citation_number: int,
) -> dict[str, object]:
    regulatory_chunk_id: str | None = None
    heading_path: list[str] = []
    file_name = document.semantic_identifier.split(" — ", 1)[0]
    try:
        user_file_id = UUID(document.document_id)
        rows = get_chunks_for_file(db_session, user_file_id)
        if 0 <= document.chunk_ind < len(rows):
            row = rows[document.chunk_ind]
            regulatory_chunk_id = row.id
            heading_path = list(row.heading_path)
        user_file = db_session.get(UserFile, user_file_id)
        if user_file is not None:
            file_name = user_file.name
    except (ValueError, TypeError):
        pass
    return {
        "citation_number": citation_number,
        "regulatory_chunk_id": regulatory_chunk_id,
        "document_id": document.document_id,
        "chunk_index": document.chunk_ind,
        "file_name": file_name,
        "semantic_identifier": document.semantic_identifier,
        "heading_path": heading_path,
        "excerpt": document.blurb[:4000],
        "score": document.score,
        "link": document.link,
    }


def _cited_sources(db_session: Session, response: object) -> list[dict[str, object]]:
    citation_info = getattr(response, "citation_info", [])
    top_documents: list[SearchDoc] = getattr(response, "top_documents", [])
    by_document: dict[str, list[SearchDoc]] = defaultdict(list)
    for document in top_documents:
        by_document[document.document_id].append(document)
    document_offsets: dict[str, int] = defaultdict(int)
    sources: list[dict[str, object]] = []
    for citation in citation_info:
        matches = by_document.get(citation.document_id, [])
        if not matches:
            continue
        offset = document_offsets[citation.document_id]
        document = matches[min(offset, len(matches) - 1)]
        document_offsets[citation.document_id] += 1
        sources.append(_source_snapshot(db_session, document, citation.citation_number))
    return sources


def _citation_metrics(
    question_snapshot: dict[str, Any], cited_sources: list[dict[str, object]]
) -> tuple[float | None, float | None]:
    expected = question_snapshot.get("expected_citations") or []
    expected_ids = {
        str(citation["chunk_id"])
        for citation in expected
        if isinstance(citation, dict)
        and citation.get("requirement", "required") == "required"
        and citation.get("chunk_id")
    }
    if not expected_ids:
        return None, None
    actual_ids = {
        str(source["regulatory_chunk_id"])
        for source in cited_sources
        if source.get("regulatory_chunk_id")
    }
    overlap = expected_ids & actual_ids
    recall = len(overlap) / len(expected_ids)
    precision = len(overlap) / len(actual_ids) if actual_ids else 0.0
    return recall, precision


def _search_doc_snapshot(document: SearchDoc) -> dict[str, object]:
    return {
        "document_id": document.document_id,
        "chunk_index": document.chunk_ind,
        "semantic_identifier": document.semantic_identifier,
        "excerpt": document.blurb[:2000],
        "score": document.score,
    }


def _execution_steps(response: object) -> list[dict[str, object]]:
    steps: list[dict[str, object]] = []
    for index, tool_call in enumerate(getattr(response, "tool_calls", []), start=1):
        steps.append(
            {
                "sequence": index,
                "kind": "tool",
                "title": tool_call.tool_name,
                "arguments": tool_call.tool_arguments,
                "reasoning": (tool_call.pre_reasoning or "")[:_MAX_REASONING_CHARS]
                or None,
                "result": tool_call.tool_result[:_MAX_TOOL_RESULT_CHARS],
                "retrieved_documents": [
                    _search_doc_snapshot(document)
                    for document in (tool_call.search_docs or [])[:50]
                ],
            }
        )
    steps.append(
        {
            "sequence": len(steps) + 1,
            "kind": "answer",
            "title": "Final answer",
            "character_count": len(getattr(response, "answer", "")),
        }
    )
    return steps


def _judge_completed_item(
    db_session: Session,
    *,
    run: BenchmarkRun,
    item: BenchmarkRunItem,
    user: User,
    persona: Persona | None,
) -> None:
    judge_llm = get_llm_for_persona(
        persona,
        user,
        LLMOverride(
            model_provider=run.judge_provider,
            model_version=run.judge_model,
            temperature=0,
        ),
    )
    with benchmark_usage_capture() as judge_usage:
        with ensure_trace(
            workflow_name="regulatory_benchmark_judge",
            group_id=str(run.id),
            metadata={"run_id": run.id, "item_id": item.id},
        ):
            result = judge_benchmark_answer(
                judge_llm,
                question_snapshot=item.question_snapshot,
                candidate_answer=item.final_result or "",
                cited_sources=item.cited_sources,
                citation_recall=item.citation_recall,
                citation_precision=item.citation_precision,
            )
    input_tokens, output_tokens, cost_cents, cost_source = _usage_cost(
        db_session, judge_usage
    )
    item.llm_calls = [
        *item.llm_calls,
        *_usage_snapshots(judge_usage, phase="judge"),
    ]
    add_benchmark_judgment(
        db_session,
        run_item=item,
        judge_provider=run.judge_provider,
        judge_model=run.judge_model,
        correctness_score=result.correctness_score,
        groundedness_score=result.groundedness_score,
        completeness_score=result.completeness_score,
        clarity_score=result.clarity_score,
        overall_score=result.overall_score,
        rationale=result.rationale,
        report=result.model_dump(mode="json"),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cost_cents=cost_cents,
        cost_source=cost_source,
    )


def _generate_item_answer(
    db_session: Session,
    *,
    run: BenchmarkRun,
    item: BenchmarkRunItem,
    user: User,
    persona: Persona | None,
    started: float,
) -> None:
    override = LLMOverride(
        model_provider=item.provider,
        model_version=item.model_id,
        temperature=0,
    )
    chat_session = create_chat_session(
        db_session,
        description=f"Benchmark run {run.id}, item {item.id}",
        user_id=user.id,
        persona_id=persona.id if persona else None,
        llm_override=override,
        project_id=item.question.project_id,
    )
    as_of_date_value = item.question_snapshot.get("as_of_date")
    as_of_date = (
        datetime.date.fromisoformat(str(as_of_date_value)) if as_of_date_value else None
    )
    request = SendMessageRequest(
        message=str(item.question_snapshot.get("prompt") or item.question.prompt),
        chat_session_id=chat_session.id,
        llm_override=override,
        internal_search_filters=BaseFilters(as_of_date=as_of_date),
        deep_research=run.deep_research,
        stream=True,
        include_citations=True,
    )
    state_container = ChatStateContainer()
    with benchmark_usage_capture() as answer_usage:
        response = gather_stream_full(
            handle_stream_message_objects(
                request, user, external_state_container=state_container
            ),
            state_container,
        )
    item.duration_ms = round((time.monotonic() - started) * 1000)
    if response.error_msg:
        raise RuntimeError(response.error_msg)

    input_tokens, output_tokens, cost_cents, cost_source = _usage_cost(
        db_session, answer_usage
    )
    item.final_result = response.answer
    item.input_tokens = input_tokens
    item.output_tokens = output_tokens
    item.total_tokens = input_tokens + output_tokens
    item.cost_cents = cost_cents
    item.cost_source = cost_source
    item.answer_reasoning = (response.pre_answer_reasoning or "")[
        :_MAX_REASONING_CHARS
    ] or None
    item.chat_session_id = chat_session.id
    item.assistant_message_id = response.message_id
    item.cited_sources = _cited_sources(db_session, response)
    item.cited_chunk_ids = [
        str(source["regulatory_chunk_id"])
        for source in item.cited_sources
        if source.get("regulatory_chunk_id")
    ]
    item.citation_recall, item.citation_precision = _citation_metrics(
        item.question_snapshot, item.cited_sources
    )
    item.execution_steps = _execution_steps(response)
    item.llm_calls = _usage_snapshots(answer_usage, phase="answer")
    # This is the idempotency boundary: a redelivery after this commit resumes
    # at judging and does not pay for the canonical chat answer twice.
    db_session.commit()


def _run_item(
    db_session: Session,
    *,
    run: BenchmarkRun,
    item: BenchmarkRunItem,
    user: User,
    persona: Persona | None,
) -> None:
    started = time.monotonic()
    item.status = BenchmarkRunItemStatus.RUNNING.value
    item.started_at = _utcnow()
    item.completed_at = None
    db_session.commit()

    try:
        if item.judgment is not None:
            item.status = BenchmarkRunItemStatus.COMPLETED.value
            return
        if item.final_result is None:
            _generate_item_answer(
                db_session,
                run=run,
                item=item,
                user=user,
                persona=persona,
                started=started,
            )
        db_session.refresh(run)
        db_session.refresh(item)
        if run.status == BenchmarkRunStatus.CANCELLED.value:
            item.status = BenchmarkRunItemStatus.CANCELLED.value
            return

        try:
            _judge_completed_item(
                db_session, run=run, item=item, user=user, persona=persona
            )
            item.status = BenchmarkRunItemStatus.COMPLETED.value
        except Exception as judge_error:
            logger.exception("Benchmark judge failed for item %s", item.id)
            db_session.rollback()
            reloaded_item = db_session.get(BenchmarkRunItem, item.id)
            assert reloaded_item is not None
            item = reloaded_item
            item.status = BenchmarkRunItemStatus.ERROR.value
            item.judge_error = str(judge_error)
            item.error_message = f"Judge failed: {judge_error}"
    except Exception as error:
        logger.exception("Benchmark run %s item %s failed", run.id, item.id)
        db_session.rollback()
        reloaded_item = db_session.get(BenchmarkRunItem, item.id)
        assert reloaded_item is not None
        item = reloaded_item
        item.status = BenchmarkRunItemStatus.ERROR.value
        item.error_message = str(error)
    finally:
        if item.duration_ms is None:
            item.duration_ms = round((time.monotonic() - started) * 1000)
        if item.status in {
            BenchmarkRunItemStatus.COMPLETED.value,
            BenchmarkRunItemStatus.ERROR.value,
            BenchmarkRunItemStatus.CANCELLED.value,
        }:
            item.completed_at = _utcnow()
        db_session.commit()


def _generate_run_report(
    db_session: Session,
    *,
    run: BenchmarkRun,
    user: User,
    persona: Persona | None,
) -> None:
    judged_items = [item for item in run.items if item.judgment is not None]
    if not judged_items:
        return
    grouped: dict[tuple[str, str], list[BenchmarkRunItem]] = defaultdict(list)
    for item in judged_items:
        grouped[(item.provider, item.model_id)].append(item)
    aggregates: list[dict[str, object]] = []
    for (provider, model_id), items in grouped.items():
        scores = [item.judgment.overall_score for item in items if item.judgment]
        recalls = [
            item.citation_recall for item in items if item.citation_recall is not None
        ]
        aggregates.append(
            {
                "provider": provider,
                "model_id": model_id,
                "average_score": sum(scores) / len(scores),
                "average_citation_recall": (
                    sum(recalls) / len(recalls) if recalls else None
                ),
                "average_duration_ms": sum(item.duration_ms or 0 for item in items)
                / len(items),
                "total_cost_cents": sum(
                    (item.cost_cents or 0)
                    + (item.judgment.cost_cents or 0 if item.judgment else 0)
                    for item in items
                ),
            }
        )
    report_items = [
        {
            "model": f"{item.provider}/{item.model_id}",
            "question": item.question_snapshot.get("title"),
            "overall_score": item.judgment.overall_score,
            "judge_summary": item.judgment.report.get("summary"),
            "strengths": item.judgment.report.get("strengths", []),
            "weaknesses": item.judgment.report.get("weaknesses", []),
            "citation_recall": item.citation_recall,
        }
        for item in judged_items
        if item.judgment
    ]
    judge_llm = get_llm_for_persona(
        persona,
        user,
        LLMOverride(
            model_provider=run.judge_provider,
            model_version=run.judge_model,
            temperature=0,
        ),
    )
    try:
        with benchmark_usage_capture() as report_usage:
            with ensure_trace(
                workflow_name="regulatory_benchmark_report",
                group_id=str(run.id),
                metadata={"run_id": run.id},
            ):
                report = generate_benchmark_run_report(
                    judge_llm,
                    run_label=run.label,
                    aggregates=aggregates,
                    items=report_items,
                )
        input_tokens, output_tokens, cost_cents, _ = _usage_cost(
            db_session, report_usage
        )
        run.report = report.model_dump(mode="json")
        run.report_input_tokens = input_tokens
        run.report_output_tokens = output_tokens
        run.report_cost_cents = cost_cents
    except Exception as error:
        logger.exception("Benchmark run %s report generation failed", run.id)
        run.report_error = str(error)


def _recover_interrupted_items(
    run: BenchmarkRun, *, recovered_at: datetime.datetime
) -> int:
    """Return abandoned in-flight items to a resumable state.

    The task-level distributed lease guarantees that no live worker owns the run
    when this executes. A persisted answer is intentionally retained so
    ``_run_item`` can continue at the judge boundary.
    """
    recovered = 0
    for item in run.items:
        if item.status != BenchmarkRunItemStatus.RUNNING.value:
            continue
        recovered += 1
        if item.judgment is not None:
            item.status = BenchmarkRunItemStatus.COMPLETED.value
            item.completed_at = recovered_at
        else:
            item.status = BenchmarkRunItemStatus.PENDING.value
            item.completed_at = None
    return recovered


def _mark_unfinished_items_error(
    run: BenchmarkRun, *, completed_at: datetime.datetime
) -> int:
    marked = 0
    for item in run.items:
        if item.status not in {
            BenchmarkRunItemStatus.PENDING.value,
            BenchmarkRunItemStatus.RUNNING.value,
        }:
            continue
        item.status = BenchmarkRunItemStatus.ERROR.value
        item.error_message = item.error_message or (
            "Benchmark execution ended before the item reached a terminal state"
        )
        item.completed_at = completed_at
        marked += 1
    return marked


def run_benchmark(db_session: Session, run_id: int) -> None:
    run = get_benchmark_run(db_session, run_id)
    if run is None:
        raise ValueError(f"Benchmark run {run_id} not found")
    if run.status not in {
        BenchmarkRunStatus.PENDING.value,
        BenchmarkRunStatus.RUNNING.value,
    }:
        return

    user = db_session.get(User, run.created_by)
    if user is None:
        run.status = BenchmarkRunStatus.ERROR.value
        run.completed_at = _utcnow()
        db_session.commit()
        raise ValueError("Benchmark run creator no longer exists")
    persona_id = get_best_persona_id_for_user(db_session, user)
    persona = db_session.get(Persona, persona_id) if persona_id is not None else None

    now = _utcnow()
    _recover_interrupted_items(run, recovered_at=now)
    run.status = BenchmarkRunStatus.RUNNING.value
    # This is also the durable recovery lease renewed at each item boundary.
    run.started_at = now
    db_session.commit()

    for item in run.items:
        db_session.refresh(run)
        if run.status == BenchmarkRunStatus.CANCELLED.value:
            break
        if item.status != BenchmarkRunItemStatus.PENDING.value:
            continue
        run.started_at = _utcnow()
        db_session.commit()
        _run_item(db_session, run=run, item=item, user=user, persona=persona)
        refresh_benchmark_run_counts(db_session, run)
        run.started_at = _utcnow()
        db_session.commit()

    db_session.refresh(run)
    if run.status != BenchmarkRunStatus.CANCELLED.value:
        finished_at = _utcnow()
        _mark_unfinished_items_error(run, completed_at=finished_at)
        db_session.flush()
        refresh_benchmark_run_counts(db_session, run)
        run.status = (
            BenchmarkRunStatus.COMPLETED.value
            if run.total_items > 0 and run.completed_items == run.total_items
            else BenchmarkRunStatus.ERROR.value
        )
        _generate_run_report(db_session, run=run, user=user, persona=persona)
        run.completed_at = finished_at
        db_session.commit()
