import datetime
import time
from collections import defaultdict
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from onyx.chat.chat_state import ChatStateContainer
from onyx.chat.models import AnswerStream
from onyx.chat.process_message import gather_stream_full, handle_stream_message_objects
from onyx.configs.constants import DEFAULT_PERSONA_ID
from onyx.context.search.models import BaseFilters, SearchDoc
from onyx.db.chat import create_chat_session
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.enums import (
    BenchmarkCostSource,
    BenchmarkRunFailureCode,
    BenchmarkRunItemPhase,
    BenchmarkRunItemStatus,
    BenchmarkRunStatus,
)
from onyx.db.models import BenchmarkRun, BenchmarkRunItem, Persona, User, UserFile
from onyx.db.regulatory_benchmark import (
    add_benchmark_judgment,
    claim_benchmark_run_item,
    get_benchmark_run,
    get_benchmark_run_for_update,
    get_benchmark_run_item,
    refresh_benchmark_run_counts,
    touch_benchmark_run_items,
)
from onyx.db.regulatory_chunks import get_chunks_for_file
from onyx.llm.constants import LlmProviderNames
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
from onyx.server.query_and_chat.models import MessageOrigin, SendMessageRequest
from onyx.tracing.framework.create import ensure_trace
from onyx.utils.logger import setup_logger

logger = setup_logger()

_MAX_TOOL_RESULT_CHARS = 20_000
_MAX_REASONING_CHARS = 30_000
_PROGRESS_HEARTBEAT_INTERVAL_SECONDS = 15
_DOCUMENT_SET_SCOPE_TOOL_ERROR = (
    "Selected Document Sets are outside this agent's knowledge scope."
)
_BENCHMARK_DOCUMENT_SET_SCOPE_ERROR = (
    "Benchmark search could not access the selected document set scope"
)


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _with_item_progress_heartbeat(
    packets: AnswerStream,
    *,
    run_id: int,
    item_id: int,
) -> AnswerStream:
    """Persist item liveness only while the chat stream makes forward progress."""
    last_heartbeat = time.monotonic()
    for packet in packets:
        now = time.monotonic()
        if now - last_heartbeat >= _PROGRESS_HEARTBEAT_INTERVAL_SECONDS:
            with get_session_with_current_tenant() as progress_session:
                touch_benchmark_run_items(
                    progress_session,
                    run_id,
                    item_ids=[item_id],
                    heartbeat_at=_utcnow(),
                )
            last_heartbeat = now
        yield packet


def _override_provider_name(
    provider_selector: str, provider_id: int | None
) -> str | None:
    return None if provider_id is not None else provider_selector


def _override_provider_type(
    provider_selector: str, provider_id: int | None
) -> str | None:
    if provider_id is not None:
        # The exact row ID is authoritative across all configured provider
        # implementations; adding an assumed type can only make it conflict.
        return None
    # A named provider whose name equals its type needs exact-name resolution.
    return (
        None
        if provider_selector == LlmProviderNames.OPENROUTER
        else LlmProviderNames.OPENROUTER
    )


def _usage_cost(
    db_session: Session, usage_calls: list[LLMCallUsage]
) -> tuple[int, int, float | None, str]:
    input_tokens = sum(call.input_tokens for call in usage_calls)
    output_tokens = sum(call.output_tokens for call in usage_calls)
    total_cost_cents = 0.0
    pricing_available = bool(usage_calls)
    for call in usage_calls:
        price = get_model_price_per_million(
            call.model, call.provider, db_session=db_session
        )
        known_rates = [
            rate
            for rate in (
                price.input_per_mtok,
                price.output_per_mtok,
                price.cache_per_mtok,
            )
            if rate is not None
        ]
        provider_cost_is_authoritative = call.cost_cents is not None and (
            call.cost_cents > 0
            or call.input_tokens + call.output_tokens == 0
            or (known_rates and all(rate == 0 for rate in known_rates))
        )
        if provider_cost_is_authoritative:
            assert call.cost_cents is not None
            total_cost_cents += call.cost_cents
            continue
        non_cached_input_tokens = max(
            call.input_tokens - call.cache_read_tokens,
            0,
        )
        input_cost, output_cost = compute_cost_cents(
            call.model,
            call.provider,
            non_cached_input_tokens,
            call.output_tokens,
            call.cache_read_tokens,
            db_session=db_session,
        )
        total_cost_cents += input_cost + output_cost
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
    by_chunk_identity: dict[tuple[str, int], SearchDoc] = {}
    for document in top_documents:
        by_document[document.document_id].append(document)
        by_chunk_identity.setdefault(
            (document.document_id, document.chunk_ind), document
        )
    document_offsets: dict[str, int] = defaultdict(int)
    sources: list[dict[str, object]] = []
    for citation in citation_info:
        citation_chunk_ind = getattr(citation, "chunk_ind", None)
        if citation_chunk_ind is not None:
            document = by_chunk_identity.get((citation.document_id, citation_chunk_ind))
            if document is None:
                continue
            sources.append(
                _source_snapshot(db_session, document, citation.citation_number)
            )
            continue
        matches = by_document.get(citation.document_id, [])
        if not matches:
            continue
        offset = document_offsets[citation.document_id]
        document = matches[min(offset, len(matches) - 1)]
        document_offsets[citation.document_id] += 1
        sources.append(_source_snapshot(db_session, document, citation.citation_number))
    return sources


def _get_item_persona(
    db_session: Session,
) -> Persona:
    # Benchmark candidates must differ only by model. The default persona is
    # the same deterministic starting point as a new unconfigured chat session.
    persona = db_session.get(Persona, DEFAULT_PERSONA_ID)
    if persona is None:
        raise ValueError("Default new-session persona is missing")
    return persona


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


def _has_document_set_scope_tool_failure(response: object) -> bool:
    return any(
        getattr(tool_call, "tool_name", None) == "internal_search"
        and _DOCUMENT_SET_SCOPE_TOOL_ERROR in str(getattr(tool_call, "tool_result", ""))
        for tool_call in getattr(response, "tool_calls", [])
    )


def _judge_completed_item(
    db_session: Session,
    *,
    run: BenchmarkRun,
    item: BenchmarkRunItem,
    user: User,
    persona: Persona | None,
) -> None:
    judge_provider_id = getattr(run, "judge_provider_id", None)
    judge_llm = get_llm_for_persona(
        persona,
        user,
        LLMOverride(
            model_provider=_override_provider_name(
                run.judge_provider, judge_provider_id
            ),
            model_provider_type=_override_provider_type(
                run.judge_provider, judge_provider_id
            ),
            model_provider_id=judge_provider_id,
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
    provider_id = getattr(item, "provider_id", None)
    override = LLMOverride(
        model_provider=_override_provider_name(item.provider, provider_id),
        model_provider_type=_override_provider_type(item.provider, provider_id),
        model_provider_id=provider_id,
        model_version=item.model_id,
        temperature=0,
    )
    item.execution_phase = BenchmarkRunItemPhase.PREPARING_SESSION.value
    item.heartbeat_at = _utcnow()
    db_session.commit()
    chat_session = create_chat_session(
        db_session,
        description=f"Benchmark run {run.id}, item {item.id}",
        user_id=user.id,
        persona_id=persona.id if persona else None,
        llm_override=override,
        benchmark_flow=True,
    )
    item.chat_session_id = chat_session.id
    item.execution_phase = (
        BenchmarkRunItemPhase.RESEARCHING.value
        if run.deep_research
        else BenchmarkRunItemPhase.ANSWERING.value
    )
    item.heartbeat_at = _utcnow()
    db_session.commit()
    as_of_date_value = item.question_snapshot.get("as_of_date")
    as_of_date = (
        datetime.date.fromisoformat(str(as_of_date_value)) if as_of_date_value else None
    )
    request = SendMessageRequest(
        message=str(item.question_snapshot.get("prompt") or item.question.prompt),
        chat_session_id=chat_session.id,
        llm_override=override,
        internal_search_filters=BaseFilters(
            document_set=[item.question.document_set.name],
            as_of_date=as_of_date,
        ),
        deep_research=run.deep_research,
        atez_search=run.search_mode == "v1",
        atez_search_v2=run.search_mode == "v2",
        stream=True,
        include_citations=True,
        origin=MessageOrigin.BENCHMARK,
    )
    state_container = ChatStateContainer()
    with benchmark_usage_capture() as answer_usage:
        response = gather_stream_full(
            _with_item_progress_heartbeat(
                handle_stream_message_objects(
                    request, user, external_state_container=state_container
                ),
                run_id=run.id,
                item_id=item.id,
            ),
            state_container,
        )
    item.duration_ms = round((time.monotonic() - started) * 1000)
    input_tokens, output_tokens, cost_cents, cost_source = _usage_cost(
        db_session, answer_usage
    )
    item.input_tokens = input_tokens
    item.output_tokens = output_tokens
    item.total_tokens = input_tokens + output_tokens
    item.cost_cents = cost_cents
    item.cost_source = cost_source
    item.answer_reasoning = (response.pre_answer_reasoning or "")[
        :_MAX_REASONING_CHARS
    ] or None
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
    if _has_document_set_scope_tool_failure(response):
        item.error_message = _BENCHMARK_DOCUMENT_SET_SCOPE_ERROR
        db_session.commit()
        raise RuntimeError(_BENCHMARK_DOCUMENT_SET_SCOPE_ERROR)
    if response.error_msg:
        # Preserve provider usage and partial execution evidence even though the
        # canonical answer boundary was not reached.
        db_session.commit()
        raise RuntimeError(response.error_msg)

    item.final_result = response.answer
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
    already_claimed: bool = False,
) -> None:
    started = time.monotonic()
    if not already_claimed:
        item.status = BenchmarkRunItemStatus.RUNNING.value
        item.started_at = _utcnow()
        item.execution_phase = BenchmarkRunItemPhase.STARTING.value
        item.heartbeat_at = item.started_at
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
            item.execution_phase = BenchmarkRunItemPhase.JUDGING.value
            item.heartbeat_at = _utcnow()
            db_session.commit()
            _judge_completed_item(
                db_session, run=run, item=item, user=user, persona=persona
            )
            db_session.refresh(run)
            if run.status != BenchmarkRunStatus.RUNNING.value:
                db_session.rollback()
                reloaded_item = db_session.get(BenchmarkRunItem, item.id)
                assert reloaded_item is not None
                item = reloaded_item
                return
            item.status = BenchmarkRunItemStatus.COMPLETED.value
        except Exception as judge_error:
            db_session.rollback()
            reloaded_run = db_session.get(BenchmarkRun, run.id)
            reloaded_item = db_session.get(BenchmarkRunItem, item.id)
            assert reloaded_item is not None
            item = reloaded_item
            if (
                reloaded_run is not None
                and reloaded_run.status == BenchmarkRunStatus.RUNNING.value
                and item.status != BenchmarkRunItemStatus.CANCELLED.value
            ):
                logger.exception("Benchmark judge failed for item %s", item.id)
                item.status = BenchmarkRunItemStatus.ERROR.value
                item.judge_error = str(judge_error)
                item.error_message = f"Judge failed: {judge_error}"
            else:
                logger.info(
                    "Ignoring late judge failure for inactive benchmark item %s",
                    item.id,
                )
    except Exception as error:
        db_session.rollback()
        reloaded_run = db_session.get(BenchmarkRun, run.id)
        reloaded_item = db_session.get(BenchmarkRunItem, item.id)
        assert reloaded_item is not None
        item = reloaded_item
        if (
            reloaded_run is not None
            and reloaded_run.status == BenchmarkRunStatus.RUNNING.value
            and item.status != BenchmarkRunItemStatus.CANCELLED.value
        ):
            logger.exception("Benchmark run %s item %s failed", run.id, item.id)
            item.status = BenchmarkRunItemStatus.ERROR.value
            item.error_message = str(error)
        else:
            logger.info(
                "Ignoring late execution failure for inactive benchmark item %s",
                item.id,
            )
    finally:
        if item.status != BenchmarkRunItemStatus.CANCELLED.value:
            if item.duration_ms is None:
                item.duration_ms = round((time.monotonic() - started) * 1000)
            if item.status in {
                BenchmarkRunItemStatus.COMPLETED.value,
                BenchmarkRunItemStatus.ERROR.value,
            }:
                item.execution_phase = None
                item.completed_at = _utcnow()
                item.heartbeat_at = item.completed_at
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
    grouped: dict[tuple[str, int | None, str], list[BenchmarkRunItem]] = defaultdict(
        list
    )
    for item in judged_items:
        grouped[(item.provider, item.provider_id, item.model_id)].append(item)
    aggregates: list[dict[str, object]] = []
    for (provider, provider_id, model_id), items in grouped.items():
        scores = [item.judgment.overall_score for item in items if item.judgment]
        recalls = [
            item.citation_recall for item in items if item.citation_recall is not None
        ]
        aggregates.append(
            {
                "provider": provider,
                "provider_id": provider_id,
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
            "provider_id": item.provider_id,
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
    judge_provider_id = getattr(run, "judge_provider_id", None)
    judge_llm = get_llm_for_persona(
        persona,
        user,
        LLMOverride(
            model_provider=_override_provider_name(
                run.judge_provider, judge_provider_id
            ),
            model_provider_type=_override_provider_type(
                run.judge_provider, judge_provider_id
            ),
            model_provider_id=judge_provider_id,
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
    when this executes. Persisted answers resume at the judge boundary, while a
    persisted pre-answer failure marker becomes terminal without another paid call.
    """
    recovered = 0
    for item in run.items:
        if item.status != BenchmarkRunItemStatus.RUNNING.value:
            continue
        recovered += 1
        if item.judgment is not None:
            item.status = BenchmarkRunItemStatus.COMPLETED.value
            item.heartbeat_at = recovered_at
            item.completed_at = recovered_at
        elif item.error_message is not None and item.final_result is None:
            item.status = BenchmarkRunItemStatus.ERROR.value
            item.execution_phase = None
            item.heartbeat_at = recovered_at
            item.completed_at = recovered_at
        else:
            item.status = BenchmarkRunItemStatus.PENDING.value
            item.execution_phase = None
            item.heartbeat_at = None
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
        item.execution_phase = None
        item.error_message = item.error_message or (
            "Benchmark execution ended before the item reached a terminal state"
        )
        item.heartbeat_at = completed_at
        item.completed_at = completed_at
        marked += 1
    return marked


def prepare_benchmark_run(db_session: Session, run_id: int) -> list[int]:
    """Move a queued run into execution and return its resumable item IDs."""
    run = get_benchmark_run_for_update(db_session, run_id)
    if run is None:
        logger.warning("Ignoring delivery for missing benchmark run %s", run_id)
        return []
    if run.status not in {
        BenchmarkRunStatus.QUEUED.value,
        BenchmarkRunStatus.RUNNING.value,
    }:
        return []

    user = db_session.get(User, run.created_by)
    if user is None:
        run.status = BenchmarkRunStatus.ERROR.value
        run.failure_code = BenchmarkRunFailureCode.EXECUTION_FAILED.value
        run.failure_message = "Benchmark run creator no longer exists"
        run.completed_at = _utcnow()
        db_session.commit()
        raise ValueError("Benchmark run creator no longer exists")

    now = _utcnow()
    _recover_interrupted_items(run, recovered_at=now)
    run.status = BenchmarkRunStatus.RUNNING.value
    if run.started_at is None:
        run.started_at = now
    run.heartbeat_at = now
    item_ids = [
        item.id
        for item in run.items
        if item.status == BenchmarkRunItemStatus.PENDING.value
    ]
    db_session.commit()
    return item_ids


def run_benchmark_item(db_session: Session, run_id: int, item_id: int) -> None:
    """Execute one item in an isolated process/session after an atomic claim."""
    started_at = _utcnow()
    if not claim_benchmark_run_item(
        db_session,
        run_id=run_id,
        item_id=item_id,
        started_at=started_at,
    ):
        logger.info("Benchmark item %s is no longer pending", item_id)
        return

    run_claimed_benchmark_item(db_session, run_id, item_id)


def run_claimed_benchmark_item(db_session: Session, run_id: int, item_id: int) -> None:
    """Execute an item already fenced as running by its coordinator."""

    run = db_session.get(BenchmarkRun, run_id)
    if run is None or run.status != BenchmarkRunStatus.RUNNING.value:
        return
    item = get_benchmark_run_item(db_session, run_id=run_id, item_id=item_id)
    if item is None:
        raise ValueError(f"Benchmark item {item_id} does not belong to run {run_id}")
    if item.status != BenchmarkRunItemStatus.RUNNING.value:
        logger.info("Benchmark item %s is no longer running", item_id)
        return
    user = db_session.get(User, run.created_by)
    if user is None:
        raise ValueError("Benchmark run creator no longer exists")
    persona = _get_item_persona(db_session)
    _run_item(
        db_session,
        run=run,
        item=item,
        user=user,
        persona=persona,
        already_claimed=True,
    )


def finalize_benchmark_run(
    db_session: Session, run_id: int, *, had_execution_timeout: bool = False
) -> None:
    """Aggregate terminal items and generate the run report exactly once."""
    run = get_benchmark_run_for_update(db_session, run_id)
    if run is None or run.status != BenchmarkRunStatus.RUNNING.value:
        return
    user = db_session.get(User, run.created_by)
    if user is None:
        raise ValueError("Benchmark run creator no longer exists")

    finished_at = _utcnow()
    _mark_unfinished_items_error(run, completed_at=finished_at)
    db_session.flush()
    refresh_benchmark_run_counts(db_session, run)
    if run.total_items > 0 and run.completed_items == run.total_items:
        run.status = BenchmarkRunStatus.COMPLETED.value
        run.failure_code = None
        run.failure_message = None
    else:
        run.status = BenchmarkRunStatus.ERROR.value
        run.failure_code = (
            BenchmarkRunFailureCode.EXECUTION_TIMEOUT.value
            if had_execution_timeout
            else BenchmarkRunFailureCode.EXECUTION_FAILED.value
        )
        run.failure_message = "One or more benchmark items failed"

    report_persona = _get_item_persona(db_session) if run.items else None
    run.completed_at = finished_at
    run.heartbeat_at = finished_at
    db_session.commit()

    # Report generation can take another provider call. Commit the terminal run
    # first so no row lock is held and a report failure cannot leave it running.
    # Sessions do not expire on commit, while item judgments are written by
    # separate child processes. Reload the graph so report aggregation observes
    # those committed relationships instead of the coordinator's stale identity map.
    db_session.expire_all()
    report_run = get_benchmark_run(db_session, run_id)
    if report_run is None:
        raise ValueError(f"Benchmark run {run_id} disappeared before reporting")
    _generate_run_report(
        db_session,
        run=report_run,
        user=user,
        persona=report_persona,
    )
    db_session.commit()


def run_benchmark(db_session: Session, run_id: int) -> None:
    run = get_benchmark_run_for_update(db_session, run_id)
    if run is None:
        logger.warning("Ignoring delivery for missing benchmark run %s", run_id)
        return
    if run.status not in {
        BenchmarkRunStatus.QUEUED.value,
        BenchmarkRunStatus.RUNNING.value,
    }:
        return

    user = db_session.get(User, run.created_by)
    if user is None:
        run.status = BenchmarkRunStatus.ERROR.value
        run.failure_code = BenchmarkRunFailureCode.EXECUTION_FAILED.value
        run.failure_message = "Benchmark run creator no longer exists"
        run.completed_at = _utcnow()
        db_session.commit()
        raise ValueError("Benchmark run creator no longer exists")
    now = _utcnow()
    _recover_interrupted_items(run, recovered_at=now)
    run.status = BenchmarkRunStatus.RUNNING.value
    if run.started_at is None:
        run.started_at = now
    run.heartbeat_at = now
    db_session.commit()

    report_persona: Persona | None = None
    for item in run.items:
        db_session.refresh(run)
        if run.status != BenchmarkRunStatus.RUNNING.value:
            break
        if item.status != BenchmarkRunItemStatus.PENDING.value:
            continue
        persona = _get_item_persona(db_session)
        if report_persona is None:
            report_persona = persona
        run.heartbeat_at = _utcnow()
        db_session.commit()
        _run_item(db_session, run=run, item=item, user=user, persona=persona)
        refresh_benchmark_run_counts(db_session, run)
        run.heartbeat_at = _utcnow()
        db_session.commit()

    db_session.refresh(run)
    if run.status == BenchmarkRunStatus.RUNNING.value:
        finished_at = _utcnow()
        _mark_unfinished_items_error(run, completed_at=finished_at)
        db_session.flush()
        refresh_benchmark_run_counts(db_session, run)
        if run.total_items > 0 and run.completed_items == run.total_items:
            run.status = BenchmarkRunStatus.COMPLETED.value
            run.failure_code = None
            run.failure_message = None
        else:
            run.status = BenchmarkRunStatus.ERROR.value
            run.failure_code = BenchmarkRunFailureCode.EXECUTION_FAILED.value
            run.failure_message = "One or more benchmark items failed"
        if report_persona is None and run.items:
            report_persona = _get_item_persona(db_session)
        _generate_run_report(
            db_session,
            run=run,
            user=user,
            persona=report_persona,
        )
        run.completed_at = finished_at
        run.heartbeat_at = finished_at
        db_session.commit()
