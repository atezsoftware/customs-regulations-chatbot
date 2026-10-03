"""Metadata-only usage reads after callers authorize chat or admin history access."""

from __future__ import annotations

from collections import defaultdict
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.configs.constants import MessageType
from onyx.db.models import AnswerGraphNode, AnswerGraphRun, ChatMessage, ChatSession
from onyx.llm.usage_cost import GenerationCost, ResponseUsage, token_count
from onyx.tracing.flows import IMAGE_FLOWS, LLMFlow
from shared_configs.contextvars import get_current_tenant_id


def get_response_usage(
    db_session: Session,
    message_ids: list[int],
    *,
    user_id: UUID | None = None,
    admin: bool = False,
) -> dict[int, ResponseUsage]:
    if not message_ids or (user_id is None and not admin):
        return {}
    statement = (
        select(
            ChatMessage.id,
            ChatMessage.processing_duration_seconds,
            AnswerGraphRun.id,
            AnswerGraphRun.status,
            AnswerGraphRun.capture_status,
            AnswerGraphRun.time_created,
            AnswerGraphRun.time_finished,
        )
        .join(ChatSession, ChatSession.id == ChatMessage.chat_session_id)
        .outerjoin(
            AnswerGraphRun,
            (AnswerGraphRun.assistant_message_id == ChatMessage.id)
            & (AnswerGraphRun.tenant_id == get_current_tenant_id()),
        )
        .where(
            ChatMessage.id.in_(message_ids),
            ChatMessage.message_type == MessageType.ASSISTANT,
        )
    )
    if not admin:
        statement = statement.where(
            ChatSession.user_id == user_id, ChatSession.deleted.is_(False)
        )
    rows = list(db_session.execute(statement))
    by_run = {row[2]: row[0] for row in rows if row[2] is not None}
    summaries = {row[0]: ResponseUsage(duration_seconds=row[1]) for row in rows}
    incomplete: set[int] = set()
    for message_id, duration, run_id, status, capture, started, finished in rows:
        summary = summaries[message_id]
        if finished is not None and started is not None:
            summary.duration_seconds = max(0, (finished - started).total_seconds())
        if run_id is not None:
            summary.status = (
                "running" if status in {"RUNNING", "FINALIZING"} else "complete"
            )
        if capture != "COMPLETE":
            incomplete.add(message_id)
    if not by_run:
        return summaries
    # Only JSON metadata, never encrypted payloads or private reasoning.
    generations = db_session.execute(
        select(
            AnswerGraphNode.run_id,
            AnswerGraphNode.attributes,
            AnswerGraphNode.capture_status,
            AnswerGraphNode.operation,
        ).where(
            AnswerGraphNode.run_id.in_(by_run), AnswerGraphNode.kind == "generation"
        )
    )
    groups: dict[int, dict[str, GenerationCost]] = defaultdict(dict)
    service_flows = {
        *IMAGE_FLOWS,
        LLMFlow.EMBED_QUERY,
        LLMFlow.EMBED_PASSAGE,
        LLMFlow.REGULATORY_EMBEDDING_BATCH,
        LLMFlow.RERANK,
        LLMFlow.INTENT_CLASSIFICATION,
        LLMFlow.STT,
        LLMFlow.TTS,
    }
    for run_id, attributes, capture, operation in generations:
        message_id = by_run[run_id]
        summary = summaries[message_id]
        if operation in service_flows:
            summary.excluded_service_calls += 1
            continue
        summary.calls += 1
        try:
            cost = GenerationCost.model_validate(attributes.get("usage_cost"))
        except (ValidationError, AttributeError):
            usage = attributes.get("usage") or {}
            cost = GenerationCost(
                model=attributes.get("model") or "unknown",
                provider=attributes.get("provider"),
                source="unavailable",
                priced_at=next(row[5] for row in rows if row[0] == message_id),
                input_tokens=token_count(usage, "input_tokens", "prompt_tokens"),
                output_tokens=token_count(usage, "output_tokens", "completion_tokens"),
                reasoning_tokens=token_count(usage, "reasoning_tokens"),
            )
        if not cost.complete or capture != "COMPLETE":
            summary.unpriced_calls += 1
            incomplete.add(message_id)
        if any(line.cost_usd is not None for line in cost.lines) or cost.complete:
            summary.known_cost_usd = (summary.known_cost_usd or 0) + cost.known_cost_usd
        key = repr(
            (
                cost.model,
                cost.provider,
                cost.source,
                cost.complete,
                [(line.category, line.usd_per_million) for line in cost.lines],
            )
        )
        previous = groups[message_id].get(key)
        if previous is None:
            groups[message_id][key] = cost
        else:
            for field in ("input_tokens", "output_tokens", "reasoning_tokens"):
                left, right = getattr(previous, field), getattr(cost, field)
                setattr(
                    previous,
                    field,
                    left + right if left is not None and right is not None else None,
                )
            previous.priced_at = max(previous.priced_at, cost.priced_at)
            previous.known_cost_usd += cost.known_cost_usd
            for left, right in zip(previous.lines, cost.lines):
                left.tokens += right.tokens
                left.cost_usd = (
                    left.cost_usd + right.cost_usd
                    if left.cost_usd is not None and right.cost_usd is not None
                    else None
                )
    for message_id, summary in summaries.items():
        summary.models = list(groups[message_id].values())
        if summary.status == "running":
            continue
        if not summary.calls or summary.known_cost_usd is None:
            summary.status = "unavailable"
        elif message_id in incomplete:
            summary.status = "partial"
        else:
            summary.status = "complete"
            summary.total_cost_usd = summary.known_cost_usd
    return summaries
