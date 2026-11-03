"""Per-workflow chat activity and captured research-model cost estimates."""

from collections.abc import Sequence
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session, aliased

from onyx.configs.constants import MessageType
from onyx.db.models import ChatMessage, ChatSession, User
from onyx.db.response_usage import get_response_usage
from onyx.db.user_usage import DELETED_USER_EXPORT_EMAIL

USAGE_WORKFLOWS = (
    "asv3",
    "asv3_exp_cropped",
    "asv3_fast",
    "asv3_fast_cropped",
    "asv3_kompozit",
    "asv3_kompozit_cropped",
)
_RESPONSE_BATCH_SIZE = 500


class WorkflowUsage(BaseModel):
    query_count: int = 0
    session_count: int = 0
    cost_cents: float | None = 0.0
    known_cost_cents: float = 0.0
    average_cost_cents_per_query: float | None = 0.0
    average_cost_cents_per_session: float | None = 0.0
    cost_status: Literal["complete", "partial", "unavailable", "running"] = "complete"
    unpriced_call_count: int = 0
    complete_query_count: int = 0
    complete_session_count: int = 0
    complete_query_cost_cents: float = 0.0
    complete_session_cost_cents: float = 0.0
    average_cost_cents_per_complete_query: float | None = 0.0
    average_cost_cents_per_complete_session: float | None = 0.0
    partial_query_count: int = 0
    running_query_count: int = 0
    unavailable_query_count: int = 0


class _WorkflowResponse(BaseModel):
    email: str
    workflow: str
    message_id: int
    user_message_id: int
    session_id: UUID


class _CapturedQueryCost(BaseModel):
    session_id: UUID
    response_count: int = 0
    complete_response_count: int = 0
    priced_response_count: int = 0
    running_response_count: int = 0
    known_cost_cents: float = 0.0
    unpriced_call_count: int = 0


class _CapturedWorkflowCost(BaseModel):
    # Alternative answers for one prompt may cross response batches.
    queries: dict[int, _CapturedQueryCost] = Field(default_factory=dict)


class _CapturedSessionCost(BaseModel):
    query_count: int = 0
    complete_query_count: int = 0
    known_cost_cents: float = 0.0


def _query_cost_status(
    cost: _CapturedQueryCost,
) -> Literal["complete", "partial", "unavailable", "running"]:
    if cost.running_response_count:
        return "running"
    if cost.complete_response_count == cost.response_count:
        return "complete"
    return "partial" if cost.priced_response_count else "unavailable"


def empty_workflow_usage() -> dict[str, WorkflowUsage]:
    return {workflow: WorkflowUsage() for workflow in USAGE_WORKFLOWS}


def _add_response_costs(
    db_session: Session,
    responses: Sequence[_WorkflowResponse],
    costs: dict[tuple[str, str], _CapturedWorkflowCost],
) -> None:
    summaries = get_response_usage(
        db_session, [response.message_id for response in responses], admin=True
    )
    for response in responses:
        workflow_cost = costs.setdefault(
            (response.email, response.workflow), _CapturedWorkflowCost()
        )
        cost = workflow_cost.queries.setdefault(
            response.user_message_id,
            _CapturedQueryCost(session_id=response.session_id),
        )
        cost.response_count += 1
        summary = summaries.get(response.message_id)
        if summary is None:
            continue
        cost.complete_response_count += summary.status == "complete"
        cost.running_response_count += summary.status == "running"
        cost.unpriced_call_count += summary.unpriced_calls
        if summary.known_cost_usd is not None:
            cost.priced_response_count += 1
            cost.known_cost_cents += summary.known_cost_usd * 100


def get_workflow_usage_by_email(
    db_session: Session,
    start: datetime,
    end: datetime,
) -> dict[str, dict[str, WorkflowUsage]]:
    """Count prompts/sessions over [start, end), using answer workflow identity.

    Costs use the same captured research LLM estimates as individual answers;
    the daily usage ledger cannot be split by workflow. Unknown historical
    workflows stay unclassified. A mixed session counts in each mode it used.
    """
    user_message = aliased(ChatMessage)
    scoped_responses = (
        select(
            func.coalesce(User.email, DELETED_USER_EXPORT_EMAIL).label("email"),
            ChatMessage.workflow.label("workflow"),
            ChatMessage.id.label("message_id"),
            user_message.id.label("user_message_id"),
            ChatSession.id.label("chat_session_id"),
        )
        .select_from(ChatMessage)
        .join(ChatSession, ChatSession.id == ChatMessage.chat_session_id)
        .join(
            user_message,
            (user_message.id == ChatMessage.parent_message_id)
            & (user_message.chat_session_id == ChatSession.id),
        )
        .outerjoin(User, ChatSession.user_id == User.id)
        .where(
            user_message.time_sent >= start,
            user_message.time_sent < end,
            user_message.message_type == MessageType.USER,
            ChatMessage.message_type == MessageType.ASSISTANT,
            ChatMessage.workflow.in_(USAGE_WORKFLOWS),
            ChatSession.benchmark_flow.is_(False),
        )
        .cte("workflow_responses")
    )
    activity = (
        select(
            scoped_responses.c.email,
            scoped_responses.c.workflow,
            func.count(func.distinct(scoped_responses.c.user_message_id)).label(
                "query_count"
            ),
            func.count(func.distinct(scoped_responses.c.chat_session_id)).label(
                "session_count"
            ),
        )
        .group_by(scoped_responses.c.email, scoped_responses.c.workflow)
        .subquery()
    )
    # Counts and response IDs share one statement snapshot, including while new
    # users/turns arrive. Graph reads remain bounded independently of report size.
    responses = db_session.execute(
        select(
            scoped_responses.c.email,
            scoped_responses.c.workflow,
            scoped_responses.c.message_id,
            scoped_responses.c.user_message_id,
            scoped_responses.c.chat_session_id,
            activity.c.query_count,
            activity.c.session_count,
        )
        .join(
            activity,
            (activity.c.email == scoped_responses.c.email)
            & (activity.c.workflow == scoped_responses.c.workflow),
        )
        .order_by(scoped_responses.c.message_id)
    ).yield_per(_RESPONSE_BATCH_SIZE)
    usage_by_email: dict[str, dict[str, WorkflowUsage]] = {}
    costs: dict[tuple[str, str], _CapturedWorkflowCost] = {}
    batch: list[_WorkflowResponse] = []
    for (
        email,
        workflow,
        message_id,
        user_message_id,
        session_id,
        query_count,
        session_count,
    ) in responses:
        if email not in usage_by_email:
            usage_by_email[email] = empty_workflow_usage()
        usage = usage_by_email[email][workflow]
        usage.query_count = query_count
        usage.session_count = session_count
        batch.append(
            _WorkflowResponse(
                email=email,
                workflow=workflow,
                message_id=message_id,
                user_message_id=user_message_id,
                session_id=session_id,
            )
        )
        if len(batch) == _RESPONSE_BATCH_SIZE:
            _add_response_costs(db_session, batch, costs)
            batch.clear()
    if batch:
        _add_response_costs(db_session, batch, costs)

    for (email, workflow), cost in costs.items():
        usage = usage_by_email[email][workflow]
        sessions: dict[UUID, _CapturedSessionCost] = {}
        priced_responses = 0
        for query in cost.queries.values():
            status = _query_cost_status(query)
            usage.known_cost_cents += query.known_cost_cents
            usage.unpriced_call_count += query.unpriced_call_count
            priced_responses += query.priced_response_count
            session = sessions.setdefault(query.session_id, _CapturedSessionCost())
            session.query_count += 1
            session.known_cost_cents += query.known_cost_cents
            if status == "complete":
                usage.complete_query_count += 1
                usage.complete_query_cost_cents += query.known_cost_cents
                session.complete_query_count += 1
            elif status == "running":
                usage.running_query_count += 1
            elif status == "partial":
                usage.partial_query_count += 1
            else:
                usage.unavailable_query_count += 1
        for session in sessions.values():
            if session.complete_query_count == session.query_count:
                usage.complete_session_count += 1
                usage.complete_session_cost_cents += session.known_cost_cents
        usage.average_cost_cents_per_complete_query = (
            usage.complete_query_cost_cents / usage.complete_query_count
            if usage.complete_query_count
            else None
        )
        usage.average_cost_cents_per_complete_session = (
            usage.complete_session_cost_cents / usage.complete_session_count
            if usage.complete_session_count
            else None
        )
        if usage.running_query_count:
            usage.cost_status = "running"
        elif usage.complete_query_count == usage.query_count:
            usage.cost_status = "complete"
        elif priced_responses:
            usage.cost_status = "partial"
        else:
            usage.cost_status = "unavailable"
        complete_cost = (
            usage.known_cost_cents if usage.cost_status == "complete" else None
        )
        usage.cost_cents = complete_cost
        usage.average_cost_cents_per_query = (
            complete_cost / usage.query_count if complete_cost is not None else None
        )
        usage.average_cost_cents_per_session = (
            complete_cost / usage.session_count if complete_cost is not None else None
        )
    return usage_by_email
