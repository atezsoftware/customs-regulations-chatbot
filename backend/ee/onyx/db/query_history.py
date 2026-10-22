from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from sqlalchemy import BinaryExpression, ColumnElement, asc, desc, distinct, or_
from sqlalchemy.orm import Session, contains_eager, joinedload
from sqlalchemy.sql import case, func, select
from sqlalchemy.sql.expression import UnaryExpression, literal

from ee.onyx.background.task_name_builders import QUERY_HISTORY_TASK_NAME_PREFIX
from onyx.configs.constants import MessageType, QAFeedbackType
from onyx.db.models import (
    ChatMessage,
    ChatMessageFeedback,
    ChatSession,
    ChatSessionFeedback,
    TaskQueueState,
    User,
)
from onyx.db.tasks import get_all_tasks_with_prefix


def _build_filter_conditions(
    start_time: datetime | None,
    end_time: datetime | None,
    feedback_filter: QAFeedbackType | None,
    session_rating: int | None = None,
    user_id: UUID | None = None,
    user_email: str | None = None,
    user_query: str | None = None,
    api_key_query: str | None = None,
    q: str | None = None,
) -> list[ColumnElement]:
    """Keep page contents and total counts under the same session predicates."""
    conditions = []

    if start_time is not None:
        conditions.append(ChatSession.time_created >= start_time)
    if end_time is not None:
        conditions.append(ChatSession.time_created <= end_time)

    if user_id is not None:
        conditions.append(ChatSession.user_id == user_id)
    email = (user_email or "").strip()
    if email:
        conditions.append(ChatSession.user.has(func.lower(User.email) == email.lower()))
    user_search = (user_query or "").strip()
    if user_search:
        conditions.append(
            or_(
                ChatSession.user.has(
                    func.lower(User.email).contains(
                        user_search.lower(), autoescape=True
                    )
                ),
                (ChatSession.api_key_id.is_not(None))
                & ChatSession.external_user_name.icontains(
                    user_search, autoescape=True
                ),
            )
        )
    key_search = (api_key_query or "").strip()
    if key_search:
        conditions.append(
            or_(
                ChatSession.api_key_name.icontains(key_search, autoescape=True),
                ChatSession.api_key_display.icontains(key_search, autoescape=True),
                ChatSession.api_key_id == int(key_search)
                if key_search.isascii()
                and key_search.isdigit()
                and len(key_search) <= 10
                and int(key_search) <= 2147483647
                else literal(False),
            )
        )
    query_search = (q or "").strip()
    if query_search:
        text_matches = or_(
            ChatSession.description.icontains(query_search, autoescape=True),
            ChatSession.messages.any(
                (ChatMessage.message_type == MessageType.USER)
                & ChatMessage.message.icontains(query_search, autoescape=True)
            ),
        )
        try:
            session_id = UUID(query_search)
        except ValueError:
            conditions.append(text_matches)
        else:
            conditions.append(or_(ChatSession.id == session_id, text_matches))

    if session_rating is not None:
        conditions.append(
            ChatSession.session_feedback.has(
                ChatSessionFeedback.rating == session_rating
            )
        )

    if feedback_filter is not None:
        feedback_subq = (
            select(ChatMessage.chat_session_id)
            .join(ChatMessageFeedback)
            .group_by(ChatMessage.chat_session_id)
            .having(
                case(
                    (
                        case(
                            {literal(feedback_filter == QAFeedbackType.LIKE): True},
                            else_=False,
                        ),
                        func.bool_and(ChatMessageFeedback.is_positive),
                    ),
                    (
                        case(
                            {literal(feedback_filter == QAFeedbackType.DISLIKE): True},
                            else_=False,
                        ),
                        func.bool_and(func.not_(ChatMessageFeedback.is_positive)),
                    ),
                    else_=func.bool_or(ChatMessageFeedback.is_positive)
                    & func.bool_or(func.not_(ChatMessageFeedback.is_positive)),
                )
            )
        )
        conditions.append(ChatSession.id.in_(feedback_subq))

    return conditions


def get_total_filtered_chat_sessions_count(
    db_session: Session,
    start_time: datetime | None,
    end_time: datetime | None,
    feedback_filter: QAFeedbackType | None,
    session_rating: int | None = None,
    user_id: UUID | None = None,
    user_email: str | None = None,
    user_query: str | None = None,
    api_key_query: str | None = None,
    q: str | None = None,
) -> int:
    conditions = _build_filter_conditions(
        start_time=start_time,
        end_time=end_time,
        feedback_filter=feedback_filter,
        session_rating=session_rating,
        user_id=user_id,
        user_email=user_email,
        user_query=user_query,
        api_key_query=api_key_query,
        q=q,
    )
    stmt = (
        select(func.count(distinct(ChatSession.id)))
        .select_from(ChatSession)
        .filter(*conditions)
    )
    return db_session.scalar(stmt) or 0


def get_page_of_chat_sessions(
    start_time: datetime | None,
    end_time: datetime | None,
    db_session: Session,
    page_num: int,
    page_size: int,
    feedback_filter: QAFeedbackType | None = None,
    session_rating: int | None = None,
    user_id: UUID | None = None,
    user_email: str | None = None,
    user_query: str | None = None,
    api_key_query: str | None = None,
    q: str | None = None,
) -> Sequence[ChatSession]:
    conditions = _build_filter_conditions(
        start_time=start_time,
        end_time=end_time,
        feedback_filter=feedback_filter,
        session_rating=session_rating,
        user_id=user_id,
        user_email=user_email,
        user_query=user_query,
        api_key_query=api_key_query,
        q=q,
    )

    subquery = (
        select(ChatSession.id)
        .filter(*conditions)
        .order_by(desc(ChatSession.time_created), ChatSession.id)
        .limit(page_size)
        .offset(page_num * page_size)
        .subquery()
    )

    stmt = (
        select(ChatSession)
        .join(subquery, ChatSession.id == subquery.c.id)
        .outerjoin(ChatMessage, ChatSession.id == ChatMessage.chat_session_id)
        .options(
            joinedload(ChatSession.user),
            joinedload(ChatSession.persona),
            joinedload(ChatSession.session_feedback),
            contains_eager(ChatSession.messages).joinedload(
                ChatMessage.chat_message_feedbacks
            ),
        )
        .order_by(
            desc(ChatSession.time_created),
            ChatSession.id,
            asc(ChatMessage.id),  # Ensure chronological message order
        )
    )

    return db_session.scalars(stmt).unique().all()


def fetch_chat_sessions_eagerly_by_time(
    start: datetime,
    end: datetime,
    db_session: Session,
    limit: int | None = 500,
    initial_time: datetime | None = None,
) -> list[ChatSession]:
    """Sorted by oldest to newest, then by message id"""

    asc_time_order: UnaryExpression = asc(ChatSession.time_created)
    message_order: UnaryExpression = asc(ChatMessage.id)

    filters: list[ColumnElement | BinaryExpression] = [
        ChatSession.time_created.between(start, end)
    ]

    if initial_time:
        filters.append(ChatSession.time_created > initial_time)

    subquery = (
        db_session.query(ChatSession.id, ChatSession.time_created)
        .filter(*filters)
        .order_by(asc_time_order)
        .limit(limit)
        .subquery()
    )

    query = (
        db_session.query(ChatSession)
        .join(subquery, ChatSession.id == subquery.c.id)
        .outerjoin(ChatMessage, ChatSession.id == ChatMessage.chat_session_id)
        .options(
            joinedload(ChatSession.user),
            joinedload(ChatSession.persona),
            joinedload(ChatSession.session_feedback),
            contains_eager(ChatSession.messages).joinedload(
                ChatMessage.chat_message_feedbacks
            ),
        )
        .order_by(asc_time_order, message_order)
    )

    chat_sessions = query.all()

    return chat_sessions


def get_all_query_history_export_tasks(
    db_session: Session,
) -> list[TaskQueueState]:
    return get_all_tasks_with_prefix(db_session, QUERY_HISTORY_TASK_NAME_PREFIX)
