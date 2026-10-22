from collections.abc import Generator
from datetime import datetime, timezone
from typing import NotRequired, TypedDict
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from ee.onyx.db.query_history import (
    get_page_of_chat_sessions,
    get_total_filtered_chat_sessions_count,
)
from onyx.configs.constants import MessageType, QAFeedbackType
from onyx.db.models import (
    ChatMessage,
    ChatMessageFeedback,
    ChatSession,
    ChatSessionFeedback,
    User,
    UserRole,
)


class HistoryFilterArguments(TypedDict):
    db_session: Session
    start_time: datetime | None
    end_time: datetime | None
    feedback_filter: QAFeedbackType | None
    user_id: NotRequired[UUID]


class RatedHistoryFilterArguments(HistoryFilterArguments):
    session_rating: int
    api_key_query: str


@pytest.fixture
def history_rows(
    db_session: Session,
) -> Generator[tuple[User, User, list[ChatSession]], None, None]:
    tag = uuid4().hex
    users = [
        User(
            id=uuid4(),
            email=f"history_{tag}{suffix}@example.com",
            hashed_password="test",
            role=UserRole.BASIC,
        )
        for suffix in ("", "_other")
    ]
    db_session.add_all(users)
    db_session.flush()
    created = datetime(2026, 10, 20, tzinfo=timezone.utc)
    sessions = [
        ChatSession(
            user=users[0],
            description="Duty exemption",
            external_user_name="Çağrı Şahin %_",
            api_key_id=17,
            api_key_name="Customer Portal %_",
            api_key_display="dn_****1234",
            time_created=created,
        ),
        ChatSession(
            user=users[0],
            description="Ordinary question",
            external_user_name="Ayşe",
            api_key_id=18,
            api_key_name="Other Portal",
            time_created=created,
        ),
        ChatSession(
            user=users[1],
            description="Another duty session",
            external_user_name="Çağrı Şahin %_",
            api_key_id=17,
            api_key_name="Customer Portal %_",
            time_created=created,
        ),
    ]
    for chat, questions in zip(
        sessions,
        [
            ["duty 50%_ exemption", "duty 50%_ follow-up"],
            ["duty 50XY exemption"],
            ["duty 50%_ exemption"],
        ],
    ):
        chat.messages = [
            ChatMessage(message=question, message_type=MessageType.USER, token_count=1)
            for question in questions
        ]
        chat.messages.append(
            ChatMessage(
                message="assistant-only-token",
                message_type=MessageType.ASSISTANT,
                token_count=1,
            )
        )
    sessions[0].session_feedback = ChatSessionFeedback(
        rating=4, submitted_by_user_id=users[0].id
    )
    sessions[0].messages[0].chat_message_feedbacks = [
        ChatMessageFeedback(is_positive=True)
    ]
    db_session.add_all(sessions)
    db_session.flush()
    try:
        yield users[0], users[1], sessions
    finally:
        db_session.rollback()


def test_query_history_filters_page_and_count_with_literal_search(
    db_session: Session, history_rows: tuple[User, User, list[ChatSession]]
) -> None:
    user, other_user, sessions = history_rows
    filters: HistoryFilterArguments = dict(
        db_session=db_session, start_time=None, end_time=None, feedback_filter=None
    )
    page = get_page_of_chat_sessions(
        **filters,
        page_num=0,
        page_size=20,
        user_query="şahin %_",
        api_key_query="portal %_",
        q="50%_",
    )
    count = get_total_filtered_chat_sessions_count(
        **filters,
        user_query="şahin %_",
        api_key_query="portal %_",
        q="50%_",
    )
    assert {chat.id for chat in page} == {sessions[0].id, sessions[2].id}
    assert count == 2
    assert len(page[0].messages) >= 2
    # The other email contains the first email's local-part; this filter is exact.
    exact = get_page_of_chat_sessions(
        **filters,
        page_num=0,
        page_size=20,
        user_email=user.email.upper(),
    )
    assert {chat.id for chat in exact} == {sessions[0].id, sessions[1].id}
    assert all(chat.user_id != other_user.id for chat in exact)


def test_query_history_account_pagination_and_combined_feedback_filters(
    db_session: Session, history_rows: tuple[User, User, list[ChatSession]]
) -> None:
    user, _, sessions = history_rows
    filters: HistoryFilterArguments = dict(
        db_session=db_session,
        start_time=None,
        end_time=None,
        feedback_filter=None,
        user_id=user.id,
    )
    count = get_total_filtered_chat_sessions_count(**filters)
    pages = [
        get_page_of_chat_sessions(page_num=page, page_size=1, **filters)
        for page in range(2)
    ]
    assert count == 2
    assert {page[0].id for page in pages} == {sessions[0].id, sessions[1].id}
    assert len({page[0].id for page in pages}) == 2
    combined: RatedHistoryFilterArguments = dict(
        db_session=db_session,
        start_time=None,
        end_time=None,
        user_id=user.id,
        feedback_filter=QAFeedbackType.LIKE,
        session_rating=4,
        api_key_query="17",
    )
    assert get_total_filtered_chat_sessions_count(**combined) == 1
    assert (
        get_page_of_chat_sessions(page_num=0, page_size=10, **combined)[0].id
        == sessions[0].id
    )


def test_query_history_searches_all_user_turns_and_session_identity(
    db_session: Session, history_rows: tuple[User, User, list[ChatSession]]
) -> None:
    user, _, sessions = history_rows
    filters: HistoryFilterArguments = dict(
        db_session=db_session,
        start_time=None,
        end_time=None,
        feedback_filter=None,
        user_id=user.id,
    )
    for query in ("follow-up", "Duty exemption", str(sessions[0].id)):
        assert get_total_filtered_chat_sessions_count(**filters, q=query) == 1
        page = get_page_of_chat_sessions(**filters, page_num=0, page_size=10, q=query)
        assert [chat.id for chat in page] == [sessions[0].id]
    assert (
        get_total_filtered_chat_sessions_count(**filters, q="assistant-only-token") == 0
    )
    assert (
        get_total_filtered_chat_sessions_count(**filters, user_query="no-such-user")
        == 0
    )
