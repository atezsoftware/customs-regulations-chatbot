"""Exercise history pagination against PostgreSQL without committing fixture data."""

from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
from sqlalchemy import event
from sqlalchemy.orm import Session

from onyx.configs.constants import MessageType
from onyx.db.chat import get_chat_sessions_by_user
from onyx.db.models import ChatMessage, ChatSession, User, UserProject, UserRole

pytestmark = pytest.mark.usefixtures("tenant_context")


@pytest.fixture
def history_owners(
    db_session: Session,
) -> Generator[tuple[UUID, UUID, int], None, None]:
    owner_ids = (uuid4(), uuid4())
    for owner_id in owner_ids:
        db_session.add(
            User(
                id=owner_id,
                email=f"history-pagination-{owner_id}@example.com",
                hashed_password="unused-test-password",
                is_active=True,
                is_superuser=False,
                is_verified=True,
                role=UserRole.BASIC,
            )
        )
    db_session.flush()
    project = UserProject(
        user_id=owner_ids[0], name="History pagination", instructions=""
    )
    db_session.add(project)
    db_session.flush()
    try:
        yield owner_ids[0], owner_ids[1], project.id
    finally:
        db_session.rollback()


def _add_session(
    db_session: Session,
    user_id: UUID,
    updated: datetime,
    *,
    recent: bool = False,
    message_type: MessageType | None = MessageType.USER,
    deleted: bool = False,
    onyxbot_flow: bool = False,
    benchmark_flow: bool = False,
    project_id: int | None = None,
    description: str = "History pagination fixture",
) -> UUID:
    chat = ChatSession(
        id=uuid4(),
        user_id=user_id,
        description=description,
        time_updated=updated,
        time_created=(
            datetime.now(timezone.utc) - timedelta(minutes=1)
            if recent
            else updated - timedelta(hours=1)
        ),
        deleted=deleted,
        onyxbot_flow=onyxbot_flow,
        benchmark_flow=benchmark_flow,
        project_id=project_id,
    )
    db_session.add(chat)
    db_session.flush()
    if message_type is not None:
        db_session.add(
            ChatMessage(
                chat_session_id=chat.id,
                message_type=message_type,
                message="Fixture message",
                token_count=2,
            )
        )
        db_session.flush()
    return chat.id


def test_page_is_filled_after_filtering_and_only_loads_requested_sessions(
    db_session: Session, history_owners: tuple[UUID, UUID, int]
) -> None:
    owner_id, _, _ = history_owners
    now = datetime.now(timezone.utc)
    # Failed chats sort ahead of the valid history and must not consume the page.
    for offset in range(12):
        _add_session(
            db_session,
            owner_id,
            now - timedelta(minutes=offset),
            message_type=MessageType.SYSTEM if offset % 2 else None,
        )
    valid_ids = [
        _add_session(db_session, owner_id, now - timedelta(hours=offset))
        for offset in range(1, 8)
    ]
    db_session.expunge_all()
    loaded_ids: list[UUID] = []

    def record_loaded_session(_: Session, instance: object) -> None:
        if isinstance(instance, ChatSession):
            loaded_ids.append(instance.id)

    event.listen(db_session, "loaded_as_persistent", record_loaded_session)
    try:
        page = get_chat_sessions_by_user(
            user_id=owner_id, deleted=False, db_session=db_session, limit=3
        )
    finally:
        event.remove(db_session, "loaded_as_persistent", record_loaded_session)

    assert [chat.id for chat in page] == valid_ids[:3]
    assert loaded_ids == valid_ids[:3]
    next_page = get_chat_sessions_by_user(
        user_id=owner_id,
        deleted=False,
        db_session=db_session,
        limit=3,
        before=page[-1].time_updated,
    )
    assert [chat.id for chat in next_page] == valid_ids[3:6]


def test_recent_empty_sessions_and_failed_chat_opt_in(
    db_session: Session, history_owners: tuple[UUID, UUID, int]
) -> None:
    owner_id, _, _ = history_owners
    now = datetime.now(timezone.utc)
    recent_id = _add_session(db_session, owner_id, now, recent=True, message_type=None)
    failed_id = _add_session(
        db_session,
        owner_id,
        now - timedelta(minutes=10),
        message_type=MessageType.SYSTEM,
    )
    valid_id = _add_session(db_session, owner_id, now - timedelta(hours=1))

    filtered = get_chat_sessions_by_user(
        user_id=owner_id, deleted=False, db_session=db_session, limit=0
    )
    assert [chat.id for chat in filtered] == [recent_id, valid_id]
    inclusive = get_chat_sessions_by_user(
        user_id=owner_id,
        deleted=False,
        db_session=db_session,
        limit=2,
        include_failed_chats=True,
    )
    assert [chat.id for chat in inclusive] == [recent_id, failed_id]


def test_history_filters_preserve_owner_and_project_boundaries(
    db_session: Session, history_owners: tuple[UUID, UUID, int]
) -> None:
    owner_id, other_owner_id, project_id = history_owners
    now = datetime.now(timezone.utc)
    _add_session(db_session, other_owner_id, now)
    _add_session(db_session, owner_id, now, deleted=True)
    _add_session(db_session, owner_id, now, onyxbot_flow=True)
    _add_session(db_session, owner_id, now, benchmark_flow=True)
    project_chat_id = _add_session(db_session, owner_id, now, project_id=project_id)
    plain_chat_id = _add_session(db_session, owner_id, now - timedelta(hours=1))

    plain = get_chat_sessions_by_user(
        user_id=owner_id,
        deleted=False,
        db_session=db_session,
        only_non_project_chats=True,
    )
    assert [chat.id for chat in plain] == [plain_chat_id]
    project = get_chat_sessions_by_user(
        user_id=owner_id,
        deleted=False,
        db_session=db_session,
        project_id=project_id,
    )
    assert [chat.id for chat in project] == [project_chat_id]


def test_name_prefix_is_filtered_before_limit_and_matches_literal_characters(
    db_session: Session, history_owners: tuple[UUID, UUID, int]
) -> None:
    owner_id, other_owner_id, _ = history_owners
    now = datetime.now(timezone.utc)
    prefix = "tariff-ai:account%_name:"
    _add_session(db_session, other_owner_id, now, description=f"{prefix}other user")
    for offset, description in enumerate(
        ["Unrelated session", "tariff-ai:accountXXname:wrong", "tariff-ai:another:"]
    ):
        _add_session(
            db_session,
            owner_id,
            now - timedelta(minutes=offset),
            description=description,
        )
    matching_ids = [
        _add_session(
            db_session,
            owner_id,
            now - timedelta(hours=offset),
            description=f"{prefix}session {offset}",
        )
        for offset in range(1, 4)
    ]
    page = get_chat_sessions_by_user(
        user_id=owner_id,
        deleted=False,
        db_session=db_session,
        limit=2,
        name_prefix=prefix,
    )
    assert [chat.id for chat in page] == matching_ids[:2]
