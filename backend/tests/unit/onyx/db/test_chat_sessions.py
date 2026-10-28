"""Guard SQL pagination boundaries; PostgreSQL tests cover returned history."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session
from sqlalchemy.sql import Select

from onyx.configs.constants import MessageType
from onyx.db.chat import get_chat_sessions_by_user
from onyx.db.models import ChatSession


@pytest.fixture
def db_session() -> MagicMock:
    session = MagicMock(spec=Session)
    session.execute.return_value.scalars.return_value.all.return_value = []
    return session


@pytest.mark.parametrize("include_failed_chats", [False, True])
def test_page_limit_is_applied_in_sql(
    db_session: MagicMock, include_failed_chats: bool
) -> None:
    chat = ChatSession(id=uuid4())
    db_session.execute.return_value.scalars.return_value.all.return_value = [chat]
    before_query = datetime.now(timezone.utc) - timedelta(minutes=5)

    result = get_chat_sessions_by_user(
        user_id=uuid4(),
        deleted=False,
        db_session=db_session,
        limit=3,
        include_failed_chats=include_failed_chats,
    )

    statement: Select[tuple[ChatSession]] = db_session.execute.call_args.args[0]
    compiled = statement.compile(dialect=postgresql.dialect())
    sql = str(compiled)
    assert "LIMIT" in sql
    assert 3 in compiled.params.values()
    assert result == [chat]
    db_session.execute.assert_called_once()

    if include_failed_chats:
        assert "chat_message" not in sql
        assert sql.count("LIMIT") == 1
    else:
        assert "chat_session.time_created >=" in sql
        assert "SELECT chat_message.id" in sql
        assert "IS NOT NULL" in sql
        assert sql.count("LIMIT") == 2
        assert "chat_message.chat_session_id = chat_session.id" in sql
        assert compiled.params["message_type_1"] == MessageType.SYSTEM
        assert (
            before_query
            <= compiled.params["time_created_1"]
            <= (datetime.now(timezone.utc) - timedelta(minutes=5))
        )


def test_zero_limit_preserves_unbounded_admin_history(db_session: MagicMock) -> None:
    get_chat_sessions_by_user(
        user_id=uuid4(), deleted=False, db_session=db_session, limit=0
    )

    statement: Select[tuple[ChatSession]] = db_session.execute.call_args.args[0]
    sql = str(statement.compile(dialect=postgresql.dialect()))
    assert sql.count("LIMIT") == 1
    assert "IS NOT NULL" in sql
    db_session.execute.assert_called_once()


@pytest.mark.parametrize(
    ("project_id", "only_non_project_chats", "project_filter"),
    [
        (None, False, None),
        (None, True, "chat_session.project_id IS NULL"),
        (7, True, "chat_session.project_id ="),
    ],
)
def test_owner_and_scope_filters_remain_in_paginated_query(
    db_session: MagicMock,
    project_id: int | None,
    only_non_project_chats: bool,
    project_filter: str | None,
) -> None:
    user_id = uuid4()
    before = datetime.now(timezone.utc) - timedelta(days=1)
    get_chat_sessions_by_user(
        user_id=user_id,
        deleted=False,
        db_session=db_session,
        before=before,
        project_id=project_id,
        only_non_project_chats=only_non_project_chats,
    )

    statement: Select[tuple[ChatSession]] = db_session.execute.call_args.args[0]
    compiled = statement.compile(dialect=postgresql.dialect())
    sql = str(compiled)
    assert "chat_session.user_id =" in sql
    assert compiled.params["user_id_1"] == user_id
    assert "chat_session.deleted = false" in sql
    assert "chat_session.onyxbot_flow IS false" in sql
    assert "chat_session.benchmark_flow IS false" in sql
    assert "chat_session.time_updated <" in sql
    assert compiled.params["time_updated_1"] == before
    assert "ORDER BY chat_session.time_updated DESC" in sql
    if project_filter is not None:
        assert project_filter in sql
    else:
        assert "chat_session.project_id" not in sql.split("WHERE", 1)[1]
    if project_id is not None:
        assert compiled.params["project_id_1"] == project_id


def test_name_prefix_is_a_bound_literal_filter(db_session: MagicMock) -> None:
    prefix = "tariff-ai:customer%_/"
    get_chat_sessions_by_user(
        user_id=uuid4(),
        deleted=False,
        db_session=db_session,
        name_prefix=prefix,
    )

    statement: Select[tuple[ChatSession]] = db_session.execute.call_args.args[0]
    compiled = statement.compile(dialect=postgresql.dialect())
    sql = str(compiled)
    assert prefix not in sql
    assert "chat_session.description LIKE" in sql
    assert "ESCAPE '/'" in sql
    assert compiled.params["description_1"] == "tariff-ai:customer/%/_//"
    db_session.execute.assert_called_once()
