from unittest.mock import MagicMock
from uuid import uuid4

from sqlalchemy.orm import Session

from onyx.db.chat_search import search_chat_sessions


def test_chat_search_excludes_benchmark_sessions() -> None:
    db_session = MagicMock(spec=Session)
    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    db_session.execute.return_value = result

    search_chat_sessions(
        user_id=uuid4(),
        db_session=db_session,
        query=None,
    )

    statement = str(db_session.execute.call_args.args[0])
    assert "chat_session.benchmark_flow IS false" in statement
