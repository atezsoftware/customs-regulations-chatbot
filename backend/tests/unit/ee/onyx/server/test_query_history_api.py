from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from ee.onyx.server.query_history.api import snapshot_from_chat_session
from onyx.configs.constants import MessageType


def _message(message_id: int, message_type: MessageType, message: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=message_id,
        message=message,
        message_type=message_type,
        chat_message_feedbacks=[],
        search_docs=[],
        time_sent=datetime.now(timezone.utc),
    )


def test_snapshot_uses_all_persisted_messages_when_latest_child_is_missing() -> None:
    chat_session = SimpleNamespace(
        id=uuid4(), user=None, persona=None, onyxbot_flow=False
    )
    persisted_messages = [
        _message(1, MessageType.SYSTEM, ""),
        _message(2, MessageType.USER, "What duty applies?"),
        _message(3, MessageType.ASSISTANT, "The applicable duty is 10%."),
    ]
    db_session = Mock()

    with patch(
        "ee.onyx.server.query_history.api.get_chat_messages_by_session",
        return_value=persisted_messages,
    ) as get_messages:
        snapshot = snapshot_from_chat_session(chat_session, db_session)

    assert snapshot is not None
    assert [message.message for message in snapshot.messages] == [
        "What duty applies?",
        "The applicable duty is 10%.",
    ]
    get_messages.assert_called_once_with(
        chat_session_id=chat_session.id,
        user_id=None,
        db_session=db_session,
        skip_permission_check=True,
        prefetch_message_details=True,
    )
