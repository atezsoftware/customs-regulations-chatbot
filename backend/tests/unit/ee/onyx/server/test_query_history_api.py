from datetime import datetime, timezone
from unittest.mock import Mock, patch
from uuid import uuid4

from ee.onyx.server.query_history.api import snapshot_from_chat_session
from ee.onyx.server.query_history.models import (
    ChatSessionMinimal,
    QuestionAnswerPairSnapshot,
)
from onyx.configs.constants import MessageType, QAFeedbackType
from onyx.db.models import (
    ChatMessage,
    ChatMessageFeedback,
    ChatSession,
    ChatSessionFeedback,
    Persona,
    User,
)
from onyx.utils.csv_utils import sanitize_csv_row


def _message(
    message_id: int,
    message_type: MessageType,
    message: str,
    model_display_name: str | None = None,
) -> ChatMessage:
    return ChatMessage(
        id=message_id,
        message=message,
        message_type=message_type,
        model_display_name=model_display_name,
        chat_message_feedbacks=[],
        search_docs=[],
        time_sent=datetime.now(timezone.utc),
    )


def test_snapshot_uses_all_persisted_messages_when_latest_child_is_missing() -> None:
    chat_session = ChatSession(
        id=uuid4(),
        user=None,
        persona=None,
        onyxbot_flow=False,
        description="History",
        persona_id=0,
        time_created=datetime.now(timezone.utc),
    )
    persisted_messages = [
        _message(1, MessageType.SYSTEM, ""),
        _message(2, MessageType.USER, "What duty applies?"),
        _message(
            3,
            MessageType.ASSISTANT,
            "The applicable duty is 10%.",
            model_display_name="GPT-5",
        ),
    ]
    db_session = Mock()

    with (
        patch(
            "ee.onyx.server.query_history.api.get_chat_messages_by_session",
            return_value=persisted_messages,
        ) as get_messages,
        patch(
            "ee.onyx.server.query_history.api.get_response_usage", return_value={}
        ) as get_usage,
    ):
        snapshot = snapshot_from_chat_session(chat_session, db_session)

    assert snapshot is not None
    assert [message.message for message in snapshot.messages] == [
        "What duty applies?",
        "The applicable duty is 10%.",
    ]
    assert snapshot.messages[1].model_display_name == "GPT-5"
    get_usage.assert_called_once_with(db_session, [1, 2, 3], admin=True)
    get_messages.assert_called_once_with(
        chat_session_id=chat_session.id,
        user_id=None,
        db_session=db_session,
        skip_permission_check=True,
        prefetch_message_details=True,
    )


def test_minimal_snapshot_includes_each_model_used_in_the_conversation() -> None:
    chat_session = ChatSession(
        id=uuid4(),
        user=User(email="admin@example.com"),
        description="Duty question",
        persona_id=1,
        persona=Persona(name="Customs Agent"),
        time_created=datetime.now(timezone.utc),
        onyxbot_flow=False,
        messages=[
            _message(1, MessageType.USER, "What duty applies?"),
            _message(
                2,
                MessageType.ASSISTANT,
                "The applicable duty is 10%.",
                model_display_name="GPT-5",
            ),
            _message(
                3,
                MessageType.ASSISTANT,
                "Here is an alternative answer.",
                model_display_name="Claude Sonnet 4",
            ),
            _message(
                4,
                MessageType.ASSISTANT,
                "A repeated model should not duplicate the table value.",
                model_display_name="GPT-5",
            ),
        ],
    )

    snapshot = ChatSessionMinimal.from_chat_session(chat_session)

    assert snapshot.model_display_names == ["GPT-5", "Claude Sonnet 4"]


def test_session_feedback_in_admin_snapshots_and_sanitized_csv_rows() -> None:
    submitted_by = uuid4()
    timestamp = datetime.now(timezone.utc)
    messages = [
        _message(1, MessageType.USER, "First question"),
        _message(2, MessageType.ASSISTANT, "First answer"),
        _message(3, MessageType.USER, "Second question"),
        _message(4, MessageType.ASSISTANT, "Second answer"),
    ]
    messages[1].chat_message_feedbacks = [
        ChatMessageFeedback(is_positive=True, feedback_text="Message feedback")
    ]
    chat_session = ChatSession(
        id=uuid4(),
        user=None,
        persona=None,
        description="Session feedback",
        persona_id=0,
        time_created=timestamp,
        onyxbot_flow=False,
        messages=messages,
    )
    chat_session.session_feedback = ChatSessionFeedback(
        chat_session_id=chat_session.id,
        rating=2,
        feedback_text="=SUM(A1)",
        submitted_by_user_id=submitted_by,
        evaluated_through_message_id=4,
        version=1,
        time_created=timestamp,
        time_updated=timestamp,
    )
    minimal = ChatSessionMinimal.from_chat_session(chat_session)
    assert minimal.feedback_type == QAFeedbackType.LIKE
    assert minimal.session_feedback is not None
    assert minimal.session_feedback.label == "Eksik"
    assert "submitted_by_user_id" not in minimal.session_feedback.model_dump()

    with (
        patch(
            "ee.onyx.server.query_history.api.get_chat_messages_by_session",
            return_value=messages,
        ),
        patch("ee.onyx.server.query_history.api.get_response_usage", return_value={}),
    ):
        snapshot = snapshot_from_chat_session(chat_session, Mock())
    assert snapshot is not None
    assert snapshot.session_feedback == minimal.session_feedback
    rows = [
        sanitize_csv_row(pair.to_json())
        for pair in QuestionAnswerPairSnapshot.from_chat_session_snapshot(snapshot)
    ]
    assert len(rows) == 2
    assert rows[0]["feedback_type"] == "like"
    assert rows[0]["feedback_text"] == "Message feedback"
    assert rows[1]["feedback_type"] == ""
    for row in rows:
        assert row["session_feedback_rating"] == "2"
        assert row["session_feedback_label"] == "Eksik"
        assert row["session_feedback_text"] == "'=SUM(A1)"
        assert row["session_feedback_updated_at"] == timestamp.isoformat()
        assert set(row) == set(QuestionAnswerPairSnapshot.model_fields)
