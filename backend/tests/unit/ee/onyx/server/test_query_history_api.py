import csv
import io
from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest

from ee.onyx.background.celery.tasks.query_history.tasks import (
    export_query_history_task,
)
from ee.onyx.server.query_history.api import (
    get_chat_session_admin,
    get_chat_session_history,
    snapshot_from_chat_session,
)
from ee.onyx.server.query_history.models import (
    ONYX_ANONYMIZED_EMAIL,
    ChatSessionMinimal,
    ChatSessionSnapshot,
    QuestionAnswerPairSnapshot,
)
from onyx.configs.constants import MessageType, QAFeedbackType, QueryHistoryType
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


def _session_with_identity(
    external_user_name: str | None, *, api_key: bool = True
) -> ChatSession:
    return ChatSession(
        id=uuid4(),
        user=User(
            email="API_KEY__Shared Application@onyxapikey.ai"
            if api_key
            else "admin@example.com"
        ),
        external_user_name=external_user_name,
        api_key_id=17 if api_key else None,
        api_key_name="Shared Application" if api_key else None,
        api_key_display="dn_abcd...wxyz" if api_key else None,
        persona=None,
        description="Application session",
        persona_id=0,
        time_created=datetime.now(timezone.utc),
        onyxbot_flow=False,
        messages=[
            _message(1, MessageType.USER, "First question"),
            _message(2, MessageType.ASSISTANT, "First answer"),
            _message(3, MessageType.USER, "Second question"),
            _message(4, MessageType.ASSISTANT, "Second answer"),
        ],
    )


@pytest.mark.parametrize(
    ("api_key", "external_user_name", "expected_name", "expected_email"),
    [
        (True, "İpek Çağrı Şahin", "İpek Çağrı Şahin", "api_key__shared application"),
        (True, None, None, "api_key__shared application"),
        (False, "İpek Çağrı Şahin", None, "admin@example.com"),
    ],
)
def test_admin_identity_uses_session_name_only_for_api_key_sessions(
    api_key: bool,
    external_user_name: str | None,
    expected_name: str | None,
    expected_email: str,
) -> None:
    chat_session = _session_with_identity(external_user_name, api_key=api_key)
    minimal = ChatSessionMinimal.from_chat_session(chat_session)
    with (
        patch(
            "ee.onyx.server.query_history.api.get_chat_messages_by_session",
            return_value=chat_session.messages,
        ),
        patch("ee.onyx.server.query_history.api.get_response_usage", return_value={}),
    ):
        snapshot = snapshot_from_chat_session(chat_session, Mock())

    assert snapshot is not None
    for result in (minimal, snapshot):
        assert result.user_name == expected_name
        assert result.user_email == expected_email
        assert result.api_key_id == (17 if api_key else None)
        assert result.api_key_name == ("Shared Application" if api_key else None)
        assert result.api_key_display == ("dn_abcd...wxyz" if api_key else None)


def test_session_identity_roundtrips_and_applies_to_every_csv_message_pair() -> None:
    chat_session = _session_with_identity("İpek Çağrı Şahin")
    chat_session.api_key_name = "=Shared Application"
    with (
        patch(
            "ee.onyx.server.query_history.api.get_chat_messages_by_session",
            return_value=chat_session.messages,
        ),
        patch("ee.onyx.server.query_history.api.get_response_usage", return_value={}),
    ):
        snapshot = snapshot_from_chat_session(chat_session, Mock())
    assert snapshot is not None
    roundtripped = ChatSessionSnapshot.model_validate_json(snapshot.model_dump_json())
    rows = [
        sanitize_csv_row(pair.to_json())
        for pair in QuestionAnswerPairSnapshot.from_chat_session_snapshot(roundtripped)
    ]
    assert len(rows) == 2
    for row in rows:
        assert row["user_name"] == "İpek Çağrı Şahin"
        assert row["api_key_id"] == "17"
        assert row["api_key_name"] == "'=Shared Application"
        assert row["api_key_display"] == "dn_abcd...wxyz"
        assert row["user_email"] == "api_key__shared application"
        assert set(row) == set(QuestionAnswerPairSnapshot.model_fields)


@pytest.mark.parametrize(
    "query_history_type", [QueryHistoryType.NORMAL, QueryHistoryType.ANONYMIZED]
)
def test_admin_list_and_detail_apply_identity_anonymization(
    query_history_type: QueryHistoryType,
) -> None:
    chat_session = _session_with_identity("İpek Çağrı Şahin")
    with (
        patch(
            "ee.onyx.server.query_history.api.ensure_query_history_is_enabled",
            return_value=query_history_type,
        ),
        patch(
            "ee.onyx.server.query_history.api.get_page_of_chat_sessions",
            return_value=[chat_session],
        ),
        patch(
            "ee.onyx.server.query_history.api.get_total_filtered_chat_sessions_count",
            return_value=1,
        ),
        patch(
            "ee.onyx.server.query_history.api.get_chat_session_by_id",
            return_value=chat_session,
        ),
        patch(
            "ee.onyx.server.query_history.api.get_chat_messages_by_session",
            return_value=chat_session.messages,
        ),
        patch("ee.onyx.server.query_history.api.get_response_usage", return_value={}),
    ):
        page = get_chat_session_history(
            page_num=0,
            page_size=10,
            feedback_type=None,
            session_rating=None,
            start_time=None,
            end_time=None,
            _=Mock(),
            db_session=Mock(),
        )
        detail = get_chat_session_admin(chat_session.id, _=Mock(), db_session=Mock())

    assert page.total_items == 1
    for result in (page.items[0], detail):
        if query_history_type == QueryHistoryType.ANONYMIZED:
            assert result.user_email == ONYX_ANONYMIZED_EMAIL
            assert result.user_name is None
            assert result.api_key_id is None
            assert result.api_key_name is None
            assert result.api_key_display is None
        else:
            assert result.user_email == "api_key__shared application"
            assert result.user_name == "İpek Çağrı Şahin"
            assert result.api_key_id == 17
            assert result.api_key_name == "Shared Application"
            assert result.api_key_display == "dn_abcd...wxyz"


@pytest.mark.parametrize(
    "query_history_type", [QueryHistoryType.NORMAL, QueryHistoryType.ANONYMIZED]
)
def test_csv_worker_preserves_or_anonymizes_all_identity_fields(
    query_history_type: QueryHistoryType,
) -> None:
    chat_session = _session_with_identity("=İpek Çağrı Şahin")
    with (
        patch(
            "ee.onyx.server.query_history.api.get_chat_messages_by_session",
            return_value=chat_session.messages,
        ),
        patch("ee.onyx.server.query_history.api.get_response_usage", return_value={}),
    ):
        snapshot = snapshot_from_chat_session(chat_session, Mock())
    assert snapshot is not None

    file_store = Mock()
    timestamp = datetime.now(timezone.utc)
    task_module = "ee.onyx.background.celery.tasks.query_history.tasks"
    with (
        patch(
            f"{task_module}.get_session_with_current_tenant", return_value=MagicMock()
        ),
        patch(f"{task_module}.mark_task_as_started_with_id"),
        patch(f"{task_module}.delete_task_with_id"),
        patch(f"{task_module}.get_default_file_store", return_value=file_store),
        patch(
            f"{task_module}.fetch_and_process_chat_session_history",
            return_value=[snapshot],
        ),
        patch(
            f"{task_module}.load_settings",
            return_value=Mock(query_history_type=query_history_type),
        ),
        patch.object(export_query_history_task.request, "id", str(uuid4())),
    ):
        export_query_history_task.run(
            start=timestamp,
            end=timestamp,
            start_time=timestamp,
            tenant_id="public",
        )

    file_store.save_file.assert_called_once()
    stream = file_store.save_file.call_args.kwargs["content"]
    rows = list(csv.DictReader(io.StringIO(stream.getvalue())))
    assert len(rows) == 2
    for row in rows:
        if query_history_type == QueryHistoryType.ANONYMIZED:
            assert row["user_email"] == ONYX_ANONYMIZED_EMAIL
            assert row["user_name"] == ""
            assert row["api_key_id"] == ""
            assert row["api_key_name"] == ""
            assert row["api_key_display"] == ""
        else:
            assert row["user_email"] == "api_key__shared application"
            assert row["user_name"] == "'=İpek Çağrı Şahin"
            assert row["api_key_id"] == "17"
            assert row["api_key_name"] == "Shared Application"
            assert row["api_key_display"] == "dn_abcd...wxyz"
