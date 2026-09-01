from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest
from sqlalchemy.orm import Session

from ee.onyx.db.usage_export import (
    get_all_empty_chat_message_entries,
    get_empty_chat_messages_entries__paginated,
    get_usage_summary,
)
from onyx.configs.constants import MessageType
from onyx.db.chat import (
    create_chat_session,
    create_new_chat_message,
    get_or_create_root_message,
)
from onyx.db.models import ChatMessage, UserUsage
from onyx.db.user_usage import get_user_activity_counts_by_email
from tests.external_dependency_unit.conftest import create_test_user


def _full_period() -> tuple[datetime, datetime]:
    return (
        datetime.fromtimestamp(0, tz=timezone.utc),
        datetime.now(tz=timezone.utc),
    )


def _make_user_message(
    db_session: Session, chat_session_id: UUID, parent: ChatMessage
) -> ChatMessage:
    return create_new_chat_message(
        chat_session_id=chat_session_id,
        parent_message=parent,
        message="user prompt",
        token_count=0,
        message_type=MessageType.USER,
        db_session=db_session,
    )


def _make_assistant_message(
    db_session: Session,
    chat_session_id: UUID,
    parent: ChatMessage,
    model_display_name: str,
) -> ChatMessage:
    msg = create_new_chat_message(
        chat_session_id=chat_session_id,
        parent_message=parent,
        message="assistant reply",
        token_count=0,
        message_type=MessageType.ASSISTANT,
        db_session=db_session,
    )
    msg.model_display_name = model_display_name
    db_session.commit()
    return msg


def test_multi_model_branch_emits_row_per_assistant_child(
    db_session: Session,
) -> None:
    """A user message answered by multiple models (multi-model branch) must
    produce one report row per assistant child so no model invocation is
    dropped — even non-preferred branches."""
    user = create_test_user(db_session, "usage-export-branch")
    chat_session = create_chat_session(
        db_session=db_session,
        description="multi-model branch",
        user_id=user.id,
        persona_id=None,
    )
    root = get_or_create_root_message(chat_session.id, db_session)

    user_msg = _make_user_message(db_session, chat_session.id, root)
    _make_assistant_message(db_session, chat_session.id, user_msg, "model-a")
    assistant_b = _make_assistant_message(
        db_session, chat_session.id, user_msg, "model-b"
    )

    # Even when one branch is marked preferred, both must still be reported.
    user_msg.preferred_response_id = assistant_b.id
    db_session.commit()

    _, skeletons = get_empty_chat_messages_entries__paginated(
        db_session, _full_period()
    )

    matching = [s for s in skeletons if s.message_id == user_msg.id]
    assert {s.llm_model for s in matching} == {"model-a", "model-b"}
    assert len(matching) == 2


def test_single_assistant_child_emits_single_row(db_session: Session) -> None:
    """The common case (one assistant reply per user message) still produces
    exactly one row with that model. Guards against the per-pair change
    inflating row counts in non-branched conversations."""
    user = create_test_user(db_session, "usage-export-single")
    chat_session = create_chat_session(
        db_session=db_session,
        description="single reply",
        user_id=user.id,
        persona_id=None,
    )
    root = get_or_create_root_message(chat_session.id, db_session)

    user_msg = _make_user_message(db_session, chat_session.id, root)
    _make_assistant_message(db_session, chat_session.id, user_msg, "only-model")

    _, skeletons = get_empty_chat_messages_entries__paginated(
        db_session, _full_period()
    )

    matching = [s for s in skeletons if s.message_id == user_msg.id]
    assert len(matching) == 1
    assert matching[0].llm_model == "only-model"


def test_orphan_user_message_emits_row_with_null_model(db_session: Session) -> None:
    """User message with no assistant reply (still streaming, errored) gets a
    single row with `llm_model=None` rather than being dropped."""
    user = create_test_user(db_session, "usage-export-orphan")
    chat_session = create_chat_session(
        db_session=db_session,
        description="orphan user message",
        user_id=user.id,
        persona_id=None,
    )
    root = get_or_create_root_message(chat_session.id, db_session)

    user_msg = _make_user_message(db_session, chat_session.id, root)

    _, skeletons = get_empty_chat_messages_entries__paginated(
        db_session, _full_period()
    )

    matching = [s for s in skeletons if s.message_id == user_msg.id]
    assert len(matching) == 1
    assert matching[0].llm_model is None


def test_usage_summary_uses_ledger_tokens_and_cost_for_query_session_rates(
    db_session: Session,
) -> None:
    user = create_test_user(db_session, "usage-summary")
    summary_time = datetime(2200, 1, 1, tzinfo=timezone.utc) + timedelta(
        seconds=user.id.int % 1_000_000_000
    )
    first_session = create_chat_session(
        db_session=db_session,
        description="first session",
        user_id=user.id,
        persona_id=None,
    )
    second_session = create_chat_session(
        db_session=db_session,
        description="second session",
        user_id=user.id,
        persona_id=None,
    )
    first_session.time_created = summary_time - timedelta(days=10)
    second_session.time_created = summary_time + timedelta(days=10)
    first_root = get_or_create_root_message(first_session.id, db_session)
    second_root = get_or_create_root_message(second_session.id, db_session)

    first_query = _make_user_message(db_session, first_session.id, first_root)
    second_query = _make_user_message(db_session, first_session.id, first_query)
    third_query = _make_user_message(db_session, second_session.id, second_root)
    first_query.token_count = 10
    second_query.token_count = 20
    third_query.token_count = 60
    first_query.time_sent = summary_time
    second_query.time_sent = summary_time
    third_query.time_sent = summary_time
    db_session.add_all(
        [
            UserUsage(
                user_id=user.id,
                window_start=summary_time,
                model="model-a",
                flow="chat_response",
                provider="provider-a",
                input_tokens=1_000,
                output_tokens=500,
                cache_read_tokens=200,
                cost_cents=25.0,
            ),
            UserUsage(
                user_id=user.id,
                window_start=summary_time,
                model="model-a",
                flow="query_rephrase",
                provider="provider-a",
                input_tokens=2_000,
                output_tokens=1_000,
                cache_read_tokens=300,
                cost_cents=75.0,
            ),
        ]
    )
    db_session.commit()

    summary = get_usage_summary(
        db_session,
        (
            summary_time - timedelta(seconds=1),
            summary_time + timedelta(seconds=1),
        ),
    )

    assert summary.total_user_queries == 3
    assert summary.total_user_sessions == 2
    assert summary.total_tokens == 4_500
    assert summary.total_cost_cents == 100.0
    assert summary.average_tokens_per_query == 1_500.0
    assert summary.average_tokens_per_session == 2_250.0
    assert summary.average_cost_cents_per_query == pytest.approx(100.0 / 3)
    assert summary.average_cost_cents_per_session == 50.0
    assert summary.average_queries_per_session == 1.5


def test_user_activity_counts_group_queries_and_sessions_by_email(
    db_session: Session,
) -> None:
    first_user = create_test_user(db_session, "usage-activity-first")
    second_user = create_test_user(db_session, "usage-activity-second")
    activity_time = datetime(2300, 1, 1, tzinfo=timezone.utc) + timedelta(
        seconds=first_user.id.int % 1_000_000_000
    )

    first_session = create_chat_session(
        db_session=db_session,
        description="first activity session",
        user_id=first_user.id,
        persona_id=None,
    )
    second_session = create_chat_session(
        db_session=db_session,
        description="second activity session",
        user_id=first_user.id,
        persona_id=None,
    )
    benchmark_session = create_chat_session(
        db_session=db_session,
        description="benchmark activity session",
        user_id=second_user.id,
        persona_id=None,
    )
    first_session.time_created = activity_time - timedelta(days=10)
    second_session.time_created = activity_time + timedelta(days=10)
    benchmark_session.time_created = activity_time
    benchmark_session.benchmark_flow = True

    first_root = get_or_create_root_message(first_session.id, db_session)
    second_root = get_or_create_root_message(second_session.id, db_session)
    benchmark_root = get_or_create_root_message(benchmark_session.id, db_session)
    first_query = _make_user_message(db_session, first_session.id, first_root)
    second_query = _make_user_message(db_session, first_session.id, first_query)
    third_query = _make_user_message(db_session, second_session.id, second_root)
    benchmark_query = _make_user_message(
        db_session, benchmark_session.id, benchmark_root
    )
    first_query.time_sent = activity_time
    second_query.time_sent = activity_time
    third_query.time_sent = activity_time
    benchmark_query.time_sent = activity_time
    db_session.commit()

    activity = get_user_activity_counts_by_email(
        db_session,
        start=activity_time - timedelta(seconds=1),
        end=activity_time + timedelta(seconds=1),
    )

    assert [row.model_dump() for row in activity] == [
        {
            "email": first_user.email,
            "query_count": 3,
            "session_count": 2,
        }
    ]


def test_usage_export_includes_session_at_period_start(db_session: Session) -> None:
    user = create_test_user(db_session, "usage-period-start")
    period_start = datetime(2250, 1, 1, tzinfo=timezone.utc) + timedelta(
        seconds=user.id.int % 1_000_000_000
    )
    chat_session = create_chat_session(
        db_session=db_session,
        description="period boundary",
        user_id=user.id,
        persona_id=None,
    )
    chat_session.time_created = period_start
    root = get_or_create_root_message(chat_session.id, db_session)
    user_message = _make_user_message(db_session, chat_session.id, root)
    db_session.commit()

    batches = list(
        get_all_empty_chat_message_entries(
            db_session,
            (period_start, period_start + timedelta(seconds=1)),
        )
    )

    assert [row.message_id for batch in batches for row in batch] == [user_message.id]
