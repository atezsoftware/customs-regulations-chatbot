from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete, event
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session

from onyx.configs.constants import MessageType
from onyx.db.models import (
    AnswerGraphNode,
    AnswerGraphRun,
    ChatMessage,
    ChatSession,
    User,
    UserRole,
    UserUsage,
)
from onyx.db.workflow_usage import get_workflow_usage_by_email
from onyx.llm.usage_cost import GenerationCost, TokenCostLine
from onyx.server.features.usage.api import export_usage
from shared_configs.contextvars import get_current_tenant_id

_START = datetime(2400, 1, 1, tzinfo=timezone.utc)
_END = _START + timedelta(days=1)


def _session(db_session: Session, user: User | None = None) -> tuple[User, ChatSession]:
    if user is None:
        user = User(
            id=uuid4(),
            email=f"workflow-{uuid4().hex}@example.com",
            hashed_password="unused",
            role=UserRole.BASIC,
        )
        db_session.add(user)
        db_session.flush()
    chat_session = ChatSession(user_id=user.id)
    db_session.add(chat_session)
    db_session.flush()
    return user, chat_session


def _message(
    db_session: Session,
    chat_session: ChatSession,
    parent: ChatMessage | None = None,
    workflow: str | None = None,
    sent: datetime = _START,
) -> ChatMessage:
    message = ChatMessage(
        chat_session_id=chat_session.id,
        parent_message_id=parent.id if parent else None,
        message_type=MessageType.ASSISTANT if parent else MessageType.USER,
        workflow=workflow,
        message="test message",
        token_count=0,
        time_sent=sent,
    )
    db_session.add(message)
    db_session.flush()
    return message


def _run(
    db_session: Session,
    user_message: ChatMessage,
    assistant: ChatMessage,
    *,
    status: str = "COMPLETE",
    capture: str = "COMPLETE",
    tenant_id: str | None = None,
) -> AnswerGraphRun:
    run = AnswerGraphRun(
        tenant_id=tenant_id or get_current_tenant_id(),
        assistant_message_id=assistant.id,
        user_message_id=user_message.id,
        chat_session_id=assistant.chat_session_id,
        trace_id=f"trace-{uuid4().hex}",
        status=status,
        capture_status=capture,
        time_created=_START,
        time_finished=_START + timedelta(seconds=1),
    )
    db_session.add(run)
    db_session.flush()
    return run


def _generation(
    db_session: Session,
    run: AnswerGraphRun,
    cost_usd: float | None,
    *,
    operation: str = "asv3_researcher",
    parent_node_id: str | None = None,
) -> str:
    node_id = uuid4().hex
    cost = GenerationCost(
        model="test-model",
        provider="test-provider",
        source="model_registry",
        priced_at=_START,
        input_tokens=100,
        output_tokens=0,
        known_cost_usd=cost_usd or 0.0,
        complete=cost_usd is not None,
        lines=[
            TokenCostLine(
                category="input",
                tokens=100,
                usd_per_million=cost_usd * 10_000 if cost_usd is not None else None,
                cost_usd=cost_usd,
            )
        ],
    )
    db_session.add(
        AnswerGraphNode(
            run_id=run.id,
            node_id=node_id,
            parent_node_id=parent_node_id,
            kind="generation",
            operation=operation,
            status="COMPLETE",
            started_at=_START,
            ended_at=_START + timedelta(seconds=1),
            capture_status="COMPLETE",
            attributes={"usage_cost": cost.model_dump(mode="json")},
        )
    )
    db_session.flush()
    return node_id


def test_workflow_usage_counts_prompts_once_and_all_nested_research_costs(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    import onyx.db.workflow_usage as workflow_usage

    monkeypatch.setattr(workflow_usage, "_RESPONSE_BATCH_SIZE", 2)
    user, first_session = _session(db_session)
    _, second_session = _session(db_session, user)
    deep_query = _message(db_session, first_session)
    first_answer = _message(db_session, first_session, deep_query, "asv3")
    first_run = _run(db_session, deep_query, first_answer)
    parent_node = _generation(db_session, first_run, 0.2)
    _generation(db_session, first_run, 0.3, parent_node_id=parent_node)
    _generation(db_session, first_run, 99.0, operation="embed_query")
    # Regenerations and alternative-model answers cost money but are one prompt.
    alternate_answer = _message(db_session, first_session, deep_query, "asv3")
    _generation(db_session, _run(db_session, deep_query, alternate_answer), 0.4)
    fast_query = _message(db_session, first_session)
    fast_answer = _message(db_session, first_session, fast_query, "asv3_exp_cropped")
    _generation(db_session, _run(db_session, fast_query, fast_answer), 0.25)
    second_query = _message(db_session, second_session)
    second_answer = _message(db_session, second_session, second_query, "asv3")
    _generation(db_session, _run(db_session, second_query, second_answer), 0.1)

    usage = get_workflow_usage_by_email(db_session, _START, _END)[user.email]

    deep = usage["asv3"]
    assert (deep.query_count, deep.session_count) == (2, 2)
    assert deep.cost_cents == pytest.approx(100.0)
    assert deep.average_cost_cents_per_query == pytest.approx(50.0)
    assert deep.average_cost_cents_per_session == pytest.approx(50.0)
    assert deep.cost_status == "complete"
    fast = usage["asv3_exp_cropped"]
    assert (fast.query_count, fast.session_count) == (1, 1)
    assert fast.cost_cents == pytest.approx(25.0)

    db_session.add(
        UserUsage(
            user_id=user.id,
            window_start=_START,
            model="test-model",
            flow="asv3_researcher",
            provider="test-provider",
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=0,
            cost_cents=1234.0,
        )
    )
    db_session.flush()
    response = export_usage(
        start=_START.date(),
        end=_START.date(),
        period_from=None,
        period_to=None,
        model=None,
        _=user,
        db_session=db_session,
    )
    exported_user = next(item for item in response.users if item.email == user.email)
    assert exported_user.totals.cost_cents == 1234.0
    assert exported_user.totals.total_user_queries == 3
    assert exported_user.totals.total_user_sessions == 2
    assert exported_user.workflow_usage == usage


@pytest.mark.parametrize(
    ("status", "capture", "unpriced", "expected"),
    [
        ("COMPLETE", "PARTIAL", False, "partial"),
        ("COMPLETE", "COMPLETE", True, "partial"),
        ("RUNNING", "COMPLETE", False, "running"),
    ],
)
def test_partial_and_running_costs_retain_known_cost_without_false_averages(
    db_session: Session, status: str, capture: str, unpriced: bool, expected: str
) -> None:
    user, chat_session = _session(db_session)
    query = _message(db_session, chat_session)
    answer = _message(db_session, chat_session, query, "asv3")
    run = _run(db_session, query, answer, status=status, capture=capture)
    _generation(db_session, run, 0.5)
    if unpriced:
        _generation(db_session, run, None)

    usage = get_workflow_usage_by_email(db_session, _START, _END)[user.email]["asv3"]

    assert usage.cost_status == expected
    assert usage.known_cost_cents == pytest.approx(50.0)
    assert usage.cost_cents is None
    assert usage.average_cost_cents_per_query is None
    assert usage.average_cost_cents_per_session is None
    assert usage.unpriced_call_count == int(unpriced)


def test_missing_and_foreign_tenant_graphs_are_unavailable(db_session: Session) -> None:
    user, chat_session = _session(db_session)
    query = _message(db_session, chat_session)
    _message(db_session, chat_session, query, "asv3")
    foreign_answer = _message(db_session, chat_session, query, "asv3_exp_cropped")
    foreign_run = _run(db_session, query, foreign_answer, tenant_id="other_tenant")
    _generation(db_session, foreign_run, 99.0)

    usage = get_workflow_usage_by_email(db_session, _START, _END)[user.email]

    for workflow in ("asv3", "asv3_exp_cropped"):
        assert usage[workflow].query_count == 1
        assert usage[workflow].cost_status == "unavailable"
        assert usage[workflow].known_cost_cents == 0.0
        assert usage[workflow].cost_cents is None


def test_workflow_activity_uses_prompt_time_excludes_benchmarks_and_unknown_modes(
    db_session: Session,
) -> None:
    user, chat_session = _session(db_session)
    query = _message(db_session, chat_session)
    # The answer may arrive after the reporting interval; its prompt defines scope.
    _message(db_session, chat_session, query, "asv3", sent=_END)
    for sent in (_START - timedelta(microseconds=1), _END):
        out_of_scope_query = _message(db_session, chat_session, sent=sent)
        _message(db_session, chat_session, out_of_scope_query, "asv3_exp_cropped")
    unknown_query = _message(db_session, chat_session)
    _message(db_session, chat_session, unknown_query)
    _, benchmark_session = _session(db_session, user)
    benchmark_session.benchmark_flow = True
    benchmark_query = _message(db_session, benchmark_session)
    _message(db_session, benchmark_session, benchmark_query, "asv3_exp_cropped")

    usage = get_workflow_usage_by_email(db_session, _START, _END)[user.email]

    assert usage["asv3"].query_count == 1
    assert usage["asv3_exp_cropped"].query_count == 0
    assert usage["asv3_exp_cropped"].session_count == 0
    assert usage["asv3_exp_cropped"].cost_cents == 0.0
    assert usage["asv3_exp_cropped"].cost_status == "complete"


def test_new_user_arriving_during_report_cannot_mismatch_counts_and_cost_groups(
    db_session: Session,
) -> None:
    user, chat_session = _session(db_session)
    query = _message(db_session, chat_session)
    _message(db_session, chat_session, query, "asv3")
    connection = db_session.connection()
    concurrent_rows: list[tuple[str, UUID, UUID]] = []

    def _insert_after_scope_snapshot(
        _connection: Connection,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        if concurrent_rows or "chat_message.workflow" not in statement:
            return
        # Commit after the first scope SELECT has its snapshot. Separate count
        # and response SELECTs would now disagree about this new user/mode.
        with Session(bind=connection.engine) as concurrent:
            new_user, new_session = _session(concurrent)
            new_query = _message(concurrent, new_session)
            _message(concurrent, new_session, new_query, "asv3_exp_cropped")
            concurrent.commit()
            concurrent_rows.append((new_user.email, new_user.id, new_session.id))

    event.listen(connection, "after_cursor_execute", _insert_after_scope_snapshot)
    try:
        usage = get_workflow_usage_by_email(db_session, _START, _END)
        assert len(concurrent_rows) == 1
        assert concurrent_rows[0][0] not in usage
        assert usage[user.email]["asv3"].query_count == 1
    finally:
        event.remove(connection, "after_cursor_execute", _insert_after_scope_snapshot)
        for _, new_user_id, new_session_id in concurrent_rows:
            with Session(bind=connection.engine) as cleanup:
                cleanup.execute(
                    delete(ChatMessage).where(
                        ChatMessage.chat_session_id == new_session_id
                    )
                )
                cleanup.execute(
                    delete(ChatSession).where(ChatSession.id == new_session_id)
                )
                cleanup.execute(delete(User).where(User.__table__.c.id == new_user_id))
                cleanup.commit()
