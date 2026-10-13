"""Execute checkpoint transactions locally with failure at the COMMIT boundary."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest
from pydantic import JsonValue
from sqlalchemy import JSON, Column, MetaData, Table, Uuid, create_engine, select
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, InterfaceError, OperationalError
from sqlalchemy.orm import Session

from onyx.asv3.harness import Harness
from onyx.asv3.models import Decision, RunContext
from onyx.asv3.parallel_checkpoint_writer import ParallelCheckpointWriter
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.workflow_variant import ASV3_EXP_CROPPED_VARIANT
from onyx.db import asv3_runs
from onyx.db.models import ChatMessage, ChatSession, ToolCall


class DatabaseFault(Exception):
    def __init__(self, sqlstate: str | None) -> None:
        self.pgcode = sqlstate
        super().__init__("Synthetic database failure")


def transient(
    sqlstate: str | None = "40001", *, disconnected: bool = False
) -> OperationalError:
    return OperationalError(
        "COMMIT", {}, DatabaseFault(sqlstate), connection_invalidated=disconnected
    )


@dataclass
class Store:
    engine: Engine
    chats: Table
    messages: Table
    calls: Table
    owner: UUID
    snapshot: dict[str, JsonValue]
    sessions: list[Session] = field(default_factory=list)
    closed: list[Session] = field(default_factory=list)
    sleeps: list[float] = field(default_factory=list)
    failures: list[tuple[BaseException, bool]] = field(default_factory=list)
    payloads: list[str] = field(default_factory=list)
    locked: list[bool] = field(default_factory=list)
    after_failure: Callable[[], None] | None = None
    commits: int = 0

    def save(self, snapshot: dict[str, JsonValue] | None = None) -> None:
        asv3_runs.save_asv3_checkpoint(
            message_id=77,
            user_id=self.owner,
            snapshot=self.snapshot if snapshot is None else snapshot,
        )

    def stored(self) -> list[str]:
        with self.engine.connect() as connection:
            return list(
                connection.scalars(select(self.calls.c.tool_call_response)).all()
            )


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> Iterator[Store]:
    metadata = MetaData()
    chats = Table(
        "chat_session",
        metadata,
        Column("id", Uuid, primary_key=True),
        Column("user_id", ChatSession.__table__.c.user_id.type),
    )
    tables: dict[str, Table] = {}
    for model in (ChatMessage, ToolCall):
        tables[model.__tablename__] = Table(
            model.__tablename__,
            metadata,
            *[
                Column(
                    column.name,
                    JSON() if isinstance(column.type, (JSONB, ARRAY)) else column.type,
                    primary_key=column.primary_key,
                    nullable=True,
                )
                for column in model.__table__.columns
            ],
        )
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    metadata.create_all(engine)
    owner, chat = uuid4(), uuid4()
    state = Store(
        engine=engine,
        chats=chats,
        messages=tables["chat_message"],
        calls=tables["tool_call"],
        owner=owner,
        snapshot={
            "run_id": "checkpoint-owner",
            "sequence": 7,
            "asv3_workflow_variant": ASV3_EXP_CROPPED_VARIANT,
            "answer": "Exact guarded answer [1].",
            "evidence": {"originals": [{"text": "Exact operative condition."}]},
        },
    )
    with engine.begin() as connection:
        connection.execute(chats.insert(), {"id": chat, "user_id": owner})
        connection.execute(
            state.messages.insert(),
            {"id": 77, "chat_session_id": chat, "message_type": "ASSISTANT"},
        )

    original_commit = Session.commit
    original_owner = asv3_runs._require_owned_message

    def commit(session: Session) -> None:
        state.commits += 1
        state.payloads.extend(
            item.tool_call_response
            for item in [*session.new, *session.dirty]
            if isinstance(item, ToolCall)
        )
        if state.failures:
            error, persisted = state.failures.pop(0)
            if persisted:
                original_commit(session)
            if state.after_failure is not None:
                state.after_failure()
            raise error
        original_commit(session)

    def owned(
        session: Session, message_id: int, user_id: UUID | None, *, lock: bool = False
    ) -> ChatMessage:
        state.locked.append(lock)
        return original_owner(session, message_id, user_id, lock=lock)

    @contextmanager
    def sessions() -> Iterator[Session]:
        with Session(engine) as session:
            state.sessions.append(session)
            try:
                yield session
            finally:
                session.close()
                state.closed.append(session)

    monkeypatch.setattr(Session, "commit", commit)
    monkeypatch.setattr(asv3_runs, "get_session_with_current_tenant", sessions)
    monkeypatch.setattr(asv3_runs, "_require_owned_message", owned)
    monkeypatch.setattr(asv3_runs.time, "sleep", state.sleeps.append)
    try:
        yield state
    finally:
        engine.dispose()


@pytest.mark.parametrize("sqlstate", sorted(asv3_runs._CHECKPOINT_TRANSIENT_SQLSTATES))
def test_transient_commit_reuses_exact_payload_in_fresh_locked_session(
    store: Store, sqlstate: str
) -> None:
    expected = asv3_runs.encode_asv3_checkpoint(store.snapshot)
    store.failures = [(transient(sqlstate), False)]
    store.after_failure = lambda: store.snapshot.update(answer="Caller changed alias")
    store.save()
    assert store.stored() == [expected]
    assert store.payloads == [expected, expected]
    assert store.commits == 2 and store.sleeps == [0.2]
    assert len({id(session) for session in store.sessions}) == 2
    assert store.closed == store.sessions
    assert store.locked == [True, True]


@pytest.mark.parametrize("existing", [False, True])
def test_lost_commit_acknowledgement_does_not_duplicate_or_rewrite(
    store: Store, existing: bool
) -> None:
    if existing:
        store.save({**store.snapshot, "sequence": 6, "answer": "Earlier answer"})
        store.commits = 0
        store.payloads.clear()
    expected = asv3_runs.encode_asv3_checkpoint(store.snapshot)
    store.failures = [(transient(None, disconnected=True), True)]
    store.save()
    assert store.stored() == [expected]
    assert store.commits == 1 and store.payloads == [expected]
    assert store.sleeps == [0.2]
    assert store.locked[-2:] == [True, True]


def test_newer_revision_is_preserved_after_uncertain_commit(store: Store) -> None:
    newer = asv3_runs.encode_asv3_checkpoint(
        {**store.snapshot, "sequence": 8, "answer": "Newer independently saved answer"}
    )

    def advance() -> None:
        with store.engine.begin() as connection:
            connection.execute(store.calls.update().values(tool_call_response=newer))

    store.failures = [(transient("08006"), True)]
    store.after_failure = advance
    store.save()
    assert store.stored() == [newer]
    assert store.commits == 1 and store.sleeps == [0.2]


def test_retry_revalidates_owner_after_transient_failure(store: Store) -> None:
    def revoke() -> None:
        with store.engine.begin() as connection:
            connection.execute(store.chats.update().values(user_id=uuid4()))

    store.failures = [(transient(), False)]
    store.after_failure = revoke
    with pytest.raises(PermissionError, match="outside the current user's session"):
        store.save()
    assert store.stored() == [] and store.commits == 1
    assert store.locked == [True, True] and store.sleeps == [0.2]


def test_foreign_run_after_uncertain_commit_is_rejected(store: Store) -> None:
    foreign = asv3_runs.encode_asv3_checkpoint(
        {**store.snapshot, "run_id": "another-run"}
    )

    def replace() -> None:
        with store.engine.begin() as connection:
            connection.execute(store.calls.update().values(tool_call_response=foreign))

    store.failures = [(transient(), True)]
    store.after_failure = replace
    with pytest.raises(ValueError, match="cannot change ASv3 run identity"):
        store.save()
    assert store.stored() == [foreign] and store.commits == 1


def test_exhaustion_raises_last_failure_after_three_attempts(store: Store) -> None:
    errors = [transient() for _ in range(3)]
    store.failures = [(error, False) for error in errors]
    with pytest.raises(OperationalError) as raised:
        store.save()
    assert raised.value is errors[-1]
    assert store.commits == 3 and store.sleeps == [0.2, 0.5]
    assert store.stored() == []
    assert store.closed == store.sessions


@pytest.mark.parametrize("variant", [None, "asv3", "asv3_tuned", "unknown"])
def test_non_cropped_retains_single_attempt(store: Store, variant: str | None) -> None:
    if variant is None:
        store.snapshot.pop("asv3_workflow_variant")
    else:
        store.snapshot["asv3_workflow_variant"] = variant
    error = transient()
    store.failures = [(error, False)]
    with pytest.raises(OperationalError) as raised:
        store.save()
    assert raised.value is error
    assert store.commits == 1 and store.sleeps == [] and store.stored() == []


@pytest.mark.parametrize(
    ("sqlstate", "disconnected"),
    [(None, False)]
    + [
        (sqlstate, disconnected)
        for sqlstate in ("57014", "28P01", "28000", "23505", "42P01")
        for disconnected in (False, True)
    ],
)
def test_explicit_permanent_or_unclassified_error_is_not_retried(
    store: Store, sqlstate: str | None, disconnected: bool
) -> None:
    error = transient(sqlstate, disconnected=disconnected)
    store.failures = [(error, False)]
    with pytest.raises(OperationalError) as raised:
        store.save()
    assert raised.value is error
    assert store.commits == 1 and store.sleeps == []


def test_interface_disconnect_recovers_but_integrity_error_does_not(
    store: Store,
) -> None:
    store.failures = [
        (
            InterfaceError(
                "COMMIT", {}, DatabaseFault(None), connection_invalidated=True
            ),
            False,
        )
    ]
    store.save()
    assert store.commits == 2
    store.sleeps.clear()
    store.failures = [(IntegrityError("COMMIT", {}, DatabaseFault("23505")), False)]
    with pytest.raises(IntegrityError):
        store.save({**store.snapshot, "sequence": 8})
    assert store.commits == 3 and store.sleeps == []


def test_encoding_failure_never_opens_a_transaction(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(asv3_runs, "MAX_CROPPED_DECODED_CHECKPOINT_BYTES", 1)
    with pytest.raises(ValueError, match="decoded payload budget"):
        store.save()
    assert store.sessions == [] and store.sleeps == []


def test_verified_harness_answer_survives_transient_checkpoint_failure(
    store: Store,
) -> None:
    guards: list[str] = []

    def guard(answer: str) -> None:
        guards.append(answer)

    def checkpoint(snapshot: dict[str, JsonValue]) -> None:
        store.save({**store.snapshot, "harness": snapshot})

    store.failures = [(transient(), False)]
    harness = Harness(
        request="Offline checkpoint recovery",
        context=RunContext(),
        registry=CapabilityRegistry([]),
        decide=lambda _view: Decision(answer="Guarded answer"),
        draft_guard=guard,
        checkpoint=checkpoint,
    )
    result = harness.run()
    assert result.answer == "Guarded answer" and guards == ["Guarded answer"]
    assert store.commits == 2 and store.sleeps == [0.2]
    assert len(store.stored()) == 1


def test_async_writer_waits_for_recovery_without_becoming_poisoned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[str] = []
    committed: list[str] = []

    def persist(**arguments: object) -> None:
        payload = arguments["payload"]
        assert isinstance(payload, str)
        attempts.append(payload)
        if len(attempts) == 1:
            raise transient()
        committed.append(payload)

    monkeypatch.setattr(asv3_runs, "_save_checkpoint_payload", persist)
    monkeypatch.setattr(asv3_runs.time, "sleep", lambda _delay: None)
    writer = ParallelCheckpointWriter(
        lambda snapshot: asv3_runs.save_asv3_checkpoint(
            message_id=77, user_id=None, snapshot=snapshot
        )
    )
    try:
        revision = writer.submit(
            {
                "run_id": "async",
                "sequence": 1,
                "asv3_workflow_variant": ASV3_EXP_CROPPED_VARIANT,
            }
        )
        writer.flush(revision)
        writer.raise_if_failed()
    finally:
        writer.close()
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    assert committed == [attempts[0]]
