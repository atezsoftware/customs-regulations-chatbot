from collections.abc import Iterator
from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest
from pydantic import JsonValue
from sqlalchemy import (
    Boolean,
    Column,
    Integer,
    MetaData,
    String,
    Table,
    Uuid,
    create_engine,
)
from sqlalchemy.orm import Session

from onyx.configs.constants import MessageType
from onyx.db import asv3_runs
from onyx.db.models import ChatSession


@dataclass
class MemoryStore:
    session: Session
    chats: Table
    messages: Table
    calls: Table
    chat_id: UUID
    owner_id: UUID

    def message(
        self,
        identifier: int,
        parent: int | None,
        kind: MessageType,
        *,
        chat_id: UUID | None = None,
    ) -> None:
        self.session.execute(
            self.messages.insert().values(
                id=identifier,
                parent_message_id=parent,
                chat_session_id=chat_id or self.chat_id,
                message_type=kind.name,
            )
        )

    def checkpoint(
        self, message_id: int, label: str, *, chat_id: UUID | None = None
    ) -> dict[str, JsonValue]:
        snapshot: dict[str, JsonValue] = {
            "version": 1,
            "request": label,
            "run_id": label,
            "session_research": {"requests": [label]},
            "evidence": {"records": [{"item": {"text": label}}]},
        }
        self.session.execute(
            self.calls.insert().values(
                parent_chat_message_id=message_id,
                chat_session_id=chat_id or self.chat_id,
                tool_id=asv3_runs.ASV3_CHECKPOINT_TOOL_ID,
                tool_call_response=asv3_runs.encode_asv3_checkpoint(snapshot),
            )
        )
        return snapshot

    def load(self, message_id: int) -> dict[str, JsonValue] | None:
        return asv3_runs.load_asv3_session_checkpoint(
            chat_session_id=self.chat_id,
            user_message_id=message_id,
            user_id=self.owner_id,
        )


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> Iterator[MemoryStore]:
    """Execute the real recursive query against isolated, in-memory tables."""
    metadata = MetaData()
    chats = Table(
        "chat_session",
        metadata,
        Column("id", Uuid, primary_key=True),
        Column("user_id", ChatSession.__table__.c.user_id.type),
        Column("deleted", Boolean, nullable=False),
    )
    messages = Table(
        "chat_message",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("parent_message_id", Integer),
        Column("chat_session_id", Uuid, nullable=False),
        Column("message_type", String, nullable=False),
    )
    calls = Table(
        "tool_call",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("parent_chat_message_id", Integer),
        Column("chat_session_id", Uuid, nullable=False),
        Column("tool_id", Integer, nullable=False),
        Column("tool_call_response", String, nullable=False),
    )
    engine = create_engine("sqlite://")
    metadata.create_all(engine)
    with Session(engine) as session:
        state = MemoryStore(session, chats, messages, calls, uuid4(), uuid4())
        session.execute(
            chats.insert().values(
                id=state.chat_id, user_id=state.owner_id, deleted=False
            )
        )
        # A nested session uses the same connection and performs no commits.
        monkeypatch.setattr(
            asv3_runs,
            "get_session_with_current_tenant",
            lambda: Session(session.connection()),
        )
        yield state
    engine.dispose()


def test_nearest_ancestor_excludes_newer_sibling_and_current_turn(
    store: MemoryStore,
) -> None:
    store.message(1, None, MessageType.ASSISTANT)
    store.checkpoint(1, "older")
    store.message(2, 1, MessageType.USER)
    store.message(3, 2, MessageType.ASSISTANT)
    expected = store.checkpoint(3, "selected branch")
    store.message(4, 2, MessageType.ASSISTANT)
    store.checkpoint(4, "newer unselected branch")
    store.message(5, 3, MessageType.USER)
    store.message(6, 5, MessageType.ASSISTANT)
    store.checkpoint(6, "current turn answer")

    assert store.load(5) == expected


def test_long_history_and_integer_id_boundaries_are_not_truncated(
    store: MemoryStore,
) -> None:
    store.message(1, None, MessageType.ASSISTANT)
    expected = store.checkpoint(1, "earliest original")
    for identifier in range(2, 22):
        store.message(
            identifier,
            identifier - 1,
            MessageType.USER if identifier % 2 else MessageType.ASSISTANT,
        )
    assert store.load(21) == expected


@pytest.mark.parametrize("foreign_parent", [False, True])
def test_no_eligible_checkpoint_does_not_use_another_chat(
    store: MemoryStore, foreign_parent: bool
) -> None:
    foreign_chat = uuid4()
    store.session.execute(
        store.chats.insert().values(
            id=foreign_chat, user_id=store.owner_id, deleted=False
        )
    )
    store.message(1, None, MessageType.ASSISTANT, chat_id=foreign_chat)
    store.checkpoint(1, "foreign original", chat_id=foreign_chat)
    store.message(2, 1 if foreign_parent else None, MessageType.USER)
    assert store.load(2) is None


def test_checkpoint_record_itself_must_belong_to_the_same_chat(
    store: MemoryStore,
) -> None:
    store.message(1, None, MessageType.ASSISTANT)
    store.checkpoint(1, "mismatched tool log", chat_id=uuid4())
    store.message(2, 1, MessageType.USER)
    assert store.load(2) is None


@pytest.mark.parametrize("failure", ["owner", "chat", "deleted", "assistant"])
def test_owned_nondeleted_current_user_message_is_required(
    store: MemoryStore, failure: str
) -> None:
    store.message(
        1,
        None,
        MessageType.ASSISTANT if failure == "assistant" else MessageType.USER,
    )
    if failure == "deleted":
        store.session.execute(store.chats.update().values(deleted=True))
    with pytest.raises(PermissionError):
        asv3_runs.load_asv3_session_checkpoint(
            chat_session_id=uuid4() if failure == "chat" else store.chat_id,
            user_message_id=1,
            user_id=uuid4() if failure == "owner" else store.owner_id,
        )


def test_cycle_fails_closed_even_with_a_nearby_checkpoint(store: MemoryStore) -> None:
    store.message(1, 2, MessageType.USER)
    store.message(2, 1, MessageType.ASSISTANT)
    store.checkpoint(2, "not safe to restore")
    store.message(3, 2, MessageType.USER)
    with pytest.raises(ValueError, match="cyclic"):
        store.load(3)
