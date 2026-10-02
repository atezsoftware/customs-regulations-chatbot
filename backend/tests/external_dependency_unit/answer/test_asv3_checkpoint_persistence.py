from uuid import uuid4

import pytest
from pydantic import JsonValue
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.configs.constants import MessageType
from onyx.db.asv3_runs import (
    ASV3_CHECKPOINT_TOOL_ID,
    checkpoint_progress_packets,
    load_asv3_checkpoint,
    save_asv3_checkpoint,
)
from onyx.db.models import ChatMessage, ChatSession, ToolCall
from tests.external_dependency_unit.conftest import create_test_user


@pytest.mark.usefixtures("tenant_context")
def test_checkpoint_ownership_sequence_and_durable_replay(db_session: Session) -> None:
    owner = create_test_user(db_session, "asv3_owner")
    stranger = create_test_user(db_session, "asv3_stranger")
    session = ChatSession(id=uuid4(), user_id=owner.id)
    db_session.add(session)
    db_session.flush()
    message = ChatMessage(
        chat_session_id=session.id,
        message="",
        token_count=0,
        message_type=MessageType.ASSISTANT,
    )
    db_session.add(message)
    db_session.commit()
    try:
        run_id = str(uuid4())
        event: dict[str, JsonValue] = {
            "type": "asv3_progress",
            "run_id": run_id,
            "event_id": str(uuid4()),
            "sequence": 1,
            "language": "tr",
            "phase": "research",
            "status": "running",
            "title": "Süreyi inceliyorum",
            "message": "Sürenin hangi olaydan başladığını kontrol ediyorum.",
        }
        snapshot: dict[str, JsonValue] = {
            "run_id": run_id,
            "sequence": 1,
            "progress": [event],
            "evidence": {"version": 1, "records": []},
        }
        save_asv3_checkpoint(message_id=message.id, user_id=owner.id, snapshot=snapshot)
        save_asv3_checkpoint(
            message_id=message.id,
            user_id=owner.id,
            snapshot={**snapshot, "sequence": 2, "request": "tamir"},
        )
        save_asv3_checkpoint(message_id=message.id, user_id=owner.id, snapshot=snapshot)
        restored = load_asv3_checkpoint(message_id=message.id, user_id=owner.id)
        assert restored is not None and restored["sequence"] == 2
        assert restored["request"] == "tamir"
        records = db_session.scalars(
            select(ToolCall).where(
                ToolCall.parent_chat_message_id == message.id,
                ToolCall.tool_id == ASV3_CHECKPOINT_TOOL_ID,
            )
        ).all()
        assert len(records) == 1
        db_session.refresh(records[0])
        assert checkpoint_progress_packets(records[0].tool_call_response) == [event]
        with pytest.raises(PermissionError):
            load_asv3_checkpoint(message_id=message.id, user_id=stranger.id)
        with pytest.raises(PermissionError):
            save_asv3_checkpoint(
                message_id=message.id, user_id=stranger.id, snapshot=snapshot
            )
        with pytest.raises(ValueError, match="run identity"):
            save_asv3_checkpoint(
                message_id=message.id,
                user_id=owner.id,
                snapshot={**snapshot, "sequence": 3, "run_id": str(uuid4())},
            )
    finally:
        db_session.delete(session)
        db_session.delete(owner)
        db_session.delete(stranger)
        db_session.commit()
