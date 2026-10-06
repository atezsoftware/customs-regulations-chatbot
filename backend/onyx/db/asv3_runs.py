"""Tenant-scoped ASv3 checkpoints stored in the existing durable tool log."""

from __future__ import annotations

import base64
import json
import zlib
from uuid import UUID

from pydantic import JsonValue
from sqlalchemy import String, cast, literal, select
from sqlalchemy.orm import Session

from onyx.asv3.parallel_checkpoint import (
    compact_parallel_checkpoint,
    restore_parallel_checkpoint,
)
from onyx.configs.constants import MessageType
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.models import ChatMessage, ChatSession, ToolCall

ASV3_CHECKPOINT_TOOL_ID = -3001
ASV3_CHECKPOINT_VERSION = 1
MAX_CHECKPOINT_BYTES = 2_500_000
MAX_DECODED_CHECKPOINT_BYTES = 8_000_000


def encode_asv3_checkpoint(snapshot: dict[str, JsonValue]) -> str:
    raw = json.dumps(
        compact_parallel_checkpoint(
            {**snapshot, "version": ASV3_CHECKPOINT_VERSION},
            max_unit_bytes=MAX_DECODED_CHECKPOINT_BYTES,
        ),
        ensure_ascii=False,
    ).encode()
    if len(raw) > MAX_DECODED_CHECKPOINT_BYTES:
        raise ValueError("ASv3 checkpoint exceeds its decoded payload budget")
    payload = json.dumps(
        {
            "version": ASV3_CHECKPOINT_VERSION,
            "encoding": "zlib-base64",
            "data": base64.b64encode(zlib.compress(raw, level=3)).decode("ascii"),
        }
    )
    if len(payload.encode()) > MAX_CHECKPOINT_BYTES:
        raise ValueError("ASv3 checkpoint exceeds its durable payload budget")
    return payload


def decode_asv3_checkpoint(payload: str) -> dict[str, JsonValue]:
    if len(payload.encode()) > MAX_DECODED_CHECKPOINT_BYTES:
        raise ValueError("ASv3 checkpoint exceeds its payload budget")
    data: dict[str, JsonValue] = json.loads(payload)
    if data.get("encoding") == "zlib-base64":
        encoded = data.get("data")
        if not isinstance(encoded, str):
            raise ValueError("Invalid compressed checkpoint")
        decoder = zlib.decompressobj()
        raw = decoder.decompress(
            base64.b64decode(encoded, validate=True), MAX_DECODED_CHECKPOINT_BYTES + 1
        )
        if (
            len(raw) > MAX_DECODED_CHECKPOINT_BYTES
            or not decoder.eof
            or decoder.unused_data
        ):
            raise ValueError("Invalid or oversized compressed checkpoint")
        data = json.loads(raw)
    if not isinstance(data, dict) or data.get("version") != ASV3_CHECKPOINT_VERSION:
        raise ValueError("Unsupported ASv3 checkpoint version")
    return restore_parallel_checkpoint(
        data, max_unit_bytes=MAX_DECODED_CHECKPOINT_BYTES
    )


def _require_owned_message(
    session: Session, message_id: int, user_id: UUID | None, *, lock: bool = False
) -> ChatMessage:
    statement = (
        select(ChatMessage)
        .join(ChatSession, ChatMessage.chat_session_id == ChatSession.id)
        .where(ChatMessage.id == message_id, ChatSession.user_id == user_id)
    )
    if lock:
        statement = statement.with_for_update(of=ChatMessage)
    message = session.scalar(statement)
    if message is None:
        raise PermissionError("ASv3 checkpoint is outside the current user's session")
    return message


def save_asv3_checkpoint(
    *, message_id: int, user_id: UUID | None, snapshot: dict[str, JsonValue]
) -> None:
    """Serialize writes on the owning message and reject stale snapshot revisions."""
    run_id = snapshot.get("run_id")
    sequence = snapshot.get("sequence")
    if not isinstance(run_id, str) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("An ASv3 checkpoint needs run identity and sequence")
    payload = encode_asv3_checkpoint(snapshot)
    with get_session_with_current_tenant() as session:
        message = _require_owned_message(session, message_id, user_id, lock=True)
        record = session.scalar(
            select(ToolCall).where(
                ToolCall.parent_chat_message_id == message_id,
                ToolCall.tool_id == ASV3_CHECKPOINT_TOOL_ID,
            )
        )
        if record is not None:
            previous = decode_asv3_checkpoint(record.tool_call_response)
            if previous["run_id"] != run_id:
                raise ValueError("A message cannot change ASv3 run identity")
            prior_sequence = previous.get("sequence")
            if not isinstance(prior_sequence, int):
                raise ValueError("Invalid checkpoint sequence")
            if prior_sequence >= sequence:
                return
            record.tool_call_response = payload
        else:
            session.add(
                ToolCall(
                    chat_session_id=message.chat_session_id,
                    parent_chat_message_id=message_id,
                    parent_tool_call_id=None,
                    turn_number=0,
                    tab_index=0,
                    tool_id=ASV3_CHECKPOINT_TOOL_ID,
                    tool_call_id=f"asv3-checkpoint:{run_id}",
                    reasoning_tokens=None,
                    tool_call_arguments={"workflow": "ASv3", "run_id": run_id},
                    tool_call_response=payload,
                    tool_call_tokens=0,
                )
            )
        session.commit()


def load_asv3_checkpoint(
    *, message_id: int, user_id: UUID | None
) -> dict[str, JsonValue] | None:
    with get_session_with_current_tenant() as session:
        _require_owned_message(session, message_id, user_id)
        payload = session.scalar(
            select(ToolCall.tool_call_response).where(
                ToolCall.parent_chat_message_id == message_id,
                ToolCall.tool_id == ASV3_CHECKPOINT_TOOL_ID,
            )
        )
        if payload is None:
            return None
        return decode_asv3_checkpoint(payload)


def load_asv3_session_checkpoint(
    *, chat_session_id: UUID, user_message_id: int, user_id: UUID | None
) -> dict[str, JsonValue] | None:
    """Read the nearest checkpoint on this owned user message's ancestor branch."""
    owned_message = (
        ChatMessage.id == user_message_id,
        ChatMessage.chat_session_id == chat_session_id,
        ChatMessage.message_type == MessageType.USER,
        ChatSession.id == chat_session_id,
        ChatSession.user_id == user_id,
        ChatSession.deleted.is_(False),
    )
    with get_session_with_current_tenant() as session:
        if (
            session.scalar(
                select(ChatMessage.id)
                .join(ChatSession, ChatMessage.chat_session_id == ChatSession.id)
                .where(*owned_message)
            )
            is None
        ):
            raise PermissionError(
                "ASv3 session memory is outside the owned message chain"
            )

        # Comma-delimited integer IDs detect cycles without truncating long histories.
        message_path = literal(",") + cast(ChatMessage.id, String) + literal(",")
        ancestors = (
            select(
                ChatMessage.id.label("message_id"),
                ChatMessage.parent_message_id,
                literal(0).label("depth"),
                message_path.label("path"),
                literal(False).label("cycle"),
            )
            .join(ChatSession, ChatMessage.chat_session_id == ChatSession.id)
            .where(*owned_message)
            .cte("asv3_session_ancestors", recursive=True)
        )
        ancestors = ancestors.union_all(
            select(
                ChatMessage.id,
                ChatMessage.parent_message_id,
                ancestors.c.depth + 1,
                ancestors.c.path + cast(ChatMessage.id, String) + literal(","),
                ancestors.c.path.contains(message_path),
            )
            .join(ancestors, ChatMessage.id == ancestors.c.parent_message_id)
            .where(
                ChatMessage.chat_session_id == chat_session_id,
                ancestors.c.cycle.is_(False),
            )
        )
        checkpoint = (
            select(ToolCall.tool_call_response)
            .join(ancestors, ToolCall.parent_chat_message_id == ancestors.c.message_id)
            .join(ChatMessage, ChatMessage.id == ancestors.c.message_id)
            .where(
                ancestors.c.depth > 0,
                ChatMessage.message_type == MessageType.ASSISTANT,
                ToolCall.chat_session_id == chat_session_id,
                ToolCall.tool_id == ASV3_CHECKPOINT_TOOL_ID,
            )
            .order_by(ancestors.c.depth, ToolCall.id.desc())
            .limit(1)
            .scalar_subquery()
        )
        cycle, payload = session.execute(
            select(
                select(ancestors.c.message_id)
                .where(ancestors.c.cycle.is_(True))
                .exists(),
                checkpoint,
            )
        ).one()
        if cycle:
            raise ValueError("ASv3 session memory has a cyclic message chain")
        return decode_asv3_checkpoint(payload) if payload is not None else None


def checkpoint_progress_packets(payload: str) -> list[dict[str, JsonValue]]:
    """History replay calls this only after the chat's access/publication fence."""
    data = decode_asv3_checkpoint(payload)
    if data.get("version") != ASV3_CHECKPOINT_VERSION:
        return []
    events = data.get("progress", [])
    if not isinstance(events, list):
        raise ValueError("Invalid ASv3 progress checkpoint")
    return [event for event in events if isinstance(event, dict)]
