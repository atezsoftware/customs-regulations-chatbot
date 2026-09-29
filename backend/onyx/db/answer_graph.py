"""Tenant-scoped persistence for answer execution graphs."""

from __future__ import annotations

import datetime
import hashlib
import time
from typing import Any
from uuid import UUID

from sqlalchemy import delete, select, tuple_
from sqlalchemy.orm import Session

from onyx.configs.constants import MessageType
from onyx.db.models import (
    AnswerGraphAccessEvent,
    AnswerGraphEdge,
    AnswerGraphNode,
    AnswerGraphPayload,
    AnswerGraphRun,
    ChatMessage,
    ChatSession,
)
from shared_configs.contextvars import get_current_tenant_id


def create_answer_graph_run(
    db_session: Session,
    *,
    trace_id: str,
    assistant_message_id: int,
    user_message_id: int,
    chat_session_id: UUID,
    model_name: str | None,
) -> UUID:
    if (
        db_session.scalar(
            select(ChatSession.deleted).where(ChatSession.id == chat_session_id)
        )
        is not False
    ):
        raise ValueError("Answer graph chat session is unavailable")
    message_rows = list(
        db_session.scalars(
            select(ChatMessage).where(
                ChatMessage.id.in_((assistant_message_id, user_message_id)),
                ChatMessage.chat_session_id == chat_session_id,
            )
        )
    )
    messages = {message.id: message for message in message_rows}
    if (
        len(messages) != 2
        or messages[assistant_message_id].message_type != MessageType.ASSISTANT
        or messages[user_message_id].message_type != MessageType.USER
    ):
        raise ValueError("Answer graph message binding is invalid")
    existing = db_session.scalar(
        select(AnswerGraphRun)
        .where(AnswerGraphRun.assistant_message_id == assistant_message_id)
        .with_for_update()
    )
    if existing is not None:
        if (
            existing.tenant_id != get_current_tenant_id()
            or existing.user_message_id != user_message_id
            or existing.chat_session_id != chat_session_id
        ):
            raise ValueError("Answer graph retry binding is invalid")
        existing.status = "RUNNING"
        existing.capture_status = "PARTIAL"
        existing.capture_error = "Assistant generation restarted"
        db_session.commit()
        return existing.id
    run = AnswerGraphRun(
        tenant_id=get_current_tenant_id(),
        trace_id=trace_id,
        assistant_message_id=assistant_message_id,
        user_message_id=user_message_id,
        chat_session_id=chat_session_id,
        model_name=model_name,
        status="RUNNING",
        capture_status="COMPLETE",
    )
    db_session.add(run)
    db_session.commit()
    return run.id


def get_answer_graph_run_by_trace(
    db_session: Session, trace_id: str
) -> AnswerGraphRun | None:
    return db_session.scalar(
        select(AnswerGraphRun).where(
            AnswerGraphRun.trace_id == trace_id,
            AnswerGraphRun.tenant_id == get_current_tenant_id(),
        )
    )


def get_answer_graph_run_by_message(
    db_session: Session, assistant_message_id: int
) -> AnswerGraphRun | None:
    return db_session.scalar(
        select(AnswerGraphRun)
        .join(ChatMessage, ChatMessage.id == AnswerGraphRun.assistant_message_id)
        .join(ChatSession, ChatSession.id == AnswerGraphRun.chat_session_id)
        .where(
            AnswerGraphRun.assistant_message_id == assistant_message_id,
            AnswerGraphRun.tenant_id == get_current_tenant_id(),
            ChatMessage.message_type == MessageType.ASSISTANT,
            ChatSession.deleted.is_(False),
        )
    )


def get_answer_graph_run(db_session: Session, run_id: UUID) -> AnswerGraphRun | None:
    return db_session.scalar(
        select(AnswerGraphRun)
        .join(ChatSession, ChatSession.id == AnswerGraphRun.chat_session_id)
        .where(
            AnswerGraphRun.id == run_id,
            AnswerGraphRun.tenant_id == get_current_tenant_id(),
            ChatSession.deleted.is_(False),
        )
    )


def record_answer_graph_access(
    db_session: Session,
    *,
    run_id: UUID,
    user_id: UUID,
    action: str,
    node_id: str | None = None,
) -> None:
    db_session.add(
        AnswerGraphAccessEvent(
            tenant_id=get_current_tenant_id(),
            run_id=run_id,
            node_id=node_id,
            user_id=user_id,
            action=action,
        )
    )
    db_session.commit()


def answer_graph_message_exists(db_session: Session, assistant_message_id: int) -> bool:
    return (
        db_session.scalar(
            select(ChatMessage.id)
            .join(ChatSession, ChatSession.id == ChatMessage.chat_session_id)
            .where(
                ChatMessage.id == assistant_message_id,
                ChatMessage.message_type == MessageType.ASSISTANT,
                ChatSession.deleted.is_(False),
            )
        )
        is not None
    )


def start_answer_graph_node(
    db_session: Session,
    *,
    run_id: UUID,
    node_id: str,
    parent_node_id: str | None,
    kind: str,
    operation: str,
    started_at: datetime.datetime,
) -> None:
    existing = db_session.get(AnswerGraphNode, (run_id, node_id))
    if existing is not None:
        return
    db_session.add(
        AnswerGraphNode(
            run_id=run_id,
            node_id=node_id,
            parent_node_id=parent_node_id,
            kind=kind,
            operation=operation,
            started_at=started_at,
            status="RUNNING",
            capture_status="COMPLETE",
            attributes={},
        )
    )
    db_session.commit()


def finish_answer_graph_node(
    db_session: Session,
    *,
    run_id: UUID,
    node_id: str,
    parent_node_id: str | None,
    kind: str,
    operation: str,
    started_at: datetime.datetime,
    ended_at: datetime.datetime,
    status: str,
    attributes: dict[str, Any],
    input_ciphertext: bytes | None,
    output_ciphertext: bytes | None,
    reasoning_ciphertext: bytes | None,
    error: str | None,
    capture_status: str,
) -> None:
    node = db_session.get(AnswerGraphNode, (run_id, node_id))
    if node is None:
        node = AnswerGraphNode(
            run_id=run_id,
            node_id=node_id,
            parent_node_id=parent_node_id,
            kind=kind,
            operation=operation,
            started_at=started_at,
        )
        db_session.add(node)
    else:
        db_session.execute(
            delete(AnswerGraphPayload).where(
                AnswerGraphPayload.run_id == run_id,
                AnswerGraphPayload.node_id == node_id,
            )
        )
        node.parent_node_id = parent_node_id
        node.kind = kind
        node.operation = operation
        node.started_at = started_at
    node.ended_at = ended_at
    node.status = status
    node.attributes = attributes
    node.has_input = input_ciphertext is not None
    node.has_output = output_ciphertext is not None
    node.has_reasoning = reasoning_ciphertext is not None
    node.error = error
    node.capture_status = capture_status
    db_session.add_all(
        AnswerGraphPayload(
            run_id=run_id, node_id=node_id, part=part, ciphertext=ciphertext
        )
        for part, ciphertext in (
            ("input", input_ciphertext),
            ("output", output_ciphertext),
            ("reasoning", reasoning_ciphertext),
        )
        if ciphertext is not None
    )
    db_session.commit()


def add_answer_graph_edge(
    db_session: Session,
    *,
    run_id: UUID,
    from_node_id: str,
    to_node_id: str,
    kind: str,
) -> None:
    if (
        db_session.get(AnswerGraphNode, (run_id, from_node_id)) is None
        or db_session.get(AnswerGraphNode, (run_id, to_node_id)) is None
    ):
        raise ValueError("Answer graph data link references an unknown node")
    existing = db_session.scalar(
        select(AnswerGraphEdge.id).where(
            AnswerGraphEdge.run_id == run_id,
            AnswerGraphEdge.from_node_id == from_node_id,
            AnswerGraphEdge.to_node_id == to_node_id,
            AnswerGraphEdge.kind == kind,
        )
    )
    if existing is None:
        db_session.add(
            AnswerGraphEdge(
                run_id=run_id,
                from_node_id=from_node_id,
                to_node_id=to_node_id,
                kind=kind,
            )
        )
        db_session.commit()


def mark_answer_graph_capture_partial(
    db_session: Session, run_id: UUID, reason: str
) -> None:
    run = db_session.get(AnswerGraphRun, run_id)
    if run is None or run.tenant_id != get_current_tenant_id():
        return
    run.capture_status = "PARTIAL"
    run.capture_error = reason
    db_session.commit()


def mark_answer_graph_node_partial(
    db_session: Session, run_id: UUID, node_id: str, reason: str
) -> None:
    node = db_session.get(AnswerGraphNode, (run_id, node_id))
    if node is None:
        mark_answer_graph_capture_partial(db_session, run_id, reason)
        return
    node.status = "FAILED"
    node.capture_status = "PARTIAL"
    node.error = reason
    node.ended_at = datetime.datetime.now(datetime.timezone.utc)
    mark_answer_graph_capture_partial(db_session, run_id, reason)


def finish_answer_graph_run(db_session: Session, run_id: UUID, *, failed: bool) -> None:
    run = db_session.get(AnswerGraphRun, run_id)
    if run is None or run.tenant_id != get_current_tenant_id():
        return
    unfinished = db_session.scalar(
        select(AnswerGraphNode.node_id)
        .where(
            AnswerGraphNode.run_id == run_id,
            AnswerGraphNode.ended_at.is_(None),
        )
        .limit(1)
    )
    if unfinished is not None:
        run.capture_status = "PARTIAL"
        run.capture_error = "One or more operations did not finish recording"
    # The persisted assistant message is written after the trace closes.
    run.status = "FAILED" if failed else "FINALIZING"
    run.time_finished = datetime.datetime.now(datetime.timezone.utc)
    db_session.commit()


def finish_answer_graph_message(
    db_session: Session, assistant_message_id: int
) -> tuple[AnswerGraphRun, ChatMessage] | None:
    run = get_answer_graph_run_by_message(db_session, assistant_message_id)
    if run is None:
        return None
    message = db_session.get(ChatMessage, assistant_message_id)
    if message is None:
        return None
    run.final_answer_sha256 = hashlib.sha256(
        message.message.encode("utf-8")
    ).hexdigest()
    run.status = "FAILED" if message.error else "COMPLETE"
    run.time_finished = datetime.datetime.now(datetime.timezone.utc)
    db_session.commit()
    return run, message


def list_answer_graph_nodes(
    db_session: Session, run_id: UUID, *, offset: int, limit: int
) -> list[AnswerGraphNode]:
    return list(
        db_session.scalars(
            select(AnswerGraphNode)
            .where(AnswerGraphNode.run_id == run_id)
            .order_by(AnswerGraphNode.started_at, AnswerGraphNode.node_id)
            .offset(offset)
            .limit(limit)
        )
    )


def list_answer_graph_edges(
    db_session: Session, run_id: UUID, *, offset: int, limit: int
) -> list[AnswerGraphEdge]:
    return list(
        db_session.scalars(
            select(AnswerGraphEdge)
            .where(AnswerGraphEdge.run_id == run_id)
            .order_by(AnswerGraphEdge.id)
            .offset(offset)
            .limit(limit)
        )
    )


def get_answer_graph_node(
    db_session: Session, run_id: UUID, node_id: str
) -> AnswerGraphNode | None:
    return db_session.get(AnswerGraphNode, (run_id, node_id))


def get_answer_graph_node_payloads(
    db_session: Session, run_id: UUID, node_id: str
) -> dict[str, bytes]:
    return {
        payload.part: payload.ciphertext
        for payload in db_session.scalars(
            select(AnswerGraphPayload).where(
                AnswerGraphPayload.run_id == run_id,
                AnswerGraphPayload.node_id == node_id,
            )
        )
    }


def latest_answer_graph_node_id(db_session: Session, run_id: UUID) -> str | None:
    return db_session.scalar(
        select(AnswerGraphNode.node_id)
        .where(
            AnswerGraphNode.run_id == run_id,
            AnswerGraphNode.kind != "answer",
        )
        .order_by(AnswerGraphNode.ended_at.desc().nulls_last())
        .limit(1)
    )


def delete_answer_graphs_for_session(
    db_session: Session, chat_session_id: UUID
) -> None:
    """Delete graph rows and their database-backed encrypted payloads."""
    run_ids = list(
        db_session.scalars(
            select(AnswerGraphRun.id).where(
                AnswerGraphRun.chat_session_id == chat_session_id,
                AnswerGraphRun.tenant_id == get_current_tenant_id(),
            )
        )
    )
    if not run_ids:
        return
    db_session.execute(delete(AnswerGraphRun).where(AnswerGraphRun.id.in_(run_ids)))
    db_session.commit()


def prune_expired_answer_graphs(
    db_session: Session,
    *,
    batch_size: int = 500,
    max_batches: int = 20,
    max_seconds: float = 30.0,
) -> tuple[int, int]:
    """Prune bounded batches: payloads after 30 days, graph rows after 90."""
    deadline = time.monotonic() + max_seconds
    deleted_runs = 0
    expired_nodes = 0
    for _ in range(max_batches):
        stale, deleted, expired = _prune_expired_answer_graphs_batch(
            db_session, batch_size=batch_size
        )
        deleted_runs += deleted
        expired_nodes += expired
        if stale + deleted + expired == 0 or time.monotonic() >= deadline:
            break
    return deleted_runs, expired_nodes


def _prune_expired_answer_graphs_batch(
    db_session: Session, *, batch_size: int
) -> tuple[int, int, int]:
    now = datetime.datetime.now(datetime.timezone.utc)
    metadata_cutoff = now - datetime.timedelta(days=90)
    payload_cutoff = now - datetime.timedelta(days=30)
    tenant_id = get_current_tenant_id()
    stale_cutoff = now - datetime.timedelta(hours=24)
    stale_runs = list(
        db_session.scalars(
            select(AnswerGraphRun)
            .where(
                AnswerGraphRun.tenant_id == tenant_id,
                AnswerGraphRun.time_created < stale_cutoff,
                AnswerGraphRun.status.in_(("RUNNING", "FINALIZING")),
            )
            .limit(batch_size)
        )
    )
    for run in stale_runs:
        run.status = "INTERRUPTED"
        run.capture_status = "PARTIAL"
        run.capture_error = "Trace or answer finalization did not finish"
        run.time_finished = now
    db_session.commit()
    old_runs = list(
        db_session.scalars(
            select(AnswerGraphRun)
            .where(
                AnswerGraphRun.tenant_id == tenant_id,
                AnswerGraphRun.time_created < metadata_cutoff,
                AnswerGraphRun.status.in_(("COMPLETE", "FAILED", "INTERRUPTED")),
            )
            .order_by(AnswerGraphRun.time_created)
            .limit(batch_size)
        )
    )
    for run in old_runs:
        db_session.delete(run)
    db_session.commit()

    expired_nodes = list(
        db_session.scalars(
            select(AnswerGraphNode)
            .join(AnswerGraphRun, AnswerGraphRun.id == AnswerGraphNode.run_id)
            .where(
                AnswerGraphRun.tenant_id == tenant_id,
                AnswerGraphRun.time_created < payload_cutoff,
                AnswerGraphRun.time_created >= metadata_cutoff,
                AnswerGraphRun.status.in_(("COMPLETE", "FAILED", "INTERRUPTED")),
                AnswerGraphNode.capture_status != "EXPIRED",
            )
            .order_by(AnswerGraphNode.started_at)
            .limit(batch_size)
        )
    )
    expired_run_ids: set[UUID] = set()
    if expired_nodes:
        db_session.execute(
            delete(AnswerGraphPayload).where(
                tuple_(AnswerGraphPayload.run_id, AnswerGraphPayload.node_id).in_(
                    [(node.run_id, node.node_id) for node in expired_nodes]
                )
            )
        )
    for node in expired_nodes:
        node.has_input = False
        node.has_output = False
        node.has_reasoning = False
        node.capture_status = "EXPIRED"
        expired_run_ids.add(node.run_id)
    db_session.commit()
    for run_id in expired_run_ids:
        unexpired = db_session.scalar(
            select(AnswerGraphNode.node_id)
            .where(
                AnswerGraphNode.run_id == run_id,
                AnswerGraphNode.capture_status != "EXPIRED",
            )
            .limit(1)
        )
        if unexpired is None:
            run = db_session.get(AnswerGraphRun, run_id)
            if run is not None:
                run.capture_status = "EXPIRED"
    db_session.commit()
    return len(stale_runs), len(old_runs), len(expired_nodes)
