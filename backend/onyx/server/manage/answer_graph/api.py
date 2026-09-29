"""Admin-only, tenant-scoped access to recorded answer execution graphs."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy.orm import Session

from onyx.auth.permissions import require_permission
from onyx.db.answer_graph import (
    answer_graph_message_exists,
    get_answer_graph_node,
    get_answer_graph_node_payloads,
    get_answer_graph_run,
    get_answer_graph_run_by_message,
    list_answer_graph_edges,
    list_answer_graph_nodes,
    record_answer_graph_access,
)
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import Permission
from onyx.db.models import AnswerGraphNode, AnswerGraphRun, User
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.server.manage.answer_graph.models import (
    AnswerGraphEdgePage,
    AnswerGraphEdgeView,
    AnswerGraphNodeDetail,
    AnswerGraphNodePage,
    AnswerGraphNodeView,
    AnswerGraphRunView,
)
from onyx.tracing.answer_graph import load_graph_part
from onyx.utils.logger import setup_logger

logger = setup_logger(__name__)

admin_router = APIRouter(prefix="/admin/answer-graphs")


def _no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "private, no-store"


def _run_view(run: AnswerGraphRun) -> AnswerGraphRunView:
    return AnswerGraphRunView(
        run_id=run.id,
        assistant_message_id=run.assistant_message_id,
        status=run.status,
        capture_status=run.capture_status,
        capture_error=run.capture_error,
        model_name=run.model_name,
        started_at=run.time_created,
        finished_at=run.time_finished,
        final_answer_sha256=run.final_answer_sha256,
    )


def _node_view(node: AnswerGraphNode) -> AnswerGraphNodeView:
    return AnswerGraphNodeView(
        node_id=node.node_id,
        parent_node_id=node.parent_node_id,
        kind=node.kind,
        operation=node.operation,
        status=node.status,
        capture_status=node.capture_status,
        started_at=node.started_at,
        ended_at=node.ended_at,
        attributes=node.attributes,
        has_input=node.has_input,
        has_output=node.has_output,
        has_reasoning=node.has_reasoning,
        error=node.error,
    )


def _visible_run(db_session: Session, run_id: UUID) -> AnswerGraphRun:
    run = get_answer_graph_run(db_session, run_id)
    if run is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Answer graph not found")
    return run


@admin_router.get("/by-message/{assistant_message_id}")
def get_graph_by_message(
    assistant_message_id: int,
    response: Response,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> AnswerGraphRunView:
    _no_store(response)
    if not answer_graph_message_exists(db_session, assistant_message_id):
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Answer graph not found")
    run = get_answer_graph_run_by_message(db_session, assistant_message_id)
    if run is None:
        return AnswerGraphRunView(
            run_id=None,
            assistant_message_id=assistant_message_id,
            status="NO_TRACE",
        )
    record_answer_graph_access(
        db_session, run_id=run.id, user_id=user.id, action="overview"
    )
    return _run_view(run)


@admin_router.get("/{run_id}/nodes")
def get_graph_nodes(
    run_id: UUID,
    response: Response,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=500),
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> AnswerGraphNodePage:
    _no_store(response)
    _visible_run(db_session, run_id)
    rows = list_answer_graph_nodes(db_session, run_id, offset=offset, limit=limit + 1)
    record_answer_graph_access(
        db_session, run_id=run_id, user_id=user.id, action="nodes_list"
    )
    has_more = len(rows) > limit
    return AnswerGraphNodePage(
        nodes=[_node_view(node) for node in rows[:limit]],
        next_offset=offset + limit if has_more else None,
    )


@admin_router.get("/{run_id}/edges")
def get_graph_edges(
    run_id: UUID,
    response: Response,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=500),
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> AnswerGraphEdgePage:
    _no_store(response)
    _visible_run(db_session, run_id)
    rows = list_answer_graph_edges(db_session, run_id, offset=offset, limit=limit + 1)
    record_answer_graph_access(
        db_session, run_id=run_id, user_id=user.id, action="edges_list"
    )
    has_more = len(rows) > limit
    return AnswerGraphEdgePage(
        edges=[
            AnswerGraphEdgeView(
                from_node_id=edge.from_node_id,
                to_node_id=edge.to_node_id,
                kind=edge.kind,
            )
            for edge in rows[:limit]
        ],
        next_offset=offset + limit if has_more else None,
    )


def _part(
    ciphertext: bytes | None,
    *,
    run_id: UUID,
    node_id: str,
    part: str,
    capture_status: str,
) -> tuple[object, str]:
    if ciphertext is None:
        if capture_status == "EXPIRED":
            return None, "EXPIRED"
        if capture_status == "PARTIAL":
            return None, "UNAVAILABLE"
        return None, "NOT_RETURNED"
    try:
        return (
            load_graph_part(ciphertext, run_id=run_id, node_id=node_id, part=part),
            "CAPTURED",
        )
    except Exception:
        logger.exception("Answer graph payload unavailable")
        return None, "UNAVAILABLE"


@admin_router.get("/{run_id}/nodes/{node_id}")
def get_graph_node_detail(
    run_id: UUID,
    node_id: str,
    response: Response,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> AnswerGraphNodeDetail:
    _no_store(response)
    _visible_run(db_session, run_id)
    node = get_answer_graph_node(db_session, run_id, node_id)
    if node is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Answer graph node not found")
    payloads = get_answer_graph_node_payloads(db_session, run_id, node_id)
    record_answer_graph_access(
        db_session,
        run_id=run_id,
        user_id=user.id,
        action="node_detail",
        node_id=node_id,
    )
    logger.info(
        "Answer graph node read by admin user=%s run=%s node=%s",
        user.id,
        run_id,
        node_id,
    )
    input_value, input_state = _part(
        payloads.get("input"),
        run_id=run_id,
        node_id=node_id,
        part="input",
        capture_status=node.capture_status,
    )
    output_value, output_state = _part(
        payloads.get("output"),
        run_id=run_id,
        node_id=node_id,
        part="output",
        capture_status=node.capture_status,
    )
    reasoning, reasoning_state = _part(
        payloads.get("reasoning"),
        run_id=run_id,
        node_id=node_id,
        part="reasoning",
        capture_status=node.capture_status,
    )
    return AnswerGraphNodeDetail(
        node=_node_view(node),
        input=input_value,
        output=output_value,
        reasoning=reasoning,
        input_state=input_state,
        output_state=output_state,
        reasoning_state=reasoning_state,
    )
