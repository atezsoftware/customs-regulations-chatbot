from datetime import datetime, timedelta, timezone
from io import BytesIO
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from onyx.auth.users import current_user
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import Permission
from onyx.db.models import AnswerGraphEdge, AnswerGraphNode, AnswerGraphRun
from onyx.error_handling.exceptions import register_onyx_exception_handlers
from onyx.server.manage.answer_graph import api, markdown_report


def test_markdown_contains_parallel_map_and_full_captured_details() -> None:
    started = datetime(2026, 9, 30, tzinfo=timezone.utc)
    root = AnswerGraphNode(
        node_id="root",
        parent_node_id=None,
        kind="step",
        operation="chat.input",
        status="COMPLETE",
        capture_status="COMPLETE",
        started_at=started,
        ended_at=started + timedelta(seconds=1),
        attributes={"agent": "Chat agent"},
    )
    agents = [
        AnswerGraphNode(
            node_id=f"agent-{number}",
            parent_node_id="root",
            kind="function",
            operation="research_agent",
            status="COMPLETE",
            capture_status="COMPLETE",
            started_at=started + timedelta(seconds=2),
            ended_at=started + timedelta(seconds=5),
            attributes={"agent": f"Research agent {number}"},
        )
        for number in (1, 2)
    ]
    search = AnswerGraphNode(
        node_id="search",
        parent_node_id="agent-1",
        kind="step",
        operation="search.bm25",
        status="COMPLETE",
        capture_status="COMPLETE",
        started_at=started + timedelta(seconds=3),
        ended_at=started + timedelta(seconds=4),
        attributes={},
    )
    payload = "a" * 70_000 + "END_OF_CHUNK"
    output = BytesIO()

    def parts(node: AnswerGraphNode) -> dict[str, tuple[object, str]]:
        return {
            "input": ({"query": "tarife", "api_key": "[REDACTED]"}, "CAPTURED"),
            "output": (
                {"chunks": [{"content": "Türkçe gümrük metni " + payload}]},
                "CAPTURED",
            )
            if node.node_id == "search"
            else (None, "NOT_RETURNED"),
            "reasoning": ("Captured provider reasoning", "CAPTURED")
            if node.node_id == "search"
            else (None, "NOT_RETURNED"),
        }

    markdown_report.write_answer_graph_markdown(
        AnswerGraphRun(
            assistant_message_id=2657, status="COMPLETE", capture_status="COMPLETE"
        ),
        [search, agents[1], root, agents[0]],
        [AnswerGraphEdge(from_node_id="agent-2", to_node_id="search", kind="data")],
        parts,
        output,
    )
    text = output.getvalue().decode("utf-8")

    assert "```mermaid\nflowchart LR" in text
    assert 'subgraph lane_2["Research agent 1"]' in text
    assert 'subgraph lane_3["Research agent 2"]' in text
    assert "n003 ==>|data| n004" in text
    assert "[step 002](#step-002)" in text
    assert "Türkçe gümrük metni" in text
    assert "END_OF_CHUNK" in text
    assert "Captured provider reasoning" in text
    assert "[REDACTED]" in text
    assert "### Output — `NOT_RETURNED`" in text


def test_markdown_rejects_oversized_export_without_silent_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(markdown_report, "_MAX_EXPORT_BYTES", 100)
    with pytest.raises(markdown_report.MarkdownExportTooLarge):
        markdown_report.write_answer_graph_markdown(
            AnswerGraphRun(
                assistant_message_id=1, status="COMPLETE", capture_status="COMPLETE"
            ),
            [],
            [],
            lambda _node: {},
            BytesIO(),
        )


def test_admin_markdown_endpoint_streams_complete_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = datetime(2026, 9, 30, tzinfo=timezone.utc)
    node = AnswerGraphNode(
        node_id="root",
        parent_node_id=None,
        kind="step",
        operation="chat.input",
        status="COMPLETE",
        capture_status="COMPLETE",
        started_at=started,
        attributes={"agent": "Chat agent"},
    )
    run = AnswerGraphRun(
        id=uuid4(),
        assistant_message_id=2657,
        status="COMPLETE",
        capture_status="COMPLETE",
    )
    monkeypatch.setattr(api, "_visible_run", lambda _db, _id: run)
    monkeypatch.setattr(
        api,
        "list_answer_graph_nodes",
        lambda _db, _id, **_kwargs: [node],
    )
    monkeypatch.setattr(
        api,
        "list_answer_graph_edges",
        lambda _db, _id, **_kwargs: [],
    )
    monkeypatch.setattr(api, "get_answer_graph_node_payloads", lambda *_: {})
    monkeypatch.setattr(api, "record_answer_graph_access", lambda *_, **__: None)
    app = FastAPI()
    register_onyx_exception_handlers(app)
    app.include_router(api.admin_router)
    admin = type(
        "Admin",
        (),
        {
            "id": uuid4(),
            "effective_permissions": [Permission.FULL_ADMIN_PANEL_ACCESS.value],
        },
    )()
    app.dependency_overrides[current_user] = lambda: admin
    app.dependency_overrides[get_session] = lambda: object()

    response = TestClient(app).get(f"/admin/answer-graphs/{run.id}/markdown")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["content-disposition"].endswith('2657.md"')
    assert "```mermaid\nflowchart LR" in response.text
    assert "## Step 001 — chat.input" in response.text
    assert "### Input — `NOT_RETURNED`" in response.text
