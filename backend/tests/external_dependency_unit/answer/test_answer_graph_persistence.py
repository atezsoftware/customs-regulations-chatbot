from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from onyx.auth.users import current_user
from onyx.configs.constants import MessageType
from onyx.db.answer_graph import (
    delete_answer_graphs_for_session,
    get_answer_graph_node_payloads,
    get_answer_graph_run_by_message,
    list_answer_graph_edges,
    list_answer_graph_nodes,
    prune_expired_answer_graphs,
)
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import Permission
from onyx.db.models import AnswerGraphNode, AnswerGraphPayload, ChatMessage, ChatSession
from onyx.error_handling.exceptions import register_onyx_exception_handlers
from onyx.server.manage.answer_graph.api import admin_router
from onyx.tracing.answer_graph import (
    AnswerGraphTracingProcessor,
    graph_step,
    load_graph_part,
    record_final_answer_message,
)
from onyx.tracing.framework import set_trace_processors
from onyx.tracing.framework.create import ChatTraceMetadata, trace


def test_persisted_answer_graph_follows_reserved_message(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANSWER_GRAPH_ENCRYPTION_KEY", "test-key-" * 6)
    chat_session = ChatSession(id=uuid4())
    db_session.add(chat_session)
    db_session.flush()
    user_message = ChatMessage(
        chat_session_id=chat_session.id,
        message="What is the rule?",
        token_count=5,
        message_type=MessageType.USER,
    )
    assistant_message = ChatMessage(
        chat_session_id=chat_session.id,
        message="The rule is cited.",
        token_count=7,
        message_type=MessageType.ASSISTANT,
    )
    db_session.add_all((user_message, assistant_message))
    db_session.commit()

    set_trace_processors([AnswerGraphTracingProcessor()])
    try:
        with trace(
            "answer_graph_test",
            metadata=ChatTraceMetadata(
                chat_session_id=str(chat_session.id),
                user_message_id=user_message.id,
                assistant_message_id=assistant_message.id,
                model_name="test-model",
            ).model_dump(),
        ):
            with graph_step(
                "chat.input",
                {"question": user_message.message, "api_key": "should-hide"},
            ) as input_step:
                input_step.output_value = {"ready": True}
            with graph_step("search.bm25", {"query": "rule"}) as search_step:
                search_step.output_value = {"hits": [1, 2], "body": "a" * 70000}
        record_final_answer_message(assistant_message.id)
        db_session.expire_all()
        run = get_answer_graph_run_by_message(db_session, assistant_message.id)
        assert run is not None
        assert run.status == "COMPLETE"
        assert run.capture_status == "COMPLETE"
        assert run.final_answer_sha256 is not None
        nodes = list_answer_graph_nodes(db_session, run.id, offset=0, limit=20)
        assert [node.operation for node in nodes] == [
            "chat.input",
            "search.bm25",
            "answer.delivered",
        ]
        search_node = next(node for node in nodes if node.operation == "search.bm25")
        assert search_node.parent_node_id == input_step.node_id
        input_node = next(node for node in nodes if node.operation == "chat.input")
        assert load_graph_part(
            get_answer_graph_node_payloads(db_session, run.id, input_node.node_id)[
                "input"
            ],
            run_id=run.id,
            node_id=input_node.node_id,
            part="input",
        ) == {"question": "What is the rule?", "api_key": "[REDACTED]"}
        assert not hasattr(search_node, "output_ciphertext")
        large_payload = db_session.scalar(
            select(AnswerGraphPayload.ciphertext).where(
                AnswerGraphPayload.run_id == run.id,
                AnswerGraphPayload.node_id == search_node.node_id,
                AnswerGraphPayload.part == "output",
            )
        )
        assert large_payload is not None and len(large_payload) > 65536
        edges = list_answer_graph_edges(db_session, run.id, offset=0, limit=20)
        assert len(edges) == 1
        assert edges[0].to_node_id == f"answer_{assistant_message.id}"
        app = FastAPI()
        register_onyx_exception_handlers(app)
        app.include_router(admin_router)
        admin = type(
            "Admin",
            (),
            {
                "id": uuid4(),
                "effective_permissions": [Permission.FULL_ADMIN_PANEL_ACCESS.value],
            },
        )()
        app.dependency_overrides[current_user] = lambda: admin
        app.dependency_overrides[get_session] = lambda: db_session
        client = TestClient(app)
        overview = client.get(f"/admin/answer-graphs/by-message/{assistant_message.id}")
        assert overview.status_code == 200
        assert overview.json()["status"] == "COMPLETE"
        assert overview.headers["cache-control"] == "private, no-store"
        detail = client.get(f"/admin/answer-graphs/{run.id}/nodes/{input_node.node_id}")
        assert detail.status_code == 200
        assert detail.json()["input"]["api_key"] == "[REDACTED]"
        search_detail = client.get(
            f"/admin/answer-graphs/{run.id}/nodes/{search_node.node_id}"
        )
        assert search_detail.json()["output"]["body"] == "a" * 70000
        run.time_created = datetime.now(timezone.utc) - timedelta(days=31)
        db_session.commit()
        assert prune_expired_answer_graphs(db_session, batch_size=1) == (0, 3)
        db_session.refresh(run)
        assert run.capture_status == "EXPIRED"
        assert (
            db_session.scalar(
                select(AnswerGraphPayload).where(AnswerGraphPayload.run_id == run.id)
            )
            is None
        )
        expired_detail = client.get(
            f"/admin/answer-graphs/{run.id}/nodes/{input_node.node_id}"
        )
        assert expired_detail.json()["input_state"] == "EXPIRED"
        delete_answer_graphs_for_session(db_session, chat_session.id)
        assert (
            db_session.scalar(
                select(AnswerGraphNode).where(AnswerGraphNode.run_id == run.id)
            )
            is None
        )
    finally:
        set_trace_processors([])
        db_session.execute(
            delete(ChatMessage).where(ChatMessage.chat_session_id == chat_session.id)
        )
        db_session.execute(delete(ChatSession).where(ChatSession.id == chat_session.id))
        db_session.commit()
