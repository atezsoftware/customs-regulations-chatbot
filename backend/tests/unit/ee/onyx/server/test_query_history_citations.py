"""Saved citation visibility and the authenticated admin preview boundary."""

import copy
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ee.onyx.db import query_history_citations as reads
from ee.onyx.server.query_history import api
from ee.onyx.server.query_history.models import MessageSnapshot
from onyx.asv3.models import EvidenceItem
from onyx.auth.users import current_user
from onyx.configs.constants import DocumentSource, MessageType, QueryHistoryType
from onyx.context.search.models import IndexFilters, SearchDoc
from onyx.db.asv3_runs import ASV3_CHECKPOINT_TOOL_ID
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import Permission
from onyx.db.models import ChatMessage, ToolCall, User
from onyx.db.models import SearchDoc as DBSearchDoc
from onyx.error_handling.exceptions import register_onyx_exception_handlers
from onyx.server.documents.models import ChunkInfo
from onyx.tools.tool_implementations.web_search.models import WEB_SEARCH_PREFIX


def message_with_originals(
    texts: tuple[str, ...] = ("  first\n", "second\n\n"),
) -> ChatMessage:
    source_id = str(uuid4())
    message = ChatMessage(
        id=6202,
        chat_session_id=uuid4(),
        message=" ".join(f"Claim [{number}]" for number in range(1, len(texts) + 1)),
        message_type=MessageType.ASSISTANT,
        model_display_name=None,
        time_sent=datetime.now(timezone.utc),
        search_docs=[],
        tool_calls=[],
        chat_message_feedbacks=[],
        citations={},
    )
    records = []
    for number, text in enumerate(texts, 1):
        chunk_id = "rc_" + str(number)
        metadata = {
            "regulatory_chunk_id": chunk_id,
            "regulatory_heading_path": ["Law", f"MADDE {240 + number}"],
            "asv3_source_display_name": "Official law title",
        }
        document = DBSearchDoc(
            id=100 + number,
            document_id=source_id,
            chunk_ind=number,
            semantic_id="stored.md",
            link=None,
            blurb=text,
            doc_metadata=metadata,
            source_type=DocumentSource.FILE,
            is_internet=False,
        )
        item = EvidenceItem(
            source_id=source_id,
            chunk_id=chunk_id,
            text=text,
            search_doc=SearchDoc(
                document_id=source_id,
                chunk_ind=number,
                semantic_identifier="stored.md",
                blurb=text,
                source_type=DocumentSource.FILE,
                boost=0,
                hidden=False,
                metadata=metadata,
                match_highlights=[],
            ),
        )
        message.search_docs.append(document)
        assert message.citations is not None
        message.citations[number] = document.id
        records.append({"citation": number, "item": item.model_dump(mode="json")})
    checkpoint = {
        "version": 1,
        "scope": IndexFilters(access_control_list=None).model_dump(mode="json"),
        "evidence": {"version": 1, "records": records},
        "research_profile": "experimental",
        "asv3_workflow_variant": "asv3_tuned",
    }
    message.tool_calls = [
        ToolCall(
            tool_id=ASV3_CHECKPOINT_TOOL_ID, tool_call_response=json.dumps(checkpoint)
        )
    ]
    return message


def test_every_citation_survives_same_file_duplicates_missing_rows_and_notice_catalogue() -> (
    None
):
    message = message_with_originals(tuple(f"Original {n}" for n in range(68)))
    message.message = "First [68], next [1]. " + message.message + " Missing [999]."
    assert message.citations is not None
    message.citations[70] = message.search_docs[0].id
    notice = DBSearchDoc(
        id=500, document_id="notice:" + str(uuid4()), semantic_id="Catalogue only"
    )
    message.search_docs.append(notice)
    assert message.tool_calls
    before = copy.deepcopy(message.tool_calls[0].tool_call_response)
    snapshot = MessageSnapshot.build(message)
    assert [row.citation_number for row in snapshot.citation_info][:2] == [68, 1]
    assert len(snapshot.citation_info) == 69
    assert snapshot.unresolved_citations == [999]
    assert all(row.text is None for row in snapshot.citation_info)
    assert snapshot.citation_info[1].reference == "Law > MADDE 241"
    assert {row.document_id for row in snapshot.citation_info} == {
        message.search_docs[0].document_id
    }
    assert snapshot.citation_info[-1].citation_number == 70
    assert snapshot.citation_info[-1].resolution_status == "unavailable"
    assert message.tool_calls[0].tool_call_response == before


@pytest.mark.parametrize("prefix", ["notice:", "catalog:"])
def test_stale_one_original_does_not_hide_the_other_and_notice_is_never_evidence(
    prefix: str,
) -> None:
    message = message_with_originals()
    message.search_docs[0].chunk_ind = 999
    result = MessageSnapshot.build(message)
    assert [row.resolution_status for row in result.citation_info] == [
        "unavailable",
        "retained",
    ]
    assert result.citation_info[0].preview_url is None
    message.search_docs[0].document_id = prefix + str(uuid4())
    result = MessageSnapshot.build(message)
    assert [row.citation_number for row in result.citation_info] == [2]
    assert result.unresolved_citations == [1]
    message.message_type = MessageType.USER
    assert MessageSnapshot.build(message).citation_info == []
    assert MessageSnapshot.build(message).unresolved_citations == []


def test_legacy_targets_are_explicitly_current_external_or_unavailable() -> None:
    message = message_with_originals(("first", "second", "third"))
    message.tool_calls = []
    message.search_docs[1].document_id = WEB_SEARCH_PREFIX + "https://example.test/law"
    message.search_docs[1].source_type = DocumentSource.WEB
    message.search_docs[1].link = "https://example.test/law"
    message.search_docs[2].chunk_ind = -3
    result = MessageSnapshot.build(message)
    assert [row.resolution_status for row in result.citation_info] == [
        "current",
        "external",
        "unavailable",
    ]
    assert result.citation_info[1].link == "https://example.test/law"
    assert result.citation_info[1].preview_url is None
    assert result.documents[0].chunk_ind == 1
    assert result.documents[0].metadata == {
        "regulatory_chunk_id": "rc_1",
        "regulatory_heading_path": ["Law", "MADDE 241"],
        "asv3_source_display_name": "Official law title",
    }


def test_missing_saved_documents_are_loaded_once_without_mutating_relationships() -> (
    None
):
    message = message_with_originals()
    original = message.search_docs.pop()
    session = MagicMock()
    session.scalars.return_value = [original]
    documents = reads.citation_documents(session, [message])
    assert set(documents) == {101, 102}
    assert len(message.search_docs) == 1
    session.scalars.assert_called_once()
    session.commit.assert_not_called()
    session.execute.assert_not_called()


def test_checkpoint_is_decoded_once_for_all_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = message_with_originals(tuple(f"Text {n}" for n in range(68)))
    decoder = MagicMock(wraps=reads.decode_asv3_checkpoint)
    monkeypatch.setattr(reads, "decode_asv3_checkpoint", decoder)
    assert all(
        row.resolution_status == "retained"
        for row in MessageSnapshot.build(message).citation_info
    )
    decoder.assert_called_once()


@pytest.mark.parametrize("heading", ["null", '["Law", null]'])
def test_malformed_optional_heading_does_not_break_history(heading: str) -> None:
    message = message_with_originals()
    message.search_docs[0].doc_metadata = {
        "regulatory_heading_path": json.loads(heading)
    }
    snapshot = MessageSnapshot.build(message)
    assert snapshot.citation_info[0].reference is None
    assert snapshot.documents[0].metadata == {}


def test_existing_string_heading_is_preserved_without_guessing_article_identity() -> (
    None
):
    message = message_with_originals()
    message.search_docs[0].doc_metadata["regulatory_heading_path"] = "Law > MADDE 241"
    snapshot = MessageSnapshot.build(message)
    assert snapshot.citation_info[0].reference == "Law > MADDE 241"
    assert (
        snapshot.documents[0].metadata["regulatory_heading_path"] == "Law > MADDE 241"
    )


@pytest.mark.parametrize(
    "payload", [None, "", '{"encoding":"zlib-base64","data":"bm90LXpsaWI="}']
)
def test_unavailable_checkpoint_never_breaks_other_saved_targets(
    payload: str | None,
) -> None:
    message = message_with_originals()
    assert message.tool_calls
    message.tool_calls[0].tool_call_response = payload  # ty: ignore[invalid-assignment]
    snapshot = MessageSnapshot.build(message)
    assert [row.resolution_status for row in snapshot.citation_info] == [
        "unavailable",
        "unavailable",
    ]
    assert snapshot.unresolved_citations == []


@pytest.fixture
def preview_client(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, MagicMock, MagicMock, User]:
    app = FastAPI()
    app.include_router(api.router, prefix="/api")
    register_onyx_exception_handlers(app)
    user = User(
        id=uuid4(), effective_permissions=[Permission.FULL_ADMIN_PANEL_ACCESS.value]
    )
    session = MagicMock()
    app.dependency_overrides[current_user] = lambda: user
    app.dependency_overrides[get_session] = lambda: session
    monkeypatch.setattr(
        api, "ensure_query_history_is_enabled", lambda **_: QueryHistoryType.NORMAL
    )
    validator = MagicMock()
    monkeypatch.setattr(
        api,
        "CorpusBroker",
        MagicMock(return_value=MagicMock(revalidate_evidence=validator)),
    )
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = list
    monkeypatch.setattr(api, "get_tokenizer", lambda *_: tokenizer)
    return TestClient(app), session, validator, user


@pytest.mark.parametrize("text", ["", " \n\t ", "Exact source\n  tail\n"])
def test_admin_http_returns_exact_retained_text_and_checks_session_binding(
    preview_client: tuple[TestClient, MagicMock, MagicMock, User],
    text: str,
) -> None:
    client, session, validator, user = preview_client
    message = message_with_originals((text,))
    session.scalar.return_value = message
    response = client.get(
        f"/api/admin/chat-session-history/{message.chat_session_id}/citation/{message.id}/1"
    )
    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    assert response.json()["content"] == text
    validator.assert_called_once()
    assert validator.call_args.args[0][0].text == text
    statement = session.scalar.call_args.args[0]
    assert set(statement.compile().params.values()) == {
        message.id,
        message.chat_session_id,
    }
    session.commit.assert_not_called()
    assert user.id != message.chat_session_id


def test_admin_http_permission_gate_and_revoked_source_never_fall_back(
    preview_client: tuple[TestClient, MagicMock, MagicMock, User],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, session, validator, user = preview_client
    message = message_with_originals()
    session.scalar.return_value = message
    path = f"/api/admin/chat-session-history/{message.chat_session_id}/citation/{message.id}/1"
    user.effective_permissions = [Permission.BASIC_ACCESS.value]
    assert client.get(path).status_code == 403
    session.scalar.assert_not_called()
    user.effective_permissions = [Permission.FULL_ADMIN_PANEL_ACCESS.value]
    validator.side_effect = PermissionError("Source access revoked")
    current_reader = MagicMock()
    monkeypatch.setattr(api, "get_chunk_info", current_reader)
    assert client.get(path).status_code == 404
    current_reader.assert_not_called()
    session.scalar.return_value = None
    assert client.get(path).status_code == 404


def test_admin_http_native_original_preserves_negative_locator(
    preview_client: tuple[TestClient, MagicMock, MagicMock, User],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, session, validator, _ = preview_client
    message = message_with_originals(("Native page\n  original",))
    assert message.tool_calls
    checkpoint = json.loads(message.tool_calls[0].tool_call_response)
    item = checkpoint["evidence"]["records"][0]["item"]
    item["chunk_id"] = None
    item["metadata"] = {
        "derived": True,
        "source_sha256": "a" * 64,
        "locator": {"page": 2},
    }
    item["search_doc"]["chunk_ind"] = -1
    message.search_docs[0].chunk_ind = -1
    message.search_docs[0].doc_metadata = {}
    message.tool_calls[0].tool_call_response = json.dumps(checkpoint)
    session.scalar.return_value = message
    reader = MagicMock()
    monkeypatch.setattr(api, "get_chunk_info", reader)
    result = client.get(
        f"/api/admin/chat-session-history/{message.chat_session_id}/citation/{message.id}/1"
    )
    assert result.status_code == 200 and result.json()["content"] == item["text"]
    assert validator.call_args.args[0][0].chunk_id is None
    reader.assert_not_called()


def test_admin_http_publication_gate_and_legacy_current_reader(
    preview_client: tuple[TestClient, MagicMock, MagicMock, User],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, session, validator, user = preview_client
    message = message_with_originals()
    message.tool_calls = []
    session.scalar.return_value = message
    reader = MagicMock(
        return_value=ChunkInfo(content="Current indexed chunk", num_tokens=3)
    )
    monkeypatch.setattr(api, "get_chunk_info", reader)
    path = f"/api/admin/chat-session-history/{message.chat_session_id}/citation/{message.id}/1"
    assert client.get(path).json()["content"] == "Current indexed chunk"
    reader.assert_called_once_with(message.search_docs[0].document_id, 1, user, session)
    validator.assert_not_called()
    monkeypatch.setattr(reads, "message_publication_available", lambda _: False)
    reader.reset_mock()
    assert client.get(path).status_code == 404
    reader.assert_not_called()
    assert reads.history_citation_info(
        message, {doc.id: doc for doc in message.search_docs}
    ) == ([], [1, 2])
