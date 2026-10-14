from datetime import datetime, timezone
from typing import cast
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.configs.constants import DocumentSource, MessageType
from onyx.db.chat import RESERVED_ASSISTANT_MESSAGE
from onyx.db.external_chat_share_messages import get_external_shared_messages
from onyx.db.models import ChatMessage, ChatSession, SearchDoc
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError


def _session() -> ChatSession:
    timestamp = datetime.now(timezone.utc)
    return ChatSession(
        id=uuid4(),
        description="Shared conversation",
        time_created=timestamp,
        time_updated=timestamp,
        external_user_name="Private name",
        api_key_id=42,
        api_key_name="Private credential",
        api_key_display="Private key display",
    )


def _message(
    session_id: UUID,
    message_id: int,
    role: MessageType,
    content: str,
    parent: int | None,
    child: int | None = None,
) -> ChatMessage:
    return ChatMessage(
        id=message_id,
        chat_session_id=session_id,
        message_type=role,
        message=content,
        parent_message_id=parent,
        latest_child_message_id=child,
        time_sent=datetime.now(timezone.utc),
        search_docs=[],
        citations=None,
        reasoning_tokens="Private reasoning",
        error=None,
        publication_read=None,
    )


def _database(messages: list[ChatMessage]) -> MagicMock:
    database = MagicMock(spec=Session)
    database.scalars.return_value.all.return_value = messages
    return database


def _conversation(session: ChatSession) -> list[ChatMessage]:
    return [
        _message(session.id, 1, MessageType.SYSTEM, "Private system prompt", None, 2),
        _message(session.id, 2, MessageType.USER, "Çağrı Şahin'in sorusu", 1, 3),
        _message(session.id, 3, MessageType.ASSISTANT, "Selected answer", 2),
    ]


def test_projection_follows_selected_branch_and_excludes_private_fields() -> None:
    session = _session()
    messages = _conversation(session)
    messages.append(_message(session.id, 4, MessageType.ASSISTANT, "Old answer", 2))
    database = _database(list(reversed(messages)))

    response = get_external_shared_messages(database, session)

    assert [message.content for message in response.messages] == [
        "Çağrı Şahin'in sorusu",
        "Selected answer",
    ]
    assert response.session.messageCount == 2
    assert response.session.isShared is True
    assert set(response.model_dump()) == {"session", "messages"}
    assert set(response.session.model_dump()) == {
        "id",
        "title",
        "createdAt",
        "updatedAt",
        "messageCount",
        "isShared",
    }
    assert set(response.messages[1].model_dump()) == {
        "id",
        "sessionId",
        "role",
        "content",
        "timestamp",
        "citations",
    }
    assert "Private" not in response.model_dump_json()
    assert "Old answer" not in response.model_dump_json()
    database.commit.assert_not_called()
    database.flush.assert_not_called()
    database.add.assert_not_called()
    query = database.scalars.call_args.args[0]
    assert query.compile().params["chat_session_id_1"] == session.id
    assert "reasoning_tokens" not in str(query)


def test_live_projection_includes_new_messages_and_hides_internal_chain_nodes() -> None:
    session = _session()
    messages = _conversation(session)
    database = _database(messages)
    assert get_external_shared_messages(database, session).session.messageCount == 2

    messages[2].latest_child_message_id = 4
    messages.extend(
        [
            _message(
                session.id, 4, MessageType.USER_REMINDER, "Private reminder", 3, 5
            ),
            _message(
                session.id, 5, MessageType.TOOL_CALL_RESPONSE, "Private tool", 4, 6
            ),
            _message(session.id, 6, MessageType.USER, "New question", 5, 7),
            _message(session.id, 7, MessageType.ASSISTANT, "New answer", 6),
        ]
    )
    response = get_external_shared_messages(database, session)
    assert [message.content for message in response.messages] == [
        "Çağrı Şahin'in sorusu",
        "Selected answer",
        "New question",
        "New answer",
    ]
    assert response.session.messageCount == 4


def test_empty_session_projection_does_not_create_root() -> None:
    database = _database([])
    response = get_external_shared_messages(database, _session())
    assert response.messages == []
    assert response.session.messageCount == 0
    database.add.assert_not_called()
    database.commit.assert_not_called()


@pytest.mark.parametrize(
    "problem",
    ["missing_child", "wrong_parent", "cycle", "multiple_roots", "foreign_session"],
)
def test_invalid_selected_branch_fails_closed(problem: str) -> None:
    session = _session()
    messages = _conversation(session)
    if problem == "missing_child":
        messages[1].latest_child_message_id = 999
    elif problem == "wrong_parent":
        messages[2].parent_message_id = 1
    elif problem == "cycle":
        messages[2].latest_child_message_id = 2
    elif problem == "multiple_roots":
        messages.append(_message(session.id, 4, MessageType.USER, "Other root", None))
    else:
        messages[2].chat_session_id = uuid4()

    with pytest.raises(OnyxError) as raised:
        get_external_shared_messages(_database(messages), session)
    assert raised.value.error_code == OnyxErrorCode.CONFLICT
    assert raised.value.headers == {"Cache-Control": "private, no-store"}
    assert "Selected answer" not in raised.value.detail


@pytest.mark.parametrize("hidden_kind", ["reserved", "error", "unavailable", "draft"])
def test_unfinished_or_unavailable_answers_do_not_expose_content_or_citations(
    hidden_kind: str,
) -> None:
    session = _session()
    messages = _conversation(session)
    answer = messages[2]
    answer.search_docs = [
        SearchDoc(id=91, semantic_id="Private source", link="/api/chat/file/private")
    ]
    answer.citations = {1: 91}
    if hidden_kind == "reserved":
        answer.message = RESERVED_ASSISTANT_MESSAGE
    elif hidden_kind == "error":
        answer.message = "Private upstream error"
        answer.error = "Private upstream error"
    elif hidden_kind == "draft":
        answer.publication_read = {
            "evidence": {
                "observation": {
                    "scope": {
                        "tenant_id": "public",
                        "environment": "test",
                        "database_identity": "test",
                    },
                    "committed_epoch": 0,
                },
                "user_file_ids": [],
            },
            "generation_done": False,
            "finalized": False,
        }
    with (
        patch(
            "onyx.db.external_chat_share_messages.message_publication_available",
            side_effect=lambda message: (
                hidden_kind != "unavailable" or message is not answer
            ),
        ),
    ):
        response = get_external_shared_messages(_database(messages), session)
    assert [message.role for message in response.messages] == ["user"]
    assert response.session.messageCount == 1
    assert "Private" not in response.model_dump_json()


def test_malformed_publication_evidence_hides_answer_without_writes() -> None:
    session = _session()
    messages = _conversation(session)
    messages[2].publication_read = {"malformed": True}
    database = _database(messages)
    response = get_external_shared_messages(database, session)
    assert [message.role for message in response.messages] == ["user"]
    database.commit.assert_not_called()


def test_publication_change_during_projection_discards_answer() -> None:
    session = _session()
    messages = _conversation(session)
    with patch(
        "onyx.db.external_chat_share_messages.message_publication_available",
        side_effect=[True, True, True, False],
    ):
        response = get_external_shared_messages(_database(messages), session)
    assert [message.role for message in response.messages] == ["user"]


@pytest.mark.parametrize(
    ("link", "expected_url"),
    [
        (
            "https://example.com/law?article=1#text",
            "https://example.com/law?article=1#text",
        ),
        ("http://example.com/law", "http://example.com/law"),
        ("/api/asv3/citation/3/1", None),
        ("https://example.com/api/chat/file/private", None),
        ("javascript:alert(1)", None),
        ("https://user:secret@example.com/law", None),
        ("http://127.0.0.1/law", None),
        ("https://localhost/law", None),
        ("https://localhost./law", None),
        ("http://127.1/law", None),
        ("https://example.com/\nprivate", None),
    ],
)
def test_citations_expose_only_bound_metadata_and_safe_internet_links(
    link: str,
    expected_url: str | None,
) -> None:
    session = _session()
    messages = _conversation(session)
    answer = messages[2]
    answer.citations = {1: 91, 2: 92, 3: 999}
    answer.search_docs = [
        SearchDoc(
            id=91,
            semantic_id="Public cited law",
            source_type=DocumentSource.WEB,
            is_internet=True,
            link=link,
            blurb="Private unprojected snippet",
        ),
        SearchDoc(
            id=92,
            semantic_id="Attached cited law",
            source_type=DocumentSource.USER_FILE,
            is_internet=True,
            link="https://example.com/attached-file-id",
        ),
        SearchDoc(id=93, semantic_id="Private uncited document"),
    ]
    response = get_external_shared_messages(_database(messages), session)
    assert [citation.model_dump() for citation in response.messages[1].citations] == [
        {"number": 1, "title": "Public cited law", "url": expected_url},
        {"number": 2, "title": "Attached cited law", "url": None},
    ]
    assert "Private" not in response.model_dump_json()


def test_jsonb_citation_keys_sort_numerically_and_skip_invalid_bindings() -> None:
    session = _session()
    messages = _conversation(session)
    answer = messages[2]
    # JSONB returns string keys despite the ORM's integer-key annotation.
    answer.citations = cast(
        dict[int, int],
        {
            "10": 91,
            "2": 92,
            "invalid": 91,
            "-1": 91,
            "0": 91,
            "3": True,
            "4": "91",
            "5": 91.0,
            "6": None,
            "7": [91],
            "8": {"id": 91},
            "9": 0,
            "11": 91,
            "011": 92,
            "12": 999,
        },
    )
    answer.search_docs = [
        SearchDoc(id=91, semantic_id="Tenth source"),
        SearchDoc(id=92, semantic_id="Second source"),
    ]
    response = get_external_shared_messages(_database(messages), session)
    assert [citation.model_dump() for citation in response.messages[1].citations] == [
        {"number": 2, "title": "Second source", "url": None},
        {"number": 10, "title": "Tenth source", "url": None},
    ]
