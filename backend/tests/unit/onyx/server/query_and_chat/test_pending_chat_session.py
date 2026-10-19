import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from onyx.configs.constants import DocumentSource, MessageType
from onyx.db import regulatory_chat_reads
from onyx.db.asv3_runs import ASV3_CHECKPOINT_TOOL_ID
from onyx.db.chat import RESERVED_ASSISTANT_MESSAGE
from onyx.db.models import (
    ChatMessage,
    ChatSession,
    ChatSessionSharedStatus,
    SearchDoc,
    Tool,
    ToolCall,
    User,
)
from onyx.regulatory.publication_reads import PublicationReadChanged
from onyx.server.query_and_chat import chat_backend
from onyx.server.query_and_chat.models import ChatSessionDetailResponse
from onyx.server.query_and_chat.streaming_models import (
    AgentResponseDelta,
    ASv3Progress,
    CitationInfo,
    OverallStop,
    Packet,
    ReasoningDelta,
    SearchToolQueriesDelta,
)

_SESSION_ID = UUID("9d0eba5a-7bcf-4f30-b445-f127851ad1f9")
_USER_ID = UUID("b010d672-dd03-4609-9d63-39d08fadcd6c")
_NOW = datetime(2026, 10, 19, tzinfo=timezone.utc)


def _message(
    message_id: int,
    *,
    message: str = RESERVED_ASSISTANT_MESSAGE,
    message_type: MessageType = MessageType.ASSISTANT,
    parent_message_id: int | None = 10,
    error: str | None = None,
    reasoning: str | None = None,
    tool_calls: list[ToolCall] | None = None,
    search_docs: list[SearchDoc] | None = None,
) -> ChatMessage:
    return cast(
        ChatMessage,
        SimpleNamespace(
            id=message_id,
            chat_session_id=_SESSION_ID,
            parent_message_id=parent_message_id,
            latest_child_message_id=None,
            message=message,
            message_type=message_type,
            error=error,
            reasoning_tokens=reasoning,
            tool_calls=tool_calls or [],
            search_docs=search_docs or [],
            citations={1: search_docs[0].id} if search_docs else {},
            chat_message_feedbacks=[],
            time_sent=_NOW,
            files=[],
            processing_duration_seconds=None,
            preferred_response_id=None,
            model_display_name="Test model",
            publication_read=None,
        ),
    )


def _source() -> SearchDoc:
    return cast(
        SearchDoc,
        SimpleNamespace(
            id=70,
            document_id="customs-law",
            chunk_ind=46,
            semantic_id="Customs Law, Article 46",
            score=1.0,
            link=None,
            blurb="Saved source passage",
            source_type=DocumentSource.FILE,
            boost=0,
            hidden=False,
            doc_metadata={},
            match_highlights=[],
            relevance_explanation=None,
            is_relevant=True,
            updated_at=None,
            primary_owners=None,
            secondary_owners=None,
            is_internet=False,
        ),
    )


def _get_session(
    monkeypatch: pytest.MonkeyPatch,
    messages: list[ChatMessage],
    *,
    run_id: int | None,
    cache_fails: bool = False,
    publication_denied: bool = False,
    invalidate_on_finalize: bool = False,
) -> ChatSessionDetailResponse:
    db_session = MagicMock()
    session = cast(
        ChatSession,
        SimpleNamespace(
            id=_SESSION_ID,
            user_id=_USER_ID,
            user=SimpleNamespace(personal_name="Test user"),
            description="Pending response",
            persona_id=0,
            persona=SimpleNamespace(name="Test agent", icon_name=None),
            current_alternate_model=None,
            temperature_override=None,
            reasoning_effort_override=None,
            time_created=_NOW,
            shared_status=ChatSessionSharedStatus.PRIVATE,
            deleted=False,
        ),
    )
    monkeypatch.setattr(
        chat_backend, "get_chat_session_by_id", MagicMock(return_value=session)
    )
    monkeypatch.setattr(
        chat_backend, "get_chat_messages_by_session", MagicMock(return_value=messages)
    )
    monkeypatch.setattr(chat_backend, "get_response_usage", MagicMock(return_value={}))
    monkeypatch.setattr(chat_backend, "get_cache_backend", MagicMock())
    monkeypatch.setattr(
        chat_backend,
        "get_processing_run_id",
        MagicMock(
            return_value=run_id,
            side_effect=RuntimeError("Cache unavailable") if cache_fails else None,
        ),
    )
    search_tool = cast(Tool, SimpleNamespace(id=71, in_code_tool_id="SearchTool"))
    monkeypatch.setattr(
        chat_backend, "get_tools_by_ids", MagicMock(return_value=[search_tool])
    )
    monkeypatch.setattr(
        chat_backend, "get_session_feedback", MagicMock(return_value=None)
    )

    available = not publication_denied

    def publication_available(_message: ChatMessage) -> bool:
        return available

    def finalize_publication(_message_id: int) -> bool:
        nonlocal available
        available = False
        return False

    monkeypatch.setattr(
        regulatory_chat_reads, "message_publication_available", publication_available
    )
    monkeypatch.setattr(
        regulatory_chat_reads,
        "message_publication_read_needs_finalization",
        MagicMock(return_value=invalidate_on_finalize),
    )
    monkeypatch.setattr(
        regulatory_chat_reads, "finalize_message_publication_read", finalize_publication
    )
    response = chat_backend.get_chat_session(
        _SESSION_ID,
        user=cast(User, SimpleNamespace(id=_USER_ID)),
        db_session=db_session,
    )
    db_session.commit.assert_not_called()
    db_session.flush.assert_not_called()
    return response


def _packets_for(response: ChatSessionDetailResponse, message_id: int) -> list[Packet]:
    assistants = [
        message
        for message in response.messages
        if message.message_type == MessageType.ASSISTANT
    ]
    index = next(
        index
        for index, message in enumerate(assistants)
        if message.message_id == message_id
    )
    return response.packets[index]


def test_active_reserved_response_omits_placeholder_without_mutating_saved_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pending = _message(11)
    previous_attempt = _message(12)

    response = _get_session(monkeypatch, [pending, previous_attempt], run_id=pending.id)

    assert response.current_run is not None
    assert response.current_run.run_id == pending.id
    assert response.messages[0].message == ""
    assert response.messages[0].error is None
    assert not any(
        isinstance(packet.obj, (AgentResponseDelta, OverallStop))
        for packet in response.packets[0]
    )
    assert pending.message == RESERVED_ASSISTANT_MESSAGE
    assert pending.error is None
    assert response.messages[1].message == RESERVED_ASSISTANT_MESSAGE
    assert any(
        isinstance(packet.obj, AgentResponseDelta)
        and packet.obj.content == RESERVED_ASSISTANT_MESSAGE
        for packet in _packets_for(response, previous_attempt.id)
    )
    assert previous_attempt.message == RESERVED_ASSISTANT_MESSAGE


def test_pending_replay_retains_reasoning_tool_progress_and_source_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    checkpoint = cast(
        ToolCall,
        SimpleNamespace(
            tool_id=ASV3_CHECKPOINT_TOOL_ID,
            turn_number=0,
            tool_call_response=json.dumps(
                {
                    "version": 1,
                    "progress": [
                        {
                            "run_id": "test-run",
                            "event_id": "test-event",
                            "sequence": 1,
                            "language": "tr",
                            "phase": "research",
                            "status": "running",
                            "title": "Kaynaklar inceleniyor",
                        }
                    ],
                }
            ),
        ),
    )
    search = cast(
        ToolCall,
        SimpleNamespace(
            tool_id=71,
            turn_number=1,
            tab_index=0,
            reasoning_tokens=None,
            tool_call_arguments={"queries": ["Customs Law Article 46"]},
            search_docs=[source],
        ),
    )
    pending = _message(
        11,
        reasoning="Saved reasoning",
        tool_calls=[checkpoint, search],
        search_docs=[source],
    )

    response = _get_session(monkeypatch, [pending], run_id=pending.id)
    objects = [packet.obj for packet in response.packets[0]]

    assert any(
        isinstance(obj, ASv3Progress) and obj.status == "running" for obj in objects
    )
    assert any(
        isinstance(obj, ReasoningDelta) and obj.reasoning == "Saved reasoning"
        for obj in objects
    )
    assert any(
        isinstance(obj, SearchToolQueriesDelta)
        and obj.queries == ["Customs Law Article 46"]
        for obj in objects
    )
    assert any(
        isinstance(obj, CitationInfo)
        and obj.document_id == source.document_id
        and obj.chunk_ind == source.chunk_ind
        for obj in objects
    )
    assert not any(
        isinstance(obj, (AgentResponseDelta, OverallStop)) for obj in objects
    )
    assert response.messages[0].citations == {1: source.document_id}
    context_docs = response.messages[0].context_docs
    assert (
        context_docs is not None and context_docs[0].document_id == source.document_id
    )


def test_user_fenced_run_retains_ambiguous_previous_attempt_placeholders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = _message(10, message="Question", message_type=MessageType.USER)
    pending = _message(11)
    completed = _message(12, message="Completed answer")
    failed = _message(13, error="Provider failed")
    historical = _message(14, parent_message_id=9)
    previous_attempt = _message(15)

    response = _get_session(
        monkeypatch,
        [user, pending, completed, failed, historical, previous_attempt],
        run_id=user.id,
    )
    details = {message.message_id: message for message in response.messages}

    assert details[pending.id].message == RESERVED_ASSISTANT_MESSAGE
    assert details[completed.id].message == completed.message
    assert details[failed.id].message == RESERVED_ASSISTANT_MESSAGE
    assert details[failed.id].error == failed.error
    assert details[historical.id].message == RESERVED_ASSISTANT_MESSAGE
    assert details[previous_attempt.id].message == RESERVED_ASSISTANT_MESSAGE
    assert any(
        isinstance(packet.obj, AgentResponseDelta)
        and packet.obj.content == RESERVED_ASSISTANT_MESSAGE
        for packet in _packets_for(response, pending.id)
    )
    assert any(
        isinstance(packet.obj, OverallStop)
        for packet in _packets_for(response, pending.id)
    )
    assert any(
        isinstance(packet.obj, AgentResponseDelta)
        and packet.obj.content == RESERVED_ASSISTANT_MESSAGE
        for packet in _packets_for(response, historical.id)
    )
    assert all(
        message.message == RESERVED_ASSISTANT_MESSAGE
        for message in [pending, failed, historical, previous_attempt]
    )


@pytest.mark.parametrize("run_id", [None, 999])
def test_idle_or_unrelated_run_retains_interrupted_placeholder(
    monkeypatch: pytest.MonkeyPatch, run_id: int | None
) -> None:
    pending = _message(11)

    response = _get_session(monkeypatch, [pending], run_id=run_id)

    assert response.messages[0].message == RESERVED_ASSISTANT_MESSAGE
    assert any(
        isinstance(packet.obj, AgentResponseDelta)
        and packet.obj.content == RESERVED_ASSISTANT_MESSAGE
        for packet in response.packets[0]
    )
    assert any(isinstance(packet.obj, OverallStop) for packet in response.packets[0])


def test_unknown_cache_state_retains_interrupted_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pending = _message(11)

    response = _get_session(monkeypatch, [pending], run_id=pending.id, cache_fails=True)

    assert response.current_run is None
    assert response.messages[0].message == RESERVED_ASSISTANT_MESSAGE
    assert any(
        isinstance(packet.obj, AgentResponseDelta)
        and packet.obj.content == RESERVED_ASSISTANT_MESSAGE
        for packet in response.packets[0]
    )


@pytest.mark.parametrize(
    ("message", "error"),
    [
        ("Completed answer", None),
        (RESERVED_ASSISTANT_MESSAGE, "Provider failed"),
        (f"{RESERVED_ASSISTANT_MESSAGE} Additional details.", None),
    ],
)
def test_active_run_preserves_real_answers_and_errors(
    monkeypatch: pytest.MonkeyPatch, message: str, error: str | None
) -> None:
    saved = _message(11, message=message, error=error)

    response = _get_session(monkeypatch, [saved], run_id=saved.id)

    assert response.messages[0].message == message
    assert response.messages[0].error == error
    assert any(
        isinstance(packet.obj, AgentResponseDelta) and packet.obj.content == message
        for packet in response.packets[0]
    )
    assert any(isinstance(packet.obj, OverallStop) for packet in response.packets[0])


@pytest.mark.parametrize("invalidate_on_finalize", [False, True])
def test_pending_projection_never_overwrites_publication_error(
    monkeypatch: pytest.MonkeyPatch, invalidate_on_finalize: bool
) -> None:
    pending = _message(11)

    response = _get_session(
        monkeypatch,
        [pending],
        run_id=pending.id,
        publication_denied=not invalidate_on_finalize,
        invalidate_on_finalize=invalidate_on_finalize,
    )

    expected_error = str(PublicationReadChanged())
    assert response.messages[0].message == expected_error
    assert response.messages[0].error == expected_error
    assert response.packets == [[]]
    assert pending.message == RESERVED_ASSISTANT_MESSAGE
