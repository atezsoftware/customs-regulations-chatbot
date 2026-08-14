from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, Mock

import pytest

from onyx.chat import process_message
from onyx.chat.chat_state import ChatStateContainer, ChatTurnSetup
from onyx.chat.models import AnswerStream, StreamingError
from onyx.configs import app_configs
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import BaseFilters
from onyx.server.query_and_chat.models import MessageResponseIDInfo, SendMessageRequest


def test_default_persona_forces_regulatory_scope_without_discarding_user_filters() -> (
    None
):
    requested_date = process_message.datetime.date(2026, 7, 1)
    requested_filters = BaseFilters(
        document_set=["Dar kapsam"],
        as_of_date=requested_date,
    )
    setup = SimpleNamespace(
        persona=SimpleNamespace(id=process_message.DEFAULT_PERSONA_ID),
        new_msg_req=SimpleNamespace(
            internal_search_filters=requested_filters,
            message="Antrepo rejiminin şartları nelerdir?",
        ),
    )

    effective_filters = process_message._global_regulatory_search_filters(
        cast(ChatTurnSetup, setup)
    )

    assert effective_filters is not None
    assert effective_filters.regulatory_chunks_only is True
    assert effective_filters.source_type == [DocumentSource.USER_FILE]
    assert effective_filters.document_set == ["Dar kapsam"]
    assert effective_filters.as_of_date == requested_date


@pytest.mark.parametrize(
    "message",
    [
        "Merhaba!",
        "SELAM 👋",
        "Teşekkür ederim.",
        "Hello, how are you?",
        "Tamam, anladım.",
    ],
)
def test_default_persona_keeps_social_messages_out_of_regulatory_research(
    message: str,
) -> None:
    setup = SimpleNamespace(
        persona=SimpleNamespace(id=process_message.DEFAULT_PERSONA_ID),
        new_msg_req=SimpleNamespace(internal_search_filters=None, message=message),
    )

    effective_filters = process_message._global_regulatory_search_filters(
        cast(ChatTurnSetup, setup)
    )

    assert effective_filters is not None
    assert effective_filters.regulatory_chunks_only is False


@pytest.mark.parametrize(
    "message",
    [
        "Antrepo nedir?",
        "GTİP 8703?",
        "Merhaba, antrepo rejiminin şartları nelerdir?",
    ],
)
def test_default_persona_keeps_substantive_short_queries_in_regulatory_research(
    message: str,
) -> None:
    setup = SimpleNamespace(
        persona=SimpleNamespace(id=process_message.DEFAULT_PERSONA_ID),
        new_msg_req=SimpleNamespace(internal_search_filters=None, message=message),
    )

    effective_filters = process_message._global_regulatory_search_filters(
        cast(ChatTurnSetup, setup)
    )

    assert effective_filters is not None
    assert effective_filters.regulatory_chunks_only is True


def test_mock_llm_response_requires_integration_mode() -> None:
    assert app_configs.INTEGRATION_TESTS_MODE is False, (
        "Unit tests expect INTEGRATION_TESTS_MODE=false."
    )
    assert process_message.INTEGRATION_TESTS_MODE is False, (
        "process_message should reflect INTEGRATION_TESTS_MODE=false in unit tests."
    )

    request = SendMessageRequest(
        message="test",
        mock_llm_response='{"name":"internal_search","arguments":{"queries":["alpha"]}}',
    )
    mock_user = Mock()
    mock_user.id = "user-id"
    mock_user.is_anonymous = False
    mock_user.email = "user@example.com"

    with pytest.raises(
        ValueError,
        match="mock_llm_response can only be used when INTEGRATION_TESTS_MODE=true",
    ):
        next(
            process_message.handle_stream_message_objects(
                new_msg_req=request,
                user=mock_user,
            )
        )


def test_gather_stream_returns_empty_answer_when_streaming_error_only() -> None:
    packets: AnswerStream = iter(
        [
            MessageResponseIDInfo(
                user_message_id=None,
                reserved_assistant_message_id=42,
            ),
            StreamingError(
                error="OpenAI quota exceeded",
                error_code="BUDGET_EXCEEDED",
                is_retryable=False,
            ),
        ]
    )

    result = process_message.gather_stream(packets)

    assert result.answer == ""
    assert result.answer_citationless == ""
    assert result.error_msg == "OpenAI quota exceeded"
    assert result.message_id == 42


def test_gather_stream_accepts_structurally_compatible_response_metadata() -> None:
    packets = cast(
        AnswerStream,
        iter(
            [
                SimpleNamespace(reserved_assistant_message_id=43),
                StreamingError(
                    error="provider stream ended",
                    error_code="PROVIDER_ERROR",
                    is_retryable=True,
                ),
            ]
        ),
    )

    result = process_message.gather_stream(packets)

    assert result.message_id == 43
    assert result.error_msg == "provider stream ended"


def test_gather_stream_surfaces_setup_error_when_no_message_was_reserved() -> None:
    packets: AnswerStream = iter(
        [
            StreamingError(
                error="provider setup failed",
                error_code="PROVIDER_ERROR",
                is_retryable=True,
            )
        ]
    )

    with pytest.raises(RuntimeError, match="provider setup failed"):
        process_message.gather_stream(packets)


def test_gather_stream_full_accepts_structurally_compatible_metadata() -> None:
    state_container = MagicMock(spec=ChatStateContainer)
    state_container.get_answer_tokens.return_value = "answer"
    state_container.get_reasoning_tokens.return_value = None
    state_container.get_tool_calls.return_value = []
    packets = cast(
        AnswerStream,
        iter([SimpleNamespace(reserved_assistant_message_id=44)]),
    )

    result = process_message.gather_stream_full(packets, state_container)

    assert result.message_id == 44
    assert result.answer == "answer"
