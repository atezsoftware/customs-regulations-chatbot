from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch
from uuid import uuid4

from onyx.configs.constants import MessageType
from onyx.db.models import ChatMessage
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.llm.override_models import LLMOverride
from onyx.secondary_llm_flows.chat_session_naming import (
    DEFAULT_CHAT_SESSION_NAME,
    get_fallback_chat_session_name,
)
from onyx.server.query_and_chat.chat_backend import (
    _generate_or_fallback_chat_session_name,
    rename_chat_session,
)
from onyx.server.query_and_chat.models import ChatRenameRequest


def _chat_message(message: str, message_type: MessageType) -> ChatMessage:
    chat_message = MagicMock()
    chat_message.message = message
    chat_message.message_type = message_type
    return cast(ChatMessage, chat_message)


def test_fallback_chat_session_name_uses_first_user_message() -> None:
    chat_history = [
        _chat_message("Assistant text", MessageType.ASSISTANT),
        _chat_message("A" * 50, MessageType.USER),
        _chat_message("Later user text", MessageType.USER),
    ]

    assert get_fallback_chat_session_name(chat_history) == f"{'A' * 40}..."


def test_fallback_chat_session_name_handles_empty_history() -> None:
    assert get_fallback_chat_session_name([]) == DEFAULT_CHAT_SESSION_NAME


def test_rate_limited_chat_naming_returns_fallback() -> None:
    first_user_message = "Explain cost-based usage metering"
    chat_history = [_chat_message(first_user_message, MessageType.USER)]
    request = MagicMock()
    user = MagicMock()
    user.id = uuid4()

    with (
        patch(
            "onyx.server.query_and_chat.chat_backend.check_token_rate_limits",
            side_effect=OnyxError(OnyxErrorCode.RATE_LIMITED),
        ),
        patch(
            "onyx.server.query_and_chat.chat_backend.get_llm_for_persona"
        ) as get_llm_for_persona,
    ):
        generated_name = _generate_or_fallback_chat_session_name(
            chat_history=chat_history,
            request=request,
            user=user,
            chat_session_id=uuid4(),
            persona=None,
            llm_override=None,
        )

    assert generated_name == first_user_message
    get_llm_for_persona.assert_not_called()


def test_dedicated_chat_naming_model_keeps_provider_type_identity() -> None:
    chat_session_id = uuid4()
    user = MagicMock()
    user.id = uuid4()
    chat_session = SimpleNamespace(persona=None, llm_override=None)
    naming_model = SimpleNamespace(
        name="shared-model",
        llm_provider=SimpleNamespace(name="Shared Provider", provider="openrouter"),
    )
    db_session = MagicMock()

    with (
        patch(
            "onyx.server.query_and_chat.chat_backend.get_session_with_current_tenant",
            return_value=nullcontext(db_session),
        ),
        patch(
            "onyx.server.query_and_chat.chat_backend.get_chat_session_by_id",
            return_value=chat_session,
        ),
        patch(
            "onyx.server.query_and_chat.chat_backend.create_chat_history_chain",
            return_value=[],
        ),
        patch(
            "onyx.server.query_and_chat.chat_backend.fetch_default_chat_naming_model",
            return_value=naming_model,
        ),
        patch(
            "onyx.server.query_and_chat.chat_backend._generate_or_fallback_chat_session_name",
            return_value="Generated name",
        ) as generate_name,
        patch("onyx.server.query_and_chat.chat_backend.update_chat_session"),
    ):
        response = rename_chat_session(
            ChatRenameRequest(chat_session_id=chat_session_id),
            MagicMock(),
            user,
        )

    assert response.new_name == "Generated name"
    assert generate_name.call_args.kwargs["llm_override"] == LLMOverride(
        model_provider="Shared Provider",
        model_provider_type="openrouter",
        model_version="shared-model",
    )
