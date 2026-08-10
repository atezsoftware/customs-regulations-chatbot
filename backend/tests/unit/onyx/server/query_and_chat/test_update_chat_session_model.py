from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from onyx.db.models import User
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.llm.override_models import LLMOverride
from onyx.server.query_and_chat.chat_backend import update_chat_session_model
from onyx.server.query_and_chat.models import UpdateChatSessionThreadRequest


def _call_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    chat_session: SimpleNamespace,
    serialized_model: str,
    provider: SimpleNamespace | None = None,
) -> MagicMock:
    db_session = MagicMock()
    resolved_provider = provider or SimpleNamespace(
        id=1,
        name=None,
        provider="vertex_ai",
        model_configurations=[
            SimpleNamespace(name="gemini-3.6-flash", is_visible=True)
        ],
    )
    provider_selection = (
        resolved_provider
        if any(
            model.name == "gemini-3.6-flash" and model.is_visible
            for model in resolved_provider.model_configurations
        )
        else None
    )
    monkeypatch.setattr(
        "onyx.server.query_and_chat.chat_backend.get_chat_session_by_id",
        lambda **_: chat_session,
    )

    def resolve_selection(
        provider_name: str | None,
        provider_type: str,
        model_name: str,
        selection_db_session: MagicMock,
    ) -> SimpleNamespace | None:
        expected_name, expected_type, expected_model = serialized_model.split(
            "__", maxsplit=2
        )
        assert provider_name == (expected_name or None)
        assert provider_type == expected_type
        assert model_name == expected_model
        assert selection_db_session is db_session
        return provider_selection

    monkeypatch.setattr(
        "onyx.server.query_and_chat.chat_backend.fetch_llm_provider_for_model_selection",
        resolve_selection,
        raising=False,
    )

    update_chat_session_model(
        UpdateChatSessionThreadRequest(
            chat_session_id=uuid4(), new_alternate_model=serialized_model
        ),
        user=cast(User, SimpleNamespace(id=uuid4())),
        db_session=db_session,
    )
    return db_session


def test_update_chat_session_model_persists_structured_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_session = SimpleNamespace(current_alternate_model=None, llm_override=None)
    serialized_model = "__vertex_ai__gemini-3.6-flash"

    db_session = _call_endpoint(monkeypatch, chat_session, serialized_model)

    assert chat_session.current_alternate_model == serialized_model
    assert chat_session.llm_override == LLMOverride(
        model_provider=None,
        model_provider_type="vertex_ai",
        model_version="gemini-3.6-flash",
    )
    db_session.add.assert_called_once_with(chat_session)
    db_session.commit.assert_called_once()


def test_frontend_nameless_selector_persists_successfully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_session = SimpleNamespace(current_alternate_model=None, llm_override=None)
    serialized_model = "vertex_ai__vertex_ai__gemini-3.6-flash"

    db_session = _call_endpoint(
        monkeypatch,
        chat_session,
        serialized_model,
    )

    assert chat_session.current_alternate_model == serialized_model
    assert chat_session.llm_override == LLMOverride(
        model_provider=None,
        model_provider_type="vertex_ai",
        model_version="gemini-3.6-flash",
    )
    db_session.commit.assert_called_once()


def test_update_chat_session_model_rejects_malformed_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_session = SimpleNamespace(
        current_alternate_model="OpenAI__openai__gpt-4o",
        llm_override=LLMOverride(model_provider="OpenAI", model_version="gpt-4o"),
    )

    with pytest.raises(OnyxError) as exc:
        _call_endpoint(monkeypatch, chat_session, "vertex_ai__missing-model")

    assert exc.value.error_code is OnyxErrorCode.INVALID_INPUT
    assert chat_session.current_alternate_model == "OpenAI__openai__gpt-4o"
    assert chat_session.llm_override == LLMOverride(
        model_provider="OpenAI", model_version="gpt-4o"
    )


def test_update_chat_session_model_rejects_unresolvable_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_session = SimpleNamespace(
        current_alternate_model="OpenAI__openai__gpt-4o",
        llm_override=LLMOverride(model_provider="OpenAI", model_version="gpt-4o"),
    )

    with pytest.raises(OnyxError) as exc:
        _call_endpoint(
            monkeypatch,
            chat_session,
            "__vertex_ai__missing-model",
            provider=SimpleNamespace(model_configurations=[]),
        )

    assert exc.value.error_code is OnyxErrorCode.INVALID_INPUT
    assert chat_session.current_alternate_model == "OpenAI__openai__gpt-4o"
    assert chat_session.llm_override == LLMOverride(
        model_provider="OpenAI", model_version="gpt-4o"
    )
