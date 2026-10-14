from uuid import uuid4

import pytest
from pydantic import ValidationError

from onyx.server.query_and_chat.models import (
    ChatSessionCreationRequest,
    SendMessageRequest,
)


def test_top_level_user_creates_session_for_constructor_and_validation() -> None:
    requests = (
        SendMessageRequest(message="Merhaba", user="  Çağrı  Şahin Öztürk  "),
        SendMessageRequest.model_validate(
            {"message": "Merhaba", "user": "  Çağrı  Şahin Öztürk  "}
        ),
    )
    for request in requests:
        assert request.user == "Çağrı  Şahin Öztürk"
        assert request.chat_session_id is None
        assert request.chat_session_info is not None
        assert request.chat_session_info.user == "Çağrı  Şahin Öztürk"


def test_top_level_user_preserves_explicit_session_creation_options() -> None:
    request = SendMessageRequest(
        message="Merhaba",
        user="  İpek Şahin  ",
        chat_session_info=ChatSessionCreationRequest(
            persona_id=7,
            project_id=12,
            description="Dış uygulama oturumu",
        ),
    )
    assert request.chat_session_info is not None
    assert request.chat_session_info.model_dump() == {
        "persona_id": 7,
        "project_id": 12,
        "description": "Dış uygulama oturumu",
        "user": "İpek Şahin",
    }


def test_nested_user_remains_compatible_without_top_level_user() -> None:
    request = SendMessageRequest.model_validate(
        {
            "message": "Merhaba",
            "chat_session_info": {"user": "  Özgür Çelik  ", "persona_id": 4},
        }
    )
    assert request.user is None
    assert request.chat_session_info is not None
    assert request.chat_session_info.user == "Özgür Çelik"
    assert request.chat_session_info.persona_id == 4


def test_equal_normalized_top_level_and_nested_names_are_allowed() -> None:
    request = SendMessageRequest(
        message="Merhaba",
        user="  Çağrı Şahin  ",
        chat_session_info=ChatSessionCreationRequest(user="Çağrı Şahin "),
    )
    assert request.user == "Çağrı Şahin"
    assert request.chat_session_info is not None
    assert request.chat_session_info.user == "Çağrı Şahin"


def test_conflicting_normalized_top_level_and_nested_names_are_rejected() -> None:
    with pytest.raises(ValidationError):
        SendMessageRequest.model_validate(
            {
                "message": "Merhaba",
                "user": "  Çağrı Şahin  ",
                "chat_session_info": {"user": " İpek Şahin "},
            }
        )


def test_existing_session_id_does_not_create_session_info_from_user() -> None:
    session_id = uuid4()
    request = SendMessageRequest(
        message="Devam sorusu",
        chat_session_id=session_id,
        user="  İpek Şahin  ",
    )
    assert request.chat_session_id == session_id
    assert request.chat_session_info is None


def test_existing_session_id_with_creation_info_remains_invalid() -> None:
    with pytest.raises(ValidationError):
        SendMessageRequest(
            message="Devam sorusu",
            chat_session_id=uuid4(),
            chat_session_info=ChatSessionCreationRequest(user="İpek Şahin"),
            user="İpek Şahin",
        )


@pytest.mark.parametrize(
    "user_fields", [{}, {"user": None}, {"user": ""}, {"user": " \t\n "}]
)
def test_omitted_null_and_blank_user_create_unnamed_sessions(
    user_fields: dict[str, str | None],
) -> None:
    creation_request = ChatSessionCreationRequest.model_validate(user_fields)
    request = SendMessageRequest.model_validate({"message": "Merhaba", **user_fields})
    assert creation_request.user is None
    assert request.user is None
    assert request.chat_session_info is not None
    assert request.chat_session_info.user is None


@pytest.mark.parametrize("invalid_user", [42, "İpek\x00Şahin", "İ" * 256])
def test_creation_and_message_models_share_strict_user_validation(
    invalid_user: object,
) -> None:
    for model, payload in (
        (ChatSessionCreationRequest, {"user": invalid_user}),
        (SendMessageRequest, {"message": "Merhaba", "user": invalid_user}),
    ):
        with pytest.raises(ValidationError):
            model.model_validate(payload)


def test_user_accepts_255_unicode_characters() -> None:
    full_name = "İ" * 255
    request = SendMessageRequest(message="Merhaba", user=full_name)
    assert request.chat_session_info is not None
    assert request.chat_session_info.user == full_name
