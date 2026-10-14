from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import delete
from sqlalchemy.orm import Session

from ee.onyx.server.query_history.api import snapshot_from_chat_session
from ee.onyx.server.query_history.models import (
    ChatSessionMinimal,
    QuestionAnswerPairSnapshot,
)
from onyx.auth.api_key import ApiKeyDescriptor
from onyx.auth.schemas import UserRole
from onyx.chat.chat_utils import create_chat_session_from_request
from onyx.chat.models import CreateChatSessionID
from onyx.chat.process_message import build_chat_turn
from onyx.configs.constants import MessageType
from onyx.db.api_key import (
    insert_api_key,
    regenerate_api_key,
    remove_api_key,
    update_api_key,
)
from onyx.db.engine.async_sql_engine import reset_sqlalchemy_async_engine
from onyx.db.models import ChatMessage, ChatSession, Persona, User
from onyx.server.api_key.models import APIKeyArgs
from onyx.server.query_and_chat.chat_backend import create_new_chat_session
from onyx.server.query_and_chat.models import (
    ChatSessionCreationRequest,
    SendMessageRequest,
)
from shared_configs.contextvars import (
    CURRENT_USAGE_CREDENTIAL_CONTEXTVAR,
    UsageCredentialIdentity,
)
from shared_configs.enums import UsageCredentialType

pytestmark = pytest.mark.usefixtures("tenant_context")


@pytest.fixture
def integration_key(
    db_session: Session,
) -> Generator[tuple[ApiKeyDescriptor, User], None, None]:
    persona = db_session.get(Persona, 0)
    created_persona = persona is None
    if created_persona:
        db_session.add(Persona(id=0, name="Default test agent", description=""))
        db_session.commit()
    key = insert_api_key(
        db_session, APIKeyArgs(name="Müşteri Uygulaması", role=UserRole.BASIC), None
    )
    key.api_key = None
    owner = db_session.get(User, key.user_id)
    assert owner is not None
    try:
        yield key, owner
    finally:
        db_session.rollback()
        session_ids = [session.id for session in owner.chat_sessions]
        db_session.execute(
            delete(ChatMessage).where(ChatMessage.chat_session_id.in_(session_ids))
        )
        db_session.execute(delete(ChatSession).where(ChatSession.user_id == owner.id))
        remove_api_key(db_session, key.api_key_id)
        if created_persona:
            db_session.execute(delete(Persona).where(Persona.id == 0))
            db_session.commit()


def _create_session(
    db_session: Session, key: ApiKeyDescriptor, owner: User, payload: dict[str, object]
) -> UUID:
    token = CURRENT_USAGE_CREDENTIAL_CONTEXTVAR.set(
        UsageCredentialIdentity(
            UsageCredentialType.API_KEY,
            str(key.api_key_id),
            key.api_key_name,
            key.api_key_display,
        )
    )
    try:
        result = create_new_chat_session(
            ChatSessionCreationRequest.model_validate(payload), owner, db_session
        )
        assert set(result.model_dump()) == {"chat_session_id"}
        return result.chat_session_id
    finally:
        CURRENT_USAGE_CREDENTIAL_CONTEXTVAR.reset(token)


def test_http_creation_captures_authenticated_key_and_accepts_utf8_name(
    db_session: Session, integration_key: tuple[ApiKeyDescriptor, User]
) -> None:
    key, owner = integration_key
    rotated = regenerate_api_key(db_session, key.api_key_id)
    bearer = rotated.api_key
    rotated.api_key = None
    assert bearer is not None

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
        await reset_sqlalchemy_async_engine()
        try:
            yield
        finally:
            await reset_sqlalchemy_async_engine()

    app = FastAPI(lifespan=lifespan)
    app.add_api_route(
        "/api/chat/create-chat-session", create_new_chat_session, methods=["POST"]
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/chat/create-chat-session",
            headers={"Authorization": f"Bearer {bearer}"},
            json={"user": "Çağrı Şahin Öztürk", "api_key_id": 999999},
        )
    assert response.status_code == 200
    assert set(response.json()) == {"chat_session_id"}
    persisted = db_session.get(ChatSession, UUID(response.json()["chat_session_id"]))
    assert persisted is not None
    assert persisted.user_id == owner.id
    assert persisted.external_user_name == "Çağrı Şahin Öztürk"
    assert persisted.api_key_id == key.api_key_id
    assert persisted.api_key_display == rotated.api_key_display


def test_one_api_key_keeps_distinct_session_users_and_later_message_identity(
    db_session: Session, integration_key: tuple[ApiKeyDescriptor, User]
) -> None:
    key, owner = integration_key
    names = ["Çağrı  Şahin Öztürk", "İpek Çığdem Yılmaz"]
    session_ids = [
        _create_session(db_session, key, owner, {"user": name}) for name in names
    ]
    for session_id in session_ids:
        # Subsequent requests identify the session without repeating the name.
        request = SendMessageRequest(message="İkinci soru", chat_session_id=session_id)
        assert request.chat_session_info is None
        for message_type, text in [
            (MessageType.USER, "Birinci soru"),
            (MessageType.ASSISTANT, "Birinci cevap"),
            (MessageType.USER, request.message),
            (MessageType.ASSISTANT, "İkinci cevap"),
        ]:
            db_session.add(
                ChatMessage(
                    chat_session_id=session_id,
                    message_type=message_type,
                    message=text,
                    token_count=2,
                    time_sent=datetime.now(timezone.utc),
                )
            )
    db_session.commit()
    # Key renaming must not rewrite the recorded integration source.
    update_api_key(
        db_session, key.api_key_id, APIKeyArgs(name="Renamed key", role=UserRole.BASIC)
    )
    db_session.expire_all()
    for session_id, name in zip(session_ids, names, strict=True):
        persisted = db_session.get(ChatSession, session_id)
        assert persisted is not None
        assert persisted.user_id == owner.id
        minimal = ChatSessionMinimal.from_chat_session(persisted)
        snapshot = snapshot_from_chat_session(persisted, db_session)
        assert snapshot is not None
        for history in [minimal, snapshot]:
            assert history.user_name == name
            assert history.api_key_id == key.api_key_id
            assert history.api_key_name == "Müşteri Uygulaması"
            assert history.api_key_display == key.api_key_display
        pairs = QuestionAnswerPairSnapshot.from_chat_session_snapshot(snapshot)
        assert len(pairs) == 2
        assert all(pair.user_name == name for pair in pairs)
        assert all(pair.api_key_id == key.api_key_id for pair in pairs)


@pytest.mark.parametrize("payload", [{}, {"user": None}, {"user": "   "}])
def test_optional_user_keeps_key_identity_without_a_name(
    db_session: Session,
    integration_key: tuple[ApiKeyDescriptor, User],
    payload: dict[str, object],
) -> None:
    key, owner = integration_key
    session_id = _create_session(db_session, key, owner, payload)
    db_session.expire_all()
    persisted = db_session.get(ChatSession, session_id)
    assert persisted is not None
    assert persisted.external_user_name is None
    minimal = ChatSessionMinimal.from_chat_session(persisted)
    assert minimal.user_name is None
    assert minimal.api_key_id == key.api_key_id
    assert minimal.user_email


def test_inline_session_creation_and_untrusted_key_fields(
    db_session: Session, integration_key: tuple[ApiKeyDescriptor, User]
) -> None:
    key, owner = integration_key
    request = SendMessageRequest.model_validate(
        {
            "message": "Soru",
            "chat_session_info": {
                "user": "  Şule İğde  ",
                "api_key_id": 999999,
                "api_key_name": "Spoofed key",
            },
        }
    )
    assert request.chat_session_info is not None
    token = CURRENT_USAGE_CREDENTIAL_CONTEXTVAR.set(
        UsageCredentialIdentity(
            UsageCredentialType.API_KEY,
            str(key.api_key_id),
            key.api_key_name,
            key.api_key_display,
        )
    )
    try:
        created = create_chat_session_from_request(
            request.chat_session_info, owner, db_session
        )
    finally:
        CURRENT_USAGE_CREDENTIAL_CONTEXTVAR.reset(token)
    db_session.expire_all()
    assert created.external_user_name == "Şule İğde"
    assert created.api_key_id == key.api_key_id
    assert created.api_key_name == key.api_key_name


@pytest.mark.parametrize(
    "payload",
    [
        {"message": "İlk soru", "user": "  Çağrı Şahin Öztürk  "},
        {
            "message": "İlk soru",
            "user": "  Çağrı Şahin Öztürk  ",
            "chat_session_info": {"description": "Dış uygulama"},
        },
        {
            "message": "İlk soru",
            "chat_session_info": {"user": "  Çağrı Şahin Öztürk  "},
        },
    ],
)
def test_first_message_creates_session_with_saved_user(
    db_session: Session,
    integration_key: tuple[ApiKeyDescriptor, User],
    payload: dict[str, object],
) -> None:
    key, owner = integration_key
    request = SendMessageRequest.model_validate(payload)
    assert request.chat_session_info is not None
    token = CURRENT_USAGE_CREDENTIAL_CONTEXTVAR.set(
        UsageCredentialIdentity(
            UsageCredentialType.API_KEY,
            str(key.api_key_id),
            key.api_key_name,
            key.api_key_display,
        )
    )
    setup = build_chat_turn(request, owner, db_session, llm_overrides=None)
    try:
        # Session creation is the first stream packet, before answer generation.
        packet = next(setup)
    finally:
        setup.close()
        CURRENT_USAGE_CREDENTIAL_CONTEXTVAR.reset(token)
    assert isinstance(packet, CreateChatSessionID)
    db_session.expire_all()
    saved = db_session.get(ChatSession, packet.chat_session_id)
    assert saved is not None
    history = ChatSessionMinimal.from_chat_session(saved)
    assert history.user_name == "Çağrı Şahin Öztürk"
    assert history.api_key_id == key.api_key_id
    assert history.api_key_name == key.api_key_name
    assert saved.user_id == owner.id
    assert saved.description == (request.chat_session_info.description or "")


@pytest.mark.parametrize(
    "name", [42, True, {"name": "Çağrı"}, "A" * 256, "Çağrı\x00Şahin"]
)
def test_invalid_session_user_is_rejected(name: object) -> None:
    with pytest.raises(ValidationError):
        ChatSessionCreationRequest.model_validate({"user": name})
