from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from hashlib import sha256
from typing import NamedTuple
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from onyx.auth.schemas import UserRole
from onyx.configs.constants import MessageType
from onyx.db.api_key import insert_api_key, regenerate_api_key, remove_api_key
from onyx.db.engine.async_sql_engine import reset_sqlalchemy_async_engine
from onyx.db.enums import ChatSessionSharedStatus, Permission
from onyx.db.models import ChatMessage, ChatSession, ExternalChatShare, Persona, User
from onyx.db.pat import create_pat
from onyx.error_handling.exceptions import register_onyx_exception_handlers
from onyx.redis.redis_pool import get_async_redis_connection
from onyx.server.api_key.models import APIKeyArgs
from onyx.server.query_and_chat.chat_backend import router as chat_router

pytestmark = pytest.mark.usefixtures("tenant_context")


class ShareContext(NamedTuple):
    client: TestClient
    owner: User
    owner_key_id: int
    owner_token: SecretStr
    other_token: SecretStr
    session_id: UUID
    last_message: ChatMessage


def _headers(token: SecretStr) -> dict[str, str]:
    return {"Authorization": f"Bearer {token.get_secret_value()}"}


def _append_message(
    db_session: Session,
    session_id: UUID,
    kind: MessageType,
    content: str,
    parent: ChatMessage | None,
) -> ChatMessage:
    message = ChatMessage(
        chat_session_id=session_id,
        parent_message_id=parent.id if parent else None,
        message_type=kind,
        message=content,
        token_count=1,
        reasoning_tokens="Private reasoning" if kind == MessageType.ASSISTANT else None,
    )
    db_session.add(message)
    db_session.flush()
    if parent is not None:
        parent.latest_child_message_id = message.id
    return message


@pytest.fixture
def share_context(db_session: Session) -> Generator[ShareContext, None, None]:
    created_persona = db_session.get(Persona, 0) is None
    if created_persona:
        db_session.add(Persona(id=0, name="Share test agent", description=""))
        db_session.commit()
    keys = [
        insert_api_key(
            db_session, APIKeyArgs(name="Tariff", role=UserRole.BASIC), None
        ),
        insert_api_key(
            db_session, APIKeyArgs(name="Other app", role=UserRole.ADMIN), None
        ),
    ]
    assert all(key.api_key for key in keys)
    tokens = [SecretStr(key.api_key or "") for key in keys]
    for key in keys:
        key.api_key = None
    owner = db_session.get(User, keys[0].user_id)
    assert owner is not None

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
        await reset_sqlalchemy_async_engine()
        try:
            yield
        finally:
            redis = await get_async_redis_connection()
            await redis.aclose(close_connection_pool=True)
            await reset_sqlalchemy_async_engine()

    app = FastAPI(lifespan=lifespan)
    register_onyx_exception_handlers(app)
    app.include_router(chat_router, prefix="/api")
    try:
        with TestClient(app) as client:
            result = client.post(
                "/api/chat/create-chat-session",
                headers=_headers(tokens[0]),
                json={"user": "Çağrı Şahin", "description": "Canlı paylaşım"},
            )
            assert result.status_code == 200
            session_id = UUID(result.json()["chat_session_id"])
            root = _append_message(db_session, session_id, MessageType.SYSTEM, "", None)
            question = _append_message(
                db_session, session_id, MessageType.USER, "İlk soru", root
            )
            answer = _append_message(
                db_session, session_id, MessageType.ASSISTANT, "İlk cevap", question
            )
            db_session.commit()
            yield ShareContext(
                client,
                owner,
                keys[0].api_key_id,
                tokens[0],
                tokens[1],
                session_id,
                answer,
            )
    finally:
        db_session.rollback()
        for key in keys:
            session_ids = list(
                db_session.scalars(
                    select(ChatSession.id).where(ChatSession.user_id == key.user_id)
                )
            )
            db_session.execute(
                delete(ChatMessage).where(ChatMessage.chat_session_id.in_(session_ids))
            )
            db_session.execute(
                delete(ChatSession).where(ChatSession.user_id == key.user_id)
            )
            remove_api_key(db_session, key.api_key_id)
        if created_persona:
            db_session.execute(delete(Persona).where(Persona.id == 0))
            db_session.commit()


def _share(context: ShareContext) -> str:
    result = context.client.post(
        f"/api/chat/chat-session/{context.session_id}/share",
        headers=_headers(context.owner_token),
    )
    assert result.status_code == 200
    assert set(result.json()) == {"shareToken", "shareUrl"}
    token = result.json()["shareToken"]
    assert (
        result.json()["shareUrl"]
        == f"https://tariff.singlewindow.io/chat/shared/{token}"
    )
    assert "no-store" in result.headers["cache-control"]
    return token


def test_live_link_reads_new_turns_without_exposing_core_ui_or_private_fields(
    db_session: Session, share_context: ShareContext
) -> None:
    context = share_context
    token = _share(context)
    assert token != str(context.session_id)
    stored = db_session.get(ExternalChatShare, sha256(token.encode()).hexdigest())
    assert stored is not None and stored.api_key_id == context.owner_key_id
    path = f"/api/chat/shared/{token}/messages"
    first = context.client.get(path, headers=_headers(context.owner_token))
    assert first.status_code == 200
    assert "no-store" in first.headers["cache-control"]
    assert [row["content"] for row in first.json()["messages"]] == [
        "İlk soru",
        "İlk cevap",
    ]
    assert set(first.json()) == {"session", "messages"}
    assert set(first.json()["messages"][1]) == {
        "id",
        "sessionId",
        "role",
        "content",
        "timestamp",
        "citations",
    }
    assert set(first.json()["session"]) == {
        "id",
        "title",
        "createdAt",
        "updatedAt",
        "messageCount",
        "isShared",
    }
    assert "Private reasoning" not in first.text
    assert "Çağrı Şahin" not in first.text

    question = _append_message(
        db_session,
        context.session_id,
        MessageType.USER,
        "İkinci soru",
        context.last_message,
    )
    _append_message(
        db_session, context.session_id, MessageType.ASSISTANT, "İkinci cevap", question
    )
    db_session.commit()
    second = context.client.get(path, headers=_headers(context.owner_token))
    assert second.status_code == 200
    assert [row["content"] for row in second.json()["messages"]] == [
        "İlk soru",
        "İlk cevap",
        "İkinci soru",
        "İkinci cevap",
    ]
    assert second.json()["session"]["messageCount"] == 4
    db_session.expire_all()
    session = db_session.get(ChatSession, context.session_id)
    assert (
        session is not None and session.shared_status == ChatSessionSharedStatus.PRIVATE
    )
    blocked = context.client.get(
        f"/api/chat/get-chat-session/{context.session_id}?is_shared=true",
        headers=_headers(context.owner_token),
    )
    assert blocked.status_code == 403


def test_links_are_owned_by_the_application_key_even_for_other_admin_keys(
    share_context: ShareContext,
) -> None:
    context = share_context
    token = _share(context)
    for method, path in [
        ("POST", f"/api/chat/chat-session/{context.session_id}/share"),
        ("GET", f"/api/chat/shared/{token}/messages"),
        ("DELETE", f"/api/chat/shared/{token}"),
    ]:
        denied = context.client.request(
            method, path, headers=_headers(context.other_token)
        )
        assert denied.status_code == 404
        assert "no-store" in denied.headers["cache-control"]
    assert context.client.get(f"/api/chat/shared/{token}/messages").status_code == 403


def test_revoke_is_idempotent_and_invalid_tokens_do_not_read_sessions(
    share_context: ShareContext,
) -> None:
    context = share_context
    token = _share(context)
    for _ in range(2):
        response = context.client.delete(
            f"/api/chat/shared/{token}", headers=_headers(context.owner_token)
        )
        assert response.status_code == 204 and response.content == b""
    for bad_token in [token, "does-not-exist", str(context.session_id)]:
        response = context.client.get(
            f"/api/chat/shared/{bad_token}/messages",
            headers=_headers(context.owner_token),
        )
        assert response.status_code == 404
        assert "no-store" in response.headers["cache-control"]


def test_rotating_key_preserves_links_and_deleting_session_closes_them(
    db_session: Session, share_context: ShareContext
) -> None:
    context = share_context
    token = _share(context)
    rotated = regenerate_api_key(db_session, context.owner_key_id)
    rotated_token = SecretStr(rotated.api_key or "")
    rotated.api_key = None
    path = f"/api/chat/shared/{token}/messages"
    assert context.client.get(path, headers=_headers(rotated_token)).status_code == 200
    session = db_session.get(ChatSession, context.session_id)
    assert session is not None
    session.deleted = True
    db_session.commit()
    assert context.client.get(path, headers=_headers(rotated_token)).status_code == 404


def test_hard_delete_removes_share_records(
    db_session: Session, share_context: ShareContext
) -> None:
    context = share_context
    token = _share(context)
    db_session.execute(
        delete(ChatMessage).where(ChatMessage.chat_session_id == context.session_id)
    )
    db_session.execute(delete(ChatSession).where(ChatSession.id == context.session_id))
    db_session.commit()
    db_session.expire_all()
    assert db_session.get(ExternalChatShare, sha256(token.encode()).hexdigest()) is None
    response = context.client.get(
        f"/api/chat/shared/{token}/messages", headers=_headers(context.owner_token)
    )
    assert response.status_code == 404


def test_read_only_key_cannot_create_or_revoke_links(
    db_session: Session, share_context: ShareContext
) -> None:
    context = share_context
    token = _share(context)
    context.owner.effective_permissions = [Permission.READ_CHAT.value]
    db_session.commit()
    assert (
        context.client.get(
            f"/api/chat/shared/{token}/messages", headers=_headers(context.owner_token)
        ).status_code
        == 200
    )
    assert (
        context.client.post(
            f"/api/chat/chat-session/{context.session_id}/share",
            headers=_headers(context.owner_token),
        ).status_code
        == 403
    )
    assert (
        context.client.delete(
            f"/api/chat/shared/{token}", headers=_headers(context.owner_token)
        ).status_code
        == 403
    )


def test_external_share_url_can_target_the_applications_own_route(
    share_context: ShareContext,
) -> None:
    context = share_context
    response = context.client.post(
        f"/api/chat/chat-session/{context.session_id}/share",
        headers=_headers(context.owner_token),
        json={"share_url_base": "https://app.example.com/custom/shares/"},
    )
    assert response.status_code == 200
    token = response.json()["shareToken"]
    assert (
        response.json()["shareUrl"] == f"https://app.example.com/custom/shares/{token}"
    )


def test_limited_keys_with_chat_permissions_work_but_pat_cannot_use_shares(
    db_session: Session, share_context: ShareContext
) -> None:
    context = share_context
    context.owner.role = UserRole.LIMITED
    context.owner.effective_permissions = [Permission.WRITE_CHAT.value]
    db_session.commit()
    token = _share(context)
    assert (
        context.client.get(
            f"/api/chat/shared/{token}/messages", headers=_headers(context.owner_token)
        ).status_code
        == 200
    )
    _, raw_pat = create_pat(db_session, context.owner.id, "Share denial", None)
    denied = context.client.get(
        f"/api/chat/shared/{token}/messages", headers=_headers(SecretStr(raw_pat))
    )
    assert denied.status_code == 403


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/shared",
        "https://user:password@example.com/shared",
        "https://example.com/shared?x=1",
        "https://example.com/shared#fragment",
    ],
)
def test_invalid_share_frontend_url_is_rejected(
    share_context: ShareContext, url: str
) -> None:
    context = share_context
    response = context.client.post(
        f"/api/chat/chat-session/{context.session_id}/share",
        headers=_headers(context.owner_token),
        json={"share_url_base": url},
    )
    assert response.status_code == 422
