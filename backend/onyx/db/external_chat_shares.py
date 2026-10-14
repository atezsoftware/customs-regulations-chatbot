"""External live shares remain separate from the core frontend's public chats."""

import hashlib
import secrets
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.models import ChatSession, ExternalChatShare
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError


def _share_not_found() -> OnyxError:
    return OnyxError(
        OnyxErrorCode.NOT_FOUND,
        "Shared conversation not found",
        headers={"Cache-Control": "no-store"},
    )


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_external_chat_share(
    db_session: Session,
    session_id: UUID,
    user_id: UUID,
    api_key_id: int,
) -> str:
    session = db_session.scalar(
        select(ChatSession)
        .where(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id,
            ChatSession.deleted.is_(False),
            ChatSession.benchmark_flow.is_(False),
        )
        .with_for_update()
    )
    if session is None:
        raise _share_not_found()
    token = secrets.token_urlsafe(32)
    db_session.add(
        ExternalChatShare(
            token_hash=_token_hash(token),
            chat_session_id=session.id,
            created_by_user_id=user_id,
            api_key_id=api_key_id,
        )
    )
    db_session.commit()
    return token


def get_external_shared_chat_session(
    db_session: Session,
    share_token: str,
    user_id: UUID,
    api_key_id: int,
) -> ChatSession:
    session = db_session.scalar(
        select(ChatSession)
        .join(ExternalChatShare, ExternalChatShare.chat_session_id == ChatSession.id)
        .where(
            ExternalChatShare.token_hash == _token_hash(share_token),
            ExternalChatShare.created_by_user_id == user_id,
            ExternalChatShare.api_key_id == api_key_id,
            ExternalChatShare.revoked_at.is_(None),
            ChatSession.user_id == user_id,
            ChatSession.deleted.is_(False),
            ChatSession.benchmark_flow.is_(False),
        )
    )
    if session is None:
        raise _share_not_found()
    return session


def revoke_external_chat_share(
    db_session: Session,
    share_token: str,
    user_id: UUID,
    api_key_id: int,
) -> None:
    share = db_session.scalar(
        select(ExternalChatShare)
        .join(ChatSession, ChatSession.id == ExternalChatShare.chat_session_id)
        .where(
            ExternalChatShare.token_hash == _token_hash(share_token),
            ExternalChatShare.created_by_user_id == user_id,
            ExternalChatShare.api_key_id == api_key_id,
            ChatSession.user_id == user_id,
        )
        .with_for_update(of=ExternalChatShare)
    )
    if share is None:
        raise _share_not_found()
    if share.revoked_at is None:
        share.revoked_at = datetime.now(timezone.utc)
    db_session.commit()
