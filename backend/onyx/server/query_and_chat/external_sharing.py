"""API-key-only live sharing for trusted external application server proxies.

The external application must authorize its individual users against their session
mapping before creating or revoking links: a shared API key represents one owner.
Public viewers call that application's proxy, which keeps the core API key private.
"""

from http import HTTPStatus
from uuid import UUID

from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

from onyx.auth.permissions import require_permission
from onyx.configs.constants import PUBLIC_API_TAGS
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import Permission
from onyx.db.external_chat_share_messages import get_external_shared_messages
from onyx.db.external_chat_shares import (
    create_external_chat_share,
    get_external_shared_chat_session,
    revoke_external_chat_share,
)
from onyx.db.models import User
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.server.query_and_chat.external_sharing_models import (
    ExternalShareCreateRequest,
    ExternalShareCreateResponse,
    ExternalSharedChatMessagesResponse,
)
from shared_configs.contextvars import get_current_usage_credential
from shared_configs.enums import UsageCredentialType

router = APIRouter()
_NO_STORE_HEADERS = {"Cache-Control": "private, no-store"}


def _authenticated_api_key_id() -> int:
    credential = get_current_usage_credential()
    if (
        credential is None
        or credential.credential_type != UsageCredentialType.API_KEY
        or credential.credential_id is None
        or not credential.credential_id.isdecimal()
        or int(credential.credential_id) <= 0
    ):
        raise OnyxError(
            OnyxErrorCode.INSUFFICIENT_PERMISSIONS,
            "External chat sharing requires an authenticated application API key.",
            headers=_NO_STORE_HEADERS,
        )
    return int(credential.credential_id)


@router.post("/chat-session/{session_id}/share", tags=PUBLIC_API_TAGS)
def create_external_share(
    session_id: UUID,
    response: Response,
    share_request: ExternalShareCreateRequest | None = None,
    user: User = Depends(
        require_permission(Permission.WRITE_CHAT, allow_anonymous=True)
    ),
    db_session: Session = Depends(get_session),
) -> ExternalShareCreateResponse:
    """Create a live link without making the session public in the core frontend."""
    response.headers.update(_NO_STORE_HEADERS)
    api_key_id = _authenticated_api_key_id()
    share_token = create_external_chat_share(
        db_session=db_session,
        session_id=session_id,
        user_id=user.id,
        api_key_id=api_key_id,
    )
    settings = share_request or ExternalShareCreateRequest()
    share_url = f"{str(settings.share_url_base).rstrip('/')}/{share_token}"
    return ExternalShareCreateResponse(shareToken=share_token, shareUrl=share_url)


@router.get("/shared/{share_token}/messages", tags=PUBLIC_API_TAGS)
def get_external_share_messages(
    share_token: str,
    response: Response,
    user: User = Depends(
        require_permission(Permission.READ_CHAT, allow_anonymous=True)
    ),
    db_session: Session = Depends(get_session),
) -> ExternalSharedChatMessagesResponse:
    """Return current published messages to the owning application's server proxy."""
    response.headers.update(_NO_STORE_HEADERS)
    api_key_id = _authenticated_api_key_id()
    chat_session = get_external_shared_chat_session(
        db_session=db_session,
        share_token=share_token,
        user_id=user.id,
        api_key_id=api_key_id,
    )
    return get_external_shared_messages(db_session, chat_session)


@router.delete(
    "/shared/{share_token}", status_code=HTTPStatus.NO_CONTENT, tags=PUBLIC_API_TAGS
)
def revoke_external_share(
    share_token: str,
    response: Response,
    user: User = Depends(
        require_permission(Permission.WRITE_CHAT, allow_anonymous=True)
    ),
    db_session: Session = Depends(get_session),
) -> None:
    """Revoke a link; the external proxy must authorize its individual user first."""
    response.headers.update(_NO_STORE_HEADERS)
    api_key_id = _authenticated_api_key_id()
    revoke_external_chat_share(
        db_session=db_session,
        share_token=share_token,
        user_id=user.id,
        api_key_id=api_key_id,
    )
