"""Read-only projection of the active conversation for an authorized share."""

from ipaddress import ip_address
from urllib.parse import urlsplit
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, load_only, raiseload, selectinload

from onyx.configs.constants import DocumentSource, MessageType
from onyx.db.chat import RESERVED_ASSISTANT_MESSAGE
from onyx.db.models import ChatMessage, ChatSession, SearchDoc
from onyx.db.regulatory_chat_reads import (
    MessagePublicationRead,
    message_publication_available,
)
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.server.query_and_chat.external_sharing_models import (
    ExternalSharedChatCitation,
    ExternalSharedChatMessage,
    ExternalSharedChatMessagesResponse,
    ExternalSharedChatSession,
)


def _invalid_active_branch() -> OnyxError:
    return OnyxError(
        OnyxErrorCode.CONFLICT,
        "Invalid active chat branch",
        headers={"Cache-Control": "private, no-store"},
    )


def _active_message_chain(
    messages: list[ChatMessage], session_id: UUID
) -> list[ChatMessage]:
    if not messages:
        return []
    roots = [message for message in messages if message.parent_message_id is None]
    messages_by_id = {message.id: message for message in messages}
    if len(roots) != 1 or len(messages_by_id) != len(messages):
        raise _invalid_active_branch()

    chain: list[ChatMessage] = []
    visited: set[int] = set()
    current = roots[0]
    while True:
        if current.id in visited or current.chat_session_id != session_id:
            raise _invalid_active_branch()
        visited.add(current.id)
        chain.append(current)
        if current.latest_child_message_id is None:
            return chain
        child = messages_by_id.get(current.latest_child_message_id)
        if child is None or child.parent_message_id != current.id:
            raise _invalid_active_branch()
        current = child


def _message_is_visible(message: ChatMessage) -> bool:
    if (
        message.message_type not in (MessageType.USER, MessageType.ASSISTANT)
        or message.error is not None
        or message.message == RESERVED_ASSISTANT_MESSAGE
        or not message_publication_available(message)
    ):
        return False
    return (
        message.message_type != MessageType.ASSISTANT
        or message.publication_read is None
        or MessagePublicationRead.model_validate(
            message.publication_read
        ).generation_done
    )


def _public_citation_url(document: SearchDoc) -> str | None:
    if (
        document.source_type
        in (DocumentSource.FILE, DocumentSource.USER_FILE, DocumentSource.CRAFT_FILE)
        or document.is_internet is not True
        or not document.link
        or any(
            character.isspace() or ord(character) < 32 for character in document.link
        )
    ):
        return None
    try:
        url = urlsplit(document.link)
        hostname = url.hostname.rstrip(".") if url.hostname else None
        if (
            url.scheme not in ("http", "https")
            or not hostname
            or url.username is not None
            or url.password is not None
            or url.path.startswith(("/api/chat/", "/api/asv3/", "/api/admin/"))
            or hostname == "localhost"
            or hostname.endswith((".localhost", ".local", ".internal"))
        ):
            return None
        # A literal address must be globally routable; single-label hosts are internal.
        try:
            address = ip_address(hostname)
        except ValueError:
            if "." not in hostname or hostname.rsplit(".", 1)[-1].isdecimal():
                return None
        else:
            if not address.is_global:
                return None
        # Accessing port also rejects malformed authority values.
        _ = url.port
    except ValueError:
        return None
    return document.link


def _normalized_citation_bindings(citations: object) -> dict[int, int]:
    if not isinstance(citations, dict):
        return {}
    bindings: dict[int, int] = {}
    ambiguous_numbers: set[int] = set()
    for key, document_id in citations.items():
        if isinstance(key, int) and not isinstance(key, bool):
            number = key
        elif isinstance(key, str) and key.isascii() and key.isdecimal():
            try:
                number = int(key)
            except ValueError:
                continue
        else:
            continue
        if (
            number <= 0
            or not isinstance(document_id, int)
            or isinstance(document_id, bool)
            or document_id <= 0
            or number in ambiguous_numbers
        ):
            continue
        if number in bindings and bindings[number] != document_id:
            bindings.pop(number)
            ambiguous_numbers.add(number)
        else:
            bindings[number] = document_id
    return bindings


def _message_citations(message: ChatMessage) -> list[ExternalSharedChatCitation]:
    documents_by_id = {document.id: document for document in message.search_docs}
    return [
        ExternalSharedChatCitation(
            number=number,
            title=document.semantic_id,
            url=_public_citation_url(document),
        )
        for number, document_id in sorted(
            _normalized_citation_bindings(message.citations).items()
        )
        if (document := documents_by_id.get(document_id)) is not None
    ]


def get_external_shared_messages(
    db_session: Session, chat_session: ChatSession
) -> ExternalSharedChatMessagesResponse:
    """Project an already authorized session without changing ownership or history."""
    messages = list(
        db_session.scalars(
            select(ChatMessage)
            .where(ChatMessage.chat_session_id == chat_session.id)
            .options(
                load_only(
                    ChatMessage.id,
                    ChatMessage.chat_session_id,
                    ChatMessage.parent_message_id,
                    ChatMessage.latest_child_message_id,
                    ChatMessage.message_type,
                    ChatMessage.message,
                    ChatMessage.time_sent,
                    ChatMessage.citations,
                    ChatMessage.error,
                    ChatMessage.publication_read,
                    raiseload=True,
                ),
                selectinload(ChatMessage.search_docs).load_only(
                    SearchDoc.id,
                    SearchDoc.semantic_id,
                    SearchDoc.link,
                    SearchDoc.source_type,
                    SearchDoc.is_internet,
                    raiseload=True,
                ),
                raiseload("*"),
            )
        ).all()
    )
    candidates: list[tuple[ChatMessage, ExternalSharedChatMessage]] = []
    for message in _active_message_chain(messages, chat_session.id):
        if not _message_is_visible(message):
            continue
        detail = ExternalSharedChatMessage(
            id=message.id,
            sessionId=chat_session.id,
            role="user" if message.message_type == MessageType.USER else "assistant",
            content=message.message,
            timestamp=message.time_sent,
            citations=_message_citations(message)
            if message.message_type == MessageType.ASSISTANT
            else [],
        )
        candidates.append((message, detail))
    # Recheck every retained message after assembling the entire live response.
    projected = [
        detail for message, detail in candidates if _message_is_visible(message)
    ]
    return ExternalSharedChatMessagesResponse(
        session=ExternalSharedChatSession(
            id=chat_session.id,
            title=chat_session.description,
            createdAt=chat_session.time_created,
            updatedAt=chat_session.time_updated,
            messageCount=len(projected),
        ),
        messages=projected,
    )
