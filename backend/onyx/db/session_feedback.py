"""Owner-only session evaluations, serialized with session deletion and other writes."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.configs.constants import MessageType
from onyx.db.chat import RESERVED_ASSISTANT_MESSAGE
from onyx.db.models import ChatMessage, ChatSession, ChatSessionFeedback
from onyx.db.regulatory_chat_reads import (
    MessagePublicationRead,
    message_publication_available,
)
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError


def get_session_feedback(
    db_session: Session, chat_session_id: UUID
) -> ChatSessionFeedback | None:
    return db_session.get(ChatSessionFeedback, chat_session_id)


def _evaluated_through_message_id(
    db_session: Session, chat_session_id: UUID
) -> int | None:
    messages = db_session.scalars(
        select(ChatMessage).where(ChatMessage.chat_session_id == chat_session_id)
    ).all()
    roots = [message for message in messages if message.parent_message_id is None]
    if not messages:
        return None
    if len(roots) != 1:
        raise OnyxError(OnyxErrorCode.CONFLICT, "Invalid active chat branch")

    messages_by_id = {message.id: message for message in messages}
    current_message: ChatMessage | None = roots[0]
    visited: set[int] = set()
    evaluated_message_id: int | None = None
    while current_message is not None:
        if current_message.id in visited:
            raise OnyxError(OnyxErrorCode.CONFLICT, "Invalid active chat branch")
        visited.add(current_message.id)
        if (
            current_message.message_type == MessageType.ASSISTANT
            and current_message.error is None
            and current_message.message != RESERVED_ASSISTANT_MESSAGE
            and message_publication_available(current_message)
            and (
                current_message.publication_read is None
                or MessagePublicationRead.model_validate(
                    current_message.publication_read
                ).generation_done
            )
        ):
            evaluated_message_id = current_message.id
        next_message_id = current_message.latest_child_message_id
        if next_message_id is None:
            break
        current_message = messages_by_id.get(next_message_id)
        if current_message is None:
            raise OnyxError(OnyxErrorCode.CONFLICT, "Invalid active chat branch")
    return evaluated_message_id


def upsert_session_feedback(
    db_session: Session,
    chat_session_id: UUID,
    user_id: UUID,
    rating: int,
    feedback_text: str | None,
    expected_version: int | None = None,
) -> ChatSessionFeedback:
    # The session lock also serializes first inserts and concurrent hard/soft deletion.
    chat_session = db_session.scalar(
        select(ChatSession)
        .where(
            ChatSession.id == chat_session_id,
            ChatSession.user_id == user_id,
            ChatSession.deleted.is_(False),
        )
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if chat_session is None:
        raise OnyxError(OnyxErrorCode.SESSION_NOT_FOUND, "Chat session not found")

    feedback = db_session.scalar(
        select(ChatSessionFeedback)
        .where(ChatSessionFeedback.chat_session_id == chat_session_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    current_version = feedback.version if feedback else 0
    if expected_version is not None and expected_version != current_version:
        raise OnyxError(OnyxErrorCode.CONFLICT, "Session feedback has changed")

    evaluated_message_id = _evaluated_through_message_id(db_session, chat_session_id)
    if feedback is None:
        feedback = ChatSessionFeedback(
            chat_session_id=chat_session_id,
            rating=rating,
            feedback_text=feedback_text,
            submitted_by_user_id=user_id,
            evaluated_through_message_id=evaluated_message_id,
            version=1,
        )
        db_session.add(feedback)
    elif (
        feedback.rating,
        feedback.feedback_text,
        feedback.evaluated_through_message_id,
    ) != (
        rating,
        feedback_text,
        evaluated_message_id,
    ):
        feedback.rating = rating
        feedback.feedback_text = feedback_text
        feedback.submitted_by_user_id = user_id
        feedback.evaluated_through_message_id = evaluated_message_id
        feedback.version += 1

    db_session.flush()
    db_session.refresh(feedback)
    # Keep this write's response stable if another request commits immediately after it.
    db_session.expunge(feedback)
    db_session.commit()
    return feedback
