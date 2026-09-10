"""Durable source authority for an answer which is still being delivered."""

from pydantic import ValidationError
from sqlalchemy.orm import Session

from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.models import ChatMessage
from onyx.document_index.publication_models import (
    PublicationModel,
    PublicationReadEvidence,
)
from onyx.regulatory.publication_reads import (
    PublicationReadChanged,
    evidence_unavailable,
    public_read_store,
)


class MessagePublicationRead(PublicationModel):
    evidence: PublicationReadEvidence
    generation_done: bool = False
    finalized: bool = False


def stage_message_publication_read(
    session: Session,
    message: ChatMessage,
    evidence: PublicationReadEvidence | None,
) -> None:
    if evidence is None:
        return
    if not public_read_store().lock_public_read(
        session, evidence.observation, evidence.user_file_ids
    ):
        raise PublicationReadChanged()
    message.publication_read = MessagePublicationRead(evidence=evidence).model_dump(
        mode="json"
    )


def mark_message_publication_generated(message_id: int) -> bool:
    return _advance_message_publication_read(message_id, finalize=False)


def finalize_message_publication_read(message_id: int) -> bool:
    return _advance_message_publication_read(message_id, finalize=True)


def _advance_message_publication_read(message_id: int, *, finalize: bool) -> bool:
    with get_session_with_current_tenant() as session:
        message = session.get(ChatMessage, message_id)
        if message is None or message.publication_read is None:
            return True
        state = MessagePublicationRead.model_validate(message.publication_read)
        if state.finalized:
            return True
        if finalize and not state.generation_done:
            return False
        if not public_read_store().lock_public_read(
            session, state.evidence.observation, state.evidence.user_file_ids
        ):
            return False
        message.publication_read = state.model_copy(
            update={"finalized": finalize, "generation_done": True}
        ).model_dump(mode="json")
        session.commit()
        return True


def message_publication_available(message: ChatMessage) -> bool:
    if message.publication_read is None:
        return True
    try:
        state = MessagePublicationRead.model_validate(message.publication_read)
    except ValidationError:
        return False
    return state.finalized or not evidence_unavailable(state.evidence)
