from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Barrier, Event
from uuid import UUID

import pytest
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from onyx.configs.constants import MessageType
from onyx.db.chat import RESERVED_ASSISTANT_MESSAGE, delete_chat_session
from onyx.db.models import ChatMessage, ChatSession, ChatSessionFeedback, User
from onyx.db.regulatory_chat_reads import MessagePublicationRead
from onyx.db.session_feedback import upsert_session_feedback
from onyx.document_index.publication_models import (
    PublicationReadEvidence,
    PublicationScope,
    ReadObservation,
)
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.server.query_and_chat.models import (
    SessionFeedbackRequest,
    SessionFeedbackResponse,
)
from tests.external_dependency_unit.conftest import create_test_user


@pytest.fixture
def feedback_session(db_session: Session) -> Generator[tuple[UUID, UUID], None, None]:
    owner = create_test_user(db_session, "session-feedback")
    chat_session = ChatSession(user_id=owner.id)
    db_session.add(chat_session)
    db_session.commit()
    session_id, owner_id = chat_session.id, owner.id
    yield session_id, owner_id
    db_session.rollback()
    db_session.execute(
        delete(ChatMessage).where(ChatMessage.chat_session_id == session_id)
    )
    db_session.execute(delete(ChatSession).where(ChatSession.id == session_id))
    owner = db_session.get(User, owner_id)
    if owner is not None:
        db_session.delete(owner)
    db_session.commit()


def test_session_feedback_update_noop_and_stale_version(
    db_session: Session, feedback_session: tuple[UUID, UUID]
) -> None:
    session_id, owner_id = feedback_session
    chat_session = db_session.get(ChatSession, session_id)
    assert chat_session is not None
    session_updated_at = chat_session.time_updated
    request = SessionFeedbackRequest(rating=3, feedback_text="  İstisna eksik.  ")
    first = SessionFeedbackResponse.from_model(
        upsert_session_feedback(
            db_session, session_id, owner_id, request.rating, request.feedback_text, 0
        )
    )
    assert first.label == "Kısmi doğru"
    assert first.feedback_text == "İstisna eksik."
    assert first.version == 1
    assert first.evaluated_through_message_id is None

    repeated = SessionFeedbackResponse.from_model(
        upsert_session_feedback(
            db_session, session_id, owner_id, request.rating, request.feedback_text, 1
        )
    )
    assert repeated == first
    updated = SessionFeedbackResponse.from_model(
        upsert_session_feedback(db_session, session_id, owner_id, 4, None, 1)
    )
    assert updated.version == 2
    assert updated.label == "Doğru"
    assert updated.feedback_text is None
    assert updated.time_created == first.time_created
    assert updated.time_updated > first.time_updated
    with pytest.raises(OnyxError) as conflict:
        upsert_session_feedback(db_session, session_id, owner_id, 1, "Stale", 1)
    assert conflict.value.error_code == OnyxErrorCode.CONFLICT
    db_session.rollback()
    persisted = db_session.get(ChatSessionFeedback, session_id)
    assert persisted is not None and persisted.rating == 4 and persisted.version == 2
    db_session.refresh(chat_session)
    assert chat_session.time_updated == session_updated_at


def _append_message(
    db_session: Session,
    session_id: UUID,
    message_type: MessageType,
    message_text: str,
    parent: ChatMessage | None = None,
) -> ChatMessage:
    message = ChatMessage(
        chat_session_id=session_id,
        parent_message_id=parent.id if parent else None,
        message_type=message_type,
        message=message_text,
        token_count=1,
    )
    db_session.add(message)
    db_session.flush()
    if parent:
        parent.latest_child_message_id = message.id
    return message


def test_feedback_scope_uses_active_branch_and_excludes_reserved_message(
    db_session: Session, feedback_session: tuple[UUID, UUID]
) -> None:
    session_id, owner_id = feedback_session
    root = _append_message(db_session, session_id, MessageType.SYSTEM, "")
    question = _append_message(db_session, session_id, MessageType.USER, "Q", root)
    completed = _append_message(
        db_session, session_id, MessageType.ASSISTANT, "A", question
    )
    alternative = _append_message(
        db_session, session_id, MessageType.ASSISTANT, "Other A", question
    )
    question.latest_child_message_id = completed.id
    db_session.commit()
    first = upsert_session_feedback(db_session, session_id, owner_id, 4, None)
    assert first.evaluated_through_message_id == completed.id

    followup = _append_message(
        db_session, session_id, MessageType.USER, "Q2", completed
    )
    reserved = _append_message(
        db_session,
        session_id,
        MessageType.ASSISTANT,
        RESERVED_ASSISTANT_MESSAGE,
        followup,
    )
    db_session.commit()
    repeated = upsert_session_feedback(db_session, session_id, owner_id, 4, None)
    assert repeated.version == 1
    assert repeated.evaluated_through_message_id == completed.id

    reserved.message = "Completed A2"
    db_session.commit()
    continued = upsert_session_feedback(db_session, session_id, owner_id, 4, None)
    assert continued.version == 2
    assert continued.evaluated_through_message_id == reserved.id
    question.latest_child_message_id = alternative.id
    db_session.commit()
    switched = upsert_session_feedback(db_session, session_id, owner_id, 4, None)
    assert switched.version == 3
    assert switched.evaluated_through_message_id == alternative.id


def test_feedback_scope_excludes_unpublished_and_malformed_regulatory_drafts(
    db_session: Session, feedback_session: tuple[UUID, UUID]
) -> None:
    session_id, owner_id = feedback_session
    root = _append_message(db_session, session_id, MessageType.SYSTEM, "")
    assistant = _append_message(
        db_session, session_id, MessageType.ASSISTANT, "Draft", root
    )
    publication = MessagePublicationRead(
        evidence=PublicationReadEvidence(
            observation=ReadObservation(
                scope=PublicationScope(
                    tenant_id="public",
                    environment="session-feedback-test",
                    database_identity="test",
                ),
                committed_epoch=0,
            )
        )
    )
    assistant.publication_read = publication.model_dump(mode="json")
    db_session.commit()
    draft = upsert_session_feedback(db_session, session_id, owner_id, 1, None)
    assert draft.evaluated_through_message_id is None
    assistant.publication_read = {"invalid": "legacy state"}
    db_session.commit()
    malformed = upsert_session_feedback(db_session, session_id, owner_id, 1, None)
    assert malformed.version == 1
    assert malformed.evaluated_through_message_id is None
    assistant.publication_read = publication.model_copy(
        update={"generation_done": True, "finalized": True}
    ).model_dump(mode="json")
    db_session.commit()
    published = upsert_session_feedback(db_session, session_id, owner_id, 1, None)
    assert published.version == 2
    assert published.evaluated_through_message_id == assistant.id


@pytest.mark.parametrize("conditional", [False, True])
def test_concurrent_first_feedback_is_serialized(
    db_session: Session, feedback_session: tuple[UUID, UUID], conditional: bool
) -> None:
    session_id, owner_id = feedback_session
    barrier = Barrier(2)
    engine = db_session.get_bind()

    def submit() -> int | OnyxErrorCode:
        with Session(engine) as worker_session:
            barrier.wait(timeout=5)
            try:
                return upsert_session_feedback(
                    worker_session,
                    session_id,
                    owner_id,
                    3,
                    None,
                    expected_version=0 if conditional else None,
                ).version
            except OnyxError as error:
                return error.error_code

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(submit) for _ in range(2)]
        results = [future.result(timeout=10) for future in futures]
    if conditional:
        assert results.count(1) == 1
        assert results.count(OnyxErrorCode.CONFLICT) == 1
    else:
        assert results == [1, 1]
    assert (
        db_session.scalar(
            select(func.count())
            .select_from(ChatSessionFeedback)
            .where(ChatSessionFeedback.chat_session_id == session_id)
        )
        == 1
    )


def test_feedback_scope_nulls_on_message_delete_and_cascades_on_session_delete(
    db_session: Session, feedback_session: tuple[UUID, UUID]
) -> None:
    session_id, owner_id = feedback_session
    root = _append_message(db_session, session_id, MessageType.SYSTEM, "")
    assistant = _append_message(
        db_session, session_id, MessageType.ASSISTANT, "A", root
    )
    db_session.commit()
    feedback = upsert_session_feedback(db_session, session_id, owner_id, 2, "Missing")
    assert feedback.evaluated_through_message_id == assistant.id
    root.latest_child_message_id = None
    db_session.flush()
    db_session.execute(delete(ChatMessage).where(ChatMessage.id == assistant.id))
    db_session.commit()
    persisted = db_session.get(ChatSessionFeedback, session_id)
    assert persisted is not None and persisted.evaluated_through_message_id is None
    db_session.execute(
        delete(ChatMessage).where(ChatMessage.chat_session_id == session_id)
    )
    db_session.execute(delete(ChatSession).where(ChatSession.id == session_id))
    db_session.commit()
    db_session.expire_all()
    assert db_session.get(ChatSessionFeedback, session_id) is None


def test_concurrent_delete_rejects_waiting_feedback_without_orphans(
    db_session: Session, feedback_session: tuple[UUID, UUID]
) -> None:
    session_id, owner_id = feedback_session
    engine = db_session.get_bind()
    started = Event()
    with Session(engine) as delete_session:
        chat_session = delete_session.scalar(
            select(ChatSession).where(ChatSession.id == session_id).with_for_update()
        )
        assert chat_session is not None

        def submit() -> OnyxErrorCode:
            with Session(engine) as worker_session:
                started.set()
                with pytest.raises(OnyxError) as failure:
                    upsert_session_feedback(
                        worker_session, session_id, owner_id, 4, None
                    )
                return failure.value.error_code

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(submit)
            assert started.wait(timeout=5)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.2)
            delete_session.delete(chat_session)
            delete_session.commit()
            assert future.result(timeout=5) == OnyxErrorCode.SESSION_NOT_FOUND
    assert db_session.get(ChatSessionFeedback, session_id) is None


def test_existing_hard_delete_waits_for_feedback_before_deleting_messages(
    db_session: Session, feedback_session: tuple[UUID, UUID]
) -> None:
    session_id, owner_id = feedback_session
    root = _append_message(db_session, session_id, MessageType.SYSTEM, "")
    assistant = _append_message(
        db_session, session_id, MessageType.ASSISTANT, "A", root
    )
    db_session.commit()
    engine = db_session.get_bind()
    started = Event()
    with Session(engine) as write_session:
        write_session.scalar(
            select(ChatSession).where(ChatSession.id == session_id).with_for_update()
        )

        def hard_delete() -> None:
            with Session(engine) as delete_session:
                started.set()
                delete_chat_session(
                    owner_id, session_id, delete_session, hard_delete=True
                )

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(hard_delete)
            assert started.wait(timeout=5)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.2)
            feedback = upsert_session_feedback(
                write_session, session_id, owner_id, 3, None
            )
            assert feedback.evaluated_through_message_id == assistant.id
            future.result(timeout=5)
    db_session.expire_all()
    assert db_session.get(ChatSessionFeedback, session_id) is None


@pytest.mark.parametrize("rating", [0, 5])
def test_database_rating_constraint(
    db_session: Session, feedback_session: tuple[UUID, UUID], rating: int
) -> None:
    session_id, owner_id = feedback_session
    upsert_session_feedback(db_session, session_id, owner_id, 4, None)
    with pytest.raises(IntegrityError):
        db_session.execute(
            update(ChatSessionFeedback)
            .where(ChatSessionFeedback.chat_session_id == session_id)
            .values(rating=rating)
        )
    db_session.rollback()
