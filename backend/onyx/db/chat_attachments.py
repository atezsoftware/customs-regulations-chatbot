"""Private chat attachments, retained outside the searchable corpus."""

from collections.abc import Mapping, Sequence
from typing import cast
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.chat.models import ChatLoadedFile
from onyx.configs.constants import DEFAULT_PERSONA_ID, FileOrigin
from onyx.db.models import ChatSession, FileRecord, User
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.file_store.file_store import get_default_file_store
from onyx.file_store.models import ChatAttachmentMetadata, FileDescriptor

MAX_SESSION_ATTACHMENTS = 10
MAX_SESSION_ATTACHMENT_BYTES = 20 * 1024 * 1024
MAX_SESSION_ATTACHMENT_TEXT_CHARS = 200_000


def lock_chat_attachment_owner(db_session: Session, user_id: UUID) -> None:
    # One lock serializes quota checks and competing draft claims for this owner.
    if (
        db_session.scalar(
            select(User.__table__.c.id)
            .where(User.__table__.c.id == user_id)
            .with_for_update()
        )
        is None
    ):
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "User not found")


def get_owned_attachment_session(
    db_session: Session, chat_session_id: UUID, user_id: UUID, *, lock: bool = False
) -> ChatSession:
    statement = select(ChatSession).where(
        ChatSession.id == chat_session_id,
        ChatSession.user_id == user_id,
        ChatSession.deleted.is_(False),
    )
    session = db_session.scalar(statement.with_for_update() if lock else statement)
    if session is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Chat session not found")
    if session.persona_id != DEFAULT_PERSONA_ID or session.project_id is not None:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            "Session attachments require the default assistant outside projects",
        )
    return session


def attachment_metadata(record: FileRecord) -> ChatAttachmentMetadata:
    if record.file_origin != FileOrigin.CHAT_UPLOAD:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Attachment not found")
    try:
        return ChatAttachmentMetadata.model_validate(record.file_metadata)
    except ValidationError as error:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Attachment not found") from error


def get_owned_chat_attachment(
    db_session: Session, file_id: str, user_id: UUID
) -> tuple[FileRecord, ChatAttachmentMetadata]:
    record = db_session.get(FileRecord, file_id)
    if record is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Attachment not found")
    metadata = attachment_metadata(record)
    if metadata.user_id != user_id:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Attachment not found")
    if metadata.chat_session_id is not None:
        get_owned_attachment_session(db_session, metadata.chat_session_id, user_id)
    return record, metadata


def get_chat_attachment_access(
    db_session: Session, file_id: str, user_id: UUID | None
) -> bool | None:
    """Fence private attachments before legacy public-session file access."""
    if file_id.startswith("chat_attachment_text_"):
        return False
    record = db_session.get(FileRecord, file_id)
    if record is None or not isinstance(record.file_metadata, Mapping):
        return None
    metadata = cast(Mapping[str, object], record.file_metadata)
    if metadata.get("kind") != "session_attachment":
        return None
    if user_id is None:
        return False
    try:
        get_owned_chat_attachment(db_session, file_id, user_id)
    except OnyxError:
        return False
    return True


def reject_session_attachment_in_generic_chat(
    db_session: Session, file_id: str, user_id: UUID | None = None
) -> None:
    access = get_chat_attachment_access(db_session, file_id, user_id)
    if access is False:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Attachment not found")
    if access is True:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            "Session attachments are supported only in ASv3 Tek and ASv3 Exp Cropped",
        )


def attachment_descriptor(record: FileRecord) -> FileDescriptor:
    metadata = attachment_metadata(record)
    return {
        "id": record.file_id,
        "type": metadata.chat_file_type,
        "name": record.display_name,
        "is_session_attachment": True,
    }


def ensure_chat_attachment_capacity(
    db_session: Session,
    user_id: UUID,
    chat_session_id: UUID | None,
    additional: Sequence[ChatAttachmentMetadata],
) -> None:
    lock_chat_attachment_owner(db_session, user_id)
    records = _list_attachment_records(
        db_session, user_id, chat_session_id, active_only=chat_session_id is None
    )
    retained = [attachment_metadata(record) for record in records] + list(additional)
    attachments = [metadata for metadata in retained if metadata.active]
    if len(attachments) > MAX_SESSION_ATTACHMENTS:
        raise ValueError(
            f"At most {MAX_SESSION_ATTACHMENTS} attachments are allowed per session or draft"
        )
    if sum(attachment.size for attachment in retained) > MAX_SESSION_ATTACHMENT_BYTES:
        if chat_session_id is not None:
            raise ValueError(
                "Retained originals exceed the 20 MiB session limit; start a new session. "
                "Removed attachments retain their originals for earlier messages."
            )
        raise ValueError("Combined attachments exceed the 20 MiB draft limit")
    if (
        sum(attachment.text_chars for attachment in attachments)
        > MAX_SESSION_ATTACHMENT_TEXT_CHARS
    ):
        raise ValueError(
            "Combined extracted text exceeds the 200,000-character session or draft limit"
        )


def _list_attachment_records(
    db_session: Session,
    user_id: UUID,
    chat_session_id: UUID | None,
    *,
    active_only: bool = True,
) -> Sequence[FileRecord]:
    session_filter = (
        FileRecord.file_metadata["chat_session_id"].as_string().is_(None)
        if chat_session_id is None
        else FileRecord.file_metadata["chat_session_id"].as_string()
        == str(chat_session_id)
    )
    statement = select(FileRecord).where(
        FileRecord.file_origin == FileOrigin.CHAT_UPLOAD,
        FileRecord.file_metadata["kind"].as_string() == "session_attachment",
        FileRecord.file_metadata["user_id"].as_string() == str(user_id),
        session_filter,
    )
    if active_only:
        statement = statement.where(
            FileRecord.file_metadata["active"].as_boolean().is_(True)
        )
    return db_session.scalars(
        statement.order_by(FileRecord.created_at, FileRecord.file_id)
    ).all()


def list_draft_chat_attachment_records(
    db_session: Session, user_id: UUID
) -> Sequence[FileRecord]:
    return _list_attachment_records(db_session, user_id, None)


def list_chat_session_attachment_records(
    db_session: Session, chat_session_id: UUID, user_id: UUID
) -> Sequence[FileRecord]:
    get_owned_attachment_session(db_session, chat_session_id, user_id)
    return _list_attachment_records(db_session, user_id, chat_session_id)


def list_chat_session_attachment_descriptors(
    db_session: Session, chat_session_id: UUID, user_id: UUID
) -> list[FileDescriptor]:
    return [
        attachment_descriptor(record)
        for record in list_chat_session_attachment_records(
            db_session, chat_session_id, user_id
        )
    ]


def canonicalize_and_bind_chat_attachments(
    db_session: Session,
    descriptors: list[FileDescriptor],
    chat_session_id: UUID,
    user_id: UUID,
) -> list[FileDescriptor]:
    lock_chat_attachment_owner(db_session, user_id)
    get_owned_attachment_session(db_session, chat_session_id, user_id, lock=True)
    file_ids = list(dict.fromkeys(descriptor["id"] for descriptor in descriptors))
    records = list(
        db_session.scalars(
            select(FileRecord)
            .where(FileRecord.file_id.in_(file_ids))
            .order_by(FileRecord.file_id)
            .with_for_update()
        ).all()
    )
    if len(records) != len(file_ids):
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Attachment not found")
    validated: list[tuple[FileRecord, ChatAttachmentMetadata]] = []
    for record in records:
        metadata = attachment_metadata(record)
        if (
            metadata.user_id != user_id
            or metadata.chat_session_id not in (None, chat_session_id)
            or not metadata.active
        ):
            raise OnyxError(OnyxErrorCode.NOT_FOUND, "Attachment not found")
        validated.append((record, metadata))
    try:
        ensure_chat_attachment_capacity(
            db_session,
            user_id,
            chat_session_id,
            [metadata for _, metadata in validated if metadata.chat_session_id is None],
        )
    except ValueError as error:
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(error)) from error
    for record, metadata in validated:
        record.file_metadata = metadata.model_copy(
            update={"chat_session_id": chat_session_id}
        ).model_dump(mode="json")
    db_session.commit()
    canonical = {record.file_id: attachment_descriptor(record) for record in records}
    return [canonical[file_id] for file_id in file_ids]


def delete_draft_chat_attachment(
    db_session: Session, file_id: str, user_id: UUID
) -> None:
    lock_chat_attachment_owner(db_session, user_id)
    record, metadata = get_owned_chat_attachment(db_session, file_id, user_id)
    if metadata.chat_session_id is not None:
        raise OnyxError(
            OnyxErrorCode.CONFLICT,
            "Use the session attachment endpoint to remove a bound file",
        )
    file_store = get_default_file_store()
    file_store.delete_file(record.file_id, error_on_missing=False)
    file_store.delete_file(metadata.plaintext_file_id, error_on_missing=False)
    db_session.commit()


def delete_chat_session_attachment_files(
    db_session: Session, chat_session_id: UUID
) -> set[str]:
    records = db_session.scalars(
        select(FileRecord).where(
            FileRecord.file_origin == FileOrigin.CHAT_UPLOAD,
            FileRecord.file_metadata["kind"].as_string() == "session_attachment",
            FileRecord.file_metadata["chat_session_id"].as_string()
            == str(chat_session_id),
        )
    ).all()
    file_store = get_default_file_store()
    removed: set[str] = set()
    for record in records:
        metadata = attachment_metadata(record)
        file_store.delete_file(record.file_id, error_on_missing=False)
        file_store.delete_file(metadata.plaintext_file_id, error_on_missing=False)
        removed.add(record.file_id)
    return removed


def remove_chat_session_attachment(
    db_session: Session, chat_session_id: UUID, file_id: str, user_id: UUID
) -> None:
    lock_chat_attachment_owner(db_session, user_id)
    get_owned_attachment_session(db_session, chat_session_id, user_id)
    record, metadata = get_owned_chat_attachment(db_session, file_id, user_id)
    if metadata.chat_session_id != chat_session_id:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Attachment not found")
    # Preserve original downloads for earlier messages while excluding future turns.
    record.file_metadata = metadata.model_copy(update={"active": False}).model_dump(
        mode="json"
    )
    db_session.commit()


def load_chat_session_attachments(
    db_session: Session, chat_session_id: UUID, user_id: UUID
) -> list[ChatLoadedFile]:
    file_store = get_default_file_store()
    loaded: list[ChatLoadedFile] = []
    for record in list_chat_session_attachment_records(
        db_session, chat_session_id, user_id
    ):
        metadata = attachment_metadata(record)
        with file_store.read_file(metadata.plaintext_file_id, mode="b") as content:
            text = content.read().decode("utf-8")

        def read_content(file_id: str = record.file_id) -> bytes:
            with file_store.read_file(file_id, mode="b") as content:
                return content.read()

        loaded.append(
            ChatLoadedFile.lazy_loaded(
                file_id=record.file_id,
                file_type=metadata.chat_file_type,
                filename=record.display_name,
                content_text=text,
                token_count=metadata.token_count,
                loader=read_content,
            )
        )
    return loaded
