import hashlib
import io
import mimetypes
from collections.abc import Iterator
from http import HTTPStatus
from pathlib import PurePosixPath
from urllib.parse import quote
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, Response, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from onyx.asv3.session_attachments import PDF_PAGE_INPUT_TOKEN_RESERVE
from onyx.auth.permissions import require_permission
from onyx.configs.constants import PUBLIC_API_TAGS, FileOrigin
from onyx.db.chat_attachments import (
    attachment_metadata,
    delete_draft_chat_attachment,
    ensure_chat_attachment_capacity,
    get_owned_attachment_session,
    get_owned_chat_attachment,
    list_chat_session_attachment_records,
    list_draft_chat_attachment_records,
    lock_chat_attachment_owner,
    remove_chat_session_attachment,
)
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import Permission
from onyx.db.models import FileRecord, User
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.file_processing.chat_attachments import (
    MAX_CHAT_ATTACHMENT_BYTES,
    extract_chat_attachment_text,
    validate_chat_attachment_pdf,
)
from onyx.file_store.file_store import get_default_file_store
from onyx.file_store.models import ChatAttachmentMetadata, ChatFileType
from onyx.natural_language_processing.utils import get_tokenizer
from onyx.server.features.chat_files.models import (
    ChatAttachmentSnapshot,
    ChatAttachmentUploadResponse,
    RejectedChatAttachment,
)
from onyx.utils.logger import setup_logger

router = APIRouter(prefix="/chat", tags=PUBLIC_API_TAGS)
logger = setup_logger()
MAX_FILES_PER_UPLOAD = 10


def _attachment_snapshot(record: FileRecord) -> ChatAttachmentSnapshot:
    metadata = attachment_metadata(record)
    return ChatAttachmentSnapshot(
        id=record.file_id,
        type=metadata.chat_file_type,
        name=record.display_name,
        token_count=metadata.token_count,
        size=metadata.size,
    )


@router.post("/files/upload")
def upload_chat_attachments(
    files: list[UploadFile] = File(...),
    chat_session_id: UUID | None = Form(None),
    user: User = Depends(require_permission(Permission.WRITE_CHAT)),
    db_session: Session = Depends(get_session),
) -> ChatAttachmentUploadResponse:
    if len(files) > MAX_FILES_PER_UPLOAD:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            f"Upload at most {MAX_FILES_PER_UPLOAD} files per request",
        )
    if chat_session_id is not None:
        lock_chat_attachment_owner(db_session, user.id)
        get_owned_attachment_session(db_session, chat_session_id, user.id, lock=True)
    response = ChatAttachmentUploadResponse()
    file_store = get_default_file_store()
    tokenizer = get_tokenizer(model_name=None, provider_type=None)
    for upload in files:
        filename = PurePosixPath((upload.filename or "").replace("\\", "/")).name
        extension = PurePosixPath(filename).suffix.lower()
        pdf_token_reserve = 0
        try:
            content = upload.file.read(MAX_CHAT_ATTACHMENT_BYTES + 1)
            if extension == ".pdf":
                pages = validate_chat_attachment_pdf(content)
                pdf_token_reserve = pages * PDF_PAGE_INPUT_TOKEN_RESERVE
                text = (
                    f"[Original PDF attached directly to the model: {pages} page(s). "
                    "No text extraction or preliminary model reading was performed.]"
                )
            else:
                text = extract_chat_attachment_text(filename, content)
        except ValueError as error:
            response.rejected_files.append(
                RejectedChatAttachment(name=filename, reason=str(error))
            )
            continue
        file_type = (
            ChatFileType.TABULAR
            if extension in {".csv", ".xlsx"}
            else ChatFileType.DOC
            if extension in {".pdf", ".docx"}
            else ChatFileType.PLAIN_TEXT
        )
        file_id = str(uuid4())
        plaintext_file_id = f"chat_attachment_text_{file_id}"
        metadata = ChatAttachmentMetadata(
            user_id=user.id,
            chat_session_id=chat_session_id,
            plaintext_file_id=plaintext_file_id,
            chat_file_type=file_type,
            token_count=pdf_token_reserve or len(tokenizer.encode(text)),
            size=len(content),
            text_chars=len(text),
            sha256=hashlib.sha256(content).hexdigest(),
        )
        try:
            ensure_chat_attachment_capacity(
                db_session, user.id, chat_session_id, [metadata]
            )
        except ValueError as error:
            response.rejected_files.append(
                RejectedChatAttachment(name=filename, reason=str(error))
            )
            continue
        file_store.save_file(
            content=io.BytesIO(text.encode("utf-8")),
            display_name=filename,
            file_origin=FileOrigin.PLAINTEXT_CACHE,
            file_type="text/plain",
            file_id=plaintext_file_id,
        )
        try:
            file_store.save_file(
                content=io.BytesIO(content),
                display_name=filename,
                file_origin=FileOrigin.CHAT_UPLOAD,
                file_type=mimetypes.guess_type(filename)[0]
                or "application/octet-stream",
                file_metadata=metadata.model_dump(mode="json"),
                file_id=file_id,
            )
        except Exception:
            try:
                file_store.delete_file(plaintext_file_id, error_on_missing=False)
            except Exception:
                logger.exception("Failed to remove unlinked attachment plaintext")
            raise
        response.files.append(
            ChatAttachmentSnapshot(
                id=file_id,
                type=file_type,
                name=filename,
                token_count=metadata.token_count,
                size=metadata.size,
            )
        )
    db_session.commit()
    return response


@router.get("/files")
def list_draft_chat_attachments(
    user: User = Depends(require_permission(Permission.READ_CHAT)),
    db_session: Session = Depends(get_session),
) -> list[ChatAttachmentSnapshot]:
    return [
        _attachment_snapshot(record)
        for record in list_draft_chat_attachment_records(db_session, user.id)
    ]


@router.get("/sessions/{chat_session_id}/files")
def list_chat_attachments(
    chat_session_id: UUID,
    user: User = Depends(require_permission(Permission.READ_CHAT)),
    db_session: Session = Depends(get_session),
) -> list[ChatAttachmentSnapshot]:
    return [
        _attachment_snapshot(record)
        for record in list_chat_session_attachment_records(
            db_session, chat_session_id, user.id
        )
    ]


@router.delete("/sessions/{chat_session_id}/files/{file_id}")
def delete_chat_attachment(
    chat_session_id: UUID,
    file_id: str,
    user: User = Depends(require_permission(Permission.WRITE_CHAT)),
    db_session: Session = Depends(get_session),
) -> Response:
    remove_chat_session_attachment(db_session, chat_session_id, file_id, user.id)
    return Response(status_code=HTTPStatus.NO_CONTENT)


@router.delete("/files/{file_id}")
def delete_draft_attachment(
    file_id: str,
    user: User = Depends(require_permission(Permission.WRITE_CHAT)),
    db_session: Session = Depends(get_session),
) -> Response:
    delete_draft_chat_attachment(db_session, file_id, user.id)
    return Response(status_code=HTTPStatus.NO_CONTENT)


@router.get("/files/{file_id}")
def download_chat_attachment(
    file_id: str,
    user: User = Depends(require_permission(Permission.READ_CHAT)),
    db_session: Session = Depends(get_session),
) -> Response:
    record, _metadata = get_owned_chat_attachment(db_session, file_id, user.id)

    def content() -> Iterator[bytes]:
        with get_default_file_store().read_file(file_id, mode="b") as stream:
            while block := stream.read(64 * 1024):
                yield block

    return StreamingResponse(
        content(),
        media_type=record.file_type,
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(record.display_name or 'attachment')}",
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )
