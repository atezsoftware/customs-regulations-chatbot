"""Verified retained full text for HTML/Markdown readers, without PDF rendering."""

import hashlib

from sqlalchemy.orm import Session

from onyx.db.document_intake_preview import intake_prepared_preview
from onyx.db.models import UserFile
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.file_store.file_store import get_default_file_store

MAX_READING_BYTES = 40_000_000


def read_document_markdown(session: Session, file: UserFile) -> str:
    store = get_default_file_store()
    prepared = intake_prepared_preview(session, file.id)
    with store.read_file(file.file_id, mode="b") as stream:
        original = stream.read(MAX_READING_BYTES + 1)
    if len(original) > MAX_READING_BYTES:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT, "Document exceeds the reading limit"
        )
    content = original
    if prepared is not None:
        if (
            prepared.original_file_id != file.file_id
            or hashlib.sha256(original).hexdigest() != prepared.raw_sha256
        ):
            raise OnyxError(OnyxErrorCode.CONFLICT, "The retained original changed")
        with store.read_file(prepared.text_file_id, mode="b") as stream:
            content = stream.read(MAX_READING_BYTES + 1)
        if (
            len(content) > MAX_READING_BYTES
            or hashlib.sha256(content).hexdigest() != prepared.text_sha256
        ):
            raise OnyxError(OnyxErrorCode.CONFLICT, "The verified reading text changed")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise OnyxError(
            OnyxErrorCode.SERVICE_UNAVAILABLE, "Verified reading text is unavailable"
        ) from error
    if content.startswith(b"%PDF-") or "\x00" in text:
        raise OnyxError(
            OnyxErrorCode.SERVICE_UNAVAILABLE, "Verified reading text is unavailable"
        )
    return text
