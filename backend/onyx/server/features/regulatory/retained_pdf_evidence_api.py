"""Admin retention of exact original PDFs, independent of publication."""

import hashlib
import io
from urllib.parse import quote
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, Response, UploadFile
from sqlalchemy.orm import Session

from onyx.auth.permissions import require_permission
from onyx.configs.constants import PUBLIC_API_TAGS, FileOrigin
from onyx.db import regulatory_retained_pdf_evidence as repository
from onyx.db.engine.sql_engine import get_session, get_session_with_current_tenant
from onyx.db.enums import Permission
from onyx.db.models import User
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.file_store.file_store import get_default_file_store
from onyx.regulatory.retained_pdf_evidence import (
    MAX_RETAINED_PDF_BYTES,
    PdfRenditionRelation,
    RetainedPdfDescriptor,
    RetainedPdfList,
    RetainedPdfReceipt,
    validate_retained_pdf,
)
from onyx.utils.logger import setup_logger

router = APIRouter(
    prefix="/files/{user_file_id}/retained-pdf-evidence", tags=PUBLIC_API_TAGS
)
_admin = require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)
logger = setup_logger()


def _discard_unattached_blob(raw_file_id: str) -> None:
    try:
        with get_session_with_current_tenant() as session:
            attached = repository.retained_pdf_blob_is_attached(session, raw_file_id)
        if not attached:
            get_default_file_store().delete_file(raw_file_id, error_on_missing=False)
    except Exception:
        logger.warning("Retained PDF cleanup deferred for owned blob %s", raw_file_id)


@router.post("")
def attach_retained_pdf(
    user_file_id: UUID,
    document_set_id: int = Form(...),
    idempotency_key: UUID = Form(...),
    expected_canonical_sha256: str = Form(...),
    relation: PdfRenditionRelation = Form(...),
    reason: str = Form(...),
    source_url: str | None = Form(default=None),
    file: UploadFile = File(...),
    user: User = Depends(_admin),
    db_session: Session = Depends(get_session),
) -> RetainedPdfReceipt:
    raw_file_id, commit_attempted = None, False
    try:
        _target, digest = repository.retained_pdf_scope(
            db_session, user, user_file_id, document_set_id
        )
        descriptor = RetainedPdfDescriptor(
            document_set_id=document_set_id,
            idempotency_key=idempotency_key,
            expected_canonical_sha256=expected_canonical_sha256,
            relation=relation,
            reason=reason,
            source_url=source_url,
            original_filename=file.filename or "original.pdf",
        )
        content = file.file.read(MAX_RETAINED_PDF_BYTES + 1)
        sha256 = validate_retained_pdf(content, file.content_type)
        existing = repository.existing_retained_pdf(
            db_session,
            user=user,
            user_file_id=user_file_id,
            descriptor=descriptor,
            sha256=sha256,
            byte_count=len(content),
        )
        if existing is not None:
            return RetainedPdfReceipt.model_validate(existing)
        if digest != expected_canonical_sha256:
            raise repository.RetainedPdfConflictError(
                "The canonical document changed; reload before attaching an original"
            )
        raw_file_id = str(uuid4())
        get_default_file_store().save_file(
            io.BytesIO(content),
            display_name=descriptor.original_filename,
            file_origin=FileOrigin.OTHER,
            file_type="application/pdf",
            file_id=raw_file_id,
            file_metadata={
                "sha256": sha256,
                "retained_pdf_user_file_id": str(user_file_id),
            },
        )
        row, created = repository.append_retained_pdf(
            db_session,
            user=user,
            user_file_id=user_file_id,
            descriptor=descriptor,
            sha256=sha256,
            byte_count=len(content),
            raw_file_id=raw_file_id,
        )
        receipt = RetainedPdfReceipt.model_validate(row)
        if not created:
            db_session.rollback()
            _discard_unattached_blob(raw_file_id)
            return receipt
        # A commit error can be ambiguous: leave its bytes for a keyed retry.
        commit_attempted = True
        db_session.commit()
        return receipt
    except Exception as error:
        db_session.rollback()
        if raw_file_id is not None and not commit_attempted:
            _discard_unattached_blob(raw_file_id)
        if isinstance(error, PermissionError):
            raise OnyxError(
                OnyxErrorCode.INSUFFICIENT_PERMISSIONS, str(error)
            ) from error
        if isinstance(error, repository.RetainedPdfConflictError):
            raise OnyxError(OnyxErrorCode.CONFLICT, str(error)) from error
        if isinstance(error, ValueError):
            raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(error)) from error
        raise


@router.get("")
def list_retained_pdfs(
    user_file_id: UUID,
    document_set_id: int,
    user: User = Depends(_admin),
    db_session: Session = Depends(get_session),
) -> RetainedPdfList:
    try:
        target, digest = repository.retained_pdf_scope(
            db_session, user, user_file_id, document_set_id, require_active=False
        )
    except PermissionError as error:
        raise OnyxError(OnyxErrorCode.INSUFFICIENT_PERMISSIONS, str(error)) from error
    except ValueError as error:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, str(error)) from error
    return RetainedPdfList(
        document_set_id=document_set_id,
        user_file_id=user_file_id,
        canonical_sha256=digest,
        canonical_generation_hash=target.regulatory_chunk_generation_hash,
        items=[
            RetainedPdfReceipt.model_validate(row)
            for row in repository.list_retained_pdfs(db_session, user_file_id)
        ],
    )


@router.get("/{evidence_id}/download")
def download_retained_pdf(
    user_file_id: UUID,
    evidence_id: UUID,
    document_set_id: int,
    user: User = Depends(_admin),
    db_session: Session = Depends(get_session),
) -> Response:
    try:
        repository.retained_pdf_scope(
            db_session, user, user_file_id, document_set_id, require_active=False
        )
        row = repository.get_retained_pdf(
            db_session, user_file_id, evidence_id, document_set_id
        )
    except PermissionError as error:
        raise OnyxError(OnyxErrorCode.INSUFFICIENT_PERMISSIONS, str(error)) from error
    except ValueError as error:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, str(error)) from error
    with get_default_file_store().read_file(row.raw_file_id) as stream:
        content = stream.read(MAX_RETAINED_PDF_BYTES + 1)
    if (
        len(content) != row.byte_count
        or hashlib.sha256(content).hexdigest() != row.sha256
    ):
        raise OnyxError(
            OnyxErrorCode.INTERNAL_ERROR, "Retained original integrity check failed"
        )
    return Response(
        content,
        media_type=row.mime_type,
        headers={
            "Content-Disposition": "attachment; filename*=UTF-8''"
            + quote(row.original_filename, safe=""),
            "Content-Security-Policy": "sandbox",
            "X-Content-Type-Options": "nosniff",
        },
    )
