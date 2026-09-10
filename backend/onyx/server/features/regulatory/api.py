"""Chunk inspection/editing endpoints for the Files panel.

Postgres `regulatory_chunk` rows are the source of truth. Content mutations
re-project the whole file; file-level validity uses an exact metadata-only
Elasticsearch patch and rejects unsafe projections.
"""

import hashlib
import io
import json
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import Response
from sqlalchemy.orm import Session

from onyx.auth.permissions import require_permission
from onyx.auth.schemas import UserRole
from onyx.background.celery.tasks.regulatory_amendments.sources import (
    enqueue_source_package,
)
from onyx.background.celery.tasks.regulatory_amendments.tasks import (
    enqueue_amendment_batch,
    enqueue_amendment_proposal_approval,
)
from onyx.configs.app_configs import MAX_AMENDMENT_SOURCE_BYTES
from onyx.configs.constants import PUBLIC_API_TAGS, FileOrigin
from onyx.db.amendment_sources import (
    attach_source_package_to_batch,
    create_source_package,
    get_source_asset,
    get_source_package,
    list_source_assets,
    mark_source_package_failed,
    require_ready_source_package,
    retry_source_package,
)
from onyx.db.document_set import get_document_set_by_id_for_user
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import AmendmentBatchStatus, Permission
from onyx.db.models import DocumentSet, User, UserFile
from onyx.db.regulatory_amendments import (
    compute_duplicate_targets,
    create_batch,
    finalize_amendment_proposal_projection,
    get_batch,
    get_proposal,
    list_batches_for_document_set,
    list_proposals_for_batch,
    queue_amendment_proposal_approval,
    reject_proposal,
    reset_amendment_proposal_approval,
    reset_failed_batch_for_retry,
    retry_amendment_proposal_projection,
)
from onyx.db.regulatory_annex_changes import list_annex_changes
from onyx.db.regulatory_chunks import (
    ValidityDateUpdate,
    delete_hierarchical_aggregates_referencing_chunk,
    get_chunk_by_id,
    get_chunks_for_file,
    is_hierarchical_aggregate_chunk,
    update_chunk,
    update_file_validity_window,
)
from onyx.db.user_file import lock_completed_user_file_for_projection
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.file_store.file_store import get_default_file_store
from onyx.regulatory.amendments.annexes import config as annex_config
from onyx.regulatory.amendments.annexes.sources import MAX_ASSET_BYTES
from onyx.regulatory.amendments.source_extraction import (
    AmendmentSourceExtractionError,
    fetch_and_extract_amendment_url,
)
from onyx.regulatory.amendments.source_extraction import (
    extract_amendment_docx as extract_amendment_docx_text,
)
from onyx.regulatory.amendments.source_extraction import (
    extract_amendment_pdf as extract_amendment_pdf_text,
)
from onyx.regulatory.pdf import render_chunk_pdf, render_document_pdf
from onyx.regulatory.projection import project_user_file_to_index
from onyx.regulatory.validity_projection import (
    compensate_regulatory_validity_patch,
    patch_user_file_validity_in_active_indices,
)
from onyx.server.features.projects.models import UserFileSnapshot
from onyx.server.features.regulatory.annex_api import router as annex_router
from onyx.server.features.regulatory.models import (
    AmendmentBatchSnapshot,
    AmendmentProposalSnapshot,
    AmendmentSourceAssetSnapshot,
    AmendmentSourceExtractionSnapshot,
    AmendmentSourcePackageSnapshot,
    AmendmentSourceUrlRequest,
    AnalyzeAmendmentRequest,
    AnalyzeAmendmentResponse,
    AnnexReviewSnapshot,
    ApproveAmendmentProposalRequest,
    CreateAmendmentSourcePackageRequest,
    RegulatoryChunkSnapshot,
    RegulatoryChunkUpdateRequest,
    RegulatoryFileValidityUpdateRequest,
    RegulatoryFileValidityUpdateResponse,
    UserFileRenameRequest,
)
from onyx.utils.logger import setup_logger
from shared_configs.contextvars import get_current_tenant_id

logger = setup_logger()

router = APIRouter(prefix="/regulatory")

_FILE_VALIDITY_REINDEX_REQUIRED = (
    "A canonical reindex is required before this file's validity can be updated safely."
)


def _get_owned_user_file(
    db_session: Session, user_file_id: UUID, user: User
) -> UserFile:
    user_file = db_session.get(UserFile, user_file_id)
    if user_file is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "File not found")
    if user_file.user_id != user.id and user.role != UserRole.ADMIN:
        raise OnyxError(OnyxErrorCode.UNAUTHORIZED, "Not your file")
    return user_file


@router.get("/files/{user_file_id}/chunks", tags=PUBLIC_API_TAGS)
def list_chunks_for_file(
    user_file_id: UUID,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> list[RegulatoryChunkSnapshot]:
    _get_owned_user_file(db_session, user_file_id, user)
    chunks = get_chunks_for_file(db_session, user_file_id)
    return [RegulatoryChunkSnapshot.from_model(chunk) for chunk in chunks]


def _pdf_response(pdf: bytes, filename: str) -> Response:
    """Inline so the browser renders it; the same bytes are what gets saved."""

    safe_name = filename.rsplit("/", 1)[-1].replace('"', "")
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{safe_name}"'},
    )


@router.get("/files/{user_file_id}/pdf", tags=PUBLIC_API_TAGS)
def get_file_pdf(
    user_file_id: UUID,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> Response:
    """The document as uploaded, before chunking, rendered for reading."""

    user_file = _get_owned_user_file(db_session, user_file_id, user)
    with get_default_file_store().read_file(user_file.file_id, mode="b") as handle:
        markdown = handle.read().decode("utf-8", errors="replace")

    pdf = render_document_pdf(name=user_file.name, markdown=markdown)
    return _pdf_response(pdf, f"{user_file.name.rsplit('.', 1)[0]}.pdf")


@router.get("/chunks/{chunk_id}/pdf", tags=PUBLIC_API_TAGS)
def get_chunk_pdf(
    chunk_id: str,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> Response:
    chunk = get_chunk_by_id(db_session, chunk_id)
    if chunk is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Chunk not found")
    # Access is granted on the file, so it is checked there.
    _get_owned_user_file(db_session, chunk.user_file_id, user)

    pdf = render_chunk_pdf(
        text=chunk.text,
        heading_path=list(chunk.heading_path),
        validity_start_date=chunk.validity_start_date,
        validity_end_date=chunk.validity_end_date,
        position=chunk.position,
    )
    return _pdf_response(pdf, f"chunk-{chunk.position + 1}.pdf")


@router.patch("/chunks/{chunk_id}", tags=PUBLIC_API_TAGS)
def patch_chunk(
    chunk_id: str,
    update_request: RegulatoryChunkUpdateRequest,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> RegulatoryChunkSnapshot:
    chunk = get_chunk_by_id(db_session, chunk_id)
    if chunk is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Chunk not found")
    user_file = _get_owned_user_file(db_session, chunk.user_file_id, user)

    if is_hierarchical_aggregate_chunk(chunk):
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            "Derived aggregate chunks cannot be edited directly.",
        )

    if update_request.text is not None and not update_request.text.strip():
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, "Chunk text cannot be emptied")

    validity_start: object = (
        None
        if update_request.clear_validity_start_date
        else (
            update_request.validity_start_date
            if update_request.validity_start_date is not None
            else "unset"
        )
    )
    validity_end: object = (
        None
        if update_request.clear_validity_end_date
        else (
            update_request.validity_end_date
            if update_request.validity_end_date is not None
            else "unset"
        )
    )

    delete_hierarchical_aggregates_referencing_chunk(
        db_session,
        user_file_id=chunk.user_file_id,
        source_chunk_id=chunk.id,
    )
    update_chunk(
        chunk,
        text=update_request.text,
        heading_path=update_request.heading_path,
        chunk_metadata=update_request.chunk_metadata,
        validity_start_date=validity_start,  # type: ignore[arg-type]
        validity_end_date=validity_end,  # type: ignore[arg-type]
    )
    db_session.flush()

    # Same transaction: if the Elasticsearch projection fails, the row edit rolls
    # back too, so Postgres and the index never diverge.
    project_user_file_to_index(db_session, user_file, get_current_tenant_id())
    db_session.commit()

    return RegulatoryChunkSnapshot.from_model(chunk)


@router.patch("/files/{user_file_id}/validity", tags=PUBLIC_API_TAGS)
def patch_file_validity(
    user_file_id: UUID,
    update_request: RegulatoryFileValidityUpdateRequest,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> RegulatoryFileValidityUpdateResponse:
    """Set explicit snapshot dates only when metadata projection is exact."""

    _get_owned_user_file(db_session, user_file_id, user)
    user_file = lock_completed_user_file_for_projection(db_session, user_file_id)
    if user_file is None:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            "Validity can only be updated for a completed file.",
        )
    if (
        update_request.validity_start_date is not None
        and update_request.clear_validity_start_date
    ) or (
        update_request.validity_end_date is not None
        and update_request.clear_validity_end_date
    ):
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            "A validity date cannot be set and cleared in the same request.",
        )

    updates_start = (
        update_request.validity_start_date is not None
        or update_request.clear_validity_start_date
    )
    updates_end = (
        update_request.validity_end_date is not None
        or update_request.clear_validity_end_date
    )
    if not updates_start and not updates_end:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            "At least one validity boundary must be set or cleared.",
        )

    validity_start: ValidityDateUpdate = (
        None
        if update_request.clear_validity_start_date
        else (update_request.validity_start_date if updates_start else "unset")
    )
    validity_end: ValidityDateUpdate = (
        None
        if update_request.clear_validity_end_date
        else (update_request.validity_end_date if updates_end else "unset")
    )
    try:
        result = update_file_validity_window(
            db_session,
            user_file_id,
            validity_start_date=validity_start,
            validity_end_date=validity_end,
        )
    except ValueError as error:
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(error)) from error

    if result.updated_chunk_count == 0:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            "The file has no unversioned indexed chunks to update.",
        )

    db_session.flush()
    if (
        result.skipped_versioned_chunk_count != 0
        or result.previous_window is None
        or result.updated_window is None
    ):
        db_session.rollback()
        raise OnyxError(
            OnyxErrorCode.CONFLICT,
            _FILE_VALIDITY_REINDEX_REQUIRED,
        )

    try:
        applied_metadata_patch = patch_user_file_validity_in_active_indices(
            db_session,
            user_file,
            previous_window=result.previous_window,
            updated_window=result.updated_window,
        )
    except Exception:
        db_session.rollback()
        raise
    if applied_metadata_patch is None:
        db_session.rollback()
        raise OnyxError(
            OnyxErrorCode.CONFLICT,
            _FILE_VALIDITY_REINDEX_REQUIRED,
        )
    try:
        db_session.commit()
    except Exception:
        logger.exception(
            "PostgreSQL commit failed after regulatory validity projection for "
            "user_file=%s",
            user_file_id,
        )
        if applied_metadata_patch is not None:
            try:
                compensate_regulatory_validity_patch(applied_metadata_patch)
            except Exception as compensation_error:
                db_session.rollback()
                raise RuntimeError(
                    "PostgreSQL commit and Elasticsearch validity compensation both "
                    f"failed for user_file={user_file_id}."
                ) from compensation_error
        db_session.rollback()
        raise

    return RegulatoryFileValidityUpdateResponse(
        updated_chunk_count=result.updated_chunk_count,
        skipped_versioned_chunk_count=result.skipped_versioned_chunk_count,
    )


@router.patch("/files/{user_file_id}", tags=PUBLIC_API_TAGS)
def rename_user_file(
    user_file_id: UUID,
    rename_request: UserFileRenameRequest,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> UserFileSnapshot:
    user_file = _get_owned_user_file(db_session, user_file_id, user)
    user_file.name = rename_request.name.strip()
    db_session.add(user_file)
    db_session.flush()

    # File name is embedded in each chunk's semantic identifier (citations),
    # so a rename re-projects the file's chunks as well.
    chunks = get_chunks_for_file(db_session, user_file_id)
    if chunks:
        project_user_file_to_index(db_session, user_file, get_current_tenant_id())
    db_session.commit()

    return UserFileSnapshot.from_model(user_file)


# =============================================================================
# Amendment (update) mechanism
#
# An admin/curator pastes amendment text scoped to a document set. It is
# segmented into atomic instructions, matched against the document set's chunks,
# and drafted into proposals — nothing writes to regulatory_chunk until a
# proposal is approved (approve_amendment_proposal owns that transaction).
# =============================================================================


def _source_extraction_snapshot(
    text: str, source_type: Literal["html", "pdf", "docx"], display_name: str
) -> AmendmentSourceExtractionSnapshot:
    return AmendmentSourceExtractionSnapshot(
        text=text,
        source_type=source_type,
        display_name=display_name,
    )


@router.post("/amendments/sources/url", tags=PUBLIC_API_TAGS)
def extract_amendment_url(
    source_request: AmendmentSourceUrlRequest,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
) -> AmendmentSourceExtractionSnapshot:
    del user
    try:
        extraction = fetch_and_extract_amendment_url(source_request.url)
    except AmendmentSourceExtractionError as exc:
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(exc)) from exc
    return _source_extraction_snapshot(
        extraction.text, extraction.source_type, extraction.display_name
    )


@router.post("/amendments/sources/pdf", tags=PUBLIC_API_TAGS)
def extract_amendment_pdf(
    file: UploadFile = File(...),
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
) -> AmendmentSourceExtractionSnapshot:
    del user
    file_name = file.filename or "amendment.pdf"
    content = file.file.read(MAX_AMENDMENT_SOURCE_BYTES + 1)
    try:
        text = extract_amendment_pdf_text(content, file_name)
    except AmendmentSourceExtractionError as exc:
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(exc)) from exc
    return _source_extraction_snapshot(text, "pdf", file_name)


@router.post("/amendments/sources/docx", tags=PUBLIC_API_TAGS)
def extract_amendment_docx(
    file: UploadFile = File(...),
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
) -> AmendmentSourceExtractionSnapshot:
    del user
    file_name = file.filename or "amendment.docx"
    content = file.file.read(MAX_AMENDMENT_SOURCE_BYTES + 1)
    try:
        text = extract_amendment_docx_text(content, file_name)
    except AmendmentSourceExtractionError as exc:
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(exc)) from exc
    return _source_extraction_snapshot(text, "docx", file_name)


def _get_editable_document_set(
    db_session: Session, document_set_id: int, user: User
) -> DocumentSet:
    document_set = get_document_set_by_id_for_user(
        db_session=db_session,
        document_set_id=document_set_id,
        user=user,
        get_editable=True,
    )
    if document_set is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Document set not found")
    return document_set


@router.post("/amendments/analyze", tags=PUBLIC_API_TAGS, status_code=202)
def analyze_amendment_text(
    analyze_request: AnalyzeAmendmentRequest,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> AmendmentBatchSnapshot:
    document_set = _get_editable_document_set(
        db_session, analyze_request.document_set_id, user
    )
    if analyze_request.source_package_id is not None:
        _require_annex_updates()
        try:
            package = require_ready_source_package(
                db_session,
                package_id=analyze_request.source_package_id,
                document_set_id=document_set.id,
                environment=annex_config.REGULATORY_ANNEX_ENVIRONMENT,
            )
            if package.created_by != user.id:
                raise ValueError("Source package owner mismatch")
        except ValueError as exc:
            raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(exc)) from exc
    user_file_ids = [user_file.id for user_file in document_set.user_files]
    if not user_file_ids:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            "This document set has no files to amend yet.",
        )

    # Commit before dispatch so every broker delivery points at a durable row.
    batch = create_batch(
        db_session,
        document_set_id=document_set.id,
        user_file_ids=user_file_ids,
        raw_text=analyze_request.raw_text,
        created_by=user.id,
    )
    if analyze_request.source_package_id is not None:
        attach_source_package_to_batch(
            db_session,
            batch=batch,
            package_id=analyze_request.source_package_id,
            environment=annex_config.REGULATORY_ANNEX_ENVIRONMENT,
        )
    db_session.commit()
    try:
        enqueue_amendment_batch(
            batch_id=batch.id,
            tenant_id=tenant_id,
        )
    except Exception:
        logger.exception(
            "Initial dispatch failed for amendment batch=%s; recovery will retry",
            batch.id,
        )
    return AmendmentBatchSnapshot.from_model(
        batch,
        annex_groups=list_annex_changes(db_session, batch.id)
        if batch.created_by == user.id
        else [],
    )


@router.get("/amendments/batches", tags=PUBLIC_API_TAGS)
def list_amendment_batches(
    document_set_id: int,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> list[AmendmentBatchSnapshot]:
    _get_editable_document_set(db_session, document_set_id, user)
    batches = list_batches_for_document_set(db_session, document_set_id)
    return [
        AmendmentBatchSnapshot.from_model(
            b,
            annex_groups=list_annex_changes(db_session, b.id)
            if b.created_by == user.id
            else [],
        )
        for b in batches
    ]


@router.get("/amendments/batches/{batch_id}/proposals", tags=PUBLIC_API_TAGS)
def list_amendment_proposals(
    batch_id: int,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> list[AmendmentProposalSnapshot]:
    batch = get_batch(db_session, batch_id)
    if batch is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Batch not found")
    _get_editable_document_set(db_session, batch.document_set_id, user)

    proposals = list_proposals_for_batch(db_session, batch_id)
    duplicates = compute_duplicate_targets(proposals)
    return [
        AmendmentProposalSnapshot.from_model(
            p, duplicate_target=duplicates.get(p.id, False)
        )
        for p in proposals
    ]


@router.get(
    "/amendments/batches/{batch_id}/analysis",
    tags=PUBLIC_API_TAGS,
)
def get_amendment_analysis(
    batch_id: int,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> AnalyzeAmendmentResponse:
    batch = get_batch(db_session, batch_id)
    if batch is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Amendment batch not found")
    _get_editable_document_set(db_session, batch.document_set_id, user)
    proposals = list_proposals_for_batch(db_session, batch_id)
    duplicates = compute_duplicate_targets(proposals)
    return AnalyzeAmendmentResponse(
        annex_groups=[
            AnnexReviewSnapshot.model_validate(group)
            for group in (
                list_annex_changes(db_session, batch_id)
                if batch.created_by == user.id
                else []
            )
        ],
        batch=AmendmentBatchSnapshot.from_model(
            batch,
            annex_groups=list_annex_changes(db_session, batch.id)
            if batch.created_by == user.id
            else [],
        ),
        proposals=[
            AmendmentProposalSnapshot.from_model(
                proposal,
                duplicate_target=duplicates.get(proposal.id, False),
            )
            for proposal in proposals
        ],
        unmatched_instructions=list(batch.unmatched_instructions),
    )


@router.post(
    "/amendments/batches/{batch_id}/retry",
    tags=PUBLIC_API_TAGS,
    status_code=202,
)
def retry_amendment_analysis(
    batch_id: int,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> AmendmentBatchSnapshot:
    batch = get_batch(db_session, batch_id)
    if batch is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Amendment batch not found")
    _get_editable_document_set(db_session, batch.document_set_id, user)
    retried = reset_failed_batch_for_retry(db_session, batch_id=batch_id)
    if retried is None:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            "Only failed amendment batches can be retried.",
        )
    try:
        enqueue_amendment_batch(batch_id=batch_id, tenant_id=tenant_id)
    except Exception:
        logger.exception(
            "Retry dispatch failed for amendment batch=%s; recovery will retry",
            batch_id,
        )
    return AmendmentBatchSnapshot.from_model(retried)


@router.post(
    "/amendments/proposals/{proposal_id}/approve",
    tags=PUBLIC_API_TAGS,
    status_code=202,
)
def approve_proposal(
    proposal_id: int,
    approval_request: ApproveAmendmentProposalRequest | None = None,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> AmendmentProposalSnapshot:
    proposal = get_proposal(db_session, proposal_id)
    if proposal is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Proposal not found")
    batch = get_batch(db_session, proposal.batch_id)
    assert batch is not None
    _get_editable_document_set(db_session, batch.document_set_id, user)
    if batch.status != AmendmentBatchStatus.ANALYZED.value:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            "Proposals can only be reviewed after analysis is complete.",
        )

    try:
        proposal = queue_amendment_proposal_approval(
            db_session,
            proposal,
            decided_by=user.id,
            reviewed_new_chunk_draft=(
                approval_request.new_chunk_draft if approval_request else None
            ),
        )
    except ValueError as e:
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(e)) from e
    db_session.commit()

    try:
        enqueue_amendment_proposal_approval(
            proposal_id=proposal.id,
            tenant_id=tenant_id,
        )
    except Exception as error:
        logger.exception("Approval dispatch failed for proposal=%s", proposal.id)
        reset_amendment_proposal_approval(
            db_session,
            proposal_id=proposal.id,
        )
        raise OnyxError(
            OnyxErrorCode.SERVICE_UNAVAILABLE,
            "Approval could not be queued. Please try again.",
        ) from error

    return AmendmentProposalSnapshot.from_model(proposal)


@router.post(
    "/amendments/proposals/{proposal_id}/retry",
    tags=PUBLIC_API_TAGS,
    status_code=202,
)
def retry_proposal_indexing(
    proposal_id: int,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> AmendmentProposalSnapshot:
    proposal = get_proposal(db_session, proposal_id)
    if proposal is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Proposal not found")
    batch = get_batch(db_session, proposal.batch_id)
    assert batch is not None
    _get_editable_document_set(db_session, batch.document_set_id, user)
    if batch.status != AmendmentBatchStatus.ANALYZED.value:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            "Proposals can only be retried after analysis is complete.",
        )
    try:
        proposal, should_enqueue = retry_amendment_proposal_projection(
            db_session,
            proposal_id=proposal_id,
        )
    except ValueError as error:
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(error)) from error
    db_session.commit()

    if should_enqueue:
        try:
            enqueue_amendment_proposal_approval(
                proposal_id=proposal.id,
                tenant_id=tenant_id,
            )
        except Exception as error:
            logger.exception(
                "Approval retry dispatch failed for proposal=%s", proposal.id
            )
            finalize_amendment_proposal_projection(
                db_session,
                proposal_id=proposal.id,
                succeeded=False,
                error_message="Indexing could not be queued. Please try again.",
            )
            db_session.commit()
            raise OnyxError(
                OnyxErrorCode.SERVICE_UNAVAILABLE,
                "Indexing could not be queued. Please try again.",
            ) from error

    return AmendmentProposalSnapshot.from_model(proposal)


@router.post("/amendments/proposals/{proposal_id}/reject", tags=PUBLIC_API_TAGS)
def reject_proposal_endpoint(
    proposal_id: int,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> AmendmentProposalSnapshot:
    proposal = get_proposal(db_session, proposal_id)
    if proposal is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Proposal not found")
    batch = get_batch(db_session, proposal.batch_id)
    assert batch is not None
    _get_editable_document_set(db_session, batch.document_set_id, user)
    if batch.status != AmendmentBatchStatus.ANALYZED.value:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            "Proposals can only be reviewed after analysis is complete.",
        )

    try:
        proposal = reject_proposal(db_session, proposal, decided_by=user.id)
    except ValueError as e:
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(e)) from e
    db_session.commit()

    return AmendmentProposalSnapshot.from_model(proposal)


def _require_annex_updates() -> None:
    if not annex_config.REGULATORY_ANNEX_UPDATES_ENABLED:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Annex source packages are disabled")


def _source_package_snapshot(
    db_session: Session, package_id: UUID, document_set_id: int
) -> AmendmentSourcePackageSnapshot:
    package = get_source_package(
        db_session,
        package_id=package_id,
        document_set_id=document_set_id,
        environment=annex_config.REGULATORY_ANNEX_ENVIRONMENT,
    )
    if package is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Source package not found")
    snapshot = AmendmentSourcePackageSnapshot.model_validate(package)
    snapshot.assets = [
        AmendmentSourceAssetSnapshot.model_validate(asset)
        for asset in list_source_assets(db_session, package_id)
    ]
    return snapshot


def _create_source_package_request(
    *,
    db_session: Session,
    document_set_id: int,
    idempotency_key: str,
    user: User,
    tenant_id: str,
    spec: dict[str, str],
    content: bytes | None,
) -> AmendmentSourcePackageSnapshot:
    request_hash = hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode() + b"\0" + (content or b"")
    ).hexdigest()
    store = get_default_file_store()
    input_file_id = (
        store.save_file(
            io.BytesIO(content),
            display_name=spec.get("display_name", "source"),
            file_origin=FileOrigin.OTHER,
            file_type=spec.get("mime_type", "application/octet-stream"),
        )
        if content is not None
        else None
    )
    try:
        package, created = create_source_package(
            db_session,
            document_set_id=document_set_id,
            environment=annex_config.REGULATORY_ANNEX_ENVIRONMENT,
            idempotency_key=idempotency_key,
            request_hash=request_hash,
            input_spec=spec,
            created_by=user.id,
            input_file_id=input_file_id,
        )
    except ValueError as exc:
        if input_file_id:
            store.delete_file(input_file_id)
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(exc)) from exc
    db_session.commit()
    if not created and input_file_id:
        store.delete_file(input_file_id)
    if created:
        try:
            enqueue_source_package(package_id=package.id, tenant_id=tenant_id)
        except Exception:
            mark_source_package_failed(
                db_session,
                package_id=package.id,
                environment=annex_config.REGULATORY_ANNEX_ENVIRONMENT,
            )
            logger.warning(
                "Source package dispatch failed; retry package %s", package.id
            )
    return _source_package_snapshot(db_session, package.id, document_set_id)


@router.post("/amendments/source-packages", tags=PUBLIC_API_TAGS, status_code=202)
def create_amendment_source_package(
    request: CreateAmendmentSourcePackageRequest,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> AmendmentSourcePackageSnapshot:
    _require_annex_updates()
    _get_editable_document_set(db_session, request.document_set_id, user)
    spec = (
        {"url": request.url}
        if request.url is not None
        else {"mime_type": "text/plain", "display_name": "amendment.txt"}
    )
    return _create_source_package_request(
        db_session=db_session,
        document_set_id=request.document_set_id,
        idempotency_key=request.idempotency_key,
        user=user,
        tenant_id=tenant_id,
        spec=spec,
        content=request.text.encode() if request.text is not None else None,
    )


@router.post(
    "/amendments/source-packages/upload", tags=PUBLIC_API_TAGS, status_code=202
)
def upload_amendment_source_package(
    file: UploadFile = File(...),
    document_set_id: int = Form(...),
    idempotency_key: str = Form(..., min_length=1, max_length=200),
    base_url: str | None = Form(default=None, max_length=8192),
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> AmendmentSourcePackageSnapshot:
    _require_annex_updates()
    _get_editable_document_set(db_session, document_set_id, user)
    content = file.file.read(MAX_ASSET_BYTES + 1)
    if not content or len(content) > MAX_ASSET_BYTES:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            "Source file must contain between 1 byte and 25 MiB",
        )
    spec = {
        "mime_type": file.content_type or "application/octet-stream",
        "display_name": file.filename or "source",
    }
    if base_url:
        spec["base_url"] = base_url
    return _create_source_package_request(
        db_session=db_session,
        document_set_id=document_set_id,
        idempotency_key=idempotency_key,
        user=user,
        tenant_id=tenant_id,
        spec=spec,
        content=content,
    )


@router.get("/amendments/source-packages/{package_id}", tags=PUBLIC_API_TAGS)
def get_amendment_source_package(
    package_id: UUID,
    document_set_id: int,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> AmendmentSourcePackageSnapshot:
    _require_annex_updates()
    _get_editable_document_set(db_session, document_set_id, user)
    return _source_package_snapshot(db_session, package_id, document_set_id)


@router.post(
    "/amendments/source-packages/{package_id}/retry",
    tags=PUBLIC_API_TAGS,
    status_code=202,
)
def retry_amendment_source_package(
    package_id: UUID,
    document_set_id: int,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> AmendmentSourcePackageSnapshot:
    _require_annex_updates()
    _get_editable_document_set(db_session, document_set_id, user)
    try:
        retry_source_package(
            db_session,
            package_id=package_id,
            document_set_id=document_set_id,
            environment=annex_config.REGULATORY_ANNEX_ENVIRONMENT,
        )
    except ValueError as exc:
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(exc)) from exc
    try:
        enqueue_source_package(package_id=package_id, tenant_id=tenant_id)
    except Exception:
        mark_source_package_failed(
            db_session,
            package_id=package_id,
            environment=annex_config.REGULATORY_ANNEX_ENVIRONMENT,
        )
        logger.warning("Source package retry dispatch failed for %s", package_id)
    return _source_package_snapshot(db_session, package_id, document_set_id)


@router.get(
    "/amendments/source-packages/{package_id}/assets/{asset_id}", tags=PUBLIC_API_TAGS
)
def download_amendment_source_asset(
    package_id: UUID,
    asset_id: UUID,
    document_set_id: int,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> Response:
    _require_annex_updates()
    _get_editable_document_set(db_session, document_set_id, user)
    _source_package_snapshot(db_session, package_id, document_set_id)
    asset = get_source_asset(db_session, package_id=package_id, asset_id=asset_id)
    if asset is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Source asset not found")
    with get_default_file_store().read_file(asset.file_id) as stream:
        content = stream.read(MAX_ASSET_BYTES + 1)
    if (
        len(content) != asset.byte_count
        or hashlib.sha256(content).hexdigest() != asset.sha256
    ):
        raise OnyxError(
            OnyxErrorCode.INTERNAL_ERROR, "Source asset integrity check failed"
        )
    return Response(
        content,
        media_type=asset.mime_type,
        headers={
            "Content-Disposition": 'attachment; filename="source"',
            "Content-Security-Policy": "sandbox",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/amendments/source-packages/{package_id}/evidence", tags=PUBLIC_API_TAGS)
def get_amendment_source_evidence(
    package_id: UUID,
    document_set_id: int,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> Response:
    _require_annex_updates()
    _get_editable_document_set(db_session, document_set_id, user)
    package = get_source_package(
        db_session,
        package_id=package_id,
        document_set_id=document_set_id,
        environment=annex_config.REGULATORY_ANNEX_ENVIRONMENT,
    )
    if package is None or package.manifest_file_id is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Source evidence is not available yet")
    with get_default_file_store().read_file(package.manifest_file_id) as stream:
        content = stream.read(150 * 1024 * 1024 + 1)
    if hashlib.sha256(content).hexdigest() != package.manifest_sha256:
        raise OnyxError(
            OnyxErrorCode.INTERNAL_ERROR, "Source manifest integrity check failed"
        )
    return Response(
        content,
        media_type="application/json",
        headers={"X-Content-Type-Options": "nosniff"},
    )


router.include_router(annex_router)
