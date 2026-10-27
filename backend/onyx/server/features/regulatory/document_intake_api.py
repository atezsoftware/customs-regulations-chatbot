"""Admin review and explicit publication of newly acquired regulatory documents."""

import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Query, Response, UploadFile
from sqlalchemy.orm import Session

from onyx.auth.permissions import require_permission
from onyx.configs.constants import PUBLIC_API_TAGS
from onyx.db import document_intake as repository
from onyx.db import regulatory_labeling as labeling_repository
from onyx.db.document_intake_sources import (
    create_markdown_source_package,
    create_zip_source_package,
    markdown_source_snapshot,
)
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import Permission
from onyx.db.labeling_configuration import (
    INTAKE_LABELING_CONFIGURATION_ERROR,
    INTAKE_LABELING_MODEL,
    get_intake_labeling_provider_options,
)
from onyx.db.models import RegulatoryDocumentIntake, User
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.regulatory.amendments.annexes import config
from onyx.regulatory.amendments.annexes.models import SourceLink
from onyx.regulatory.document_intake.api_models import (
    DocumentLineageSnapshot,
    IntakeApproveRequest,
    IntakeBatchSnapshot,
    IntakeCapabilities,
    IntakeChunkEditRequest,
    IntakeChunksEditRequest,
    IntakeCreateRequest,
    IntakeExistingParentCandidate,
    IntakePredecessorCandidate,
    IntakePredecessorSelectionRequest,
    IntakePublicationReviewRequest,
    IntakeReprepareRequest,
    IntakeReprepareSnapshot,
)
from onyx.regulatory.labeling.api_models import (
    LabelingCounts,
    LabelingProviderSummary,
    LabelingSetup,
)
from onyx.server.features.regulatory.models import (
    AmendmentSourcePackageSnapshot,
    IntakeChunkReviewPage,
    IntakeChunkReviewSnapshot,
    RegulatoryChunkSnapshot,
)
from onyx.utils.logger import setup_logger
from shared_configs.contextvars import get_current_tenant_id

router = APIRouter(prefix="/document-intake", tags=PUBLIC_API_TAGS)
logger = setup_logger()
_admin = require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)


def _require_enabled() -> None:
    if not config.REGULATORY_ANNEX_UPDATES_ENABLED:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "New document intake is disabled")


def _get_batch(
    session: Session, batch_id: UUID, user: User
) -> RegulatoryDocumentIntake:
    _require_enabled()
    try:
        return repository.get_intake(session, batch_id, user)
    except ValueError as exc:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, str(exc)) from exc


def _snapshot(session: Session, batch: RegulatoryDocumentIntake) -> IntakeBatchSnapshot:
    try:
        snapshot = IntakeBatchSnapshot.model_validate(
            repository.intake_snapshot(session, batch)
        )
        session.commit()
        return snapshot
    except repository.IntakeConflictError as exc:
        session.rollback()
        raise OnyxError(OnyxErrorCode.CONFLICT, str(exc)) from exc


def _enqueue_batch(
    session: Session, batch: RegulatoryDocumentIntake, tenant_id: str
) -> None:
    from onyx.background.celery.tasks.regulatory_amendments.document_intake import (
        enqueue_intake_batch,
    )

    try:
        if batch.stage == "reprepare":
            from onyx.db.document_intake_reprepare import latest_reprepare

            request = latest_reprepare(session, batch.id)
            if request is None:
                raise ValueError("No durable reprepare request exists")
            enqueue_intake_batch(
                batch_id=batch.id, tenant_id=tenant_id, reprepare_request_id=request.id
            )
        else:
            enqueue_intake_batch(batch_id=batch.id, tenant_id=tenant_id)
    except Exception:
        logger.exception("Document intake dispatch failed for batch=%s", batch.id)
        repository.mark_intake_dispatch_failed(
            session,
            batch,
            "Preparation delivery is pending; automatic recovery will retry",
        )
        session.commit()


def _enqueue_indexes(
    session: Session,
    batch: RegulatoryDocumentIntake,
    deliveries: list[tuple[UUID, UUID]],
    tenant_id: str,
) -> None:
    from onyx.background.celery.tasks.regulatory_amendments.document_intake import (
        enqueue_intake_indexes,
    )

    try:
        enqueue_intake_indexes(deliveries=deliveries, tenant_id=tenant_id)
    except Exception:
        logger.exception(
            "Document intake index dispatch failed; durable requests remain"
        )
        repository.mark_intake_index_dispatch_failed(
            session, batch, "Some index requests could not be queued; retry this import"
        )
        session.commit()


@router.get("/capabilities")
def intake_capabilities(
    user: User = Depends(_admin),
    db_session: Session = Depends(get_session),
) -> IntakeCapabilities:
    if not config.REGULATORY_ANNEX_UPDATES_ENABLED:
        return IntakeCapabilities(
            enabled=False, error="New document intake is disabled"
        )
    try:
        target = repository.resolve_intake_target(db_session, user)
    except (ValueError, repository.IntakeConflictError) as exc:
        return IntakeCapabilities(enabled=False, error=str(exc))
    return IntakeCapabilities(
        enabled=True,
        target_document_set_id=target.id,
        target_document_set_name=target.name,
    )


@router.get("/labeling-setup")
def intake_labeling_setup(
    user: User = Depends(_admin),
    db_session: Session = Depends(get_session),
) -> LabelingSetup:
    _require_enabled()
    try:
        repository.resolve_intake_target(db_session, user)
    except ValueError as exc:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, str(exc)) from exc
    options = get_intake_labeling_provider_options(db_session, user=user)
    providers = [option for option in options if not option["configuration_error"]][:1]
    configuration_errors = (
        []
        if providers
        else list(
            dict.fromkeys(
                option["configuration_error"]
                for option in options
                if option["configuration_error"]
            )
        )
        or [INTAKE_LABELING_CONFIGURATION_ERROR]
    )
    settings = labeling_repository.get_label_settings(db_session)
    return LabelingSetup(
        model=INTAKE_LABELING_MODEL,
        default_label_count=settings.taxonomy.label_count,
        taxonomies=[
            labeling_repository.taxonomy_summary(row)
            for row in labeling_repository.list_taxonomies(db_session)
        ],
        providers=[LabelingProviderSummary(**option) for option in providers],
        counts=LabelingCounts(files=0, canonical_chunks=0, derived_chunks=0),
        active_run_id=None,
        warnings=[],
        configuration_errors=configuration_errors,
    )


@router.get("/markdown-template")
def markdown_intake_template(
    user: User = Depends(_admin),
    db_session: Session = Depends(get_session),
) -> Response:
    from onyx.regulatory.document_intake.markdown_package import (
        build_markdown_package_template,
    )

    _require_enabled()
    try:
        repository.resolve_intake_target(db_session, user)
    except ValueError as exc:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, str(exc)) from exc
    return Response(
        content=build_markdown_package_template(),
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="legislation-markdown-example.zip"'
        },
    )


@router.post("/source-packages/markdown", status_code=202)
def upload_markdown_intake_package(
    file: UploadFile = File(...),
    idempotency_key: UUID = Form(...),
    user: User = Depends(_admin),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> AmendmentSourcePackageSnapshot:
    from onyx.background.celery.tasks.regulatory_amendments.sources import (
        enqueue_source_package,
    )
    from onyx.db.amendment_sources import mark_source_package_failed
    from onyx.regulatory.document_intake.markdown_package import (
        MAX_MARKDOWN_UPLOAD_BYTES,
    )

    _require_enabled()
    name = file.filename or ""
    if len(name) > 255 or not name.lower().endswith((".md", ".zip")):
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            "Upload a Markdown .md file or a Markdown .zip package",
        )
    content = file.file.read(MAX_MARKDOWN_UPLOAD_BYTES + 1)
    if not content or len(content) > MAX_MARKDOWN_UPLOAD_BYTES:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            "Markdown upload must contain between 1 byte and 16 MiB",
        )
    try:
        package, created = create_markdown_source_package(
            db_session,
            user=user,
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            idempotency_key=idempotency_key,
            display_name=name,
            content=content,
        )
        db_session.commit()
    except (ValueError, repository.IntakeConflictError) as exc:
        db_session.rollback()
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(exc)) from exc
    if created:
        try:
            enqueue_source_package(package_id=package.id, tenant_id=tenant_id)
        except Exception:
            logger.exception(
                "Markdown source dispatch failed for package=%s", package.id
            )
            mark_source_package_failed(
                db_session,
                package_id=package.id,
                environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            )
    return markdown_source_snapshot(db_session, package)


@router.post("/source-packages/zip", status_code=202)
def upload_zip_intake_package(
    file: UploadFile = File(...),
    idempotency_key: UUID = Form(...),
    user: User = Depends(_admin),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> AmendmentSourcePackageSnapshot:
    from onyx.background.celery.tasks.regulatory_amendments.sources import (
        enqueue_source_package,
    )
    from onyx.db.amendment_sources import mark_source_package_failed
    from onyx.regulatory.document_intake.zip_package import MAX_ZIP_UPLOAD_BYTES

    _require_enabled()
    name = file.filename or ""
    if len(name) > 255 or not name.lower().endswith(".zip"):
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT, "Upload a mixed document .zip package"
        )
    content = file.file.read(MAX_ZIP_UPLOAD_BYTES + 1)
    if not content or len(content) > MAX_ZIP_UPLOAD_BYTES:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            "ZIP upload must contain between 1 byte and 25 MiB",
        )
    try:
        package, created = create_zip_source_package(
            db_session,
            user=user,
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            idempotency_key=idempotency_key,
            display_name=name,
            content=content,
        )
        db_session.commit()
    except (ValueError, repository.IntakeConflictError) as exc:
        db_session.rollback()
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(exc)) from exc
    if created:
        try:
            enqueue_source_package(package_id=package.id, tenant_id=tenant_id)
        except Exception:
            logger.exception("ZIP source dispatch failed for package=%s", package.id)
            mark_source_package_failed(
                db_session,
                package_id=package.id,
                environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            )
    return markdown_source_snapshot(db_session, package)


@router.post("/batches", status_code=202)
def create_intake_batch(
    request: IntakeCreateRequest,
    user: User = Depends(_admin),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> IntakeBatchSnapshot:
    _require_enabled()
    try:
        batch, created = repository.create_intake(
            db_session,
            source_package_id=request.source_package_id,
            idempotency_key=request.idempotency_key,
            user=user,
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
        )
        db_session.commit()
    except repository.IntakeConflictError as exc:
        db_session.rollback()
        raise OnyxError(OnyxErrorCode.CONFLICT, str(exc)) from exc
    except ValueError as exc:
        db_session.rollback()
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(exc)) from exc
    if created:
        _enqueue_batch(db_session, batch, tenant_id)
    return _snapshot(db_session, batch)


@router.get("/source-packages/{package_id}/links")
def get_intake_source_links(
    package_id: UUID,
    user: User = Depends(_admin),
    db_session: Session = Depends(get_session),
) -> list[SourceLink]:
    _require_enabled()
    try:
        return repository.intake_source_links(
            db_session, package_id, user, config.REGULATORY_ANNEX_ENVIRONMENT
        )
    except ValueError as exc:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, str(exc)) from exc
    except repository.IntakeConflictError as exc:
        raise OnyxError(OnyxErrorCode.CONFLICT, str(exc)) from exc


@router.get("/batches")
def list_intake_batches(
    user: User = Depends(_admin),
    db_session: Session = Depends(get_session),
) -> list[IntakeBatchSnapshot]:
    _require_enabled()
    try:
        batches = repository.list_intakes(db_session, user)
    except ValueError as exc:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, str(exc)) from exc
    return [_snapshot(db_session, batch) for batch in batches]


@router.get("/batches/{batch_id}")
def get_intake_batch(
    batch_id: UUID,
    user: User = Depends(_admin),
    db_session: Session = Depends(get_session),
) -> IntakeBatchSnapshot:
    return _snapshot(db_session, _get_batch(db_session, batch_id, user))


@router.post("/batches/{batch_id}/retry", status_code=202)
def retry_intake_batch(
    batch_id: UUID,
    user: User = Depends(_admin),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> IntakeBatchSnapshot:
    batch = _get_batch(db_session, batch_id, user)
    try:
        deliveries = repository.retry_intake(db_session, batch)
        db_session.commit()
    except repository.IntakeConflictError as exc:
        db_session.rollback()
        raise OnyxError(OnyxErrorCode.CONFLICT, str(exc)) from exc
    except ValueError as exc:
        db_session.rollback()
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(exc)) from exc
    if deliveries:
        _enqueue_indexes(db_session, batch, deliveries, tenant_id)
    elif batch.stage == "labeling":
        _enqueue_edited_labels(db_session, batch, batch.labeling_run_id, tenant_id)
    elif batch.stage not in {"indexing", "lineage", "completed"}:
        _enqueue_batch(db_session, batch, tenant_id)
    return _snapshot(db_session, batch)


@router.get("/batches/{batch_id}/reprepare")
def get_intake_reprepare(
    batch_id: UUID,
    user: User = Depends(_admin),
    db_session: Session = Depends(get_session),
) -> IntakeReprepareSnapshot:
    from onyx.db.document_intake_reprepare import latest_reprepare, reprepare_snapshot

    batch = _get_batch(db_session, batch_id, user)
    previous = latest_reprepare(db_session, batch.id)
    audit = (
        dict(
            request_id=previous.id,
            status=previous.status,
            previous_labeling_run_id=previous.previous_labeling_run_id,
            documents={
                key: {
                    field: value
                    for field, value in entry.items()
                    if field != "canonical"
                }
                for key, entry in previous.documents.items()
            },
        )
        if previous
        else {}
    )
    try:
        view = reprepare_snapshot(db_session, batch)
        result = IntakeReprepareSnapshot.model_validate(
            {"eligible": True, "expected_sha256": view["expected_sha256"], **audit}
        )
        db_session.commit()
        return result
    except (repository.IntakeConflictError, ValueError) as exc:
        db_session.rollback()
        return IntakeReprepareSnapshot.model_validate(
            {"eligible": False, "reason": str(exc), **audit}
        )


@router.post("/batches/{batch_id}/reprepare", status_code=202)
def reprepare_intake_batch(
    batch_id: UUID,
    request: IntakeReprepareRequest,
    user: User = Depends(_admin),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> IntakeBatchSnapshot:
    from onyx.db.document_intake_reprepare import request_reprepare

    batch = _get_batch(db_session, batch_id, user)
    try:
        audit = request_reprepare(
            db_session,
            batch,
            expected_sha256=request.expected_sha256,
            idempotency_key=request.idempotency_key,
        )
        db_session.commit()
    except repository.IntakeConflictError as exc:
        db_session.rollback()
        raise OnyxError(OnyxErrorCode.CONFLICT, str(exc)) from exc
    except ValueError as exc:
        db_session.rollback()
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(exc)) from exc
    if audit.status in {"queued", "failed"}:
        _enqueue_batch(db_session, batch, tenant_id)
    return _snapshot(db_session, batch)


@router.post("/batches/{batch_id}/approve", status_code=202)
def approve_intake_batch(
    batch_id: UUID,
    request: IntakeApproveRequest,
    user: User = Depends(_admin),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> IntakeBatchSnapshot:
    batch = _get_batch(db_session, batch_id, user)
    try:
        deliveries = repository.approve_intake(
            db_session, batch, expected_review_sha256=request.expected_review_sha256
        )
        db_session.commit()
    except repository.IntakeConflictError as exc:
        db_session.rollback()
        raise OnyxError(OnyxErrorCode.CONFLICT, str(exc)) from exc
    except ValueError as exc:
        db_session.rollback()
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(exc)) from exc
    _enqueue_indexes(db_session, batch, deliveries, tenant_id)
    return _snapshot(db_session, batch)


@router.get("/batches/{batch_id}/documents/{document_id}/chunks")
def get_intake_chunk_page(
    batch_id: UUID,
    document_id: UUID,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    user: User = Depends(_admin),
    db_session: Session = Depends(get_session),
) -> IntakeChunkReviewPage:
    batch = _get_batch(db_session, batch_id, user)
    try:
        rows, total = repository.intake_chunk_page(
            db_session, batch, document_id, offset=offset, limit=limit
        )
    except ValueError as exc:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, str(exc)) from exc
    diagnostics = repository.intake_chunk_labeling_diagnostics(
        db_session, batch, [chunk.id for chunk, _labels in rows]
    )
    items: list[IntakeChunkReviewSnapshot] = []
    for chunk, labels in rows:
        item = IntakeChunkReviewSnapshot.model_validate(
            RegulatoryChunkSnapshot.from_model(chunk).model_dump()
        )
        item.chunk_metadata["labels"] = list(labels)
        diagnostic = diagnostics.get(chunk.id, {})
        item.labeling_status = diagnostic.get("status")
        item.labeling_error = diagnostic.get("error")
        items.append(item)
    return IntakeChunkReviewPage(items=items, total=total, offset=offset, limit=limit)


def _enqueue_edited_labels(
    db_session: Session,
    batch: RegulatoryDocumentIntake,
    run_id: UUID | None,
    tenant_id: str,
) -> None:
    if run_id is None:
        return
    from onyx.background.celery.tasks.regulatory_labeling.tasks import (
        enqueue_labeling_run,
    )

    try:
        enqueue_labeling_run(run_id=run_id, tenant_id=tenant_id)
    except Exception:
        logger.exception("Edited intake labeling dispatch failed batch=%s", batch.id)
        repository.mark_intake_label_dispatch_failed(db_session, batch.id)
        db_session.commit()


@router.put("/batches/{batch_id}/documents/{document_id}/publication-review")
def set_intake_publication_review(
    batch_id: UUID,
    document_id: UUID,
    request: IntakePublicationReviewRequest,
    user: User = Depends(_admin),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> IntakeBatchSnapshot:
    from onyx.db.document_intake_review import set_publication_review

    batch = _get_batch(db_session, batch_id, user)
    try:
        run_id = set_publication_review(
            db_session,
            batch,
            document_id=document_id,
            expected_review_sha256=request.expected_review_sha256,
            disposition=request.disposition,
            duplicate_of_document_id=request.duplicate_of_document_id,
            existing_parent_file_id=request.existing_parent_file_id,
            reason=request.reason,
            user=user,
        )
        db_session.commit()
    except (
        repository.IntakeConflictError,
        labeling_repository.LabelingStateConflictError,
    ) as exc:
        db_session.rollback()
        raise OnyxError(OnyxErrorCode.CONFLICT, str(exc)) from exc
    except ValueError as exc:
        db_session.rollback()
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(exc)) from exc
    _enqueue_edited_labels(db_session, batch, run_id, tenant_id)
    return _snapshot(db_session, batch)


@router.get("/batches/{batch_id}/documents/{document_id}/existing-parent-candidates")
def list_intake_existing_parent_candidates(
    batch_id: UUID,
    document_id: UUID,
    search: str | None = Query(default=None, max_length=500),
    limit: int = Query(default=20, ge=1, le=50),
    user: User = Depends(_admin),
    db_session: Session = Depends(get_session),
) -> list[IntakeExistingParentCandidate]:
    from onyx.db.document_intake_review import existing_parent_candidates

    batch = _get_batch(db_session, batch_id, user)
    try:
        return [
            IntakeExistingParentCandidate.model_validate(row)
            for row in existing_parent_candidates(
                db_session, batch, document_id, search, limit
            )
        ]
    except ValueError as exc:
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(exc)) from exc


@router.patch("/batches/{batch_id}/chunks")
def edit_intake_chunks(
    batch_id: UUID,
    request: IntakeChunksEditRequest,
    user: User = Depends(_admin),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> IntakeBatchSnapshot:
    from onyx.db.document_intake_edit import IntakeChunkEdit
    from onyx.db.document_intake_edit import edit_intake_chunks as edit_chunks

    batch = _get_batch(db_session, batch_id, user)
    try:
        run_id = edit_chunks(
            db_session,
            batch,
            expected_review_sha256=request.expected_review_sha256,
            edits=[
                IntakeChunkEdit(
                    document_id=edit.document_id,
                    chunk_id=edit.chunk_id,
                    updates=edit.updates.model_dump(exclude_unset=True),
                )
                for edit in request.edits
            ],
        )
        db_session.commit()
    except (
        repository.IntakeConflictError,
        labeling_repository.LabelingStateConflictError,
    ) as exc:
        db_session.rollback()
        raise OnyxError(OnyxErrorCode.CONFLICT, str(exc)) from exc
    except ValueError as exc:
        db_session.rollback()
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(exc)) from exc
    _enqueue_edited_labels(db_session, batch, run_id, tenant_id)
    return _snapshot(db_session, batch)


@router.patch("/batches/{batch_id}/documents/{document_id}/chunks/{chunk_id}")
def edit_intake_chunk(
    batch_id: UUID,
    document_id: UUID,
    chunk_id: str,
    request: IntakeChunkEditRequest,
    user: User = Depends(_admin),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> IntakeBatchSnapshot:
    from onyx.db.document_intake_edit import edit_intake_chunk as edit_chunk

    batch = _get_batch(db_session, batch_id, user)
    try:
        run_id = edit_chunk(
            db_session,
            batch,
            document_id=document_id,
            chunk_id=chunk_id,
            expected_review_sha256=request.expected_review_sha256,
            updates=request.model_dump(
                exclude_unset=True, exclude={"expected_review_sha256"}
            ),
        )
        db_session.commit()
    except (
        repository.IntakeConflictError,
        labeling_repository.LabelingStateConflictError,
    ) as exc:
        db_session.rollback()
        raise OnyxError(OnyxErrorCode.CONFLICT, str(exc)) from exc
    except ValueError as exc:
        db_session.rollback()
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(exc)) from exc
    _enqueue_edited_labels(db_session, batch, run_id, tenant_id)
    return _snapshot(db_session, batch)


@router.get("/batches/{batch_id}/documents/{document_id}/predecessor-candidates")
def get_intake_predecessor_candidates(
    batch_id: UUID,
    document_id: UUID,
    query: str | None = Query(default=None, max_length=200),
    user: User = Depends(_admin),
    db_session: Session = Depends(get_session),
) -> list[IntakePredecessorCandidate]:
    from onyx.db.document_lineage import predecessor_candidates

    batch = _get_batch(db_session, batch_id, user)
    try:
        return [
            IntakePredecessorCandidate.model_validate(candidate)
            for candidate in predecessor_candidates(
                db_session, batch, document_id, query=query
            )
        ]
    except repository.IntakeConflictError as exc:
        raise OnyxError(OnyxErrorCode.CONFLICT, str(exc)) from exc
    except ValueError as exc:
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(exc)) from exc


@router.put("/batches/{batch_id}/documents/{document_id}/predecessor-selection")
def select_intake_predecessor(
    batch_id: UUID,
    document_id: UUID,
    request: IntakePredecessorSelectionRequest,
    user: User = Depends(_admin),
    db_session: Session = Depends(get_session),
) -> IntakeBatchSnapshot:
    from onyx.db.document_lineage import set_predecessor_selection

    batch = _get_batch(db_session, batch_id, user)
    try:
        repository.refresh_intake(db_session, batch)
        if (
            batch.status != "ready"
            or batch.review_sha256 != request.expected_review_sha256
            or repository.review_digest(db_session, batch)
            != request.expected_review_sha256
        ):
            raise repository.IntakeConflictError(
                "The document review changed; reload before choosing a predecessor"
            )
        set_predecessor_selection(
            db_session,
            batch,
            document_id,
            predecessor_user_file_id=request.predecessor_user_file_id,
            relation=request.relation,
            effective_date=request.effective_date,
            evidence=request.evidence,
        )
        batch.review_sha256 = repository.review_digest(db_session, batch)
        db_session.commit()
    except repository.IntakeConflictError as exc:
        db_session.rollback()
        raise OnyxError(OnyxErrorCode.CONFLICT, str(exc)) from exc
    except ValueError as exc:
        db_session.rollback()
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(exc)) from exc
    return _snapshot(db_session, batch)


@router.get("/files/{file_id}/lineage")
def get_document_lineage(
    file_id: UUID,
    as_of: datetime.date | None = None,
    user: User = Depends(_admin),
    db_session: Session = Depends(get_session),
) -> DocumentLineageSnapshot:
    from onyx.db.document_lineage import file_lineage_snapshot, reconcile_file_lineage

    _require_enabled()
    try:
        target = repository.resolve_intake_target(db_session, user)
        reconcile_file_lineage(db_session, target.id, file_id)
        from onyx.db.document_intake_review import file_attachment_relations

        snapshot = DocumentLineageSnapshot.model_validate(
            {
                **file_lineage_snapshot(db_session, target.id, file_id, as_of=as_of),
                **file_attachment_relations(db_session, target.id, file_id),
            }
        )
        db_session.commit()
        return snapshot
    except ValueError as exc:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, str(exc)) from exc
