"""Authorized immutable grouped Updates review and source-text revision APIs."""

from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from sqlalchemy.orm import Session

from onyx.auth.permissions import require_permission
from onyx.background.celery.tasks.regulatory_amendments.annex_publication import (
    enqueue_annex_publication,
)
from onyx.background.celery.tasks.regulatory_amendments.tasks import (
    enqueue_amendment_batch,
)
from onyx.db.amendment_sources import list_source_assets, require_ready_source_package
from onyx.db.document_set import get_document_set_by_id_for_user
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import Permission
from onyx.db.models import AmendmentBatch, AnnexChangeSet, User
from onyx.db.regulatory_amendments import get_batch
from onyx.db.regulatory_annex_changes import (
    create_source_text_revision,
    get_annex_review_evidence,
    get_scoped_annex_review,
    list_annex_changes,
    list_annex_review_revisions,
    queue_annex_publication,
    reject_annex_review,
)
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.file_store.file_store import get_default_file_store
from onyx.regulatory.amendments.annexes import config
from onyx.regulatory.amendments.annexes.corrections import (
    read_frozen_evidence,
)
from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft
from onyx.server.features.regulatory.models import (
    AmendmentBatchSnapshot,
    AmendmentSourceTextRevisionRequest,
    AnnexCapabilities,
    AnnexReviewDecisionRequest,
    AnnexReviewEditRequest,
    AnnexReviewSnapshot,
    AnnexSourceTextSnapshot,
)
from onyx.utils.logger import setup_logger
from shared_configs.contextvars import get_current_tenant_id

router = APIRouter(prefix="/amendments")
logger = setup_logger()


def _authorized_batch(session: Session, batch_id: int, user: User) -> AmendmentBatch:
    if not config.REGULATORY_ANNEX_UPDATES_ENABLED:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Annex updates are disabled")
    batch = get_batch(session, batch_id)
    if (
        batch is None
        or batch.created_by != user.id
        or get_document_set_by_id_for_user(
            db_session=session,
            document_set_id=batch.document_set_id,
            user=user,
            get_editable=True,
        )
        is None
    ):
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Amendment batch not found")
    return batch


def _authorized_review(
    session: Session, batch_id: int, review_id: UUID, user: User
) -> AnnexChangeSet:
    _authorized_batch(session, batch_id, user)
    try:
        return get_scoped_annex_review(
            session,
            change_set_id=review_id,
            batch_id=batch_id,
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            created_by=user.id,
        )
    except ValueError as exc:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, str(exc)) from exc


@router.get("/capabilities")
def annex_capabilities(
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
) -> AnnexCapabilities:
    del user
    enabled = config.REGULATORY_ANNEX_UPDATES_ENABLED
    return AnnexCapabilities(
        enabled=enabled,
        grouped_review=enabled,
        immutable_review_revisions=enabled,
        asynchronous_source_preparation=enabled,
    )


@router.get("/batches/{batch_id}/annex-groups")
def list_groups(
    batch_id: int,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> list[AnnexReviewSnapshot]:
    _authorized_batch(db_session, batch_id, user)
    return [
        AnnexReviewSnapshot.model_validate(group)
        for group in list_annex_changes(db_session, batch_id)
    ]


@router.get("/batches/{batch_id}/annex-groups/{review_id}")
def get_group(
    batch_id: int,
    review_id: UUID,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> AnnexReviewSnapshot:
    return AnnexReviewSnapshot.model_validate(
        _authorized_review(db_session, batch_id, review_id, user)
    )


@router.get("/batches/{batch_id}/annex-groups/{review_id}/revisions")
def get_revisions(
    batch_id: int,
    review_id: UUID,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> list[AnnexReviewSnapshot]:
    review = _authorized_review(db_session, batch_id, review_id, user)
    return [
        AnnexReviewSnapshot.model_validate(item)
        for item in list_annex_review_revisions(
            db_session, logical_group_id=review.logical_group_id
        )
    ]


@router.get("/batches/{batch_id}/annex-groups/{review_id}/evidence/{evidence_id}")
def get_evidence(
    batch_id: int,
    review_id: UUID,
    evidence_id: UUID,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> Response:
    batch = _authorized_batch(db_session, batch_id, user)
    _authorized_review(db_session, batch_id, review_id, user)
    try:
        evidence = get_annex_review_evidence(
            db_session,
            change_set_id=review_id,
            evidence_id=evidence_id,
            document_set_id=batch.document_set_id,
            created_by=user.id,
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
        )
        content = read_frozen_evidence(get_default_file_store(), evidence)
    except ValueError as exc:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, str(exc)) from exc
    return Response(
        content,
        media_type=evidence.mime_type,
        headers={
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": "attachment; filename=annex-evidence",
            "Content-Security-Policy": "sandbox",
        },
    )


@router.post("/batches/{batch_id}/annex-groups/{review_id}/edit", status_code=202)
@router.post("/batches/{batch_id}/annex-groups/{review_id}/revalidate", status_code=202)
def edit_group(
    batch_id: int,
    review_id: UUID,
    request: AnnexReviewEditRequest,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> AnnexReviewSnapshot:
    from onyx.background.celery.tasks.regulatory_amendments.annex_preparation import (
        enqueue_review_preparation,
    )
    from onyx.db.regulatory_annex_preparation import queue_review_preparation

    _authorized_review(db_session, batch_id, review_id, user)
    try:
        review = queue_review_preparation(
            db_session,
            review_id=review_id,
            expected_review_sha256=request.expected_review_sha256,
            corrections=request.corrections,
            corrected_by=user.id,
            tenant_id=get_current_tenant_id(),
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            database_identity=config.ANNEX_DATABASE_IDENTITY,
        )
    except ValueError as exc:
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(exc)) from exc
    try:
        enqueue_review_preparation(
            review_id=review.id, tenant_id=get_current_tenant_id()
        )
    except Exception:
        logger.exception(
            "Annex review preparation dispatch deferred review=%s", review.id
        )
    return AnnexReviewSnapshot.model_validate(review)


def _queue_review(
    session: Session,
    batch_id: int,
    review_id: UUID,
    request: AnnexReviewDecisionRequest,
    user: User,
    tenant_id: str,
    retry: bool,
) -> AnnexReviewSnapshot:
    _authorized_review(session, batch_id, review_id, user)
    try:
        from onyx.db.regulatory_annex_execution import validate_publication_retry
        from onyx.regulatory.amendments.annexes.analysis import (
            validate_live_review_configuration,
            validate_live_review_runtime,
        )

        review = _authorized_review(session, batch_id, review_id, user)
        recovery = retry and validate_publication_retry(
            session,
            change_set_id=review_id,
            expected_review_sha256=request.expected_review_sha256,
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            tenant_id=tenant_id,
            database_identity=config.ANNEX_DATABASE_IDENTITY,
        )
        draft = AnnexChangeDraft.model_validate(review.review_payload)
        if recovery:
            validate_live_review_runtime(draft)
        else:
            validate_live_review_configuration(draft)
        intent = queue_annex_publication(
            session,
            change_set_id=review_id,
            expected_review_sha256=request.expected_review_sha256,
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            tenant_id=tenant_id,
            database_identity=config.ANNEX_DATABASE_IDENTITY,
            decided_by=user.id,
            retry=retry,
        )
        if recovery:
            # The producer may return an existing active intent without committing.
            session.commit()
    except ValueError as exc:
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(exc)) from exc
    try:
        enqueue_annex_publication(intent)
    except Exception:
        logger.exception(
            "Annex publication dispatch failed; immutable intent=%s remains queued",
            intent.id,
        )
    return AnnexReviewSnapshot.model_validate(
        _authorized_review(session, batch_id, review_id, user)
    )


@router.post("/batches/{batch_id}/annex-groups/{review_id}/approve")
def approve_group(
    batch_id: int,
    review_id: UUID,
    request: AnnexReviewDecisionRequest,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> AnnexReviewSnapshot:
    return _queue_review(
        db_session, batch_id, review_id, request, user, tenant_id, False
    )


@router.post("/batches/{batch_id}/annex-groups/{review_id}/retry")
def retry_group(
    batch_id: int,
    review_id: UUID,
    request: AnnexReviewDecisionRequest,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> AnnexReviewSnapshot:
    review = _authorized_review(db_session, batch_id, review_id, user)
    if (
        review.status in ("pending", "blocked", "rejected", "failed")
        and not review.publication_generation
    ):
        draft = AnnexChangeDraft.model_validate(review.review_payload)
        if (
            draft.issues
            or draft.publication is None
            or (
                review.preparation is not None
                and review.preparation.status in ("queued", "running", "failed")
            )
        ):
            return edit_group(
                batch_id=batch_id,
                review_id=review_id,
                request=AnnexReviewEditRequest(
                    expected_review_sha256=request.expected_review_sha256
                ),
                user=user,
                db_session=db_session,
            )
        from onyx.db.regulatory_annex_changes import resume_unpublished_annex_review
        from onyx.regulatory.amendments.annexes.analysis import (
            validate_live_review_configuration,
        )

        try:
            draft = AnnexChangeDraft.model_validate(review.review_payload)
            if not draft.issues and draft.impact is not None and draft.impact.ready:
                validate_live_review_configuration(draft)
            resumed = resume_unpublished_annex_review(
                db_session,
                change_set_id=review_id,
                expected_review_sha256=request.expected_review_sha256,
                environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            )
            return AnnexReviewSnapshot.model_validate(resumed)
        except ValueError as exc:
            raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(exc)) from exc
    return _queue_review(
        db_session, batch_id, review_id, request, user, tenant_id, True
    )


@router.post("/batches/{batch_id}/annex-groups/{review_id}/reject")
def reject_group(
    batch_id: int,
    review_id: UUID,
    request: AnnexReviewDecisionRequest,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> AnnexReviewSnapshot:
    _authorized_review(db_session, batch_id, review_id, user)
    try:
        review = reject_annex_review(
            db_session,
            change_set_id=review_id,
            expected_review_sha256=request.expected_review_sha256,
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            decided_by=user.id,
        )
    except ValueError as exc:
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(exc)) from exc
    return AnnexReviewSnapshot.model_validate(review)


@router.post("/batches/{batch_id}/source-revisions")
def edit_source_text(
    batch_id: int,
    request: AmendmentSourceTextRevisionRequest,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> AmendmentBatchSnapshot:
    _authorized_batch(db_session, batch_id, user)
    try:
        revised = create_source_text_revision(
            db_session,
            batch_id=batch_id,
            raw_text=request.raw_text,
            expected_source_text_sha256=request.expected_source_text_sha256,
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            created_by=user.id,
            source_package_id=request.source_package_id,
        )
    except ValueError as exc:
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(exc)) from exc
    try:
        enqueue_amendment_batch(batch_id=revised.id, tenant_id=tenant_id)
    except Exception:
        logger.exception(
            "Source revision batch=%s awaits recovery dispatch", revised.id
        )
    return AmendmentBatchSnapshot.from_model(revised)


@router.get("/source-packages/{package_id}/text")
def get_source_text(
    package_id: UUID,
    document_set_id: int,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> AnnexSourceTextSnapshot:
    from onyx.regulatory.amendments.annexes.analysis import (
        read_original_source_text,
        read_source_graph,
    )

    if (
        not config.REGULATORY_ANNEX_UPDATES_ENABLED
        or get_document_set_by_id_for_user(
            db_session=db_session,
            document_set_id=document_set_id,
            user=user,
            get_editable=True,
        )
        is None
    ):
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Document set not found")
    try:
        package = require_ready_source_package(
            db_session,
            package_id=package_id,
            document_set_id=document_set_id,
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
        )
        if package.created_by != user.id:
            raise ValueError("source package owner mismatch")
        store = get_default_file_store()
        assets = list_source_assets(db_session, package.id)
        # A source without a frozen manifest (e.g. a plain pasted-text package)
        # has no link graph to mark attachments with; fall back to the plain
        # join rather than failing the whole preview.
        links = (
            read_source_graph(
                store,
                manifest_file_id=package.manifest_file_id,
                manifest_sha256=package.manifest_sha256,
                assets=assets,
            )
            if package.manifest_file_id and package.manifest_sha256
            else None
        )
        text, digest = read_original_source_text(store, assets, links=links)
    except ValueError as exc:
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(exc)) from exc
    return AnnexSourceTextSnapshot(
        package_id=package.id,
        manifest_sha256=package.manifest_sha256 or "",
        original_text=text,
        original_text_sha256=digest,
    )
