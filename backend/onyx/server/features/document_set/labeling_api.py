from __future__ import annotations

from uuid import UUID, uuid5

from fastapi import APIRouter, Depends, Query
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from onyx.auth.permissions import require_permission
from onyx.db import regulatory_labeling as repository
from onyx.db.document_set import get_document_set_by_id_for_user
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import Permission
from onyx.db.labeling_configuration import (
    LabelingBatchGateway,
    get_labeling_provider_options,
    resolve_labeling_gateway,
    resolve_labeling_provider_binding,
)
from onyx.db.models import RegulatoryLabelingRun, RegulatoryLabelSettings, User
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.regulatory.indexing_jobs.models import (
    IndexingGatewayError,
    IndexingGatewayHTTPError,
)
from onyx.regulatory.labeling.api_models import (
    LabelingItemsPage,
    LabelingProviderSummary,
    LabelingRunCreate,
    LabelingRunSnapshot,
    LabelingSetup,
    LabelSettingsSnapshot,
    LabelSettingsUpdate,
    TaxonomyCreate,
    TaxonomySummary,
)
from onyx.regulatory.labeling.gemini_inline_batch import GeminiInlineBatchAccessError
from onyx.regulatory.labeling.provider import DEFAULT_MODEL, TaxonomyDefinition
from onyx.utils.logger import setup_logger
from shared_configs.contextvars import get_current_tenant_id

router = APIRouter(prefix="/manage/admin/document-set/{document_set_id}/labeling")
labeling_admin = require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)
logger = setup_logger()


def _probe_labeling_gateway(gateway: LabelingBatchGateway) -> None:
    try:
        gateway.probe_gemini_read_access()
    except GeminiInlineBatchAccessError as error:
        detail = (
            "Enable the Gemini Developer API for the Batch key's project before starting labeling."
            if error.reason_code == "SERVICE_DISABLED"
            else "Gemini Batch access could not be verified. Check this connection's Batch API key and project access in Language Models."
        )
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, detail) from None
    except IndexingGatewayHTTPError as error:
        if error.status_code in {400, 401, 403, 404}:
            raise OnyxError(
                OnyxErrorCode.INVALID_INPUT,
                "Gemini Vertex Batch access could not be verified. Check this connection's service account, project, location, and Cloud Storage access in Language Models.",
            ) from None
        raise OnyxError(
            OnyxErrorCode.BAD_GATEWAY,
            "Gemini Batch access could not be checked. Try again shortly.",
        ) from None
    except IndexingGatewayError:
        raise OnyxError(
            OnyxErrorCode.BAD_GATEWAY,
            "Gemini Batch access could not be checked. Try again shortly.",
        ) from None


def _check_access(session: Session, document_set_id: int, user: User) -> None:
    if (
        get_document_set_by_id_for_user(
            db_session=session,
            document_set_id=document_set_id,
            user=user,
            get_editable=True,
        )
        is None
    ):
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Document set not found")


def _get_run(
    session: Session, document_set_id: int, run_id: UUID
) -> RegulatoryLabelingRun:
    run = repository.get_run(session, document_set_id=document_set_id, run_id=run_id)
    if run is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Labeling run not found")
    return run


def _wake(run_id: UUID, tenant_id: str) -> None:
    from onyx.background.celery.tasks.regulatory_labeling.tasks import (
        enqueue_labeling_run,
    )

    try:
        enqueue_labeling_run(run_id=run_id, tenant_id=tenant_id)
    except Exception:
        logger.warning(
            "Labeling broker delivery failed; durable recovery will retry",
            extra={"run_id": str(run_id)},
        )


def _label_settings_snapshot(
    settings: RegulatoryLabelSettings,
) -> LabelSettingsSnapshot:
    taxonomy = TaxonomyDefinition.model_validate(settings.taxonomy.definition)
    return LabelSettingsSnapshot(
        revision=settings.revision,
        taxonomy_id=str(settings.taxonomy_id),
        labels=taxonomy.labels,
        updated_at=settings.updated_at,
    )


@router.get("/setup")
def labeling_setup(
    document_set_id: int,
    user: User = Depends(labeling_admin),
    db_session: Session = Depends(get_session),
) -> LabelingSetup:
    _check_access(db_session, document_set_id, user)
    counts, warnings = repository.get_labeling_counts(db_session, document_set_id)
    providers = get_labeling_provider_options(db_session, user=user)
    label_settings = repository.get_label_settings(db_session)
    if not providers:
        warnings.append(
            "Configure an accessible Gemini connection with an enabled model before starting labeling."
        )
    active_run = repository.get_active_run_id(db_session, document_set_id)
    return LabelingSetup(
        model=DEFAULT_MODEL,
        default_label_count=label_settings.taxonomy.label_count,
        taxonomies=[
            repository.taxonomy_summary(row)
            for row in repository.list_taxonomies(db_session)
        ],
        providers=[LabelingProviderSummary(**option) for option in providers],
        counts=counts,
        warnings=warnings,
        active_run_id=str(active_run) if active_run is not None else None,
    )


@router.get("/label-settings")
def get_label_settings(
    document_set_id: int,
    user: User = Depends(labeling_admin),
    db_session: Session = Depends(get_session),
) -> LabelSettingsSnapshot:
    _check_access(db_session, document_set_id, user)
    return _label_settings_snapshot(repository.get_label_settings(db_session))


@router.put("/label-settings")
def update_label_settings(
    document_set_id: int,
    body: LabelSettingsUpdate,
    user: User = Depends(labeling_admin),
    db_session: Session = Depends(get_session),
) -> LabelSettingsSnapshot:
    _check_access(db_session, document_set_id, user)
    try:
        settings = repository.update_label_settings(
            db_session,
            labels=body.labels,
            expected_revision=body.expected_revision,
            updated_by_id=user.id,
        )
        result = _label_settings_snapshot(settings)
        db_session.commit()
    except repository.LabelingStateConflictError as error:
        db_session.rollback()
        raise OnyxError(OnyxErrorCode.CONFLICT, str(error)) from None
    except ValueError as error:
        db_session.rollback()
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(error)) from None
    except IntegrityError:
        db_session.rollback()
        raise OnyxError(
            OnyxErrorCode.CONFLICT,
            "Label settings changed concurrently; reload the latest labels",
        ) from None
    return result


@router.post("/taxonomies")
def upload_label_taxonomy(
    document_set_id: int,
    body: TaxonomyCreate,
    user: User = Depends(labeling_admin),
    db_session: Session = Depends(get_session),
) -> TaxonomySummary:
    _check_access(db_session, document_set_id, user)
    try:
        taxonomy = TaxonomyDefinition(name=body.name, labels=body.labels)
        row = repository.create_taxonomy(
            db_session, taxonomy=taxonomy, created_by_id=user.id
        )
        result = repository.taxonomy_summary(row)
        db_session.commit()
    except ValueError:
        db_session.rollback()
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            "Invalid taxonomy: use unique label IDs and keep the vocabulary within 256 KiB",
        ) from None
    except IntegrityError:
        db_session.rollback()
        raise OnyxError(
            OnyxErrorCode.CONFLICT,
            "This taxonomy was uploaded concurrently; refresh the available versions",
        ) from None
    return result


def _start_run(
    session: Session,
    document_set_id: int,
    body: LabelingRunCreate,
    user: User,
    tenant_id: str,
    *,
    retry_of_id: UUID | None = None,
) -> LabelingRunSnapshot:
    existing = repository.get_run_by_idempotency(
        session, document_set_id, body.idempotency_key
    )
    if existing is not None:
        original_configuration_id = existing.provider_binding.get(
            "model_configuration_id"
        )
        if body.taxonomy_id is not None:
            same_taxonomy = existing.taxonomy_id == body.taxonomy_id
        elif existing.uses_current_labels:
            same_taxonomy = True
        else:
            same_taxonomy = (
                existing.taxonomy_id
                == repository.get_label_settings(session).taxonomy_id
            )
        if (
            not same_taxonomy
            or original_configuration_id != body.model_configuration_id
        ):
            raise OnyxError(
                OnyxErrorCode.CONFLICT,
                "The idempotency key was already used with different parameters",
            )
        return repository.run_snapshot(existing)
    taxonomy = (
        repository.get_taxonomy(session, body.taxonomy_id)
        if body.taxonomy_id is not None
        else (repository.get_label_settings(session).taxonomy)
    )
    if taxonomy is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Label taxonomy not found")
    try:
        binding = resolve_labeling_provider_binding(
            session, body.model_configuration_id, user=user
        )
        gateway = resolve_labeling_gateway(
            session, body.model_configuration_id, user=user, expected_binding=binding
        )
        taxonomy_id = taxonomy.id
        session.commit()
        _probe_labeling_gateway(gateway)
        _check_access(session, document_set_id, user)
        current_binding = resolve_labeling_provider_binding(
            session, body.model_configuration_id, user=user
        )
        if current_binding.fingerprint != binding.fingerprint:
            raise repository.LabelingStateConflictError(
                "The Gemini Batch connection changed while access was checked; start again."
            )
        taxonomy = repository.get_taxonomy(session, taxonomy_id)
        if taxonomy is None:
            raise ValueError("The selected label snapshot is no longer available")
        run, _created = repository.create_labeling_run(
            session,
            document_set_id=document_set_id,
            taxonomy=taxonomy,
            model_configuration_id=body.model_configuration_id,
            model=DEFAULT_MODEL,
            provider_binding=binding.model_dump(mode="json"),
            requested_by_id=user.id,
            idempotency_key=body.idempotency_key,
            retry_of_id=retry_of_id,
            uses_current_labels=body.taxonomy_id is None,
        )
        result = repository.run_snapshot(run)
        session.commit()
    except repository.LabelingStateConflictError as error:
        session.rollback()
        raise OnyxError(OnyxErrorCode.CONFLICT, str(error)) from None
    except ValueError as error:
        session.rollback()
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(error)) from None
    except IntegrityError:
        session.rollback()
        raise OnyxError(
            OnyxErrorCode.CONFLICT,
            "Another labeling request changed this document set; refresh its runs",
        ) from None
    _wake(UUID(result.id), tenant_id)
    return result


@router.post("/runs")
def start_labeling(
    document_set_id: int,
    body: LabelingRunCreate,
    user: User = Depends(labeling_admin),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> LabelingRunSnapshot:
    _check_access(db_session, document_set_id, user)
    return _start_run(db_session, document_set_id, body, user, tenant_id)


@router.get("/runs")
def list_labeling_runs(
    document_set_id: int,
    user: User = Depends(labeling_admin),
    db_session: Session = Depends(get_session),
) -> list[LabelingRunSnapshot]:
    _check_access(db_session, document_set_id, user)
    return [
        repository.run_snapshot(row)
        for row in repository.list_runs(db_session, document_set_id)
    ]


@router.get("/runs/{run_id}")
def get_labeling_run(
    document_set_id: int,
    run_id: UUID,
    user: User = Depends(labeling_admin),
    db_session: Session = Depends(get_session),
) -> LabelingRunSnapshot:
    _check_access(db_session, document_set_id, user)
    return repository.run_snapshot(_get_run(db_session, document_set_id, run_id))


@router.get("/runs/{run_id}/items")
def get_labeling_items(
    document_set_id: int,
    run_id: UUID,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    user: User = Depends(labeling_admin),
    db_session: Session = Depends(get_session),
) -> LabelingItemsPage:
    _check_access(db_session, document_set_id, user)
    page = repository.list_items(
        db_session,
        document_set_id=document_set_id,
        run_id=run_id,
        offset=offset,
        limit=limit,
    )
    if page is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Labeling run not found")
    return page


@router.post("/runs/{run_id}/cancel")
def cancel_labeling(
    document_set_id: int,
    run_id: UUID,
    user: User = Depends(labeling_admin),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> LabelingRunSnapshot:
    _check_access(db_session, document_set_id, user)
    run = repository.request_cancellation(
        db_session, document_set_id=document_set_id, run_id=run_id
    )
    if run is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Labeling run not found")
    result = repository.run_snapshot(run)
    db_session.commit()
    _wake(run_id, tenant_id)
    return result


@router.post("/runs/{run_id}/retry")
def retry_labeling(
    document_set_id: int,
    run_id: UUID,
    user: User = Depends(labeling_admin),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> LabelingRunSnapshot:
    _check_access(db_session, document_set_id, user)
    previous = _get_run(db_session, document_set_id, run_id)
    retry_key = uuid5(run_id, "labeling-retry")
    existing_retry = repository.get_run_by_idempotency(
        db_session, document_set_id, retry_key
    )
    if existing_retry is not None:
        return repository.run_snapshot(existing_retry)
    if previous.status not in {"failed", "cancelled", "completed_with_errors"}:
        raise OnyxError(
            OnyxErrorCode.CONFLICT,
            "Only failed, cancelled or incomplete runs can be retried",
        )
    if previous.model_configuration_id is None:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            "The previous provider was removed; select a provider and start a new run",
        )
    return _start_run(
        db_session,
        document_set_id,
        LabelingRunCreate(
            taxonomy_id=previous.taxonomy_id,
            model_configuration_id=previous.model_configuration_id,
            idempotency_key=retry_key,
        ),
        user,
        tenant_id,
        retry_of_id=run_id,
    )
