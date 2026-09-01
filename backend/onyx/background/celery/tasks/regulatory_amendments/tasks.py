import datetime
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import copy_context
from threading import Event, Thread
from typing import Any
from uuid import UUID

from celery import Celery, Task, shared_task
from sqlalchemy.orm import Session

from onyx.background.celery.queue_names import REGULATORY_AMENDMENT_QUEUE
from onyx.configs.constants import OnyxCeleryPriority, OnyxCeleryTask
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.enums import AmendmentProposalStatus
from onyx.db.models import UserFile
from onyx.db.regulatory_amendments import (
    approve_amendment_proposal,
    claim_batch_for_analysis,
    claim_stale_batches_for_recovery,
    finalize_amendment_proposal_projection,
    get_proposal,
    mark_batch_failed,
    recover_stale_amendment_proposal_approvals,
    reset_amendment_proposal_approval,
    touch_amendment_proposal_approval,
    touch_batch_heartbeat,
)
from onyx.db.search_settings import get_current_search_settings
from onyx.regulatory.amendments.job import run_amendment_batch
from onyx.regulatory.projection import project_amendment_to_index
from onyx.utils.logger import setup_logger
from shared_configs.enums import EmbeddingProvider

logger = setup_logger()

_DELIVERY_EXPIRES_SECONDS = 24 * 60 * 60
_STALE_HEARTBEAT_SECONDS = 10 * 60
_HEARTBEAT_INTERVAL_SECONDS = 60
_SAFE_FAILURE_MESSAGE = (
    "Analysis failed. Retry to resume from the last completed instruction."
)
_SAFE_APPROVAL_FAILURE_MESSAGE = "Indexing failed. The approval was not published."
_AMENDMENT_EMBEDDING_MODEL = "gemini-embedding-2"
_AMENDMENT_EMBEDDING_DIMENSION = 1024


def validate_amendment_projection_search_settings(
    db_session: Session,
    *,
    expected_id: int | None = None,
    for_update: bool = False,
) -> int:
    search_settings = (
        get_current_search_settings(db_session, for_update=True)
        if for_update
        else get_current_search_settings(db_session)
    )
    if expected_id is not None and search_settings.id != expected_id:
        raise RuntimeError("Search settings changed during amendment projection")
    if search_settings.provider_type is not EmbeddingProvider.GOOGLE:
        raise RuntimeError(
            "Amendment indexing requires the active Google Gemini provider"
        )
    if search_settings.model_name != _AMENDMENT_EMBEDDING_MODEL:
        raise RuntimeError(f"Amendment indexing requires {_AMENDMENT_EMBEDDING_MODEL}")
    if search_settings.final_embedding_dim != _AMENDMENT_EMBEDDING_DIMENSION:
        raise RuntimeError(
            "Amendment indexing requires 1024-dimensional Elasticsearch vectors"
        )
    return search_settings.id


@contextmanager
def _renew_batch_lease(*, batch_id: int, lease_generation: int) -> Generator[None]:
    """Renew a claimed batch from a separate DB session during long LLM calls."""

    stop = Event()

    def renew() -> None:
        while not stop.wait(_HEARTBEAT_INTERVAL_SECONDS):
            try:
                with get_session_with_current_tenant() as db_session:
                    if not touch_batch_heartbeat(
                        db_session,
                        batch_id=batch_id,
                        lease_generation=lease_generation,
                    ):
                        logger.warning(
                            "Stopped heartbeat for amendment batch=%s lease=%s",
                            batch_id,
                            lease_generation,
                        )
                        return
            except Exception:
                logger.exception(
                    "Heartbeat failed for amendment batch=%s lease=%s",
                    batch_id,
                    lease_generation,
                )

    # TenantAwareTask stores tenant scope in contextvars. Threads do not inherit
    # that context automatically, so run the watchdog in an explicit copy.
    context = copy_context()
    thread = Thread(target=lambda: context.run(renew), daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2)


@contextmanager
def _renew_amendment_approval(*, proposal_id: int) -> Generator[None]:
    """Keep recovery from redelivering a live full-file Gemini projection."""

    stop = Event()

    def renew() -> None:
        while not stop.wait(_HEARTBEAT_INTERVAL_SECONDS):
            try:
                with get_session_with_current_tenant() as db_session:
                    if not touch_amendment_proposal_approval(
                        db_session,
                        proposal_id=proposal_id,
                    ):
                        logger.warning(
                            "Stopped approval heartbeat for proposal=%s",
                            proposal_id,
                        )
                        return
            except Exception:
                logger.exception(
                    "Approval heartbeat failed for proposal=%s",
                    proposal_id,
                )

    context = copy_context()
    thread = Thread(target=lambda: context.run(renew), daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2)


def enqueue_amendment_batch(
    celery_app: Celery | Any | None = None, *, batch_id: int, tenant_id: str
) -> None:
    if celery_app is None:
        from onyx.background.celery.versioned_apps.client import app as celery_app

    errors: list[Exception] = []
    for countdown in (None, 5):
        options: dict[str, Any] = {
            "kwargs": {"batch_id": batch_id, "tenant_id": tenant_id},
            "queue": REGULATORY_AMENDMENT_QUEUE,
            "priority": OnyxCeleryPriority.HIGH,
            "expires": _DELIVERY_EXPIRES_SECONDS,
            "retry": False,
        }
        if countdown is not None:
            options["countdown"] = countdown
        try:
            celery_app.send_task(OnyxCeleryTask.REGULATORY_AMENDMENT_RUN, **options)
        except Exception as error:
            errors.append(error)
            logger.warning(
                "Amendment batch %s dispatch failed (countdown=%s)",
                batch_id,
                countdown,
                exc_info=True,
            )
    if len(errors) == 2:
        raise RuntimeError("All amendment dispatch attempts failed") from errors[0]


def enqueue_amendment_proposal_approval(
    celery_app: Celery | Any | None = None,
    *,
    proposal_id: int,
    tenant_id: str,
) -> None:
    if celery_app is None:
        from onyx.background.celery.versioned_apps.client import app as celery_app

    celery_app.send_task(
        OnyxCeleryTask.REGULATORY_AMENDMENT_APPROVE,
        kwargs={"proposal_id": proposal_id, "tenant_id": tenant_id},
        queue=REGULATORY_AMENDMENT_QUEUE,
        priority=OnyxCeleryPriority.HIGH,
        expires=_DELIVERY_EXPIRES_SECONDS,
        retry=False,
    )


@shared_task(
    name=OnyxCeleryTask.REGULATORY_AMENDMENT_RUN,
    ignore_result=True,
    trail=False,
    acks_late=True,
    reject_on_worker_lost=True,
)
def regulatory_amendment_run(
    *,
    batch_id: int,
    tenant_id: str,  # noqa: ARG001 - TenantAwareTask consumes it
) -> None:
    with get_session_with_current_tenant() as db_session:
        lease = claim_batch_for_analysis(db_session, batch_id=batch_id)
    if lease is None:
        return
    try:
        with _renew_batch_lease(
            batch_id=batch_id,
            lease_generation=lease.generation,
        ):
            run_amendment_batch(
                batch_id=batch_id,
                lease_generation=lease.generation,
            )
    except Exception:
        logger.exception("Amendment batch %s failed", batch_id)
        with get_session_with_current_tenant() as db_session:
            mark_batch_failed(
                db_session,
                batch_id=batch_id,
                lease_generation=lease.generation,
                error_message=_SAFE_FAILURE_MESSAGE,
            )
        raise


@shared_task(
    name=OnyxCeleryTask.REGULATORY_AMENDMENT_APPROVE,
    ignore_result=True,
    trail=False,
    acks_late=True,
    reject_on_worker_lost=True,
)
def regulatory_amendment_approve(
    *,
    proposal_id: int,
    tenant_id: str,  # noqa: ARG001 - TenantAwareTask consumes it
) -> None:
    """Apply and project one approval outside the Cloudflare request window."""

    version_applied = False
    try:
        with get_session_with_current_tenant() as db_session:
            proposal = get_proposal(db_session, proposal_id)
            if proposal is None:
                logger.warning("Amendment proposal %s no longer exists", proposal_id)
                return
            if proposal.status != AmendmentProposalStatus.APPROVING.value:
                logger.info(
                    "Skipping amendment approval proposal=%s status=%s",
                    proposal_id,
                    proposal.status,
                )
                return

            result = approve_amendment_proposal(db_session, proposal)
            user_file_id = UUID(str(result.new_chunk.user_file_id))
            new_chunk_id = str(result.new_chunk.id)
            old_chunk_id = (
                str(result.old_chunk.id) if result.old_chunk is not None else None
            )
            db_session.commit()
            version_applied = True
            logger.info(
                "Amendment approval version committed proposal=%s user_file=%s",
                proposal_id,
                user_file_id,
            )

        with _renew_amendment_approval(proposal_id=proposal_id):
            with get_session_with_current_tenant() as db_session:
                user_file = db_session.get(UserFile, user_file_id)
                if user_file is None:
                    raise RuntimeError(f"User file {user_file_id} no longer exists")
                current_search_settings_id = (
                    validate_amendment_projection_search_settings(db_session)
                )
                projected_chunk_count = project_amendment_to_index(
                    db_session,
                    user_file,
                    tenant_id,
                    old_chunk_id=old_chunk_id,
                    new_chunk_id=new_chunk_id,
                    current_search_settings_id=current_search_settings_id,
                )
                if projected_chunk_count <= 0:
                    raise RuntimeError("Amendment projection did not write any chunks")
                validate_amendment_projection_search_settings(
                    db_session,
                    expected_id=current_search_settings_id,
                    for_update=True,
                )
                if not finalize_amendment_proposal_projection(
                    db_session,
                    proposal_id=proposal_id,
                    succeeded=True,
                ):
                    raise RuntimeError("amendment proposal could not be finalized")
                db_session.commit()
        logger.info(
            "Amendment approval projected proposal=%s user_file=%s",
            proposal_id,
            user_file_id,
        )
    except Exception:
        logger.exception("Amendment proposal %s approval failed", proposal_id)
        if version_applied:
            with get_session_with_current_tenant() as db_session:
                finalize_amendment_proposal_projection(
                    db_session,
                    proposal_id=proposal_id,
                    succeeded=False,
                    error_message=_SAFE_APPROVAL_FAILURE_MESSAGE,
                )
                db_session.commit()
        else:
            with get_session_with_current_tenant() as db_session:
                reset_amendment_proposal_approval(
                    db_session,
                    proposal_id=proposal_id,
                )
        raise


@shared_task(
    name=OnyxCeleryTask.REGULATORY_AMENDMENT_RECOVER_STALE,
    bind=True,
    ignore_result=True,
    trail=False,
)
def regulatory_amendment_recover_stale(
    self: Task,
    *,
    tenant_id: str,  # noqa: ARG001 - TenantAwareTask consumes it
) -> None:
    now = datetime.datetime.now(datetime.timezone.utc)
    with get_session_with_current_tenant() as db_session:
        batch_ids = claim_stale_batches_for_recovery(
            db_session,
            stale_before=now - datetime.timedelta(seconds=_STALE_HEARTBEAT_SECONDS),
            claimed_at=now,
        )
    with get_session_with_current_tenant() as db_session:
        proposal_ids = recover_stale_amendment_proposal_approvals(
            db_session,
            stale_before=now - datetime.timedelta(seconds=_STALE_HEARTBEAT_SECONDS),
            recovered_at=now,
        )

    for batch_id in batch_ids:
        try:
            enqueue_amendment_batch(
                self.app,
                batch_id=batch_id,
                tenant_id=tenant_id,
            )
        except Exception:
            logger.exception("Failed to recover amendment batch %s", batch_id)
    for proposal_id in proposal_ids:
        try:
            enqueue_amendment_proposal_approval(
                self.app,
                proposal_id=proposal_id,
                tenant_id=tenant_id,
            )
        except Exception:
            logger.exception(
                "Failed to recover amendment approval proposal=%s",
                proposal_id,
            )
