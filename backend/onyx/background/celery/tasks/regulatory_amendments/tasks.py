import datetime
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import copy_context
from threading import Event, Thread
from typing import Any
from uuid import UUID

from celery import Celery, shared_task

from onyx.background.celery.tasks.regulatory_indexing.tasks import (
    enqueue_prepared_regulatory_indexing_job,
)
from onyx.configs.constants import OnyxCeleryPriority, OnyxCeleryQueues, OnyxCeleryTask
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.enums import AmendmentProposalStatus
from onyx.db.regulatory_amendments import (
    approve_amendment_proposal,
    claim_batch_for_analysis,
    claim_stale_batches_for_recovery,
    get_proposal,
    link_amendment_proposal_indexing_job,
    mark_batch_failed,
    recover_stale_amendment_proposal_approvals,
    reset_amendment_proposal_approval,
    touch_batch_heartbeat,
)
from onyx.regulatory.amendments.job import run_amendment_batch
from onyx.regulatory.indexing_jobs.preparation import (
    prepare_regulatory_indexing_job_from_chunks,
)
from onyx.utils.logger import setup_logger

logger = setup_logger()

_DELIVERY_EXPIRES_SECONDS = 24 * 60 * 60
_STALE_HEARTBEAT_SECONDS = 10 * 60
_HEARTBEAT_INTERVAL_SECONDS = 60
_SAFE_FAILURE_MESSAGE = (
    "Analysis failed. Retry to resume from the last completed instruction."
)


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


def enqueue_amendment_batch(
    celery_app: Celery | Any | None = None, *, batch_id: int, tenant_id: str
) -> None:
    if celery_app is None:
        from onyx.background.celery.versioned_apps.client import app as celery_app

    errors: list[Exception] = []
    for countdown in (None, 5):
        options: dict[str, Any] = {
            "kwargs": {"batch_id": batch_id, "tenant_id": tenant_id},
            "queue": OnyxCeleryQueues.REGULATORY_AMENDMENT,
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
        queue=OnyxCeleryQueues.REGULATORY_AMENDMENT,
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
            db_session.commit()
            version_applied = True
            logger.info(
                "Amendment approval version committed proposal=%s user_file=%s",
                proposal_id,
                user_file_id,
            )

        with get_session_with_current_tenant() as db_session:
            job_id = prepare_regulatory_indexing_job_from_chunks(
                user_file_id,
                tenant_id,
                db_session,
            )

        with get_session_with_current_tenant() as db_session:
            if not link_amendment_proposal_indexing_job(
                db_session,
                proposal_id=proposal_id,
                job_id=job_id,
            ):
                raise RuntimeError("amendment proposal could not be linked to indexing")
            db_session.commit()

        enqueue_prepared_regulatory_indexing_job(
            job_id=job_id,
            tenant_id=tenant_id,
        )
        logger.info(
            "Amendment approval indexing dispatched proposal=%s user_file=%s job=%s",
            proposal_id,
            user_file_id,
            job_id,
        )
    except Exception:
        logger.exception("Amendment proposal %s approval failed", proposal_id)
        if not version_applied:
            with get_session_with_current_tenant() as db_session:
                reset_amendment_proposal_approval(
                    db_session,
                    proposal_id=proposal_id,
                )
        raise


@shared_task(
    name=OnyxCeleryTask.REGULATORY_AMENDMENT_RECOVER_STALE,
    ignore_result=True,
    trail=False,
)
def regulatory_amendment_recover_stale(
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

    from onyx.background.celery.versioned_apps.client import app as celery_app

    for batch_id in batch_ids:
        try:
            enqueue_amendment_batch(
                celery_app,
                batch_id=batch_id,
                tenant_id=tenant_id,
            )
        except Exception:
            logger.exception("Failed to recover amendment batch %s", batch_id)
    for proposal_id in proposal_ids:
        try:
            enqueue_amendment_proposal_approval(
                celery_app,
                proposal_id=proposal_id,
                tenant_id=tenant_id,
            )
        except Exception:
            logger.exception(
                "Failed to recover amendment approval proposal=%s",
                proposal_id,
            )
