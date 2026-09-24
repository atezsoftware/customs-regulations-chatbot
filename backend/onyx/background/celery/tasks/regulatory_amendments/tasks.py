import datetime
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import copy_context
from threading import Event, Thread
from typing import Any

from celery import Celery, Task, shared_task
from sqlalchemy.orm import Session

from onyx.background.celery.queue_names import REGULATORY_AMENDMENT_QUEUE
from onyx.configs.constants import OnyxCeleryPriority, OnyxCeleryTask
from onyx.db.amendment_resources import defer_analysis, parallel_analysis_allowed
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.regulatory_amendments import (
    claim_batch_for_analysis,
    claim_stale_batches_for_recovery,
    mark_batch_failed,
    recover_stale_amendment_proposal_approvals,
    touch_amendment_proposal_approval,
    touch_batch_heartbeat,
)
from onyx.db.search_settings import get_current_search_settings
from onyx.regulatory.amendments.annexes import config as annex_config
from onyx.regulatory.amendments.memory_budget import ResourcePressure
from onyx.regulatory.amendments.supervision import run_supervised_amendment
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
            "kwargs": {
                "batch_id": batch_id,
                "tenant_id": tenant_id,
                "environment": annex_config.REGULATORY_ANNEX_ENVIRONMENT,
                "database_identity": annex_config.ANNEX_DATABASE_IDENTITY,
            },
            "queue": annex_config.analysis_delivery_queue_name(),
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
    tenant_id: str,
    environment: str | None = None,
    database_identity: str | None = None,
) -> None:
    from shared_configs.contextvars import get_current_tenant_id

    if environment is not None or database_identity is not None:
        if (
            environment != annex_config.REGULATORY_ANNEX_ENVIRONMENT
            or database_identity != annex_config.ANNEX_DATABASE_IDENTITY
            or not tenant_id
            or tenant_id != get_current_tenant_id()
        ):
            raise ValueError("amendment analysis worker scope mismatch")
    with get_session_with_current_tenant() as db_session:
        lease = claim_batch_for_analysis(db_session, batch_id=batch_id)
    if lease is None:
        return
    try:
        with get_session_with_current_tenant() as db_session:
            parallel = parallel_analysis_allowed(db_session, batch_id)
        with _renew_batch_lease(
            batch_id=batch_id,
            lease_generation=lease.generation,
        ):
            run_supervised_amendment(
                batch_id=batch_id,
                lease_generation=lease.generation,
                tenant_id=tenant_id,
                parallel=parallel,
            )
    except ResourcePressure as error:
        with get_session_with_current_tenant() as db_session:
            defer_analysis(
                db_session,
                batch_id=batch_id,
                lease_generation=lease.generation,
                reason=str(error),
                started=error.started,
            )
        logger.info("Amendment batch=%s waiting for resources: %s", batch_id, error)
    except Exception as error:
        logger.exception("Amendment batch %s failed", batch_id)
        with get_session_with_current_tenant() as db_session:
            mark_batch_failed(
                db_session,
                batch_id=batch_id,
                lease_generation=lease.generation,
                error_message=_SAFE_FAILURE_MESSAGE,
                failure=error,
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

    from onyx.db.regulatory_writer_publication import record_owned_amendment_failure
    from onyx.regulatory.writer_publication import approve_owned_amendment

    try:
        with get_session_with_current_tenant() as db_session:
            current_id = validate_amendment_projection_search_settings(db_session)
        with _renew_amendment_approval(proposal_id=proposal_id):
            approve_owned_amendment(proposal_id, tenant_id, current_id)
    except Exception as error:
        logger.exception("Amendment proposal %s approval failed", proposal_id)
        record_owned_amendment_failure(proposal_id, tenant_id, error)
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
