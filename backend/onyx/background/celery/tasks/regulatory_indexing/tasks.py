from __future__ import annotations

import datetime
import math
from uuid import UUID

from celery import Celery, Task, shared_task

from onyx.configs import app_configs
from onyx.configs.constants import (
    CELERY_REGULATORY_INDEXING_MAX_TASK_EXPIRES,
    CELERY_REGULATORY_INDEXING_TASK_EXPIRES,
    OnyxCeleryPriority,
    OnyxCeleryQueues,
    OnyxCeleryTask,
)
from onyx.db import regulatory_indexing_jobs as indexing_job_repository
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.regulatory.indexing_jobs.orchestrator import (
    OrchestrationDeliveryKind,
    OrchestrationOutcome,
    run_preclaimed_regulatory_indexing_step,
    run_regulatory_indexing_step,
)


def _delivery_expiry(countdown_seconds: float) -> int:
    if not math.isfinite(countdown_seconds) or countdown_seconds < 0:
        raise ValueError("countdown_seconds must be finite and non-negative")
    expires = max(
        CELERY_REGULATORY_INDEXING_TASK_EXPIRES,
        math.ceil(countdown_seconds) + 60,
    )
    if expires > CELERY_REGULATORY_INDEXING_MAX_TASK_EXPIRES:
        raise ValueError("regulatory indexing countdown exceeds the delivery bound")
    return expires


def enqueue_regulatory_indexing_step(
    celery_app: Celery,
    *,
    job_id: UUID,
    expected_generation: int,
    tenant_id: str,
    delivery_kind: OrchestrationDeliveryKind,
    countdown_seconds: float = 0,
) -> None:
    if not tenant_id.strip():
        raise ValueError("tenant_id must not be empty")
    expires = _delivery_expiry(countdown_seconds)
    kwargs = {
        "job_id": str(job_id),
        "expected_generation": expected_generation,
        "tenant_id": tenant_id,
        "delivery_kind": delivery_kind.value,
    }
    if countdown_seconds > 0:
        celery_app.send_task(
            OnyxCeleryTask.REGULATORY_INDEXING_RUN_STEP,
            kwargs=kwargs,
            queue=OnyxCeleryQueues.REGULATORY_INDEXING,
            priority=OnyxCeleryPriority.MEDIUM,
            countdown=countdown_seconds,
            expires=expires,
        )
    else:
        celery_app.send_task(
            OnyxCeleryTask.REGULATORY_INDEXING_RUN_STEP,
            kwargs=kwargs,
            queue=OnyxCeleryQueues.REGULATORY_INDEXING,
            priority=OnyxCeleryPriority.MEDIUM,
            expires=expires,
        )


@shared_task(
    name=OnyxCeleryTask.REGULATORY_INDEXING_RUN_STEP,
    bind=True,
    ignore_result=True,
)
def regulatory_indexing_run_step(
    self: Task,
    *,
    job_id: str,
    expected_generation: int,
    tenant_id: str,
    delivery_kind: str = OrchestrationDeliveryKind.NORMAL.value,
) -> None:
    kind = OrchestrationDeliveryKind(delivery_kind)
    parsed_job_id = UUID(job_id)
    result = (
        run_preclaimed_regulatory_indexing_step(
            parsed_job_id,
            expected_generation,
            tenant_id,
        )
        if kind is OrchestrationDeliveryKind.PRECLAIMED
        else run_regulatory_indexing_step(
            parsed_job_id,
            expected_generation,
            tenant_id,
        )
    )
    if result.outcome is not OrchestrationOutcome.NEXT_STEP:
        return
    if result.expected_generation is None:
        raise RuntimeError("next regulatory indexing step has no generation")
    enqueue_regulatory_indexing_step(
        self.app,
        job_id=result.job_id,
        expected_generation=result.expected_generation,
        tenant_id=tenant_id,
        delivery_kind=OrchestrationDeliveryKind.NORMAL,
        countdown_seconds=result.countdown_seconds or 0,
    )


def _claim_stale_jobs(
    tenant_id: str,
) -> list[indexing_job_repository.RegulatoryIndexingJobClaim]:
    if not tenant_id.strip():
        raise ValueError("tenant_id must not be empty")
    claimed_at = datetime.datetime.now(datetime.timezone.utc)
    stale_before = claimed_at - datetime.timedelta(
        seconds=app_configs.REGULATORY_INDEXING_LEASE_SECONDS
    )
    with get_session_with_current_tenant() as db_session:
        return indexing_job_repository.claim_stale_regulatory_indexing_jobs(
            db_session,
            stale_before=stale_before,
            claimed_at=claimed_at,
        )


@shared_task(
    name=OnyxCeleryTask.REGULATORY_INDEXING_RECOVER_STALE,
    bind=True,
    ignore_result=True,
)
def regulatory_indexing_recover_stale(self: Task, *, tenant_id: str) -> None:
    if not app_configs.REGULATORY_BATCH_INDEXING_ENABLED:
        return
    for claim in _claim_stale_jobs(tenant_id):
        enqueue_regulatory_indexing_step(
            self.app,
            job_id=claim.job_id,
            expected_generation=claim.lease_generation,
            tenant_id=tenant_id,
            delivery_kind=OrchestrationDeliveryKind.PRECLAIMED,
        )
