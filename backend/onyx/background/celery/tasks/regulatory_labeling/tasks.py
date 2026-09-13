from __future__ import annotations

import math
from uuid import UUID

from celery import Celery, Task, shared_task

from onyx.configs.constants import (
    CELERY_REGULATORY_INDEXING_MAX_TASK_EXPIRES,
    CELERY_REGULATORY_INDEXING_TASK_EXPIRES,
    OnyxCeleryPriority,
    OnyxCeleryQueues,
    OnyxCeleryTask,
)
from onyx.db import regulatory_labeling as repository
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.regulatory.labeling.orchestrator import (
    LabelingStepOutcome,
    run_labeling_step,
)


def _delivery_expiry(countdown_seconds: float) -> int:
    if not math.isfinite(countdown_seconds) or countdown_seconds < 0:
        raise ValueError("countdown_seconds must be finite and non-negative")
    expires = max(
        CELERY_REGULATORY_INDEXING_TASK_EXPIRES,
        math.ceil(countdown_seconds) + 60,
    )
    if expires > CELERY_REGULATORY_INDEXING_MAX_TASK_EXPIRES:
        raise ValueError("labeling countdown exceeds the delivery bound")
    return expires


def enqueue_labeling_step(
    celery_app: Celery,
    *,
    run_id: UUID,
    expected_generation: int,
    tenant_id: str,
    countdown_seconds: float = 0,
) -> None:
    if not tenant_id.strip():
        raise ValueError("tenant_id must not be empty")
    kwargs = {
        "run_id": str(run_id),
        "expected_generation": expected_generation,
        "tenant_id": tenant_id,
    }
    if countdown_seconds > 0:
        celery_app.send_task(
            OnyxCeleryTask.REGULATORY_LABELING_RUN_STEP,
            kwargs=kwargs,
            queue=OnyxCeleryQueues.REGULATORY_INDEXING,
            priority=OnyxCeleryPriority.MEDIUM,
            expires=_delivery_expiry(countdown_seconds),
            countdown=countdown_seconds,
        )
    else:
        celery_app.send_task(
            OnyxCeleryTask.REGULATORY_LABELING_RUN_STEP,
            kwargs=kwargs,
            queue=OnyxCeleryQueues.REGULATORY_INDEXING,
            priority=OnyxCeleryPriority.MEDIUM,
            expires=_delivery_expiry(countdown_seconds),
        )


def enqueue_labeling_run(*, run_id: UUID, tenant_id: str) -> None:
    from onyx.background.celery.versioned_apps.client import app as celery_app

    with get_session_with_current_tenant() as session:
        run = repository.get_run_for_delivery(session, run_id)
        if run is None:
            raise ValueError("The labeling run is unavailable")
        generation = run.lease_generation
    enqueue_labeling_step(
        celery_app,
        run_id=run_id,
        expected_generation=generation,
        tenant_id=tenant_id,
    )


@shared_task(
    name=OnyxCeleryTask.REGULATORY_LABELING_RUN_STEP,
    bind=True,
    ignore_result=True,
)
def regulatory_labeling_run_step(
    self: Task,
    *,
    run_id: str,
    expected_generation: int,
    tenant_id: str,
) -> None:
    result = run_labeling_step(UUID(run_id), expected_generation, tenant_id)
    if result.outcome in (LabelingStepOutcome.SKIPPED, LabelingStepOutcome.TERMINAL):
        return
    if result.expected_generation is None:
        raise RuntimeError("Next labeling step has no lease generation")
    enqueue_labeling_step(
        self.app,
        run_id=result.run_id,
        expected_generation=result.expected_generation,
        tenant_id=tenant_id,
        countdown_seconds=result.countdown_seconds,
    )


@shared_task(
    name=OnyxCeleryTask.REGULATORY_LABELING_RECOVER_STALE,
    bind=True,
    ignore_result=True,
)
def regulatory_labeling_recover_stale(self: Task, *, tenant_id: str) -> None:
    if not tenant_id.strip():
        raise ValueError("tenant_id must not be empty")
    with get_session_with_current_tenant() as session:
        recoverable = repository.recoverable_runs(session, limit=100)
    for run in recoverable:
        enqueue_labeling_step(
            self.app,
            run_id=run.run_id,
            expected_generation=run.generation,
            tenant_id=tenant_id,
        )
