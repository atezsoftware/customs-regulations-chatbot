from __future__ import annotations

from unittest.mock import MagicMock, call, patch
from uuid import uuid4

import pytest

from onyx.background.celery.tasks.regulatory_indexing.beat_schedule import (
    PRODUCTION_LITE_TASK_TEMPLATES,
)
from onyx.background.celery.tasks.regulatory_indexing.tasks import (
    enqueue_regulatory_indexing_step,
    regulatory_indexing_recover_stale,
    regulatory_indexing_run_step,
)
from onyx.configs.constants import (
    CELERY_REGULATORY_INDEXING_TASK_EXPIRES,
    OnyxCeleryPriority,
    OnyxCeleryQueues,
    OnyxCeleryTask,
)
from onyx.db.enums import RegulatoryIndexingStage
from onyx.db.regulatory_indexing_jobs import RegulatoryIndexingJobClaim
from onyx.regulatory.indexing_jobs.orchestrator import (
    OrchestrationDeliveryKind,
    OrchestrationOutcome,
    OrchestrationResult,
)


def test_enqueue_includes_tenant_queue_priority_and_bounded_expiry() -> None:
    celery_app = MagicMock()
    job_id = uuid4()

    enqueue_regulatory_indexing_step(
        celery_app,
        job_id=job_id,
        expected_generation=7,
        tenant_id="tenant-a",
        delivery_kind=OrchestrationDeliveryKind.NORMAL,
        countdown_seconds=30,
    )

    celery_app.send_task.assert_called_once_with(
        OnyxCeleryTask.REGULATORY_INDEXING_RUN_STEP,
        kwargs={
            "job_id": str(job_id),
            "expected_generation": 7,
            "tenant_id": "tenant-a",
            "delivery_kind": OrchestrationDeliveryKind.NORMAL.value,
        },
        queue=OnyxCeleryQueues.REGULATORY_INDEXING,
        priority=OnyxCeleryPriority.MEDIUM,
        countdown=30,
        expires=CELERY_REGULATORY_INDEXING_TASK_EXPIRES,
    )
    assert CELERY_REGULATORY_INDEXING_TASK_EXPIRES > 30


def test_run_step_emits_at_most_one_followup_message() -> None:
    task_app = MagicMock()
    job_id = uuid4()
    result = OrchestrationResult(
        job_id=job_id,
        outcome=OrchestrationOutcome.NEXT_STEP,
        expected_generation=5,
        countdown_seconds=12,
    )
    with (
        patch(
            "onyx.background.celery.tasks.regulatory_indexing.tasks.run_regulatory_indexing_step",
            return_value=result,
        ) as run_normal,
        patch(
            "onyx.background.celery.tasks.regulatory_indexing.tasks.run_preclaimed_regulatory_indexing_step"
        ) as run_preclaimed,
        patch.object(regulatory_indexing_run_step, "app", task_app),
    ):
        regulatory_indexing_run_step.run(
            job_id=str(job_id),
            expected_generation=4,
            tenant_id="tenant-a",
            delivery_kind=OrchestrationDeliveryKind.NORMAL.value,
        )

    run_normal.assert_called_once_with(job_id, 4, "tenant-a")
    run_preclaimed.assert_not_called()
    assert task_app.send_task.call_count == 1


def test_recovery_sends_preclaimed_generation_without_double_claim() -> None:
    first_id = uuid4()
    second_id = uuid4()
    claims = [
        RegulatoryIndexingJobClaim(
            job_id=first_id,
            stage=RegulatoryIndexingStage.CONTEXT_WAIT,
            lease_generation=8,
            recovery_token=uuid4(),
        ),
        RegulatoryIndexingJobClaim(
            job_id=second_id,
            stage=RegulatoryIndexingStage.EMBEDDING,
            lease_generation=3,
            recovery_token=uuid4(),
        ),
    ]
    task_app = MagicMock()
    with (
        patch(
            "onyx.background.celery.tasks.regulatory_indexing.tasks._claim_stale_jobs",
            return_value=claims,
        ) as claim_stale,
        patch.object(regulatory_indexing_recover_stale, "app", task_app),
        patch(
            "onyx.background.celery.tasks.regulatory_indexing.tasks.app_configs.REGULATORY_BATCH_INDEXING_ENABLED",
            True,
        ),
    ):
        regulatory_indexing_recover_stale.run(tenant_id="tenant-a")

    claim_stale.assert_called_once_with("tenant-a")
    expected_calls = [
        call(
            OnyxCeleryTask.REGULATORY_INDEXING_RUN_STEP,
            kwargs={
                "job_id": str(claim.job_id),
                "expected_generation": claim.lease_generation,
                "recovery_token": str(claim.recovery_token),
                "tenant_id": "tenant-a",
                "delivery_kind": OrchestrationDeliveryKind.PRECLAIMED.value,
            },
            queue=OnyxCeleryQueues.REGULATORY_INDEXING,
            priority=OnyxCeleryPriority.MEDIUM,
            expires=CELERY_REGULATORY_INDEXING_TASK_EXPIRES,
        )
        for claim in claims
    ]
    assert task_app.send_task.call_args_list == expected_calls


def test_periodic_recovery_republishes_stale_job_after_initial_broker_failure() -> None:
    job_id = uuid4()
    initial_app = MagicMock()
    initial_app.send_task.side_effect = ConnectionError("broker unavailable")

    recovery_template = next(
        template
        for template in PRODUCTION_LITE_TASK_TEMPLATES
        if template["task"] == OnyxCeleryTask.REGULATORY_INDEXING_RECOVER_STALE
    )
    assert recovery_template["schedule"].total_seconds() == 60
    assert recovery_template["options"]["queue"] == OnyxCeleryQueues.REGULATORY_INDEXING

    with pytest.raises(ConnectionError, match="broker unavailable"):
        enqueue_regulatory_indexing_step(
            initial_app,
            job_id=job_id,
            expected_generation=1,
            tenant_id="tenant-a",
            delivery_kind=OrchestrationDeliveryKind.NORMAL,
        )

    claim = RegulatoryIndexingJobClaim(
        job_id=job_id,
        stage=RegulatoryIndexingStage.PREPARING,
        lease_generation=2,
        recovery_token=uuid4(),
    )
    recovery_app = MagicMock()
    with (
        patch(
            "onyx.background.celery.tasks.regulatory_indexing.tasks._claim_stale_jobs",
            return_value=[claim],
        ),
        patch.object(regulatory_indexing_recover_stale, "app", recovery_app),
        patch(
            "onyx.background.celery.tasks.regulatory_indexing.tasks.app_configs.REGULATORY_BATCH_INDEXING_ENABLED",
            True,
        ),
    ):
        regulatory_indexing_recover_stale.run(tenant_id="tenant-a")

    recovery_app.send_task.assert_called_once_with(
        OnyxCeleryTask.REGULATORY_INDEXING_RUN_STEP,
        kwargs={
            "job_id": str(job_id),
            "expected_generation": 2,
            "recovery_token": str(claim.recovery_token),
            "tenant_id": "tenant-a",
            "delivery_kind": OrchestrationDeliveryKind.PRECLAIMED.value,
        },
        queue=OnyxCeleryQueues.REGULATORY_INDEXING,
        priority=OnyxCeleryPriority.MEDIUM,
        expires=CELERY_REGULATORY_INDEXING_TASK_EXPIRES,
    )
