"""Durable revalidation; HTTP requests only enqueue scoped preparation."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import copy_context
from threading import Event, Thread
from time import monotonic
from uuid import UUID

from celery import shared_task

from onyx.configs.constants import OnyxCeleryPriority
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.regulatory_annex_preparation import (
    claim_review_preparation,
    pending_review_preparations,
    touch_review_preparation,
)
from onyx.regulatory.amendments.annexes import config
from onyx.regulatory.amendments.annexes.models import (
    AnnexChangeDraft,
    AnnexElementCorrection,
)
from onyx.utils.logger import setup_logger

logger = setup_logger()
TASK_NAME = "prepare_annex_review"


def enqueue_review_preparation(*, review_id: UUID, tenant_id: str) -> None:
    from onyx.background.celery.versioned_apps.client import app

    app.send_task(
        TASK_NAME,
        kwargs={
            "review_id": str(review_id),
            "tenant_id": tenant_id,
            "environment": config.REGULATORY_ANNEX_ENVIRONMENT,
            "database_identity": config.ANNEX_DATABASE_IDENTITY,
        },
        queue=config.analysis_delivery_queue_name(),
        priority=OnyxCeleryPriority.MEDIUM,
        expires=3600,
        retry=False,
    )


@contextmanager
def _heartbeat(review_id: UUID, generation: int) -> Iterator[None]:
    stop = Event()

    def renew() -> None:
        while not stop.wait(60):
            try:
                touch_review_preparation(review_id=review_id, generation=generation)
            except Exception:
                logger.exception(
                    "Annex preparation heartbeat failed review=%s", review_id
                )
                return

    context = copy_context()
    thread = Thread(target=lambda: context.run(renew), daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2)


def recover_review_preparations(
    *, tenant_id: str, environment: str, database_identity: str
) -> int:
    ids = pending_review_preparations(
        tenant_id=tenant_id,
        environment=environment,
        database_identity=database_identity,
    )
    for review_id in ids:
        enqueue_review_preparation(review_id=review_id, tenant_id=tenant_id)
    return len(ids)


@shared_task(
    name=TASK_NAME, ignore_result=True, acks_late=True, reject_on_worker_lost=True
)
def prepare_annex_review(
    *, review_id: str, tenant_id: str, environment: str, database_identity: str
) -> None:
    from onyx.db.regulatory_annex_changes import (
        require_current_annex_review,
        revise_annex_review,
    )
    from onyx.llm.factory import get_default_llm, get_default_llm_with_vision
    from onyx.regulatory.amendments.annexes.analysis import (
        prepare_review_context,
        validate_live_review_runtime,
    )
    from onyx.regulatory.amendments.annexes.corrections import revalidate_annex_review
    from onyx.regulatory.amendments.annexes.preparation_progress import (
        PreparationObserver,
        observe_preparation,
    )
    from onyx.regulatory.amendments.annexes.publication_preparation import (
        prepare_publication_review,
    )
    from shared_configs.contextvars import get_current_tenant_id

    if (
        environment != config.REGULATORY_ANNEX_ENVIRONMENT
        or database_identity != config.ANNEX_DATABASE_IDENTITY
        or not tenant_id
        or tenant_id != get_current_tenant_id()
    ):
        raise ValueError("review preparation worker scope mismatch")
    identifier = UUID(review_id)
    with get_session_with_current_tenant() as session:
        job = claim_review_preparation(
            session,
            review_id=identifier,
            tenant_id=tenant_id,
            environment=environment,
            database_identity=database_identity,
        )
    if job is None:
        return
    started = monotonic()

    def progress(stage: str, completed: int, total: int) -> None:
        if monotonic() - started > 7200:
            raise TimeoutError("review preparation exceeded its resumable work window")
        touch_review_preparation(
            review_id=identifier,
            generation=job.generation,
            stage=stage,
            completed=completed,
            total=total,
        )

    def checkpoint(draft: AnnexChangeDraft) -> None:
        touch_review_preparation(
            review_id=identifier, generation=job.generation, checkpoint=draft
        )

    try:
        with (
            _heartbeat(identifier, job.generation),
            observe_preparation(
                PreparationObserver(checkpoint=checkpoint, progress=progress)
            ),
        ):
            with get_session_with_current_tenant() as session:
                review = require_current_annex_review(
                    session,
                    change_set_id=identifier,
                    expected_review_sha256=job.expected_review_sha256,
                    environment=environment,
                    allow_preparation=True,
                )
                original = AnnexChangeDraft.model_validate(review.review_payload)
            progress("evidence", 0, 0)
            if job.checkpoint is not None:
                draft = AnnexChangeDraft.model_validate(job.checkpoint).model_copy(
                    update={"date_resolution": original.date_resolution}
                )
                # A saved comparison is reusable only under its frozen live configuration.
                validate_live_review_runtime(draft)
                draft = (
                    prepare_publication_review(draft)
                    if draft.impact is not None
                    else prepare_review_context(draft)
                )
            else:
                draft = revalidate_annex_review(
                    draft=original,
                    corrections=[
                        AnnexElementCorrection.model_validate(item)
                        for item in job.corrections
                    ],
                    corrected_by=job.corrected_by,
                    llm=get_default_llm(),
                    vision_llm=get_default_llm_with_vision(),
                )
            progress("validating", 0, 0)
            with get_session_with_current_tenant() as session:
                revise_annex_review(
                    session,
                    change_set_id=identifier,
                    expected_review_sha256=job.expected_review_sha256,
                    draft=draft,
                    environment=environment,
                    preparation_generation=job.generation,
                )
    except Exception as error:
        logger.exception("Annex review preparation failed review=%s", identifier)
        touch_review_preparation(
            review_id=identifier,
            generation=job.generation,
            error=f"Review preparation failed ({type(error).__name__}). Retry resumes saved preparation.",
        )
        raise
