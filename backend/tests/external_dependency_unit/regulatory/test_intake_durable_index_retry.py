"""An approved durable handoff remains retryable after asynchronous failure."""

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import cast
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.background.celery.tasks.user_file_processing import tasks
from onyx.db import document_intake as intake
from onyx.db import document_intake_durable_indexing as durable
from onyx.db import regulatory_indexing_jobs as jobs
from onyx.db.enums import (
    IndexModelStatus,
    RegulatoryIndexingJobStatus,
    RegulatoryIndexingProviderCleanupPhase,
    RegulatoryIndexingStage,
    UserFileProjectionRepairStatus,
)
from onyx.db.models import (
    DocumentSet,
    KVStore,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryFilePublication,
    RegulatoryIndexingItem,
    RegulatoryIndexingJob,
    SearchSettings,
    User,
    UserFile,
    UserFileProjectionRepair,
)
from onyx.db.user_file import recover_user_file_index_requests
from onyx.document_index.publication_models import PublicationScope, publication_digest
from onyx.regulatory.amendments.annexes import config
from onyx.regulatory.indexing_jobs import preparation
from onyx.regulatory.indexing_jobs.models import RegulatoryInputHashVersion
from onyx.utils.special_types import JSON_ro
from shared_configs.contextvars import get_current_tenant_id
from tests.external_dependency_unit.regulatory.test_document_intake import (
    _existing_rows,
)
from tests.external_dependency_unit.regulatory.test_intake_recovered_index_delivery import (
    _pending_pair,
)
from tests.external_dependency_unit.regulatory.test_intake_recovered_index_delivery import (
    owned_session as owned_session,
)
from tests.external_dependency_unit.regulatory.test_intake_recovered_index_delivery import (
    owner as owner,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)
from tests.unit.onyx.regulatory.indexing_jobs.test_preparation import _snapshot


def _complete_cleanup(session: Session, job_id: UUID) -> None:
    for phase in (
        RegulatoryIndexingProviderCleanupPhase.GCS_CLEANUP,
        RegulatoryIndexingProviderCleanupPhase.COMPLETE,
    ):
        now = datetime.now(timezone.utc)
        claims = jobs.claim_due_regulatory_provider_cleanups(
            session, stale_before=now, claimed_at=now
        )
        claim = next(claim for claim in claims if claim.job_id == job_id)
        assert jobs.consume_regulatory_provider_cleanup_delivery(
            session,
            job_id=job_id,
            cleanup_generation=claim.cleanup_generation,
            cleanup_token=claim.cleanup_token,
            consumed_at=now,
        )
        if phase == RegulatoryIndexingProviderCleanupPhase.COMPLETE:
            assert jobs.complete_regulatory_provider_cleanup(
                session,
                job_id=job_id,
                cleanup_generation=claim.cleanup_generation,
                now=now,
            )
        else:
            assert jobs.advance_regulatory_provider_cleanup(
                session,
                job_id=job_id,
                cleanup_generation=claim.cleanup_generation,
                expected_phase=phase,
                next_phase=RegulatoryIndexingProviderCleanupPhase.COMPLETE,
                now=now,
            )


def _retained_job(session: Session, job: RegulatoryIndexingJob) -> tuple[object, ...]:
    session.refresh(job)
    return (
        {column.name: getattr(job, column.name) for column in job.__table__.columns},
        [
            {
                column.name: getattr(item, column.name)
                for column in item.__table__.columns
            }
            for item in session.scalars(
                select(RegulatoryIndexingItem).where(
                    RegulatoryIndexingItem.job_id == job.id
                )
            )
        ],
    )


@pytest.mark.parametrize(
    "handoff",
    [
        "normal",
        "rotated_crash",
        "binding_crash",
        "source_changed",
        "config_changed",
        "foreign_job",
        "restart_binding_crash",
    ],
)
def test_async_failure_retains_approved_intent_for_a_fenced_successor(
    owned_session: Session,
    owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    handoff: str,
) -> None:
    session = owned_session
    batch, subject, sibling, attempt = _pending_pair(session, owner, monkeypatch, False)
    sibling_before = _existing_rows(session, [sibling.user_file_id])
    file = session.get_one(UserFile, subject.user_file_id)
    rows = list(
        session.scalars(
            select(RegulatoryChunk).where(RegulatoryChunk.user_file_id == file.id)
        )
    )
    settings = session.scalar(
        select(SearchSettings).where(SearchSettings.status == IndexModelStatus.PRESENT)
    )
    if settings is None:
        settings = SearchSettings(
            model_name="fixture",
            model_dim=3,
            normalize=False,
            status=IndexModelStatus.PRESENT,
            index_name="fixture",
        )
        session.add(settings)
        session.flush()
    snapshot = _snapshot(
        input_content_hash=preparation.regulatory_chunks_content_hash(rows)
    ).model_copy(
        update={
            "input_hash_version": RegulatoryInputHashVersion.CHUNK_ROWS_V3,
            "chunk_generation_hash": file.regulatory_chunk_generation_hash,
            "search_settings_id": settings.id,
        }
    )
    scope = PublicationScope(
        tenant_id=get_current_tenant_id(),
        environment=config.REGULATORY_ANNEX_ENVIRONMENT,
        database_identity=config.ANNEX_DATABASE_IDENTITY,
    )
    session.get_one(RegulatoryFilePublication, file.id).scope_key = publication_digest(
        scope.model_dump(mode="json")
    )
    session.commit()
    monkeypatch.setattr(tasks.app_configs, "REGULATORY_BATCH_INDEXING_ENABLED", True)
    monkeypatch.setattr(
        preparation, "resolve_regulatory_indexing_snapshot", lambda *_a, **_k: snapshot
    )
    deliveries: list[UUID] = []
    monkeypatch.setattr(
        tasks,
        "_enqueue_durable_regulatory_indexing",
        lambda **kw: deliveries.append(kw["job_id"]),
    )

    def prepare_items(
        *, job_id: UUID, expected_generation: int, tenant_id: str, db_session: Session
    ) -> UUID:
        assert tenant_id == get_current_tenant_id()
        assert jobs.persist_regulatory_indexing_preparation(
            db_session,
            job_id=job_id,
            expected_generation=expected_generation,
            prepare_items=lambda: [
                jobs.RegulatoryIndexingPreparedItem(
                    regulatory_chunk_id=row.id,
                    request_hash="b" * 64,
                    skip_context=False,
                )
                for row in rows
            ],
            resolved_input_hash_version="chunk-rows-v3",
            now=datetime.now(timezone.utc),
        )
        return job_id

    monkeypatch.setattr(
        preparation,
        "prepare_claimed_regulatory_indexing_job_from_chunks",
        prepare_items,
    )

    def deliver(token: UUID) -> None:
        tasks.index_user_file_impl(
            user_file_id=str(file.id),
            tenant_id=get_current_tenant_id(),
            index_request_attempt_id=str(token),
        )

    def rotate() -> UUID:
        request = session.get_one(
            UserFileProjectionRepair, file.id, populate_existing=True
        )
        request.updated_at = datetime.now(timezone.utc) - timedelta(minutes=5)
        session.commit()
        return dict(
            recover_user_file_index_requests(
                session, stale_before=datetime.now(timezone.utc) - timedelta(minutes=1)
            )
        )[file.id]

    if handoff == "binding_crash":

        def interrupted_binding(*_a: object, **_k: object) -> None:
            raise RuntimeError("binding interrupted")

        with monkeypatch.context() as patch:
            patch.setattr(durable, "bind_intake_durable_job", interrupted_binding)
            with pytest.raises(RuntimeError, match="binding interrupted"):
                deliver(attempt)
        assert (
            list(
                session.scalars(
                    select(RegulatoryIndexingJob).where(
                        RegulatoryIndexingJob.user_file_id == file.id
                    )
                )
            )
            == []
        )
        assert not deliveries
        session.expire_all()
        intake.refresh_intake(session, batch)
        attempt = dict(intake.retry_intake(session, batch))[file.id]
        session.commit()
    if handoff == "rotated_crash":
        attempt = rotate()

        def interrupted_ack(*_a: object, **_k: object) -> None:
            raise RuntimeError("worker disappeared before ACK")

        with monkeypatch.context() as patch:
            patch.setattr(tasks, "finish_user_file_index_request", interrupted_ack)
            with pytest.raises(RuntimeError, match="worker disappeared"):
                deliver(attempt)
        attempt = rotate()
    else:
        deliver(attempt)
    assert len(deliveries) == 1
    job = session.get_one(RegulatoryIndexingJob, deliveries[0], populate_existing=True)
    assert jobs.claim_regulatory_indexing_job(
        session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.CONTEXT_SUBMIT,
        expected_generation=job.lease_generation,
        now=datetime.now(timezone.utc),
    )
    session.refresh(job)
    assert jobs.fail_regulatory_indexing_job(
        session,
        job_id=job.id,
        expected_stage=RegulatoryIndexingStage.CONTEXT_SUBMIT,
        expected_generation=job.lease_generation,
        error_code="context:test",
        error_message="Fake provider rejected this batch",
        now=datetime.now(timezone.utc),
    )
    request = session.get_one(UserFileProjectionRepair, file.id, populate_existing=True)
    assert request.status == UserFileProjectionRepairStatus.FAILED
    intake.refresh_intake(session, batch)
    assert batch.status == "failed"
    assert _existing_rows(session, [sibling.user_file_id]) == sibling_before
    with pytest.raises(intake.IntakeConflictError, match="cleanup"):
        intake.retry_intake(session, batch)
    _complete_cleanup(session, job.id)
    original = _retained_job(session, job)
    if handoff in {"source_changed", "config_changed", "foreign_job"}:
        if handoff == "source_changed":
            rows[0].text += "changed after approval"
        else:
            proof = session.get_one(KVStore, "intake_durable_index_job:" + str(attempt))
            value = dict(cast(Mapping[str, JSON_ro], proof.value))
            value["config_sha256" if handoff == "config_changed" else "job_id"] = (
                "f" * 64 if handoff == "config_changed" else str(uuid4())
            )
            proof.value = value
        session.commit()
        request_before = session.get_one(UserFileProjectionRepair, file.id).attempt_id
        with pytest.raises(intake.IntakeConflictError):
            intake.retry_intake(session, batch)
        assert (
            session.get_one(UserFileProjectionRepair, file.id).attempt_id
            == request_before
        )
        assert _retained_job(session, job) == original
        assert _existing_rows(session, [sibling.user_file_id]) == sibling_before
        return
    for ordinal in (1, 2):
        replacements = dict(intake.retry_intake(session, batch))
        assert set(replacements) == {file.id}
        anchor = replacements[file.id]
        session.commit()
        # A scheduler redelivery retains one restart identity for this approved Retry.
        token = rotate()

        def current_config_must_not_be_used(*_a: object, **_k: object) -> None:
            raise AssertionError("retry must use its frozen configuration")

        monkeypatch.setattr(
            preparation,
            "resolve_regulatory_indexing_snapshot",
            current_config_must_not_be_used,
        )
        if handoff == "restart_binding_crash" and ordinal == 1:

            def interrupted_successor_binding(*_a: object, **_k: object) -> None:
                raise RuntimeError("successor binding interrupted")

            with monkeypatch.context() as patch:
                patch.setattr(
                    durable, "bind_intake_durable_job", interrupted_successor_binding
                )
                with pytest.raises(RuntimeError, match="successor binding interrupted"):
                    deliver(token)
            assert list(
                session.scalars(
                    select(RegulatoryIndexingJob.id).where(
                        RegulatoryIndexingJob.user_file_id == file.id
                    )
                )
            ) == [job.id]
            session.expire_all()
            intake.refresh_intake(session, batch)
            anchor = dict(intake.retry_intake(session, batch))[file.id]
            session.commit()
            token = rotate()
        deliver(token)
        successor = session.get_one(
            RegulatoryIndexingJob, deliveries[-1], populate_existing=True
        )
        assert successor.id != job.id
        assert successor.restart_ordinal == ordinal
        assert successor.restart_of_job_id == job.id
        assert successor.restart_attempt_id == anchor
        assert successor.config_snapshot == job.config_snapshot
        assert successor.stage == RegulatoryIndexingStage.CONTEXT_SUBMIT.value
        assert successor.status == RegulatoryIndexingJobStatus.QUEUED.value
        before = list(
            session.scalars(
                select(RegulatoryIndexingJob.id).where(
                    RegulatoryIndexingJob.user_file_id == file.id
                )
            )
        )
        deliver(token)
        assert (
            list(
                session.scalars(
                    select(RegulatoryIndexingJob.id).where(
                        RegulatoryIndexingJob.user_file_id == file.id
                    )
                )
            )
            == before
        )
        assert _retained_job(session, job) == original
        assert _existing_rows(session, [sibling.user_file_id]) == sibling_before
        assert jobs.claim_regulatory_indexing_job(
            session,
            job_id=successor.id,
            expected_stage=RegulatoryIndexingStage.CONTEXT_SUBMIT,
            expected_generation=successor.lease_generation,
            now=datetime.now(timezone.utc),
        )
        session.refresh(successor)
        assert jobs.fail_regulatory_indexing_job(
            session,
            job_id=successor.id,
            expected_stage=RegulatoryIndexingStage.CONTEXT_SUBMIT,
            expected_generation=successor.lease_generation,
            error_code="context:test",
            error_message="successor provider failure",
            now=datetime.now(timezone.utc),
        )
        _complete_cleanup(session, successor.id)
        intake.refresh_intake(session, batch)
        assert batch.status == "failed"
        job = successor
        original = _retained_job(session, job)
