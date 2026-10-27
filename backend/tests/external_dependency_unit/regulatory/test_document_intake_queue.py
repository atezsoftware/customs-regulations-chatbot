"""Durable preparation delivery and label-slot waiting use real PostgreSQL locks."""

from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import document_intake_queue as queue
from onyx.db import document_intake_reprepare as reprepare
from onyx.db import regulatory_labeling as labeling
from onyx.db.models import (
    AmendmentSourcePackage,
    DocumentSet,
    LLMProvider,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeReprepare,
    RegulatoryLabelingRun,
    User,
    UserRole,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    _ENVIRONMENT,
    _admitted,
    _completed_labels,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_owner as intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_session as intake_session,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    LabelingDatabase,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)


@pytest.mark.parametrize(
    "case",
    [
        "queued",
        "grace",
        "live",
        "expired_claim",
        "orphan_expiry",
        "failed",
        "ready",
        "foreign",
    ],
)
def test_only_due_unclaimed_scoped_intents_are_reserved(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    case: str,
) -> None:
    user, target, model = intake_owner
    batch, members = _admitted(intake_session, user, target, model)
    now = datetime.now(timezone.utc)
    batch.updated_at = now - timedelta(seconds=1000)
    if case == "grace":
        batch.updated_at = now - timedelta(seconds=915)
    elif case in {"live", "expired_claim"}:
        batch.lease_token = uuid4()
        batch.lease_expires_at = now + timedelta(seconds=60 if case == "live" else -60)
    elif case == "orphan_expiry":
        batch.lease_expires_at = now - timedelta(seconds=60)
    elif case in {"failed", "ready"}:
        batch.status = case
    elif case == "foreign":
        intake_session.get_one(
            AmendmentSourcePackage, batch.source_package_id
        ).environment = "foreign"
    intake_session.flush()
    before_chunks = [
        (row.id, row.text) for row in intake_session.scalars(select(RegulatoryChunk))
    ]
    deliveries = queue.reserve_intake_preparation_deliveries(
        intake_session, environment=_ENVIRONMENT, now=now
    )
    assert [delivery.batch_id for delivery in deliveries] == (
        [batch.id] if case == "queued" else []
    )
    assert all(delivery.reprepare_request_id is None for delivery in deliveries)
    assert [
        (row.id, row.text) for row in intake_session.scalars(select(RegulatoryChunk))
    ] == before_chunks
    assert len(members) == 2
    if case == "queued":
        intake.refresh_intake(intake_session, batch)
        assert batch.status == "preparing"
        assert intake.intake_snapshot(intake_session, batch)["stage"] == "queued"
        assert (
            queue.reserve_intake_preparation_deliveries(
                intake_session, environment=_ENVIRONMENT, now=now
            )
            == []
        )
        # A lost send is selected after the bounded reservation interval.
        assert (
            len(
                queue.reserve_intake_preparation_deliveries(
                    intake_session,
                    environment=_ENVIRONMENT,
                    now=now + timedelta(seconds=931),
                )
            )
            == 1
        )
    elif case == "orphan_expiry":
        intake.refresh_intake(intake_session, batch)
        assert batch.status == "failed"


@pytest.mark.parametrize("kind", ["initial", "reprepare"])
def test_late_deliveries_never_reclaim_expired_paid_work(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    kind: str,
) -> None:
    batch, _ = _admitted(intake_session, *intake_owner)
    old_token = uuid4()
    batch.lease_token = old_token
    batch.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    request = None
    if kind == "reprepare":
        request = RegulatoryDocumentIntakeReprepare(
            intake_id=batch.id,
            idempotency_key=uuid4(),
            expected_sha256="a" * 64,
            status="running",
            documents={},
        )
        intake_session.add(request)
        batch.stage = "reprepare"
    intake_session.flush()
    assert (
        reprepare.claim_reprepare(intake_session, batch.id, request.id)
        if request
        else intake.claim_intake(intake_session, batch.id)
    ) is None
    assert batch.lease_token == old_token
    intake.refresh_intake(intake_session, batch)
    assert batch.status == "failed"
    assert (
        reprepare.claim_reprepare(intake_session, batch.id, request.id)
        if request
        else intake.claim_intake(intake_session, batch.id)
    ) is None
    intake.retry_intake(intake_session, batch)
    assert (
        reprepare.claim_reprepare(intake_session, batch.id, request.id)
        if request
        else intake.claim_intake(intake_session, batch.id)
    ) is not None


def test_reprepare_delivery_uses_exact_latest_durable_request(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
) -> None:
    batch, _ = _admitted(intake_session, *intake_owner)
    now = datetime.now(timezone.utc)
    requests = [
        RegulatoryDocumentIntakeReprepare(
            intake_id=batch.id,
            idempotency_key=uuid4(),
            expected_sha256="a" * 64,
            status="queued",
            documents={},
            created_at=now + timedelta(seconds=index),
        )
        for index in range(2)
    ]
    intake_session.add_all(requests)
    batch.stage = "reprepare"
    batch.updated_at = now - timedelta(seconds=1000)
    intake_session.flush()
    assert reprepare.claim_reprepare(intake_session, batch.id, requests[0].id) is None
    assert batch.lease_token is None
    deliveries = queue.reserve_intake_preparation_deliveries(
        intake_session, environment=_ENVIRONMENT, now=now
    )
    assert deliveries == [queue.IntakePreparationDelivery(batch.id, requests[1].id)]
    assert intake.intake_snapshot(intake_session, batch)["stage"] == "queued_reprepare"
    assert (
        reprepare.claim_reprepare(intake_session, batch.id, requests[1].id) is not None
    )


def test_busy_label_slot_waits_then_resumes_only_prepared_intake_files(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old, old_members = _admitted(intake_session, *intake_owner)
    active = _completed_labels(intake_session, old, old_members)
    active.status = "running"
    batch, members = _admitted(intake_session, *intake_owner)
    token = intake.claim_intake(intake_session, batch.id)
    assert token is not None
    for member in members:
        intake.mark_intake_file_prepared(intake_session, batch.id, token, member.id)
    before = [
        (row.id, row.text, dict(row.chunk_metadata))
        for row in intake_session.scalars(select(RegulatoryChunk))
    ]
    assert intake.start_intake_labeling(intake_session, batch.id, token) is None
    assert batch.stage == "waiting_labeling" and batch.status == "preparing"
    assert batch.lease_token is None and batch.lease_expires_at is None
    assert batch.labeling_run_id is None
    assert (
        intake_session.scalar(select(func.count()).select_from(RegulatoryLabelingRun))
        == 1
    )
    now = datetime.now(timezone.utc)
    batch.updated_at = now - timedelta(seconds=120)
    assert (
        queue.resume_waiting_intake_labeling(
            intake_session, environment=_ENVIRONMENT, now=now
        )
        == []
    )
    active.status = "completed"
    binding = SimpleNamespace(
        model="gemini-3.5-flash",
        model_dump=lambda **_kwargs: {"model_configuration_id": intake_owner[2].id},
    )
    monkeypatch.setattr(
        intake, "resolve_intake_labeling_provider_binding", lambda *_a, **_k: binding
    )
    monkeypatch.setattr(
        labeling,
        "get_label_settings",
        lambda *_a: SimpleNamespace(taxonomy=active.taxonomy),
    )
    resumed = queue.resume_waiting_intake_labeling(
        intake_session, environment=_ENVIRONMENT, now=now + timedelta(seconds=61)
    )
    assert len(resumed) == 1 and resumed[0] == batch.labeling_run_id
    run = intake_session.get_one(RegulatoryLabelingRun, resumed[0])
    assert set(run.file_ids) == {str(member.user_file_id) for member in members}
    assert batch.stage == "labeling" and batch.status == "preparing"
    assert [
        (row.id, row.text, dict(row.chunk_metadata))
        for row in intake_session.scalars(select(RegulatoryChunk))
    ] == before
    assert (
        queue.resume_waiting_intake_labeling(
            intake_session, environment=_ENVIRONMENT, now=now + timedelta(seconds=122)
        )
        == []
    )


@pytest.fixture
def committed_queue(
    labeling_database: LabelingDatabase,
) -> Generator[tuple[Engine, str, list[UUID]], None, None]:
    """Commit only inside the dedicated schema removed by labeling_database."""
    environment = f"queue-contention-{uuid4()}"
    with Session(labeling_database.engine) as session:
        user = User(
            id=uuid4(),
            email=f"queue-{uuid4()}@test.local",
            hashed_password="unused",
            role=UserRole.ADMIN,
        )
        session.add(user)
        session.flush()
        target = DocumentSet(name=f"queue-{uuid4()}", user_id=user.id, is_public=False)
        model = ModelConfiguration(
            llm_provider=LLMProvider(name=f"queue-{uuid4()}", provider="vertex_ai"),
            name="fixture",
            is_visible=True,
        )
        session.add_all([target, model])
        session.flush()
        batches = []
        for index in range(2):
            package = AmendmentSourcePackage(
                document_set_id=target.id,
                created_by=user.id,
                environment=environment,
                idempotency_key=str(uuid4()),
                request_hash="a" * 64,
                input_spec={},
                status="ready",
            )
            session.add(package)
            session.flush()
            batch = RegulatoryDocumentIntake(
                source_package_id=package.id,
                document_set_id=target.id,
                created_by=user.id,
                idempotency_key=uuid4(),
                labeling_model_configuration_id=model.id,
                status="preparing",
                stage="chunking",
                updated_at=datetime.now(timezone.utc) - timedelta(seconds=2000 - index),
            )
            session.add(batch)
            batches.append(batch)
        session.commit()
        ids = [batch.id for batch in batches]
    yield labeling_database.engine, environment, ids


def test_concurrent_recovery_skips_locked_intent_then_recovers_it(
    committed_queue: tuple[Engine, str, list[UUID]],
) -> None:
    engine, environment, ids = committed_queue
    with Session(engine) as owner, Session(engine) as control:
        owner.get_one(RegulatoryDocumentIntake, ids[0], with_for_update=True)
        reserved = queue.reserve_intake_preparation_deliveries(
            control, environment=environment
        )
        control.commit()
        assert [item.batch_id for item in reserved] == [ids[1]]
        owner.rollback()
    with Session(engine) as control:
        reserved = queue.reserve_intake_preparation_deliveries(
            control, environment=environment
        )
        control.commit()
        assert [item.batch_id for item in reserved] == [ids[0]]


def test_concurrent_duplicate_deliveries_claim_only_once(
    committed_queue: tuple[Engine, str, list[UUID]],
) -> None:
    engine, _environment, ids = committed_queue
    barrier = Barrier(2)

    def claim() -> UUID | None:
        with Session(engine) as session:
            barrier.wait(timeout=5)
            token = intake.claim_intake(session, ids[0])
            session.commit()
            return token

    with ThreadPoolExecutor(max_workers=2) as pool:
        tokens = list(pool.map(lambda _index: claim(), range(2)))
    assert sum(token is not None for token in tokens) == 1


def test_cached_preclaim_row_cannot_overwrite_another_delivery_claim(
    committed_queue: tuple[Engine, str, list[UUID]],
) -> None:
    engine, _environment, ids = committed_queue
    with Session(engine) as stale:
        cached = stale.get_one(RegulatoryDocumentIntake, ids[0])
        assert cached.lease_token is None
        with Session(engine) as owner:
            token = intake.claim_intake(owner, ids[0])
            owner.commit()
        assert token is not None
        assert intake.claim_intake(stale, ids[0]) is None
        assert cached.lease_token == token
