"""Committed metadata intents exclude concurrent and stale owners in PostgreSQL."""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier
from typing import NamedTuple
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import document_intake_metadata_recovery as recovery
from onyx.db import labeling_configuration
from onyx.db.document_intake_metadata_checkpoint import metadata_execution_lock
from onyx.db.models import (
    DocumentSet,
    LLMProvider,
    ModelConfiguration,
    RegulatoryDocumentIntake,
    User,
    UserRole,
)
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR
from tests.external_dependency_unit.regulatory import test_document_intake as fixtures
from tests.external_dependency_unit.regulatory.test_intake_metadata_checkpoint import (
    ready_checkpoint,
)
from tests.external_dependency_unit.regulatory.test_intake_metadata_recovery import (
    canonical_digest,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    LabelingDatabase,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as _labeling_database,
)

labeling_database = _labeling_database


class _Work(NamedTuple):
    database: LabelingDatabase
    batch_id: UUID
    old_token: UUID
    environment: str
    now: datetime
    source_digest: str


@pytest.fixture
def committed_work(
    labeling_database: LabelingDatabase, monkeypatch: pytest.MonkeyPatch
) -> Iterator[_Work]:
    context = CURRENT_TENANT_ID_CONTEXTVAR.set(labeling_database.schema)
    environment = f"metadata-recovery-{uuid4().hex}"
    monkeypatch.setattr(fixtures, "_ENVIRONMENT", environment)
    monkeypatch.setattr(recovery.config, "REGULATORY_ANNEX_ENVIRONMENT", environment)
    try:
        with Session(labeling_database.engine, expire_on_commit=False) as session:
            user = User(
                id=uuid4(),
                email=f"recovery-{uuid4()}@test.local",
                hashed_password="unused",
                role=UserRole.ADMIN,
                is_active=True,
                is_verified=True,
                is_superuser=False,
            )
            target = session.scalar(
                select(DocumentSet).where(DocumentSet.name == "PC Külliyatı")
            )
            if target is None:
                target = DocumentSet(
                    name="PC Külliyatı",
                    user_id=user.id,
                    is_public=False,
                    is_up_to_date=True,
                )
                session.add(target)
            provider = LLMProvider(name=f"recovery-{uuid4()}", provider="vertex_ai")
            model = ModelConfiguration(
                llm_provider=provider, name="recovery-fixture", is_visible=True
            )
            session.add_all([user, target, model])
            session.flush()
            monkeypatch.setattr(
                intake, "resolve_intake_target", lambda *_a, **_k: target
            )
            monkeypatch.setattr(
                intake,
                "resolve_intake_labeling_provider_binding",
                lambda *_a, **_k: None,
            )
            batch, _members, old, binding = ready_checkpoint(
                session, (user, target, model)
            )
            monkeypatch.setattr(
                labeling_configuration,
                "resolve_intake_labeling_provider_binding",
                lambda *_a, **_k: binding,
            )
            now = datetime.now(timezone.utc)
            batch.lease_expires_at = now - timedelta(seconds=1)
            batch.updated_at = now - timedelta(seconds=1000)
            session.flush()
            digest = canonical_digest(session)
            session.commit()
            yield _Work(labeling_database, batch.id, old, environment, now, digest)
    finally:
        CURRENT_TENANT_ID_CONTEXTVAR.reset(context)


def test_two_recovery_ticks_reserve_one_committed_attempt(
    committed_work: _Work,
) -> None:
    work = committed_work
    barrier = Barrier(2)

    def reserve() -> list[UUID]:
        context = CURRENT_TENANT_ID_CONTEXTVAR.set(work.database.schema)
        try:
            with Session(work.database.engine) as session:
                barrier.wait(timeout=10)
                result = recovery.reserve_intake_metadata_deliveries(
                    session, environment=work.environment, now=work.now
                )
                session.commit()
                return result
        finally:
            CURRENT_TENANT_ID_CONTEXTVAR.reset(context)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: reserve(), range(2)))
    assert [identifier for result in results for identifier in result] == [
        work.batch_id
    ]
    with Session(work.database.engine) as session:
        batch = session.get_one(RegulatoryDocumentIntake, work.batch_id)
        assert batch.metadata_recovery_count == 1 and batch.lease_token is None
        assert canonical_digest(session) == work.source_digest


def test_alive_execution_mutex_prevents_takeover_even_after_lease_deadline(
    committed_work: _Work,
) -> None:
    work = committed_work
    with metadata_execution_lock(work.batch_id) as acquired:
        assert acquired
        with Session(work.database.engine) as session:
            assert (
                recovery.reserve_intake_metadata_deliveries(
                    session, environment=work.environment, now=work.now
                )
                == []
            )
            session.commit()
            batch = session.get_one(RegulatoryDocumentIntake, work.batch_id)
            assert (
                batch.metadata_recovery_count == 0
                and batch.lease_token == work.old_token
            )
    with Session(work.database.engine) as session:
        assert recovery.reserve_intake_metadata_deliveries(
            session, environment=work.environment, now=work.now
        ) == [work.batch_id]
        session.commit()


def test_stale_cached_failure_cannot_poison_new_pending_or_claimed_owner(
    committed_work: _Work,
) -> None:
    work = committed_work
    with Session(work.database.engine, expire_on_commit=False) as stale:
        cached = stale.get_one(RegulatoryDocumentIntake, work.batch_id)
        assert cached.lease_token == work.old_token
        stale.commit()
        with Session(work.database.engine) as fresh:
            assert recovery.reserve_intake_metadata_deliveries(
                fresh, environment=work.environment, now=work.now
            ) == [work.batch_id]
            fresh.commit()
        intake.fail_intake(
            stale, work.batch_id, work.old_token, "obsolete worker error"
        )
        stale.commit()
        assert cached.status == "preparing" and cached.lease_token is None
        with Session(work.database.engine) as fresh:
            token = recovery.claim_intake_metadata_recovery(
                fresh, work.batch_id, now=work.now
            )
            assert token is not None
            fresh.commit()
        intake.fail_intake(
            stale, work.batch_id, work.old_token, "obsolete worker error"
        )
        stale.commit()
        assert cached.status == "preparing" and cached.lease_token == token
        assert canonical_digest(stale) == work.source_digest
