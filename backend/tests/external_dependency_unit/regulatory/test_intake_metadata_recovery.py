"""Expired frozen metadata owns bounded attempts; broker hints own no state."""

import json
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from uuid import UUID

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import document_intake_metadata_recovery as recovery
from onyx.db import labeling_configuration
from onyx.db.document_intake_metadata_checkpoint import require_metadata_lease
from onyx.db.models import (
    DocumentSet,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryFilePublication,
    User,
)
from onyx.regulatory.document_intake.metadata_checkpoint import MetadataLeaseLost
from tests.external_dependency_unit.regulatory.test_document_intake import _ENVIRONMENT
from tests.external_dependency_unit.regulatory.test_document_intake_metadata import (
    intake_owner as intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake_metadata import (
    intake_session as intake_session,
)
from tests.external_dependency_unit.regulatory.test_intake_metadata_checkpoint import (
    ready_checkpoint,
)


def expired_checkpoint(
    session: Session,
    owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[RegulatoryDocumentIntake, UUID, datetime]:
    batch, _members, old, binding = ready_checkpoint(session, owner)
    monkeypatch.setattr(
        labeling_configuration,
        "resolve_intake_labeling_provider_binding",
        lambda *_a, **_k: binding,
    )
    monkeypatch.setattr(recovery.config, "REGULATORY_ANNEX_ENVIRONMENT", _ENVIRONMENT)
    now = datetime.now(timezone.utc)
    batch.lease_expires_at = now - timedelta(seconds=1)
    batch.updated_at = now - timedelta(seconds=1000)
    session.flush()
    return batch, old, now


def canonical_digest(session: Session) -> str:
    return sha256(
        json.dumps(
            [
                dict(row)
                for row in session.execute(
                    select(RegulatoryChunk.__table__).order_by(RegulatoryChunk.id)
                ).mappings()
            ],
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()


def test_expired_fence_lost_hint_redelivery_and_three_attempt_cap(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = intake_session
    batch, old, now = expired_checkpoint(session, intake_owner, monkeypatch)
    before, checkpoint = canonical_digest(session), batch.metadata_checkpoint
    assert recovery.reserve_intake_metadata_deliveries(
        session, environment=_ENVIRONMENT, now=now
    ) == [batch.id]
    assert batch.metadata_recovery_count == 1 and batch.stage == "metadata"
    assert batch.lease_token is batch.lease_expires_at is None
    with pytest.raises(MetadataLeaseLost):
        require_metadata_lease(session, batch.id, old)
    assert (
        recovery.reserve_intake_metadata_deliveries(
            session, environment=_ENVIRONMENT, now=now
        )
        == []
    )
    assert recovery.reserve_intake_metadata_deliveries(
        session, environment=_ENVIRONMENT, now=now + timedelta(seconds=931)
    ) == [batch.id]
    assert batch.metadata_recovery_count == 1
    # UI polling cannot turn a checkpoint-backed broker wait into terminal failure.
    batch.updated_at = now - timedelta(seconds=2000)
    session.flush()
    intake.refresh_intake(session, batch)
    assert batch.status == "preparing" and batch.lease_token is None
    for attempt in range(1, 4):
        token = recovery.claim_intake_metadata_recovery(session, batch.id, now=now)
        assert token is not None and token != old
        assert (
            recovery.claim_intake_metadata_recovery(session, batch.id, now=now) is None
        )
        assert batch.metadata_recovery_count == attempt
        old = token
        batch.lease_expires_at = now - timedelta(seconds=1)
        session.flush()
        delivery = recovery.reserve_intake_metadata_deliveries(
            session, environment=_ENVIRONMENT, now=now
        )
        assert delivery == ([batch.id] if attempt < 3 else [])
    assert batch.status == "failed" and batch.metadata_recovery_count == 3
    assert "limit" in (batch.error or "")
    assert (
        batch.metadata_checkpoint == checkpoint and canonical_digest(session) == before
    )
    assert (
        recovery.reserve_intake_metadata_deliveries(
            session, environment=_ENVIRONMENT, now=now + timedelta(days=1)
        )
        == []
    )


@pytest.mark.parametrize(
    "kind",
    [
        "legacy",
        "live",
        "failed",
        "foreign",
        "writer",
        "source",
        "binding",
        "checkpoint",
        "recipe",
    ],
)
def test_unrecoverable_or_unowned_work_never_becomes_a_recovery(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    session = intake_session
    batch, token, now = expired_checkpoint(session, intake_owner, monkeypatch)
    environment = _ENVIRONMENT
    if kind == "legacy":
        batch.metadata_checkpoint = None
    elif kind == "live":
        batch.lease_expires_at = now + timedelta(seconds=60)
    elif kind == "failed":
        batch.status = "failed"
    elif kind == "foreign":
        environment = "foreign"
    elif kind == "writer":
        publication = session.scalar(select(RegulatoryFilePublication))
        assert publication is not None
        publication.epoch = 1
    elif kind == "source":
        row = session.scalar(select(RegulatoryChunk).order_by(RegulatoryChunk.id))
        assert row is not None
        row.text += " changed source"
    elif kind == "binding":
        monkeypatch.setattr(
            labeling_configuration,
            "resolve_intake_labeling_provider_binding",
            lambda *_a, **_k: (_ for _ in ()).throw(ValueError("changed model")),
        )
    elif kind == "recipe":
        monkeypatch.setattr(
            "onyx.regulatory.document_intake.metadata_job.metadata_recipe_sha256",
            lambda: "f" * 64,
        )
    else:
        batch.metadata_checkpoint = {"broken": True}
    session.flush()
    before = canonical_digest(session)
    assert (
        recovery.reserve_intake_metadata_deliveries(
            session,
            environment=environment,
            now=now,
        )
        == []
    )
    assert batch.metadata_recovery_count == 0 and canonical_digest(session) == before
    if kind in {"writer", "source", "binding", "checkpoint", "recipe"}:
        assert batch.status == "failed" and batch.lease_token is None
    else:
        assert batch.lease_token == token


@pytest.mark.parametrize("value", [-1, 4])
def test_database_enforces_recovery_count_cap(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    value: int,
) -> None:
    batch, _old, _now = expired_checkpoint(intake_session, intake_owner, monkeypatch)
    with pytest.raises(IntegrityError), intake_session.begin_nested():
        batch.metadata_recovery_count = value
        intake_session.flush()
