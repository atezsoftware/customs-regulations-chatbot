"""Expired delivery succession must retain the intake's approved retry authority."""

from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import cast
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import document_intake_publication_recovery as recovery
from onyx.db.enums import UserFileProjectionRepairStatus, UserFileStatus
from onyx.db.models import (
    DocumentSet,
    KVStore,
    LLMProvider,
    ModelConfiguration,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    User,
    UserFile,
    UserFileProjectionRepair,
    UserRole,
)
from onyx.db.user_file import (
    finish_user_file_index_request,
    recover_user_file_index_requests,
    start_user_file_index_request,
)
from onyx.utils.special_types import JSON_ro
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR
from tests.external_dependency_unit.conftest import create_test_user
from tests.external_dependency_unit.regulatory.test_document_intake import (
    _existing_rows,
)
from tests.external_dependency_unit.regulatory.test_document_intake_index_retry import (
    _failed_initial_publication,
)
from tests.external_dependency_unit.regulatory.test_document_intake_publication_recovery import (
    _fixture,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    LabelingDatabase,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)


@pytest.fixture
def owned_session(labeling_database: LabelingDatabase) -> Iterator[Session]:
    token = CURRENT_TENANT_ID_CONTEXTVAR.set(labeling_database.schema)
    try:
        with labeling_database.engine.connect() as connection:
            connection.execute(
                text(f'SET search_path TO "{labeling_database.schema}", public')
            )
            connection.commit()
            try:
                with Session(connection) as session:
                    yield session
            finally:
                connection.rollback()
                connection.execute(text("RESET search_path"))
                connection.commit()
    finally:
        CURRENT_TENANT_ID_CONTEXTVAR.reset(token)


@pytest.fixture
def owner(
    owned_session: Session, monkeypatch: pytest.MonkeyPatch
) -> tuple[User, DocumentSet, ModelConfiguration]:
    session = owned_session
    user = create_test_user(session, "recovered-intake", UserRole.ADMIN)
    target = session.scalar(
        select(DocumentSet).where(DocumentSet.name == "PC Külliyatı")
    )
    if target is None:
        target = DocumentSet(name="PC Külliyatı", user_id=user.id, is_public=True)
        session.add(target)
    provider = LLMProvider(name=f"retry-{uuid4()}", provider="vertex_ai")
    model = ModelConfiguration(llm_provider=provider, name="fixture", is_visible=True)
    session.add(model)
    session.flush()
    monkeypatch.setattr(
        intake, "resolve_intake_labeling_provider_binding", lambda *_a, **_k: None
    )
    return user, target, model


def _pending_pair(
    session: Session,
    owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    closed_manifest: bool,
) -> tuple[
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryDocumentIntakeFile,
    UUID,
]:
    if closed_manifest:
        batch, subject, sibling, _manifest = _fixture(session, owner, monkeypatch)
    else:
        batch, members, _tokens = _failed_initial_publication(session, owner)
        subject, sibling = members
    first = dict(intake.retry_intake(session, batch))
    completed = session.get_one(UserFile, sibling.user_file_id)
    completed.status = UserFileStatus.COMPLETED
    sibling.status = "completed"
    sibling_request = session.get_one(UserFileProjectionRepair, completed.id)
    sibling_request.status = UserFileProjectionRepairStatus.SUCCEEDED
    request = session.get_one(UserFileProjectionRepair, subject.user_file_id)
    request.updated_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    session.commit()
    return batch, subject, sibling, first[subject.user_file_id]


@pytest.mark.parametrize("closed_manifest", [False, True])
def test_failed_recovered_delivery_can_retry_without_reindexing_completed_sibling(
    owned_session: Session,
    owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    closed_manifest: bool,
) -> None:
    session = owned_session
    batch, subject, sibling, approved = _pending_pair(
        session, owner, monkeypatch, closed_manifest
    )
    completed = session.get_one(UserFile, sibling.user_file_id)
    sibling_before = _existing_rows(session, [completed.id])
    retired = [approved]
    for _ in range(2):
        rotated = dict(
            recover_user_file_index_requests(
                session, stale_before=datetime.now(timezone.utc) - timedelta(minutes=1)
            )
        )[subject.user_file_id]
        assert rotated not in retired
        assert subject.approval_attempt_id == approved
        assert not start_user_file_index_request(
            session, subject.user_file_id, retired[-1]
        )
        retired.append(rotated)
        request = session.get_one(UserFileProjectionRepair, subject.user_file_id)
        request.updated_at = datetime.now(timezone.utc) - timedelta(minutes=5)
        session.commit()
    rotated = retired[-1]
    if closed_manifest:
        # Automatic redelivery must not silently authorize rebuilding frozen work.
        with pytest.raises(intake.IntakeConflictError):
            recovery.recovery_for_attempt(session, subject.user_file_id, rotated)
    assert start_user_file_index_request(session, subject.user_file_id, rotated)
    assert finish_user_file_index_request(
        session, subject.user_file_id, rotated, failure_code="index:ValueError"
    )
    intake.refresh_intake(session, batch)
    assert batch.status == "failed"
    replacement = dict(intake.retry_intake(session, batch))
    assert set(replacement) == {subject.user_file_id}
    assert replacement[subject.user_file_id] != rotated
    assert subject.approval_attempt_id == replacement[subject.user_file_id]
    assert not start_user_file_index_request(session, subject.user_file_id, rotated)
    assert _existing_rows(session, [completed.id]) == sibling_before
    if closed_manifest:
        assert (
            recovery.recovery_for_attempt(
                session, subject.user_file_id, replacement[subject.user_file_id]
            )
            is not None
        )


@pytest.mark.parametrize("closed_manifest", [False, True])
@pytest.mark.parametrize("proof_change", ["missing", "different_review"])
def test_retry_rejects_unproved_rotated_attempts(
    owned_session: Session,
    owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    closed_manifest: bool,
    proof_change: str,
) -> None:
    session = owned_session
    batch, subject, _sibling, approved = _pending_pair(
        session, owner, monkeypatch, closed_manifest
    )
    rotated = dict(
        recover_user_file_index_requests(
            session, stale_before=datetime.now(timezone.utc) - timedelta(minutes=1)
        )
    )[subject.user_file_id]
    assert start_user_file_index_request(session, subject.user_file_id, rotated)
    assert finish_user_file_index_request(
        session, subject.user_file_id, rotated, failure_code="index:ValueError"
    )
    intake.refresh_intake(session, batch)
    proof = session.scalar(
        select(KVStore).where(KVStore.value["attempt_id"].astext == str(rotated))
    )
    assert proof is not None and isinstance(proof.value, Mapping)
    if proof_change == "missing":
        session.delete(proof)
    else:
        proof.value = cast(JSON_ro, {**proof.value, "approved_review_sha256": "0" * 64})
    session.flush()
    before = _existing_rows(session, [subject.user_file_id])
    with pytest.raises(intake.IntakeConflictError):
        intake.retry_intake(session, batch)
    assert _existing_rows(session, [subject.user_file_id]) == before
    assert subject.approval_attempt_id == approved


@pytest.mark.parametrize("locked_model", [RegulatoryDocumentIntake, UserFile])
def test_busy_intake_or_file_skips_recovery_without_rotating_authority(
    owned_session: Session,
    labeling_database: LabelingDatabase,
    owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    locked_model: type[RegulatoryDocumentIntake] | type[UserFile],
) -> None:
    session = owned_session
    batch, subject, _sibling, approved = _pending_pair(
        session, owner, monkeypatch, False
    )
    engine = labeling_database.engine
    locked_id = (
        batch.id if locked_model is RegulatoryDocumentIntake else subject.user_file_id
    )

    def recover() -> list[tuple[UUID, UUID]]:
        with Session(engine) as worker:
            return recover_user_file_index_requests(
                worker, stale_before=datetime.now(timezone.utc) - timedelta(minutes=1)
            )

    with Session(engine) as blocking:
        blocking.scalar(
            select(locked_model).where(locked_model.id == locked_id).with_for_update()
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert subject.user_file_id not in dict(
                executor.submit(recover).result(timeout=10)
            )
        session.expire_all()
        assert (
            session.get_one(UserFileProjectionRepair, subject.user_file_id).attempt_id
            == approved
        )
        assert subject.approval_attempt_id == approved
        assert not list(
            session.scalars(
                select(KVStore).where(
                    KVStore.value["user_file_id"].astext == str(subject.user_file_id)
                )
            )
        )


def test_scheduler_does_not_authorize_an_unrecorded_foreign_attempt(
    owned_session: Session,
    owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = owned_session
    _batch, subject, _sibling, approved = _pending_pair(
        session, owner, monkeypatch, False
    )
    foreign = uuid4()
    request = session.get_one(UserFileProjectionRepair, subject.user_file_id)
    request.attempt_id = foreign
    session.commit()
    deliveries = recover_user_file_index_requests(
        session, stale_before=datetime.now(timezone.utc) - timedelta(minutes=1)
    )
    assert subject.user_file_id not in dict(deliveries)
    assert request.attempt_id == foreign and subject.approval_attempt_id == approved
    assert not list(
        session.scalars(
            select(KVStore).where(
                KVStore.value["user_file_id"].astext == str(subject.user_file_id)
            )
        )
    )
