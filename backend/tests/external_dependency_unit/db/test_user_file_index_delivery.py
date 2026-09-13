"""Explicit indexing intent survives loss of its disposable broker delivery."""

from collections.abc import Iterator
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db.enums import UserFileStatus
from onyx.db.models import UserFile, UserFileProjectionRepair
from onyx.db.regulatory_publication import PublicationStore
from onyx.document_index.publication_models import FileOwnership
from onyx.server.features.document_set import api
from tests.external_dependency_unit.conftest import create_test_user


@pytest.mark.parametrize("bulk", [False, True])
def test_index_request_survives_broker_failure(db_session: Session, bulk: bool) -> None:
    user = create_test_user(db_session, "durable_index")
    file = UserFile(
        id=uuid4(),
        user_id=user.id,
        file_id="test",
        name="test.txt",
        file_type="text/plain",
        status=UserFileStatus.CHUNKED,
    )
    db_session.add(file)
    db_session.commit()

    visible_intents: list[bool] = []

    def unavailable_broker(*_args: object, **_kwargs: object) -> None:
        # An independent transaction must see accepted intent before publication.
        with Session(db_session.get_bind()) as observer:
            visible_intents.append(
                observer.get(UserFileProjectionRepair, file.id) is not None
            )
        raise RuntimeError("broker unavailable")

    with (
        patch.object(api, "_get_editable_document_set_or_raise"),
        patch.object(
            api, "get_user_file_for_document_set_management", return_value=file
        ),
        patch.object(api, "fetch_user_files_for_document_set", return_value=[file]),
        patch.object(
            api,
            "_enqueue_user_file_indexing",
            side_effect=unavailable_broker,
        ),
    ):
        if bulk:
            assert (
                api.index_document_set_chunked_files(
                    1, user, db_session, "public"
                ).queued
                == 1
            )
        else:
            api.index_document_set_file(1, file.id, user, db_session, "public")
    assert visible_intents == [True]
    db_session.expire_all()
    assert db_session.get(UserFileProjectionRepair, file.id) is not None
    assert file.status is UserFileStatus.CHUNKED


@pytest.fixture
def requested_file(db_session: Session) -> UserFile:
    from onyx.db.user_file import claim_user_file_projection_repair

    user = create_test_user(db_session, "index_recovery")
    file = UserFile(
        id=uuid4(),
        user_id=user.id,
        file_id="test",
        name="test.txt",
        file_type="text/plain",
        status=UserFileStatus.CHUNKED,
    )
    db_session.add(file)
    db_session.commit()
    assert claim_user_file_projection_repair(db_session, file.id, initial_index=True)
    db_session.commit()
    return file


def test_expiry_rotates_token_and_rejects_stale_or_duplicate_workers(
    db_session: Session, requested_file: UserFile
) -> None:
    from datetime import datetime, timedelta, timezone

    from onyx.background.celery.tasks.user_file_processing import tasks
    from onyx.db.user_file import recover_user_file_index_requests

    identifier = requested_file.id
    repair = db_session.get(UserFileProjectionRepair, identifier)
    assert repair is not None
    stale_token = repair.attempt_id
    deliveries = recover_user_file_index_requests(
        db_session, stale_before=datetime.now(timezone.utc) + timedelta(seconds=1)
    )
    token = dict(deliveries)[identifier]
    assert token != stale_token

    def publish(*_args: object, **_kwargs: object) -> int:
        with Session(db_session.get_bind()) as other:
            file = other.get(UserFile, identifier)
            assert file is not None
            file.status = UserFileStatus.COMPLETED
            other.commit()
        return 3

    with (
        patch.object(tasks.app_configs, "REGULATORY_BATCH_INDEXING_ENABLED", False),
        patch(
            "onyx.regulatory.writer_publication.republish_user_file",
            side_effect=publish,
        ) as project,
    ):
        for attempt in (stale_token, token, token):
            tasks.index_user_file_impl(
                user_file_id=str(identifier),
                tenant_id="public",
                index_request_attempt_id=str(attempt),
            )
    assert project.call_count == 1
    db_session.refresh(repair)
    assert repair.status.value == "SUCCEEDED"


def test_live_publication_excludes_recovery_until_lease_release(
    db_session: Session, requested_file: UserFile
) -> None:
    from datetime import datetime, timedelta, timezone

    from onyx.db.user_file import (
        claim_user_file_projection_repair,
        recover_user_file_index_requests,
        start_user_file_index_request,
    )
    from onyx.regulatory.amendments.annexes.publication_execution import LEASE_TTL

    identifier = requested_file.id
    repair = db_session.get(UserFileProjectionRepair, identifier)
    assert repair is not None
    token = repair.attempt_id
    assert start_user_file_index_request(db_session, identifier, token)
    authority = _publication_store()
    owner = authority.acquire(identifier, owner_id=uuid4(), ttl=LEASE_TTL)
    future = datetime.now(timezone.utc) + timedelta(hours=3)
    try:
        assert identifier not in dict(
            recover_user_file_index_requests(db_session, stale_before=future)
        )
        assert (
            claim_user_file_projection_repair(
                db_session, identifier, initial_index=True, now=future
            )
            is None
        )
        db_session.rollback()
    finally:
        authority.release(owner)
    assert identifier in dict(
        recover_user_file_index_requests(db_session, stale_before=future)
    )


@pytest.mark.parametrize(
    "status",
    [
        UserFileStatus.CANCELED,
        UserFileStatus.DELETING,
        UserFileStatus.FAILED,
        UserFileStatus.COMPLETED,
    ],
)
def test_terminal_file_state_is_preserved(
    db_session: Session, requested_file: UserFile, status: UserFileStatus
) -> None:
    from onyx.background.celery.tasks.user_file_processing import tasks

    requested_file.status = status
    db_session.commit()
    repair = db_session.get(UserFileProjectionRepair, requested_file.id)
    assert repair is not None
    with patch("onyx.regulatory.writer_publication.republish_user_file") as project:
        if status is UserFileStatus.COMPLETED:
            tasks.index_user_file_impl(
                user_file_id=str(requested_file.id),
                tenant_id="public",
                index_request_attempt_id=str(repair.attempt_id),
            )
        else:
            with pytest.raises(ValueError):
                tasks.index_user_file_impl(
                    user_file_id=str(requested_file.id),
                    tenant_id="public",
                    index_request_attempt_id=str(repair.attempt_id),
                )
    project.assert_not_called()
    db_session.refresh(requested_file)
    assert requested_file.status is status


def test_unrequested_review_file_is_not_recovered(db_session: Session) -> None:
    from datetime import datetime, timedelta, timezone

    from onyx.db.user_file import recover_user_file_index_requests

    user = create_test_user(db_session, "review_only")
    file = UserFile(
        id=uuid4(),
        user_id=user.id,
        file_id="review",
        name="review.txt",
        file_type="text/plain",
        status=UserFileStatus.CHUNKED,
    )
    db_session.add(file)
    db_session.commit()
    deliveries = recover_user_file_index_requests(
        db_session, stale_before=datetime.now(timezone.utc) + timedelta(hours=3)
    )
    assert file.id not in dict(deliveries)


def test_terminal_projection_error_is_not_automatically_retried(
    db_session: Session, requested_file: UserFile
) -> None:
    from datetime import datetime, timedelta, timezone

    from onyx.background.celery.tasks.user_file_processing import tasks
    from onyx.db.user_file import recover_user_file_index_requests

    repair = db_session.get(UserFileProjectionRepair, requested_file.id)
    assert repair is not None
    with (
        patch.object(tasks.app_configs, "REGULATORY_BATCH_INDEXING_ENABLED", False),
        patch(
            "onyx.regulatory.writer_publication.republish_user_file",
            side_effect=ValueError("invalid configuration"),
        ),
        pytest.raises(ValueError),
    ):
        tasks.index_user_file_impl(
            user_file_id=str(requested_file.id),
            tenant_id="public",
            index_request_attempt_id=str(repair.attempt_id),
        )
    db_session.refresh(requested_file)
    assert requested_file.status is UserFileStatus.FAILED
    assert requested_file.id not in dict(
        recover_user_file_index_requests(
            db_session, stale_before=datetime.now(timezone.utc) + timedelta(hours=3)
        )
    )


def test_scheduler_redelivers_expired_intent_with_expiration(
    db_session: Session, requested_file: UserFile
) -> None:
    from datetime import datetime, timedelta, timezone
    from unittest.mock import MagicMock

    from onyx.background.celery.tasks.user_file_processing import tasks
    from onyx.configs.constants import OnyxCeleryTask

    repair = db_session.get(UserFileProjectionRepair, requested_file.id)
    assert repair is not None
    repair.updated_at = datetime.now(timezone.utc) - timedelta(days=1)
    db_session.commit()
    redis = MagicMock()
    redis.lock.return_value.acquire.return_value = True
    with (
        patch.object(tasks, "get_redis_client", return_value=redis),
        patch.object(tasks, "celery_get_broker_client", return_value=redis),
        patch.object(tasks, "celery_get_queue_length", return_value=0),
        patch.object(tasks.check_user_file_processing.app, "send_task") as send,
    ):
        tasks.check_user_file_processing.run(tenant_id="public")
    deliveries = [
        call
        for call in send.call_args_list
        if call.args[0] == OnyxCeleryTask.INDEX_SINGLE_USER_FILE
        and call.kwargs["kwargs"]["user_file_id"] == str(requested_file.id)
    ]
    assert len(deliveries) == 1
    assert deliveries[0].kwargs["expires"] > 0
    db_session.refresh(repair)
    assert deliveries[0].kwargs["kwargs"]["index_request_attempt_id"] == str(
        repair.attempt_id
    )


def test_batch_mode_hands_explicit_intent_to_durable_job(
    db_session: Session, requested_file: UserFile
) -> None:
    from onyx.background.celery.tasks.user_file_processing import tasks

    repair = db_session.get(UserFileProjectionRepair, requested_file.id)
    assert repair is not None
    job_id = uuid4()
    with (
        patch.object(tasks.app_configs, "REGULATORY_BATCH_INDEXING_ENABLED", True),
        patch.object(
            tasks, "prepare_regulatory_indexing_job_from_chunks", return_value=job_id
        ) as prepare,
        patch.object(tasks, "_enqueue_durable_regulatory_indexing") as enqueue,
        patch("onyx.regulatory.writer_publication.republish_user_file") as project,
    ):
        tasks.index_user_file_impl(
            user_file_id=str(requested_file.id),
            tenant_id="public",
            index_request_attempt_id=str(repair.attempt_id),
        )
    assert prepare.call_count == 1
    assert enqueue.call_args.kwargs["job_id"] == job_id
    project.assert_not_called()
    db_session.refresh(repair)
    assert repair.status.value == "SUCCEEDED"


def _publication_store() -> PublicationStore:
    from onyx.db.regulatory_publication import PublicationStore
    from onyx.document_index.publication_models import PublicationScope
    from onyx.regulatory.amendments.annexes import config

    return PublicationStore(
        PublicationScope(
            tenant_id="public",
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            database_identity=config.ANNEX_DATABASE_IDENTITY,
        )
    )


def test_crashed_publication_lease_remains_recoverable(
    db_session: Session, requested_file: UserFile
) -> None:
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from onyx.background.celery.tasks.user_file_processing import tasks
    from onyx.db.user_file import recover_user_file_index_requests
    from onyx.regulatory import writer_publication
    from onyx.regulatory.amendments.annexes.publication_execution import LEASE_TTL

    authority = _publication_store()
    owner = authority.acquire(requested_file.id, owner_id=uuid4(), ttl=LEASE_TTL)
    repair = db_session.get(UserFileProjectionRepair, requested_file.id)
    assert repair is not None
    with (
        patch.object(tasks.app_configs, "REGULATORY_BATCH_INDEXING_ENABLED", False),
        patch("onyx.document_index.elasticsearch.client.ElasticsearchClient") as index,
    ):
        try:
            tasks.index_user_file_impl(
                user_file_id=str(requested_file.id),
                tenant_id="public",
                index_request_attempt_id=str(repair.attempt_id),
            )
        finally:
            authority.release(owner)
    index.assert_not_called()
    db_session.refresh(repair)
    db_session.refresh(requested_file)
    assert repair.status.value == "PENDING"
    assert requested_file.status is UserFileStatus.CHUNKED
    deliveries = recover_user_file_index_requests(
        db_session, stale_before=datetime.now(timezone.utc) + timedelta(minutes=3)
    )
    token = dict(deliveries)[requested_file.id]
    inputs = SimpleNamespace(
        file=requested_file, canonical=[object()], settings=[], bindings=[]
    )
    with (
        patch.object(tasks.app_configs, "REGULATORY_BATCH_INDEXING_ENABLED", False),
        patch(
            "onyx.db.regulatory_writer_publication.load_owned_writer_inputs",
            return_value=inputs,
        ),
        patch("onyx.document_index.elasticsearch.client.ElasticsearchClient"),
        patch(
            "onyx.regulatory.writer_projection.prepare_owned_correction",
            return_value=MagicMock(),
        ),
        patch.object(writer_publication, "execute_writer_publication") as publish,
    ):
        tasks.index_user_file_impl(
            user_file_id=str(requested_file.id),
            tenant_id="public",
            index_request_attempt_id=str(token),
        )
    assert publish.call_count == 1
    db_session.refresh(repair)
    assert repair.status.value == "SUCCEEDED"


def test_concurrent_deletion_does_not_wait_on_index_request_transaction(
    db_session: Session, requested_file: UserFile
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import contextmanager
    from threading import Event

    from sqlalchemy import text

    from onyx.background.celery.tasks.user_file_processing import tasks
    from onyx.db import regulatory_writer_publication as deletion
    from onyx.db.engine.sql_engine import get_session_with_tenant
    from onyx.regulatory.amendments.annexes.publication_execution import LEASE_TTL

    requested_file.chunk_count = 0
    db_session.commit()
    identifier = requested_file.id
    repair = db_session.get(UserFileProjectionRepair, identifier)
    assert repair is not None
    token = str(repair.attempt_id)
    paused, resume, deleted = Event(), Event(), Event()
    original_status = tasks.get_user_file_index_request_status

    def pause_after_status(session: Session, file_id: UUID) -> UserFileStatus | None:
        status = original_status(session, file_id)
        paused.set()
        assert resume.wait(8)
        return status

    @contextmanager
    def bounded_session(*, tenant_id: str) -> Iterator[Session]:
        with get_session_with_tenant(tenant_id=tenant_id) as session:
            session.execute(text("SET LOCAL statement_timeout = '4s'"))
            yield session

    def run_index() -> None:
        try:
            tasks.index_user_file_impl(
                user_file_id=str(identifier),
                tenant_id="public",
                index_request_attempt_id=token,
            )
        except ValueError:
            pass  # Deletion can legitimately win before publication acquisition.

    def run_delete(owner: FileOwnership) -> None:
        deletion.finish_owned_deletion(owner, without_index_authority=True)
        deleted.set()

    authority = _publication_store()
    with (
        patch.object(
            tasks, "get_user_file_index_request_status", side_effect=pause_after_status
        ),
        patch.object(tasks.app_configs, "REGULATORY_BATCH_INDEXING_ENABLED", False),
        patch.object(deletion, "get_session_with_tenant", side_effect=bounded_session),
        patch("onyx.document_index.elasticsearch.client.ElasticsearchClient") as index,
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        indexing = executor.submit(run_index)
        assert paused.wait(3)
        owner = authority.acquire(identifier, owner_id=uuid4(), ttl=LEASE_TTL)
        deletion.begin_owned_deletion(owner)
        deleting = executor.submit(run_delete, owner)
        try:
            completed_before_resume = deleted.wait(2)
        finally:
            resume.set()
        indexing.result(timeout=8)
        deleting.result(timeout=8)
    assert completed_before_resume
    index.assert_not_called()
    db_session.expire_all()
    assert db_session.get(UserFile, identifier) is None


def test_superseded_claim_cannot_enter_publication_or_finalize_current_request(
    db_session: Session, requested_file: UserFile
) -> None:
    from datetime import datetime, timedelta, timezone

    from onyx.db.user_file import (
        UserFileIndexRequestSuperseded,
        finish_user_file_index_request,
        recover_user_file_index_requests,
        start_user_file_index_request,
    )
    from onyx.regulatory.writer_publication import republish_user_file

    identifier = requested_file.id
    repair = db_session.get(UserFileProjectionRepair, identifier)
    assert repair is not None
    old_token = repair.attempt_id
    assert start_user_file_index_request(db_session, identifier, old_token)
    deliveries = recover_user_file_index_requests(
        db_session, stale_before=datetime.now(timezone.utc) + timedelta(minutes=3)
    )
    token = dict(deliveries)[identifier]
    assert start_user_file_index_request(db_session, identifier, token)
    with (
        patch("onyx.document_index.elasticsearch.client.ElasticsearchClient") as index,
        pytest.raises(UserFileIndexRequestSuperseded),
    ):
        republish_user_file(
            identifier,
            "public",
            include_chunked=True,
            index_request_attempt_id=old_token,
        )
    index.assert_not_called()
    assert not finish_user_file_index_request(db_session, identifier, old_token)
    assert not finish_user_file_index_request(
        db_session, identifier, old_token, failure_code="stale_failure"
    )
    assert not finish_user_file_index_request(
        db_session, identifier, old_token, retry=True
    )
    db_session.refresh(repair)
    db_session.refresh(requested_file)
    assert repair.status.value == "RUNNING"
    assert repair.attempt_id == token
    assert requested_file.status is UserFileStatus.CHUNKED


def test_first_publication_acquisition_cannot_race_recovery_of_absent_authority(
    db_session: Session, requested_file: UserFile
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime, timedelta, timezone
    from threading import Event
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from sqlalchemy import Select
    from sqlalchemy.sql import Executable

    from onyx.db.models import RegulatoryFilePublication
    from onyx.db.user_file import (
        UserFileIndexRequestSuperseded,
        recover_user_file_index_requests,
        start_user_file_index_request,
        validate_user_file_index_request,
    )
    from onyx.regulatory import writer_publication

    identifier = requested_file.id
    repair = db_session.get(UserFileProjectionRepair, identifier)
    assert repair is not None
    old_token = repair.attempt_id
    assert db_session.get(RegulatoryFilePublication, identifier) is None
    assert start_user_file_index_request(db_session, identifier, old_token)
    repair.updated_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    db_session.commit()
    absence_seen, resume_recovery = Event(), Event()
    validated, resume_writer = Event(), Event()

    def recover() -> list[tuple[UUID, UUID]]:
        with Session(db_session.get_bind()) as session:
            original_scalar = session.scalar

            def pause_after_absence(statement: Executable) -> object:
                if (
                    isinstance(statement, Select)
                    and str(statement).startswith(
                        "SELECT user_file_projection_repair.user_file_id,"
                    )
                    and identifier in statement.compile().params.values()
                ):
                    absence_seen.set()
                    assert resume_recovery.wait(5)
                return original_scalar(statement)

            with patch.object(session, "scalar", side_effect=pause_after_absence):
                return recover_user_file_index_requests(
                    session,
                    stale_before=datetime.now(timezone.utc) - timedelta(minutes=1),
                )

    def pause_after_validation(owner: FileOwnership, token: UUID) -> None:
        validate_user_file_index_request(owner, token)
        validated.set()
        assert resume_writer.wait(5)

    def publish() -> None:
        try:
            writer_publication.republish_user_file(
                identifier,
                "public",
                include_chunked=True,
                index_request_attempt_id=old_token,
            )
        except UserFileIndexRequestSuperseded:
            pass

    inputs = SimpleNamespace(
        file=requested_file, canonical=[object()], settings=[], bindings=[]
    )
    with (
        patch(
            "onyx.db.user_file.validate_user_file_index_request",
            side_effect=pause_after_validation,
        ),
        patch(
            "onyx.db.regulatory_writer_publication.load_owned_writer_inputs",
            return_value=inputs,
        ),
        patch("onyx.document_index.elasticsearch.client.ElasticsearchClient"),
        patch(
            "onyx.regulatory.writer_projection.prepare_owned_correction",
            return_value=MagicMock(),
        ),
        patch.object(writer_publication, "execute_writer_publication") as write_index,
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        recovering = executor.submit(recover)
        assert absence_seen.wait(3)
        publishing = executor.submit(publish)
        # With synchronization, acquisition waits until recovery has committed.
        validated.wait(1)
        resume_recovery.set()
        deliveries = recovering.result(timeout=5)
        resume_writer.set()
        publishing.result(timeout=5)
    assert dict(deliveries)[identifier] != old_token
    write_index.assert_not_called()
