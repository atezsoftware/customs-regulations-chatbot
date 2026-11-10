"""A slow metadata provider cannot silently lose the owned preparation lease."""

from collections.abc import Iterator
from contextlib import contextmanager
from threading import TIMEOUT_MAX, Event, Lock, Thread, current_thread, get_ident
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.orm import Session

from onyx.db import document_intake_metadata as repository
from onyx.regulatory.document_intake import metadata_job
from onyx.regulatory.document_intake.metadata_checkpoint import MetadataLeaseLost
from onyx.regulatory.document_intake.metadata_lease import maintain_metadata_lease
from shared_configs.contextvars import (
    CURRENT_TENANT_ID_CONTEXTVAR,
    get_current_tenant_id,
)
from tests.unit.onyx.regulatory.test_intake_metadata import _sources

_RECIPE = "56c3504da668fd7299ffd8bd3af7bf438b3224bbf98fc590e3b94e5f0cac45f2"


@pytest.mark.parametrize(
    "interval", [0, -1, float("nan"), float("inf"), -float("inf"), TIMEOUT_MAX * 2]
)
def test_invalid_interval_never_starts_renewal(interval: float) -> None:
    heartbeat = MagicMock()
    with pytest.raises(ValueError, match="interval"):
        with maintain_metadata_lease(heartbeat, interval_seconds=interval):
            pytest.fail("Invalid interval entered the lease scope")
    heartbeat.assert_not_called()


def test_periodic_renewal_uses_copied_tenant_and_stops_before_scope_exit() -> None:
    renewed = Event()
    calls: list[tuple[str, Thread]] = []
    token = CURRENT_TENANT_ID_CONTEXTVAR.set("metadata_tenant")

    def heartbeat() -> None:
        calls.append((get_current_tenant_id(), current_thread()))
        if len(calls) >= 3:
            renewed.set()

    try:
        with maintain_metadata_lease(heartbeat, interval_seconds=0.005) as owned:
            CURRENT_TENANT_ID_CONTEXTVAR.set("different_parent_tenant")
            assert renewed.wait(2), "Slow I/O received no periodic lease renewal"
        count = len(calls)
        assert count >= 3
        assert all(tenant == "metadata_tenant" for tenant, _thread in calls)
        assert all(not thread.is_alive() for _tenant, thread in calls)
        assert all(thread is not current_thread() for _tenant, thread in calls)
        with pytest.raises(RuntimeError, match="closed"):
            owned()
        assert not Event().wait(0.03)
        assert len(calls) == count
    finally:
        CURRENT_TENANT_ID_CONTEXTVAR.reset(token)


def test_manual_and_periodic_renewals_do_not_overlap() -> None:
    entered, release, manual_done = Event(), Event(), Event()
    lock = Lock()
    active = maximum = 0

    def heartbeat() -> None:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(active, maximum)
        entered.set()
        assert release.wait(2)
        with lock:
            active -= 1

    with maintain_metadata_lease(heartbeat, interval_seconds=0.005) as owned:
        assert entered.wait(2)

        def manual() -> None:
            owned()
            manual_done.set()

        caller = Thread(target=manual)
        caller.start()
        try:
            assert not manual_done.wait(0.03)
        finally:
            release.set()
            caller.join(2)
        assert manual_done.is_set() and not caller.is_alive()
    assert maximum == 1 and active == 0


def test_first_owner_failure_propagates_and_prevents_further_renewal() -> None:
    failed = Event()
    error = MetadataLeaseLost("Lease was replaced")
    calls = 0

    def heartbeat() -> None:
        nonlocal calls
        calls += 1
        failed.set()
        raise error

    with pytest.raises(MetadataLeaseLost) as raised:
        with maintain_metadata_lease(heartbeat, interval_seconds=0.005) as owned:
            assert failed.wait(2)
            with pytest.raises(MetadataLeaseLost) as direct:
                owned()
            assert direct.value is error
    assert raised.value is error
    assert calls == 1


def test_primary_extraction_error_is_not_masked_by_owner_failure() -> None:
    failed = Event()

    def heartbeat() -> None:
        failed.set()
        raise MetadataLeaseLost("Lease lost during I/O")

    primary = ValueError("Provider returned invalid metadata")
    with pytest.raises(ValueError) as raised:
        with maintain_metadata_lease(heartbeat, interval_seconds=0.005):
            assert failed.wait(2)
            raise primary
    assert raised.value is primary


def _job_boundary(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, list[MagicMock]]:
    sessions: list[MagicMock] = []

    @contextmanager
    def session_scope() -> Iterator[Session]:
        session = MagicMock(spec=Session)
        session.info = {"thread": get_ident()}
        sessions.append(session)
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr(metadata_job, "get_session_with_current_tenant", session_scope)
    monkeypatch.setattr(
        repository,
        "load_metadata_work",
        lambda *_args: (
            SimpleNamespace(created_by=uuid4(), labeling_model_configuration_id=7),
            [(uuid4(), _sources())],
        ),
    )
    monkeypatch.setattr(
        metadata_job,
        "resolve_intake_labeling_provider_binding",
        lambda *_args, **_kwargs: SimpleNamespace(fingerprint="f" * 64),
    )
    monkeypatch.setattr(
        metadata_job,
        "require_current_metadata_checkpoint",
        lambda *_args, **_kwargs: SimpleNamespace(
            recipe_sha256=_RECIPE, extraction_mode=None
        ),
    )
    monkeypatch.setattr(metadata_job, "resolve_labeling_gateway", MagicMock())
    monkeypatch.setattr(metadata_job, "metadata_recipe_sha256", lambda: _RECIPE)
    monkeypatch.setattr(
        metadata_job,
        "maintain_metadata_lease",
        lambda heartbeat: maintain_metadata_lease(heartbeat, interval_seconds=0.005),
    )
    apply = MagicMock()
    monkeypatch.setattr(repository, "apply_metadata_results", apply)
    return apply, sessions


@pytest.mark.parametrize("lose_owner", [False, True])
def test_metadata_job_keeps_renewing_during_slow_io_and_never_applies_after_loss(
    monkeypatch: pytest.MonkeyPatch, lose_owner: bool
) -> None:
    apply, sessions = _job_boundary(monkeypatch)
    renewed = Event()
    calls: list[tuple[Session, str, Thread]] = []
    result: metadata_job.MetadataResults = []

    def heartbeat(session: Session, _id: object, _token: object) -> None:
        assert session.info["thread"] == get_ident()
        calls.append((session, get_current_tenant_id(), current_thread()))
        if lose_owner:
            renewed.set()
            raise MetadataLeaseLost("Metadata owner changed")
        if len(calls) >= 3:
            renewed.set()

    def extract(**arguments: Any) -> metadata_job.MetadataResults:
        assert arguments["chunks"] == _sources()
        assert renewed.wait(2)
        return result

    monkeypatch.setattr(repository, "heartbeat_metadata_with_timeout", heartbeat)
    monkeypatch.setattr(metadata_job, "_extract_metadata_document", extract)
    token = CURRENT_TENANT_ID_CONTEXTVAR.set("metadata_tenant")
    try:
        if lose_owner:
            with pytest.raises(MetadataLeaseLost, match="owner changed"):
                metadata_job.run_intake_metadata_locked(
                    intake_id=uuid4(), lease_token=uuid4(), tenant_id="metadata_tenant"
                )
            apply.assert_not_called()
        else:
            metadata_job.run_intake_metadata_locked(
                intake_id=uuid4(), lease_token=uuid4(), tenant_id="metadata_tenant"
            )
            apply.assert_called_once()
            assert apply.call_args.kwargs["results"] is result
        assert calls and all(
            tenant == "metadata_tenant" for _session, tenant, _thread in calls
        )
        assert all(not thread.is_alive() for _session, _tenant, thread in calls)
        assert len({id(session) for session, _tenant, _thread in calls}) == len(calls)
        assert all(session.close.call_count == 1 for session in sessions)
    finally:
        CURRENT_TENANT_ID_CONTEXTVAR.reset(token)


def test_db_timeout_is_transaction_local_and_keeps_the_ownership_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = MagicMock(spec=Session)
    events: list[str] = []
    session.execute.side_effect = lambda statement: events.append(str(statement))
    guard = MagicMock(side_effect=lambda *_args: events.append("owner"))
    monkeypatch.setattr(repository, "heartbeat_metadata", guard)
    intake_id, token = uuid4(), uuid4()
    repository.heartbeat_metadata_with_timeout(session, intake_id, token)
    assert events == [
        "SET LOCAL lock_timeout = '5s'",
        "SET LOCAL statement_timeout = '10s'",
        "owner",
    ]
    guard.assert_called_once_with(session, intake_id, token)
    session.commit.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        OperationalError("heartbeat", {}, RuntimeError("lock timeout")),
        SQLAlchemyTimeoutError("Pool checkout timed out"),
    ],
)
def test_db_timeout_cannot_authorize_a_metadata_write(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    session = MagicMock(spec=Session)
    session.execute.side_effect = error
    guard = MagicMock()
    monkeypatch.setattr(repository, "heartbeat_metadata", guard)
    with pytest.raises(MetadataLeaseLost) as raised:
        repository.heartbeat_metadata_with_timeout(session, uuid4(), uuid4())
    assert raised.value.__cause__ is error
    guard.assert_not_called()
    session.commit.assert_not_called()


def test_current_metadata_semantic_contract_recipe_is_frozen() -> None:
    assert metadata_job.metadata_recipe_sha256() == _RECIPE
