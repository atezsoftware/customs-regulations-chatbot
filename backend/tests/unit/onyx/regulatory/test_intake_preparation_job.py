"""Duplicate source workers never enter another execution's preparation."""

from collections.abc import Callable
from contextlib import nullcontext
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from onyx.regulatory.document_intake import job, reprepare_job


@pytest.mark.parametrize("reprepare", [False, True])
def test_live_preparation_mutex_rejects_duplicate_before_claim(
    monkeypatch: pytest.MonkeyPatch,
    reprepare: bool,
) -> None:
    claim = MagicMock(return_value=None)
    module = reprepare_job if reprepare else job
    monkeypatch.setattr(
        module.repository, "claim_reprepare" if reprepare else "claim_intake", claim
    )
    monkeypatch.setattr(module, "get_session_with_current_tenant", MagicMock())
    monkeypatch.setattr(
        module,
        "preparation_execution_lock",
        lambda _id: nullcontext(False),
        raising=False,
    )
    result = (
        reprepare_job.run_document_intake_reprepare(
            intake_id=uuid4(), request_id=uuid4(), tenant_id="test"
        )
        if reprepare
        else job.run_document_intake(intake_id=uuid4(), tenant_id="test")
    )
    assert result is None
    claim.assert_not_called()


@pytest.mark.parametrize("lose_owner", [False, True])
def test_slow_source_preparation_keeps_lease_and_stops_before_metadata_after_loss(
    monkeypatch: pytest.MonkeyPatch, lose_owner: bool
) -> None:
    from threading import Event, current_thread, main_thread

    from onyx.db import document_intake_preparation_lease as preparation
    from onyx.regulatory.document_intake import metadata_job
    from onyx.regulatory.document_intake.metadata_checkpoint import MetadataLeaseLost
    from onyx.regulatory.document_intake.metadata_lease import maintain_metadata_lease

    token, run_id = uuid4(), uuid4()
    renewed = Event()
    outcomes: list[str] = []
    monkeypatch.setattr(
        job, "preparation_execution_lock", lambda _id: nullcontext(True)
    )
    monkeypatch.setattr(job, "get_session_with_current_tenant", MagicMock())
    monkeypatch.setattr(preparation, "get_session_with_current_tenant", MagicMock())
    monkeypatch.setattr(job.repository, "claim_intake", lambda *_args: token)
    monkeypatch.setattr(
        preparation,
        "maintain_metadata_lease",
        lambda heartbeat: maintain_metadata_lease(heartbeat, interval_seconds=0.005),
    )

    def heartbeat(*_args: object, **_kwargs: object) -> None:
        if current_thread() is not main_thread():
            renewed.set()
            if lose_owner:
                raise MetadataLeaseLost("Source ownership changed")

    def slow_sources(*, check_owner: Callable[[], None], **_arguments: object) -> None:
        assert renewed.wait(2)
        check_owner()
        outcomes.append("prepared")

    def metadata(**_arguments: object) -> None:
        outcomes.append("metadata")

    def ready(*_arguments: object) -> object:
        outcomes.append("ready")
        return run_id

    monkeypatch.setattr(preparation, "heartbeat_preparation", heartbeat)
    monkeypatch.setattr(job, "_prepare_intake_sources", slow_sources)
    monkeypatch.setattr(metadata_job, "run_intake_metadata", metadata)
    monkeypatch.setattr(job.repository, "start_intake_labeling", ready)
    result = job.run_document_intake(intake_id=uuid4(), tenant_id="test")
    assert outcomes == ([] if lose_owner else ["prepared", "metadata", "ready"])
    assert result == (None if lose_owner else run_id)
