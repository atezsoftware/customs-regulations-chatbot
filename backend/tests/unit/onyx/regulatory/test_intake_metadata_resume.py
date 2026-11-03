"""A metadata recovery delivery uses its frozen stage and preserves lost ownership."""

from contextlib import nullcontext
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from onyx.db import document_intake_metadata_recovery as recovery
from onyx.regulatory.document_intake import metadata_resume_job as job
from onyx.regulatory.document_intake.metadata_checkpoint import MetadataLeaseLost


@pytest.mark.parametrize("claimed", [False, True])
def test_only_claimed_metadata_work_runs_and_starts_normal_labeling(
    monkeypatch: pytest.MonkeyPatch, claimed: bool
) -> None:
    batch_id, token, run_id = uuid4(), uuid4(), uuid4()
    session = MagicMock()
    claim = MagicMock(return_value=token if claimed else None)
    execute, start, fail = MagicMock(), MagicMock(return_value=run_id), MagicMock()
    monkeypatch.setattr(job, "metadata_execution_lock", lambda _id: nullcontext(True))
    monkeypatch.setattr(
        job, "get_session_with_current_tenant", lambda: nullcontext(session)
    )
    monkeypatch.setattr(recovery, "claim_intake_metadata_recovery", claim)
    monkeypatch.setattr(job, "run_intake_metadata_locked", execute)
    monkeypatch.setattr(job.intake, "start_intake_labeling", start)
    monkeypatch.setattr(job.intake, "fail_intake", fail)
    assert job.run_document_intake_metadata_recovery(
        intake_id=batch_id, tenant_id="test"
    ) == (run_id if claimed else None)
    fail.assert_not_called()
    if claimed:
        execute.assert_called_once_with(
            intake_id=batch_id, lease_token=token, tenant_id="test"
        )
        start.assert_called_once_with(session, batch_id, token)
    else:
        execute.assert_not_called()
        start.assert_not_called()


@pytest.mark.parametrize(
    "error", [MetadataLeaseLost("expired"), ValueError("invalid evidence")]
)
def test_lost_metadata_owner_leaves_recovery_intent_but_permanent_error_fails(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    batch_id, token = uuid4(), uuid4()
    session, fail = MagicMock(), MagicMock()
    monkeypatch.setattr(job, "metadata_execution_lock", lambda _id: nullcontext(True))
    monkeypatch.setattr(
        job, "get_session_with_current_tenant", lambda: nullcontext(session)
    )
    monkeypatch.setattr(
        recovery, "claim_intake_metadata_recovery", lambda *_args: token
    )
    monkeypatch.setattr(job, "run_intake_metadata_locked", MagicMock(side_effect=error))
    monkeypatch.setattr(job.intake, "fail_intake", fail)
    if isinstance(error, MetadataLeaseLost):
        assert (
            job.run_document_intake_metadata_recovery(
                intake_id=batch_id, tenant_id="test"
            )
            is None
        )
        fail.assert_not_called()
    else:
        with pytest.raises(ValueError, match="invalid evidence"):
            job.run_document_intake_metadata_recovery(
                intake_id=batch_id, tenant_id="test"
            )
        fail.assert_called_once_with(session, batch_id, token, "invalid evidence")


def test_active_execution_does_not_claim_or_dispatch_another_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim, execute = MagicMock(), MagicMock()
    monkeypatch.setattr(job, "metadata_execution_lock", lambda _id: nullcontext(False))
    monkeypatch.setattr(recovery, "claim_intake_metadata_recovery", claim)
    monkeypatch.setattr(job, "run_intake_metadata_locked", execute)
    assert (
        job.run_document_intake_metadata_recovery(intake_id=uuid4(), tenant_id="test")
        is None
    )
    claim.assert_not_called()
    execute.assert_not_called()
