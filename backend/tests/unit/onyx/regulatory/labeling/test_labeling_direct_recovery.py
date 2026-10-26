import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db.regulatory_labeling import RunLease
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchGateway
from onyx.regulatory.labeling import orchestrator


def _state(
    monkeypatch: pytest.MonkeyPatch, *, transport: str, expired: bool
) -> tuple[RunLease, SimpleNamespace, MagicMock, MagicMock]:
    lease = RunLease(run_id=uuid4(), generation=1, token=uuid4())
    deadline = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
        minutes=-1 if expired else 5
    )
    shard = SimpleNamespace(
        id=uuid4(),
        status="reconcile_required",
        submission_key="frozen-original-submission",
        reconcile_until=deadline,
        remote_job_name=None,
        failure_count=0,
    )
    session = MagicMock(spec=Session)
    session.__enter__.return_value = session
    monkeypatch.setattr(
        orchestrator, "get_session_with_current_tenant", lambda: session
    )
    monkeypatch.setattr(
        orchestrator.repository,
        "load_claimed_run",
        lambda *_args: SimpleNamespace(provider_binding={"transport": transport}),
    )
    monkeypatch.setattr(
        orchestrator,
        "LabelingProviderBinding",
        SimpleNamespace(
            model_validate=lambda payload: SimpleNamespace(
                transport=payload["transport"]
            )
        ),
    )
    monkeypatch.setattr(
        orchestrator.repository, "load_claimed_shard", lambda *_args: shard
    )
    monkeypatch.setattr(
        orchestrator.repository, "next_cancellable_shard", lambda *_args: shard
    )
    recorded = MagicMock()
    monkeypatch.setattr(orchestrator.repository, "record_shard_state", recorded)
    monkeypatch.setattr(orchestrator.repository, "release_run", MagicMock())
    gateway = MagicMock(spec=VertexBatchGateway)
    gateway.reconcile_submission.return_value = None
    return lease, shard, gateway, recorded


@pytest.mark.parametrize("expired", [False, True])
def test_interrupted_direct_submission_resumes_same_cache_without_deadline_wait(
    monkeypatch: pytest.MonkeyPatch, expired: bool
) -> None:
    lease, shard, gateway, recorded = _state(
        monkeypatch, transport="vertex_direct_v1", expired=expired
    )
    result = orchestrator._submit_or_reconcile(lease, shard.id, gateway, "public")
    gateway.reconcile_submission.assert_called_once_with("frozen-original-submission")
    assert recorded.call_args.kwargs["status"] == "prepared"
    assert shard.submission_key == "frozen-original-submission"
    assert result.outcome == orchestrator.LabelingStepOutcome.NEXT


@pytest.mark.parametrize(
    "transport", ["vertex_gcs_v1", "gemini_files_v1", "gemini_inline_api_key_v1"]
)
@pytest.mark.parametrize("expired", [False, True])
def test_existing_async_transports_retain_reconciliation_deadline(
    monkeypatch: pytest.MonkeyPatch, transport: str, expired: bool
) -> None:
    lease, shard, gateway, recorded = _state(
        monkeypatch, transport=transport, expired=expired
    )
    orchestrator._submit_or_reconcile(lease, shard.id, gateway, "public")
    assert recorded.call_args.kwargs["status"] == (
        "failed" if expired else "reconcile_required"
    )


def test_direct_cancellation_does_not_wait_for_a_nonexistent_remote_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease, _shard, gateway, recorded = _state(
        monkeypatch, transport="vertex_direct_v1", expired=False
    )
    result = orchestrator._cancel(lease, gateway)
    gateway.reconcile_submission.assert_not_called()
    assert recorded.call_args.kwargs["status"] == "cancelled"
    assert result.countdown_seconds == 0


def test_existing_async_cancellation_still_waits_for_unknown_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease, _shard, gateway, recorded = _state(
        monkeypatch, transport="vertex_gcs_v1", expired=False
    )
    result = orchestrator._cancel(lease, gateway)
    gateway.reconcile_submission.assert_called_once()
    assert recorded.call_args.kwargs["status"] == "reconcile_required"
    assert result.countdown_seconds == orchestrator.LABELING_POLL_SECONDS


@pytest.mark.parametrize(
    "failures", [0, orchestrator.LABELING_MAX_PROVIDER_FAILURES - 1]
)
def test_unexpected_direct_interruption_has_a_bounded_retry_budget(
    monkeypatch: pytest.MonkeyPatch, failures: int
) -> None:
    lease, shard, gateway, recorded = _state(
        monkeypatch, transport="vertex_direct_v1", expired=False
    )
    shard.status = "prepared"
    shard.failure_count = failures
    monkeypatch.setattr(
        orchestrator.repository,
        "mark_shard_submitting",
        lambda *_args, **_kwargs: shard,
    )
    monkeypatch.setattr(
        orchestrator.repository, "load_shard_requests", lambda *_args: []
    )
    gateway.submit.side_effect = RuntimeError("private SDK diagnostic")
    orchestrator._submit_or_reconcile(lease, shard.id, gateway, "public")
    state = recorded.call_args.kwargs
    assert state["status"] == ("prepared" if failures == 0 else "failed")
    assert state["increment_failure"] is True
    assert "private SDK diagnostic" not in state["error"]
