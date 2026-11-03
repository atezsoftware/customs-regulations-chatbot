"""Incident retry retains normal GET/retry semantics and never stages label runs."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db.models import RegulatoryDocumentIntake, RegulatoryLabelingRun


@pytest.fixture
def operator(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    root = Path(__file__).resolve().parents[4] / "onyx/db"
    for name in ("_new_document_rg_operator", "_rg_labeling_retry_20261102"):
        spec = importlib.util.spec_from_file_location(name, root / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return module


@pytest.fixture
def run(operator: ModuleType) -> RegulatoryLabelingRun:
    return RegulatoryLabelingRun(
        id=operator.scope.LABELING_RUN,
        document_set_id=15,
        file_ids=[operator.scope.MAIN_FILE, operator.scope.PDF_FILE],
        status="completed_with_errors",
        stage="finished",
        total_chunks=662,
        completed_chunks=619,
        failed_chunks=43,
        stale_chunks=0,
        derived_chunks=35,
        unresolved_derived_chunks=2,
        cancel_requested=False,
        lease_token=None,
        lease_expires_at=None,
    )


def test_current_operator_cannot_mutate_before_final_sha_is_pinned(
    operator: ModuleType, run: RegulatoryLabelingRun
) -> None:
    operator.require_failed_run(run)
    assert operator.RETRY_IMAGE_SHA is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", uuid4()),
        ("status", "running"),
        ("stage", "projecting"),
        ("file_ids", []),
        ("completed_chunks", 618),
        ("failed_chunks", 44),
        ("stale_chunks", 1),
        ("derived_chunks", 36),
        ("unresolved_derived_chunks", 1),
        ("cancel_requested", True),
        ("lease_token", uuid4()),
    ],
)
def test_failed_run_identity_counters_and_owner_are_exact(
    operator: ModuleType, run: RegulatoryLabelingRun, field: str, value: object
) -> None:
    setattr(run, field, value)
    with pytest.raises(RuntimeError, match="terminal labeling receipt"):
        operator.require_failed_run(run)


@pytest.mark.parametrize("failure", [None, "get", "deliveries", "publication"])
def test_retry_uses_normal_get_then_retry_and_keeps_publication_unchanged(
    operator: ModuleType, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    session = MagicMock(spec=Session)
    batch = RegulatoryDocumentIntake(
        id=operator.scope.BATCH,
        status="preparing",
        stage="labeling",
        labeling_run_id=operator.scope.LABELING_RUN,
    )
    calls: list[str] = []
    monkeypatch.setattr(operator, "retry_plan", lambda *_: {"verified": True})
    epochs = MagicMock(
        side_effect=[[{"epoch": 0}], [{"epoch": 1 if failure == "publication" else 0}]]
    )
    monkeypatch.setattr(operator.scope, "unpublished_epoch_scope", epochs)
    verify = MagicMock()
    monkeypatch.setattr(operator.scope, "verify_scope", verify)

    def normal_get(*_args: object) -> dict[str, object]:
        calls.append("normal_get")
        batch.status = "partial"
        return {
            "status": "partial",
            "stage": "labeling",
            "labeling_failed_count": 42 if failure == "get" else 43,
        }

    def normal_retry(*_args: object) -> list[tuple[object, object]]:
        calls.append("normal_retry")
        batch.status, batch.stage, batch.error = "preparing", "chunking", None
        return [(uuid4(), uuid4())] if failure == "deliveries" else []

    monkeypatch.setattr(operator.intake, "intake_snapshot", normal_get)
    monkeypatch.setattr(operator.intake, "retry_intake", normal_retry)
    if failure:
        with pytest.raises(RuntimeError):
            operator.request_retry(session, batch)
        assert calls == (
            ["normal_get"] if failure == "get" else ["normal_get", "normal_retry"]
        )
    else:
        result = operator.request_retry(session, batch)
        assert calls == ["normal_get", "normal_retry"]
        assert result["operator_action"] == "normal_labeling_retry_requested"
        assert (
            result["delivery"]
            == "standard_enqueue_intake_batch_without_reprepare_request"
        )
        assert result["verified"] is True
        verify.assert_called_once_with(session, batch)
    session.commit.assert_not_called()
