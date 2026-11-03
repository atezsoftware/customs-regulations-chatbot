"""Only normal GET may expose a fully verified replacement review as ready."""

import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
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
    module: ModuleType | None = None
    for name in ("_new_document_rg_operator", "_rg_review_refresh_20261103"):
        spec = importlib.util.spec_from_file_location(name, root / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    assert module is not None
    return module


@pytest.fixture
def completed(
    operator: ModuleType,
) -> tuple[RegulatoryDocumentIntake, RegulatoryLabelingRun]:
    identifier = uuid4()
    batch = RegulatoryDocumentIntake(
        labeling_run_id=identifier, labeling_model_configuration_id=7
    )
    run = RegulatoryLabelingRun(
        id=identifier,
        idempotency_key=operator.uuid5(
            operator.scope.BATCH, str(operator.scope.LABELING_RUN)
        ),
        document_set_id=15,
        requested_by_id=operator.UUID(operator.scope.OWNER),
        model_configuration_id=7,
        file_ids=[operator.scope.MAIN_FILE, operator.scope.PDF_FILE],
        status="completed",
        stage="finished",
        total_chunks=651,
        completed_chunks=651,
        failed_chunks=0,
        stale_chunks=0,
        derived_chunks=46,
        unresolved_derived_chunks=0,
        error=None,
        cancel_requested=False,
        lease_token=None,
        lease_expires_at=None,
    )
    return batch, run


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", uuid4()),
        ("idempotency_key", uuid4()),
        ("document_set_id", 16),
        ("requested_by_id", uuid4()),
        ("model_configuration_id", 8),
        ("file_ids", []),
        ("status", "completed_with_errors"),
        ("stage", "waiting"),
        ("total_chunks", 662),
        ("completed_chunks", 650),
        ("failed_chunks", 1),
        ("stale_chunks", 1),
        ("derived_chunks", 35),
        ("unresolved_derived_chunks", 1),
        ("error", "failure"),
        ("cancel_requested", True),
        ("lease_token", uuid4()),
    ],
)
def test_completed_replacement_requires_exact_identity_and_zero_failures(
    operator: ModuleType,
    completed: tuple[RegulatoryDocumentIntake, RegulatoryLabelingRun],
    field: str,
    value: object,
) -> None:
    batch, run = completed
    operator.validate_completed_run(run, batch)
    setattr(run, field, value)
    with pytest.raises(RuntimeError, match="exact completed"):
        operator.validate_completed_run(run, batch)


@pytest.mark.parametrize("missing", ["enabled", "run", "source"])
def test_mutation_gate_stops_before_database_access(
    operator: ModuleType, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    monkeypatch.setattr(operator, "REFRESH_REVIEW_ENABLED", missing != "enabled")
    monkeypatch.setattr(
        operator, "PINNED_REVIEW_RUN_ID", None if missing == "run" else uuid4()
    )
    monkeypatch.setattr(
        operator,
        "PINNED_REVIEW_SOURCE_SHA256",
        None if missing == "source" else "a" * 64,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review",
            "--expected-image-sha",
            operator.FINAL_IMAGE_SHA,
            "--observed-image-sha",
            operator.FINAL_IMAGE_SHA,
            "--refresh-review",
        ],
    )
    initialize = MagicMock(
        side_effect=AssertionError("No database before both exact pins")
    )
    monkeypatch.setattr(operator.SqlEngine, "init_engine", initialize)
    with pytest.raises(RuntimeError, match="disabled"):
        operator.main()
    initialize.assert_not_called()


@pytest.mark.parametrize(
    "failure", [None, "status", "digest", "source", "epoch", "attachment"]
)
def test_only_normal_get_can_transition_the_verified_review(
    operator: ModuleType, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    run_id = uuid4()
    batch = RegulatoryDocumentIntake(
        labeling_run_id=run_id, status="preparing", stage="labeling"
    )
    session = MagicMock(spec=Session)
    monkeypatch.setattr(operator, "REFRESH_REVIEW_ENABLED", True)
    monkeypatch.setattr(operator, "PINNED_REVIEW_RUN_ID", run_id)
    monkeypatch.setattr(operator, "PINNED_REVIEW_SOURCE_SHA256", "s")
    monkeypatch.setattr(
        operator,
        "review_plan",
        lambda *_: {"canonical_source_sha256": "s", "review_sha256": "r"},
    )
    epochs = MagicMock(
        side_effect=[[{"epoch": 0}], [{"epoch": 1 if failure == "epoch" else 0}]]
    )
    monkeypatch.setattr(operator.scope, "unpublished_epoch_scope", epochs)
    monkeypatch.setattr(operator.scope, "verify_scope", MagicMock())
    monkeypatch.setattr(
        operator, "source_digest", lambda *_: "changed" if failure == "source" else "s"
    )
    monkeypatch.setattr(
        operator.intake,
        "review_digest",
        lambda *_: "changed" if failure == "digest" else "r",
    )
    calls: list[str] = []

    def normal_get(*_args: object) -> dict[str, object]:
        calls.append("normal_get")
        batch.status, batch.stage, batch.error, batch.review_sha256 = (
            "ready",
            "review",
            None,
            "r",
        )
        return {
            "status": "partial" if failure == "status" else "ready",
            "stage": "review",
            "labeling_failed_count": 0,
            "active_document_count": 2,
            "documents": [
                {
                    "id": operator.scope.MAIN_DOCUMENT,
                    "user_file_id": operator.scope.MAIN_FILE,
                    "role": "root",
                    "document_links": []
                    if failure == "attachment"
                    else [{"target_document_id": operator.scope.PDF_DOCUMENT}],
                },
                {
                    "id": operator.scope.PDF_DOCUMENT,
                    "user_file_id": operator.scope.PDF_FILE,
                    "role": "annex",
                    "parent_document_id": operator.scope.MAIN_DOCUMENT,
                },
            ],
            "edit_review_sha256": "edit",
            "publication_review_sha256": "publication",
        }

    monkeypatch.setattr(operator.intake, "intake_snapshot", normal_get)
    if failure:
        with pytest.raises(RuntimeError):
            operator.refresh_review(session, batch)
    else:
        proof = operator.refresh_review(session, batch)
        assert proof["operator_action"] == "normal_get_review_refreshed"
        assert proof["status"] == "ready" and proof["stage"] == "review"
    assert calls == ["normal_get"]
    session.commit.assert_not_called()


def test_read_only_plan_never_calls_normal_get_or_commits(
    operator: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = MagicMock(spec=Session)

    @contextmanager
    def scoped_session() -> Iterator[Session]:
        yield session

    monkeypatch.setattr(operator, "get_session_with_current_tenant", scoped_session)
    monkeypatch.setattr(operator, "POSTGRES_DB", "customs-regulations-dev")
    monkeypatch.setattr(operator.scope.config, "REGULATORY_ANNEX_ENVIRONMENT", "dev")
    monkeypatch.setattr(operator, "get_current_tenant_id", lambda: "public")
    monkeypatch.setattr(operator, "set_is_ee_based_on_env_variable", lambda: None)
    monkeypatch.setattr(operator.SqlEngine, "init_engine", MagicMock())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review",
            "--expected-image-sha",
            operator.FINAL_IMAGE_SHA,
            "--observed-image-sha",
            operator.FINAL_IMAGE_SHA,
        ],
    )
    monkeypatch.setattr(operator, "review_plan", lambda *_: {"verified": True})
    normal_get = MagicMock(side_effect=AssertionError("Read-only plan cannot refresh"))
    monkeypatch.setattr(operator.intake, "intake_snapshot", normal_get)
    operator.main()
    assert str(session.execute.call_args.args[0]) == "SET TRANSACTION READ ONLY"
    session.rollback.assert_called_once()
    session.commit.assert_not_called()
    normal_get.assert_not_called()
