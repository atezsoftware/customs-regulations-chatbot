"""The incident operator requires an exact reviewed snapshot and never GET-promotes."""

import hashlib
import importlib.util
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db.models import (
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeReprepare,
    RegulatoryLabelingRun,
)


@pytest.fixture
def operator(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    directory = Path(__file__).resolve().parents[4] / "onyx/db"
    module: ModuleType | None = None
    for name in ("_new_document_rg_operator", "_rg_reprepare_v5_20261103"):
        spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    assert module is not None
    return module


@pytest.fixture
def terminal(
    operator: ModuleType,
) -> tuple[RegulatoryDocumentIntake, RegulatoryLabelingRun]:
    batch = RegulatoryDocumentIntake(
        id=operator.scope.BATCH,
        document_set_id=15,
        created_by=operator.UUID(operator.scope.OWNER),
        labeling_model_configuration_id=7,
        labeling_run_id=operator.OLD_RUN,
        status="partial",
        stage="labeling",
        error="Labeling has unresolved or failed chunks",
    )
    run = RegulatoryLabelingRun(
        id=operator.OLD_RUN,
        document_set_id=15,
        requested_by_id=batch.created_by,
        model_configuration_id=7,
        file_ids=[operator.scope.MAIN_FILE, operator.scope.PDF_FILE],
        status="completed_with_errors",
        stage="finished",
        total_chunks=651,
        completed_chunks=640,
        failed_chunks=11,
        stale_chunks=0,
        derived_chunks=46,
        unresolved_derived_chunks=0,
        cancel_requested=False,
    )
    return batch, run


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", uuid4()),
        ("document_set_id", 16),
        ("requested_by_id", uuid4()),
        ("model_configuration_id", 8),
        ("file_ids", []),
        ("status", "completed"),
        ("stage", "running"),
        ("total_chunks", 652),
        ("completed_chunks", 639),
        ("failed_chunks", 12),
        ("stale_chunks", 1),
        ("derived_chunks", 45),
        ("unresolved_derived_chunks", 1),
        ("cancel_requested", True),
        ("lease_token", uuid4()),
    ],
)
def test_original_terminal_scope_cannot_drift(
    operator: ModuleType,
    terminal: tuple[RegulatoryDocumentIntake, RegulatoryLabelingRun],
    field: str,
    value: object,
) -> None:
    batch, run = terminal
    operator.require_old_run(run, batch)
    setattr(run, field, value)
    with pytest.raises(RuntimeError, match="terminal v4"):
        operator.require_old_run(run, batch)


def test_policy_keeps_table_2048_and_body_512(operator: ModuleType) -> None:
    assert operator.policy("5") == {**operator.policy("4"), "chunking_version": "5"}
    assert operator.KEY == operator.uuid5(
        operator.scope.BATCH, "logical-matrix-v5-20261103"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("canonical_text", "altered source"),
        ("canonical_text_sha256", "f" * 64),
        ("source_span_exact", False),
    ],
)
def test_strict_matrix_canary_rejects_changed_exact_source_before_database(
    operator: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    source = (
        Path(__file__).resolve().parents[4]
        / "tests/fixtures/regulatory/rg_v5_largest_matrix_canary.json"
    )
    fixture = json.loads(source.read_text())
    fixture[field] = value
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(fixture, ensure_ascii=False))
    forbidden = MagicMock(
        side_effect=AssertionError("Invalid source must not reach DB or provider")
    )
    monkeypatch.setattr(operator.scope, "get_session_with_current_tenant", forbidden)
    with pytest.raises(RuntimeError, match="logical matrix fixture changed"):
        operator.scope.strict_v4_canary(
            path, hashlib.sha256(path.read_bytes()).hexdigest(), chunking_version="5"
        )
    forbidden.assert_not_called()


@pytest.mark.parametrize("missing", ["enabled", "image", "snapshot"])
def test_mutation_requires_separate_explicit_pins(
    operator: ModuleType, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    monkeypatch.setattr(operator, "REPREPARE_ENABLED", missing != "enabled")
    monkeypatch.setattr(
        operator, "FINAL_IMAGE_SHA", None if missing == "image" else "f" * 40
    )
    monkeypatch.setattr(
        operator, "APPROVED_PLAN_SHA", None if missing == "snapshot" else "a" * 64
    )
    session = MagicMock(spec=Session)
    with pytest.raises(RuntimeError, match="Mutation disabled"):
        operator.request_once(session, MagicMock())
    session.scalar.assert_not_called()


def test_existing_request_never_retries_or_reenqueues(
    operator: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(operator, "REPREPARE_ENABLED", True)
    monkeypatch.setattr(operator, "FINAL_IMAGE_SHA", "f" * 40)
    monkeypatch.setattr(operator, "APPROVED_PLAN_SHA", "a" * 64)
    request = MagicMock(id=uuid4(), status="failed")
    monkeypatch.setattr(operator, "existing_request", lambda *_: request)
    forbidden = MagicMock(
        side_effect=AssertionError("A duplicate must not replay preparation")
    )
    monkeypatch.setattr(operator, "reprepare_plan", forbidden)
    monkeypatch.setattr(operator, "request_reprepare", forbidden)
    report, enqueue = operator.request_once(MagicMock(spec=Session), MagicMock())
    assert report["action"] == "existing_v5_request_inspected_only" and enqueue is None
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "changed", ["batch", "previous", "snapshot", "version", "formatter", "policy"]
)
def test_existing_key_is_bound_to_exact_incident_and_target_policy(
    operator: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    changed: str,
) -> None:
    monkeypatch.setattr(operator, "APPROVED_PLAN_SHA", "a" * 64)
    documents = {
        key: {
            "target_chunking_version": "5",
            "target_formatter_version": 4,
            "target_preparation_policy": operator.policy("5"),
        }
        for key in (operator.scope.MAIN_DOCUMENT, operator.scope.PDF_DOCUMENT)
    }
    request = RegulatoryDocumentIntakeReprepare(
        id=uuid4(),
        intake_id=operator.scope.BATCH,
        previous_labeling_run_id=operator.OLD_RUN,
        expected_sha256="a" * 64,
        idempotency_key=operator.KEY,
        documents=documents,
    )
    session = MagicMock(spec=Session)
    session.scalar.return_value = request
    assert operator.existing_request(session) is request
    if changed in {"batch", "previous", "snapshot"}:
        setattr(
            request,
            {
                "batch": "intake_id",
                "previous": "previous_labeling_run_id",
                "snapshot": "expected_sha256",
            }[changed],
            "b" * 64 if changed == "snapshot" else uuid4(),
        )
    else:
        entry = documents[operator.scope.PDF_DOCUMENT]
        entry[
            {
                "version": "target_chunking_version",
                "formatter": "target_formatter_version",
                "policy": "target_preparation_policy",
            }[changed]
        ] = {"version": "4", "formatter": 5, "policy": operator.policy("4")}[changed]
    with pytest.raises(RuntimeError, match="reviewed identity"):
        operator.existing_request(session)


@pytest.mark.parametrize(
    "status,stage",
    [("preparing", "labeling"), ("ready", "review"), ("partial", "indexing")],
)
def test_preview_requires_autonomous_partial_before_normal_locked_snapshot(
    operator: ModuleType,
    terminal: tuple[RegulatoryDocumentIntake, RegulatoryLabelingRun],
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    stage: str,
) -> None:
    batch, run = terminal
    batch.status, batch.stage = status, stage
    session = MagicMock(spec=Session)
    session.get_one.return_value = run
    monkeypatch.setattr(operator, "require_files", lambda *_: None)
    locked_preview = MagicMock(side_effect=AssertionError("Not eligible for preview"))
    monkeypatch.setattr(operator, "reprepare_snapshot", locked_preview)
    with pytest.raises(RuntimeError, match="autonomous recovery"):
        operator.reprepare_plan(session, batch)
    locked_preview.assert_not_called()


@pytest.mark.parametrize("operation", ["status", "plan", "progress", "acceptance"])
def test_read_operations_never_commit_or_call_intake_get(
    operator: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(operator, "FINAL_IMAGE_SHA", "f" * 40)
    monkeypatch.setattr(operator, "POSTGRES_DB", "customs-regulations-dev")
    monkeypatch.setattr(operator.scope.config, "REGULATORY_ANNEX_ENVIRONMENT", "dev")
    monkeypatch.setattr(operator, "get_current_tenant_id", lambda: "public")
    monkeypatch.setattr(operator, "set_is_ee_based_on_env_variable", lambda: None)
    monkeypatch.setattr(operator.SqlEngine, "init_engine", lambda **_: None)
    session = MagicMock(spec=Session)

    @contextmanager
    def database() -> Iterator[Session]:
        yield session

    monkeypatch.setattr(operator, "get_session_with_current_tenant", database)
    monkeypatch.setattr(operator, "reprepare_plan", lambda *_: {"action": "plan"})
    monkeypatch.setattr(operator, "terminal_status", lambda *_: {"action": "status"})
    monkeypatch.setattr(operator, "progress", lambda *_, **__: {"action": operation})
    forbidden = MagicMock(
        side_effect=AssertionError("No review GET or transition is allowed")
    )
    monkeypatch.setattr(operator.intake, "intake_snapshot", forbidden)
    monkeypatch.setattr(operator.intake, "refresh_intake", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "operator",
            "--operation",
            operation,
            "--expected-image-sha",
            "f" * 40,
            "--observed-image-sha",
            "f" * 40,
        ],
    )
    operator.main()
    assert '"action"' in capsys.readouterr().out
    session.commit.assert_not_called()
    session.rollback.assert_called_once()
    forbidden.assert_not_called()
    statements = [str(call.args[0]) for call in session.execute.call_args_list]
    assert ("SET TRANSACTION READ ONLY" in statements) is (operation != "plan")
