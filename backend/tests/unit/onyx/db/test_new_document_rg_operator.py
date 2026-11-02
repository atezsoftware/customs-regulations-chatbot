"""Incident operations must fail closed before a standard reprepare is requested."""

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db import document_intake_reprepare as repository
from onyx.db.models import RegulatoryDocumentIntake, RegulatoryDocumentIntakeReprepare
from onyx.regulatory.document_intake.models import IntakePreparationPolicy


@pytest.fixture
def operator() -> ModuleType:
    path = Path(__file__).resolve().parents[4] / "onyx/db/_new_document_rg_operator.py"
    spec = importlib.util.spec_from_file_location("rg_operator_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def batch(operator: ModuleType) -> RegulatoryDocumentIntake:
    return RegulatoryDocumentIntake(
        id=operator.BATCH,
        status="failed",
        stage="metadata",
        error="New-document metadata failed strict validation after bounded repair attempts",
        labeling_run_id=None,
        review_sha256=None,
    )


def test_reprepare_uses_reviewed_policy_and_source_guards(
    operator: ModuleType,
    batch: RegulatoryDocumentIntake,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = MagicMock(spec=Session)
    policy = IntakePreparationPolicy().model_dump(mode="json")
    documents = [operator.MAIN_DOCUMENT, operator.PDF_DOCUMENT]
    preview = {
        "expected_sha256": "a" * 64,
        "target_preparation_policy": policy,
        "documents": [{"document_id": value} for value in documents],
    }
    request = RegulatoryDocumentIntakeReprepare(
        id=uuid4(),
        status="queued",
        documents={
            value: {"target_chunking_version": "4", "target_preparation_policy": policy}
            for value in documents
        },
    )
    monkeypatch.setattr(repository, "latest_reprepare", MagicMock(return_value=None))
    monkeypatch.setattr(
        repository, "reprepare_snapshot", MagicMock(return_value=preview)
    )
    scope = MagicMock()
    monkeypatch.setattr(operator, "verify_scope", scope)
    monkeypatch.setattr(
        operator, "pdf_chunk_verification", lambda _: {"chunking_version": "3"}
    )

    def queue(*_args: object, **_kwargs: object) -> RegulatoryDocumentIntakeReprepare:
        batch.stage = "reprepare"
        return request

    queue_request = MagicMock(side_effect=queue)
    monkeypatch.setattr(repository, "request_reprepare", queue_request)
    report, request_id = operator.request_v4_reprepare(session, batch)

    queue_request.assert_called_once_with(
        session, batch, expected_sha256="a" * 64, idempotency_key=operator.REPREPARE_KEY
    )
    scope.assert_called_once_with(session, batch)
    assert request_id == request.id
    assert report["expected_review_sha256"] == "a" * 64
    assert report["publication_or_index_requested"] is False
    session.commit.assert_not_called()


@pytest.mark.parametrize("status", ["queued", "running", "prepared"])
def test_existing_request_is_inspected_without_duplicate_delivery(
    operator: ModuleType,
    batch: RegulatoryDocumentIntake,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    request = RegulatoryDocumentIntakeReprepare(
        id=uuid4(), idempotency_key=operator.REPREPARE_KEY, status=status
    )
    monkeypatch.setattr(repository, "latest_reprepare", MagicMock(return_value=request))
    queue = MagicMock()
    monkeypatch.setattr(repository, "request_reprepare", queue)
    report, delivery = operator.request_v4_reprepare(MagicMock(spec=Session), batch)
    assert report["operator_action"] == "existing_reprepare_inspected_only"
    assert delivery is None
    queue.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "preparing"),
        ("stage", "chunking"),
        ("error", "different"),
        ("review_sha256", "a" * 64),
        ("labeling_run_id", uuid4()),
    ],
)
def test_changed_incident_state_cannot_request_reprepare(
    operator: ModuleType,
    batch: RegulatoryDocumentIntake,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    setattr(batch, field, value)
    monkeypatch.setattr(repository, "latest_reprepare", MagicMock(return_value=None))
    preview = MagicMock()
    monkeypatch.setattr(repository, "reprepare_snapshot", preview)
    with pytest.raises(RuntimeError):
        operator.request_v4_reprepare(MagicMock(spec=Session), batch)
    preview.assert_not_called()


def test_canary_rejects_changed_wire_before_any_provider_call(
    operator: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "wire.txt"
    path.write_text("title: none | text: | changed source |")
    sessions = MagicMock()
    monkeypatch.setattr(operator, "get_session_with_current_tenant", sessions)
    with pytest.raises(RuntimeError, match="Fixture size or SHA changed"):
        operator.strict_v4_canary(path, "a" * 64)
    sessions.assert_not_called()


@pytest.mark.parametrize("change", ["policy", "documents", "review_sha"])
def test_changed_review_snapshot_cannot_queue(
    operator: ModuleType,
    batch: RegulatoryDocumentIntake,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    preview = {
        "expected_sha256": "a" * 64,
        "target_preparation_policy": IntakePreparationPolicy().model_dump(mode="json"),
        "documents": [
            {"document_id": operator.MAIN_DOCUMENT},
            {"document_id": operator.PDF_DOCUMENT},
        ],
    }
    if change == "policy":
        preview["target_preparation_policy"] = None
    elif change == "documents":
        preview["documents"] = [{"document_id": operator.PDF_DOCUMENT}]
    else:
        preview["expected_sha256"] = "invalid"
    monkeypatch.setattr(repository, "latest_reprepare", MagicMock(return_value=None))
    monkeypatch.setattr(repository, "reprepare_snapshot", MagicMock(return_value=preview))
    monkeypatch.setattr(
        operator, "pdf_chunk_verification", lambda _: {"chunking_version": "3"}
    )
    queue = MagicMock()
    monkeypatch.setattr(repository, "request_reprepare", queue)
    with pytest.raises(RuntimeError):
        operator.request_v4_reprepare(MagicMock(spec=Session), batch)
    queue.assert_not_called()
