import datetime
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from onyx.auth.users import current_user
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import Permission
from onyx.db.labeling_configuration import LabelingProviderOption
from onyx.db.models import RegulatoryDocumentIntake
from onyx.error_handling.exceptions import register_onyx_exception_handlers
from onyx.server.features.regulatory import document_intake_api as api
from shared_configs.contextvars import get_current_tenant_id


class _User:
    def __init__(self, *, admin: bool) -> None:
        self.id = uuid4()
        self.effective_permissions = (
            [Permission.FULL_ADMIN_PANEL_ACCESS.value] if admin else []
        )


def _client(session: MagicMock, *, admin: bool = True) -> TestClient:
    app = FastAPI()
    register_onyx_exception_handlers(app)
    app.include_router(api.router, prefix="/regulatory")
    app.dependency_overrides[current_user] = lambda: _User(admin=admin)
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_current_tenant_id] = lambda: "tenant"
    return TestClient(app)


def _snapshot(batch_id: UUID) -> dict[str, object]:
    return {
        "id": str(batch_id),
        "source_package_id": str(uuid4()),
        "status": "ready",
        "stage": "review",
        "target_document_set_id": 8,
        "target_document_set_name": "PC Külliyatı",
        "review_sha256": "a" * 64,
        "issues": [],
        "documents": [],
    }


@pytest.fixture(autouse=True)
def _enable_intake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api.config, "REGULATORY_ANNEX_UPDATES_ENABLED", True)


@pytest.mark.parametrize(
    "path",
    [
        "/capabilities",
        "/labeling-setup",
        "/batches",
        f"/batches/{uuid4()}",
        f"/batches/{uuid4()}/documents/{uuid4()}/chunks",
    ],
)
def test_intake_read_routes_require_admin(path: str) -> None:
    response = _client(MagicMock(spec=Session), admin=False).get(
        f"/regulatory/document-intake{path}"
    )
    assert response.status_code == 403


@pytest.mark.parametrize("action", ["create", "retry", "approve"])
def test_intake_mutations_require_admin(action: str) -> None:
    path = "/regulatory/document-intake/batches"
    body: dict[str, str | int] = {
        "source_package_id": str(uuid4()),
        "idempotency_key": str(uuid4()),
    }
    if action != "create":
        path += f"/{uuid4()}/{action}"
        body = {"expected_review_sha256": "a" * 64} if action == "approve" else {}
    response = _client(MagicMock(spec=Session), admin=False).post(path, json=body)
    assert response.status_code == 403


def test_capabilities_report_missing_fixed_target_without_enabling_intake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_target(*_args: object) -> None:
        raise ValueError("PC Külliyatı is missing")

    monkeypatch.setattr(api.repository, "resolve_intake_target", missing_target)
    response = _client(MagicMock(spec=Session)).get(
        "/regulatory/document-intake/capabilities"
    )
    assert response.status_code == 200
    assert response.json() == {
        "enabled": False,
        "target_document_set_id": None,
        "target_document_set_name": "PC Külliyatı",
        "error": "PC Külliyatı is missing",
    }


@pytest.mark.parametrize("created", [True, False])
def test_create_dispatches_only_after_commit_and_never_compares(
    monkeypatch: pytest.MonkeyPatch, created: bool
) -> None:
    batch_id, package_id = uuid4(), uuid4()
    batch = SimpleNamespace(id=batch_id)
    session = MagicMock(spec=Session)
    events: list[str] = []
    session.commit.side_effect = lambda: events.append("commit")
    create = MagicMock(return_value=(batch, created))
    monkeypatch.setattr(api.repository, "create_intake", create)
    monkeypatch.setattr(
        api.repository, "intake_snapshot", lambda *_args: _snapshot(batch_id)
    )
    monkeypatch.setattr(api, "_enqueue_batch", lambda *_args: events.append("dispatch"))
    response = _client(session).post(
        "/regulatory/document-intake/batches",
        json={
            "source_package_id": str(package_id),
            "idempotency_key": str(uuid4()),
        },
    )
    assert response.status_code == 202
    assert create.call_args.kwargs["source_package_id"] == package_id
    assert "labeling_model_configuration_id" not in create.call_args.kwargs
    assert events == (
        ["commit", "dispatch", "commit"] if created else ["commit", "commit"]
    )


def test_stale_approval_returns_conflict_without_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_id = uuid4()
    session = MagicMock(spec=Session)
    monkeypatch.setattr(
        api.repository, "get_intake", lambda *_args: SimpleNamespace(id=batch_id)
    )
    approve = MagicMock(
        side_effect=api.repository.IntakeConflictError("Review changed")
    )
    monkeypatch.setattr(api.repository, "approve_intake", approve)
    dispatch = MagicMock()
    monkeypatch.setattr(api, "_enqueue_indexes", dispatch)
    response = _client(session).post(
        f"/regulatory/document-intake/batches/{batch_id}/approve",
        json={"expected_review_sha256": "a" * 64},
    )
    assert response.status_code == 409
    session.rollback.assert_called_once()
    dispatch.assert_not_called()


def test_approve_queues_only_durable_intake_file_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_id, file_id, attempt_id = uuid4(), uuid4(), uuid4()
    session = MagicMock(spec=Session)
    events: list[str] = []
    session.commit.side_effect = lambda: events.append("commit")
    monkeypatch.setattr(
        api.repository, "get_intake", lambda *_args: SimpleNamespace(id=batch_id)
    )
    approve = MagicMock(return_value=[(file_id, attempt_id)])
    monkeypatch.setattr(api.repository, "approve_intake", approve)
    monkeypatch.setattr(
        api.repository, "intake_snapshot", lambda *_args: _snapshot(batch_id)
    )

    def indexes(
        _session: Session,
        _batch: object,
        deliveries: list[tuple[UUID, UUID]],
        tenant: str,
    ) -> None:
        assert deliveries == [(file_id, attempt_id)]
        assert tenant == "tenant"
        events.append("index")

    monkeypatch.setattr(api, "_enqueue_indexes", indexes)
    monkeypatch.setattr(api, "_enqueue_batch", lambda *_args: events.append("observe"))
    response = _client(session).post(
        f"/regulatory/document-intake/batches/{batch_id}/approve",
        json={"expected_review_sha256": "b" * 64},
    )
    assert response.status_code == 202
    assert approve.call_args.kwargs["expected_review_sha256"] == "b" * 64
    assert events == ["commit", "index", "commit"]


@pytest.mark.parametrize("assigned_labels", [[], ["subject:customs"]])
def test_review_chunk_labels_overlay_does_not_mutate_canonical_metadata(
    monkeypatch: pytest.MonkeyPatch, assigned_labels: list[str]
) -> None:
    batch_id, document_id = uuid4(), uuid4()
    now = datetime.datetime.now(datetime.timezone.utc)
    metadata = {"source_file": "annex.pdf"}
    chunk = SimpleNamespace(
        id="chunk-1",
        user_file_id=uuid4(),
        text="MADDE 1 - Rule.",
        position=0,
        chunk_type="article",
        heading_path=["MADDE 1"],
        chunk_metadata=metadata,
        validity_start_date=None,
        validity_end_date=None,
        status="active",
        source="original",
        supersedes_chunk_id=None,
        superseded_by_chunk_id=None,
        created_at=now,
        updated_at=now,
    )
    monkeypatch.setattr(
        api.repository, "get_intake", lambda *_args: SimpleNamespace(id=batch_id)
    )
    page = MagicMock(return_value=([(chunk, assigned_labels)], 1))
    monkeypatch.setattr(api.repository, "intake_chunk_page", page)
    diagnostics = MagicMock(
        return_value={
            "chunk-1": {
                "status": "failed",
                "error": "The model abstained: no supported label was identified.",
            }
        }
    )
    monkeypatch.setattr(
        api.repository, "intake_chunk_labeling_diagnostics", diagnostics
    )
    response = _client(MagicMock(spec=Session)).get(
        f"/regulatory/document-intake/batches/{batch_id}/documents/{document_id}/chunks?offset=0&limit=10"
    )
    assert response.status_code == 200
    assert response.json()["items"][0]["chunk_metadata"]["labels"] == assigned_labels
    assert response.json()["items"][0]["labeling_status"] == "failed"
    assert (
        response.json()["items"][0]["labeling_error"]
        == "The model abstained: no supported label was identified."
    )
    assert diagnostics.call_args.args[2] == ["chunk-1"]
    assert metadata == {"source_file": "annex.pdf"}


def test_partial_review_exposes_edit_receipt_without_an_approval_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_id = uuid4()
    snapshot = {
        **_snapshot(batch_id),
        "status": "partial",
        "stage": "labeling",
        "review_sha256": None,
        "edit_review_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        api.repository, "get_intake", lambda *_args: SimpleNamespace(id=batch_id)
    )
    monkeypatch.setattr(api.repository, "intake_snapshot", lambda *_args: snapshot)
    response = _client(MagicMock(spec=Session)).get(
        f"/regulatory/document-intake/batches/{batch_id}"
    )
    assert response.status_code == 200
    assert response.json()["status"] == "partial"
    assert response.json()["review_sha256"] is None
    assert response.json()["edit_review_sha256"] == "c" * 64


def test_create_rejects_user_selected_labeling_model_and_invalid_review_hash() -> None:
    client = _client(MagicMock(spec=Session))
    response = client.post(
        "/regulatory/document-intake/batches",
        json={
            "source_package_id": str(uuid4()),
            "idempotency_key": str(uuid4()),
            "labeling_model_configuration_id": 7,
        },
    )
    assert response.status_code == 422
    approval = client.post(
        f"/regulatory/document-intake/batches/{uuid4()}/approve",
        json={"expected_review_sha256": "stale"},
    )
    assert approval.status_code == 422


def test_unknown_or_foreign_intake_returns_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get = MagicMock(side_effect=ValueError("New document intake not found"))
    monkeypatch.setattr(api.repository, "get_intake", get)
    response = _client(MagicMock(spec=Session)).get(
        f"/regulatory/document-intake/batches/{uuid4()}"
    )
    assert response.status_code == 404


def test_index_retry_only_redelivers_the_durable_intake_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_id, file_id, attempt_id = uuid4(), uuid4(), uuid4()
    batch = SimpleNamespace(id=batch_id, stage="indexing")
    session = MagicMock(spec=Session)
    monkeypatch.setattr(api.repository, "get_intake", lambda *_args: batch)
    monkeypatch.setattr(
        api.repository, "retry_intake", lambda *_args: [(file_id, attempt_id)]
    )
    monkeypatch.setattr(
        api.repository, "intake_snapshot", lambda *_args: _snapshot(batch_id)
    )
    indexes, preparation = MagicMock(), MagicMock()
    monkeypatch.setattr(api, "_enqueue_indexes", indexes)
    monkeypatch.setattr(api, "_enqueue_batch", preparation)
    response = _client(session).post(
        f"/regulatory/document-intake/batches/{batch_id}/retry"
    )
    assert response.status_code == 202
    indexes.assert_called_once_with(session, batch, [(file_id, attempt_id)], "tenant")
    preparation.assert_not_called()


@pytest.mark.parametrize("indexing", [False, True])
def test_broker_failure_records_a_retryable_visible_state(
    monkeypatch: pytest.MonkeyPatch, indexing: bool
) -> None:
    from onyx.background.celery.tasks.regulatory_amendments import document_intake

    batch = cast(RegulatoryDocumentIntake, SimpleNamespace(id=uuid4()))
    session = MagicMock(spec=Session)
    failed = MagicMock()
    dispatch = MagicMock(side_effect=RuntimeError("Broker unavailable"))
    if indexing:
        monkeypatch.setattr(document_intake, "enqueue_intake_indexes", dispatch)
        monkeypatch.setattr(api.repository, "mark_intake_index_dispatch_failed", failed)
        api._enqueue_indexes(session, batch, [(uuid4(), uuid4())], "tenant")
    else:
        monkeypatch.setattr(document_intake, "enqueue_intake_batch", dispatch)
        monkeypatch.setattr(api.repository, "mark_intake_dispatch_failed", failed)
        api._enqueue_batch(session, batch, "tenant")
    failed.assert_called_once()
    assert failed.call_args.args[:2] == (session, batch)
    session.commit.assert_called_once()


def test_intake_labeling_setup_uses_only_new_scope_and_never_bulk_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api.repository, "resolve_intake_target", lambda *_args: SimpleNamespace(id=8)
    )
    monkeypatch.setattr(
        api,
        "get_intake_labeling_provider_options",
        lambda *_args, **_kwargs: [
            {"id": 7, "name": "Gemini", "configuration_error": None}
        ],
    )
    monkeypatch.setattr(
        api.labeling_repository,
        "get_label_settings",
        lambda *_args: SimpleNamespace(taxonomy=SimpleNamespace(label_count=255)),
    )
    monkeypatch.setattr(api.labeling_repository, "list_taxonomies", lambda *_args: [])
    monkeypatch.setattr(
        api.labeling_repository,
        "get_labeling_counts",
        lambda *_args: pytest.fail("New intake read existing corpus counts"),
    )
    response = _client(MagicMock(spec=Session)).get(
        "/regulatory/document-intake/labeling-setup"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["providers"] == [
        {"id": 7, "name": "Gemini", "configuration_error": None}
    ]
    assert body["counts"] == {"files": 0, "canonical_chunks": 0, "derived_chunks": 0}
    assert body["active_run_id"] is None
    assert body["configuration_errors"] == []
    assert body["model"] == "gemini-3.5-flash"


@pytest.mark.parametrize("available", [False, True])
def test_intake_setup_exposes_only_automatic_usable_connection(
    monkeypatch: pytest.MonkeyPatch, available: bool
) -> None:
    monkeypatch.setattr(
        api.repository, "resolve_intake_target", lambda *_args: SimpleNamespace(id=8)
    )
    options: list[LabelingProviderOption] = [
        {"id": 6, "name": "Broken", "configuration_error": "Credentials missing"}
    ]
    if available:
        options.extend(
            [
                {"id": 7, "name": "Google", "configuration_error": None},
                {"id": 8, "name": "Other", "configuration_error": None},
            ]
        )
    monkeypatch.setattr(
        api, "get_intake_labeling_provider_options", lambda *_args, **_kwargs: options
    )
    monkeypatch.setattr(
        api.labeling_repository,
        "get_label_settings",
        lambda *_args: SimpleNamespace(taxonomy=SimpleNamespace(label_count=255)),
    )
    monkeypatch.setattr(api.labeling_repository, "list_taxonomies", lambda *_args: [])
    body = (
        _client(MagicMock(spec=Session))
        .get("/regulatory/document-intake/labeling-setup")
        .json()
    )
    assert body["model"] == "gemini-3.5-flash"
    assert [option["id"] for option in body["providers"]] == ([7] if available else [])
    assert body["configuration_errors"] == (
        [] if available else ["Credentials missing"]
    )


@pytest.mark.parametrize("changed", [False, True])
def test_chunk_edit_commits_before_scoped_relabel_dispatch(
    monkeypatch: pytest.MonkeyPatch, changed: bool
) -> None:
    from onyx.background.celery.tasks.regulatory_labeling import tasks
    from onyx.db import document_intake_edit

    batch_id, document_id, run_id = uuid4(), uuid4(), uuid4()
    session = MagicMock(spec=Session)
    events: list[str] = []
    session.commit.side_effect = lambda: events.append("commit")
    monkeypatch.setattr(
        api.repository, "get_intake", lambda *_args: SimpleNamespace(id=batch_id)
    )
    monkeypatch.setattr(
        api.repository, "intake_snapshot", lambda *_args: _snapshot(batch_id)
    )
    edit = MagicMock(return_value=run_id if changed else None)
    monkeypatch.setattr(document_intake_edit, "edit_intake_chunk", edit)
    dispatch = MagicMock(side_effect=lambda **_kwargs: events.append("dispatch"))
    monkeypatch.setattr(tasks, "enqueue_labeling_run", dispatch)
    response = _client(session).patch(
        f"/regulatory/document-intake/batches/{batch_id}/documents/{document_id}/chunks/source-chunk",
        json={
            "expected_review_sha256": "a" * 64,
            "text": "Corrected MADDE 1",
            "chunk_metadata": {"article_no": "1", "article_title": "Başvuru"},
            "validity_end_date": None,
        },
    )
    assert response.status_code == 200
    assert edit.call_args.kwargs["updates"] == {
        "text": "Corrected MADDE 1",
        "chunk_metadata": {"article_no": "1", "article_title": "Başvuru"},
        "validity_end_date": None,
    }
    assert edit.call_args.kwargs["document_id"] == document_id
    assert events == (
        ["commit", "dispatch", "commit"] if changed else ["commit", "commit"]
    )
    if changed:
        dispatch.assert_called_once_with(run_id=run_id, tenant_id="tenant")


@pytest.mark.parametrize("invalid", [False, True])
def test_chunk_edit_rejects_conflict_or_unexpected_fields(
    monkeypatch: pytest.MonkeyPatch, invalid: bool
) -> None:
    from onyx.background.celery.tasks.regulatory_labeling import tasks
    from onyx.db import document_intake_edit

    session = MagicMock(spec=Session)
    monkeypatch.setattr(
        api.repository, "get_intake", lambda *_args: SimpleNamespace(id=uuid4())
    )
    edit = MagicMock(side_effect=api.repository.IntakeConflictError("Review changed"))
    monkeypatch.setattr(document_intake_edit, "edit_intake_chunk", edit)
    dispatch = MagicMock()
    monkeypatch.setattr(tasks, "enqueue_labeling_run", dispatch)
    body: dict[str, object] = {"expected_review_sha256": "a" * 64, "text": "Text"}
    if invalid:
        body["approval_attempt_id"] = str(uuid4())
    response = _client(session).patch(
        f"/regulatory/document-intake/batches/{uuid4()}/documents/{uuid4()}/chunks/source-chunk",
        json=body,
    )
    assert response.status_code == (422 if invalid else 409)
    dispatch.assert_not_called()
    if invalid:
        edit.assert_not_called()
    else:
        session.rollback.assert_called_once()


@pytest.mark.parametrize("stale", [False, True])
def test_predecessor_selection_uses_review_lock_and_updates_digest(
    monkeypatch: pytest.MonkeyPatch, stale: bool
) -> None:
    from onyx.db import document_lineage

    batch_id, document_id, predecessor = uuid4(), uuid4(), uuid4()
    batch = SimpleNamespace(id=batch_id, status="ready", review_sha256="a" * 64)
    session = MagicMock(spec=Session)
    monkeypatch.setattr(api.repository, "get_intake", lambda *_args: batch)
    monkeypatch.setattr(api.repository, "refresh_intake", lambda *_args: None)
    monkeypatch.setattr(
        api.repository, "intake_snapshot", lambda *_args: _snapshot(batch_id)
    )
    digest = MagicMock(side_effect=["b" * 64] if stale else ["a" * 64, "c" * 64])
    monkeypatch.setattr(api.repository, "review_digest", digest)
    selection = MagicMock()
    monkeypatch.setattr(document_lineage, "set_predecessor_selection", selection)
    response = _client(session).put(
        f"/regulatory/document-intake/batches/{batch_id}/documents/{document_id}/predecessor-selection",
        json={
            "expected_review_sha256": "a" * 64,
            "predecessor_user_file_id": str(predecessor),
            "relation": "repeals",
            "effective_date": "2026-10-26",
            "evidence": "MADDE 9 explicitly repeals the previous law",
        },
    )
    assert response.status_code == (409 if stale else 200)
    if stale:
        selection.assert_not_called()
        session.rollback.assert_called_once()
    else:
        assert batch.review_sha256 == "c" * 64
        assert selection.call_args.kwargs["predecessor_user_file_id"] == predecessor
        assert selection.call_args.kwargs["effective_date"] == datetime.date(
            2026, 10, 26
        )


@pytest.mark.parametrize("method", ["patch", "put", "get"])
def test_new_review_routes_require_admin(method: str) -> None:
    path = f"/regulatory/document-intake/batches/{uuid4()}/documents/{uuid4()}"
    if method == "patch":
        path += "/chunks/source-chunk"
        body = {"expected_review_sha256": "a" * 64, "text": "Text"}
    elif method == "put":
        path += "/predecessor-selection"
        body = {"expected_review_sha256": "a" * 64, "predecessor_user_file_id": None}
    else:
        path += "/predecessor-candidates"
        body = None
    client = _client(MagicMock(spec=Session), admin=False)
    response = client.request(method.upper(), path, json=body)
    assert response.status_code == 403


def test_file_lineage_is_scoped_to_editable_fixed_document_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.db import document_lineage

    file_id = uuid4()
    session = MagicMock(spec=Session)
    monkeypatch.setattr(
        api.repository, "resolve_intake_target", lambda *_args: SimpleNamespace(id=8)
    )
    snapshot = MagicMock(
        return_value={
            "user_file_id": str(file_id),
            "title": "Old law",
            "status": "repealed",
            "effective_date": "2026-10-26",
            "predecessors": [],
            "successors": [],
        }
    )
    monkeypatch.setattr(document_lineage, "file_lineage_snapshot", snapshot)
    monkeypatch.setattr(document_lineage, "reconcile_file_lineage", lambda *_args: None)
    response = _client(session).get(
        f"/regulatory/document-intake/files/{file_id}/lineage?as_of=2026-10-26"
    )
    assert response.status_code == 200
    assert response.json()["status"] == "repealed"
    snapshot.assert_called_once_with(
        session, 8, file_id, as_of=datetime.date(2026, 10, 26)
    )


@pytest.mark.parametrize("changed", [False, True])
def test_bulk_chunk_edit_commits_all_edits_then_dispatches_one_run(
    monkeypatch: pytest.MonkeyPatch, changed: bool
) -> None:
    from onyx.background.celery.tasks.regulatory_labeling import tasks
    from onyx.db import document_intake_edit

    batch_id, main_id, annex_id, run_id = uuid4(), uuid4(), uuid4(), uuid4()
    session = MagicMock(spec=Session)
    events: list[str] = []
    session.commit.side_effect = lambda: events.append("commit")
    monkeypatch.setattr(
        api.repository, "get_intake", lambda *_args: SimpleNamespace(id=batch_id)
    )
    snapshot = _snapshot(batch_id)
    snapshot["labeling_failed_count"] = 5
    monkeypatch.setattr(api.repository, "intake_snapshot", lambda *_args: snapshot)
    edit = MagicMock(return_value=run_id if changed else None)
    monkeypatch.setattr(document_intake_edit, "edit_intake_chunks", edit)
    dispatch = MagicMock(side_effect=lambda **_kwargs: events.append("dispatch"))
    monkeypatch.setattr(tasks, "enqueue_labeling_run", dispatch)
    response = _client(session).patch(
        f"/regulatory/document-intake/batches/{batch_id}/chunks",
        json={
            "expected_review_sha256": "a" * 64,
            "edits": [
                {
                    "document_id": str(main_id),
                    "chunk_id": "main-date",
                    "updates": {
                        "validity_start_date": "2025-01-05",
                        "validity_end_date": None,
                    },
                },
                {
                    "document_id": str(annex_id),
                    "chunk_id": "annex-formula",
                    "updates": {
                        "text": "SEC = 0,46 x c^(0,63)",
                        "chunk_metadata": {"article_title": "Calculation"},
                    },
                },
            ],
        },
    )
    assert response.status_code == 200
    edit.assert_called_once()
    edits = edit.call_args.kwargs["edits"]
    assert len(edits) == 2
    assert edits[0].document_id == main_id
    assert edits[0].updates == {
        "validity_start_date": datetime.date(2025, 1, 5),
        "validity_end_date": None,
    }
    assert edits[1].document_id == annex_id
    assert edits[1].updates == {
        "text": "SEC = 0,46 x c^(0,63)",
        "chunk_metadata": {"article_title": "Calculation"},
    }
    assert edit.call_args.kwargs["expected_review_sha256"] == "a" * 64
    assert response.json()["labeling_failed_count"] == 5
    assert events == (
        ["commit", "dispatch", "commit"] if changed else ["commit", "commit"]
    )
    if changed:
        dispatch.assert_called_once_with(run_id=run_id, tenant_id="tenant")
    else:
        dispatch.assert_not_called()


@pytest.mark.parametrize("invalid", [False, True])
def test_bulk_chunk_edit_rolls_back_and_never_dispatches_on_failure(
    monkeypatch: pytest.MonkeyPatch, invalid: bool
) -> None:
    from onyx.background.celery.tasks.regulatory_labeling import tasks
    from onyx.db import document_intake_edit

    session = MagicMock(spec=Session)
    monkeypatch.setattr(
        api.repository, "get_intake", lambda *_args: SimpleNamespace(id=uuid4())
    )
    error = (
        ValueError("Invalid final edit")
        if invalid
        else api.repository.IntakeConflictError("Review changed")
    )
    edit = MagicMock(side_effect=error)
    monkeypatch.setattr(document_intake_edit, "edit_intake_chunks", edit)
    dispatch = MagicMock()
    monkeypatch.setattr(tasks, "enqueue_labeling_run", dispatch)
    response = _client(session).patch(
        f"/regulatory/document-intake/batches/{uuid4()}/chunks",
        json={
            "expected_review_sha256": "a" * 64,
            "edits": [
                {
                    "document_id": str(uuid4()),
                    "chunk_id": identifier,
                    "updates": {"text": "Text"},
                }
                for identifier in ("first", "last")
            ],
        },
    )
    assert response.status_code == (400 if invalid else 409)
    session.rollback.assert_called_once()
    session.commit.assert_not_called()
    dispatch.assert_not_called()


@pytest.mark.parametrize(
    "invalid",
    ["empty", "too_many", "duplicate", "empty_updates", "protected", "unexpected"],
)
def test_bulk_chunk_edit_request_is_bounded_and_rejects_unknown_fields(
    monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    from onyx.db import document_intake_edit

    edit = MagicMock()
    monkeypatch.setattr(document_intake_edit, "edit_intake_chunks", edit)
    entry: dict[str, object] = {
        "document_id": str(uuid4()),
        "chunk_id": "chunk",
        "updates": {"text": "Text"},
    }
    entries = [entry]
    body: dict[str, object] = {"expected_review_sha256": "a" * 64, "edits": entries}
    if invalid == "empty":
        entries.clear()
    elif invalid == "too_many":
        entries[:] = [{**entry, "chunk_id": f"chunk-{index}"} for index in range(257)]
    elif invalid == "duplicate":
        entries.append({**entry, "document_id": str(uuid4())})
    elif invalid == "empty_updates":
        entry["updates"] = {}
    elif invalid == "protected":
        entry["updates"] = {"approval_attempt_id": str(uuid4())}
    else:
        body["labeling_run_id"] = str(uuid4())
    response = _client(MagicMock(spec=Session)).patch(
        f"/regulatory/document-intake/batches/{uuid4()}/chunks", json=body
    )
    assert response.status_code == 422
    edit.assert_not_called()


def test_bulk_chunk_edit_requires_admin() -> None:
    response = _client(MagicMock(spec=Session), admin=False).patch(
        f"/regulatory/document-intake/batches/{uuid4()}/chunks",
        json={
            "expected_review_sha256": "a" * 64,
            "edits": [
                {
                    "document_id": str(uuid4()),
                    "chunk_id": "chunk",
                    "updates": {"text": "Text"},
                }
            ],
        },
    )
    assert response.status_code == 403


@pytest.mark.parametrize("created", [True, False])
def test_mixed_zip_upload_uses_fixed_intake_request_and_commits_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    created: bool,
) -> None:
    from onyx.background.celery.tasks.regulatory_amendments import sources
    from onyx.server.features.regulatory.models import AmendmentSourcePackageSnapshot

    package_id, request_id = uuid4(), uuid4()
    package = SimpleNamespace(id=package_id)
    events: list[str] = []
    session = MagicMock(spec=Session)
    session.commit.side_effect = lambda: events.append("commit")
    create = MagicMock(return_value=(package, created))
    monkeypatch.setattr(api, "create_zip_source_package", create)
    snapshot = AmendmentSourcePackageSnapshot(
        id=package_id,
        document_set_id=8,
        status="processing",
        asset_count=0,
        total_bytes=0,
        issues=[],
        manifest_sha256=None,
        created_at=datetime.datetime.now(datetime.timezone.utc),
        updated_at=datetime.datetime.now(datetime.timezone.utc),
    )
    monkeypatch.setattr(api, "markdown_source_snapshot", lambda *_args: snapshot)
    dispatch = MagicMock(side_effect=lambda **_kwargs: events.append("dispatch"))
    monkeypatch.setattr(sources, "enqueue_source_package", dispatch)
    response = _client(session).post(
        "/regulatory/document-intake/source-packages/zip",
        data={"idempotency_key": str(request_id)},
        files={"file": ("mixed.zip", b"PK-source", "application/zip")},
    )
    assert response.status_code == 202
    assert response.json()["document_set_id"] == 8
    assert create.call_args.kwargs["content"] == b"PK-source"
    assert create.call_args.kwargs["idempotency_key"] == request_id
    assert "document_set_id" not in create.call_args.kwargs
    assert events == (["commit", "dispatch"] if created else ["commit"])
    if created:
        dispatch.assert_called_once_with(package_id=package_id, tenant_id="tenant")


@pytest.mark.parametrize(
    "name,content", [("source.md", b"# Main"), ("source.zip", b"")]
)
def test_mixed_zip_upload_rejects_wrong_extension_and_empty_input(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    content: bytes,
) -> None:
    create = MagicMock()
    monkeypatch.setattr(api, "create_zip_source_package", create)
    response = _client(MagicMock(spec=Session)).post(
        "/regulatory/document-intake/source-packages/zip",
        data={"idempotency_key": str(uuid4())},
        files={"file": (name, content)},
    )
    assert response.status_code == 400
    create.assert_not_called()


def test_mixed_zip_upload_requires_admin() -> None:
    response = _client(MagicMock(spec=Session), admin=False).post(
        "/regulatory/document-intake/source-packages/zip",
        data={"idempotency_key": str(uuid4())},
        files={"file": ("source.zip", b"PK-source")},
    )
    assert response.status_code == 403
