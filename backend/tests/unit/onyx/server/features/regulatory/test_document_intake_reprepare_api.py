"""Reprepare admission requires admin, a fresh receipt and a server-owned scope."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db import document_intake_reprepare as repository
from onyx.db.models import RegulatoryDocumentIntake
from onyx.server.features.regulatory import document_intake_api as api
from tests.unit.onyx.server.features.regulatory.test_document_intake_api import (
    _client,
    _snapshot,
)


@pytest.mark.parametrize("method", ["get", "post"])
def test_reprepare_requires_admin(method: str) -> None:
    client = _client(MagicMock(spec=Session), admin=False)
    path = f"/regulatory/document-intake/batches/{uuid4()}/reprepare"
    response = (
        client.get(path)
        if method == "get"
        else client.post(
            path, json={"expected_sha256": "a" * 64, "idempotency_key": str(uuid4())}
        )
    )
    assert response.status_code == 403


@pytest.mark.parametrize("stale", [False, True])
def test_reprepare_commits_before_dispatch_and_conflicts_never_dispatch(
    monkeypatch: pytest.MonkeyPatch, stale: bool
) -> None:
    monkeypatch.setattr(api.config, "REGULATORY_ANNEX_UPDATES_ENABLED", True)
    session = MagicMock(spec=Session)
    batch_id = uuid4()
    batch = SimpleNamespace(id=batch_id, stage="reprepare")
    monkeypatch.setattr(api.repository, "get_intake", lambda *_args: batch)
    monkeypatch.setattr(
        api.repository, "intake_snapshot", lambda *_args: _snapshot(batch_id)
    )
    request = MagicMock(return_value=SimpleNamespace(status="queued"))
    if stale:
        request.side_effect = api.repository.IntakeConflictError(
            "The document snapshot changed"
        )
    monkeypatch.setattr(repository, "request_reprepare", request)
    events = []
    session.commit.side_effect = lambda: events.append("commit")
    monkeypatch.setattr(api, "_enqueue_batch", lambda *_args: events.append("dispatch"))
    response = _client(session).post(
        f"/regulatory/document-intake/batches/{batch_id}/reprepare",
        json={"expected_sha256": "a" * 64, "idempotency_key": str(uuid4())},
    )
    assert response.status_code == (409 if stale else 202)
    assert events == ([] if stale else ["commit", "dispatch", "commit"])
    assert "user_file_ids" not in request.call_args.kwargs


def test_reprepare_rejects_caller_file_allowlist() -> None:
    response = _client(MagicMock(spec=Session)).post(
        f"/regulatory/document-intake/batches/{uuid4()}/reprepare",
        json={
            "expected_sha256": "a" * 64,
            "idempotency_key": str(uuid4()),
            "user_file_ids": [str(uuid4())],
        },
    )
    assert response.status_code == 422


def test_reprepare_dispatch_carries_only_the_durable_request_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.background.celery.tasks.regulatory_amendments import (
        document_intake as delivery,
    )

    request_id, batch_id = uuid4(), uuid4()
    monkeypatch.setattr(
        repository, "latest_reprepare", lambda *_args: SimpleNamespace(id=request_id)
    )
    enqueue = MagicMock()
    monkeypatch.setattr(delivery, "enqueue_intake_batch", enqueue)
    api._enqueue_batch(
        MagicMock(spec=Session),
        cast(RegulatoryDocumentIntake, SimpleNamespace(id=batch_id, stage="reprepare")),
        "tenant",
    )
    enqueue.assert_called_once_with(
        batch_id=batch_id, tenant_id="tenant", reprepare_request_id=request_id
    )
