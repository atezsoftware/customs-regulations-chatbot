"""Publication review uses the same authorized, commit-before-dispatch boundary."""

from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db import document_intake_review as review
from onyx.server.features.regulatory import document_intake_api as api
from tests.unit.onyx.server.features.regulatory.test_document_intake_api import (
    _client,
    _snapshot,
)


@pytest.fixture(autouse=True)
def _enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api.config, "REGULATORY_ANNEX_UPDATES_ENABLED", True)


@pytest.mark.parametrize("candidate", [False, True])
def test_publication_review_routes_require_admin(candidate: bool) -> None:
    path = f"/regulatory/document-intake/batches/{uuid4()}/documents/{uuid4()}"
    client = _client(MagicMock(spec=Session), admin=False)
    response = (
        client.get(path + "/existing-parent-candidates")
        if candidate
        else client.put(
            path + "/publication-review",
            json={
                "expected_review_sha256": "a" * 64,
                "disposition": "legislation",
                "reason": "Reviewed the original annex",
            },
        )
    )
    assert response.status_code == 403


@pytest.mark.parametrize("replacement", [False, True])
def test_scope_and_parent_review_commit_before_one_label_dispatch(
    monkeypatch: pytest.MonkeyPatch, replacement: bool
) -> None:
    session = MagicMock(spec=Session)
    batch_id, document_id, duplicate_id, run_id = (uuid4() for _ in range(4))
    batch = SimpleNamespace(id=batch_id)
    events: list[object] = []
    session.commit.side_effect = lambda: events.append("commit")
    save = MagicMock(return_value=run_id if replacement else None)
    monkeypatch.setattr(review, "set_publication_review", save)
    monkeypatch.setattr(api.repository, "get_intake", lambda *_args: batch)
    monkeypatch.setattr(
        api.repository, "intake_snapshot", lambda *_args: _snapshot(batch_id)
    )
    monkeypatch.setattr(
        api,
        "_enqueue_edited_labels",
        lambda _s, _b, identifier, tenant: events.append((identifier, tenant)),
    )
    response = _client(session).put(
        f"/regulatory/document-intake/batches/{batch_id}/documents/{document_id}/publication-review",
        json={
            "expected_review_sha256": "a" * 64,
            "disposition": "source_evidence",
            "duplicate_of_document_id": str(duplicate_id),
            "existing_parent_file_id": None,
            "reason": "HTML is the retained official rendition of the selected PDF",
        },
    )
    assert response.status_code == 200
    assert events == ["commit", (run_id if replacement else None, "tenant"), "commit"]
    assert save.call_args.kwargs["document_id"] == document_id
    assert save.call_args.kwargs["duplicate_of_document_id"] == duplicate_id
    assert save.call_args.kwargs["expected_review_sha256"] == "a" * 64


@pytest.mark.parametrize("conflict", [False, True])
def test_invalid_or_stale_review_rolls_back_without_dispatch(
    monkeypatch: pytest.MonkeyPatch, conflict: bool
) -> None:
    session = MagicMock(spec=Session)
    batch_id, document_id = uuid4(), uuid4()
    monkeypatch.setattr(
        api.repository, "get_intake", lambda *_args: SimpleNamespace(id=batch_id)
    )
    error = (
        api.repository.IntakeConflictError("Review changed")
        if conflict
        else ValueError("Select an owned target")
    )
    monkeypatch.setattr(review, "set_publication_review", MagicMock(side_effect=error))
    dispatch = MagicMock()
    monkeypatch.setattr(api, "_enqueue_edited_labels", dispatch)
    response = _client(session).put(
        f"/regulatory/document-intake/batches/{batch_id}/documents/{document_id}/publication-review",
        json={
            "expected_review_sha256": "a" * 64,
            "disposition": "legislation",
            "reason": "Reviewed",
        },
    )
    assert response.status_code == (409 if conflict else 400)
    session.rollback.assert_called_once()
    session.commit.assert_not_called()
    dispatch.assert_not_called()


def test_parent_candidates_preserve_scope_and_perform_no_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = MagicMock(spec=Session)
    batch_id, document_id, parent_id = uuid4(), uuid4(), uuid4()
    batch = SimpleNamespace(id=batch_id)
    monkeypatch.setattr(api.repository, "get_intake", lambda *_args: batch)
    candidates = MagicMock(
        return_value=[
            {
                "user_file_id": str(parent_id),
                "display_name": "Own legal title",
                "source_url": None,
                "chunk_count": 318,
            }
        ]
    )
    monkeypatch.setattr(review, "existing_parent_candidates", candidates)
    dispatch = MagicMock()
    monkeypatch.setattr(api, "_enqueue_edited_labels", dispatch)
    response = _client(session).get(
        f"/regulatory/document-intake/batches/{batch_id}/documents/{document_id}/existing-parent-candidates?search=2022&limit=5"
    )
    assert response.status_code == 200
    assert response.json()[0]["user_file_id"] == str(parent_id)
    candidates.assert_called_once_with(session, batch, document_id, "2022", 5)
    session.commit.assert_not_called()
    dispatch.assert_not_called()
