import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.background.celery.tasks.regulatory_amendments import sources
from onyx.regulatory.document_intake.markdown_package import (
    parse_markdown_package,
)
from onyx.server.features.regulatory import document_intake_api as api
from onyx.server.features.regulatory.models import AmendmentSourcePackageSnapshot
from tests.unit.onyx.server.features.regulatory.test_document_intake_api import (
    _client,
)


@pytest.fixture(autouse=True)
def _enable_intake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api.config, "REGULATORY_ANNEX_UPDATES_ENABLED", True)


def _snapshot(package: SimpleNamespace) -> AmendmentSourcePackageSnapshot:
    now = datetime.datetime.now(datetime.timezone.utc)
    return AmendmentSourcePackageSnapshot(
        id=package.id,
        document_set_id=7,
        status=package.status,
        asset_count=0,
        total_bytes=0,
        issues=[],
        manifest_sha256=None,
        created_at=now,
        updated_at=now,
    )


def test_markdown_source_upload_and_template_require_admin() -> None:
    client = _client(MagicMock(spec=Session), admin=False)
    assert (
        client.get("/regulatory/document-intake/markdown-template").status_code == 403
    )
    response = client.post(
        "/regulatory/document-intake/source-packages/markdown",
        data={"idempotency_key": str(uuid4())},
        files={"file": ("ana.md", b"# Ana belge", "text/markdown")},
    )
    assert response.status_code == 403


@pytest.mark.parametrize("created", [False, True])
def test_markdown_upload_commits_before_queuing_and_reuses_source_identity(
    monkeypatch: pytest.MonkeyPatch, created: bool
) -> None:
    session = MagicMock(spec=Session)
    events: list[str] = []
    session.commit.side_effect = lambda: events.append("commit")
    package = SimpleNamespace(id=uuid4(), status="processing")
    create = MagicMock(return_value=(package, created))
    monkeypatch.setattr(api, "create_markdown_source_package", create)
    monkeypatch.setattr(
        api, "markdown_source_snapshot", lambda *_args: _snapshot(package)
    )
    monkeypatch.setattr(
        sources, "enqueue_source_package", lambda **_kwargs: events.append("dispatch")
    )
    response = _client(session).post(
        "/regulatory/document-intake/source-packages/markdown",
        data={"idempotency_key": str(uuid4())},
        files={"file": ("ana.md", b"# Ana belge", "text/markdown")},
    )
    assert response.status_code == 202
    assert response.json()["id"] == str(package.id)
    assert create.call_args.kwargs["content"] == b"# Ana belge"
    assert "document_set_id" not in create.call_args.kwargs
    assert events == (["commit", "dispatch"] if created else ["commit"])


@pytest.mark.parametrize(("name", "content"), [("ana.pdf", b"pdf"), ("ana.md", b"")])
def test_invalid_markdown_upload_does_not_create_a_source_request(
    monkeypatch: pytest.MonkeyPatch, name: str, content: bytes
) -> None:
    create = MagicMock()
    monkeypatch.setattr(api, "create_markdown_source_package", create)
    response = _client(MagicMock(spec=Session)).post(
        "/regulatory/document-intake/source-packages/markdown",
        data={"idempotency_key": str(uuid4())},
        files={"file": (name, content, "application/octet-stream")},
    )
    assert response.status_code == 400
    create.assert_not_called()


def test_downloaded_template_follows_the_actual_parser_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api.repository, "resolve_intake_target", lambda *_args: object()
    )
    response = _client(MagicMock(spec=Session)).get(
        "/regulatory/document-intake/markdown-template"
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    parsed = parse_markdown_package(response.content, "example.zip")
    assert parsed.status == "ready" and len(parsed.assets) == 3
    assert len({asset.sha256 for asset in parsed.assets}) == 3
    assert all(asset.mime_type == "text/markdown" for asset in parsed.assets)
    assert all(link.target_asset_hash for link in parsed.links)
