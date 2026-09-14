import json
from collections.abc import Generator
from contextlib import contextmanager
from hashlib import sha256
from io import BytesIO
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from fastapi import UploadFile
from pypdf import PdfWriter
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.datastructures import Headers

from onyx.auth.schemas import UserRole
from onyx.db.engine.sql_engine import get_sqlalchemy_engine
from onyx.db.models import AmendmentSourcePackage, DocumentSet, User
from onyx.file_store.file_store import get_default_file_store
from onyx.regulatory.amendments.annexes.source_limits import (
    SOURCE_PACKAGE_LEASE_MARGIN_SECONDS,
    source_preparation_seconds,
)
from tests.external_dependency_unit.conftest import create_test_user


@pytest.fixture
def source_session(
    db_session: Session,  # noqa: ARG001
    tenant_context: None,  # noqa: ARG001
) -> Generator[Session, None, None]:
    engine = get_sqlalchemy_engine()
    with engine.connect() as connection:
        transaction = connection.begin()
        with Session(connection, join_transaction_mode="create_savepoint") as session:
            yield session
        transaction.rollback()


def _editable_source_set(session: Session) -> tuple[User, DocumentSet]:
    user = create_test_user(session, "annex_source_modes", UserRole.ADMIN)
    document_set = DocumentSet(
        name=f"annex-source-modes-{uuid4()}",
        description="source mode test",
        user_id=user.id,
        is_up_to_date=True,
        is_public=False,
    )
    session.add(document_set)
    session.commit()
    return user, document_set


def _configure_api(monkeypatch: pytest.MonkeyPatch) -> list[tuple[UUID, str]]:
    from onyx.server.features.regulatory import api

    delivered: list[tuple[UUID, str]] = []
    monkeypatch.setattr(api.annex_config, "REGULATORY_ANNEX_UPDATES_ENABLED", True)
    monkeypatch.setattr(api.annex_config, "REGULATORY_ANNEX_ENVIRONMENT", "local-test")
    monkeypatch.setattr(
        api,
        "enqueue_source_package",
        lambda package_id, tenant_id: delivered.append((package_id, tenant_id)),
    )
    return delivered


def _run_with_source_session(
    monkeypatch: pytest.MonkeyPatch,
    session: Session,
    package_id: UUID,
) -> list[int]:
    from onyx.regulatory.amendments.annexes import job

    @contextmanager
    def session_context() -> Generator[Session, None, None]:
        yield session

    monkeypatch.setattr(job, "get_session_with_current_tenant", session_context)
    original_extend = job.extend_source_package_lease
    recorded_lease_seconds: list[int] = []

    def record_extension(
        db_session: Session,
        *,
        package_id: UUID,
        environment: str,
        lease_token: UUID,
        lease_seconds: int,
    ) -> bool:
        recorded_lease_seconds.append(lease_seconds)
        return original_extend(
            db_session,
            package_id=package_id,
            environment=environment,
            lease_token=lease_token,
            lease_seconds=lease_seconds,
        )

    monkeypatch.setattr(job, "extend_source_package_lease", record_extension)
    job.run_source_package(package_id=package_id, environment="local-test")
    return recorded_lease_seconds


def _assert_lease_budget(actual: int, *, pdf_pages: int) -> None:
    expected = (
        source_preparation_seconds(pdf_pages) + SOURCE_PACKAGE_LEASE_MARGIN_SECONDS
    )
    assert expected - 1 <= actual <= expected


def test_pasted_text_extends_lease_and_persists_verified_outputs(
    source_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from onyx.db.amendment_sources import (
        list_source_assets,
        require_ready_source_package,
    )
    from onyx.server.features.regulatory import api
    from onyx.server.features.regulatory.models import (
        CreateAmendmentSourcePackageRequest,
    )

    delivered = _configure_api(monkeypatch)
    user, document_set = _editable_source_set(source_session)
    source_text = "MADDE 1 — İthalat beyannamesi elektronik ortamda sunulur."
    created = api.create_amendment_source_package(
        CreateAmendmentSourcePackageRequest(
            document_set_id=document_set.id,
            idempotency_key=str(uuid4()),
            text=source_text,
        ),
        user,
        source_session,
        "public",
    )
    assert delivered == [(created.id, "public")]

    lease_seconds = _run_with_source_session(monkeypatch, source_session, created.id)
    assert len(lease_seconds) == 1
    _assert_lease_budget(lease_seconds[0], pdf_pages=0)
    source_session.expire_all()
    package = require_ready_source_package(
        source_session,
        package_id=created.id,
        document_set_id=document_set.id,
        environment="local-test",
    )
    assets = list_source_assets(source_session, package.id)
    assert len(assets) == 1
    assert assets[0].mime_type == "text/plain"
    assert assets[0].text_sha256 == sha256(source_text.encode()).hexdigest()
    assert package.manifest_file_id is not None
    with get_default_file_store().read_file(package.manifest_file_id) as stream:
        manifest = json.load(stream)
    assert manifest["status"] == "ready"
    assert manifest["assets"][0]["text"] == source_text


def test_uploaded_pdf_sizes_lease_and_persists_verified_outputs(
    source_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from onyx.db.amendment_sources import (
        list_source_assets,
        require_ready_source_package,
    )
    from onyx.regulatory.amendments import pdf_vision
    from onyx.server.features.regulatory import api

    delivered = _configure_api(monkeypatch)
    user, document_set = _editable_source_set(source_session)
    writer = PdfWriter()
    for _ in range(12):
        writer.add_blank_page(width=300, height=300)
    content = BytesIO()
    writer.write(content)
    pdf_bytes = content.getvalue()
    uploaded = UploadFile(
        file=BytesIO(pdf_bytes),
        filename="amendment.pdf",
        headers=Headers({"content-type": "application/pdf"}),
    )
    created = api.upload_amendment_source_package(
        uploaded,
        document_set.id,
        str(uuid4()),
        None,
        user,
        source_session,
        "public",
    )
    assert delivered == [(created.id, "public")]

    prepared = MagicMock(
        side_effect=lambda asset, **_kwargs: asset.model_copy(
            update={"text": "Frozen twelve-page PDF transcript"}
        )
    )
    monkeypatch.setattr(pdf_vision, "prepare_pdf_source", prepared)
    lease_seconds = _run_with_source_session(monkeypatch, source_session, created.id)
    assert len(lease_seconds) == 1
    _assert_lease_budget(lease_seconds[0], pdf_pages=12)
    prepared.assert_called_once()
    source_session.expire_all()
    package = require_ready_source_package(
        source_session,
        package_id=created.id,
        document_set_id=document_set.id,
        environment="local-test",
    )
    assets = list_source_assets(source_session, package.id)
    assert len(assets) == 1
    assert assets[0].mime_type == "application/pdf"
    assert assets[0].sha256 == sha256(pdf_bytes).hexdigest()
    assert (
        assets[0].text_sha256
        == sha256(b"Frozen twelve-page PDF transcript").hexdigest()
    )
    assert package.manifest_file_id is not None


def test_worker_fails_safely_when_completion_loses_its_lease(
    source_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from onyx.regulatory.amendments.annexes import job
    from onyx.server.features.regulatory import api
    from onyx.server.features.regulatory.models import (
        CreateAmendmentSourcePackageRequest,
    )

    _configure_api(monkeypatch)
    user, document_set = _editable_source_set(source_session)
    created = api.create_amendment_source_package(
        CreateAmendmentSourcePackageRequest(
            document_set_id=document_set.id,
            idempotency_key=str(uuid4()),
            text="A bounded pasted-text source.",
        ),
        user,
        source_session,
        "public",
    )
    monkeypatch.setattr(job, "finish_source_package", lambda *_args, **_kwargs: False)
    with pytest.raises(RuntimeError, match="source_preparation_lease_lost"):
        _run_with_source_session(monkeypatch, source_session, created.id)
    source_session.expire_all()
    package = source_session.scalar(
        select(AmendmentSourcePackage).where(AmendmentSourcePackage.id == created.id)
    )
    assert package is not None
    assert package.status == "failed"
    assert package.issues[0]["code"] == "acquisition_failed"
    assert "source_preparation_lease_lost" not in json.dumps(package.issues)


def test_legacy_image_without_frozen_text_requires_new_nonretryable_preparation(
    source_session: Session,
) -> None:
    from onyx.db.amendment_sources import (
        create_source_package,
        mark_source_package_failed,
    )

    _user, document_set = _editable_source_set(source_session)
    package, _ = create_source_package(
        source_session,
        document_set_id=document_set.id,
        environment="local-test",
        idempotency_key=str(uuid4()),
        request_hash="a" * 64,
        input_spec={"mime_type": "image/png", "display_name": "amendment.png"},
        created_by=None,
    )
    mark_source_package_failed(
        source_session,
        package_id=package.id,
        environment="local-test",
        failure=ValueError("image_source_requires_new_preparation"),
    )
    source_session.refresh(package)
    assert package.status == "failed"
    assert package.issues[0]["code"] == "image_source_requires_new_preparation"
    assert package.issues[0]["retryable"] is False
