from collections.abc import Generator
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.engine.sql_engine import get_sqlalchemy_engine
from onyx.db.models import AmendmentSourcePackage, DocumentSet


@pytest.fixture
def source_session(
    db_session: Session,  # noqa: ARG001
    tenant_context: None,  # noqa: ARG001
    request: pytest.FixtureRequest,
) -> Generator[Session, None, None]:
    if "live_review" in request.fixturenames:
        from tests.external_dependency_unit.regulatory.publication_fixtures import (
            committed_review_session,
        )

        with committed_review_session() as session:
            yield session
        return
    engine = get_sqlalchemy_engine()
    with engine.connect() as connection:
        transaction = connection.begin()
        with Session(connection, join_transaction_mode="create_savepoint") as session:
            yield session
        transaction.rollback()


def test_idempotent_request_and_changed_input_conflict(source_session: Session) -> None:
    from onyx.db.amendment_sources import create_source_package

    document_set_id = source_session.scalar(select(DocumentSet.id).limit(1))
    if document_set_id is None:
        document_set = DocumentSet(
            name=f"annex-{uuid4()}", description="annex test", is_up_to_date=True
        )
        source_session.add(document_set)
        source_session.flush()
        document_set_id = document_set.id
    key = str(uuid4())

    def create(request_hash: str) -> tuple[AmendmentSourcePackage, bool]:
        return create_source_package(
            source_session,
            document_set_id=document_set_id,
            environment="local-test",
            idempotency_key=key,
            request_hash=request_hash,
            input_spec={"url": "https://example.gov/update"},
            created_by=None,
        )

    first, created = create("a" * 64)
    repeated, created_again = create("a" * 64)
    assert first.id == repeated.id
    assert created and not created_again
    with pytest.raises(ValueError, match="idempotency"):
        create("b" * 64)


def test_only_complete_same_set_and_environment_source_can_be_analyzed(
    source_session: Session,
) -> None:
    from onyx.db.amendment_sources import (
        create_source_package,
        require_ready_source_package,
    )

    document_set = DocumentSet(
        name=f"annex-{uuid4()}", description="annex test", is_up_to_date=True
    )
    source_session.add(document_set)
    source_session.flush()
    package, _ = create_source_package(
        source_session,
        document_set_id=document_set.id,
        environment="local-test",
        idempotency_key=str(uuid4()),
        request_hash="a" * 64,
        input_spec={"text": "Annex"},
        created_by=None,
    )
    for document_set_id, environment in [
        (document_set.id, "local-test"),
        (document_set.id + 1, "local-test"),
        (document_set.id, "different"),
    ]:
        with pytest.raises(ValueError):
            require_ready_source_package(
                source_session,
                package_id=package.id,
                document_set_id=document_set_id,
                environment=environment,
            )


def test_acquisition_job_persists_ready_manifest_and_immutable_versioned_bytes(
    source_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import contextmanager
    from hashlib import sha256

    from sqlalchemy import update
    from sqlalchemy.exc import DBAPIError

    from onyx.db.amendment_sources import (
        create_source_package,
        list_source_assets,
        require_ready_source_package,
    )
    from onyx.db.models import RegulatorySourceAsset
    from onyx.regulatory.amendments.annexes import job
    from onyx.regulatory.amendments.annexes.sources import DownloadedSource

    @contextmanager
    def session_context() -> Generator[Session, None, None]:
        yield source_session

    monkeypatch.setattr(job, "get_session_with_current_tenant", session_context)
    document_set = DocumentSet(
        name=f"annex-{uuid4()}", description="annex test", is_up_to_date=True
    )
    source_session.add(document_set)
    source_session.flush()
    bodies = [b"<main>First amendment</main>", b"<main>Changed amendment</main>"]
    packages = []
    for body in bodies:
        monkeypatch.setattr(
            job,
            "download_source",
            lambda url, body=body, **_kwargs: DownloadedSource(body, "text/html", url),
        )
        package, _ = create_source_package(
            source_session,
            document_set_id=document_set.id,
            environment="local-test",
            idempotency_key=str(uuid4()),
            request_hash=sha256(body).hexdigest(),
            input_spec={"url": "https://example.gov/update"},
            created_by=None,
        )
        job.run_source_package(package_id=package.id, environment="local-test")
        source_session.expire_all()
        verified = require_ready_source_package(
            source_session,
            package_id=package.id,
            document_set_id=document_set.id,
            environment="local-test",
        )
        assert verified.asset_count == 1
        assert verified.manifest_sha256
        packages.append(package.id)
    first = list_source_assets(source_session, packages[0])[0]
    second = list_source_assets(source_session, packages[1])[0]
    assert first.sha256 != second.sha256
    assert first.file_id != second.file_id
    with source_session.begin_nested():
        with pytest.raises(DBAPIError, match="immutable"):
            source_session.execute(
                update(RegulatorySourceAsset)
                .where(RegulatorySourceAsset.id == first.id)
                .values(sha256="b" * 64)
            )
        source_session.rollback()


def test_scoped_api_idempotency_and_analysis_reject_incomplete_package(
    source_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from onyx.auth.schemas import UserRole
    from onyx.error_handling.exceptions import OnyxError
    from onyx.server.features.regulatory import api
    from onyx.server.features.regulatory.models import (
        AnalyzeAmendmentRequest,
        CreateAmendmentSourcePackageRequest,
    )
    from tests.external_dependency_unit.conftest import create_test_user

    monkeypatch.setattr(api.annex_config, "REGULATORY_ANNEX_UPDATES_ENABLED", True)
    user = create_test_user(source_session, "annex_source", UserRole.ADMIN)
    document_set = DocumentSet(
        name=f"annex-{uuid4()}", description="annex test", is_up_to_date=True
    )
    other_set = DocumentSet(
        name=f"annex-{uuid4()}", description="annex test", is_up_to_date=True
    )
    source_session.add_all([document_set, other_set])
    source_session.flush()
    delivered = []
    monkeypatch.setattr(
        api,
        "enqueue_source_package",
        lambda package_id, tenant_id: delivered.append((package_id, tenant_id)),
    )
    request = CreateAmendmentSourcePackageRequest(
        document_set_id=document_set.id,
        idempotency_key=str(uuid4()),
        url="https://example.gov/update",
    )
    first = api.create_amendment_source_package(request, user, source_session, "public")
    repeated = api.create_amendment_source_package(
        request, user, source_session, "public"
    )
    assert first.id == repeated.id
    assert first.status == "processing"
    assert delivered == [(first.id, "public")]
    with pytest.raises(OnyxError, match="complete"):
        api.analyze_amendment_text(
            AnalyzeAmendmentRequest(
                document_set_id=document_set.id,
                source_package_id=first.id,
                raw_text="amendment",
            ),
            user,
            source_session,
            "public",
        )
    with pytest.raises(OnyxError, match="not found"):
        api.get_amendment_source_package(first.id, other_set.id, user, source_session)


def test_retry_refreshes_locked_state_before_clearing_worker_lease(
    source_session: Session,
) -> None:
    import datetime

    from sqlalchemy import update

    from onyx.db.amendment_sources import create_source_package, retry_source_package
    from onyx.db.models import AmendmentSourcePackage

    document_set = DocumentSet(
        name=f"annex-{uuid4()}", description="annex test", is_up_to_date=True
    )
    source_session.add(document_set)
    source_session.flush()
    package, _ = create_source_package(
        source_session,
        document_set_id=document_set.id,
        environment="local-test",
        idempotency_key=str(uuid4()),
        request_hash="a" * 64,
        input_spec={"url": "https://example.gov/update"},
        created_by=None,
    )
    source_session.execute(
        update(AmendmentSourcePackage)
        .where(AmendmentSourcePackage.id == package.id)
        .values(
            lease_token=uuid4(),
            lease_expires_at=datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(minutes=5),
        )
        .execution_options(synchronize_session=False)
    )
    with pytest.raises(ValueError, match="still running"):
        retry_source_package(
            source_session,
            package_id=package.id,
            document_set_id=document_set.id,
            environment="local-test",
        )


@pytest.mark.parametrize("page_count", [12, 50])
def test_source_lease_protects_long_pdf_preparation_from_duplicate_claims(
    source_session: Session, monkeypatch: pytest.MonkeyPatch, page_count: int
) -> None:
    import datetime
    from types import SimpleNamespace

    from onyx.db import amendment_sources
    from onyx.regulatory.amendments.annexes.source_limits import (
        source_preparation_seconds,
    )

    document_set = DocumentSet(
        name=f"annex-{uuid4()}", description="annex test", is_up_to_date=True
    )
    source_session.add(document_set)
    source_session.flush()
    package, _ = amendment_sources.create_source_package(
        source_session,
        document_set_id=document_set.id,
        environment="local-test",
        idempotency_key=str(uuid4()),
        request_hash="a" * 64,
        input_spec={"url": "https://example.gov/update.htm"},
        created_by=None,
    )
    started = datetime.datetime.now(datetime.timezone.utc)
    claimed = amendment_sources.claim_source_package(
        source_session, package_id=package.id, environment="local-test"
    )
    assert claimed is not None
    assert amendment_sources.extend_source_package_lease(
        source_session,
        package_id=package.id,
        environment="local-test",
        lease_token=claimed[1],
        lease_seconds=source_preparation_seconds(page_count) + 300,
    )
    fake_datetime = SimpleNamespace(
        datetime=SimpleNamespace(
            now=lambda _tz: started + datetime.timedelta(minutes=20)
        ),
        timezone=datetime.timezone,
        timedelta=datetime.timedelta,
    )
    monkeypatch.setattr(amendment_sources, "datetime", fake_datetime)
    assert (
        amendment_sources.claim_source_package(
            source_session, package_id=package.id, environment="local-test"
        )
        is None
    )
    with pytest.raises(ValueError, match="still running"):
        amendment_sources.retry_source_package(
            source_session,
            package_id=package.id,
            document_set_id=document_set.id,
            environment="local-test",
        )
    fake_datetime.datetime.now = lambda _tz: started + datetime.timedelta(minutes=30)
    subsequent = amendment_sources.claim_source_package(
        source_session, package_id=package.id, environment="local-test"
    )
    assert (subsequent is None) is (page_count == 50)


@pytest.mark.parametrize("ownership", ["wrong_token", "expired", "finished"])
def test_source_lease_extension_rejects_lost_ownership(
    source_session: Session, ownership: str
) -> None:
    import datetime

    from onyx.db import amendment_sources

    document_set = DocumentSet(
        name=f"annex-{uuid4()}", description="annex test", is_up_to_date=True
    )
    source_session.add(document_set)
    source_session.flush()
    package, _ = amendment_sources.create_source_package(
        source_session,
        document_set_id=document_set.id,
        environment="local-test",
        idempotency_key=str(uuid4()),
        request_hash="a" * 64,
        input_spec={"url": "https://example.gov/update.htm"},
        created_by=None,
    )
    claimed = amendment_sources.claim_source_package(
        source_session, package_id=package.id, environment="local-test"
    )
    assert claimed is not None
    if ownership == "expired":
        package.lease_expires_at = datetime.datetime.now(
            datetime.timezone.utc
        ) - datetime.timedelta(seconds=1)
    elif ownership == "finished":
        package.status = "failed"
    source_session.commit()
    previous_expiry = package.lease_expires_at
    assert not amendment_sources.extend_source_package_lease(
        source_session,
        package_id=package.id,
        environment="local-test",
        lease_token=uuid4() if ownership == "wrong_token" else claimed[1],
        lease_seconds=1500,
    )
    source_session.refresh(package)
    assert package.lease_expires_at == previous_expiry


def test_source_preparation_timeout_has_a_specific_safe_failure_code(
    source_session: Session,
) -> None:
    from onyx.db.amendment_sources import (
        create_source_package,
        mark_source_package_failed,
    )

    document_set = DocumentSet(
        name=f"annex-{uuid4()}", description="annex test", is_up_to_date=True
    )
    source_session.add(document_set)
    source_session.flush()
    package, _ = create_source_package(
        source_session,
        document_set_id=document_set.id,
        environment="local-test",
        idempotency_key=str(uuid4()),
        request_hash="a" * 64,
        input_spec={"url": "https://example.gov/update.htm"},
        created_by=None,
    )
    mark_source_package_failed(
        source_session,
        package_id=package.id,
        environment="local-test",
        failure=TimeoutError("private payload must not appear"),
    )
    source_session.refresh(package)
    assert package.status == "failed"
    assert package.issues[0]["code"] == "source_preparation_timeout"
    assert package.issues[0]["retryable"] is True
    assert "TimeoutError" in package.issues[0]["failure_detail"]
    assert "private payload" not in package.issues[0]["failure_detail"]


def test_retry_retains_distinct_url_occurrences_and_retries_unresolved_child(
    source_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    from contextlib import contextmanager
    from io import BytesIO

    from pypdf import PdfWriter

    from onyx.db.amendment_sources import (
        create_source_package,
        get_source_package,
        retry_source_package,
    )
    from onyx.file_store.file_store import get_default_file_store
    from onyx.regulatory.amendments import pdf_vision
    from onyx.regulatory.amendments.annexes import job
    from onyx.regulatory.amendments.annexes.models import AcquiredAsset
    from onyx.regulatory.amendments.annexes.sources import (
        DownloadedSource,
        SourceAcquisitionError,
    )

    @contextmanager
    def session_context() -> Generator[Session, None, None]:
        yield source_session

    monkeypatch.setattr(job, "get_session_with_current_tenant", session_context)

    def prepare_fixture(asset: AcquiredAsset, **_kwargs: object) -> AcquiredAsset:
        return asset.model_copy(update={"text": "Frozen PDF fixture"})

    monkeypatch.setattr(pdf_vision, "prepare_pdf_source", prepare_fixture)
    document_set = DocumentSet(
        name=f"annex-{uuid4()}", description="annex test", is_up_to_date=True
    )
    source_session.add(document_set)
    source_session.flush()
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    pdf = BytesIO()
    writer.write(pdf)
    root = "https://example.gov/update"
    missing = "https://example.gov/b/annex.pdf"
    fixtures = {
        root: DownloadedSource(
            b'<a href="a/step">Annex A</a><a href="b/step">Annex B</a>',
            "text/html",
            root,
        ),
        **{
            f"https://example.gov/{part}/step": DownloadedSource(
                b'<a href="annex.pdf">Ek 1</a>',
                "text/html",
                f"https://example.gov/{part}/step",
            )
            for part in ("a", "b")
        },
        **{
            f"https://example.gov/{part}/annex.pdf": DownloadedSource(
                pdf.getvalue(),
                "application/pdf",
                f"https://example.gov/{part}/annex.pdf",
            )
            for part in ("a", "b")
        },
    }
    calls: list[str] = []
    available = False

    def download(url: str, **_kwargs: object) -> DownloadedSource:
        calls.append(url)
        if url == missing and not available:
            raise SourceAcquisitionError("404")
        return fixtures[url]

    monkeypatch.setattr(job, "download_source", download)
    package, _ = create_source_package(
        source_session,
        document_set_id=document_set.id,
        environment="local-test",
        idempotency_key=str(uuid4()),
        request_hash="a" * 64,
        input_spec={"url": root},
        created_by=None,
    )
    package_id, document_set_id = package.id, document_set.id
    for attempt in range(3):
        if attempt:
            retry_source_package(
                source_session,
                package_id=package_id,
                document_set_id=document_set_id,
                environment="local-test",
            )
            calls.clear()
        available = attempt == 2
        job.run_source_package(package_id=package_id, environment="local-test")
        source_session.expire_all()
        updated = get_source_package(
            source_session,
            package_id=package_id,
            document_set_id=document_set_id,
            environment="local-test",
        )
        assert updated is not None and updated.manifest_file_id is not None
        assert updated.status == ("ready" if available else "partial")
        with get_default_file_store().read_file(updated.manifest_file_id) as stream:
            manifest = json.load(stream)
        assert len(manifest["links"]) == 4
        if attempt:
            assert calls == [missing]
        if not available:
            assert manifest["issues"][0]["code"] == "404"
            unresolved = [
                link for link in manifest["links"] if link["target_asset_hash"] is None
            ]
            assert len(unresolved) == 1
            assert unresolved[0]["parent_url"] == "https://example.gov/b/step"
            assert unresolved[0]["requested_url"] == missing
        else:
            assert manifest["issues"] == []
