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
) -> Generator[Session, None, None]:
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
