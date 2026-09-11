import datetime
import json
from collections.abc import Generator
from contextlib import nullcontext
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy import JSON, Column, MetaData, Table, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session

from onyx.db.amendment_sources import mark_source_package_failed
from onyx.db.models import AmendmentSourcePackage


@pytest.fixture
def source_row() -> Generator[tuple[Session, AmendmentSourcePackage], None, None]:
    # Execute the production mapped UPDATE without external services. Only DDL's
    # PostgreSQL JSONB type is adapted; this does not test PostgreSQL lock races.
    engine = create_engine("sqlite://")
    table = Table(
        AmendmentSourcePackage.__tablename__,
        MetaData(),
        *(
            Column(
                column.name,
                JSON() if isinstance(column.type, JSONB) else column.type,
                primary_key=column.primary_key,
                nullable=column.nullable,
            )
            for column in AmendmentSourcePackage.__table__.columns
        ),
    )
    table.create(engine)
    with Session(engine) as session:
        now = datetime.datetime.now(datetime.timezone.utc)
        package = AmendmentSourcePackage(
            id=uuid4(),
            document_set_id=23,
            environment="fixture",
            idempotency_key="owned",
            request_hash="a" * 64,
            input_spec={"text": "original"},
            input_file_id="original-file",
            status="processing",
            issues=[],
            lease_token=uuid4(),
            created_at=now,
            updated_at=now,
        )
        session.add(package)
        session.commit()
        yield session, package
    engine.dispose()


@pytest.mark.parametrize(
    "mode",
    [
        "owned",
        "wrong_environment",
        "wrong_package",
        "stale_lease",
        "terminal",
        "already_failed",
        "unclaimed",
        "claimed_without_lease",
    ],
)
def test_source_failure_receipt_is_written_only_by_current_owner(
    source_row: tuple[Session, AmendmentSourcePackage],
    mode: str,
) -> None:
    session, package = source_row
    if mode == "terminal":
        package.status = "ready"
    if mode == "already_failed":
        package.status = "failed"
        package.issues = [{"code": "previous_failure"}]
    if mode == "unclaimed":
        package.lease_token = None
    session.commit()
    before = (
        package.status,
        list(package.issues),
        package.input_spec.copy(),
        package.input_file_id,
        package.lease_token,
    )
    token = (
        uuid4()
        if mode == "stale_lease"
        else None
        if mode == "claimed_without_lease"
        else package.lease_token
    )
    mark_source_package_failed(
        session,
        package_id=uuid4() if mode == "wrong_package" else package.id,
        environment="other" if mode == "wrong_environment" else "fixture",
        lease_token=token,
        failure=ValueError("SECRET_SOURCE_AND_CREDENTIAL"),
    )
    session.refresh(package)
    if mode in {"owned", "unclaimed"}:
        assert package.status == "failed"
        receipt = package.issues[0]
        assert receipt["code"] == "acquisition_failed" and receipt["retryable"] is True
        detail = receipt["failure_detail"]
        assert "SECRET" not in detail and len(detail) <= 4000
        assert json.loads(detail)["stage"] == "source_package"
        assert json.loads(detail)["exceptions"][0]["type"] == "ValueError"
    else:
        assert (package.status, package.issues) == before[:2]
    assert (package.input_spec, package.input_file_id, package.lease_token) == before[
        2:
    ]


def test_source_worker_passes_actual_exception_to_durable_guard(
    source_row: tuple[Session, AmendmentSourcePackage],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.regulatory.amendments.annexes import job

    session, package = source_row
    failure = ValueError("SECRET_WORKER_INPUT")
    monkeypatch.setattr(
        job, "get_session_with_current_tenant", lambda: nullcontext(session)
    )
    monkeypatch.setattr(
        job,
        "claim_source_package",
        lambda *_args, **_kwargs: (package, package.lease_token),
    )
    monkeypatch.setattr(job, "list_source_assets", lambda *_args: [])
    monkeypatch.setattr(job, "get_default_file_store", MagicMock(side_effect=failure))
    with pytest.raises(ValueError) as captured:
        job.run_source_package(package_id=package.id, environment="fixture")
    assert captured.value is failure
    session.refresh(package)
    assert package.status == "failed"
    detail = package.issues[0]["failure_detail"]
    assert "SECRET" not in detail
    frames = json.loads(detail)["exceptions"][0]["frames"]
    assert any(frame["function"] == "run_source_package" for frame in frames)


def test_canary_retains_source_failure_before_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.db.regulatory_annex_acceptance import CanaryRun
    from onyx.regulatory.amendments.annexes import acceptance_canary
    from onyx.regulatory.amendments.annexes.dev_acceptance import safe_failure_detail

    detail = safe_failure_detail("source_package", ValueError("SECRET"))
    result = {
        "status": "failed",
        "issues": [
            {"code": "acquisition_failed", "retryable": True, "failure_detail": detail}
        ],
    }
    run = CanaryRun(
        release_sha="a" * 40, user_id=uuid4(), package_id=uuid4(), document_set_id=23
    )
    monkeypatch.setattr(
        acceptance_canary, "request_json", lambda *_args, **_kwargs: result
    )
    monkeypatch.setattr(acceptance_canary.time, "monotonic", lambda: 0)
    saved = MagicMock()
    monkeypatch.setattr(acceptance_canary, "save_canary", saved)
    with pytest.raises(ValueError, match="fictional_source_package_failed"):
        acceptance_canary.wait_package(MagicMock(), run, 10)
    assert run.evidence == {"source_package_status": "failed", "worker_failure": detail}
    saved.assert_called_once_with(run)


def test_source_receipt_validation_paths_are_bounded_and_private() -> None:
    from pydantic import ValidationError
    from pydantic_core import PydanticCustomError

    from onyx.regulatory.amendments.annexes.dev_acceptance import (
        safe_failure_detail as legacy_helper,
    )
    from onyx.regulatory.failure_details import safe_failure_detail

    assert legacy_helper is safe_failure_detail
    error = ValidationError.from_exception_data(
        "SECRET_TITLE",
        [
            {
                "type": "list_type",
                "loc": ("elements", 0, "box"),
                "input": "SECRET_INPUT",
            },
            {
                "type": PydanticCustomError("SECRET_TYPE", "SECRET_MESSAGE"),
                "loc": ("SECRET_FIELD",),
                "input": "SECRET_INPUT",
            },
        ],
    )
    detail = safe_failure_detail("source_package", error)
    assert "SECRET" not in detail and len(detail) <= 4000
    assert json.loads(detail)["exceptions"][0]["validation_errors"] == [
        {"loc": ["elements", 0, "box"], "type": "list_type"},
        {"loc": ["unknown"], "type": "unknown"},
    ]


def test_dispatch_failure_without_exception_retains_existing_issue_shape(
    source_row: tuple[Session, AmendmentSourcePackage],
) -> None:
    session, package = source_row
    package.lease_token = None
    session.commit()
    mark_source_package_failed(session, package_id=package.id, environment="fixture")
    session.refresh(package)
    assert package.issues == [{"code": "acquisition_failed", "retryable": True}]
