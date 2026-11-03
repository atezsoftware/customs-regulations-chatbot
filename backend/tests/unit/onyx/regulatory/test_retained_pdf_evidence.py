"""Provider-free receipt transactions on isolated SQL and controlled raw blobs."""

import hashlib
import importlib.util
import io
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock
from uuid import uuid4

import pytest
import sqlalchemy as sa
from fastapi import UploadFile
from pypdf import PdfWriter
from sqlalchemy.orm import Session
from starlette.datastructures import Headers

from onyx.db import regulatory_retained_pdf_evidence as repository
from onyx.db.enums import Permission
from onyx.db.models import RegulatoryRetainedPdfEvidence as Evidence
from onyx.db.models import User
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.regulatory.retained_pdf_evidence import (
    RetainedPdfDescriptor,
    validate_retained_pdf,
)
from onyx.server.features.regulatory import retained_pdf_evidence_api as api


def _pdf(width: int = 72) -> bytes:
    writer, output = PdfWriter(), io.BytesIO()
    writer.add_blank_page(width=width, height=72)
    writer.write(output)
    return output.getvalue()


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Any]:
    metadata = sa.MetaData()
    parents = {
        name: sa.Table(name, metadata, sa.Column(column, kind, primary_key=True))
        for name, column, kind in (
            ("user_file", "id", sa.Uuid),
            ("document_set", "id", sa.Integer),
            ("user", "id", sa.Uuid),
            ("file_record", "file_id", sa.String),
        )
    }
    cast(sa.Table, Evidence.__table__).to_metadata(metadata)
    engine = sa.create_engine("sqlite:///" + str(tmp_path / "receipts.sqlite"))
    metadata.create_all(engine)
    context = SimpleNamespace(
        file_id=uuid4(),
        user=SimpleNamespace(id=uuid4()),
        key=uuid4(),
        digest="a" * 64,
        file=SimpleNamespace(regulatory_chunk_generation_hash="generation-v1"),
        saved={},
        deleted=[],
        save_calls=0,
    )
    with engine.begin() as connection:
        for name, value in (
            ("user_file", context.file_id),
            ("document_set", 15),
            ("user", context.user.id),
        ):
            connection.execute(parents[name].insert().values(id=value))

    def save_file(content: io.BytesIO, *, file_id: str, **_kwargs: Any) -> str:
        context.save_calls += 1
        context.saved[file_id] = content.read()
        with engine.begin() as connection:
            connection.execute(parents["file_record"].insert().values(file_id=file_id))
        return file_id

    def delete_file(file_id: str, **_kwargs: Any) -> None:
        context.deleted.append(file_id)
        context.saved.pop(file_id)
        with engine.begin() as connection:
            connection.execute(
                parents["file_record"]
                .delete()
                .where(parents["file_record"].c.file_id == file_id)
            )

    @contextmanager
    def current_session() -> Iterator[Session]:
        with Session(engine) as session:
            yield session

    store = SimpleNamespace(
        save_file=save_file,
        delete_file=delete_file,
        read_file=lambda file_id: io.BytesIO(context.saved[file_id]),
    )
    context.store = store
    monkeypatch.setattr(api, "get_default_file_store", lambda: store)
    monkeypatch.setattr(api, "get_session_with_current_tenant", current_session)
    monkeypatch.setattr(
        repository,
        "retained_pdf_scope",
        lambda *_args, **_kwargs: (context.file, context.digest),
    )
    with Session(engine) as session:
        context.session = session
        yield context
    engine.dispose()


def _attach(context: Any, content: bytes | None = None, **changes: Any) -> Any:
    values: dict[str, Any] = dict(
        user_file_id=context.file_id,
        document_set_id=15,
        idempotency_key=context.key,
        expected_canonical_sha256="a" * 64,
        relation="equivalent_rendition",
        reason="Reviewed official source parity",
        source_url="https://example.gov/original.pdf",
        user=context.user,
        db_session=context.session,
        file=UploadFile(
            filename="Özgün kaynak.pdf",
            file=io.BytesIO(content or _pdf()),
            headers=Headers({"content-type": "application/pdf"}),
        ),
    )
    return api.attach_retained_pdf(**{**values, **changes})


def test_append_download_list_and_exact_retry_preserve_original_anchor(
    harness: Any,
) -> None:
    before = vars(harness.file).copy()
    receipt = _attach(harness)
    assert receipt.sha256 == hashlib.sha256(_pdf()).hexdigest()
    assert receipt.canonical_generation_hash == "generation-v1"
    harness.digest = "b" * 64
    assert _attach(harness) == receipt
    assert harness.save_calls == 1 and harness.deleted == []
    listing = api.list_retained_pdfs(harness.file_id, 15, harness.user, harness.session)
    assert listing.canonical_sha256 == "b" * 64 and listing.items == [receipt]
    response = api.download_retained_pdf(
        harness.file_id, receipt.id, 15, harness.user, harness.session
    )
    assert bytes(response.body) == _pdf()
    assert "filename*=UTF-8''" in response.headers["content-disposition"]
    assert vars(harness.file) == before


@pytest.mark.parametrize(
    "changes",
    [
        {"reason": "Different review"},
        {"relation": "historical_rendition"},
        {"expected_canonical_sha256": "b" * 64},
        {"idempotency_key": uuid4()},
    ],
)
def test_changed_descriptor_cannot_reuse_same_bytes(
    harness: Any, changes: dict[str, Any]
) -> None:
    receipt = _attach(harness)
    with pytest.raises(OnyxError) as error:
        _attach(harness, **changes)
    assert error.value.error_code == OnyxErrorCode.CONFLICT
    assert harness.save_calls == 1 and harness.deleted == []
    assert harness.session.get(Evidence, receipt.id).reason == receipt.reason


def test_changed_bytes_same_key_and_stale_new_anchor_fail_before_upload(
    harness: Any,
) -> None:
    _attach(harness)
    with pytest.raises(OnyxError):
        _attach(harness, _pdf(144))
    harness.digest = "c" * 64
    with pytest.raises(OnyxError):
        _attach(harness, _pdf(144), idempotency_key=uuid4())
    assert harness.save_calls == 1


@pytest.mark.parametrize("phase", ["save", "flush", "commit-before", "commit-after"])
def test_failures_never_delete_bytes_after_a_commit_attempt(
    harness: Any,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    original_commit = harness.session.commit
    if phase == "save":
        original_save = harness.store.save_file

        def fail_save(*args: Any, **kwargs: Any) -> None:
            original_save(*args, **kwargs)
            raise RuntimeError("store acknowledgement lost")

        monkeypatch.setattr(harness.store, "save_file", fail_save)
    elif phase == "flush":
        original_flush = harness.session.flush

        def fail_flush(*args: Any, **kwargs: Any) -> None:
            if any(isinstance(row, Evidence) for row in harness.session.new):
                raise RuntimeError("flush failed")
            original_flush(*args, **kwargs)

        monkeypatch.setattr(harness.session, "flush", fail_flush)
    else:

        def fail_commit() -> None:
            if phase == "commit-after":
                original_commit()
            raise RuntimeError("commit acknowledgement lost")

        monkeypatch.setattr(harness.session, "commit", fail_commit)
    with pytest.raises(RuntimeError):
        _attach(harness)
    assert len(harness.deleted) == (1 if phase in ("save", "flush") else 0)
    if phase == "commit-after":
        monkeypatch.setattr(harness.session, "commit", original_commit)
        assert _attach(harness).sha256 == hashlib.sha256(_pdf()).hexdigest()
        assert harness.save_calls == 1


def test_resolved_duplicate_race_cleans_only_its_new_pointer(
    harness: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _attach(harness)
    original = repository.existing_retained_pdf
    calls = 0

    def racing_lookup(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return None if calls == 1 else original(*args, **kwargs)

    monkeypatch.setattr(repository, "existing_retained_pdf", racing_lookup)
    assert _attach(harness) == receipt
    assert harness.save_calls == 2 and len(harness.deleted) == 1
    assert len(harness.saved) == 1 and next(iter(harness.saved.values())) == _pdf()


def test_corrupt_download_and_cross_target_receipt_fail_closed(harness: Any) -> None:
    receipt = _attach(harness)
    pointer = harness.session.get(Evidence, receipt.id).raw_file_id
    harness.saved[pointer] = b"corrupt"
    with pytest.raises(OnyxError):
        api.download_retained_pdf(
            harness.file_id, receipt.id, 15, harness.user, harness.session
        )
    with pytest.raises(ValueError):
        repository.get_retained_pdf(harness.session, uuid4(), receipt.id, 15)


@pytest.mark.parametrize(
    "field,value",
    [
        ("reason", "Changed review"),
        ("relation", "historical_rendition"),
        ("canonical_sha256", "b" * 64),
        ("canonical_generation_hash", "changed"),
        ("byte_count", 1),
        ("mime_type", "text/plain"),
    ],
)
def test_stored_receipt_tampering_cannot_be_returned(
    harness: Any, field: str, value: Any
) -> None:
    receipt = _attach(harness)
    row = harness.session.get(Evidence, receipt.id)
    setattr(row, field, value)
    with harness.session.no_autoflush:
        with pytest.raises(repository.RetainedPdfConflictError):
            repository.get_retained_pdf(
                harness.session, harness.file_id, receipt.id, 15
            )
        with pytest.raises(repository.RetainedPdfConflictError):
            repository.list_retained_pdfs(harness.session, harness.file_id)


@pytest.mark.parametrize("failure", ["permission", "set", "membership", "canonical"])
def test_current_scope_requires_admin_editable_pc_and_canonical_membership(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    session, user = Mock(spec=Session), cast(User, SimpleNamespace(id=uuid4()))
    monkeypatch.setattr(
        repository,
        "get_effective_permissions",
        lambda _user: (
            set() if failure == "permission" else {Permission.FULL_ADMIN_PANEL_ACCESS}
        ),
    )
    monkeypatch.setattr(
        repository,
        "resolve_intake_target",
        lambda *_args: SimpleNamespace(id=99 if failure == "set" else 15),
    )
    session.get.side_effect = [
        SimpleNamespace(),
        None if failure == "membership" else object(),
    ]
    session.scalar.return_value = None if failure == "canonical" else "canonical-chunk"
    with pytest.raises(PermissionError if failure == "permission" else ValueError):
        repository.retained_pdf_scope(session, user, uuid4(), 15)


def test_native_pdf_and_descriptor_validation_never_converts() -> None:
    assert (
        validate_retained_pdf(_pdf(), "application/pdf")
        == hashlib.sha256(_pdf()).hexdigest()
    )
    for content, mime in ((b"invalid", "application/pdf"), (_pdf(), "text/plain")):
        with pytest.raises(ValueError):
            validate_retained_pdf(content, mime)
    with pytest.raises(ValueError):
        RetainedPdfDescriptor(
            document_set_id=15,
            expected_canonical_sha256="a" * 64,
            idempotency_key=uuid4(),
            original_filename="../source.pdf",
            relation="historical_rendition",
            reason="reviewed",
        )


def test_historical_read_scope_preserves_access_without_active_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, user, file = (
        Mock(spec=Session),
        cast(User, SimpleNamespace(id=uuid4())),
        SimpleNamespace(),
    )
    monkeypatch.setattr(
        repository,
        "get_effective_permissions",
        lambda _user: {Permission.FULL_ADMIN_PANEL_ACCESS},
    )
    monkeypatch.setattr(
        repository, "resolve_intake_target", lambda *_args: SimpleNamespace(id=15)
    )
    monkeypatch.setattr(repository, "canonical_scope_digest", lambda *_args: "a" * 64)
    session.get.side_effect = [file, object()]
    assert repository.retained_pdf_scope(
        session, user, uuid4(), 15, require_active=False
    ) == (file, "a" * 64)
    session.scalar.assert_not_called()


def test_migration_matches_additive_model_and_preserves_parent_deletion_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = (
        Path(__file__).resolve().parents[4]
        / "alembic/versions/2fcae483a930_retain_raw_pdf_evidence.py"
    )
    spec = importlib.util.spec_from_file_location("retained_pdf_migration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    operations = Mock()
    monkeypatch.setattr(module, "op", operations)
    module.upgrade()
    args = operations.create_table.call_args.args
    columns = {
        column.name: column for column in args[1:] if isinstance(column, sa.Column)
    }
    assert set(columns) == set(Evidence.__table__.columns.keys())
    assert next(iter(columns["user_file_id"].foreign_keys)).ondelete == "CASCADE"
    assert next(iter(columns["created_by"].foreign_keys)).ondelete == "SET NULL"
    assert not operations.execute.called and operations.create_table.call_count == 1
    module.downgrade()
    operations.drop_table.assert_called_once_with(Evidence.__tablename__)
