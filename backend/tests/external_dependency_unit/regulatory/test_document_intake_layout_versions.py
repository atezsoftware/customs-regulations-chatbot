"""Real retained sources keep their target version and source-cell identity."""

from copy import deepcopy
from datetime import date
from functools import partial
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4
from zipfile import ZipFile

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import document_intake_edit as editing
from onyx.db import document_intake_reprepare as repository
from onyx.db.document_intake_sources import create_zip_source_package
from onyx.db.models import (
    DocumentSet,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryFilePublication,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    User,
    UserFile,
)
from onyx.db.regulatory_annex_changes import capture_canonical_scope
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.indexing.embedder import DefaultIndexingEmbedder
from onyx.natural_language_processing.utils import TiktokenTokenizer
from onyx.regulatory.amendments.annexes import job as source_job
from onyx.regulatory.amendments.annexes.models import AnnexCanonicalSnapshot
from onyx.regulatory.document_intake import job as intake_job
from onyx.regulatory.document_intake import metadata_job, reprepare_job
from tests.external_dependency_unit.regulatory.test_document_intake import _ENVIRONMENT
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_owner as intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_session as intake_session,
)
from tests.external_dependency_unit.regulatory.test_document_intake_markdown import (
    native_markdown_runtime as native_markdown_runtime,
)
from tests.external_dependency_unit.regulatory.test_document_intake_reprepare import (
    _fixture,
    _mark_ready,
    _request,
)
from tests.unit.onyx.regulatory.test_document_intake_html_layout import _HTML
from tests.unit.onyx.regulatory.test_document_intake_table_continuations import (
    FRONT_TABLE,
    REAL_ROW,
)


@pytest.mark.parametrize("legacy_queued", [False, True])
def test_reprepare_worker_uses_server_frozen_versions_without_rebasing_old_requests(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
    legacy_queued: bool,
) -> None:
    batch, _run = _fixture(
        intake_session, intake_owner, native_markdown_runtime, monkeypatch
    )
    members = intake.intake_files(intake_session, batch)
    ids = {row.id: (row.user_file_id, row.source_asset_id) for row in members}
    request = _request(intake_session, batch)
    assert all(
        entry["target_chunking_version"] == "3"
        and entry["target_formatter_version"] == 4
        for entry in request.documents.values()
    )
    if legacy_queued:
        audit = deepcopy(request.documents)
        for entry in audit.values():
            entry.pop("target_chunking_version")
            entry.pop("target_formatter_version")
        request.documents = audit
        intake_session.commit()
    expected = ("2", 3) if legacy_queued else ("3", 4)
    observed = []
    prepare = reprepare_job.prepare_intake_file

    def capture(**kwargs: Any) -> Any:
        observed.append((kwargs["chunking_version"], kwargs["formatter_version"]))
        return prepare(**kwargs)

    monkeypatch.setattr(reprepare_job, "prepare_intake_file", capture)
    run_id = reprepare_job.run_document_intake_reprepare(
        intake_id=batch.id, request_id=request.id, tenant_id="public"
    )
    assert run_id is not None and observed == [expected] * len(members)
    intake_session.expire_all()
    assert {
        row.id: (row.user_file_id, row.source_asset_id)
        for row in intake.intake_files(intake_session, batch)
    } == ids
    assert {
        row.chunk_metadata["intake_chunking_version"]
        for row in intake_session.scalars(
            select(RegulatoryChunk).where(
                RegulatoryChunk.user_file_id.in_([row.user_file_id for row in members])
            )
        )
    } == {expected[0]}


def test_table_fragment_source_identity_survives_bulk_dates_labels_and_canonical_receipt(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.external_dependency_unit.regulatory import (
        test_document_intake_reprepare as fixtures,
    )

    source = ("# EK-3\n\n" + FRONT_TABLE).encode()
    monkeypatch.setattr(
        fixtures,
        "_markdown_input",
        lambda *_args: ("ek-3.md", source, {sha256(source).hexdigest(): source}),
    )
    monkeypatch.setattr(
        DefaultIndexingEmbedder,
        "from_db_search_settings",
        lambda **_kwargs: SimpleNamespace(
            embedding_model=SimpleNamespace(
                tokenizer=TiktokenTokenizer("text-embedding-3-small")
            )
        ),
    )
    batch, run_id = _fixture(
        intake_session,
        intake_owner,
        native_markdown_runtime,
        monkeypatch,
        legacy_chunks=False,
    )
    monkeypatch.setattr(
        editing,
        "resolve_intake_labeling_provider_binding",
        intake.resolve_intake_labeling_provider_binding,
    )
    _mark_ready(intake_session, batch, run_id)
    member = intake.intake_files(intake_session, batch)[0]
    rows = list(
        intake_session.scalars(
            select(RegulatoryChunk)
            .where(RegulatoryChunk.user_file_id == member.user_file_id)
            .order_by(RegulatoryChunk.position)
        )
    )
    fragments = [row for row in rows if "intake_table_row" in row.chunk_metadata]
    assert len(fragments) == 2
    assert "".join(row.text for row in fragments) == REAL_ROW
    original = {
        row.id: deepcopy(row.chunk_metadata["intake_table_row"]) for row in fragments
    }
    assert batch.review_sha256
    fresh_run = editing.edit_intake_chunks(
        intake_session,
        batch,
        expected_review_sha256=batch.review_sha256,
        edits=[
            editing.IntakeChunkEdit(
                member.id, row.id, {"validity_start_date": date(2025, 5, 9)}
            )
            for row in fragments
        ],
    )
    assert fresh_run is not None and fresh_run != run_id
    intake_session.flush()
    for row in fragments:
        intake_session.refresh(row)
        assert row.chunk_metadata["intake_table_row"] == original[row.id]
    for item in intake_session.scalars(
        select(RegulatoryLabelingItem).where(
            RegulatoryLabelingItem.run_id == fresh_run,
            RegulatoryLabelingItem.regulatory_chunk_id.in_(original),
        )
    ):
        metadata = item.source_snapshot["chunk_metadata"]
        assert isinstance(metadata, dict)
        typed_metadata = cast(dict[str, object], metadata)
        assert typed_metadata["intake_table_row"] == original[item.regulatory_chunk_id]
    snapshots = capture_canonical_scope(intake_session, member.user_file_id)
    for snapshot in snapshots:
        restored = AnnexCanonicalSnapshot.model_validate_json(
            snapshot.model_dump_json()
        )
        if snapshot.id in original:
            assert restored.metadata["intake_table_row"] == original[snapshot.id]
        if snapshot.metadata.get("chunk_variant") == "hierarchical_aggregate":
            assert "intake_table_row" not in restored.metadata
    with pytest.raises(ValueError, match="legal chunk metadata"):
        from onyx.db.document_intake_metadata import manual_edit_metadata

        manual_edit_metadata(
            fragments[0], {"chunk_metadata": {"intake_table_row": None}}
        )


def test_initial_retry_fills_only_virgin_annex_then_explicit_reprepare_upgrades_same_files(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user, _target, model = intake_owner
    model_id = model.id
    contents = {
        "main.html": _HTML.replace(
            "</body>", "<p><a href='ek-3.md'>EK-3 için tıklayınız</a></p></body>"
        ).encode(),
        "ek-3.md": ("# EK-3\n\n" + FRONT_TABLE).encode(),
    }
    archive = BytesIO()
    with ZipFile(archive, "w") as zipped:
        for path, content in contents.items():
            zipped.writestr(path, content)
    package, _ = create_zip_source_package(
        intake_session,
        user=user,
        environment=_ENVIRONMENT,
        idempotency_key=uuid4(),
        display_name="same-sources.zip",
        content=archive.getvalue(),
        store=native_markdown_runtime,
    )
    package_id = package.id
    intake_session.commit()
    source_job.run_source_package(package_id=package_id, environment=_ENVIRONMENT)
    batch, _ = intake.create_intake(
        intake_session,
        source_package_id=package_id,
        idempotency_key=uuid4(),
        labeling_model_configuration_id=model_id,
        user=user,
        environment=_ENVIRONMENT,
    )
    batch_id = batch.id
    intake_session.commit()
    monkeypatch.setattr(
        intake,
        "resolve_intake_labeling_provider_binding",
        lambda *_args, **_kwargs: SimpleNamespace(
            model="gemini-3.5-flash",
            model_dump=lambda **_options: {"model_configuration_id": model_id},
        ),
    )
    monkeypatch.setattr(metadata_job, "run_intake_metadata", lambda **_kwargs: None)
    monkeypatch.setattr(reprepare_job, "run_intake_metadata", lambda **_kwargs: None)
    monkeypatch.setattr(
        DefaultIndexingEmbedder,
        "from_db_search_settings",
        lambda **_kwargs: SimpleNamespace(
            embedding_model=SimpleNamespace(
                tokenizer=TiktokenTokenizer("text-embedding-3-small")
            )
        ),
    )
    prepare = intake_job.prepare_intake_file
    with monkeypatch.context() as old_runtime:
        old_runtime.setattr(
            intake_job,
            "prepare_intake_file",
            partial(prepare, formatter_version=3, chunking_version="2"),
        )
        with pytest.raises(ValueError, match="table row exceeds"):
            intake_job.run_document_intake(intake_id=batch_id, tenant_id="public")
    intake_session.expire_all()
    batch = intake_session.get_one(type(batch), batch_id)
    members = intake.intake_files(intake_session, batch)
    root = next(member for member in members if member.role == "root")
    annex = next(member for member in members if member.role == "annex")
    ids = {
        member.id: (member.user_file_id, member.source_asset_id) for member in members
    }
    baseline = [
        row.model_dump(mode="json")
        for row in capture_canonical_scope(intake_session, root.user_file_id)
    ]
    original_receipt = deepcopy(
        intake_session.get_one(
            RegulatoryFilePublication, root.user_file_id
        ).original_ingestion_receipt
    )
    assert baseline and not capture_canonical_scope(intake_session, annex.user_file_id)
    assert all(row["metadata"]["intake_chunking_version"] == "2" for row in baseline)
    assert batch.status == "failed" and batch.stage == "chunking"
    intake.retry_intake(intake_session, batch)
    intake_session.commit()
    run_id = intake_job.run_document_intake(intake_id=batch_id, tenant_id="public")
    assert run_id is not None
    intake_session.expire_all()
    assert [
        row.model_dump(mode="json")
        for row in capture_canonical_scope(intake_session, root.user_file_id)
    ] == baseline
    assert (
        intake_session.get_one(
            RegulatoryFilePublication, root.user_file_id
        ).original_ingestion_receipt
        == original_receipt
    )
    assert {
        row.metadata["intake_chunking_version"]
        for row in capture_canonical_scope(intake_session, annex.user_file_id)
    } == {"3"}
    run = intake_session.get_one(RegulatoryLabelingRun, run_id)
    run.status, run.stage = "failed", "finished"
    batch.status, batch.stage = "partial", "labeling"
    intake_session.commit()
    monkeypatch.setattr(
        repository,
        "get_session_with_tenant",
        intake_job.get_session_with_current_tenant,
    )
    monkeypatch.setattr(
        reprepare_job,
        "get_session_with_current_tenant",
        intake_job.get_session_with_current_tenant,
    )
    monkeypatch.setattr(
        reprepare_job, "get_default_file_store", lambda: native_markdown_runtime
    )
    request = _request(intake_session, batch)
    assert (
        reprepare_job.run_document_intake_reprepare(
            intake_id=batch_id, request_id=request.id, tenant_id="public"
        )
        is not None
    )
    intake_session.expire_all()
    assert {
        member.id: (member.user_file_id, member.source_asset_id)
        for member in intake.intake_files(intake_session, batch)
    } == ids
    for member in members:
        assert intake_session.get_one(UserFile, member.user_file_id).file_id
        assert {
            row.metadata["intake_chunking_version"]
            for row in capture_canonical_scope(intake_session, member.user_file_id)
        } == {"3"}
    repaired = capture_canonical_scope(intake_session, root.user_file_id)
    assert {row.metadata.get("article_no") for row in repaired} >= {
        "3",
        "4",
        "27",
        "28",
    }
