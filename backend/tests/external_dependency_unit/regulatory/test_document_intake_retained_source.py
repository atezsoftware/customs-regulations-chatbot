import datetime
import json
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from onyx.auth.schemas import UserRole
from onyx.configs.constants import FileOrigin
from onyx.db import document_intake as intake
from onyx.db import document_intake_edit as editing
from onyx.db import document_intake_metadata as metadata
from onyx.db import regulatory_labeling as labeling
from onyx.db.document_intake_preview import intake_prepared_preview
from onyx.db.enums import UserFileProjectionRepairStatus, UserFileStatus
from onyx.db.models import (
    AmendmentSourcePackage,
    DocumentSet,
    DocumentSet__UserFile,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryDocumentIntakeFile,
    RegulatoryDocumentIntakeReprepare,
    RegulatoryFilePublication,
    RegulatoryLabelingRun,
    RegulatorySourceAsset,
    User,
    UserFile,
    UserFileProjectionRepair,
)
from onyx.db.regulatory_original_ingestion import original_attachment_receipt
from onyx.db.regulatory_writer_publication import canonical_scope_digest
from onyx.file_store.file_store import FileStore
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.regulatory.amendments.annexes.models import SourceLink
from onyx.regulatory.document_intake import reprepare_job
from onyx.regulatory.document_intake.metadata import (
    METADATA_AUDIT_KEY,
    ChunkLegalMetadata,
    LegalMetadataFact,
    LegalMetadataResult,
    metadata_input_hash,
)
from onyx.regulatory.document_intake.models import IntakeSourceAsset
from onyx.regulatory.document_intake.preparation import clone_intake_original
from tests.external_dependency_unit.conftest import create_test_user
from tests.external_dependency_unit.regulatory.test_document_intake import (
    _ENVIRONMENT,
    _package,
)
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
    _request,
)


def test_deleted_import_retains_source_and_history_and_cannot_be_approved(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
) -> None:
    owner, target, model = intake_owner
    package, assets, links = _package(intake_session, owner, target)
    batch, _created = intake.create_intake(
        intake_session,
        source_package_id=package.id,
        idempotency_key=uuid4(),
        labeling_model_configuration_id=model.id,
        user=owner,
        environment=_ENVIRONMENT,
    )
    store = PostgresBackedFileStore()
    bound_store = cast(
        FileStore,
        SimpleNamespace(
            read_file=lambda identifier: store.read_file(
                identifier, db_session=intake_session
            ),
            save_file=lambda content, **kwargs: store.save_file(
                content, **kwargs, db_session=intake_session
            ),
        ),
    )
    originals = {}
    for asset in assets:
        with store.read_file(asset.file_id, db_session=intake_session) as stream:
            originals[asset.file_id] = stream.read()
    copies = {
        asset.id: clone_intake_original(
            intake_id=batch.id,
            asset=IntakeSourceAsset.model_validate(asset),
            store=bound_store,
        )
        for asset in assets
    }
    members = intake.admit_intake_assets(
        intake_session,
        batch,
        assets,
        links,
        root_hash=assets[0].sha256,
        owned_file_ids=copies,
    )
    member = next(item for item in members if item.role == "annex")
    file = intake_session.get_one(UserFile, member.user_file_id)
    assert file.file_id == copies[member.source_asset_id]
    assert file.file_id not in originals
    batch.status, batch.stage = "ready", "review"
    batch.review_sha256 = intake.review_digest(intake_session, batch)
    reviewed = batch.review_sha256
    assert reviewed is not None
    original_name = file.name
    stored_copy = file.file_id
    intake_session.execute(
        delete(DocumentSet__UserFile).where(
            DocumentSet__UserFile.user_file_id == file.id
        )
    )
    store.delete_file(stored_copy, db_session=intake_session)
    intake_session.delete(file)
    intake_session.flush()

    snapshot = intake.intake_snapshot(intake_session, batch)
    deleted = next(
        item for item in snapshot["documents"] if item["id"] == str(member.id)
    )
    assert deleted["status"] == "deleted"
    assert deleted["display_name"] == original_name
    assert deleted["parent_document_id"] is not None
    assert intake.intake_files(intake_session, batch)
    for source_id, original in originals.items():
        with store.read_file(source_id, db_session=intake_session) as stream:
            assert stream.read() == original
    with pytest.raises(intake.IntakeConflictError, match="membership changed"):
        intake.approve_intake(intake_session, batch, reviewed)


def test_completed_approved_same_generation_reprepare_keeps_original_after_reviewed_dates(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = intake_session
    batch, _old_run = _fixture(
        session, intake_owner, native_markdown_runtime, monkeypatch, legacy_chunks=False
    )
    members = intake.intake_files(session, batch)
    originals: dict[UUID, dict[str, Any]] = {}
    for member in members:
        original = session.get_one(
            RegulatoryFilePublication, member.user_file_id
        ).original_ingestion_receipt
        assert original is not None
        originals[member.user_file_id] = deepcopy(original)
    request = _request(session, batch)
    request_id = request.id

    def enrich(*, intake_id: UUID, lease_token: UUID, tenant_id: str) -> None:
        assert tenant_id == "public"
        with reprepare_job.get_session_with_current_tenant() as current:
            _batch, work = metadata.load_metadata_work(current, intake_id, lease_token)
            for document_id, chunks in work:
                source = next(
                    (chunk for chunk in chunks if "01/01/2027" in chunk.text), None
                )
                result = LegalMetadataResult(
                    chunks=[
                        ChunkLegalMetadata(
                            chunk_id=chunk.chunk_id,
                            fields=[
                                LegalMetadataFact(
                                    field="validity_start_date",
                                    value="2027-01-01",
                                    source_chunk_id=source.chunk_id,
                                    evidence_quote=source.text,
                                )
                            ]
                            if source
                            else [],
                        )
                        for chunk in chunks
                    ]
                )
                metadata.apply_metadata_results(
                    current,
                    intake_id=intake_id,
                    lease_token=lease_token,
                    document_id=document_id,
                    expected_input_sha256=metadata_input_hash(chunks),
                    results=[("a" * 64, result)],
                )
            current.commit()

    monkeypatch.setattr(reprepare_job, "run_intake_metadata", enrich)
    run_id = reprepare_job.run_document_intake_reprepare(
        intake_id=batch.id, request_id=request_id, tenant_id="public"
    )
    assert run_id is not None
    session.expire_all()
    request = session.get_one(RegulatoryDocumentIntakeReprepare, request_id)
    prepared_documents = deepcopy(request.documents)
    for member in members:
        file = session.get_one(UserFile, member.user_file_id)
        assert (
            file.regulatory_chunk_generation_hash
            == originals[file.id]["generation_hash"]
        )
        assert (
            request.documents[str(member.id)]["after_receipt"]["generation_hash"]
            == originals[file.id]["generation_hash"]
        )

    def complete_labels(identifier: UUID) -> None:
        current = session.get_one(RegulatoryLabelingRun, identifier)
        lease = labeling.claim_run(
            session,
            run_id=identifier,
            expected_generation=current.lease_generation,
            lease_seconds=300,
        )
        assert lease is not None
        for item in labeling.prepare_next_item_page(session, lease, limit=128):
            item.status, item.labels = "completed", ["Gümrük"]
            assignments: list[dict[str, object]] = [
                {"label_id": "Gümrük", "evidence_quote": item.text_snapshot}
            ]
            item.assignments = assignments
        current.completed_chunks = current.total_chunks
        while labeling.has_pending_derived_projections(session, lease):
            labeling.project_derived_labels(session, lease, limit=128)
        labeling.release_run(
            session, lease, status="completed", stage="finished", finished=True
        )
        session.flush()
        intake.refresh_intake(session, batch)
        assert batch.status == "ready" and batch.review_sha256
        session.commit()

    complete_labels(run_id)
    root = next(member for member in members if member.role == "root")
    row = session.scalar(
        select(RegulatoryChunk)
        .where(
            RegulatoryChunk.user_file_id == root.user_file_id,
            RegulatoryChunk.chunk_type != "hierarchical_aggregate",
        )
        .order_by(RegulatoryChunk.position)
    )
    assert row is not None and row.validity_start_date == datetime.date(2027, 1, 1)
    monkeypatch.setattr(
        editing,
        "resolve_intake_labeling_provider_binding",
        intake.resolve_intake_labeling_provider_binding,
    )
    edit_review = batch.review_sha256
    assert edit_review is not None
    edited_run_id = editing.edit_intake_chunk(
        session,
        batch,
        document_id=root.id,
        chunk_id=row.id,
        expected_review_sha256=edit_review,
        updates={"validity_start_date": datetime.date(2028, 1, 1)},
    )
    assert edited_run_id is not None
    complete_labels(edited_run_id)
    assert (
        row.chunk_metadata[METADATA_AUDIT_KEY]["fields"]["validity_start_date"][
            "source"
        ]
        == "manual"
    )
    approved = batch.review_sha256
    assert approved is not None
    deliveries = intake.approve_intake(session, batch, approved)
    for identifier, token in deliveries:
        file = session.get_one(UserFile, identifier)
        file.status = UserFileStatus.COMPLETED
        publication = session.get_one(RegulatoryFilePublication, identifier)
        publication.epoch = 1
        repair = session.get_one(UserFileProjectionRepair, identifier)
        assert repair.attempt_id == token
        repair.status = UserFileProjectionRepairStatus.SUCCEEDED
    session.flush()
    intake.refresh_intake(session, batch)
    assert (
        batch.status == "completed" and intake.review_digest(session, batch) == approved
    )
    session.commit()
    for member in members:
        receipt = original_attachment_receipt(session, member.user_file_id)
        assert receipt is not None
        assert receipt.raw_sha256 == originals[member.user_file_id]["raw_sha256"]
        assert receipt.canonical_sha256 == canonical_scope_digest(
            session, member.user_file_id
        )

    for mutation in (
        "text",
        "date",
        "file",
        "generation",
        "token",
        "member",
        "batch",
        "gate",
        "epoch",
        "ordinal",
        "supersession",
        "source",
        "unassociated",
    ):
        savepoint = session.begin_nested()
        try:
            file = session.get_one(UserFile, root.user_file_id)
            publication = session.get_one(RegulatoryFilePublication, file.id)
            if mutation == "text":
                row.text += " Unreviewed change."
            elif mutation == "date":
                row.validity_start_date = datetime.date(2030, 1, 1)
            elif mutation == "file":
                file.file_id = "unreviewed-original"
            elif mutation == "generation":
                file.regulatory_chunk_generation_hash = "0" * 64
            elif mutation == "token":
                root.approval_attempt_id = None
            elif mutation == "member":
                members[-1].status = "indexing"
            elif mutation == "batch":
                batch.status = "approving"
            elif mutation == "gate":
                publication.gate_closed = True
            elif mutation == "epoch":
                publication.epoch = 0
            elif mutation == "ordinal":
                row.projection_ordinal += 100
            elif mutation == "supersession":
                row.superseded_by_chunk_id = next(
                    chunk.id
                    for chunk in session.scalars(select(RegulatoryChunk))
                    if chunk.id != row.id
                )
            elif mutation == "source":
                row.source = "amendment"
            else:
                session.execute(
                    delete(RegulatoryDocumentIntakeFile).where(
                        RegulatoryDocumentIntakeFile.intake_id == batch.id
                    )
                )
            session.flush()
            assert original_attachment_receipt(session, root.user_file_id) is None, (
                mutation
            )
        finally:
            savepoint.rollback()
            session.expire_all()
        assert original_attachment_receipt(session, root.user_file_id) is not None
    assert (
        session.get_one(RegulatoryDocumentIntakeReprepare, request_id).documents
        == prepared_documents
    )
    for member in members:
        assert (
            session.get_one(
                RegulatoryFilePublication, member.user_file_id
            ).original_ingestion_receipt
            == originals[member.user_file_id]
        )


def test_intake_preview_binds_owned_binary_and_verified_text_and_fails_closed(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
) -> None:
    session = intake_session
    owner, target, model = intake_owner
    store = PostgresBackedFileStore()
    package = AmendmentSourcePackage(
        document_set_id=target.id,
        created_by=owner.id,
        environment=_ENVIRONMENT,
        idempotency_key=str(uuid4()),
        request_hash=uuid4().hex * 2,
        input_spec={"url": "https://example.gov/source.docx"},
        status="processing",
    )
    session.add(package)
    session.flush()
    binary = b"PK\x03\x04\xff\xfe frozen office original"
    markdown = "# Yerel Markdown\n\nKaynak metin değişmez.".encode()
    prepared = "# Türkçe kaynak\n\nİçerik — hazırlanmış belge metni.".encode()
    assets = []
    for name, raw, mime_type, text in (
        (
            "source.docx",
            binary,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            prepared,
        ),
        ("native.md", markdown, "text/markdown", markdown),
        (
            "missing.docx",
            binary + b" missing",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            None,
        ),
    ):
        raw_id = store.save_file(
            BytesIO(raw), name, FileOrigin.USER_FILE, mime_type, db_session=session
        )
        text_id = (
            store.save_file(
                BytesIO(text),
                name + ".md",
                FileOrigin.OTHER,
                "text/markdown",
                db_session=session,
            )
            if text
            else None
        )
        assets.append(
            RegulatorySourceAsset(
                package_id=package.id,
                sha256=sha256(raw).hexdigest(),
                file_id=raw_id,
                text_file_id=text_id,
                text_sha256=sha256(text).hexdigest() if text else None,
                mime_type=mime_type,
                display_name=name,
                byte_count=len(raw),
            )
        )
    session.add_all(assets)
    session.flush()
    links = [
        SourceLink(
            parent_asset_hash=assets[0].sha256,
            target_asset_hash=asset.sha256,
            parent_url="https://example.gov/source.docx",
            requested_url=f"https://example.gov/{asset.display_name}",
            source_field="href",
            label=asset.display_name,
            kind="url",
        )
        for asset in assets[1:]
    ]
    manifest = json.dumps(
        {
            "status": "ready",
            "assets": [{"sha256": asset.sha256} for asset in assets],
            "links": [link.model_dump(mode="json") for link in links],
            "issues": [],
        }
    ).encode()
    package.manifest_file_id = store.save_file(
        BytesIO(manifest),
        "manifest.json",
        FileOrigin.OTHER,
        "application/json",
        db_session=session,
    )
    package.manifest_sha256 = sha256(manifest).hexdigest()
    package.asset_count, package.total_bytes = (
        len(assets),
        sum(asset.byte_count for asset in assets),
    )
    package.status = "ready"
    batch, _created = intake.create_intake(
        session,
        source_package_id=package.id,
        idempotency_key=uuid4(),
        labeling_model_configuration_id=model.id,
        user=owner,
        environment=_ENVIRONMENT,
    )
    bound_store = cast(
        FileStore,
        SimpleNamespace(
            read_file=lambda identifier: store.read_file(
                identifier, db_session=session
            ),
            save_file=lambda content, **kwargs: store.save_file(
                content, **kwargs, db_session=session
            ),
        ),
    )
    copies = {
        asset.id: clone_intake_original(
            intake_id=batch.id,
            asset=IntakeSourceAsset.model_validate(asset),
            store=bound_store,
        )
        for asset in assets
    }
    members = intake.admit_intake_assets(
        session, batch, assets, links, root_hash=assets[0].sha256, owned_file_ids=copies
    )
    root = next(member for member in members if member.role == "root")
    source = intake_prepared_preview(session, root.user_file_id)
    assert source is not None
    assert source.original_file_id == copies[assets[0].id] != assets[0].file_id
    assert source.raw_sha256 == sha256(binary).hexdigest()
    assert source.text_file_id == assets[0].text_file_id
    assert source.text_sha256 == sha256(prepared).hexdigest()
    with store.read_file(source.original_file_id, db_session=session) as handle:
        assert handle.read() == binary
    with store.read_file(source.text_file_id, db_session=session) as handle:
        assert handle.read().decode() == prepared.decode()
    native = next(
        member for member in members if member.source_asset_id == assets[1].id
    )
    missing = next(
        member for member in members if member.source_asset_id == assets[2].id
    )
    assert intake_prepared_preview(session, native.user_file_id) is None
    with pytest.raises(ValueError, match="verified prepared text"):
        intake_prepared_preview(session, missing.user_file_id)
    other = create_test_user(session, "preview-other-owner", UserRole.ADMIN)
    for mutation in ("owner", "package", "missing-file"):
        savepoint = session.begin_nested()
        try:
            file = session.get_one(UserFile, root.user_file_id)
            if mutation == "owner":
                file.user_id = other.id
            elif mutation == "package":
                package.status = "failed"
            else:
                session.delete(file)
            session.flush()
            with pytest.raises(ValueError, match="source.*(?:binding|unavailable)"):
                intake_prepared_preview(session, root.user_file_id)
        finally:
            savepoint.rollback()
            session.expire_all()
        assert intake_prepared_preview(session, root.user_file_id) == source
    with store.read_file(copies[assets[0].id], db_session=session) as handle:
        assert handle.read() == binary
