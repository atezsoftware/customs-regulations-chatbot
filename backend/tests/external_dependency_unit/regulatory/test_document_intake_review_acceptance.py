"""Publication review keeps every source while publishing only confirmed documents."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
from typing import Any, Literal
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.configs.constants import FileOrigin
from onyx.db import document_intake as intake
from onyx.db import document_intake_edit as editing
from onyx.db import regulatory_labeling as labeling
from onyx.db.enums import UserFileStatus
from onyx.db.labeling_configuration import LabelingProviderBinding
from onyx.db.models import (
    AmendmentSourcePackage,
    DocumentSet,
    DocumentSet__UserFile,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryFilePublication,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryPublicationOrdinal,
    RegulatorySourceAsset,
    User,
    UserFile,
    UserFileProjectionRepair,
)
from onyx.db.regulatory_writer_publication import canonical_scope_digest
from onyx.file_processing.original_ingestion import OriginalIngestionReceipt
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.regulatory.amendments.annexes.models import SourceLink
from onyx.regulatory.indexing_jobs.models import VertexAuthenticationMode
from tests.external_dependency_unit.regulatory.test_document_intake import (
    _completed_labels,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_owner as intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_session as intake_session,
)


def _table_rows(session: Session, *models: Any) -> str:
    return json.dumps(
        {
            model.__tablename__: sorted(
                (
                    dict(row)
                    for row in session.execute(select(model.__table__)).mappings()
                ),
                key=str,
            )
            for model in models
        },
        sort_keys=True,
        default=str,
    )


def _three_document_review(
    session: Session,
    owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    *,
    failed_alias: bool = False,
    standalone_annex: bool = False,
    manual_date: bool = False,
    annex_via_alias: bool = False,
) -> tuple[
    RegulatoryDocumentIntake,
    dict[str, RegulatoryDocumentIntakeFile],
    RegulatoryLabelingRun,
    LabelingProviderBinding,
]:
    user, target, model = owner
    store = PostgresBackedFileStore()
    package = AmendmentSourcePackage(
        document_set_id=target.id,
        created_by=user.id,
        environment="document-intake-regression",
        idempotency_key=str(uuid4()),
        request_hash=uuid4().hex * 2,
        input_spec={"url": "https://example.gov/20250105-3.htm"},
        status="processing",
    )
    session.add(package)
    session.flush()
    assets: dict[str, RegulatorySourceAsset] = {}
    for key, name, mime, raw, text in (
        (
            "root",
            "20250105-3.htm.pdf",
            "application/pdf",
            b"%PDF-1.7\nretained-root",
            "SGM-2024/5. MADDE 1 — Başvuruda EK-1 belgeleri sunulur.",
        ),
        (
            "annex",
            "20250105-3-1.pdf",
            "application/pdf",
            b"%PDF-1.7\nretained-annex",
            "EK-1 — Başvuruda sunulacak belgeler ve teknik koşullar.",
        ),
        (
            "alias",
            "20250105-3.htm",
            "text/html",
            b"<html>same-law-retained</html>",
            "SGM-2024/5. MADDE 1 — Başvuruda EK-1 belgeleri sunulur.",
        ),
    ):
        if standalone_annex and key != "annex":
            continue
        raw_id = store.save_file(
            BytesIO(raw), name, FileOrigin.USER_FILE, mime, db_session=session
        )
        text_bytes = text.encode()
        text_id = store.save_file(
            BytesIO(text_bytes),
            name + ".txt",
            FileOrigin.OTHER,
            "text/plain",
            db_session=session,
        )
        assets[key] = RegulatorySourceAsset(
            package_id=package.id,
            sha256=sha256(raw).hexdigest(),
            file_id=raw_id,
            text_file_id=text_id,
            text_sha256=sha256(text_bytes).hexdigest(),
            mime_type=mime,
            display_name=name,
            byte_count=len(raw),
            original_url="https://example.gov/" + name,
            final_url="https://example.gov/" + name,
        )
    if standalone_annex:
        assets = {"annex": assets["annex"]}
    session.add_all(assets.values())
    session.flush()
    links = [
        SourceLink(
            parent_asset_hash=assets[
                "alias" if annex_via_alias and key == "annex" else "root"
            ].sha256,
            target_asset_hash=assets[key].sha256,
            parent_url=assets[
                "alias" if annex_via_alias and key == "annex" else "root"
            ].original_url,
            requested_url=assets[key].original_url,
            source_field="href",
            label="Ekleri için tıklayınız"
            if key == "annex"
            else "Resmî Gazete asıl metni",
            kind="url",
        )
        for key in (() if standalone_annex else ("annex", "alias"))
    ]
    manifest = json.dumps(
        {
            "status": "ready",
            "assets": [{"sha256": item.sha256} for item in assets.values()],
            "links": [link.model_dump(mode="json") for link in links],
            "issues": [],
        },
    ).encode()
    package.manifest_file_id = store.save_file(
        BytesIO(manifest),
        "source-manifest.json",
        FileOrigin.OTHER,
        "application/json",
        db_session=session,
    )
    package.manifest_sha256 = sha256(manifest).hexdigest()
    package.asset_count, package.total_bytes, package.status = (
        len(assets),
        sum(a.byte_count for a in assets.values()),
        "ready",
    )
    session.flush()
    batch, _created = intake.create_intake(
        session,
        source_package_id=package.id,
        idempotency_key=uuid4(),
        labeling_model_configuration_id=model.id,
        user=user,
        environment=package.environment,
    )
    members = intake.admit_intake_assets(
        session,
        batch,
        list(assets.values()),
        links,
        root_hash=assets["annex" if standalone_annex else "root"].sha256,
    )
    by_asset = {member.source_asset_id: member for member in members}
    named = {key: by_asset[asset.id] for key, asset in assets.items()}
    for key, member in named.items():
        asset = assets[key]
        assert asset.text_file_id
        with store.read_file(asset.text_file_id, db_session=session) as stream:
            text = stream.read().decode()
        chunk = RegulatoryChunk(
            id=f"rc_review_{uuid4().hex}",
            user_file_id=member.user_file_id,
            text=text,
            position=0,
            projection_ordinal=0,
            heading_path=[key],
            chunk_metadata={"source_asset_id": str(asset.id)},
            chunk_type="article",
            status="active",
            source="indexed",
        )
        session.add(chunk)
        session.flush()
        file = session.get_one(UserFile, member.user_file_id)
        file.regulatory_chunk_generation_hash = "a" * 64
        receipt = OriginalIngestionReceipt(
            file_id=file.file_id,
            raw_sha256=asset.sha256,
            documents_sha256="b" * 64,
            plaintext_sha256=asset.text_sha256 or "",
            canonical_sha256=canonical_scope_digest(session, file.id),
            generation_hash="a" * 64,
        )
        session.add(
            RegulatoryFilePublication(
                user_file_id=file.id,
                scope_key="review-acceptance",
                owner_id=uuid4(),
                fencing_token=1,
                lease_expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
                epoch=0,
                gate_closed=False,
                next_ordinal=1,
                original_ingestion_receipt=receipt.model_dump(mode="json"),
            )
        )
        session.add(
            RegulatoryPublicationOrdinal(
                user_file_id=file.id,
                allocation_key=f"canonical:{chunk.id}",
                ordinal=0,
            )
        )
        if manual_date and key == "root":
            chunk.validity_start_date = date(2025, 1, 5)
    seed = _completed_labels(session, batch, members)
    binding = LabelingProviderBinding(
        model_configuration_id=model.id,
        provider_id=model.llm_provider.id,
        project="review-test",
        location="global",
        authentication_mode=VertexAuthenticationMode.WORKLOAD_IDENTITY,
        credential_identity="fixture",
        model="gemini-3.5-flash",
        transport="vertex_direct_v1",
    )
    monkeypatch.setattr(
        editing, "resolve_intake_labeling_provider_binding", lambda *_a, **_k: binding
    )
    monkeypatch.setattr(
        intake, "resolve_intake_labeling_provider_binding", lambda *_a, **_k: binding
    )
    from onyx.db import document_intake_review as review

    def retained_digest(pointer: str) -> str:
        with store.read_file(pointer, db_session=session) as stream:
            return sha256(stream.read()).hexdigest()

    monkeypatch.setattr(review, "_blob_sha256", retained_digest)
    run, _created = labeling.create_labeling_run(
        session,
        document_set_id=target.id,
        taxonomy=seed.taxonomy,
        model_configuration_id=model.id,
        model=binding.model,
        provider_binding=binding.model_dump(mode="json"),
        requested_by_id=user.id,
        idempotency_key=uuid4(),
        target_file_ids=[member.user_file_id for member in members],
    )
    lease = labeling.claim_run(
        session, run_id=run.id, expected_generation=0, lease_seconds=300
    )
    assert lease
    for item in labeling.prepare_next_item_page(session, lease, limit=128):
        item.status, item.labels = "completed", ["Gümrük"]
        assignment: dict[str, object] = {
            "label": "Gümrük",
            "evidence": item.text_snapshot,
        }
        request: dict[str, object] = {"fixture": True}
        item.assignments = [assignment]
        item.request_hash, item.request_payload = uuid4().hex * 2, request
        if failed_alias and item.user_file_id == named["alias"].user_file_id:
            item.status, item.error, item.labels, item.assignments = (
                "failed",
                "invalid_evidence",
                [],
                [],
            )
    run.status, run.stage = (
        ("completed_with_errors" if failed_alias else "completed"),
        "finished",
    )
    run.completed_chunks, run.failed_chunks = (
        (len(members) - 1, 1) if failed_alias else (len(members), 0)
    )
    run.lease_token, run.lease_expires_at = None, None
    batch.labeling_run_id, batch.status, batch.stage = run.id, "preparing", "labeling"
    session.flush()
    intake.refresh_intake(session, batch)
    assert batch.status == ("partial" if failed_alias else "ready")
    return batch, named, run, binding


def test_legacy_review_digest_has_no_empty_publication_decision_fields(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.document_index.publication_models import publication_digest

    batch, _members, run, _binding = _three_document_review(
        intake_session, intake_owner, monkeypatch
    )
    members = intake.intake_files(intake_session, batch)
    chunks = list(
        intake_session.scalars(
            select(RegulatoryChunk)
            .where(
                RegulatoryChunk.user_file_id.in_(
                    [member.user_file_id for member in members]
                ),
            )
            .order_by(RegulatoryChunk.id)
        )
    )
    labels = {
        item.regulatory_chunk_id: item.labels
        for item in intake_session.scalars(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == run.id
            ),
        )
    }
    previous_payload = {
        "source_package_id": str(batch.source_package_id),
        "files": [
            {
                "id": str(item.user_file_id),
                "asset": str(item.source_asset_id),
                "role": item.role,
                "parent_document_id": str(item.parent_document_id)
                if item.parent_document_id
                else None,
                "relationships": item.relationships,
            }
            for item in members
        ],
        "chunks": [
            {
                "id": row.id,
                "file_id": str(row.user_file_id),
                "text": row.text,
                "chunk_type": row.chunk_type,
                "position": row.position,
                "metadata": row.chunk_metadata,
                "headings": list(row.heading_path),
                "status": row.status,
                "start": str(row.validity_start_date),
                "end": str(row.validity_end_date),
                "labels": labels.get(row.id, []),
            }
            for row in chunks
        ],
    }
    assert intake.review_digest(intake_session, batch) == publication_digest(
        previous_payload
    )


def _decision(
    session: Session,
    batch: RegulatoryDocumentIntake,
    member: RegulatoryDocumentIntakeFile,
    user: User,
    *,
    disposition: Literal["legislation", "source_evidence"] = "legislation",
    duplicate: UUID | None = None,
    existing_parent: UUID | None = None,
    expected: str | None = None,
) -> UUID | None:
    from onyx.db import document_intake_review as review

    digest = expected or review.publication_review_digest(session, batch)
    assert digest
    return review.set_publication_review(
        session,
        batch,
        document_id=member.id,
        expected_review_sha256=digest,
        disposition=disposition,
        duplicate_of_document_id=duplicate,
        existing_parent_file_id=existing_parent,
        reason="Asıl PDF ile aynı hukuki metnin HTML kaynak gösterimidir."
        if duplicate
        else "Resmî kaynakta bu ana belgeye ait ayrı ek olarak gösterilmiştir.",
        user=user,
    )


def _finish_reused_run(
    session: Session, batch: RegulatoryDocumentIntake
) -> RegulatoryLabelingRun:
    run = session.get_one(RegulatoryLabelingRun, batch.labeling_run_id)
    items = list(
        session.scalars(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == run.id
            )
        )
    )
    assert items and all(item.status == "completed" for item in items)
    run.status, run.stage, run.completed_chunks = "completed", "finished", len(items)
    run.lease_token, run.lease_expires_at = None, None
    session.flush()
    intake.refresh_intake(session, batch)
    assert batch.status == "ready" and batch.review_sha256
    return run


def test_failed_html_alias_retains_all_evidence_reuses_two_labels_and_never_indexes(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.db import document_intake_review as review
    from onyx.db.user_file import claim_user_file_projection_repair

    session = intake_session
    user, target, _model = intake_owner
    batch, members, previous, _binding = _three_document_review(
        session, intake_owner, monkeypatch, failed_alias=True
    )
    immutable = _table_rows(
        session,
        AmendmentSourcePackage,
        RegulatorySourceAsset,
        RegulatoryChunk,
        RegulatoryFilePublication,
    )
    graph = {
        member.id: (member.role, member.parent_document_id, list(member.relationships))
        for member in members.values()
    }
    old_results = {
        item.regulatory_chunk_id: (
            item.request_hash,
            item.assignments,
            item.source_snapshot,
        )
        for item in session.scalars(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == previous.id,
                RegulatoryLabelingItem.status == "completed",
            )
        )
    }
    accepted_receipt = review.publication_review_digest(session, batch)
    assert accepted_receipt
    replacement_id = _decision(
        session,
        batch,
        members["alias"],
        user,
        disposition="source_evidence",
        duplicate=members["root"].id,
        expected=accepted_receipt,
    )
    assert replacement_id and replacement_id != previous.id
    decision = review.latest_publication_decisions(session, batch.id)[
        members["alias"].id
    ]
    assert decision.source_snapshot["previous_labeling_run_id"] == str(previous.id)
    assert (
        decision.source_snapshot["accepted_publication_review_sha256"]
        == accepted_receipt
    )
    replacement = _finish_reused_run(session, batch)
    selected = {members[key].user_file_id for key in ("root", "annex")}
    assert {
        item.user_file_id for item in review.effective_intake_files(session, batch)
    } == selected
    assert set(replacement.file_ids) == {str(identifier) for identifier in selected}
    for item in session.scalars(
        select(RegulatoryLabelingItem).where(
            RegulatoryLabelingItem.run_id == replacement.id
        )
    ):
        assert (
            item.request_hash,
            item.assignments,
            item.source_snapshot,
        ) == old_results[item.regulatory_chunk_id]
    assert (
        _table_rows(
            session,
            AmendmentSourcePackage,
            RegulatorySourceAsset,
            RegulatoryChunk,
            RegulatoryFilePublication,
        )
        == immutable
    )
    reviewed = batch.review_sha256
    assert reviewed
    assert {
        file_id for file_id, _token in intake.approve_intake(session, batch, reviewed)
    } == selected
    assert intake.review_digest(session, batch) == reviewed
    alias_id = members["alias"].user_file_id
    assert session.get(DocumentSet__UserFile, (target.id, alias_id)) is None
    assert all(
        session.get(DocumentSet__UserFile, (target.id, identifier))
        for identifier in selected
    )
    assert (
        session.get(UserFile, alias_id)
        and session.get(UserFileProjectionRepair, alias_id) is None
    )
    for identifier in selected:
        session.get_one(UserFile, identifier).status = UserFileStatus.COMPLETED
    session.flush()
    intake.refresh_intake(session, batch)
    assert batch.status == "completed"
    assert members["alias"].status == "source_evidence"
    assert (
        _table_rows(
            session,
            AmendmentSourcePackage,
            RegulatorySourceAsset,
            RegulatoryChunk,
            RegulatoryFilePublication,
        )
        == immutable
    )
    assert {
        member.id: (member.role, member.parent_document_id, list(member.relationships))
        for member in members.values()
    } == graph
    for include_approving in (False, True):
        assert intake.pending_intake_file_ids(
            session, [alias_id, *selected], include_approving=include_approving
        ) == {alias_id}
    assert (
        claim_user_file_projection_repair(session, alias_id, initial_index=True) is None
    )
    assert (
        claim_user_file_projection_repair(session, alias_id, initial_index=False)
        is None
    )
    assert session.get(UserFileProjectionRepair, alias_id) is None
    from onyx.db.document_set import link_user_file_to_document_set

    try:
        relinked = link_user_file_to_document_set(
            session, target.id, session.get_one(UserFile, alias_id)
        )
    except ValueError:
        relinked = False
    assert not relinked
    assert session.get(DocumentSet__UserFile, (target.id, alias_id)) is None


def test_alias_review_is_reversible_with_append_only_decisions_before_approval(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlalchemy.exc import DBAPIError

    from onyx.db import document_intake_review as review
    from onyx.db.models import RegulatoryDocumentIntakeReviewDecision

    session = intake_session
    user, target, _model = intake_owner
    batch, members, _previous, _binding = _three_document_review(
        session, intake_owner, monkeypatch
    )
    source = _table_rows(
        session,
        AmendmentSourcePackage,
        RegulatorySourceAsset,
        RegulatoryChunk,
        UserFile,
    )
    _decision(
        session,
        batch,
        members["alias"],
        user,
        disposition="source_evidence",
        duplicate=members["root"].id,
    )
    _finish_reused_run(session, batch)
    first = review.latest_publication_decisions(session, batch.id)[members["alias"].id]
    frozen = _table_rows(session, RegulatoryDocumentIntakeReviewDecision)
    _decision(session, batch, members["alias"], user)
    latest = review.latest_publication_decisions(session, batch.id)[members["alias"].id]
    assert latest.id != first.id
    assert latest.previous_decision_id == first.id
    assert (
        latest.disposition == "legislation" and latest.duplicate_of_document_id is None
    )
    assert len(review.effective_intake_files(session, batch)) == 3
    assert all(
        session.get(DocumentSet__UserFile, (target.id, member.user_file_id))
        for member in members.values()
    )
    assert (
        _table_rows(
            session,
            AmendmentSourcePackage,
            RegulatorySourceAsset,
            RegulatoryChunk,
            UserFile,
        )
        == source
    )
    assert frozen != _table_rows(session, RegulatoryDocumentIntakeReviewDecision)
    with pytest.raises(DBAPIError), session.begin_nested():
        session.delete(first)
        session.flush()
    with pytest.raises(DBAPIError), session.begin_nested():
        first.reason = "Altered historical decision"
        session.flush()


def test_selected_two_initial_index_failures_retry_without_touching_alias_or_labels(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.db.models import RegulatoryDocumentIntakeReviewDecision
    from onyx.db.user_file import (
        finish_user_file_index_request,
        start_user_file_index_request,
    )

    session = intake_session
    user, _target, _model = intake_owner
    batch, members, _previous, _binding = _three_document_review(
        session, intake_owner, monkeypatch, failed_alias=True
    )
    _decision(
        session,
        batch,
        members["alias"],
        user,
        disposition="source_evidence",
        duplicate=members["root"].id,
    )
    _finish_reused_run(session, batch)
    reviewed = batch.review_sha256
    assert reviewed
    old_tokens = dict(intake.approve_intake(session, batch, reviewed))
    assert len(old_tokens) == 2
    session.commit()
    immutable = _table_rows(
        session,
        RegulatoryDocumentIntakeReviewDecision,
        RegulatoryChunk,
        RegulatoryLabelingRun,
        RegulatoryLabelingItem,
        RegulatorySourceAsset,
        RegulatoryFilePublication,
    )
    for identifier, token in old_tokens.items():
        assert start_user_file_index_request(session, identifier, token)
        assert finish_user_file_index_request(
            session, identifier, token, failure_code="context:test"
        )
    intake.refresh_intake(session, batch)
    assert batch.status == "failed" and batch.stage == "indexing"
    fresh = dict(intake.retry_intake(session, batch))
    assert fresh.keys() == old_tokens.keys()
    assert all(fresh[identifier] != old_tokens[identifier] for identifier in fresh)
    assert members["alias"].approval_attempt_id is None
    assert session.get(UserFileProjectionRepair, members["alias"].user_file_id) is None
    assert batch.review_sha256 == reviewed
    assert (
        _table_rows(
            session,
            RegulatoryDocumentIntakeReviewDecision,
            RegulatoryChunk,
            RegulatoryLabelingRun,
            RegulatoryLabelingItem,
            RegulatorySourceAsset,
            RegulatoryFilePublication,
        )
        == immutable
    )


def test_selected_manual_date_original_receipt_remains_available_but_unreviewed_change_fails(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.db.regulatory_original_ingestion import original_attachment_receipt

    session = intake_session
    user, _target, _model = intake_owner
    batch, members, _run, _binding = _three_document_review(
        session, intake_owner, monkeypatch, manual_date=True
    )
    _decision(
        session,
        batch,
        members["alias"],
        user,
        disposition="source_evidence",
        duplicate=members["root"].id,
    )
    _finish_reused_run(session, batch)
    reviewed = batch.review_sha256
    assert reviewed
    approved = dict(intake.approve_intake(session, batch, reviewed))
    for identifier in approved:
        session.get_one(UserFile, identifier).status = UserFileStatus.COMPLETED
        session.get_one(RegulatoryFilePublication, identifier).epoch = 2
    session.flush()
    intake.refresh_intake(session, batch)
    assert batch.status == "completed"
    file_id = members["root"].user_file_id
    publication = session.get_one(RegulatoryFilePublication, file_id)
    immutable = json.dumps(publication.original_ingestion_receipt, sort_keys=True)
    receipt = original_attachment_receipt(session, file_id)
    assert receipt and receipt.canonical_sha256 == canonical_scope_digest(
        session, file_id
    )
    assert receipt.file_id == session.get_one(UserFile, file_id).file_id
    assert (
        json.dumps(publication.original_ingestion_receipt, sort_keys=True) == immutable
    )
    chunk = session.scalar(
        select(RegulatoryChunk).where(RegulatoryChunk.user_file_id == file_id)
    )
    assert chunk
    chunk.text += " Onaydan sonra kaydedilmemiş değişiklik."
    session.flush()
    assert original_attachment_receipt(session, file_id) is None
    assert (
        json.dumps(publication.original_ingestion_receipt, sort_keys=True) == immutable
    )


def test_alias_effective_parent_mapping_retains_original_graph_and_links_to_publication_parent(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = intake_session
    user, _target, _model = intake_owner
    batch, members, _run, _binding = _three_document_review(
        session, intake_owner, monkeypatch, annex_via_alias=True
    )
    assert members["annex"].parent_document_id == members["alias"].id
    graph = {
        member.id: (member.parent_document_id, list(member.relationships))
        for member in members.values()
    }
    _decision(
        session,
        batch,
        members["alias"],
        user,
        disposition="source_evidence",
        duplicate=members["root"].id,
    )
    _finish_reused_run(session, batch)
    snapshot = intake.intake_snapshot(session, batch)
    annex = next(
        document
        for document in snapshot["documents"]
        if document["id"] == str(members["annex"].id)
    )
    assert annex["parent_document_id"] == str(members["root"].id)
    assert {
        member.id: (member.parent_document_id, list(member.relationships))
        for member in members.values()
    } == graph


@pytest.mark.parametrize("descendant", ["alias", "annex"])
def test_alias_to_its_descendant_cannot_create_effective_parent_cycle(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    descendant: str,
) -> None:
    from onyx.db.models import RegulatoryDocumentIntakeReviewDecision

    session = intake_session
    user, _target, _model = intake_owner
    batch, members, _run, _binding = _three_document_review(
        session, intake_owner, monkeypatch
    )
    before = _table_rows(
        session,
        RegulatoryDocumentIntakeReviewDecision,
        RegulatoryDocumentIntakeFile,
        RegulatoryLabelingRun,
        RegulatoryChunk,
    )
    with pytest.raises((ValueError, intake.IntakeConflictError)):
        _decision(
            session,
            batch,
            members["root"],
            user,
            disposition="source_evidence",
            duplicate=members[descendant].id,
        )
    assert (
        _table_rows(
            session,
            RegulatoryDocumentIntakeReviewDecision,
            RegulatoryDocumentIntakeFile,
            RegulatoryLabelingRun,
            RegulatoryChunk,
        )
        == before
    )


@pytest.mark.parametrize(
    "invalid",
    [
        "stale",
        "self",
        "foreign_document",
        "foreign_target",
        "active_lease",
        "published",
        "source_drift",
    ],
)
def test_alias_invalid_scope_or_changed_review_leaves_no_decision_or_replacement_run(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    invalid: str,
) -> None:
    from onyx.db import document_intake_review as review
    from onyx.db.models import RegulatoryDocumentIntakeReviewDecision

    session = intake_session
    user, _target, _model = intake_owner
    batch, members, run, _binding = _three_document_review(
        session, intake_owner, monkeypatch
    )
    expected = review.publication_review_digest(session, batch)
    assert expected
    member = members["alias"]
    target_id = members["root"].id
    if invalid == "stale":
        expected = "0" * 64
    elif invalid == "self":
        target_id = member.id
    elif invalid in {"foreign_document", "foreign_target"}:
        _other, foreign, _run, _binding = _three_document_review(
            session, intake_owner, monkeypatch
        )
        if invalid == "foreign_document":
            member = foreign["alias"]
        else:
            target_id = foreign["root"].id
    elif invalid == "active_lease":
        run.lease_token, run.lease_expires_at = (
            uuid4(),
            datetime.now(timezone.utc) + timedelta(minutes=5),
        )
    elif invalid == "published":
        session.get_one(RegulatoryFilePublication, member.user_file_id).epoch = 1
    else:
        member.relationships = [
            {**member.relationships[0], "label": "changed frozen source label"}
        ]
    session.flush()
    before = _table_rows(
        session,
        RegulatoryDocumentIntakeReviewDecision,
        RegulatoryLabelingRun,
        RegulatoryChunk,
        UserFile,
        DocumentSet__UserFile,
    )
    with pytest.raises((ValueError, intake.IntakeConflictError)):
        _decision(
            session,
            batch,
            member,
            user,
            disposition="source_evidence",
            duplicate=target_id,
            expected=expected,
        )
    assert (
        _table_rows(
            session,
            RegulatoryDocumentIntakeReviewDecision,
            RegulatoryLabelingRun,
            RegulatoryChunk,
            UserFile,
            DocumentSet__UserFile,
        )
        == before
    )
