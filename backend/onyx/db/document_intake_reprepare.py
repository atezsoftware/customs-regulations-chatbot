"""Explicit rechunking of an unapproved intake, with append-only before/after proof."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import delete, or_, select
from sqlalchemy.orm import Session

from onyx.connectors.models import Document
from onyx.db import document_intake as intake
from onyx.db.engine.sql_engine import get_session_with_tenant
from onyx.db.enums import UserFileStatus
from onyx.db.models import (
    AmendmentSourcePackage,
    DocumentSet,
    DocumentSet__UserFile,
    RegulatoryAnnexElementChunk,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryDocumentIntakeReprepare,
    RegulatoryDocumentLineageReview,
    RegulatoryFilePublication,
    RegulatoryIndexingJob,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryLabelingShard,
    RegulatorySourceAsset,
    RegulatoryTemporalProjection,
    UserFile,
    UserFileProjectionRepair,
)
from onyx.db.regulatory_annex_changes import capture_canonical_scope
from onyx.db.regulatory_canonical_revisions import retain_canonical_revisions
from onyx.db.regulatory_publication import PublicationStore
from onyx.db.regulatory_writer_publication import canonical_scope_digest
from onyx.document_index.publication_models import FileOwnership, publication_digest
from onyx.file_processing.original_ingestion import (
    LoadedUserFileDocuments,
    OriginalIngestionReceipt,
    documents_sha256,
)
from onyx.natural_language_processing.utils import BaseTokenizer
from onyx.regulatory.amendments.annexes.models import AnnexCanonicalSnapshot
from onyx.regulatory.document_intake.metadata import METADATA_AUDIT_KEY


def latest_reprepare(
    session: Session, intake_id: UUID
) -> RegulatoryDocumentIntakeReprepare | None:
    return session.scalar(
        select(RegulatoryDocumentIntakeReprepare)
        .where(RegulatoryDocumentIntakeReprepare.intake_id == intake_id)
        .order_by(
            RegulatoryDocumentIntakeReprepare.created_at.desc(),
            RegulatoryDocumentIntakeReprepare.id.desc(),
        )
        .limit(1)
    )


def _guard_file(
    session: Session,
    batch: RegulatoryDocumentIntake,
    member: RegulatoryDocumentIntakeFile,
    *,
    owned: bool = False,
) -> tuple[UserFile, RegulatoryFilePublication, list[AnnexCanonicalSnapshot]]:
    publication = session.get(
        RegulatoryFilePublication, member.user_file_id, with_for_update=True
    )
    file = session.get_one(UserFile, member.user_file_id, with_for_update=True)
    now = datetime.now(timezone.utc)
    if (
        file.status != UserFileStatus.CHUNKED
        or member.approval_attempt_id is not None
        or not session.get(DocumentSet__UserFile, (batch.document_set_id, file.id))
        or publication is None
        or publication.epoch != 0
        or publication.gate_closed
        or publication.writer_manifest is not None
        or not publication.original_ingestion_receipt
        or (not owned and publication.lease_expires_at > now)
    ):
        raise intake.IntakeConflictError(
            "Only unpublished, idle intake files can be prepared again"
        )
    for model in (
        UserFileProjectionRepair,
        RegulatoryIndexingJob,
        RegulatoryTemporalProjection,
    ):
        if (
            session.scalar(select(model).where(model.user_file_id == file.id).limit(1))
            is not None
        ):
            raise intake.IntakeConflictError(
                "An indexing or publication history prevents reprepare"
            )
    snapshots = capture_canonical_scope(session, file.id)
    if not snapshots:
        raise intake.IntakeConflictError("The original canonical chunks are missing")
    ids = [row.id for row in snapshots]
    if (
        session.scalar(
            select(RegulatoryAnnexElementChunk)
            .where(RegulatoryAnnexElementChunk.chunk_id.in_(ids))
            .limit(1)
        )
        is not None
    ):
        raise intake.IntakeConflictError("Annex publication history prevents reprepare")
    if (
        session.scalar(
            select(RegulatoryChunk)
            .where(
                RegulatoryChunk.user_file_id != file.id,
                or_(
                    RegulatoryChunk.supersedes_chunk_id.in_(ids),
                    RegulatoryChunk.superseded_by_chunk_id.in_(ids),
                ),
            )
            .limit(1)
        )
        is not None
    ):
        raise intake.IntakeConflictError("Another document references these chunks")
    for chunk in snapshots:
        audit = chunk.metadata.get(METADATA_AUDIT_KEY, {})
        fields = audit.get("fields", {}) if isinstance(audit, dict) else {}
        if (
            chunk.source != "indexed"
            or chunk.status != "active"
            or chunk.supersedes_chunk_id
            or chunk.superseded_by_chunk_id
            or "intake_edit_revision" in chunk.metadata
            or any(
                isinstance(value, dict) and value.get("source") == "manual"
                for value in fields.values()
            )
            or any(
                value is not None and fields.get(key, {}).get("source") != "llm"
                for key, value in (
                    ("validity_start_date", chunk.validity_start_date),
                    ("validity_end_date", chunk.validity_end_date),
                )
            )
        ):
            raise intake.IntakeConflictError(
                "Manual edits or legal version history must be preserved; reprepare is unavailable"
            )
    return file, publication, snapshots


def _guard_batch(
    session: Session, batch: RegulatoryDocumentIntake, *, running: bool = False
) -> list[RegulatoryDocumentIntakeFile]:
    session.get_one(DocumentSet, batch.document_set_id, with_for_update=True)
    session.refresh(batch, with_for_update=True)
    if (
        batch.status
        not in ({"preparing"} if running else {"failed", "partial", "ready"})
        or batch.stage == "indexing"
        or (
            not running
            and batch.lease_expires_at
            and batch.lease_expires_at > datetime.now(timezone.utc)
        )
    ):
        raise intake.IntakeConflictError(
            "Only an idle, unapproved ready, failed or incomplete intake can be prepared again"
        )
    members = intake.intake_files(session, batch)
    package = session.get_one(AmendmentSourcePackage, batch.source_package_id)
    if (
        package.status != "ready"
        or package.issues
        or package.document_set_id != batch.document_set_id
        or package.created_by != batch.created_by
    ):
        raise intake.IntakeConflictError("The verified source package scope changed")
    if not members:
        raise intake.IntakeConflictError("No admitted documents exist")
    if (
        session.scalar(
            select(RegulatoryDocumentLineageReview)
            .where(
                RegulatoryDocumentLineageReview.intake_document_id.in_(
                    [m.id for m in members]
                ),
                RegulatoryDocumentLineageReview.status != "draft",
            )
            .limit(1)
        )
        is not None
    ):
        raise intake.IntakeConflictError(
            "An approved document relationship prevents reprepare"
        )
    run = (
        session.get(RegulatoryLabelingRun, batch.labeling_run_id)
        if batch.labeling_run_id
        else None
    )
    from onyx.db.regulatory_labeling import get_active_run_id

    if get_active_run_id(session, batch.document_set_id) is not None:
        raise intake.IntakeConflictError(
            "Wait for active document-set labeling work to finish"
        )
    if run is not None:
        if (
            run.status
            not in {"completed", "completed_with_errors", "failed", "cancelled"}
            or (
                run.lease_token
                and run.lease_expires_at
                and run.lease_expires_at > datetime.now(timezone.utc)
            )
            or session.scalar(
                select(RegulatoryLabelingItem.id)
                .where(
                    RegulatoryLabelingItem.run_id == run.id,
                    RegulatoryLabelingItem.status == "submitted",
                )
                .limit(1)
            )
            or session.scalar(
                select(RegulatoryLabelingShard.id)
                .where(
                    RegulatoryLabelingShard.run_id == run.id,
                    RegulatoryLabelingShard.status.in_(
                        ("submitting", "reconcile_required", "submitted")
                    ),
                )
                .limit(1)
            )
        ):
            raise intake.IntakeConflictError(
                "Wait for current provider labeling work to finish or reconcile"
            )
    if batch.status == "ready" and (
        batch.stage != "review"
        or run is None
        or run.status != "completed"
        or not batch.review_sha256
        or intake.review_digest(session, batch) != batch.review_sha256
    ):
        raise intake.IntakeConflictError(
            "The unapproved ready document review changed; reload before reprepare"
        )
    return members


def reprepare_snapshot(
    session: Session, batch: RegulatoryDocumentIntake
) -> dict[str, Any]:
    members = _guard_batch(session, batch)
    payload = []
    for member in members:
        file, publication, snapshots = _guard_file(session, batch, member)
        asset = session.get_one(RegulatorySourceAsset, member.source_asset_id)
        payload.append(
            dict(
                document_id=str(member.id),
                user_file_id=str(file.id),
                original_file_id=file.file_id,
                source_asset_id=str(asset.id),
                source_sha256=asset.sha256,
                parent_document_id=str(member.parent_document_id)
                if member.parent_document_id
                else None,
                role=member.role,
                relationships=member.relationships,
                source_receipt=publication.original_ingestion_receipt,
                canonical=[row.model_dump(mode="json") for row in snapshots],
            )
        )
    digest = publication_digest(
        dict(
            intake_id=str(batch.id),
            source_package_id=str(batch.source_package_id),
            labeling_run_id=str(batch.labeling_run_id)
            if batch.labeling_run_id
            else None,
            documents=payload,
        )
    )
    return {"expected_sha256": digest, "documents": payload}


def request_reprepare(
    session: Session,
    batch: RegulatoryDocumentIntake,
    *,
    expected_sha256: str,
    idempotency_key: UUID,
) -> RegulatoryDocumentIntakeReprepare:
    session.get_one(DocumentSet, batch.document_set_id, with_for_update=True)
    session.refresh(batch, with_for_update=True)
    existing = session.scalar(
        select(RegulatoryDocumentIntakeReprepare).where(
            RegulatoryDocumentIntakeReprepare.idempotency_key == idempotency_key
        )
    )
    if existing is not None:
        if (
            existing.intake_id != batch.id
            or existing.expected_sha256 != expected_sha256
        ):
            raise intake.IntakeConflictError(
                "Reprepare request identity belongs to another snapshot"
            )
        return existing
    prior = latest_reprepare(session, batch.id)
    if prior is not None and prior.status in {"queued", "running", "failed"}:
        raise intake.IntakeConflictError(
            "Resume the existing reprepare request instead"
        )
    view = reprepare_snapshot(session, batch)
    if view["expected_sha256"] != expected_sha256:
        raise intake.IntakeConflictError(
            "The document snapshot changed; reload before reprepare"
        )
    audit = {}
    for item in view["documents"]:
        revisions = retain_canonical_revisions(
            session, capture_canonical_scope(session, UUID(item["user_file_id"]))
        )
        audit[item["document_id"]] = {
            **item,
            "previous_review_sha256": batch.review_sha256,
            "before_revision_ids": [str(value) for value in revisions.values()],
            "status": "pending",
        }
    request = RegulatoryDocumentIntakeReprepare(
        intake_id=batch.id,
        idempotency_key=idempotency_key,
        expected_sha256=expected_sha256,
        previous_labeling_run_id=batch.labeling_run_id,
        status="queued",
        documents=audit,
    )
    session.add(request)
    batch.labeling_run_id = None
    batch.review_sha256 = None
    batch.status, batch.stage, batch.error = "preparing", "reprepare", None
    batch.lease_token, batch.lease_expires_at = None, None
    session.flush()
    return request


def claim_reprepare(session: Session, intake_id: UUID, request_id: UUID) -> UUID | None:
    batch = session.get_one(RegulatoryDocumentIntake, intake_id, with_for_update=True)
    request = session.get_one(
        RegulatoryDocumentIntakeReprepare, request_id, with_for_update=True
    )
    now = datetime.now(timezone.utc)
    if (
        request.intake_id != batch.id
        or request.status == "prepared"
        or batch.stage != "reprepare"
        or batch.status not in {"preparing", "failed"}
        or (batch.lease_expires_at and batch.lease_expires_at > now)
    ):
        return None
    batch.status, batch.error = "preparing", None
    batch.lease_token, batch.lease_expires_at = uuid4(), now + timedelta(minutes=10)
    request.status, request.error = "running", None
    session.flush()
    return batch.lease_token


def persist_owned_reprepared_intake_file(
    owner: FileOwnership,
    documents: list[Document],
    tokenizer: BaseTokenizer,
    contextual: bool,
    generation_hash: str,
    *,
    intake_id: UUID,
    request_id: UUID,
    lease_token: UUID,
) -> int:
    from onyx.regulatory.indexing import documents_to_regulatory_chunks

    with get_session_with_tenant(tenant_id=owner.scope.tenant_id) as session:
        batch = session.get_one(RegulatoryDocumentIntake, intake_id)
        members = _guard_batch(session, batch, running=True)
        request = session.get_one(
            RegulatoryDocumentIntakeReprepare, request_id, with_for_update=True
        )
        member = next(
            (m for m in members if m.user_file_id == owner.user_file_id), None
        )
        if (
            member is None
            or batch.stage != "reprepare"
            or batch.lease_token != lease_token
            or request.intake_id != batch.id
            or request.status != "running"
        ):
            raise intake.IntakeConflictError("Reprepare ownership changed")
        entry = request.documents.get(str(member.id))
        if entry is None:
            raise intake.IntakeConflictError(
                "Document is outside the requested reprepare scope"
            )
        if (
            entry["user_file_id"] != str(member.user_file_id)
            or entry["source_asset_id"] != str(member.source_asset_id)
            or entry["parent_document_id"]
            != (str(member.parent_document_id) if member.parent_document_id else None)
            or entry["role"] != member.role
            or entry["relationships"] != member.relationships
        ):
            raise intake.IntakeConflictError(
                "The frozen document relationships changed"
            )
        asset = session.get_one(RegulatorySourceAsset, member.source_asset_id)
        if (
            asset.package_id != batch.source_package_id
            or asset.sha256 != entry["source_sha256"]
        ):
            raise intake.IntakeConflictError("The frozen source asset changed")
        if entry["status"] == "prepared":
            return len(capture_canonical_scope(session, member.user_file_id))
        PublicationStore(owner.scope).lock_owned_snapshot(session, owner)
        file, publication, before = _guard_file(session, batch, member, owned=True)
        if (
            publication.original_ingestion_receipt != entry["source_receipt"]
            or file.file_id != entry["original_file_id"]
            or [row.model_dump(mode="json") for row in before] != entry["canonical"]
        ):
            raise intake.IntakeConflictError(
                "The source or canonical snapshot changed during reprepare"
            )
        if not isinstance(documents, LoadedUserFileDocuments):
            raise ValueError("Verified original extraction is required for reprepare")
        proof = documents.original_extraction
        old = OriginalIngestionReceipt.model_validate(entry["source_receipt"])
        if (
            proof.file_id != file.file_id
            or proof.raw_sha256 != old.raw_sha256
            or proof.documents_sha256 != documents_sha256(documents)
            or proof.plaintext_sha256 != old.plaintext_sha256
        ):
            raise ValueError(
                "Reprepare must use the identical original source and text"
            )
        retain_canonical_revisions(session, before)
        session.execute(
            delete(RegulatoryChunk).where(RegulatoryChunk.user_file_id == file.id),
            execution_options={"synchronize_session": "fetch"},
        )
        session.flush()
        chunks = documents_to_regulatory_chunks(
            documents,
            session,
            tokenizer,
            enable_contextual_rag=contextual,
            publication_owner=owner,
        )
        after = capture_canonical_scope(session, file.id)
        if not after:
            raise ValueError("Reprepare produced no canonical chunks")
        revisions = retain_canonical_revisions(session, after)
        receipt = OriginalIngestionReceipt(
            **proof.model_dump(),
            canonical_sha256=canonical_scope_digest(session, file.id),
            generation_hash=generation_hash,
        ).model_dump(mode="json")
        file.regulatory_chunk_generation_hash = generation_hash
        file.chunk_count = len(chunks)
        member.status, member.error = "chunked", None
        entries = deepcopy(request.documents)
        entries[str(member.id)].update(
            status="prepared",
            after_revision_ids=[str(value) for value in revisions.values()],
            after_receipt=receipt,
        )
        request.documents = entries
        batch.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)
        session.commit()
        return len(chunks)


def reprepared_original_receipt(
    session: Session, user_file_id: UUID, original: OriginalIngestionReceipt
) -> OriginalIngestionReceipt:
    """A new canonical receipt never replaces immutable original extraction evidence."""
    member = session.scalar(
        select(RegulatoryDocumentIntakeFile).where(
            RegulatoryDocumentIntakeFile.user_file_id == user_file_id
        )
    )
    if member is None:
        return original
    request = session.scalar(
        select(RegulatoryDocumentIntakeReprepare)
        .where(
            RegulatoryDocumentIntakeReprepare.intake_id == member.intake_id,
            RegulatoryDocumentIntakeReprepare.status == "prepared",
        )
        .order_by(
            RegulatoryDocumentIntakeReprepare.created_at.desc(),
            RegulatoryDocumentIntakeReprepare.id.desc(),
        )
        .limit(1)
    )
    if request is None:
        return original
    entry = request.documents.get(str(member.id), {})
    if entry.get("status") != "prepared" or not entry.get("after_receipt"):
        raise ValueError("Prepared intake receipt is incomplete")
    receipt = OriginalIngestionReceipt.model_validate(entry["after_receipt"])
    if (
        receipt.file_id != original.file_id
        or receipt.raw_sha256 != original.raw_sha256
        or receipt.plaintext_sha256 != original.plaintext_sha256
    ):
        raise ValueError("Prepared intake receipt changed immutable source evidence")
    return receipt


def finish_reprepare(
    session: Session, intake_id: UUID, request_id: UUID, token: UUID
) -> None:
    batch = session.get_one(RegulatoryDocumentIntake, intake_id, with_for_update=True)
    request = session.get_one(
        RegulatoryDocumentIntakeReprepare, request_id, with_for_update=True
    )
    if (
        batch.lease_token != token
        or request.intake_id != batch.id
        or any(v["status"] != "prepared" for v in request.documents.values())
    ):
        raise intake.IntakeConflictError(
            "Every requested document must finish reprepare"
        )
    request.status = "prepared"
    batch.stage = "chunking"
    session.flush()


def fail_reprepare(
    session: Session, intake_id: UUID, request_id: UUID, token: UUID, error: str
) -> None:
    batch = session.get_one(RegulatoryDocumentIntake, intake_id, with_for_update=True)
    if batch.lease_token != token:
        return
    request = session.get_one(
        RegulatoryDocumentIntakeReprepare, request_id, with_for_update=True
    )
    request.status, request.error = "failed", error[:1000]
    intake.fail_intake(session, intake_id, token, error)
