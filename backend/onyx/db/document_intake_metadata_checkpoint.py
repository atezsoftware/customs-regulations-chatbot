"""Short metadata lease transactions and an execution mutex across provider calls."""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db.engine.sql_engine import get_sqlalchemy_engine
from onyx.db.enums import UserFileStatus
from onyx.db.labeling_configuration import LabelingProviderBinding
from onyx.db.models import (
    AmendmentSourcePackage,
    DocumentSet__UserFile,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryFilePublication,
    RegulatoryIndexingJob,
    RegulatorySourceAsset,
    RegulatoryTemporalProjection,
    UserFile,
    UserFileProjectionRepair,
)
from onyx.file_processing.original_ingestion import OriginalIngestionReceipt
from onyx.regulatory.document_intake.metadata_checkpoint import (
    MetadataCheckpoint,
    MetadataLeaseLost,
    metadata_digest,
)
from onyx.regulatory.document_intake.metadata_review import ReviewedMetadataEdit
from shared_configs.contextvars import get_current_tenant_id


def _execution_key(intake_id: UUID) -> int:
    return int.from_bytes(
        sha256(
            f"intake-metadata:{get_current_tenant_id()}:{intake_id}".encode()
        ).digest()[:8],
        byteorder="big",
        signed=True,
    )


def try_reserve_metadata_execution(session: Session, intake_id: UUID) -> bool:
    return bool(
        session.scalar(
            select(func.pg_try_advisory_xact_lock(_execution_key(intake_id)))
        )
    )


@contextmanager
def metadata_execution_lock(intake_id: UUID) -> Iterator[bool]:
    # AUTOCOMMIT keeps the session lock without retaining a transaction snapshot.
    key = _execution_key(intake_id)
    with (
        get_sqlalchemy_engine()
        .connect()
        .execution_options(isolation_level="AUTOCOMMIT") as connection
    ):
        acquired = bool(connection.scalar(select(func.pg_try_advisory_lock(key))))
        try:
            yield acquired
        finally:
            if acquired:
                try:
                    connection.execute(select(func.pg_advisory_unlock(key)))
                except Exception:
                    connection.invalidate()


def require_metadata_lease(
    session: Session,
    intake_id: UUID,
    token: UUID,
    *,
    allow_preparation_stage: bool = False,
) -> RegulatoryDocumentIntake:
    batch = session.get_one(
        RegulatoryDocumentIntake,
        intake_id,
        with_for_update=True,
        populate_existing=True,
    )
    if (
        batch.status != "preparing"
        or (
            batch.stage != "metadata"
            and not (
                allow_preparation_stage and batch.stage in {"chunking", "reprepare"}
            )
        )
        or batch.lease_token != token
        or batch.lease_expires_at is None
        or batch.lease_expires_at <= datetime.now(timezone.utc)
    ):
        raise MetadataLeaseLost("The metadata preparation lease expired or changed")
    return batch


def _source_snapshot(
    session: Session,
    batch: RegulatoryDocumentIntake,
    *,
    reviewed: ReviewedMetadataEdit | None = None,
    for_review: bool = False,
) -> tuple[dict[str, Any], dict[str, list[str]], set[str]]:
    from onyx.db.document_intake_metadata import (
        metadata_chunks,
        metadata_receipts_current,
        validate_metadata_receipt_evidence,
    )
    from onyx.db.document_intake_metadata_review import (
        canonical_metadata_source_digest,
        require_reviewed_document,
        require_reviewed_metadata_owner,
        reviewed_metadata_edit,
    )
    from onyx.db.document_intake_reprepare import reprepared_original_receipt
    from onyx.regulatory.document_intake.metadata import (
        TARGETS_PER_REQUEST,
        build_metadata_request,
        metadata_input_hash,
    )

    package = session.get_one(
        AmendmentSourcePackage, batch.source_package_id, populate_existing=True
    )
    members = list(
        session.scalars(
            select(RegulatoryDocumentIntakeFile)
            .where(RegulatoryDocumentIntakeFile.intake_id == batch.id)
            .order_by(RegulatoryDocumentIntakeFile.id)
            .execution_options(populate_existing=True)
        )
    )
    reviewed = reviewed or reviewed_metadata_edit(batch)
    if (
        package.status != "ready"
        or package.issues
        or package.created_by != batch.created_by
        or package.document_set_id != batch.document_set_id
        or not package.manifest_sha256
        or not members
        or len(members) != package.asset_count
        or (batch.labeling_run_id is not None and reviewed is None)
    ):
        raise intake.IntakeConflictError("The frozen metadata source package changed")
    if reviewed is not None:
        if batch.labeling_run_id is not None:
            from onyx.db.document_intake_review import effective_intake_files

            members = effective_intake_files(session, batch)
        if not set(reviewed.documents) <= {member.id for member in members}:
            raise intake.IntakeConflictError(
                "The reviewed metadata document scope changed"
            )
        require_reviewed_metadata_owner(
            session, batch, reviewed, {str(member.user_file_id) for member in members}
        )
    documents, requests, completed = {}, {}, set()
    for member in members:
        file = session.get_one(UserFile, member.user_file_id, populate_existing=True)
        publication = session.get(
            RegulatoryFilePublication, file.id, populate_existing=True
        )
        asset = session.get_one(
            RegulatorySourceAsset, member.source_asset_id, populate_existing=True
        )
        if (
            member.status != "chunked"
            or member.approval_attempt_id is not None
            or file.status != UserFileStatus.CHUNKED
            or file.user_id != batch.created_by
            or session.get(
                DocumentSet__UserFile,
                (batch.document_set_id, file.id),
                populate_existing=True,
            )
            is None
            or asset.package_id != package.id
            or publication is None
            or publication.epoch != 0
            or publication.gate_closed
            or publication.writer_manifest is not None
            or publication.lease_expires_at > datetime.now(timezone.utc)
            or not publication.original_ingestion_receipt
        ):
            raise intake.IntakeConflictError(
                "Metadata recovery requires idle unpublished owned sources"
            )
        original = OriginalIngestionReceipt.model_validate(
            publication.original_ingestion_receipt
        )
        effective = reprepared_original_receipt(session, file.id, original)
        if (
            original.file_id != file.file_id
            or original.raw_sha256 != asset.sha256
            or effective.generation_hash != file.regulatory_chunk_generation_hash
        ):
            raise intake.IntakeConflictError(
                "The original metadata source receipt changed"
            )
        for model in (
            RegulatoryIndexingJob,
            RegulatoryTemporalProjection,
            UserFileProjectionRepair,
        ):
            if (
                session.scalar(
                    select(model).where(model.user_file_id == file.id).limit(1)
                )
                is not None
            ):
                raise intake.IntakeConflictError(
                    "Publication history prevents metadata recovery"
                )
        rows = list(
            session.scalars(
                select(RegulatoryChunk)
                .where(RegulatoryChunk.user_file_id == file.id)
                .order_by(RegulatoryChunk.position, RegulatoryChunk.id)
                .execution_options(populate_existing=True)
            )
        )
        if not rows or any(
            row.status != "active"
            or row.source != "indexed"
            or row.supersedes_chunk_id
            or row.superseded_by_chunk_id
            or ("intake_edit_revision" in row.chunk_metadata and reviewed is None)
            for row in rows
        ):
            raise intake.IntakeConflictError(
                "The original metadata canonical source changed"
            )
        chunks = metadata_chunks(rows)
        input_hash = metadata_input_hash(chunks)
        if reviewed is not None and member.id in reviewed.documents:
            require_reviewed_document(reviewed.documents[member.id], file.id, rows)
        if not chunks:
            raise intake.IntakeConflictError(
                "Metadata requires canonical source targets"
            )
        if metadata_receipts_current(rows, input_hash):
            if not for_review:
                validate_metadata_receipt_evidence(rows, chunks)
            completed.add(str(member.id))
        elif (
            reviewed is not None
            and batch.labeling_run_id is not None
            and (
                member.id not in reviewed.documents
                or not reviewed.documents[member.id].refresh_metadata
            )
        ):
            raise intake.IntakeConflictError(
                "Unreviewed post-label metadata cannot be regenerated"
            )
        requests[str(member.id)] = (
            []
            if for_review or str(member.id) in completed
            else [
                build_metadata_request(
                    chunks,
                    [
                        chunk.chunk_id
                        for chunk in chunks[offset : offset + TARGETS_PER_REQUEST]
                    ],
                ).request_hash
                for offset in range(0, len(chunks), TARGETS_PER_REQUEST)
            ]
        )
        documents[str(member.id)] = {
            "user_file_id": str(file.id),
            "source_asset_id": str(asset.id),
            "source_file_id": asset.file_id,
            "text_file_id": asset.text_file_id,
            "text_sha256": asset.text_sha256,
            "original_url": asset.original_url,
            "final_url": asset.final_url,
            "raw_sha256": asset.sha256,
            "original_file_id": file.file_id,
            "generation_hash": effective.generation_hash,
            "original_receipt_sha256": metadata_digest(
                original.model_dump(mode="json")
            ),
            "input_sha256": input_hash,
            "role": member.role,
            "parent_document_id": str(member.parent_document_id)
            if member.parent_document_id
            else None,
            "relationships": member.relationships,
            "canonical_source_sha256": canonical_metadata_source_digest(rows),
        }
    return (
        {
            "tenant_id": get_current_tenant_id(),
            "intake_id": str(batch.id),
            "package_id": str(package.id),
            "environment": package.environment,
            "document_set_id": batch.document_set_id,
            "created_by": str(batch.created_by),
            "manifest_sha256": package.manifest_sha256,
            "manifest_file_id": package.manifest_file_id,
            "documents": documents,
            **({"reviewed_edit": reviewed.model_dump(mode="json")} if reviewed else {}),
        },
        requests,
        completed,
    )


def _validated_checkpoint(
    session: Session,
    batch: RegulatoryDocumentIntake,
    binding: LabelingProviderBinding,
    recipe_sha256: str,
    *,
    capture: bool = False,
    rebase_recipe: bool = False,
) -> MetadataCheckpoint:
    try:
        source, requests, completed = _source_snapshot(session, batch)
        current = MetadataCheckpoint.build(
            recipe_sha256=recipe_sha256,
            provider_binding=binding.model_dump(mode="json"),
            source_snapshot=source,
            request_hashes=requests,
        )
        if batch.metadata_checkpoint is None and capture:
            batch.metadata_checkpoint = current.model_dump(mode="json")
        saved = MetadataCheckpoint.model_validate(batch.metadata_checkpoint)
        saved.validate_checksum()
        if (
            saved.provider_binding != current.provider_binding
            or saved.source_snapshot != current.source_snapshot
            or set(saved.request_hashes) != set(requests)
            or any(
                saved.request_hashes[key] != hashes
                for key, hashes in requests.items()
                if key not in completed
            )
        ):
            raise ValueError(
                "The frozen metadata source, recipe or provider binding changed"
            )
        if saved.recipe_sha256 != current.recipe_sha256:
            if not rebase_recipe:
                raise ValueError(
                    "The frozen metadata source or request contract changed"
                )
            saved = MetadataCheckpoint.build(
                recipe_sha256=current.recipe_sha256,
                provider_binding=current.provider_binding,
                source_snapshot=current.source_snapshot,
                request_hashes=current.request_hashes,
                predecessor=saved,
            )
            batch.metadata_checkpoint = saved.model_dump(mode="json")
    except (ValidationError, ValueError) as error:
        raise intake.IntakeConflictError(str(error)) from error
    return saved


def rebase_metadata_checkpoint_for_retry(
    session: Session, batch: RegulatoryDocumentIntake
) -> None:
    """An explicit idle Retry may adopt a validator with identical frozen requests."""
    if batch.metadata_checkpoint is None:
        return
    if batch.labeling_run_id is not None:
        from onyx.db.document_intake_metadata_review import reviewed_metadata_edit

        if reviewed_metadata_edit(batch) is None:
            return
    if (
        batch.status not in {"failed", "partial"}
        or batch.stage != "metadata"
        or batch.lease_token is not None
        or batch.lease_expires_at is not None
        or not try_reserve_metadata_execution(session, batch.id)
    ):
        raise intake.IntakeConflictError("Metadata execution is not idle for Retry")
    from onyx.db.labeling_configuration import resolve_intake_labeling_provider_binding
    from onyx.db.models import User
    from onyx.regulatory.document_intake.metadata_job import metadata_recipe_sha256

    binding = resolve_intake_labeling_provider_binding(
        session,
        batch.labeling_model_configuration_id,
        user=session.get_one(User, batch.created_by),
    )
    _validated_checkpoint(
        session, batch, binding, metadata_recipe_sha256(), rebase_recipe=True
    )


def require_current_metadata_checkpoint(
    session: Session,
    intake_id: UUID,
    lease_token: UUID,
    *,
    provider_binding: LabelingProviderBinding,
    recipe_sha256: str,
    capture: bool,
) -> MetadataCheckpoint:
    batch = require_metadata_lease(session, intake_id, lease_token)
    saved = _validated_checkpoint(
        session, batch, provider_binding, recipe_sha256, capture=capture
    )
    session.flush()
    return saved


def validate_metadata_recovery_source(
    session: Session, batch: RegulatoryDocumentIntake
) -> None:
    from onyx.db.labeling_configuration import resolve_intake_labeling_provider_binding
    from onyx.db.models import User
    from onyx.regulatory.document_intake.metadata_job import metadata_recipe_sha256

    try:
        binding = resolve_intake_labeling_provider_binding(
            session,
            batch.labeling_model_configuration_id,
            user=session.get_one(User, batch.created_by),
        )
    except ValueError as error:
        raise intake.IntakeConflictError(str(error)) from error
    _validated_checkpoint(session, batch, binding, metadata_recipe_sha256())
