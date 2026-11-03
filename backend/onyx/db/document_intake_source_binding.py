"""Current owned source identity for intake-only readers and semantic repair."""

from collections.abc import Mapping
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.models import (
    DocumentSet__UserFile,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryFilePublication,
    RegulatorySourceAsset,
    UserFile,
)
from onyx.file_processing.original_ingestion import OriginalIngestionReceipt
from onyx.regulatory.document_intake.embedding_policy import (
    embedding_token_limit_for_metadata,
)


def require_owned_v4_intake_source(
    session: Session,
    *,
    user_file_id: UUID,
    metadata: Mapping[str, Any],
    chunk_type: str | None,
    document_set_id: int | None = None,
) -> RegulatorySourceAsset:
    if metadata.get("intake_chunking_version") not in ("4", "5"):
        raise ValueError("Only frozen whole-table intake sources support this reader")
    embedding_token_limit_for_metadata(metadata, chunk_type)
    return require_owned_intake_source(
        session, user_file_id=user_file_id, document_set_id=document_set_id
    )


def require_owned_intake_source(
    session: Session,
    *,
    user_file_id: UUID,
    document_set_id: int | None = None,
) -> RegulatorySourceAsset:
    member = session.scalar(
        select(RegulatoryDocumentIntakeFile)
        .where(RegulatoryDocumentIntakeFile.user_file_id == user_file_id)
        .execution_options(populate_existing=True)
    )
    if member is None:
        raise ValueError("A current owned intake source is required")
    batch = session.get(
        RegulatoryDocumentIntake, member.intake_id, populate_existing=True
    )
    asset = session.get(
        RegulatorySourceAsset, member.source_asset_id, populate_existing=True
    )
    file = session.get(UserFile, user_file_id, populate_existing=True)
    publication = session.get(
        RegulatoryFilePublication, user_file_id, populate_existing=True
    )
    if batch is None or asset is None or file is None:
        raise ValueError("The current owned intake source is unavailable")
    if (
        asset.package_id != batch.source_package_id
        or asset.mime_type != file.file_type
        or (document_set_id is not None and batch.document_set_id != document_set_id)
        or session.scalar(
            select(DocumentSet__UserFile).where(
                DocumentSet__UserFile.document_set_id == batch.document_set_id,
                DocumentSet__UserFile.user_file_id == user_file_id,
            )
        )
        is None
        or publication is None
        or not publication.original_ingestion_receipt
    ):
        raise ValueError("The current retained intake source binding changed")
    receipt = OriginalIngestionReceipt.model_validate(
        publication.original_ingestion_receipt
    )
    if receipt.file_id != file.file_id or receipt.raw_sha256 != asset.sha256:
        raise ValueError("The current original intake source changed")
    return asset
