"""Frozen prepared-text pointers for an intake original's reading preview."""

import re
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.models import (
    AmendmentSourcePackage,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatorySourceAsset,
    UserFile,
)


@dataclass(frozen=True)
class IntakePreparedPreview:
    original_file_id: str
    raw_sha256: str
    text_file_id: str
    text_sha256: str


def intake_prepared_preview(
    session: Session, user_file_id: UUID
) -> IntakePreparedPreview | None:
    member = session.scalar(
        select(RegulatoryDocumentIntakeFile)
        .where(RegulatoryDocumentIntakeFile.user_file_id == user_file_id)
        .execution_options(populate_existing=True)
    )
    if member is None:
        return None
    batch = session.get(
        RegulatoryDocumentIntake, member.intake_id, populate_existing=True
    )
    asset = session.get(
        RegulatorySourceAsset, member.source_asset_id, populate_existing=True
    )
    file = session.get(UserFile, user_file_id, populate_existing=True)
    if batch is None or asset is None or file is None:
        raise ValueError("The imported original's source is unavailable")
    package = session.get(
        AmendmentSourcePackage, batch.source_package_id, populate_existing=True
    )
    if (
        asset.package_id != batch.source_package_id
        or file.user_id != batch.created_by
        or package is None
        or package.status != "ready"
        or package.issues
        or package.document_set_id != batch.document_set_id
        or package.created_by != batch.created_by
    ):
        raise ValueError("The imported original's source binding changed")
    if asset.mime_type == "text/markdown":
        return None
    if (
        not asset.text_file_id
        or not asset.text_sha256
        or not re.fullmatch(r"[0-9a-f]{64}", asset.sha256)
        or not re.fullmatch(r"[0-9a-f]{64}", asset.text_sha256)
    ):
        raise ValueError("The imported original has no verified prepared text")
    return IntakePreparedPreview(
        original_file_id=file.file_id,
        raw_sha256=asset.sha256,
        text_file_id=asset.text_file_id,
        text_sha256=asset.text_sha256,
    )
