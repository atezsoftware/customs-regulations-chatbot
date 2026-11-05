"""Immutable source references shared by approval receipts and read-side lineage."""

from onyx.db.models import (
    AmendmentSourcePackage,
    RegulatoryDocumentIntakeFile,
    RegulatorySourceAsset,
    UserFile,
)


def intake_source_reference(
    package: AmendmentSourcePackage,
    member: RegulatoryDocumentIntakeFile,
    asset: RegulatorySourceAsset,
    file: UserFile,
) -> dict[str, object]:
    return {
        "package_id": str(package.id),
        "asset_id": str(asset.id),
        "asset_sha256": asset.sha256,
        "asset_file_id": asset.file_id,
        "prepared_file_id": asset.text_file_id,
        "prepared_sha256": asset.text_sha256,
        "original_file_id": file.file_id,
        "manifest_file_id": package.manifest_file_id,
        "manifest_sha256": package.manifest_sha256,
        "role": member.role,
        "parent_document_id": str(member.parent_document_id)
        if member.parent_document_id
        else None,
        "relationships": member.relationships,
    }
