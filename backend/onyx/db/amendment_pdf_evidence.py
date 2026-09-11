"""Scope checks for immutable PDF source evidence attached to ordinary proposals."""

from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from onyx.db.amendment_sources import list_source_assets, require_ready_source_package
from onyx.db.models import AmendmentBatch, AmendmentProposal
from onyx.regulatory.amendments.annexes import config
from onyx.regulatory.amendments.pdf_vision import (
    PDF_EVIDENCE_KEY,
    PdfBatchSource,
    PdfOriginal,
    PdfProposalEvidence,
    digest,
    snapshot_digest,
    validate_pdf_frozen_references,
)


def load_batch_pdf_source(
    session: Session, batch: AmendmentBatch
) -> PdfBatchSource | None:
    if not config.REGULATORY_ANNEX_UPDATES_ENABLED or batch.source_package_id is None:
        return None
    package = require_ready_source_package(
        session,
        package_id=batch.source_package_id,
        document_set_id=batch.document_set_id,
        environment=config.REGULATORY_ANNEX_ENVIRONMENT,
    )
    if (
        package.created_by != batch.created_by
        or not package.manifest_file_id
        or not package.manifest_sha256
    ):
        raise ValueError("pdf_source_package_authority_mismatch")
    if batch.source_text_sha256 != digest(batch.raw_text):
        raise ValueError("pdf_source_text_authority_mismatch")
    originals = [
        PdfOriginal(asset_id=asset.id, file_id=asset.file_id, sha256=asset.sha256)
        for asset in list_source_assets(session, package.id)
        if asset.mime_type == "application/pdf"
    ]
    if not originals:
        return None
    return PdfBatchSource(
        batch_id=batch.id,
        package_id=package.id,
        source_text_sha256=digest(batch.raw_text),
        manifest_file_id=package.manifest_file_id,
        manifest_sha256=package.manifest_sha256,
        originals=originals,
        user_file_ids=[UUID(value) for value in batch.user_file_ids],
    )


def validate_pdf_proposal_authority(
    session: Session, proposal: AmendmentProposal, draft: dict[str, Any]
) -> None:
    raw = proposal.old_chunk_snapshot.get(PDF_EVIDENCE_KEY)
    if raw is None:
        return
    evidence = PdfProposalEvidence.model_validate(raw)
    batch = session.get(AmendmentBatch, proposal.batch_id, populate_existing=True)
    if batch is None or batch.superseded_by_batch_id is not None:
        raise ValueError("pdf_proposal_source_batch_changed")
    source = load_batch_pdf_source(session, batch)
    if (
        source is None
        or evidence.batch_id != batch.id
        or evidence.package_id != source.package_id
        or evidence.source_text_sha256 != source.source_text_sha256
        or evidence.manifest_sha256 != source.manifest_sha256
        or evidence.old_chunk_id != proposal.old_chunk_id
        or evidence.old_chunk_id != proposal.old_chunk_snapshot.get("id")
        or evidence.old_snapshot_sha256 != snapshot_digest(proposal.old_chunk_snapshot)
        or evidence.draft_text_sha256 != digest(draft["text"])
        or str(draft["user_file_id"]) not in batch.user_file_ids
    ):
        raise ValueError("pdf_proposal_evidence_changed_reanalysis_required")
    originals = {item.asset_id: item.sha256 for item in source.originals}
    if any(
        originals.get(page.asset_id) != page.source_sha256 for page in evidence.pages
    ):
        raise ValueError("pdf_proposal_original_changed")

    from onyx.file_store.file_store import get_default_file_store

    validate_pdf_frozen_references(source, evidence, get_default_file_store())
