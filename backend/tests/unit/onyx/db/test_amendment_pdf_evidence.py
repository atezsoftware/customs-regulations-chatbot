"""PDF proposal evidence is checked against current batch, source and target."""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from onyx.db import amendment_pdf_evidence as dal
from onyx.db.models import (
    AmendmentBatch,
    AmendmentProposal,
    AmendmentSourcePackage,
    RegulatorySourceAsset,
)
from onyx.regulatory.amendments.pdf_vision import (
    PDF_EVIDENCE_KEY,
    PdfPageReference,
    PdfProposalEvidence,
    digest,
    snapshot_digest,
)


def scope(monkeypatch: pytest.MonkeyPatch):
    user_id, file_id, package_id, asset_id = uuid4(), uuid4(), uuid4(), uuid4()
    batch = AmendmentBatch(
        id=44,
        document_set_id=18,
        source_package_id=package_id,
        created_by=user_id,
        raw_text="MADDE 3 update",
        source_text_sha256=digest("MADDE 3 update"),
        user_file_ids=[str(file_id)],
        superseded_by_batch_id=None,
    )
    package = AmendmentSourcePackage(
        id=package_id,
        document_set_id=18,
        created_by=user_id,
        manifest_file_id="manifest",
        manifest_sha256="a" * 64,
    )
    asset = RegulatorySourceAsset(
        id=asset_id,
        package_id=package_id,
        file_id="original",
        sha256="b" * 64,
        mime_type="application/pdf",
    )
    monkeypatch.setattr(dal.config, "REGULATORY_ANNEX_UPDATES_ENABLED", True)
    monkeypatch.setattr(
        dal, "require_ready_source_package", MagicMock(return_value=package)
    )
    monkeypatch.setattr(dal, "list_source_assets", MagicMock(return_value=[asset]))
    session = MagicMock()
    session.get.return_value = batch
    return batch, package, asset, session, file_id


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "package_owner",
        "source_text",
        "source_revision",
        "manifest",
        "original",
        "draft",
        "target",
        "old_snapshot",
        "foreign_file",
        "malformed",
    ],
)
def test_pdf_approval_requires_original_source_and_target_binding(
    monkeypatch: pytest.MonkeyPatch, mutation: str | None
) -> None:
    batch, package, asset, session, file_id = scope(monkeypatch)
    snapshot = {
        "id": "old",
        "user_file_id": str(file_id),
        "text": "old 5%",
        "metadata": {},
        "heading_path": [],
    }
    receipt = PdfProposalEvidence(
        batch_id=44,
        package_id=package.id,
        source_text_sha256=digest(batch.raw_text),
        manifest_sha256="a" * 64,
        old_chunk_id="old",
        old_snapshot_sha256=snapshot_digest(snapshot),
        draft_text_sha256=digest("new 17%"),
        pages=[
            PdfPageReference(
                asset_id=asset.id,
                source_sha256="b" * 64,
                extraction_sha256="c" * 64,
                page=1,
                positions=[0, 1],
            )
        ],
    )
    snapshot[PDF_EVIDENCE_KEY] = receipt.model_dump(mode="json")
    proposal = AmendmentProposal(
        id=2, batch_id=44, old_chunk_id="old", old_chunk_snapshot=snapshot
    )
    draft = {"text": "new 17%", "user_file_id": str(file_id)}
    if mutation == "package_owner":
        package.created_by = uuid4()
    if mutation == "source_text":
        batch.raw_text = "changed source"
    if mutation == "source_revision":
        batch.superseded_by_batch_id = 45
    if mutation == "manifest":
        package.manifest_sha256 = "d" * 64
    if mutation == "original":
        asset.sha256 = "e" * 64
    if mutation == "draft":
        draft["text"] = "new 99%"
    if mutation == "target":
        proposal.old_chunk_id = "other"
    if mutation == "old_snapshot":
        snapshot["text"] = "changed baseline"
    if mutation == "foreign_file":
        draft["user_file_id"] = str(uuid4())
    if mutation == "malformed":
        snapshot[PDF_EVIDENCE_KEY] = {"unexpected": True}
    if mutation:
        with pytest.raises(ValueError):
            dal.validate_pdf_proposal_authority(session, proposal, draft)
    else:
        dal.validate_pdf_proposal_authority(session, proposal, draft)
    session.add.assert_not_called()
    session.commit.assert_not_called()


def test_legacy_text_proposal_does_not_load_pdf_authority() -> None:
    session = MagicMock()
    proposal = AmendmentProposal(old_chunk_snapshot={"id": "old"})
    dal.validate_pdf_proposal_authority(session, proposal, {"text": "new"})
    session.get.assert_not_called()
