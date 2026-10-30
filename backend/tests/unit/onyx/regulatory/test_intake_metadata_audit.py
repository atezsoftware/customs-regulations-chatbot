"""A recovered metadata result preserves the request that supplied each fact."""

from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

from pytest import MonkeyPatch
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import document_intake_metadata as repository
from onyx.db.models import RegulatoryChunk, RegulatoryDocumentIntake
from onyx.regulatory.document_intake.metadata import (
    METADATA_AUDIT_KEY,
    ChunkLegalMetadata,
    LegalMetadataFact,
    LegalMetadataResult,
    metadata_input_hash,
)


def test_applied_metadata_distinguishes_retained_current_and_manual_fact_sources(
    monkeypatch: MonkeyPatch,
) -> None:
    original_hash, recovery_hash = "a" * 64, "b" * 64
    manual_provenance = {"source": "manual", "basis": "user_review", "stale": False}
    chunk = RegulatoryChunk(
        id="rc_metadata_audit",
        user_file_id=uuid4(),
        text="MADDE 1 — Belge.\nResmî Gazete Sayı : 31529",
        heading_path=["Belge", "MADDE 1"],
        chunk_type="article",
        chunk_metadata={
            "document_title": "Elle incelenmiş başlık",
            METADATA_AUDIT_KEY: {
                "fields": {"document_title": manual_provenance},
                "warnings": [],
                "input_stale": False,
            },
        },
    )
    batch = RegulatoryDocumentIntake(id=uuid4())
    session = cast(Session, MagicMock(spec=Session))
    monkeypatch.setattr(
        intake, "load_intake_job", MagicMock(return_value=(batch, None, []))
    )
    monkeypatch.setattr(repository, "_load_rows", MagicMock(return_value=[chunk]))
    monkeypatch.setattr(repository, "refresh_intake_document_name", MagicMock())
    monkeypatch.setattr(repository, "heartbeat_metadata", MagicMock())
    result = LegalMetadataResult(
        chunks=[
            ChunkLegalMetadata(
                chunk_id=chunk.id,
                fields=[
                    LegalMetadataFact(
                        field="publication_number",
                        value="31529",
                        source_chunk_id=chunk.id,
                        evidence_quote="Resmî Gazete Sayı : 31529",
                        retained_request_hash=original_hash,
                    ),
                    LegalMetadataFact(
                        field="article_no",
                        value="1",
                        source_chunk_id=chunk.id,
                        evidence_quote="MADDE 1 — Belge.",
                    ),
                    LegalMetadataFact(
                        field="document_title",
                        value="Belge",
                        source_chunk_id=chunk.id,
                        evidence_quote="MADDE 1 — Belge.",
                        retained_request_hash=original_hash,
                    ),
                ],
            )
        ]
    )
    repository.apply_metadata_results(
        session,
        intake_id=batch.id,
        lease_token=uuid4(),
        document_id=uuid4(),
        expected_input_sha256=metadata_input_hash(repository.metadata_chunks([chunk])),
        results=[(recovery_hash, result)],
    )

    audit = chunk.chunk_metadata[METADATA_AUDIT_KEY]
    retained = audit["fields"]["publication_number"]
    assert chunk.chunk_metadata["publication_number"] == "31529"
    assert retained["request_hash"] == original_hash
    assert retained["retained_from_validated_fact"] is True
    assert retained["source_chunk_id"] == chunk.id
    assert retained["evidence_quote"] == "Resmî Gazete Sayı : 31529"
    assert audit["fields"]["article_no"]["request_hash"] == recovery_hash
    assert "retained_from_validated_fact" not in audit["fields"]["article_no"]
    assert chunk.chunk_metadata["document_title"] == "Elle incelenmiş başlık"
    assert audit["fields"]["document_title"] == manual_provenance
    assert audit["request_hash"] == recovery_hash
