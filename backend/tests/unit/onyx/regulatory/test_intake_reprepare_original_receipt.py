from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db import (
    document_intake_original_ingestion,
    document_intake_reprepare,
    regulatory_original_ingestion,
)
from onyx.db.enums import UserFileStatus
from onyx.file_processing.original_ingestion import OriginalIngestionReceipt


@pytest.mark.parametrize("reprepared", [False, True, "same_generation"])
def test_unchanged_generation_keeps_original_read_without_intake_audit_query(
    monkeypatch: pytest.MonkeyPatch, reprepared: bool | str
) -> None:
    original = OriginalIngestionReceipt(
        file_id="original",
        raw_sha256="a" * 64,
        documents_sha256="b" * 64,
        plaintext_sha256="c" * 64,
        canonical_sha256="d" * 64,
        generation_hash="e" * 64,
    )
    new = original.model_copy(
        update={
            "generation_hash": original.generation_hash
            if reprepared == "same_generation"
            else "f" * 64,
            "canonical_sha256": "9" * 64,
        }
    )
    session = MagicMock(spec=Session)
    session.get.side_effect = [
        SimpleNamespace(
            original_ingestion_receipt=original.model_dump(mode="json"),
            gate_closed=False,
        ),
        SimpleNamespace(
            file_id="original",
            status=UserFileStatus.COMPLETED,
            regulatory_chunk_generation_hash=new.generation_hash
            if reprepared
            else original.generation_hash,
        ),
    ]
    resolver = MagicMock(return_value=new)
    monkeypatch.setattr(
        document_intake_original_ingestion, "approved_intake_original_receipt", resolver
    )
    monkeypatch.setattr(
        "onyx.db.regulatory_writer_publication.canonical_scope_digest",
        lambda *_args: (
            new.canonical_sha256 if reprepared else original.canonical_sha256
        ),
    )
    assert regulatory_original_ingestion.original_attachment_receipt(
        session, uuid4()
    ) == (new if reprepared else original)
    assert resolver.call_count == (1 if reprepared else 0)


def test_ordinary_generation_drift_preserves_matching_original_eligibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = OriginalIngestionReceipt(
        file_id="original",
        raw_sha256="a" * 64,
        documents_sha256="b" * 64,
        plaintext_sha256="c" * 64,
        canonical_sha256="d" * 64,
        generation_hash="e" * 64,
    )
    session = MagicMock(spec=Session)
    session.get.side_effect = [
        SimpleNamespace(
            original_ingestion_receipt=original.model_dump(mode="json"),
            gate_closed=False,
        ),
        SimpleNamespace(
            file_id=original.file_id,
            status=UserFileStatus.COMPLETED,
            regulatory_chunk_generation_hash="f" * 64,
        ),
    ]
    resolver = MagicMock(return_value=original)
    approved = MagicMock()
    monkeypatch.setattr(
        document_intake_reprepare, "reprepared_original_receipt", resolver
    )
    monkeypatch.setattr(
        document_intake_original_ingestion, "approved_intake_original_receipt", approved
    )
    monkeypatch.setattr(
        "onyx.db.regulatory_writer_publication.canonical_scope_digest",
        lambda *_args: original.canonical_sha256,
    )
    assert (
        regulatory_original_ingestion.original_attachment_receipt(session, uuid4())
        == original
    )
    resolver.assert_called_once()
    approved.assert_not_called()
