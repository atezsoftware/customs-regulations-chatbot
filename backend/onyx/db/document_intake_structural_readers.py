"""Read-only retained HTML evidence for intake-only structural label readers."""

import hashlib
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.document_intake_source_binding import require_owned_v4_intake_source
from onyx.db.models import (
    RegulatoryDocumentIntakeFile,
    RegulatorySourceAsset,
)
from onyx.db.regulatory_annex_changes import capture_canonical_scope
from onyx.file_store.file_store import FileStore, get_default_file_store
from onyx.regulatory.amendments.annexes.models import AnnexCanonicalSnapshot
from onyx.regulatory.amendments.annexes.sources import MAX_ASSET_BYTES
from onyx.regulatory.document_intake.embedding_policy import (
    embedding_token_limit_for_metadata,
)
from onyx.regulatory.document_intake.structural_readers import (
    NativeChapterReaderProof,
    build_native_chapter_reader_proofs,
    native_chapter_reader_candidate,
    validate_native_chapter_reader_proof,
)


def validate_current_structural_reader_proof(
    session: Session,
    proof: NativeChapterReaderProof,
    rows: Sequence[AnnexCanonicalSnapshot] | None = None,
) -> NativeChapterReaderProof:
    rows = (
        rows
        if rows is not None
        else capture_canonical_scope(session, proof.user_file_id)
    )
    governed = {proof.heading_chunk_id, *proof.governed_chunk_ids}
    selected = [row for row in rows if row.id in governed]
    if len(selected) != len(governed):
        raise ValueError("The native reader v4 scope is incomplete")
    version = selected[0].metadata.get("intake_chunking_version")
    if version not in ("4", "5") or any(
        row.metadata.get("intake_chunking_version") != version for row in selected
    ):
        raise ValueError(
            "Native chapter readers require one frozen whole-table profile"
        )
    for row in selected:
        embedding_token_limit_for_metadata(row.metadata, row.chunk_type)
    asset = require_owned_v4_intake_source(
        session,
        user_file_id=proof.user_file_id,
        metadata=selected[0].metadata,
        chunk_type=selected[0].chunk_type,
    )
    if asset.mime_type != "text/html":
        raise ValueError("Native chapter readers require retained HTML")
    return validate_native_chapter_reader_proof(
        proof,
        rows=rows,
        expected_source_sha256=asset.sha256,
        expected_source_asset_id=asset.id,
        user_file_id=proof.user_file_id,
    )


def load_intake_structural_reader_proofs(
    session: Session, *, intake_id: UUID, store: FileStore | None = None
) -> dict[str, NativeChapterReaderProof]:
    """Verify retained original bytes once per HTML source at labeling admission."""
    members = session.scalars(
        select(RegulatoryDocumentIntakeFile)
        .join(
            RegulatorySourceAsset,
            RegulatorySourceAsset.id == RegulatoryDocumentIntakeFile.source_asset_id,
        )
        .where(
            RegulatoryDocumentIntakeFile.intake_id == intake_id,
            RegulatorySourceAsset.mime_type == "text/html",
        )
        .order_by(RegulatoryDocumentIntakeFile.id)
    )
    result: dict[str, NativeChapterReaderProof] = {}
    for member in members:
        rows = capture_canonical_scope(session, member.user_file_id)
        candidates = [
            row
            for row in rows
            if native_chapter_reader_candidate(row)
            and row.metadata.get("intake_chunking_version") in ("4", "5")
        ]
        if not candidates:
            continue
        asset = require_owned_v4_intake_source(
            session,
            user_file_id=member.user_file_id,
            metadata=candidates[0].metadata,
            chunk_type=candidates[0].chunk_type,
        )
        if asset.mime_type != "text/html":
            raise ValueError("Native chapter readers require retained HTML")
        file_store = store or get_default_file_store()
        with file_store.read_file(asset.file_id) as stream:
            raw = stream.read(MAX_ASSET_BYTES + 1)
        if (
            len(raw) != asset.byte_count
            or len(raw) > MAX_ASSET_BYTES
            or hashlib.sha256(raw).hexdigest() != asset.sha256
        ):
            raise ValueError("Native chapter reader retained raw source changed")
        proofs = build_native_chapter_reader_proofs(
            raw_html=raw,
            expected_sha256=asset.sha256,
            user_file_id=member.user_file_id,
            source_asset_id=asset.id,
            rows=rows,
        )
        for proof in proofs.values():
            result[proof.heading_chunk_id] = validate_current_structural_reader_proof(
                session, proof, rows
            )
    return result
