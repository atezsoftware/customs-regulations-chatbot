"""Demand-driven reconciliation of approved text with retrievable originals."""

import datetime
import hashlib
import json
from uuid import UUID

from sqlalchemy.orm import Session

from onyx.db.regulatory_annexes import (
    canonical_chunk_lineage_keys,
    get_effective_annex_revision,
    get_or_create_annex,
    load_legacy_annex_chunks,
    persist_annex_revision,
    require_annex_file_scope,
)
from onyx.file_store.file_store import FileStore
from onyx.regulatory.amendments.annexes.models import (
    AnnexBaseline,
    AnnexExtraction,
    AnnexLocator,
    AnnexOriginalEvidence,
    ExtractedAnnexElement,
)
from onyx.regulatory.amendments.annexes.sources import inspect_source
from onyx.regulatory.chunk_evidence import chunk_evidence
from onyx.utils.process_isolation import run_in_isolated_process

MAX_ORIGINAL_BYTES = 25 * 1024 * 1024


def _verify_original(content: bytes, mime: str) -> None:
    from onyx.regulatory.amendments.annexes.source_parser import (
        apply_source_process_limits,
    )

    apply_source_process_limits()
    inspect_source(content, mime)


def verify_legacy_original(store: FileStore, file_id: str) -> AnnexOriginalEvidence:
    evidence = AnnexOriginalEvidence(file_id=file_id)
    try:
        size = store.get_file_size(file_id)
        if size is None:
            evidence.issue = "original_unavailable"
            return evidence
        if size > MAX_ORIGINAL_BYTES:
            evidence.issue = "original_size_limit"
            return evidence
        record = store.read_file_record(file_id)
        evidence.mime_type = record.file_type
        if record.file_type in ("text/markdown", "text/plain"):
            evidence.issue = "canonical_only_source"
            return evidence
        if record.file_type == "image/gif":
            evidence.issue = "legacy_image_format_unsupported"
            return evidence
        with store.read_file(file_id) as stream:
            content = stream.read(MAX_ORIGINAL_BYTES + 1)
        if len(content) != size:
            evidence.issue = "original_size_mismatch"
            return evidence
        run_in_isolated_process(_verify_original, content, record.file_type, timeout=30)
        evidence.sha256 = hashlib.sha256(content).hexdigest()
        evidence.available = True
    except Exception:
        # FileStore metadata alone does not prove S3/the PG large object exists.
        # Keep the approved transcript and make visual unavailability explicit.
        evidence.issue = "original_unavailable"
    return evidence


def prepare_legacy_baseline(
    session: Session,
    store: FileStore,
    *,
    document_set_id: int,
    user_file_id: UUID,
    annex_label: str,
    as_of_date: datetime.date,
) -> AnnexBaseline:
    file = require_annex_file_scope(session, document_set_id, user_file_id)
    rows = load_legacy_annex_chunks(
        session,
        document_set_id=document_set_id,
        user_file_id=user_file_id,
        annex_label=annex_label,
        as_of_date=as_of_date,
    )
    if not rows:
        raise ValueError("legacy annex scope has no canonical chunks")
    originals: list[AnnexOriginalEvidence] = []
    seen: set[str] = set()
    elements: list[ExtractedAnnexElement] = []
    lineage_keys = canonical_chunk_lineage_keys(session, rows)
    for row in rows:
        references = chunk_evidence(row.chunk_metadata)
        image_id = references.image_file_id
        for image_file_id in dict.fromkeys(
            [*references.image_file_ids, *([image_id] if image_id else [])]
        ):
            if image_file_id not in seen:
                seen.add(image_file_id)
                originals.append(verify_legacy_original(store, image_file_id))
        elements.append(
            ExtractedAnnexElement(
                canonical_chunk_id=row.id,
                bound_to_regulatory_chunk_id=row.chunk_metadata.get(
                    "bound_to_regulatory_chunk_id"
                ),
                canonical_role="supporting"
                if row.source == "indexed"
                and row.chunk_metadata.get("chunk_variant") == "image_companion"
                else "authoritative",
                kind="image_region" if image_id else "text",
                text=row.text,
                semantic_key=lineage_keys[row.id],
                evidence_kind="canonical",
                image_file_id=image_id if isinstance(image_id, str) else None,
                locator=AnnexLocator(path=" > ".join(row.heading_path)),
            )
        )
    for original in originals:
        original.canonical_chunk_ids = [
            row.id
            for row in rows
            if row.chunk_metadata.get("image_file_id") == original.file_id
            or original.file_id in row.chunk_metadata.get("image_file_ids", [])
        ]
    corrections = [row.id for row in rows if row.source == "amendment"]
    issues: list[str] = (
        ["raw_original_may_precede_approved_correction"] if corrections else []
    )
    if (
        file.file_type not in ("text/markdown", "text/plain")
        and file.file_id not in seen
    ):
        original = verify_legacy_original(store, file.file_id)
        original.canonical_chunk_ids = [row.id for row in rows]
        originals.append(original)
    else:
        issues.append("canonical_only_user_file")
    canonical_text = "\n\n".join(
        element.text
        for element in elements
        if element.canonical_role == "authoritative"
    )
    hash_input = {
        "document_set_id": document_set_id,
        "user_file_id": str(user_file_id),
        "rows": [
            {
                "id": row.id,
                "text": row.text,
                "metadata": row.chunk_metadata,
                "start": str(row.validity_start_date),
                "end": str(row.validity_end_date),
            }
            for row in rows
        ],
        "originals": [item.model_dump() for item in originals],
    }
    digest = hashlib.sha256(
        json.dumps(hash_input, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    baseline = AnnexBaseline(
        canonical_amendment_chunk_ids=corrections,
        baseline_sha256=digest,
        canonical_text=canonical_text,
        elements=elements,
        originals=originals,
        visual_evidence_available=bool(originals)
        and all(
            item.available
            and (item.mime_type or "").startswith(("image/", "application/pdf"))
            for item in originals
        ),
        issues=issues,
    )
    annex = get_or_create_annex(
        session,
        document_set_id=document_set_id,
        user_file_id=user_file_id,
        annex_label=annex_label,
    )
    previous = get_effective_annex_revision(session, annex.id, as_of_date)
    starts = [
        row.validity_start_date for row in rows if row.validity_start_date is not None
    ]
    ends = [row.validity_end_date for row in rows if row.validity_end_date is not None]
    revision = persist_annex_revision(
        session,
        annex_id=annex.id,
        extraction=AnnexExtraction(
            source_sha256=hashlib.sha256(canonical_text.encode()).hexdigest(),
            mime_type="text/markdown",
            elements=elements,
        ),
        baseline_sha256=digest,
        effective_start=max(starts) if starts else None,
        effective_end=min(ends) if ends else None,
        approved_at=max(row.updated_at for row in rows),
        predecessor_revision_id=previous.id if previous else None,
        chunk_ids_by_position={position: [row.id] for position, row in enumerate(rows)},
        baseline_snapshot=baseline.model_dump(mode="json"),
    )
    baseline.revision_id = str(revision.id)
    return baseline
