"""Revalidate intake-only semantic repair evidence before using accepted labels."""

import json
from collections.abc import Mapping, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.document_intake_source_binding import (
    require_owned_intake_source,
    require_owned_v4_intake_source,
)
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryLabelTaxonomy,
)
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from onyx.regulatory.labeling.intake_abstention_repair import (
    ABSTENTION_REPAIR_KEY,
    AbstentionRepairAudit,
    validate_stored_abstention_repair_audit,
)
from onyx.regulatory.labeling.intake_structural_repair import (
    legacy_structural_repair_candidate,
)
from onyx.regulatory.labeling.provider import TaxonomyDefinition, build_labeling_request

SEMANTIC_REPAIR_KEY = ABSTENTION_REPAIR_KEY


def require_current_intake_repair_item(
    session: Session, *, item: RegulatoryLabelingItem, run: RegulatoryLabelingRun
) -> None:
    """Check target identity; consumers also retain their full context-current guard."""
    if item.run_id != run.id or str(item.user_file_id) not in run.file_ids:
        raise ValueError("Semantic repair differs from its frozen run scope")
    row = session.scalar(
        select(RegulatoryChunk)
        .where(
            RegulatoryChunk.id == item.regulatory_chunk_id,
            RegulatoryChunk.user_file_id == item.user_file_id,
        )
        .execution_options(populate_existing=True)
    )
    if (
        row is None
        or row.status != "active"
        or row.superseded_by_chunk_id is not None
        or row.chunk_metadata.get("chunk_variant") != "atomic"
    ):
        raise ValueError("Semantic repair requires a current canonical atomic source")
    expected = {
        "position": row.position,
        "heading_path": row.heading_path,
        "chunk_type": row.chunk_type,
        "chunk_metadata": row.chunk_metadata,
        "validity_start_date": row.validity_start_date.isoformat()
        if row.validity_start_date
        else None,
        "validity_end_date": row.validity_end_date.isoformat()
        if row.validity_end_date
        else None,
    }
    if (
        row.text != item.text_snapshot
        or any(
            item.source_snapshot.get(key) != value for key, value in expected.items()
        )
        or item.canonical_text_sha256 != context_hash(row.text)
    ):
        raise ValueError("Semantic repair canonical source snapshot changed")
    if row.chunk_metadata.get("intake_chunking_version") in ("4", "5"):
        require_owned_v4_intake_source(
            session,
            user_file_id=item.user_file_id,
            metadata=row.chunk_metadata,
            chunk_type=row.chunk_type,
            document_set_id=run.document_set_id,
        )
    elif legacy_structural_repair_candidate(
        chunk_id=row.id, text=row.text, source_snapshot=item.source_snapshot
    ):
        require_owned_intake_source(
            session,
            user_file_id=item.user_file_id,
            document_set_id=run.document_set_id,
        )
    else:
        raise ValueError("Legacy semantic repair requires a verified structural source")


def validate_item_semantic_repair_audit(
    session: Session,
    *,
    item: RegulatoryLabelingItem,
    run: RegulatoryLabelingRun,
    audit: AbstentionRepairAudit | Mapping[str, object] | None = None,
    labels: Sequence[str] | None = None,
    assignments: Sequence[Mapping[str, object]] | None = None,
) -> None:
    """An absent audit is legacy; a present invalid audit can never be ignored.

    This verifies exact frozen context text and target bindings without recursively
    validating neighbors. Apply/search/reuse retain their current-context check.
    """
    present = SEMANTIC_REPAIR_KEY in item.source_snapshot
    if audit is None and not present:
        return
    if (labels is None) != (assignments is None):
        raise ValueError("Semantic repair prospective assignments are incomplete")
    stored = AbstentionRepairAudit.model_validate(
        audit if audit is not None else item.source_snapshot[SEMANTIC_REPAIR_KEY]
    )
    if (
        audit is not None
        and present
        and item.source_snapshot[SEMANTIC_REPAIR_KEY] != stored.model_dump(mode="json")
    ):
        raise ValueError("A semantic repair attempt has already been recorded")
    require_current_intake_repair_item(session, item=item, run=run)
    taxonomy_row = session.get_one(
        RegulatoryLabelTaxonomy, run.taxonomy_id, populate_existing=True
    )
    taxonomy = TaxonomyDefinition.model_validate(taxonomy_row.definition)
    if taxonomy_row.version_hash != taxonomy.version_hash:
        raise ValueError("Semantic repair frozen taxonomy changed")
    if run.provider_binding.get("model") != run.model or stored.model != run.model:
        raise ValueError("Semantic repair frozen model changed")
    if (
        not item.request_payload
        or not item.context_sha256
        or item.context_snapshot is None
    ):
        raise ValueError("Semantic repair requires a frozen base request and context")
    base = VertexBatchRequest.model_validate(item.request_payload)
    rebuilt = build_labeling_request(
        chunk_id=item.regulatory_chunk_id,
        text=item.text_snapshot,
        context=item.context_snapshot,
        taxonomy=taxonomy,
        source_hash=item.canonical_text_sha256 or "",
    )
    if base != rebuilt or item.request_hash != base.request_hash:
        raise ValueError("Semantic repair base request or frozen context changed")
    # Parsing again also rejects a malformed request payload before trusting audit.
    if json.loads(base.prompt).get("interpretation_context") != item.context_snapshot:
        raise ValueError("Semantic repair exact context changed")
    validate_stored_abstention_repair_audit(
        stored,
        base_request=base,
        text=item.text_snapshot,
        chunk_id=item.regulatory_chunk_id,
        source_hash=item.canonical_text_sha256 or "",
        taxonomy=taxonomy,
        provider_binding=run.provider_binding,
        labels=item.labels if labels is None else labels,
        assignments=item.assignments if assignments is None else assignments,
    )
