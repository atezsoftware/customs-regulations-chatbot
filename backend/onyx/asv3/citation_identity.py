"""Bind one citation number to its retained original and captured scope."""

from pydantic import JsonValue

from onyx.asv3.models import EvidenceItem
from onyx.context.search.models import IndexFilters


def saved_item_from_checkpoint(
    checkpoint: dict[str, JsonValue], number: int
) -> tuple[EvidenceItem, IndexFilters]:
    scope = checkpoint.get("scope")
    evidence = checkpoint.get("evidence")
    if number < 1 or not isinstance(scope, dict) or not isinstance(evidence, dict):
        raise ValueError("Invalid citation checkpoint")
    if evidence.get("version") != 1:
        raise ValueError("Unsupported citation ledger version")
    records = evidence.get("records")
    if not isinstance(records, list):
        raise ValueError("Missing citation ledger")
    record = next(
        (
            row
            for row in records
            if isinstance(row, dict) and row.get("citation") == number
        ),
        None,
    )
    if record is None:
        raise ValueError("Citation not found")
    item = EvidenceItem.model_validate(record.get("item"))
    if item.search_doc is None or item.search_doc.document_id != item.source_id:
        raise ValueError("Citation source identity differs from saved evidence")
    if item.chunk_id is None:
        if (
            not item.metadata.get("derived")
            or not item.metadata.get("source_sha256")
            or item.search_doc.chunk_ind != -number
        ):
            raise ValueError("Citation is not verified native evidence")
    elif (
        item.search_doc.chunk_ind < 0
        or item.search_doc.metadata.get("regulatory_chunk_id") != item.chunk_id
    ):
        raise ValueError("Citation canonical identity differs from saved evidence")
    return item, IndexFilters.model_validate(scope)
