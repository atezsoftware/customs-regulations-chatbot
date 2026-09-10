"""Dated durable items retain canonical identity and independent projection identity."""

from collections.abc import Sequence
from datetime import date
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from onyx.db.models import RegulatoryChunk, RegulatoryIndexingItem
from onyx.regulatory.amendments.annexes.models import AnnexCanonicalSnapshot
from onyx.regulatory.amendments.annexes.staging import canonical_snapshot_rows


class DurableProjectionInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    representation: AnnexCanonicalSnapshot
    context_rows: list[AnnexCanonicalSnapshot]
    reference_date: date | None
    canonical_revision_id: UUID | None = None
    canonical_base_sha256: str | None = None


def projection_input(item: RegulatoryIndexingItem) -> DurableProjectionInput | None:
    payload = getattr(item, "projection_input", None)
    return (
        DurableProjectionInput.model_validate(payload) if payload is not None else None
    )


def ordered_projection_items(
    *,
    job_id: UUID,
    user_file_id: UUID,
    rows: Sequence[RegulatoryChunk],
    items: Sequence[RegulatoryIndexingItem],
    require_complete: bool = True,
) -> list[tuple[RegulatoryChunk, RegulatoryIndexingItem]]:
    if not rows:
        raise ValueError("regulatory indexing job has no canonical chunks")
    if any(row.user_file_id != user_file_id for row in rows):
        raise ValueError("canonical chunk belongs to a different user file")
    row_by_id = {row.id: row for row in rows}
    if len(row_by_id) != len(rows):
        raise ValueError("canonical chunks contain duplicate ids")
    identities: set[str] = set()
    covered: set[str] = set()
    ordered: list[tuple[RegulatoryChunk, RegulatoryIndexingItem]] = []
    for item in items:
        if item.job_id != job_id:
            raise ValueError("indexing item belongs to a different job")
        row = row_by_id.get(item.regulatory_chunk_id)
        if row is None:
            raise ValueError("indexing item has no canonical chunk")
        identifier = getattr(item, "projection_id", None)
        key = f"projection:{identifier}" if identifier else f"canonical:{row.id}"
        if key in identities:
            raise ValueError("indexing items contain duplicate projection identities")
        identities.add(key)
        covered.add(row.id)
        frozen = projection_input(item)
        if identifier is not None and frozen is None:
            raise ValueError("dated projection has no frozen input")
        if frozen is not None:
            if identifier is None or (
                frozen.representation.id != row.id
                or UUID(frozen.representation.user_file_id) != user_file_id
                or any(
                    UUID(value.user_file_id) != user_file_id
                    for value in frozen.context_rows
                )
            ):
                raise ValueError("durable projection input scope mismatch")
            row = canonical_snapshot_rows([frozen.representation])[0]
        ordered.append((row, item))
    if require_complete and covered != set(row_by_id):
        raise ValueError("embedded items do not exactly cover canonical chunks")
    return sorted(
        ordered,
        key=lambda pair: (
            pair[0].position,
            pair[0].id,
            getattr(pair[1], "projection_ordinal", None) or 0,
        ),
    )


def projection_ordinal(
    row: RegulatoryChunk, item: RegulatoryIndexingItem, legacy_offset: int
) -> int:
    ordinal = getattr(item, "projection_ordinal", None)
    if ordinal is None:
        ordinal = getattr(row, "projection_ordinal", None)
    return ordinal if ordinal is not None else legacy_offset
