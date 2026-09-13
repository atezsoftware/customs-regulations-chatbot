from __future__ import annotations

import datetime
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from typing import cast as type_cast
from uuid import UUID, uuid4

from sqlalchemy import (
    Integer,
    and_,
    case,
    cast,
    func,
    insert,
    literal,
    or_,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session, load_only, selectinload
from sqlalchemy.sql import Select
from sqlalchemy.sql.elements import ColumnElement

from onyx.db.enums import RegulatoryChunkStatus
from onyx.db.models import (
    DocumentSet,
    DocumentSet__UserFile,
    RegulatoryChunk,
    RegulatoryDerivedLabelProjection,
    RegulatoryIndexingItem,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryLabelingShard,
    RegulatoryLabelSettings,
    RegulatoryLabelTaxonomy,
    User,
    UserFile,
)
from onyx.regulatory.amendments.annexes.context_dependencies import (
    canonical_dependency_ids,
    context_hash,
)
from onyx.regulatory.chunker import (
    ATOMIC_CHUNK_VARIANT,
    HIERARCHICAL_AGGREGATE_CHUNK_VARIANT,
)
from onyx.regulatory.contextual import (
    context_reference_date,
    validity_window_contains,
    visible_regulatory_snapshot_for_target,
)
from onyx.regulatory.labeling.api_models import (
    LabelingCounts,
    LabelingItemSnapshot,
    LabelingItemsPage,
    LabelingRunSnapshot,
    TaxonomySummary,
)
from onyx.regulatory.labeling.domain import (
    LabelingChunkView,
    bounded_document_context,
)
from onyx.regulatory.labeling.provider import LabelDefinition, TaxonomyDefinition

_ACTIVE_RUN_STATUSES = ("queued", "running")
_TERMINAL_ITEM_STATUSES = ("completed", "failed", "stale", "cancelled")
_CONTEXT_BYTE_LIMIT = 384 * 1024
_CONTEXT_NEIGHBOR_LIMIT = 24
_CONTEXT_CANDIDATE_LIMIT = 1024
_PROJECTION_SOURCE_LIMIT = 1024


class LabelingStateConflictError(RuntimeError):
    pass


class LabelingCancellationRequested(LabelingStateConflictError):
    pass


@dataclass(frozen=True, slots=True)
class RunLease:
    run_id: UUID
    generation: int
    token: UUID


@dataclass(frozen=True, slots=True)
class RecoverableRun:
    run_id: UUID
    generation: int


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    item_id: UUID
    request_hash: str
    request_payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class PreparedShard:
    ordinal: int
    item_ids: tuple[UUID, ...]
    submission_key: str


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _is_aggregate(row: RegulatoryChunk) -> bool:
    return (
        row.chunk_type == HIERARCHICAL_AGGREGATE_CHUNK_VARIANT
        or row.chunk_metadata.get("chunk_variant")
        == HIERARCHICAL_AGGREGATE_CHUNK_VARIANT
    )


def _is_atomic(row: RegulatoryChunk) -> bool:
    variant = row.chunk_metadata.get("chunk_variant")
    if variant == ATOMIC_CHUNK_VARIANT:
        return True
    if _is_aggregate(row):
        return False
    try:
        return not canonical_dependency_ids(row)
    except ValueError:
        return False


def _atomic_sql_expression() -> ColumnElement[bool]:
    variant = RegulatoryChunk.chunk_metadata["chunk_variant"].as_string()
    binding = RegulatoryChunk.chunk_metadata["bound_to_regulatory_chunk_id"].as_string()
    source_value = RegulatoryChunk.chunk_metadata["source_regulatory_chunk_ids"]
    source_count = case(
        (
            func.jsonb_typeof(source_value) == "array",
            func.jsonb_array_length(source_value),
        ),
        (source_value.is_(None), 0),
        else_=1,
    )
    return or_(
        variant == ATOMIC_CHUNK_VARIANT,
        and_(
            variant.is_(None),
            or_(
                RegulatoryChunk.chunk_type.is_(None),
                RegulatoryChunk.chunk_type != HIERARCHICAL_AGGREGATE_CHUNK_VARIANT,
            ),
            binding.is_(None),
            source_count == 0,
        ),
    )


def _snapshot_position(snapshot: Mapping[str, object]) -> int:
    value = snapshot.get("position")
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("Labeling source snapshot has no integer position")
    return value


def _snapshot_context_window(snapshot: Mapping[str, object]) -> tuple[int, int]:
    lower = snapshot.get("context_lower_position")
    upper = snapshot.get("context_upper_position")
    if (
        not isinstance(lower, int)
        or isinstance(lower, bool)
        or not isinstance(upper, int)
        or isinstance(upper, bool)
        or lower > upper
    ):
        raise ValueError("Labeling source snapshot has no valid context window")
    return lower, upper


def _snapshot_string_list(snapshot: Mapping[str, object], key: str) -> list[str]:
    value = snapshot.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Labeling source snapshot has invalid {key}")
    return type_cast(list[str], value)


def _chunk_view(row: RegulatoryChunk) -> LabelingChunkView:
    return LabelingChunkView(
        id=row.id,
        position=row.position,
        heading_path=tuple(row.heading_path),
        text=row.text,
    )


def _snapshot_date(snapshot: Mapping[str, object], key: str) -> datetime.date | None:
    value = snapshot.get(key)
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    if isinstance(value, str):
        return datetime.date.fromisoformat(value)
    raise ValueError(f"Labeling source snapshot has invalid {key}")


def _bounded_snapshot_context_items(
    rows: Sequence[RegulatoryLabelingItem],
    target: RegulatoryLabelingItem,
) -> list[RegulatoryLabelingItem]:
    reference_date = context_reference_date(
        _snapshot_date(target.source_snapshot, "validity_start_date"),
        _snapshot_date(target.source_snapshot, "validity_end_date"),
    )
    candidates_by_position: dict[int, list[RegulatoryLabelingItem]] = defaultdict(list)
    for candidate in rows:
        if validity_window_contains(
            _snapshot_date(candidate.source_snapshot, "validity_start_date"),
            _snapshot_date(candidate.source_snapshot, "validity_end_date"),
            reference_date,
        ):
            candidates_by_position[
                _snapshot_position(candidate.source_snapshot)
            ].append(candidate)

    target_position = _snapshot_position(target.source_snapshot)
    visible: list[RegulatoryLabelingItem] = []
    for position in sorted(set(candidates_by_position) | {target_position}):
        candidates = candidates_by_position.get(position, [])
        if position == target_position:
            visible.append(target)
        elif len(candidates) == 1:
            visible.append(candidates[0])
    visible.sort(
        key=lambda item: (
            _snapshot_position(item.source_snapshot),
            item.regulatory_chunk_id,
        )
    )
    if len(visible) <= _CONTEXT_NEIGHBOR_LIMIT + 2:
        return visible
    index = next(
        index
        for index, item in enumerate(visible)
        if item.regulatory_chunk_id == target.regulatory_chunk_id
    )
    radius = _CONTEXT_NEIGHBOR_LIMIT // 2
    nearby = visible[max(0, index - radius) : index + radius + 1]
    selected = {
        item.regulatory_chunk_id: item for item in [visible[0], *nearby, visible[-1]]
    }
    return sorted(
        selected.values(),
        key=lambda item: (
            _snapshot_position(item.source_snapshot),
            item.regulatory_chunk_id,
        ),
    )


def _bounded_current_context_rows(
    rows: Sequence[RegulatoryChunk], target: RegulatoryChunk
) -> list[RegulatoryChunk]:
    visible = visible_regulatory_snapshot_for_target(rows, target)
    if len(visible) <= _CONTEXT_NEIGHBOR_LIMIT + 2:
        return visible
    index = next(index for index, row in enumerate(visible) if row.id == target.id)
    radius = _CONTEXT_NEIGHBOR_LIMIT // 2
    nearby = visible[max(0, index - radius) : index + radius + 1]
    selected = {row.id: row for row in [visible[0], *nearby, visible[-1]]}
    return sorted(selected.values(), key=lambda row: (row.position, row.id))


def _source_context_hash(rows: Sequence[RegulatoryChunk]) -> str:
    return context_hash(
        [
            {
                "id": row.id,
                "position": row.position,
                "heading_path": row.heading_path,
                "text_sha256": context_hash(row.text),
                "validity_start_date": row.validity_start_date,
                "validity_end_date": row.validity_end_date,
                "status": row.status,
            }
            for row in sorted(rows, key=lambda value: (value.position, value.id))
        ]
    )


def _bounded_context_for_target(
    rows: Sequence[RegulatoryChunk],
    target: RegulatoryChunk,
    advisory: str | None,
) -> str:
    visible = visible_regulatory_snapshot_for_target(rows, target)
    ordered = sorted(visible, key=lambda row: (row.position, row.id))
    index = next(index for index, row in enumerate(ordered) if row.id == target.id)
    selected = ordered
    if len(ordered) > _CONTEXT_NEIGHBOR_LIMIT + 2:
        radius = _CONTEXT_NEIGHBOR_LIMIT // 2
        nearby = ordered[max(0, index - radius) : index + radius + 1]
        selected = list(
            {row.id: row for row in [ordered[0], *nearby, ordered[-1]]}.values()
        )
    context = bounded_document_context(
        [_chunk_view(row) for row in selected],
        target_id=target.id,
        max_utf8_bytes=_CONTEXT_BYTE_LIMIT,
    )
    if advisory and advisory.strip():
        advisory_block = (
            "\n\n[Existing generated context; advisory only]\n" + advisory.strip()
        )
        available = _CONTEXT_BYTE_LIMIT - len(context.encode("utf-8"))
        if available > 0:
            context += advisory_block.encode("utf-8")[:available].decode(
                "utf-8", errors="ignore"
            )
    return context


def _active_rows_for_document_set(
    session: Session, document_set_id: int
) -> list[RegulatoryChunk]:
    return list(
        session.scalars(
            select(RegulatoryChunk)
            .join(
                DocumentSet__UserFile,
                DocumentSet__UserFile.user_file_id == RegulatoryChunk.user_file_id,
            )
            .join(UserFile, UserFile.id == RegulatoryChunk.user_file_id)
            .where(
                DocumentSet__UserFile.document_set_id == document_set_id,
                RegulatoryChunk.status == RegulatoryChunkStatus.ACTIVE.value,
            )
            .order_by(
                RegulatoryChunk.user_file_id,
                RegulatoryChunk.position,
                RegulatoryChunk.id,
            )
        ).all()
    )


def _latest_context_advisories(
    session: Session, chunk_ids: Sequence[str]
) -> dict[str, str]:
    if not chunk_ids:
        return {}
    rows = session.execute(
        select(RegulatoryIndexingItem)
        .where(
            RegulatoryIndexingItem.regulatory_chunk_id.in_(chunk_ids),
            RegulatoryIndexingItem.context.is_not(None),
            RegulatoryIndexingItem.status.in_(("CONTEXT_READY", "EMBEDDED")),
            RegulatoryIndexingItem.effective_end.is_(None),
        )
        .order_by(RegulatoryIndexingItem.updated_at.desc())
    ).scalars()
    result: dict[str, str] = {}
    for row in rows:
        if row.regulatory_chunk_id in result or not row.context:
            continue
        value = row.context.get("raw_contextual_text") or row.context.get(
            "contextual_text"
        )
        if isinstance(value, str) and value.strip():
            result[row.regulatory_chunk_id] = value
    return result


def taxonomy_summary(row: RegulatoryLabelTaxonomy) -> TaxonomySummary:
    return TaxonomySummary(
        id=str(row.id),
        name=row.name,
        version_hash=row.version_hash,
        label_count=row.label_count,
        created_at=row.created_at,
    )


def run_snapshot(row: RegulatoryLabelingRun) -> LabelingRunSnapshot:
    return LabelingRunSnapshot(
        id=str(row.id),
        document_set_id=row.document_set_id,
        taxonomy_id=str(row.taxonomy_id),
        taxonomy_name=row.taxonomy.name,
        model=row.model,
        status=row.status,
        stage=row.stage,
        total_chunks=row.total_chunks,
        completed_chunks=row.completed_chunks,
        failed_chunks=row.failed_chunks,
        stale_chunks=row.stale_chunks,
        derived_chunks=row.derived_chunks,
        unresolved_derived_chunks=row.unresolved_derived_chunks,
        created_at=row.created_at,
        updated_at=row.updated_at,
        finished_at=row.finished_at,
        error=row.error,
        cancel_requested=row.cancel_requested,
    )


def create_taxonomy(
    session: Session,
    *,
    taxonomy: TaxonomyDefinition,
    created_by_id: UUID | None,
) -> RegulatoryLabelTaxonomy:
    existing = session.scalar(
        select(RegulatoryLabelTaxonomy).where(
            RegulatoryLabelTaxonomy.version_hash == taxonomy.version_hash
        )
    )
    if existing is not None:
        return existing
    row = RegulatoryLabelTaxonomy(
        name=taxonomy.name,
        version_hash=taxonomy.version_hash,
        definition=taxonomy.model_dump(mode="json"),
        label_count=len(taxonomy.labels),
        created_by_id=created_by_id,
    )
    session.add(row)
    session.flush()
    return row


def list_taxonomies(session: Session) -> list[RegulatoryLabelTaxonomy]:
    return list(
        session.scalars(
            select(RegulatoryLabelTaxonomy)
            .order_by(
                RegulatoryLabelTaxonomy.created_at.desc(),
                RegulatoryLabelTaxonomy.id.desc(),
            )
            .limit(100)
        ).all()
    )


def get_or_create_taxonomy(
    session: Session, *, taxonomy: TaxonomyDefinition, created_by_id: UUID | None
) -> RegulatoryLabelTaxonomy:
    session.execute(
        pg_insert(RegulatoryLabelTaxonomy)
        .values(
            id=uuid4(),
            name=taxonomy.name,
            version_hash=taxonomy.version_hash,
            definition=taxonomy.model_dump(mode="json"),
            label_count=len(taxonomy.labels),
            created_by_id=created_by_id,
        )
        .on_conflict_do_nothing(index_elements=[RegulatoryLabelTaxonomy.version_hash])
    )
    return session.scalars(
        select(RegulatoryLabelTaxonomy).where(
            RegulatoryLabelTaxonomy.version_hash == taxonomy.version_hash
        )
    ).one()


def get_taxonomy(session: Session, taxonomy_id: UUID) -> RegulatoryLabelTaxonomy | None:
    return session.get(RegulatoryLabelTaxonomy, taxonomy_id)


def get_label_settings(session: Session) -> RegulatoryLabelSettings:
    return session.scalars(
        select(RegulatoryLabelSettings)
        .options(selectinload(RegulatoryLabelSettings.taxonomy))
        .where(RegulatoryLabelSettings.id == 1)
        .execution_options(populate_existing=True)
    ).one()


def update_label_settings(
    session: Session,
    *,
    labels: Sequence[LabelDefinition],
    expected_revision: int,
    updated_by_id: UUID,
) -> RegulatoryLabelSettings:
    settings = session.scalars(
        select(RegulatoryLabelSettings)
        .options(selectinload(RegulatoryLabelSettings.taxonomy))
        .where(RegulatoryLabelSettings.id == 1)
        .with_for_update(of=RegulatoryLabelSettings)
        .execution_options(populate_existing=True)
    ).one()
    if settings.revision != expected_revision:
        raise LabelingStateConflictError(
            "Label settings changed while you were editing. Reload the latest labels before saving."
        )
    definition = TaxonomyDefinition(name=settings.taxonomy.name, labels=list(labels))
    if definition.version_hash == settings.taxonomy.version_hash:
        return settings
    settings.taxonomy = get_or_create_taxonomy(
        session, taxonomy=definition, created_by_id=updated_by_id
    )
    settings.revision += 1
    settings.updated_by_id = updated_by_id
    settings.updated_at = datetime.datetime.now(datetime.timezone.utc)
    session.flush()
    return settings


def get_labeling_counts(
    session: Session, document_set_id: int
) -> tuple[LabelingCounts, list[str]]:
    atomic_expression = _atomic_sql_expression()
    file_count = (
        session.scalar(
            select(func.count(DocumentSet__UserFile.user_file_id)).where(
                DocumentSet__UserFile.document_set_id == document_set_id
            )
        )
        or 0
    )
    canonical_count, derived_count, covered_file_count = session.execute(
        select(
            func.count().filter(atomic_expression),
            func.count().filter(~atomic_expression),
            func.count(func.distinct(RegulatoryChunk.user_file_id)).filter(
                atomic_expression
            ),
        )
        .join(
            DocumentSet__UserFile,
            DocumentSet__UserFile.user_file_id == RegulatoryChunk.user_file_id,
        )
        .where(
            DocumentSet__UserFile.document_set_id == document_set_id,
            RegulatoryChunk.status == RegulatoryChunkStatus.ACTIVE.value,
        )
    ).one()
    unsupported = max(0, file_count - covered_file_count)
    warnings: list[str] = []
    if unsupported:
        warnings.append(
            f"{unsupported} file(s) have no active canonical regulatory chunks."
        )
    if not canonical_count:
        warnings.append("This document set has no canonical chunks to label.")
    return (
        LabelingCounts(
            files=file_count,
            canonical_chunks=canonical_count,
            derived_chunks=derived_count,
        ),
        warnings,
    )


def get_active_run_id(session: Session, document_set_id: int) -> UUID | None:
    return session.scalar(
        select(RegulatoryLabelingRun.id)
        .where(
            RegulatoryLabelingRun.document_set_id == document_set_id,
            RegulatoryLabelingRun.status.in_(_ACTIVE_RUN_STATUSES),
        )
        .order_by(RegulatoryLabelingRun.created_at.desc())
        .limit(1)
    )


def get_run_by_idempotency(
    session: Session, document_set_id: int, idempotency_key: UUID
) -> RegulatoryLabelingRun | None:
    return session.scalar(
        select(RegulatoryLabelingRun)
        .options(selectinload(RegulatoryLabelingRun.taxonomy))
        .where(
            RegulatoryLabelingRun.document_set_id == document_set_id,
            RegulatoryLabelingRun.idempotency_key == idempotency_key,
        )
    )


def get_run_for_delivery(
    session: Session, run_id: UUID
) -> RegulatoryLabelingRun | None:
    return session.get(RegulatoryLabelingRun, run_id)


def create_labeling_run(
    session: Session,
    *,
    document_set_id: int,
    taxonomy: RegulatoryLabelTaxonomy,
    model_configuration_id: int,
    model: str,
    provider_binding: dict[str, object],
    requested_by_id: UUID | None,
    idempotency_key: UUID,
    retry_of_id: UUID | None = None,
    uses_current_labels: bool = False,
) -> tuple[RegulatoryLabelingRun, bool]:
    document_set = session.scalar(
        select(DocumentSet).where(DocumentSet.id == document_set_id).with_for_update()
    )
    if document_set is None:
        raise ValueError("Document set not found")
    existing = session.scalar(
        select(RegulatoryLabelingRun)
        .options(selectinload(RegulatoryLabelingRun.taxonomy))
        .where(
            RegulatoryLabelingRun.document_set_id == document_set_id,
            RegulatoryLabelingRun.idempotency_key == idempotency_key,
        )
    )
    if existing is not None:
        same_taxonomy = (
            uses_current_labels and existing.uses_current_labels
        ) or existing.taxonomy_id == taxonomy.id
        if (
            not same_taxonomy
            or existing.provider_binding.get("model_configuration_id")
            != model_configuration_id
        ):
            raise LabelingStateConflictError(
                "The idempotency key was already used with different parameters"
            )
        return existing, False
    if get_active_run_id(session, document_set_id) is not None:
        raise LabelingStateConflictError(
            "A labeling run is already active for this document set"
        )

    file_ids = list(
        session.scalars(
            select(DocumentSet__UserFile.user_file_id)
            .where(DocumentSet__UserFile.document_set_id == document_set_id)
            .order_by(DocumentSet__UserFile.user_file_id)
        ).all()
    )
    atomic_expression = _atomic_sql_expression()
    frozen_rows = session.execute(
        select(RegulatoryChunk.id, atomic_expression.label("is_atomic"))
        .where(
            RegulatoryChunk.user_file_id.in_(file_ids),
            RegulatoryChunk.status == RegulatoryChunkStatus.ACTIVE.value,
        )
        .order_by(RegulatoryChunk.id)
        .with_for_update(of=RegulatoryChunk)
    ).all()
    atomic_ids = [row.id for row in frozen_rows if row.is_atomic]
    derived_ids = [row.id for row in frozen_rows if not row.is_atomic]
    if not atomic_ids:
        raise ValueError("This document set has no canonical chunks to label")
    snapshot_hash = context_hash(
        {
            "document_set_id": document_set_id,
            "taxonomy": taxonomy.version_hash,
            "provider_binding": provider_binding,
            "file_ids": [str(file_id) for file_id in file_ids],
            "canonical_chunk_ids": atomic_ids,
            "derived_chunk_ids": derived_ids,
        }
    )
    run = RegulatoryLabelingRun(
        document_set_id=document_set_id,
        taxonomy_id=taxonomy.id,
        model_configuration_id=model_configuration_id,
        requested_by_id=requested_by_id,
        retry_of_id=retry_of_id,
        idempotency_key=idempotency_key,
        uses_current_labels=uses_current_labels,
        model=model,
        provider_binding=provider_binding,
        file_ids=[str(file_id) for file_id in file_ids],
        snapshot_hash=snapshot_hash,
        total_chunks=len(atomic_ids),
    )
    session.add(run)
    session.flush()
    source_snapshot = func.jsonb_build_object(
        literal("position"),
        RegulatoryChunk.position,
        literal("heading_path"),
        RegulatoryChunk.heading_path,
        literal("validity_start_date"),
        RegulatoryChunk.validity_start_date,
        literal("validity_end_date"),
        RegulatoryChunk.validity_end_date,
        literal("chunk_type"),
        RegulatoryChunk.chunk_type,
        literal("chunk_metadata"),
        RegulatoryChunk.chunk_metadata,
    )
    item_source = select(
        func.gen_random_uuid(),
        literal(run.id),
        RegulatoryChunk.id,
        RegulatoryChunk.user_file_id,
        RegulatoryChunk.text,
        source_snapshot,
    ).where(
        RegulatoryChunk.id.in_(atomic_ids),
    )
    session.execute(
        insert(RegulatoryLabelingItem).from_select(
            [
                "id",
                "run_id",
                "regulatory_chunk_id",
                "user_file_id",
                "text_snapshot",
                "source_snapshot",
            ],
            item_source,
        )
    )
    run.total_chunks = (
        session.scalar(
            select(func.count()).where(RegulatoryLabelingItem.run_id == run.id)
        )
        or 0
    )
    if not run.total_chunks:
        raise ValueError("This document set has no canonical chunks to label")
    derived_source = select(
        func.gen_random_uuid(),
        literal(run.id),
        RegulatoryChunk.id,
        RegulatoryChunk.user_file_id,
        RegulatoryChunk.text,
        source_snapshot,
    ).where(
        RegulatoryChunk.id.in_(derived_ids),
    )
    session.execute(
        insert(RegulatoryDerivedLabelProjection).from_select(
            [
                "id",
                "run_id",
                "regulatory_chunk_id",
                "user_file_id",
                "text_snapshot",
                "source_snapshot",
            ],
            derived_source,
        )
    )
    # Store an exact content hash without requiring pgcrypto in the database.
    # The worker fills both hashes and bounded context in small pages.
    run.derived_chunks = (
        session.scalar(
            select(func.count()).where(
                RegulatoryDerivedLabelProjection.run_id == run.id
            )
        )
        or 0
    )
    session.flush()
    session.refresh(run)
    return run, True


def get_run(
    session: Session, *, document_set_id: int, run_id: UUID
) -> RegulatoryLabelingRun | None:
    return session.scalar(
        select(RegulatoryLabelingRun)
        .options(selectinload(RegulatoryLabelingRun.taxonomy))
        .where(
            RegulatoryLabelingRun.id == run_id,
            RegulatoryLabelingRun.document_set_id == document_set_id,
        )
    )


def list_runs(session: Session, document_set_id: int) -> list[RegulatoryLabelingRun]:
    return list(
        session.scalars(
            select(RegulatoryLabelingRun)
            .options(selectinload(RegulatoryLabelingRun.taxonomy))
            .where(RegulatoryLabelingRun.document_set_id == document_set_id)
            .order_by(RegulatoryLabelingRun.created_at.desc())
            .limit(100)
        ).all()
    )


def list_items(
    session: Session,
    *,
    document_set_id: int,
    run_id: UUID,
    offset: int,
    limit: int,
) -> LabelingItemsPage | None:
    run_exists = session.scalar(
        select(RegulatoryLabelingRun.id).where(
            RegulatoryLabelingRun.id == run_id,
            RegulatoryLabelingRun.document_set_id == document_set_id,
        )
    )
    if run_exists is None:
        return None
    total = (
        session.scalar(
            select(func.count()).where(RegulatoryLabelingItem.run_id == run_id)
        )
        or 0
    )
    rows = session.scalars(
        select(RegulatoryLabelingItem)
        .where(RegulatoryLabelingItem.run_id == run_id)
        .order_by(
            RegulatoryLabelingItem.user_file_id,
            RegulatoryLabelingItem.regulatory_chunk_id,
        )
        .offset(offset)
        .limit(limit)
    ).all()
    return LabelingItemsPage(
        items=[
            LabelingItemSnapshot(
                chunk_id=row.regulatory_chunk_id,
                file_id=str(row.user_file_id),
                status=row.status,
                labels=list(row.labels),
                error=row.error,
            )
            for row in rows
        ],
        total=total,
    )


def request_cancellation(
    session: Session, *, document_set_id: int, run_id: UUID
) -> RegulatoryLabelingRun | None:
    run = get_run(session, document_set_id=document_set_id, run_id=run_id)
    if run is None:
        return None
    if run.status not in _ACTIVE_RUN_STATUSES:
        return run
    run.cancel_requested = True
    run.next_retry_at = None
    session.flush()
    return run


def claim_run(
    session: Session,
    *,
    run_id: UUID,
    expected_generation: int,
    lease_seconds: int,
    now: datetime.datetime | None = None,
) -> RunLease | None:
    claimed_at = now or _utcnow()
    token = uuid4()
    generation = expected_generation + 1
    claimed = session.execute(
        update(RegulatoryLabelingRun)
        .where(
            RegulatoryLabelingRun.id == run_id,
            RegulatoryLabelingRun.status.in_(_ACTIVE_RUN_STATUSES),
            RegulatoryLabelingRun.lease_generation == expected_generation,
            or_(
                RegulatoryLabelingRun.lease_token.is_(None),
                RegulatoryLabelingRun.lease_expires_at <= claimed_at,
            ),
            or_(
                RegulatoryLabelingRun.next_retry_at.is_(None),
                RegulatoryLabelingRun.next_retry_at <= claimed_at,
            ),
        )
        .values(
            status="running",
            lease_generation=generation,
            lease_token=token,
            lease_expires_at=claimed_at + datetime.timedelta(seconds=lease_seconds),
            next_retry_at=None,
            updated_at=claimed_at,
        )
        .returning(RegulatoryLabelingRun.id)
    ).scalar_one_or_none()
    if claimed is None:
        return None
    return RunLease(run_id=run_id, generation=generation, token=token)


def _lease_run(session: Session, lease: RunLease) -> RegulatoryLabelingRun:
    run = session.scalar(
        select(RegulatoryLabelingRun)
        .options(selectinload(RegulatoryLabelingRun.taxonomy))
        .where(
            RegulatoryLabelingRun.id == lease.run_id,
            RegulatoryLabelingRun.lease_generation == lease.generation,
            RegulatoryLabelingRun.lease_token == lease.token,
        )
        .with_for_update()
    )
    if run is None:
        raise LabelingStateConflictError("The labeling run lease is no longer owned")
    return run


def load_claimed_run(session: Session, lease: RunLease) -> RegulatoryLabelingRun:
    return _lease_run(session, lease)


def get_claimed_run_requester(session: Session, lease: RunLease) -> User | None:
    run = _lease_run(session, lease)
    if run.requested_by_id is None:
        return None
    return session.get(User, run.requested_by_id)


def load_claimed_shard(
    session: Session, lease: RunLease, shard_id: UUID
) -> RegulatoryLabelingShard:
    _lease_run(session, lease)
    shard = session.get(RegulatoryLabelingShard, shard_id)
    if shard is None or shard.run_id != lease.run_id:
        raise LabelingStateConflictError("The labeling shard is outside the run")
    return shard


def release_run(
    session: Session,
    lease: RunLease,
    *,
    stage: str | None = None,
    status: str | None = None,
    retry_after_seconds: float | None = None,
    error: str | None = None,
    finished: bool = False,
) -> int:
    values: dict[str, object] = {
        "lease_token": None,
        "lease_expires_at": None,
        "updated_at": _utcnow(),
    }
    if stage is not None:
        values["stage"] = stage
    if status is not None:
        values["status"] = status
    if retry_after_seconds is not None:
        values["next_retry_at"] = _utcnow() + datetime.timedelta(
            seconds=retry_after_seconds
        )
    else:
        values["next_retry_at"] = None
    if error is not None:
        values["error"] = error[:4000]
    if finished:
        values["finished_at"] = _utcnow()
        values["stage"] = "finished"
    result = type_cast(
        CursorResult[Any],
        session.execute(
            update(RegulatoryLabelingRun)
            .where(
                RegulatoryLabelingRun.id == lease.run_id,
                RegulatoryLabelingRun.lease_generation == lease.generation,
                RegulatoryLabelingRun.lease_token == lease.token,
            )
            .values(**values)
        ),
    )
    count = result.rowcount
    if count != 1:
        raise LabelingStateConflictError("The labeling run lease is no longer owned")
    return lease.generation


def recoverable_runs(
    session: Session,
    *,
    now: datetime.datetime | None = None,
    limit: int = 100,
) -> list[RecoverableRun]:
    current = now or _utcnow()
    rows = session.execute(
        select(RegulatoryLabelingRun.id, RegulatoryLabelingRun.lease_generation)
        .where(
            RegulatoryLabelingRun.status.in_(_ACTIVE_RUN_STATUSES),
            or_(
                RegulatoryLabelingRun.next_retry_at.is_(None),
                RegulatoryLabelingRun.next_retry_at <= current,
            ),
            or_(
                RegulatoryLabelingRun.lease_token.is_(None),
                RegulatoryLabelingRun.lease_expires_at <= current,
            ),
        )
        .order_by(RegulatoryLabelingRun.updated_at)
        .limit(limit)
    ).all()
    return [RecoverableRun(run_id=row[0], generation=row[1]) for row in rows]


def pending_items(session: Session, lease: RunLease) -> list[RegulatoryLabelingItem]:
    _lease_run(session, lease)
    return list(
        session.scalars(
            select(RegulatoryLabelingItem)
            .where(
                RegulatoryLabelingItem.run_id == lease.run_id,
                RegulatoryLabelingItem.status == "pending",
            )
            .order_by(
                RegulatoryLabelingItem.user_file_id,
                RegulatoryLabelingItem.regulatory_chunk_id,
            )
        ).all()
    )


def prepare_next_item_page(
    session: Session,
    lease: RunLease,
    *,
    limit: int,
) -> list[RegulatoryLabelingItem]:
    """Freeze bounded contexts for one page without loading the whole corpus."""

    _lease_run(session, lease)
    position = cast(
        RegulatoryLabelingItem.source_snapshot["position"].as_string(), Integer
    )
    targets = list(
        session.scalars(
            select(RegulatoryLabelingItem)
            .where(
                RegulatoryLabelingItem.run_id == lease.run_id,
                RegulatoryLabelingItem.status == "pending",
                RegulatoryLabelingItem.request_hash.is_(None),
            )
            .order_by(
                RegulatoryLabelingItem.user_file_id, position, RegulatoryLabelingItem.id
            )
            .limit(limit)
        ).all()
    )
    if not targets:
        return []
    advisories = _latest_context_advisories(
        session, [target.regulatory_chunk_id for target in targets]
    )
    targets_by_file: dict[UUID, list[RegulatoryLabelingItem]] = defaultdict(list)
    for target in targets:
        targets_by_file[target.user_file_id].append(target)

    for file_id, file_targets in targets_by_file.items():
        target_positions = [
            _snapshot_position(target.source_snapshot) for target in file_targets
        ]
        lower = min(target_positions) - (_CONTEXT_NEIGHBOR_LIMIT // 2)
        upper = max(target_positions) + (_CONTEXT_NEIGHBOR_LIMIT // 2)
        nearby = list(
            session.scalars(
                select(RegulatoryLabelingItem)
                .where(
                    RegulatoryLabelingItem.run_id == lease.run_id,
                    RegulatoryLabelingItem.user_file_id == file_id,
                    position.between(lower, upper),
                )
                .order_by(position, RegulatoryLabelingItem.id)
                .limit(_CONTEXT_CANDIDATE_LIMIT + 1)
            ).all()
        )
        if len(nearby) > _CONTEXT_CANDIDATE_LIMIT:
            for target in file_targets:
                target.status = "failed"
                target.error = "The canonical context window is too ambiguous"
            continue
        first = session.scalar(
            select(RegulatoryLabelingItem)
            .where(
                RegulatoryLabelingItem.run_id == lease.run_id,
                RegulatoryLabelingItem.user_file_id == file_id,
            )
            .order_by(position, RegulatoryLabelingItem.id)
            .limit(1)
        )
        last = session.scalar(
            select(RegulatoryLabelingItem)
            .where(
                RegulatoryLabelingItem.run_id == lease.run_id,
                RegulatoryLabelingItem.user_file_id == file_id,
            )
            .order_by(position.desc(), RegulatoryLabelingItem.id.desc())
            .limit(1)
        )
        file_items = list(
            {
                item.regulatory_chunk_id: item
                for item in [
                    *([first] if first else []),
                    *nearby,
                    *([last] if last else []),
                ]
            }.values()
        )
        for target in file_targets:
            selected = _bounded_snapshot_context_items(file_items, target)
            views = [
                LabelingChunkView(
                    id=item.regulatory_chunk_id,
                    position=_snapshot_position(item.source_snapshot),
                    heading_path=tuple(
                        _snapshot_string_list(item.source_snapshot, "heading_path")
                    ),
                    text=item.text_snapshot,
                )
                for item in selected
            ]
            advisory = advisories.get(target.regulatory_chunk_id)
            context = bounded_document_context(
                views,
                target_id=target.regulatory_chunk_id,
                max_utf8_bytes=_CONTEXT_BYTE_LIMIT,
            )
            context += "\n[Context selection is bounded to frozen neighboring chunks.]"
            if advisory and advisory.strip():
                advisory_block = (
                    "\n\n[Existing generated context; advisory only]\n"
                    + advisory.strip()
                )
                available = _CONTEXT_BYTE_LIMIT - len(context.encode("utf-8"))
                if available > 0:
                    context += advisory_block.encode("utf-8")[:available].decode(
                        "utf-8", errors="ignore"
                    )
            member_fingerprint = [
                {
                    "id": item.regulatory_chunk_id,
                    "text_sha256": context_hash(item.text_snapshot),
                    "source_snapshot": item.source_snapshot,
                }
                for item in selected
            ]
            target.canonical_text_sha256 = context_hash(target.text_snapshot)
            target.context_snapshot = context
            target.context_sha256 = context_hash(
                {
                    "members": member_fingerprint,
                    "context": context,
                    "advisory_sha256": context_hash(advisory or ""),
                }
            )
            target.source_snapshot = {
                **target.source_snapshot,
                "context_lower_position": lower,
                "context_upper_position": upper,
                "context_member_ids": [item.regulatory_chunk_id for item in selected],
                "advisory_sha256": context_hash(advisory or ""),
            }
    session.flush()
    return [target for target in targets if target.status == "pending"]


def store_prepared_shards(
    session: Session,
    lease: RunLease,
    *,
    requests: Sequence[PreparedRequest],
    shards: Sequence[PreparedShard],
    failed_items: Mapping[UUID, str],
) -> str:
    run = _lease_run(session, lease)
    if run.cancel_requested:
        raise LabelingCancellationRequested("The labeling run was cancelled")
    if run.stage != "preparing":
        raise LabelingStateConflictError("The labeling run is no longer preparing")
    request_by_id = {request.item_id: request for request in requests}
    for item_id, error in failed_items.items():
        session.execute(
            update(RegulatoryLabelingItem)
            .where(
                RegulatoryLabelingItem.run_id == run.id,
                RegulatoryLabelingItem.id == item_id,
                RegulatoryLabelingItem.status == "pending",
            )
            .values(status="failed", error=error[:4000], updated_at=_utcnow())
        )
    for shard in shards:
        shard_row = RegulatoryLabelingShard(
            run_id=run.id,
            ordinal=shard.ordinal,
            item_ids=[str(item_id) for item_id in shard.item_ids],
            submission_key=shard.submission_key,
        )
        session.add(shard_row)
        session.flush()
        for item_id in shard.item_ids:
            request = request_by_id[item_id]
            session.execute(
                update(RegulatoryLabelingItem)
                .where(
                    RegulatoryLabelingItem.run_id == run.id,
                    RegulatoryLabelingItem.id == item_id,
                    RegulatoryLabelingItem.status == "pending",
                )
                .values(
                    request_hash=request.request_hash,
                    request_payload=request.request_payload,
                    shard_id=shard_row.id,
                    updated_at=_utcnow(),
                )
            )
    remaining = session.scalar(
        select(func.count()).where(
            RegulatoryLabelingItem.run_id == run.id,
            RegulatoryLabelingItem.status == "pending",
            RegulatoryLabelingItem.request_hash.is_(None),
        )
    )
    run.stage = "preparing" if remaining else "submitting"
    _refresh_counts(session, run)
    session.flush()
    return run.stage


def next_shard_ordinal(session: Session, lease: RunLease) -> int:
    _lease_run(session, lease)
    maximum = session.scalar(
        select(func.max(RegulatoryLabelingShard.ordinal)).where(
            RegulatoryLabelingShard.run_id == lease.run_id
        )
    )
    return int(maximum) + 1 if maximum is not None else 0


def set_claimed_run_stage(session: Session, lease: RunLease, stage: str) -> None:
    run = _lease_run(session, lease)
    run.stage = stage
    session.flush()


def next_due_shard(
    session: Session, lease: RunLease, *, max_in_flight: int
) -> RegulatoryLabelingShard | None:
    run = _lease_run(session, lease)
    now = _utcnow()
    in_flight = (
        session.scalar(
            select(func.count()).where(
                RegulatoryLabelingShard.run_id == run.id,
                RegulatoryLabelingShard.status.in_(
                    ("submitting", "reconcile_required", "submitted")
                ),
            )
        )
        or 0
    )
    submission_statuses = (
        ("prepared", "submitting", "reconcile_required")
        if in_flight < max_in_flight
        else ("submitting", "reconcile_required")
    )
    submission = session.scalar(
        select(RegulatoryLabelingShard)
        .where(
            RegulatoryLabelingShard.run_id == run.id,
            RegulatoryLabelingShard.status.in_(submission_statuses),
            or_(
                RegulatoryLabelingShard.next_retry_at.is_(None),
                RegulatoryLabelingShard.next_retry_at <= now,
            ),
        )
        .order_by(RegulatoryLabelingShard.ordinal)
        .limit(1)
    )
    if submission is not None:
        return submission
    return session.scalar(
        select(RegulatoryLabelingShard)
        .where(
            RegulatoryLabelingShard.run_id == run.id,
            RegulatoryLabelingShard.status == "submitted",
            or_(
                RegulatoryLabelingShard.next_retry_at.is_(None),
                RegulatoryLabelingShard.next_retry_at <= now,
            ),
        )
        .order_by(
            RegulatoryLabelingShard.next_retry_at, RegulatoryLabelingShard.ordinal
        )
        .limit(1)
    )


def load_shard_requests(
    session: Session, lease: RunLease, shard_id: UUID
) -> list[RegulatoryLabelingItem]:
    _lease_run(session, lease)
    return list(
        session.scalars(
            select(RegulatoryLabelingItem)
            .where(
                RegulatoryLabelingItem.run_id == lease.run_id,
                RegulatoryLabelingItem.shard_id == shard_id,
            )
            .order_by(RegulatoryLabelingItem.id)
        ).all()
    )


def next_cancellable_shard(
    session: Session, lease: RunLease
) -> RegulatoryLabelingShard | None:
    _lease_run(session, lease)
    return session.scalar(
        select(RegulatoryLabelingShard)
        .where(
            RegulatoryLabelingShard.run_id == lease.run_id,
            RegulatoryLabelingShard.status.not_in(("succeeded", "failed", "cancelled")),
        )
        .order_by(RegulatoryLabelingShard.ordinal)
        .limit(1)
    )


def mark_shard_submitting(
    session: Session,
    lease: RunLease,
    *,
    shard_id: UUID,
    reconcile_seconds: int,
) -> RegulatoryLabelingShard:
    run = _lease_run(session, lease)
    if run.cancel_requested:
        raise LabelingCancellationRequested("The labeling run was cancelled")
    shard = session.get(RegulatoryLabelingShard, shard_id)
    if shard is None or shard.run_id != lease.run_id or shard.status != "prepared":
        raise LabelingStateConflictError("The labeling shard is not prepared")
    shard.status = "submitting"
    shard.attempt_count += 1
    shard.reconcile_until = _utcnow() + datetime.timedelta(seconds=reconcile_seconds)
    shard.next_retry_at = None
    for item in load_shard_requests(session, lease, shard_id):
        item.status = "submitted"
    session.flush()
    return shard


def record_shard_state(
    session: Session,
    lease: RunLease,
    *,
    shard_id: UUID,
    status: str,
    remote_job_name: str | None = None,
    input_uri: str | None = None,
    output_uri: str | None = None,
    retry_after_seconds: float | None = None,
    error: str | None = None,
    increment_failure: bool = False,
) -> None:
    _lease_run(session, lease)
    shard = session.get(RegulatoryLabelingShard, shard_id)
    if shard is None or shard.run_id != lease.run_id:
        raise LabelingStateConflictError("The labeling shard is outside the run")
    shard.status = status
    shard.remote_job_name = remote_job_name or shard.remote_job_name
    shard.input_uri = input_uri or shard.input_uri
    shard.output_uri = output_uri or shard.output_uri
    shard.next_retry_at = (
        _utcnow() + datetime.timedelta(seconds=retry_after_seconds)
        if retry_after_seconds is not None
        else None
    )
    shard.error = error[:4000] if error else None
    if increment_failure:
        shard.failure_count += 1
    if status in ("failed", "cancelled"):
        item_status = "cancelled" if status == "cancelled" else "failed"
        session.execute(
            update(RegulatoryLabelingItem)
            .where(
                RegulatoryLabelingItem.run_id == lease.run_id,
                RegulatoryLabelingItem.shard_id == shard_id,
                RegulatoryLabelingItem.status.not_in(_TERMINAL_ITEM_STATUSES),
            )
            .values(status=item_status, error=shard.error, updated_at=_utcnow())
        )


def _current_file_atomics(
    session: Session, *, document_set_id: int, user_file_id: UUID
) -> list[RegulatoryChunk]:
    rows = list(
        session.scalars(
            select(RegulatoryChunk)
            .join(
                DocumentSet__UserFile,
                and_(
                    DocumentSet__UserFile.user_file_id == RegulatoryChunk.user_file_id,
                    DocumentSet__UserFile.document_set_id == document_set_id,
                ),
            )
            .where(
                RegulatoryChunk.user_file_id == user_file_id,
                RegulatoryChunk.status == RegulatoryChunkStatus.ACTIVE.value,
            )
            .order_by(RegulatoryChunk.position, RegulatoryChunk.id)
        ).all()
    )
    return [row for row in rows if _is_atomic(row)]


def _current_file_context_candidates(
    session: Session,
    *,
    document_set_id: int,
    user_file_id: UUID,
    lower: int,
    upper: int,
) -> tuple[list[RegulatoryChunk], frozenset[str], bool]:
    conditions = (
        DocumentSet__UserFile.document_set_id == document_set_id,
        RegulatoryChunk.user_file_id == user_file_id,
        RegulatoryChunk.status == RegulatoryChunkStatus.ACTIVE.value,
        _atomic_sql_expression(),
    )
    nearby = list(
        session.scalars(
            select(RegulatoryChunk)
            .join(
                DocumentSet__UserFile,
                DocumentSet__UserFile.user_file_id == RegulatoryChunk.user_file_id,
            )
            .where(*conditions, RegulatoryChunk.position.between(lower, upper))
            .order_by(RegulatoryChunk.position, RegulatoryChunk.id)
            .limit(_CONTEXT_CANDIDATE_LIMIT + 1)
        ).all()
    )
    if len(nearby) > _CONTEXT_CANDIDATE_LIMIT:
        return nearby[:_CONTEXT_CANDIDATE_LIMIT], frozenset(), False
    first = session.scalar(
        select(RegulatoryChunk)
        .join(
            DocumentSet__UserFile,
            DocumentSet__UserFile.user_file_id == RegulatoryChunk.user_file_id,
        )
        .where(*conditions)
        .order_by(RegulatoryChunk.position, RegulatoryChunk.id)
        .limit(1)
    )
    last = session.scalar(
        select(RegulatoryChunk)
        .join(
            DocumentSet__UserFile,
            DocumentSet__UserFile.user_file_id == RegulatoryChunk.user_file_id,
        )
        .where(*conditions)
        .order_by(RegulatoryChunk.position.desc(), RegulatoryChunk.id.desc())
        .limit(1)
    )
    boundaries = frozenset(row.id for row in (first, last) if row is not None)
    candidates = {
        row.id: row
        for row in [*nearby, *([first] if first else []), *([last] if last else [])]
    }
    return (
        sorted(candidates.values(), key=lambda row: (row.position, row.id)),
        boundaries,
        True,
    )


def _snapshot_matches_row(snapshot: Mapping[str, object], row: RegulatoryChunk) -> bool:
    return (
        snapshot.get("position") == row.position
        and snapshot.get("heading_path") == row.heading_path
        and snapshot.get("chunk_type") == row.chunk_type
        and snapshot.get("chunk_metadata") == row.chunk_metadata
        and snapshot.get("validity_start_date")
        == (row.validity_start_date.isoformat() if row.validity_start_date else None)
        and snapshot.get("validity_end_date")
        == (row.validity_end_date.isoformat() if row.validity_end_date else None)
    )


def _item_snapshot_is_current(
    item: RegulatoryLabelingItem,
    *,
    snapshot_items: Mapping[str, RegulatoryLabelingItem],
    current_rows: Mapping[str, RegulatoryChunk],
    current_file_rows: Mapping[UUID, Sequence[RegulatoryChunk]],
    current_boundary_ids: Mapping[UUID, frozenset[str]],
    current_windows_complete: Mapping[UUID, bool],
    advisory_hashes: Mapping[str, str],
) -> bool:
    try:
        member_ids = _snapshot_string_list(item.source_snapshot, "context_member_ids")
    except ValueError:
        return False
    current_target = current_rows.get(item.regulatory_chunk_id)
    if current_target is None:
        return False
    if not current_windows_complete.get(item.user_file_id, False):
        return False
    try:
        lower, upper = _snapshot_context_window(item.source_snapshot)
    except ValueError:
        return False
    boundary_ids = current_boundary_ids.get(item.user_file_id, frozenset())
    context_rows = [
        row
        for row in current_file_rows.get(item.user_file_id, ())
        if lower <= row.position <= upper or row.id in boundary_ids
    ]
    expected_member_ids = [
        row.id for row in _bounded_current_context_rows(context_rows, current_target)
    ]
    if member_ids != expected_member_ids:
        return False
    for member_id in member_ids:
        frozen = snapshot_items.get(member_id)
        current = current_rows.get(member_id)
        if (
            frozen is None
            or current is None
            or current.user_file_id != item.user_file_id
            or frozen.text_snapshot != current.text
            or not _snapshot_matches_row(frozen.source_snapshot, current)
        ):
            return False
    return advisory_hashes.get(
        item.regulatory_chunk_id, context_hash("")
    ) == item.source_snapshot.get("advisory_sha256")


def apply_shard_results(
    session: Session,
    lease: RunLease,
    *,
    shard_id: UUID,
    outcomes: Mapping[str, tuple[list[str], list[dict[str, object]]] | str],
) -> None:
    run = _lease_run(session, lease)
    if run.cancel_requested:
        raise LabelingCancellationRequested("The labeling run was cancelled")
    shard = session.get(RegulatoryLabelingShard, shard_id)
    if shard is None or shard.run_id != run.id:
        raise LabelingStateConflictError("The labeling shard is outside the run")
    items = load_shard_requests(session, lease, shard_id)
    required_ids = {
        member_id
        for item in items
        for member_id in _snapshot_string_list(
            item.source_snapshot, "context_member_ids"
        )
    }
    snapshot_items = {
        item.regulatory_chunk_id: item
        for item in session.scalars(
            select(RegulatoryLabelingItem)
            .options(
                load_only(
                    RegulatoryLabelingItem.regulatory_chunk_id,
                    RegulatoryLabelingItem.user_file_id,
                    RegulatoryLabelingItem.text_snapshot,
                    RegulatoryLabelingItem.source_snapshot,
                )
            )
            .where(
                RegulatoryLabelingItem.run_id == run.id,
                RegulatoryLabelingItem.regulatory_chunk_id.in_(required_ids),
            )
        ).all()
    }
    current_file_rows: dict[UUID, list[RegulatoryChunk]] = {}
    current_boundary_ids: dict[UUID, frozenset[str]] = {}
    current_windows_complete: dict[UUID, bool] = {}
    for file_id in {item.user_file_id for item in items}:
        file_items = [item for item in items if item.user_file_id == file_id]
        windows = [
            _snapshot_context_window(item.source_snapshot) for item in file_items
        ]
        rows, boundaries, complete = _current_file_context_candidates(
            session,
            document_set_id=run.document_set_id,
            user_file_id=file_id,
            lower=min(lower for lower, _upper in windows),
            upper=max(upper for _lower, upper in windows),
        )
        current_file_rows[file_id] = rows
        current_boundary_ids[file_id] = boundaries
        current_windows_complete[file_id] = complete
    current_by_id = {row.id: row for rows in current_file_rows.values() for row in rows}
    advisories = _latest_context_advisories(
        session, [item.regulatory_chunk_id for item in items]
    )
    for item in items:
        outcome = outcomes.get(item.request_hash or "")
        if outcome is None:
            item.status = "failed"
            item.error = "The provider output omitted this request"
            continue
        advisory_hashes = {
            chunk_id: context_hash(value) for chunk_id, value in advisories.items()
        }
        if not _item_snapshot_is_current(
            item,
            snapshot_items=snapshot_items,
            current_rows=current_by_id,
            current_file_rows=current_file_rows,
            current_boundary_ids=current_boundary_ids,
            current_windows_complete=current_windows_complete,
            advisory_hashes=advisory_hashes,
        ):
            item.status = "stale"
            item.error = "The canonical context changed before labels were applied"
            continue
        if isinstance(outcome, str):
            item.status = "failed"
            item.error = outcome[:4000]
            continue
        labels, assignments = outcome
        item.labels = list(dict.fromkeys(labels))
        item.assignments = assignments
        item.status = "completed"
        item.error = None
    shard.status = "succeeded"
    shard.error = None
    _refresh_counts(session, run)


def all_shards_terminal(session: Session, lease: RunLease) -> bool:
    _lease_run(session, lease)
    count = session.scalar(
        select(func.count()).where(
            RegulatoryLabelingShard.run_id == lease.run_id,
            RegulatoryLabelingShard.status.not_in(("succeeded", "failed", "cancelled")),
        )
    )
    return not count


def cancel_claimed_run(session: Session, lease: RunLease) -> None:
    run = _lease_run(session, lease)
    session.execute(
        update(RegulatoryLabelingItem)
        .where(
            RegulatoryLabelingItem.run_id == run.id,
            RegulatoryLabelingItem.status.not_in(_TERMINAL_ITEM_STATUSES),
        )
        .values(status="cancelled", error="Cancelled by administrator")
    )
    session.execute(
        update(RegulatoryLabelingShard)
        .where(
            RegulatoryLabelingShard.run_id == run.id,
            RegulatoryLabelingShard.status.not_in(("succeeded", "failed", "cancelled")),
        )
        .values(status="cancelled", error="Cancelled by administrator")
    )
    _refresh_counts(session, run)


def _refresh_counts(session: Session, run: RegulatoryLabelingRun) -> None:
    counts: dict[str, int] = {
        status: count
        for status, count in session.execute(
            select(RegulatoryLabelingItem.status, func.count())
            .where(RegulatoryLabelingItem.run_id == run.id)
            .group_by(RegulatoryLabelingItem.status)
        ).all()
    }
    run.completed_chunks = int(counts.get("completed", 0))
    run.failed_chunks = int(counts.get("failed", 0))
    run.stale_chunks = int(counts.get("stale", 0))


def _projection_item_query() -> Select[tuple[RegulatoryLabelingItem]]:
    return select(RegulatoryLabelingItem).options(
        load_only(
            RegulatoryLabelingItem.regulatory_chunk_id,
            RegulatoryLabelingItem.user_file_id,
            RegulatoryLabelingItem.canonical_text_sha256,
            RegulatoryLabelingItem.text_snapshot,
            RegulatoryLabelingItem.source_snapshot,
            RegulatoryLabelingItem.status,
            RegulatoryLabelingItem.labels,
            RegulatoryLabelingItem.assignments,
            RegulatoryLabelingItem.error,
        )
    )


def _explicit_projection_dependencies(
    target: RegulatoryDerivedLabelProjection,
) -> list[str]:
    raw_metadata = target.source_snapshot.get("chunk_metadata", {})
    if not isinstance(raw_metadata, dict):
        raise ValueError("invalid_explicit_lineage")
    metadata = type_cast(dict[str, object], raw_metadata)
    sources = metadata.get("source_regulatory_chunk_ids", [])
    binding = metadata.get("bound_to_regulatory_chunk_id")
    if not isinstance(sources, list) or not all(
        isinstance(value, str) for value in sources
    ):
        raise ValueError("invalid_explicit_lineage")
    if binding is not None and not isinstance(binding, str):
        raise ValueError("invalid_explicit_lineage")
    typed_sources = type_cast(list[str], sources)
    return list(dict.fromkeys([*typed_sources, *([binding] if binding else [])]))


def _current_projection_target(
    session: Session,
    *,
    run: RegulatoryLabelingRun,
    target: RegulatoryDerivedLabelProjection,
) -> RegulatoryChunk | None:
    row = session.scalar(
        select(RegulatoryChunk)
        .join(
            DocumentSet__UserFile,
            DocumentSet__UserFile.user_file_id == RegulatoryChunk.user_file_id,
        )
        .where(
            DocumentSet__UserFile.document_set_id == run.document_set_id,
            RegulatoryChunk.id == target.regulatory_chunk_id,
            RegulatoryChunk.status == RegulatoryChunkStatus.ACTIVE.value,
        )
    )
    if (
        row is None
        or row.user_file_id != target.user_file_id
        or _is_atomic(row)
        or row.text != target.text_snapshot
        or not _snapshot_matches_row(target.source_snapshot, row)
    ):
        return None
    return row


def _load_projection_item(
    session: Session,
    *,
    run_id: UUID,
    chunk_id: str,
) -> RegulatoryLabelingItem | None:
    query = _projection_item_query()
    return session.scalar(
        query.where(
            RegulatoryLabelingItem.run_id == run_id,
            RegulatoryLabelingItem.regulatory_chunk_id == chunk_id,
        )
    )


def _load_nested_projection(
    session: Session,
    *,
    run_id: UUID,
    chunk_id: str,
) -> RegulatoryDerivedLabelProjection | None:
    return session.scalar(
        select(RegulatoryDerivedLabelProjection).where(
            RegulatoryDerivedLabelProjection.run_id == run_id,
            RegulatoryDerivedLabelProjection.regulatory_chunk_id == chunk_id,
        )
    )


def _expand_projection_dependencies(
    session: Session,
    *,
    run: RegulatoryLabelingRun,
    dependency_ids: Sequence[str],
    user_file_id: UUID,
    item_cache: dict[str, RegulatoryLabelingItem | None],
    target_cache: dict[str, RegulatoryDerivedLabelProjection | None],
    seen: frozenset[str],
    visited: set[str],
    depth: int,
) -> tuple[str, ...]:
    if depth > 64:
        raise ValueError("explicit_lineage_too_deep")
    resolved: list[str] = []
    for dependency_id in dependency_ids:
        if dependency_id in seen:
            raise ValueError("cyclic_explicit_lineage")
        visited.add(dependency_id)
        if len(visited) > _PROJECTION_SOURCE_LIMIT:
            raise ValueError("explicit_lineage_too_broad")
        if dependency_id not in item_cache:
            item_cache[dependency_id] = _load_projection_item(
                session, run_id=run.id, chunk_id=dependency_id
            )
        item = item_cache[dependency_id]
        if item is not None:
            if item.user_file_id != user_file_id:
                raise ValueError("cross_file_explicit_dependency")
            resolved.append(dependency_id)
            continue
        if dependency_id not in target_cache:
            target_cache[dependency_id] = _load_nested_projection(
                session, run_id=run.id, chunk_id=dependency_id
            )
        nested = target_cache[dependency_id]
        if nested is None:
            raise ValueError("missing_explicit_dependency")
        if nested.user_file_id != user_file_id:
            raise ValueError("cross_file_explicit_dependency")
        if _current_projection_target(session, run=run, target=nested) is None:
            raise ValueError("nested_derived_target_changed")
        resolved.extend(
            _expand_projection_dependencies(
                session,
                run=run,
                dependency_ids=_explicit_projection_dependencies(nested),
                user_file_id=user_file_id,
                item_cache=item_cache,
                target_cache=target_cache,
                seen=seen | {dependency_id},
                visited=visited,
                depth=depth + 1,
            )
        )
    return tuple(dict.fromkeys(resolved))


def _legacy_projection_sources(
    session: Session,
    *,
    run_id: UUID,
    target: RegulatoryDerivedLabelProjection,
) -> tuple[tuple[str, ...], list[RegulatoryLabelingItem], str | None]:
    query = _projection_item_query()
    candidates = list(
        session.scalars(
            query.where(
                RegulatoryLabelingItem.run_id == run_id,
                RegulatoryLabelingItem.user_file_id == target.user_file_id,
                func.length(func.btrim(RegulatoryLabelingItem.text_snapshot)) >= 32,
                func.strpos(
                    literal(target.text_snapshot), RegulatoryLabelingItem.text_snapshot
                )
                > 0,
            )
            .order_by(RegulatoryLabelingItem.regulatory_chunk_id)
            .limit(_PROJECTION_SOURCE_LIMIT + 1)
        ).all()
    )
    if len(candidates) > _PROJECTION_SOURCE_LIMIT:
        return (), [], "legacy_containment_too_broad"
    ids_by_text: dict[str, list[str]] = defaultdict(list)
    for item in candidates:
        ids_by_text[item.text_snapshot].append(item.regulatory_chunk_id)
    if any(len(ids) != 1 for ids in ids_by_text.values()):
        return (), [], "ambiguous_legacy_containment"
    source_ids = tuple(sorted(ids[0] for ids in ids_by_text.values()))
    if not source_ids:
        return (), [], "no_legacy_containment"
    return source_ids, candidates, None


def _revalidate_projection_sources(
    session: Session,
    *,
    run: RegulatoryLabelingRun,
    source_items: Sequence[RegulatoryLabelingItem],
    already_revalidated: set[str],
) -> None:
    completed = [
        item
        for item in source_items
        if item.status == "completed"
        and item.regulatory_chunk_id not in already_revalidated
    ]
    if not completed:
        return
    required_ids = {
        member_id
        for item in completed
        for member_id in _snapshot_string_list(
            item.source_snapshot, "context_member_ids"
        )
    }
    query = _projection_item_query()
    snapshot_items = {
        item.regulatory_chunk_id: item
        for item in session.scalars(
            query.where(
                RegulatoryLabelingItem.run_id == run.id,
                RegulatoryLabelingItem.regulatory_chunk_id.in_(required_ids),
            )
        ).all()
    }
    advisories = _latest_context_advisories(
        session, [item.regulatory_chunk_id for item in completed]
    )
    advisory_hashes = {
        chunk_id: context_hash(value) for chunk_id, value in advisories.items()
    }
    window_cache: dict[
        tuple[UUID, int, int], tuple[list[RegulatoryChunk], frozenset[str], bool]
    ] = {}
    for item in completed:
        lower, upper = _snapshot_context_window(item.source_snapshot)
        key = (item.user_file_id, lower, upper)
        if key not in window_cache:
            window_cache[key] = _current_file_context_candidates(
                session,
                document_set_id=run.document_set_id,
                user_file_id=item.user_file_id,
                lower=lower,
                upper=upper,
            )
        rows, boundaries, complete = window_cache[key]
        current_rows = {row.id: row for row in rows}
        if not _item_snapshot_is_current(
            item,
            snapshot_items=snapshot_items,
            current_rows=current_rows,
            current_file_rows={item.user_file_id: rows},
            current_boundary_ids={item.user_file_id: boundaries},
            current_windows_complete={item.user_file_id: complete},
            advisory_hashes=advisory_hashes,
        ):
            item.status = "stale"
            item.labels = []
            item.assignments = []
            item.error = "The canonical source changed before label projection"
        already_revalidated.add(item.regulatory_chunk_id)


def project_derived_labels(
    session: Session, lease: RunLease, *, limit: int = 128
) -> int:
    if limit < 1:
        raise ValueError("projection page limit must be positive")
    run = _lease_run(session, lease)
    if run.cancel_requested:
        raise LabelingCancellationRequested("The labeling run was cancelled")
    targets = list(
        session.scalars(
            select(RegulatoryDerivedLabelProjection)
            .where(
                RegulatoryDerivedLabelProjection.run_id == run.id,
                RegulatoryDerivedLabelProjection.resolution == "pending",
            )
            .order_by(
                RegulatoryDerivedLabelProjection.user_file_id,
                RegulatoryDerivedLabelProjection.regulatory_chunk_id,
            )
            .limit(limit)
        ).all()
    )
    item_cache: dict[str, RegulatoryLabelingItem | None] = {}
    target_cache: dict[str, RegulatoryDerivedLabelProjection | None] = {
        target.regulatory_chunk_id: target for target in targets
    }
    revalidated_source_ids: set[str] = set()
    for target in targets:
        target.derived_text_sha256 = context_hash(target.text_snapshot)
        source_ids: tuple[str, ...] = ()
        source_items: list[RegulatoryLabelingItem] = []
        resolution_kind = "unresolved"
        unresolved_reason: str | None = None
        if _current_projection_target(session, run=run, target=target) is None:
            unresolved_reason = "derived_target_changed"
        else:
            try:
                dependencies = _explicit_projection_dependencies(target)
                if dependencies:
                    source_ids = _expand_projection_dependencies(
                        session,
                        run=run,
                        dependency_ids=dependencies,
                        user_file_id=target.user_file_id,
                        item_cache=item_cache,
                        target_cache=target_cache,
                        seen=frozenset({target.regulatory_chunk_id}),
                        visited=set(),
                        depth=0,
                    )
                    source_items = type_cast(
                        list[RegulatoryLabelingItem],
                        [item_cache[source_id] for source_id in source_ids],
                    )
                    resolution_kind = "lineage"
                else:
                    source_ids, source_items, unresolved_reason = (
                        _legacy_projection_sources(
                            session, run_id=run.id, target=target
                        )
                    )
                    if source_ids:
                        resolution_kind = "legacy_containment"
            except ValueError as error:
                unresolved_reason = str(error)
        _revalidate_projection_sources(
            session,
            run=run,
            source_items=source_items,
            already_revalidated=revalidated_source_ids,
        )
        available = [item for item in source_items if item.status == "completed"]
        if len(available) != len(source_ids) or not source_ids:
            available = []
            source_ids = ()
            resolution_kind = "unresolved"
            unresolved_reason = unresolved_reason or "source_label_result_unavailable"
        labels = sorted({label for item in available for label in item.labels})
        target.labels = labels
        provenance: dict[str, object] = {
            label: [
                {
                    "canonical_chunk_id": item.regulatory_chunk_id,
                    "canonical_text_sha256": item.canonical_text_sha256,
                    "assignments": [
                        assignment
                        for assignment in item.assignments
                        if assignment.get("label_id") == label
                    ],
                }
                for item in available
                if label in item.labels
            ]
            for label in labels
        }
        target.provenance = provenance
        target.resolution = resolution_kind
        target.unresolved_reason = unresolved_reason

    run.unresolved_derived_chunks = (
        session.scalar(
            select(func.count()).where(
                RegulatoryDerivedLabelProjection.run_id == run.id,
                RegulatoryDerivedLabelProjection.resolution == "unresolved",
            )
        )
        or 0
    )
    run.error = (
        f"{run.unresolved_derived_chunks} derived chunk(s) could not be projected safely"
        if run.unresolved_derived_chunks
        else None
    )
    _refresh_counts(session, run)
    return len(targets)


def has_pending_derived_projections(session: Session, lease: RunLease) -> bool:
    run = _lease_run(session, lease)
    return (
        session.scalar(
            select(RegulatoryDerivedLabelProjection.id)
            .where(
                RegulatoryDerivedLabelProjection.run_id == run.id,
                RegulatoryDerivedLabelProjection.resolution == "pending",
            )
            .limit(1)
        )
        is not None
    )


def final_status(session: Session, lease: RunLease) -> str:
    run = _lease_run(session, lease)
    _refresh_counts(session, run)
    if run.cancel_requested:
        return "cancelled"
    if run.completed_chunks == 0 and (run.failed_chunks or run.stale_chunks):
        return "failed"
    if run.failed_chunks or run.stale_chunks:
        return "completed_with_errors"
    if run.unresolved_derived_chunks:
        return "completed_with_errors"
    return "completed"
