"""Uncitable ASv3 planning rows; originals require the public evidence reader."""

import json
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import date
from threading import RLock
from time import perf_counter
from typing import Any
from uuid import UUID

from sqlalchemy import Integer, Text, and_, cast, column, func, or_, select, true
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session

from onyx.db.models import RegulatoryChunk, RegulatoryTemporalProjection
from onyx.document_index.publication_models import (
    PublicationIndexSnapshot,
    ReadObservation,
)
from onyx.regulatory.amendments.annexes.models import AnnexTemporalProjection
from onyx.regulatory.publication_reads import require_publication_files
from onyx.tracing.answer_graph import graph_step


@dataclass(frozen=True, slots=True)
class ASv3CandidateInventoryRow:
    binding_id: UUID
    payload_sha256: str
    source_id: UUID
    canonical_chunk_id: str
    ordinal: int
    semantic_position: int
    text: str
    heading_path: tuple[str, ...]
    image_file_id: str | None
    effective_start: date | None
    effective_end: date | None
    derived_role: str
    canonical_metadata: dict[str, Any]
    closure_metadata: dict[str, Any]

    def verify(self, binding: AnnexTemporalProjection) -> None:
        source = json.loads(binding.projection.source_json)
        if (
            binding.id != self.binding_id
            or source["regulatory_chunk_id"] != self.canonical_chunk_id
            or source["document_id"] != str(self.source_id)
            or binding.projection.ordinal != self.ordinal
            or binding.semantic_position != self.semantic_position
            or binding.representation_text != self.text
            or tuple(source.get("heading_path") or ()) != self.heading_path
            or source.get("image_file_id") != self.image_file_id
            or binding.effective_start != self.effective_start
            or binding.effective_end != self.effective_end
            or binding.derived_role != self.derived_role
        ):
            raise ValueError("ASv3 planning inventory differs from verified original")
        metadata = {**source, **binding.representation_metadata}
        if any(
            metadata.get(key) != value for key, value in self.closure_metadata.items()
        ):
            raise ValueError("ASv3 planning structure differs from verified original")


@dataclass
class ASv3SourceInventoryScope:
    scope_key: str
    max_cache_bytes: int
    check_active: Callable[[], None]
    entries: dict[tuple[str, ...], tuple[ASv3CandidateInventoryRow, ...]] = field(
        default_factory=dict
    )
    retained_bytes: int = 0
    query_indexes: dict[UUID, PublicationIndexSnapshot] = field(default_factory=dict)
    lock: RLock = field(default_factory=RLock)


_CURRENT_SCOPE: ContextVar[ASv3SourceInventoryScope | None] = ContextVar(
    "asv3_source_inventory_scope", default=None
)


def current_asv3_source_inventory_scope() -> ASv3SourceInventoryScope | None:
    return _CURRENT_SCOPE.get()


@contextmanager
def asv3_source_inventory_scope(
    *,
    scope_key: str,
    max_cache_bytes: int = 8 * 1024 * 1024,
    check_active: Callable[[], None] = lambda: None,
) -> Iterator[ASv3SourceInventoryScope]:
    if not scope_key or max_cache_bytes < 0:
        raise ValueError("ASv3 source inventory requires a bounded authorized scope")
    scope = ASv3SourceInventoryScope(scope_key, max_cache_bytes, check_active)
    token = _CURRENT_SCOPE.set(scope)
    try:
        yield scope
    finally:
        _CURRENT_SCOPE.reset(token)
        with scope.lock:
            scope.entries.clear()
            scope.query_indexes.clear()
            scope.retained_bytes = 0


_STRUCTURAL_KEYS = (
    "article_no",
    "article_title",
    "paragraph_no",
    "clause_label",
    "chunk_type",
)


def read_asv3_candidate_inventory(
    session: Session,
    *,
    source_id: UUID,
    index: PublicationIndexSnapshot,
    as_of_date: date,
    observation: ReadObservation,
    canonical_chunk_ids: tuple[str, ...] | None = None,
) -> tuple[ASv3CandidateInventoryRow, ...]:
    """Keep full candidate breadth without reading vector/revision bodies into Python."""
    scope = current_asv3_source_inventory_scope()
    if scope is None:
        raise ValueError("Uncitable inventory requires an explicit ASv3 planning scope")
    key = (
        scope.scope_key,
        str(source_id),
        index.model_dump_json(),
        as_of_date.isoformat(),
        observation.model_dump_json(),
        json.dumps(sorted(canonical_chunk_ids))
        if canonical_chunk_ids is not None
        else "all",
    )
    scope.check_active()
    require_publication_files(observation, (source_id,))
    with scope.lock:
        previous_index = scope.query_indexes.get(source_id)
        if previous_index is not None and not previous_index.matches_temporal_index(
            index
        ):
            raise ValueError("ASv3 source physical index changed during one invocation")
        scope.query_indexes[source_id] = index
        cached = scope.entries.get(key)
        if cached is not None:
            require_publication_files(observation, (source_id,))
            return cached
        binding_fields = (
            func.jsonb_to_record(RegulatoryTemporalProjection.payload)
            .table_valued(
                column("semantic_position", Integer),
                column("representation_text", Text),
                column("derived_role", Text),
                column("index", JSONB),
                column("projection", JSONB),
                column("representation_metadata", JSONB),
            )
            .render_derived(with_types=True)
            .lateral("binding_fields")
        )
        projection_fields = (
            func.jsonb_to_record(binding_fields.c.projection)
            .table_valued(
                column("source_json", Text),
                column("embedding_config_json", Text),
                column("observed_index", JSONB),
            )
            .render_derived(with_types=True)
            .lateral("projection_fields")
        )
        source_fields = (
            func.jsonb_to_record(cast(projection_fields.c.source_json, JSONB))
            .table_valued(
                column("regulatory_chunk_id", Text),
                column("document_id", Text),
                column("heading_path", JSONB),
                column("image_file_id", Text),
                *[column(name, JSONB) for name in _STRUCTURAL_KEYS],
            )
            .render_derived(with_types=True)
            .lateral("source_fields")
        )
        canonical_fields = (
            func.jsonb_to_record(RegulatoryChunk.chunk_metadata)
            .table_valued(
                *[column(name, JSONB) for name in _STRUCTURAL_KEYS],
            )
            .render_derived(with_types=True)
            .lateral("canonical_fields")
        )
        representation_fields = (
            func.jsonb_to_record(binding_fields.c.representation_metadata)
            .table_valued(
                *[column(name, JSONB) for name in _STRUCTURAL_KEYS],
            )
            .render_derived(with_types=True)
            .lateral("representation_fields")
        )
        statement = (
            select(
                RegulatoryTemporalProjection.id,
                RegulatoryTemporalProjection.payload_sha256,
                RegulatoryTemporalProjection.canonical_chunk_id,
                RegulatoryTemporalProjection.projection_ordinal,
                RegulatoryTemporalProjection.effective_start,
                RegulatoryTemporalProjection.effective_end,
                binding_fields.c.semantic_position.label("semantic_position"),
                binding_fields.c.representation_text.label("representation_text"),
                binding_fields.c.derived_role.label("derived_role"),
                binding_fields.c.index.label("binding_index"),
                projection_fields.c.embedding_config_json.label(
                    "encoder_configuration"
                ),
                projection_fields.c.observed_index.label("observed_index"),
                source_fields.c.regulatory_chunk_id.label("source_chunk_id"),
                source_fields.c.document_id.label("source_document_id"),
                source_fields.c.heading_path.label("heading_path"),
                source_fields.c.image_file_id.label("image_file_id"),
                *[
                    canonical_fields.c[name].label("canonical_" + name)
                    for name in _STRUCTURAL_KEYS
                ],
                RegulatoryChunk.chunk_type.label("live_chunk_type"),
                RegulatoryChunk.heading_path.label("canonical_heading_path"),
                *[
                    source_fields.c[name].label("source_" + name)
                    for name in _STRUCTURAL_KEYS
                ],
                *[
                    representation_fields.c[name].label("representation_" + name)
                    for name in _STRUCTURAL_KEYS
                ],
                *[
                    binding_fields.c.representation_metadata.has_key(name).label(
                        "has_" + name
                    )
                    for name in _STRUCTURAL_KEYS
                ],
            )
            .select_from(RegulatoryTemporalProjection)
            .join(
                RegulatoryChunk,
                and_(
                    RegulatoryChunk.id
                    == RegulatoryTemporalProjection.canonical_chunk_id,
                    RegulatoryChunk.user_file_id == source_id,
                ),
            )
            .join(binding_fields, true())
            .join(projection_fields, true())
            .join(source_fields, true())
            .join(canonical_fields, true())
            .join(representation_fields, true())
            .where(
                RegulatoryTemporalProjection.user_file_id == source_id,
                RegulatoryTemporalProjection.index_uuid == index.index_uuid,
                RegulatoryTemporalProjection.retired_at.is_(None),
                or_(
                    RegulatoryTemporalProjection.effective_start.is_(None),
                    RegulatoryTemporalProjection.effective_start <= as_of_date,
                ),
                or_(
                    RegulatoryTemporalProjection.effective_end.is_(None),
                    RegulatoryTemporalProjection.effective_end > as_of_date,
                ),
                or_(
                    RegulatoryChunk.validity_start_date.is_(None),
                    RegulatoryChunk.validity_start_date <= as_of_date,
                ),
                or_(
                    RegulatoryChunk.validity_end_date.is_(None),
                    RegulatoryChunk.validity_end_date > as_of_date,
                ),
            )
            .order_by(RegulatoryTemporalProjection.projection_ordinal)
            .execution_options(stream_results=True, yield_per=128)
        )
        if canonical_chunk_ids is not None:
            statement = statement.where(
                RegulatoryTemporalProjection.canonical_chunk_id.in_(canonical_chunk_ids)
            )
        rows: list[ASv3CandidateInventoryRow] = []
        row_bytes = 0
        index_snapshots: dict[str, PublicationIndexSnapshot] = {}
        with graph_step(
            "asv3.source_inventory",
            {
                "source_id": str(source_id),
                "as_of_date": as_of_date.isoformat(),
                "index_uuid": index.index_uuid,
                "citable": False,
            },
        ) as step:
            started = perf_counter()
            result = session.execute(statement)
            execute_seconds = perf_counter() - started
            fetch_seconds = selection_seconds = 0.0
            iterator = iter(result)
            try:
                while True:
                    scope.check_active()
                    started = perf_counter()
                    try:
                        row = next(iterator)._mapping
                    except StopIteration:
                        fetch_seconds += perf_counter() - started
                        break
                    fetch_seconds += perf_counter() - started
                    started = perf_counter()
                    serialized_index = json.dumps(row["binding_index"], sort_keys=True)
                    binding_index = index_snapshots.get(serialized_index)
                    if binding_index is None:
                        binding_index = PublicationIndexSnapshot.model_validate(
                            row["binding_index"]
                        )
                        index_snapshots[serialized_index] = binding_index
                    if not binding_index.matches_temporal_index(index):
                        continue
                    config = row["encoder_configuration"]
                    if config is not None:
                        if not index.accepts_encoder_configuration(json.loads(config)):
                            raise ValueError(
                                "temporal binding encoder receipt is not accepted"
                            )
                    else:
                        observed = PublicationIndexSnapshot.model_validate(
                            row["observed_index"]
                        )
                        if not observed.matches_temporal_index(index):
                            raise ValueError(
                                "temporal observation index is not accepted"
                            )
                    if row["source_chunk_id"] != row["canonical_chunk_id"] or row[
                        "source_document_id"
                    ] != str(source_id):
                        raise ValueError("ASv3 planning source identity mismatch")
                    canonical = {
                        name: row["canonical_" + name] for name in _STRUCTURAL_KEYS
                    }
                    canonical["chunk_type"] = row["live_chunk_type"]
                    canonical["heading_path"] = row["canonical_heading_path"]
                    closure = {
                        name: row["representation_" + name]
                        if row["has_" + name]
                        else row["source_" + name]
                        for name in _STRUCTURAL_KEYS
                    }
                    candidate = ASv3CandidateInventoryRow(
                        row["id"],
                        row["payload_sha256"],
                        source_id,
                        row["canonical_chunk_id"],
                        row["projection_ordinal"],
                        int(row["semantic_position"]),
                        row["representation_text"],
                        tuple(row["heading_path"] or ()),
                        row["image_file_id"],
                        row["effective_start"],
                        row["effective_end"],
                        row["derived_role"],
                        canonical,
                        closure,
                    )
                    rows.append(candidate)
                    row_bytes += len(repr(candidate).encode()) + 512
                    selection_seconds += perf_counter() - started
            finally:
                result.close()
            step.output_value = {
                "rows": len(rows),
                "inventory_bytes": row_bytes,
                "sql_execute_seconds": execute_seconds,
                "fetch_seconds": fetch_seconds,
                "selection_seconds": selection_seconds,
                "full_payload_validations": 0,
                "cache_admitted": scope.retained_bytes + row_bytes
                <= scope.max_cache_bytes,
                "cache_limit_bytes": scope.max_cache_bytes,
            }
        scope.check_active()
        require_publication_files(observation, (source_id,))
        inventory = tuple(rows)
        if scope.retained_bytes + row_bytes <= scope.max_cache_bytes:
            scope.entries[key] = inventory
            scope.retained_bytes += row_bytes
    return inventory


def iter_verified_asv3_inventory_members(
    session: Session,
    *,
    rows: tuple[ASv3CandidateInventoryRow, ...],
    index: PublicationIndexSnapshot,
    as_of_date: date,
    observation: ReadObservation,
) -> Generator[AnnexTemporalProjection, None, None]:
    from onyx.db.regulatory_public_reads import iter_public_temporal_bindings

    if not rows:
        return
    source_id = rows[0].source_id
    if any(row.source_id != source_id for row in rows):
        raise ValueError("Selected planning originals must share one source")
    scope = current_asv3_source_inventory_scope()
    if scope is None:
        raise ValueError("Selected planning originals require their ASv3 scope")
    scope.check_active()
    require_publication_files(observation, (source_id,))
    by_ordinal = {row.ordinal: row for row in rows}
    canonical_by_id = {row.canonical_chunk_id: row.canonical_metadata for row in rows}
    current_metadata = {
        row.id: {
            **{name: row._mapping[name] for name in _STRUCTURAL_KEYS},
            "chunk_type": row.live_chunk_type,
            "heading_path": row.heading_path,
        }
        for row in session.execute(
            select(
                RegulatoryChunk.id,
                RegulatoryChunk.heading_path,
                RegulatoryChunk.chunk_type.label("live_chunk_type"),
                *[
                    RegulatoryChunk.chunk_metadata[name].label(name)
                    for name in _STRUCTURAL_KEYS
                ],
            ).where(
                RegulatoryChunk.user_file_id == source_id,
                RegulatoryChunk.id.in_(tuple(canonical_by_id)),
            )
        )
    }
    if current_metadata != canonical_by_id:
        raise ValueError("Selected ASv3 canonical selection metadata changed")
    seen: set[int] = set()
    with graph_step(
        "asv3.selected_original_validation",
        {
            "source_id": str(source_id),
            "selected_count": len(by_ordinal),
            "as_of_date": as_of_date.isoformat(),
        },
    ) as step:
        original_bytes = 0
        bindings = iter_public_temporal_bindings(
            session,
            source_id,
            index=index,
            as_of_date=as_of_date,
            projection_ordinals=tuple(by_ordinal),
            expected_payload_sha256={
                ordinal: row.payload_sha256 for ordinal, row in by_ordinal.items()
            },
        )
        try:
            for binding in bindings:
                scope.check_active()
                row = by_ordinal[binding.projection.ordinal]
                row.verify(binding)
                seen.add(binding.projection.ordinal)
                original_bytes += len(binding.projection.source_json.encode())
                yield binding
        finally:
            bindings.close()
            step.output_value = {
                "full_payload_validations": len(seen),
                "selected_source_bytes": original_bytes,
            }
    if seen != by_ordinal.keys():
        raise ValueError("Selected ASv3 original is no longer available")
    scope.check_active()
    require_publication_files(observation, (source_id,))
