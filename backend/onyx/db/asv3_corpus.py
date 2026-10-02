"""Read-only, ACL and publication fenced corpus access for ASv3."""

import json
from collections.abc import Callable, Generator
from dataclasses import dataclass
from datetime import date
from typing import Any
from uuid import UUID

from sqlalchemy import (
    JSON,
    Integer,
    Text,
    and_,
    case,
    cast,
    column,
    func,
    or_,
    select,
    true,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session

from onyx.access.access import get_access_for_user_files, get_acl_for_user
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import IndexFilters
from onyx.db.document_set import (
    filter_document_set_names_by_user_access,
    get_document_set_by_id_for_user,
)
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.models import (
    DocumentSet,
    DocumentSet__UserFile,
    FileRecord,
    Persona__UserFile,
    Project__UserFile,
    RegulatoryChunk,
    RegulatoryTemporalProjection,
    User,
    UserFile,
)
from onyx.db.regulatory_public_reads import (
    iter_public_temporal_bindings,
    qualified_file_ids,
    resolve_public_query_index,
)
from onyx.document_index.publication_models import PublicationIndexSnapshot
from onyx.regulatory.amendments.annexes.models import AnnexTemporalProjection
from onyx.regulatory.publication_reads import (
    filter_publication_read,
    observe_publication_read,
    require_publication_files,
)


class CorpusScopeUnavailable(ValueError):
    """A requested fence cannot safely be evaluated by the corpus broker."""


PC_CORPUS_NAME = "PC Külliyatı"


def resolve_pc_corpus_scope(
    session: Session, *, user: User, filters: IndexFilters
) -> IndexFilters:
    """Resolve and pin the active tenant's mandatory set without relaxing caller fences."""
    from shared_configs.contextvars import get_current_tenant_id

    tenant_id = get_current_tenant_id()
    if filters.tenant_id is not None and filters.tenant_id != tenant_id:
        raise PermissionError(
            "Tenant filter differs from the active authorized tenant."
        )
    if filters.source_type and any(
        value != DocumentSource.USER_FILE for value in filters.source_type
    ):
        raise CorpusScopeUnavailable(
            "ASv3 PC corpus requires the USER_FILE source type."
        )
    forced = filters.forced_document_set
    if forced and PC_CORPUS_NAME not in forced:
        raise CorpusScopeUnavailable(
            "Requested document sets do not include the mandatory PC corpus."
        )
    candidates = list(
        session.scalars(
            select(DocumentSet)
            .where(
                DocumentSet.name == PC_CORPUS_NAME, DocumentSet.is_deleting.is_(False)
            )
            .limit(2)
        )
    )
    if len(candidates) != 1:
        raise CorpusScopeUnavailable(
            "Mandatory PC corpus is missing or ambiguous in the current tenant."
        )
    corpus = candidates[0]
    if (
        filters.asv3_document_set_id is not None
        and filters.asv3_document_set_id != corpus.id
    ):
        raise PermissionError("Pinned PC corpus identity changed.")
    allowed = get_document_set_by_id_for_user(
        session, corpus.id, user, get_editable=False
    )
    if allowed is None or allowed.name != PC_CORPUS_NAME:
        raise PermissionError("Mandatory PC corpus is outside the authorized scope.")
    scoped = filters.model_copy(
        deep=True,
        update={
            "tenant_id": tenant_id,
            "asv3_document_set_id": corpus.id,
            "source_type": [DocumentSource.USER_FILE],
            "forced_document_set": [PC_CORPUS_NAME],
        },
    )
    requested = set(scoped.document_set or []) | {PC_CORPUS_NAME}
    if requested - filter_document_set_names_by_user_access(
        session, list(requested), user
    ):
        raise PermissionError("Document set is outside the authorized scope.")
    return scoped


def bind_pc_corpus_scope(*, user: User, filters: IndexFilters) -> IndexFilters:
    with get_session_with_current_tenant() as session:
        return resolve_pc_corpus_scope(session, user=user, filters=filters)


@dataclass(frozen=True)
class CorpusSource:
    id: UUID
    name: str
    file_id: str


@dataclass(frozen=True)
class CorpusChunk:
    id: str
    source_id: UUID
    text: str
    position: int
    projection_ordinal: int
    heading_path: tuple[str, ...]
    metadata: dict[str, Any]
    validity_start: date | None
    validity_end: date | None
    status: str


@dataclass(frozen=True)
class CorpusClosureRead:
    source: CorpusSource
    chunks: list[CorpusChunk]
    members: dict[str, tuple[str, ...]]
    complete: dict[str, bool]
    continuation: dict[str, tuple[str, ...]]
    outline_rows: int
    outline_truncated: bool


def _binding_chunk(
    source_id: UUID, binding: AnnexTemporalProjection, as_of: date
) -> CorpusChunk:
    payload = json.loads(binding.projection.source_json)
    return CorpusChunk(
        payload["regulatory_chunk_id"],
        source_id,
        binding.representation_text,
        binding.semantic_position,
        binding.projection.ordinal,
        tuple(payload.get("heading_path") or ()),
        {
            **payload,
            **binding.representation_metadata,
            "read_as_of_date": as_of.isoformat(),
        },
        binding.effective_start,
        binding.effective_end,
        "active",
    )


def iter_source_chunks_by_ids(
    session: Session,
    *,
    user: User,
    filters: IndexFilters,
    source_id: UUID,
    chunk_ids: tuple[str, ...],
    index: PublicationIndexSnapshot | None,
    check_active: Callable[[], None],
) -> Generator[CorpusChunk, None, None]:
    """Stream exact retained originals under the same live source/snapshot fences."""
    if not chunk_ids:
        return
    check_active()
    require_source(session, user=user, filters=filters, source_id=source_id)
    observation = observe_publication_read()
    effective_date = filters.as_of_date or date.today()
    qualified = source_id in qualified_file_ids(session, (source_id,))
    if qualified:
        if index is None:
            raise CorpusScopeUnavailable(
                "Source requires a verified query index snapshot."
            )
        bindings = iter_public_temporal_bindings(
            session,
            source_id,
            index=index,
            as_of_date=effective_date,
            canonical_chunk_ids=tuple(dict.fromkeys(chunk_ids)),
        )
        try:
            for binding in bindings:
                check_active()
                if binding.derived_role != "canonical":
                    continue
                chunk = _binding_chunk(source_id, binding, effective_date)
                if filter_publication_read(
                    observation, [chunk], lambda value: str(value.source_id)
                ):
                    yield chunk
        finally:
            bindings.close()
    else:
        statement = select(RegulatoryChunk).where(
            RegulatoryChunk.user_file_id == source_id,
            RegulatoryChunk.id.in_(chunk_ids),
            RegulatoryChunk.chunk_type.is_distinct_from("hierarchical_aggregate"),
        )
        if filters.as_of_date is None:
            statement = statement.where(RegulatoryChunk.status == "active")
        else:
            statement = statement.where(
                or_(
                    RegulatoryChunk.validity_start_date.is_(None),
                    RegulatoryChunk.validity_start_date <= effective_date,
                ),
                or_(
                    RegulatoryChunk.validity_end_date.is_(None),
                    RegulatoryChunk.validity_end_date > effective_date,
                ),
            )
        rows = session.scalars(
            statement.order_by(
                RegulatoryChunk.position, RegulatoryChunk.id
            ).execution_options(yield_per=16, populate_existing=True)
        )
        try:
            for row in rows:
                check_active()
                chunk = CorpusChunk(
                    row.id,
                    source_id,
                    row.text,
                    row.position,
                    row.projection_ordinal,
                    tuple(row.heading_path or ()),
                    {
                        **(row.chunk_metadata or {}),
                        "read_as_of_date": filters.as_of_date.isoformat()
                        if filters.as_of_date
                        else None,
                    },
                    row.validity_start_date,
                    row.validity_end_date,
                    row.status,
                )
                if filter_publication_read(
                    observation, [chunk], lambda value: str(value.source_id)
                ):
                    yield chunk
        finally:
            rows.close()
    check_active()
    require_source(session, user=user, filters=filters, source_id=source_id)
    require_publication_files(observation, (source_id,))


def read_search_source_closures(
    session: Session,
    *,
    user: User,
    filters: IndexFilters,
    source_id: UUID,
    center_ids: tuple[str, ...],
    index: PublicationIndexSnapshot | None,
    check_active: Callable[[], None],
    max_outline_rows: int = 20000,
    max_outline_bytes: int = 4 * 1024 * 1024,
    max_chars: int = 64000,
) -> CorpusClosureRead:
    """Plan local closures once from structure, then hydrate only their exact members."""
    from onyx.db.regulatory_chunks import (
        RegulatoryChunkSiblingCandidate,
        _candidate_article_no,
        _has_overlapping_visible_positions,
        _project_candidates,
        _provision_span_for_seed,
    )

    if min(max_outline_rows, max_outline_bytes, max_chars) < 1:
        raise ValueError("Closure read budgets must be positive.")
    check_active()
    source = require_source(session, user=user, filters=filters, source_id=source_id)
    observation = observe_publication_read()
    qualified = source_id in qualified_file_ids(session, (source_id,))
    effective_date = filters.as_of_date or date.today()
    if qualified and index is None:
        raise CorpusScopeUnavailable("Source requires a verified query index snapshot.")

    from onyx.db.asv3_candidate_inventory import (
        current_asv3_source_inventory_scope,
        iter_verified_asv3_inventory_members,
        read_asv3_candidate_inventory,
    )

    planning_inventory = None
    if qualified and current_asv3_source_inventory_scope() is not None:
        assert index is not None
        planning_inventory = read_asv3_candidate_inventory(
            session,
            source_id=source_id,
            index=index,
            as_of_date=effective_date,
            observation=observation,
        )

    def outline() -> Generator[RegulatoryChunkSiblingCandidate, None, None]:
        if qualified:
            assert index is not None
            if planning_inventory is not None:
                for row in planning_inventory:
                    check_active()
                    if row.derived_role != "canonical":
                        continue
                    metadata = row.closure_metadata
                    yield RegulatoryChunkSiblingCandidate(
                        regulatory_chunk_id=row.canonical_chunk_id,
                        user_file_id=source_id,
                        position=row.semantic_position,
                        text="",
                        status="active",
                        heading_path=row.heading_path,
                        article_no=str(metadata["article_no"])
                        if metadata.get("article_no") is not None
                        else None,
                        article_title=str(metadata["article_title"])
                        if metadata.get("article_title") is not None
                        else None,
                        chunk_type=metadata["chunk_type"]
                        if isinstance(metadata.get("chunk_type"), str)
                        else None,
                        paragraph_no=str(metadata["paragraph_no"])
                        if metadata.get("paragraph_no") is not None
                        else None,
                        clause_label=str(metadata["clause_label"])
                        if metadata.get("clause_label") is not None
                        else None,
                        validity_start_date=row.effective_start,
                        validity_end_date=row.effective_end,
                        projection_ordinal=row.ordinal,
                    )
                return
            bindings = iter_public_temporal_bindings(
                session, source_id, index=index, as_of_date=effective_date
            )
            try:
                for binding in bindings:
                    check_active()
                    if binding.derived_role != "canonical":
                        continue
                    payload = json.loads(binding.projection.source_json)
                    metadata = {**payload, **binding.representation_metadata}
                    yield RegulatoryChunkSiblingCandidate(
                        regulatory_chunk_id=payload["regulatory_chunk_id"],
                        user_file_id=source_id,
                        position=binding.semantic_position,
                        text="",
                        status="active",
                        heading_path=tuple(payload.get("heading_path") or ()),
                        article_no=str(metadata["article_no"])
                        if metadata.get("article_no") is not None
                        else None,
                        article_title=str(metadata["article_title"])
                        if metadata.get("article_title") is not None
                        else None,
                        chunk_type=str(metadata["chunk_type"])
                        if isinstance(metadata.get("chunk_type"), str)
                        else None,
                        paragraph_no=str(metadata["paragraph_no"])
                        if metadata.get("paragraph_no") is not None
                        else None,
                        clause_label=str(metadata["clause_label"])
                        if metadata.get("clause_label") is not None
                        else None,
                        validity_start_date=binding.effective_start,
                        validity_end_date=binding.effective_end,
                        projection_ordinal=binding.projection.ordinal,
                    )
                    # Frozen text/payload is never retained by the outline.
                    del payload, metadata
            finally:
                bindings.close()
            return
        statement = select(
            RegulatoryChunk.id,
            RegulatoryChunk.position,
            RegulatoryChunk.heading_path,
            RegulatoryChunk.chunk_metadata,
            RegulatoryChunk.chunk_type,
            RegulatoryChunk.status,
            RegulatoryChunk.validity_start_date,
            RegulatoryChunk.validity_end_date,
            RegulatoryChunk.projection_ordinal,
        ).where(
            RegulatoryChunk.user_file_id == source_id,
            RegulatoryChunk.chunk_type.is_distinct_from("hierarchical_aggregate"),
        )
        if filters.as_of_date is None:
            statement = statement.where(RegulatoryChunk.status == "active")
        else:
            statement = statement.where(
                or_(
                    RegulatoryChunk.validity_start_date.is_(None),
                    RegulatoryChunk.validity_start_date <= effective_date,
                ),
                or_(
                    RegulatoryChunk.validity_end_date.is_(None),
                    RegulatoryChunk.validity_end_date > effective_date,
                ),
            )
        rows = session.execute(
            statement.order_by(
                RegulatoryChunk.position, RegulatoryChunk.id
            ).execution_options(yield_per=128)
        )
        try:
            for row in rows:
                check_active()
                metadata = row.chunk_metadata or {}
                yield RegulatoryChunkSiblingCandidate(
                    regulatory_chunk_id=row.id,
                    user_file_id=source_id,
                    position=row.position,
                    text="",
                    status=row.status,
                    heading_path=tuple(row.heading_path or ()),
                    article_no=str(metadata["article_no"])
                    if metadata.get("article_no") is not None
                    else None,
                    article_title=str(metadata["article_title"])
                    if metadata.get("article_title") is not None
                    else None,
                    chunk_type=row.chunk_type,
                    paragraph_no=str(metadata["paragraph_no"])
                    if metadata.get("paragraph_no") is not None
                    else None,
                    clause_label=str(metadata["clause_label"])
                    if metadata.get("clause_label") is not None
                    else None,
                    validity_start_date=row.validity_start_date,
                    validity_end_date=row.validity_end_date,
                    projection_ordinal=row.projection_ordinal,
                )
        finally:
            rows.close()

    candidates = []
    outline_bytes = 0
    truncated = False
    stream = outline()
    try:
        for candidate in stream:
            size = len(repr(candidate).encode("utf-8"))
            if (
                len(candidates) >= max_outline_rows
                or outline_bytes + size > max_outline_bytes
            ):
                truncated = True
                break
            candidates.append(candidate)
            outline_bytes += size
    finally:
        stream.close()
    by_file, by_id = _project_candidates(candidates, as_of_date=filters.as_of_date)
    rows = by_file.get(source_id, [])
    members: dict[str, tuple[str, ...]] = {}
    proven: dict[str, bool] = {}
    for center in dict.fromkeys(center_ids):
        seed = by_id.get(center)
        if seed is None or truncated:
            members[center], proven[center] = (center,), False
            continue
        span = _provision_span_for_seed(
            rows, seed.structural_index, as_of_date=filters.as_of_date
        )
        selected_positions = {rows[i].position for i in span}
        overlap_indices = {
            i for i, row in enumerate(rows) if row.position in selected_positions
        }
        if _has_overlapping_visible_positions(
            rows, overlap_indices, as_of_date=filters.as_of_date
        ):
            raise CorpusScopeUnavailable(
                "Retrieved provision has overlapping visible versions."
            )
        members[center] = tuple(rows[i].regulatory_chunk_id for i in sorted(span))
        # Unidentified outer rows cannot be asserted to be complete continuation.
        uncertain_edge = any(
            0 <= edge < len(rows) and _candidate_article_no(rows[edge]) is None
            for edge in (min(span) - 1, max(span) + 1)
        )
        proven[center] = _candidate_article_no(seed) is not None and not uncertain_edge
    outline_rows = len(candidates)
    del candidates, by_file, by_id, rows
    requested = set(center_ids)
    requested.update(identifier for group in members.values() for identifier in group)

    def hydrate(identifiers: tuple[str, ...]) -> Generator[CorpusChunk, None, None]:
        if not identifiers:
            return
        if qualified:
            assert index is not None
            if planning_inventory is not None:
                requested_ids = set(identifiers)
                bindings = iter_verified_asv3_inventory_members(
                    session,
                    rows=tuple(
                        row
                        for row in planning_inventory
                        if row.canonical_chunk_id in requested_ids
                        and row.derived_role == "canonical"
                    ),
                    index=index,
                    as_of_date=effective_date,
                    observation=observation,
                )
            else:
                bindings = iter_public_temporal_bindings(
                    session,
                    source_id,
                    index=index,
                    as_of_date=effective_date,
                    canonical_chunk_ids=identifiers,
                )
            try:
                for binding in bindings:
                    check_active()
                    if binding.derived_role == "canonical":
                        yield _binding_chunk(source_id, binding, effective_date)
            finally:
                bindings.close()
            return
        statement = select(RegulatoryChunk).where(
            RegulatoryChunk.user_file_id == source_id,
            RegulatoryChunk.id.in_(identifiers),
            RegulatoryChunk.chunk_type.is_distinct_from("hierarchical_aggregate"),
        )
        if filters.as_of_date is None:
            statement = statement.where(RegulatoryChunk.status == "active")
        else:
            statement = statement.where(
                or_(
                    RegulatoryChunk.validity_start_date.is_(None),
                    RegulatoryChunk.validity_start_date <= effective_date,
                ),
                or_(
                    RegulatoryChunk.validity_end_date.is_(None),
                    RegulatoryChunk.validity_end_date > effective_date,
                ),
            )
        result = session.scalars(
            statement.order_by(
                RegulatoryChunk.position, RegulatoryChunk.id
            ).execution_options(yield_per=16)
        )
        try:
            for row in result:
                check_active()
                yield CorpusChunk(
                    row.id,
                    source_id,
                    row.text,
                    row.position,
                    row.projection_ordinal,
                    tuple(row.heading_path or ()),
                    {
                        **(row.chunk_metadata or {}),
                        "read_as_of_date": filters.as_of_date.isoformat()
                        if filters.as_of_date
                        else None,
                    },
                    row.validity_start_date,
                    row.validity_end_date,
                    row.status,
                )
        finally:
            result.close()

    chunks: list[CorpusChunk] = []
    characters = 0
    hydrated_ids: set[str] = set()
    for identifiers in (
        tuple(dict.fromkeys(center_ids)),
        tuple(requested - set(center_ids)),
    ):
        texts = hydrate(identifiers)
        try:
            for chunk in texts:
                if chunk.id in hydrated_ids:
                    raise CorpusScopeUnavailable(
                        "Retrieved canonical center has ambiguous visible bindings."
                    )
                if characters + len(chunk.text) > max_chars:
                    break
                chunks.append(chunk)
                hydrated_ids.add(chunk.id)
                characters += len(chunk.text)
        finally:
            texts.close()
    check_active()
    chunks = filter_publication_read(
        observation, chunks, lambda row: str(row.source_id)
    )
    require_source(session, user=user, filters=filters, source_id=source_id)
    require_publication_files(observation, (source_id,))
    present = {chunk.id for chunk in chunks}
    continuation = {
        center: tuple(identifier for identifier in group if identifier not in present)
        for center, group in members.items()
    }
    return CorpusClosureRead(
        source,
        chunks,
        members,
        {center: proven[center] and not continuation[center] for center in members},
        continuation,
        outline_rows,
        truncated,
    )


def source_chunk_position(
    session: Session,
    *,
    source_id: UUID,
    chunk_id: str,
    index: PublicationIndexSnapshot | None = None,
    as_of: date | None = None,
) -> int | None:
    if index is not None:
        return session.scalar(
            select(
                cast(
                    RegulatoryTemporalProjection.payload["semantic_position"].astext,
                    Integer,
                )
            )
            .where(
                RegulatoryTemporalProjection.user_file_id == source_id,
                RegulatoryTemporalProjection.canonical_chunk_id == chunk_id,
                RegulatoryTemporalProjection.index_uuid == index.index_uuid,
                RegulatoryTemporalProjection.retired_at.is_(None),
                RegulatoryTemporalProjection.payload["derived_role"].astext
                == "canonical",
                or_(
                    RegulatoryTemporalProjection.effective_start.is_(None),
                    RegulatoryTemporalProjection.effective_start
                    <= (as_of or date.today()),
                ),
                or_(
                    RegulatoryTemporalProjection.effective_end.is_(None),
                    RegulatoryTemporalProjection.effective_end
                    > (as_of or date.today()),
                ),
            )
            .limit(1)
        )
    return session.scalar(
        select(RegulatoryChunk.position).where(
            RegulatoryChunk.user_file_id == source_id, RegulatoryChunk.id == chunk_id
        )
    )


def source_sibling_ids(
    session: Session,
    *,
    user: User,
    filters: IndexFilters,
    seed: CorpusChunk,
    index: PublicationIndexSnapshot | None,
) -> tuple[str, ...]:
    """Select all immediate-parent siblings from metadata, without loading text."""
    source_id = seed.source_id
    require_source(session, user=user, filters=filters, source_id=source_id)
    observation = observe_publication_read()
    as_of = filters.as_of_date or date.today()
    parent = seed.heading_path[:-1]
    qualified = source_id in qualified_file_ids(session, (source_id,))
    if qualified:
        if index is None:
            raise CorpusScopeUnavailable(
                "Local context requires a verified query index snapshot."
            )
        identifier = RegulatoryTemporalProjection.canonical_chunk_id
        position = cast(
            RegulatoryTemporalProjection.payload["semantic_position"].astext, Integer
        )
        ordinal = RegulatoryTemporalProjection.projection_ordinal
        parent_fields = (
            func.json_to_record(
                cast(
                    RegulatoryTemporalProjection.payload["projection"][
                        "source_json"
                    ].astext,
                    JSON,
                )
            )
            .table_valued(column("heading_path", JSON))
            .render_derived(with_types=True)
            .lateral("parent_fields")
        )
        headings = cast(parent_fields.c.heading_path, JSONB)
        statement = (
            select(identifier)
            .join(
                RegulatoryChunk,
                and_(
                    RegulatoryChunk.id == identifier,
                    RegulatoryChunk.user_file_id == source_id,
                ),
            )
            .join(parent_fields, true())
            .where(
                RegulatoryTemporalProjection.user_file_id == source_id,
                RegulatoryTemporalProjection.index_uuid == index.index_uuid,
                RegulatoryTemporalProjection.retired_at.is_(None),
                RegulatoryTemporalProjection.payload["derived_role"].astext
                == "canonical",
                or_(
                    RegulatoryChunk.validity_start_date.is_(None),
                    RegulatoryChunk.validity_start_date <= as_of,
                ),
                or_(
                    RegulatoryChunk.validity_end_date.is_(None),
                    RegulatoryChunk.validity_end_date > as_of,
                ),
                or_(
                    RegulatoryTemporalProjection.effective_start.is_(None),
                    RegulatoryTemporalProjection.effective_start <= as_of,
                ),
                or_(
                    RegulatoryTemporalProjection.effective_end.is_(None),
                    RegulatoryTemporalProjection.effective_end > as_of,
                ),
            )
        )
        parent_expression = headings.op("-", return_type=JSONB)(-1)
    else:
        identifier, position, ordinal = (
            RegulatoryChunk.id,
            RegulatoryChunk.position,
            RegulatoryChunk.projection_ordinal,
        )
        statement = select(identifier).where(
            RegulatoryChunk.user_file_id == source_id,
            RegulatoryChunk.chunk_type.is_distinct_from("hierarchical_aggregate"),
        )
        if filters.as_of_date is None:
            statement = statement.where(RegulatoryChunk.status == "active")
        else:
            statement = statement.where(
                or_(
                    RegulatoryChunk.validity_start_date.is_(None),
                    RegulatoryChunk.validity_start_date <= as_of,
                ),
                or_(
                    RegulatoryChunk.validity_end_date.is_(None),
                    RegulatoryChunk.validity_end_date > as_of,
                ),
            )
        headings = RegulatoryChunk.heading_path
        parent_expression = headings.op("-", return_type=JSONB)(-1)
    # An unknown heading is not evidence of a shared parent.
    parent_filter = (
        and_(func.jsonb_array_length(headings) > 0, parent_expression == list(parent))
        if seed.heading_path
        else identifier == seed.id
    )
    ids = tuple(
        session.scalars(
            statement.where(parent_filter).order_by(
                position, func.coalesce(ordinal, 0), identifier
            )
        )
    )
    require_source(session, user=user, filters=filters, source_id=source_id)
    require_publication_files(observation, (source_id,))
    return tuple(dict.fromkeys(ids))


def source_provision_position(
    session: Session,
    *,
    user: User,
    filters: IndexFilters,
    source_id: UUID,
    article: str,
    qualifier: str | None,
    index: PublicationIndexSnapshot | None,
) -> int | None:
    """Locate an article from fenced structural metadata without hydrating a source."""
    from onyx.regulatory.heading_path import parse_regulatory_article_heading

    if not article.isdecimal() or len(article) > 4:
        return None
    require_source(session, user=user, filters=filters, source_id=source_id)
    observation = observe_publication_read()
    as_of = filters.as_of_date or date.today()
    if source_id in qualified_file_ids(session, (source_id,)):
        if index is None:
            raise CorpusScopeUnavailable(
                "Provision locator requires a verified index snapshot."
            )
        headings = cast(
            RegulatoryTemporalProjection.payload["projection"]["source_json"].astext,
            JSONB,
        )["heading_path"]
        position = cast(
            RegulatoryTemporalProjection.payload["semantic_position"].astext, Integer
        )
        statement = select(position, headings).where(
            RegulatoryTemporalProjection.user_file_id == source_id,
            RegulatoryTemporalProjection.index_uuid == index.index_uuid,
            RegulatoryTemporalProjection.retired_at.is_(None),
            RegulatoryTemporalProjection.payload["derived_role"].astext == "canonical",
            or_(
                RegulatoryTemporalProjection.effective_start.is_(None),
                RegulatoryTemporalProjection.effective_start <= as_of,
            ),
            or_(
                RegulatoryTemporalProjection.effective_end.is_(None),
                RegulatoryTemporalProjection.effective_end > as_of,
            ),
        )
    else:
        headings, position = RegulatoryChunk.heading_path, RegulatoryChunk.position
        statement = select(position, headings).where(
            RegulatoryChunk.user_file_id == source_id,
            RegulatoryChunk.chunk_type.is_distinct_from("hierarchical_aggregate"),
        )
        if filters.as_of_date is None:
            statement = statement.where(RegulatoryChunk.status == "active")
        else:
            statement = statement.where(
                or_(
                    RegulatoryChunk.validity_start_date.is_(None),
                    RegulatoryChunk.validity_start_date <= as_of,
                ),
                or_(
                    RegulatoryChunk.validity_end_date.is_(None),
                    RegulatoryChunk.validity_end_date > as_of,
                ),
            )
    # SQL only narrows candidates; exact own-heading identity is checked below.
    pattern = rf"(madde[[:space:]]+{article}([^0-9]|$)|{article}[[:space:].]+madde)"
    rows = session.execute(
        statement.where(cast(headings, Text).op("~*")(pattern))
        .order_by(position)
        .limit(64)
    )
    result = None
    for row in rows:
        for heading in reversed(row[1] or []):
            own = parse_regulatory_article_heading(str(heading))
            if own:
                if (own.article_no, own.qualifier) == (article, qualifier):
                    result = int(row[0])
                break
        if result is not None:
            break
    require_source(session, user=user, filters=filters, source_id=source_id)
    require_publication_files(observation, (source_id,))
    return result


def _bounded_temporal_bindings(
    session: Session,
    source_id: UUID,
    index: PublicationIndexSnapshot,
    as_of: date,
    start: int,
    limit: int,
) -> list[AnnexTemporalProjection]:
    from onyx.db.regulatory_canonical_revisions import (
        validate_temporal_canonical_revisions,
    )
    from onyx.document_index.publication_models import (
        accepts_publication_projection,
        publication_digest,
    )

    position = cast(
        RegulatoryTemporalProjection.payload["semantic_position"].astext, Integer
    )
    rows = list(
        session.scalars(
            select(RegulatoryTemporalProjection)
            .join(
                RegulatoryChunk,
                RegulatoryChunk.id == RegulatoryTemporalProjection.canonical_chunk_id,
            )
            .where(
                RegulatoryTemporalProjection.user_file_id == source_id,
                RegulatoryTemporalProjection.index_uuid == index.index_uuid,
                RegulatoryTemporalProjection.retired_at.is_(None),
                RegulatoryTemporalProjection.payload["derived_role"].astext
                == "canonical",
                position >= start,
                or_(
                    RegulatoryTemporalProjection.effective_start.is_(None),
                    RegulatoryTemporalProjection.effective_start <= as_of,
                ),
                or_(
                    RegulatoryTemporalProjection.effective_end.is_(None),
                    RegulatoryTemporalProjection.effective_end > as_of,
                ),
                or_(
                    RegulatoryChunk.validity_start_date.is_(None),
                    RegulatoryChunk.validity_start_date <= as_of,
                ),
                or_(
                    RegulatoryChunk.validity_end_date.is_(None),
                    RegulatoryChunk.validity_end_date > as_of,
                ),
            )
            .order_by(position, RegulatoryTemporalProjection.projection_ordinal)
            .limit(limit)
        )
    )
    validate_temporal_canonical_revisions(session, rows)
    bindings = []
    for row in rows:
        if publication_digest(row.payload) != row.payload_sha256:
            raise CorpusScopeUnavailable("Temporal binding payload changed.")
        binding = AnnexTemporalProjection.model_validate(row.payload)
        if not binding.index.matches_temporal_index(
            index
        ) or not accepts_publication_projection(index, binding.projection):
            raise CorpusScopeUnavailable(
                "Temporal binding has no accepted publication/encoder receipt."
            )
        bindings.append(binding)
    return bindings


def _validate_filters(session: Session, user: User, filters: IndexFilters) -> None:
    from shared_configs.contextvars import get_current_tenant_id

    if filters.tenant_id is not None and filters.tenant_id != get_current_tenant_id():
        raise PermissionError(
            "Tenant filter differs from the active authorized tenant."
        )
    if filters.regulatory_source_hint or filters.regulatory_lookup_heading:
        raise CorpusScopeUnavailable(
            "Named-source/heading scopes require indexed structural lookup."
        )
    if filters.regulatory_candidate_ids is not None:
        raise CorpusScopeUnavailable("Candidate-only scopes require indexed search.")
    if filters.tags or filters.created_at_range or filters.updated_at_range:
        raise CorpusScopeUnavailable("Tag/time metadata scopes require indexed search.")
    if filters.hierarchy_node_ids:
        raise CorpusScopeUnavailable("Hierarchy scopes require indexed search.")
    if filters.source_type and any(
        value != DocumentSource.USER_FILE for value in filters.source_type
    ):
        raise CorpusScopeUnavailable("This broker reads regulatory user files only.")
    if filters.asv3_document_set_id is None or filters.forced_document_set != [
        PC_CORPUS_NAME
    ]:
        raise CorpusScopeUnavailable("Mandatory PC corpus scope has not been bound.")
    pinned = get_document_set_by_id_for_user(
        session, filters.asv3_document_set_id, user, get_editable=False
    )
    if pinned is None or pinned.name != PC_CORPUS_NAME:
        raise PermissionError("Pinned PC corpus is missing, changed or inaccessible.")
    requested = set(filters.document_set or []) | set(filters.forced_document_set or [])
    allowed = filter_document_set_names_by_user_access(session, list(requested), user)
    if requested - allowed:
        raise PermissionError("Document set is outside the authorized scope.")


def _source_statement(filters: IndexFilters) -> Any:
    statement = select(UserFile.id, UserFile.name, UserFile.file_id)
    set_membership = select(DocumentSet__UserFile.user_file_id).join(
        DocumentSet, DocumentSet.id == DocumentSet__UserFile.document_set_id
    )
    if filters.asv3_document_set_id is not None:
        statement = statement.where(
            UserFile.id.in_(
                set_membership.where(
                    DocumentSet.id == filters.asv3_document_set_id,
                    DocumentSet.is_deleting.is_(False),
                )
            )
        )
    knowledge = []
    if filters.document_set:
        knowledge.append(
            UserFile.id.in_(
                set_membership.where(DocumentSet.name.in_(filters.document_set))
            )
        )
    if filters.attached_document_ids:
        knowledge.append(
            UserFile.id.in_([UUID(value) for value in filters.attached_document_ids])
        )
    if knowledge:
        statement = statement.where(or_(*knowledge))
    if filters.forced_document_set:
        statement = statement.where(
            UserFile.id.in_(
                set_membership.where(DocumentSet.name.in_(filters.forced_document_set))
            )
        )
    if filters.project_id_filter is not None:
        statement = statement.where(
            UserFile.id.in_(
                select(Project__UserFile.user_file_id).where(
                    Project__UserFile.project_id == filters.project_id_filter
                )
            )
        )
    if filters.persona_id_filter is not None:
        statement = statement.where(
            UserFile.id.in_(
                select(Persona__UserFile.user_file_id).where(
                    Persona__UserFile.persona_id == filters.persona_id_filter
                )
            )
        )
    return statement


def find_sources(
    session: Session,
    *,
    user: User,
    filters: IndexFilters,
    query: str = "",
    source_ids: tuple[UUID, ...] | None = None,
    offset: int = 0,
    limit: int = 50,
) -> tuple[list[CorpusSource], bool]:
    """Page metadata before text hydration; every returned source passes actual ACL."""
    if offset < 0 or not 1 <= limit <= 100:
        raise ValueError("Source pages require offset >= 0 and limit 1..100.")
    _validate_filters(session, user, filters)
    statement = _source_statement(filters)
    if source_ids is not None:
        statement = statement.where(UserFile.id.in_(source_ids))
    # Corpus filenames commonly transliterate Turkish letters and use underscores.
    translation = str.maketrans("ÇĞİÖŞÜÂÎÛçğıöşüâîû", "CGIOSUAIUcgiosuaiu")
    normalized_query = query.translate(translation).lower().strip()
    normalized_name = func.lower(
        func.translate(UserFile.name, "ÇĞİÖŞÜÂÎÛçğıöşüâîû", "CGIOSUAIUcgiosuaiu")
    )
    order = []
    if normalized_query:
        escaped_terms = [
            term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            for term in normalized_query.split()[:12]
        ]
        for term in escaped_terms:
            statement = statement.where(
                normalized_name.ilike("%" + term + "%", escape="\\")
            )
        basename = func.regexp_replace(normalized_name, "^.*/", "")
        order = [
            case(
                (
                    and_(
                        *(
                            basename.ilike("%" + term + "%", escape="\\")
                            for term in escaped_terms
                        )
                    ),
                    0,
                ),
                else_=1,
            ),
            func.length(basename),
        ]
    records = session.execute(
        statement.order_by(*order, UserFile.id).offset(offset).limit(limit + 1)
    ).all()
    access = get_access_for_user_files([str(row.id) for row in records], session)
    user_acl = get_acl_for_user(user, session)
    sources = [
        CorpusSource(row.id, row.name, row.file_id)
        for row in records[:limit]
        if row.id and str(row.id) in access and access[str(row.id)].to_acl() & user_acl
    ]
    retained = filter_publication_read(
        observe_publication_read(), sources, lambda row: str(row.id)
    )
    return retained, len(records) > limit


def require_source(
    session: Session, *, user: User, filters: IndexFilters, source_id: UUID
) -> CorpusSource:
    rows, _ = find_sources(
        session, user=user, filters=filters, source_ids=(source_id,), limit=1
    )
    if not rows:
        raise PermissionError("Source is unavailable or outside the authorized scope.")
    require_publication_files(observe_publication_read(), (source_id,))
    return rows[0]


def resolve_source_query_index(
    session: Session, source_id: UUID
) -> PublicationIndexSnapshot | None:
    """Resolve only authorized source metadata; never hydrate corpus-wide text/vectors."""
    if source_id not in qualified_file_ids(session, (source_id,)):
        return None
    from onyx.db.search_settings import get_current_search_settings

    name = get_current_search_settings(session).index_name
    values = list(
        session.execute(
            select(
                RegulatoryTemporalProjection.index_uuid,
                RegulatoryTemporalProjection.payload["index"]["index_name"].astext,
            )
            .where(
                RegulatoryTemporalProjection.user_file_id == source_id,
                RegulatoryTemporalProjection.retired_at.is_(None),
                RegulatoryTemporalProjection.payload["index"]["index_name"].astext
                == name,
            )
            .distinct()
            .limit(3)
        )
    )
    if len(values) != 1:
        raise CorpusScopeUnavailable(
            "Source has missing or ambiguous query index authority."
        )
    index_uuid, name = values[0]
    return resolve_public_query_index(name, index_uuid, file_ids=(source_id,))


def read_source_chunks(
    session: Session,
    *,
    user: User,
    filters: IndexFilters,
    source_id: UUID,
    start: int = 0,
    limit: int = 50,
    query_indexes: dict[UUID, PublicationIndexSnapshot] | None = None,
    historical_inventory: bool = False,
) -> tuple[CorpusSource, list[CorpusChunk], bool]:
    if start < 0 or not 1 <= limit <= 100:
        raise ValueError("Chunk pages require start >= 0 and limit 1..100.")
    source = require_source(session, user=user, filters=filters, source_id=source_id)
    observation = observe_publication_read()
    if source_id in qualified_file_ids(session, (source_id,)):
        if historical_inventory:
            raise CorpusScopeUnavailable(
                "Version inventory requires retained publication bindings."
            )
        index = (query_indexes or {}).get(source_id)
        if index is None:
            raise CorpusScopeUnavailable(
                "Source requires a verified query index snapshot."
            )
        bindings = _bounded_temporal_bindings(
            session,
            source_id,
            index,
            filters.as_of_date or date.today(),
            start,
            limit + 1,
        )
        import json

        chunks = []
        for binding in sorted(bindings, key=lambda item: item.semantic_position):
            if binding.semantic_position < start:
                continue
            payload = json.loads(binding.projection.source_json)
            chunks.append(
                CorpusChunk(
                    payload["regulatory_chunk_id"],
                    source_id,
                    binding.representation_text,
                    binding.semantic_position,
                    binding.projection.ordinal,
                    tuple(payload.get("heading_path") or []),
                    {
                        **payload,
                        **binding.representation_metadata,
                        "read_as_of_date": (
                            filters.as_of_date or date.today()
                        ).isoformat(),
                    },
                    binding.effective_start,
                    binding.effective_end,
                    "active",
                )
            )
        has_more = len(chunks) > limit
        chunks = chunks[:limit]
    else:
        statement = select(RegulatoryChunk).where(
            RegulatoryChunk.user_file_id == source_id,
            RegulatoryChunk.position >= start,
            RegulatoryChunk.chunk_type.is_distinct_from("hierarchical_aggregate"),
        )
        if not historical_inventory:
            if filters.as_of_date is None:
                statement = statement.where(RegulatoryChunk.status == "active")
            else:
                statement = statement.where(
                    or_(
                        RegulatoryChunk.validity_start_date.is_(None),
                        RegulatoryChunk.validity_start_date <= filters.as_of_date,
                    ),
                    or_(
                        RegulatoryChunk.validity_end_date.is_(None),
                        RegulatoryChunk.validity_end_date > filters.as_of_date,
                    ),
                )
        rows = list(
            session.scalars(
                statement.order_by(RegulatoryChunk.position, RegulatoryChunk.id).limit(
                    limit + 1
                )
            )
        )
        has_more = len(rows) > limit
        chunks = [
            CorpusChunk(
                row.id,
                source_id,
                row.text,
                row.position,
                row.projection_ordinal,
                tuple(row.heading_path),
                {
                    **row.chunk_metadata,
                    "read_as_of_date": filters.as_of_date.isoformat()
                    if filters.as_of_date
                    else None,
                },
                row.validity_start_date,
                row.validity_end_date,
                row.status,
            )
            for row in rows[:limit]
        ]
    chunks = filter_publication_read(
        observation, chunks, lambda row: str(row.source_id)
    )
    require_publication_files(observation, (source_id,))
    return source, chunks, has_more


def source_diagnostic(
    session: Session, *, user: User, filters: IndexFilters, source_id: UUID
) -> dict[str, Any]:
    source = require_source(session, user=user, filters=filters, source_id=source_id)
    counts = session.execute(
        select(RegulatoryChunk.status, func.count(RegulatoryChunk.id))
        .where(RegulatoryChunk.user_file_id == source_id)
        .group_by(RegulatoryChunk.status)
    ).all()
    return {
        "source_id": str(source.id),
        "name": source.name,
        "canonical_status_counts": {
            str(status): int(count) for status, count in counts
        },
        "qualified_publication": source.id in qualified_file_ids(session, (source.id,)),
        "index_presence": "not_observed",
        "absence_proven": False,
    }


def original_source_record(
    session: Session, *, user: User, filters: IndexFilters, source_id: UUID
) -> tuple[CorpusSource, str]:
    from onyx.db.regulatory_original_ingestion import unavailable_original_file_ids

    source = require_source(session, user=user, filters=filters, source_id=source_id)
    if filters.as_of_date is not None or unavailable_original_file_ids(
        session, (source_id,)
    ):
        raise CorpusScopeUnavailable(
            "The original cannot prove this dated/versioned source; use retained canonical evidence."
        )
    record = session.get(FileRecord, source.file_id)
    if record is None:
        raise CorpusScopeUnavailable("Original source file record is unavailable.")
    return source, record.file_type


def research_code_enabled(session: Session) -> bool:
    from onyx.db.code_interpreter import fetch_code_interpreter_server

    return bool(fetch_code_interpreter_server(session).server_enabled)
