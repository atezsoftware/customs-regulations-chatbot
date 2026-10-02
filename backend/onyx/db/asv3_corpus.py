"""Read-only, ACL and publication fenced corpus access for ASv3."""

from dataclasses import dataclass
from datetime import date
from typing import Any
from uuid import UUID

from sqlalchemy import Integer, cast, func, or_, select
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
    if query:
        for term in query.split()[:12]:
            statement = statement.where(
                UserFile.name.ilike(
                    "%" + term.replace("%", "\\%").replace("_", "\\_") + "%",
                    escape="\\",
                )
            )
    records = session.execute(
        statement.order_by(UserFile.id).offset(offset).limit(limit + 1)
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
