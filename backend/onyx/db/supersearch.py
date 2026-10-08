"""Bind Supersearch to the authorized tenant-local PC Külliyatı identity."""

import json
from collections.abc import Callable
from datetime import date
from typing import Any
from uuid import UUID

from sqlalchemy import Select, or_, select
from sqlalchemy.orm import Session

from onyx.context.search.models import IndexFilters
from onyx.db.asv3_corpus import (
    PC_CORPUS_NAME,
    CorpusScopeUnavailable,
    require_source,
    resolve_pc_corpus_scope,
    resolve_source_query_index,
)
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.models import RegulatoryChunk, User
from onyx.db.regulatory_public_reads import (
    iter_public_temporal_bindings,
    qualified_file_ids,
)
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.regulatory.publication_reads import (
    observe_publication_read,
    require_publication_files,
)

_MAX_SEARCH_CENTERS = 128
_MAX_AGGREGATE_MEMBERS = 4096


def bind_supersearch_pc_scope(
    *,
    user: User,
    filters: IndexFilters,
    document_set_names_override: list[str] | None = None,
) -> IndexFilters:
    """Keep caller narrowing inside a mandatory membership fence for every read.

    The bound corpus ID fences the broker's source IDs by actual database
    membership; the mandatory set name independently fences indexed searches.
    Caller knowledge, ACL, temporal, label and publication filters are retained.
    """
    for requested in (filters.forced_document_set, document_set_names_override):
        if requested is not None and PC_CORPUS_NAME not in requested:
            raise OnyxError(
                OnyxErrorCode.INVALID_INPUT,
                "Supersearch requires the PC Külliyatı document set; the requested "
                "scope does not intersect that corpus.",
            )
    try:
        with get_session_with_current_tenant() as session:
            return resolve_pc_corpus_scope(session, user=user, filters=filters)
    except PermissionError as error:
        raise OnyxError(
            OnyxErrorCode.INSUFFICIENT_PERMISSIONS,
            f"Supersearch PC Külliyatı scope is unavailable: {error}",
        ) from error
    except CorpusScopeUnavailable as error:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            f"Supersearch PC Külliyatı scope is unavailable: {error}",
        ) from error


def _stored_center_members(
    center_id: str, chunk_type: str | None, metadata: dict[str, Any]
) -> tuple[str, ...]:
    if chunk_type != "hierarchical_aggregate":
        return (center_id,)
    members = metadata.get("source_regulatory_chunk_ids")
    if (
        not isinstance(members, list)
        or not members
        or len(members) > _MAX_AGGREGATE_MEMBERS
    ):
        raise CorpusScopeUnavailable(
            "Supersearch aggregate has no complete bounded canonical membership."
        )
    canonical_members: list[str] = []
    for member in members:
        if not isinstance(member, str) or not member:
            raise CorpusScopeUnavailable(
                "Supersearch aggregate has invalid canonical member identities."
            )
        canonical_members.append(member)
    return tuple(dict.fromkeys(canonical_members))


def _all_center_members(resolved: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    members = dict.fromkeys(member for values in resolved.values() for member in values)
    if len(members) > _MAX_AGGREGATE_MEMBERS:
        raise CorpusScopeUnavailable(
            "Supersearch center batch exceeds the complete canonical membership limit."
        )
    return tuple(members)


def resolve_supersearch_center_ids(
    session: Session,
    *,
    user: User,
    filters: IndexFilters,
    source_id: UUID,
    center_ids: tuple[str, ...],
    check_active: Callable[[], None],
) -> dict[str, tuple[str, ...]]:
    """Resolve search aggregates from authoritative stored membership, without text.

    Children must all be visible atomic chunks of the same authorized PC source.
    Qualified files use accepted dated publication bindings; stale projected
    membership never grants access or supplies canonical evidence.
    """
    if not center_ids:
        return {}
    if len(center_ids) > _MAX_SEARCH_CENTERS:
        raise ValueError("Supersearch center reads require at most 128 identities.")
    check_active()
    require_source(session, user=user, filters=filters, source_id=source_id)
    observation = observe_publication_read()
    effective_date = filters.as_of_date or date.today()
    resolved: dict[str, tuple[str, ...]] = dict.fromkeys(center_ids, ())
    valid_children: set[str] = set()
    qualified = source_id in qualified_file_ids(session, (source_id,))
    if qualified:
        index = resolve_source_query_index(session, source_id)
        if index is None:
            raise CorpusScopeUnavailable(
                "Supersearch source requires a verified query index snapshot."
            )
        bindings = iter_public_temporal_bindings(
            session,
            source_id,
            index=index,
            as_of_date=effective_date,
            canonical_chunk_ids=center_ids,
        )
        try:
            for binding in bindings:
                check_active()
                payload = json.loads(binding.projection.source_json)
                identifier = payload["regulatory_chunk_id"]
                if not isinstance(identifier, str) or identifier not in resolved:
                    raise CorpusScopeUnavailable(
                        "Supersearch publication binding differs from its requested center."
                    )
                metadata = {**payload, **binding.representation_metadata}
                chunk_type = metadata.get("chunk_type")
                if chunk_type is not None and not isinstance(chunk_type, str):
                    raise CorpusScopeUnavailable(
                        "Supersearch publication center has an invalid chunk type."
                    )
                if (
                    chunk_type != "hierarchical_aggregate"
                    and binding.derived_role != "canonical"
                ):
                    continue
                members = _stored_center_members(identifier, chunk_type, metadata)
                if resolved.get(identifier) and resolved[identifier] != members:
                    raise CorpusScopeUnavailable(
                        "Supersearch center has ambiguous dated membership."
                    )
                resolved[identifier] = members
                if chunk_type != "hierarchical_aggregate":
                    valid_children.add(identifier)
        finally:
            bindings.close()
        children = tuple(
            member
            for member in _all_center_members(resolved)
            if member not in valid_children
        )
        bindings = iter_public_temporal_bindings(
            session,
            source_id,
            index=index,
            as_of_date=effective_date,
            canonical_chunk_ids=children,
        )
        try:
            for binding in bindings:
                check_active()
                if binding.derived_role == "canonical":
                    payload = json.loads(binding.projection.source_json)
                    metadata = {**payload, **binding.representation_metadata}
                    if metadata.get("chunk_type") != "hierarchical_aggregate":
                        valid_children.add(payload["regulatory_chunk_id"])
        finally:
            bindings.close()
    else:

        def visible_statement(
            identifiers: tuple[str, ...],
        ) -> Select[tuple[str, str | None, dict[str, Any]]]:
            statement = select(
                RegulatoryChunk.id,
                RegulatoryChunk.chunk_type,
                RegulatoryChunk.chunk_metadata,
            ).where(
                RegulatoryChunk.user_file_id == source_id,
                RegulatoryChunk.id.in_(identifiers),
            )
            if filters.as_of_date is None:
                return statement.where(RegulatoryChunk.status == "active")
            return statement.where(
                or_(
                    RegulatoryChunk.validity_start_date.is_(None),
                    RegulatoryChunk.validity_start_date <= effective_date,
                ),
                or_(
                    RegulatoryChunk.validity_end_date.is_(None),
                    RegulatoryChunk.validity_end_date > effective_date,
                ),
            )

        for row in session.execute(visible_statement(center_ids)):
            check_active()
            resolved[row.id] = _stored_center_members(
                row.id, row.chunk_type, row.chunk_metadata or {}
            )
            if row.chunk_type != "hierarchical_aggregate":
                valid_children.add(row.id)
        children = tuple(
            member
            for member in _all_center_members(resolved)
            if member not in valid_children
        )
        if children:
            valid_children.update(
                row.id
                for row in session.execute(
                    visible_statement(children).where(
                        RegulatoryChunk.chunk_type.is_distinct_from(
                            "hierarchical_aggregate"
                        )
                    )
                )
            )
    if any(set(members) - valid_children for members in resolved.values()):
        raise CorpusScopeUnavailable(
            "Supersearch aggregate membership is missing, dated out, or outside "
            "the authorized canonical source."
        )
    check_active()
    require_source(session, user=user, filters=filters, source_id=source_id)
    require_publication_files(observation, (source_id,))
    return resolved
