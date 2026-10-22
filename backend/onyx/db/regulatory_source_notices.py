"""Catalogue-only regulatory status reads within the ordinary corpus ACL scope."""

from typing import Literal, cast
from uuid import UUID

from pydantic import JsonValue
from sqlalchemy import Boolean, Table, func, inspect, or_, select
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session

from onyx.context.search.models import IndexFilters
from onyx.db.asv3_corpus import (
    PC_CORPUS_NAME,
    CorpusScopeUnavailable,
    require_source,
    resolve_pc_corpus_scope,
)
from onyx.db.models import RegulatorySourceNotice, User
from onyx.regulatory.legistation_metadata import LegistationMetadata
from onyx.regulatory.source_notice_identity import (
    notice_identity_queries,
    notice_title_alias,
)
from onyx.regulatory.source_notices import SourceNotice


def create_source_notice_registry(connection: Connection) -> bool:
    """Create only this new table, or refuse a previously created, drifted shape."""
    table = cast(Table, RegulatorySourceNotice.__table__)
    inspector = inspect(connection)
    if not inspector.has_table(table.name):
        table.create(connection, checkfirst=False)
        return True
    actual = {column["name"]: column for column in inspector.get_columns(table.name)}
    expected = {column.name: column for column in table.columns}
    if actual.keys() != expected.keys() or any(
        bool(actual[name]["nullable"]) != column.nullable
        or str(actual[name]["type"].compile(dialect=connection.dialect)).upper()
        != str(column.type.compile(dialect=connection.dialect)).upper()
        for name, column in expected.items()
    ):
        raise ValueError(
            "Existing notice registry schema differs from the approved table."
        )
    primary = inspector.get_pk_constraint(table.name).get("constrained_columns")
    unique = {
        (value["name"], tuple(value["column_names"]))
        for value in inspector.get_unique_constraints(table.name)
    }
    if primary != ["id"] or unique != {
        ("uq_regulatory_source_notice_identity", ("document_set_id", "identity_key"))
    }:
        raise ValueError(
            "Existing notice registry identity constraints are unavailable."
        )
    checks = {value["name"] for value in inspector.get_check_constraints(table.name)}
    if checks != {
        "ck_regulatory_source_notice_revision",
        "ck_regulatory_source_notice_content_state",
    }:
        raise ValueError("Existing notice registry content constraints differ.")
    foreign_keys = {
        (
            tuple(value["constrained_columns"]),
            value["referred_table"],
            tuple(value["referred_columns"]),
            value.get("options", {}).get("ondelete"),
        )
        for value in inspector.get_foreign_keys(table.name)
    }
    if foreign_keys != {
        (("document_set_id",), "document_set", ("id",), "RESTRICT"),
        (("user_file_id",), "user_file", ("id",), "RESTRICT"),
    }:
        raise ValueError("Existing notice registry ownership constraints differ.")
    indexes = {
        value["name"]: (value["column_names"], bool(value["unique"]))
        for value in inspector.get_indexes(table.name)
        if not value.get("duplicates_constraint")
    }
    if indexes != {
        "ix_regulatory_source_notice_instrument": (
            [
                "document_set_id",
                "instrument_type",
                "instrument_year",
                "instrument_number",
            ],
            False,
        )
    }:
        raise ValueError("Existing notice registry lookup indexes differ.")
    return False


def _notice_from_row(row: RegulatorySourceNotice) -> SourceNotice:
    return SourceNotice(
        id=str(row.id),
        document_set_id=row.document_set_id,
        identity_key=row.identity_key,
        instrument_type=row.instrument_type,
        instrument_number=row.instrument_number,
        instrument_year=row.instrument_year,
        instrument_date=row.instrument_date.isoformat()
        if row.instrument_date
        else None,
        catalog_title=row.catalog_title,
        aliases=list(row.aliases),
        legistation_metadata=LegistationMetadata.model_validate(
            row.legistation_metadata
        ),
        provenance=cast(dict[str, JsonValue], dict(row.provenance)),
        content_state=cast(Literal["metadata_only", "materialized"], row.content_state),
        user_file_id=str(row.user_file_id) if row.user_file_id else None,
        revision=row.revision,
    )


def find_source_notices(
    session: Session,
    user: User,
    filters: IndexFilters,
    *,
    queries: list[str] | None = None,
    notice_ids: list[str] | None = None,
    limit: int = 20,
) -> list[SourceNotice]:
    """Resolve explicit typed identities; a catalogue entry never proves original text."""
    with session.no_autoflush:
        return _find_source_notices(
            session, user, filters, queries=queries, notice_ids=notice_ids, limit=limit
        )


def _find_source_notices(
    session: Session,
    user: User,
    filters: IndexFilters,
    *,
    queries: list[str] | None,
    notice_ids: list[str] | None,
    limit: int,
) -> list[SourceNotice]:
    if not 1 <= limit <= 100:
        raise ValueError("Notice lookup limit must be 1..100.")
    if queries is None and notice_ids is None:
        return []
    if (queries is not None and len(queries) > 100) or (
        notice_ids is not None and len(notice_ids) > 100
    ):
        raise ValueError("Notice lookup accepts at most 100 targets.")
    scoped = resolve_pc_corpus_scope(session, user=user, filters=filters)
    if (
        scoped.tags
        or scoped.created_at_range
        or scoped.updated_at_range
        or scoped.hierarchy_node_ids
        or scoped.regulatory_source_hint
        or scoped.regulatory_lookup_heading
        or scoped.regulatory_candidate_ids is not None
    ):
        raise CorpusScopeUnavailable(
            "Metadata-only notices cannot satisfy this narrower scope."
        )
    # Notices have set ownership, but no file/project/persona membership to infer.
    file_only = (
        scoped.project_id_filter is not None
        or scoped.persona_id_filter is not None
        or bool(scoped.attached_document_ids)
        and PC_CORPUS_NAME not in (scoped.document_set or [])
    )
    if (
        scoped.document_set
        and PC_CORPUS_NAME not in scoped.document_set
        and not scoped.attached_document_ids
    ):
        return []
    query_groups: list[tuple[set[str], str | None]] = []
    for query in queries or []:
        identities = notice_identity_queries(query)
        query_groups.append(
            (
                set(identities),
                notice_title_alias(query)
                if query.strip().casefold() not in identities
                else None,
            )
        )
    scope_conditions = [
        RegulatorySourceNotice.document_set_id == scoped.asv3_document_set_id,
    ]
    if file_only:
        scope_conditions.append(RegulatorySourceNotice.user_file_id.is_not(None))
    title_aliases = {alias for _, alias in query_groups if alias is not None}
    matched_titles: dict[str, set[UUID]] = {}
    if title_aliases:
        title_candidates = list(
            session.execute(
                select(
                    RegulatorySourceNotice.id,
                    RegulatorySourceNotice.catalog_title,
                    RegulatorySourceNotice.aliases,
                )
                .where(
                    *scope_conditions,
                    or_(
                        RegulatorySourceNotice.identity_key.not_like("catalog:%"),
                        *(
                            func.jsonb_exists(
                                RegulatorySourceNotice.aliases, alias, type_=Boolean
                            )
                            for alias in sorted(title_aliases)
                        ),
                    ),
                )
                .limit(1001)
            )
        )
        if len(title_candidates) > 1000:
            raise CorpusScopeUnavailable(
                "Exact notice title candidates exceed the bounded lookup."
            )
        for identifier, title, aliases in title_candidates:
            exact = title_aliases.intersection(aliases)
            old_title = notice_title_alias(title)
            if old_title in title_aliases:
                exact.add(old_title)
            for alias in exact:
                matched_titles.setdefault(alias, set()).add(identifier)
    keys: set[str] = set()
    preferred_identifiers: set[UUID] = set()
    for identities, title_alias in query_groups:
        if title_alias is not None and title_alias in matched_titles:
            preferred_identifiers.update(matched_titles[title_alias])
        else:
            keys.update(identities)
            if title_alias is not None:
                keys.add(title_alias)
    identifiers = set(preferred_identifiers)
    for value in notice_ids or []:
        try:
            identifiers.add(UUID(value))
        except ValueError:
            continue
    matches = []
    if keys:
        matches.append(RegulatorySourceNotice.identity_key.in_(sorted(keys)))
        matches.extend(
            func.jsonb_exists(RegulatorySourceNotice.aliases, key, type_=Boolean)
            for key in sorted(keys)
        )
    if identifiers:
        matches.append(RegulatorySourceNotice.id.in_(identifiers))
    if not matches:
        return []
    statement = select(RegulatorySourceNotice).where(
        *scope_conditions,
        or_(*matches),
    )
    records = list(
        session.scalars(statement.order_by(RegulatorySourceNotice.id).limit(limit))
    )
    result: list[SourceNotice] = []
    for record in records:
        if record.user_file_id is not None:
            try:
                require_source(
                    session, user=user, filters=scoped, source_id=record.user_file_id
                )
            except (PermissionError, CorpusScopeUnavailable):
                continue
        result.append(_notice_from_row(record))
    return result
