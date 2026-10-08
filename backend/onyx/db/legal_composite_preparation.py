"""Persistent original source types; preparation never runs on the chat path."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from datetime import date
from uuid import UUID, uuid4

from sqlalchemy import and_, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from onyx.context.search.models import IndexFilters
from onyx.db.asv3_corpus import CorpusScopeUnavailable, CorpusSource, require_source
from onyx.db.legal_composite_sources import (
    MAX_OPENING_BATCH_SOURCES,
    SOURCE_PAGE_SIZE,
    SourceClassification,
    SourceKind,
    SourceLaneCatalogue,
    _routing_opening_batch,
    classify_source,
    find_source_inventory_page,
    source_scope_sha256,
)
from onyx.db.models import (
    LegalCompositeSourceKind,
    LegalCompositeSourceState,
    RegulatoryChunk,
    RegulatoryTemporalProjection,
    User,
)

ALGORITHM_VERSION = "original-opening-v1"


def _window_condition(as_of: date | None) -> ColumnElement[bool]:
    if as_of is None:
        return LegalCompositeSourceKind.window_key == "current"
    return and_(
        LegalCompositeSourceKind.window_key != "current",
        or_(
            LegalCompositeSourceKind.effective_start.is_(None),
            LegalCompositeSourceKind.effective_start <= as_of,
        ),
        or_(
            LegalCompositeSourceKind.effective_end.is_(None),
            LegalCompositeSourceKind.effective_end > as_of,
        ),
    )


def _prepared_records(
    session: Session, sources: list[CorpusSource], as_of: date | None
) -> list[SourceClassification]:
    ids = tuple(source.id for source in sources)
    if not ids:
        return []
    revisions = {
        identifier: revision
        for identifier, revision in session.execute(
            select(
                LegalCompositeSourceState.user_file_id,
                LegalCompositeSourceState.revision,
            ).where(LegalCompositeSourceState.user_file_id.in_(ids))
        ).all()
    }
    rows = session.execute(
        select(
            LegalCompositeSourceKind.id,
            LegalCompositeSourceKind.user_file_id,
            LegalCompositeSourceKind.source_revision,
            LegalCompositeSourceKind.window_key,
            LegalCompositeSourceKind.source_kind,
            LegalCompositeSourceKind.classification["uncertain"]
            .as_boolean()
            .label("uncertain"),
            LegalCompositeSourceKind.classification[
                "opening_identity_sha256"
            ].astext.label("opening_identity"),
        )
        .join(
            LegalCompositeSourceState,
            and_(
                LegalCompositeSourceState.user_file_id
                == LegalCompositeSourceKind.user_file_id,
                LegalCompositeSourceState.revision
                == LegalCompositeSourceKind.source_revision,
            ),
        )
        .where(
            LegalCompositeSourceKind.user_file_id.in_(ids),
            LegalCompositeSourceKind.algorithm_version == ALGORITHM_VERSION,
            _window_condition(as_of),
        )
    ).all()
    by_id = {row.user_file_id: row for row in rows}
    if len(by_id) != len(rows):
        raise CorpusScopeUnavailable("Prepared source validity windows overlap.")
    result: list[SourceClassification] = []
    for source in sources:
        prepared = by_id.get(source.id)
        kind = SourceKind(prepared.source_kind) if prepared else SourceKind.UNKNOWN
        result.append(
            SourceClassification(
                source_id=source.id,
                name=source.name,
                kind=kind,
                original_kind=kind if kind != SourceKind.UNKNOWN else None,
                method="prepared_original_opening"
                if prepared
                else "preparation_pending",
                uncertain=kind == SourceKind.UNKNOWN
                or bool(prepared and prepared.uncertain),
                observed_document_types=(),
                opening_identity_sha256=prepared.opening_identity if prepared else None,
                prepared=True,
                prepared_revision=prepared.source_revision
                if prepared
                else revisions.get(source.id),
                prepared_window=prepared.window_key if prepared else None,
                preparation_id=prepared.id if prepared else None,
            )
        )
    return result


def load_prepared_source_lane_catalogue(
    session: Session,
    *,
    user: User,
    filters: IndexFilters,
    check_active: Callable[[], None],
) -> SourceLaneCatalogue:
    """Read compact stored types with current scope/ACL; never read chunk text."""
    records: list[SourceClassification] = []
    offset, more = 0, True
    while more:
        check_active()
        sources, more = find_source_inventory_page(
            session, user=user, filters=filters, offset=offset, limit=SOURCE_PAGE_SIZE
        )
        records.extend(_prepared_records(session, sources, filters.as_of_date))
        offset += SOURCE_PAGE_SIZE
    missing = sum(row.preparation_id is None for row in records)
    return SourceLaneCatalogue(
        user_id=user.id,
        scope_sha256=source_scope_sha256(user, filters),
        records=tuple(records),
        complete=missing == 0,
        limitations=(
            (
                f"{missing} sources await offline type preparation and remain searchable in every source-kind lane.",
            )
            if missing
            else ()
        ),
    )


def revalidate_prepared_source_classification(
    session: Session,
    *,
    user: User,
    filters: IndexFilters,
    recorded: SourceClassification,
    check_active: Callable[[], None],
) -> CorpusSource:
    """Check live access and the transactional revision, without reopening text."""
    check_active()
    source = require_source(
        session, user=user, filters=filters, source_id=recorded.source_id
    )
    current = _prepared_records(session, [source], filters.as_of_date)[0]
    if (
        current.preparation_id,
        current.prepared_revision,
        current.kind,
        current.uncertain,
    ) != (
        recorded.preparation_id,
        recorded.prepared_revision,
        recorded.kind,
        recorded.uncertain,
    ):
        raise CorpusScopeUnavailable(
            "Prepared source identity changed during research."
        )
    return source


def revalidate_prepared_sources(
    session: Session,
    *,
    user: User,
    filters: IndexFilters,
    recorded: list[SourceClassification],
    check_active: Callable[[], None],
) -> dict[UUID, CorpusSource]:
    """Batch live authorization and revision checks; exclude only changed sources."""
    expected = {row.source_id: row for row in recorded}
    valid: dict[UUID, CorpusSource] = {}
    identifiers = tuple(expected)
    for start in range(0, len(identifiers), SOURCE_PAGE_SIZE):
        check_active()
        sources, _ = find_source_inventory_page(
            session,
            user=user,
            filters=filters,
            source_ids=identifiers[start : start + SOURCE_PAGE_SIZE],
            limit=SOURCE_PAGE_SIZE,
        )
        for source, current in zip(
            sources,
            _prepared_records(session, sources, filters.as_of_date),
            strict=True,
        ):
            previous = expected[source.id]
            if (
                current.preparation_id,
                current.prepared_revision,
                current.kind,
                current.uncertain,
            ) == (
                previous.preparation_id,
                previous.prepared_revision,
                previous.kind,
                previous.uncertain,
            ):
                valid[source.id] = source
    return valid


def _boundaries(session: Session, ids: tuple[UUID, ...]) -> dict[UUID, list[date]]:
    values: dict[UUID, set[date]] = defaultdict(set)
    statements = (
        select(
            RegulatoryChunk.user_file_id,
            RegulatoryChunk.validity_start_date,
            RegulatoryChunk.validity_end_date,
        )
        .where(RegulatoryChunk.user_file_id.in_(ids))
        .distinct(),
        select(
            RegulatoryTemporalProjection.user_file_id,
            RegulatoryTemporalProjection.effective_start,
            RegulatoryTemporalProjection.effective_end,
        )
        .where(
            RegulatoryTemporalProjection.user_file_id.in_(ids),
            RegulatoryTemporalProjection.retired_at.is_(None),
        )
        .distinct(),
    )
    for statement in statements:
        for source_id, start, end in session.execute(statement):
            if start is not None:
                values[source_id].add(start)
            if end is not None:
                values[source_id].add(end)
    return {source_id: sorted(values[source_id]) for source_id in ids}


def prepare_source_kinds(
    session: Session,
    *,
    user: User,
    filters: IndexFilters,
    check_active: Callable[[], None],
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[int, int]:
    """Resume a tenant preparation; commit only new/current source versions per batch.

    All historical validity intervals are retained. Repeating this command reads
    only the small revision/type tables for already prepared source versions.
    """
    offset, more, prepared_count, skipped_count = 0, True, 0, 0
    while more:
        check_active()
        sources, more = find_source_inventory_page(
            session, user=user, filters=filters, offset=offset, limit=SOURCE_PAGE_SIZE
        )
        offset += SOURCE_PAGE_SIZE
        for start in range(0, len(sources), MAX_OPENING_BATCH_SOURCES):
            group = sources[start : start + MAX_OPENING_BATCH_SOURCES]
            ids = tuple(source.id for source in group)
            if not ids:
                continue
            session.execute(
                insert(LegalCompositeSourceState)
                .values(
                    [{"user_file_id": source_id, "revision": 1} for source_id in ids]
                )
                .on_conflict_do_nothing(index_elements=["user_file_id"])
            )
            revisions = {
                identifier: revision
                for identifier, revision in session.execute(
                    select(
                        LegalCompositeSourceState.user_file_id,
                        LegalCompositeSourceState.revision,
                    ).where(LegalCompositeSourceState.user_file_id.in_(ids))
                ).all()
            }
            complete = set(
                session.scalars(
                    select(LegalCompositeSourceKind.user_file_id)
                    .join(
                        LegalCompositeSourceState,
                        and_(
                            LegalCompositeSourceState.user_file_id
                            == LegalCompositeSourceKind.user_file_id,
                            LegalCompositeSourceState.revision
                            == LegalCompositeSourceKind.source_revision,
                        ),
                    )
                    .where(
                        LegalCompositeSourceKind.user_file_id.in_(ids),
                        LegalCompositeSourceKind.algorithm_version == ALGORITHM_VERSION,
                        LegalCompositeSourceKind.window_key == "current",
                    )
                )
            )
            skipped_count += len(complete)
            pending = [source for source in group if source.id not in complete]
            pending_ids = tuple(source.id for source in pending)
            boundaries = _boundaries(session, pending_ids) if pending else {}
            jobs: dict[
                date | None, list[tuple[CorpusSource, date | None, date | None]]
            ] = defaultdict(list)
            for source in pending:
                cuts: list[date | None] = [None, *boundaries[source.id], None]
                for lower, upper in zip(cuts, cuts[1:]):
                    if upper == date.min:
                        continue
                    jobs[lower or date.min].append((source, lower, upper))
                # Current and historical windows commit atomically for this source batch.
                jobs[None].append((source, None, None))
            for observed_at, window_sources in jobs.items():
                check_active()
                dated = filters.model_copy(update={"as_of_date": observed_at})
                openings = _routing_opening_batch(
                    session,
                    tuple(source.id for source, _, _ in window_sources),
                    dated,
                    check_active,
                )
                for source, lower, upper in window_sources:
                    opening = openings[source.id]
                    record = classify_source(
                        source,
                        (),
                        opening_texts=opening.texts
                        if opening.identity_available
                        else (),
                    )
                    key = (
                        "current"
                        if observed_at is None
                        else f"{lower or '-infinity'}/{upper or 'infinity'}"
                    )
                    statement = insert(LegalCompositeSourceKind).values(
                        id=uuid4(),
                        user_file_id=source.id,
                        source_revision=revisions[source.id],
                        algorithm_version=ALGORITHM_VERSION,
                        window_key=key,
                        effective_start=lower,
                        effective_end=upper,
                        source_kind=record.kind.value,
                        classification=record.model_copy(
                            update={"opening_witnesses": opening.witnesses}
                        ).model_dump(mode="json"),
                    )
                    session.execute(
                        statement.on_conflict_do_nothing(
                            constraint="uq_legal_composite_kind_version_window"
                        )
                    )
            session.commit()
            prepared_count += len(pending)
            if on_progress:
                on_progress(prepared_count, skipped_count)
    return prepared_count, skipped_count


def main() -> None:
    """Explicit, resumable preparation outside the request path."""
    import argparse
    import json

    from sqlalchemy import text

    from onyx.auth.schemas import UserRole
    from onyx.db.asv3_corpus import resolve_pc_corpus_scope
    from onyx.db.engine.sql_engine import SqlEngine, get_session_with_current_tenant
    from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--expected-database", required=True)
    parser.add_argument("--document-set-id", required=True, type=int)
    parser.add_argument("--user-id", required=True, type=UUID)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--file-id", type=UUID, action="append")
    selection.add_argument("--all-sources", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    SqlEngine.init_engine(pool_size=2, max_overflow=0)
    tenant_token = CURRENT_TENANT_ID_CONTEXTVAR.set(args.tenant)
    try:
        with get_session_with_current_tenant() as session:
            actual_database = session.execute(
                text("SELECT current_database()")
            ).scalar_one()
            if actual_database != args.expected_database:
                raise PermissionError(
                    "Preparation database does not match the expected environment"
                )
            user = session.get(User, args.user_id)
            if user is None or not user.is_active or user.role != UserRole.ADMIN:
                raise PermissionError(
                    "An active authorized admin is required for shared source preparation"
                )
            filters = resolve_pc_corpus_scope(
                session,
                user=user,
                filters=IndexFilters(
                    access_control_list=[],
                    tenant_id=args.tenant,
                    asv3_document_set_id=args.document_set_id,
                    attached_document_ids=[str(value) for value in args.file_id]
                    if args.file_id
                    else None,
                    regulatory_chunks_only=True,
                ),
            )

            def progress(prepared: int, skipped: int) -> None:
                print(
                    json.dumps({"prepared": prepared, "unchanged_skipped": skipped}),
                    flush=True,
                )

            if args.apply:
                prepared, skipped = prepare_source_kinds(
                    session,
                    user=user,
                    filters=filters,
                    check_active=lambda: None,
                    on_progress=progress,
                )
                progress(prepared, skipped)
            else:
                catalogue = load_prepared_source_lane_catalogue(
                    session,
                    user=user,
                    filters=filters,
                    check_active=lambda: None,
                )
                print(json.dumps(catalogue.provenance()), flush=True)
    finally:
        CURRENT_TENANT_ID_CONTEXTVAR.reset(tenant_token)


if __name__ == "__main__":
    main()
