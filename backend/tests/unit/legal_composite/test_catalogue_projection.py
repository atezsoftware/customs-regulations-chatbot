"""LC catalogue projection preserves public guards and emits aggregate timings only."""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from onyx.context.search.models import IndexFilters
from onyx.db import legal_composite_catalogue as catalogue
from onyx.db import legal_composite_preparation as preparation
from onyx.db import legal_composite_sources as sources
from onyx.db import regulatory_publication as publication
from onyx.db.models import User
from onyx.document_index.publication_models import PublicationScope, ReadObservation
from onyx.regulatory.amendments.annexes.config import ANNEX_DATABASE_IDENTITY


def scope() -> PublicationScope:
    return PublicationScope(
        tenant_id="owned-catalogue-test",
        environment="test",
        database_identity=ANNEX_DATABASE_IDENTITY,
    )


def session_for_scope(observed: PublicationScope) -> MagicMock:
    session = MagicMock(spec=Session)
    connection = session.connection.return_value
    connection.get_execution_options.return_value = {
        "schema_translate_map": {None: observed.tenant_id}
    }
    host_port, database = observed.database_identity.rsplit("/", 1)
    host, port = host_port.rsplit(":", 1)
    connection.engine.url = SimpleNamespace(
        host=host, port=int(port), database=database
    )
    connection.get_isolation_level.return_value = "READ COMMITTED"
    return session


def test_scalar_publication_predicate_matches_default_with_fresh_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_scope = scope()
    default = publication.PublicationStore(observed_scope)
    timings = catalogue.CatalogueTimings()
    scalar = catalogue._ScalarPublicationStore(observed_scope, timings)
    observation = ReadObservation(scope=observed_scope, committed_epoch=5)
    identifiers = tuple(uuid4() for _ in range(5))
    opened: list[MagicMock] = []
    queries: list[Any] = []
    state = {"closed": False, "epoch": 5}

    @contextmanager
    def fresh(*, tenant_id: str) -> Iterator[Session]:
        assert tenant_id == observed_scope.tenant_id
        session = session_for_scope(observed_scope)
        rows = [
            SimpleNamespace(
                user_file_id=identifiers[0],
                scope_key=default.scope_key,
                gate_closed=state["closed"],
                epoch=state["epoch"],
            ),
            SimpleNamespace(
                user_file_id=identifiers[1],
                scope_key="foreign-scope",
                gate_closed=False,
                epoch=999,
            ),
            SimpleNamespace(
                user_file_id=identifiers[2],
                scope_key="foreign-scope",
                gate_closed=True,
                epoch=0,
            ),
            SimpleNamespace(
                user_file_id=identifiers[3],
                scope_key=default.scope_key,
                gate_closed=False,
                epoch=4,
            ),
        ]
        session.scalars.side_effect = lambda statement: (
            queries.append(statement),
            rows,
        )[1]
        session.execute.side_effect = lambda statement: (
            queries.append(statement),
            SimpleNamespace(all=lambda: rows),
        )[1]
        opened.append(session)
        yield cast(Session, session)

    monkeypatch.setattr(publication, "get_session_with_tenant", fresh)
    monkeypatch.setattr(catalogue, "get_session_with_tenant", fresh)
    for closed, epoch in [(False, 5), (True, 5), (False, 6)]:
        state.update(closed=closed, epoch=epoch)
        assert scalar.unavailable(observation, identifiers) == default.unavailable(
            observation, identifiers
        )
    assert len(opened) == 6 and len({id(session) for session in opened}) == 6
    assert identifiers[1] not in scalar.unavailable(observation, identifiers)
    assert identifiers[4] not in scalar.unavailable(observation, identifiers)
    scalar_sql = str(queries[0].compile(dialect=postgresql.dialect()))
    default_sql = str(queries[1].compile(dialect=postgresql.dialect()))
    assert list(queries[0].selected_columns.keys()) == [
        "user_file_id",
        "scope_key",
        "gate_closed",
        "epoch",
    ]
    assert (
        "writer_manifest" not in scalar_sql
        and "original_ingestion_receipt" not in scalar_sql
    )
    assert (
        "writer_manifest" in default_sql and "original_ingestion_receipt" in default_sql
    )


@pytest.mark.parametrize("bad_guard", ["tenant", "database", "isolation"])
def test_scalar_projection_keeps_real_session_scope_validation(
    bad_guard: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed_scope = scope()
    store = catalogue._ScalarPublicationStore(
        observed_scope, catalogue.CatalogueTimings()
    )
    session = session_for_scope(observed_scope)
    if bad_guard == "tenant":
        session.connection.return_value.get_execution_options.return_value = {
            "schema_translate_map": {None: "another-tenant"}
        }
    elif bad_guard == "database":
        session.connection.return_value.engine.url.database = "another-database"
    else:
        session.connection.return_value.get_isolation_level.return_value = (
            "REPEATABLE READ"
        )

    @contextmanager
    def fresh(*, tenant_id: str) -> Iterator[Session]:
        assert tenant_id == observed_scope.tenant_id
        yield cast(Session, session)

    monkeypatch.setattr(catalogue, "get_session_with_tenant", fresh)
    with pytest.raises(ValueError, match="mismatch|READ COMMITTED"):
        store.unavailable(
            ReadObservation(scope=observed_scope, committed_epoch=0), (uuid4(),)
        )
    session.execute.assert_not_called()


def test_mismatched_observation_cannot_open_a_publication_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = catalogue._ScalarPublicationStore(scope(), catalogue.CatalogueTimings())
    connect = MagicMock()
    monkeypatch.setattr(catalogue, "get_session_with_tenant", connect)
    wrong_scope = scope().model_copy(update={"environment": "another-environment"})
    with pytest.raises(ValueError, match="read observation scope mismatch"):
        store.unavailable(
            ReadObservation(scope=wrong_scope, committed_epoch=0), (uuid4(),)
        )
    connect.assert_not_called()


@pytest.mark.parametrize("as_of", [None, date(2020, 1, 1)])
def test_catalogue_scalar_optin_preserves_inventory_acl_types_dates_and_unknown(
    as_of: date | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = cast(User, SimpleNamespace(id=uuid4()))
    filters = IndexFilters(
        access_control_list=["owned-user"],
        asv3_document_set_id=15,
        forced_document_set=["PC Külliyatı"],
        as_of_date=as_of,
    )
    rows = [
        SimpleNamespace(
            id=uuid4(), name=f"private-source-{index}", file_id=f"private-file-{index}"
        )
        for index in range(1001)
    ]
    denied, unavailable = rows[1].id, rows[2].id
    known, uncertain = rows[0].id, rows[3].id
    prepared_rows = [
        SimpleNamespace(
            id=uuid4(),
            user_file_id=known,
            source_revision=3,
            window_key="current" if as_of is None else "historical-window",
            source_kind=sources.SourceKind.STATUTE.value,
            uncertain=False,
            opening_identity="a" * 64,
        ),
        SimpleNamespace(
            id=uuid4(),
            user_file_id=uncertain,
            source_revision=3,
            window_key="current" if as_of is None else "historical-window",
            source_kind=sources.SourceKind.REGULATION.value,
            uncertain=True,
            opening_identity="b" * 64,
        ),
    ]
    query_sets: list[list[Any]] = [[], []]
    permissions: list[list[str]] = []

    def validate(
        _session: Session, actual_user: User, actual_filters: IndexFilters
    ) -> None:
        assert actual_user is user and actual_filters is filters

    monkeypatch.setattr(sources, "_validate_filters", validate)
    monkeypatch.setattr(
        sources,
        "get_acl_for_user",
        lambda actual_user, _session: (
            {"owned-user"} if actual_user is user else pytest.fail("Wrong user")
        ),
    )

    def access(ids: list[str], _session: Session) -> dict[str, Any]:
        permissions.append(ids)
        return {
            identifier: SimpleNamespace(to_acl=lambda: {"owned-user"})
            for identifier in ids
            if identifier != str(denied)
        }

    monkeypatch.setattr(sources, "get_access_for_user_files", access)
    observation = ReadObservation(scope=scope(), committed_epoch=7)
    monkeypatch.setattr(sources, "observe_publication_read", lambda: observation)
    default_filter = MagicMock(
        side_effect=lambda observed, candidates, _identity, **_kwargs: (
            [row for row in candidates if row.id != unavailable]
            if observed is observation
            else pytest.fail("Wrong observation")
        )
    )
    monkeypatch.setattr(sources, "filter_publication_read", default_filter)
    store = catalogue._ScalarPublicationStore(scope(), catalogue.CatalogueTimings())
    monkeypatch.setattr(catalogue, "public_read_store", lambda: store)

    @contextmanager
    def publication_session(*, tenant_id: str) -> Iterator[Session]:
        assert tenant_id == scope().tenant_id
        session = session_for_scope(scope())
        session.execute.return_value.all.return_value = [
            SimpleNamespace(
                user_file_id=unavailable,
                scope_key=store.scope_key,
                gate_closed=True,
                epoch=7,
            )
        ]
        yield cast(Session, session)

    monkeypatch.setattr(catalogue, "get_session_with_tenant", publication_session)

    def loader_session(query_set: list[Any]) -> Session:
        session = MagicMock(spec=Session)

        def execute(statement: Any) -> Any:
            query_set.append(statement)
            table = next(iter(statement.selected_columns)).table.name
            if table == "user_file":
                offset = statement._offset_clause.value
                limit = statement._limit_clause.value
                result = rows[offset : offset + limit]
            elif table == "legal_composite_source_state":
                result = [(row.id, 3) for row in rows]
            else:
                result = prepared_rows
            return SimpleNamespace(all=lambda: result)

        session.execute.side_effect = execute
        return cast(Session, session)

    baseline = preparation.load_prepared_source_lane_catalogue(
        loader_session(query_sets[0]),
        user=user,
        filters=filters,
        check_active=lambda: None,
    )
    assert default_filter.call_count == 2
    timings = catalogue.CatalogueTimings()
    actual = catalogue.load_prepared_source_lane_catalogue(
        loader_session(query_sets[1]),
        user=user,
        filters=filters,
        check_active=lambda: None,
        timings=timings,
    )
    assert actual == baseline
    assert default_filter.call_count == 2
    assert len(actual.records) == 999 and not actual.complete
    unknown = next(row for row in actual.records if row.preparation_id is None)
    assert all(unknown.admits(kind) for kind in sources.SourceKind)
    unclear = next(row for row in actual.records if row.source_id == uncertain)
    assert all(unclear.admits(kind) for kind in sources.SourceKind)
    for left, right in zip(query_sets[0], query_sets[1], strict=True):
        before = left.compile(dialect=postgresql.dialect())
        after = right.compile(dialect=postgresql.dialect())
        assert str(before) == str(after) and before.params == after.params
    type_query = str(query_sets[1][2].compile(dialect=postgresql.dialect()))
    assert ("effective_start" in type_query) == (as_of is not None)
    assert len(permissions[0]) == 1001 and permissions[0] == permissions[2]
    assert timings.snapshot()["inventory_page_count"] == 2
    aggregate = str(timings.snapshot())
    assert all(
        str(row.id) not in aggregate
        and row.name not in aggregate
        and row.file_id not in aggregate
        for row in rows
    )


def test_timing_collector_accepts_only_known_finite_aggregate_stages() -> None:
    timings = catalogue.CatalogueTimings()
    timings.record("inventory_page_seconds", 1.0)
    timings.record("inventory_page_seconds", 2.0)
    assert timings.snapshot()["stage_seconds"] == {"inventory_page_seconds": 3.0}
    assert timings.snapshot()["durations_overlap"] is True
    for name, value in [
        ("private-source", 1.0),
        ("inventory_page_seconds", -1.0),
        ("inventory_page_seconds", float("nan")),
        ("inventory_page_seconds", float("inf")),
    ]:
        with pytest.raises(ValueError, match="Invalid aggregate"):
            timings.record(name, value)
    assert "private-source" not in str(timings.snapshot())


def test_scalar_loader_is_explicit_lc_runtime_optin() -> None:
    import inspect

    from onyx.legal_composite import runtime

    assert (
        runtime.load_prepared_source_lane_catalogue
        is catalogue.load_prepared_source_lane_catalogue
    )
    assert (
        preparation.load_prepared_source_lane_catalogue
        is catalogue.load_default_catalogue
    )
    parameters = inspect.signature(sources.find_source_inventory_page).parameters
    assert parameters["publication_store"].default is None
    assert parameters["record_timing"].default is None
    parameters = inspect.signature(
        preparation.load_prepared_source_lane_catalogue
    ).parameters
    assert parameters["publication_store"].default is None
    assert parameters["record_timing"].default is None


def test_public_catalogue_summary_has_fixed_aggregate_keys_and_no_nested_double_count() -> (
    None
):
    timings = catalogue.CatalogueTimings()
    for stage, seconds in {
        "catalogue_total_seconds": 12.34,
        "inventory_page_seconds": 10.5,
        "file_acl_seconds": 2.25,
        "user_acl_seconds": 0.25,
        "publication_guard_seconds": 6.0,
        "publication_scalar_query_seconds": 4.0,
        "publication_session_scope_seconds": 1.0,
        "prepared_records_seconds": 1.75,
    }.items():
        timings.record(stage, seconds)
    assert timings.safe_summary() == (
        "catalogue total=12.34s inventory=10.50s acl=2.50s pub=6.00s types=1.75s pages=1"
    )
    assert len(timings.safe_summary()) <= 160
    assert "publication_scalar_query" not in timings.safe_summary()
    assert "publication_session_scope" not in timings.safe_summary()
    assert "source" not in timings.safe_summary()
    huge = catalogue.CatalogueTimings()
    huge.record("catalogue_total_seconds", 1e308)
    assert "total=1.00e+308s" in huge.safe_summary()
    with pytest.raises(ValueError, match="Invalid aggregate"):
        huge.record("catalogue_total_seconds", 1e308)
    assert "total=1.00e+308s" in huge.safe_summary()
