"""Exercise concurrent FK-index recovery on disposable local tenant schemas."""

import importlib.util
import time
from argparse import Namespace
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Connection, Engine, create_engine, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.pool import NullPool
from sqlalchemy.schema import CreateSchema, DropSchema

from onyx.configs.app_configs import POSTGRES_HOST, POSTGRES_PORT
from onyx.db.engine.sql_engine import build_connection_string

EXPECTED = {
    "ix_regulatory_chunk_supersedes_chunk_id": (
        "regulatory_chunk",
        "supersedes_chunk_id",
    ),
    "ix_regulatory_chunk_superseded_by_chunk_id": (
        "regulatory_chunk",
        "superseded_by_chunk_id",
    ),
    "ix_user_file_batch_item_regulatory_chunk_id": (
        "user_file_batch_item",
        "regulatory_chunk_id",
    ),
    "ix_regulatory_indexing_item_regulatory_chunk_id": (
        "regulatory_indexing_item",
        "regulatory_chunk_id",
    ),
    "ix_amendment_proposal_applied_new_chunk_id": (
        "amendment_proposal",
        "applied_new_chunk_id",
    ),
    "ix_regulatory_annex_element_chunk_chunk_id": (
        "regulatory_annex_element_chunk",
        "chunk_id",
    ),
}


@pytest.fixture
def migration() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[3]
        / "alembic/versions/576f0377f747_index_regulatory_chunk_incoming_foreign_.py"
    )
    spec = importlib.util.spec_from_file_location("regulatory_chunk_fk_indexes", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.INDEXES == EXPECTED
    return module


@pytest.fixture
def local_engine() -> Generator[Engine, None, None]:
    assert POSTGRES_HOST in {"localhost", "127.0.0.1"}
    assert str(POSTGRES_PORT) == "25432"
    engine = create_engine(
        build_connection_string(db_api="psycopg2"), poolclass=NullPool
    )
    try:
        yield engine
    finally:
        engine.dispose()


def _quoted(connection: Connection, value: str) -> str:
    return connection.dialect.identifier_preparer.quote_schema(value)


@pytest.fixture
def tenant_schema(
    local_engine: Engine, request: pytest.FixtureRequest
) -> Generator[str, None, None]:
    # A quote exercises tenant qualification on the separate AUTOCOMMIT connection.
    schema = (
        'tenant_rg_fk_"_' if getattr(request, "param", True) else "tenant_rg_fk_"
    ) + uuid4().hex
    with local_engine.begin() as connection:
        connection.execute(CreateSchema(schema))
        quoted = _quoted(connection, schema)
        connection.exec_driver_sql(
            f"CREATE TABLE {quoted}.regulatory_chunk ("
            "id text PRIMARY KEY, supersedes_chunk_id text, "
            "superseded_by_chunk_id text, "
            f"FOREIGN KEY (supersedes_chunk_id) REFERENCES {quoted}.regulatory_chunk(id) ON DELETE SET NULL, "
            f"FOREIGN KEY (superseded_by_chunk_id) REFERENCES {quoted}.regulatory_chunk(id) ON DELETE SET NULL)"
        )
        for table, column in sorted(set(EXPECTED.values())):
            if table == "regulatory_chunk":
                continue
            connection.exec_driver_sql(
                f"CREATE TABLE {quoted}.{table} ("
                f"{column} text REFERENCES {quoted}.regulatory_chunk(id) "
                + (
                    "ON DELETE SET NULL)"
                    if table == "amendment_proposal"
                    else "ON DELETE CASCADE)"
                )
            )
        connection.exec_driver_sql(
            f"INSERT INTO {quoted}.regulatory_chunk (id) VALUES ('source'), ('other')"
        )
        for table, column in sorted(set(EXPECTED.values())):
            if table != "regulatory_chunk":
                connection.exec_driver_sql(
                    f"INSERT INTO {quoted}.{table} ({column}) VALUES ('source'), ('source')"
                )
    try:
        yield schema
    finally:
        with local_engine.begin() as connection:
            connection.execute(DropSchema(schema, cascade=True))


@pytest.fixture
def neighbor_schema(local_engine: Engine) -> Generator[str, None, None]:
    schema = "tenant_rg_fk_neighbor_" + uuid4().hex
    with local_engine.begin() as connection:
        connection.execute(CreateSchema(schema))
        quoted = _quoted(connection, schema)
        connection.exec_driver_sql(
            f"CREATE TABLE {quoted}.regulatory_chunk (supersedes_chunk_id text)"
        )
        connection.exec_driver_sql(
            f"CREATE INDEX ix_regulatory_chunk_supersedes_chunk_id ON {quoted}.regulatory_chunk (supersedes_chunk_id)"
        )
    try:
        yield schema
    finally:
        with local_engine.begin() as connection:
            connection.execute(DropSchema(schema, cascade=True))


def _run_migration(
    migration: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    engine: Engine,
    schema: str,
    operation: str,
) -> None:
    with engine.connect() as bind:
        bind.exec_driver_sql(f"SET search_path TO {_quoted(bind, schema)}")
        monkeypatch.setattr(migration.op, "get_bind", lambda: bind)
        getattr(migration, operation)()


def _indexes(engine: Engine, schema: str) -> dict[str, dict[str, object]]:
    with engine.connect() as connection:
        result = connection.execute(
            text(
                "SELECT c.relname AS name, c.oid, t.relname AS table_name, "
                "i.indisvalid, i.indisready, i.indisunique, am.amname, "
                "i.indpred IS NULL AS unfiltered, i.indexprs IS NULL AS unexpressed, "
                "ARRAY(SELECT a.attname FROM unnest(i.indkey) WITH ORDINALITY k(attnum, ord) "
                "JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=k.attnum "
                "ORDER BY k.ord) AS columns "
                "FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid "
                "JOIN pg_class t ON t.oid=i.indrelid JOIN pg_namespace n ON n.oid=t.relnamespace "
                "JOIN pg_am am ON am.oid=c.relam WHERE n.nspname=:schema"
            ),
            {"schema": schema},
        ).mappings()
        return {str(row["name"]): dict(row) for row in result}


def _data(engine: Engine, schema: str) -> dict[str, list[str]]:
    with engine.connect() as connection:
        quoted = _quoted(connection, schema)
        return {
            table: list(
                connection.scalars(
                    text(
                        f"SELECT row_to_json(t)::text FROM {quoted}.{table} t ORDER BY row_to_json(t)::text"
                    )
                )
            )
            for table in sorted({table for table, _column in EXPECTED.values()})
        }


def test_partial_upgrade_restart_and_downgrade_preserve_data(
    migration: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    local_engine: Engine,
    tenant_schema: str,
    neighbor_schema: str,
) -> None:
    before = _data(local_engine, tenant_schema)
    neighbor_indexes = _indexes(local_engine, neighbor_schema)
    with local_engine.begin() as connection:
        quoted = _quoted(connection, tenant_schema)
        for name, (table, column) in list(EXPECTED.items())[:2]:
            connection.exec_driver_sql(
                f"CREATE INDEX {name} ON {quoted}.{table} ({column})"
            )
        connection.exec_driver_sql(
            f"CREATE INDEX ix_owned_fk_test_sentinel ON {quoted}.regulatory_chunk (id)"
        )
    initial = _indexes(local_engine, tenant_schema)
    _run_migration(migration, monkeypatch, local_engine, tenant_schema, "upgrade")
    indexes = _indexes(local_engine, tenant_schema)
    for name, (table, column) in EXPECTED.items():
        index = indexes[name]
        assert index["table_name"] == table
        assert index["columns"] == [column]
        assert index["indisvalid"] and index["indisready"]
        assert not index["indisunique"]
        assert index["amname"] == "btree"
        assert index["unfiltered"] and index["unexpressed"]
    assert all(indexes[name]["oid"] == index["oid"] for name, index in initial.items())
    _run_migration(migration, monkeypatch, local_engine, tenant_schema, "upgrade")
    assert _indexes(local_engine, tenant_schema) == indexes
    assert _data(local_engine, tenant_schema) == before
    _run_migration(migration, monkeypatch, local_engine, tenant_schema, "downgrade")
    _run_migration(migration, monkeypatch, local_engine, tenant_schema, "downgrade")
    remaining = _indexes(local_engine, tenant_schema)
    assert not EXPECTED.keys() & remaining.keys()
    assert (
        remaining["ix_owned_fk_test_sentinel"] == initial["ix_owned_fk_test_sentinel"]
    )
    assert _data(local_engine, tenant_schema) == before
    assert _indexes(local_engine, neighbor_schema) == neighbor_indexes


def test_wrong_unique_same_name_fails_closed(
    migration: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    local_engine: Engine,
    tenant_schema: str,
) -> None:
    name = "ix_regulatory_chunk_supersedes_chunk_id"
    with local_engine.begin() as connection:
        quoted = _quoted(connection, tenant_schema)
        connection.exec_driver_sql(
            f"CREATE UNIQUE INDEX {name} ON {quoted}.regulatory_chunk (supersedes_chunk_id)"
        )
    before = _indexes(local_engine, tenant_schema)
    with pytest.raises(
        (RuntimeError, ValueError), match="(?i)definition|unexpected|mismatch"
    ):
        _run_migration(migration, monkeypatch, local_engine, tenant_schema, "upgrade")
    assert _indexes(local_engine, tenant_schema)[name] == before[name]


def test_interrupted_exact_concurrent_index_is_rebuilt(
    migration: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    local_engine: Engine,
    tenant_schema: str,
) -> None:
    name = "ix_regulatory_indexing_item_regulatory_chunk_id"
    before = _data(local_engine, tenant_schema)
    with (
        local_engine.connect() as blocker,
        local_engine.connect().execution_options(
            isolation_level="AUTOCOMMIT"
        ) as builder,
    ):
        quoted = _quoted(blocker, tenant_schema)
        blocker.exec_driver_sql(
            f"INSERT INTO {quoted}.regulatory_indexing_item (regulatory_chunk_id) VALUES ('other')"
        )
        builder_pid = builder.scalar(text("SELECT pg_backend_pid()"))
        assert isinstance(builder_pid, int)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                builder.exec_driver_sql,
                f"CREATE INDEX CONCURRENTLY {name} ON {quoted}.regulatory_indexing_item (regulatory_chunk_id)",
            )
            try:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    index = _indexes(local_engine, tenant_schema).get(name)
                    if index is not None and index["indisvalid"] is False:
                        break
                    assert not future.done(), (
                        "Concurrent build finished before its owned writer lock was released"
                    )
                    time.sleep(0.02)
                else:
                    pytest.fail("Concurrent build did not register its invalid index")
                with local_engine.connect() as inspector:
                    assert (
                        inspector.scalar(
                            text(
                                "SELECT pg_cancel_backend(:pid) WHERE :pid <> pg_backend_pid()"
                            ),
                            {"pid": builder_pid},
                        )
                        is True
                    )
                with pytest.raises(DBAPIError, match="canceling statement"):
                    future.result(timeout=10)
            finally:
                blocker.rollback()
    invalid = _indexes(local_engine, tenant_schema)[name]
    assert invalid["indisvalid"] is False
    assert not invalid["indisunique"]
    _run_migration(migration, monkeypatch, local_engine, tenant_schema, "upgrade")
    rebuilt = _indexes(local_engine, tenant_schema)[name]
    assert rebuilt["indisvalid"] and rebuilt["indisready"]
    assert rebuilt["oid"] != invalid["oid"]
    assert rebuilt["columns"] == ["regulatory_chunk_id"]
    assert _data(local_engine, tenant_schema) == before


@pytest.mark.parametrize("tenant_schema", [False], indirect=True)
@pytest.mark.parametrize("runner", ["injected", "normal_async"])
def test_real_alembic_env_stamps_after_concurrent_indexes(
    local_engine: Engine,
    tenant_schema: str,
    runner: str,
) -> None:
    backend = Path(__file__).resolve().parents[3]
    config = Config(str(backend / "alembic.ini"))
    config.set_main_option("script_location", str(backend / "alembic"))
    config.attributes["configure_logger"] = False
    if runner == "injected":
        config.attributes.update(connection=local_engine, schema_name=tenant_schema)
    else:
        config.attributes["onyx_target_url"] = build_connection_string(db_api="asyncpg")
        config.cmd_opts = Namespace(x=[f"schemas={tenant_schema}"])
    with local_engine.begin() as connection:
        quoted = _quoted(connection, tenant_schema)
        connection.exec_driver_sql(
            f"CREATE TABLE {quoted}.alembic_version (version_num varchar(32) PRIMARY KEY)"
        )
        connection.exec_driver_sql(
            f"INSERT INTO {quoted}.alembic_version VALUES ('e363f58e0eaf')"
        )
    before = _data(local_engine, tenant_schema)
    if runner == "normal_async":
        with local_engine.begin() as connection:
            quoted = _quoted(connection, tenant_schema)
            connection.exec_driver_sql(
                f"CREATE UNIQUE INDEX ix_regulatory_chunk_supersedes_chunk_id ON {quoted}.regulatory_chunk (supersedes_chunk_id)"
            )
        with pytest.raises(RuntimeError, match="expected chunk reference definition"):
            command.upgrade(config, "576f0377f747")
        with local_engine.begin() as connection:
            quoted = _quoted(connection, tenant_schema)
            assert (
                connection.scalar(
                    text(f"SELECT version_num FROM {quoted}.alembic_version")
                )
                == "e363f58e0eaf"
            )
            connection.exec_driver_sql(
                f"DROP INDEX {quoted}.ix_regulatory_chunk_supersedes_chunk_id"
            )
    command.upgrade(config, "576f0377f747")
    with local_engine.begin() as connection:
        quoted = _quoted(connection, tenant_schema)
        assert (
            connection.scalar(text(f"SELECT version_num FROM {quoted}.alembic_version"))
            == "576f0377f747"
        )
        connection.exec_driver_sql(
            f"INSERT INTO {quoted}.regulatory_chunk (id) VALUES ('after-version-stamp')"
        )
        connection.exec_driver_sql(
            f"DELETE FROM {quoted}.regulatory_chunk WHERE id='after-version-stamp'"
        )
    indexes = _indexes(local_engine, tenant_schema)
    assert all(
        indexes[name]["indisvalid"] and indexes[name]["indisready"] for name in EXPECTED
    )
    command.upgrade(config, "576f0377f747")
    assert _indexes(local_engine, tenant_schema) == indexes
    command.downgrade(config, "e363f58e0eaf")
    with local_engine.connect() as connection:
        quoted = _quoted(connection, tenant_schema)
        assert (
            connection.scalar(text(f"SELECT version_num FROM {quoted}.alembic_version"))
            == "e363f58e0eaf"
        )
    assert not EXPECTED.keys() & _indexes(local_engine, tenant_schema).keys()
    assert _data(local_engine, tenant_schema) == before
