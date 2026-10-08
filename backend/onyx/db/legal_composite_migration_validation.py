"""Exercise source-kind DDL with the same asyncpg driver used by Alembic."""

import importlib.util
from pathlib import Path
from typing import Protocol, cast
from uuid import UUID, uuid4

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import Connection, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from onyx.db.engine.sql_engine import build_connection_string


class _Migration(Protocol):
    op: Operations

    def upgrade(self) -> None: ...

    def downgrade(self) -> None: ...


async def validate_source_kind_migration() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "alembic/versions/4ecf8032a916_persist_original_source_kinds_for_legal_.py"
    )
    spec = importlib.util.spec_from_file_location("source_kind_migration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    migration = cast(_Migration, module)
    schema = "lc_migration_test_" + uuid4().hex
    source_a, source_b = uuid4(), uuid4()
    engine = create_async_engine(
        build_connection_string(db_api="asyncpg"), poolclass=NullPool
    )

    def upgrade(connection: Connection) -> None:
        migration.op = Operations(MigrationContext.configure(connection))
        migration.upgrade()

    def downgrade(connection: Connection) -> None:
        migration.op = Operations(MigrationContext.configure(connection))
        migration.downgrade()

    try:
        async with engine.connect() as connection:
            transaction = await connection.begin()
            try:
                for statement in (
                    f"CREATE SCHEMA {schema}",
                    f"SET LOCAL search_path TO {schema}",
                    "CREATE TABLE user_file (id uuid PRIMARY KEY)",
                    "CREATE TABLE regulatory_chunk (id int PRIMARY KEY, user_file_id uuid REFERENCES user_file(id) ON DELETE CASCADE, text text, position int, chunk_type text, status text, validity_start_date date, validity_end_date date, metadata jsonb)",
                    "CREATE TABLE regulatory_temporal_projection (id int PRIMARY KEY, user_file_id uuid REFERENCES user_file(id) ON DELETE CASCADE, canonical_chunk_id text, canonical_revision_id uuid, index_uuid text, effective_start date, effective_end date, retired_at timestamptz, projection_ordinal int, payload jsonb)",
                ):
                    await connection.execute(text(statement))
                await connection.run_sync(upgrade)
                await connection.execute(
                    text("INSERT INTO user_file VALUES (:a), (:b)"),
                    {"a": source_a, "b": source_b},
                )
                await connection.execute(
                    text(
                        "INSERT INTO regulatory_chunk(id,user_file_id,text,position) VALUES (1,:a,'first',0),(2,:a,'second',1)"
                    ),
                    {"a": source_a},
                )

                async def revision(source: UUID = source_a) -> int:
                    return int(
                        (
                            await connection.execute(
                                text(
                                    "SELECT revision FROM legal_composite_source_state WHERE user_file_id=:source"
                                ),
                                {"source": source},
                            )
                        ).scalar_one()
                    )

                assert await revision() == 1
                await connection.execute(
                    text("UPDATE regulatory_chunk SET text=text, metadata='{}'")
                )
                assert await revision() == 1
                await connection.execute(
                    text("UPDATE regulatory_chunk SET text=text || ' changed'")
                )
                assert await revision() == 2
                await connection.execute(
                    text(
                        "INSERT INTO regulatory_temporal_projection(id,user_file_id,payload) VALUES (1,:a,'{}')"
                    ),
                    {"a": source_a},
                )
                assert await revision() == 3
                await connection.execute(
                    text(
                        "UPDATE regulatory_temporal_projection SET payload=jsonb_build_object('representation_text','amended')"
                    )
                )
                assert await revision() == 4
                await connection.execute(
                    text("UPDATE regulatory_chunk SET user_file_id=:b WHERE id=2"),
                    {"b": source_b},
                )
                assert await revision() == 5 and await revision(source_b) == 1
                await connection.execute(
                    text("DELETE FROM user_file WHERE id=:a"), {"a": source_a}
                )
                assert (
                    await connection.execute(
                        text("SELECT count(*) FROM legal_composite_source_state")
                    )
                ).scalar_one() == 1
                await connection.run_sync(downgrade)
                assert (
                    await connection.execute(
                        text("SELECT text FROM regulatory_chunk WHERE id=2")
                    )
                ).scalar_one() == "second changed"
            finally:
                await transaction.rollback()
    finally:
        await engine.dispose()
