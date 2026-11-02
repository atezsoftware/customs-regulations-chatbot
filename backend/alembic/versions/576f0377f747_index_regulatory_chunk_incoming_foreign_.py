"""Index incoming chunk references so canonical replacement avoids repeated scans.

Revision ID: 576f0377f747
Revises: e363f58e0eaf
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "576f0377f747"
down_revision: str | None = "e363f58e0eaf"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEXES: dict[str, tuple[str, str]] = {
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


def _index_state(
    conn: sa.engine.Connection, schema: str, name: str, table: str, column: str
) -> bool | None:
    quote = conn.dialect.identifier_preparer.quote_identifier
    row = (
        conn.execute(
            sa.text("""
            SELECT i.indisvalid, i.indisready,
                   n.nspname = :schema AND t.relname = :table
                   AND am.amname = 'btree' AND NOT i.indisunique
                   AND i.indnkeyatts = 1 AND i.indnatts = 1
                   AND i.indpred IS NULL AND i.indexprs IS NULL
                   AND a.attname = :column AND opc.opcdefault
                   AND i.indoption[0] = 0
                   AND i.indcollation[0] = a.attcollation AS expected_definition,
                   EXISTS (SELECT 1 FROM pg_stat_progress_create_index p
                           WHERE p.index_relid = i.indexrelid) AS building
            FROM pg_index i
            JOIN pg_class idx ON idx.oid = i.indexrelid
            JOIN pg_class t ON t.oid = i.indrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            JOIN pg_am am ON am.oid = idx.relam
            LEFT JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = i.indkey[0]
            LEFT JOIN pg_opclass opc ON opc.oid = i.indclass[0]
            WHERE i.indexrelid = to_regclass(:qualified_name)
        """),
            {
                "schema": schema,
                "table": table,
                "column": column,
                "qualified_name": f"{quote(schema)}.{quote(name)}",
            },
        )
        .mappings()
        .one_or_none()
    )
    if row is None:
        return None
    if not row["expected_definition"]:
        raise RuntimeError(
            f"Index {name} differs from the expected chunk reference definition"
        )
    if row["building"]:
        raise RuntimeError(f"Index {name} still has an active concurrent build")
    return bool(row["indisvalid"] and row["indisready"])


def _migration_connection() -> tuple[sa.engine.Connection, str]:
    bind = op.get_bind()
    schema = bind.execute(sa.text("SELECT current_schema()")).scalar_one()
    # env.py owns the outer transaction; release its snapshot before concurrent DDL.
    bind.commit()
    return bind, schema


def upgrade() -> None:
    bind, schema = _migration_connection()
    with bind.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        quote = conn.dialect.identifier_preparer.quote_identifier
        for name, (table, column) in INDEXES.items():
            state = _index_state(conn, schema, name, table, column)
            if state is True:
                continue
            if state is False:
                conn.exec_driver_sql(
                    f"DROP INDEX CONCURRENTLY {quote(schema)}.{quote(name)}"
                )
            conn.exec_driver_sql(
                f"CREATE INDEX CONCURRENTLY {quote(name)} ON {quote(schema)}.{quote(table)} ({quote(column)})"
            )
            if _index_state(conn, schema, name, table, column) is not True:
                raise RuntimeError(f"Index {name} did not become valid")


def downgrade() -> None:
    bind, schema = _migration_connection()
    with bind.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        quote = conn.dialect.identifier_preparer.quote_identifier
        for name, (table, column) in reversed(INDEXES.items()):
            if _index_state(conn, schema, name, table, column) is not None:
                conn.exec_driver_sql(
                    f"DROP INDEX CONCURRENTLY {quote(schema)}.{quote(name)}"
                )
