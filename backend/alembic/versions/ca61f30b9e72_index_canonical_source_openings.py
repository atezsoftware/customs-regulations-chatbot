"""Index canonical opening order and retired-inclusive file qualification.

Revision ID: ca61f30b9e72
Revises: 898eaa5536e4
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "ca61f30b9e72"
down_revision: str | None = "898eaa5536e4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEXES = {
    "ix_temporal_projection_file_qualification": "(user_file_id)",
    "ix_temporal_projection_canonical_opening": (
        "(user_file_id, index_uuid, ((payload ->> 'semantic_position')::integer), "
        "projection_ordinal) WHERE retired_at IS NULL "
        "AND (payload ->> 'derived_role') = 'canonical' "
        "AND ((payload ->> 'semantic_position')::integer) >= 0"
    ),
}


def _index_state(conn: sa.engine.Connection, schema: str, name: str) -> bool | None:
    qualified = conn.dialect.identifier_preparer.quote_schema(schema) + "." + name
    row = conn.execute(
        sa.text(
            "SELECT i.indisvalid FROM pg_index i "
            "WHERE i.indexrelid = to_regclass(:qualified_name)"
        ),
        {"qualified_name": qualified},
    ).one_or_none()
    return row[0] if row is not None else None


def _release_migration_snapshot() -> tuple[sa.engine.Connection, str]:
    bind = op.get_bind()
    schema = bind.execute(sa.text("SELECT current_schema()"), {}).scalar_one()
    bind.commit()
    return bind, schema


def upgrade() -> None:
    bind, schema = _release_migration_snapshot()
    quoted_schema = bind.dialect.identifier_preparer.quote_schema(schema)
    # The dedicated connection preserves concurrent readers and tenant isolation.
    with bind.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        for name, definition in INDEXES.items():
            state = _index_state(conn, schema, name)
            if state is True:
                continue
            if state is False:
                conn.exec_driver_sql(f"DROP INDEX CONCURRENTLY {quoted_schema}.{name}")
            conn.exec_driver_sql(
                f"CREATE INDEX CONCURRENTLY {name} "
                f"ON {quoted_schema}.regulatory_temporal_projection {definition}"
            )


def downgrade() -> None:
    bind, schema = _release_migration_snapshot()
    quoted_schema = bind.dialect.identifier_preparer.quote_schema(schema)
    with bind.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        for name in reversed(INDEXES):
            if _index_state(conn, schema, name) is not None:
                conn.exec_driver_sql(f"DROP INDEX CONCURRENTLY {quoted_schema}.{name}")
