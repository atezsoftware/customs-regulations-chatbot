"""Index immutable parent-family locators without rewriting text or vectors.

Revision ID: 898eaa5536e4
Revises: d47e3a92b601
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "898eaa5536e4"
down_revision: str | None = "d47e3a92b601"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEX_NAME = "ix_temporal_projection_frozen_parent"


def _index_state(conn: sa.engine.Connection, schema: str) -> bool | None:
    qualified = conn.dialect.identifier_preparer.quote_schema(schema) + "." + INDEX_NAME
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
    with bind.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        state = _index_state(conn, schema)
        if state is True:
            return
        if state is False:
            conn.exec_driver_sql(
                f"DROP INDEX CONCURRENTLY {quoted_schema}.{INDEX_NAME}"
            )
        conn.exec_driver_sql(
            f"CREATE INDEX CONCURRENTLY {INDEX_NAME} "
            f"ON {quoted_schema}.regulatory_temporal_projection "
            "(user_file_id, index_uuid, "
            "md5((((payload #>> '{projection,source_json}')::jsonb "
            "-> 'heading_path') - -1)::text)) WHERE retired_at IS NULL"
        )


def downgrade() -> None:
    bind, schema = _release_migration_snapshot()
    quoted_schema = bind.dialect.identifier_preparer.quote_schema(schema)
    with bind.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        if _index_state(conn, schema) is None:
            return
        conn.exec_driver_sql(f"DROP INDEX CONCURRENTLY {quoted_schema}.{INDEX_NAME}")
