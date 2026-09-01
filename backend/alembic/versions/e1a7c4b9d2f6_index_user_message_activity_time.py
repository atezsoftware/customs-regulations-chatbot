"""index user message activity time

Revision ID: e1a7c4b9d2f6
Revises: c9d4e7f1a2b6
Create Date: 2026-09-01

The usage dashboard counts user messages over a selected time range and groups
them by chat session. This partial index keeps that query bounded to user
messages and supports both the time predicate and session join.

chat_message is hot, so the index is built concurrently. A dedicated
AUTOCOMMIT connection is required because the tenant migration environment
starts a transaction while setting search_path.
"""

import sqlalchemy as sa
from alembic import op


revision = "e1a7c4b9d2f6"
down_revision = "c9d4e7f1a2b6"
branch_labels = None
depends_on = None

INDEX_NAME = "ix_chat_message_user_time_sent_session_id"


def _index_state(conn: sa.engine.Connection, schema: str) -> bool | None:
    row = conn.execute(
        sa.text(
            "SELECT i.indisvalid FROM pg_index i "
            "WHERE i.indexrelid = to_regclass(:qualified_name)"
        ),
        {"qualified_name": f'"{schema}"."{INDEX_NAME}"'},
    ).one_or_none()
    return row[0] if row is not None else None


def _release_migration_snapshot() -> tuple[sa.engine.Connection, str]:
    bind = op.get_bind()
    schema = bind.execute(sa.text("SELECT current_schema()"), {}).scalar_one()
    bind.commit()
    return bind, schema


def upgrade() -> None:
    bind, schema = _release_migration_snapshot()
    with bind.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        state = _index_state(conn, schema)
        if state is True:
            return
        if state is False:
            conn.exec_driver_sql(f'DROP INDEX CONCURRENTLY "{schema}"."{INDEX_NAME}"')
        conn.exec_driver_sql(
            f'CREATE INDEX CONCURRENTLY "{INDEX_NAME}" '
            f'ON "{schema}".chat_message (time_sent, chat_session_id) '
            "WHERE message_type = 'USER'"
        )


def downgrade() -> None:
    bind, schema = _release_migration_snapshot()
    with bind.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        if _index_state(conn, schema) is None:
            return
        conn.exec_driver_sql(f'DROP INDEX CONCURRENTLY "{schema}"."{INDEX_NAME}"')
