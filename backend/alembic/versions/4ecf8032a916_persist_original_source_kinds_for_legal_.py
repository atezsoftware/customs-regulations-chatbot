"""persist original source kinds for legal composite

Revision ID: 4ecf8032a916
Revises: ca61f30b9e72
Create Date: 2026-10-08 21:57:41.054236

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "4ecf8032a916"
down_revision: str | None = "ca61f30b9e72"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE_FIELDS = {
    "regulatory_chunk": (
        "user_file_id",
        "text",
        "position",
        "chunk_type",
        "status",
        "validity_start_date",
        "validity_end_date",
    ),
    "regulatory_temporal_projection": (
        "user_file_id",
        "canonical_chunk_id",
        "canonical_revision_id",
        "index_uuid",
        "effective_start",
        "effective_end",
        "retired_at",
        "projection_ordinal",
        "payload ->> 'representation_text'",
        "payload ->> 'semantic_position'",
        "payload ->> 'derived_role'",
    ),
}


def _invalidate_trigger(table: str, operation: str, schema: str) -> None:
    name = f"lc_kind_{table}_{operation}"
    if operation == "insert":
        references = "REFERENCING NEW TABLE AS new_rows"
        changed = "SELECT DISTINCT user_file_id FROM new_rows"
    elif operation == "delete":
        references = "REFERENCING OLD TABLE AS old_rows"
        changed = "SELECT DISTINCT user_file_id FROM old_rows"
    else:
        references = "REFERENCING OLD TABLE AS old_rows NEW TABLE AS new_rows"
        comparisons = " OR ".join(
            f"(o.{field}) IS DISTINCT FROM (n.{field})" for field in TABLE_FIELDS[table]
        )
        changed = (
            "SELECT DISTINCT unnest(ARRAY[o.user_file_id, n.user_file_id]) AS user_file_id "
            "FROM old_rows o JOIN new_rows n ON o.id = n.id "
            f"WHERE {comparisons}"
        )
    op.execute(
        sa.text(f"""
        CREATE FUNCTION {schema}.{name}() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            INSERT INTO {schema}.legal_composite_source_state AS state
                (user_file_id, revision, updated_at)
            SELECT changed.user_file_id, 1, now()
            FROM ({changed}) AS changed
            JOIN {schema}.user_file f ON f.id = changed.user_file_id
            ORDER BY changed.user_file_id
            ON CONFLICT (user_file_id) DO UPDATE
              SET revision = state.revision + 1, updated_at = now();
            RETURN NULL;
        END $$;
        CREATE TRIGGER {name} AFTER {operation.upper()} ON {schema}.{table}
        {references} FOR EACH STATEMENT EXECUTE FUNCTION {schema}.{name}();
    """)
    )


def upgrade() -> None:
    op.create_table(
        "legal_composite_source_state",
        sa.Column("user_file_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("revision", sa.BigInteger(), nullable=False, server_default="1"),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(["user_file_id"], ["user_file.id"], ondelete="CASCADE"),
    )
    op.create_table(
        "legal_composite_source_kind",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_file_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_revision", sa.BigInteger(), nullable=False),
        sa.Column("algorithm_version", sa.Text(), nullable=False),
        sa.Column("window_key", sa.Text(), nullable=False),
        sa.Column("effective_start", sa.Date()),
        sa.Column("effective_end", sa.Date()),
        sa.Column("source_kind", sa.Text(), nullable=False),
        sa.Column("classification", postgresql.JSONB(), nullable=False),
        sa.Column(
            "prepared_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(["user_file_id"], ["user_file.id"], ondelete="CASCADE"),
        sa.UniqueConstraint(
            "user_file_id",
            "source_revision",
            "algorithm_version",
            "window_key",
            name="uq_legal_composite_kind_version_window",
        ),
        sa.CheckConstraint(
            "source_kind IN ('constitution','statute','treaty','presidential_decree','regulation','communique','circular','judicial_decision','executive_decision','private_ruling','other','unknown')",
            name="legal_composite_source_kind_check",
        ),
        sa.CheckConstraint(
            "effective_end IS NULL OR effective_start IS NULL OR effective_end > effective_start",
            name="legal_composite_kind_window_check",
        ),
    )
    op.create_index(
        "ix_legal_composite_kind_lookup",
        "legal_composite_source_kind",
        ["user_file_id", "algorithm_version", "source_revision"],
    )
    bind = op.get_bind()
    schema = bind.dialect.identifier_preparer.quote_schema(
        bind.execute(sa.text("SELECT current_schema()")).scalar_one()
    )
    for table in TABLE_FIELDS:
        for operation in ("insert", "update", "delete"):
            _invalidate_trigger(table, operation, schema)


def downgrade() -> None:
    bind = op.get_bind()
    schema = bind.dialect.identifier_preparer.quote_schema(
        bind.execute(sa.text("SELECT current_schema()")).scalar_one()
    )
    for table in TABLE_FIELDS:
        for operation in ("insert", "update", "delete"):
            name = f"lc_kind_{table}_{operation}"
            op.execute(sa.text(f"DROP TRIGGER {name} ON {schema}.{table}"))
            op.execute(sa.text(f"DROP FUNCTION {schema}.{name}()"))
    op.drop_table("legal_composite_source_kind")
    op.drop_table("legal_composite_source_state")
