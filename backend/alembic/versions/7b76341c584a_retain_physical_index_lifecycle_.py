"""Retain physical index lifecycle operation authority

Revision ID: 7b76341c584a
Revises: 601b08f82b8c
Create Date: 2026-09-11 00:23:42.596939

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "7b76341c584a"
down_revision = "601b08f82b8c"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "regulatory_physical_index_operation",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("scope_key", sa.Text(), nullable=False),
        sa.Column("index_name", sa.Text(), nullable=False),
        sa.Column("index_uuid", sa.Text(), nullable=False),
        sa.Column("operation", sa.Text(), nullable=False),
        sa.Column("request", postgresql.JSONB(), nullable=False),
        sa.Column("request_sha256", sa.Text(), nullable=False),
        sa.Column("phase", sa.Text(), nullable=False),
        sa.Column("owner_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("result_index_uuid", sa.Text(), nullable=True),
        sa.Column("terminal_evidence", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "uq_active_physical_index_operation",
        "regulatory_physical_index_operation",
        ["index_name"],
        unique=True,
        postgresql_where=sa.text("completed_at IS NULL"),
    )
    op.execute("""CREATE FUNCTION protect_physical_index_operation() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF ROW(OLD.scope_key, OLD.index_name, OLD.index_uuid, OLD.operation, OLD.request, OLD.request_sha256)
         IS DISTINCT FROM ROW(NEW.scope_key, NEW.index_name, NEW.index_uuid, NEW.operation, NEW.request, NEW.request_sha256) THEN
        RAISE EXCEPTION 'physical index operation authority is immutable';
      END IF;
      IF OLD.terminal_evidence IS NOT NULL AND OLD.terminal_evidence IS DISTINCT FROM NEW.terminal_evidence THEN
        RAISE EXCEPTION 'physical terminal evidence is immutable';
      END IF;
      RETURN NEW;
    END; $$""")
    op.execute(
        "CREATE TRIGGER immutable_physical_index_operation BEFORE UPDATE ON regulatory_physical_index_operation FOR EACH ROW EXECUTE FUNCTION protect_physical_index_operation()"
    )


def downgrade() -> None:
    op.execute("""DO $$ BEGIN IF EXISTS (SELECT 1 FROM regulatory_physical_index_operation WHERE completed_at IS NULL) THEN
      RAISE EXCEPTION 'pending physical index operations require a data-preserving migration';
    END IF; END $$""")
    op.drop_table("regulatory_physical_index_operation")
    op.execute("DROP FUNCTION protect_physical_index_operation()")
