"""Retain canonical revision authority independently of the live file.

Revision ID: 1fa78a83fa88
Revises: d421936e44b3
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "1fa78a83fa88"
down_revision = "d421936e44b3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "regulatory_canonical_revision",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("user_file_id", sa.Uuid(), nullable=False),
        sa.Column("canonical_chunk_id", sa.Text(), nullable=False),
        sa.Column("payload_sha256", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "canonical_chunk_id", "payload_sha256", name="uq_canonical_revision_payload"
        ),
    )
    op.create_index(
        "ix_regulatory_canonical_revision_user_file_id",
        "regulatory_canonical_revision",
        ["user_file_id"],
    )
    op.create_index(
        "ix_regulatory_canonical_revision_canonical_chunk_id",
        "regulatory_canonical_revision",
        ["canonical_chunk_id"],
    )
    op.add_column(
        "regulatory_temporal_projection",
        sa.Column("canonical_revision_id", sa.Uuid(), nullable=True),
    )
    op.create_foreign_key(
        "fk_temporal_canonical_revision",
        "regulatory_temporal_projection",
        "regulatory_canonical_revision",
        ["canonical_revision_id"],
        ["id"],
    )
    op.drop_constraint(
        "annex_change_set_user_file_id_fkey", "annex_change_set", type_="foreignkey"
    )
    op.drop_constraint(
        "regulatory_temporal_projection_user_file_id_fkey",
        "regulatory_temporal_projection",
        type_="foreignkey",
    )
    op.drop_constraint(
        "regulatory_temporal_projection_canonical_chunk_id_fkey",
        "regulatory_temporal_projection",
        type_="foreignkey",
    )
    op.execute("""CREATE FUNCTION reject_canonical_revision_update() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN RAISE EXCEPTION 'canonical revision authority is immutable'; END; $$""")
    op.execute(
        "CREATE TRIGGER immutable_canonical_revision BEFORE UPDATE ON regulatory_canonical_revision FOR EACH ROW EXECUTE FUNCTION reject_canonical_revision_update()"
    )
    op.execute("""CREATE FUNCTION protect_temporal_canonical_revision() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF OLD.canonical_revision_id IS NOT NULL AND NEW.canonical_revision_id IS DISTINCT FROM OLD.canonical_revision_id THEN
        RAISE EXCEPTION 'temporal canonical revision authority is immutable';
      END IF;
      RETURN NEW;
    END; $$""")
    op.execute(
        "CREATE TRIGGER immutable_temporal_canonical_revision BEFORE UPDATE ON regulatory_temporal_projection FOR EACH ROW EXECUTE FUNCTION protect_temporal_canonical_revision()"
    )


def downgrade() -> None:
    # Refuse loss of correction evidence or detached review history.
    op.execute("""DO $$ BEGIN IF EXISTS (SELECT 1 FROM regulatory_canonical_revision) THEN
      RAISE EXCEPTION 'retained canonical revisions require a data-preserving migration';
    END IF; END $$""")
    op.create_foreign_key(
        "annex_change_set_user_file_id_fkey",
        "annex_change_set",
        "user_file",
        ["user_file_id"],
        ["id"],
    )
    op.create_foreign_key(
        "regulatory_temporal_projection_user_file_id_fkey",
        "regulatory_temporal_projection",
        "user_file",
        ["user_file_id"],
        ["id"],
    )
    op.create_foreign_key(
        "regulatory_temporal_projection_canonical_chunk_id_fkey",
        "regulatory_temporal_projection",
        "regulatory_chunk",
        ["canonical_chunk_id"],
        ["id"],
    )
    op.execute(
        "DROP TRIGGER immutable_temporal_canonical_revision ON regulatory_temporal_projection"
    )
    op.execute("DROP FUNCTION protect_temporal_canonical_revision()")
    op.drop_constraint(
        "fk_temporal_canonical_revision",
        "regulatory_temporal_projection",
        type_="foreignkey",
    )
    op.drop_column("regulatory_temporal_projection", "canonical_revision_id")
    op.drop_table("regulatory_canonical_revision")
    op.execute("DROP FUNCTION reject_canonical_revision_update()")
