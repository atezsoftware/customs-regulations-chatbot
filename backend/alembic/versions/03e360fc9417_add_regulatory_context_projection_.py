"""add regulatory context projection history

Revision ID: 03e360fc9417
Revises: e7c376db8d4c
Create Date: 2026-09-09
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "03e360fc9417"
down_revision = "e7c376db8d4c"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for name, hash_column, constraint in (
        ("regulatory_context_snapshot", "sha256", "uq_context_snapshot_file_hash"),
        (
            "regulatory_context_generation",
            "request_sha256",
            "uq_context_generation_file_hash",
        ),
    ):
        op.create_table(
            name,
            sa.Column("id", sa.UUID(), primary_key=True),
            sa.Column(
                "user_file_id",
                sa.UUID(),
                sa.ForeignKey("user_file.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(hash_column, sa.Text(), nullable=False),
            sa.Column("payload", postgresql.JSONB(), nullable=False),
            sa.UniqueConstraint("user_file_id", hash_column, name=constraint),
        )
    op.create_table(
        "regulatory_context_projection",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column(
            "canonical_chunk_id",
            sa.String(),
            sa.ForeignKey("regulatory_chunk.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "source_snapshot_id",
            sa.UUID(),
            sa.ForeignKey("regulatory_context_snapshot.id"),
            nullable=False,
        ),
        sa.Column("payload_sha256", sa.Text(), nullable=False),
        sa.Column("embedding_input_sha256", sa.Text(), nullable=False),
        sa.Column("embedding_config_sha256", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("effective_start", sa.Date()),
        sa.Column("effective_end", sa.Date()),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "canonical_chunk_id", "payload_sha256", name="uq_context_projection_payload"
        ),
        sa.CheckConstraint(
            "effective_end IS NULL OR effective_start IS NULL OR effective_end > effective_start",
            name="context_projection_dates_check",
        ),
    )
    op.create_index(
        "ix_context_projection_effective",
        "regulatory_context_projection",
        ["canonical_chunk_id", "effective_start", "effective_end"],
    )
    op.create_table(
        "regulatory_context_projection_call",
        sa.Column(
            "projection_id",
            sa.UUID(),
            sa.ForeignKey("regulatory_context_projection.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "generation_id",
            sa.UUID(),
            sa.ForeignKey("regulatory_context_generation.id"),
            primary_key=True,
        ),
    )
    op.execute("""CREATE FUNCTION reject_context_evidence_update() RETURNS trigger AS $$
    BEGIN
      IF TG_TABLE_NAME = 'regulatory_context_projection' THEN
        IF (to_jsonb(NEW) - ARRAY['effective_start','effective_end','published_at']) IS DISTINCT FROM
           (to_jsonb(OLD) - ARRAY['effective_start','effective_end','published_at']) THEN
          RAISE EXCEPTION 'context projection evidence is immutable';
        END IF;
      ELSE
        RAISE EXCEPTION 'context source evidence is immutable';
      END IF;
      RETURN NEW;
    END; $$ LANGUAGE plpgsql""")
    for table in (
        "regulatory_context_snapshot",
        "regulatory_context_generation",
        "regulatory_context_projection",
    ):
        op.execute(
            f"CREATE TRIGGER immutable_context_evidence BEFORE UPDATE ON {table} FOR EACH ROW EXECUTE FUNCTION reject_context_evidence_update()"
        )


def downgrade() -> None:
    op.drop_table("regulatory_context_projection_call")
    op.drop_table("regulatory_context_projection")
    op.drop_table("regulatory_context_generation")
    op.drop_table("regulatory_context_snapshot")
    op.execute("DROP FUNCTION IF EXISTS reject_context_evidence_update()")
