"""add grouped annex review staging

Revision ID: bde9f17c588a
Revises: 03e360fc9417
Create Date: 2026-09-10 00:41:00.420929

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "bde9f17c588a"
down_revision = "03e360fc9417"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "annex_change_set",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "batch_id",
            sa.Integer(),
            sa.ForeignKey("amendment_batch.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("instruction_index", sa.Integer(), nullable=False),
        sa.Column("instruction_indices", postgresql.JSONB(), nullable=False),
        sa.Column("environment", sa.Text(), nullable=False),
        sa.Column(
            "user_file_id", sa.Uuid(), sa.ForeignKey("user_file.id"), nullable=True
        ),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("review_sha256", sa.Text(), nullable=False),
        sa.Column("review_payload", postgresql.JSONB(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("decided_by", sa.Uuid(), sa.ForeignKey("user.id"), nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "publication_generation", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "batch_id", "instruction_index", name="uq_annex_change_instruction"
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'blocked', 'approving', 'preparing', 'publishing', 'approved', 'rejected', 'failed')",
            name="annex_change_status_check",
        ),
    )
    op.create_index(
        "ix_annex_change_pending_publication",
        "annex_change_set",
        ["environment", "status", "heartbeat_at"],
    )
    op.create_table(
        "annex_change_item",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "change_set_id",
            sa.Uuid(),
            sa.ForeignKey("annex_change_set.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("operation", sa.Text(), nullable=False),
        sa.Column("old_chunk_ids", postgresql.JSONB(), nullable=False),
        sa.Column("prospective_chunk_ids", postgresql.JSONB(), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.UniqueConstraint(
            "change_set_id", "position", name="uq_annex_change_item_position"
        ),
    )

    op.create_table(
        "annex_change_evidence",
        sa.Column(
            "change_set_id",
            sa.Uuid(),
            sa.ForeignKey("annex_change_set.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("evidence_id", sa.Uuid(), primary_key=True),
        sa.Column("file_id", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
    )
    op.execute("""CREATE FUNCTION reject_annex_review_update() RETURNS trigger AS $$
    BEGIN
      IF TG_TABLE_NAME = 'annex_change_set' THEN
        IF (to_jsonb(NEW) - ARRAY['status','error_message','decided_by','decided_at','publication_generation','heartbeat_at','updated_at']) IS DISTINCT FROM
           (to_jsonb(OLD) - ARRAY['status','error_message','decided_by','decided_at','publication_generation','heartbeat_at','updated_at']) THEN
          RAISE EXCEPTION 'annex review evidence is immutable';
        END IF;
      ELSE
        RAISE EXCEPTION 'annex review association is immutable';
      END IF;
      RETURN NEW;
    END; $$ LANGUAGE plpgsql""")
    for table in ("annex_change_set", "annex_change_item", "annex_change_evidence"):
        op.execute(
            f"CREATE TRIGGER immutable_annex_review BEFORE UPDATE ON {table} FOR EACH ROW EXECUTE FUNCTION reject_annex_review_update()"
        )


def downgrade() -> None:
    op.drop_table("annex_change_evidence")
    op.drop_table("annex_change_item")
    op.drop_table("annex_change_set")
    op.execute("DROP FUNCTION IF EXISTS reject_annex_review_update()")
