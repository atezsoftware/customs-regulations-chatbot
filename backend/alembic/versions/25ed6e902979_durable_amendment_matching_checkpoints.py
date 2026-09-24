"""durable amendment matching checkpoints

Revision ID: 25ed6e902979
Revises: f43b8c129e70
Create Date: 2026-09-24 12:23:52.476952

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "25ed6e902979"
down_revision = "f43b8c129e70"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "amendment_match_checkpoint",
        sa.Column(
            "batch_id",
            sa.Integer(),
            sa.ForeignKey("amendment_batch.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("instruction_index", sa.Integer(), primary_key=True),
        sa.Column("input_sha256", sa.Text(), nullable=False),
        sa.Column("lease_generation", sa.Integer(), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("source_fingerprints", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "instruction_index >= 0 AND lease_generation >= 0",
            name="amendment_match_checkpoint_indices_check",
        ),
    )


def downgrade() -> None:
    op.drop_table("amendment_match_checkpoint")
