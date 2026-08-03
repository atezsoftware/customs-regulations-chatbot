"""add amendment batches and proposals

Revision ID: 9ec902e08372
Revises: 9ce718a30332
Create Date: 2026-07-31 16:31:26.544190

The update mechanism: an admin pastes amendment text scoped to a directory
(user_project). It's segmented into atomic instructions
(amendment_batch), each becomes an individually approvable/rejectable
amendment_proposal with a frozen old/new chunk snapshot. Nothing writes to
regulatory_chunk until a proposal is approved (see
onyx/db/regulatory_amendments.py's approve_amendment_proposal).
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "9ec902e08372"
down_revision = "9ce718a30332"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "amendment_batch",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "project_id",
            sa.Integer(),
            sa.ForeignKey("user_project.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("reference_date", sa.Date(), nullable=True),
        sa.Column(
            "status", sa.Text(), nullable=False, server_default=sa.text("'analyzing'")
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_by",
            sa.Uuid(),
            sa.ForeignKey("user.id"),
            nullable=True,
        ),
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
        sa.CheckConstraint(
            "status IN ('analyzing', 'analyzed', 'failed')",
            name="amendment_batch_status_check",
        ),
    )
    op.create_index("ix_amendment_batch_project_id", "amendment_batch", ["project_id"])

    op.create_table(
        "amendment_proposal",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "batch_id",
            sa.Integer(),
            sa.ForeignKey("amendment_batch.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("instruction_index", sa.Integer(), nullable=False),
        sa.Column("instruction_text", sa.Text(), nullable=False),
        sa.Column(
            "old_chunk_id",
            sa.Text(),
            sa.ForeignKey("regulatory_chunk.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "old_chunk_snapshot",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "new_chunk_draft",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("match_confidence", sa.Float(), nullable=True),
        sa.Column("match_rationale", sa.Text(), nullable=True),
        sa.Column("date_rationale", sa.Text(), nullable=True),
        sa.Column(
            "status", sa.Text(), nullable=False, server_default=sa.text("'pending'")
        ),
        sa.Column(
            "applied_new_chunk_id",
            sa.Text(),
            sa.ForeignKey("regulatory_chunk.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "decided_by",
            sa.Uuid(),
            sa.ForeignKey("user.id"),
            nullable=True,
        ),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.CheckConstraint(
            "status IN ('pending', 'approved', 'rejected')",
            name="amendment_proposal_status_check",
        ),
    )
    op.create_index(
        "ix_amendment_proposal_batch_id", "amendment_proposal", ["batch_id"]
    )
    op.create_index("ix_amendment_proposal_status", "amendment_proposal", ["status"])
    op.create_index(
        "ix_amendment_proposal_old_chunk_id", "amendment_proposal", ["old_chunk_id"]
    )


def downgrade() -> None:
    op.drop_table("amendment_proposal")
    op.drop_table("amendment_batch")
