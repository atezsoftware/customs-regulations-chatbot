"""Durable annex review preparation.

Revision ID: fbe4b3871f5f
Revises: b27e6a4c1d90
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "fbe4b3871f5f"
down_revision = "b27e6a4c1d90"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "annex_review_preparation",
        sa.Column(
            "review_id",
            sa.Uuid(),
            sa.ForeignKey("annex_change_set.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("expected_review_sha256", sa.Text(), nullable=False),
        sa.Column("request_sha256", sa.Text(), nullable=False),
        sa.Column("corrections", postgresql.JSONB(), nullable=False),
        sa.Column("corrected_by", sa.Uuid(), sa.ForeignKey("user.id"), nullable=False),
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("environment", sa.Text(), nullable=False),
        sa.Column("database_identity", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("stage", sa.Text(), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("checkpoint", postgresql.JSONB(), nullable=True),
        sa.Column("completed_chunks", sa.Integer(), nullable=False),
        sa.Column("total_chunks", sa.Integer(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "result_review_id",
            sa.Uuid(),
            sa.ForeignKey("annex_change_set.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'failed')",
            name="annex_review_preparation_status_check",
        ),
    )
    op.create_index(
        "ix_annex_review_preparation_recovery",
        "annex_review_preparation",
        ["environment", "status", "heartbeat_at"],
    )


def downgrade() -> None:
    op.drop_table("annex_review_preparation")
