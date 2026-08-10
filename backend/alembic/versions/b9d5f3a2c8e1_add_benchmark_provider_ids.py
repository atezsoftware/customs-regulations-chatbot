"""Persist exact provider identity for benchmark models.

Revision ID: b9d5f3a2c8e1
Revises: a8c4e2f1b7d9
"""

import sqlalchemy as sa
from alembic import op

revision = "b9d5f3a2c8e1"
down_revision = "a8c4e2f1b7d9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "benchmark_run",
        sa.Column("judge_provider_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "benchmark_run_item",
        sa.Column("provider_id", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("benchmark_run_item", "provider_id")
    op.drop_column("benchmark_run", "judge_provider_id")
