"""add benchmark dispatch state

Revision ID: c3f7a1e5d9b2
Revises: b9d5f3a2c8e1
Create Date: 2026-08-11
"""

import sqlalchemy as sa
from alembic import op

revision = "c3f7a1e5d9b2"
down_revision = "b9d5f3a2c8e1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("benchmark_run_status_check", "benchmark_run", type_="check")
    op.create_check_constraint(
        "benchmark_run_status_check",
        "benchmark_run",
        "status IN ('pending', 'queued', 'running', 'completed', 'error', 'cancelled')",
    )
    op.add_column(
        "benchmark_run",
        sa.Column("queued_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "benchmark_run",
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column("benchmark_run", sa.Column("failure_code", sa.Text(), nullable=True))
    op.add_column(
        "benchmark_run", sa.Column("failure_message", sa.Text(), nullable=True)
    )


def downgrade() -> None:
    op.execute("UPDATE benchmark_run SET status = 'pending' WHERE status = 'queued'")
    for column_name in (
        "failure_message",
        "failure_code",
        "heartbeat_at",
        "queued_at",
    ):
        op.drop_column("benchmark_run", column_name)
    op.drop_constraint("benchmark_run_status_check", "benchmark_run", type_="check")
    op.create_check_constraint(
        "benchmark_run_status_check",
        "benchmark_run",
        "status IN ('pending', 'running', 'completed', 'error', 'cancelled')",
    )
