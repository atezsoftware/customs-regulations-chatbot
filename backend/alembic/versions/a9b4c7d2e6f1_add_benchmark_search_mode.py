"""add benchmark search mode

Revision ID: a9b4c7d2e6f1
Revises: 9883f79b2386
Create Date: 2026-08-25
"""

import sqlalchemy as sa
from alembic import op


revision = "a9b4c7d2e6f1"
down_revision = "9883f79b2386"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "benchmark_run",
        sa.Column(
            "search_mode",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'v2'"),
        ),
    )
    op.create_check_constraint(
        "benchmark_run_search_mode_check",
        "benchmark_run",
        "search_mode IN ('v1', 'v2')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "benchmark_run_search_mode_check", "benchmark_run", type_="check"
    )
    op.drop_column("benchmark_run", "search_mode")
