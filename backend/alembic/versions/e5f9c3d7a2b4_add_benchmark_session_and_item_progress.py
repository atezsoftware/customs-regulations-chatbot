"""add benchmark session classification and item progress

Revision ID: e5f9c3d7a2b4
Revises: d4e8b2c6f1a3
Create Date: 2026-08-11
"""

import sqlalchemy as sa
from alembic import op

revision = "e5f9c3d7a2b4"
down_revision = "d4e8b2c6f1a3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chat_session",
        sa.Column(
            "benchmark_flow",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.add_column(
        "benchmark_run_item",
        sa.Column("execution_phase", sa.Text(), nullable=True),
    )
    op.add_column(
        "benchmark_run_item",
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_check_constraint(
        "benchmark_run_item_execution_phase_check",
        "benchmark_run_item",
        "execution_phase IS NULL OR execution_phase IN "
        "('starting', 'answering', 'researching', 'judging')",
    )

    # Linked sessions cover completed answers. The strict description pattern
    # also covers sessions created by an item that was interrupted mid-stream,
    # before the legacy runner persisted its chat_session_id.
    op.execute(
        sa.text(
            """
            UPDATE chat_session AS session
            SET benchmark_flow = true
            WHERE EXISTS (
                SELECT 1
                FROM benchmark_run_item AS item
                WHERE item.chat_session_id = session.id
            )
               OR session.description ~ '^Benchmark run [0-9]+, item [0-9]+$'
            """
        )
    )


def downgrade() -> None:
    op.drop_constraint(
        "benchmark_run_item_execution_phase_check",
        "benchmark_run_item",
        type_="check",
    )
    op.drop_column("benchmark_run_item", "heartbeat_at")
    op.drop_column("benchmark_run_item", "execution_phase")
    op.drop_column("chat_session", "benchmark_flow")
