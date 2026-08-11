"""allow benchmark preparing session execution phase

Revision ID: f7b2e4c6a8d1
Revises: e5f9c3d7a2b4
Create Date: 2026-08-11
"""

import sqlalchemy as sa
from alembic import op

revision = "f7b2e4c6a8d1"
down_revision = "e5f9c3d7a2b4"
branch_labels = None
depends_on = None

_CONSTRAINT_NAME = "benchmark_run_item_execution_phase_check"
_TABLE_NAME = "benchmark_run_item"


def upgrade() -> None:
    op.drop_constraint(_CONSTRAINT_NAME, _TABLE_NAME, type_="check")
    op.create_check_constraint(
        _CONSTRAINT_NAME,
        _TABLE_NAME,
        "execution_phase IS NULL OR execution_phase IN "
        "('starting', 'preparing_session', 'answering', 'researching', 'judging')",
    )


def downgrade() -> None:
    op.execute(
        sa.text(
            "UPDATE benchmark_run_item "
            "SET execution_phase = 'starting' "
            "WHERE execution_phase = 'preparing_session'"
        )
    )
    op.drop_constraint(_CONSTRAINT_NAME, _TABLE_NAME, type_="check")
    op.create_check_constraint(
        _CONSTRAINT_NAME,
        _TABLE_NAME,
        "execution_phase IS NULL OR execution_phase IN "
        "('starting', 'answering', 'researching', 'judging')",
    )
