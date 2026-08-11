"""normalize cancelled benchmark items

Revision ID: d4e8b2c6f1a3
Revises: c3f7a1e5d9b2
Create Date: 2026-08-11
"""

import sqlalchemy as sa
from alembic import op

revision = "d4e8b2c6f1a3"
down_revision = "c3f7a1e5d9b2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        sa.text(
            """
            UPDATE benchmark_run_item AS item
            SET status = 'cancelled',
                completed_at = COALESCE(item.completed_at, run.completed_at, NOW())
            FROM benchmark_run AS run
            WHERE item.run_id = run.id
              AND run.status = 'cancelled'
              AND item.status IN ('pending', 'running')
            """
        )
    )


def downgrade() -> None:
    # This is a terminal-state data repair; the previous inconsistent state is
    # neither meaningful nor safely reconstructable.
    pass
