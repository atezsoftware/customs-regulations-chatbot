"""allow async amendment approval

Revision ID: a8c1d4e7f2b9
Revises: f30912f19544
Create Date: 2026-08-31
"""

from alembic import op


revision = "a8c1d4e7f2b9"
down_revision = "f30912f19544"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "amendment_proposal_status_check",
        "amendment_proposal",
        type_="check",
    )
    op.create_check_constraint(
        "amendment_proposal_status_check",
        "amendment_proposal",
        "status IN ('pending', 'approving', 'approved', 'rejected')",
    )


def downgrade() -> None:
    op.execute(
        "UPDATE amendment_proposal SET status = 'pending', decided_by = NULL, "
        "decided_at = NULL WHERE status = 'approving'"
    )
    op.drop_constraint(
        "amendment_proposal_status_check",
        "amendment_proposal",
        type_="check",
    )
    op.create_check_constraint(
        "amendment_proposal_status_check",
        "amendment_proposal",
        "status IN ('pending', 'approved', 'rejected')",
    )
