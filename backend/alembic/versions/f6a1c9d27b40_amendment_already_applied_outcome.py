"""Retain an audited outcome for an identical previously applied version.

Revision ID: f6a1c9d27b40
Revises: d8b7a14f902e
"""

from alembic import op
import sqlalchemy as sa

revision = "f6a1c9d27b40"
down_revision = "d8b7a14f902e"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "amendment_proposal_status_check", "amendment_proposal", type_="check"
    )
    op.create_check_constraint(
        "amendment_proposal_status_check",
        "amendment_proposal",
        "status IN ('pending', 'approving', 'approval_failed', 'approved', 'rejected', 'already_applied')",
    )


def downgrade() -> None:
    if op.get_bind().scalar(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM amendment_proposal WHERE status = 'already_applied')"
        )
    ):
        raise ValueError(
            "Cannot remove audited already_applied outcomes; retain this schema revision"
        )
    op.drop_constraint(
        "amendment_proposal_status_check", "amendment_proposal", type_="check"
    )
    op.create_check_constraint(
        "amendment_proposal_status_check",
        "amendment_proposal",
        "status IN ('pending', 'approving', 'approval_failed', 'approved', 'rejected')",
    )
