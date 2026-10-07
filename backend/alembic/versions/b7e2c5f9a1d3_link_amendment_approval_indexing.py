"""link amendment approval indexing

Revision ID: b7e2c5f9a1d3
Revises: a8c1d4e7f2b9
Create Date: 2026-08-31
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "b7e2c5f9a1d3"
down_revision = "a8c1d4e7f2b9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "amendment_proposal",
        sa.Column(
            "approval_indexing_job_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
    )
    op.add_column(
        "amendment_proposal",
        sa.Column("approval_error", sa.Text(), nullable=True),
    )
    op.create_foreign_key(
        "fk_amendment_proposal_approval_indexing_job_id",
        "amendment_proposal",
        "regulatory_indexing_job",
        ["approval_indexing_job_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_amendment_proposal_approval_indexing_job_id",
        "amendment_proposal",
        ["approval_indexing_job_id"],
    )
    op.drop_constraint(
        "amendment_proposal_status_check",
        "amendment_proposal",
        type_="check",
    )
    op.create_check_constraint(
        "amendment_proposal_status_check",
        "amendment_proposal",
        "status IN ('pending', 'approving', 'approval_failed', 'approved', 'rejected')",
    )


def downgrade() -> None:
    op.execute(
        "UPDATE amendment_proposal SET status = 'approving', "
        "approval_error = NULL WHERE status = 'approval_failed'"
    )
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
    op.drop_index(
        "ix_amendment_proposal_approval_indexing_job_id",
        table_name="amendment_proposal",
    )
    op.drop_constraint(
        "fk_amendment_proposal_approval_indexing_job_id",
        "amendment_proposal",
        type_="foreignkey",
    )
    op.drop_column("amendment_proposal", "approval_error")
    op.drop_column("amendment_proposal", "approval_indexing_job_id")
