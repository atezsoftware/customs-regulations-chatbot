"""add regulatory retry cleanup state and reconcile submission constraints

Revision ID: 9883f79b2386
Revises: b7e9d2c4f6a8
Create Date: 2026-08-20 15:34:47.611087

"""

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision = "9883f79b2386"
down_revision = "b7e9d2c4f6a8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "regulatory_indexing_job_submission_state_check",
        "regulatory_indexing_job",
        type_="check",
    )
    op.create_check_constraint(
        "regulatory_indexing_job_submission_state_check",
        "regulatory_indexing_job",
        "vertex_submission_state IN ('NONE', 'SUBMITTING', "
        "'RECONCILE_REQUIRED', 'RECONCILED_ABSENT', "
        "'RETRY_CLEANUP_REQUIRED', 'SUBMITTED')",
    )
    # Some environments applied an earlier b7 artifact before its source gained
    # the manual-reconciliation value. Recreate the constraint deterministically.
    op.drop_constraint(
        "regulatory_indexing_job_openrouter_submission_state_check",
        "regulatory_indexing_job",
        type_="check",
    )
    op.create_check_constraint(
        "regulatory_indexing_job_openrouter_submission_state_check",
        "regulatory_indexing_job",
        "openrouter_submission_state IN ('NONE', 'SUBMITTING', "
        "'RECONCILE_REQUIRED', 'RECONCILED_ABSENT', "
        "'MANUAL_RECONCILE_REQUIRED', 'SUBMITTED')",
    )


def downgrade() -> None:
    op.execute(
        sa.text(
            "UPDATE regulatory_indexing_job "
            "SET vertex_submission_state = 'SUBMITTED' "
            "WHERE vertex_submission_state = 'RETRY_CLEANUP_REQUIRED'"
        )
    )
    op.drop_constraint(
        "regulatory_indexing_job_submission_state_check",
        "regulatory_indexing_job",
        type_="check",
    )
    op.create_check_constraint(
        "regulatory_indexing_job_submission_state_check",
        "regulatory_indexing_job",
        "vertex_submission_state IN ('NONE', 'SUBMITTING', "
        "'RECONCILE_REQUIRED', 'RECONCILED_ABSENT', 'SUBMITTED')",
    )
