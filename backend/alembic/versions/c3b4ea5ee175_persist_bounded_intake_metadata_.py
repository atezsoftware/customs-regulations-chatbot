"""Persist bounded intake metadata recovery checkpoints.

Revision ID: c3b4ea5ee175
Revises: 576f0377f747
Create Date: 2026-11-03 13:50:47.606891

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "c3b4ea5ee175"
down_revision = "576f0377f747"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "regulatory_document_intake",
        sa.Column(
            "metadata_checkpoint", postgresql.JSONB(none_as_null=True), nullable=True
        ),
    )
    op.add_column(
        "regulatory_document_intake",
        sa.Column(
            "metadata_recovery_count", sa.Integer(), nullable=False, server_default="0"
        ),
    )
    op.create_check_constraint(
        "ck_regulatory_intake_metadata_recovery_count",
        "regulatory_document_intake",
        "metadata_recovery_count BETWEEN 0 AND 3",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_regulatory_intake_metadata_recovery_count",
        "regulatory_document_intake",
        type_="check",
    )
    op.drop_column("regulatory_document_intake", "metadata_recovery_count")
    op.drop_column("regulatory_document_intake", "metadata_checkpoint")
