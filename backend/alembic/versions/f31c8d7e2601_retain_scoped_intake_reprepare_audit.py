"""Retain scoped intake reprepare audit.

Revision ID: f31c8d7e2601
Revises: e21b7c6d2601
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "f31c8d7e2601"
down_revision = "e21b7c6d2601"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "regulatory_document_intake_reprepare",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "intake_id",
            sa.Uuid(),
            sa.ForeignKey("regulatory_document_intake.id"),
            nullable=False,
        ),
        sa.Column("idempotency_key", sa.Uuid(), nullable=False, unique=True),
        sa.Column("expected_sha256", sa.String(64), nullable=False),
        sa.Column(
            "previous_labeling_run_id",
            sa.Uuid(),
            sa.ForeignKey("regulatory_labeling_run.id"),
        ),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("documents", postgresql.JSONB(), nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'prepared', 'failed')",
            name="intake_reprepare_status_check",
        ),
    )
    op.create_index(
        "ix_regulatory_document_intake_reprepare_intake_id",
        "regulatory_document_intake_reprepare",
        ["intake_id"],
    )


def downgrade() -> None:
    op.drop_table("regulatory_document_intake_reprepare")
