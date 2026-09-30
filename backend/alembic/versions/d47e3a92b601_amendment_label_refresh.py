"""Track durable labeling refresh after amendment publication.

Revision ID: d47e3a92b601
Revises: a8f5d2c31e70
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "d47e3a92b601"
down_revision: str | None = "a8f5d2c31e70"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "regulatory_amendment_label_refresh",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "proposal_id",
            sa.Integer(),
            sa.ForeignKey("amendment_proposal.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "document_set_id",
            sa.Integer(),
            sa.ForeignKey("document_set.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_file_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("user_file.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("new_chunk_ids", postgresql.JSONB(), nullable=False),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("regulatory_labeling_run.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("next_retry_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.String(4000), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "proposal_id",
            "document_set_id",
            name="uq_regulatory_amendment_label_refresh_proposal_set",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name="regulatory_amendment_label_refresh_status_check",
        ),
        sa.CheckConstraint(
            "attempt_count >= 0",
            name="regulatory_amendment_label_refresh_attempt_check",
        ),
    )
    op.create_index(
        "ix_regulatory_amendment_label_refresh_due",
        "regulatory_amendment_label_refresh",
        ["status", "next_retry_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_regulatory_amendment_label_refresh_due",
        table_name="regulatory_amendment_label_refresh",
    )
    op.drop_table("regulatory_amendment_label_refresh")
