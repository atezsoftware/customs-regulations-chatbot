"""add durable user file projection repair state

Revision ID: b1d7e4a9c2f6
Revises: a9b4c7d2e6f1
Create Date: 2026-08-25
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "b1d7e4a9c2f6"
down_revision = "a9b4c7d2e6f1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_file_projection_repair",
        sa.Column("user_file_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("attempt_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.String(length=9), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED')",
            name="user_file_projection_repair_status_check",
        ),
        sa.ForeignKeyConstraint(["user_file_id"], ["user_file.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_file_id"),
        sa.UniqueConstraint("attempt_id"),
    )


def downgrade() -> None:
    op.drop_table("user_file_projection_repair")
