"""add projection repair failure code

Revision ID: c2e8f5b1d3a7
Revises: b1d7e4a9c2f6
Create Date: 2026-08-25
"""

import sqlalchemy as sa
from alembic import op

revision = "c2e8f5b1d3a7"
down_revision = "b1d7e4a9c2f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "user_file_projection_repair",
        sa.Column("failure_code", sa.String(length=128), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("user_file_projection_repair", "failure_code")
