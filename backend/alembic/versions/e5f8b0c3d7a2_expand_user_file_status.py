"""expand user file status for batch processing

Revision ID: e5f8b0c3d7a2
Revises: d4e7a9b2c6f1
"""

import sqlalchemy as sa
from alembic import op

revision = "e5f8b0c3d7a2"
down_revision = "d4e7a9b2c6f1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "user_file",
        "status",
        existing_type=sa.String(length=10),
        type_=sa.String(length=32),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "user_file",
        "status",
        existing_type=sa.String(length=32),
        type_=sa.String(length=10),
        existing_nullable=False,
    )
