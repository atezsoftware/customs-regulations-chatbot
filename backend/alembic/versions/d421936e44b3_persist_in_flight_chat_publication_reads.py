"""Persist in-flight chat publication reads.

Revision ID: d421936e44b3
Revises: 8c88cdd6317e
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "d421936e44b3"
down_revision = "8c88cdd6317e"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chat_message", sa.Column("publication_read", postgresql.JSONB(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("chat_message", "publication_read")
