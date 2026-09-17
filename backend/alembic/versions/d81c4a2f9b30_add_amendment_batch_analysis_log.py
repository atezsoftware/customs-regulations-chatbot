"""add amendment batch analysis log

Revision ID: d81c4a2f9b30
Revises: c6b70d871979
Create Date: 2026-09-17 01:10:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "d81c4a2f9b30"
down_revision = "c6b70d871979"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "amendment_batch",
        sa.Column(
            "analysis_log",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column("amendment_batch", "analysis_log")
